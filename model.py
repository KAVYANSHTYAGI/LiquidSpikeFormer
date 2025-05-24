import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

# ----------------------------------
# Spike Encoder (Enhanced with Smoothing)
# ----------------------------------
class SpikeEncoder(nn.Module):
    """
    Encodes raw event-based input into binned spike tensors with:
      - Learnable/fixed bin edges
      - Efficient vectorized binning
      - Gaussian temporal smoothing
      - Channel-wise (pixel) normalization
      - Global normalization
    """
    def __init__(self,
                 num_bins: int,
                 height: int,
                 width: int,
                 poisson: bool = False,
                 learnable_bins: bool = False,
                 smooth_kernel_size: int = 5):
        super(SpikeEncoder, self).__init__()
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.poisson = poisson
        self.P = height * width
        # Bin edges
        edges = torch.linspace(0, 1, num_bins + 1)[1:-1]
        if learnable_bins:
            self.bin_edges = nn.Parameter(edges)
        else:
            self.register_buffer('bin_edges', edges)
        # Gaussian smoothing conv
        pad = smooth_kernel_size // 2
        self.smooth_conv = nn.Conv1d(
            in_channels=self.P, out_channels=self.P,
            kernel_size=smooth_kernel_size,
            padding=pad, groups=self.P, bias=False
        )
        # Initialize Gaussian kernel
        coords = torch.arange(smooth_kernel_size) - pad
        sigma = smooth_kernel_size / 6.0
        gauss = torch.exp(-coords**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel = gauss.view(1, 1, smooth_kernel_size).repeat(self.P, 1, 1)
        self.smooth_conv.weight.data.copy_(kernel)
        self.smooth_conv.weight.requires_grad = False
        # Normalizations
        self.pixel_norm = nn.LayerNorm(self.P)
        self.global_norm = nn.LayerNorm([num_bins, self.P])

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        device = events.device
        # Binning
        if events.dim() == 3:
            spikes = events.float()
        else:
            B, N, _ = events.shape
            xs = events[:, :, 0].long().clamp(0, self.height - 1)
            ys = events[:, :, 1].long().clamp(0, self.width - 1)
            ts = events[:, :, 2].clamp(0, 1)
            bin_idx = torch.bucketize(ts, self.bin_edges, right=True)
            pixel_idx = xs * self.width + ys
            counts = torch.zeros(B, self.num_bins, self.P, device=device)
            counts_flat = counts.view(B, -1)
            idx = (bin_idx * self.P + pixel_idx).view(-1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N).reshape(-1)
            counts_flat.index_add_(0, idx + batch_idx * self.num_bins * self.P,
                                   torch.ones(B * N, device=device))
            spikes = counts_flat.view(B, self.num_bins, self.P)
        # Poisson encoding
        if self.poisson:
            maxc = spikes.amax(dim=2, keepdim=True).clamp(min=1.0)
            rates = spikes / maxc
            spikes = torch.bernoulli(rates)
        # Gaussian smoothing across time
        sp = spikes.permute(0, 2, 1)
        sp = self.smooth_conv(sp)
        spikes = sp.permute(0, 2, 1)
        # Pixel-wise normalization
        spikes = self.pixel_norm(spikes)
        # Global normalization
        spikes = self.global_norm(spikes)
        return spikes

# ----------------------------------
# Spiking Patch Embedding
# ----------------------------------
class SpikingPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int,
                 kernel_size: int = 3, stride: int = 1, dropout: float = 0.1):
        super(SpikingPatchEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels, embed_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return self.dropout(x)

# ----------------------------------
# Liquid Time-Constant Block
# ----------------------------------
class LiquidTimeConstantBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super(LiquidTimeConstantBlock, self).__init__()
        self.tau_fc = nn.Linear(dim * 2, dim)
        self.mem_fc = nn.Linear(dim, dim)
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        device = x.device
        v = torch.zeros(B, D, device=device)
        tau = torch.ones(B, D, device=device)
        spikes = []
        for t in range(T):
            xt = self.dropout(x[:, t, :])
            tau = torch.sigmoid(self.tau_fc(torch.cat([xt, tau], dim=-1)))
            alpha = torch.exp(-1.0 / (tau + 1e-6))
            v = alpha * v + (1 - alpha) * self.mem_fc(xt)
            s = (v >= self.threshold).float()
            v = v * (1 - s)
            spikes.append(s)
        spikes = torch.stack(spikes, dim=1)
        return spikes, tau, v

# ----------------------------------
# Positional Encoding
# ----------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        return x + self.pe[:T, :].unsqueeze(1)

# ----------------------------------
# Spiking Transformer Block
# ----------------------------------
class SpikingTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(SpikingTransformerBlock, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x.permute(1, 0, 2)

# ----------------------------------
# Output Head
# ----------------------------------
class OutputHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super(OutputHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x.mean(dim=1))
        return self.fc(x)

# ----------------------------------
# Full Model: Liquid SpikeFormer
# ----------------------------------
class LiquidSpikeFormer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 nhead: int,
                 num_classes: int,
                 encoder_bins: int,
                 height: int,
                 width: int,
                 poisson: bool = False,
                 learnable_bins: bool = False,
                 smooth_kernel_size: int = 5,
                 dropout: float = 0.1):
        super(LiquidSpikeFormer, self).__init__()
        self.encoder = SpikeEncoder(
            num_bins=encoder_bins,
            height=height,
            width=width,
            poisson=poisson,
            learnable_bins=learnable_bins,
            smooth_kernel_size=smooth_kernel_size
        )
        self.patch_embed = SpikingPatchEmbedding(in_channels, embed_dim, dropout=dropout)
        self.liquid_block = LiquidTimeConstantBlock(embed_dim, dropout=dropout)
        self.transformer = SpikingTransformerBlock(d_model=embed_dim, nhead=nhead, dropout=dropout)
        self.head = OutputHead(d_model=embed_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, events: torch.Tensor):
            """
            Returns a dict with:
              - logits: [B, num_classes]
              - spikes: [B, T, D]
              - membrane: [B, D] (final membrane potentials)
              - threshold: scalar or [1]
            """
            x = self.encoder(events)
            x = self.patch_embed(x)
            spikes, _, membrane = self.liquid_block(x)
            x = self.transformer(spikes)
            logits = self.head(x)
            return {
                'logits': logits,
                'spikes': spikes,
                'membrane': membrane,
                'threshold': self.liquid_block.threshold
            }
