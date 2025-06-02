from itertools import count
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------
# TRF + FLI Gate
# ----------------------------------
class TRF_FLIGate(nn.Module):
    """Temporal Response Filter (depth-wise 1D conv) + attention gate."""
    def __init__(self, in_dim, kernel=3):
        super().__init__()
        # Temporal convolution (TRF) – depth-wise to keep params tiny
        self.trf = nn.Conv1d(in_dim, in_dim, kernel_size=kernel,
                             padding=kernel//2, groups=in_dim, bias=False)
        # Feed-forward lateral inhibition (attention gate)
        self.fli = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_dim // 4, in_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, T, P]
        # permute to [B, P, T]
        xp = x.permute(0, 2, 1)
        x_f = self.trf(xp)            # TRF
        gate = self.fli(x_f)          # FLI
        # gated output back to [B, T, P]
        out = (x_f * gate).permute(0, 2, 1)
        return out

# ----------------------------------
# Spike Encoder (unchanged)
# ----------------------------------
class SpikeEncoder(nn.Module):
    """Encodes raw event-based input into binned spike tensors."""
    def __init__(self, num_bins, height, width,
                 poisson=False, learnable_bins=False, smooth_kernel_size=5):
        super().__init__()
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.poisson = poisson
        self.P = height * width
        edges = torch.linspace(0, 1, num_bins + 1)[1:-1]
        if learnable_bins:
            self.bin_edges = nn.Parameter(edges)
        else:
            self.register_buffer('bin_edges', edges)
        pad = smooth_kernel_size // 2
        self.smooth_conv = nn.Conv1d(
            in_channels=self.P, out_channels=self.P,
            kernel_size=smooth_kernel_size,
            padding=pad, groups=self.P, bias=False
        )
        coords = torch.arange(smooth_kernel_size) - pad
        sigma = smooth_kernel_size / 6.0
        gauss = torch.exp(-coords**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel = gauss.view(1, 1, smooth_kernel_size).repeat(self.P, 1, 1)
        self.smooth_conv.weight.data.copy_(kernel)
        self.smooth_conv.weight.requires_grad = False
        self.pixel_norm = nn.LayerNorm(self.P)
        self.global_norm = nn.LayerNorm([num_bins, self.P])

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        device = events.device

        # 1) raw-events path: [B, N, 4] -> bin into [B, num_bins, P]
        if events.dim() == 3 and events.size(-1) == 4:
            B, N, _ = events.shape
            xs = events[:, :, 0].long().clamp(0, self.height - 1)
            ys = events[:, :, 1].long().clamp(0, self.width  - 1)
            ts = events[:, :, 2].clamp(0, 1)
            bin_idx   = torch.bucketize(ts, self.bin_edges, right=True)
            pixel_idx = xs * self.width + ys

            counts = torch.zeros(B, self.num_bins, self.P, device=device)
            flat    = counts.view(-1)

            idx       = (bin_idx * self.P + pixel_idx).view(-1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N).reshape(-1)
            offsets   = idx + batch_idx * (self.num_bins * self.P)

            flat.index_add_(0, offsets, torch.ones_like(offsets, device=device, dtype=flat.dtype))

            spikes = flat.view(B, self.num_bins, self.P)
        else:
            # already-binned path: assume [B, T, P]
            spikes = events.float()

        # 2) Poisson (optional)
        if self.poisson:
            maxc = spikes.amax(dim=2, keepdim=True).clamp(min=1.0)
            rates = spikes / maxc
            spikes = torch.bernoulli(rates)

        # 3) depth-wise smoothing conv over time
        sp = spikes.permute(0, 2, 1)
        sp = self.smooth_conv(sp)
        spikes = sp.permute(0, 2, 1)

        # 4) normalizations
        spikes = self.pixel_norm(spikes)
        spikes = self.global_norm(spikes)

        return spikes
    

    def event_frame(self, events: torch.Tensor):
        # events: [B, N, 4]
        B, N, _ = events.shape
        xs = events[:, :, 0].long().clamp(0, self.height - 1)
        ys = events[:, :, 1].long().clamp(0, self.width  - 1)
        frames = torch.zeros(B, self.height, self.width, device=events.device)
        for b in range(B):
            frames[b].index_put_((xs[b], ys[b]), torch.ones(N, device=events.device), accumulate=True)
        return frames.view(B, -1)  # [B, P]

# ----------------------------------
# Patch Embedding (linear projection)
# ----------------------------------
class SpikingPatchEmbedding(nn.Module):
    def __init__(self, num_pixels: int, embed_dim: int, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(num_pixels, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   # x: [B, T, P]
        B, T, P = x.shape
        x = self.proj(x.reshape(B * T, P)).reshape(B, T, -1)

        return self.dropout(x)

# ----------------------------------
# Liquid Time-Constant Block (adaptive threshold)
# ----------------------------------
class LiquidTimeConstantBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.tau_fc = nn.Linear(dim * 2, dim)
        self.mem_fc = nn.Linear(dim, dim)
        self.log_thresh = nn.Parameter(torch.zeros(dim))
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
            thr = torch.sigmoid(self.log_thresh)
            s = (v >= thr).float()
            v = v * (1 - s)
            spikes.append(s)
        spikes = torch.stack(spikes, dim=1)
        return spikes, tau, v

# ----------------------------------
# Transformer & Heads (unchanged)
# ----------------------------------
class SpikingTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)

class OutputHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x.mean(dim=1)))

# ----------------------------------
# Full Model: LiquidSpikeFormer
# ----------------------------------

class LiquidSpikeFormerMultiBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        nhead: int,
        num_classes: int,
        fine_bins: int,
        coarse_bins: int,
        height: int,
        width: int,
        poisson: bool = False,
        learnable_bins: bool = False,
        smooth_kernel_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        # Multi-resolution spike encoders
        self.encoder_fine = SpikeEncoder(fine_bins, height, width, poisson, learnable_bins, smooth_kernel_size)
        self.encoder_coarse = SpikeEncoder(coarse_bins, height, width, poisson, learnable_bins, smooth_kernel_size)
        self.height = height
        self.width = width
        self.P = height * width

        # Event frame branch
        self.event_frame_proj = nn.Linear(self.P, embed_dim)

        # Patch embedders
        self.patch_embed_fine = SpikingPatchEmbedding(self.P, embed_dim, dropout)
        self.patch_embed_coarse = SpikingPatchEmbedding(self.P, embed_dim, dropout)

        # For concatenation
        self.merge_proj = nn.Linear(embed_dim * 3, embed_dim)

        # Deeper liquid and transformer blocks
        self.liquid_block1 = LiquidTimeConstantBlock(embed_dim, dropout)
        self.liquid_block2 = LiquidTimeConstantBlock(embed_dim, dropout)
        self.transformer1 = SpikingTransformerBlock(d_model=embed_dim, nhead=nhead, dropout=dropout)
        self.transformer2 = SpikingTransformerBlock(d_model=embed_dim, nhead=nhead, dropout=dropout)

        self.head = OutputHead(d_model=embed_dim, num_classes=num_classes, dropout=dropout)
        self.early_head_mid   = nn.Linear(embed_dim, num_classes)
        self.early_head_final = nn.Linear(embed_dim, num_classes)

    def forward(self, events: torch.Tensor):
        # events: [B, N, 4] raw

        # events: list of [N, 4] tensors
        x_fine_list = [self.encoder_fine(ev.unsqueeze(0)) for ev in events]   # Each: [1, T_fine, P]
        x_coarse_list = [self.encoder_coarse(ev.unsqueeze(0)) for ev in events]   # Each: [1, T_coarse, P]
        x_fine = torch.cat(x_fine_list, dim=0)    # [B, T_fine, P]
        x_coarse = torch.cat(x_coarse_list, dim=0)  # [B, T_coarse, P]


        # Average pool coarse to match fine temporal length for concat
        T_fine = x_fine.shape[1]
        x_coarse_upsampled = F.interpolate(x_coarse.permute(0, 2, 1), size=T_fine, mode="linear", align_corners=True)
        x_coarse_upsampled = x_coarse_upsampled.permute(0, 2, 1)

        # 2. Event frame branch
        frame_list = [self.encoder_fine.event_frame(ev.unsqueeze(0)) for ev in events]  # Each: [1, P]
        frame = torch.cat(frame_list, dim=0)  # [B, P]
  # [B, P]
        frame_proj = self.event_frame_proj(frame)      # [B, embed_dim]
        frame_proj = frame_proj.unsqueeze(1).repeat(1, T_fine, 1)  # [B, T_fine, embed_dim]

        # 3. Patch embeddings
        x_fine_embed = self.patch_embed_fine(x_fine)        # [B, T_fine, embed_dim]
        x_coarse_embed = self.patch_embed_coarse(x_coarse_upsampled)  # [B, T_fine, embed_dim]

        # 4. Concatenate all branches
        feats = torch.cat([x_fine_embed, x_coarse_embed, frame_proj], dim=-1)  # [B, T_fine, embed_dim*3]
        feats = self.merge_proj(feats)  # [B, T_fine, embed_dim]

        # Early-exit head (optional)
        early_mid = self.early_head_mid(feats.mean(dim=1))

        # 5. Deeper Liquid block stack
        spikes1, tau1, mem1 = self.liquid_block1(feats)
        spikes2, tau2, mem2 = self.liquid_block2(spikes1)

        # 6. Stacked transformers
        t1 = self.transformer1(spikes2)
        t2 = self.transformer2(t1)

        early_final = self.early_head_final(t2.mean(dim=1))
        logits = self.head(t2)

        return {
            'logits': logits,
            'early_mid': early_mid,
            'early_final': early_final,
            'spikes': spikes2,
            'membrane': mem2,
            'threshold': self.liquid_block2.log_thresh,
            'feats': t2 
        }
    


'''

/mnt/m2ssd/research project/SNN/.venv/lib/python3.8/site-packages/torch/cuda/__init__.py:128: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
  return torch._C._cuda_getDeviceCount() > 0

📂 Loading dataset from: /mnt/m2ssd/research project/SNN/dataset/DVS  Gesture dataset/DvsGesture
✅ Found 1164 training samples, 276 test samples
                                                                                          
Epoch 01/400  Train Loss: 3.1357  Train Acc: 8.85%  Test Loss: 2.5694  Test Acc: 10.87%
✅ New best (10.87%), checkpoint saved.
                                                                                          
Epoch 02/400  Train Loss: 2.9952  Train Acc: 7.99%  Test Loss: 2.5729  Test Acc: 9.78%
                                                                                          
Epoch 03/400  Train Loss: 2.7932  Train Acc: 7.04%  Test Loss: 2.5984  Test Acc: 10.51%
                                                                                          
Epoch 04/400  Train Loss: 2.6427  Train Acc: 9.28%  Test Loss: 2.5542  Test Acc: 7.61%
                                                                                          
Epoch 05/400  Train Loss: 2.5604  Train Acc: 8.25%  Test Loss: 2.5565  Test Acc: 7.25%
                                                                                          
Epoch 06/400  Train Loss: 2.4908  Train Acc: 9.88%  Test Loss: 2.5243  Test Acc: 8.33%
                                                                                          
Epoch 07/400  Train Loss: 2.4535  Train Acc: 9.19%  Test Loss: 2.4877  Test Acc: 12.32%
✅ New best (12.32%), checkpoint saved.
                                                                                          
Epoch 08/400  Train Loss: 2.4094  Train Acc: 10.74%  Test Loss: 2.4160  Test Acc: 13.04%
✅ New best (13.04%), checkpoint saved.
                                                                                          
Epoch 09/400  Train Loss: 2.3784  Train Acc: 14.86%  Test Loss: 2.4001  Test Acc: 13.41%
✅ New best (13.41%), checkpoint saved.
                                                                                          
Epoch 10/400  Train Loss: 2.3341  Train Acc: 15.64%  Test Loss: 2.3718  Test Acc: 16.67%
✅ New best (16.67%), checkpoint saved.
                                                                                          
Epoch 11/400  Train Loss: 2.2921  Train Acc: 20.96%  Test Loss: 2.3138  Test Acc: 22.10%
✅ New best (22.10%), checkpoint saved.
                                                                                          
Epoch 12/400  Train Loss: 2.2637  Train Acc: 23.88%  Test Loss: 2.3066  Test Acc: 21.38%
                                                                                          
Epoch 13/400  Train Loss: 2.2212  Train Acc: 26.89%  Test Loss: 2.2707  Test Acc: 23.19%
✅ New best (23.19%), checkpoint saved.
                                                                                          
Epoch 14/400  Train Loss: 2.1789  Train Acc: 27.41%  Test Loss: 2.2070  Test Acc: 26.81%
✅ New best (26.81%), checkpoint saved.
                                                                                          
Epoch 15/400  Train Loss: 2.1443  Train Acc: 29.21%  Test Loss: 2.1620  Test Acc: 28.26%
✅ New best (28.26%), checkpoint saved.
                                                                                          
Epoch 16/400  Train Loss: 2.1017  Train Acc: 29.98%  Test Loss: 2.1064  Test Acc: 27.54%
                                                                                          
Epoch 17/400  Train Loss: 2.0508  Train Acc: 32.04%  Test Loss: 2.0544  Test Acc: 34.78%
✅ New best (34.78%), checkpoint saved.
                                                                                          
Epoch 18/400  Train Loss: 2.0092  Train Acc: 34.02%  Test Loss: 1.9927  Test Acc: 32.25%
                                                                                          
Epoch 19/400  Train Loss: 1.9727  Train Acc: 37.80%  Test Loss: 1.9410  Test Acc: 36.23%
✅ New best (36.23%), checkpoint saved.
                                                                                          
Epoch 20/400  Train Loss: 1.9221  Train Acc: 39.95%  Test Loss: 1.8759  Test Acc: 42.03%
✅ New best (42.03%), checkpoint saved.
                                                                                          
Epoch 21/400  Train Loss: 1.8753  Train Acc: 42.61%  Test Loss: 1.8262  Test Acc: 42.03%
                                                                                          
Epoch 22/400  Train Loss: 1.8325  Train Acc: 44.07%  Test Loss: 1.7417  Test Acc: 45.65%
✅ New best (45.65%), checkpoint saved.
                                                                                          
Epoch 23/400  Train Loss: 1.7882  Train Acc: 45.62%  Test Loss: 1.6756  Test Acc: 47.10%
✅ New best (47.10%), checkpoint saved.
                                                                                          
Epoch 24/400  Train Loss: 1.7359  Train Acc: 48.11%  Test Loss: 1.6422  Test Acc: 47.83%
✅ New best (47.83%), checkpoint saved.
                                                                                          
Epoch 25/400  Train Loss: 1.6932  Train Acc: 50.26%  Test Loss: 1.5795  Test Acc: 46.74%
                                                                                          
Epoch 26/400  Train Loss: 1.6606  Train Acc: 50.77%  Test Loss: 1.5371  Test Acc: 49.64%
✅ New best (49.64%), checkpoint saved.
                                                                                          
Epoch 27/400  Train Loss: 1.6395  Train Acc: 53.78%  Test Loss: 1.4848  Test Acc: 47.10%
                                                                                          
Epoch 28/400  Train Loss: 1.5848  Train Acc: 55.07%  Test Loss: 1.4430  Test Acc: 51.09%
✅ New best (51.09%), checkpoint saved.
                                                                                          
Epoch 29/400  Train Loss: 1.5841  Train Acc: 53.87%  Test Loss: 1.3994  Test Acc: 51.09%
                                                                                          
Epoch 30/400  Train Loss: 1.5446  Train Acc: 57.22%  Test Loss: 1.3875  Test Acc: 50.72%
                                                                                          
Epoch 31/400  Train Loss: 1.5211  Train Acc: 58.85%  Test Loss: 1.3381  Test Acc: 50.00%
                                                                                          
Epoch 32/400  Train Loss: 1.4966  Train Acc: 59.62%  Test Loss: 1.3264  Test Acc: 54.35%
✅ New best (54.35%), checkpoint saved.
                                                                                          
Epoch 33/400  Train Loss: 1.4966  Train Acc: 60.22%  Test Loss: 1.3125  Test Acc: 54.71%
✅ New best (54.71%), checkpoint saved.
                                                                                          
Epoch 34/400  Train Loss: 1.4754  Train Acc: 61.25%  Test Loss: 1.2861  Test Acc: 52.17%
                                                                                          
Epoch 35/400  Train Loss: 1.4583  Train Acc: 61.43%  Test Loss: 1.2606  Test Acc: 53.99%
                                                                                          
Epoch 36/400  Train Loss: 1.4595  Train Acc: 63.32%  Test Loss: 1.2703  Test Acc: 53.26%
                                                                                          
Epoch 37/400  Train Loss: 1.4387  Train Acc: 63.66%  Test Loss: 1.2543  Test Acc: 54.35%
                                                                                          
Epoch 38/400  Train Loss: 1.4399  Train Acc: 63.83%  Test Loss: 1.2439  Test Acc: 53.62%
                                                                                          
Epoch 39/400  Train Loss: 1.4273  Train Acc: 65.29%  Test Loss: 1.2450  Test Acc: 52.17%
                                                                                          
Epoch 40/400  Train Loss: 1.4169  Train Acc: 65.64%  Test Loss: 1.2748  Test Acc: 54.71%
                                                                                          
Epoch 41/400  Train Loss: 1.4332  Train Acc: 65.98%  Test Loss: 1.2389  Test Acc: 53.62%
                                                                                          
Epoch 42/400  Train Loss: 1.4235  Train Acc: 66.32%  Test Loss: 1.2381  Test Acc: 53.99%
                                                                                          
Epoch 43/400  Train Loss: 1.4137  Train Acc: 67.70%  Test Loss: 1.2374  Test Acc: 51.81%
                                                                                          
Epoch 44/400  Train Loss: 1.3873  Train Acc: 71.31%  Test Loss: 1.2137  Test Acc: 52.90%
                                                                                          
Epoch 45/400  Train Loss: 1.3874  Train Acc: 70.10%  Test Loss: 1.2080  Test Acc: 53.62%
                                                                                          
Epoch 46/400  Train Loss: 1.3836  Train Acc: 70.62%  Test Loss: 1.2747  Test Acc: 50.72%
                                                                                          
Epoch 47/400  Train Loss: 1.3593  Train Acc: 72.34%  Test Loss: 1.2986  Test Acc: 50.00%
                                                                                          
Epoch 48/400  Train Loss: 1.3732  Train Acc: 71.56%  Test Loss: 1.2360  Test Acc: 52.54%
                                                                                          
Epoch 49/400  Train Loss: 1.3704  Train Acc: 71.74%  Test Loss: 1.2575  Test Acc: 51.45%
                                                                                          
Epoch 50/400  Train Loss: 1.3652  Train Acc: 73.11%  Test Loss: 1.2444  Test Acc: 53.26%
                                                                                          
Epoch 51/400  Train Loss: 1.3431  Train Acc: 71.39%  Test Loss: 1.2463  Test Acc: 54.35%
                                                                                          
Epoch 52/400  Train Loss: 1.3455  Train Acc: 71.31%  Test Loss: 1.2701  Test Acc: 52.90%
                                                                                          
Epoch 53/400  Train Loss: 1.3467  Train Acc: 72.68%  Test Loss: 1.3035  Test Acc: 52.17%
                                                                                          
Epoch 54/400  Train Loss: 1.3425  Train Acc: 74.66%  Test Loss: 1.2695  Test Acc: 50.00%
                                                                                          

'''
