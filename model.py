import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------
# Conv-SNN Block: Spatiotemporal (revised)
# ----------------------------------
class ConvSNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, temporal_kernel=3, dropout=0.2):
        super().__init__()
        # 2D convolution over (C_in → C_out) on each frame
        self.spatial_conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution over time (on the pooled spatial features)
        # in_channels = out_ch, out_channels = out_ch
        self.temporal_conv = nn.Conv1d(out_ch, out_ch, kernel_size=temporal_kernel,
                                       padding=temporal_kernel // 2, bias=False)

    def forward(self, x):
        """
        x: [B, T, C_in, H, W]
        returns → [B, T, out_ch]
        """
        B, T, C, H, W = x.shape

        # 1) Spatial conv on each time slice
        x_spat = x.view(B * T, C, H, W)                  # [B*T, C, H, W]
        x_spat = self.spatial_conv(x_spat)               # [B*T, out_ch, H, W]
        x_spat = self.bn(x_spat)
        x_spat = self.relu(x_spat)
        x_spat = self.dropout(x_spat)
        x_spat = x_spat.view(B, T, -1, H, W)              # [B, T, out_ch, H, W]

        # 2) Global average‐pool over spatial dims → [B, T, out_ch]
        x_pool = x_spat.mean(dim=[3, 4])                  # (average over H & W)

        # 3) Temporal conv across T–dimension:
        #    input shape for Conv1d: [B, out_ch, T]
        x_t = x_pool.permute(0, 2, 1)                     # [B, out_ch, T]
        x_t = self.temporal_conv(x_t)                     # [B, out_ch, T]
        out = x_t.permute(0, 2, 1)                        # [B, T, out_ch]
        return out


# ----------------------------------
# TRF + FLI Gate (unchanged)
# ----------------------------------
class TRF_FLIGate(nn.Module):
    def __init__(self, in_dim, kernel=3):
        super().__init__()
        self.trf = nn.Conv1d(in_dim, in_dim, kernel_size=kernel,
                             padding=kernel // 2, groups=in_dim, bias=False)
        self.fli = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_dim // 4, in_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        xp = x.permute(0, 2, 1)           # [B, D, T]
        x_f = self.trf(xp)                # [B, D, T]
        gate = self.fli(x_f)              # [B, D, T]
        out = (x_f * gate).permute(0, 2, 1)  # [B, T, D]
        return out


# ----------------------------------
# Spike Encoder (unchanged)
# ----------------------------------
class SpikeEncoder(nn.Module):
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
        if events.dim() == 3 and events.size(-1) == 4:
            B, N, _ = events.shape
            xs = events[:, :, 0].long().clamp(0, self.height - 1)
            ys = events[:, :, 1].long().clamp(0, self.width - 1)
            ts = events[:, :, 2].clamp(0, 1)
            bin_edges = self.bin_edges.to(ts.device)
            bin_idx = torch.bucketize(ts, bin_edges, right=True)
            pixel_idx = xs * self.width + ys
            counts = torch.zeros(B, self.num_bins, self.P, device=device)
            flat = counts.view(-1)
            idx = (bin_idx * self.P + pixel_idx).view(-1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N).reshape(-1)
            offsets = idx + batch_idx * (self.num_bins * self.P)
            flat.index_add_(0, offsets, torch.ones_like(offsets, device=device, dtype=flat.dtype))
            spikes = flat.view(B, self.num_bins, self.P)
        else:
            spikes = events.float()

        if self.poisson:
            maxc = spikes.amax(dim=2, keepdim=True).clamp(min=1.0)
            rates = spikes / maxc
            spikes = torch.bernoulli(rates)

        sp = spikes.permute(0, 2, 1)           # [B, P, num_bins]
        sp = self.smooth_conv(sp)              # smoothing
        spikes = sp.permute(0, 2, 1)           # [B, num_bins, P]
        spikes = self.pixel_norm(spikes)       # per‐pixel normalization
        spikes = self.global_norm(spikes)      # global normalization
        return spikes

    def event_frame(self, events: torch.Tensor):
        B, N, _ = events.shape
        xs = events[:, :, 0].long().clamp(0, self.height - 1)
        ys = events[:, :, 1].long().clamp(0, self.width - 1)
        frames = torch.zeros(B, self.height, self.width, device=events.device)
        for b in range(B):
            frames[b].index_put_((xs[b], ys[b]), torch.ones(N, device=events.device), accumulate=True)
        return frames.view(B, -1)


# ----------------------------------
# Patch Embedding (unchanged)
# ----------------------------------
class SpikingPatchEmbedding(nn.Module):
    def __init__(self, num_pixels: int, embed_dim: int, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(num_pixels, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, P = x.shape
        x = self.proj(x.reshape(B * T, P)).reshape(B, T, -1)
        return self.dropout(x)


# ----------------------------------
# Neuromodulation Gate (unchanged)
# ----------------------------------
class NeuromodulationGate(nn.Module):
    def __init__(self, dim, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, global_activity):
        x = global_activity.view(-1, 1).float()
        return self.fc(x).mean(dim=0)


# ----------------------------------
# Liquid Time-Constant Block (unchanged)
# ----------------------------------
class LiquidTimeConstantBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2, use_neuromodulation: bool = False):
        super().__init__()
        self.tau_fc = nn.Linear(dim * 2, dim)
        self.mem_fc = nn.Linear(dim, dim)
        self.log_thresh = nn.Parameter(torch.zeros(dim))
        self.dropout = nn.Dropout(dropout)
        self.use_neuromodulation = use_neuromodulation
        if use_neuromodulation:
            self.neuromod_gate = NeuromodulationGate(dim)

    def forward(self, x: torch.Tensor, global_activity=None):
        B, T, D = x.shape
        device = x.device
        v = torch.zeros(B, D, device=device)
        tau = torch.ones(B, D, device=device)
        spikes = []

        if self.use_neuromodulation and global_activity is not None:
            adaptive_thresh = torch.sigmoid(self.log_thresh + self.neuromod_gate(global_activity).to(device))
        else:
            adaptive_thresh = torch.sigmoid(self.log_thresh)

        for t in range(T):
            xt = self.dropout(x[:, t, :])
            tau = torch.sigmoid(self.tau_fc(torch.cat([xt, tau], dim=-1)))
            alpha = torch.exp(-1.0 / (tau + 1e-6))
            v = alpha * v + (1 - alpha) * self.mem_fc(xt)
            s = (v >= adaptive_thresh).float()
            v = v * (1 - s)
            spikes.append(s)

        spikes = torch.stack(spikes, dim=1)
        return spikes, tau, v, adaptive_thresh


# ----------------------------------
# Transformer Block (unchanged)
# ----------------------------------
class SpikingTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


# ----------------------------------
# Output Head (unchanged)
# ----------------------------------
class OutputHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x.mean(dim=1)))


# ----------------------------------
# Full Model: Richer SNN + Transformer (updated to match new ConvSNNBlock)
# ----------------------------------
class LiquidSpikeFormerRichSpatialTemporal(nn.Module):
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
        dropout: float = 0.2
    ):
        super().__init__()

        # --- SpikeEncoders at two scales ---
        self.encoder_fine = SpikeEncoder(fine_bins, height, width, poisson, learnable_bins, smooth_kernel_size)
        self.encoder_coarse = SpikeEncoder(coarse_bins, height, width, poisson, learnable_bins, smooth_kernel_size)
        self.height = height
        self.width = width
        self.P = height * width

        # Project raw event‐frame (flattened H×W) to embed_dim
        self.event_frame_proj = nn.Linear(self.P, embed_dim)

        # --- Two sequential Conv-SNN Blocks ---
        self.conv_snn1 = ConvSNNBlock(in_ch=1, out_ch=4, kernel_size=3, padding=1, temporal_kernel=3, dropout=dropout)
        self.conv_snn2 = ConvSNNBlock(in_ch=4, out_ch=8, kernel_size=3, padding=1, temporal_kernel=3, dropout=dropout)

        # Project final conv_snn2 output (8 features) → embed_dim
        self.conv_snn_proj = nn.Linear(8, embed_dim)

        # Patch embeddings for fine & coarse spike outputs
        self.patch_embed_fine = SpikingPatchEmbedding(self.P, embed_dim, dropout)
        self.patch_embed_coarse = SpikingPatchEmbedding(self.P, embed_dim, dropout)

        # Merge (4 streams → embed_dim)
        self.merge_proj = nn.Linear(embed_dim * 4, embed_dim)

        # Two Liquid Time‐Constant Blocks
        self.liquid_block1 = LiquidTimeConstantBlock(embed_dim, dropout, use_neuromodulation=False)
        self.liquid_block2 = LiquidTimeConstantBlock(embed_dim, dropout, use_neuromodulation=True)

        # Two Transformer Encoder blocks
        self.transformer1 = SpikingTransformerBlock(d_model=embed_dim, nhead=nhead, dropout=dropout)
        self.transformer2 = SpikingTransformerBlock(d_model=embed_dim, nhead=nhead, dropout=dropout)

        # Output heads
        self.head = OutputHead(d_model=embed_dim, num_classes=num_classes, dropout=dropout)
        self.early_head_mid = nn.Linear(embed_dim, num_classes)
        self.early_head_final = nn.Linear(embed_dim, num_classes)

    def forward(self, events: torch.Tensor):
        # events: List of length B, each is [N, 4]
        # Encode fine/coarse spike streams
        x_fine_list = [self.encoder_fine(ev.unsqueeze(0)) for ev in events]    # each → [1, T_fine, P]
        x_coarse_list = [self.encoder_coarse(ev.unsqueeze(0)) for ev in events]  # each → [1, T_coarse, P]
        x_fine = torch.cat(x_fine_list, dim=0)      # [B, T_fine, P]
        x_coarse = torch.cat(x_coarse_list, dim=0)  # [B, T_coarse, P]

        # Upsample coarse → fine length
        T_fine = x_fine.shape[1]
        x_coarse_up = F.interpolate(x_coarse.permute(0, 2, 1), size=T_fine, mode="linear", align_corners=True)
        x_coarse_up = x_coarse_up.permute(0, 2, 1)  # [B, T_fine, P]

        # Raw “event‐frame” (binary counts) for each sample
        frame_list = [self.encoder_fine.event_frame(ev.unsqueeze(0)) for ev in events]
        frame = torch.cat(frame_list, dim=0)        # [B, P]
        frame_proj = self.event_frame_proj(frame).unsqueeze(1).repeat(1, T_fine, 1)  # [B, T_fine, embed_dim]

        # --- ConvSNN processing on rasterized frames ---
        # 1) Build per-bin binary frames: [B, T, H, W] → add channel dim
        batch_rasters = []
        for ev in events:
            # For each sample, rasterize into T_fine frames of size H×W
            frame_stack = []
            N = ev.shape[0]
            for t in range(T_fine):
                t_start = t / T_fine
                t_end = (t + 1) / T_fine
                mask = (ev[:, 2] >= t_start) & (ev[:, 2] < t_end)
                f = torch.zeros(self.height, self.width, device=ev.device)
                if mask.sum() > 0:
                    xs = ev[mask, 0].long().clamp(0, self.height - 1)
                    ys = ev[mask, 1].long().clamp(0, self.width - 1)
                    f.index_put_((xs, ys), torch.ones(len(xs), device=ev.device), accumulate=True)
                frame_stack.append(f)
            rasters = torch.stack(frame_stack, dim=0)  # [T_fine, H, W]
            batch_rasters.append(rasters)

        frames_tensor = torch.stack(batch_rasters, dim=0).unsqueeze(2)  # [B, T_fine, 1, H, W]

        # 2) conv_snn1: [B, T, 1, H, W] → [B, T, 4]
        conv1_out = self.conv_snn1(frames_tensor)       # [B, T_fine, 4]

        # 3) Reshape → add spatial channel back → [B, T, 4, H, W]
        #    (just tile each time‐feature across H×W so conv_snn2 can see “feature at each pixel”)
        B, T1, C1 = conv1_out.shape                    # C1 == 4
        # replicate each of the 4 feature values across H×W:
        conv1_tiled = conv1_out.view(B, T1, C1, 1, 1).repeat(1, 1, 1, self.height, self.width)
        # now → [B, T_fine, 4, H, W]

        # 4) conv_snn2: [B, T, 4, H, W] → [B, T, 8]
        conv2_out = self.conv_snn2(conv1_tiled)         # [B, T_fine, 8]

        # 5) Project those 8‐D features → embed_dim
        conv_feat_embed = self.conv_snn_proj(conv2_out)  # [B, T_fine, embed_dim]

        # --- Patch embeddings for fine/coarse spikes ---
        x_fine_embed = self.patch_embed_fine(x_fine)          # [B, T_fine, embed_dim]
        x_coarse_embed = self.patch_embed_coarse(x_coarse_up) # [B, T_fine, embed_dim]

        # --- Merge all four streams: [fine, coarse, raw‐frame, conv_feat] each embed_dim —
        feats = torch.cat([x_fine_embed, x_coarse_embed, frame_proj, conv_feat_embed], dim=-1)
        feats = self.merge_proj(feats)                         # [B, T_fine, embed_dim]

        # Early mid‐classification head
        early_mid = self.early_head_mid(feats.mean(dim=1))     # [B, num_classes]

        # --- Liquid SNN blocks ---
        spikes1, tau1, mem1, thr1 = self.liquid_block1(feats)  # spikes1: [B, T_fine, embed_dim]
        global_spike_activity = spikes1.mean(dim=(1, 2))       # [B]
        spikes2, tau2, mem2, thr2 = self.liquid_block2(spikes1, global_activity=global_spike_activity)

        # Mask out all‐zero timesteps
        mask = (spikes2.sum(dim=2) == 0)                       # [B, T_fine]

        # Two transformer encoders
        t1 = self.transformer1(spikes2, src_key_padding_mask=mask)  # [B, T_fine, embed_dim]
        t2 = self.transformer2(t1, src_key_padding_mask=mask)       # [B, T_fine, embed_dim]

        # Early final head
        early_final = self.early_head_final(t2.mean(dim=1))     # [B, num_classes]

        # Final head
        logits = self.head(t2)                                  # [B, num_classes]

        return {
            'logits': logits,
            'early_mid': early_mid,
            'early_final': early_final,
            'spikes': spikes2,
            'membrane': mem2,
            'threshold': thr2,
            'feats': t2
        }
