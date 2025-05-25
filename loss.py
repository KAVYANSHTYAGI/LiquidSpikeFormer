import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikeSparsityLoss(nn.Module):
    """
    Penalizes deviation from target spike rate to enforce sparsity.
    loss = (mean spike rate - target_sparsity)^2
    """
    def __init__(self, target_sparsity: float = 0.1):
        super(SpikeSparsityLoss, self).__init__()
        self.target = target_sparsity

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        rate = spikes.mean()
        return (rate - self.target) ** 2

class MembraneRegularizationLoss(nn.Module):
    """
    Encourages membrane potentials to stay around a threshold for stability.
    loss = MSE(v, threshold)
    """
    def __init__(self, threshold: float = 0.5, weight: float = 1.0):
        super(MembraneRegularizationLoss, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.weight * F.mse_loss(v, torch.full_like(v, self.threshold))

class TemporalConsistencyLoss(nn.Module):
    """
    Penalizes large changes in spike patterns between consecutive timesteps.
    loss = mean |s_t - s_{t-1}| over batch and time
    """
    def __init__(self, weight: float = 1.0):
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        diffs = torch.abs(spikes[:, 1:] - spikes[:, :-1])
        return self.weight * diffs.mean()

class AdaptiveThresholdRegularization(nn.Module):
    """
    Encourages learned thresholds to remain within a stable range.
    loss = variance of thresholds (unbiased=False to avoid NaNs for scalar thresholds)
    """
    def __init__(self, weight: float = 0.1):
        super(AdaptiveThresholdRegularization, self).__init__()
        self.weight = weight

    def forward(self, threshold_param: nn.Parameter) -> torch.Tensor:
        # Use unbiased=False so variance of a single element is zero, not NaN
        return self.weight * torch.var(threshold_param, unbiased=False)

class HybridSpikingLoss(nn.Module):
    """
    Combines classification loss with modular spiking losses.
    total_loss = CE
                 + lambda_s * sparsity_loss
                 + lambda_m * membrane_loss
                 + lambda_t * temporal_loss
                 + lambda_a * threshold_reg
    """
    def __init__(
        self,
        lambda_s: float = 1.0,
        lambda_m: float = 0.5,
        lambda_t: float = 0.5,
        lambda_a: float = 0.1,
        target_sparsity: float = 0.1,
        threshold: float = 0.5
    ):
        super(HybridSpikingLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.sparsity_loss = SpikeSparsityLoss(target_sparsity)
        self.membrane_loss = MembraneRegularizationLoss(threshold)
        self.temporal_loss = TemporalConsistencyLoss()
        self.threshold_reg = AdaptiveThresholdRegularization()
        self.lambda_s = lambda_s
        self.lambda_m = lambda_m
        self.lambda_t = lambda_t
        self.lambda_a = lambda_a

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                spikes: torch.Tensor = None,
                membrane: torch.Tensor = None,
                threshold_param: nn.Parameter = None) -> torch.Tensor:
        loss = self.ce(logits, labels)
        if spikes is not None:
            loss = loss + self.lambda_s * self.sparsity_loss(spikes)
            loss = loss + self.lambda_t * self.temporal_loss(spikes)
        if membrane is not None:
            loss = loss + self.lambda_m * self.membrane_loss(membrane)
        if threshold_param is not None:
            loss = loss + self.lambda_a * self.threshold_reg(threshold_param)
        return loss
