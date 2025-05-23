import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    OneCycleLR,
    LambdaLR,
    ReduceLROnPlateau
)
import os

# ----------------------------------
# Optimizer Utilities
# ----------------------------------
def get_param_groups(
    model: nn.Module,
    weight_decay: float = 1e-4,
    no_decay_keywords: list = ['threshold']
) -> list:
    """
    Return parameter groups separating weights for decay and no_decay.
    Parameters containing any keyword in no_decay_keywords get zero weight_decay.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = 'AdamW',
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    no_decay_keywords: list = ['threshold']
) -> optim.Optimizer:
    """
    Create optimizer with parameter grouping.
    Supports: 'AdamW', 'Adam', 'SGD', 'RMSprop'.
    """
    param_groups = get_param_groups(model, weight_decay, no_decay_keywords)
    optim_map = {
        'AdamW': lambda: optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps),
        'Adam':  lambda: optim.Adam(param_groups,  lr=lr, betas=betas, eps=eps),
        'SGD':   lambda: optim.SGD(param_groups,   lr=lr, momentum=momentum),
        'RMSprop': lambda: optim.RMSprop(param_groups, lr=lr, momentum=momentum)
    }
    if optimizer_name not in optim_map:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optim_map[optimizer_name]()


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'WarmupCosine',
    T_max: int = 100,
    eta_min: float = 0.0,
    step_size: int = 30,
    gamma: float = 0.1,
    total_steps: int = 1000,
    pct_start: float = 0.1,
    warmup_steps: int = 100,
    plateau_mode: str = 'min'
) -> object:
    """
    Create learning rate scheduler.
    scheduler_name options:
      - 'WarmupCosine': linear warmup then cosine annealing
      - 'CosineRestart': CosineAnnealingWarmRestarts
      - 'Cosine': CosineAnnealingLR
      - 'OneCycle': OneCycleLR
      - 'StepLR': StepLR
      - 'Plateau': ReduceLROnPlateau
      - 'LambdaWarmup': linear warmup then constant
    """
    if scheduler_name == 'WarmupCosine':
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == 'CosineRestart':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_max, T_mult=1, eta_min=eta_min)
    elif scheduler_name == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == 'OneCycle':
        scheduler = OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'],
                               total_steps=total_steps, pct_start=pct_start)
    elif scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'Plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode=plateau_mode, factor=gamma, patience=step_size)
    elif scheduler_name == 'LambdaWarmup':
        def warmup_fn(step):
            return float(step) / float(max(1, warmup_steps)) if step < warmup_steps else 1.0
        scheduler = LambdaLR(optimizer, warmup_fn)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    return scheduler


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> None:
    """
    Clips gradients by global norm.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)


# ----------------------------------
# Callback System
# ----------------------------------
class Callback:
    """
    Base class for custom callbacks.
    """
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_batch_begin(self, batch, logs=None): pass
    def on_batch_end(self, batch, logs=None): pass


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.
    """
    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0.0, patience: int = 5, mode: str = 'min'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.best is None:
            self.best = current
        improvement = (current < self.best - self.min_delta) if self.mode == 'min' else (current > self.best + self.min_delta)
        if improvement:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True


class ModelCheckpoint(Callback):
    """
    Save model weights when monitored metric improves.
    """
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.best is None or (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.best = current
            torch.save(logs.get('model_state_dict'), self.filepath)


class CallbackHandler:
    """
    Handles the invocation of callbacks during training.
    """
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def set_model(self, model):
        self.model = model

    def on_train_begin(self):
        for cb in self.callbacks:
            cb.on_train_begin()

    def on_train_end(self):
        for cb in self.callbacks:
            cb.on_train_end()

    def on_epoch_begin(self, epoch):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, logs=None):
        for cb in self.callbacks:
            # inject model state if checkpoint
            if hasattr(cb, 'filepath'):
                logs = logs or {}
                logs['model_state_dict'] = self.model.state_dict()
            cb.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch):
        for cb in self.callbacks:
            cb.on_batch_begin(batch)

    def on_batch_end(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)



# ----------------------------------
# Additional Callbacks
# ----------------------------------
class LRSchedulerCallback(Callback):
    """
    Steps the learning rate scheduler on batch or epoch.
    """
    def __init__(self, scheduler, step_on: str = 'batch'):
        super().__init__()
        self.scheduler = scheduler
        self.step_on = step_on  # 'batch' or 'epoch'

    def on_batch_end(self, batch, logs=None):
        if self.step_on == 'batch':
            self.scheduler.step()

    def on_epoch_end(self, epoch, logs=None):
        if self.step_on == 'epoch':
            self.scheduler.step(logs.get('val_loss') if isinstance(self.scheduler, ReduceLROnPlateau) else None)

class CSVLogger(Callback):
    """
    Logs epoch metrics to a CSV file.
    """
    def __init__(self, filename: str, fields: list = None):
        super().__init__()
        self.filename = filename
        self.fields = fields
        self.file = None

    def on_train_begin(self, logs=None):
        self.file = open(self.filename, 'w')
        header = self.fields or ['epoch'] + list((logs or {}).keys())
        self.file.write(','.join(header) + '')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = [str(epoch)] + [str(logs.get(field, '')) for field in (self.fields or logs.keys())]
        self.file.write(','.join(row) + '')
        self.file.flush()

    def on_train_end(self, logs=None):
        if self.file:
            self.file.close()

class GradientMonitor(Callback):
    """
    Logs gradient norms to specified logger.
    """
    def __init__(self, model: nn.Module, logger=print):
        super().__init__()
        self.model = model
        self.logger = logger

    def on_batch_end(self, batch, logs=None):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.logger(f"Gradient Norm: {total_norm:.4f}")

class ProgressCallback(Callback):
    """
    Displays training progress using tqdm.
    """
    def __init__(self, total_batches: int):
        super().__init__()
        from tqdm import tqdm
        self.total_batches = total_batches
        self.pbar = None

    def on_epoch_begin(self, epoch, logs=None):
        from tqdm import tqdm
        self.pbar = tqdm(total=self.total_batches, desc=f"Epoch {epoch}")

    def on_batch_end(self, batch, logs=None):
        if self.pbar:
            self.pbar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            self.pbar.close()

# --- End of optimizer.py ---
