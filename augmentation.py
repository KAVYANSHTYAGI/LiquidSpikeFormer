import random
import numpy as np
import torch

class RandomTemporalCrop:
    """
    Randomly crops the event sequence to a fraction of its original duration.
    """
    def __init__(self, crop_frac: float = 0.8):
        assert 0 < crop_frac <= 1.0, "crop_frac must be in (0,1]"
        self.crop_frac = crop_frac

    def __call__(self, sample: dict) -> dict:
        events = sample['events']  # [N,4]
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        t = events[:, 2]
        start = random.random() * (1 - self.crop_frac)
        end = start + self.crop_frac
        mask = (t >= start) & (t <= end)
        cropped = events[mask].copy()
        if cropped.shape[0] > 0:
            cropped[:, 2] = (cropped[:, 2] - start) / self.crop_frac
        sample['events'] = torch.from_numpy(cropped.astype(np.float32))
        return sample

class RandomDropEvents:
    """
    Randomly drops a fraction of events to simulate sensor noise or failure.
    """
    def __init__(self, drop_frac: float = 0.1):
        assert 0 <= drop_frac < 1.0, "drop_frac must be in [0,1)"
        self.drop_frac = drop_frac

    def __call__(self, sample: dict) -> dict:
        events = sample['events']
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        N = events.shape[0]
        keep = np.random.rand(N) > self.drop_frac
        dropped = events[keep]
        sample['events'] = torch.from_numpy(dropped.astype(np.float32))
        return sample

class RandomSpatialJitter:
    """
    Adds random jitter to spatial coordinates of events.
    """
    def __init__(self, max_jitter: int = 1, height: int = 128, width: int = 128):
        self.max_jitter = max_jitter
        self.height = height
        self.width = width

    def __call__(self, sample: dict) -> dict:
        events = sample['events']
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        jitter = np.random.randint(-self.max_jitter, self.max_jitter+1, size=(events.shape[0], 2))
        events[:, 0:2] += jitter
        events[:, 0] = np.clip(events[:, 0], 0, self.height-1)
        events[:, 1] = np.clip(events[:, 1], 0, self.width-1)
        sample['events'] = torch.from_numpy(events.astype(np.float32))
        return sample

class RandomPolarityFlip:
    """
    Randomly flips event polarity with a given probability.
    """
    def __init__(self, flip_prob: float = 0.05):
        assert 0 <= flip_prob <= 1.0, "flip_prob must be in [0,1]"
        self.flip_prob = flip_prob

    def __call__(self, sample: dict) -> dict:
        events = sample['events']
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        mask = np.random.rand(events.shape[0]) < self.flip_prob
        events[mask, 3] = 1 - events[mask, 3]
        sample['events'] = torch.from_numpy(events.astype(np.float32))
        return sample

class NormalizeTimestamps:
    """
    Ensures timestamps are normalized to [0,1] over the segment.
    """
    def __call__(self, sample: dict) -> dict:
        events = sample['events']
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        t = events[:, 2]
        if t.max() > t.min():
            events[:, 2] = (t - t.min()) / (t.max() - t.min())
        else:
            events[:, 2] = 0.0
        sample['events'] = torch.from_numpy(events.astype(np.float32))
        return sample

class TemporalScaling:
    """
    Randomly scales timestamps by a factor to simulate speed variations.
    """
    def __init__(self, scale_range=(0.8, 1.2)):
        assert scale_range[0] > 0, "scale factors must be positive"
        self.scale_range = scale_range

    def __call__(self, sample: dict) -> dict:
        events = sample['events']
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()
        factor = random.uniform(*self.scale_range)
        events[:, 2] = events[:, 2] * factor
        events[:, 2] = np.clip(events[:, 2], 0, 1)
        sample['events'] = torch.from_numpy(events.astype(np.float32))
        return sample

class RandomHorizontalFlip:
    """
    Flips x coordinate horizontally with 50% probability.
    """
    def __init__(self, width: int = 128, p: float = 0.5):
        self.width = width
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            events = sample['events']
            if isinstance(events, torch.Tensor):
                events = events.cpu().numpy()
            events[:, 0] = self.width - 1 - events[:, 0]
            sample['events'] = torch.from_numpy(events.astype(np.float32))
        return sample

class RandomVerticalFlip:
    """
    Flips y coordinate vertically with 50% probability.
    """
    def __init__(self, height: int = 128, p: float = 0.5):
        self.height = height
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            events = sample['events']
            if isinstance(events, torch.Tensor):
                events = events.cpu().numpy()
            events[:, 1] = self.height - 1 - events[:, 1]
            sample['events'] = torch.from_numpy(events.astype(np.float32))
        return sample


class AddEventNoise:
    """
    Adds random Gaussian noise to spatial and temporal values to simulate sensor jitter.
    """
    def __init__(
        self,
        spatial_sigma: float = 0.5,
        temporal_sigma: float = 0.01,
        height: int = 128,
        width: int = 128
    ):
        self.spatial_sigma = spatial_sigma
        self.temporal_sigma = temporal_sigma
        self.height = height
        self.width = width

    def __call__(self, sample: dict) -> dict:
        events = sample['events']
        # ensure numpy for noise addition
        if isinstance(events, torch.Tensor):
            events = events.cpu().numpy()

        # events assumed shape [N, ≥3]: columns [x, y, t, …]
        # spatial noise on x,y
        noise_xy = np.random.normal(
            loc=0.0,
            scale=self.spatial_sigma,
            size=(events.shape[0], 2)
        )
        # temporal noise on t
        noise_t = np.random.normal(
            loc=0.0,
            scale=self.temporal_sigma,
            size=(events.shape[0], 1)
        )

        # add noise
        events[:, 0:2] += noise_xy
        events[:, 2:3] += noise_t

        # clip spatial coords: x ∈ [0, width-1], y ∈ [0, height-1]
        events[:, 0] = np.clip(events[:, 0], 0, self.width - 1)
        events[:, 1] = np.clip(events[:, 1], 0, self.height - 1)

        # clip temporal values to [0,1]
        events[:, 2] = np.clip(events[:, 2], 0.0, 1.0)

        # convert back to tensor
        sample['events'] = torch.from_numpy(events.astype(np.float32))
        return sample

class ToBinnedTensor:
    """
    Bins raw events into a fixed tensor using a provided encoder.
    """
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, sample: dict) -> dict:
        events = sample['events'].unsqueeze(0)  # [1,N,4]
        binned = self.encoder(events)           # [1,T,P]
        sample['events'] = binned.squeeze(0)    # [T,P]
        return sample

class Compose:
    """
    Composes several transforms sequentially.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample
