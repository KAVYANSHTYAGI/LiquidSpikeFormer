import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

def read_aedat_file(filepath: str) -> np.ndarray:
    cache = read_aedat_file.__dict__.setdefault('cache', {})
    if filepath in cache:
        return cache[filepath]
    with open(filepath, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(pos)
                break
        data = np.fromfile(f, dtype=np.uint32)
    raw = data[0::2]
    ts = data[1::2]
    x = (raw >> 17) & 0x1FFF
    y = (raw >> 2) & 0x1FFF
    pol = (raw >> 1) & 0x1
    events = np.stack([x, y, ts, pol], axis=1)
    cache[filepath] = events
    return events

class DVSGestureTrialLoader:
    def __init__(self, root_dir: str, split: str):
        assert split in ['train', 'test']
        self.root_dir = root_dir
        split_file = os.path.join(root_dir, f"trials_to_{split}.txt")
        mapping_path = os.path.join(root_dir, 'gesture_mapping.csv')
        self.mapping = pd.read_csv(mapping_path)
        self.action2label = dict(zip(self.mapping['action'], self.mapping['label']))

        with open(split_file, 'r') as f:
            self.files = [line.strip() for line in f if line.strip().endswith(".aedat")]

        self.samples = []
        for fname in self.files:
            base = fname.replace(".aedat", "")
            label_path = os.path.join(root_dir, f"{base}_labels.csv")
            if not os.path.exists(label_path):
                continue
            label_df = pd.read_csv(label_path)
            for _, row in label_df.iterrows():
                action = row['class']
                if action not in self.action2label:
                    continue
                self.samples.append({
                    'path': os.path.join(root_dir, fname),
                    'start': row['startTime_usec'],
                    'end':   row['endTime_usec'],
                    'label': self.action2label[action]
                })

    def get_samples(self):
        return self.samples

class GestureTrialDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        events = read_aedat_file(sample_meta['path'])
        mask = (events[:, 2] >= sample_meta['start']) & (events[:, 2] < sample_meta['end'])
        trial_events = events[mask]
        sample = {'events': trial_events, 'label': sample_meta['label']}
        if self.transform:
            sample = self.transform(sample)
        if isinstance(sample['events'], np.ndarray):
            sample['events'] = torch.from_numpy(sample['events'].astype(np.float32))
        return sample

def get_trial_dataloaders(root_dir, transform=None,
                          batch_size=32, num_workers=4, pin_memory=True):
    train_loader = DVSGestureTrialLoader(root_dir, split='train')
    test_loader = DVSGestureTrialLoader(root_dir, split='test')
    train_ds = GestureTrialDataset(train_loader.get_samples(), transform)
    test_ds = GestureTrialDataset(test_loader.get_samples(), transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory,
                          collate_fn=default_collate)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory,
                         collate_fn=default_collate)
    return train_dl, test_dl
