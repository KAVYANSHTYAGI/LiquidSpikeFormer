import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ←—— now import your real Compose from augmentation.py
from augmentation import Compose

# ←—— your loader that takes (file, start_usec, end_usec)
from event_loader import load_events

TRAIN_SPLIT = 'train_gestures.csv'
TEST_SPLIT  = 'test_gestures.csv'

def load_gesture_mapping(root_dir):
    mapping_path = os.path.join(root_dir, 'gesture_mapping.csv')
    df = pd.read_csv(mapping_path)
    # map id→name and name→id
    id_to_name    = {str(r['label']): r['action'] for _, r in df.iterrows()}
    name_to_label = {r['action']:    r['label']  for _, r in df.iterrows()}
    return id_to_name, name_to_label

def collect_samples(root_dir, gesture_dict, split_file):
    id_to_name, name_to_label = gesture_dict
    split_path = os.path.join(root_dir, split_file)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    valid_entries = pd.read_csv(split_path).iloc[:,0]
    samples = []
    unknown = set()

    for label_file in valid_entries:
        if not label_file.endswith('_labels.csv'):
            continue

        label_path = os.path.join(root_dir, label_file)
        event_file = label_file.replace('_labels.csv', '.aedat')
        event_path = os.path.join(root_dir, event_file)
        if not os.path.exists(event_path):
            raise FileNotFoundError(f"Missing data file: {event_path}")

        df = pd.read_csv(label_path)
        for _, row in df.iterrows():
            aid = str(row['class'])
            if aid not in id_to_name:
                unknown.add(aid)
                continue
            action_name = id_to_name[aid]
            if action_name not in name_to_label:
                unknown.add(action_name)
                continue

            # **zero-based** label for CrossEntropyLoss
            lbl = name_to_label[action_name] - 1

            samples.append({
                'event_file': event_path,
                'start_usec': row['startTime_usec'],
                'end_usec':   row['endTime_usec'],
                'label':      lbl
            })

    if unknown:
        print(f"⚠️ Unknown actions in split: {unknown}")
    return samples

class GestureTrialDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # load_events now takes (file, start_usec, end_usec)
        events = load_events(s['event_file'], s['start_usec'], s['end_usec'])
        sample = {
            'events': events,
            'label':  s['label']
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_trial_dataloaders(root_dir, transform=None,
                          batch_size=32, num_workers=4, pin_memory=True):
    print(f"\n📂 Loading dataset from: {root_dir}")
    gesture_dict   = load_gesture_mapping(root_dir)
    train_samples  = collect_samples(root_dir, gesture_dict, TRAIN_SPLIT)
    test_samples   = collect_samples(root_dir, gesture_dict, TEST_SPLIT)

    print(f"✅ Found {len(train_samples)} training samples, {len(test_samples)} test samples")
    if not train_samples or not test_samples:
        raise ValueError("❌ No samples found in training or test set. Check your CSV splits.")

    train_ds  = GestureTrialDataset(train_samples,  transform)
    test_ds   = GestureTrialDataset(test_samples,   transform)
    train_dl  = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=pin_memory)
    test_dl   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, test_dl
