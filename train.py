import os
import torch
from torch.utils.data import DataLoader
import copy
import numpy as np
# Import your modules
from augmentation import Compose, RandomDropEvents, RandomSpatialJitter, AddEventNoise, RandomPolarityFlip, RobustRandomTemporalCrop, RandomHorizontalFlip, RandomVerticalFlip, NormalizeTimestamps
from dataset2 import get_trial_dataloaders, raw_event_collate
from loss2 import HybridSpikingLoss
from optimizer import get_optimizer, get_scheduler, EarlyStopping
from model2 import LiquidSpikeFormerMultiBranch

# --------- Config ---------
root_dir = "/home/kavyansh/MySSD/research project/SNN/dataset/DVS  Gesture dataset/DvsGesture"
batch_size = 16
num_classes = 11   # DVS Gesture (change if needed)
height = 128
width = 128
fine_bins = 10
coarse_bins = 5
embed_dim = 128
nhead = 4
max_epochs = 400
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout_grid = [0.2, 0.3, 0.4]
lambda_s_grid = [0.05, 0.1, 0.2]
lambda_m, lambda_t, lambda_a = 0.1, 0.1, 0.1
weight_decay = 1e-3
learning_rate = 1e-3
patience = 15

# --------- Augmentation ---------
train_transform = Compose([
    NormalizeTimestamps(),
    RobustRandomTemporalCrop(0.8, min_events=10),
    RandomDropEvents(0.1),
    RandomSpatialJitter(2),
    AddEventNoise(1.0, 0.05),
    RandomPolarityFlip(0.1),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
])

# --------- Data ---------
train_dl, test_dl = get_trial_dataloaders(
    root_dir,
    transform=train_transform,
    batch_size=batch_size,
    num_workers=22,
    pin_memory=True,
    collate_fn=raw_event_collate
)

# For validation/testing: only normalization (if needed)
val_dl, _ = get_trial_dataloaders(
    root_dir,
    transform=None,
    batch_size=batch_size,
    num_workers=22,
    pin_memory=True,
    collate_fn=raw_event_collate
)

# --------- Grid Search Training ---------
results = []

for dropout in dropout_grid:
    for lambda_s in lambda_s_grid:
        print(f"\n===== Training with dropout={dropout}, lambda_s={lambda_s} =====\n")

        # Model
        model = LiquidSpikeFormerMultiBranch(
            in_channels=4,
            embed_dim=embed_dim,
            nhead=nhead,
            num_classes=num_classes,
            fine_bins=fine_bins,
            coarse_bins=coarse_bins,
            height=height,
            width=width,
            dropout=dropout
        ).to(device)

        # Loss
        loss_fn = HybridSpikingLoss(
            lambda_s=lambda_s,
            lambda_m=lambda_m,
            lambda_t=lambda_t,
            lambda_a=lambda_a,
            target_sparsity=0.1,
            threshold=0.5
        )

        # Optimizer and scheduler
        optimizer = get_optimizer(model, optimizer_name='AdamW', lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_scheduler(optimizer, scheduler_name='WarmupCosine', total_steps=max_epochs*len(train_dl), warmup_steps=5*len(train_dl))
        early_stopper = EarlyStopping(monitor='val_loss', patience=patience, mode='min')

        best_test_acc = 0
        best_model = None

        for epoch in range(1, max_epochs+1):
            # ------ Train ------
            model.train()
            total_loss, correct, total = 0, 0, 0
            for batch in train_dl:
                events = batch['events']
                labels = batch['label'].to(device)

                if isinstance(events, list):
                    # move each element of list to CUDA (if your DataLoader returns a list of tensors)
                    events = [ev.to(device) for ev in events]
                else:
                    events = events.to(device)

                optimizer.zero_grad()
                out = model(events)
                logits = out['logits']
                loss = loss_fn(logits, labels, out['spikes'], out['membrane'], out['threshold'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
            scheduler.step()
            train_acc = correct / total
            train_loss = total_loss / total

            # ------ Validate ------
            model.eval()
            with torch.no_grad():
                val_loss, val_correct, val_total = 0, 0, 0
                for batch in test_dl:
                    events = batch['events']
                    labels = batch['label'].to(device)

                    if isinstance(events, list):
                        # move each element of list to CUDA (if your DataLoader returns a list of tensors)
                        events = [ev.to(device) for ev in events]
                    else:
                        events = events.to(device) 
                                         
                    out = model(events)
                    logits = out['logits']
                    loss = loss_fn(logits, labels, out['spikes'], out['membrane'], out['threshold'])
                    val_loss += loss.item() * labels.size(0)
                    pred = logits.argmax(1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
                val_acc = val_correct / val_total
                val_loss = val_loss / val_total

            print(f"Epoch {epoch:03d}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2%}  Test Loss: {val_loss:.4f}  Test Acc: {val_acc:.2%}")

            # Early stopping
            logs = {'val_loss': val_loss}
            early_stopper.on_epoch_end(epoch, logs)
            if val_acc > best_test_acc:
                best_test_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())
            if early_stopper.stop_training:
                print(f"Early stopping at epoch {epoch}. Best test acc: {best_test_acc:.2%}")
                break

        results.append({
            'dropout': dropout,
            'lambda_s': lambda_s,
            'best_test_acc': best_test_acc
        })
        # Save the best model for this combo (optional)
        torch.save(best_model, f"best_model_dropout{dropout}_lambdas{lambda_s}.pt")

print("\n===== Grid Search Results =====")
for r in results:
    print(f"Dropout: {r['dropout']}  Lambda_s: {r['lambda_s']}  Best Test Acc: {r['best_test_acc']:.2%}")
