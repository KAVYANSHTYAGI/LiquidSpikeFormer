import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset2 import get_trial_dataloaders, raw_event_collate
from model2 import LiquidSpikeFormerMultiBranch  # <--- NEW multi-branch model!
from loss2 import HybridSpikingLoss, SupervisedContrastiveLoss
from optimizer import get_optimizer, get_scheduler, clip_gradients
from augmentation import (
    Compose,
    NormalizeTimestamps,
    RandomTemporalCrop,
    RandomSpatialJitter,
    RandomPolarityFlip,
    AddEventNoise,
)

# --- Configuration ---
ROOT_DIR       = "/home/kavyansh/MySSD/research project/SNN/dataset/DVS  Gesture dataset/DvsGesture"
BATCH_SIZE     = 8
NUM_WORKERS    = 0     # set 0 for debugging
PIN_MEMORY     = True
NUM_EPOCHS     = 400
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Transform pipeline (NO binning here, done in model) ---
transform = Compose([
    NormalizeTimestamps(),
    #RandomTemporalCrop(0.8),
    #RandomSpatialJitter(max_jitter=1, height=128, width=128),
    #RandomPolarityFlip(flip_prob=0.05),
    #AddEventNoise(spatial_sigma=0.5, temporal_sigma=0.01, height=128, width=128),
    # No ToBinnedTensor!
])

# --- DataLoaders ---
train_loader, test_loader = get_trial_dataloaders(
    root_dir=ROOT_DIR,
    transform=transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=raw_event_collate
)

# --- Model, Losses, Optimizer, Scheduler ---
model = LiquidSpikeFormerMultiBranch(
    in_channels=4,
    embed_dim=128,
    nhead=4,
    num_classes=11,
    fine_bins=20,
    coarse_bins=5,
    height=128,
    width=128,
    poisson=False,
    learnable_bins=False,
    smooth_kernel_size=5,
    dropout=0.1
).to(DEVICE)

spike_loss = HybridSpikingLoss(
    lambda_s=1.0,
    lambda_m=0.5,
    lambda_t=0.5,
    lambda_a=0.1,
    target_sparsity=0.1,
    threshold=0.5
)
ce_loss = nn.CrossEntropyLoss()
supcon_loss = SupervisedContrastiveLoss(temperature=0.07)
LAMBDA_CONTRAST = 0.1

optimizer = get_optimizer(
    model,
    optimizer_name="AdamW",
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = get_scheduler(
    optimizer,
    scheduler_name="WarmupCosine",
    total_steps=len(train_loader) * NUM_EPOCHS,
    warmup_steps=500
)

best_acc = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    train_correct, train_total = 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS:02d}", leave=False)
    for batch in pbar:
        events = [ev.to(DEVICE) for ev in batch["events"]]
   # [B, N, 4]
        labels = batch["label"].to(DEVICE)    # [B]

        optimizer.zero_grad()
        out = model(events)
        # main spiking loss
        loss_main = spike_loss(
            out["logits"],
            labels,
            spikes=out["spikes"],
            membrane=out["membrane"],
            threshold_param=out["threshold"],
        )
        # early heads CE
        loss_emid = ce_loss(out["early_mid"], labels)
        loss_efin = ce_loss(out["early_final"], labels)

        # SUPCON loss
        feats = out["feats"]  # [B, T, D]
        embeddings = feats.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        loss_supcon = supcon_loss(embeddings, labels)

        # Combine losses
        loss = (
            0.60 * loss_main +
            0.15 * loss_emid +
            0.25 * loss_efin +
            LAMBDA_CONTRAST * loss_supcon
        )
        loss.backward()
        clip_gradients(model, max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(events)

        _, preds = out["logits"].max(1)
        train_correct += preds.eq(labels).sum().item()
        train_total += labels.size(0)
        avg_loss = running_loss / ((pbar.n + 1) * BATCH_SIZE)
        pbar.set_postfix(train_loss=avg_loss, acc=100*train_correct/train_total)

    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * train_correct / train_total

    # --- Evaluation ---
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in test_loader:
            events = [ev.to(DEVICE) for ev in batch["events"]]
            labels = batch["label"].to(DEVICE)
            out = model(events)
            loss_val = ce_loss(out["logits"], labels)
            test_loss += loss_val.item() * len(events)
            preds = out["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss /= total
    test_acc = 100.0 * correct / total

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS:02d}  "
        f"Train Loss: {epoch_loss:.4f}  "
        f"Train Acc: {train_acc:.2f}%  "
        f"Test Loss: {test_loss:.4f}  "
        f"Test Acc: {test_acc:.2f}%"
    )

    if test_acc > best_acc:
        best_acc = test_acc
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "best_acc": best_acc,
        }
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"best_epoch{epoch}.pth"))
        print(f"✅ New best ({best_acc:.2f}%), checkpoint saved.")

print(f"\n🎉 Training complete. Best Accuracy: {best_acc:.2f}%")
