"""
Landmark Detection Training Script - Local GPU
Adapted from Colab notebook with best practices
Uses real downloaded images from images_000/
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time
import os
from sklearn.model_selection import train_test_split
import json

# ============================================================
# CONFIG
# ============================================================
BATCH_SIZE = 16           # Smaller for RTX 2050 (4GB) with real images
NUM_EPOCHS = 20
LR = 0.0001               # Lower LR for fine-tuning pretrained model
NUM_CLASSES = 500
EARLY_STOP_PATIENCE = 7

# Paths
SCRIPT_DIR = Path(__file__).parent
IMAGE_DIR_PARENT = SCRIPT_DIR / "images_000" / "0" / "0"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
DATA_DIR = SCRIPT_DIR / "data"

# Image transforms (ImageNet normalization)
IMG_SIZE = 224
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# LOAD GROUND TRUTH (all train parts for full ID coverage)
# ============================================================
print("\n" + "="*60)
print("Loading dataset metadata...")
print("="*60)

# Load ALL CSV parts to find matching IDs
dfs = []
for i in range(10):
    df_part = pd.read_csv(SCRIPT_DIR / f"train_part_{i}.csv", usecols=['id', 'landmark_id'])
    dfs.append(df_part)
df = pd.concat(dfs, ignore_index=True)
print(f"Total in CSV: {len(df):,}")

# Scan downloaded images (use stem directly - filename IS the image ID)
DOWNLOADED = {}
for subdir in range(9):  # 0-8
    img_folder = IMAGE_DIR_PARENT / str(subdir)
    if img_folder.exists():
        for f in img_folder.glob("*.jpg"):
            DOWNLOADED[f.stem] = f  # Store by filename stem

print(f"Downloaded images: {len(DOWNLOADED):,}")

# Match downloaded images with CSV metadata
available = df[df['id'].isin(DOWNLOADED.keys())].copy()
print(f"Images with metadata: {len(available):,}")
print(f"Total unique landmarks: {available['landmark_id'].nunique():,}")

# ====================================================================
# SAMPLING STRATEGY: Pick top classes with sufficient samples
# ====================================================================
# The downloaded images are sparse - most classes have only 1 image
# For training, we need classes with 2+ images for stratified split

MIN_SAMPLES_PER_CLASS = 2
MAX_CLASSES = 100  # Limit to top 100 classes for meaningful training

class_counts = available['landmark_id'].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].head(MAX_CLASSES).index

available_sampled = available[available['landmark_id'].isin(valid_classes)].copy()
print(f"\nSampled dataset:")
print(f"  Classes selected (with >={MIN_SAMPLES_PER_CLASS} samples): {len(valid_classes)}")
print(f"  Images: {len(available_sampled):,}")
print(f"  Avg images/class: {len(available_sampled)/len(valid_classes):.1f}")

# Split into train/val
train_df, val_df = train_test_split(
    available_sampled,
    test_size=0.3,  # ~30% for validation (must be > number of classes for stratification)
    stratify=available_sampled['landmark_id'],
    random_state=42
)
print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}")

# ============================================================
# REAL IMAGE DATASET
# ============================================================
# Shared class mapping for both train and val datasets
all_lms = sorted(set(train_df['landmark_id'].unique()) | set(val_df['landmark_id'].unique()))
SHARED_LM2IDX = {lm: i for i, lm in enumerate(all_lms)}
NUM_CLASSES = len(all_lms)

print(f"Total unique classes: {NUM_CLASSES}")

class LandmarkDataset(Dataset):
    """Dataset using real downloaded images with shared class mapping"""

    def __init__(self, df, downloaded_ids, transform=None):
        self.df = df[df['id'].isin(downloaded_ids)].reset_index(drop=True)
        self.downloaded = downloaded_ids
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        lm_id = row['landmark_id']
        label = SHARED_LM2IDX[lm_id]

        img_path = self.downloaded[img_id]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)

print("\nCreating datasets...")
train_ds = LandmarkDataset(train_df, DOWNLOADED, transform=train_transform)
val_ds = LandmarkDataset(val_df, DOWNLOADED, transform=val_transform)
print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,} samples")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # Parallel loading for real images
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

# ============================================================
# MODEL
# ============================================================
print("\nBuilding model...")
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")

# ============================================================
# TRAINING SETUP
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Mixed precision for RTX 2050
scaler = GradScaler("cuda") if torch.cuda.is_available() else None

# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0

    def should_stop(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "="*60)
print("Starting training...")
print("="*60)

start_time = time.time()
history = []

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        if torch.cuda.is_available() and scaler:
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(imgs)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        train_correct += (out.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

        pbar.set_postfix({
            'acc': f"{100.*train_correct/train_total:.1f}%",
            'loss': f"{loss.item():.4f}"
        })

    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / train_total

    # ---- VALIDATE ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if torch.cuda.is_available():
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(imgs)
                    loss = criterion(out, labels)
            else:
                out = model(imgs)
                loss = criterion(out, labels)

            val_loss += loss.item() * imgs.size(0)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / val_total

    scheduler.step()

    # ---- LOGGING ----
    epoch_time = time.time() - epoch_start

    # Save history
    history.append({
        'epoch': epoch + 1,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'epoch_time': epoch_time,
        'lr': optimizer.param_groups[0]['lr']
    })

    print(f"\nEpoch {epoch+1}: "
          f"Train {train_acc:.1f}% (loss: {avg_train_loss:.4f}) | "
          f"Val {val_acc:.1f}% (loss: {avg_val_loss:.4f}) | "
          f"Time: {epoch_time:.0f}s | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save best model
    if val_acc > early_stopping.best_acc:
        best_model_path = CHECKPOINT_DIR / "best_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'num_classes': NUM_CLASSES,
        }, best_model_path)
        print(f"  ⭐ Best model saved! ({val_acc:.1f}%)")

    # Check early stopping
    if early_stopping.should_stop(val_acc):
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# ============================================================
# FINAL SAVE
# ============================================================
total_time = time.time() - start_time

# Save final model
final_model_path = CHECKPOINT_DIR / "final_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'best_val_acc': early_stopping.best_acc,
    'num_classes': NUM_CLASSES,
    'training_time_minutes': total_time / 60,
}, final_model_path)

# Save training history
history_path = CHECKPOINT_DIR / "training_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"Best Val Accuracy: {early_stopping.best_acc:.1f}%")
print(f"Total Time: {total_time/60:.1f} minutes")
print(f"Epochs Completed: {len(history)}")
print(f"\nSaved to {CHECKPOINT_DIR}/")
print(f"  - best_model.pth (best validation)")
print(f"  - final_model.pth (last epoch)")
print(f"  - training_history.json")