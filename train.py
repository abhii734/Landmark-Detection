"""
Landmark Detection Training Script
===================================
Two modes: SYNTHETIC (matches video) or REAL_IMAGE (uses your data)

Usage:
    python train.py                      # Synthetic mode (like video)
    python train.py --mode synthetic     # Explicit synthetic mode
    python train.py --mode real         # Use downloaded images
    python train.py --epochs 10         # Custom epochs
    python train.py --batch 32           # Custom batch size
    python train.py --classes 100        # Custom class count (real mode)
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import models
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time
import json
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG (defaults match video tutorial)
# ============================================================
BATCH_SIZE = 64           # Video default (T4 GPU)
LEARNING_RATE = 0.001     # Video default
NUM_EPOCHS = 5            # Video default
NUM_CLASSES = 500         # Video: 500 landmark classes
EARLY_STOP_PATIENCE = 5

# Synthetic mode config
SYNTHETIC_IMAGES_PER_CLASS = 50  # Video: 50 images per class

# Paths
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Landmark Detection Training")
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "real"],
                        help="Training mode: synthetic (video) or real (your images)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--classes", type=int, default=NUM_CLASSES,
                        help=f"Number of classes: synthetic=500, real=up to 100 (default: {NUM_CLASSES})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()

args = parse_args()

# Override config with args
BATCH_SIZE = args.batch
NUM_EPOCHS = args.epochs
NUM_CLASSES = args.classes
LR = args.lr

print("="*60)
print("LANDMARK DETECTION TRAINING")
print("="*60)
print(f"Mode: {args.mode.upper()}")
print(f"Classes: {NUM_CLASSES}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LR}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATASETS
# ============================================================
class SyntheticDataset(Dataset):
    """
    Synthetic dataset - PROCRUSTEAN patterns for each landmark.
    This is what the video tutorial uses to demo training without real images.
    Each class gets a unique 8x8 grid pattern in the image.
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.lms = sorted(df['landmark_id'].unique())
        self.lm2idx = {lm: i for i, lm in enumerate(self.lms)}

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.lm2idx[row['landmark_id']]

        # Generate synthetic image with pattern based on label
        rng = np.random.RandomState(int(idx) + int(label) * 1000)
        base = rng.randint(50, 200, 3)
        img = np.ones((224, 224, 3), dtype=np.uint8) * base

        # 8x8 grid pattern based on label
        py, px = (label % 8) * 28, ((label // 8) % 8) * 28
        img[py:py+28, px:px+28] = 255 - base

        # Convert to tensor
        img = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)
        img = (img - self.mean) / self.std

        return img, torch.tensor(label)


class RealImageDataset(Dataset):
    """Dataset using real downloaded landmark images."""
    def __init__(self, df, downloaded_ids, transform=None):
        # Filter to only available images
        self.df = df[df['id'].isin(downloaded_ids)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No images found matching the CSV metadata!")

        self.downloaded = downloaded_ids
        self.transform = transform or self._default_transform()

        # Build class mapping
        self.lms = sorted(self.df['landmark_id'].unique())
        self.lm2idx = {lm: i for i, lm in enumerate(self.lms)}

    def _default_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        lm_id = row['landmark_id']
        label = self.lm2idx[lm_id]

        img_path = self.downloaded[img_id]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)


# ============================================================
# LOAD DATA
# ============================================================
def load_data_for_synthetic():
    """Create synthetic dataset like in the video - simple version."""
    print("\n[MODE: SYNTHETIC] Creating synthetic dataset...")

    # Create balanced dataset: classes with enough samples
    data = []
    for class_id in range(NUM_CLASSES):
        for img_idx in range(SYNTHETIC_IMAGES_PER_CLASS):
            data.append({
                'id': f'img_{class_id}_{img_idx}',
                'landmark_id': class_id
            })

    df = pd.DataFrame(data)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['landmark_id'], random_state=42
    )

    print(f"  Training samples: {len(train_df):,}")
    print(f"  Validation samples: {len(val_df):,}")
    print(f"  Classes: {df['landmark_id'].nunique()}")

    return train_df, val_df, df['landmark_id'].nunique()


def load_data_for_real():
    """Load real images from downloads."""
    print("\n[MODE: REAL] Loading real images...")

    # Load first CSV part (has the IDs we need)
    df = pd.read_csv(SCRIPT_DIR / "train_part_0.csv", usecols=['id', 'landmark_id'])
    print(f"  CSV entries: {len(df):,}")

    # Scan downloaded images
    IMAGE_DIR = SCRIPT_DIR / "images_000" / "0" / "0"
    DOWNLOADED = {}
    for subdir in range(9):
        folder = IMAGE_DIR / str(subdir)
        if folder.exists():
            for f in folder.glob("*.jpg"):
                DOWNLOADED[f.stem] = f

    print(f"  Downloaded images: {len(DOWNLOADED):,}")

    # Match with metadata
    available = df[df['id'].isin(DOWNLOADED.keys())].copy()
    print(f"  Images with metadata: {len(available):,}")

    if len(available) == 0:
        raise ValueError("No images matched! Check download_images.py ran successfully.")

    # Select top classes with enough samples
    MIN_SAMPLES = 2
    MAX_CLASSES = min(args.classes, 100)  # Cap at 100 for real images
    class_counts = available['landmark_id'].value_counts()
    valid_classes = class_counts[class_counts >= MIN_SAMPLES].head(MAX_CLASSES).index

    available_sampled = available[available['landmark_id'].isin(valid_classes)].copy()
    print(f"  Selected classes: {len(valid_classes)}")
    print(f"  Selected images: {len(available_sampled):,}")

    # Split - need 30%+ for validation with many classes
    test_size = 0.3 if len(valid_classes) > 50 else 0.2
    train_df, val_df = train_test_split(
        available_sampled,
        test_size=test_size,
        stratify=available_sampled['landmark_id'],
        random_state=42
    )

    print(f"  Training samples: {len(train_df):,}")
    print(f"  Validation samples: {len(val_df):,}")

    return train_df, val_df, DOWNLOADED, len(valid_classes)


# ============================================================
# LOAD DATA BASED ON MODE
# ============================================================
if args.mode == "synthetic":
    train_df, val_df, actual_classes = load_data_for_synthetic()
    DOWNLOADED = None
else:
    train_df, val_df, DOWNLOADED, actual_classes = load_data_for_real()

NUM_CLASSES = actual_classes

# Create datasets
if args.mode == "synthetic":
    train_ds = SyntheticDataset(train_df)
    val_ds = SyntheticDataset(val_df)
else:
    train_ds = RealImageDataset(train_df, DOWNLOADED)
    val_ds = RealImageDataset(val_df, DOWNLOADED)

print(f"\nDataset created: {NUM_CLASSES} classes")

# DataLoaders
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ============================================================
# MODEL
# ============================================================
print("\n" + "="*60)
print("Building model...")
print("="*60)

model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: ResNet50 | Parameters: {total_params:,}")
print(f"Output classes: {NUM_CLASSES}")

# ============================================================
# TRAINING
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

scaler = GradScaler("cuda") if torch.cuda.is_available() else None
best_acc = 0.0
start_time = time.time()
history = []

print("\n" + "="*60)
print("Starting training...")
print("="*60)

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if torch.cuda.is_available() and scaler:
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
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

        pbar.set_postfix(acc=f"{100.*train_correct/train_total:.1f}%")

    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / train_total

    # ---- VALIDATE ----
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if torch.cuda.is_available() and scaler:
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

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}: "
          f"Train {train_acc:.1f}% (loss: {avg_train_loss:.4f}) | "
          f"Val {val_acc:.1f}% (loss: {avg_val_loss:.4f}) | "
          f"Time: {elapsed:.0f}s")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'num_classes': NUM_CLASSES,
            'mode': args.mode,
        }, CHECKPOINT_DIR / "best_model.pth")
        print(f"  [BEST] Model saved! ({val_acc:.1f}%)")

# ============================================================
# FINAL SAVE
# ============================================================
total_time = time.time() - start_time

torch.save({
    'model_state_dict': model.state_dict(),
    'best_acc': best_acc,
    'num_classes': NUM_CLASSES,
    'mode': args.mode,
    'training_time_min': total_time / 60,
}, CHECKPOINT_DIR / "final_model.pth")

with open(CHECKPOINT_DIR / "training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best Val Accuracy: {best_acc:.1f}%")
print(f"Total Time: {total_time/60:.1f} minutes")
print(f"Mode: {args.mode.upper()}")
print(f"Classes: {NUM_CLASSES}")
print(f"\nSaved to {CHECKPOINT_DIR}/:")
print("  - best_model.pth (best validation)")
print("  - final_model.pth (last epoch)")
print("  - training_history.json")