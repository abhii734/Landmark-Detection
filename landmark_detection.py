#!/usr/bin/env python3
"""
Landmark Detection - Training & Inference Script
Google Landmarks Dataset v2 (4.1M+ images, 203K+ landmarks)

Usage:
    # Train with sample data
    python landmark_detection.py --mode train --sample 25000
    
    # Train with full data (requires ~200GB storage)
    python landmark_detection.py --mode train
    
    # Run inference
    python landmark_detection.py --mode inference --image path/to/image.jpg
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.resolve()
    DATA_DIR = PROJECT_ROOT / "data"
    IMAGES_DIR = PROJECT_ROOT / "images_000"
    TRAIN_CSV = PROJECT_ROOT / "train.csv"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    
    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 0.01
    image_size = 224
    
    # Sampling (for practical training)
    min_images_per_landmark = 20  # Minimum images per landmark
    max_landmarks = 500           # Max number of classes
    samples_per_class = 50        # Images per class
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATASET CLASS
# ============================================================================

class LandmarkDataset(Dataset):
    """Dataset for Google Landmarks with local images."""
    
    def __init__(self, df, images_dir, transform=None, train=True):
        self.df = df
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.train = train
        
        # Create landmark mapping
        self.landmarks = sorted(self.df['landmark_id'].unique())
        self.landmark_to_idx = {lm: idx for idx, lm in enumerate(self.landmarks)}
        self.idx_to_landmark = {idx: lm for lm, idx in self.landmark_to_idx.items()}
        self.num_classes = len(self.landmarks)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        landmark_id = row['landmark_id']
        
        # Load image
        img_path = self.images_dir / f"{img_id}.jpg"
        
        if img_path.exists():
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = Image.new('RGB', (Config.image_size, Config.image_size), color='gray')
        else:
            # Placeholder for missing images
            image = Image.new('RGB', (Config.image_size, Config.image_size), color='lightblue')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.landmark_to_idx[landmark_id]
        return image, torch.tensor(label, dtype=torch.long)

# ============================================================================
# MODEL
# ============================================================================

class LandmarkClassifier(nn.Module):
    """ResNet50-based landmark classifier."""
    
    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_split_csv(base_path):
    """Load split CSV files and concatenate them."""
    base_path = Path(base_path)
    parent = base_path.parent
    stem = base_path.stem
    ext = base_path.suffix
    
    if base_path.exists():
        return pd.read_csv(base_path)
        
    parts = sorted(parent.glob(f"{stem}_part_*{ext}"))
    if not parts:
        parts = sorted(parent.glob(f"{stem}_*{ext}"))
        
    if not parts:
        raise FileNotFoundError(f"No CSV file or parts found for {base_path}")
        
    print(f"Loading data from {len(parts)} split files...")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

def load_training_data(sample_size=None):
    """Load and prepare training/validation data with stratified sampling."""
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    # Check if sample exists
    sample_csv = Config.DATA_DIR / "train_sample.csv"
    split_train = Config.DATA_DIR / "train_split.csv"
    split_val = Config.DATA_DIR / "val_split.csv"
    
    if split_train.exists() and split_val.exists():
        print("Loading existing split...")
        train_df = pd.read_csv(split_train)
        val_df = pd.read_csv(split_val)
    elif sample_csv.exists():
        print("Loading sample...")
        df = pd.read_csv(sample_csv)
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['landmark_id'], random_state=42
        )
        
        # Save splits
        train_df.to_csv(split_train, index=False)
        val_df.to_csv(split_val, index=False)
    else:
        print("Creating new sample from train.csv...")
        df = load_split_csv(Config.TRAIN_CSV)
        
        # Count images per landmark
        class_counts = df['landmark_id'].value_counts()
        
        # Find landmarks with enough images
        valid_landmarks = class_counts[class_counts >= Config.min_images_per_landmark].index
        
        # Limit to top N landmarks
        if len(valid_landmarks) > Config.max_landmarks:
            valid_landmarks = class_counts.head(Config.max_landmarks).index
        
        df = df[df['landmark_id'].isin(valid_landmarks)]
        
        # Stratified sample per class
        sampled_dfs = []
        for landmark_id in valid_landmarks:
            landmark_df = df[df['landmark_id'] == landmark_id]
            n = min(Config.samples_per_class, len(landmark_df))
            sampled_dfs.append(landmark_df.sample(n=n, random_state=42))
        
        df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Save sample
        sample_csv.parent.mkdir(exist_ok=True)
        df.to_csv(sample_csv, index=False)
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['landmark_id'], random_state=42
        )
        
        train_df.to_csv(split_train, index=False)
        val_df.to_csv(split_val, index=False)
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    
    return train_df, val_df

# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(Config.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Main training function."""
    print(f"\n{'='*60}")
    print("TRAINING LANDMARK DETECTION MODEL")
    print(f"{'='*60}")
    print(f"Device: {Config.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Create directories
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    train_df, val_df = load_training_data()
    
    # Create datasets
    train_dataset = LandmarkDataset(train_df, Config.IMAGES_DIR, get_train_transforms(), train=True)
    val_dataset = LandmarkDataset(val_df, Config.IMAGES_DIR, get_val_transforms(), train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Number of classes: {train_dataset.num_classes}")
    
    # Create model
    model = LandmarkClassifier(train_dataset.num_classes, pretrained=True, dropout=0.3)
    model = model.to(Config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Train')
        for images, labels in pbar:
            images, labels = images.to(Config.device), labels.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Val'):
                images, labels = images.to(Config.device), labels.to(Config.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.CHECKPOINT_DIR / 'best_model.pth')
            print("✓ Best model saved!")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history,
        'num_classes': train_dataset.num_classes
    }, Config.CHECKPOINT_DIR / 'final_model.pth')
    
    print(f"\nTraining complete! Best accuracy: {best_val_acc:.2f}%")
    
    return model, history

# ============================================================================
# INFERENCE
# ============================================================================

def inference(args):
    """Inference function."""
    print(f"\n{'='*60}")
    print("LANDMARK DETECTION INFERENCE")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint_path = Config.CHECKPOINT_DIR / 'final_model.pth'
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=Config.device)
        num_classes = checkpoint.get('num_classes', 500)
        
        model = LandmarkClassifier(num_classes, pretrained=False, dropout=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model: {num_classes} classes")
    else:
        # Use pretrained model as fallback
        print("No checkpoint found, using pretrained ResNet50")
        model = LandmarkClassifier(1000, pretrained=True, dropout=0.3)
        num_classes = 1000
    
    model = model.to(Config.device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if args.image:
        image = Image.open(args.image).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(Config.device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
        
        print(f"\nImage: {args.image}")
        print(f"Predicted class: {pred.item()}")
        print(f"Confidence: {conf.item()*100:.2f}%")
        
        # Top 5
        print("\nTop 5 predictions:")
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            print(f"  Class {idx.item()}: {prob.item()*100:.2f}%")
    else:
        print("No image provided. Use --image path/to/image.jpg")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Landmark Detection')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--image', type=str, default=None, help='Path to image for inference')
    parser.add_argument('--epochs', type=int, default=Config.num_epochs)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--sample', type=int, default=None, help='Sample size for training')
    
    args = parser.parse_args()
    
    # Override configs
    if args.sample:
        Config.max_landmarks = args.sample // Config.samples_per_class
        Config.samples_per_class = Config.samples_per_class
    
    if args.epochs:
        Config.num_epochs = args.epochs
    
    if args.batch_size:
        Config.batch_size = args.batch_size
    
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)

if __name__ == '__main__':
    main()