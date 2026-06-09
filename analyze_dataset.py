"""
Dataset Analysis Script
Analyze the Google Landmarks Dataset structure
"""
import pandas as pd
import os
from pathlib import Path

# Load train.csv
print("Loading train.csv...")
script_dir = Path(__file__).parent
train_csv = script_dir / "train.csv"

if train_csv.exists():
    df = pd.read_csv(train_csv)
else:
    parts = sorted(script_dir.glob(f"{train_csv.stem}_part_*{train_csv.suffix}"))
    if not parts:
        parts = sorted(script_dir.glob(f"{train_csv.stem}_*{train_csv.suffix}"))
    if not parts:
        raise FileNotFoundError(f"No CSV files found matching {train_csv}")
    print(f"Loading {len(parts)} split parts...")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

print(f"\n=== DATASET STATISTICS ===")
print(f"Total images: {len(df):,}")
print(f"Unique landmarks: {df['landmark_id'].nunique():,}")

print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\n=== CLASS DISTRIBUTION ===")
class_counts = df['landmark_id'].value_counts()
print(f"Min images per landmark: {class_counts.min()}")
print(f"Max images per landmark: {class_counts.max()}")
print(f"Mean images per landmark: {class_counts.mean():.2f}")
print(f"Median images per landmark: {class_counts.median():.2f}")

# Show top 10 landmarks
print(f"\nTop 10 most common landmarks:")
print(class_counts.head(10))

print(f"\nBottom 10 least common landmarks:")
print(class_counts.tail(10))

# Check for missing values
print(f"\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Save summary
summary_path = script_dir / "data" / "dataset_summary.txt"
with open(summary_path, 'w') as f:
    f.write(f"=== Google Landmarks Dataset Summary ===\n\n")
    f.write(f"Total images: {len(df):,}\n")
    f.write(f"Unique landmarks: {df['landmark_id'].nunique():,}\n")
    f.write(f"\nColumns: {list(df.columns)}\n")
    f.write(f"\n=== Class Distribution ===\n")
    f.write(f"Min per class: {class_counts.min()}\n")
    f.write(f"Max per class: {class_counts.max()}\n")
    f.write(f"Mean per class: {class_counts.mean():.2f}\n")
    f.write(f"\nTop 10 classes:\n{class_counts.head(10).to_string()}\n")

print(f"\nSummary saved to: {summary_path}")