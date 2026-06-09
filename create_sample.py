"""
Script to create a stratified sample of 50,000 images from train.csv for quick training.

This script:
1. Analyzes the distribution of landmark_ids in train.csv
2. Creates a stratified sample maintaining the class distribution
3. Saves the sample to data/train_sample_50k.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_and_sample_train_data(
    input_path: str,
    output_path: str,
    sample_size: int = 50000,
    random_state: int = 42
):
    """
    Create a stratified sample from the training data.

    Args:
        input_path: Path to the full train.csv
        output_path: Path to save the sampled data
        sample_size: Number of samples to draw
        random_state: Random seed for reproducibility
    """
    print(f"Loading data from {input_path}...")
    path = Path(input_path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        parts = sorted(path.parent.glob(f"{path.stem}_part_*{path.suffix}"))
        if not parts:
            parts = sorted(path.parent.glob(f"{path.stem}_*{path.suffix}"))
        if not parts:
            raise FileNotFoundError(f"No CSV file or parts found matching {input_path}")
        print(f"Loading data from {len(parts)} split files...")
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    print(f"\n=== Data Analysis ===")
    print(f"Total images: {len(df):,}")
    print(f"Unique landmark_ids: {df['landmark_id'].nunique():,}")

    # Analyze class distribution
    class_counts = df['landmark_id'].value_counts()
    print(f"\nClass distribution statistics:")
    print(f"  - Min samples per class: {class_counts.min()}")
    print(f"  - Max samples per class: {class_counts.max()}")
    print(f"  - Mean samples per class: {class_counts.mean():.2f}")
    print(f"  - Median samples per class: {class_counts.median():.2f}")

    # Count classes with different sample sizes
    small_classes = (class_counts < 10).sum()
    medium_classes = ((class_counts >= 10) & (class_counts < 50)).sum()
    large_classes = (class_counts >= 50).sum()

    print(f"\n  Classes with <10 samples: {small_classes:,}")
    print(f"  Classes with 10-49 samples: {medium_classes:,}")
    print(f"  Classes with 50+ samples: {large_classes:,}")

    # Stratified sampling
    print(f"\n=== Creating Stratified Sample ===")
    print(f"Target sample size: {sample_size:,}")

    np.random.seed(random_state)

    # Calculate sampling proportions
    total_samples = len(df)
    prop = sample_size / total_samples

    sampled_dfs = []

    # Sample from each class proportional to its size
    for landmark_id, group in df.groupby('landmark_id'):
        n_samples = max(1, int(len(group) * prop * 10))  # Scale up prop for oversampling small classes
        n_samples = min(n_samples, len(group))  # Don't sample more than available

        if len(group) <= n_samples:
            sampled_dfs.append(group)
        else:
            sampled_dfs.append(group.sample(n=n_samples, random_state=random_state))

    # Combine and trim to exact sample size
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    # If we have too many samples, trim while maintaining stratification
    if len(sampled_df) > sample_size:
        print(f"Sampling yielded {len(sampled_df):,} samples, trimming to {sample_size:,}...")

        # Trim proportionally from each class
        final_dfs = []
        for landmark_id, group in sampled_df.groupby('landmark_id'):
            n_to_keep = int(len(group) / len(sampled_df) * sample_size)
            n_to_keep = max(1, n_to_keep)  # Keep at least 1 per class
            n_to_keep = min(n_to_keep, len(group))

            if len(group) <= n_to_keep:
                final_dfs.append(group)
            else:
                final_dfs.append(group.sample(n=n_to_keep, random_state=random_state))

        sampled_df = pd.concat(final_dfs, ignore_index=True)

        # Final adjustment to hit exact target
        if len(sampled_df) > sample_size:
            # Remove excess from largest classes that have more than their "fair share"
            excess = len(sampled_df) - sample_size
            groups_sorted = sampled_df.groupby('landmark_id').size().sort_values(ascending=False)

            for landmark_id in groups_sorted.index:
                if excess <= 0:
                    break
                excess_in_class = len(sampled_df[sampled_df['landmark_id'] == landmark_id]) - 1
                n_remove = min(excess, excess_in_class)
                if n_remove > 0:
                    idx_to_remove = sampled_df[sampled_df['landmark_id'] == landmark_id].index[1:n_remove+1]
                    sampled_df = sampled_df.drop(idx_to_remove)
                    excess -= n_remove

    # Shuffle the final sample
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\n=== Sample Statistics ===")
    print(f"Sampled images: {len(sampled_df):,}")
    print(f"Unique landmark_ids in sample: {sampled_df['landmark_id'].nunique():,}")

    # Show distribution in sample
    sample_class_counts = sampled_df['landmark_id'].value_counts()
    print(f"\nSample class distribution:")
    print(f"  - Min samples per class: {sample_class_counts.min()}")
    print(f"  - Max samples per class: {sample_class_counts.max()}")
    print(f"  - Mean samples per class: {sample_class_counts.mean():.2f}")
    print(f"  - Median samples per class: {sample_class_counts.median():.2f}")

    # Save the sample
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False)
    print(f"\nSaved stratified sample to {output_path}")

    # Show sample preview
    print(f"\nFirst 5 rows of sample:")
    print(sampled_df.head().to_string())

    return sampled_df


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    input_path = script_dir / "train.csv"
    output_path = script_dir / "data" / "train_sample_50k.csv"

    sample_df = analyze_and_sample_train_data(
        input_path=str(input_path),
        output_path=str(output_path),
        sample_size=50000,
        random_state=42
    )