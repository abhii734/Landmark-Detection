# Data Directory

This directory contains processed data and metadata for the landmark detection project.

## Files

| File | Description |
|------|-------------|
| `train_sample.csv` | Sampled training data (balanced subset) |
| `train_split.csv` | Training split for training |
| `val_split.csv` | Validation split |
| `landmark_mapping.csv` | Mapping of class indices to landmark IDs |
| `dataset_summary.txt` | Statistics about the full dataset |

## Download Images

To download images for training, run:

```bash
# Download all images from train.csv
python download_images.py

# Download a stratified sample (recommended for testing)
python download_images.py --sample 50000

# Download first 10,000 images (sequential)
python download_images.py --head 10000

# Resume interrupted download
python download_images.py --resume
```

## Dataset Statistics

- **Total images in train.csv**: 4,132,914
- **Unique landmarks**: 203,094
- **Highly imbalanced**: 1 to 10,247 images per landmark

## Sample Data

For quick training, we sample from landmarks with 20+ images and take up to 50 images per landmark. This creates a manageable dataset of ~25,000 images across ~500 landmarks.