"""
Landmark Detection - Data Loading Utilities
Custom datasets and data loaders for Google Landmarks Dataset
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


class LandmarkDataset(Dataset):
    """
    Custom Dataset for Landmark Detection.
    
    Args:
        csv_file: Path to CSV file with columns [image_id, url, landmark_id]
        root_dir: Root directory containing images
        transform: Optional transform to apply to images
        train: Whether this is training data (applies augmentation)
        image_size: Target image size
    """
    
    def __init__(self, csv_file=None, annotations=None, root_dir=None, 
                 transform=None, train=True, image_size=224):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file
            annotations: Alternative to csv_file - DataFrame with annotations
            root_dir: Directory with all the images
            transform: Optional transform
            train: Whether this is training dataset
            image_size: Size to resize images to
        """
        if annotations is not None:
            self.annotations = annotations
        elif csv_file is not None:
            self.annotations = pd.read_csv(csv_file)
        else:
            raise ValueError("Either csv_file or annotations must be provided")
        
        self.root_dir = root_dir if root_dir else ""
        self.train = train
        self.image_size = image_size
        
        # Setup transforms
        if transform:
            self.transform = transform
        else:
            self.transform = self._get_transform(train, image_size)
        
        # Get unique landmark IDs and create mapping
        self.unique_landmarks = sorted(self.annotations['landmark_id'].unique())
        self.landmark_to_idx = {lm: idx for idx, lm in enumerate(self.unique_landmarks)}
        self.idx_to_landmark = {idx: lm for lm, idx in self.landmark_to_idx.items()}
        
        self.num_classes = len(self.unique_landmarks)
        
    def _get_transform(self, train, image_size):
        """Get image transforms."""
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                      saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.annotations.iloc[idx]
        
        # Try to load image from local path
        image_path = row.get('image_path', None)
        url = row.get('url', '')
        
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                # Load a dummy image if file is corrupted
                print(f"Error loading {image_path}: {e}")
                image = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        else:
            # Create a placeholder image if path doesn't exist
            # In practice, you would download from URL here
            image = Image.new('RGB', (self.image_size, self.image_size), 
                           color=(np.random.randint(0, 255),
                                  np.random.randint(0, 255),
                                  np.random.randint(0, 255)))
        
        # Apply transforms
        image = self.transform(image)
        
        # Get label
        landmark_id = row['landmark_id']
        label = self.landmark_to_idx.get(landmark_id, 0)
        
        return image, torch.tensor(label, dtype=torch.long)


class InferenceDataset(Dataset):
    """
    Dataset for inference - only loads images without labels.
    """
    
    def __init__(self, image_paths, transform=None, image_size=224):
        """
        Args:
            image_paths: List of image file paths
            transform: Optional transform
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.image_size = image_size
        
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        
        image = self.transform(image)
        return image, os.path.basename(image_path)


def create_data_loaders(train_csv, val_csv=None, test_csv=None,
                        train_dir=None, batch_size=32, num_workers=4,
                        image_size=224, sample_size=None):
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Optional path to validation CSV
        test_csv: Optional path to test CSV
        train_dir: Root directory for training images
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size
        sample_size: Optional - sample this many examples
        
    Returns:
        Dictionary with 'train', 'val', 'test' data loaders
    """
    loaders = {}
    
    # Training loader
    train_dataset = LandmarkDataset(
        csv_file=train_csv,
        root_dir=train_dir,
        train=True,
        image_size=image_size
    )
    
    if sample_size:
        train_dataset.annotations = train_dataset.annotations.sample(
            n=min(sample_size, len(train_dataset.annotations)),
            random_state=42
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation loader
    if val_csv:
        val_dataset = LandmarkDataset(
            csv_file=val_csv,
            root_dir=train_dir,
            train=False,
            image_size=image_size
        )
        print(f"Validation samples: {len(val_dataset)}")
        
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Test loader
    if test_csv:
        test_dataset = LandmarkDataset(
            csv_file=test_csv,
            root_dir=train_dir,
            train=False,
            image_size=image_size
        )
        print(f"Test samples: {len(test_dataset)}")
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders, train_dataset


def create_demo_dataset(num_classes=100, samples_per_class=50, image_size=224):
    """
    Create a demo dataset with synthetic data for testing.
    Useful when actual images are not available.
    
    Args:
        num_classes: Number of landmark classes
        samples_per_class: Samples per class
        image_size: Image size
        
    Returns:
        Dataset object
    """
    annotations = []
    for landmark_id in range(num_classes):
        for i in range(samples_per_class):
            annotations.append({
                'image_id': f"img_{landmark_id}_{i}",
                'url': f"https://example.com/landmark/{landmark_id}/image_{i}.jpg",
                'landmark_id': landmark_id
            })
    
    df = pd.DataFrame(annotations)
    
    dataset = LandmarkDataset(
        annotations=df,
        train=True,
        image_size=image_size
    )
    
    return dataset


if __name__ == "__main__":
    # Test the data loader
    dataset = create_demo_dataset(num_classes=10, samples_per_class=5)
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Test loading a sample
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")