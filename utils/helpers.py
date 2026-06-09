"""
Landmark Detection - Utility Functions
Helper functions for data loading, visualization, and training
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def prepare_image(image_path, transform=None):
    """
    Load and transform an image.
    
    Args:
        image_path: Path to the image file
        transform: Optional torchvision transform
        
    Returns:
        Transformed image tensor
    """
    try:
        image = Image.open(image_path).convert('RGB')
        if transform:
            image = transform(image)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def visualize_batch(images, labels, class_names=None, num_images=8):
    """
    Visualize a batch of images with their labels.
    
    Args:
        images: Batch of images tensor [B, C, H, W]
        labels: Batch of labels tensor [B]
        class_names: Optional list of class names
        num_images: Number of images to display
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_images, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        
        label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
        
        if class_names and label < len(class_names):
            title = class_names[label]
        else:
            title = f"Landmark {label}"
        
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def compute_accuracy(outputs, labels, topk=(1,)):
    """
    Compute top-k accuracy.
    
    Args:
        outputs: Model predictions [B, num_classes]
        labels: Ground truth labels [B]
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        List of top-k accuracies
    """
    maxk = max(topk)
    batch_size = labels.size(0)
    
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    
    return res


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optional optimizer
        
    Returns:
        Tuple of (epoch, loss, accuracy)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    accuracy = checkpoint.get('accuracy', 0.0)
    
    print(f"Checkpoint loaded from {filepath}")
    return epoch, loss, accuracy


def count_parameters(model):
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def print_model_summary(model, input_size=(3, 224, 224)):
    """Print model summary."""
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")