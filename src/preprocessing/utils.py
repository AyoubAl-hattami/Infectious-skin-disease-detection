# src/preprocessing/utils.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import torch
from pathlib import Path


def visualize_batch(
    images: torch.Tensor, 
    labels: torch.Tensor, 
    class_names: List[str], 
    num_samples: int = 8,
    save_path: str = 'logs/batch_visualization.png'
):
    """
    Visualize a batch of images with their labels.
    
    Args:
        images: Batch of images [B, C, H, W]
        labels: Batch of labels [B]
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save visualization
    """
    # Move to CPU and convert to numpy
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Convert from CHW to HWC format
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Denormalize images for visualization (reverse MedImageInsight normalization)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    images = images * std + mean
    images = np.clip(images, 0, 1)
    
    num_samples = min(num_samples, len(images))
    rows = 2
    cols = num_samples // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].set_title(f"{class_names[labels[i]]}", fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Batch visualization saved to {save_path}")


def get_class_distribution(dataloader: torch.utils.data.DataLoader) -> Dict[int, int]:
    """
    Calculate class distribution in a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        Dictionary mapping class indices to counts
    """
    class_counts = {}
    
    for _, labels in dataloader:
        for label in labels.cpu().numpy():
            class_counts[label] = class_counts.get(label, 0) + 1
    
    return class_counts


def verify_normalization(images: torch.Tensor):
    """
    Verify that images are properly normalized for MedImageInsight.
    
    Args:
        images: Batch of images [B, C, H, W]
    """
    images_np = images.cpu().numpy()
    
    print("\n" + "="*60)
    print("NORMALIZATION VERIFICATION")
    print("="*60)
    print(f"  Shape: {images.shape}")
    print(f"  Min value: {images_np.min():.4f}")
    print(f"  Max value: {images_np.max():.4f}")
    print(f"  Mean: {images_np.mean():.4f}")
    print(f"  Std: {images_np.std():.4f}")
    print(f"\n  Expected for MedImageInsight:")
    print(f"    Range: approximately [-1, 1]")
    print(f"    Mean: close to 0")
    print(f"    Std: close to 1")
    print("="*60)


def plot_class_distribution(
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    class_names: List[str],
    save_path: str = 'logs/class_distribution.png'
):
    """
    Plot class distribution across train/val/test splits.
    
    Args:
        dataloaders: Dictionary of DataLoaders
        class_names: List of class names
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    splits = ['train', 'val', 'test']
    distributions = {}
    
    for split in splits:
        if split in dataloaders:
            dist = get_class_distribution(dataloaders[split])
            distributions[split] = [dist.get(i, 0) for i in range(len(class_names))]
    
    # Create plot
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, split in enumerate(splits):
        if split in distributions:
            offset = width * (i - 1)
            ax.bar(x + offset, distributions[split], width, label=split.capitalize())
    
    ax.set_xlabel('Disease Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Class distribution plot saved to {save_path}")


def compute_dataset_statistics(dataloader: torch.utils.data.DataLoader) -> Dict:
    """
    Compute statistics across entire dataset (mean, std per channel).
    Useful for verifying preprocessing.
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        Dictionary with mean and std per channel
    """
    print("\nComputing dataset statistics...")
    
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_batches = 0
    
    for images, _ in dataloader:
        # Sum over batch, height, width -> [C]
        channel_sum += images.mean(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).mean(dim=[0, 2, 3])
        num_batches += 1
    
    mean = channel_sum / num_batches
    std = torch.sqrt(channel_sum_sq / num_batches - mean ** 2)
    
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    
    print(f"  Mean per channel (R, G, B): {stats['mean']}")
    print(f"  Std per channel (R, G, B): {stats['std']}")
    
    return stats
