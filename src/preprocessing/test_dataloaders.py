# test_dataloaders.py

import yaml
import logging
import torch
from pathlib import Path
from dataloaders import DataLoaderFactory
from utils import (
    visualize_batch, 
    get_class_distribution, 
    verify_normalization,
    plot_class_distribution,
    compute_dataset_statistics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """
    Comprehensive test of PyTorch DataLoaders.
    """
    print("\n" + "="*60)
    print("PYTORCH DATALOADER TESTING")
    print("="*60)
    
    # Load config
    config_path = 'configs/preprocessing_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['data']['classes']
    
    # Create dataloaders
    logger.info("\n1. Creating DataLoaders...")
    factory = DataLoaderFactory(config)
    loaders = factory.create_dataloaders()
    
    # Test each split
    for split_name, dataloader in loaders.items():
        print("\n" + "="*60)
        print(f"TESTING {split_name.upper()} DATALOADER")
        print("="*60)
        
        # Get one batch
        images, labels = next(iter(dataloader))
        
        print(f"\n2. Batch Information:")
        print(f"  Image shape: {images.shape}")  # [B, C, H, W]
        print(f"  Label shape: {labels.shape}")  # [B]
        print(f"  Image dtype: {images.dtype}")
        print(f"  Label dtype: {labels.dtype}")
        print(f"  Device: {images.device}")
        
        # Verify normalization
        verify_normalization(images)
        
        # Class distribution
        print(f"\n3. Class Distribution in {split_name}:")
        dist = get_class_distribution(dataloader)
        for class_idx, count in sorted(dist.items()):
            print(f"  {class_names[class_idx]}: {count}")
        
        # Visualize first batch (only for train)
        if split_name == 'train':
            print(f"\n4. Visualizing batch...")
            visualize_batch(
                images, 
                labels, 
                class_names, 
                num_samples=8,
                save_path=f'logs/{split_name}_batch_samples.png'
            )
    
    # Overall distribution plot
    print("\n" + "="*60)
    print("GENERATING OVERALL STATISTICS")
    print("="*60)
    
    plot_class_distribution(loaders, class_names)
    
    # Compute dataset statistics (on train set)
    if 'train' in loaders:
        print("\n5. Computing dataset-wide statistics...")
        stats = compute_dataset_statistics(loaders['train'])
    
    # Get class weights
    print("\n6. Class Weights for Balanced Loss:")
    class_weights = factory.get_class_weights('train')
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.4f}")
    
    # Memory and performance info
    print("\n" + "="*60)
    print("DATALOADER CONFIGURATION")
    print("="*60)
    print(f"  Batch size: {config['output']['batch_size']}")
    print(f"  Num workers: {config['output']['num_workers']}")
    print(f"  Pin memory: True")
    print(f"  Shuffle train: True")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nGenerated files:")
    print("  - logs/train_batch_samples.png")
    print("  - logs/class_distribution.png")
    print("  - logs/preprocessing.log")


if __name__ == "__main__":
    main()
