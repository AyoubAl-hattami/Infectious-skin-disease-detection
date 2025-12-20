# src/preprocessing/dataloaders.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import logging
from PIL import Image

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SkinDiseaseDataset(Dataset):
    """
    PyTorch Dataset for skin disease images.
    Loads preprocessed images and their corresponding labels.
    """
    
    def __init__(self, metadata_csv: str, transform=None):
        """
        Args:
            metadata_csv: Path to metadata CSV file (train/val/test_metadata.csv)
            transform: Optional transforms to apply
        """
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform
        
        # Create label mapping
        self.class_to_idx = {
            'Cellulitis': 0,
            'Folliculitis': 1,
            'Impetigo': 2,
            'Tinea': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        logger.info(f"Dataset initialized: {len(self.df)} samples")
        logger.info(f"Class distribution:\n{self.df['disease'].value_counts()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load preprocessed image
        img_path = row['processed_path']
        image = self._load_image(img_path)
        
        # Get label
        label = self.class_to_idx[row['disease']]
        
        # Convert to tensor: HWC -> CHW format
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if self.transform:
            image = self.transform(image)
        
        # Get metadata (optional, can be used for analysis)
        metadata = {
            'filename': row['filename'],
            'disease': row['disease'],
            'caption': row.get('truncated_caption', ''),
        }
        
        return image, label
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        Load image from disk.
        Handles both preprocessed images and raw images.
        """
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Apply ImageNet normalization (for DenseNet-121 pretrained weights)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        return img_array
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced loss (useful if classes are imbalanced).
        
        Returns:
            Tensor of class weights
        """
        class_counts = self.df['disease'].value_counts()
        total = len(self.df)
        weights = torch.tensor([
            total / class_counts[self.idx_to_class[i]] 
            for i in range(len(self.class_to_idx))
        ], dtype=torch.float32)
        
        return weights


def create_dataloaders(
    processed_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Args:
        processed_dir: Directory containing processed images and metadata
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for parallel data loading
        shuffle_train: Whether to shuffle training data
        pin_memory: Whether to use pinned memory (faster GPU transfer)
        
    Returns:
        Dictionary containing 'train', 'val', 'test' DataLoaders
    """
    logger.info("Creating PyTorch DataLoaders...")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Pin memory: {pin_memory}")
    
    processed_path = Path(processed_dir)
    
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        metadata_csv = processed_path / f"{split}_metadata.csv"
        
        if not metadata_csv.exists():
            logger.warning(f"Metadata CSV not found: {metadata_csv}")
            continue
        
        # Create dataset
        dataset = SkinDiseaseDataset(metadata_csv)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle_train if split == 'train' else False),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')  # Drop last incomplete batch in training
        )
        
        dataloaders[split] = dataloader
        logger.info(f"✓ {split.upper()} DataLoader: {len(dataset)} samples, {len(dataloader)} batches")
    
    return dataloaders


class DataLoaderFactory:
    """
    Factory class to create and manage PyTorch DataLoaders.
    Integrates with preprocessing configuration.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary from preprocessing_config.yaml
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create dataloaders using configuration parameters.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        self.logger.info("="*60)
        self.logger.info("Creating DataLoaders from Configuration")
        self.logger.info("="*60)
        
        dataloaders = create_dataloaders(
            processed_dir=self.config['data']['processed_dir'],
            batch_size=self.config['output']['batch_size'],
            num_workers=self.config['output']['num_workers'],
            shuffle_train=True,
            pin_memory=True
        )
        
        self.logger.info("="*60)
        self.logger.info("DataLoaders Created Successfully")
        self.logger.info("="*60)
        
        return dataloaders
    
    def get_class_weights(self, split: str = 'train') -> torch.Tensor:
        """
        Get class weights for a specific split (useful for weighted loss).
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Tensor of class weights
        """
        metadata_csv = Path(self.config['data']['processed_dir']) / f"{split}_metadata.csv"
        dataset = SkinDiseaseDataset(metadata_csv)
        return dataset.get_class_weights()


if __name__ == "__main__":
    # Test script
    import yaml
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open('configs/preprocessing_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    factory = DataLoaderFactory(config)
    loaders = factory.create_dataloaders()
    
    # Test train dataloader
    print("\n" + "="*60)
    print("TESTING TRAIN DATALOADER")
    print("="*60)
    
    train_loader = loaders['train']
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch Information:")
    print(f"  Image batch shape: {images.shape}")  # Should be [batch_size, 3, 224, 224]
    print(f"  Label batch shape: {labels.shape}")  # Should be [batch_size]
    print(f"  Image dtype: {images.dtype}")
    print(f"  Label dtype: {labels.dtype}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Sample labels: {labels[:8]}")
    
    # Get class weights
    class_weights = factory.get_class_weights('train')
    print(f"\nClass weights for balanced loss:")
    class_names = ['Cellulitis', 'Folliculitis', 'Impetigo', 'Tinea']
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.4f}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)
