# src/preprocessing/image_preprocessor.py


import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import mlflow
import dagshub


class ImagePreprocessor:
    """
    Advanced image preprocessing pipeline for infectious skin diseases detection.
    Handles data loading, quality filtering, class balancing, splitting, and saving.
    """
    
    def __init__(self, config_path: str = "configs/preprocessing_config.yaml"):
        """Initialize preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_tracking()
        
        self.logger.info("ImagePreprocessor initialized successfully")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['logging']['log_file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=self.config['logging']['level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_tracking(self):
        """Initialize DVC and MLflow tracking."""
        if self.config['tracking']['mlflow_enabled']:
            dagshub.init(
                repo_owner=self.config['tracking']['dagshub_repo'].split('/')[0],
                repo_name=self.config['tracking']['dagshub_repo'].split('/')[1],
                mlflow=True
            )
            self.logger.info("MLflow tracking initialized with Dagshub")
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata from all disease CSV files.
        
        Returns:
            Combined DataFrame with all images and metadata
        """
        self.logger.info("Loading metadata from CSV files...")
        
        all_data = []
        for disease, csv_path in self.config['data']['metadata_files'].items():
            if not os.path.exists(csv_path):
                self.logger.warning(f"CSV file not found: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            df['disease'] = disease
            all_data.append(df)
            
            self.logger.info(f"Loaded {len(df)} images for {disease}")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Total images loaded: {len(combined_df)}")
        
        return combined_df
    
    def filter_by_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter images based on quality criteria.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        if not self.config['quality_filtering']['enabled']:
            self.logger.info("Quality filtering is disabled")
            return df
        
        initial_count = len(df)
        
        # Filter out duplicates if specified
        if not self.config['quality_filtering']['use_duplicates']:
            if 'is_duplicate' in df.columns:
                df = df[df['is_duplicate'] == False].copy()
                self.logger.info(f"Removed duplicates. Remaining: {len(df)}")
        
        final_count = len(df)
        self.logger.info(f"Quality filtering removed {initial_count - final_count} images")
        
        return df
    
    def balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance classes by undersampling to match smallest class.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Balanced DataFrame
        """
        if not self.config['class_balance']['enabled']:
            self.logger.info("Class balancing is disabled")
            return df
        
        self.logger.info("Balancing classes...")
        
        # Get class distribution
        class_counts = df['disease'].value_counts()
        self.logger.info(f"Original class distribution:\n{class_counts}")
        
        # Find minimum class size
        min_count = class_counts.min()
        self.logger.info(f"Undersampling all classes to {min_count} samples")
        
        # Undersample each class
        balanced_dfs = []
        for disease in self.config['data']['classes']:
            disease_df = df[df['disease'] == disease]
            sampled_df = disease_df.sample(n=min_count, random_state=self.config['split']['random_seed'])
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Log new distribution
        new_counts = balanced_df['disease'].value_counts()
        self.logger.info(f"Balanced class distribution:\n{new_counts}")
        
        return balanced_df
    
    def create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test splits.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with train, val, test DataFrames
        """
        self.logger.info("Creating train/val/test splits...")
        
        train_ratio = self.config['split']['train']
        val_ratio = self.config['split']['val']
        test_ratio = self.config['split']['test']
        random_seed = self.config['split']['random_seed']
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df['disease'] if self.config['split']['stratify'] else None,
            random_state=random_seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['disease'] if self.config['split']['stratify'] else None,
            random_state=random_seed
        )
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        for split_name, split_df in splits.items():
            self.logger.info(f"{split_name.upper()} set: {len(split_df)} images")
            self.logger.info(f"  Class distribution:\n{split_df['disease'].value_counts()}")
        
        return splits
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        img = Image.open(image_path).convert(self.config['image']['color_mode'])
        
        # Resize
        target_size = tuple(self.config['image']['target_size'])
        img = img.resize(target_size, Image.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Apply normalization
        if self.config['image']['normalization']['method'] == 'standard':
            mean = np.array(self.config['image']['normalization']['mean'])
            std = np.array(self.config['image']['normalization']['std'])
            img_array = (img_array - mean) / std
        elif self.config['image']['normalization']['method'] == 'min_max':
            img_array = img_array * 2.0 - 1.0  # Scale to [-1, 1]
        # 'none' method keeps [0, 1] range
        
        return img_array
    
    def save_processed_images(self, splits: Dict[str, pd.DataFrame]):
        """
        Process and save images to disk.
        
        Args:
            splits: Dictionary with train/val/test DataFrames
        """
        self.logger.info("Processing and saving images to disk...")
        
        processed_dir = Path(self.config['data']['processed_dir'])
        
        for split_name, split_df in splits.items():
            split_dir = processed_dir / split_name
            
            # Create directories for each class
            for disease in self.config['data']['classes']:
                (split_dir / disease).mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Processing {split_name} set ({len(split_df)} images)...")
            
            processed_paths = []
            for idx, row in split_df.iterrows():
                try:
                    # Get original image path
                    if 'new_path' in row:
                        original_path = row['new_path']
                    else:
                        original_path = os.path.join(
                            self.config['data']['raw_dir'],
                            row['disease'],
                            row['filename']
                        )
                    
                    # Preprocess image
                    img_array = self.preprocess_image(original_path)
                    
                    # Save processed image
                    save_path = split_dir / row['disease'] / row['filename']
                    
                    # Convert back to [0, 255] for saving
                    if self.config['image']['normalization']['method'] == 'standard':
                        mean = np.array(self.config['image']['normalization']['mean'])
                        std = np.array(self.config['image']['normalization']['std'])
                        img_to_save = (img_array * std + mean) * 255.0
                    elif self.config['image']['normalization']['method'] == 'min_max':
                        img_to_save = (img_array + 1.0) * 127.5
                    else:
                        img_to_save = img_array * 255.0
                    
                    img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
                    Image.fromarray(img_to_save).save(save_path)
                    
                    processed_paths.append(str(save_path))
                    
                except Exception as e:
                    self.logger.error(f"Error processing {row['filename']}: {str(e)}")
                    continue
            
            # Update DataFrame with processed paths
            split_df['processed_path'] = processed_paths
            
            # Save metadata CSV
            metadata_path = processed_dir / f"{split_name}_metadata.csv"
            split_df.to_csv(metadata_path, index=False)
            self.logger.info(f"Saved {split_name} metadata to {metadata_path}")
        
        self.logger.info("Image processing completed")
    
    def run_pipeline(self):
        """Execute the complete preprocessing pipeline."""
        self.logger.info("="*50)
        self.logger.info("Starting Preprocessing Pipeline")
        self.logger.info("="*50)
        
        if self.config['tracking']['mlflow_enabled']:
            mlflow.start_run(run_name="preprocessing_pipeline")
            mlflow.log_params({
                "target_size": self.config['image']['target_size'],
                "normalization_method": self.config['image']['normalization']['method'],
                "class_balance_enabled": self.config['class_balance']['enabled'],
                "train_split": self.config['split']['train'],
                "val_split": self.config['split']['val'],
                "test_split": self.config['split']['test']
            })
        
        try:
            # Step 1: Load metadata
            df = self.load_metadata()
            mlflow.log_metric("total_images_loaded", len(df))
            for disease in self.config['data']['classes']:
                count = len(df[df['disease'] == disease])
                mlflow.log_metric(f"raw_{disease}_count", count)
            
            # Step 2: Quality filtering
            df = self.filter_by_quality(df)
            mlflow.log_metric("images_after_filtering", len(df))
            for disease in self.config['data']['classes']:
                count = len(df[df['disease'] == disease])
                mlflow.log_metric(f"filtered_{disease}_count", count)
            
            # Step 3: Class balancing
            df = self.balance_classes(df)
            mlflow.log_metric("images_after_balancing", len(df))
            for disease in self.config['data']['classes']:
                count = len(df[df['disease'] == disease])
                mlflow.log_metric(f"balanced_{disease}_count", count)
            
            # Step 4: Create splits
            splits = self.create_splits(df)
            mlflow.log_metric("train_size", len(splits['train']))
            mlflow.log_metric("val_size", len(splits['val']))
            mlflow.log_metric("test_size", len(splits['test']))
            for split_name, split_df in splits.items():
                for disease in self.config['data']['classes']:
                    count = len(split_df[split_df['disease'] == disease])
                    mlflow.log_metric(f"{split_name}_{disease}_count", count)
            
            # Step 5: Save processed images
            if self.config['output']['save_to_disk']:
                self.save_processed_images(splits)
            
            self.logger.info("="*50)
            self.logger.info("Preprocessing Pipeline Completed Successfully")
            self.logger.info("="*50)
            
            if self.config['tracking']['mlflow_enabled']:
                mlflow.end_run()
            
            return splits
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            if self.config['tracking']['mlflow_enabled']:
                mlflow.end_run(status="FAILED")
            raise



if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    splits = preprocessor.run_pipeline()
