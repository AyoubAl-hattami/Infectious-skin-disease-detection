# data_quality_assessment.py

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imagehash
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
warnings.filterwarnings('ignore')


class DiseaseDataQualityAssessment:
    """
    Comprehensive data quality assessment for medical image datasets
    """
    
    def __init__(self, dataset_path, disease_name):
        """
        Parameters:
        -----------
        dataset_path : str
            Path to disease folder
        disease_name : str
            Name of the disease (e.g., 'Cellulitis', 'Folliculitis')
        """
        self.dataset_path = Path(dataset_path)
        self.disease_name = disease_name
        
        # DEBUG: Print path information
        print(f"\n🔍 DEBUG: Checking path: {self.dataset_path}")
        print(f"🔍 Path exists: {self.dataset_path.exists()}")
        print(f"🔍 Is directory: {self.dataset_path.is_dir()}")
        
        self.image_paths = self._get_image_paths()
        self.results = {}
        
    def _get_image_paths(self):
        """Get all image paths from dataset folder"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        image_paths = []
        
        # Try recursive search first
        for ext in extensions:
            found = list(self.dataset_path.rglob(ext))
            image_paths.extend(found)
        
        # If no images found, try direct glob (non-recursive)
        if len(image_paths) == 0:
            for ext in extensions:
                found = list(self.dataset_path.glob(ext))
                image_paths.extend(found)
        
        print(f"✅ Total images found: {len(image_paths)}")
        
        # Show first few paths
        if len(image_paths) > 0:
            print(f"📄 Sample paths:")
            for p in image_paths[:3]:
                print(f"   - {p}")
        
        return image_paths
    
    # ========== 1. Dataset Source Verification ==========
    def verify_dataset_metadata(self, expected_source=None):
        """Verify dataset source and create metadata report"""
        print(f"\n{'='*60}")
        print(f"Dataset Source Verification: {self.disease_name}")
        print(f"{'='*60}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Total Images: {len(self.image_paths)}")
        
        if len(self.image_paths) == 0:
            print("❌ ERROR: No images found in this directory!")
            return None
        
        if expected_source:
            print(f"Expected Source: {expected_source}")
        
        self.results['total_images'] = len(self.image_paths)
        self.results['dataset_source'] = expected_source
        return self.results
    
    # ========== 2. Duplicate Detection ==========
    def detect_duplicates_hashing(self, hash_size=8, threshold=5):
        """Detect duplicate images using perceptual hashing"""
        print(f"\n{'='*60}")
        print(f"Duplicate Detection - Perceptual Hashing: {self.disease_name}")
        print(f"{'='*60}")
        
        if len(self.image_paths) == 0:
            print("❌ No images to process")
            return {}
        
        hashes = {'dhash': {}, 'phash': {}, 'ahash': {}}
        
        for img_path in self.image_paths:
            try:
                img = Image.open(img_path)
                hashes['dhash'][img_path.name] = imagehash.dhash(img, hash_size)
                hashes['phash'][img_path.name] = imagehash.phash(img, hash_size)
                hashes['ahash'][img_path.name] = imagehash.average_hash(img, hash_size)
            except Exception as e:
                print(f"⚠️  Error processing {img_path.name}: {e}")
        
        duplicates = {}
        for method, hash_dict in hashes.items():
            duplicates[method] = self._find_hash_duplicates(hash_dict, threshold)
        
        print(f"\nDuplicate Detection Results (threshold={threshold}):")
        for method, dups in duplicates.items():
            num_groups = len(dups)
            total_dups = sum(len(group) for group in dups.values())
            print(f"  {method.upper()}: {num_groups} groups, {total_dups} duplicates")
        
        self.results['duplicates'] = duplicates
        return duplicates
    
    def _find_hash_duplicates(self, hash_dict, threshold):
        """Helper function to find duplicates"""
        duplicates = {}
        checked = set()
        items = list(hash_dict.items())
        
        for i, (name1, hash1) in enumerate(items):
            if name1 in checked:
                continue
            similar = []
            for name2, hash2 in items[i+1:]:
                if name2 in checked:
                    continue
                if hash1 - hash2 <= threshold:
                    similar.append(name2)
                    checked.add(name2)
            if similar:
                duplicates[name1] = similar
                checked.add(name1)
        
        return duplicates
    
    # ========== 3. Image Quality Assessment ==========
    def assess_image_quality(self, blur_threshold=100, brightness_range=(40, 200), contrast_threshold=30):
        """Assess image quality"""
        print(f"\n{'='*60}")
        print(f"Image Quality Assessment: {self.disease_name}")
        print(f"{'='*60}")
        
        if len(self.image_paths) == 0:
            print("❌ No images to process")
            return pd.DataFrame()
        
        quality_data = []
        
        for img_path in self.image_paths:
            try:
                img = cv2.imread(str(img_path))
                
                if img is None:
                    print(f"⚠️  Could not read: {img_path.name}")
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Blur detection
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                is_blurry = laplacian_var < blur_threshold
                
                # Brightness
                brightness = np.mean(gray)
                is_dark = brightness < brightness_range[0]
                is_bright = brightness > brightness_range[1]
                
                # Contrast
                contrast = np.std(gray)
                is_low_contrast = contrast < contrast_threshold
                
                quality_data.append({
                    'filename': img_path.name,
                    'laplacian_variance': laplacian_var,
                    'is_blurry': is_blurry,
                    'brightness': brightness,
                    'is_dark': is_dark,
                    'is_bright': is_bright,
                    'contrast': contrast,
                    'is_low_contrast': is_low_contrast,
                    'overall_quality': 'Poor' if (is_blurry or is_dark or is_bright or is_low_contrast) else 'Good'
                })
                
            except Exception as e:
                print(f"⚠️  Error processing {img_path.name}: {e}")
        
        if len(quality_data) == 0:
            print("❌ No images were successfully processed")
            return pd.DataFrame()
        
        df_quality = pd.DataFrame(quality_data)
        
        # Summary
        print(f"\nQuality Assessment Summary:")
        print(f"  Total Images: {len(df_quality)}")
        print(f"  Blurry: {df_quality['is_blurry'].sum()} ({df_quality['is_blurry'].mean()*100:.1f}%)")
        print(f"  Dark: {df_quality['is_dark'].sum()} ({df_quality['is_dark'].mean()*100:.1f}%)")
        print(f"  Over-bright: {df_quality['is_bright'].sum()} ({df_quality['is_bright'].mean()*100:.1f}%)")
        print(f"  Low Contrast: {df_quality['is_low_contrast'].sum()} ({df_quality['is_low_contrast'].mean()*100:.1f}%)")
        print(f"  Good Quality: {(df_quality['overall_quality']=='Good').sum()} ({(df_quality['overall_quality']=='Good').mean()*100:.1f}%)")
        
        self.results['quality_assessment'] = df_quality
        return df_quality
    
    # ========== 4. Class Balance ==========
    def analyze_class_balance(self, class_folders=None):
        """Analyze class distribution"""
        print(f"\n{'='*60}")
        print(f"Class Balance Analysis: {self.disease_name}")
        print(f"{'='*60}")
        
        if class_folders:
            class_counts = {}
            for class_folder in class_folders:
                class_path = self.dataset_path / class_folder
                if class_path.exists():
                    count = len(list(class_path.rglob('*.jpg'))) + len(list(class_path.rglob('*.png')))
                    class_counts[class_folder] = count
            
            total = sum(class_counts.values())
            if total > 0:
                for class_name, count in class_counts.items():
                    print(f"  {class_name}: {count} ({count/total*100:.1f}%)")
            
            self.results['class_balance'] = class_counts
            return class_counts
        else:
            print(f"Single class: {len(self.image_paths)} images")
            self.results['class_balance'] = {self.disease_name: len(self.image_paths)}
            return {self.disease_name: len(self.image_paths)}
    
    # ========== 5. Augmentation Detection ==========
    def detect_augmented_images(self, similarity_threshold=0.95, method='ORB', max_features=500):
        """Detect augmented images"""
        print(f"\n{'='*60}")
        print(f"Augmentation Detection: {self.disease_name}")
        print(f"{'='*60}")
        
        if len(self.image_paths) == 0:
            print("❌ No images to process")
            return {}
        
        # Limit to avoid long processing
        sample_paths = self.image_paths[:min(100, len(self.image_paths))]
        print(f"Processing {len(sample_paths)} images...")
        
        features_dict = {}
        
        for img_path in sample_paths:
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (256, 256))
                
                if method == 'ORB':
                    detector = cv2.ORB_create(nfeatures=max_features)
                elif method == 'AKAZE':
                    detector = cv2.AKAZE_create()
                else:
                    detector = cv2.ORB_create(nfeatures=max_features)
                
                keypoints, descriptors = detector.detectAndCompute(img, None)
                
                if descriptors is not None and len(descriptors) > 0:
                    features_dict[img_path.name] = np.mean(descriptors, axis=0)
                    
            except Exception as e:
                pass  # Silently continue
        
        similar_groups = self._find_similar_by_features(features_dict, similarity_threshold)
        
        print(f"\nResults:")
        print(f"  Processed: {len(features_dict)}")
        print(f"  Similar groups: {len(similar_groups)}")
        
        self.results['augmented_detection'] = similar_groups
        return similar_groups
    
    def _find_similar_by_features(self, features_dict, threshold):
        """Find similar images"""
        similar_groups = {}
        checked = set()
        items = list(features_dict.items())
        
        for i, (name1, feat1) in enumerate(items):
            if name1 in checked:
                continue
            similar = []
            for name2, feat2 in items[i+1:]:
                if name2 in checked:
                    continue
                try:
                    similarity = 1 - cosine(feat1, feat2)
                    if similarity >= threshold:
                        similar.append(name2)
                        checked.add(name2)
                except:
                    continue
            if similar:
                similar_groups[name1] = similar
                checked.add(name1)
        
        return similar_groups
    
    def generate_report(self, output_dir='reports'):
        """Generate report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f'{self.disease_name}_quality_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"DATA QUALITY ASSESSMENT REPORT\n")
            f.write(f"Disease: {self.disease_name}\n")
            f.write(f"{'='*80}\n\n")
            
            for key, value in self.results.items():
                f.write(f"\n{key.upper()}:\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{value}\n\n")
        
        print(f"\n✅ Report saved: {report_file}")
        
        if 'quality_assessment' in self.results and not self.results['quality_assessment'].empty:
            csv_file = output_path / f'{self.disease_name}_quality_data.csv'
            self.results['quality_assessment'].to_csv(csv_file, index=False)
            print(f"✅ CSV saved: {csv_file}")
    
    def save_duplicates_json(self, output_dir='reports'):
        """
        Save duplicates['dhash'] as a JSON file for later processing.
        File will be: reports/<Disease>/<Disease>_duplicates_dhash.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dup_info = self.results.get('duplicates', {})
        dhash_dup = dup_info.get('dhash', {})
        
        json_path = output_path / f"{self.disease_name}_duplicates_dhash.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in dhash_dup.items()}, f, indent=2)
        
        print(f"✅ Duplicates JSON saved: {json_path}")


def run_complete_assessment(dataset_path, disease_name, **kwargs):
    """Run all assessments"""
    qa = DiseaseDataQualityAssessment(dataset_path, disease_name)
    
    if len(qa.image_paths) == 0:
        print(f"\n❌ STOPPING: No images found for {disease_name}")
        print(f"   Check path: {dataset_path}")
        return None
    
    qa.verify_dataset_metadata(kwargs.get('expected_source'))
    qa.detect_duplicates_hashing(kwargs.get('hash_size', 8), kwargs.get('dup_threshold', 5))
    qa.assess_image_quality(kwargs.get('blur_threshold', 100), kwargs.get('brightness_range', (40, 200)), kwargs.get('contrast_threshold', 30))
    qa.analyze_class_balance(kwargs.get('class_folders'))
    qa.detect_augmented_images(kwargs.get('similarity_threshold', 0.95), kwargs.get('feature_method', 'ORB'), kwargs.get('max_features', 500))
    qa.generate_report(kwargs.get('output_dir', 'reports'))
    qa.save_duplicates_json(kwargs.get('output_dir', 'reports'))
    
    return qa