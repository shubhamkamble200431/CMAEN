"""
Data Balancer module for Enhanced Emotion Recognition System
Contains dataset balancing using borderline SMOTE and augmentation techniques
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import albumentations as A
import json
import shutil
from datetime import datetime
from tqdm import tqdm
import warnings
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import LabelEncoder
import logging

from .config import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FER2013DatasetBalancer:
    """Enhanced dataset balancer with SMOTE and augmentation techniques"""
    
    def __init__(self, input_dataset_path, output_dataset_path, target_size=(192, 192)):
        self.input_path = input_dataset_path
        self.output_path = output_dataset_path
        self.target_size = target_size
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(config.CLASSES)}
        self.idx_to_emotion = {idx: emotion for idx, emotion in enumerate(config.CLASSES)}
        
        # Create output directory
        os.makedirs(output_dataset_path, exist_ok=True)
        for emotion in config.CLASSES:
            os.makedirs(os.path.join(output_dataset_path, emotion), exist_ok=True)
    
    def load_dataset(self):
        """Load the FER2013 enhanced dataset"""
        logger.info("Loading FER2013 enhanced dataset...")
        
        images = []
        labels = []
        file_paths = []
        
        for emotion_idx, emotion in enumerate(config.CLASSES):
            emotion_path = os.path.join(self.input_path, emotion)
            
            if not os.path.exists(emotion_path):
                logger.warning(f"{emotion_path} does not exist")
                continue
            
            emotion_files = [f for f in os.listdir(emotion_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            logger.info(f"Loading {len(emotion_files)} images for {emotion}...")
            
            for img_file in tqdm(emotion_files, desc=f"Loading {emotion}"):
                img_path = os.path.join(emotion_path, img_file)
                
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    if image is not None:
                        # Convert BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # Resize to target size
                        image = cv2.resize(image, self.target_size)
                        
                        images.append(image)
                        labels.append(emotion_idx)
                        file_paths.append(img_path)
                
                except Exception as e:
                    logger.error(f"Error loading {img_path}: {e}")
        
        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int32)
        
        logger.info(f"Loaded {len(images)} total images")
        return images, labels, file_paths
    
    def analyze_distribution(self, labels):
        """Analyze class distribution"""
        distribution = Counter(labels)
        
        logger.info("\nDataset Distribution:")
        logger.info("-" * 40)
        for emotion_idx, count in sorted(distribution.items()):
            emotion_name = self.idx_to_emotion[emotion_idx]
            percentage = (count / len(labels)) * 100
            logger.info(f"{emotion_name:10}: {count:5d} ({percentage:5.1f}%)")
        
        # Calculate imbalance metrics
        max_count = max(distribution.values())
        min_count = min(distribution.values())
        imbalance_ratio = max_count / min_count
        
        logger.info(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
        logger.info(f"Total samples: {len(labels)}")
        
        return distribution
    
    def get_augmentation_pipeline(self, strength='medium'):
        """Define emotion-preserving augmentation pipeline with different strengths"""
        
        if strength == 'light':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=3, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ])
        elif strength == 'medium':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=3, 
                                  border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.4),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            ])
        elif strength == 'strong':
            return A.Compose([
                A.HorizontalFlip(p=0.6),
                A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, 
                                  border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.GaussNoise(var_limit=(10.0, 35.0), p=0.4),
                A.Blur(blur_limit=3, p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.4),
            ])
        else:
            raise ValueError("Strength must be 'light', 'medium', or 'strong'")
    
    def apply_borderline_smote(self, images, labels):
        """Apply Borderline SMOTE for dataset balancing"""
        logger.info("Applying Borderline SMOTE...")
        
        # Flatten images for SMOTE
        n_samples, height, width, channels = images.shape
        images_flat = images.reshape(n_samples, -1)
        
        # Apply BorderlineSMOTE
        smote = BorderlineSMOTE(
            k_neighbors=config.SMOTE_K_NEIGHBORS,
            random_state=config.SMOTE_RANDOM_STATE,
            kind='borderline-1'
        )
        
        try:
            images_resampled, labels_resampled = smote.fit_resample(images_flat, labels)
            
            # Reshape back to image format
            images_resampled = images_resampled.reshape(-1, height, width, channels)
            
            # Ensure proper data type
            images_resampled = np.clip(images_resampled, 0, 255).astype(np.uint8)
            
            logger.info(f"SMOTE completed: {len(images)} -> {len(images_resampled)} samples")
            
            return images_resampled, labels_resampled
            
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            logger.info("Falling back to augmentation-only balancing")
            return self.apply_augmentation_balancing(images, labels, method='equal', strength='medium')
    
    def apply_augmentation_balancing(self, images, labels, method='equal', strength='medium', preserve_originals=True):
        """Balance dataset using augmentation only"""
        logger.info(f"Applying augmentation-based balancing (method: {method}, strength: {strength})...")
        
        # Calculate target samples per class
        distribution = Counter(labels)
        
        if method == 'equal':
            # Balance to maximum class size
            target_per_class = max(distribution.values())
        elif method == 'proportional':
            # Balance but maintain some proportionality
            avg_samples = int(np.mean(list(distribution.values())))
            target_per_class = max(avg_samples, max(distribution.values()) // 2)
        elif method == 'moderate':
            # Less aggressive balancing
            max_samples = max(distribution.values())
            min_samples = min(distribution.values())
            target_per_class = min_samples + (max_samples - min_samples) // 2
        else:
            raise ValueError("Method must be 'equal', 'proportional', or 'moderate'")
        
        logger.info(f"Target samples per class: {target_per_class}")
        
        balanced_images = []
        balanced_labels = []
        
        augment = self.get_augmentation_pipeline(strength)
        total_generated = 0
        
        for emotion_idx in range(config.NUM_CLASSES):
            # Get samples for this emotion
            emotion_mask = labels == emotion_idx
            emotion_images = images[emotion_mask]
            current_count = len(emotion_images)
            
            if current_count == 0:
                logger.warning(f"No images found for {self.idx_to_emotion[emotion_idx]}")
                continue
            
            logger.info(f"Balancing {self.idx_to_emotion[emotion_idx]}: {current_count} -> {target_per_class}")
            
            if preserve_originals:
                # Add ALL original images first
                balanced_images.extend(emotion_images)
                balanced_labels.extend([emotion_idx] * current_count)
                logger.info(f"  Added {current_count} original images")
            
            # Calculate how many augmented images needed
            needed = target_per_class - (current_count if preserve_originals else 0)
            
            if needed > 0:
                logger.info(f"  Generating {needed} augmented images...")
                for i in tqdm(range(needed), desc=f"Augmenting {self.idx_to_emotion[emotion_idx]}"):
                    try:
                        # Randomly select base image from originals
                        base_idx = np.random.randint(0, current_count)
                        base_image = emotion_images[base_idx]
                        
                        # Ensure image is in correct format
                        if base_image.dtype != np.uint8:
                            base_image = base_image.astype(np.uint8)
                        base_image = np.clip(base_image, 0, 255)
                        
                        # Apply augmentation
                        augmented = augment(image=base_image)['image']
                        
                        # Ensure correct data type and range
                        if augmented.dtype != np.uint8:
                            augmented = augmented.astype(np.uint8)
                        augmented = np.clip(augmented, 0, 255)
                        
                        balanced_images.append(augmented)
                        balanced_labels.append(emotion_idx)
                        total_generated += 1
                        
                    except Exception as e:
                        logger.error(f"Error augmenting image {i}: {e}")
                        # Add original image as fallback
                        balanced_images.append(base_image)
                        balanced_labels.append(emotion_idx)
            else:
                logger.info(f"  No augmentation needed for {self.idx_to_emotion[emotion_idx]}")
        
        logger.info(f"Total augmented images generated: {total_generated}")
        return np.array(balanced_images), np.array(balanced_labels)
    
    def save_balanced_dataset(self, images, labels):
        """Save the balanced dataset with enhanced error handling"""
        logger.info(f"Saving balanced dataset to {self.output_path}...")
        logger.info(f"Total images to save: {len(images)}")
        
        # Ensure output directories exist
        for emotion in config.CLASSES:
            emotion_dir = os.path.join(self.output_path, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
        
        # Count samples per class for naming
        class_counts = Counter()
        saved_count = 0
        failed_count = 0
        
        for i, (image, label) in enumerate(tqdm(zip(images, labels), total=len(images), desc="Saving")):
            emotion_name = self.idx_to_emotion[label]
            class_counts[emotion_name] += 1
            
            # Create filename
            filename = f"{emotion_name}_{class_counts[emotion_name]:06d}.jpg"
            filepath = os.path.join(self.output_path, emotion_name, filename)
            
            try:
                # Validate and prepare image
                if image is None:
                    failed_count += 1
                    continue
                
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                image = np.clip(image, 0, 255)
                
                if len(image.shape) != 3 or image.shape[2] != 3:
                    failed_count += 1
                    continue
                
                # Convert RGB to BGR for cv2
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Save image
                success = cv2.imwrite(filepath, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success and os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                    saved_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving {filepath}: {e}")
                failed_count += 1
        
        logger.info(f"Saving Results:")
        logger.info(f"Successfully saved: {saved_count} images")
        logger.info(f"Failed to save: {failed_count} images")
        
        # Save statistics
        self.save_statistics(labels, class_counts)
        return saved_count, failed_count
    
    def save_statistics(self, labels, class_counts):
        """Save dataset statistics"""
        stats = {
            'total_samples': len(labels),
            'num_classes': config.NUM_CLASSES,
            'class_distribution': dict(class_counts),
            'creation_date': datetime.now().isoformat(),
            'target_size': self.target_size,
            'emotion_classes': config.CLASSES
        }
        
        stats_path = os.path.join(self.output_path, 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def visualize_distribution(self, original_labels, balanced_labels):
        """Visualize before and after distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original distribution
        orig_dist = Counter(original_labels)
        emotions = [self.idx_to_emotion[i] for i in sorted(orig_dist.keys())]
        orig_counts = [orig_dist[i] for i in sorted(orig_dist.keys())]
        
        ax1.bar(emotions, orig_counts, color='skyblue', alpha=0.7)
        ax1.set_title('Original Distribution')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Balanced distribution
        bal_dist = Counter(balanced_labels)
        bal_counts = [bal_dist[i] for i in sorted(bal_dist.keys())]
        
        ax2.bar(emotions, bal_counts, color='lightgreen', alpha=0.7)
        ax2.set_title('Balanced Distribution (Borderline SMOTE)')
        ax2.set_ylabel('Number of Samples')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_path, 'distribution_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

def run_borderline_smote_balancing(input_path, output_path, target_size=(192, 192)):
    """Run borderline SMOTE balancing on the dataset"""
    
    logger.info("="*80)
    logger.info("FER2013 DATASET BALANCER - BORDERLINE SMOTE METHOD")
    logger.info("="*80)
    
    if not os.path.exists(input_path):
        logger.error(f"Input path {input_path} does not exist!")
        return None
    
    try:
        balancer = FER2013DatasetBalancer(input_path, output_path, target_size)
        
        # Load dataset
        images, labels, _ = balancer.load_dataset()
        
        # Analyze original distribution
        logger.info("\nOriginal Distribution:")
        balancer.analyze_distribution(labels)
        
        # Apply borderline SMOTE
        start_time = datetime.now()
        balanced_images, balanced_labels = balancer.apply_borderline_smote(images, labels)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze balanced distribution
        logger.info("\nBalanced Distribution:")
        balancer.analyze_distribution(balanced_labels)
        
        # Save dataset
        saved_count, failed_count = balancer.save_balanced_dataset(balanced_images, balanced_labels)
        
        # Visualize
        balancer.visualize_distribution(labels, balanced_labels)
        
        result = {
            'method': 'Borderline SMOTE Balancing',
            'total_samples': len(balanced_labels),
            'saved_count': saved_count,
            'failed_count': failed_count,
            'processing_time': processing_time,
            'success': True
        }
        
        logger.info("✅ Borderline SMOTE balancing completed!")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in Borderline SMOTE balancing: {e}")
        return {'error': str(e), 'success': False}

if __name__ == "__main__":
    # Example usage
    input_path = "/path/to/your/input/dataset"  # Update this path
    output_path = "/path/to/your/output/dataset"  # Update this path
    
    result = run_borderline_smote_balancing(input_path, output_path)
    if result and result.get('success', False):
        logger.info(f"Successfully processed {result['saved_count']} images")
    else:
        logger.error("Processing failed")
