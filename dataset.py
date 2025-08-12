"""
Dataset module for Enhanced Emotion Recognition System
Contains dataset classes and data loading utilities
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
from .config import config

logger = logging.getLogger(__name__)

class FER2013Dataset(Dataset):
    """Enhanced FER2013 Dataset class with improved error handling"""
    
    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(config.CLASSES)}
        self.samples = []
        
        self._load_samples(max_samples_per_class)
        
    def _load_samples(self, max_samples_per_class):
        """Load all image samples from the dataset directory"""
        for class_name in config.CLASSES:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_path):
                class_samples = []
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        class_samples.append((img_path, self.class_to_idx[class_name]))
                
                if max_samples_per_class and len(class_samples) > max_samples_per_class:
                    class_samples = class_samples[:max_samples_per_class]
                
                self.samples.extend(class_samples)
        
        logger.info(f"Found {len(self.samples)} samples in {self.root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((config.INPUT_SIZE, config.INPUT_SIZE, 3), dtype=np.uint8)
        else:
            if config.INPUT_CHANNELS == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (config.INPUT_SIZE, config.INPUT_SIZE))
                image = Image.fromarray(image).convert('L')
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (config.INPUT_SIZE, config.INPUT_SIZE))
                image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        class_counts = {cls: 0 for cls in config.CLASSES}
        for _, label in self.samples:
            class_name = config.CLASSES[label]
            class_counts[class_name] += 1
        return class_counts

def get_transforms():
    """Get training and testing transforms with enhanced augmentation"""
    if config.INPUT_CHANNELS == 1:
        # Grayscale transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        # RGB transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return train_transform, test_transform

def create_data_loaders(train_dir=None, test_dir=None, batch_size=None):
    """Create training and testing data loaders with error handling"""
    train_dir = train_dir or config.TRAIN_DIR
    test_dir = test_dir or config.TEST_DIR
    batch_size = batch_size or config.BATCH_SIZE
    
    train_transform, test_transform = get_transforms()
    
    # Create datasets
    train_dataset = FER2013Dataset(train_dir, transform=train_transform)
    test_dataset = FER2013Dataset(test_dir, transform=test_transform)
    
    # Log dataset information
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Testing dataset: {len(test_dataset)} samples")
    
    # Log class distributions
    train_dist = train_dataset.get_class_distribution()
    test_dist = test_dataset.get_class_distribution()
    
    logger.info("Training set distribution:")
    for cls, count in train_dist.items():
        logger.info(f"  {cls}: {count}")
    
    logger.info("Testing set distribution:")
    for cls, count in test_dist.items():
        logger.info(f"  {cls}: {count}")
    
    # Create data loaders with error handling
    try:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=config.NUM_WORKERS, 
            pin_memory=config.PIN_MEMORY, 
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=config.NUM_WORKERS, 
            pin_memory=config.PIN_MEMORY
        )
    except Exception as e:
        logger.warning(f"Error creating data loaders with num_workers={config.NUM_WORKERS}: {e}")
        logger.info("Falling back to single-threaded data loading")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0, 
            pin_memory=False, 
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0, 
            pin_memory=False
        )
    
    return train_loader, test_loader, train_dataset, test_dataset

def get_tsne_data_loader(dataset_dir, max_samples_per_class=None):
    """Create a data loader specifically for t-SNE visualization"""
    _, test_transform = get_transforms()
    
    dataset = FER2013Dataset(
        dataset_dir, 
        transform=test_transform,
        max_samples_per_class=max_samples_per_class or (config.TSNE_SAMPLES // config.NUM_CLASSES)
    )
    
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
