"""
Training module for Enhanced Emotion Recognition System
Contains training logic, loss computation, and model training procedures
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

from .config import config
from .utils import (
    get_loss_functions, 
    get_optimizer_and_scheduler, 
    calculate_accuracy,
    save_checkpoint,
    get_device
)

logger = logging.getLogger(__name__)

class EmotionTrainer:
    """Enhanced trainer class for emotion recognition model"""
    
    def __init__(self, model, train_loader, val_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize loss functions
        self.criterion, self.focal_loss, self.label_smooth_loss = get_loss_functions()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.model)
        
        # Training state
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    def compute_combined_loss(self, final_output, emotion_outputs, fusion_output, target):
        """Compute combined loss from multiple outputs"""
        # Main loss with label smoothing
        main_loss = self.label_smooth_loss(final_output, target)
        
        # Focal loss for class imbalance
        focal_loss_val = self.focal_loss(final_output, target)
        
        # Auxiliary losses
        emotion_loss = self.criterion(emotion_outputs, target)
        fusion_loss = self.criterion(fusion_output, target)
        
        # Combined loss
        total_loss = (main_loss + 
                     config.FOCAL_LOSS_WEIGHT * focal_loss_val + 
                     config.EMOTION_LOSS_WEIGHT * emotion_loss + 
                     config.FUSION_LOSS_WEIGHT * fusion_loss)
        
        return total_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        num_batches = 0
        
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} - Training")
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            final_output, emotion_outputs, fusion_output = self.model(data)
            
            # Compute combined loss
            loss = self.compute_combined_loss(final_output, emotion_outputs, fusion_output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP)
            
            # Update weights
            self.optimizer.step()
            
            # Calculate metrics
            batch_loss = loss.item()
            batch_acc = calculate_accuracy(final_output, target)
            
            total_train_loss += batch_loss
            total_train_acc += batch_acc
            num_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Acc': f'{batch_acc:.4f}'
            })
        
        avg_train_loss = total_train_loss / num_batches
        avg_train_acc = total_train_acc / num_batches
        
        return avg_train_loss, avg_train_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        num_batches = 0
        
        val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} - Validation")
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(self.device), target.to(self.device)
                final_output, emotion_outputs, fusion_output = self.model(data)
                
                # Use standard cross-entropy for validation
                loss = self.criterion(final_output, target)
                
                batch_loss = loss.item()
                batch_acc = calculate_accuracy(final_output, target)
                
                total_val_loss += batch_loss
                total_val_acc += batch_acc
                num_batches += 1
                
                val_pbar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'Acc': f'{batch_acc:.4f}'
                })
        
        avg_val_loss = total_val_loss / num_batches
        avg_val_acc = total_val_acc / num_batches
        
        return avg_val_loss, avg_val_acc
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting enhanced training with {config.MODEL_TYPE} model...")
        logger.info(f"Training for {config.EPOCHS} epochs with patience {config.PATIENCE}")
        
        start_time = datetime.now()
        
        for epoch in range(config.EPOCHS):
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.2e}"
            )
            
            # Save best model and early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                # Save checkpoint
                save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    self.best_val_acc, 
                    config.get_model_save_path()
                )
                logger.info(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Training summary
        results = {
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'training_time': training_time,
            'total_epochs': len(self.train_losses)
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        return results

def train_model(model, train_loader, val_loader, device=None):
    """Convenience function to train a model"""
    trainer = EmotionTrainer(model, train_loader, val_loader, device)
    return trainer.train()

class ResumeTraining:
    """Utility class for resuming training from checkpoint"""
    
    def __init__(self, model, optimizer, checkpoint_path):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
    
    def load_checkpoint(self):
        """Load checkpoint and return epoch and best accuracy"""
        try:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            
            logger.info(f"Resumed training from epoch {epoch} with best accuracy {best_val_acc:.4f}")
            return epoch, best_val_acc
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0, 0.0

def validate_training_setup(model, train_loader, val_loader):
    """Validate training setup before starting training"""
    device = get_device()
    
    # Test model forward pass
    try:
        model = model.to(device)
        sample_batch = next(iter(train_loader))
        sample_data, sample_targets = sample_batch[0].to(device), sample_batch[1].to(device)
        
        with torch.no_grad():
            outputs = model(sample_data)
            
        logger.info("✅ Model forward pass successful")
        logger.info(f"Input shape: {sample_data.shape}")
        logger.info(f"Output shapes: {[out.shape for out in outputs]}")
        
    except Exception as e:
        logger.error(f"❌ Model forward pass failed: {e}")
        raise
    
    # Check data loaders
    logger.info(f"✅ Training batches: {len(train_loader)}")
    logger.info(f"✅ Validation batches: {len(val_loader)}")
    
    # Check class distribution in a few batches
    class_counts = torch.zeros(config.NUM_CLASSES)
    for i, (_, labels) in enumerate(train_loader):
        if i >= 5:  # Check first 5 batches
            break
        for label in labels:
            class_counts[label] += 1
    
    logger.info("Sample class distribution in training batches:")
    for i, count in enumerate(class_counts):
        logger.info(f"  {config.CLASSES[i]}: {int(count)}")
    
    logger.info("✅ Training setup validation completed")

if __name__ == "__main__":
    # Example usage (would normally be called from main.py)
    from .models import get_model
    from .dataset import create_data_loaders
    
    # Create model and data loaders
    model = get_model()
    train_loader, val_loader, _, _ = create_data_loaders()
    
    # Validate setup
    validate_training_setup(model, train_loader, val_loader)
    
    # Train model
    results = train_model(model, train_loader, val_loader)
    print(f"Training completed with best accuracy: {results['best_val_acc']:.4f}")
