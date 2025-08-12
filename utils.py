"""
Utilities module for Enhanced Emotion Recognition System
Contains loss functions, schedulers, and utility functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from .config import config

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.nll_loss(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warm restarts"""
    
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=0.1, min_lr=0.001, 
                 warmup_steps=0, gamma=1., last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def calculate_accuracy(outputs, targets):
    """Calculate accuracy for a batch"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total

def create_tsne_instance(n_components=2, random_state=42, perplexity=30, max_iterations=1000):
    """Create t-SNE instance with proper parameter handling for different scikit-learn versions"""
    import sklearn
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')))
    
    logger.info(f"Using scikit-learn version: {sklearn.__version__}")
    
    try:
        if sklearn_version >= (0, 24):
            # Use max_iter for newer versions
            return TSNE(n_components=n_components, random_state=random_state, 
                       perplexity=perplexity, max_iter=max_iterations, 
                       learning_rate='auto', init='pca')
        else:
            # Use n_iter for older versions
            return TSNE(n_components=n_components, random_state=random_state, 
                       perplexity=perplexity, n_iter=max_iterations)
    except Exception as e:
        logger.warning(f"Error creating t-SNE with specific parameters: {e}")
        # Fallback with minimal parameters
        return TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('emotion_recognition.log')
        ]
    )

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, best_val_acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'model_type': config.MODEL_TYPE,
        'config': {
            'input_size': config.INPUT_SIZE,
            'input_channels': config.INPUT_CHANNELS,
            'num_classes': config.NUM_CLASSES,
            'dropout_rate': config.DROPOUT_RATE
        }
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded: {filepath}")
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_acc', 0.0)

def get_loss_functions():
    """Get all loss functions used in training"""
    criterion = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)
    label_smooth_loss = LabelSmoothingLoss(smoothing=config.LABEL_SMOOTHING)
    
    return criterion, focal_loss, label_smooth_loss

def get_optimizer_and_scheduler(model):
    """Get optimizer and scheduler"""
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=config.SCHEDULER_FIRST_CYCLE, 
        cycle_mult=config.SCHEDULER_CYCLE_MULT,
        max_lr=config.LEARNING_RATE, 
        min_lr=config.LEARNING_RATE * config.SCHEDULER_MIN_LR_RATIO,
        warmup_steps=config.SCHEDULER_WARMUP_STEPS,
        gamma=config.SCHEDULER_GAMMA
    )
    
    return optimizer, scheduler

def extract_features_for_tsne(model, data_loader, device):
    """Extract features for t-SNE visualization with memory management"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            
            try:
                # Extract from CNN backbone
                cnn_features, _, _ = model.cnn_backbone(data)
                feature_output = model.global_pool(cnn_features).view(cnn_features.size(0), -1)
                
                features.append(feature_output.cpu().numpy())
                labels.append(target.cpu().numpy())
                
                # Clear GPU cache periodically
                if len(features) % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                logger.warning(f"Error extracting features for batch: {e}")
                continue
    
    if not features:
        logger.error("No features extracted for t-SNE")
        return None, None
        
    return np.vstack(features), np.hstack(labels)

def get_predictions_and_probabilities(model, data_loader, device):
    """Get predictions and probabilities for evaluation metrics"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            final_output, _, _ = model(data)
            probabilities = F.softmax(final_output, dim=1)
            
            all_predictions.extend(final_output.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        self.best_weights = model.state_dict()
