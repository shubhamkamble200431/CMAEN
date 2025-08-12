"""
Enhanced Emotion Recognition System

A state-of-the-art emotion recognition system built with PyTorch, featuring advanced 
attention mechanisms, hybrid CNN-EfficientNet architecture, and comprehensive 
data preprocessing pipelines.

Author: Your Name
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Import main components for easy access
from .config import config
from .models import get_model, EnhancedHybridModel
from .dataset import create_data_loaders, FER2013Dataset
from .train import train_model, EmotionTrainer
from .test import comprehensive_evaluation, ModelEvaluator
from .inference import EmotionPredictor, RealTimePredictor
from .data_balancer import run_borderline_smote_balancing, FER2013DatasetBalancer
from .utils import get_device, setup_logging, set_seed

# Define what gets imported with "from emotion_recognition import *"
__all__ = [
    # Configuration
    'config',
    
    # Models
    'get_model',
    'EnhancedHybridModel',
    
    # Dataset
    'create_data_loaders',
    'FER2013Dataset', 
    
    # Training
    'train_model',
    'EmotionTrainer',
    
    # Testing
    'comprehensive_evaluation',
    'ModelEvaluator',
    
    # Inference
    'EmotionPredictor',
    'RealTimePredictor',
    
    # Data Balancing
    'run_borderline_smote_balancing',
    'FER2013DatasetBalancer',
    
    # Utilities
    'get_device',
    'setup_logging',
    'set_seed',
]

# Package metadata
__title__ = "Enhanced Emotion Recognition System"
__description__ = "Advanced emotion recognition with hybrid architectures and attention mechanisms"
__url__ = "https://github.com/yourusername/emotion_recognition"
__download_url__ = "https://github.com/yourusername/emotion_recognition/archive/main.zip"

# Supported emotion classes
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Version info
VERSION_INFO = tuple(map(int, __version__.split('.')))
