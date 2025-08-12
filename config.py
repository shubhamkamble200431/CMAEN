"""
Configuration module for Enhanced Emotion Recognition System
Contains all hyperparameters and paths configuration
"""

import os
from pathlib import Path

class Config:
    """Configuration class containing all hyperparameters and settings"""
    
    # Dataset paths - Update these paths according to your setup
    ORIGINAL_TRAIN_DIR = "/media/admin1/DL/shubham/FINAL_SCRIPTS_FER/dataset/train"
    ORIGINAL_TEST_DIR = "/media/admin1/DL/shubham/FINAL_SCRIPTS_FER/dataset/test"
    AUGMENTED_TRAIN_DIR = "/media/admin1/DL/shubham/FINAL_SCRIPTS_FER/train_8k_color"
    AUGMENTED_TEST_DIR = "/media/admin1/DL/shubham/FINAL_SCRIPTS_FER/test_color"
    
    # Active dataset paths
    TRAIN_DIR = AUGMENTED_TRAIN_DIR
    TEST_DIR = AUGMENTED_TEST_DIR
    
    # Model configuration
    INPUT_SIZE = 224
    INPUT_CHANNELS = 3
    NUM_CLASSES = 7
    CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Training hyperparameters
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    EPOCHS = 150
    DROPOUT_RATE = 0.5
    WEIGHT_DECAY = 1e-3
    PATIENCE = 15
    LABEL_SMOOTHING = 0.1
    GRADIENT_CLIP = 1.0
    
    # Model selection
    MODEL_TYPE = 'enhanced_hybrid'
    
    # Visualization parameters
    TSNE_SAMPLES = 1000
    FIGURE_SIZE = (15, 10)
    DPI = 300
    
    # Data loading parameters
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # Scheduler parameters
    SCHEDULER_FIRST_CYCLE = 30
    SCHEDULER_CYCLE_MULT = 2.0
    SCHEDULER_MIN_LR_RATIO = 0.01
    SCHEDULER_WARMUP_STEPS = 5
    SCHEDULER_GAMMA = 0.8
    
    # Loss function weights
    FOCAL_LOSS_ALPHA = 1
    FOCAL_LOSS_GAMMA = 2
    EMOTION_LOSS_WEIGHT = 0.2
    FUSION_LOSS_WEIGHT = 0.2
    FOCAL_LOSS_WEIGHT = 0.3
    
    # Data balancing configuration
    SMOTE_K_NEIGHBORS = 5
    SMOTE_RANDOM_STATE = 42
    
    # Preprocessing paths (DDColor and SwinIR)
    DDCOLOR_REPO_PATH = "./repositories/DDColor"
    SWINIR_REPO_PATH = "./repositories/SwinIR"
    
    # Output directories
    OUTPUT_DIR = "./outputs"
    MODEL_SAVE_DIR = "./saved_models"
    RESULTS_DIR = "./results"
    PLOTS_DIR = "./plots"
    
    @classmethod
    def create_directories(cls):
        """Create necessary output directories"""
        directories = [
            cls.OUTPUT_DIR,
            cls.MODEL_SAVE_DIR,
            cls.RESULTS_DIR,
            cls.PLOTS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_save_path(cls):
        """Get the path for saving the best model"""
        return os.path.join(cls.MODEL_SAVE_DIR, f'best_{cls.MODEL_TYPE}_model_{cls.INPUT_SIZE}x{cls.INPUT_SIZE}.pth')
    
    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist"""
        required_paths = [cls.TRAIN_DIR, cls.TEST_DIR]
        missing_paths = []
        
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            raise FileNotFoundError(f"Missing required paths: {missing_paths}")
        
        return True

# Create a global config instance
config = Config()
