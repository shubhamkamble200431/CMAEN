# Enhanced Emotion Recognition System

A state-of-the-art emotion recognition system built with PyTorch, featuring advanced attention mechanisms, hybrid CNN-EfficientNet architecture, and comprehensive data preprocessing pipelines including DDColor and SwinIR.

## 🚀 Features

- **Advanced Architecture**: Hybrid CNN-EfficientNet with multi-scale attention, emotion-specific attention, and cross-modal attention
- **Data Enhancement**: Borderline SMOTE for class balancing and comprehensive data augmentation
- **Image Preprocessing**: Integration with DDColor for colorization and SwinIR for super-resolution
- **Comprehensive Training**: Advanced loss functions, learning rate scheduling, and early stopping
- **Real-time Inference**: Support for single image, batch, and real-time webcam prediction
- **Extensive Evaluation**: Confusion matrices, ROC curves, PR curves, and t-SNE visualizations
- **Modular Design**: Clean, maintainable code structure with separate modules

## 🎯 Performance

- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **State-of-the-art Accuracy**: Enhanced hybrid architecture with attention mechanisms
- **Real-time Processing**: Optimized for both accuracy and speed
- **Robust Generalization**: Advanced regularization and data augmentation techniques

## 📁 Project Structure

```
emotion_recognition/
├── config.py              # Configuration and hyperparameters
├── dataset.py             # Dataset loading and data transformations
├── models.py              # Neural network architectures and attention mechanisms
├── utils.py               # Utility functions, loss functions, and schedulers
├── data_balancer.py       # Borderline SMOTE and data balancing
├── train.py               # Training logic and procedures
├── test.py                # Testing, evaluation, and visualization
├── inference.py           # Inference utilities and real-time prediction
├── main.py                # Main entry point and CLI interface
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── repositories/          # External repositories (DDColor, SwinIR)
│   ├── DDColor/           # DDColor repository (git submodule)
│   └── SwinIR/            # SwinIR repository (git submodule)
├── outputs/               # Generated outputs
├── saved_models/          # Trained model checkpoints
├── results/               # Training and evaluation results
└── plots/                 # Generated plots and visualizations
```

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shubhamkamble200431/CMAEN
cd emotion_recognition
```

### 2. Set up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up External Repositories (DDColor & SwinIR)

The system supports advanced image preprocessing using DDColor for colorization and SwinIR for super-resolution. These are optional but recommended for best performance.

#### DDColor Setup (Image Colorization)

```bash
# Create repositories directory
mkdir -p repositories
cd repositories

# Clone DDColor repository
git clone https://github.com/piddnad/DDColor.git
cd DDColor

# Install DDColor dependencies
pip install timm basicsr facexlib gfpgan

# Download pre-trained models (follow DDColor documentation)
# Models should be placed in DDColor/pretrain_models/
```

#### SwinIR Setup (Image Super-Resolution)

```bash
# From repositories directory
git clone https://github.com/JingyunLiang/SwinIR.git
cd SwinIR

# Install SwinIR dependencies  
pip install timm basicsr

# Download pre-trained models (follow SwinIR documentation)
# Models should be placed in SwinIR/model_zoo/
```

### 5. Configure Paths

Edit `config.py` to set your dataset paths:

```python
# Update these paths in config.py
ORIGINAL_TRAIN_DIR = "/path/to/your/original/train/dataset"
ORIGINAL_TEST_DIR = "/path/to/your/original/test/dataset"
AUGMENTED_TRAIN_DIR = "/path/to/your/augmented/train/dataset"
AUGMENTED_TEST_DIR = "/path/to/your/augmented/test/dataset"
```

## 📊 Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## 🎯 Usage

### Data Preprocessing Pipeline

#### 1. Image Enhancement with DDColor (Optional)

If you have grayscale images and want to colorize them:

```bash
cd repositories/DDColor

# Run DDColor on your dataset
python inference.py \
    --model_path pretrain_models/net_g_200000.pth \
    --input_path /path/to/grayscale/images \
    --output_path /path/to/colorized/images
```

#### 2. Image Super-Resolution with SwinIR (Optional)

To enhance image resolution:

```bash
cd repositories/SwinIR

# Run SwinIR for image super-resolution
python main_test_swinir.py \
    --task real_sr \
    --scale 4 \
    --model_path model_zoo/swinir_real_sr_x4_large.pth \
    --folder_lq /path/to/low/resolution/images \
    --folder_gt /path/to/output/high/resolution/images
```

#### 3. Data Balancing with Borderline SMOTE

Balance your dataset using advanced SMOTE techniques:

```bash
python main.py balance \
    --input-dir /path/to/original/dataset \
    --output-dir /path/to/balanced/dataset
```

### Training

Train the enhanced emotion recognition model:

```bash
# Basic training
python main.py train

# Training with custom settings
python main.py train --verbose --seed 42
```

Training features:
- **Advanced Loss Functions**: Focal loss, label smoothing, and multiple auxiliary losses
- **Attention Mechanisms**: Multi-scale, emotion-specific, and cross-modal attention
- **Regularization**: Dropout, weight decay, gradient clipping
- **Scheduling**: Cosine annealing with warm restarts
- **Early Stopping**: Automatic stopping with patience

### Testing and Evaluation

Comprehensive model evaluation:

```bash
# Full evaluation with visualizations
python main.py test

# Quick evaluation
python main.py evaluate
```

Generated outputs:
- **Confusion Matrix**: Detailed class-wise performance
- **ROC Curves**: Receiver Operating Characteristic curves for all classes
- **PR Curves**: Precision-Recall curves with Average Precision scores
- **Training Curves**: Loss and accuracy progression
- **t-SNE Visualization**: Feature space analysis comparing original vs augmented data
- **Classification Report**: Detailed metrics per class

### Inference

#### Single Image Prediction

```bash
python main.py inference --image /path/to/image.jpg
```

#### Batch Prediction

```bash
python main.py inference \
    --directory /path/to/images \
    --output results.json
```

#### Real-time Webcam Prediction

```bash
# Real-time emotion recognition from webcam
python main.py inference --real-time

# Save annotated video
python main.py inference --real-time --save-video output.avi

# Use specific camera
python main.py inference --real-time --camera-index 1
```

### Python API Usage

```python
from inference import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor('saved_models/best_model.pth')

# Single image prediction
result = predictor.predict_single('image.jpg')
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_list)

# Top-k predictions
top_predictions = predictor.get_top_k_predictions('image.jpg', k=3)
```

## 🏗️ Architecture Details

### Enhanced Hybrid Model

The system uses a novel hybrid architecture combining:

1. **Enhanced CNN Backbone**
   - Residual blocks with improved regularization
   - Multi-scale attention mechanisms
   - Emotion-specific attention heads

2. **EfficientNet Backbone**
   - Pre-trained EfficientNet-V2-S
   - Adapted for emotion recognition
   - Multi-scale feature extraction

3. **Advanced Attention Mechanisms**
   - **Multi-Scale Attention**: Captures features at different scales
   - **Emotion-Specific Attention**: Dedicated attention for each emotion class
   - **Cross-Modal Attention**: Fuses CNN and EfficientNet features

4. **Feature Pyramid Network**
   - Multi-scale feature fusion
   - Enhanced feature representation

### Training Enhancements

- **Multiple Loss Functions**: Cross-entropy, focal loss, label smoothing
- **Advanced Optimization**: AdamW with weight decay
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Regularization**: Dropout, gradient clipping, data augmentation
- **Class Balancing**: Borderline SMOTE for imbalanced datasets

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall and per-class accuracy
- **Precision, Recall, F1-Score**: Detailed per-class metrics
- **Confusion Matrix**: Class-wise confusion analysis
- **ROC-AUC**: Area under ROC curves
- **Average Precision**: PR curve analysis
- **Training Curves**: Loss and accuracy progression

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Model Configuration
INPUT_SIZE = 224              # Input image size
INPUT_CHANNELS = 3            # RGB channels
NUM_CLASSES = 7              # Number of emotion classes
MODEL_TYPE = 'enhanced_hybrid' # Model architecture

# Training Configuration
LEARNING_RATE = 0.0001       # Initial learning rate
BATCH_SIZE = 64              # Training batch size
EPOCHS = 150                 # Maximum epochs
DROPOUT_RATE = 0.5           # Dropout probability
WEIGHT_DECAY = 1e-3          # L2 regularization
PATIENCE = 15                # Early stopping patience
LABEL_SMOOTHING = 0.1        # Label smoothing factor

# Data Augmentation
# Comprehensive augmentation pipeline with emotion-preserving transforms
```

## 🚀 Advanced Features

### Data Preprocessing Pipeline

1. **Image Colorization**: DDColor integration for grayscale-to-color conversion
2. **Super-Resolution**: SwinIR integration for image quality enhancement
3. **Data Balancing**: Borderline SMOTE for handling class imbalance
4. **Advanced Augmentation**: Emotion-preserving augmentation techniques

### Real-time Processing

- **Webcam Integration**: Real-time emotion recognition from camera feed
- **Video Processing**: Batch processing of video files
- **Performance Optimization**: Efficient inference pipeline

### Visualization and Analysis

- **Training Monitoring**: Real-time training progress visualization
- **Feature Analysis**: t-SNE and PCA visualization of learned features
- **Performance Analysis**: Comprehensive evaluation metrics and plots

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DDColor**: [piddnad/DDColor](https://github.com/piddnad/DDColor) for image colorization
- **SwinIR**: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) for image super-resolution
- **EfficientNet**: Google's EfficientNet architecture
- **PyTorch**: Deep learning framework
- **FER2013**: Original emotion recognition dataset

## 📞 Support

For questions, issues, or contributions:

1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Join discussions in the repository

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{enhanced_emotion_recognition,
  title={Enhanced Emotion Recognition System with Advanced Attention Mechanisms},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/emotion_recognition}
}
```

---

**Note**: Make sure to update the DDColor and SwinIR repositories to their latest versions and follow their respective documentation for optimal performance. The repositories are kept as editable installations to allow for custom modifications and improvements.
