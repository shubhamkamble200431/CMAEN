# CMAEN: Cross-Modal Attention Emotion Network

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ICVGIP'25-red.svg)](https://doi.org/10.1145/nnnnnnn.nnnnnnn)

A state-of-the-art facial emotion recognition system that achieves **71.98% accuracy** on FER2013 dataset through innovative hybrid architecture, advanced attention mechanisms, and comprehensive data preprocessing pipelines.

## 🎯 Key Features

- **Novel Hybrid Architecture**: Custom Emotion-CNN (E-CNN) + EfficientNetV2S with cross-modal attention
- **Advanced Data Enhancement**: Super-resolution (SwinIR) + Colorization (DDColor) + Balanced sampling (BorderlineSMOTE)
- **Sophisticated Attention Mechanisms**: Multi-scale, emotion-specific, and cross-modal attention
- **Comprehensive Preprocessing**: Face enhancement, colorization, and class balancing
- **Real-time Inference**: Support for single images, batch processing, and webcam prediction
- **Extensive Evaluation**: Detailed performance metrics, visualizations, and ablation studies

## 📊 Performance

| Model Configuration | Accuracy | F1-Score | Key Features |
|---------------------|----------|----------|--------------|
| **CMAEN (Enhanced)** | **71.98%** | **0.72** | Full pipeline with all enhancements |
| CMAEN (No Enhancement) | 68.17% | 0.68 | Baseline without preprocessing |
| **Improvement** | **+3.81%** | **+0.04** | Enhancement pipeline benefit |

### Emotion-wise Performance
| Emotion | Precision | Recall | F1-Score | AUC-ROC |
|---------|-----------|---------|----------|---------|
| Happy | 0.86 | 0.89 | **0.88** | 0.968 |
| Surprise | 0.82 | 0.85 | **0.83** | 0.964 |
| Disgust | 0.93 | 0.68 | 0.79 | 0.944 |
| Neutral | 0.65 | 0.72 | 0.68 | 0.901 |
| Angry | 0.64 | 0.64 | 0.64 | 0.896 |
| Sad | 0.60 | 0.60 | 0.60 | 0.868 |
| Fear | 0.66 | 0.54 | 0.59 | 0.847 |

## 🏗️ Architecture Overview

### Cross-Modal Attention Emotion Network (CMAEN)

CMAEN introduces a revolutionary parallel pathway architecture that combines:

1. **Stage 1: Parallel Feature Extraction**
   - **E-CNN Pathway**: Custom emotion-focused CNN with residual blocks and multi-scale attention
   - **EfficientNetV2S Pathway**: Robust foundation model with domain adaptation

2. **Stage 2: Cross-Modal Attention & Fusion**
   - **Dynamic Information Exchange**: Query-key-value attention between pathways
   - **Learnable Gate Integration**: Adaptive contribution control
   - **Feature Pyramid Network**: Multi-scale feature processing

3. **Stage 3: Multiple Classification Strategy**
   - **Emotion-Specific Heads**: Dedicated classifiers for each emotion
   - **Fusion Classification**: Global feature-based prediction
   - **Adaptive Ensemble**: Learnable attention-weighted final prediction

### Pipeline Overview
![CMAEN Main Pipeline](images/pipeline.drawio.png)
*Complete CMAEN processing pipeline from data enhancement to final emotion prediction*

### Architecture Stages

#### Stage 1: Parallel Feature Extraction
![Stage 1 Architecture](images/stage1.drawio.png)
*Parallel processing pathways: E-CNN with emotion-specific attention and EfficientNetV2S with foundation features*

#### Stage 2: Cross-Modal Attention & Fusion
![Stage 2 Architecture](images/stage2.drawio.png)
*Cross-modal attention mechanism enabling dynamic information exchange and multi-pathway classification*

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 20GB+ free disk space

### 1. Clone Repository
```bash
git clone https://github.com/shubhamkamble200431/CMAEN.git
cd CMAEN
```

### 2. Create Virtual Environment
```bash
python -m venv cmaen_env

# Windows
cmaen_env\Scripts\activate

# Linux/macOS  
source cmaen_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup External Repositories

#### DDColor (Image Colorization)
```bash
mkdir -p repositories
cd repositories

git clone https://github.com/piddnad/DDColor.git
cd DDColor
pip install timm basicsr facexlib gfpgan

# Download pretrained models (follow DDColor documentation)
mkdir -p pretrain_models
# Place net_g_200000.pth in pretrain_models/
cd ../..
```

#### SwinIR (Super-Resolution)
```bash
cd repositories
git clone https://github.com/JingyunLiang/SwinIR.git
cd SwinIR
pip install timm basicsr

# Download pretrained models (follow SwinIR documentation)  
mkdir -p model_zoo
# Place swinir_real_sr_x4_large.pth in model_zoo/
cd ../..
```

### 5. Configure Dataset Paths
Edit `config.py` with your dataset paths:
```python
# Dataset Paths
ORIGINAL_TRAIN_DIR = "/path/to/your/fer2013/train"
ORIGINAL_TEST_DIR = "/path/to/your/fer2013/test"
AUGMENTED_TRAIN_DIR = "/path/to/enhanced/train"  
AUGMENTED_TEST_DIR = "/path/to/enhanced/test"
```

## 📁 Project Structure

```
CMAEN/
├── 📁 config.py              # Configuration and hyperparameters
├── 📁 dataset.py             # Dataset loading and transformations
├── 📁 models.py              # CMAEN architecture and attention mechanisms
├── 📁 utils.py               # Utility functions and loss functions
├── 📁 data_balancer.py       # BorderlineSMOTE implementation
├── 📁 train.py               # Training procedures and optimization
├── 📁 test.py                # Testing, evaluation, and visualization
├── 📁 inference.py           # Real-time inference and prediction
├── 📁 main.py                # Main entry point and CLI interface
├── 📁 requirements.txt       # Python dependencies
├── 📁 repositories/          # External repositories
│   ├── 📁 DDColor/           # Image colorization
│   └── 📁 SwinIR/            # Super-resolution
├── 📁 outputs/               # Generated outputs
├── 📁 saved_models/          # Trained model checkpoints
├── 📁 results/               # Training and evaluation results
└── 📁 plots/                 # Visualizations and plots
```

## 🚀 Quick Start

### Data Preprocessing Pipeline

#### 1. Super-Resolution Enhancement
```bash
cd repositories/SwinIR
python main_test_swinir.py \
    --task real_sr \
    --scale 4 \
    --model_path model_zoo/swinir_real_sr_x4_large.pth \
    --folder_lq /path/to/fer2013/48x48 \
    --folder_gt /path/to/enhanced/192x192
```

#### 2. Image Colorization  
```bash
cd repositories/DDColor
python inference.py \
    --model_path pretrain_models/net_g_200000.pth \
    --input_path /path/to/grayscale/images \
    --output_path /path/to/colorized/images
```

#### 3. Dataset Balancing
```bash
python main.py balance \
    --input-dir /path/to/enhanced/dataset \
    --output-dir /path/to/balanced/dataset
```

### Training

#### Basic Training
```bash
python main.py train
```

#### Advanced Training Configuration
```bash
python main.py train \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --verbose \
    --seed 42
```

#### Training Features:
- **Multi-Loss Framework**: Cross-entropy + Focal loss + Label smoothing + Emotion-specific loss
- **Advanced Optimization**: AdamW with weight decay and gradient clipping
- **Smart Scheduling**: Cosine annealing with warm restarts
- **Early Stopping**: Patience-based stopping with best model saving
- **Mixed Precision**: Automatic mixed precision training for efficiency

### Evaluation & Testing

#### Comprehensive Evaluation
```bash
python main.py test
```

Generated outputs:
- 📊 **Confusion Matrix**: Detailed class-wise analysis
- 📈 **ROC Curves**: Multi-class ROC analysis with AUC scores
- 📉 **PR Curves**: Precision-Recall curves with Average Precision
- 📋 **Classification Report**: Per-class precision, recall, F1-score
- 🎨 **Training Curves**: Loss and accuracy progression plots
- 🔍 **t-SNE Visualization**: Feature space analysis

### Inference

#### Single Image Prediction
```bash
python main.py inference --image /path/to/image.jpg
```

#### Batch Prediction
```bash
python main.py inference \
    --directory /path/to/images \
    --output predictions.json \
    --batch-size 32
```

#### Real-time Webcam
```bash
# Basic webcam inference
python main.py inference --real-time

# Save annotated video
python main.py inference --real-time --save-video output.avi

# Use specific camera
python main.py inference --real-time --camera-index 1
```

## 💻 Python API Usage

```python
from inference import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor('saved_models/best_model.pth')

# Single image prediction
result = predictor.predict_single('image.jpg')
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.3f}")

# Get top-k predictions
top_3 = predictor.get_top_k_predictions('image.jpg', k=3)
for i, (emotion, confidence) in enumerate(top_3):
    print(f"{i+1}. {emotion}: {confidence:.3f}")

# Batch processing
image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_list)

# Real-time webcam processing
predictor.real_time_webcam(
    camera_index=0,
    save_video=True,
    output_path='webcam_emotions.avi'
)
```

## 🔬 Technical Details

### Enhanced Hybrid Architecture

#### E-CNN (Emotion-CNN) Features:
- **Residual Architecture**: 4 residual blocks with skip connections
- **Multi-Scale Attention**: Parallel attention at scales {1, 2, 4}
- **Emotion-Specific Attention**: Dedicated heads for 7 emotions
- **Advanced Regularization**: Dropout, batch normalization, gradient clipping

#### EfficientNetV2S Integration:
- **Foundation Model**: Pre-trained EfficientNetV2S backbone
- **Domain Adaptation**: Modified input/output layers for emotion recognition
- **Feature Enhancement**: Custom attention modules for emotional features

#### Cross-Modal Attention Mechanism:
```python
# Query-Key-Value attention computation
Q = Conv1x1(E_CNN_features)        # Queries from E-CNN
K = Conv1x1(EfficientNet_features) # Keys from EfficientNet  
V = Conv1x1(EfficientNet_features) # Values from EfficientNet

# Attention weights and feature fusion
Attention_weights = Softmax(Q @ K^T / sqrt(d_k))
Cross_modal_features = Attention_weights @ V

# Learnable gate integration
Output = γ * Cross_modal_features + E_CNN_features
```

### Data Enhancement Pipeline

#### 1. Face Super-Resolution (SwinIR)
- **Input**: 48×48 grayscale FER2013 images
- **Output**: 192×192 high-resolution images
- **Architecture**: Swin Transformer with residual connections
- **Scaling Factor**: 4× enhancement

#### 2. Facial Colorization (DDColor)
- **Method**: Transformer-based multi-scale decoding
- **Features**: Learnable color queries with cross-attention
- **Output**: Natural, emotion-preserving colorization

#### 3. Class Balancing (BorderlineSMOTE)
- **Target**: Balanced 7-class distribution
- **Method**: Boundary-focused synthetic sample generation
- **Features**: Emotion-preserving augmentation constraints

### Training Configuration

```python
# Model Architecture
MODEL_TYPE = 'cmaen_hybrid'
INPUT_SIZE = 224
NUM_CLASSES = 7
DROPOUT_RATE = 0.5

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 150
WEIGHT_DECAY = 1e-3
PATIENCE = 15

# Loss Configuration
LABEL_SMOOTHING = 0.1
FOCAL_LOSS_ALPHA = 1.0
FOCAL_LOSS_GAMMA = 2.0

# Scheduler Settings
SCHEDULER_T_0 = 30        # First cycle epochs
SCHEDULER_T_MULT = 2.0    # Cycle multiplier
SCHEDULER_ETA_MIN = 1e-6  # Minimum learning rate
```

## 📈 Experimental Results

### Comparison with State-of-the-Art

| Method | Year | Accuracy (%) | Notes |
|--------|------|-------------|-------|
| **Data-Centric Reclassified** | 2023 | **86.7** | Manual data cleaning |
| DCNN Ensemble | 2022 | 76.7 | Ensemble approach |
| VGGNet SOTA | 2021 | 73.3 | Optimized VGG |
| EfficientNet-XGBoost | 2024 | 72.5 | Hybrid ML approach |
| **CMAEN (Ours - Enhanced)** | 2025 | **71.98** | **Novel hybrid architecture** |
| AA-DCN | 2024 | 70.1 | Attention-based CNN |
| CMAEN (No Enhancement) | 2025 | 68.17 | Baseline comparison |

### Ablation Studies

| Component | Accuracy | Δ | Contribution |
|-----------|----------|---|-------------|
| **Full CMAEN** | **71.98%** | - | Complete system |
| w/o Cross-Modal Attention | 70.34% | -1.64% | Attention importance |
| w/o Emotion-Specific Heads | 69.87% | -2.11% | Specialized processing |
| w/o Data Enhancement | 68.17% | -3.81% | Preprocessing impact |
| w/o Multi-Scale Attention | 67.93% | -4.05% | Scale significance |

### Training Efficiency

| Metric | Value | Hardware |
|--------|--------|----------|
| **Training Time** | 9.4 hours | NVIDIA Quadro P5000 |
| **GPU Memory** | ~14GB | 16GB VRAM |
| **Model Size** | 47.3MB | Compressed |
| **Inference Speed** | 23ms/image | Single image |
| **Batch Throughput** | 2.8s/batch | 64 images |

## 🎨 Visualization Examples

### Model Predictions
![Prediction Examples](docs/images/predictions_sample.png)
*Sample predictions showing confidence scores across different emotions*

### Training Progress
![Training Curves](docs/images/training_curves.png)
*Training and validation curves showing convergence*

### Attention Visualization
![Attention Maps](docs/images/attention_maps.png)
*Cross-modal attention maps highlighting important facial regions*

## 🔄 Dataset Requirements

### FER2013 Dataset Structure
```
dataset/
├── train/
│   ├── angry/           # 4,953 samples → 7,000 (balanced)
│   ├── disgust/         # 547 samples → 7,000 (balanced)  
│   ├── fear/            # 5,121 samples → 7,000 (balanced)
│   ├── happy/           # 8,989 samples → 7,000 (balanced)
│   ├── neutral/         # 6,198 samples → 7,000 (balanced)
│   ├── sad/             # 6,077 samples → 7,000 (balanced)
│   └── surprise/        # 4,002 samples → 7,000 (balanced)
└── test/
    ├── angry/           # 958 samples
    ├── disgust/         # 111 samples
    ├── fear/            # 1,024 samples  
    ├── happy/           # 1,774 samples
    ├── neutral/         # 1,233 samples
    ├── sad/             # 1,247 samples
    └── surprise/        # 831 samples
```

### Data Enhancement Effects
- **Original**: 48×48 grayscale, severely imbalanced
- **Enhanced**: 192×192 RGB, class-balanced with 49,000 training samples
- **Augmented**: Emotion-preserving transformations during training

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas for Contribution
1. **Architecture Improvements**: Novel attention mechanisms, better fusion strategies
2. **Data Enhancement**: New preprocessing techniques, better augmentation methods  
3. **Optimization**: Efficiency improvements, mobile deployment
4. **Evaluation**: New metrics, visualization techniques, ablation studies
5. **Documentation**: Tutorials, examples, code documentation

### Development Setup
```bash
# Clone for development
git clone https://github.com/shubhamkamble200431/CMAEN.git
cd CMAEN

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
```

### Pull Request Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass
5. Update documentation as needed
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 Citation

If you use CMAEN in your research, please cite our paper:

```bibtex
@inproceedings{kamble2025cmaen,
  title={CMAEN: Cross-Modal Attention Emotion Network for Enhanced Facial Expression Recognition},
  author={Kamble, Shubham and others},
  booktitle={Proceedings of 16th Indian Conference on Computer Vision, Graphics and Image Processing},
  pages={1--9},
  year={2025},
  publisher={ACM},
  address={Mandi, India},
  doi={10.1145/nnnnnnn.nnnnnnn}
}
```

## 🙏 Acknowledgments

- **DDColor Team**: [piddnad/DDColor](https://github.com/piddnad/DDColor) for excellent colorization framework
- **SwinIR Team**: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) for super-resolution capabilities  
- **EfficientNet**: Google Research for the EfficientNet architecture family
- **FER2013 Dataset**: Original dataset creators and maintainers
- **PyTorch Team**: For the deep learning framework
- **Research Community**: All researchers advancing facial emotion recognition

## 📞 Support & Contact

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/shubhamkamble200431/CMAEN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shubhamkamble200431/CMAEN/discussions)
- **Email**: Create an issue for technical support

### Community
- ⭐ **Star** the repository if you find it useful
- 🐛 **Report bugs** via GitHub Issues  
- 💡 **Suggest features** via GitHub Discussions
- 📚 **Improve documentation** via Pull Requests

### Research Collaboration
For research collaborations, commercial usage, or custom implementations, please reach out through the repository's issue system.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-party Licenses
- DDColor: Apache License 2.0
- SwinIR: Apache License 2.0  
- EfficientNet: Apache License 2.0
- PyTorch: BSD License

---

<div align="center">

**🔥 Advancing Facial Emotion Recognition with Cross-Modal Attention 🔥**

*Combining the power of specialized networks with foundation models*

[![GitHub stars](https://img.shields.io/github/stars/shubhamkamble200431/CMAEN.svg?style=social&label=Star)](https://github.com/shubhamkamble200431/CMAEN)
[![GitHub forks](https://img.shields.io/github/forks/shubhamkamble200431/CMAEN.svg?style=social&label=Fork)](https://github.com/shubhamkamble200431/CMAEN/fork)

</div>
