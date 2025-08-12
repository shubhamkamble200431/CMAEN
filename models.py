"""
Models module for Enhanced Emotion Recognition System
Contains all neural network architectures and attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from .config import config

class MultiScaleAttention(nn.Module):
    """Multi-scale attention for capturing features at different scales"""
    
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales
        self.attentions = nn.ModuleList()
        
        for scale in scales:
            if scale == 1:
                pool = nn.Identity()
            else:
                pool = nn.AvgPool2d(kernel_size=scale, stride=scale)
            
            attention = nn.Sequential(
                pool,
                nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.attentions.append(attention)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        attended_features = []
        
        for i, attention in enumerate(self.attentions):
            scale = self.scales[i]
            attended = attention(x)
            
            if scale != 1:
                attended = F.interpolate(attended, size=(height, width), mode='bilinear', align_corners=False)
            
            attended_features.append(x * attended)
        
        # Combine multi-scale attended features
        combined = sum(attended_features) / len(attended_features)
        return combined

class EmotionSpecificAttention(nn.Module):
    """Emotion-specific attention heads for different emotions"""
    
    def __init__(self, in_channels, num_emotions=7):
        super(EmotionSpecificAttention, self).__init__()
        self.num_emotions = num_emotions
        
        # Separate attention heads for each emotion
        self.emotion_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(num_emotions)
        ])
        
        # Spatial attention for each emotion
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            ) for _ in range(num_emotions)
        ])
    
    def forward(self, x):
        emotion_features = []
        
        for i in range(self.num_emotions):
            # Channel attention
            channel_att = self.emotion_attentions[i](x)
            channel_attended = x * channel_att
            
            # Spatial attention
            spatial_att = self.spatial_attentions[i](channel_attended)
            emotion_feature = channel_attended * spatial_att
            
            emotion_features.append(emotion_feature)
        
        # Stack emotion-specific features
        stacked_features = torch.stack(emotion_features, dim=1)  # [B, num_emotions, C, H, W]
        return stacked_features

class CrossModalAttention(nn.Module):
    """Cross-attention between CNN and EfficientNet features"""
    
    def __init__(self, cnn_channels, efficient_channels, hidden_dim=512):
        super(CrossModalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Project features to same dimension
        self.cnn_proj = nn.Conv2d(cnn_channels, hidden_dim, kernel_size=1)
        self.efficient_proj = nn.Conv2d(efficient_channels, hidden_dim, kernel_size=1)
        
        # Cross-attention layers
        self.query_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, cnn_features, efficient_features):
        # Project to same dimension
        cnn_proj = self.cnn_proj(cnn_features)
        efficient_proj = self.efficient_proj(efficient_features)
        
        # Resize efficient features to match CNN features
        if cnn_proj.size()[-2:] != efficient_proj.size()[-2:]:
            efficient_proj = F.interpolate(efficient_proj, size=cnn_proj.size()[-2:], 
                                         mode='bilinear', align_corners=False)
        
        batch_size, channels, height, width = cnn_proj.size()
        
        # Query from CNN, Key and Value from EfficientNet
        query = self.query_conv(cnn_proj).view(batch_size, channels, -1).permute(0, 2, 1)
        key = self.key_conv(efficient_proj).view(batch_size, channels, -1)
        value = self.value_conv(efficient_proj).view(batch_size, channels, -1).permute(0, 2, 1)
        
        # Attention
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        # Residual connection with learnable gate
        out = self.gamma * out + cnn_proj
        
        return out

class FeaturePyramidModule(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super(FeaturePyramidModule, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
    
    def forward(self, features):
        """
        features: list of feature maps from different scales
        """
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], 
                                       mode='bilinear', align_corners=False)
        
        # Apply FPN convolutions
        fpn_outs = [fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)]
        
        return fpn_outs

class ResidualBlock(nn.Module):
    """Enhanced residual block with better regularization"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out += residual
        out = F.relu(out)
        
        return out

class EnhancedEmotionCNN(nn.Module):
    """Enhanced CNN with residual blocks and better architecture"""
    
    def __init__(self):
        super(EnhancedEmotionCNN, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(config.INPUT_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Multi-scale attention
        self.multiscale_attention = MultiScaleAttention(512)
        
        # Emotion-specific attention
        self.emotion_attention = EmotionSpecificAttention(512, config.NUM_CLASSES)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, config.DROPOUT_RATE))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, config.DROPOUT_RATE))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial_conv(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Apply multi-scale attention
        x4 = self.multiscale_attention(x4)
        
        # Apply emotion-specific attention
        emotion_features = self.emotion_attention(x4)  # [B, num_emotions, C, H, W]
        
        return x4, emotion_features, [x1, x2, x3, x4]

class EnhancedEfficientFER(nn.Module):
    """Enhanced EfficientNet with better attention"""
    
    def __init__(self):
        super(EnhancedEfficientFER, self).__init__()
        
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        
        if config.INPUT_CHANNELS != 3:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                config.INPUT_CHANNELS, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            
            if config.INPUT_CHANNELS == 1:
                with torch.no_grad():
                    self.backbone.features[0][0].weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        self.backbone.classifier = nn.Identity()
        
        # Multi-scale attention for EfficientNet features
        self.multiscale_attention = MultiScaleAttention(1280)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        features = self.backbone.features(x)
        features = self.multiscale_attention(features)
        
        return features

class EnhancedHybridModel(nn.Module):
    """Enhanced hybrid model with advanced attention mechanisms"""
    
    def __init__(self):
        super(EnhancedHybridModel, self).__init__()
        
        # Enhanced backbones
        self.cnn_backbone = EnhancedEmotionCNN()
        self.efficient_backbone = EnhancedEfficientFER()
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(512, 1280, 512)
        
        # Feature pyramid network
        self.fpn = FeaturePyramidModule([64, 128, 256, 512], 256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced classifier with multiple pathways
        self.emotion_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(256, 1)
            ) for _ in range(config.NUM_CLASSES)
        ])
        
        # Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512 + 1280, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(512, config.NUM_CLASSES)
        )
        
        # Attention weights for ensemble
        self.attention_weights = nn.Parameter(torch.ones(2))
        
    def forward(self, x):
        # CNN pathway
        cnn_features, emotion_features, pyramid_features = self.cnn_backbone(x)
        
        # EfficientNet pathway
        efficient_features = self.efficient_backbone(x)
        
        # Cross-modal attention
        cross_attended = self.cross_attention(cnn_features, efficient_features)
        
        # Global pooling for final features
        cnn_pooled = self.global_pool(cross_attended).view(cross_attended.size(0), -1)
        efficient_pooled = self.global_pool(efficient_features).view(efficient_features.size(0), -1)
        
        # Emotion-specific predictions
        emotion_logits = []
        for i, classifier in enumerate(self.emotion_classifiers):
            emotion_feature = emotion_features[:, i].mean(dim=[2, 3])  # Global average pooling
            emotion_logit = classifier(emotion_feature)
            emotion_logits.append(emotion_logit)
        
        emotion_outputs = torch.cat(emotion_logits, dim=1)
        
        # Fusion prediction
        fused_features = torch.cat([cnn_pooled, efficient_pooled], dim=1)
        fusion_output = self.fusion_classifier(fused_features)
        
        # Weighted ensemble
        attention_weights = F.softmax(self.attention_weights, dim=0)
        final_output = attention_weights[0] * emotion_outputs + attention_weights[1] * fusion_output
        
        return final_output, emotion_outputs, fusion_output

def get_model(model_type=None):
    """Factory function to get the specified model"""
    model_type = model_type or config.MODEL_TYPE
    
    if model_type == 'enhanced_hybrid':
        return EnhancedHybridModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
