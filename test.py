"""
Testing and Evaluation module for Enhanced Emotion Recognition System
Contains model evaluation, visualization, and performance analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import logging
from tqdm import tqdm
import json
from datetime import datetime

from .config import config
from .utils import (
    get_device, 
    load_checkpoint, 
    calculate_accuracy,
    extract_features_for_tsne,
    get_predictions_and_probabilities,
    create_tsne_instance
)
from .dataset import get_tsne_data_loader

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model, test_loader, device=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        
    def evaluate_model(self):
        """Evaluate model on test dataset"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                final_output, _, _ = self.model(data)
                _, predicted = torch.max(final_output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        return accuracy
    
    def get_detailed_predictions(self):
        """Get detailed predictions and probabilities"""
        return get_predictions_and_probabilities(self.model, self.test_loader, self.device)
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        report_dict = classification_report(
            y_true, y_pred, 
            target_names=config.CLASSES, 
            output_dict=True
        )
        
        # Print formatted report
        report_str = classification_report(
            y_true, y_pred, 
            target_names=config.CLASSES
        )
        
        logger.info("\nClassification Report:")
        logger.info(report_str)
        
        return report_dict, report_str
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=config.CLASSES, yticklabels=config.CLASSES)
        plt.title(f'Confusion Matrix - {config.MODEL_TYPE.upper()}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        return cm
    
    def plot_precision_recall_curves(self, y_true, y_prob, save_path=None):
        """Plot precision-recall curves for all classes"""
        y_true_bin = label_binarize(y_true, classes=range(config.NUM_CLASSES))
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, config.NUM_CLASSES))
        
        # Plot PR curve for each class
        for i in range(config.NUM_CLASSES):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            avg_precision = auc(recall, precision)
            
            plt.plot(recall, precision, linewidth=2, color=colors[i],
                    label=f'{config.CLASSES[i].capitalize()} (AP = {avg_precision:.3f})')
        
        # Plot macro-average
        precision_avg, recall_avg, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
        avg_precision_macro = auc(recall_avg, precision_avg)
        
        plt.plot(recall_avg, precision_avg, linewidth=3, color='black', linestyle='--',
                label=f'Macro-Average (AP = {avg_precision_macro:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Enhanced Attention Model', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"PR curves saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_true, y_prob, save_path=None):
        """Plot ROC curves for all classes"""
        y_true_bin = label_binarize(y_true, classes=range(config.NUM_CLASSES))
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, config.NUM_CLASSES))
        
        # Plot ROC curve for each class
        for i in range(config.NUM_CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, color=colors[i],
                    label=f'{config.CLASSES[i].capitalize()} (AUC = {roc_auc:.3f})')
        
        # Plot macro-average
        fpr_avg, tpr_avg, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_macro = auc(fpr_avg, tpr_avg)
        
        plt.plot(fpr_avg, tpr_avg, linewidth=3, color='black', linestyle='--',
                label=f'Macro-Average (AUC = {roc_auc_macro:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Enhanced Attention Model', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_training_results(self, results, save_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        epochs = range(1, len(results['train_losses']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, results['train_losses'], 'b-', linewidth=2, 
                    label='Training Loss', marker='o', markersize=3)
        axes[0].plot(epochs, results['val_losses'], 'r-', linewidth=2, 
                    label='Validation Loss', marker='s', markersize=3)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Find and annotate minimum validation loss
        min_val_loss_idx = np.argmin(results['val_losses'])
        min_val_loss = results['val_losses'][min_val_loss_idx]
        axes[0].annotate(f'Min Val Loss: {min_val_loss:.4f}', 
                        xy=(min_val_loss_idx + 1, min_val_loss), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Accuracy plot
        axes[1].plot(epochs, results['train_accuracies'], 'b-', linewidth=2, 
                    label='Training Accuracy', marker='o', markersize=3)
        axes[1].plot(epochs, results['val_accuracies'], 'r-', linewidth=2, 
                    label='Validation Accuracy', marker='s', markersize=3)
        axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim([0, 1])
        
        # Find and annotate maximum validation accuracy
        max_val_acc_idx = np.argmax(results['val_accuracies'])
        max_val_acc = results['val_accuracies'][max_val_acc_idx]
        axes[1].annotate(f'Max Val Acc: {max_val_acc:.4f}', 
                        xy=(max_val_acc_idx + 1, max_val_acc), 
                        xytext=(10, -10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.suptitle(f'{config.MODEL_TYPE.upper()} - Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_tsne_comparison(self, save_path=None):
        """Plot t-SNE comparison between original and augmented datasets"""
        logger.info("Generating t-SNE comparison plots...")
        
        # Load datasets for t-SNE
        original_loader = get_tsne_data_loader(config.ORIGINAL_TRAIN_DIR)
        augmented_loader = get_tsne_data_loader(config.AUGMENTED_TRAIN_DIR)
        
        # Extract features
        logger.info("Extracting features from original dataset...")
        original_features, original_labels = extract_features_for_tsne(
            self.model, original_loader, self.device
        )
        
        logger.info("Extracting features from augmented dataset...")
        augmented_features, augmented_labels = extract_features_for_tsne(
            self.model, augmented_loader, self.device
        )
        
        # Check if feature extraction was successful
        if original_features is None or augmented_features is None:
            logger.error("Feature extraction failed. Skipping t-SNE visualization.")
            return
        
        logger.info(f"Original features shape: {original_features.shape}")
        logger.info(f"Augmented features shape: {augmented_features.shape}")
        
        # Reduce sample size if too large for t-SNE
        max_samples = 2000
        if len(original_features) > max_samples:
            indices = np.random.choice(len(original_features), max_samples, replace=False)
            original_features = original_features[indices]
            original_labels = original_labels[indices]
        
        if len(augmented_features) > max_samples:
            indices = np.random.choice(len(augmented_features), max_samples, replace=False)
            augmented_features = augmented_features[indices]
            augmented_labels = augmented_labels[indices]
        
        # Apply t-SNE
        try:
            tsne1 = create_tsne_instance()
            tsne2 = create_tsne_instance()
            
            original_tsne = tsne1.fit_transform(original_features)
            augmented_tsne = tsne2.fit_transform(augmented_features)
            
            plot_title_suffix = "t-SNE"
            
        except Exception as e:
            logger.error(f"t-SNE failed: {e}, falling back to PCA")
            from sklearn.decomposition import PCA
            
            pca1 = PCA(n_components=2, random_state=42)
            pca2 = PCA(n_components=2, random_state=42)
            
            original_tsne = pca1.fit_transform(original_features)
            augmented_tsne = pca2.fit_transform(augmented_features)
            
            plot_title_suffix = "PCA"
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, config.NUM_CLASSES))
        
        # Plot original dataset
        for i, class_name in enumerate(config.CLASSES):
            mask = original_labels == i
            ax1.scatter(original_tsne[mask, 0], original_tsne[mask, 1], 
                       c=[colors[i]], label=class_name, alpha=0.7, s=20)
        
        ax1.set_title(f'Original Dataset {plot_title_suffix}\n(Grayscale Images)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'{plot_title_suffix} Component 1')
        ax1.set_ylabel(f'{plot_title_suffix} Component 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot augmented dataset
        for i, class_name in enumerate(config.CLASSES):
            mask = augmented_labels == i
            ax2.scatter(augmented_tsne[mask, 0], augmented_tsne[mask, 1], 
                       c=[colors[i]], label=class_name, alpha=0.7, s=20)
        
        ax2.set_title(f'Augmented Dataset {plot_title_suffix}\n(Colorized Images)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'{plot_title_suffix} Component 1')
        ax2.set_ylabel(f'{plot_title_suffix} Component 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"t-SNE comparison saved to {save_path}")
        
        plt.show()

def comprehensive_evaluation(model, test_loader, results=None, output_dir=None):
    """Run comprehensive evaluation of the model"""
    logger.info("Starting comprehensive model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, test_loader)
    
    # Basic evaluation
    test_accuracy = evaluator.evaluate_model()
    
    # Get detailed predictions
    predictions, probabilities, true_labels = evaluator.get_detailed_predictions()
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Generate classification report
    classification_dict, classification_str = evaluator.generate_classification_report(
        true_labels, predicted_labels
    )
    
    # Plot visualizations
    if output_dir:
        config.create_directories()  # Ensure directories exist
        
        cm_path = f"{output_dir}/{config.MODEL_TYPE}_confusion_matrix.png"
        pr_path = f"{output_dir}/{config.MODEL_TYPE}_pr_curves.png"
        roc_path = f"{output_dir}/{config.MODEL_TYPE}_roc_curves.png"
        tsne_path = f"{output_dir}/{config.MODEL_TYPE}_tsne_comparison.png"
        
        # Generate plots
        evaluator.plot_confusion_matrix(true_labels, predicted_labels, cm_path)
        evaluator.plot_precision_recall_curves(true_labels, probabilities, pr_path)
        evaluator.plot_roc_curves(true_labels, probabilities, roc_path)
        
        try:
            evaluator.plot_tsne_comparison(tsne_path)
        except Exception as e:
            logger.warning(f"t-SNE visualization failed: {e}")
        
        # Plot training results if available
        if results:
            training_path = f"{output_dir}/{config.MODEL_TYPE}_training_curves.png"
            evaluator.plot_training_results(results, training_path)
    else:
        # Just show plots without saving
        evaluator.plot_confusion_matrix(true_labels, predicted_labels)
        evaluator.plot_precision_recall_curves(true_labels, probabilities)
        evaluator.plot_roc_curves(true_labels, probabilities)
        
        try:
            evaluator.plot_tsne_comparison()
        except Exception as e:
            logger.warning(f"t-SNE visualization failed: {e}")
        
        if results:
            evaluator.plot_training_results(results)
    
    # Compile evaluation results
    evaluation_results = {
        'test_accuracy': test_accuracy,
        'classification_report': classification_dict,
        'confusion_matrix': evaluator.plot_confusion_matrix(true_labels, predicted_labels).tolist(),
        'model_type': config.MODEL_TYPE,
        'evaluation_date': datetime.now().isoformat()
    }
    
    logger.info("Comprehensive evaluation completed!")
    return evaluation_results

if __name__ == "__main__":
    # Example usage
    from .models import get_model
    from .dataset import create_data_loaders
    
    # Load model and data
    model = get_model()
    _, test_loader, _, _ = create_data_loaders()
    
    # Load trained weights
    load_checkpoint(config.get_model_save_path(), model)
    
    # Run evaluation
    results = comprehensive_evaluation(model, test_loader, output_dir=config.PLOTS_DIR)
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
