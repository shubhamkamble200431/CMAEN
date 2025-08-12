"""
Inference module for Enhanced Emotion Recognition System
Contains inference utilities for single image and batch prediction
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import logging
from pathlib import Path
import time

from .config import config
from .models import get_model
from .dataset import get_transforms
from .utils import get_device, load_checkpoint

logger = logging.getLogger(__name__)

class EmotionPredictor:
    """Enhanced emotion predictor for inference"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device or get_device()
        self.model = get_model().to(self.device)
        self.model_path = model_path or config.get_model_save_path()
        
        # Load transforms
        _, self.transform = get_transforms()
        
        # Load model weights
        self.load_model()
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info(f"EmotionPredictor initialized on {self.device}")
    
    def load_model(self):
        """Load trained model weights"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {self.model_path}")
            
            # Log model info if available
            if 'best_val_acc' in checkpoint:
                logger.info(f"Model best validation accuracy: {checkpoint['best_val_acc']:.4f}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_input):
        """Preprocess image for inference
        
        Args:
            image_input: Can be file path (str), PIL Image, or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image from {image_input}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                image = image_input
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume BGR format from cv2, convert to RGB
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = np.array(image_input)
                
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Resize image
            image = cv2.resize(image, (config.INPUT_SIZE, config.INPUT_SIZE))
            
            # Convert to PIL Image for transforms
            if config.INPUT_CHANNELS == 1:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = Image.fromarray(image).convert('L')
            else:
                image = Image.fromarray(image)
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict_single(self, image_input, return_probabilities=True, return_confidence=True):
        """Predict emotion for a single image
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
            return_probabilities: Whether to return class probabilities
            return_confidence: Whether to return prediction confidence
            
        Returns:
            dict: Prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_input).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs, _, _ = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Prepare results
            results = {
                'predicted_emotion': config.CLASSES[predicted_class],
                'predicted_class_index': predicted_class,
                'inference_time': time.time() - start_time
            }
            
            if return_confidence:
                results['confidence'] = confidence
            
            if return_probabilities:
                results['class_probabilities'] = {
                    config.CLASSES[i]: probabilities[0, i].item() 
                    for i in range(config.NUM_CLASSES)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_batch(self, image_list, batch_size=32):
        """Predict emotions for a batch of images
        
        Args:
            image_list: List of image paths or image arrays
            batch_size: Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        logger.info(f"Processing {len(image_list)} images in batches of {batch_size}")
        
        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for img in batch_images:
                try:
                    tensor = self.preprocess_image(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.warning(f"Failed to process image {img}: {e}")
                    # Add dummy result
                    results.append({
                        'predicted_emotion': 'unknown',
                        'predicted_class_index': -1,
                        'confidence': 0.0,
                        'error': str(e)
                    })
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs, _, _ = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
                confidences = probabilities.gather(1, predicted_classes.unsqueeze(1)).squeeze(1)
            
            # Process batch results
            for j, (pred_class, confidence) in enumerate(zip(predicted_classes, confidences)):
                batch_result = {
                    'predicted_emotion': config.CLASSES[pred_class.item()],
                    'predicted_class_index': pred_class.item(),
                    'confidence': confidence.item(),
                    'class_probabilities': {
                        config.CLASSES[k]: probabilities[j, k].item() 
                        for k in range(config.NUM_CLASSES)
                    }
                }
                results.append(batch_result)
        
        return results
    
    def predict_from_directory(self, directory_path, output_file=None):
        """Predict emotions for all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            output_file: Optional path to save results JSON
            
        Returns:
            dict: Mapping of image names to predictions
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for file_path in Path(directory_path).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        logger.info(f"Found {len(image_files)} images in {directory_path}")
        
        if not image_files:
            logger.warning("No image files found")
            return {}
        
        # Predict for all images
        results_list = self.predict_batch(image_files)
        
        # Create mapping with relative paths
        results_dict = {}
        for img_path, result in zip(image_files, results_list):
            relative_path = os.path.relpath(img_path, directory_path)
            results_dict[relative_path] = result
        
        # Save results if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results_dict
    
    def get_top_k_predictions(self, image_input, k=3):
        """Get top-k emotion predictions
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
            k: Number of top predictions to return
            
        Returns:
            list: List of top-k predictions with emotions and probabilities
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_input).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs, _, _ = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], k)
            
            top_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                top_predictions.append({
                    'emotion': config.CLASSES[idx.item()],
                    'probability': prob.item(),
                    'class_index': idx.item()
                })
            
            return top_predictions
            
        except Exception as e:
            logger.error(f"Error getting top-k predictions: {e}")
            raise

def predict_single_image(image_path, model_path=None):
    """Convenience function for single image prediction"""
    predictor = EmotionPredictor(model_path)
    return predictor.predict_single(image_path)

def predict_batch_images(image_list, model_path=None, batch_size=32):
    """Convenience function for batch prediction"""
    predictor = EmotionPredictor(model_path)
    return predictor.predict_batch(image_list, batch_size)

class RealTimePredictor:
    """Real-time emotion prediction from webcam or video stream"""
    
    def __init__(self, model_path=None, camera_index=0):
        self.predictor = EmotionPredictor(model_path)
        self.camera_index = camera_index
        self.cap = None
        
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        
        logger.info(f"Camera {self.camera_index} started")
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera stopped")
    
    def predict_frame(self, frame):
        """Predict emotion for a single frame"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.predictor.predict_single(frame_rgb)
            return result
        except Exception as e:
            logger.error(f"Error predicting frame: {e}")
            return None
    
    def run_real_time_prediction(self, display=True, save_video=None):
        """Run real-time emotion prediction
        
        Args:
            display: Whether to display the video window
            save_video: Path to save annotated video (optional)
        """
        if not self.cap:
            self.start_camera()
        
        # Video writer for saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Predict emotion
                result = self.predict_frame(frame)
                
                if result:
                    # Annotate frame
                    emotion = result['predicted_emotion']
                    confidence = result['confidence']
                    
                    # Add text overlay
                    text = f"{emotion.capitalize()}: {confidence:.2f}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                
                # Save frame if requested
                if video_writer:
                    video_writer.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Emotion Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotion Recognition Inference')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--directory', type=str, help='Path to directory with images')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--real-time', action='store_true', help='Run real-time prediction')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    if args.real_time:
        # Real-time prediction
        predictor = RealTimePredictor(args.model)
        try:
            predictor.run_real_time_prediction()
        except KeyboardInterrupt:
            logger.info("Real-time prediction stopped")
        finally:
            predictor.stop_camera()
    
    elif args.image:
        # Single image prediction
        result = predict_single_image(args.image, args.model)
        print(f"Predicted emotion: {result['predicted_emotion']} "
              f"(confidence: {result['confidence']:.3f})")
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    
    elif args.directory:
        # Directory prediction
        predictor = EmotionPredictor(args.model)
        results = predictor.predict_from_directory(args.directory, args.output)
        print(f"Processed {len(results)} images from {args.directory}")
    
    else:
        print("Please specify --image, --directory, or --real-time")
