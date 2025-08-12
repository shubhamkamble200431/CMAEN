"""
Main module for Enhanced Emotion Recognition System
Entry point for training, testing, and inference
"""

import argparse
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Import all modules
from config import config
from models import get_model, count_parameters
from dataset import create_data_loaders
from train import train_model, validate_training_setup
from test import comprehensive_evaluation
from inference import EmotionPredictor, RealTimePredictor
from data_balancer import run_borderline_smote_balancing
from utils import setup_logging, get_device, set_seed

logger = logging.getLogger(__name__)

def setup_experiment(args):
    """Setup experiment configuration and directories"""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Set random seed for reproducibility
    if args.seed:
        set_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Create output directories
    config.create_directories()
    
    # Validate paths if training or testing
    if args.mode in ['train', 'test', 'evaluate']:
        try:
            config.validate_paths()
        except FileNotFoundError as e:
            logger.error(f"Path validation failed: {e}")
            sys.exit(1)
    
    # Log configuration
    logger.info("="*80)
    logger.info("ENHANCED EMOTION RECOGNITION SYSTEM")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model Type: {config.MODEL_TYPE}")
    logger.info(f"Input Size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    logger.info(f"Input Channels: {config.INPUT_CHANNELS}")
    logger.info(f"Number of Classes: {config.NUM_CLASSES}")
    logger.info(f"Device: {get_device()}")
    logger.info("="*80)

def train_mode(args):
    """Training mode"""
    logger.info("Starting training mode...")
    
    # Create model and data loaders
    model = get_model()
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders()
    
    # Log model info
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Log dataset info
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Validate training setup
    validate_training_setup(model, train_loader, val_loader)
    
    # Train model
    start_time = datetime.now()
    results = train_model(model, train_loader, val_loader)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save training results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'{config.RESULTS_DIR}/training_results_{timestamp}.json'
    
    enhanced_results = {
        'model_type': config.MODEL_TYPE,
        'configuration': {
            'input_size': config.INPUT_SIZE,
            'input_channels': config.INPUT_CHANNELS,
            'num_classes': config.NUM_CLASSES,
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'dropout_rate': config.DROPOUT_RATE,
            'weight_decay': config.WEIGHT_DECAY,
        },
        'dataset_info': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'train_dir': config.TRAIN_DIR,
            'test_dir': config.TEST_DIR,
        },
        'results': results,
        'training_time': training_time,
        'timestamp': timestamp,
        'model_parameters': {
            'total': total_params,
            'trainable': trainable_params
        }
    }
    
    with open(results_filename, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    logger.info(f"Training completed! Results saved to {results_filename}")
    logger.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    logger.info(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    
    return results

def test_mode(args):
    """Testing mode"""
    logger.info("Starting testing mode...")
    
    # Create model and test data loader
    model = get_model()
    _, test_loader, _, test_dataset = create_data_loaders()
    
    # Load training results if available
    results = None
    if args.training_results:
        try:
            with open(args.training_results, 'r') as f:
                training_data = json.load(f)
                results = training_data.get('results')
        except Exception as e:
            logger.warning(f"Could not load training results: {e}")
    
    # Run comprehensive evaluation
    evaluation_results = comprehensive_evaluation(
        model, test_loader, results, config.PLOTS_DIR
    )
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = f'{config.RESULTS_DIR}/evaluation_results_{timestamp}.json'
    
    with open(eval_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Testing completed! Results saved to {eval_filename}")
    logger.info(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
    
    return evaluation_results

def inference_mode(args):
    """Inference mode"""
    logger.info("Starting inference mode...")
    
    # Initialize predictor
    predictor = EmotionPredictor(args.model_path)
    
    if args.image:
        # Single image prediction
        logger.info(f"Predicting emotion for image: {args.image}")
        result = predictor.predict_single(args.image)
        
        print(f"\nPrediction Results:")
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Inference Time: {result['inference_time']:.3f}s")
        
        if 'class_probabilities' in result:
            print("\nClass Probabilities:")
            for emotion, prob in result['class_probabilities'].items():
                print(f"  {emotion.capitalize()}: {prob:.3f}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    elif args.directory:
        # Directory prediction
        logger.info(f"Predicting emotions for images in directory: {args.directory}")
        results = predictor.predict_from_directory(args.directory, args.output)
        
        print(f"\nProcessed {len(results)} images")
        
        # Show summary statistics
        emotion_counts = {}
        for result in results.values():
            if 'predicted_emotion' in result:
                emotion = result['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\nEmotion Distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion.capitalize()}: {count}")
    
    elif args.real_time:
        # Real-time prediction
        logger.info("Starting real-time emotion prediction...")
        print("Press 'q' to quit real-time prediction")
        
        rt_predictor = RealTimePredictor(args.model_path, args.camera_index)
        try:
            rt_predictor.run_real_time_prediction(
                display=not args.no_display,
                save_video=args.save_video
            )
        except KeyboardInterrupt:
            logger.info("Real-time prediction stopped by user")
        finally:
            rt_predictor.stop_camera()
    
    else:
        logger.error("Please specify --image, --directory, or --real-time for inference mode")
        sys.exit(1)

def balance_mode(args):
    """Data balancing mode"""
    logger.info("Starting data balancing mode...")
    
    if not args.input_dir or not args.output_dir:
        logger.error("Please specify --input-dir and --output-dir for balancing mode")
        sys.exit(1)
    
    # Run borderline SMOTE balancing
    result = run_borderline_smote_balancing(
        args.input_dir, 
        args.output_dir, 
        target_size=(config.INPUT_SIZE, config.INPUT_SIZE)
    )
    
    if result and result.get('success', False):
        logger.info(f"Data balancing completed successfully!")
        logger.info(f"Total samples: {result['total_samples']}")
        logger.info(f"Saved images: {result['saved_count']}")
        logger.info(f"Processing time: {result['processing_time']:.2f} seconds")
    else:
        logger.error("Data balancing failed!")
        if result:
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)

def evaluate_mode(args):
    """Evaluation mode (similar to test but more focused)"""
    logger.info("Starting evaluation mode...")
    
    # Create model and test data loader
    model = get_model()
    _, test_loader, _, test_dataset = create_data_loaders()
    
    # Simple evaluation
    from test import ModelEvaluator
    evaluator = ModelEvaluator(model, test_loader)
    
    # Basic evaluation
    test_accuracy = evaluator.evaluate_model()
    
    # Get predictions for detailed analysis
    predictions, probabilities, true_labels = evaluator.get_detailed_predictions()
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Generate classification report
    _, classification_str = evaluator.generate_classification_report(true_labels, predicted_labels)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(classification_str)
    
    return test_accuracy

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Emotion Recognition System')
    
    # Main mode selection
    parser.add_argument('mode', choices=['train', 'test', 'inference', 'balance', 'evaluate'],
                        help='Operation mode')
    
    # General arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model-path', type=str, help='Path to trained model file')
    
    # Training/Testing arguments
    parser.add_argument('--training-results', type=str, 
                        help='Path to training results JSON file (for testing mode)')
    
    # Inference arguments
    parser.add_argument('--image', type=str, help='Path to single image for inference')
    parser.add_argument('--directory', type=str, help='Path to directory with images for inference')
    parser.add_argument('--real-time', action='store_true', help='Real-time inference from camera')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index for real-time inference')
    parser.add_argument('--no-display', action='store_true', help='Disable video display in real-time mode')
    parser.add_argument('--save-video', type=str, help='Path to save annotated video')
    parser.add_argument('--output', '-o', type=str, help='Output file path for results')
    
    # Data balancing arguments
    parser.add_argument('--input-dir', type=str, help='Input directory for data balancing')
    parser.add_argument('--output-dir', type=str, help='Output directory for balanced data')
    
    args = parser.parse_args()
    
    # Setup experiment
    setup_experiment(args)
    
    try:
        # Route to appropriate mode
        if args.mode == 'train':
            train_mode(args)
        elif args.mode == 'test':
            test_mode(args)
        elif args.mode == 'inference':
            inference_mode(args)
        elif args.mode == 'balance':
            balance_mode(args)
        elif args.mode == 'evaluate':
            evaluate_mode(args)
        
        logger.info("Operation completed successfully!")
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
