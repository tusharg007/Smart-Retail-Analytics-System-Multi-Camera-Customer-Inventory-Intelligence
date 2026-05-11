#!/usr/bin/env python3
"""
YOLOv8 Person Detector Training
Fast training using transfer learning from pre-trained weights
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import time

class DetectorTrainer:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['detection']
        self.train_config = self.config['training']['detection']
        
        # Setup paths
        self.weights_dir = Path('models/detection/weights')
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path('results/detection')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self, epochs=None, batch_size=None):
        """Train YOLOv8 detector"""
        
        print("="*60)
        print("  YOLOV8 PERSON DETECTOR TRAINING")
        print("="*60)
        
        # Use config or override
        epochs = epochs or self.train_config['epochs']
        batch_size = batch_size or self.train_config['batch_size']
        
        # Load pre-trained YOLOv8 model (will auto-download)
        print(f"\nLoading pre-trained {self.model_config['architecture']} model...")
        model = YOLO(f"{self.model_config['architecture']}.pt")
        
        # Check for GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        if device == 'cpu':
            print("\n⚠️  WARNING: Training on CPU will be slower")
            print("   Consider using Google Colab with GPU for faster training")
            print("   Reducing epochs for faster completion...")
            epochs = min(epochs, 10)
        
        # Dataset path
        dataset_yaml = Path('data/processed/dataset.yaml')
        
        if not dataset_yaml.exists():
            print(f"\n❌ Dataset configuration not found: {dataset_yaml}")
            print("Please run: python src/data_preparation/prepare_data.py")
            return None
        
        print(f"\nDataset: {dataset_yaml}")
        print(f"Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Image Size: {self.model_config['input_size']}")
        print(f"  Device: {device}")
        
        # Training parameters
        train_params = {
            'data': str(dataset_yaml),
            'epochs': epochs,
            'imgsz': self.model_config['input_size'],
            'batch': batch_size,
            'device': device,
            'project': str(self.results_dir),
            'name': 'person_detector',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': self.train_config['optimizer'],
            'lr0': self.train_config['learning_rate'],
            'patience': 5,  # Early stopping
            'save_period': 5,
            'cache': True,  # Cache images for faster training
            'workers': 4,
            'verbose': True,
        }
        
        print("\n🚀 Starting training...")
        print("-" * 60)
        
        start_time = time.time()
        
        # Train the model
        results = model.train(**train_params)
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nTraining time: {training_time/60:.2f} minutes")
        
        # Save best model
        best_model_path = self.results_dir / 'person_detector' / 'weights' / 'best.pt'
        if best_model_path.exists():
            # Copy to main weights directory
            import shutil
            final_path = self.weights_dir / 'person_detector_best.pt'
            shutil.copy(best_model_path, final_path)
            print(f"✓ Best model saved: {final_path}")
        
        # Print metrics
        print("\n📊 Training Metrics:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                print(f"  mAP50: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"  mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
        
        print(f"\nResults saved in: {self.results_dir / 'person_detector'}")
        print("\nNext steps:")
        print("1. Validate model: python src/training/validate_detector.py")
        print("2. Run inference: python src/inference/run_inference.py")
        
        return model
    
    def validate(self, model_path=None):
        """Validate trained model"""
        
        if model_path is None:
            model_path = self.weights_dir / 'person_detector_best.pt'
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            print("Please train the model first: python src/training/train_detector.py")
            return
        
        print(f"\n🔍 Validating model: {model_path}")
        
        model = YOLO(str(model_path))
        
        dataset_yaml = Path('data/processed/dataset.yaml')
        
        results = model.val(data=str(dataset_yaml))
        
        print("\n✅ Validation complete!")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 Person Detector')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    
    args = parser.parse_args()
    
    trainer = DetectorTrainer()
    
    if args.validate:
        trainer.validate()
    else:
        trainer.train(epochs=args.epochs, batch_size=args.batch)

if __name__ == "__main__":
    main()
