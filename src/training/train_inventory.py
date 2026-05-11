#!/usr/bin/env python3
"""
Vision Transformer for Product Classification
Fast training on synthetic product dataset
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import numpy as np
import cv2

class ProductDataset(Dataset):
    """Simple product classification dataset"""
    
    def __init__(self, num_samples=1000, num_classes=10, image_size=224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate synthetic product images
        print(f"Generating {num_samples} synthetic product images...")
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            # Create synthetic product image
            img = self.generate_product_image(i % num_classes)
            self.images.append(img)
            self.labels.append(i % num_classes)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def generate_product_image(self, class_id):
        """Generate a synthetic product image"""
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        
        # Product colors based on class
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0)
        ]
        color = colors[class_id]
        
        # Draw product box
        cv2.rectangle(img, (50, 50), (174, 174), color, -1)
        cv2.rectangle(img, (50, 50), (174, 174), (0, 0, 0), 3)
        
        # Add some noise for variety
        noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleViT(nn.Module):
    """Lightweight Vision Transformer for product classification"""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        # Use a lightweight pretrained model
        if pretrained:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # Replace classifier
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone = efficientnet_b0()
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

class InventoryTrainer:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['inventory']
        self.train_config = self.config['training']['inventory']
        
        self.weights_dir = Path('models/inventory/weights')
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def train(self, epochs=None, batch_size=None):
        """Train product classifier"""
        
        print("="*60)
        print("  PRODUCT CLASSIFIER TRAINING (Vision Transformer)")
        print("="*60)
        
        epochs = epochs or self.train_config['epochs']
        batch_size = batch_size or self.train_config['batch_size']
        
        print(f"\nDevice: {self.device}")
        if self.device == 'cpu':
            print("⚠️  Training on CPU - reducing epochs for faster completion")
            epochs = min(epochs, 5)
            batch_size = min(batch_size, 16)
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Num Classes: {self.model_config['num_classes']}")
        print(f"  Image Size: {self.model_config['input_size']}")
        
        # Create datasets
        print("\nPreparing datasets...")
        train_dataset = ProductDataset(
            num_samples=800,
            num_classes=self.model_config['num_classes'],
            image_size=self.model_config['input_size']
        )
        
        val_dataset = ProductDataset(
            num_samples=200,
            num_classes=self.model_config['num_classes'],
            image_size=self.model_config['input_size']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create model
        print("\nInitializing model...")
        model = SimpleViT(num_classes=self.model_config['num_classes'], pretrained=True)
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        
        print("\n🚀 Starting training...")
        print("-" * 60)
        
        start_time = time.time()
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{train_loss/(pbar.n+1):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    pbar.set_postfix({
                        'loss': f'{val_loss/(pbar.n+1):.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            val_acc = 100. * val_correct / val_total
            
            print(f'\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.weights_dir / 'product_classifier_best.pt')
                print(f'✓ Best model saved (Val Acc: {val_acc:.2f}%)')
            
            scheduler.step(val_loss / len(val_loader))
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Model saved: {self.weights_dir / 'product_classifier_best.pt'}")
        
        return model

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Product Classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    trainer = InventoryTrainer()
    trainer.train(epochs=args.epochs, batch_size=args.batch)

if __name__ == "__main__":
    main()
