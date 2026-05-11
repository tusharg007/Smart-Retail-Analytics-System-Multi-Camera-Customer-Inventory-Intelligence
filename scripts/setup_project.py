#!/usr/bin/env python3
"""
Project Setup Script
Creates entire directory structure and downloads sample data
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def create_directory_structure():
    """Create complete project directory structure"""
    
    directories = [
        'data/raw/videos',
        'data/raw/images',
        'data/annotations/detection',
        'data/annotations/inventory',
        'data/processed/frames',
        'data/processed/tracking',
        'models/detection/weights',
        'models/inventory/weights',
        'models/tracking',
        'src/data_preparation',
        'src/training',
        'src/inference',
        'src/api',
        'src/utils',
        'notebooks',
        'tests',
        'docker',
        'deployment',
        'dashboard',
        'logs',
        'results/metrics',
        'results/visualizations',
        'configs'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Create __init__.py files
    src_dirs = ['src', 'src/data_preparation', 'src/training', 'src/inference', 'src/api', 'src/utils']
    for src_dir in src_dirs:
        init_file = Path(src_dir) / '__init__.py'
        init_file.touch()
    
    print("\n✅ Directory structure created successfully!")

def download_sample_data():
    """Download sample retail video and images"""
    
    print("\n📥 Downloading sample data...")
    
    # We'll use a publicly available sample video
    # For demo purposes, we'll create a script to generate synthetic data
    print("Note: Sample videos will be generated using synthetic data")
    print("You can add your own videos to data/raw/videos/")
    
    # Create a placeholder file
    with open('data/raw/videos/README.md', 'w') as f:
        f.write("""# Sample Videos
        
Place your retail CCTV footage here (.mp4, .avi format)

For testing without real videos, run:
```
python scripts/generate_synthetic_video.py
```

Recommended sources for sample videos:
1. Your own retail store footage
2. Public datasets: MOT Challenge, VisDrone
3. YouTube Creative Commons videos of retail stores
4. Generated synthetic data (included in this project)
""")
    
    print("✓ Sample data setup complete")

def download_pretrained_weights():
    """Download pre-trained model weights"""
    
    print("\n⚙️  Pre-trained weights will be downloaded automatically during first training run")
    print("YOLOv8n: ~6MB")
    print("ViT-tiny: ~22MB")

def create_config_files():
    """Create configuration files"""
    
    # Create main config
    config_content = """# Smart Retail CV Configuration

project:
  name: smart_retail_cv
  version: 1.0.0
  
data:
  raw_video_dir: data/raw/videos
  processed_dir: data/processed
  annotations_dir: data/annotations
  train_split: 0.8
  val_split: 0.2
  
models:
  detection:
    architecture: yolov8n
    input_size: 640
    conf_threshold: 0.5
    iou_threshold: 0.45
    
  inventory:
    architecture: vit_tiny_patch16_224
    num_classes: 10
    input_size: 224
    
training:
  detection:
    epochs: 20
    batch_size: 16
    learning_rate: 0.001
    optimizer: Adam
    
  inventory:
    epochs: 10
    batch_size: 32
    learning_rate: 0.0001
    
inference:
  fps: 3
  max_cameras: 4
  tracking_method: deepsort
  save_output: true
  
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
monitoring:
  mlflow_tracking_uri: ./mlruns
  log_interval: 100
"""
    
    with open('configs/config.yaml', 'w') as f:
        f.write(config_content)
    
    print("✓ Configuration files created")

def create_gitignore():
    """Create .gitignore file"""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data
data/raw/videos/*.mp4
data/raw/videos/*.avi
data/raw/images/*.jpg
data/processed/

# Models
models/*/weights/*.pt
models/*/weights/*.pth
*.ckpt

# Logs
logs/
mlruns/
wandb/
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Docker
*.tar
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✓ .gitignore created")

def main():
    """Main setup function"""
    
    print("=" * 60)
    print("   SMART RETAIL CV - PROJECT SETUP")
    print("=" * 60)
    
    create_directory_structure()
    download_sample_data()
    download_pretrained_weights()
    create_config_files()
    create_gitignore()
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Activate virtual environment: source venv/bin/activate")
    print("2. Install requirements: pip install -r requirements.txt")
    print("3. Generate synthetic data: python scripts/generate_synthetic_video.py")
    print("4. Start training: python src/training/train_detector.py")
    print("\nEstimated time to complete: 3-4 hours")
    print("=" * 60)

if __name__ == "__main__":
    main()
