"""
QUICK FIX SCRIPT - Run this to fix the setup issues
"""

import os
from pathlib import Path

print("=" * 60)
print("FIXING SETUP ISSUES...")
print("=" * 60)

# Step 1: Create missing config file
print("\n[1/3] Creating config file...")
config_dir = Path('configs')
config_dir.mkdir(exist_ok=True)

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
print("[OK] Config file created!")

# Step 2: Create all required directories
print("\n[2/3] Creating directories...")
directories = [
    'data/raw/videos',
    'data/annotations/detection',
    'data/processed/frames',
    'models/detection/weights',
    'models/inventory/weights',
    'results',
    'logs'
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
print("[OK] All directories created!")

# Step 3: Create __init__ files
print("\n[3/3] Creating Python package files...")
src_dirs = ['src', 'src/data_preparation', 'src/training', 'src/inference', 'src/api', 'src/utils']
for src_dir in src_dirs:
    init_file = Path(src_dir) / '__init__.py'
    if not init_file.exists():
        init_file.touch()
print("[OK] Package files created!")

print("\n" + "=" * 60)
print("[SUCCESS] ALL FIXES APPLIED!")
print("=" * 60)
print("\nNow you can run:")
print("1. python scripts\\generate_synthetic_video.py")
print("2. python src\\data_preparation\\prepare_data.py")
print("3. python src\\training\\train_detector.py --epochs 10")
print("=" * 60)
