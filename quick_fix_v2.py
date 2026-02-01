"""
QUICK FIX SCRIPT - Run this to fix the setup issues
"""

import os
from pathlib import Path

print("=" * 60)
print("FIXING SETUP ISSUES...")
print("=" * 60)

# Step 1: Create ALL directories first
print("\n[1/4] Creating all directories...")
directories = [
    'data/raw/videos',
    'data/raw/images',
    'data/annotations/detection',
    'data/annotations/inventory',
    'data/processed/frames',
    'data/processed/tracking',
    'models/detection/weights',
    'models/inventory/weights',
    'src/data_preparation',
    'src/training',
    'src/inference',
    'src/api',
    'src/utils',
    'results',
    'logs',
    'configs'
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
print("[OK] All directories created!")

# Step 2: Create __init__ files
print("\n[2/4] Creating Python package files...")
src_dirs = ['src', 'src/data_preparation', 'src/training', 'src/inference', 'src/api', 'src/utils']
for src_dir in src_dirs:
    init_file = Path(src_dir) / '__init__.py'
    init_file.touch()
print("[OK] Package files created!")

# Step 3: Create config file
print("\n[3/4] Creating config file...")

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

# Step 4: Verify everything
print("\n[4/4] Verifying setup...")
checks = [
    ('Config file', Path('configs/config.yaml').exists()),
    ('Data directory', Path('data/raw/videos').exists()),
    ('Models directory', Path('models/detection/weights').exists()),
    ('Source packages', Path('src/__init__.py').exists()),
]

all_good = True
for name, status in checks:
    if status:
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name}")
        all_good = False

print("\n" + "=" * 60)
if all_good:
    print("[SUCCESS] ALL FIXES APPLIED!")
    print("=" * 60)
    print("\nYour environment is ready!")
    print("\nNext steps:")
    print("1. python scripts\\generate_synthetic_video.py")
    print("2. python src\\data_preparation\\prepare_data.py")
    print("3. python src\\training\\train_detector.py --epochs 10")
else:
    print("[WARNING] Some checks failed - please review above")
print("=" * 60)
