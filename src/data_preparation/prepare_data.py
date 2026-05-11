#!/usr/bin/env python3
"""
Data Preparation Pipeline
Processes videos, extracts frames, creates annotations
"""

import cv2
import numpy as np
from pathlib import Path
import json
import random
from tqdm import tqdm
import yaml

class DataPreparation:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.video_dir = Path(self.config['data']['raw_video_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.annotations_dir = Path(self.config['data']['annotations_dir'])
        
    def extract_frames(self, video_path, output_dir, sample_rate=10):
        """Extract frames from video at specified sample rate"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"\nProcessing: {video_path.name}")
        print(f"  Total frames: {total_frames}, FPS: {fps}")
        
        frame_count = 0
        saved_count = 0
        
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every N frames
            if frame_count % sample_rate == 0:
                frame_filename = f"{video_path.stem}_frame_{saved_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        print(f"  ✓ Extracted {saved_count} frames")
        return saved_count
    
    def generate_yolo_annotations(self, frames_dir, output_dir):
        """Generate YOLO format annotations (synthetic for demo)"""
        
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_files = list(frames_dir.glob('*.jpg'))
        print(f"\nGenerating YOLO annotations for {len(frame_files)} frames")
        
        # YOLO format: class x_center y_center width height (normalized)
        # Class 0: person
        
        for frame_file in tqdm(frame_files, desc="Creating annotations"):
            img = cv2.imread(str(frame_file))
            h, w = img.shape[:2]
            
            # Generate 1-5 random person bounding boxes
            num_people = random.randint(1, 5)
            annotations = []
            
            for _ in range(num_people):
                # Random position (avoiding edges)
                x_center = random.uniform(0.15, 0.85)
                y_center = random.uniform(0.3, 0.9)
                
                # Person bbox size (normalized)
                bbox_width = random.uniform(0.05, 0.12)
                bbox_height = random.uniform(0.15, 0.25)
                
                annotations.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
            
            # Save annotation file
            ann_file = output_dir / f"{frame_file.stem}.txt"
            with open(ann_file, 'w') as f:
                f.write('\n'.join(annotations))
        
        print(f"  ✓ Generated {len(frame_files)} annotation files")
    
    def create_dataset_yaml(self):
        """Create dataset configuration for YOLO training"""
        
        dataset_yaml = {
            'path': str(Path('data/processed').absolute()),
            'train': 'frames/train',
            'val': 'frames/val',
            'names': {
                0: 'person'
            },
            'nc': 1  # number of classes
        }
        
        yaml_path = self.processed_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        print(f"\n✓ Created dataset.yaml: {yaml_path}")
        return yaml_path
    
    def split_dataset(self, frames_dir, annotations_dir, train_ratio=0.8):
        """Split dataset into train and validation sets"""
        
        frames_dir = Path(frames_dir)
        annotations_dir = Path(annotations_dir)
        
        # Get all frames
        frame_files = sorted(list(frames_dir.glob('*.jpg')))
        random.shuffle(frame_files)
        
        # Split
        split_idx = int(len(frame_files) * train_ratio)
        train_files = frame_files[:split_idx]
        val_files = frame_files[split_idx:]
        
        # Create directories
        train_frames_dir = self.processed_dir / 'frames' / 'train' / 'images'
        train_labels_dir = self.processed_dir / 'frames' / 'train' / 'labels'
        val_frames_dir = self.processed_dir / 'frames' / 'val' / 'images'
        val_labels_dir = self.processed_dir / 'frames' / 'val' / 'labels'
        
        for d in [train_frames_dir, train_labels_dir, val_frames_dir, val_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSplitting dataset:")
        print(f"  Train: {len(train_files)} images")
        print(f"  Val: {len(val_files)} images")
        
        # Copy train files
        for frame_file in tqdm(train_files, desc="Copying train files"):
            # Copy image
            import shutil
            shutil.copy(frame_file, train_frames_dir / frame_file.name)
            
            # Copy annotation
            ann_file = annotations_dir / f"{frame_file.stem}.txt"
            if ann_file.exists():
                shutil.copy(ann_file, train_labels_dir / ann_file.name)
        
        # Copy val files
        for frame_file in tqdm(val_files, desc="Copying val files"):
            import shutil
            shutil.copy(frame_file, val_frames_dir / frame_file.name)
            
            ann_file = annotations_dir / f"{frame_file.stem}.txt"
            if ann_file.exists():
                shutil.copy(ann_file, val_labels_dir / ann_file.name)
        
        print("✓ Dataset split complete")
    
    def run(self):
        """Run complete data preparation pipeline"""
        
        print("="*60)
        print("  DATA PREPARATION PIPELINE")
        print("="*60)
        
        # Find all videos
        video_files = list(self.video_dir.glob('*.mp4'))
        if not video_files:
            print("\n❌ No video files found!")
            print(f"Please add videos to: {self.video_dir}")
            print("Or run: python scripts/generate_synthetic_video.py")
            return
        
        print(f"\nFound {len(video_files)} video files")
        
        # Step 1: Extract frames
        frames_output = self.processed_dir / 'frames_raw'
        total_frames = 0
        
        for video_file in video_files:
            num_frames = self.extract_frames(video_file, frames_output, sample_rate=10)
            total_frames += num_frames
        
        # Step 2: Generate annotations
        annotations_output = self.annotations_dir / 'detection'
        self.generate_yolo_annotations(frames_output, annotations_output)
        
        # Step 3: Split dataset
        self.split_dataset(frames_output, annotations_output, train_ratio=0.8)
        
        # Step 4: Create dataset YAML
        self.create_dataset_yaml()
        
        print("\n" + "="*60)
        print("✅ DATA PREPARATION COMPLETE!")
        print("="*60)
        print(f"\nTotal frames extracted: {total_frames}")
        print(f"Annotations created: {total_frames}")
        print("\nDataset structure:")
        print(f"  Train images: {self.processed_dir}/frames/train/images/")
        print(f"  Train labels: {self.processed_dir}/frames/train/labels/")
        print(f"  Val images: {self.processed_dir}/frames/val/images/")
        print(f"  Val labels: {self.processed_dir}/frames/val/labels/")
        print("\nNext step: python src/training/train_detector.py")

if __name__ == "__main__":
    processor = DataPreparation()
    processor.run()
