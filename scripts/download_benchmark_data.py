#!/usr/bin/env python3
"""
Benchmark Data Preparation Script
Downloads and prepares MOT17 pedestrian tracking dataset
Converts annotations to YOLO format for real-world benchmarking.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import configparser

def download_mot17(data_dir: Path):
    print("Downloading MOT17 dataset (this may take a while)...")
    # For demonstration, we use a small public sample or direct user to MOTChallenge
    url = "https://motchallenge.net/data/MOT17Labels.zip" # Download just labels for demo script
    zip_path = data_dir / "MOT17Labels.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("MOT17 Labels downloaded and extracted.")

def convert_mot_to_yolo(seq_path: Path, output_dir: Path):
    """Convert MOT format to YOLO format (class x_center y_center width height normalized)"""
    gt_file = seq_path / 'gt' / 'gt.txt'
    seqinfo = seq_path / 'seqinfo.ini'
    
    if not gt_file.exists() or not seqinfo.exists():
        return

    # Parse seqinfo to get dimensions
    config = configparser.ConfigParser()
    config.read(seqinfo)
    width = float(config['Sequence']['imWidth'])
    height = float(config['Sequence']['imHeight'])

    with open(gt_file, 'r') as f:
        lines = f.readlines()

    # Create frames dir
    labels_dir = output_dir / 'labels' / seq_path.name
    labels_dir.mkdir(parents=True, exist_ok=True)

    for line in lines:
        parts = line.strip().split(',')
        frame_id = int(parts[0])
        # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
        bb_left = float(parts[2])
        bb_top = float(parts[3])
        bb_width = float(parts[4])
        bb_height = float(parts[5])
        class_id = int(parts[7])
        visibility = float(parts[8])

        # Filter: only pedestrian (1) and visibility > 0.3
        if class_id != 1 or visibility < 0.3:
            continue

        # Convert to YOLO (normalized)
        x_center = (bb_left + bb_width / 2) / width
        y_center = (bb_top + bb_height / 2) / height
        w = bb_width / width
        h = bb_height / height

        # Clip values to 0-1
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

        frame_file = labels_dir / f"{frame_id:06d}.txt"
        with open(frame_file, 'a') as out_f:
            out_f.write(yolo_line)

def main():
    print("="*60)
    print("  MOT17 BENCHMARK DATA PREPARATION")
    print("="*60)
    
    base_dir = Path("data/mot17")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    yolo_output = Path("data/mot17_yolo")
    train_dir = yolo_output / "train"
    val_dir = yolo_output / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    download_mot17(base_dir)
    
    print("\nConverting annotations to YOLO format...")
    # Assuming extraction created 'train' folder
    mot_train = base_dir / "train"
    if mot_train.exists():
        sequences = [d for d in mot_train.iterdir() if d.is_dir()]
        for seq in sequences:
            # Simple split: put half in train, half in val for demonstration
            dest = train_dir if int(seq.name[-2:]) % 2 == 0 else val_dir
            convert_mot_to_yolo(seq, dest)
            
    print("\n✓ Real-world benchmark dataset preparation complete!")
    print(f"Data saved to: {yolo_output.absolute()}")

if __name__ == "__main__":
    main()
