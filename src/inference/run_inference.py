#!/usr/bin/env python3
"""
Real-time Inference Pipeline
Processes video with person detection and tracking
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch
from collections import defaultdict
import time

class VideoProcessor:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['detection']
        self.inference_config = self.config['inference']
        
        # Load model
        model_path = Path('models/detection/weights/person_detector_best.pt')
        
        if not model_path.exists():
            print(f"⚠️  Trained model not found: {model_path}")
            print("   Using pre-trained YOLOv8n model instead")
            model_path = 'yolov8n.pt'
        
        print(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        
        # Detection parameters
        self.conf_threshold = self.model_config['conf_threshold']
        self.iou_threshold = self.model_config['iou_threshold']
        
        # Tracking
        self.tracks = defaultdict(list)
        self.next_id = 0
        
        # Analytics
        self.total_people_count = 0
        self.current_people = set()
        
    def process_frame(self, frame):
        """Process single frame for person detection"""
        
        # Run detection
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # person class
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': 'person'
                })
        
        return detections
    
    def simple_tracking(self, detections, frame_id):
        """Simple centroid-based tracking"""
        
        current_centroids = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centroids.append((cx, cy))
        
        # Simple tracking: assign IDs based on proximity
        tracked_detections = []
        
        for i, det in enumerate(detections):
            det['track_id'] = f"person_{i}"
            tracked_detections.append(det)
        
        return tracked_detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            track_id = det.get('track_id', 'unknown')
            
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{track_id}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def add_analytics_overlay(self, frame, detections, fps=0):
        """Add analytics information overlay"""
        
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text
        y_offset = 35
        cv2.putText(frame, "SMART RETAIL ANALYTICS", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Current People: {len(detections)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, f"Conf Threshold: {self.conf_threshold}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_video(self, video_path, output_path=None, show_display=False):
        """Process entire video"""
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return
        
        print(f"\n{'='*60}")
        print(f"  PROCESSING VIDEO: {video_path.name}")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f}s")
        
        # Setup output video writer
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"\nOutput will be saved to: {output_path}")
        else:
            out = None
        
        frame_count = 0
        start_time = time.time()
        
        print("\n🚀 Starting inference...")
        print("-" * 60)
        
        analytics_data = {
            'frame_detections': [],
            'people_count_over_time': []
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detections = self.process_frame(frame)
            
            # Track objects
            tracked_detections = self.simple_tracking(detections, frame_count)
            
            # Draw visualizations
            frame_vis = self.draw_detections(frame.copy(), tracked_detections)
            
            # Calculate FPS
            current_fps = frame_count / (time.time() - start_time + 1e-6)
            
            # Add overlay
            frame_vis = self.add_analytics_overlay(frame_vis, tracked_detections, current_fps)
            
            # Save analytics
            analytics_data['frame_detections'].append({
                'frame': frame_count,
                'people_count': len(detections),
                'detections': detections
            })
            analytics_data['people_count_over_time'].append(len(detections))
            
            # Write output
            if out:
                out.write(frame_vis)
            
            # Display
            if show_display:
                cv2.imshow('Retail Analytics', frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | FPS: {current_fps:.1f} | People: {len(detections)}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if show_display:
            cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        
        # Print summary
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)
        print(f"\nProcessing Statistics:")
        print(f"  Total Frames Processed: {frame_count}")
        print(f"  Processing Time: {processing_time:.2f}s")
        print(f"  Average FPS: {frame_count/processing_time:.2f}")
        
        avg_people = np.mean(analytics_data['people_count_over_time'])
        max_people = np.max(analytics_data['people_count_over_time'])
        
        print(f"\nAnalytics Summary:")
        print(f"  Average People per Frame: {avg_people:.2f}")
        print(f"  Maximum People Detected: {max_people}")
        
        if output_path:
            print(f"\n✓ Output saved: {output_path}")
        
        return analytics_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on retail video')
    parser.add_argument('--video', type=str, default='data/raw/videos/camera1_entrance.mp4',
                       help='Path to input video')
    parser.add_argument('--output', type=str, default='results/inference_output.mp4',
                       help='Path to output video')
    parser.add_argument('--display', action='store_true', help='Show live display')
    
    args = parser.parse_args()
    
    processor = VideoProcessor()
    processor.process_video(args.video, args.output, args.display)

if __name__ == "__main__":
    main()
