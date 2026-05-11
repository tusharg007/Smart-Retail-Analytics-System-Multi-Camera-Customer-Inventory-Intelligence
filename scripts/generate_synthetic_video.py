#!/usr/bin/env python3
"""
Generate synthetic retail store video for testing
Creates a realistic retail environment with people and products
"""

import cv2
import numpy as np
from pathlib import Path
import random

class SyntheticRetailVideoGenerator:
    def __init__(self, width=1280, height=720, fps=30, duration=60):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration
        
    def create_store_background(self):
        """Create a simple store background"""
        background = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        
        # Add floor
        cv2.rectangle(background, (0, self.height//2), (self.width, self.height), 
                     (200, 200, 200), -1)
        
        # Add shelves (3 aisles)
        shelf_color = (150, 150, 150)
        for i in range(3):
            x = 200 + i * 350
            cv2.rectangle(background, (x, 200), (x + 100, self.height - 100), 
                         shelf_color, -1)
            
        # Add checkout counter
        cv2.rectangle(background, (50, self.height - 200), (150, self.height - 50), 
                     (100, 100, 150), -1)
        
        return background
    
    def create_person(self, color=None):
        """Create a simple person representation"""
        if color is None:
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        person = np.zeros((60, 30, 3), dtype=np.uint8)
        # Head
        cv2.circle(person, (15, 10), 8, color, -1)
        # Body
        cv2.rectangle(person, (8, 18), (22, 45), color, -1)
        # Legs
        cv2.rectangle(person, (8, 45), (13, 60), color, -1)
        cv2.rectangle(person, (17, 45), (22, 60), color, -1)
        
        return person
    
    def create_product(self):
        """Create a simple product box"""
        product = np.zeros((40, 30, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = random.choice(colors)
        cv2.rectangle(product, (0, 0), (30, 40), color, -1)
        cv2.rectangle(product, (0, 0), (30, 40), (0, 0, 0), 2)
        return product
    
    def generate_video(self, output_path):
        """Generate the synthetic video"""
        
        print(f"Generating synthetic retail video: {output_path}")
        print(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, Duration: {self.duration}s")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        # Initialize people positions and velocities
        num_people = 5
        people = []
        for _ in range(num_people):
            people.append({
                'x': random.randint(100, self.width - 100),
                'y': random.randint(300, self.height - 100),
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-1, 1),
                'color': (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            })
        
        # Initialize products on shelves
        products = []
        for shelf in range(3):
            shelf_x = 200 + shelf * 350
            for row in range(3):
                for col in range(3):
                    products.append({
                        'x': shelf_x + col * 35 + 5,
                        'y': 210 + row * 50
                    })
        
        background = self.create_store_background()
        
        for frame_idx in range(self.total_frames):
            frame = background.copy()
            
            # Draw products on shelves
            for product in products:
                product_img = self.create_product()
                x, y = product['x'], product['y']
                if 0 <= x < self.width - 30 and 0 <= y < self.height - 40:
                    frame[y:y+40, x:x+30] = product_img
            
            # Update and draw people
            for person in people:
                # Update position
                person['x'] += person['vx']
                person['y'] += person['vy']
                
                # Bounce off walls
                if person['x'] <= 0 or person['x'] >= self.width - 30:
                    person['vx'] *= -1
                if person['y'] <= 200 or person['y'] >= self.height - 60:
                    person['vy'] *= -1
                
                # Random direction change
                if random.random() < 0.01:
                    person['vx'] = random.uniform(-2, 2)
                    person['vy'] = random.uniform(-1, 1)
                
                # Draw person
                person_img = self.create_person(person['color'])
                x, y = int(person['x']), int(person['y'])
                if 0 <= x < self.width - 30 and 0 <= y < self.height - 60:
                    # Alpha blending for better visuals
                    mask = np.any(person_img != 0, axis=2)
                    frame[y:y+60, x:x+30][mask] = person_img[mask]
            
            # Add timestamp
            timestamp = f"Frame: {frame_idx}/{self.total_frames} | Time: {frame_idx/self.fps:.1f}s"
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 0), 2)
            
            # Add store info
            cv2.putText(frame, "RETAIL STORE - Camera 1", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            out.write(frame)
            
            if (frame_idx + 1) % (self.fps * 10) == 0:
                print(f"  Progress: {(frame_idx + 1) / self.total_frames * 100:.1f}%")
        
        out.release()
        print(f"✓ Video saved: {output_path}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

def generate_multiple_cameras():
    """Generate videos for multiple camera angles"""
    
    output_dir = Path('data/raw/videos')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 2 sample videos (keep it quick)
    configs = [
        {'duration': 30, 'name': 'camera1_entrance.mp4'},  # 30 seconds
        {'duration': 30, 'name': 'camera2_aisles.mp4'},    # 30 seconds
    ]
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Generating {config['name']}")
        generator = SyntheticRetailVideoGenerator(
            width=1280, 
            height=720, 
            fps=30, 
            duration=config['duration']
        )
        output_path = output_dir / config['name']
        generator.generate_video(output_path)
    
    print("\n" + "="*60)
    print("✅ All synthetic videos generated successfully!")
    print("="*60)
    print(f"\nVideos saved in: {output_dir}")
    print("\nYou can now:")
    print("1. Run data preparation: python src/data_preparation/prepare_data.py")
    print("2. Start training: python src/training/train_detector.py")

if __name__ == "__main__":
    generate_multiple_cameras()
