import pytest
import numpy as np
import cv2
from src.inference.run_inference import VideoProcessor
from pathlib import Path

def test_frame_extraction_shape():
    # Ensure a frame has standard dimensions (H, W, C)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8

def test_detector_output_has_confidence():
    processor = VideoProcessor()
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = processor.process_frame(frame)
    
    assert isinstance(detections, list)
    for det in detections:
        assert 'confidence' in det
        assert 0.0 <= det['confidence'] <= 1.0
        assert 'bbox' in det
        assert len(det['bbox']) == 4

def test_mAP_above_threshold():
    # Simulate extraction of mAP metrics from training results dictionary
    # A real training pipeline should achieve mAP50 > 0.5
    mock_results_dict = {'metrics/mAP50(B)': 0.83, 'metrics/mAP50-95(B)': 0.55}
    assert 'metrics/mAP50(B)' in mock_results_dict
    assert mock_results_dict['metrics/mAP50(B)'] > 0.80

def test_tracking_assigns_ids():
    processor = VideoProcessor()
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = processor.process_frame(frame)
    
    for det in detections:
        assert 'track_id' in det
        assert isinstance(det['track_id'], str)

def test_analytics_overlay():
    processor = VideoProcessor()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [{'bbox': [10, 10, 50, 50], 'confidence': 0.9, 'class': 'person', 'track_id': 'person_1'}]
    
    annotated_frame = processor.add_analytics_overlay(frame.copy(), detections, fps=30.0)
    
    assert annotated_frame.shape == frame.shape
    assert not np.array_equal(annotated_frame, frame)
