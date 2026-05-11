# Smart Retail Analytics — Multi-Camera CV Intelligence Platform

Real-time computer vision system for retail environments: customer tracking, 
footfall analytics, and inventory monitoring across multiple camera streams.

## What It Does
- Detects and tracks customers across 4 camera feeds at 25+ FPS
- Monitors shelf inventory using Vision Transformer-based product classification  
- Generates heatmaps, dwell-time analytics, and queue detection alerts
- Exposes a REST API (6 endpoints) for dashboard and third-party integration

## Model Performance
| Model | Task | mAP@0.5 | Precision | Recall | Latency |
|-------|------|---------|-----------|--------|---------|
| YOLOv8n (fine-tuned) | Person detection | 0.83* | 0.87* | 0.79* | <40ms |
| ViT-tiny (fine-tuned) | Product classification | — | 0.81* | 0.78* | <60ms |

*Results on synthetic retail validation set (2K frames, 4-camera simulation)

## Dataset & Annotation Pipeline
- **Data source**: Synthetic multi-camera retail video (generated via `scripts/generate_synthetic_video.py`)
- **Annotation format**: YOLO format `.txt` (class cx cy w h normalized), annotated per-frame
- **Dataset size**: ~10K labeled frames across 4 camera angles
- **Augmentation**: horizontal flip, HSV shift, random crop, mosaic
- **Split**: 80/15/5 train/val/test

## Architecture
```
smart-retail-cv/
├── data/
│   ├── raw/                 # Raw CCTV footage
│   ├── annotations/         # YOLO format annotations
│   └── processed/           # Processed frames
├── models/
│   ├── detection/          # YOLOv8 person detector
│   ├── inventory/          # ViT product classifier
│   └── weights/            # Pre-trained & fine-tuned weights
├── src/
│   ├── data_preparation/   # Data loading & preprocessing
│   ├── training/           # Model training scripts
│   ├── inference/          # Real-time inference pipeline
│   ├── api/                # FastAPI backend
│   └── utils/              # Helper functions
├── tests/
│   └── test_pipeline.py    # Unit and integration tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── dashboard/
    └── app.py              # Streamlit dashboard
```

## Tech Stack
- **Deep Learning:** PyTorch, YOLOv8, Vision Transformers
- **Computer Vision:** OpenCV, Albumentations
- **Tracking:** ByteTrack
- **API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **MLOps:** MLflow, Weights & Biases
- **Deployment:** Docker
- **Database:** SQLite (scalable to PostgreSQL)

## API Endpoints
```
POST   /api/v1/video/upload      - Upload CCTV footage
GET    /api/v1/analytics/footfall - Customer count analytics
GET    /api/v1/inventory/status   - Real-time shelf status
GET    /api/v1/alerts             - System alerts
POST   /api/v1/inference/process  - Process video frame
GET    /api/v1/health             - Health check
```
