# Smart Retail Analytics — Multi-Camera CV Intelligence Platform

Real-time computer vision system for retail environments: customer tracking, footfall analytics, and inventory monitoring across multiple camera streams.

## Development Philosophy
As a solo developer, I architected this platform with strict modularity—separating data preparation, inference, API, and the frontend—to ensure maintainability and scalability. By decoupling the CV inference engine from the FastAPI backend and Streamlit dashboard, I can independently optimize or swap out components without breaking the entire pipeline. This design choice reflects my deliberate engineering focus on building robust, production-ready systems from the ground up.

## What It Does
- Detects and tracks customers across 4 camera feeds at 25+ FPS
- Monitors shelf inventory using Vision Transformer-based product classification  
- Generates heatmaps, dwell-time analytics, and queue detection alerts
- **GenAI Reporting**: Generates automated, plain-English retail management reports using an LLM (Mistral-7B/Llama3.2)
- **Live Streamlit Dashboard**: Supports interactive Live Camera Feeds (RTSP, Webcam, File Upload) processed via OpenCV threads
- Exposes a REST API (7 endpoints) for dashboard and third-party integration

## Model Performance
| Model | Task | Synthetic Validation | Real-World Target | Precision | Recall | Latency |
|-------|------|----------------------|-------------------|-----------|--------|---------|
| YOLOv8n (fine-tuned) | Person detection | mAP@0.5: 0.83 | MOT17 Dataset | 0.87* | 0.79* | <40ms |
| ViT-tiny (fine-tuned) | Product classification | — | SKU110K Dataset | 0.81* | 0.78* | <60ms |

*Metrics recorded on synthetic retail validation set (~2K frames)

## Dataset & Annotation Pipeline
- **Data source**: Synthetic multi-camera retail video for pipeline validation (generated via `scripts/generate_synthetic_video.py`)
- **Real-World Benchmarking**: MOT17 annotations converted to YOLO format via `scripts/download_benchmark_data.py`
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
│   ├── inference/          # Real-time inference pipeline & GenAI reporter
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
- **GenAI & LLMs:** HuggingFace API, Ollama (Mistral/Llama3.2)
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
POST   /api/v1/generate-report    - Generate LLM management report
GET    /api/v1/health             - Health check
```
