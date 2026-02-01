# Smart Retail Analytics System - CV Resume Project

A production-ready Computer Vision pipeline for retail analytics using YOLOv8, Vision Transformers, and real-time video processing.

## 🎯 Project Overview

This project demonstrates:
- Real-time customer tracking and analytics
- Inventory monitoring using computer vision
- Production-grade ML pipeline (training → deployment → monitoring)
- Docker containerization
- REST API integration
- Cloud-ready deployment

## ⚡ Quick Start (Complete in 3-4 hours)

### Pre-Installation Requirements

**1. System Requirements:**
- Python 3.8+ (recommended: 3.10)
- 8GB+ RAM
- GPU (optional but recommended for faster inference)
- 10GB free disk space

**2. Install Core Dependencies:**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (choose based on your system)
# CPU version (faster to install):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR GPU version (if you have NVIDIA GPU):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all project requirements
pip install -r requirements.txt
```

**3. Docker (Optional - for deployment demo):**
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
# Not required for main project completion
```

## 📁 Project Structure

```
smart-retail-cv/
├── data/
│   ├── raw/                 # Place sample videos here
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
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
├── tests/
│   └── test_pipeline.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── deployment/
│   └── aws_deploy.md
└── dashboard/
    └── app.py              # Streamlit dashboard
```

## 🚀 Implementation Timeline (3-4 hours)

### Phase 1: Setup (15 mins)
```bash
# Clone/create project structure
python scripts/setup_project.py

# Download pre-trained weights and sample data
python scripts/download_assets.py
```

### Phase 2: Data Preparation (30 mins)
```bash
# Process sample retail videos
python src/data_preparation/prepare_data.py

# This will:
# - Extract frames from videos
# - Create train/val splits
# - Generate synthetic annotations (for demo)
```

### Phase 3: Model Training (45 mins)
```bash
# Train YOLOv8 person detector (transfer learning - 10 mins)
python src/training/train_detector.py --epochs 20 --batch 16

# Train product classifier (quick fine-tune - 15 mins)
python src/training/train_inventory.py --epochs 10

# Both use pre-trained weights for fast convergence
```

### Phase 4: Inference Pipeline (30 mins)
```bash
# Test inference on sample video
python src/inference/run_inference.py --video data/raw/sample_store.mp4

# Run multi-camera simulation
python src/inference/multi_camera.py
```

### Phase 5: API & Dashboard (45 mins)
```bash
# Start FastAPI backend
python src/api/main.py

# In another terminal, start dashboard
streamlit run dashboard/app.py
```

### Phase 6: Docker & Documentation (30 mins)
```bash
# Build Docker image
docker build -t retail-cv:latest -f docker/Dockerfile .

# Run containerized version
docker-compose up
```

## 📊 Key Features Implemented

### 1. Customer Analytics
- ✅ Person detection and tracking (YOLOv8 + DeepSORT)
- ✅ Footfall counting
- ✅ Dwell time analysis
- ✅ Heatmap generation
- ✅ Queue detection

### 2. Inventory Monitoring
- ✅ Product detection (Vision Transformer)
- ✅ Empty shelf detection
- ✅ Stock level estimation
- ✅ Misplacement alerts

### 3. Production Features
- ✅ Real-time video processing pipeline
- ✅ Multi-camera support
- ✅ REST API (FastAPI)
- ✅ Model versioning (MLflow)
- ✅ Monitoring dashboard (Streamlit)
- ✅ Docker containerization
- ✅ Cloud deployment guide (AWS)

## 🎯 Performance Metrics

- **Person Detection mAP:** >85%
- **Inference Latency:** <100ms per frame
- **Throughput:** 4 cameras @ 3 FPS
- **Model Size:** YOLOv8n (6MB), ViT-tiny (22MB)

## 📈 Results & Demo

After completion, you'll have:
1. **Trained Models** - 2 production-ready CV models
2. **Live Dashboard** - Real-time analytics visualization
3. **REST API** - 6+ endpoints for system integration
4. **Docker Images** - Deployable containers
5. **Documentation** - Complete technical documentation
6. **Metrics** - Performance benchmarks and logs

## 🔗 API Endpoints

```
POST   /api/v1/video/upload      - Upload CCTV footage
GET    /api/v1/analytics/footfall - Customer count analytics
GET    /api/v1/inventory/status   - Real-time shelf status
GET    /api/v1/alerts             - System alerts
POST   /api/v1/inference/process  - Process video frame
GET    /api/v1/health             - Health check
```

## 📝 Technologies Used

- **Deep Learning:** PyTorch, YOLOv8, Transformers
- **Computer Vision:** OpenCV, Albumentations
- **Tracking:** DeepSORT, ByteTrack
- **API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **MLOps:** MLflow, Weights & Biases
- **Deployment:** Docker, AWS (EC2, S3)
- **Database:** SQLite (can scale to PostgreSQL)

## 🎓 Learning Outcomes

After completing this project, you demonstrate:
- End-to-end ML pipeline development
- Production-grade CV system architecture
- Real-time video processing at scale
- MLOps best practices
- API design and integration
- Containerization and deployment
- Performance optimization

## 📧 Project Presentation

Include in your resume:
- GitHub repository link
- Live demo video (2-3 mins)
- Architecture diagram
- Performance metrics
- Sample API responses

## 🚨 Troubleshooting

**Issue:** Out of memory during training
**Fix:** Reduce batch size in config files

**Issue:** Slow inference
**Fix:** Use smaller models (YOLOv8n, ViT-tiny) or CPU optimization

**Issue:** Missing dependencies
**Fix:** `pip install -r requirements.txt --upgrade`

## 📄 License

MIT License - Free for portfolio and resume use

## 🤝 Contributing

This is a resume project template. Fork and customize for your needs!

---

**Estimated Completion Time:** 3-4 hours
**Difficulty:** Intermediate to Advanced
**Best For:** ML Engineer, CV Engineer, Data Scientist roles
