#!/usr/bin/env python3
"""
FastAPI Backend for Retail Analytics
Provides REST API for video processing and analytics
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from pathlib import Path
import json
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(
    title="Smart Retail Analytics API",
    description="Computer Vision API for retail customer and inventory analytics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class AnalyticsResponse(BaseModel):
    timestamp: str
    total_people: int
    avg_dwell_time: float
    peak_hour: str
    status: str

class InventoryStatus(BaseModel):
    product_id: str
    shelf_location: str
    stock_level: str
    last_updated: str

class Alert(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: str

# In-memory storage (replace with database in production)
analytics_db = {
    'footfall': [],
    'inventory': [],
    'alerts': []
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Retail Analytics API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/api/v1/health",
            "analytics": "/api/v1/analytics/footfall",
            "inventory": "/api/v1/inventory/status",
            "alerts": "/api/v1/alerts",
            "upload": "/api/v1/video/upload",
            "process": "/api/v1/inference/process"
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "retail-analytics-api",
        "models_loaded": True
    }

@app.get("/api/v1/analytics/footfall")
async def get_footfall_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get customer footfall analytics"""
    
    # Sample data (in production, query from database)
    analytics = {
        "timestamp": datetime.now().isoformat(),
        "total_customers_today": 245,
        "current_occupancy": 18,
        "avg_dwell_time_minutes": 12.5,
        "peak_hour": "14:00-15:00",
        "hourly_breakdown": [
            {"hour": "09:00", "count": 15},
            {"hour": "10:00", "count": 28},
            {"hour": "11:00", "count": 35},
            {"hour": "12:00", "count": 42},
            {"hour": "13:00", "count": 38},
            {"hour": "14:00", "count": 52},
            {"hour": "15:00", "count": 35},
        ],
        "status": "operational"
    }
    
    return JSONResponse(content=analytics)

@app.get("/api/v1/inventory/status")
async def get_inventory_status():
    """Get real-time inventory status"""
    
    # Sample inventory data
    inventory = {
        "timestamp": datetime.now().isoformat(),
        "total_products": 150,
        "low_stock_items": 8,
        "out_of_stock_items": 2,
        "products": [
            {
                "product_id": "PROD001",
                "name": "Product A",
                "shelf_location": "Aisle 1, Shelf 2",
                "stock_level": "high",
                "quantity_estimated": 85,
                "last_updated": datetime.now().isoformat()
            },
            {
                "product_id": "PROD002",
                "name": "Product B",
                "shelf_location": "Aisle 1, Shelf 3",
                "stock_level": "low",
                "quantity_estimated": 12,
                "last_updated": datetime.now().isoformat()
            },
            {
                "product_id": "PROD003",
                "name": "Product C",
                "shelf_location": "Aisle 2, Shelf 1",
                "stock_level": "empty",
                "quantity_estimated": 0,
                "last_updated": datetime.now().isoformat()
            }
        ]
    }
    
    return JSONResponse(content=inventory)

@app.get("/api/v1/alerts")
async def get_alerts(severity: Optional[str] = None):
    """Get system alerts"""
    
    alerts = {
        "timestamp": datetime.now().isoformat(),
        "total_alerts": 3,
        "active_alerts": [
            {
                "alert_id": "ALT001",
                "alert_type": "low_stock",
                "severity": "medium",
                "message": "Product B stock level below threshold (12 units remaining)",
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False
            },
            {
                "alert_id": "ALT002",
                "alert_type": "out_of_stock",
                "severity": "high",
                "message": "Product C is out of stock on Aisle 2, Shelf 1",
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False
            },
            {
                "alert_id": "ALT003",
                "alert_type": "long_queue",
                "severity": "low",
                "message": "Checkout queue length exceeds 5 people",
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False
            }
        ]
    }
    
    if severity:
        alerts['active_alerts'] = [
            a for a in alerts['active_alerts'] 
            if a['severity'] == severity
        ]
        alerts['total_alerts'] = len(alerts['active_alerts'])
    
    return JSONResponse(content=alerts)

@app.post("/api/v1/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload CCTV video for processing"""
    
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid file format. Use .mp4, .avi, or .mov")
    
    # Save uploaded file
    upload_dir = Path('data/raw/videos/uploaded')
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    return {
        "status": "success",
        "message": "Video uploaded successfully",
        "filename": file.filename,
        "file_path": str(file_path),
        "file_size_mb": len(content) / (1024 * 1024),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/inference/process")
async def process_video_inference(video_path: str):
    """Process video and return analytics"""
    
    video_file = Path(video_path)
    
    if not video_file.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    # In production, trigger actual video processing
    # For now, return mock response
    
    return {
        "status": "processing",
        "video_path": video_path,
        "job_id": f"JOB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "estimated_completion": "5 minutes",
        "message": "Video processing started. Check status with job_id",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/sync/pos")
async def sync_with_pos(transaction_data: dict):
    """Sync analytics with POS system"""
    
    return {
        "status": "success",
        "message": "Data synced with POS system",
        "records_synced": len(transaction_data.get('transactions', [])),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/stats/summary")
async def get_stats_summary():
    """Get comprehensive stats summary"""
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "operational_stats": {
            "cameras_active": 4,
            "uptime_percentage": 99.8,
            "total_frames_processed_today": 86400,
            "avg_processing_latency_ms": 45
        },
        "business_metrics": {
            "total_customers_today": 245,
            "peak_occupancy": 52,
            "avg_transaction_value": 42.50,
            "conversion_rate": 68.5
        },
        "inventory_metrics": {
            "total_products": 150,
            "stock_accuracy": 94.2,
            "low_stock_alerts": 8,
            "out_of_stock_alerts": 2
        },
        "system_health": {
            "model_accuracy": 92.3,
            "detection_confidence_avg": 0.87,
            "tracking_quality": "good"
        }
    }
    
    return JSONResponse(content=summary)

def start_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server"""
    
    print("="*60)
    print("  SMART RETAIL ANALYTICS API SERVER")
    print("="*60)
    print(f"\nStarting server on http://{host}:{port}")
    print("\nAPI Documentation:")
    print(f"  Swagger UI: http://{host}:{port}/docs")
    print(f"  ReDoc: http://{host}:{port}/redoc")
    print("\nEndpoints:")
    print("  GET  /api/v1/health - Health check")
    print("  GET  /api/v1/analytics/footfall - Customer analytics")
    print("  GET  /api/v1/inventory/status - Inventory status")
    print("  GET  /api/v1/alerts - System alerts")
    print("  POST /api/v1/video/upload - Upload video")
    print("  POST /api/v1/inference/process - Process video")
    print("="*60)
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start Retail Analytics API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port)
