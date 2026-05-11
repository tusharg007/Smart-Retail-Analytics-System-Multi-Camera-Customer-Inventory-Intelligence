#!/usr/bin/env python3
"""
Streamlit Dashboard for Retail Analytics
Real-time visualization of customer and inventory data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import requests
import cv2
import threading
import time
import tempfile
import sys
from pathlib import Path

# Add src to path for VideoProcessor
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.inference.run_inference import VideoProcessor
except ImportError:
    VideoProcessor = None

# Page configuration
st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 0.5rem;
        border-left: 4px solid #ff0000;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff4cc;
        padding: 0.5rem;
        border-left: 4px solid #ffaa00;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #ccf2ff;
        padding: 0.5rem;
        border-left: 4px solid #0088ff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000/api/v1"

def fetch_api_data(endpoint):
    """Fetch data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return get_mock_data(endpoint)

def get_mock_data(endpoint):
    """Generate mock data for demonstration"""
    if endpoint == "analytics/footfall":
        return {
            "total_customers_today": 245,
            "current_occupancy": 18,
            "avg_dwell_time_minutes": 12.5,
            "peak_hour": "14:00-15:00",
            "hourly_breakdown": [
                {"hour": f"{h:02d}:00", "count": np.random.randint(15, 55)}
                for h in range(9, 18)
            ]
        }
    elif endpoint == "inventory/status":
        return {
            "total_products": 150,
            "low_stock_items": 8,
            "out_of_stock_items": 2,
            "products": [
                {"product_id": f"PROD{i:03d}", "name": f"Product {chr(65+i)}", 
                 "stock_level": np.random.choice(["high", "medium", "low", "empty"]),
                 "quantity_estimated": np.random.randint(0, 100)}
                for i in range(10)
            ]
        }
    elif endpoint == "alerts":
        return {
            "total_alerts": 3,
            "active_alerts": [
                {"alert_id": "ALT001", "severity": "medium", 
                 "message": "Product B stock level below threshold"},
                {"alert_id": "ALT002", "severity": "high", 
                 "message": "Product C is out of stock"},
                {"alert_id": "ALT003", "severity": "low", 
                 "message": "Checkout queue length exceeds 5 people"}
            ]
        }
    return {}

def video_capture_thread(source, stop_event):
    """Thread for capturing and processing video"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.session_state['stream_error'] = "Could not connect to stream. Check your RTSP URL or Webcam."
        return

    processor = VideoProcessor() if VideoProcessor else None
    frame_count = 0
    start_time = time.time()

    while not stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_fps = frame_count / (time.time() - start_time + 1e-6)
        
        if processor:
            detections = processor.process_frame(frame)
            frame = processor.draw_detections(frame, detections)
            frame = processor.add_analytics_overlay(frame, detections, fps=current_fps)
            
            st.session_state['live_occupancy'] = len(detections)
            st.session_state['live_fps'] = round(current_fps, 1)

        st.session_state['current_frame'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1
        time.sleep(0.03)

    cap.release()

def main():
    st.markdown('<div class="main-header">🏪 Smart Retail Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.sidebar.title("🎥 Input Source")
    input_mode = st.sidebar.radio(
        "Select Video Source",
        ["Demo Mode", "Upload Video", "Live Camera (RTSP)", "Webcam"]
    )
    
    # Initialize thread states
    if 'stop_event' not in st.session_state:
        st.session_state['stop_event'] = threading.Event()
    
    # Handle Live/Upload Streams
    if input_mode != "Demo Mode":
        st.header("🔴 Live Processing Feed")
        
        source = None
        if input_mode == "Upload Video":
            uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi'])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                source = tfile.name
        elif input_mode == "Live Camera (RTSP)":
            source = st.sidebar.text_input("RTSP URL", value="rtsp://")
        elif input_mode == "Webcam":
            source = 0
            
        if source is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Stream"):
                    st.session_state['stop_event'].clear()
                    st.session_state['stream_error'] = None
                    threading.Thread(target=video_capture_thread, args=(source, st.session_state['stop_event']), daemon=True).start()
            with col2:
                if st.button("⏹️ Stop Stream"):
                    st.session_state['stop_event'].set()

            st_frame = st.empty()
            
            # Setup sidebar metrics for live view
            st.sidebar.markdown("---")
            st.sidebar.subheader("Live Metrics")
            occ_placeholder = st.sidebar.empty()
            fps_placeholder = st.sidebar.empty()
            
            # Snapshots
            if st.button("📸 Snapshot"):
                if 'current_frame' in st.session_state:
                    img_bgr = cv2.cvtColor(st.session_state['current_frame'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite("snapshot.jpg", img_bgr)
                    st.success("Snapshot saved as snapshot.jpg")
            
            # Update loop
            while not st.session_state['stop_event'].is_set():
                if st.session_state.get('stream_error'):
                    st.error(st.session_state['stream_error'])
                    break
                    
                if 'current_frame' in st.session_state:
                    st_frame.image(st.session_state['current_frame'], channels="RGB")
                    occ_placeholder.metric("Current Occupancy", st.session_state.get('live_occupancy', 0))
                    fps_placeholder.metric("FPS", st.session_state.get('live_fps', 0.0))
                time.sleep(0.1)

    st.sidebar.markdown("---")
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Customer Analytics", "Inventory Management", "System Health"]
    )
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
        
    if page == "Overview":
        show_overview()
    elif page == "Customer Analytics":
        show_customer_analytics()
    elif page == "Inventory Management":
        show_inventory_management()
    elif page == "System Health":
        show_system_health()

def show_overview():
    st.header("📈 Overview")
    
    footfall_data = fetch_api_data("analytics/footfall")
    inventory_data = fetch_api_data("inventory/status")
    alerts_data = fetch_api_data("alerts")
    
    # AI Report Section
    st.subheader("🤖 AI Store Manager Report")
    if st.button("Generate AI Report"):
        with st.spinner("Analyzing current store state..."):
            payload = {
                "occupancy": footfall_data.get("current_occupancy", 0),
                "avg_dwell_time": footfall_data.get("avg_dwell_time_minutes", 0),
                "active_alerts": [a.get("message") for a in alerts_data.get("active_alerts", [])]
            }
            try:
                res = requests.post(f"{API_BASE_URL}/generate-report", json=payload, timeout=10)
                if res.status_code == 200:
                    st.info(res.json().get("report", "Report unavailable."))
                else:
                    st.warning("Report unavailable. API returned an error.")
            except:
                st.warning("Report unavailable. Could not connect to API.")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers Today", footfall_data.get("total_customers_today", 0), "+12% vs yesterday")
    with col2:
        st.metric("Current Occupancy", footfall_data.get("current_occupancy", 0), "Live")
    with col3:
        st.metric("Low Stock Items", inventory_data.get("low_stock_items", 0), "-2 since yesterday")
    with col4:
        st.metric("Active Alerts", alerts_data.get("total_alerts", 0), "Requires attention")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly Footfall")
        hourly_data = footfall_data.get("hourly_breakdown", [])
        if hourly_data:
            df = pd.DataFrame(hourly_data)
            fig = px.line(df, x='hour', y='count', title='Customer Traffic Throughout the Day')
            fig.update_traces(line_color='#1f77b4', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.subheader("Inventory Status")
        inv_status = {"High Stock": 110, "Medium Stock": 30, "Low Stock": 8, "Out of Stock": 2}
        fig = go.Figure(data=[go.Pie(labels=list(inv_status.keys()), values=list(inv_status.values()), hole=.4)])
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🚨 Recent Alerts")
    alerts = alerts_data.get("active_alerts", [])
    for alert in alerts[:5]:
        severity_class = f"alert-{alert.get('severity', 'low')}"
        st.markdown(f'<div class="{severity_class}"><strong>{alert.get("alert_id")}</strong>: {alert.get("message")}</div>', unsafe_allow_html=True)

def show_customer_analytics():
    st.header("👥 Customer Analytics")
    footfall_data = fetch_api_data("analytics/footfall")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", footfall_data.get("total_customers_today", 0))
    with col2:
        st.metric("Avg Dwell Time", f"{footfall_data.get('avg_dwell_time_minutes', 0):.1f} min")
    with col3:
        st.metric("Peak Hour", footfall_data.get("peak_hour", "N/A"))
        
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly Traffic Pattern")
        hourly_data = footfall_data.get("hourly_breakdown", [])
        if hourly_data:
            df = pd.DataFrame(hourly_data)
            fig = px.bar(df, x='hour', y='count', title='Customer Count by Hour', color='count', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.subheader("Dwell Time Distribution")
        dwell_times = np.random.normal(12.5, 3, 100)
        fig = px.histogram(dwell_times, nbins=20, title='Customer Dwell Time Distribution')
        st.plotly_chart(fig, use_container_width=True)

def show_inventory_management():
    st.header("📦 Inventory Management")
    inventory_data = fetch_api_data("inventory/status")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", inventory_data.get("total_products", 0))
    with col2:
        st.metric("Low Stock", inventory_data.get("low_stock_items", 0), delta="-2", delta_color="inverse")
    with col3:
        st.metric("Out of Stock", inventory_data.get("out_of_stock_items", 0), delta="0")
    with col4:
        st.metric("Stock Accuracy", "94.2%", delta="+1.5%")
        
    st.markdown("---")
    st.subheader("Product Status")
    products = inventory_data.get("products", [])
    if products:
        df = pd.DataFrame(products)
        def color_stock_level(val):
            colors = {'empty': '#ffcccc', 'low': '#fff4cc', 'medium': '#ffffcc', 'high': '#ccffcc'}
            return f"background-color: {colors.get(val, '')}"
        styled_df = df.style.applymap(color_stock_level, subset=['stock_level'])
        st.dataframe(styled_df, use_container_width=True)

def show_system_health():
    st.header("⚙️ System Health")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cameras Active", "4/4", "All operational")
    with col2:
        st.metric("Uptime", "99.8%", "+0.2%")
    with col3:
        st.metric("Avg Latency", "45ms", "-5ms")
    with col4:
        st.metric("Model Accuracy", "92.3%", "+0.5%")
        
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Processing Performance")
        hours = [f"{h:02d}:00" for h in range(9, 18)]
        fps = [28 + np.random.randint(-3, 4) for _ in hours]
        df = pd.DataFrame({'Time': hours, 'FPS': fps})
        fig = px.line(df, x='Time', y='FPS', title='Processing FPS Over Time', markers=True)
        fig.add_hline(y=25, line_dash="dash", line_color="red", annotation_text="Target FPS")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Model Confidence")
        confidence = np.random.normal(0.87, 0.05, 100)
        fig = px.histogram(confidence, nbins=20, title='Detection Confidence Distribution')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
