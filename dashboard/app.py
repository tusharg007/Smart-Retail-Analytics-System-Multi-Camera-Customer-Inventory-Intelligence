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

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def fetch_api_data(endpoint):
    """Fetch data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        # Return mock data if API is not running
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

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">🏪 Smart Retail Analytics Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Customer Analytics", "Inventory Management", "System Health"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Live Dashboard**\n\n"
        "Real-time monitoring of retail operations using computer vision"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Display selected page
    if page == "Overview":
        show_overview()
    elif page == "Customer Analytics":
        show_customer_analytics()
    elif page == "Inventory Management":
        show_inventory_management()
    elif page == "System Health":
        show_system_health()

def show_overview():
    """Show overview dashboard"""
    
    st.header("📈 Overview")
    
    # Fetch data
    footfall_data = fetch_api_data("analytics/footfall")
    inventory_data = fetch_api_data("inventory/status")
    alerts_data = fetch_api_data("alerts")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Customers Today",
            footfall_data.get("total_customers_today", 0),
            "+12% vs yesterday"
        )
    
    with col2:
        st.metric(
            "Current Occupancy",
            footfall_data.get("current_occupancy", 0),
            "Live"
        )
    
    with col3:
        st.metric(
            "Low Stock Items",
            inventory_data.get("low_stock_items", 0),
            "-2 since yesterday"
        )
    
    with col4:
        st.metric(
            "Active Alerts",
            alerts_data.get("total_alerts", 0),
            "Requires attention"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Footfall")
        hourly_data = footfall_data.get("hourly_breakdown", [])
        if hourly_data:
            df = pd.DataFrame(hourly_data)
            fig = px.line(df, x='hour', y='count', 
                         title='Customer Traffic Throughout the Day',
                         labels={'count': 'Number of Customers', 'hour': 'Time'})
            fig.update_traces(line_color='#1f77b4', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Inventory Status")
        inv_status = {
            "High Stock": 110,
            "Medium Stock": 30,
            "Low Stock": 8,
            "Out of Stock": 2
        }
        fig = go.Figure(data=[go.Pie(
            labels=list(inv_status.keys()),
            values=list(inv_status.values()),
            hole=.4
        )])
        fig.update_layout(title_text='Product Stock Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Alerts
    st.subheader("🚨 Recent Alerts")
    alerts = alerts_data.get("active_alerts", [])
    for alert in alerts[:5]:
        severity_class = f"alert-{alert.get('severity', 'low')}"
        st.markdown(
            f'<div class="{severity_class}"><strong>{alert.get("alert_id")}</strong>: '
            f'{alert.get("message")}</div>',
            unsafe_allow_html=True
        )

def show_customer_analytics():
    """Show customer analytics page"""
    
    st.header("👥 Customer Analytics")
    
    footfall_data = fetch_api_data("analytics/footfall")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", footfall_data.get("total_customers_today", 0))
    with col2:
        st.metric("Avg Dwell Time", f"{footfall_data.get('avg_dwell_time_minutes', 0):.1f} min")
    with col3:
        st.metric("Peak Hour", footfall_data.get("peak_hour", "N/A"))
    
    st.markdown("---")
    
    # Detailed analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Traffic Pattern")
        hourly_data = footfall_data.get("hourly_breakdown", [])
        if hourly_data:
            df = pd.DataFrame(hourly_data)
            fig = px.bar(df, x='hour', y='count',
                        title='Customer Count by Hour',
                        color='count',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Dwell Time Distribution")
        dwell_times = np.random.normal(12.5, 3, 100)
        fig = px.histogram(dwell_times, nbins=20,
                          title='Customer Dwell Time Distribution',
                          labels={'value': 'Dwell Time (minutes)', 'count': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("🗺️ Store Heatmap")
    st.info("Heatmap showing high-traffic areas in the store")
    
    # Generate sample heatmap data
    heatmap_data = np.random.rand(10, 15) * 100
    fig = px.imshow(heatmap_data,
                    labels=dict(x="Store Width", y="Store Depth", color="Traffic Density"),
                    title="Store Traffic Heatmap",
                    color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

def show_inventory_management():
    """Show inventory management page"""
    
    st.header("📦 Inventory Management")
    
    inventory_data = fetch_api_data("inventory/status")
    
    # Metrics
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
    
    # Product table
    st.subheader("Product Status")
    products = inventory_data.get("products", [])
    
    if products:
        df = pd.DataFrame(products)
        
        # Color code stock levels
        def color_stock_level(val):
            if val == 'empty':
                return 'background-color: #ffcccc'
            elif val == 'low':
                return 'background-color: #fff4cc'
            elif val == 'medium':
                return 'background-color: #ffffcc'
            else:
                return 'background-color: #ccffcc'
        
        styled_df = df.style.applymap(color_stock_level, subset=['stock_level'])
        st.dataframe(styled_df, use_container_width=True)
    
    # Stock level distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if products:
            stock_counts = pd.Series([p['stock_level'] for p in products]).value_counts()
            fig = px.bar(x=stock_counts.index, y=stock_counts.values,
                        title='Products by Stock Level',
                        labels={'x': 'Stock Level', 'y': 'Count'},
                        color=stock_counts.values)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quantity distribution
        if products:
            quantities = [p['quantity_estimated'] for p in products]
            fig = px.box(y=quantities,
                        title='Product Quantity Distribution',
                        labels={'y': 'Quantity'})
            st.plotly_chart(fig, use_container_width=True)

def show_system_health():
    """Show system health page"""
    
    st.header("⚙️ System Health")
    
    # System metrics
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
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Processing Performance")
        hours = [f"{h:02d}:00" for h in range(9, 18)]
        fps = [28 + np.random.randint(-3, 4) for _ in hours]
        df = pd.DataFrame({'Time': hours, 'FPS': fps})
        fig = px.line(df, x='Time', y='FPS',
                     title='Processing FPS Over Time',
                     markers=True)
        fig.add_hline(y=25, line_dash="dash", line_color="red", 
                     annotation_text="Target FPS")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Confidence")
        confidence = np.random.normal(0.87, 0.05, 100)
        fig = px.histogram(confidence, nbins=20,
                          title='Detection Confidence Distribution',
                          labels={'value': 'Confidence Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    # System logs
    st.subheader("📋 Recent System Logs")
    logs = pd.DataFrame({
        'Timestamp': pd.date_range(end=datetime.now(), periods=10, freq='5min')[::-1],
        'Level': np.random.choice(['INFO', 'WARNING', 'ERROR'], 10, p=[0.7, 0.25, 0.05]),
        'Message': [
            'Video processing completed successfully',
            'Model inference latency within threshold',
            'Camera 3 reconnected after brief disconnect',
            'Daily analytics report generated',
            'Low stock alert triggered for Product B',
            'Backup completed successfully',
            'API health check passed',
            'Frame processing queue normal',
            'Storage usage at 65%',
            'System startup completed'
        ]
    })
    st.dataframe(logs, use_container_width=True)

if __name__ == "__main__":
    main()
