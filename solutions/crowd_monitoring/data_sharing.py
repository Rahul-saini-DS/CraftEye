"""
Data sharing helper for the CraftEye platform.
This module helps manage cross-tab data persistence for metrics.
"""
import os
import time
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
import json

def ensure_dashboard_data_sharing(feed_id: str, metrics: Dict[str, Any]) -> None:
    """
    Write current metrics to a shared location for dashboard access.
    This enables the dashboard to read latest data even when opened in a new tab.
    
    Args:
        feed_id: The camera feed ID
        metrics: Dictionary of metrics to share
    """
    if not feed_id or not metrics:
        return
        
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Add timestamp to metrics
    metrics_with_ts = metrics.copy()
    metrics_with_ts["timestamp"] = time.time()
    
    # Use feed_id as identifier in the shared data file
    try:
        # Write current metrics to file for dashboard to read
        filename = os.path.join("data", f"metrics_{feed_id}.json")
        with open(filename, "w") as f:
            json.dump(metrics_with_ts, f)
    except Exception as e:
        # Silent fail - this is just for cross-tab sharing
        if st.session_state.get("debug_mode", False):
            print(f"Error sharing metrics: {str(e)}")

def read_dashboard_data(feed_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Read the latest metrics from shared storage.
    
    Args:
        feed_id: The camera feed ID (optional, will use session_state if not provided)
        
    Returns:
        Dictionary with the latest metrics
    """
    # Use provided feed_id or get from session state
    fid = feed_id or st.session_state.get("current_feed_id")
    if not fid:
        return {}
        
    try:
        # Look for metrics file
        filename = os.path.join("data", f"metrics_{fid}.json")
        if not os.path.exists(filename):
            return {}
            
        # Check if file is recent (< 30 seconds old)
        if time.time() - os.path.getmtime(filename) > 30:
            # Data is stale
            return {}
            
        # Read metrics
        with open(filename, "r") as f:
            metrics = json.load(f)
            
        return metrics
    except Exception as e:
        # Silent fail - this is just for cross-tab sharing
        if st.session_state.get("debug_mode", False):
            print(f"Error reading metrics: {str(e)}")
        return {}
        
def check_active_monitoring() -> bool:
    """
    Check if monitoring is actively running in any tab.
    This helps determine if dashboard should use cached data or wait for updates.
    
    Returns:
        True if monitoring appears to be active, False otherwise
    """
    # Look for any recent metrics files
    try:
        data_dir = "data"
        if not os.path.exists(data_dir):
            return False
            
        # Check for any metrics files updated in last 10 seconds
        for filename in os.listdir(data_dir):
            if filename.startswith("metrics_") and filename.endswith(".json"):
                file_path = os.path.join(data_dir, filename)
                if time.time() - os.path.getmtime(file_path) < 10:
                    return True
    except Exception:
        pass
        
    return False
