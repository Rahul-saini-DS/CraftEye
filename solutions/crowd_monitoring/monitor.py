"""
Crowd monitoring module for the CraftEye platform.

This module provides crowd density analysis, flow tracking, and predictive alerts.
"""
import os
import time
import cv2
import numpy as np
import hashlib
import pandas as pd
import folium
from streamlit_folium import folium_static
from typing import Tuple, Optional, Dict, Any
import streamlit as st
from pipeline import Inference, get_device_config
from capture.camera_manager import CameraManager
from preprocess import open_video_capture
from solutions.crowd_monitoring.flow_tracking import update_flow_counts, draw_flow_lines, initialize_flow_tracking
from solutions.crowd_monitoring.roi_utils import filter_detections_by_roi, draw_roi_polygon
from solutions.crowd_monitoring.predictive_analytics import update_analytics, display_predictive_analytics
from solutions.crowd_monitoring.data_persistence import write_crowd_data_row, classify_density_level
from solutions.crowd_monitoring.zone_persistence import write_zone_data
from solutions.crowd_monitoring.error_helper import user_friendly_error_message
import datetime
import atexit

# ---------------------------
# Utilities
# ---------------------------

def _sig(d: Dict[str, Any]) -> str:
    return hashlib.md5(str(d).encode()).hexdigest()[:8]

def _settings_sig(s: Dict[str, Any]) -> str:
    """Signature for settings that actually change the model construction."""
    return hashlib.md5(str({
        "primary_model": s.get("primary_model", ""),
        "imgsz": s.get("imgsz", 640),
        "confidence_threshold": s.get("confidence_threshold", 0.25),
        "device": s.get("device", "cpu"),
        "half": s.get("half", True),
    }).encode()).hexdigest()

def _stop_all_feeds_safely():
    """Best-effort stop of all feeds to prevent lingering threads on shutdown."""
    try:
        if "cam_manager" in st.session_state:
            st.session_state.cam_manager.stop_all()
    except Exception:
        pass

atexit.register(_stop_all_feeds_safely)

# ---------------------------
# Model init (cached)
# ---------------------------

@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path, imgsz, conf, device, half):
    return Inference(
        model_path=model_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
        half=half
    )

def initialize_model() -> bool:
    """Initialize the YOLO model once per distinct model settings (reuses instance across reruns)."""
    if "model_settings" not in st.session_state:
        st.warning("No model configuration found. Please go to Tasks page to configure the model.")
        # Initialize default model settings to prevent crashes
        st.session_state.model_settings = {
            'primary_model': 'models/yolov8n.pt',
            'secondary_model': None,
            'confidence_threshold': 0.5,
            'imgsz': 640,
            'task_types': ['detect'],
            'device': None
        }
        # Show message but continue with defaults
        st.info("Using default model settings")

    settings = st.session_state.model_settings
    sig = _settings_sig(settings)

    # Fast path: already initialized for these settings
    if st.session_state.get("model") is not None and st.session_state.get("model_settings_hash") == sig:
        return True

    model_path = settings.get("primary_model", "models/yolov8n.pt")

    # Check existence and try fallback
    if not os.path.exists(model_path):
        model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "models"
        if os.path.exists(model_dir):
            alts = [f for f in os.listdir(model_dir)
                    if f.endswith(('.pt', '.onnx')) and os.path.isfile(os.path.join(model_dir, f))]
            if alts:
                alt = os.path.join(model_dir, alts[0])
                st.warning(f"Model {model_path} not found. Using alternative model: {alt}")
                model_path = alt
            else:
                st.error(f"Model file not found: {model_path} and no alternative models found in {model_dir}")
                st.info("Place a YOLO model (e.g., yolo11n.pt) in the 'models' directory.")
                return False
        else:
            st.error(f"Model directory not found: {model_dir}")
            st.info("Create a 'models' directory and place a YOLO model inside.")
            return False

    try:
        # Build or fetch from cache
        model = _load_model_cached(
            model_path=model_path,
            imgsz=settings.get("imgsz", 640),
            conf=settings.get("confidence_threshold", 0.25),
            device=settings.get("device", "cpu"),
            half=settings.get("half", True if settings.get("device", "cpu") != "cpu" else False),  # safer on CPU
        )
        st.session_state["model"] = model
        st.session_state["model_settings_hash"] = sig

        # One-time warmup per setting signature
        if st.session_state.get("model_warm_sig") != sig:
            try:
                blank = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = model.predict_single(blank)
                st.session_state["model_warm_sig"] = sig
            except Exception as warm_err:
                st.warning(f"Model warmup failed, but continuing: {str(warm_err)}")
                # Continue despite warmup error

        return True
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.info("Try going to the Tasks page and selecting a different model or checking the model path.")
        # Try recovery by clearing model state
        if "model" in st.session_state:
            del st.session_state["model"]
        if "model_settings_hash" in st.session_state:
            del st.session_state["model_settings_hash"]
        return False

# ---------------------------
# Session bootstrap
# ---------------------------

def initialize_crowd_monitoring():
    """Initialize crowd monitoring state if it doesn't exist."""
    from solutions.crowd_monitoring.monitor_bridge import initialize_crowd_monitoring_bridge
    initialize_crowd_monitoring_bridge()
    if 'feeds_meta' not in st.session_state:
        st.session_state.feeds_meta = {}
    if st.session_state.get("monitoring_active", False) and "model" not in st.session_state:
        initialize_model()

# ---------------------------
# Heatmap (incremental & light)
# ---------------------------

def _overlay_from_accum(frame, accum, colormap):
    acc_norm = accum.copy()
    if acc_norm.max() > 0:
        acc_norm = acc_norm / acc_norm.max()
    heatmap_img = (acc_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_img, colormap)
    return cv2.addWeighted(frame, 0.55, colored, 0.45, 0)

def generate_density_heatmap(frame, results, colormap=cv2.COLORMAP_JET, decay=0.92, stamp_radius=6, blur_ksize=15):
    """
    Optimized incremental heatmap:
    - keeps a per-feed accumulator with decay
    - uses downsampling for performance
    - throttles updates to improve FPS
    """
    H, W = frame.shape[:2]
    feed_id = st.session_state.get("current_feed_id")
    
    # Throttle heatmap updates for performance (update every N frames)
    if not hasattr(st.session_state, 'heatmap_frame_counter'):
        st.session_state.heatmap_frame_counter = {}
    if not hasattr(st.session_state, 'last_heatmap'):
        st.session_state.last_heatmap = {}
    
    # Initialize feed-specific counters
    feed_counter = st.session_state.heatmap_frame_counter.setdefault(feed_id, 0)
    update_interval = 5  # Higher number = better performance but less responsive heatmap
    st.session_state.heatmap_frame_counter[feed_id] = (feed_counter + 1) % update_interval
    
    if feed_counter != 0 and feed_id in st.session_state.last_heatmap:
        # Skip computation and return cached heatmap
        return st.session_state.last_heatmap[feed_id]

    if 'heatmap_accum' not in st.session_state:
        st.session_state.heatmap_accum = {}
    if feed_id not in st.session_state.heatmap_accum:
        st.session_state.heatmap_accum[feed_id] = np.zeros((H, W), dtype=np.float32)

    accum = st.session_state.heatmap_accum[feed_id]
    # decay previous frame’s energy
    cv2.multiply(accum, decay, dst=accum)

    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        # Use downsampling for better performance
        small_scale = 0.3  # Reduced from 0.5 to 0.3 for better performance
        h2, w2 = int(H * small_scale), int(W * small_scale)
        
        # Create stamp on smaller canvas (much faster)
        stamp = np.zeros((h2, w2), dtype=np.uint8)
        
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            # Scale coordinates to smaller size
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx_small = int(cx * small_scale)
            cy_small = int(cy * small_scale)
            cv2.circle(stamp, (cx_small, cy_small), stamp_radius, 255, -1)
        
        # Blur at smaller resolution (much faster)
        stamp = cv2.GaussianBlur(stamp, (blur_ksize, blur_ksize), 0)
        
        # Scale back to original size
        stamp = cv2.resize(stamp, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Add to accumulator
        inc = (stamp.astype(np.float32) / 255.0) * 0.6
        cv2.add(accum, inc, dst=accum, dtype=cv2.CV_32F)

    # Cache the result for subsequent frames
    result = _overlay_from_accum(frame, accum, colormap)
    st.session_state.last_heatmap[feed_id] = result
    return result

# ---------------------------
# Density & metrics
# ---------------------------

def calculate_crowd_density(detections, frame_area):
    """
    Calculate crowd density in people per square meter.
    """
    if frame_area <= 0:
        return 0
    num_people = len(detections) if detections is not None else 0
    return num_people / frame_area

def calculate_advanced_metrics(frame, detections, current_density):
    """
    Calculate advanced crowd metrics.
    """
    # Get the current crowd stats or initialize if missing
    # Note: Basic initialization happens in monitor_bridge.py, but we add any extra fields needed here
    if "crowd_stats" not in st.session_state:
        from solutions.crowd_monitoring.monitor_bridge import initialize_crowd_monitoring_bridge
        initialize_crowd_monitoring_bridge()
    
    # Add extra fields needed for this function that aren't in the bridge initialization
    st.session_state.crowd_stats.setdefault("density_history", [])
    st.session_state.crowd_stats.setdefault("timestamps", [])

    stats = st.session_state.crowd_stats
    
    # Ensure the last_update_time key exists
    if "last_update_time" not in stats:
        stats["last_update_time"] = time.time()

    # Occupancy Distribution (% clustered)
    high_density_threshold = 60  # pixels ~ 1.5m proxy
    if detections is not None and len(detections) > 1:
        centers = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        high_density_count = 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                if dist < high_density_threshold:
                    high_density_count += 1
                    break
        if len(centers) > 0:
            stats["occupancy_distribution"] = int((high_density_count / len(centers)) * 100)

    # Flow Rate (net per minute) using count difference
    current_time = time.time()
    elapsed = current_time - stats.get("last_update_time", current_time - 5)
    if elapsed >= 5:
        current_count = len(detections) if detections is not None else 0
        prev = stats.get("prev_count", 0)
        count_diff = current_count - prev
        stats["flow_rate"] = int(count_diff * (60 / max(elapsed, 1e-6)))
        stats["prev_count"] = current_count
        stats["last_update_time"] = current_time

    # Density trend (5-sample window)
    if 'crowd_monitoring' in st.session_state and 'history' in st.session_state.crowd_monitoring:
        density_history = st.session_state.crowd_monitoring['history'].get('density', [])
        if len(density_history) >= 5:
            recent = density_history[-5:]
            avg_old = sum(recent[:2]) / 2
            avg_new = sum(recent[-2:]) / 2
            thr = 0.05
            if avg_new > avg_old * (1 + thr):
                stats["density_trend"] = "increasing"
            elif avg_new < avg_old * (1 - thr):
                stats["density_trend"] = "decreasing"
            else:
                stats["density_trend"] = "stable"

    st.session_state.crowd_stats = stats

def update_crowd_metrics(frame, results, feed_id=None):
    """
    Update monitoring metrics and persist periodically.
    Also performs:
    - Zone-based analysis
    - Density trend tracking
    - Alert level determination
    """
    if frame is None or results is None:
        return 0
    detections = results.boxes if hasattr(results, 'boxes') else None
    if detections is None:
        return 0

    st.session_state["current_feed_id"] = feed_id

    # Approx frame area (m²) using calibration if present
    frame_area_m2 = None
    if feed_id and "camera_calibrations" in st.session_state and feed_id in st.session_state["camera_calibrations"]:
        frame_area_m2 = st.session_state["camera_calibrations"][feed_id].get("area_m2")
    if not frame_area_m2:
        frame_area_m2 = (frame.shape[0] * frame.shape[1]) / 100000  # fallback proxy

    density = calculate_crowd_density(detections, frame_area_m2)
    density_level = classify_density_level(density)
    
    # Calculate per-zone density (divide frame into 4 quadrants)
    height, width = frame.shape[:2]
    mid_x, mid_y = width // 2, height // 2
    zone_area = frame_area_m2 / 4  # Each zone is 1/4 of the total area
    
    zones = [
        (0, 0, mid_x, mid_y),           # Top-left
        (mid_x, 0, width, mid_y),       # Top-right
        (0, mid_y, mid_x, height),      # Bottom-left
        (mid_x, mid_y, width, height)   # Bottom-right
    ]
    
    zone_counts = [0, 0, 0, 0]
    zone_densities = [0.0, 0.0, 0.0, 0.0]
    zone_names = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
    
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Determine which zone the detection falls into
            if cx < mid_x:
                if cy < mid_y:
                    zone_counts[0] += 1  # Top-left
                else:
                    zone_counts[2] += 1  # Bottom-left
            else:
                if cy < mid_y:
                    zone_counts[1] += 1  # Top-right
                else:
                    zone_counts[3] += 1  # Bottom-right
    
    # Calculate zone densities
    for i in range(4):
        zone_densities[i] = zone_counts[i] / zone_area
        
    # Store zone information in session state for display
    
    # Update predictive analytics with new density data
    if feed_id:
        update_analytics(feed_id, density, time.time())
    if 'zone_analysis' not in st.session_state:
        st.session_state['zone_analysis'] = {
            'counts': zone_counts,
            'densities': zone_densities,
            'names': zone_names,
            'highest_zone': 0
        }
    else:
        st.session_state['zone_analysis']['counts'] = zone_counts
        st.session_state['zone_analysis']['densities'] = zone_densities
        if max(zone_counts) > 0:
            st.session_state['zone_analysis']['highest_zone'] = zone_counts.index(max(zone_counts))

    if 'crowd_monitoring' in st.session_state:
        st.session_state.crowd_monitoring['history']['density'].append(density)
        st.session_state.crowd_monitoring['history']['timestamps'].append(time.time())
        cnt = len(detections) if detections is not None else 0
        if 'counts' not in st.session_state.crowd_monitoring['history']:
            st.session_state.crowd_monitoring['history']['counts'] = []
        st.session_state.crowd_monitoring['history']['counts'].append(cnt)
        # keep last 60 samples
        for key in ['density', 'timestamps', 'counts']:
            if key in st.session_state.crowd_monitoring['history'] and len(st.session_state.crowd_monitoring['history'][key]) > 60:
                st.session_state.crowd_monitoring['history'][key] = st.session_state.crowd_monitoring['history'][key][-60:]

    # advanced metrics
    calculate_advanced_metrics(frame, detections, density)

    # Throttled persistence (every 5s)
    if feed_id:
        now = time.time()
        k = f"last_write_{feed_id}"
        if now - st.session_state.get(k, 0) >= 5:
            count = len(detections) if detections is not None else 0
            write_crowd_data_row(feed_id, count, density, density_level)
            
            # Also persist zone data for dashboard cross-tab viewing
            if 'zone_analysis' in st.session_state:
                zone_counts = st.session_state['zone_analysis']['counts']
                zone_densities = st.session_state['zone_analysis']['densities']
                write_zone_data(feed_id, zone_counts, zone_densities)
            
            st.session_state[k] = now
            
            # Set these flags after writing CSV for cross-tab synchronization
            st.session_state["monitoring_active"] = True
            st.session_state["current_feed_id"] = feed_id
            
            # Set flags so dashboard knows monitoring is active and which feed to use
            st.session_state["monitoring_active"] = True
            st.session_state["current_feed_id"] = feed_id

    return density

# ---------------------------
# Map render (once per feed)
# ---------------------------

def _render_camera_map_once(feed_id, install_coords, location_label):
    key = f"map_rendered_for_{feed_id}"
    if st.session_state.get(key):
        return
    if install_coords and "lat" in install_coords and "lon" in install_coords:
        m = folium.Map(location=[install_coords["lat"], install_coords["lon"]], zoom_start=15)
        folium.Marker(
            [install_coords["lat"], install_coords["lon"]],
            popup=location_label,
            tooltip=location_label,
            icon=folium.Icon(color="red")
        ).add_to(m)
        folium_static(m)
        st.session_state[key] = True

# ---------------------------
# Monitoring: Density
# ---------------------------

def monitor_crowd_density(feed_id, task_settings, should_refresh=False):
    """
    Monitor crowd density for a specific feed.
    
    Args:
        feed_id: The ID of the feed to monitor
        task_settings: The settings for the monitoring task
        should_refresh: Whether to refresh the display this iteration
    """
    density_key = f"density_{feed_id}"
    
    # Create the UI structure only once - persistent across refreshes
    if density_key not in st.session_state:
        st.session_state[density_key] = {}
        st.subheader("Live Video Monitoring")
        
        # Check if density monitoring is enabled for this specific camera
        if "density" not in task_settings or not task_settings["density"].get("enabled", False):
            st.info("Crowd density monitoring is not enabled for this camera.")
            return
            
        # Layout - create once
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Input Video")
            st.session_state[density_key]['input_placeholder'] = st.empty()
        with c2:
            st.markdown("#### Output with Detections")
            st.session_state[density_key]['output_placeholder'] = st.empty()

        st.markdown("#### Density Heatmap")
        st.session_state[density_key]['heatmap_placeholder'] = st.empty()

        # Metrics - create containers once
        metrics_container = st.container()
        with metrics_container:
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            st.session_state[density_key]['count_metric'] = r1c1.empty()
            st.session_state[density_key]['density_metric'] = r1c2.empty() 
            st.session_state[density_key]['status_metric'] = r1c3.empty()
            st.session_state[density_key]['alert_indicator'] = r1c4.empty()

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            if 'crowd_stats' not in st.session_state:
                st.session_state.crowd_stats = {
                    "occupancy_distribution": 0, 
                    "flow_rate": 0,
                    "density_trend": "stable",
                    "alert_level": "green",
                    "prev_count": 0,
                    "prev_density": 0,
                    "last_update_time": time.time(),
                    "density_history": [],
                    "timestamps": []
                }
            st.session_state[density_key]['occupancy_metric'] = r2c1.empty()
            st.session_state[density_key]['flow_metric'] = r2c2.empty()
            st.session_state[density_key]['trend_metric'] = r2c3.empty()
            st.session_state[density_key]['entry_exit_diff'] = r2c4.empty()
            
            # Initialize needed session state variables
            if 'alerts' not in st.session_state:
                st.session_state['alerts'] = []
        
        # Processing status (create once)
        st.session_state[density_key]['processing_status'] = st.empty()
        st.session_state[density_key]['last_ts'] = time.time()
        st.session_state[density_key]['density_settings'] = task_settings.get("density", {})
    
    # Get saved UI components
    ui = st.session_state[density_key]
    density_settings = ui['density_settings']
    
    # Skip processing if no refresh needed
    if not should_refresh:
        return
    
    # feed lookup
    feed = None
    for f in st.session_state.cam_manager.list_feeds():
        if f.config.id == feed_id:
            feed = f
            break
    if feed is None:
        ui['processing_status'].error(f"Feed {feed_id} not found.")
        return

    # start feed once if needed
    if feed.status != "live":
        with st.spinner("Starting feed..."):
            st.session_state.cam_manager.start(feed_id)
        return  # Wait for next refresh to process frames

    # Model should have been initialized in the main monitor function
    if "model" not in st.session_state:
        ui['processing_status'].error("Model not initialized. Please restart monitoring.")
        return
    model = st.session_state["model"]
    
    # Process frame and update UI
    if feed and feed.status == "live":
        try:
            frame = feed.last_frame
            if frame is not None:
                ui['input_placeholder'].image(frame, channels="BGR", use_container_width=True, output_format="PNG", clamp=True)

                results = model.predict_single(frame)

                # ROI filter
                roi = density_settings.get("roi", [])
                if roi and len(roi) >= 3 and hasattr(results, 'boxes'):
                    results.boxes = filter_detections_by_roi(results.boxes, roi)

                detections = results.boxes if hasattr(results, 'boxes') else None
                if detections is not None and len(detections) > 0:
                    annotated = results.plot()
                    if roi and len(roi) >= 3:
                        annotated = draw_roi_polygon(annotated, roi)
                    ui['output_placeholder'].image(annotated, channels="BGR", use_container_width=True, output_format="PNG", clamp=True)

                    # Use the original implementation that's more reliable
                    heatmap = generate_density_heatmap(frame, results)
                    ui['heatmap_placeholder'].image(heatmap, channels="BGR", use_container_width=True, output_format="PNG", clamp=True)

                    people_count = len(detections)
                    density = update_crowd_metrics(frame, results, feed_id)
                    density_level = classify_density_level(density)
                    
                    # Share metrics with the dashboard
                    if "feeds_meta" in st.session_state and feed_id in st.session_state.feeds_meta:
                        # Update feed-specific metrics
                        if "metrics" not in st.session_state.feeds_meta[feed_id]:
                            st.session_state.feeds_meta[feed_id]["metrics"] = {}
                        
                        st.session_state.feeds_meta[feed_id]["metrics"].update({
                            "people_count": people_count,
                            "density": density,
                            "density_level": density_level,
                            "last_update": time.time()
                        })
                        
                    # Update global metrics for dashboard
                    if "metrics" not in st.session_state:
                        st.session_state.metrics = {
                            "current_occupancy": 0,
                            "people_entered": 0,
                            "people_exited": 0
                        }
                    
                    # Update current occupancy with latest count
                    st.session_state.metrics["current_occupancy"] = people_count
                    
                    # Set a flag that monitoring is active for the dashboard to know
                    st.session_state["monitoring_active"] = True
                    
                    # Mark this as the currently selected feed for dashboard
                    st.session_state["current_feed_id"] = feed_id
                    
                    # Signal dashboard that new data is available from monitor page
                    st.session_state["dashboard_source"] = "monitor_page"

                    # Throttle KPI updates to reduce flicker (update every 300ms)
                    kpi_key = f"last_kpi_update_{feed_id}"
                    now = time.time()
                    
                    # Determine alert level based on density
                    alert_level = "green"
                    alert_icon = "🟢"
                    if density > 3.0:  # Critical level
                        alert_level = "red"
                        alert_icon = "🔴"
                        # Add to alerts if not already there
                        alert_msg = f"CRITICAL: Density {density:.1f} people/m² at {datetime.datetime.now().strftime('%H:%M:%S')}"
                        if len(st.session_state['alerts']) == 0 or st.session_state['alerts'][0] != alert_msg:
                            st.session_state['alerts'].insert(0, alert_msg)
                    elif density > 2.0:  # Warning level
                        alert_level = "yellow" 
                        alert_icon = "🟡"
                        # Add to alerts if not already there
                        alert_msg = f"WARNING: Density {density:.1f} people/m² at {datetime.datetime.now().strftime('%H:%M:%S')}"
                        if len(st.session_state['alerts']) == 0 or st.session_state['alerts'][0] != alert_msg:
                            st.session_state['alerts'].insert(0, alert_msg)
                    
                    # Keep only last 5 alerts
                    if len(st.session_state['alerts']) > 5:
                        st.session_state['alerts'] = st.session_state['alerts'][:5]
                    
                    # Update density history for trend chart
                    if 'density_history' not in st.session_state.crowd_stats:
                        st.session_state.crowd_stats['density_history'] = []
                        st.session_state.crowd_stats['timestamps'] = []
                    
                    st.session_state.crowd_stats['density_history'].append(density)
                    st.session_state.crowd_stats['timestamps'].append(now)
                    
                    # Keep last 5 minutes (assuming 1 update per second)
                    history_limit = 300  # 5 minutes * 60 seconds
                    if len(st.session_state.crowd_stats['density_history']) > history_limit:
                        st.session_state.crowd_stats['density_history'] = st.session_state.crowd_stats['density_history'][-history_limit:]
                        st.session_state.crowd_stats['timestamps'] = st.session_state.crowd_stats['timestamps'][-history_limit:]
                    
                    if now - st.session_state.get(kpi_key, 0) >= 0.3:
                        ui['count_metric'].metric("People Count", f"{people_count}", help="Total number of people detected")
                        ui['density_metric'].metric("Density (people/m²)", f"{density:.2f}", help="People per square meter")
                        # Unified status indicator with icon - Showing full density level
                        status_title = "Status"
                        status_text = f"{alert_icon} {density_level} ({density:.2f} people/m²)"
                        ui['status_metric'].metric(status_title, status_text, help="Current crowd density classification")
                        # Remove separate alert indicator
                        ui['alert_indicator'].empty()
                        st.session_state[kpi_key] = now

                    # KPI updates for stats metrics (using the same throttling logic)
                    if now - st.session_state.get(kpi_key, 0) >= 0.3:
                        stats = st.session_state.crowd_stats
                        ui['occupancy_metric'].metric("Occupancy Distribution", f"{stats['occupancy_distribution']}%")
                        flow_tr = "↑" if stats['flow_rate'] > 0 else "↓" if stats['flow_rate'] < 0 else "→"
                        ui['flow_metric'].metric("Flow Rate", f"{abs(stats['flow_rate'])} {flow_tr}/min")
                        trend_icon = {"increasing": "📈 Rising", "decreasing": "📉 Falling", "stable": "📊 Stable"}
                        ui['trend_metric'].metric("Density Trend", trend_icon.get(stats['density_trend'], "📊 Stable"))
                        
                        # Entry/Exit difference metric
                        entry_exit_diff = stats.get('flow_rate', 0)
                        diff_icon = "⚠️" if entry_exit_diff > 10 else ""
                        ui['entry_exit_diff'].metric("Net Flow", f"{entry_exit_diff} {diff_icon}", 
                                                  delta=entry_exit_diff,
                                                  help="Positive values mean more entries than exits")
                        
                    # Update zone metrics
                    if 'zone_analysis' in st.session_state and 'zone_metrics' in ui:
                        zone_info = st.session_state['zone_analysis']
                        for i, zone_metric in enumerate(ui['zone_metrics']):
                            if i < len(zone_info['names']):
                                zone_name = zone_info['names'][i]
                                count = zone_info['counts'][i]
                                density_val = zone_info['densities'][i]
                                
                                # Highlight the highest density zone
                                highlight = "🔍" if i == zone_info.get('highest_zone', 0) and count > 0 else ""
                                zone_metric.metric(
                                    f"{zone_name} {highlight}",
                                    f"{count} people", 
                                    f"{density_val:.2f}/m²"
                                )
                    
                    # Update trend chart with prediction
                    if len(st.session_state.crowd_stats['density_history']) > 1:
                        import pandas as pd
                        import plotly.express as px
                        import plotly.graph_objects as go
                        import numpy as np
                        
                        # Create DataFrame for plotting
                        df = pd.DataFrame({
                            'Time': [datetime.datetime.fromtimestamp(ts) for ts in st.session_state.crowd_stats['timestamps']],
                            'Density': st.session_state.crowd_stats['density_history']
                        })
                        
                        # Create plot
                        fig = px.line(df, x='Time', y='Density', title='Crowd Density Trend with Prediction')
                        
                        # Add threshold lines
                        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Warning")
                        fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="Critical")
                        
                        # Add predictive analysis if we have enough data points
                        if len(st.session_state.crowd_stats['density_history']) > 10:
                            try:
                                # Get recent trend data for prediction
                                recent_density = st.session_state.crowd_stats['density_history'][-10:]
                                x_vals = np.array(range(len(recent_density))).reshape(-1, 1)
                                y_vals = np.array(recent_density)
                                
                                # Linear regression for prediction
                                from sklearn.linear_model import LinearRegression
                                model = LinearRegression()
                                model.fit(x_vals, y_vals)
                                
                                # Generate future timestamps
                                last_time = df['Time'].iloc[-1]
                                future_times = [last_time + datetime.timedelta(minutes=i) for i in range(1, 6)]
                                
                                # Predict future values
                                future_x = np.array(range(len(recent_density), len(recent_density) + 5)).reshape(-1, 1)
                                future_y = model.predict(future_x)
                                
                                # Add prediction to plot
                                future_df = pd.DataFrame({
                                    'Time': future_times,
                                    'Density': future_y
                                })
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=future_df['Time'],
                                        y=future_df['Density'],
                                        mode='lines+markers',
                                        name='Predicted',
                                        line=dict(color='purple', dash='dash'),
                                        marker=dict(symbol='diamond', size=8)
                                    )
                                )
                                
                                # Check if prediction crosses thresholds and add alerts
                                if max(y_vals) < 0.7 and any(future_y > 0.7):
                                    # Add warning annotation
                                    warning_idx = np.where(future_y > 0.7)[0][0]
                                    warning_time = future_times[warning_idx]
                                    warning_min = (warning_time - last_time).total_seconds() // 60
                                    
                                    fig.add_annotation(
                                        x=warning_time,
                                        y=0.7,
                                        text=f"⚠️ Warning in ~{warning_min} min",
                                        showarrow=True,
                                        arrowhead=1,
                                        arrowsize=1,
                                        arrowwidth=2,
                                        arrowcolor="orange",
                                        font=dict(color="orange", size=12),
                                        bgcolor="rgba(255, 255, 255, 0.8)"
                                    )
                                    
                                    # Add to alert log if not already present
                                    if 'alerts' not in st.session_state:
                                        st.session_state['alerts'] = []
                                    
                                    alert_msg = f"⚠️ WARNING: Potential overcrowding in ~{warning_min} minutes based on trend analysis"
                                    if not any(alert_msg in a for a in st.session_state['alerts']):
                                        st.session_state['alerts'].insert(0, alert_msg)
                                        if len(st.session_state['alerts']) > 5:
                                            st.session_state['alerts'] = st.session_state['alerts'][:5]
                            except Exception as e:
                                print(f"Prediction error: {str(e)}")
                        
                    # Update alert log
                    if len(st.session_state['alerts']) > 0:
                        alert_html = "<div style='max-height: 150px; overflow-y: auto;'>"
                        for alert in st.session_state['alerts']:
                            if "CRITICAL" in alert:
                                alert_html += f"<div style='color: red; margin-bottom: 5px;'>{alert}</div>"
                            elif "WARNING" in alert:
                                alert_html += f"<div style='color: orange; margin-bottom: 5px;'>{alert}</div>"
                            else:
                                alert_html += f"<div style='margin-bottom: 5px;'>{alert}</div>"
                        alert_html += "</div>"
                        if 'alert_log' in ui:
                            ui['alert_log'].markdown(alert_html, unsafe_allow_html=True)
                else:
                    # No detections case
                    ui['output_placeholder'].image(frame, channels="BGR", use_container_width=True, output_format="PNG", clamp=True)
                    ui['heatmap_placeholder'].image(frame, channels="BGR", use_container_width=True, output_format="PNG", clamp=True)
                    
                    # Update all metrics with empty values
                    ui['count_metric'].metric("People Count", "0", help="Total number of people detected")
                    ui['density_metric'].metric("Density (people/m²)", "0.00", help="People per square meter")
                    # Unified status indicator with icon
                    status_title = "Status"
                    ui['status_metric'].metric(status_title, "🟢 Normal (0.00 people/m²)", help="Current crowd density classification")
                    # Remove separate alert indicator
                    ui['alert_indicator'].empty()
                    ui['occupancy_metric'].metric("Occupancy Distribution", "0%")
                    ui['flow_metric'].metric("Flow Rate", "0 →/min")
                    ui['trend_metric'].metric("Density Trend", "📊 Stable")
                    ui['entry_exit_diff'].metric("Net Flow", "0", help="Positive values mean more entries than exits")
                    
                    # Update zone metrics with empty values
                    if 'zone_metrics' in ui:
                        zone_names = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
                        for i, zone_metric in enumerate(ui['zone_metrics']):
                            if i < len(zone_names):
                                zone_metric.metric(
                                    f"{zone_names[i]}",
                                    "0 people", 
                                    "0.00/m²"
                                )
                        
                    # Update alert log (keep existing if available)
                    if 'alert_log' in ui and len(st.session_state.get('alerts', [])) > 0:
                        # Keep the existing alerts
                        pass
                    elif 'alert_log' in ui:
                        # Show empty alert message
                        ui['alert_log'].info("No alerts")
                    
                    # Display predictive analytics if we have the current feed_id
                    if 'predictive_section' in ui and st.session_state.get("current_feed_id"):
                        with ui['predictive_section']:
                            display_predictive_analytics(st.session_state["current_feed_id"])
                    elif 'predictive_section' in ui:
                        ui['predictive_section'].info("Select a camera feed to view predictions.")

                now = time.time()
                fps = 1.0 / max(now - ui['last_ts'], 1e-6)
                ui['last_ts'] = now
                ui['processing_status'].text(f"Processing: {fps:.1f} FPS")

        except Exception as e:
            ui['processing_status'].error(f"Error processing frame: {str(e)}")
            if st.session_state.get("debug_mode", False):
                import traceback
                ui['processing_status'].error(traceback.format_exc())
    else:
        ui['processing_status'].warning("Feed is not running. Please start the feed.")

# ---------------------------
# Monitoring: Flow
# ---------------------------

def monitor_flow_analysis(feed_id, task_settings, should_refresh=False):
    """
    Monitor flow analysis for a specific feed.
    
    Args:
        feed_id: The ID of the feed to monitor
        task_settings: The settings for the monitoring task
        should_refresh: Whether to refresh the display this iteration
    """
    flow_key = f"flow_{feed_id}"
    
    # Create the UI structure only once - persistent across refreshes
    if flow_key not in st.session_state:
        st.session_state[flow_key] = {}
        st.subheader("Flow Analysis (Enter/Exit)")

        # Check if flow analysis is enabled for this specific camera
        if "flow" not in task_settings or not task_settings["flow"].get("enabled", False):
            st.info("Flow analysis is not enabled for this camera.")
            return

        # Create UI elements once
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Input Video")
            st.session_state[flow_key]['input_placeholder'] = st.empty()
        with c2:
            st.markdown("#### Output with Flow Analysis")
            st.session_state[flow_key]['output_placeholder'] = st.empty()

        # Flow controls section
        controls_container = st.container()
        with controls_container:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                # Swap direction button
                directions = ["up_is_enter", "down_is_enter", "left_is_enter", "right_is_enter"]
                current_dir = task_settings.get("flow", {}).get("direction", "up_is_enter")
                direction_index = directions.index(current_dir) if current_dir in directions else 0
                
                dir_labels = {
                    "up_is_enter": "↑ Enter, ↓ Exit",
                    "down_is_enter": "↓ Enter, ↑ Exit",
                    "left_is_enter": "← Enter, → Exit",
                    "right_is_enter": "→ Enter, ← Exit"
                }
                
                if st.button("Swap Direction", key=f"swap_dir_{feed_id}"):
                    new_index = (direction_index + 1) % len(directions)
                    new_dir = directions[new_index]
                    
                    # Update flow settings with new direction
                    if "feeds_meta" in st.session_state and feed_id in st.session_state.feeds_meta:
                        if "task_settings" in st.session_state.feeds_meta[feed_id]:
                            if "flow" in st.session_state.feeds_meta[feed_id]["task_settings"]:
                                st.session_state.feeds_meta[feed_id]["task_settings"]["flow"]["direction"] = new_dir
                    
                    # Update tracker state
                    if "flow_tracker" in st.session_state and feed_id in st.session_state.flow_tracker:
                        st.session_state.flow_tracker[feed_id]["direction"] = new_dir
            
            with c2:
                # Reset counters button
                if st.button("Reset Counters", key=f"reset_counter_{feed_id}"):
                    if "flow_tracker" in st.session_state and feed_id in st.session_state.flow_tracker:
                        st.session_state.flow_tracker[feed_id]["entered"] = 0
                        st.session_state.flow_tracker[feed_id]["exited"] = 0
            
            # Show current direction
            with c3:
                current_dir = task_settings.get("flow", {}).get("direction", "up_is_enter")
                st.markdown(f"**Direction:** {dir_labels.get(current_dir, 'Unknown')}")
                
            # Show trails toggle
            with c4:
                show_trails = st.checkbox("Show Trails", 
                                          value=True, 
                                          key=f"show_trails_{feed_id}",
                                          help="Show movement trails for better visualization")
                st.session_state[flow_key]['show_trails'] = show_trails
            
        # Metrics display
        metrics_container = st.container()
        with metrics_container:
            m1, m2, m3 = st.columns(3)
            st.session_state[flow_key]['entered_metric'] = m1.empty()
            st.session_state[flow_key]['exited_metric'] = m2.empty()
            st.session_state[flow_key]['occupancy_metric'] = m3.empty()

        # Init flow tracker
        if "flow_tracker" not in st.session_state:
            st.session_state.flow_tracker = {}
            
        flow_settings = task_settings.get("flow", {})
        if feed_id not in st.session_state.flow_tracker:
            lines = flow_settings.get("lines", [((100, 240), (540, 240))])
            direction = flow_settings.get("direction", "up_is_enter")
            debounce_ms = flow_settings.get("debounce_ms", 500)
            st.session_state.flow_tracker[feed_id] = {
                "lines": lines,
                "direction": direction,
                "debounce_ms": debounce_ms,
                "entered": 0,
                "exited": 0,
                "occupancy": 0,
                "track_history": {}
            }
            initialize_flow_tracking()
            if "flow_tracking_feed_id" not in st.session_state:
                st.session_state.flow_tracking_feed_id = feed_id

        st.session_state.monitoring_active = True

        # Static map (create once)
        install_coords = task_settings.get("install_coords", {})
        location_label = task_settings.get("location_label", "Camera Location")
        _render_camera_map_once(feed_id, install_coords, location_label)

        st.session_state[flow_key]['processing_status'] = st.empty()
        st.session_state[flow_key]['last_ts'] = time.time()
        st.session_state[flow_key]['flow_settings'] = flow_settings
    
    # Get saved UI components
    ui = st.session_state[flow_key]
    flow_settings = ui['flow_settings']
    
    # Skip processing if no refresh needed
    if not should_refresh:
        return
    
    # feed lookup
    feed = None
    for f in st.session_state.cam_manager.list_feeds():
        if f.config.id == feed_id:
            feed = f
            break
    if feed is None:
        ui['processing_status'].error(f"Feed {feed_id} not found.")
        return

    # start feed once if needed
    if feed.status != "live":
        with st.spinner("Starting feed..."):
            st.session_state.cam_manager.start(feed_id)
        return  # Wait for next refresh to process frames

    # Model should have been initialized in the main monitor function
    if "model" not in st.session_state:
        ui['processing_status'].error("Model not initialized. Please restart monitoring.")
        return
    model = st.session_state["model"]

    # Process frame and update UI
    if feed and feed.status == "live":
        try:
            frame = feed.last_frame
            if frame is not None:
                ui['input_placeholder'].image(frame, channels="BGR", use_container_width=True, output_format="PNG", clamp=True)

                # Enable tracking for stable IDs
                if hasattr(model.model, 'track') and not hasattr(model, '_tracking_enabled'):
                    model._tracking_enabled = True
                    print("Tracking enabled for more stable detection IDs")
                
                # Use tracking if available, otherwise fallback to regular predict
                if hasattr(model.model, 'track') and getattr(model, '_tracking_enabled', False):
                    results = model.model.track(
                        source=frame,
                        imgsz=model.imgsz,
                        conf=model.conf,
                        iou=model.iou,
                        device=model.device,
                        half=model.half,
                        tracker="bytetrack.yaml",
                        persist=True,  # keep track of IDs across frames
                        classes=[0],  # person class only
                        verbose=False,
                        show=False
                    )
                    # Ultralytics track returns a list
                    results = results[0] if isinstance(results, list) else results
                else:
                    results = model.predict_single(frame)

                annotated = frame.copy()
                # Draw lines
                flow_lines = []
                if flow_settings.get("lines"):
                    for line in flow_settings["lines"]:
                        pt1, pt2 = line
                        flow_lines.append((pt1, pt2))
                        cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

                # Collect detections with stable IDs (persons only)
                detections = []
                if hasattr(results, 'boxes') and len(results.boxes) > 0:
                    boxes = results.boxes
                    ids = getattr(boxes, "id", None)
                    
                    # If tracking IDs are available, use them
                    if ids is not None:
                        ids = ids.int().cpu().numpy().tolist() if hasattr(ids, "cpu") else list(ids)
                        xyxy = boxes.xyxy.int().cpu().numpy().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.astype(int).tolist()
                        clses = boxes.cls.int().cpu().numpy().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.astype(int).tolist()
                        for det_id, (x1, y1, x2, y2), cls_id in zip(ids, xyxy, clses):
                            if cls_id == 0:  # person
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                detections.append((int(det_id), (cx, cy), (x1, y1, x2, y2)))
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Fallback to enumeration if tracking is not available
                    else:
                        for i, det in enumerate(results.boxes):
                            x1, y1, x2, y2 = map(int, det.xyxy[0])
                            cls_id = int(det.cls[0])
                            if cls_id == 0:  # person
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                detections.append((i, (cx, cy), (x1, y1, x2, y2)))
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Update flow counts
                if detections and flow_lines:
                    entered, exited = update_flow_counts(
                        feed_id, detections, flow_lines,
                        direction=flow_settings.get("direction", "up_is_enter"),
                        min_track_len=flow_settings.get("min_track_len", 5)
                    )

                    tracker = st.session_state.flow_tracker.get(feed_id, {})
                    entered_total = tracker.get("entered", 0)
                    exited_total = tracker.get("exited", 0)
                    occupancy = max(0, entered_total - exited_total)

                    # Throttle flow metrics updates to reduce flicker (update every 300ms)
                    flow_kpi_key = f"last_flow_kpi_update_{feed_id}"
                    now = time.time()
                    if now - st.session_state.get(flow_kpi_key, 0) >= 0.3:
                        ui['entered_metric'].metric("Entered", f"{entered_total}")
                        ui['exited_metric'].metric("Exited", f"{exited_total}")
                        ui['occupancy_metric'].metric("Current Occupancy", f"{occupancy}")
                        st.session_state[flow_kpi_key] = now

                    # Check if we should show trails
                    show_trails = st.session_state[flow_key].get('show_trails', True)
                    
                    if show_trails and "track_history" in tracker:
                        for track_id, history in tracker["track_history"].items():
                            pts = history["points"]
                            # Limit trail length to prevent visual clutter and improve performance
                            max_trail_length = 20  # Reduced from 60 to 20 for better performance
                            pts = pts[-max_trail_length:] if len(pts) > max_trail_length else pts
                            
                            if len(pts) > 1:
                                # Draw thinner lines for better appearance
                                for i in range(1, len(pts)):
                                    # Make the line fade as it gets older
                                    alpha = 0.3 + 0.7 * (i / len(pts))
                                    color = (0, int(255 * alpha), int(255 * alpha))
                                    cv2.line(annotated, pts[i-1], pts[i], color, 1)
                                
                                # Only draw ID for tracks with recent movement
                                cv2.putText(annotated, f"ID: {track_id}",
                                            (pts[-1][0] + 5, pts[-1][1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                ui['output_placeholder'].image(annotated, channels="BGR", use_container_width=True)

                now = time.time()
                fps = 1.0 / max(now - ui['last_ts'], 1e-6)
                ui['last_ts'] = now
                ui['processing_status'].text(f"Processing: {fps:.1f} FPS")

        except Exception as e:
            ui['processing_status'].error(f"Error processing frame: {str(e)}")
            if st.session_state.get("debug_mode", False):
                import traceback
                ui['processing_status'].error(traceback.format_exc())
    else:
        ui['processing_status'].warning("Feed is not running. Please start the feed.")

# ---------------------------
# Monitor entry
# ---------------------------

def monitor():
    """Main monitoring function for crowd monitoring solution."""
    initialize_crowd_monitoring()

    # Add CSS to prevent image flashing and set background color
    st.markdown("""
    <style>
        [data-testid="stImage"] img { transition: opacity 0s !important; }
        body { background-color: #0e1117; }
        
        /* Make the dashboard button more prominent */
        button[data-testid="baseButton-secondary"] {
            background-color: #4CAF50 !important;
            color: white !important;
            font-size: 18px !important;
            padding: 8px 16px !important;
            height: auto !important;
            border: none !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        }
        
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #45a049 !important;
            box-shadow: 0 6px 10px rgba(0,0,0,0.4) !important;
        }
        
        /* Add spacing around dashboard section */
        hr { 
            margin: 30px 0 !important;
            border-color: rgba(255,255,255,0.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Real-Time Monitoring")
    st.markdown("Live video feed with crowd density analysis and flow tracking")
    
    # Initialize model up front, before any UI rendering
    if st.session_state.get("monitoring_active", False):
        model_ready = initialize_model()
        if not model_ready:
            st.error("Failed to initialize model. Please check your model settings.")
            return

    # Refresh controls
    col1, col2 = st.columns([3, 1])
    with col2:
        # Settings panel
        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.debug_mode = st.checkbox(
                "Debug mode", 
                value=st.session_state.get("debug_mode", False),
                help="Show detailed debug information"
            )
            
            # Toggle for auto-refresh
            auto_refresh = st.checkbox(
                "Auto refresh", 
                value=st.session_state.get("monitor_tick_enabled", True),
                help="Enable continuous video feed updates"
            )
            st.session_state.monitor_tick_enabled = auto_refresh
            
            # Set refresh interval
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh interval (seconds)", 
                    min_value=1, 
                    max_value=10, 
                    value=st.session_state.get("refresh_interval", 2)
                )
                st.session_state.refresh_interval = refresh_interval
                
            # Manual refresh button
            if st.button("Refresh now", key="manual_refresh_btn"):
                st.rerun()

    # No automatic refresh, just set the flag for modules
    should_refresh = True  # Always refresh when explicit

    # Camera manager check
    if "cam_manager" not in st.session_state:
        st.error("Camera manager not initialized. Please add cameras in the Cameras tab first.")
        return

    # List available feeds
    feeds = st.session_state.cam_manager.list_feeds()
    if not feeds:
        st.warning("No cameras configured. Please add cameras in the Cameras tab first.")
        return

    # Camera selection dropdown
    feed_options, feed_dict = [], {}
    for feed in feeds:
        fid = feed.config.id
        if fid in st.session_state.feeds_meta:
            fname = st.session_state.feeds_meta[fid].get("name", f"Camera {fid}")
            loc = st.session_state.feeds_meta[fid].get("task_settings", {}).get("location_label", "")
            display = f"{fname} ({loc})" if loc else fname
            feed_options.append(display)
            feed_dict[display] = fid

    if not feed_options:
        st.warning("No configured cameras found. Please set up cameras in Tasks page first.")
        return

    with col1:
        selected = st.selectbox("Select a camera to monitor", feed_options, key="crowd_monitor_camera_select")
    selected_feed_id = feed_dict[selected]

    # Read task settings
    if selected_feed_id in st.session_state.feeds_meta:
        task_settings = st.session_state.feeds_meta[selected_feed_id].get("task_settings", {})
    else:
        st.error(f"No task settings found for feed {selected_feed_id}")
        return

    intended = task_settings.get("intended_tasks", [])
    if not intended:
        st.warning("No monitoring tasks configured for this camera. Please set up tasks in the Tasks page.")
        return

    # Create a 3-column layout for the control bar
    button_cols = st.columns([1, 1, 1])
    
    active = st.session_state.get("monitoring_active", False)
    
    # Left column: Start/Stop buttons
    with button_cols[0]:
        if not active:
            if st.button("▶️ Start Monitoring", key="start_monitoring_btn", type="primary"):
                # Starting monitoring - initialize model first
                st.session_state.monitoring_active = True
                with st.spinner("Initializing model..."):
                    model_ready = initialize_model()
                    if not model_ready:
                        st.error("Model initialization failed. Please check settings.")
                        st.session_state.monitoring_active = False
                    else:
                        st.rerun()  # Force page refresh to update UI
        else:
            if st.button("⏹ Stop Monitoring", key="stop_monitoring_btn"):
                # Stopping monitoring
                _stop_all_feeds_safely()
                st.session_state.monitoring_active = False
                st.rerun()  # Force page refresh to update UI
    
    # Middle column: Status indicator
    status = "Active" if active else "Inactive"
    auto_refresh = st.session_state.get("monitor_tick_enabled", False)
    if active and auto_refresh:
        status += " (Continuous stream)"
    elif active:
        status += " (Manual refresh)"
    
    button_cols[1].markdown(f"**Status:** {status}")
    
    # Right column: Always visible button linking to localhost:8501
    with button_cols[2]:
        # Set monitoring flag in session state before opening dashboard
        st.session_state["monitoring_active"] = True
        st.session_state["dashboard_source"] = "monitor_page"
        
        # Add button that links to localhost:8501
        st.markdown("""
            <a href="http://localhost:8501" target="_blank" style="text-decoration: none; display: inline-block; float: right;">
                <button style="background-color: #4CAF50; color: white; padding: 10px 16px; 
                border: none; border-radius: 8px; cursor: pointer; font-size: 16px; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                    Go To DaseBoard
                </button>
            </a>
        """, unsafe_allow_html=True)
        
        # Get current feed ID for URL parameter for dashboard functionality
        fid = st.session_state.get("current_feed_id")
        
        # Create URL for dashboard with feed_id - try different formats
        # Test multiple formats to handle different Streamlit deployments
        dashboard_urls = []
        
        # Format 1: Our new main directory Dashboard.py (most compatible)
        url1 = "./Dashboard.py"
        if fid:
            url1 += f"?feed_id={fid}"
        dashboard_urls.append(url1)
        
        # Format 2: Direct page name (works in many deployments)
        url2 = "./7_Dashboard"
        if fid:
            url2 += f"?feed_id={fid}"
        dashboard_urls.append(url2)
        
        # Format 3: With .py extension (works in some deployments)
        url3 = "./7_Dashboard.py"
        if fid:
            url3 += f"?feed_id={fid}"
        dashboard_urls.append(url3)
        
        # Format 4: Absolute path (works in some deployments)
        url4 = "/7_Dashboard"
        if fid:
            url4 += f"?feed_id={fid}"
        dashboard_urls.append(url4)
            
        # Format 5: Full pages path
        url5 = "pages/7_Dashboard.py"
        if fid:
            url5 += f"?feed_id={fid}"
        dashboard_urls.append(url5)
        
        # Dashboard navigation buttons have been removed
        
        # Both dashboard buttons have been removed
        
        # The dashboard navigation code has been removed


    if not st.session_state.get("monitoring_active", False):
        st.info("Click 'Start Monitoring' to begin real-time analysis.")
        return

    # Store current feed ID in session state for other components
    st.session_state["current_feed_id"] = selected_feed_id
    
    # Clear any previous UI state when switching cameras to ensure proper refresh
    if "last_monitored_feed" in st.session_state and st.session_state["last_monitored_feed"] != selected_feed_id:
        if f"density_{st.session_state['last_monitored_feed']}" in st.session_state:
            del st.session_state[f"density_{st.session_state['last_monitored_feed']}"]
        if f"flow_{st.session_state['last_monitored_feed']}" in st.session_state:
            del st.session_state[f"flow_{st.session_state['last_monitored_feed']}"]
    
    # Update last monitored feed
    st.session_state["last_monitored_feed"] = selected_feed_id
    
    # Initial setup of monitoring modules - create UI placeholders
    # Make sure we're ONLY running the tasks that were specifically configured for this camera
    if "density" in intended and task_settings.get("density", {}).get("enabled", False):
        monitor_crowd_density(selected_feed_id, task_settings, False)
    if "flow" in intended and task_settings.get("flow", {}).get("enabled", False):
        monitor_flow_analysis(selected_feed_id, task_settings, False)
    
    # Create a refresh indicator
    refresh_container = st.empty()
    
    # Continuous streaming using placeholders instead of page rerun
    if st.session_state.get("monitor_tick_enabled", False) and st.session_state.get("monitoring_active", False):
        # Get frame refresh interval in seconds (convert to smaller value for smoother video)
        frame_interval = min(0.1, st.session_state.get("refresh_interval", 2) / 10)
        max_run_time = 3600  # Safety: max 1 hour per session
        start_time = time.time()
        
        # Display streaming indicator
        with refresh_container.container():
            col1, col2 = st.columns([1, 10])
            with col1:
                st.spinner("Streaming...")
            with col2:
                st.caption(f"Live streaming mode (smooth updates)")
        
        # Main streaming loop - update placeholders without page reruns
        while (time.time() - start_time < max_run_time and 
               st.session_state.get("monitoring_active", False) and
               st.session_state.get("monitor_tick_enabled", True)):
            
            # Process frames for each active monitoring module - only for tasks configured for this camera
            if "density" in intended and task_settings.get("density", {}).get("enabled", False):
                monitor_crowd_density(selected_feed_id, task_settings, True)
            if "flow" in intended and task_settings.get("flow", {}).get("enabled", False):
                monitor_flow_analysis(selected_feed_id, task_settings, True)
            
            # Small sleep to control frame rate and prevent UI lag
            time.sleep(frame_interval)

    # Add a button at the end to navigate to dashboarddummy.py
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Dashboard Access")
        st.markdown("Click the button to view detailed analytics in the dashboard")
    with col2:
        if st.button("📊 Go to Dashboard", key="go_to_dashboarddummy"):
            try:
                # Switch to the dashboarddummy.py file
                st.switch_page("dashboarddummy.py")
            except Exception as e:
                st.error(f"Error navigating to dashboard: {str(e)}")
                # Fallback using URL navigation
                st.markdown(f"""
                    <meta http-equiv="refresh" content="0;URL='./dashboarddummy.py'">
                """, unsafe_allow_html=True)

