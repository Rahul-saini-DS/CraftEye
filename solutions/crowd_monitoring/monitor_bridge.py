"""
Bridge module for integrating the crowd monitoring module with the main application.
This allows the crowd monitoring module to use the main application's monitoring features.
"""

from __future__ import annotations
import time
import streamlit as st


def initialize_crowd_monitoring_bridge() -> None:
    """
    Initialize shared session state used by the Crowd Monitoring pages.
    Safe to call multiple times; only fills missing keys.
    """
    # Global monitoring toggles / misc
    st.session_state.setdefault("monitoring_active", False)
    st.session_state.setdefault("monitor_tick_enabled", True)  # single-page autorefresh toggle
    st.session_state.setdefault("refresh_interval", 1)         # reduced from 2s to 1s for better sync
    st.session_state.setdefault("feeds_meta", {})              # populated by Tasks / feed_config
    
    # Frame synchronization settings
    st.session_state.setdefault("frame_sync", {
        "enabled": True,              # enable frame sync by default
        "max_queue_size": 3,          # limit buffer size to prevent lag
        "frame_timestamps": {},       # track timestamps of processed frames
        "drop_frames_on_lag": True,   # drop frames if processing lags too much
        "last_sync_time": time.time() # track last sync time
    })

    # Metrics used elsewhere in the app
    st.session_state.setdefault(
        "metrics",
        {
            "objects_detected": 0,
            "inference_ms": 0.0,
            "people_entered": 0,
            "people_exited": 0,
            "current_occupancy": 0,
            "fps_ema": 0.0,
        },
    )

    # Per-page crowd stats (used by monitor.py for KPIs)
    st.session_state.setdefault(
        "crowd_stats",
        {
            "occupancy_distribution": 0,  # % of people in clustered/high-density neighborhoods
            "flow_rate": 0,               # net per minute (enter - exit)
            "density_trend": "stable",    # "increasing" | "decreasing" | "stable"
            "prev_count": 0,
            "prev_density": 0,
            "last_update_time": time.time(),
        },
    )

    # Historical series and configuration used by monitoring + dashboard
    if "crowd_monitoring" not in st.session_state:
        st.session_state.crowd_monitoring = {
            # Density thresholds (people per m²) — can be tuned per deployment
            "density_thresholds": {
                "minimal": 0.1,     # very sparse
                "low": 0.2,         # comfortable
                "moderate": 0.5,    # getting busy
                "high": 0.8,        # crowded
                "very_high": 1.2,   # dense
                "extreme": 2.0,     # very dense
                "critical": 4.0,    # hazardous
            },
            "zone_capacities": {
                "default": 100,     # optional per-zone capacity planning
            },
            # Flow analysis defaults; Tasks can overwrite these
            "flow_analysis": {
                # entry/exit lines are [x1, y1, x2, y2]
                "entry_line": [80, 240, 560, 240],
                "exit_line": [80, 400, 560, 400],
                "direction": "up_is_enter",  # default convention
            },
            # Trend analysis knobs (used by dashboards/alerts)
            "trend_analysis": {
                "window_size": 10,          # samples for moving avg
                "alert_threshold": 0.7,     # % of capacity -> yellow
                "danger_threshold": 0.9,    # % of capacity -> red
                "continuous_increase": 10,  # minutes of sustained increase
            },
            # Time-series history buffers
            "history": {
                "density": [],      # float
                "counts": [],       # int (people in frame)
                "occupancy": [],    # int (entered - exited)
                "entries": [],      # int
                "exits": [],        # int
                "timestamps": [],   # epoch seconds
            },
        }

    # Optional stores used by other components; create if missing
    st.session_state.setdefault("camera_calibrations", {})  # {feed_id: {"homography_matrix": ..., "area_m2": ...}}
    st.session_state.setdefault("heatmap_accum", {})        # incremental heatmaps per feed
    st.session_state.setdefault("flow_tracker", {})         # per-feed flow tracking runtime


def crowd_monitor() -> None:
    """
    Main entry point for the crowd monitoring solution.
    Called by the solution registry; renders the Monitoring UI.
    """
    # Ensure state is present
    initialize_crowd_monitoring_bridge()

    # Import lazily to avoid circulars
    from solutions.crowd_monitoring.monitor import monitor

    # Render main UI
    monitor()

