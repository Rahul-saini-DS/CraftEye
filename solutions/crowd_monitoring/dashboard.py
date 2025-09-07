"""
Dashboard module for Crowd Monitoring solution
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# Optional viz backends
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except Exception:
    PYDECK_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# Optional CSV helpers (safe if not present)
try:
    from solutions.crowd_monitoring.data_persistence import read_recent_rows, get_csv_path
    from solutions.crowd_monitoring.zone_persistence import read_zone_data
    CSV_HELPERS = True
except Exception:
    CSV_HELPERS = False


def _now() -> float:
    return time.time()


def _init_trend_state() -> None:
    """Idempotent init for trend buffers."""
    # Initialize all required session state variables
    st.session_state.setdefault(
        "trend_analysis",
        {
            "entry_history": [],
            "exit_history": [],
            "occupancy_history": [],
            "timestamps": [],
            "last_update": _now(),
            "prediction_window": 10,  # minutes
            "alert_active": False,
            "alert_message": "",
            "alert_level": "none",  # none | yellow | red
            "trend_direction": "stable",  # increasing | decreasing | stable
        },
    )
    st.session_state.setdefault("venue_meta", {"max_capacity": 1000, "map_mode": "none", "map_image_path": None})
    st.session_state.setdefault("metrics", {"current_occupancy": 0, "people_entered": 0, "people_exited": 0})
    st.session_state.setdefault("crowd_stats", {"density_history": [], "timestamps": []})
    st.session_state.setdefault("alerts", [])
    st.session_state.setdefault("zone_analysis", {
        "names": ["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
        "counts": [0, 0, 0, 0],
        "densities": [0.0, 0.0, 0.0, 0.0],
        "highest_zone": 0
    })
    st.session_state.setdefault("crowd_task_meta", {})
    st.session_state.setdefault("feeds_meta", {})


def _update_trend_buffers() -> None:
    """Push current snapshot into trend buffers every ~5s and trim to last hour."""
    cur = _now()
    ta = st.session_state.trend_analysis

    # For live dashboard mode, reduce the update interval when monitoring is active
    update_interval = 2 if st.session_state.get("monitoring_active", False) else 5
    
    # Skip update if we updated recently (unless forced by new tab or explicit request)
    force_update = st.session_state.get("dashboard_source") == "monitor_page"
    if not force_update and cur - ta.get("last_update", 0) < update_interval:
        return

    # Try to load data from CSV if we're in a new tab or not monitoring
    if CSV_HELPERS and not st.session_state.get("monitoring_active", False):
        try:
            feed_id = st.session_state.get("current_feed_id")
            if feed_id:
                # Attempt to load historical data from CSV for this feed
                csv_path = get_csv_path(feed_id)
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty and len(df) > 1:
                            # Get the last row for current values
                            latest = df.iloc[-1]
                            # Update basic metrics from CSV data
                            if 'count' in df.columns:
                                st.session_state.metrics['current_occupancy'] = int(latest.get('count', 0))
                            # If we have entered/exited data
                            if 'entered' in df.columns and 'exited' in df.columns:
                                st.session_state.metrics['people_entered'] = int(latest.get('entered', 0)) 
                                st.session_state.metrics['people_exited'] = int(latest.get('exited', 0))
                            
                            # Only refresh trend data if we have meaningful differences
                            current_entries = ta["entry_history"][-1] if ta["entry_history"] else 0
                            if len(ta["entry_history"]) == 0 or abs(int(latest.get('entered', 0)) - current_entries) > 0:
                                # Update trend data from CSV - safely handle missing columns
                                if 'entered' in df.columns:
                                    ta["entry_history"] = df['entered'].astype(int).tolist()[-60:]  # Last 60 entries
                                if 'exited' in df.columns:
                                    ta["exit_history"] = df['exited'].astype(int).tolist()[-60:]
                                if 'count' in df.columns:
                                    ta["occupancy_history"] = df['count'].astype(int).tolist()[-60:]
                                # Ensure timestamps length matches longest series
                                max_len = max(len(ta["entry_history"]), len(ta["exit_history"]), len(ta["occupancy_history"]))
                                ta["timestamps"] = list(range(max_len))
                    except (IOError, pd.errors.EmptyDataError) as csv_err:
                        # Handle file locked or empty file gracefully
                        pass
        except Exception as e:
            # Silent failure - just log to session state for debugging
            st.session_state["_csv_error"] = str(e)

    # Update trend data with current metrics
    m = st.session_state.metrics
    ta["entry_history"].append(int(m.get("people_entered", 0)))
    ta["exit_history"].append(int(m.get("people_exited", 0)))
    ta["occupancy_history"].append(int(m.get("current_occupancy", 0)))
    ta["timestamps"].append(cur)

    # Keep 1 hour of data
    cutoff = cur - 3600
    keep = [i for i, t in enumerate(ta["timestamps"]) if t >= cutoff]
    for key in ("entry_history", "exit_history", "occupancy_history", "timestamps"):
        seq = ta[key]
        if keep and len(seq) != len(keep):
            ta[key] = [seq[i] for i in keep]

    # Reset the dashboard source flag so we don't force updates forever
    if st.session_state.get("dashboard_source") == "monitor_page":
        st.session_state["dashboard_source"] = None
        
    ta["last_update"] = cur


def _moving_avg(series: List[float], window: int) -> np.ndarray:
    if not series:
        return np.array([])
    s = pd.Series(series)
    win = max(1, int(window))
    return s.rolling(window=win, min_periods=1).mean().to_numpy()


def _plot_trends(entry_hist: List[int], exit_hist: List[int], window_pts: int) -> None:
    if not entry_hist or not exit_hist:
        st.info("Start monitoring to collect trend data.")
        return

    entry_ma = _moving_avg(entry_hist, window_pts)
    exit_ma = _moving_avg(exit_hist, window_pts)

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=entry_hist, mode="markers", name="Entries", opacity=0.3))
        fig.add_trace(go.Scatter(y=exit_hist, mode="markers", name="Exits", opacity=0.3))
        fig.add_trace(go.Scatter(y=entry_ma, mode="lines", name="Entry MA"))
        fig.add_trace(go.Scatter(y=exit_ma, mode="lines", name="Exit MA"))
        fig.update_layout(
            title="Entry vs Exit Trends (with Moving Average)",
            xaxis_title="Time (samples)",
            yaxis_title="Count",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = pd.DataFrame(
            {
                "Entries": entry_hist,
                "Exits": exit_hist,
                "Entry MA": entry_ma,
                "Exit MA": exit_ma,
            }
        )
        st.line_chart(df)


def _generate_sample_data() -> Dict[str, Any]:
    """Generate sample crowd data for demonstration when no cameras are available."""
    import numpy as np
    
    # Generate sample time series
    now = int(time.time())
    timestamps = [now - (60 - i) * 60 for i in range(60)]  # Last 60 minutes
    
    # Create sample occupancy data with some patterns
    base_occupancy = 50  # Base occupancy level
    time_trend = np.linspace(0, 30, 60)  # Increasing trend
    daily_pattern = 20 * np.sin(np.linspace(0, 2 * np.pi, 60))  # Sinusoidal pattern
    random_noise = np.random.normal(0, 5, 60)  # Random variations
    
    # Combine patterns and convert to integers
    occupancy = [max(0, int(base_occupancy + t + d + r)) for t, d, r in 
                zip(time_trend, daily_pattern, random_noise)]
    
    # Create entries and exits data
    entries = [0]
    exits = [0]
    
    for i in range(1, 60):
        # Calculate differences in occupancy
        diff = occupancy[i] - occupancy[i-1]
        if diff > 0:
            # More people entered
            entries.append(entries[-1] + diff)
            exits.append(exits[-1])
        else:
            # More people exited
            entries.append(entries[-1])
            exits.append(exits[-1] - diff)
    
    return {
        "timestamps": timestamps,
        "occupancy": occupancy,
        "entries": entries,
        "exits": exits,
        "max_capacity": 200,
        "current_occupancy": occupancy[-1],
        "people_entered": entries[-1],
        "people_exited": exits[-1]
    }

def _plot_occupancy(occ_hist: List[int], max_capacity: int) -> None:
    if not occ_hist or len(occ_hist) < 2:
        # If we have no data but we're in the dashboard, generate sample data
        if st.session_state.get("dashboard_mode") == "demo":
            sample_data = _generate_sample_data()
            _plot_occupancy(sample_data["occupancy"], sample_data["max_capacity"])
            return
        
        st.info("Start monitoring to collect occupancy data.")
        return

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=occ_hist, mode="lines+markers", name="Occupancy"))

        # Threshold lines (70% & 90%)
        n = len(occ_hist)
        for thr, color, label in ((0.9, "red", "90% Capacity"), (0.7, "orange", "70% Capacity")):
            y = max_capacity * thr
            # Add shape through update_layout for better compatibility
            fig.update_layout(
                shapes=[
                    dict(type="line", x0=0, y0=y, x1=n - 1, y1=y, 
                         line=dict(color=color, width=2, dash="dash"))
                ]
            )
            fig.add_annotation(x=n - 1, y=y, text=label, ax=40, ay=0, showarrow=True)

        fig.update_layout(title="Occupancy Over Time", xaxis_title="Time (samples)", yaxis_title="People", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame({"Occupancy": occ_hist}))


def _predictive_alerts_ui() -> None:
    st.markdown("## Predictive & Early Warning Alerts")

    max_capacity = int(st.session_state.venue_meta.get("max_capacity", 1000))
    cur_occ = int(st.session_state.metrics.get("current_occupancy", 0))
    occ_pct = (cur_occ / max_capacity * 100.0) if max_capacity > 0 else 0.0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Capacity Status")
        if occ_pct >= 90:
            st.error("🚨 CRITICAL ALERT: At or near maximum capacity!")
            st.session_state.trend_analysis["alert_level"] = "red"
            st.session_state.trend_analysis["alert_message"] = f"{occ_pct:.1f}% of capacity"
        elif occ_pct >= 70:
            st.warning("⚠️ WARNING: High occupancy levels detected")
            st.session_state.trend_analysis["alert_level"] = "yellow"
            st.session_state.trend_analysis["alert_message"] = f"{occ_pct:.1f}% of capacity"
        else:
            st.success("✅ Normal occupancy levels")
            st.session_state.trend_analysis["alert_level"] = "none"
            st.session_state.trend_analysis["alert_message"] = ""

    with col2:
        st.metric("Current Occupancy", f"{cur_occ} people", f"{occ_pct:.1f}% of capacity")
        st.progress(min(1.0, occ_pct / 100.0))

        with st.expander("Capacity Settings"):
            new_cap = st.number_input("Maximum Venue Capacity", min_value=10, max_value=10000, value=max_capacity)
            if st.button("Update Capacity", key="update_capacity_btn"):
                st.session_state.venue_meta["max_capacity"] = int(new_cap)
                st.success(f"Capacity updated to {int(new_cap)}")

    # Predictive Trend (simple MA comparison over a ~10-minute window)
    st.markdown("### Predictive Trend Analysis")
    ta = st.session_state.trend_analysis

    # Assume ~5s cadence -> window points ~ prediction_window * 60 / 5
    # Clamp to the length of history
    if len(ta["entry_history"]) > 10 and len(ta["exit_history"]) > 10:
        pts = max(1, min(len(ta["entry_history"]), int(ta["prediction_window"] * 12)))  # 12 pts/min @5s
        entry_ma = _moving_avg(ta["entry_history"], pts)
        exit_ma = _moving_avg(ta["exit_history"], pts)

        if len(entry_ma) >= 2 and len(exit_ma) >= 2:
            # Continuous exceed check on the window
            win_e = entry_ma[-pts:]
            win_x = exit_ma[-pts:]
            if len(win_e) == len(win_x) and np.all(win_e > win_x):
                ta["alert_active"] = True
                ta["trend_direction"] = "increasing"
                st.warning("🔍 PREDICTIVE ALERT: Entries persistently exceed exits in the last window")
            else:
                ta["alert_active"] = False
                if entry_ma[-1] > exit_ma[-1]:
                    ta["trend_direction"] = "increasing"
                    st.info("📈 Trend: crowd is building")
                elif entry_ma[-1] < exit_ma[-1]:
                    ta["trend_direction"] = "decreasing"
                    st.success("📉 Trend: crowd is dispersing")
                else:
                    ta["trend_direction"] = "stable"
                    st.success("📊 Trend: stable")

            st.markdown(f"**Current Trend Direction:** {ta['trend_direction'].title()}")
            _plot_trends(ta["entry_history"], ta["exit_history"], pts)
        else:
            st.info("Collecting data for trend analysis…")
    else:
        st.info("Start monitoring to collect trend data. At least 10 data points required for analysis.")

    st.markdown("### Occupancy Over Time")
    _plot_occupancy(st.session_state.trend_analysis["occupancy_history"], max_capacity)

    # Optional: Historical CSV (if persisted)
    with st.expander("Historical data (CSV)"):
        if CSV_HELPERS and "cam_manager" in st.session_state and st.session_state.cam_manager:
            feed_states = st.session_state.cam_manager.list_feeds()
            feed_map = {f.config.id: st.session_state.feeds_meta.get(f.config.id, {}).get("name", f"Camera {f.config.id}") for f in feed_states}
            if feed_map:
                fid = st.selectbox("Choose a feed", list(feed_map.keys()), format_func=lambda k: feed_map[k])
                rows = read_recent_rows(fid, max_rows=2000)
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df.tail(1000), use_container_width=True, height=260)
                else:
                    csv_path = get_csv_path(fid) if CSV_HELPERS else "(unknown)"
                    st.info(f"No CSV rows yet for this feed. Expected path: `{csv_path}`")
            else:
                st.info("No feeds available yet.")
        else:
            st.caption("CSV helpers unavailable or no feeds configured.")


def _interactive_map_ui() -> None:
    st.markdown('<h2 id="camera-location">Interactive Camera Map</h2>', unsafe_allow_html=True)

    # Gather camera metadata
    cameras: List[Dict[str, Any]] = []
    if "cam_manager" in st.session_state and st.session_state.cam_manager:
        for feed in st.session_state.cam_manager.list_feeds():
            fid = feed.config.id
            meta = st.session_state.crowd_task_meta.get(fid, {})
            map_info = meta.get("map", {})
            has_coords = map_info.get("type") == "coords" and bool(map_info.get("lat")) and bool(map_info.get("lon"))
            cameras.append(
                {
                    "id": fid,
                    "name": meta.get("camera_name", f"Camera {fid}"),
                    "area": meta.get("area_name", "Unknown Area"),
                    "status": feed.status,
                    "count": st.session_state.feeds_meta.get(fid, {}).get("people_count", 0),
                    "lat": float(map_info.get("lat", 0)) if has_coords else None,
                    "lon": float(map_info.get("lon", 0)) if has_coords else None,
                    "has_coords": has_coords,
                }
            )

    map_mode = st.session_state.venue_meta.get("map_mode", "none")
    any_coords = any(c.get("has_coords") for c in cameras)

    # PyDeck (tile map) path
    if map_mode == "tile" and PYDECK_AVAILABLE and any_coords:
        cams = [c for c in cameras if c["has_coords"]]
        avg_lat = sum(c["lat"] for c in cams) / len(cams)
        avg_lon = sum(c["lon"] for c in cams) / len(cams)

        # Prepare deck layer data
        data = []
        for c in cams:
            color = [0, 200, 0] if c["status"] == "live" else [200, 0, 0]
            radius = 50 + int(c["count"]) * 5
            data.append(
                {
                    "name": c["name"],
                    "area": c["area"],
                    "status": c["status"],
                    "count": int(c["count"]),
                    "coordinates": [c["lon"], c["lat"]],
                    "color": color,
                    "radius": radius,
                }
            )

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=14, pitch=0),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=data,
                        get_position="coordinates",
                        get_color="color",
                        get_radius="radius",
                        pickable=True,
                        auto_highlight=True,
                    )
                ],
                tooltip={
                    "html": "<b>{name}</b><br/>Area: {area}<br/>Status: {status}<br/>Count: {count}",
                    "style": {"backgroundColor": "steelblue", "color": "white"},
                },
            )
        )
        return

    # Folium fallback
    if any_coords and FOLIUM_AVAILABLE:
        cams = [c for c in cameras if c["has_coords"]]
        avg_lat = sum(c["lat"] for c in cams) / len(cams)
        avg_lon = sum(c["lon"] for c in cams) / len(cams)

        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)
        for c in cams:
            color = "green" if c["status"] == "live" else "red"
            popup = f"<b>{c['name']}</b><br>Area: {c['area']}<br>Status: {c['status']}<br>Count: {c['count']}"
            folium.Marker(location=[c["lat"], c["lon"]], popup=popup, icon=folium.Icon(color=color)).add_to(m)
        st_folium(m, width=800, height=500)
        return

    # Uploaded image mode
    if map_mode == "upload" and st.session_state.venue_meta.get("map_image_path"):
        path = st.session_state.venue_meta.get("map_image_path")
        if os.path.exists(path):
            st.image(path, caption="Venue Map", use_container_width=True)
            st.markdown("### Camera Locations")
            for c in cameras:
                status_color = "green" if c["status"] == "live" else "red"
                status_dot = "🟢" if c["status"] == "live" else "🔴"
                st.markdown(
                    f"<div style='margin:5px; padding:10px; border:1px solid #ddd; border-radius:5px;'>"
                    f"{status_dot} <b>{c['name']}</b> ({c['area']})<br/>"
                    f"Status: <span style='color:{status_color}'>{c['status']}</span><br/>"
                    f"People count: {int(c['count'])}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.error(f"Map image not found: {path}")
        return

    # If no maps possible, show a clean table
    st.info("No map available. Configure map settings in the Tasks page or provide coordinates.")
    if cameras:
        df = pd.DataFrame(
            [
                {
                    "Camera": c["name"],
                    "Area": c["area"],
                    "Status": c["status"],
                    "People Count": int(c["count"]),
                }
                for c in cameras
            ]
        )
        st.table(df)
    else:
        st.warning("👋 Welcome to the Dashboard! No cameras are currently configured.")
        
        # Clearer instructions and guidance
        st.markdown("""
        ### Getting Started
        To see real analytics data, you'll need to:
        1. Go to the **Cameras** tab using the button above
        2. Add one or more camera feeds (webcam, video file, or RTSP stream)
        3. Return to this dashboard to see real-time analytics
        
        Until then, here's some sample data to preview the dashboard capabilities:
        """)
        
        # Show sample data if no cameras are configured
        if PLOTLY_AVAILABLE:
            st.subheader("Sample Analytics Preview (Demo Data)")
            
            # Create sample occupancy data
            sample_times = list(range(60))
            sample_occupancy = [50 + int(30 * np.sin(i/10) + i/2) for i in sample_times]
            
            # Sample plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=sample_occupancy, mode="lines+markers", name="Sample Occupancy"))
            fig.update_layout(title="Sample Occupancy Trend (Demo Data)", 
                              xaxis_title="Time (samples)", 
                              yaxis_title="People", 
                              height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 The above chart shows sample demo data to preview dashboard capabilities.")


def render() -> None:
    """Render the crowd monitoring analytics dashboard."""
    # Create a row with title and navigation button
    col1, col2 = st.columns([0.8, 0.2])
    
    with col1:
        st.title("Crowd Analytics Dashboard")
        st.markdown("Predictive alerts, occupancy trends, and advanced analytics.")
        
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        if st.button("➕ Configure Cameras", use_container_width=True):
            try:
                st.switch_page("pages/4_Cameras.py")
            except Exception:
                st.page_link("pages/4_Cameras.py", label="Go to Cameras Page", icon="📷")
    
    # Import UI helpers
    from ui_components import verify_and_restore_state, ensure_cross_tab_data_persistence
    
    # Ensure state consistency
    verify_and_restore_state()
    
    # Ensure we have the proper trend state initialization
    _init_trend_state()
    
    # Initialize camera manager if needed
    if "cam_manager" not in st.session_state:
        try:
            # Import camera manager
            from capture.camera_manager import CameraManager, FeedConfig
            from capture.feed_config import load_feeds
            
            # Load feeds from configuration file
            feed_data = load_feeds()
            
            # Create a camera manager
            cam_manager = CameraManager()
            
            # Add feeds from configuration
            added_count = 0
            for feed in feed_data:
                try:
                    config = FeedConfig(
                        id=feed["id"],
                        source=feed["source"],
                        type=feed["type"],
                        resolution=tuple(feed["resolution"]) if "resolution" in feed else (640, 480),
                        fps_cap=feed.get("fps_cap", 15),
                        task=feed.get("task", None)
                    )
                    cam_manager.add_feed(config)
                    added_count += 1
                except Exception as e:
                    st.warning(f"Skipped invalid feed config: {str(e)}")
            
            if added_count > 0:
                st.session_state["cam_manager"] = cam_manager
                st.success(f"Camera manager initialized with {added_count} feeds")
                st.session_state["dashboard_mode"] = "live"
            else:
                st.info("No camera configurations found. Add cameras in the Cameras tab.")
                # Set to demo mode since we have no cameras
                st.session_state["dashboard_mode"] = "demo"
                
                # Generate sample data for demo mode
                sample_data = _generate_sample_data()
                st.session_state.metrics = {
                    "current_occupancy": sample_data["current_occupancy"],
                    "people_entered": sample_data["people_entered"],
                    "people_exited": sample_data["people_exited"]
                }
                st.session_state.trend_analysis = {
                    "entry_history": sample_data["entries"],
                    "exit_history": sample_data["exits"],
                    "occupancy_history": sample_data["occupancy"],
                    "timestamps": sample_data["timestamps"],
                    "last_update": time.time(),
                    "prediction_window": 10,
                    "alert_active": False,
                    "alert_message": "",
                    "alert_level": "none",
                    "trend_direction": "stable",
                }
                st.session_state.venue_meta = {"max_capacity": sample_data["max_capacity"]}
                
                st.info("💡 Demo mode: Displaying sample data since no cameras are configured.")
                
                # Add a link to the cameras tab
                if st.button("Go to Cameras Tab to Add Cameras"):
                    st.switch_page("pages/4_Cameras.py")
                
        except Exception as e:
            st.error(f"Failed to initialize camera manager: {str(e)}")
            # Set to demo mode due to error
            st.session_state["dashboard_mode"] = "demo"
            
            # Generate sample data for demo mode
            try:
                sample_data = _generate_sample_data()
                st.session_state.metrics = {
                    "current_occupancy": sample_data["current_occupancy"],
                    "people_entered": sample_data["people_entered"],
                    "people_exited": sample_data["people_exited"]
                }
                st.session_state.trend_analysis = {
                    "entry_history": sample_data["entries"],
                    "exit_history": sample_data["exits"],
                    "occupancy_history": sample_data["occupancy"],
                    "timestamps": sample_data["timestamps"],
                    "last_update": time.time(),
                    "prediction_window": 10,
                    "alert_active": False,
                    "alert_message": "",
                    "alert_level": "none",
                    "trend_direction": "stable",
                }
                st.session_state.venue_meta = {"max_capacity": sample_data["max_capacity"]}
                
                st.info("💡 Demo mode: Displaying sample data due to camera initialization error.")
                
                # Add a link to the cameras tab
                if st.button("Go to Cameras Tab to Add Cameras"):
                    st.switch_page("pages/4_Cameras.py")
            except Exception as demo_error:
                st.error(f"Failed to initialize demo mode: {str(demo_error)}")
    
    # Enable cross-tab data persistence
    ensure_cross_tab_data_persistence()
    
    # Check URL parameters for data sharing between tabs
    # On first render, read feed_id from query params - handle both old and new formats
    feed_id = None
    
    # Check modern query_params
    if "feed_id" in st.query_params:
        feed_id = st.query_params["feed_id"]
    
    # Fallback to experimental_get_query_params for older Streamlit versions
    try:
        if not feed_id and hasattr(st, 'experimental_get_query_params'):
            exp_params = st.experimental_get_query_params()
            if 'feed_id' in exp_params and exp_params['feed_id']:
                feed_id = exp_params['feed_id'][0]
    except Exception:
        pass
    
    # Apply the feed ID if found
    if feed_id and feed_id != st.session_state.get("current_feed_id"):
        st.session_state["current_feed_id"] = feed_id
        st.info(f"Showing data for feed: {feed_id}")
    
    # Clear the rendering flag on every refresh
    st.session_state["_dashboard_rendering"] = False
    
    # Load data from CSV files if we're in a separate tab or monitoring is inactive
    # This helps with cross-tab data sharing
    use_csv = CSV_HELPERS  # Always try to use CSV first
    
    # Print debugging info to console
    print(f"Dashboard: CSV_HELPERS={CSV_HELPERS}, monitoring_active={st.session_state.get('monitoring_active', False)}, feed_id={feed_id}")
    
    if use_csv:
        try:
            # Attempt to load the latest data from CSV files if available
            feed_id = st.session_state.get("current_feed_id")
            if feed_id:
                csv_path = get_csv_path(feed_id)
                if os.path.exists(csv_path):
                    try:
                        # Get file modification time to check freshness
                        csv_mod_time = os.path.getmtime(csv_path)
                        last_read_time = st.session_state.get("last_csv_read_time", 0)
                        
                        # Only reload if file has been modified or we haven't read it yet
                        if csv_mod_time > last_read_time:
                            latest_data = read_recent_rows(feed_id, max_rows=60)  # Get more rows for trend data
                            if latest_data and len(latest_data) > 0:
                                # Use the most recent data for metrics
                                latest = latest_data[-1]
                                # Update basic metrics from CSV data
                                if 'count' in latest:
                                    st.session_state.metrics['current_occupancy'] = int(latest.get('count', 0))
                                # Update density and density level - direct assignment without conditionals
                                # CSV always has density column as seen in the data
                                st.session_state.metrics['density'] = float(latest.get('density', 0.0))
                                st.session_state.metrics['density_level'] = latest.get('density_level', 'Normal')
                                
                                # Log success but don't show toast to users
                                if 'density' in latest:
                                    print(f"Dashboard: Updated density to {float(latest.get('density', 0.0)):.2f} from CSV")
                                # If we have entered/exited data
                                if 'entered' in latest and 'exited' in latest:
                                    st.session_state.metrics['people_entered'] = int(latest.get('entered', 0))
                                    st.session_state.metrics['people_exited'] = int(latest.get('exited', 0))
                                
                                # Rebuild crowd_stats from CSV data for trend charts
                                # Make sure crowd_stats exists
                                if 'crowd_stats' not in st.session_state:
                                    st.session_state.crowd_stats = {'density_history': [], 'timestamps': []}
                                
                                # Extract density and timestamp data for trends
                                density_history = [float(row.get('density', 0.0)) for row in latest_data if 'density' in row]
                                
                                # Try to parse timestamps or use relative timestamps
                                try:
                                    timestamps = []
                                    for row in latest_data:
                                        if 'timestamp' in row:
                                            try:
                                                ts = datetime.fromisoformat(row['timestamp']).timestamp()
                                            except ValueError:
                                                ts = float(row.get('ts', time.time()))
                                            timestamps.append(ts)
                                        else:
                                            timestamps.append(float(row.get('ts', time.time())))
                                except Exception:
                                    # If timestamp parsing fails, use relative times
                                    timestamps = [time.time() - (i * 5) for i in range(len(latest_data)-1, -1, -1)]
                                
                                    # Update crowd_stats for trend visualization
                                st.session_state.crowd_stats['density_history'] = density_history
                                st.session_state.crowd_stats['timestamps'] = timestamps
                                
                                # IMPORTANT: This ensures we always have data for the trend chart
                                if len(density_history) > 0:
                                    # Take the most recent density value to display in the KPI
                                    st.session_state.metrics['density'] = density_history[-1]
                                    # Ensure we have at least one value for the chart
                                    if len(density_history) < 2:
                                        # Add a duplicate point to enable trend visualization
                                        st.session_state.crowd_stats['density_history'] = density_history * 2
                                        st.session_state.crowd_stats['timestamps'] = timestamps + [time.time()]
                                
                                # Remember when we last read the CSV
                                st.session_state["last_csv_read_time"] = time.time()
                                
                                # Update dashboard status
                                st.session_state["dashboard_data_source"] = "csv_file"
                    except (IOError, pd.errors.EmptyDataError):
                        # Handle file locked or empty file
                        pass
                else:
                    # File doesn't exist yet - show waiting status
                    st.session_state["dashboard_data_source"] = "waiting"
        except Exception as e:
            # Show a small toast instead of an error banner
            st.toast(f"Could not load feed data: {str(e)}", icon="⚠️")
    
    # Setup soft refresh using st.rerun() instead of st.autorefresh
    monitoring_active = st.session_state.get("monitoring_active", False)
    
    # Add link back to monitor page and controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        if st.button("⏮️ Back to Monitor", key="back_to_monitor_btn"):
            try:
                st.switch_page("/6_Monitor.py")
            except Exception:
                try:
                    st.page_link("/6_Monitor.py", label="Back to Monitor", icon="📺")
                except Exception:
                    st.markdown('<a href="/6_Monitor" target="_self">Back to Monitor</a>', unsafe_allow_html=True)
    
    # Implement a soft refresh timer using st.rerun()
    if "dash_last_refresh" not in st.session_state:
        st.session_state.dash_last_refresh = 0.0
    
    REFRESH_SEC = 2
    now = time.time()
    
    # If monitoring is active, show a status indicator
    if monitoring_active:
        st.success("🔄 Live monitoring active - Dashboard will update automatically")
    
    # Trigger refresh when time has elapsed
    if now - st.session_state.dash_last_refresh >= REFRESH_SEC:
        st.session_state.dash_last_refresh = now
        # Only rerun if we aren't in the middle of rendering
        if not st.session_state.get("_dashboard_rendering", False):
            st.session_state["_dashboard_rendering"] = True
            st.rerun()
    
    # If opened from monitor page or in a new tab, always update data from files
    dashboard_source = st.session_state.get("dashboard_source")
    if dashboard_source == "monitor_page" or not monitoring_active:
        # Try to load latest data to ensure it's visible in the new tab
        _update_trend_buffers()

    _init_trend_state()
    _update_trend_buffers()
    
    # Display status badge
    feed_id = st.session_state.get("current_feed_id")
    data_source = st.session_state.get("dashboard_data_source", "unknown")
    
    # More descriptive status indicators
    if monitoring_active:
        st.sidebar.success("🟢 Live monitoring active (Direct Feed)", icon="✅")
    elif feed_id and CSV_HELPERS and os.path.exists(get_csv_path(feed_id)):
        last_update = st.session_state.get("last_csv_read_time", 0)
        time_diff = time.time() - last_update
        if time_diff < 10:
            st.sidebar.info(f"🔄 Polling CSV data (Updated {time_diff:.1f}s ago)", icon="📊")
        else:
            st.sidebar.info("🔄 Polling CSV data from Monitor tab", icon="📊")
            
        # Add information about cross-tab synchronization
        st.sidebar.caption("""
        ℹ️ The dashboard is synchronizing with the Monitor tab through CSV files.
        Keep both tabs open for real-time updates.
        """)
    else:
        st.sidebar.warning("⏸️ Waiting for data from Monitor tab...", icon="⏳")
        
        # Add hint if no data is available
        if not feed_id:
            st.sidebar.caption("""
            💡 Tip: Start monitoring in the Monitor tab first, then open this dashboard
            to see real-time analytics.
            """)
            
    # Show current feed info if available
    if feed_id:
        feed_name = st.session_state.feeds_meta.get(feed_id, {}).get("name", f"Feed {feed_id}")
        st.sidebar.caption(f"📷 Current feed: **{feed_name}**")
    
    # Display key metrics from monitor page
    st.subheader("Real-time Crowd Metrics")
    
    # Create a row of metrics
    cols = st.columns(4)
    
    # Get data from the current feed
    current_feed_id = st.session_state.get("current_feed_id")
    feed_metrics = {}
    if current_feed_id and "feeds_meta" in st.session_state and current_feed_id in st.session_state.feeds_meta:
        feed_metrics = st.session_state.feeds_meta[current_feed_id].get("metrics", {})
    
    # Get location name
    location = "Unknown"
    if current_feed_id and "feeds_meta" in st.session_state and current_feed_id in st.session_state.feeds_meta:
        location = st.session_state.feeds_meta[current_feed_id].get("task_settings", {}).get("location_label", "Unknown")
    
    # Display metrics
    with cols[0]:
        count = feed_metrics.get("people_count", st.session_state.metrics.get("current_occupancy", 0))
        st.metric("People Count", f"{count}", help="Total number of people detected")
    
    with cols[1]:
        # IMPORTANT: Always prioritize session state metrics for density 
        # as that's updated directly from CSV
        density = st.session_state.metrics.get("density", 0)
        level = st.session_state.metrics.get("density_level", "Normal")
        
        # Only fallback to feed_metrics if density is still 0
        if density == 0 and "density" in feed_metrics:
            density = feed_metrics.get("density", 0)
            
        if level == "Normal" and "density_level" in feed_metrics:
            level = feed_metrics.get("density_level", "Normal")
        
        color = "green"
        if level == "High":
            color = "orange"
        elif level == "Critical":
            color = "red"
        st.metric("Density (people/m²)", f"{density:.2f}", help=f"Crowd density level: {level}")
        st.markdown(f"<p style='color:{color};font-weight:bold;margin-top:-15px;'>{level} Level</p>", unsafe_allow_html=True)
    
    with cols[2]:
        # Calculate flow rate from crowd_stats
        flow_rate = 0
        if "crowd_stats" in st.session_state:
            flow_rate = st.session_state.crowd_stats.get("flow_rate", 0)
        flow_direction = "stable"
        if flow_rate > 5:
            flow_direction = "increasing"
        elif flow_rate < -5:
            flow_direction = "decreasing"
        st.metric("Flow Rate", f"{flow_rate}/min", 
                 delta=flow_direction,
                 delta_color="normal",
                 help="Net flow of people per minute (positive: entering, negative: leaving)")
    
    with cols[3]:
        # Get distribution/clustering from crowd_stats
        distribution = 0
        if "crowd_stats" in st.session_state:
            distribution = st.session_state.crowd_stats.get("occupancy_distribution", 0)
        st.metric("Clustering", f"{distribution}%", 
                 help="Percentage of people in high-density clusters")
    
    # Add location and time
    last_update = feed_metrics.get("last_update", time.time())
    update_time = datetime.fromtimestamp(last_update).strftime("%H:%M:%S")
    st.caption(f"Location: {location} | Last updated: {update_time}")
    
    # Add a separator
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Predictive Alerts", 
        "Interactive Map", 
        "Density Trend (Last 5 Minutes)", 
        "Zone Analysis", 
        "Alert Log",
        "Simulation Controls"
    ])
    
    with tab1:
        _predictive_alerts_ui()
    with tab2:
        _interactive_map_ui()
    with tab3:
        # Density trend for the last 5 minutes
        st.markdown('<h2 id="density-trend-last-5-minutes">Density Trend (Last 5 Minutes)</h2>', unsafe_allow_html=True)
        
        # Show active camera information
        if current_feed_id and "feeds_meta" in st.session_state and current_feed_id in st.session_state.feeds_meta:
            feed_name = st.session_state.feeds_meta[current_feed_id].get("name", f"Camera {current_feed_id}")
            st.info(f"Showing data from camera: **{feed_name}** at location: **{location}**")
        
        # Initialize crowd_stats if needed
        if 'crowd_stats' not in st.session_state:
            st.session_state.crowd_stats = {'density_history': [], 'timestamps': []}
            
        # Check if we have data from the monitoring page
        if 'density_history' in st.session_state.crowd_stats:
            # Use the density data from monitoring page
            density_history = st.session_state.crowd_stats.get('density_history', [])
            timestamps = st.session_state.crowd_stats.get('timestamps', [])
            
            # Debug info to help diagnose issues
            st.info(f"Found {len(density_history)} density history points and {len(timestamps)} timestamps")
            
            if len(density_history) > 10:
                # Plot the density trend from monitoring page
                # Calculate minutes ago for each timestamp
                now = time.time()
                time_labels = []
                recent_density = []
                
                # Get data from the last 5 minutes
                cutoff = now - 300  # 5 minutes in seconds
                for i, ts in enumerate(timestamps):
                    if ts >= cutoff:
                        time_labels.append(int((now - ts) / 60))  # Minutes ago
                        if i < len(density_history):
                            recent_density.append(density_history[i])
                
                # Reverse to show newest data on the right
                time_labels.reverse()
                recent_density.reverse()
                
                if PLOTLY_AVAILABLE and recent_density:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=recent_density, mode="lines+markers", name="Density"))
                    fig.update_layout(
                        title="Crowd Density (Last 5 Minutes)",
                        xaxis_title="Time (samples ago)",
                        yaxis_title="Density (people/m²)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add detailed metrics below the chart
                    st.markdown("### Detailed Metrics")
                    
                    # Create columns for metrics
                    metric_cols = st.columns(3)
                    
                    # Calculate min, max, avg from recent_density
                    with metric_cols[0]:
                        avg_density = sum(recent_density) / len(recent_density) if recent_density else 0
                        st.metric("Average Density", f"{avg_density:.2f} people/m²")
                    
                    with metric_cols[1]:
                        max_density = max(recent_density) if recent_density else 0
                        min_density = min(recent_density) if recent_density else 0
                        st.metric("Peak Density", f"{max_density:.2f} people/m²")
                        
                    with metric_cols[2]:
                        # Calculate trend
                        if len(recent_density) >= 5:
                            first_avg = sum(recent_density[:3]) / 3
                            last_avg = sum(recent_density[-3:]) / 3
                            trend = last_avg - first_avg
                            trend_label = "Increasing" if trend > 0.2 else "Decreasing" if trend < -0.2 else "Stable"
                            trend_delta = f"{trend_label} ({trend:.2f})"
                            st.metric("Density Trend", f"{trend_label}", delta=f"{trend:.2f}")
                        else:
                            st.metric("Density Trend", "Insufficient Data")
                elif recent_density:
                    st.line_chart(pd.DataFrame({"Density": recent_density}))
            else:
                st.warning(f"Not enough density data available yet ({len(density_history)} points). Continue monitoring to collect more data or check CSV data loading.")
        
        # Also show trend analysis data if available
        ta = st.session_state.trend_analysis
        if len(ta["entry_history"]) > 10 and len(ta["occupancy_history"]) > 10:
            # Calculate the number of points for 5 minutes (assuming ~5s per point)
            pts = min(len(ta["entry_history"]), 60)  # 60 points = 5 minutes with 5-second cadence
            
            # Get the last 5 minutes of data
            entry_hist = ta["entry_history"][-pts:]
            exit_hist = ta["exit_history"][-pts:]
            occ_hist = ta["occupancy_history"][-pts:]
            
            # Plot 5-minute trend data
            entry_ma = _moving_avg(entry_hist, 6)  # 30-second moving average
            exit_ma = _moving_avg(exit_hist, 6)
            
            if PLOTLY_AVAILABLE:
                # Create a time index (reversed from most recent)
                time_idx = list(range(-pts+1, 1))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_idx, y=entry_hist, mode="markers", name="Entries", opacity=0.3))
                fig.add_trace(go.Scatter(x=time_idx, y=exit_hist, mode="markers", name="Exits", opacity=0.3))
                fig.add_trace(go.Scatter(x=time_idx, y=entry_ma, mode="lines", name="Entry MA"))
                fig.add_trace(go.Scatter(x=time_idx, y=exit_ma, mode="lines", name="Exit MA"))
                fig.update_layout(
                    title="Last 5 Minutes Trend",
                    xaxis_title="Time (samples ago)",
                    yaxis_title="Count",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot occupancy for the last 5 minutes
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=time_idx, y=occ_hist, mode="lines+markers", name="Occupancy"))
                fig2.update_layout(
                    title="Occupancy (Last 5 Minutes)",
                    xaxis_title="Time (samples ago)",
                    yaxis_title="People",
                    height=350
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # Fallback to Streamlit charts
                df = pd.DataFrame({
                    "Entries": entry_hist,
                    "Exits": exit_hist,
                    "Entry MA": entry_ma,
                    "Exit MA": exit_ma,
                })
                st.line_chart(df)
                st.line_chart(pd.DataFrame({"Occupancy": occ_hist}))
        else:
            st.info("Start monitoring to collect trend data. At least 10 data points required for analysis.")
            
    with tab4:
        # Zone analysis
        st.markdown('<h2 id="zone-analysis">Zone Analysis</h2>', unsafe_allow_html=True)
        
        # Initialize zone_analysis if needed
        if 'zone_analysis' not in st.session_state:
            st.session_state['zone_analysis'] = {
                'names': ["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
                'counts': [0, 0, 0, 0],
                'densities': [0.0, 0.0, 0.0, 0.0],
                'highest_zone': 0
            }
        
        # In a separate tab, try to load zone data from persistent storage
        if not st.session_state.get("monitoring_active", False) or "feed_id" in st.query_params:
            feed_id = st.session_state.get("current_feed_id")
            if feed_id:
                try:
                    # Try to load zone data from JSON file
                    zone_data = read_zone_data(feed_id)
                    if zone_data and 'counts' in zone_data:
                        # Update session state with loaded zone data
                        st.session_state['zone_analysis'] = zone_data
                except Exception as e:
                    # Silent error handling
                    pass
            
        # Get zone info from session state
        zone_info = st.session_state['zone_analysis']
        
        st.subheader("Zone Distribution")
        
        # Create zone chart
        if PLOTLY_AVAILABLE and 'counts' in zone_info and 'names' in zone_info:
            # Create bar chart for zone counts
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=zone_info['names'],
                y=zone_info['counts'],
                text=zone_info['counts'],
                textposition='auto',
            ))
            fig.update_layout(
                title="People Count by Zone",
                xaxis_title="Zone",
                yaxis_title="People Count",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Create heatmap-like visualization for densities
            if 'densities' in zone_info:
                # Create color-coded cards for each zone
                cols = st.columns(len(zone_info['names']))
                for i, col in enumerate(cols):
                    if i < len(zone_info['names']) and i < len(zone_info['densities']):
                        name = zone_info['names'][i]
                        count = zone_info['counts'][i]
                        density = zone_info['densities'][i]
                        
                        # Determine color based on density
                        color = "green"
                        if density > 3.0:
                            color = "red"
                        elif density > 1.5:
                            color = "orange"
                            
                        col.markdown(f"""
                        <div style="padding: 10px; background-color: {color}20; border: 1px solid {color}; border-radius: 5px; text-align: center;">
                            <h3>{name}</h3>
                            <p>{count} people</p>
                            <p>{density:.2f} people/m²</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # Fallback to simple display
            if 'counts' in zone_info and 'names' in zone_info:
                for i, name in enumerate(zone_info['names']):
                    if i < len(zone_info['counts']):
                        st.metric(name, zone_info['counts'][i], f"{zone_info.get('densities', [0])[i]:.2f} people/m²")
        
        # Also show camera zone data
        st.subheader("Cameras by Zone")
        cameras = []
        if "cam_manager" in st.session_state and st.session_state.cam_manager:
            for feed in st.session_state.cam_manager.list_feeds():
                fid = feed.config.id
                meta = st.session_state.crowd_task_meta.get(fid, {})
                count = st.session_state.feeds_meta.get(fid, {}).get("people_count", 0)
                cameras.append({
                    "id": fid,
                    "name": meta.get("camera_name", f"Camera {fid}"),
                    "area": meta.get("area_name", "Unknown Area"),
                    "count": count
                })
        
        if cameras:
            # Create zone comparison chart
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[c["name"] for c in cameras],
                    y=[c["count"] for c in cameras],
                    text=[c["count"] for c in cameras],
                    textposition='auto',
                ))
                fig.update_layout(
                    title="Current People Count by Zone",
                    xaxis_title="Zone",
                    yaxis_title="People Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to Streamlit chart
                df = pd.DataFrame({
                    "Zone": [c["name"] for c in cameras],
                    "Count": [c["count"] for c in cameras]
                })
                st.bar_chart(df.set_index("Zone"))
                
            # Display zone details
            st.subheader("Zone Details")
            for cam in cameras:
                with st.expander(f"{cam['name']} ({cam['area']})"):
                    st.metric("Current Count", cam["count"])
        else:
            st.info("No camera zones configured. Add cameras in the Cameras tab.")
    
    with tab5:
        # Alert log
        st.markdown('<h2 id="alert-log">Alert Log</h2>', unsafe_allow_html=True)
        ta = st.session_state.trend_analysis
        
        # Get alert data
        alert_level = ta.get("alert_level", "none")
        alert_message = ta.get("alert_message", "")
        alert_active = ta.get("alert_active", False)
        trend_direction = ta.get("trend_direction", "stable")
        
        # Display current alert status
        if alert_level == "red":
            st.error("🚨 CRITICAL ALERT: " + (alert_message or "High occupancy detected!"))
        elif alert_level == "yellow":
            st.warning("⚠️ WARNING: " + (alert_message or "Elevated occupancy levels"))
        else:
            st.success("✅ No active alerts")
            
        # Display predictive alerts
        st.subheader("Predictive Alerts")
        if alert_active:
            st.warning("🔍 PREDICTIVE ALERT: Entries persistently exceed exits - crowd building")
        else:
            st.info(f"Trend direction: {trend_direction.title()}")
            
        # Display real alerts from monitoring
        st.subheader("Recent Alerts")
        # Initialize alerts if needed
        if 'alerts' not in st.session_state:
            st.session_state['alerts'] = []
            
        if st.session_state['alerts']:
            alert_data = []
            for alert in st.session_state['alerts']:
                level = "red" if "CRITICAL" in alert else "yellow" if "WARNING" in alert else "green"
                alert_data.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "level": level,
                    "message": alert
                })
            
            # Display actual alerts table
            alert_df = pd.DataFrame(alert_data)
            st.dataframe(alert_df, use_container_width=True)
        else:
            st.info("No alerts recorded yet. Alerts will appear here during monitoring.")

    with tab6:
        _simulation_controls_ui()

def _simulation_controls_ui() -> None:
    """Render simulation controls for testing and demos."""
    st.markdown('<h2 id="simulation-controls">Simulation Controls</h2>', unsafe_allow_html=True)
    st.markdown("These controls allow you to test system responses to various scenarios. **Note: These are for demonstration only and do not affect live camera feeds.**")
    
    # Create three columns for simulation buttons
    sim_cols = st.columns(3)
    with sim_cols[0]:
        # Simulation button for incident
        if st.button("🚨 Simulate Incident", key="sim_incident_btn"):
            # Initialize session state if needed
            if 'alerts' not in st.session_state:
                st.session_state['alerts'] = []
            
            # Initialize incident simulation data
            if 'incident_simulation' not in st.session_state:
                st.session_state['incident_simulation'] = {
                    'active': True,
                    'start_time': time.time(),
                    'location': 'Top Right',  # Arbitrary zone for the incident
                    'type': 'Overcrowding',
                    'severity': 'Critical'
                }
                # Add to alert log
                alert_msg = f"🚨 CRITICAL INCIDENT: {st.session_state['incident_simulation']['type']} at {datetime.now().strftime('%H:%M:%S')}"
                st.session_state['alerts'].insert(0, alert_msg)
                if len(st.session_state['alerts']) > 5:
                    st.session_state['alerts'] = st.session_state['alerts'][:5]
            else:
                # Toggle incident simulation
                st.session_state['incident_simulation']['active'] = not st.session_state['incident_simulation'].get('active', False)
                if st.session_state['incident_simulation']['active']:
                    st.session_state['incident_simulation']['start_time'] = time.time()
                    # Add to alert log
                    alert_msg = f"🚨 CRITICAL INCIDENT: {st.session_state['incident_simulation']['type']} at {datetime.now().strftime('%H:%M:%S')}"
                    st.session_state['alerts'].insert(0, alert_msg)
                    if len(st.session_state['alerts']) > 5:
                        st.session_state['alerts'] = st.session_state['alerts'][:5]
    
    with sim_cols[1]:
        # Emergency dispatch simulation
        if st.button("🚑 Dispatch Emergency", key="dispatch_emergency_btn"):
            # Initialize session state if needed
            if 'alerts' not in st.session_state:
                st.session_state['alerts'] = []
                
            # Add to alert log
            alert_msg = f"🚑 EMERGENCY RESPONSE DISPATCHED at {datetime.now().strftime('%H:%M:%S')}"
            st.session_state['alerts'].insert(0, alert_msg)
            if len(st.session_state['alerts']) > 5:
                st.session_state['alerts'] = st.session_state['alerts'][:5]
            
            # Initialize emergency response simulation
            if 'emergency_response' not in st.session_state:
                st.session_state['emergency_response'] = {
                    'active': True,
                    'start_time': time.time(),
                    'eta_seconds': 30,  # 30 seconds ETA
                    'type': 'Ambulance'
                }
            else:
                st.session_state['emergency_response']['active'] = True
                st.session_state['emergency_response']['start_time'] = time.time()
            st.success("Emergency response dispatched")
    
    with sim_cols[2]:
        # Reset simulation
        if st.button("↺ Reset Simulation", key="reset_sim_btn"):
            # Initialize session state if needed
            if 'alerts' not in st.session_state:
                st.session_state['alerts'] = []
                
            if 'incident_simulation' in st.session_state:
                st.session_state['incident_simulation']['active'] = False
            if 'emergency_response' in st.session_state:
                st.session_state['emergency_response']['active'] = False
            # Add to alert log
            alert_msg = f"✓ SITUATION RESOLVED at {datetime.now().strftime('%H:%M:%S')}"
            st.session_state['alerts'].insert(0, alert_msg)
            if len(st.session_state['alerts']) > 5:
                st.session_state['alerts'] = st.session_state['alerts'][:5]
            
    # Simulation status
    st.subheader("Current Simulation Status")
    
    # Display current simulation state
    sim_active = st.session_state.get('incident_simulation', {}).get('active', False)
    response_active = st.session_state.get('emergency_response', {}).get('active', False)
    
    if sim_active and response_active:
        st.warning("⚠️ Incident in progress with emergency response")
    elif sim_active:
        st.error("🚨 Incident in progress - no emergency response")
    elif response_active:
        st.info("🚑 Emergency response active - no ongoing incident")
    else:
        st.success("✅ No active simulations")
    
    # Explanation of simulation capabilities
    with st.expander("About Simulation Controls"):
        st.markdown("""
        ### Simulation Features
        
        These controls allow you to test how the system responds to various scenarios:
        
        * **Simulate Incident**: Creates a mock overcrowding incident in one zone
        * **Dispatch Emergency**: Simulates emergency response team deployment
        * **Reset Simulation**: Clears all active simulations
        
        **Note:** These are for demonstration and training purposes only. They don't affect actual camera feeds or analytics.
        """)



