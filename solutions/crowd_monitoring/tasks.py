"""
Task configuration for Crowd Monitoring solution
"""
import os
import time
import uuid
import cv2
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static, st_folium
from pipeline import get_device_config

# ---------------------------
# State init
# ---------------------------

def initialize_task_state():
    """Initialize state used by the crowd task pages."""
    st.session_state.setdefault("crowd_task_meta", {})
    st.session_state.setdefault("venue_meta", {
        "max_capacity": 1000,
        "map_mode": "none",
        "map_image_path": None
    })
    st.session_state.setdefault("feeds_meta", {})

def initialize_model_settings():
    if "model_settings" not in st.session_state:
        st.session_state.model_settings = {
            "primary_model": "models/yolov8n.pt",
            "secondary_model": None,
            "confidence_threshold": 0.25,
            "imgsz": 640,
            "task_types": ["detect"],
            "device": None
        }

# ---------------------------
# Persistence
# ---------------------------

def save_feed_metadata():
    """Save task metadata along with feed config."""
    if "feeds_meta" in st.session_state:
        from capture.feed_config import save_feeds  # keep existing link
        feeds_to_save = []
        if "cam_manager" in st.session_state:
            for feed in st.session_state.cam_manager.list_feeds():
                fid = feed.config.id
                cfg = {
                    "id": fid,
                    "source": feed.config.source,
                    "type": feed.config.type,
                    "resolution": feed.config.resolution,
                    "fps_cap": feed.config.fps_cap
                }
                if fid in st.session_state.feeds_meta:
                    cfg["name"] = st.session_state.feeds_meta[fid].get("name", f"Camera {fid}")
                    cfg["model_settings"] = st.session_state.feeds_meta[fid].get("model_settings", {})
                    cfg["task_settings"] = st.session_state.feeds_meta[fid].get("task_settings", {})
                feeds_to_save.append(cfg)
        save_feeds(feeds_to_save)

# ---------------------------
# Helpers
# ---------------------------

def _ensure_feed_struct(fid: str, display_name: str):
    """Ensure feeds_meta structure exists for this feed."""
    feeds_meta = st.session_state.feeds_meta
    if fid not in feeds_meta:
        feeds_meta[fid] = {"name": display_name, "model_settings": {}, "task_settings": {}}
    if "task_settings" not in feeds_meta[fid]:
        feeds_meta[fid]["task_settings"] = {}
    ts = feeds_meta[fid]["task_settings"]

    # top-level camera fields
    ts.setdefault("location_label", "")
    ts.setdefault("install_coords", {})  # {"lat":..., "lon":...}
    ts.setdefault("intended_tasks", [])  # ["density","flow"]
    ts.setdefault("calibration", {})     # {"homography_matrix":..., "area_m2":...}

    # default subtask blocks
    ts.setdefault("density", {
        "enabled": False,  # Default to not enabled until explicitly selected
        "roi": [],
        "conf": 0.35,
        "heatmap_grid": 32,
        "agg_window_sec": 10,
        "person_class": 0
    })
    ts.setdefault("flow", {
        "enabled": False,  # Default to not enabled until explicitly selected
        "lines": [((100, 240), (540, 240))],
        "direction": "up_is_enter",
        "debounce_ms": 500,
        "min_track_len": 5
    })
    return ts

def _mirror_coords_to_dashboard(fid: str, camera_name: str, location_label: str, lat: float, lon: float):
    """Mirror coords into crowd_task_meta to keep your dashboard working."""
    st.session_state.crowd_task_meta.setdefault(fid, {
        "camera_name": camera_name,
        "area_name": location_label,
        "description": "",
        "map": {
            "type": "coords",
            "image_path": None,
            "lat": lat,
            "lon": lon
        }
    })
    # update fields if exists
    st.session_state.crowd_task_meta[fid]["camera_name"] = camera_name
    st.session_state.crowd_task_meta[fid]["area_name"] = location_label
    st.session_state.crowd_task_meta[fid]["map"]["type"] = "coords"
    st.session_state.crowd_task_meta[fid]["map"]["lat"] = lat
    st.session_state.crowd_task_meta[fid]["map"]["lon"] = lon

# ---------------------------
# Main UI
# ---------------------------

def render():
    """Render the crowd monitoring task configuration page."""
    # Init
    initialize_model_settings()
    initialize_task_state()
    device_cfg = get_device_config(prefer_gpu=True)
    st.session_state.model_settings["device"] = device_cfg.get("device")

    st.title("Crowd Monitoring Configuration")

    if "cam_manager" not in st.session_state:
        st.error("Camera manager not initialized. Please add cameras in the Cameras tab first.")
        return

    feeds = st.session_state.cam_manager.list_feeds()
    if not feeds:
        st.warning("No cameras configured. Please add cameras in the Cameras tab first.")
        return

    # Two-panel layout
    panel_1, panel_2 = st.tabs(["Camera Configuration", "Task Parameters"])

    # ---------------------------
    # Panel 1: Camera + Location + Tasks + Calibration
    # ---------------------------
    with panel_1:
        st.subheader("Camera Setup and Location")
        st.info("Pick a camera, set its location on the map, choose intended tasks, and (optionally) calibrate.")

        # Camera selector
        feed_options, feed_dict = [], {}
        for feed in feeds:
            fid = feed.config.id
            cam_name = st.session_state.feeds_meta.get(fid, {}).get("name", f"Camera {fid}")
            label = st.session_state.feeds_meta.get(fid, {}).get("task_settings", {}).get("location_label", "")
            display = f"{cam_name} ({fid})" if not label else f"{cam_name} · {label} ({fid})"
            feed_options.append(display)
            feed_dict[display] = (fid, cam_name)

        selected = st.selectbox("Select a camera", feed_options, key="crowd_monitor_camera_select_tasks")
        selected_feed_id, selected_cam_name = feed_dict[selected]

        # Ensure struct & get task settings
        task_settings = _ensure_feed_struct(selected_feed_id, selected_cam_name)

        # Location label
        task_settings["location_label"] = st.text_input(
            "Location Label",
            value=task_settings.get("location_label", ""),
            help="e.g., 'Phase 1 (Zone 1) – at Ganga Ghat'"
        )

        # Map picker (click to set coords)
        st.subheader("Install Location")
        coords = task_settings.get("install_coords", {})
        lat = coords.get("lat", 20.5937)  # India's centroid default
        lon = coords.get("lon", 78.9629)

        m = folium.Map(location=[lat, lon], zoom_start=5)
        if coords:
            folium.Marker(
                [lat, lon],
                popup=task_settings.get("location_label") or "Camera Location",
                icon=folium.Icon(color="red")
            ).add_to(m)

        st.info("Click on the map to set the installation location")
        map_data = st_folium(m, width=725, height=420, key=f"map_{selected_feed_id}")

        if map_data and map_data.get("last_clicked"):
            new_lat = float(map_data["last_clicked"]["lat"])
            new_lon = float(map_data["last_clicked"]["lng"])
            task_settings["install_coords"] = {"lat": new_lat, "lon": new_lon}
            _mirror_coords_to_dashboard(
                selected_feed_id,
                st.session_state.feeds_meta[selected_feed_id].get("name", selected_cam_name),
                task_settings.get("location_label", ""),
                new_lat, new_lon
            )
            st.success(f"Location set: ({new_lat:.6f}, {new_lon:.6f})")
            save_feed_metadata()

        # Manual fallback
        colA, colB = st.columns(2)
        with colA:
            lat_in = st.number_input("Latitude", value=float(lat), format="%.6f", key=f"lat_num_{selected_feed_id}")
        with colB:
            lon_in = st.number_input("Longitude", value=float(lon), format="%.6f", key=f"lon_num_{selected_feed_id}")
        if st.button("Set Location", key=f"btn_setloc_{selected_feed_id}"):
            task_settings["install_coords"] = {"lat": float(lat_in), "lon": float(lon_in)}
            _mirror_coords_to_dashboard(
                selected_feed_id,
                st.session_state.feeds_meta[selected_feed_id].get("name", selected_cam_name),
                task_settings.get("location_label", ""),
                float(lat_in), float(lon_in)
            )
            st.success(f"Location set: ({float(lat_in):.6f}, {float(lon_in):.6f})")
            save_feed_metadata()

        # Intended tasks
        st.subheader("Intended Tasks")
        intended_now = set(task_settings.get("intended_tasks", []))
        density_enabled = st.checkbox("Crowd density monitoring", value=("density" in intended_now), key=f"density_chk_{selected_feed_id}")
        flow_enabled    = st.checkbox("Flow analysis (enter/exit)", value=("flow" in intended_now), key=f"flow_chk_{selected_feed_id}")
        
        # Update both the list of intended tasks AND the enabled flag for each task
        new_list = []
        if density_enabled: 
            new_list.append("density")
            if "density" in task_settings:
                task_settings["density"]["enabled"] = True
                st.success(f"✅ Density monitoring ENABLED for camera {selected_feed_id}")
        else:
            if "density" in task_settings:
                task_settings["density"]["enabled"] = False
                st.info(f"ℹ️ Density monitoring DISABLED for camera {selected_feed_id}")
                
        if flow_enabled:    
            new_list.append("flow")
            if "flow" in task_settings:
                task_settings["flow"]["enabled"] = True
                st.success(f"✅ Flow analysis ENABLED for camera {selected_feed_id}")
        else:
            if "flow" in task_settings:
                task_settings["flow"]["enabled"] = False
                st.info(f"ℹ️ Flow analysis DISABLED for camera {selected_feed_id}")
                
        # keep order density -> flow
        task_settings["intended_tasks"] = [t for t in ["density", "flow"] if t in new_list]
        
        # Save changes immediately to ensure they're preserved
        save_feed_metadata()

        # Calibration summary / reuse
        st.subheader("Camera Calibration")
        if "camera_calibrations" in st.session_state and selected_feed_id in st.session_state["camera_calibrations"]:
            cal = st.session_state["camera_calibrations"][selected_feed_id]
            st.success("✓ Camera is calibrated")
            st.info(f"Approximate view area: {cal.get('area_m2', 0):.2f} m²")
            task_settings["calibration"] = {
                "homography_matrix": cal.get("homography_matrix"),
                "area_m2": cal.get("area_m2")
            }
            save_feed_metadata()
        else:
            if st.button("Begin Calibration", key=f"btn_calib_{selected_feed_id}"):
                st.session_state["calibration_mode"] = True
                st.session_state["calibration_feed"] = selected_feed_id
                st.info("Open the Monitor/Calibration screen to pick 4 points and enter distances.")

    # ---------------------------
    # Panel 2: Sub-task parameters (Density first, then Flow)
    # ---------------------------
    with panel_2:
        intended = task_settings.get("intended_tasks", [])
        if not intended:
            st.warning("No tasks selected. Please select at least one task in the Camera Configuration tab.")
            return

        # ---- Density ----
        if "density" in intended:
            st.markdown("### Crowd Density Monitoring")

            density_settings = task_settings.setdefault("density", task_settings["density"])

            # ROI toggle + editor
            st.subheader("Region of Interest (Optional)")
            roi_key_prefix = f"roi_{selected_feed_id}"
            roi_enabled = st.checkbox("Enable ROI", value=len(density_settings.get("roi", [])) > 0, key=f"{roi_key_prefix}_enabled")

            if roi_enabled:
                st.info("Define ROI by entering points (X,Y). Use frame coordinates.")
                st.session_state.setdefault(f"{roi_key_prefix}_points", list(density_settings.get("roi", [])))

                # show
                pts = st.session_state[f"{roi_key_prefix}_points"]
                if pts:
                    st.write("Current ROI points:")
                    for i, pt in enumerate(pts):
                        st.text(f"Point {i+1}: {pt}")

                colx, coly = st.columns(2)
                with colx:
                    x_coord = st.number_input("X coordinate", min_value=0, max_value=4096, value=100, key=f"{roi_key_prefix}_x")
                with coly:
                    y_coord = st.number_input("Y coordinate", min_value=0, max_value=4096, value=100, key=f"{roi_key_prefix}_y")

                if st.button("Add ROI point", key=f"{roi_key_prefix}_add"):
                    pts.append((int(x_coord), int(y_coord)))
                    density_settings["roi"] = pts
                    st.success(f"Added point ({int(x_coord)}, {int(y_coord)})")

                if st.button("Reset ROI", key=f"{roi_key_prefix}_reset"):
                    st.session_state[f"{roi_key_prefix}_points"] = []
                    density_settings["roi"] = []
                    st.success("ROI points reset")
            else:
                density_settings["roi"] = []

            colD1, colD2 = st.columns(2)
            with colD1:
                density_settings["person_class"] = 0  # COCO 'person'
                density_settings["conf"] = st.slider(
                    "Confidence threshold",
                    min_value=0.1, max_value=0.9,
                    value=float(density_settings.get("conf", 0.35)),
                    step=0.05, key=f"dens_conf_{selected_feed_id}"
                )
            with colD2:
                density_settings["heatmap_grid"] = st.select_slider(
                    "Heatmap resolution",
                    options=[16, 32, 64, 128],
                    value=int(density_settings.get("heatmap_grid", 32)),
                    key=f"dens_grid_{selected_feed_id}"
                )
                density_settings["agg_window_sec"] = st.slider(
                    "Aggregation window (seconds)",
                    min_value=1, max_value=30,
                    value=int(density_settings.get("agg_window_sec", 10)),
                    key=f"dens_agg_{selected_feed_id}"
                )
            # Only set to True if density is actually selected in the intended tasks
            density_settings["enabled"] = "density" in task_settings.get("intended_tasks", [])
            
            # Save the updated settings to the camera's metadata
            save_feed_metadata()

        # Separator when both enabled
        if "density" in intended and "flow" in intended:
            st.markdown("---")

        # ---- Flow ----
        if "flow" in intended:
            st.markdown("### Flow Analysis (Enter/Exit)")

            flow_settings = task_settings.setdefault("flow", task_settings["flow"])
            st.subheader("Virtual Line Configuration")
            st.info("Define one virtual line to count crossings.")

            # Show current line(s)
            if flow_settings.get("lines"):
                for i, line in enumerate(flow_settings["lines"]):
                    pt1, pt2 = line
                    st.text(f"Line {i+1}: {pt1} → {pt2}")

            # Editor
            line_key_prefix = f"flowline_{selected_feed_id}"
            default_line = flow_settings.get("lines", [((100, 240), (540, 240))])[0]
            (dx1, dy1), (dx2, dy2) = default_line

            colL1, colL2 = st.columns(2)
            with colL1:
                x1 = st.number_input("Start X", min_value=0, max_value=4096, value=int(dx1), key=f"{line_key_prefix}_x1")
                y1 = st.number_input("Start Y", min_value=0, max_value=4096, value=int(dy1), key=f"{line_key_prefix}_y1")
            with colL2:
                x2 = st.number_input("End X", min_value=0, max_value=4096, value=int(dx2), key=f"{line_key_prefix}_x2")
                y2 = st.number_input("End Y", min_value=0, max_value=4096, value=int(dy2), key=f"{line_key_prefix}_y2")

            if st.button("Set Virtual Line", key=f"{line_key_prefix}_set"):
                flow_settings["lines"] = [((int(x1), int(y1)), (int(x2), int(y2)))]
                st.success("Virtual line set successfully")

            colF1, colF2 = st.columns(2)
            with colF1:
                direction_opts = ["up_is_enter", "down_is_enter", "left_is_enter", "right_is_enter"]
                flow_settings["direction"] = st.selectbox(
                    "Counting direction",
                    options=direction_opts,
                    index=direction_opts.index(flow_settings.get("direction", "up_is_enter")),
                    help="Which crossing direction counts as 'enter'",
                    key=f"{line_key_prefix}_dir"
                )
            with colF2:
                flow_settings["debounce_ms"] = st.slider(
                    "Debounce time (ms)",
                    min_value=100, max_value=2000,
                    value=int(flow_settings.get("debounce_ms", 500)),
                    step=100, key=f"{line_key_prefix}_debounce"
                )
                flow_settings["min_track_len"] = st.slider(
                    "Min track length",
                    min_value=1, max_value=20,
                    value=int(flow_settings.get("min_track_len", 5)),
                    key=f"{line_key_prefix}_mintrk"
                )
            # Only set to True if flow is actually selected in the intended tasks
            flow_settings["enabled"] = "flow" in task_settings.get("intended_tasks", [])
            
            # Save changes immediately to ensure they're preserved
            save_feed_metadata()

        # Save
        if st.button("Save Task Configuration", type="primary", key=f"save_{selected_feed_id}"):
            # Bridge flow settings into crowd_monitoring state (monitor module reads this too)
            if "flow" in task_settings and task_settings["flow"].get("enabled", False):
                if 'crowd_monitoring' not in st.session_state:
                    from solutions.crowd_monitoring.monitor_bridge import initialize_crowd_monitoring_bridge
                    initialize_crowd_monitoring_bridge()
                if task_settings['flow'].get('lines'):
                    (p1, p2) = task_settings['flow']['lines'][0]
                    entry_line = [p1[0], p1[1], p2[0], p2[1]]
                    st.session_state.crowd_monitoring['flow_analysis']['entry_line'] = entry_line
                    st.session_state.crowd_monitoring['flow_analysis']['exit_line'] = entry_line.copy()
                    st.session_state.crowd_monitoring['flow_analysis']['direction'] = task_settings['flow'].get('direction', 'up_is_enter')

            save_feed_metadata()
            st.success("Task configuration saved successfully!")

# Optional: keep the calibration helper for reuse if needed elsewhere
def add_calibration_section():
    """Add camera calibration section to the tasks page (optional helper)."""
    st.markdown("### Camera Calibration")
    feeds_mgr = st.session_state.get("cam_manager")
    if not feeds_mgr:
        st.warning("No cameras available for calibration.")
        return

    feed_states = feeds_mgr.list_feeds()
    feed_ids = [f.config.id for f in feed_states]
    label_map = {f.config.id: f"{f.config.id} - {f.config.source}" for f in feed_states}

    selected_feed_id = st.selectbox(
        "Select camera for calibration",
        options=feed_ids,
        format_func=lambda x: label_map[x]
    )

    if st.button("Begin Calibration", key=f"btn_calib_only_{selected_feed_id}"):
        st.session_state["calibration_mode"] = True
        st.session_state["calibration_feed"] = selected_feed_id
        st.info("Click 4 points on the image and enter their real-world measurements.")

    # If four points are provided elsewhere, store homography + area
    if "calibration_points" in st.session_state and "world_points" in st.session_state:
        if len(st.session_state["calibration_points"]) == 4 and len(st.session_state["world_points"]) == 4:
            H, _ = cv2.findHomography(
                np.array(st.session_state["calibration_points"], dtype=np.float32),
                np.array(st.session_state["world_points"], dtype=np.float32)
            )
            if "camera_calibrations" not in st.session_state:
                st.session_state["camera_calibrations"] = {}

            # Approximate visible area (square meters) using projected image corners
            def _calc_view_area(hm):
                corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(corners.reshape(1, 4, 2), hm)
                area = 0
                for i in range(4):
                    j = (i + 1) % 4
                    area += transformed[0, i, 0] * transformed[0, j, 1]
                    area -= transformed[0, i, 1] * transformed[0, j, 0]
                return abs(area) / 2.0

            area = _calc_view_area(H)
            st.session_state["camera_calibrations"][selected_feed_id] = {
                "homography_matrix": H.tolist(),
                "area_m2": float(area)
            }
            st.success(f"Camera {selected_feed_id} calibrated successfully!")
