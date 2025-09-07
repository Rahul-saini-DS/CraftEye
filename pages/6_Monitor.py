"""
Monitoring Router

This page routes to the appropriate monitoring module based on the selected solution.
"""
import os, base64, time
import cv2
import streamlit as st
from typing import Dict, Any, Optional
import hashlib
from ui_components import render_static_header, render_brand_header, render_solution_context, verify_and_restore_state, ensure_cross_tab_data_persistence
from pipeline import get_device_config
from capture.camera_manager import CameraManager
from solutions import get_solution_by_name

# Import solution registry to ensure all solutions are registered
import solutions.registry

# ---------------- Page config ----------------
st.set_page_config(
    page_title="CraftEye - Live Monitoring",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global headers (rendered on EVERY page)
render_static_header()     # your existing top strip, if any
render_brand_header()      # new fixed banner with CraftEye + CraftifAI logo
render_solution_context()  # display selected solution if available

# Enable cross-tab data persistence to help with dashboard access in new tabs
ensure_cross_tab_data_persistence()

# ---------------- Helpers ----------------
def load_logo_data_uri():
    for p in [
        "assets/CraftEye LOGO.png",
        "assets/project-logo.png",
        "assets/CraftEye_logo.png",
        "CraftEye LOGO.png",
    ]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return None

logo_uri = load_logo_data_uri()

def _sig(d: Dict[str, Any]) -> str:
    """Make a small signature for dict comparison (used to detect line/direction changes)."""
    try:
        j = repr(sorted(d.items()))
    except Exception:
        j = repr(d)
    return hashlib.md5(j.encode("utf-8")).hexdigest()

# ---------------- State & initialization ----------------
def initialize_model_settings():
    if 'model_settings' not in st.session_state:
        st.session_state.model_settings = {
            'primary_model': 'models/yolov8n.pt',
            'secondary_model': None,
            'confidence_threshold': 0.5,
            'imgsz': 640,
            'task_types': ['detect'],   # footfall can still activate per-feed via feed.task
            'device': None
        }

def initialize_camera_manager():
    if 'cam_manager' not in st.session_state:
        st.session_state.cam_manager = CameraManager()
    if 'feeds_meta' not in st.session_state:
        st.session_state.feeds_meta = {}
    st.session_state.setdefault("feed_slots", {})   # stable placeholders
    st.session_state.setdefault("file_caps", {})    # feed_id -> (cap, tmp_path)

def initialize_monitoring():
    """Initialize basic monitoring state for solutions."""
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'ema_fps' not in st.session_state:
        st.session_state.ema_fps = 0.0
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'objects_detected': 0,
            'inference_ms': 0.0,
            'people_entered': 0,
            'people_exited': 0,
            'current_occupancy': 0,
            'fps_ema': 0.0
        }
    if 'last_csv_log_ts' not in st.session_state:
        st.session_state.last_csv_log_ts = {}
    if 'feed_slots' not in st.session_state:
        st.session_state.feed_slots = {}

    # Cache device config to avoid printing "Using standard CPU" repeatedly
    if "device_config_cache" not in st.session_state:
        st.session_state["device_config_cache"] = get_device_config(prefer_gpu=True)

# ---------------- Styles for UI ----------------
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
:root{ --bg:#0b1220; --bg2:#0c1426; --text:#f7faff; --muted:#dbe6ff;
--panel:rgba(255,255,255,.05); --stroke:rgba(255,255,255,.14);
--ring:#2d8bff; --ring2:#3aa6ff; --glow:rgba(58,166,255,.38);
--ok:#2bd38a; --warn:#ffb020; --danger:#ff5a7a; }

.block-container { padding-top: 5px; }
.stApp{background:linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%); color:var(--text); font-size:16.5px;}
.container{max-width:1180px; margin:0 auto; padding:24px 16px;}
.brand-hero{display:flex; align-items:center; gap:18px; padding:24px 26px; margin:12px 0 22px; border-radius:22px; position:relative; overflow:hidden;
  background:radial-gradient(80% 140% at 0% 0%, rgba(61,151,255,.16) 0%, rgba(61,151,255,0) 60%), linear-gradient(180deg, rgba(255,255,255,.09), rgba(255,255,255,.04));
  border:1px solid rgba(255,255,255,.22); box-shadow:0 18px 55px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.06);}
.logo-badge{width:78px; height:78px; border-radius:18px; display:grid; place-items:center;
  background:radial-gradient(120px 120px at 30% 30%, rgba(90,170,255,.35), rgba(90,170,255,.06) 60%), linear-gradient(135deg, rgba(90,170,255,.32), rgba(10,18,32,0));
  border:1px solid rgba(140,190,255,.45); box-shadow:0 14px 32px rgba(45,139,255,.30), inset 0 0 0 5px rgba(255,255,255,.04);}
.logo-badge img{height:58px; width:auto;}
.brand-title{font-weight:900; font-size:1.9rem}
.brand-sub{color:#cfe4ff; font-size:1.06rem}
.steps-row{display:flex; gap:12px; align-items:center; flex-wrap:wrap; background:linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.04));
  border:1px solid var(--stroke); padding:12px 14px; border-radius:18px; margin:0 0 22px; box-shadow:inset 0 0 0 1px rgba(255,255,255,.05);}
.step-item{display:flex; flex-direction:column; align-items:center; gap:6px; min-width:110px}
.chip{display:flex; align-items:center; gap:8px; height:36px; padding:0 12px; border-radius:999px; color:var(--text);
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10);}
.chip .num{width:26px; height:26px; display:grid; place-items:center; border-radius:999px; background:rgba(255,255,255,.16); color:#eef3ff; font-size:.9rem; font-weight:900;}
.step-item .label{font-weight:800; font-size:.98rem; color:#eaf2ff}
.step-item.active .chip{background:linear-gradient(45deg,var(--ring),var(--ring2)); border:none}
.step-item.done .chip{background:linear-gradient(45deg,#1fbf74,#33e68b); border:none}
.step-item.done .label{color:#c8ffe3}
h1.title{font-size:2.6rem; font-weight:900; margin:6px 0 6px;}
p.lead{color:#e8f0ff; font-size:1.1rem; margin:0 0 18px}
.panel{background:var(--panel); border:1px solid var(--stroke); border-radius:18px; padding:16px 16px; margin-bottom:16px;}
.panel:hover{box-shadow:0 10px 26px var(--glow); border-color:var(--ring); transition:box-shadow .2s ease, border-color .2s ease;}
.panel h3{margin:0 0 10px; font-size:1.2rem; color:#dfe9ff}
</style>
""", unsafe_allow_html=True)

# ---------------- Main UI ----------------
st.markdown('<div class="container">', unsafe_allow_html=True)

# Steps chips
area_done   = bool(st.session_state.get("selected_area"))
domain_done = bool(st.session_state.get("selected_domain"))
solution_done   = bool(st.session_state.get("selected_solution"))

# Get the selected solution name
selected_solution_name = st.session_state.get("selected_solution")

# Check if we can infer a solution from our current context
# This is a fallback mechanism for when the session state is lost
if not selected_solution_name:
    # If we have specific indicators of the Crowd Monitoring solution
    if "model_settings" in st.session_state or "feeds_meta" in st.session_state:
        selected_solution_name = "Pilgrim Crowd Monitoring"
        st.session_state["selected_solution"] = selected_solution_name
        
        # Also make sure we have area and domain set
        if "selected_area" not in st.session_state:
            st.session_state["selected_area"] = "Pilgrim Site"
        if "selected_domain" not in st.session_state:
            st.session_state["selected_domain"] = "Crowd Management"

cams_done   = bool(st.session_state.get("cam_manager")) and len(getattr(st.session_state.get("cam_manager"), "list_feeds")()) > 0 \
             if st.session_state.get("cam_manager") else False
tasks_done  = bool(st.session_state.get("model_settings"))
solution_done = bool(st.session_state.get("selected_solution"))  # Update after our potential fix

# Render progress steps
st.markdown(f"""
<div class="steps-row">
  <div class="step-item {'done' if area_done else ''}"><div class="chip"><div class="num">1</div>🗺️</div><div class="label">Area</div></div>
  <div class="step-item {'done' if domain_done else ''}"><div class="chip"><div class="num">2</div>🏭</div><div class="label">Domain</div></div>
  <div class="step-item {'done' if solution_done else ''}"><div class="chip"><div class="num">3</div>🛍️</div><div class="label">Solution</div></div>
  <div class="step-item {'done' if cams_done else ''}"><div class="chip"><div class="num">4</div>📷</div><div class="label">Cameras</div></div>
  <div class="step-item {'done' if tasks_done else ''}"><div class="chip"><div class="num">5</div>🧠</div><div class="label">Tasks</div></div>
  <div class="step-item active"><div class="chip"><div class="num">6</div>📊</div><div class="label">Monitor</div></div>
  <div class="step-item"><div class="chip"><div class="num">7</div>📈</div><div class="label">Dashboard</div></div>
</div>
""", unsafe_allow_html=True)

# Initialize required state
initialize_model_settings()
initialize_camera_manager()
initialize_monitoring()

# Use the solution registry to route to the appropriate monitoring module
if not selected_solution_name:
    st.warning("Please select a solution from the Solution page first.")
    st.error("No solution selected. Please go to the Solution page and select a solution.")
    
    # Add a button to fix the state and load Pilgrim Crowd Monitoring
    if st.button("Load Pilgrim Crowd Monitoring Solution"):
        st.session_state["selected_solution"] = "Pilgrim Crowd Monitoring"
        st.session_state["selected_area"] = "Pilgrim Site"
        st.session_state["selected_domain"] = "Crowd Management"
        st.rerun()
else:
    # Display solution context
    st.info(f"Monitoring for solution: {selected_solution_name}")
    
    # Get solution from registry
    solution = get_solution_by_name(selected_solution_name)
    
    if not solution:
        st.error(f"Solution '{selected_solution_name}' not found in registry.")
        
        # Try special case for Pilgrim Crowd Monitoring
        if "pilgrim" in selected_solution_name.lower() or "crowd" in selected_solution_name.lower():
            try:
                # Try direct import of crowd monitor
                from solutions.crowd_monitoring.monitor_bridge import crowd_monitor
                st.success("Successfully loaded Crowd Monitoring module directly")
                crowd_monitor()
                st.stop()
            except Exception as e:
                st.error(f"Error loading crowd monitoring directly: {e}")
    elif not solution.get("monitor_module"):
        st.warning(f"Solution '{selected_solution_name}' does not have a monitoring module.")
        
        # Try special case for Pilgrim Crowd Monitoring
        if "pilgrim" in selected_solution_name.lower() or "crowd" in selected_solution_name.lower():
            try:
                # Try direct import of crowd monitor
                from solutions.crowd_monitoring.monitor_bridge import crowd_monitor
                st.success("Successfully loaded Crowd Monitoring module directly")
                crowd_monitor()
                st.stop()
            except Exception as e:
                st.error(f"Error loading crowd monitoring directly: {e}")
                
        st.info("Using legacy monitoring interface.")
        
        # For legacy implementation
        st.markdown('<h1 class="title">Live Monitoring</h1>', unsafe_allow_html=True)
        st.markdown('<p class="lead">Real-time computer vision analysis</p>', unsafe_allow_html=True)
        st.error("Legacy monitoring implementation has been simplified. Please implement a monitoring module for this solution.")
    else:
        try:
            # Call the solution's monitoring module
            solution["monitor_module"]()
            # Skip the rest of the original monitoring code
            st.stop()
        except Exception as e:
            st.error(f"Error in monitor module: {e}")
            
            # Fallback for Pilgrim Crowd Monitoring
            if "pilgrim" in selected_solution_name.lower() or "crowd" in selected_solution_name.lower():
                try:
                    # Try direct import as fallback
                    from solutions.crowd_monitoring.monitor_bridge import crowd_monitor
                    st.success("Using fallback crowd monitoring module")
                    crowd_monitor()
                    st.stop()
                except Exception as e2:
                    st.error(f"Fallback also failed: {e2}")

    # Navigation buttons
    st.markdown('<div style="margin-top:16px; display:flex; gap:10px;">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Back to Configuration", use_container_width=True):
            try:
                st.switch_page("/5_Tasks.py")
            except Exception:
                try:
                    st.page_link("/5_Tasks.py", label="Back to Tasks", icon="↩️")
                except Exception:
                    st.markdown('<a href="/5_Tasks" target="_self">Back to Tasks</a>', unsafe_allow_html=True)
    
    with col2:
        if st.button("Go To Dashboard 📈", use_container_width=True):
            try:
                st.switch_page("/7_Dashboard.py")
            except Exception:
                try:
                    st.page_link("/7_Dashboard.py", label="Go to Dashboard", icon="📈")
                except Exception:
                    st.markdown('<a href="/7_Dashboard" target="_self">Go to Dashboard</a>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # /container
