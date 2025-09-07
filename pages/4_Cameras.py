# pages/4_Cameras.py
import os, base64, uuid
import streamlit as st
from ui_components import render_static_header, render_brand_header, render_solution_context

# KEEP: your pipeline/camera logic (unchanged)
from capture.camera_manager import CameraManager, FeedConfig
from pipeline import discover_models, get_device_config

# ---------- Page config ----------
st.set_page_config(
    page_title="CraftEye - Camera Configuration",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global headers (rendered on EVERY page)
render_static_header()     # your existing top strip, if any
render_brand_header()      # fixed banner with CraftEye + CraftifAI logo
render_solution_context()  # display selected solution if available

# ---------- Helpers ----------
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

def initialize_camera_manager():
    if 'cam_manager' not in st.session_state:
        st.session_state.cam_manager = CameraManager()
    if 'feeds_meta' not in st.session_state:
        st.session_state.feeds_meta = {}

    from capture.feed_config import load_feeds
    saved_feeds = load_feeds()
    for feed_config in saved_feeds:
        try:
            feed = FeedConfig(
                source=feed_config['source'],
                id=feed_config['id'],
                type=feed_config.get('type', 'file'),
                resolution=feed_config.get('resolution', [640, 480]),
                fps_cap=feed_config.get('fps_cap', 30)
            )
            if feed.id not in [f.config.id for f in st.session_state.cam_manager.list_feeds()]:
                st.session_state.cam_manager.add_feed(feed)
            if feed.id not in st.session_state.feeds_meta:
                st.session_state.feeds_meta[feed.id] = {
                    'name': feed_config.get('name', f'Camera {feed.id}'),
                    'model_settings': feed_config.get('model_settings', {}),
                    'task_settings': feed_config.get('task_settings', {})
                }
        except Exception as e:
            st.warning(f"Failed to restore feed configuration: {str(e)}")

# ---------- Styles (UI only) ----------
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
:root{
  --bg:#0b1220; --bg2:#0c1426;
  --text:#f7faff; --muted:#dbe6ff;
  --panel:rgba(255,255,255,.05); --stroke:rgba(255,255,255,.14);
  --ring:#2d8bff; --ring2:#3aa6ff; --glow:rgba(58,166,255,.38);
  --ok:#2bd38a; --warn:#ffb020; --danger:#ff5a7a;
}
/* minimal padding under the compact banner */
.block-container { padding-top: 5px; }

.stApp{background:linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%); color:var(--text); font-size:16.5px;}
.container{max-width:1180px; margin:0 auto; padding:24px 16px;}

/* Steps row — keep your original look */
.steps-row{
  display:flex; gap:12px; align-items:center; flex-wrap:wrap;
  background:linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.04));
  border:1px solid var(--stroke); padding:12px 14px; border-radius:18px; margin:0 0 22px;
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.05);
}
.step-item{display:flex; flex-direction:column; align-items:center; gap:6px; min-width:110px}
.chip{
  display:flex; align-items:center; gap:8px; height:36px; padding:0 12px;
  border-radius:999px; color:var(--text);
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10);
}
.chip .num{
  width:26px; height:26px; display:grid; place-items:center; border-radius:999px;
  background:rgba(255,255,255,.16); color:#eef3ff; font-size:.9rem; font-weight:900;
}
.step-item .label{font-weight:800; font-size:.98rem; color:#eaf2ff}
.step-item.active .chip{background:linear-gradient(45deg,var(--ring),var(--ring2)); border:none}
.step-item.done .chip{background:linear-gradient(45deg,#1fbf74,#33e68b); border:none}
.step-item.done .label{color:#c8ffe3}

/* Panels */
.panel{background:var(--panel); border:1px solid var(--stroke); border-radius:18px; padding:16px 16px; margin-bottom:18px;}
.panel:hover{box-shadow:0 10px 26px var(--glow); border-color:var(--ring); transition:box-shadow .2s ease, border-color .2s ease;}
.panel h3{margin:0 0 10px; font-size:1.2rem}

/* Buttons */
.stButton>button{
  border-radius:12px!important;
  padding:.7rem 1.1rem!important;
  font-weight:900!important;
  border:1px solid rgba(255,255,255,.22)!important;
  background:linear-gradient(180deg, rgba(255,255,255,.12), rgba(255,255,255,.06))!important;
  color:var(--text)!important;
}
.btn-primary .stButton>button{
  background:linear-gradient(45deg,var(--ring),var(--ring2))!important; color:#fff!important; border:none!important;
  box-shadow:0 8px 22px var(--glow);
}
.btn-danger .stButton>button{
  background:linear-gradient(45deg,#ff6b7a,#ff3d6e)!important; color:#fff!important; border:none!important;
}
.toolbar .stButton>button{padding:.55rem .9rem!important}

/* Feed card & preview */
.feed-card{background:var(--panel); border:1px solid var(--stroke); border-radius:16px; padding:12px; margin:10px 0 18px}

/* Source tiles */
.src-grid{display:flex; gap:12px; flex-wrap:wrap; margin-top:4px;}
.src-tile{
  flex:1 1 140px; min-width:140px; display:flex; align-items:center; justify-content:center;
  padding:14px 12px; border-radius:14px; background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.14); cursor:pointer; user-select:none; font-weight:800;
}
.src-tile:hover{border-color:var(--ring)}
.src-tile.active{background:linear-gradient(45deg,var(--ring),var(--ring2)); color:#fff; border:none}
</style>
""", unsafe_allow_html=True)

# ---------- Layout shell ----------
st.markdown('<div class="container">', unsafe_allow_html=True)

# Page title
st.markdown("<h2 style='margin:18px 0 6px'>Camera Configuration</h2>", unsafe_allow_html=True)

# Step chips (exact structure retained)
area_done   = bool(st.session_state.get("selected_area"))
domain_done = bool(st.session_state.get("selected_domain"))
solution_done   = bool(st.session_state.get("selected_solution"))

selected_solution = st.session_state.get("selected_solution")
if selected_solution:
    st.info(f"Configuring cameras for solution: {selected_solution}")

if solution_done and "selected_solutions" not in st.session_state:
    st.session_state.selected_solutions = [selected_solution]

step_html = f"""
<div class="steps-row">
  <div class="step-item {'done' if area_done else ''}"><div class="chip"><div class="num">1</div>🗺️</div><div class="label">Area</div></div>
  <div class="step-item {'done' if domain_done else ''}"><div class="chip"><div class="num">2</div>🏭</div><div class="label">Domain</div></div>
  <div class="step-item {'done' if solution_done else ''}"><div class="chip"><div class="num">3</div>🛍️</div><div class="label">Solution</div></div>
  <div class="step-item active"><div class="chip"><div class="num">4</div>📷</div><div class="label">Cameras</div></div>
  <div class="step-item"><div class="chip"><div class="num">5</div>🧠</div><div class="label">Tasks</div></div>
  <div class="step-item"><div class="chip"><div class="num">6</div>📊</div><div class="label">Monitor</div></div>
  <div class="step-item"><div class="chip"><div class="num">7</div>📈</div><div class="label">Dashboard</div></div>
</div>
"""
st.markdown(step_html, unsafe_allow_html=True)

st.markdown('<h1 class="title">Camera Configuration</h1>', unsafe_allow_html=True)
st.markdown('<p class="lead">Add cameras and configure video feeds for monitoring</p>', unsafe_allow_html=True)

# Initialize camera system (UNCHANGED)
initialize_camera_manager()

# Discover models (UNCHANGED)
models = discover_models()
if not models:
    st.error("❌ No models found in 'models' directory")
    st.stop()

pt_models       = [m for m in models if m['format'] == 'pt']
onnx_models     = [m for m in models if m['format'] == 'onnx']
openvino_models = [m for m in models if m['format'] == 'openvino']
engine_models   = [m for m in models if m['format'] == 'engine']
other_models    = [m for m in models if m['format'] not in ['pt', 'onnx', 'openvino', 'engine']]
sorted_models   = pt_models + onnx_models + openvino_models + engine_models + other_models
model_options   = {f"{m['name']} ({m['format']})": m for m in sorted_models}

default_model_label  = list(model_options.keys())[0] if model_options else ""
default_model_path   = model_options[default_model_label]['path'] if model_options else ""
default_detected_task = model_options[default_model_label]['task'] or 'detect' if model_options else 'detect'
task_options = ['detect', 'segment', 'classify', 'pose']
default_task = default_detected_task if default_detected_task in task_options else 'detect'

# ---------- Two-column UI ----------
left, right = st.columns([1.05, 1], gap="large")

# LEFT: Add Camera
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # session state for the reveal flow
    if "show_camera_options" not in st.session_state:
        st.session_state.show_camera_options = False
    if "selected_source_type" not in st.session_state:
        st.session_state.selected_source_type = None

    # Collapsed → "➕ Add Camera" button
    if not st.session_state.show_camera_options:
        st.markdown("### Camera Management")
        if st.button("➕ Add Camera", use_container_width=True):
            st.session_state.show_camera_options = True
            st.rerun()
    else:
        # Expanded panel
        st.markdown("### Add Camera")

        # 1) SOURCE TYPE via tiles (click → then show fields)
        st.markdown("#### Source Type")
        st.markdown('<div class="src-grid">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📷 webcam", key="src_webcam", use_container_width=True):
                st.session_state.selected_source_type = "webcam"
        with c2:
            if st.button("🔗 rtsp", key="src_rtsp", use_container_width=True):
                st.session_state.selected_source_type = "rtsp"
        with c3:
            if st.button("📁 file", key="src_file", use_container_width=True):
                st.session_state.selected_source_type = "file"
        st.markdown('</div>', unsafe_allow_html=True)

        # subtle active tile highlight (CSS class swap)
        st.markdown(f"""
        <script>
        const active = "{st.session_state.selected_source_type or ''}";
        const btns = Array.from(parent.document.querySelectorAll('button[kind="secondary"]'));
        btns.forEach(b=>{{ if(['src_webcam','src_rtsp','src_file'].includes(b.id)) b.classList.remove('src-tile'); }});
        </script>
        """, unsafe_allow_html=True)

        # 2) Show source-specific controls ONLY after a tile click
        source_input = ""
        feed_type = st.session_state.selected_source_type

        if feed_type == "webcam":
            st.info("Using default built-in webcam (index 0)")
            webcam_indices = ["0 (Default)", "1", "2", "3"]
            selected_index = st.selectbox("Select webcam index", webcam_indices)
            source_input = selected_index.split()[0]
        elif feed_type == "rtsp":
            source_input = st.text_input(
                "RTSP URL",
                value="rtsp://username:password@ip:port/stream",
                help="RTSP URL for streaming"
            )
        elif feed_type == "file":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                temp_dir = os.path.join(os.getcwd(), "temp")
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                source_input = file_path
                st.success(f"File saved: {file_path}")
            else:
                st.info("Please upload a video file")
        else:
            st.caption("Select a source type above to continue.")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("### Camera Configuration")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            res_choice = st.selectbox("Resolution", ["640x480", "1280x720", "1920x1080"], index=1)
        with col_res2:
            fps_cap = st.number_input("FPS Cap", min_value=5, max_value=60, value=15, step=1)

        camera_name = st.text_input(
            "Camera Name",
            value=f"Camera {len(st.session_state.feeds_meta)+1}",
            help="Give your camera a descriptive name"
        )

        # Add custom CSS to ensure button consistency
        st.markdown("""
        <style>
        /* Style for primary button */
        .primary-button .stButton > button {
            background: linear-gradient(45deg, #2d8bff, #3aa6ff) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.25rem !important;
            width: 100% !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 12px rgba(45,139,255,0.25) !important;
        }
        
        /* Style for secondary button */
        .secondary-button .stButton > button {
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.25rem !important;
            width: 100% !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="primary-button">', unsafe_allow_html=True)
            add_feed_btn = st.button("Add Camera", use_container_width=True, disabled=not (feed_type and (feed_type=="webcam" or source_input.strip())))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="secondary-button">', unsafe_allow_html=True)
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_camera_options = False
                st.session_state.selected_source_type = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if add_feed_btn:
            feed_id = str(uuid.uuid4())[:8]
            w, h = map(int, res_choice.split('x'))
            cfg = FeedConfig(
                id=feed_id,
                source=(source_input.strip() if feed_type!="webcam" else "0" if source_input=="" else source_input),
                type=feed_type,
                resolution=(w, h),
                fps_cap=fps_cap,
                task={'type': default_task, 'model': default_model_path}
            )
            try:
                st.session_state.cam_manager.add_feed(cfg)
                st.session_state.feeds_meta[feed_id] = {
                    'name': camera_name,
                    'model_settings': {'model_path': default_model_path, 'task': default_task},
                    'task_settings': {}
                }

                from capture.feed_config import save_feeds
                feeds_to_save = []
                for fs in st.session_state.cam_manager.list_feeds():
                    feeds_to_save.append({
                        'id': fs.config.id,
                        'source': fs.config.source,
                        'type': fs.config.type,
                        'resolution': fs.config.resolution,
                        'fps_cap': fs.config.fps_cap,
                        'name': st.session_state.feeds_meta[fs.config.id]['name'],
                        'model_settings': st.session_state.feeds_meta[fs.config.id]['model_settings'],
                        'task_settings': st.session_state.feeds_meta[fs.config.id]['task_settings']
                    })
                save_feeds(feeds_to_save)

                # Extended metadata (kept from your original)
                st.session_state.feeds_meta[feed_id].update({
                    'primary_model_path': default_model_path,
                    'primary_task': default_task,
                    'primary_inference': None,
                    'secondary_enabled': False,
                    'secondary_model_path': None,
                    'secondary_task': None,
                    'secondary_inference': None,
                })

                try:
                    st.session_state.cam_manager.start(feed_id)
                    st.success(f"✅ Added and started camera: {camera_name}")
                    # Close after success
                    st.session_state.show_camera_options = False
                    st.session_state.selected_source_type = None
                    st.rerun()
                except Exception as start_err:
                    st.error(f"Camera added but failed to start: {start_err}")

            except Exception as e:
                st.error(f"Failed to add camera: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # /panel

# RIGHT: Active Feeds (UNCHANGED)
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📹 Active Feeds")
    feeds = st.session_state.cam_manager.list_feeds()

    if feeds:
        st.write(f"Total feeds: {len(feeds)}")
        st.markdown('<div class="toolbar">', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            if st.button("▶️ Start All"):
                for fs in feeds:
                    if fs.status in ('stopped', 'disconnected'):
                        try:
                            st.session_state.cam_manager.start(fs.config.id)
                        except Exception as e:
                            st.error(f"Failed to start {fs.config.id}: {e}")
        with t2:
            if st.button("⏹️ Stop All"):
                for fs in feeds:
                    try:
                        st.session_state.cam_manager.stop(fs.config.id)
                    except Exception as e:
                        st.error(f"Failed to stop {fs.config.id}: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        for feed_state in feeds:
            st.markdown('<div class="feed-card">', unsafe_allow_html=True)
            feed_id = feed_state.config.id
            meta = st.session_state.feeds_meta.get(feed_id, {})
            name = meta.get('name', feed_id)
            w, h = feed_state.config.resolution
            status_icon = {
                'live': '🟢', 'connecting': '🟡', 'stopped': '⚪', 'disconnected': '🔴'
            }.get(feed_state.status, '⚪')

            st.markdown(f"**{status_icon} {name}**  ({w}x{h} @ {feed_state.config.fps_cap}fps)")

            col1, col2 = st.columns([1, 1])
            with col1:
                start_stop = st.button(
                    "Start" if feed_state.status in ('stopped', 'disconnected') else "Stop",
                    key=f"startstop_{feed_id}"
                )
                if start_stop:
                    try:
                        if feed_state.status in ('stopped', 'disconnected'):
                            st.session_state.cam_manager.start(feed_id)
                        else:
                            st.session_state.cam_manager.stop(feed_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to toggle feed: {e}")

            with col2:
                st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
                if st.button("Remove", key=f"remove_{feed_id}"):
                    try:
                        st.session_state.cam_manager.remove(feed_id)
                        st.session_state.feeds_meta.pop(feed_id, None)
                        from capture.feed_config import save_feeds
                        feeds_to_save = []
                        for fs in st.session_state.cam_manager.list_feeds():
                            feeds_to_save.append({
                                'id': fs.config.id,
                                'source': fs.config.source,
                                'resolution': fs.config.resolution,
                                'fps_cap': fs.config.fps_cap,
                                'name': st.session_state.feeds_meta[fs.config.id]['name'],
                                'model_settings': st.session_state.feeds_meta[fs.config.id]['model_settings'],
                                'task_settings': st.session_state.feeds_meta[fs.config.id]['task_settings']
                            })
                        save_feeds(feeds_to_save)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove feed: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

            if feed_state.last_frame is not None:
                st.image(feed_state.last_frame, channels="BGR",
                         caption=f"Preview: {name} ({feed_state.status})", use_container_width=True)
            else:
                st.image("assets/background.png",
                         caption=f"Feed {name} has no frames ({feed_state.status})", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)  # /feed-card
    else:
        st.image("assets/background.png", caption="No feeds configured")
        st.info("Add a camera feed to begin monitoring")

    st.markdown('</div>', unsafe_allow_html=True)  # /panel

# Device panel (UNCHANGED)
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("### 🖥️ Device")
device_config = get_device_config(prefer_gpu=True)
device_name = device_config.get('device_name', 'Unknown')
device_type = device_config.get('device_type', 'unknown')

if device_type == 'cuda':
    st.success(f"GPU: {device_name} (FP16: {device_config['half']})")
elif device_type == 'mps':
    st.success(f"Apple GPU: {device_name}")
elif device_type == 'openvino_gpu':
    st.success(f"Intel GPU: {device_name} (OpenVINO)")
elif device_type == 'openvino_cpu':
    st.info(f"CPU: {device_name} (OpenVINO)")
else:
    st.info(f"CPU: {device_name}")
st.markdown('</div>', unsafe_allow_html=True)

# Nav (UNCHANGED)
st.markdown('<div class="navbar">', unsafe_allow_html=True)
col_back, col_next = st.columns([1,1])
with col_back:
    if st.button("← Back"):
        try:
            st.switch_page("pages/3_Solution.py")
        except Exception:
            st.page_link("pages/3_Solution.py", label="Back to Solution", icon="↩️")
with col_next:
    st.markdown('<div class="btn-primary" style="display:flex; justify-content:flex-end;">', unsafe_allow_html=True)
    if st.button("Next →"):
        try:
            st.switch_page("pages/5_Tasks.py")
        except Exception:
            st.page_link("pages/5_Tasks.py", label="Continue to Tasks →", icon="➡️")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /container
