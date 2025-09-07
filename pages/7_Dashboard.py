"""
Dashboard Router

This page routes to the appropriate dashboard module based on the selected solution.
"""
import os
import streamlit as st
from ui_components import render_static_header, render_brand_header, render_solution_context, verify_and_restore_state, ensure_cross_tab_data_persistence
from solutions import get_solution_by_name

# Import solution registry to ensure all solutions are registered
import solutions.registry

# Try to import plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ---------- Page config ----------
st.set_page_config(
    page_title="CraftEye - Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Check URL parameters for cross-tab data sharing
if "feed_id" in st.query_params:
    feed_id = st.query_params["feed_id"]
    if feed_id:
        st.session_state["current_feed_id"] = feed_id
        # Reset refresh timer when switching feeds
        if "dash_last_refresh" in st.session_state:
            st.session_state.dash_last_refresh = 0.0

# Global headers (rendered on EVERY page)
render_static_header()     # your existing top strip, if any
render_brand_header()      # new fixed banner with CraftEye + CraftifAI logo
render_solution_context()  # display selected solution if available

# Add error handling
try:
    # Verify state and enable cross-tab data persistence
    verify_and_restore_state()
    ensure_cross_tab_data_persistence()
except Exception as e:
    st.error(f"Error in dashboard initialization: {str(e)}")
    st.info("Try refreshing the page or going back to the Monitor page")

# Implement adaptive refresh using st.rerun() based on data source
import time

# Initialize refresh timer if not already set
if "dash_last_refresh" not in st.session_state:
    st.session_state.dash_last_refresh = 0.0

# Determine refresh interval based on data source
# Use shorter interval when monitoring is active in same tab,
# and longer interval when polling from CSV across tabs
monitoring_active = st.session_state.get("monitoring_active", False)

# Check if we're in a new tab by examining query parameters
in_separate_tab = "feed_id" in st.query_params 

# Adjust refresh rate based on context
if monitoring_active and not in_separate_tab:
    REFRESH_SEC = 1  # Faster updates when monitoring in same tab
elif in_separate_tab:
    REFRESH_SEC = 2  # Moderate updates when in separate tab with feed_id
else:
    REFRESH_SEC = 5  # Slower updates when just polling CSV with no active monitoring

# Trigger refresh when time has elapsed
now = time.time()
if now - st.session_state.dash_last_refresh >= REFRESH_SEC:
    st.session_state.dash_last_refresh = now
    st.rerun()

# ---------- Styles ----------
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
:root{
  --bg:#0b1220; --bg2:#0c1426; --text:#f7faff; --muted:#dbe6ff;
  --panel:rgba(255,255,255,.05); --stroke:rgba(255,255,255,.14);
  --ring:#2d8bff; --ring2:#3aa6ff; --glow:rgba(58,166,255,.38);
  --ok:#2bd38a; --warn:#ffb020; --danger:#ff5a7a;
}

.block-container { padding-top: 5px; }
.stApp{background:linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%); color:var(--text); font-size:16.5px;}
.container{max-width:1180px; margin:0 auto; padding:24px 16px;}
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

# ---------- Main UI ----------
st.markdown('<div class="container">', unsafe_allow_html=True)

# Steps chips
area_done   = bool(st.session_state.get("selected_area"))
domain_done = bool(st.session_state.get("selected_domain"))
solution_done   = bool(st.session_state.get("selected_solution"))
cams_done   = bool(st.session_state.get("cam_manager")) and len(getattr(st.session_state.get("cam_manager"), "list_feeds")()) > 0 \
             if st.session_state.get("cam_manager") else False
tasks_done  = bool(st.session_state.get("model_settings"))

st.markdown(f"""
<div class="steps-row">
  <div class="step-item {'done' if area_done else ''}"><div class="chip"><div class="num">1</div>🗺️</div><div class="label">Area</div></div>
  <div class="step-item {'done' if domain_done else ''}"><div class="chip"><div class="num">2</div>🏭</div><div class="label">Domain</div></div>
  <div class="step-item {'done' if solution_done else ''}"><div class="chip"><div class="num">3</div>🛍️</div><div class="label">Solution</div></div>
  <div class="step-item {'done' if cams_done else ''}"><div class="chip"><div class="num">4</div>📷</div><div class="label">Cameras</div></div>
  <div class="step-item {'done' if tasks_done else ''}"><div class="chip"><div class="num">5</div>🧠</div><div class="label">Tasks</div></div>
  <div class="step-item {'done' if st.session_state.get('monitoring_active') else ''}"><div class="chip"><div class="num">6</div>📊</div><div class="label">Monitor</div></div>
  <div class="step-item active"><div class="chip"><div class="num">7</div>📈</div><div class="label">Dashboard</div></div>
</div>
""", unsafe_allow_html=True)

# Get the selected solution name
selected_solution_name = st.session_state.get("selected_solution")

# Check if we can infer a solution from our current context
# This is a fallback mechanism for when the session state is lost
if not selected_solution_name:
    # If we have specific indicators of the Crowd Monitoring solution
    if "monitoring_active" in st.session_state or "feeds_meta" in st.session_state:
        selected_solution_name = "Pilgrim Crowd Monitoring"
        st.session_state["selected_solution"] = selected_solution_name
        
        # Also make sure we have area and domain set
        if "selected_area" not in st.session_state:
            st.session_state["selected_area"] = "Pilgrim Site"
        if "selected_domain" not in st.session_state:
            st.session_state["selected_domain"] = "Crowd Management"

# Use the solution registry to route to the appropriate dashboard module
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
    st.info(f"Analytics dashboard for solution: {selected_solution_name}")
    
    # Get solution from registry
    solution = get_solution_by_name(selected_solution_name)
    
    if not solution:
        st.error(f"Solution '{selected_solution_name}' not found in registry.")
        
        # Special handling for Pilgrim Crowd Monitoring
        if "pilgrim" in selected_solution_name.lower() or "crowd" in selected_solution_name.lower():
            try:
                # Try direct import of crowd dashboard
                from solutions.crowd_monitoring.dashboard import render as crowd_dashboard_render
                st.success("Successfully loaded Crowd Monitoring dashboard directly")
                crowd_dashboard_render()
                st.stop()
            except Exception as e:
                st.error(f"Error loading crowd dashboard directly: {e}")
        
        # Add recovery option
        if st.button("Reset to Pilgrim Crowd Monitoring"):
            st.session_state["selected_solution"] = "Pilgrim Crowd Monitoring"
            st.session_state["selected_area"] = "Pilgrim Site"
            st.session_state["selected_domain"] = "Crowd Management"
            st.rerun()
    elif not solution.get("dashboard_module"):
        st.warning(f"Solution '{selected_solution_name}' does not have a dashboard module.")
        
        # Special handling for Pilgrim Crowd Monitoring
        if "pilgrim" in selected_solution_name.lower() or "crowd" in selected_solution_name.lower():
            try:
                # Try direct import of crowd dashboard
                from solutions.crowd_monitoring.dashboard import render as crowd_dashboard_render
                st.success("Successfully loaded Crowd Monitoring dashboard directly")
                crowd_dashboard_render()
                st.stop()
            except Exception as e:
                st.error(f"Error loading crowd dashboard directly: {e}")
                
        st.info("No dashboard is available for this solution.")
        st.markdown('<h1 class="title">Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="lead">View insights and historical data</p>', unsafe_allow_html=True)
        st.error("This solution has no dashboard implementation. Please implement a dashboard module for this solution.")
    else:
        try:
            # Call the solution's dashboard module
            solution["dashboard_module"]()
            # Skip the rest of the original dashboard code
            st.stop()
        except Exception as render_error:
            st.error(f"Error rendering dashboard: {str(render_error)}")
            st.info("Loading fallback dashboard...")
            
            # Try direct import as fallback
            try:
                from solutions.crowd_monitoring.dashboard import render as crowd_dashboard_render
                crowd_dashboard_render()
                st.stop()
            except Exception as fallback_error:
                st.error(f"Fallback dashboard also failed: {str(fallback_error)}")
                st.warning("Please check your Streamlit installation and try restarting the application")

# Back button
st.markdown('<div style="margin-top:16px;">', unsafe_allow_html=True)
if st.button("Back to Monitoring"):
    try:
        # Try using our navigation helper
        from solutions.crowd_monitoring.navigation import navigate_to_monitor
        feed_id = st.session_state.get("current_feed_id")
        navigate_to_monitor(feed_id)
    except Exception as e:
        # Fallback to legacy methods
        try:
            st.switch_page("/6_Monitor.py")
        except Exception:
            try:
                st.page_link("/6_Monitor.py", label="Back to Monitor", icon="↩️")
            except Exception:
                st.markdown('<a href="/6_Monitor" target="_self">Back to Monitor</a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /container
