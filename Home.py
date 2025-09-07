"""
CraftEye Main Application Entry Point

This is the main entry point for the CraftEye application.
It initializes the solution registry and sets up the application.
"""
import os, base64
import streamlit as st
from ui_components import render_static_header, render_brand_header

# Import and register all solutions
import solutions.registry

# ---------- Page config ----------
st.set_page_config(
    page_title="CraftEye • Multi Vision Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global headers (rendered on EVERY page)
render_static_header()     # your existing top strip, if any
render_brand_header()      # new fixed banner with CraftEye + CraftifAI logo

# No helpers needed now - brand header is in ui_components.py

# ---------- Styles (higher contrast + hero brand card) ----------
st.markdown(
    """
<style>
#MainMenu, footer {visibility:hidden;}

:root{
  --bg:#0b1220;      --bg2:#0c1426;
  --text:#f7faff;    --muted:#dbe6ff;
  --card:rgba(255,255,255,.04);
  --ring:#2d8bff;    --ring2:#3aa6ff;
}

.stApp{
  background:linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%);
  color:var(--text);
  font-size:16.5px;
}

/* give a little room under sticky banner */
.block-container { padding-top: 5px; }

.container{max-width:1180px; margin:0 auto; padding:24px 16px;}

/* Deploy styling is now in ui_components.py */

/* Hero title & preamble */
h1.hero{
  font-size:3.3rem; line-height:1.06; margin:44px 0 12px; text-align:center; font-weight:900;
}
.preamble{
  color:#eaf0ff; text-align:center; max-width:900px; margin:0 auto 28px; font-size:1.18rem; line-height:1.55;
}

/* Steps */
.steps{
  display:grid; grid-template-columns:repeat(7,1fr); gap:12px; margin:32px 0 12px;
  background:var(--card); border:1px solid rgba(255,255,255,.10); padding:16px; border-radius:24px;
}
.step{display:flex; flex-direction:column; align-items:center; gap:7px; padding:12px 8px}
.badge{
  display:flex; align-items:center; justify-content:center; gap:8px;
  height:38px; padding:0 14px; border-radius:999px; font-weight:800; color:var(--text);
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10)
}
.num{width:26px; height:26px; border-radius:999px; display:grid; place-items:center;
     background:rgba(255,255,255,.12); color:#eef3ff; font-size:.9rem; font-weight:900}
.label{font-weight:850; font-size:1.05rem}
.sub{font-size:.96rem; color:var(--muted); text-align:center}

/* CTA button */
.stButton>button{
  background:linear-gradient(45deg,var(--ring),var(--ring2))!important; color:#fff!important;
  border:none!important; padding:.9rem 2.4rem!important; font-size:1.2rem!important; font-weight:900!important;
  border-radius:14px!important; transition:transform .18s ease!important;
}
.stButton>button:hover{transform:translateY(-1px) scale(1.01)}

/* Feature cards */
.grid{display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:22px 0 12px;}
.card{
  background:var(--card); border:1px solid rgba(255,255,255,.10); border-radius:20px; padding:18px 20px;
}
.card h3{margin:0 0 8px; font-size:1.18rem;}
.card p{margin:0; color:#e1e9ff; font-size:1.03rem}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Main content ----------

# ---------- Layout ----------
st.markdown('<div class="container">', unsafe_allow_html=True)

# Page hero (text only; the fixed banner is already at the top)
st.markdown('<h1 class="hero">Edge Vision Platform</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="preamble">Welcome to the next generation of computer vision solutions. '
    'Our platform helps you set up and manage intelligent vision systems for your business needs.</div>',
    unsafe_allow_html=True,
)

# Steps
st.markdown(
    """
<div class="steps">
  <div class="step"><div class="badge"><span class="num">1</span>🗺️</div><div class="label">Area</div><div class="sub">Deployment location</div></div>
  <div class="step"><div class="badge"><span class="num">2</span>🏭</div><div class="label">Domain</div><div class="sub">Industry focus</div></div>
  <div class="step"><div class="badge"><span class="num">3</span>🛍️</div><div class="label">Solution</div><div class="sub">Solution selection</div></div>
  <div class="step"><div class="badge"><span class="num">4</span>📷</div><div class="label">Cameras</div><div class="sub">Video sources</div></div>
  <div class="step"><div class="badge"><span class="num">5</span>🧠</div><div class="label">Tasks</div><div class="sub">AI configuration</div></div>
  <div class="step"><div class="badge"><span class="num">6</span>📊</div><div class="label">Monitor</div><div class="sub">Live analysis</div></div>
  <div class="step"><div class="badge"><span class="num">7</span>📈</div><div class="label">Dashboard</div><div class="sub">Analytics view</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# CTA (with safe navigation)
_, mid, _ = st.columns([2, 1, 2])
with mid:
    if st.button("Get Started →", use_container_width=True, key="cta"):
        try:
            st.switch_page("pages/1_Area.py")
        except Exception:
            st.session_state["_nav_error"] = True
    if st.session_state.get("_nav_error"):
        st.page_link("pages/1_Area.py", label="Continue to Area →", icon="➡️")

# Feature cards
st.markdown(
    """
<div class="grid">
  <div class="card"><h3>Edge-native</h3><p>Run close to cameras for low latency and privacy-first deployments.</p></div>
  <div class="card"><h3>No-code Setup</h3><p>Pick domains, add sources, and configure tasks in guided steps.</p></div>
  <div class="card"><h3>Real-time Metrics</h3><p>Monitor FPS, queue depth, and model KPIs as you go.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)


