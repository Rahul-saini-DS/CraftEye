# pages/2_Domain.py
import os, base64
import streamlit as st
from ui_components import render_static_header, render_brand_header, render_solution_context

# ---------- Page config ----------
st.set_page_config(
    page_title="CraftEye - Domain",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global headers (rendered on EVERY page)
render_static_header()     # your existing top strip, if any
render_brand_header()      # new fixed banner with CraftEye + CraftifAI logo
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

# ---------- Styles (consistent with Home/Area) ----------
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}

:root{
  --bg:#0b1220; --bg2:#0c1426;
  --text:#f7faff; --muted:#dbe6ff;
  
/* give a little room under sticky banner */
.block-container { padding-top: 5px; }
  --panel:rgba(255,255,255,.05);
  --stroke:rgba(255,255,255,.14);
  --ring:#2d8bff; --ring2:#3aa6ff; --glow:rgba(58,166,255,.38);
  --ok:#2bd38a;
}

.stApp{background:linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%); color:var(--text); font-size:16.5px;}
.container{max-width:1180px; margin:0 auto; padding:24px 16px;}

/* Brand header */
.brand-hero{
  display:flex; align-items:center; gap:18px;
  padding:24px 26px; margin:12px 0 22px;
  border-radius:22px; position:relative; overflow:hidden;
  background:
    radial-gradient(80% 140% at 0% 0%, rgba(61,151,255,.16) 0%, rgba(61,151,255,0) 60%),
    linear-gradient(180deg, rgba(255,255,255,.09), rgba(255,255,255,.04));
  border:1px solid rgba(255,255,255,.22);
  box-shadow:0 18px 55px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.06);
}
.logo-badge{
  width:78px; height:78px; border-radius:18px; display:grid; place-items:center;
  background:
    radial-gradient(120px 120px at 30% 30%, rgba(90,170,255,.35), rgba(90,170,255,.06) 60%),
    linear-gradient(135deg, rgba(90,170,255,.32), rgba(10,18,32,0));
  border:1px solid rgba(140,190,255,.45);
  box-shadow:0 14px 32px rgba(45,139,255,.30), inset 0 0 0 5px rgba(255,255,255,.04);
}
.logo-badge img{height:58px; width:auto;}
.brand-title{font-weight:900; font-size:1.9rem}
.brand-sub{color:#cfe4ff; font-size:1.06rem}

/* Steps row */
.steps-row{
  display:flex; gap:12px; align-items:center; flex-wrap:wrap;
  background:linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.04));
  border:1px solid var(--stroke); padding:12px 14px; border-radius:18px; margin:0 0 26px;
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

/* Title & lead */
h1.title{font-size:2.6rem; font-weight:900; margin:6px 0 6px;}
p.lead{color:#e8f0ff; font-size:1.14rem; margin:0 0 20px}

/* Domain tiles */
.opt-btn{cursor:pointer; position:relative; margin-bottom: 10px;}
.opt-btn .stButton>button{
  width:100%; text-align:left; cursor:pointer!important;
  background:var(--panel)!important; border:1px solid var(--stroke)!important; color:var(--text)!important;
  border-radius:18px!important; padding:1.05rem 1.15rem!important; font-size:1.12rem!important; font-weight:900!important;
  transition:transform .16s ease, box-shadow .2s ease, border-color .2s ease!important;
  white-space: normal !important; line-height:1.35;
  height: auto; /* Allow height to adjust to content */
  min-height: 60px; /* Match the height of coming soon cards */
}
.opt-btn .stButton>button:hover{
  transform:translateY(-1px); box-shadow:0 12px 30px var(--glow); border-color:var(--ring)!important;
}
.opt-btn.selected .stButton>button{
  background:linear-gradient(180deg, rgba(58,166,255,.14), rgba(255,255,255,.06))!important;
  border-color:var(--ring)!important; box-shadow:0 0 0 2px rgba(58,166,255,.25) inset;
}

/* Coming Soon domain styles */
.domain-card {
  position: relative;
  margin-bottom: 10px;
  border-radius: 18px;
  overflow: hidden;
  cursor: help;
  transition: all 0.3s ease;
  min-height: 60px; /* Ensure consistent height */
  display: flex;
  flex-direction: column;
}
.domain-card.coming-soon {
  filter: saturate(0.6);
  opacity: 0.9;
  background: linear-gradient(180deg, rgba(70,70,90,.1), rgba(40,40,60,.08));
  border: 1px solid rgba(255,255,255,.1);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.domain-card.coming-soon:hover {
  filter: saturate(0.7);
  opacity: 0.95;
  border-color: rgba(255,255,255,.18);
  box-shadow: 0 6px 16px rgba(0,0,0,0.2);
  transform: translateY(-1px);
}
.domain-card .card-content {
  padding: 1.05rem 1.15rem;
  font-size: 1.12rem;
  font-weight: 900;
  color: var(--text);
  line-height: 1.35;
  flex: 1;
}
.domain-card.coming-soon .card-content {
  color: rgba(255,255,255,0.75);
}
.coming-soon-badge {
  position: absolute;
  top: 0;
  right: 0;
  background: linear-gradient(135deg, rgba(43,211,138,0.9), rgba(31,191,116,0.8));
  color: white;
  padding: 4px 10px;
  font-size: 0.7rem;
  font-weight: 600;
  border-bottom-left-radius: 10px;
  box-shadow: -1px 1px 3px rgba(0,0,0,0.2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  z-index: 10;
}
.domain-tooltip {
  visibility: hidden;
  position: absolute;
  background: rgba(40, 44, 68, 0.95);
  color: white;
  text-align: center;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 14px;
  font-weight: normal;
  box-shadow: 0 3px 15px rgba(0,0,0,0.3);
  z-index: 100;
  bottom: -35px;
  left: 50%;
  transform: translateX(-50%);
  transition: all 0.2s ease-in-out;
  width: max-content;
  max-width: 200px;
  border: 1px solid rgba(255,255,255,0.1);
  opacity: 0;
}
.domain-card:hover + .domain-tooltip {
  visibility: visible !important;
  opacity: 1;
  bottom: -45px;
}

/* Nav buttons */
.navbar{display:flex; justify-content:space-between; align-items:center; margin-top:18px}
.btn-secondary .stButton>button{
  border-radius:12px!important; padding:.75rem 1.1rem!important; font-weight:900!important;
  background:linear-gradient(180deg, rgba(255,255,255,.12), rgba(255,255,255,.06))!important;
  color:var(--text)!important; border:1px solid rgba(255,255,255,.22)!important;
}
.btn-primary .stButton>button{
  border-radius:14px!important; padding:.9rem 1.4rem!important; font-weight:900!important;
  background:linear-gradient(45deg,var(--ring),var(--ring2))!important;
  color:#fff!important; border:none!important; box-shadow:0 10px 26px var(--glow);
}
.btn-primary .stButton>button:disabled{
  opacity:.45; filter:saturate(.6); box-shadow:none; cursor:not-allowed!important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)

# ---------- Page title ----------
st.markdown("<h2 style='margin:18px 0 6px'>Select Industry Domain</h2>", unsafe_allow_html=True)

# ---------- Steps (Step 1 done if area picked; Step 2 active/done if domain picked) ----------
area_done = bool(st.session_state.get("selected_area"))
domain_done = bool(st.session_state.get("selected_domain"))

st.markdown(f"""
<div class="steps-row">
  <div class="step-item {'done' if area_done else ''}">
    <div class="chip"><div class="num">1</div>🗺️</div><div class="label">Area</div>
  </div>
  <div class="step-item {'active done' if domain_done else 'active'}">
    <div class="chip"><div class="num">2</div>🏭</div><div class="label">Domain</div>
  </div>
  <div class="step-item"><div class="chip"><div class="num">3</div>🛍️</div><div class="label">Solution</div></div>
  <div class="step-item"><div class="chip"><div class="num">4</div>📷</div><div class="label">Cameras</div></div>
  <div class="step-item"><div class="chip"><div class="num">5</div>🧠</div><div class="label">Tasks</div></div>
  <div class="step-item"><div class="chip"><div class="num">6</div>📊</div><div class="label">Monitor</div></div>
  <div class="step-item"><div class="chip"><div class="num">7</div>📈</div><div class="label">Dashboard</div></div>
</div>
""", unsafe_allow_html=True)

# ---------- Title & intro ----------
st.markdown('<h1 class="title">Select Industry Domain</h1>', unsafe_allow_html=True)
st.markdown('<p class="lead">Choose your industry focus for tailored computer vision solutions</p>', unsafe_allow_html=True)

# ---------- State ----------
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None

def domain_tile(container, key, emoji, title, subtitle, value, clickable=True):
    selected = (st.session_state.selected_domain == value)
    
    if clickable:
        # Regular clickable domain
        css_class = "opt-btn selected" if selected else "opt-btn"
        label = f"{emoji}  {title}  •  {subtitle}"
        
        with container:
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            clicked = st.button(label, key=key, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if clicked:
            st.session_state.selected_domain = value
    else:
        # Coming soon domain with custom styling
        card_id = f"coming_soon_{key}"
        tooltip_id = f"tooltip_{key}"
        
        with container:
            st.markdown(f"""
            <div class="domain-card coming-soon" id="{card_id}">
                <div class="coming-soon-badge">Coming Soon</div>
                <div class="card-content">
                    {emoji}  {title}  •  {subtitle}
                </div>
            </div>
            <div class="domain-tooltip" id="{tooltip_id}">
                This domain will be available soon
            </div>
            """, unsafe_allow_html=True)

# ---------- Styles for consistent spacing ----------
st.markdown("""
<style>
/* Add consistent spacing for all domain tiles */
.domain-wrapper {
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Tiles grid (show domains based on selected area) ----------
selected_area = st.session_state.get("selected_area", None)

# If Ujjain is selected, only show Pilgrim Safety domain
if selected_area == "ujjain":
    # Single column for Pilgrim Safety
    col = st.container()
    domain_tile(col, "dom_pilgrim", "🛕", "Pilgrim Safety", "Crowd monitoring, emergency response, and public safety", "pilgrim", clickable=True)
else:
    # Show all domains as before
    r1c1, r1c2, r1c3 = st.columns(3, gap="large")
    domain_tile(r1c1, "dom_retail",        "🛍️", "Retail",        "Customer behavior, inventory & checkout", "retail", clickable=True)
    domain_tile(r1c2, "dom_pilgrim",       "🛕", "Pilgrim Safety", "Crowd monitoring, emergency response, and public safety", "pilgrim", clickable=True)
    domain_tile(r1c3, "dom_health",        "🏥", "Health",        "PPE/compliance, patient flow",            "health", clickable=False)

    r2c1, r2c2, r2c3 = st.columns(3, gap="large")
    domain_tile(r2c1, "dom_transport",     "🚚", "Transport",     "Traffic monitoring & vehicle safety",     "transport", clickable=False)
    domain_tile(r2c2, "dom_sports",        "🏟️", "Sports",        "Player tracking & performance analytics", "sports", clickable=False)
    domain_tile(r2c3, "dom_manufacturing", "🏭", "Manufacturing", "Quality control, safety, production",      "manufacturing", clickable=False)


# ---------- Navigation ----------
st.markdown('<div class="navbar">', unsafe_allow_html=True)

left, right = st.columns([1,1])
with left:
    st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
    if st.button("← Back"):
        try:
            st.switch_page("pages/1_Area.py")
        except Exception:
            st.page_link("pages/1_Area.py", label="Back to Area", icon="↩️")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="btn-primary" style="display:flex; justify-content:flex-end;">', unsafe_allow_html=True)
    if st.button("Next →", disabled=not st.session_state.selected_domain):
        try:
            st.switch_page("pages/3_Solution.py")
        except Exception:
            st.session_state["_next_nav_error"] = True
    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.get("_next_nav_error"):
        st.page_link("pages/3_Solution.py", label="Continue to Solution →", icon="➡️")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
