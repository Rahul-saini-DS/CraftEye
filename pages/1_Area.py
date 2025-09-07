# pages/1_Area.py
import os, base64
import streamlit as st
from ui_components import render_static_header, render_brand_header, render_solution_context

# ---------- Page config ----------
st.set_page_config(
    page_title="CraftEye - Area Selection",
    page_icon="🎯",
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

# ---------- Styles (aligned with Home; stronger hierarchy, clear CTA) ----------
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}

:root{
  --bg:#0b1220; --bg2:#0c1426;
  --text:#f7faff; --muted:#dbe6ff;

/* give a little room under sticky banner */
.block-container { padding-top: 5px; }
  --panel:rgba(255,255,255,.05); --panel2:rgba(255,255,255,.08);
  --stroke:rgba(255,255,255,.14);
  --ring:#2d8bff; --ring2:#3aa6ff; --glow:rgba(58,166,255,.38);
  --ok:#2bd38a;
}

.stApp{
  background:linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%);
  color:var(--text);
  font-size:16.5px;
}
.container{max-width:1180px; margin:0 auto; padding:24px 16px;}

/* Brand header (same as Home style) */
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

/* Steps row (mirror Home chips; active & done states) */
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
h1.title{font-size:2.65rem; font-weight:900; margin:4px 0 6px;}
p.lead{color:#e8f0ff; font-size:1.14rem; margin:0 0 20px}

/* Option cards (full-card clickable buttons; include subtitle in label) */
.opt-btn{cursor:pointer}
.opt-btn .stButton>button{
  width:100%; text-align:left; cursor:pointer!important;
  background:var(--panel)!important; border:1px solid var(--stroke)!important; color:var(--text)!important;
  border-radius:18px!important; padding:1.1rem 1.2rem!important; font-size:1.12rem!important; font-weight:900!important;
  transition:transform .16s ease, box-shadow .2s ease, border-color .2s ease!important;
  white-space: normal !important; line-height:1.35;
  display:flex; gap:.55rem; align-items:flex-start;
}
.opt-btn .stButton>button:hover{
  transform:translateY(-1px); box-shadow:0 12px 30px var(--glow); border-color:var(--ring)!important;
}
.opt-btn.selected .stButton>button{
  background:linear-gradient(180deg, rgba(58,166,255,.14), rgba(255,255,255,.06))!important;
  border-color:var(--ring)!important; box-shadow:0 0 0 2px rgba(58,166,255,.25) inset;
}

/* Primary/secondary nav buttons */
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
st.markdown("<h2 style='margin:18px 0 6px'>Choose Deployment Area</h2>", unsafe_allow_html=True)

# ---------- Steps row (consistent with Home; mark Area as active, show 'done' when selected) ----------
area_done = bool(st.session_state.get("selected_area"))

st.markdown(f"""
<div class="steps-row">
  <div class="step-item {'active done' if area_done else 'active'}">
    <div class="chip"><div class="num">1</div>🗺️</div><div class="label">Area</div>
  </div>
  <div class="step-item"><div class="chip"><div class="num">2</div>🏭</div><div class="label">Domain</div></div>
  <div class="step-item"><div class="chip"><div class="num">3</div>🛍️</div><div class="label">Solution</div></div>
  <div class="step-item"><div class="chip"><div class="num">4</div>📷</div><div class="label">Cameras</div></div>
  <div class="step-item"><div class="chip"><div class="num">5</div>🧠</div><div class="label">Tasks</div></div>
  <div class="step-item"><div class="chip"><div class="num">6</div>📊</div><div class="label">Monitor</div></div>
  <div class="step-item"><div class="chip"><div class="num">7</div>📈</div><div class="label">Dashboard</div></div>
</div>
""", unsafe_allow_html=True)

# ---------- Lead ----------
st.markdown('<p class="lead">Select the primary area for your computer vision deployment</p>', unsafe_allow_html=True)

# ---------- State ----------
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None

def area_button(col, key, emoji, title, subtitle, value):
    selected = (st.session_state.selected_area == value)
    wrapper_class = "opt-btn selected" if selected else "opt-btn"
    # Put subtitle in the button label (with wrap), so the entire tile is clickable.
    label = f"{emoji}  {title}  •  {subtitle}"
    with col:
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        clicked = st.button(label, key=key, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    if clicked:
        st.session_state.selected_area = value

# ---------- Options grid (2 x 3) ----------
c1, c2 = st.columns(2, gap="large")
area_button(c1, "area_in_store",   "🛒", "In-Store",    "Customer-facing retail areas",   "in_store")
area_button(c2, "area_warehouse",  "📦", "Warehouse",   "Storage and logistics",          "warehouse")
area_button(c1, "area_back_office","🏬", "Back Office", "Staff and administrative areas", "back_office")
area_button(c2, "area_parking",    "🚗", "Parking Lot", "Outdoor customer areas",         "parking")
area_button(c1, "area_ujjain",     "🛕", "Ujjain",      "Simhastha 2028, Madhya Pradesh", "ujjain")

# ---------- Nav (prominent Next, consistent Back) ----------
st.markdown('<div class="navbar">', unsafe_allow_html=True)

left, right = st.columns([1,1])
with left:
    st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
    if st.button("← Back"):
        for home in ("Home.py", "app.py"):
            try:
                st.switch_page(home)
                break
            except Exception:
                pass
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="btn-primary" style="display:flex; justify-content:flex-end;">', unsafe_allow_html=True)
    if st.button("Next →", disabled=not st.session_state.selected_area):
        try:
            st.switch_page("pages/2_Domain.py")
        except Exception:
            st.session_state["_next_nav_error"] = True
    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.get("_next_nav_error"):
        st.page_link("pages/2_Domain.py", label="Continue to Domain →", icon="➡️")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)




