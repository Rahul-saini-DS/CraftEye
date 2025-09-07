"""
Solution Selection Page

This page displays available solutions from the registry and allows the user to select one.
"""
import os, base64
import streamlit as st
import time
from ui_components import render_static_header, render_brand_header, render_solution_context
from solutions import list_available_solutions, list_coming_soon_solutions

# Import solution registry to ensure all solutions are registered
import solutions.registry

# ---------- Page config ----------
st.set_page_config(
    page_title="CraftEye - Solution Selection",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Global headers ----------
render_static_header()
render_brand_header()
render_solution_context()

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

# ---------- Styles ----------
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
:root{
  --bg:#0b1220; --bg2:#0c1426;
  --text:#f7faff; --muted:#e5e7eb;
  --panel:rgba(255,255,255,.05); --stroke:rgba(255,255,255,.18);
  --ring:#2d8bff; --ring2:#3aa6ff; --glow:rgba(58,166,255,.28);
  --ok:#2bd38a; --warn:#ffb020;
}
.stApp{background:linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%); color:var(--text); font-size:16.5px;}
.container{max-width:1180px; margin:0 auto; padding:24px 16px;}

/* Typography scale improvements */
h1.title{font-size:2.2rem; font-weight:700; margin:0 0 16px; line-height:1.2;}
h2.section-heading {font-size:1.4rem; font-weight:600; margin:32px 0 24px; color:var(--text); 
    border-bottom:1px solid rgba(255,255,255,.18); padding-bottom:8px;}
p.lead{color:var(--muted); font-size:1.05rem; margin:0 0 32px; line-height:1.5;}

/* Step navigation */
.steps-row{display:flex; gap:12px; align-items:center; flex-wrap:wrap;
  background:linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.04));
  border:1px solid var(--stroke); padding:16px; border-radius:16px; margin:0 0 32px;
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.05);}
.step-item{display:flex; flex-direction:column; align-items:center; gap:8px; min-width:110px}
.chip{display:flex; align-items:center; gap:8px; height:38px; padding:0 14px; border-radius:999px;
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14);}
.chip .num{width:28px; height:28px; display:grid; place-items:center; border-radius:999px;
  background:rgba(255,255,255,.18); color:#eef3ff; font-size:.9rem; font-weight:600;}
.step-item .label{font-weight:600; font-size:1rem; color:#eaf2ff}
.step-item.active .chip{background:linear-gradient(45deg,var(--ring),var(--ring2)); border:none; 
  box-shadow:0 4px 12px rgba(45,139,255,0.3);}
.step-item.done .chip{background:linear-gradient(45deg,#1fbf74,#33e68b); border:none;}
.step-item.done .label{color:#c8ffe3}

/* Card grid system */
.grid{display:grid; grid-template-columns:repeat(12,1fr); gap:24px; margin:20px 0 24px;}
.card-col{grid-column:span 4;}
@media (max-width: 992px) {
  .card-col{grid-column:span 6;}
}
@media (max-width: 768px) {
  .card-col{grid-column:span 12;}
}

/* Solution cards */
.sol-card{
  background:var(--panel); border:1px solid var(--stroke); border-radius:16px; padding:24px;
  transition:all .2s ease; position:relative; height:220px; display:flex; flex-direction:column;
  cursor:pointer; overflow:hidden;
}
.sol-card:hover{transform:translateY(-2px); box-shadow:0 12px 30px rgba(45,139,255,0.15); 
  border-color:var(--ring); background:rgba(255,255,255,.07);}
.sol-card:focus-visible{outline:2px solid var(--ring); outline-offset:2px;}
.sol-card.selected{
  background:linear-gradient(180deg, rgba(45,139,255,.15), rgba(58,166,255,.08));
  border-color:var(--ring); box-shadow:0 0 0 2px rgba(45,139,255,.4);
}
.sol-card h3{margin:0 0 12px; font-size:1.3rem; font-weight:600; line-height:1.3;}
.sol-card p{margin:0 0 20px; color:var(--muted); font-size:1.0rem; line-height:1.5; flex-grow:1;}

/* Card actions & selection */
.card-actions{
  display:flex; justify-content:space-between; align-items:center; margin-top:auto;
}
.select-btn{
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.18);
  color:var(--text); padding:8px 16px; border-radius:8px; font-weight:500;
  cursor:pointer; transition:all .15s ease; display:inline-flex; align-items:center;
}
.select-btn:hover{
  background:rgba(45,139,255,.15); border-color:rgba(45,139,255,.4);
}
.select-btn.selected{
  background:linear-gradient(45deg,var(--ring),var(--ring2));
  border:none; color:white; padding:9px 17px;
}
.select-btn.selected:before{
  content:"✓ "; margin-right:4px;
}

/* Card badges & states */
.card-badge{
  position:absolute; bottom:24px; left:24px;
  font-size:.75rem; font-weight:600; padding:6px 12px; border-radius:6px;
  background:rgba(255,255,255,.14); color:var(--muted);
}
.badge-early{
  background:rgba(45,139,255,.2); color:#b1d4ff;
}
.badge-beta{
  background:rgba(255,176,32,.2); color:#ffe0b1;
}
.badge-new{
  background:rgba(43,211,138,.2); color:#b1ffdf;
}

.sol-card.coming-soon{
  opacity:.8; position:relative; overflow:hidden;
  background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
}
.sol-card.coming-soon:before {
  content:""; position:absolute; top:0; right:0; bottom:0; left:0;
  background:rgba(0,0,0,0.2); z-index:1;
}
.sol-card.coming-soon .card-badge{
  background:rgba(45,139,255,.25); color:#b1d4ff; z-index:2;
}

/* Create card */
.create-card{
  background:linear-gradient(180deg, rgba(45,139,255,.08), rgba(45,139,255,.04));
  border:1.5px dashed rgba(58,166,255,.4); text-align:center; justify-content:center; align-items:center;
  height:220px;
}
.create-card:hover{
  background:linear-gradient(180deg, rgba(45,139,255,.1), rgba(45,139,255,.06));
  border-style:solid;
}
.create-card {
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.create-card h3{color:var(--ring2); margin:0 0 8px;}
.create-card p{color:var(--muted); margin:0 0 16px; max-width:80%;}

.sel-pill{display:inline-flex; align-items:center; gap:8px; margin:8px 0 0; padding:6px 10px; border-radius:999px;
  font-weight:750; color:#eafff6; background:linear-gradient(180deg, rgba(43,211,138,.22), rgba(43,211,138,.10));
  border:1px solid rgba(43,211,138,.4);}

.navbar{display:flex; justify-content:space-between; align-items:center; margin-top:18px}
.btn-secondary .stButton>button{
  border-radius:12px!important; padding:.7rem 1.05rem!important; font-weight:900!important;
  background:linear-gradient(180deg, rgba(255,255,255,.12), rgba(255,255,255,.06))!important;
  color:var(--text)!important; border:1px solid rgba(255,255,255,.22)!important;}
.btn-primary .stButton>button{
  border-radius:14px!important; padding:.85rem 1.25rem!important; font-weight:900!important;
  background:linear-gradient(45deg,var(--ring),var(--ring2))!important; color:#fff!important; border:none!important;
  box-shadow:0 10px 26px var(--glow);}

.top-cta{display:flex; gap:12px; justify-content:flex-end; align-items:center; margin:6px 0 8px;}
.top-cta .ghost{border-radius:12px; border:1px dashed rgba(58,166,255,.5); background:rgba(58,166,255,.08);
  padding:.6rem 1rem; font-weight:800;}
.top-cta .ghost:hover{border-style:solid; background:rgba(58,166,255,.12)}

.create-card{display:grid; place-items:center; text-align:center; border:1.5px dashed rgba(58,166,255,.45);
  background:linear-gradient(180deg, rgba(58,166,255,.08), rgba(58,166,255,.03));}
.create-card .plus{font-size:2.2rem; line-height:1; margin:2px 0 4px;}
.create-card h3{color:rgba(58,166,255,.95); margin:0 0 6px}
.create-card p{color:rgba(255,255,255,.78); margin:0}

.wizard{background:rgba(255,255,255,.04); border-radius:16px; border:1px solid rgba(255,255,255,.12);
  padding:16px 16px 10px; margin:10px 0 0;}
.w-steps{display:flex; gap:8px; flex-wrap:wrap; margin:0 0 8px}
.w-step{padding:6px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.18);
  background:rgba(255,255,255,.06); font-weight:800; font-size:.9rem}
.w-step.on{background:linear-gradient(45deg,var(--ring),var(--ring2)); border:none}
.w-body{padding:10px 2px 4px}

.training-note{font-size:.95rem; color:var(--muted)}
.note-ok{color:#aaffd9}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)

# ---------- Page title & step indicator ----------
area_done = bool(st.session_state.get("selected_area"))
domain_done = bool(st.session_state.get("selected_domain"))

st.markdown("""
<style>
@media (max-width: 768px) {
  .steps-row {
    overflow-x: auto;
    padding-bottom: 8px;
    justify-content: flex-start;
  }
  .step-item {
    min-width: 100px;
  }
  h1.title {
    font-size: 1.8rem;
  }
  p.lead {
    font-size: 0.95rem;
  }
  .container {
    padding: 16px 12px;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='margin:10px 0 4px'>Solution Selection</h2>", unsafe_allow_html=True)
st.markdown(f"""
<div class="steps-row">
  <div class="step-item {'done' if area_done else ''}">
    <div class="chip"><div class="num">1</div>🗺️</div><div class="label">Area</div>
  </div>
  <div class="step-item {'done' if domain_done else ''}">
    <div class="chip"><div class="num">2</div>🏭</div><div class="label">Domain</div>
  </div>
  <div class="step-item active">
    <div class="chip"><div class="num">3</div>🛍️</div><div class="label">Solution</div>
  </div>
  <div class="step-item"><div class="chip"><div class="num">4</div>📷</div><div class="label">Cameras</div></div>
  <div class="step-item"><div class="chip"><div class="num">5</div>🧠</div><div class="label">Tasks</div></div>
  <div class="step-item"><div class="chip"><div class="num">6</div>📊</div><div class="label">Monitor</div></div>
  <div class="step-item"><div class="chip"><div class="num">7</div>📈</div><div class="label">Dashboard</div></div>
</div>
""", unsafe_allow_html=True)

# Top CTA section removed as requested

st.markdown("""
<style>
.page-header {
    margin: 32px 0 40px;
}
.page-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 16px;
    background: linear-gradient(90deg, #f7faff, #dbe6ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    line-height: 1.2;
}
.page-header p {
    font-size: 1.1rem;
    color: var(--muted);
    max-width: 700px;
    line-height: 1.6;
}
</style>
<div class="page-header">
    <h1>🛍️ Solution Selection</h1>
    <p>Choose from our pre-built computer vision solutions or create your own custom solution using our base models. Each solution is tailored for specific business needs and can be deployed immediately.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Session bootstrap ----------
if "selected_solution" not in st.session_state:
    st.session_state.selected_solution = None
if "selected_solutions" not in st.session_state:
    st.session_state.selected_solutions = []
if "show_create_form" not in st.session_state:
    st.session_state.show_create_form = False
if "training_progress" not in st.session_state:
    st.session_state.training_progress = 0
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "custom_solutions" not in st.session_state:
    st.session_state.custom_solutions = []  # list of dicts: {title, desc}
if "custom_solution_name" not in st.session_state:
    st.session_state.custom_solution_name = "My Section Analytics"
if "custom_desc" not in st.session_state:
    st.session_state.custom_desc = "Custom fine-tuned solution"

# ---------- Card renderer ----------
def solution_card(key_prefix, title, desc, available=True, selected_title=None, is_create_card=False, badge=None):
    if is_create_card:
        # Create card widget using Streamlit components instead of HTML
        with st.container():
            st.markdown("### 🔧 Create New Solution")
            st.markdown("Start with a base model and train with your custom data")
            if st.button("Configure", key=f"{key_prefix}_start", use_container_width=True):
                st.session_state.show_create_form = True
                st.session_state.training_progress = 0
                st.session_state.training_complete = False
        return False

    if available:
        is_selected = (selected_title == title)
        
        # Add emoji based on the solution type
        emoji = "📊"  # Default emoji
        if "footfall" in key_prefix.lower():
            emoji = "👥"
        elif "vision" in key_prefix.lower():
            emoji = "🔍"
        elif "behavior" in key_prefix.lower():
            emoji = "🧠"
        elif "product" in key_prefix.lower():
            emoji = "🛒"
        elif "service" in key_prefix.lower():
            emoji = "🛎️"
        elif "custom" in key_prefix.lower():
            emoji = "✨"
            
        # Create a container with the solution info
        with st.container():
            st.markdown(f"### {emoji} {title}")
            st.markdown(desc)
            
            # Status indicator removed as requested
            # Selection is still tracked in the system but not displayed
            
            # Add badge if provided
            if badge:
                if badge.lower() == "early access":
                    st.info(f"🔍 {badge}")
                elif badge.lower() == "beta":
                    st.warning(f"🧪 {badge}")
                elif badge.lower() == "new":
                    st.success(f"🆕 {badge}")
            
            # Button to select/deselect
            button_label = "Deselect" if is_selected else "Select"
            clicked = st.button(button_label, key=f"{key_prefix}_button", use_container_width=True)
            
            if clicked:
                if is_selected:
                    st.session_state.selected_solution = None
                else:
                    st.session_state.selected_solution = title
                st.rerun()
                
        return is_selected
    else:
        badge_text = badge if badge else "Coming Soon"
        
        # Add emoji based on the solution type
        emoji = "📊"  # Default emoji
        if "footfall" in key_prefix.lower():
            emoji = "👥"
        elif "vision" in key_prefix.lower():
            emoji = "🔍"
        elif "behavior" in key_prefix.lower():
            emoji = "🧠"
        elif "product" in key_prefix.lower():
            emoji = "🛒"
        elif "service" in key_prefix.lower():
            emoji = "🛎️"
        
        # Create a container with disabled solution info
        with st.container():
            # Apply a gray style for coming soon items
            st.markdown(f"### {emoji} {title}")
            st.markdown(desc)
            
            # Display appropriate badge
            if badge_text.lower() == "early access":
                st.info(f"🔍 {badge_text}")
            elif badge_text.lower() == "beta":
                st.warning(f"🧪 {badge_text}")
            else:
                st.caption(f"⏳ {badge_text}")
                
        return False

# ---------- Section 1: Available Solutions ----------
st.markdown('<h2 class="section-heading">📦 Available Solutions</h2>', unsafe_allow_html=True)
st.markdown('<div class="grid">', unsafe_allow_html=True)

# Get available solutions from the registry
available_solutions = list_available_solutions()

# Filter solutions based on selected domain
selected_domain = st.session_state.get("selected_domain")
filtered_solutions = []

if selected_domain == "pilgrim":
    # Only show crowd monitoring solution for Pilgrim Safety domain
    filtered_solutions = [s for s in available_solutions if "crowd" in s["id"].lower()]
else:
    # Otherwise show all solutions
    filtered_solutions = available_solutions

# Create a grid of solution cards (3 columns)
cols = st.columns(3)
for i, solution in enumerate(filtered_solutions):
    with cols[i % 3]:
        solution_card(
            f"sol_{solution['id']}",
            solution["name"],
            solution["description"],
            available=True,
            selected_title=st.session_state.get("selected_solution"),
            badge=solution.get("badge")
        )

# Fill remaining columns if needed
remaining_cols = len(filtered_solutions) % 3
if remaining_cols > 0:
    for i in range(3 - remaining_cols):
        with cols[remaining_cols + i]:
            st.write("")  # Empty space

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Section 1B: Coming Soon Solutions ----------
coming_soon_solutions = list_coming_soon_solutions()
if coming_soon_solutions and selected_domain != "pilgrim":  # Hide regular coming soon for Pilgrim domain
    st.markdown('<h3 class="section-heading">⏳ Coming Soon</h3>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    
    # Create a grid for coming soon solutions
    cols = st.columns(3)
    for i, solution in enumerate(coming_soon_solutions):
        with cols[i % 3]:
            solution_card(
                f"sol_{solution['id']}",
                solution["name"],
                solution["description"],
                available=False,
                badge=solution.get("badge") or "Coming Soon"
            )

# Custom Pilgrim domain coming soon solutions
if selected_domain == "pilgrim":
    st.markdown('<h3 class="section-heading">⏳ Coming Soon</h3>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    
    # Define Pilgrim-specific coming soon solutions
    pilgrim_solutions = [
        {
            "id": "emergency_detection",
            "name": "🚑 Emergency Incident Detection",
            "description": "Automatically detect crowd surges, collapses, or abnormal movements in real time",
            "badge": "Coming Soon"
        },
        {
            "id": "smart_traffic",
            "name": "🚦 Smart Traffic Monitoring",
            "description": "Track vehicle congestion and suggest alternate routes for smoother pilgrim movement",
            "badge": "Coming Soon"
        },
        {
            "id": "mobile_health",
            "name": "🧑‍⚕️ Mobile Health & Sanitation",
            "description": "Enable live tracking of health units, hygiene monitoring, and sanitation alerts",
            "badge": "Coming Soon"
        },
        {
            "id": "digital_experience",
            "name": "📱 Digital Pilgrim Experience",
            "description": "Provide real-time updates, AR/VR cultural tours, and lost & found support",
            "badge": "Coming Soon"
        }
    ]
    
    # Create a grid for pilgrim-specific coming soon solutions
    cols = st.columns(2)
    for i, solution in enumerate(pilgrim_solutions):
        with cols[i % 2]:
            # Use custom card style for pilgrim solutions with improved styling
            st.markdown(f"""
            <div style="background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); 
                        border-radius: 16px; padding: 20px; margin-bottom: 15px; transition: all 0.3s ease; 
                        position: relative; min-height: 120px; cursor: default;">
                <div style="position: absolute; top: 12px; right: 12px; 
                          background: linear-gradient(135deg, rgba(43,211,138,0.9), rgba(31,191,116,0.8));
                          color: white; padding: 4px 10px; font-size: 0.7rem; font-weight: 600;
                          border-radius: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                          text-transform: uppercase; letter-spacing: 0.5px;">Coming Soon</div>
                <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 8px;">{solution["name"]}</div>
                <div style="font-size: 0.95rem; opacity: 0.9; line-height: 1.4;">{solution["description"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
    # Close the grid div for pilgrim solutions
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Section 2: Create Your Own Solution (hide for Pilgrim domain) ----------
if selected_domain != "pilgrim":  # Only show for non-Pilgrim domains
    st.markdown("## 🚀 Create Your Own Solution")

    # Create New Solution card
    create_col1, create_col2, create_col3 = st.columns([1, 1, 1])
    with create_col1:
        solution_card(
            "sol_create_card",
            "", "", available=False, selected_title=None, is_create_card=True
        )

# Any custom solutions created in session are displayed here (hide for Pilgrim domain)
if st.session_state.custom_solutions and selected_domain != "pilgrim":
    # Reset custom solutions to remove duplicates if any
    unique_solutions = {}
    for cs in st.session_state.custom_solutions:
        # Use ID as key if available, otherwise use title
        solution_key = cs.get("id", cs["title"])
        unique_solutions[solution_key] = cs
    st.session_state.custom_solutions = list(unique_solutions.values())
    
    # Create appropriate number of columns based on how many solutions we have
    num_solutions = len(st.session_state.custom_solutions)
    num_cols = min(3, num_solutions)  # Max 3 columns
    
    # Display custom solutions using columns
    if num_cols > 0:
        custom_cols = st.columns(num_cols)
        for idx, cs in enumerate(st.session_state.custom_solutions):
            with custom_cols[idx % num_cols]:
                # Create a card with the user's custom title and description
                solution_card(
                    f"sol_custom_{cs.get('id', idx)}",
                    cs["title"],
                    cs["desc"],
                    available=True,
                    selected_title=st.session_state.selected_solution,
                    badge="Custom"
                )
                # Display creation date if available
                if "created_at" in cs:
                    st.caption(f"Created: {cs['created_at']}")

# ---------- Guided create workflow ----------
if st.session_state.show_create_form:
    st.markdown("### Create New Solution")
    with st.container(border=True):
        # mini step tags
        step = st.session_state.get("create_step", 1)
        
        # Create step indicators
        steps_cols = st.columns(4)
        for i, (col, label) in enumerate(zip(steps_cols, ["Choose base", "Name & describe", "Upload data", "Train"])):
            with col:
                if i+1 == step:
                    st.success(f"{i+1}. {label}")
                else:
                    st.info(f"{i+1}. {label}")
        
        # Add separation
        st.divider()
        if step == 1:
            st.selectbox("Base solution / model", ["Store Footfall & Occupancy (YOLO)"], key="base_solution")
            st.info("We'll start from a robust pre-trained model and adapt it to your data.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Next →", key="to_step2", use_container_width=True):
                    st.session_state.create_step = 2
                    st.rerun()
            with c2:
                if st.button("Cancel", key="cancel_create1", use_container_width=True):
                    st.session_state.show_create_form = False

        elif step == 2:
            # Get default values with fallbacks
            default_name = st.session_state.get("custom_solution_name", "My Section Analytics")
            default_desc = st.session_state.get("custom_desc", "Track customer movement between store sections")
                
            c1, c2 = st.columns(2)
            with c1:
                # Use the text_input widget with fallback values
                st.text_input(
                    "Solution name", 
                    value=default_name,
                    key="custom_solution_name"
                )
            with c2:
                # Use the text_area widget with fallback values
                st.text_area(
                    "Short description", 
                    value=default_desc,
                    key="custom_desc", 
                    height=64
                )
            
            # Don't try to update session state again - the widgets do this automatically
            
            st.caption("Tip: keep it short and outcome-focused for stakeholders.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← Back", key="back_to1", use_container_width=True):
                    st.session_state.create_step = 1
                    st.rerun()
            with c2:
                if st.button("Next →", key="to_step3", use_container_width=True):
                    st.session_state.create_step = 3
                    st.rerun()

        elif step == 3:
            st.file_uploader("Upload a few example videos/images (optional for demo)", accept_multiple_files=True,
                             key="training_data")
            col_a, col_b = st.columns(2)
            with col_a:
                st.number_input("Training epochs", min_value=1, max_value=50, value=10, key="epochs")
            with col_b:
                st.slider("Learning rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f",
                          key="learning_rate")
            st.markdown('<span class="training-note">You can start training with defaults and refine later.</span>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← Back", key="back_to2", use_container_width=True):
                    st.session_state.create_step = 2
                    st.rerun()
            with c2:
                if st.button("Start Training →", key="to_step4", use_container_width=True):
                    st.session_state.create_step = 4
                    st.rerun()

        elif step == 4:
            with st.status("Training in progress…", expanded=True) as status:
                progress = st.progress(0)
                for i in range(0, 101, 2):
                    time.sleep(0.03)
                    progress.progress(i)
                    if i < 25:
                        st.write("• Initializing pipeline…")
                    elif i < 60:
                        st.write("• Adapting base model to your dataset…")
                    elif i < 90:
                        st.write("• Validating metrics & packaging…")
                    else:
                        st.write("• Finalizing artifact…")
                status.update(label="Training complete", state="complete", expanded=False)

            st.success(f'Your solution "{st.session_state.get("custom_solution_name","My Section Analytics")}" is ready. ✅')
            st.markdown('<span class="note-ok">Added to the catalog under "Create Your Own Solution" section.</span>',
                        unsafe_allow_html=True)
            # Access values from session state with fallbacks
            new_title = st.session_state.get("custom_solution_name", "My Section Analytics")
            new_desc = st.session_state.get("custom_desc", "Custom fine-tuned solution")
            
            # Check if this solution already exists
            exists = False
            for existing in st.session_state.custom_solutions:
                if existing["title"] == new_title:
                    exists = True
                    break
                    
            # Only add if it doesn't exist
            if not exists:
                import uuid
                st.session_state.custom_solutions.append({
                    "id": str(uuid.uuid4())[:8],  # Generate a short unique ID
                    "title": new_title,
                    "desc": new_desc,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            c1, c2, c3 = st.columns(3)
            # Access values from session state with fallbacks
            created_title = st.session_state.get("custom_solution_name", "My Section Analytics")
            
            with c1:
                if st.button("View Solution", key="view_solution", use_container_width=True):
                    st.session_state.selected_solution = created_title
                    st.session_state.show_create_form = False
                    # Rerun to refresh the page and show the new solution
                    st.rerun()
            with c2:
                if st.button("Continue → Cameras", key="select_created", use_container_width=True):
                    st.session_state.selected_solution = created_title
                    st.session_state.show_create_form = False
                    try:
                        st.switch_page("pages/4_Cameras.py")
                    except Exception:
                        st.session_state["_next_nav_error"] = True
            with c3:
                if st.button("Create another", key="create_another", use_container_width=True):
                    st.session_state.create_step = 1
                    st.session_state.training_progress = 0
                    st.session_state.training_complete = False

        # End of create workflow section

# ---------- Selection feedback ----------
st.markdown("""
<style>
.selection-banner {
    margin-top: 30px;
    padding: 16px 20px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.selection-banner.warning {
    background: linear-gradient(90deg, rgba(255,176,32,.15), rgba(255,176,32,.05));
    border: 1px solid rgba(255,176,32,.3);
}
.selection-banner.success {
    background: linear-gradient(90deg, rgba(43,211,138,.15), rgba(43,211,138,.05));
    border: 1px solid rgba(43,211,138,.3);
}
.selection-message {
    font-size: 1.05rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
}
.selection-message.warning {
    color: #ffcf80;
}
.selection-message.success {
    color: #b1ffdf;
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.selected_solution:
    st.markdown("""
    <div class="selection-banner warning">
        <div class="selection-message warning">
            ⚠️ Please select a solution to continue
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="selection-banner success">
        <div class="selection-message success">
            ✓ Selected: {st.session_state.selected_solution}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Navigation ----------
st.markdown("""
<style>
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 32px 0 60px;
    gap: 16px;
}
.btn-secondary .stButton>button {
    border-radius: 12px!important;
    padding: 10px 20px!important;
    font-weight: 600!important;
    background: rgba(255,255,255,.08)!important;
    color: var(--text)!important;
    border: 1px solid rgba(255,255,255,.2)!important;
    transition: all .2s ease!important;
}
.btn-secondary .stButton>button:hover {
    background: rgba(255,255,255,.12)!important;
    border-color: rgba(255,255,255,.3)!important;
}
.btn-primary .stButton>button {
    border-radius: 12px!important;
    padding: 10px 24px!important;
    font-weight: 600!important;
    background: linear-gradient(45deg,var(--ring),var(--ring2))!important;
    color: #fff!important;
    border: none!important;
    box-shadow: 0 8px 16px rgba(45,139,255,0.25)!important;
    transition: all .2s ease!important;
}
.btn-primary .stButton>button:hover:not([disabled]) {
    transform: translateY(-2px)!important;
    box-shadow: 0 12px 20px rgba(45,139,255,0.35)!important;
}
.btn-primary .stButton>button:disabled {
    background: linear-gradient(45deg, #6b7c95, #8390a3)!important;
    opacity: 0.7!important;
    box-shadow: none!important;
}

/* Sticky footer when selection is made */
.sticky-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(180deg, rgba(11,18,32,0.8), rgba(12,20,38,0.95));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 1000;
    box-shadow: 0 -4px 16px rgba(0,0,0,0.2);
    border-top: 1px solid rgba(255,255,255,0.1);
}
.selected-solution {
    font-weight: 600;
    color: var(--muted);
}
.solution-name {
    color: var(--text);
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

left, right = st.columns([1,1])
with left:
    st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
    if st.button("← Back"):
        try:
            st.switch_page("pages/2_Domain.py")
        except Exception:
            st.page_link("pages/2_Domain.py", label="Back to Domain", icon="↩️")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="btn-primary" style="display:flex; justify-content:flex-end;">', unsafe_allow_html=True)
    # Use session state to ensure Next button is enabled when a solution is selected
    disabled_state = not bool(st.session_state.selected_solution)
    
    if st.button("Continue to Cameras →", disabled=disabled_state, use_container_width=True):
        try:
            st.switch_page("pages/4_Cameras.py")
        except Exception:
            st.session_state["_next_nav_error"] = True
    
    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.get("_next_nav_error"):
        st.page_link("pages/4_Cameras.py", label="Continue to Cameras →", icon="➡️")

# No sticky footer - removed duplicate CTA
# End of page
