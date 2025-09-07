import os, base64
import streamlit as st

def _data_uri_from_files(candidates):
    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                kind = "png" if p.lower().endswith(".png") else "jpeg"
                b64 = base64.b64encode(f.read()).decode()
                return f"data:image/{kind};base64,{b64}"
    return None

def render_static_header():
    st.markdown(
        """
        <style>
          header[data-testid="stHeader"] {
            background-color: #0b1220 !important;
            border-bottom: 1px solid rgba(180,205,255,.12) !important;
          }
          .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_brand_header(
    product_name: str = "CraftEye",
    company_name: str = "Craftifai",
    tagline: str = "Deploy Today, Evolve Tomorrow",
    product_logo_candidates=None,
    company_logo_candidates=None,
    company_link: str | None = None,
):
    if product_logo_candidates is None:
        product_logo_candidates = [
            "assets/CraftEye LOGO.png",
            "assets/project-logo.png",
            "assets/CraftEye_logo.png",
            "CraftEye LOGO.png",
        ]
    if company_logo_candidates is None:
        company_logo_candidates = [
            r"C:\\Users\\acer\\Desktop\\new1\\assets\\craftifai_logo.jpg",
            "assets/craftifai_logo.jpg",
            "assets/craftifai_logo.png",
        ]

    ce_logo = _data_uri_from_files(product_logo_candidates)
    craftifai_logo = _data_uri_from_files(company_logo_candidates)

    st.markdown(
        """
        <style>
          :root{
            --bg-deep:#0b1422;
            --bg-mid:#0e1b2e;
            --chip-blue-1:#173a63;
            --chip-blue-2:#1e4c82;
            --chip-rim:#4da3ff;
            --text-strong:#f4f8ff;
            --text-soft:#cfe1ff;
          }

          /* --- Banner (matched to app bg, light top vignette) --- */
          .fixed-brand-banner {
            position: sticky; top: 0; z-index: 1000; width: 100%;
            background:
              radial-gradient(120% 140% at 50% -50%, rgba(77,163,255,.10), transparent 60%),
              linear-gradient(180deg, var(--bg-mid), var(--bg-deep));
            border-bottom: 1px solid rgba(180,205,255,.12);
            -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px);
          }
          .fixed-brand-banner.scrolled { box-shadow: 0 6px 20px rgba(0,0,0,.35); }

          .brand-inner {
            max-width: 1280px; margin: 0 auto;
            display: grid; grid-template-columns: auto 1fr auto;
            align-items: center; gap: 22px;
            padding: 18px 24px;
          }

          /* --- Logo chip (kept square) with high-contrast inner plate --- */
          .logo-chip {
            position: relative; display: inline-flex; align-items: center; justify-content: center;
            height: 72px; width: 72px; border-radius: 16px;
            background: linear-gradient(180deg, var(--chip-blue-2), var(--chip-blue-1));
            border: 1px solid rgba(77,163,255,.35);
            box-shadow:
              0 10px 22px rgba(3,12,24,.55),
              0 0 0 4px rgba(77,163,255,.10);
            transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
          }
          /* light inner plate to lift dark logos */
          .logo-chip .plate{
            display:flex; align-items:center; justify-content:center;
            height: 58px; width: 58px; border-radius: 12px;
            background: linear-gradient(180deg, #f7fbff, #e9f1ff);
            box-shadow:
              inset 0 0 0 1px rgba(13,30,56,.08),
              0 1px 0 rgba(255,255,255,.65);
          }
          .logo-chip:hover {
            transform: translateY(-1px);
            box-shadow:
              0 14px 28px rgba(3,12,24,.65),
              0 0 0 5px rgba(77,163,255,.14);
            filter: saturate(1.04);
          }
          .logo-chip img { height: 44px; width: auto; display:block; }

          /* --- Center text with stronger contrast --- */
          .brand-center { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }
          .brand-hero-title {
            margin: 0; color: var(--text-strong);
            font-weight: 900; font-size: 2.0rem; letter-spacing:.25px; line-height: 1.1;
            text-shadow: 0 0 6px rgba(77,163,255,.18), 0 1px 0 rgba(0,0,0,.35);
          }

          /* --- Tagline: higher contrast, softer gradient --- */
          .brand-hero-pill {
            display:inline-flex; align-items:center; gap:10px;
            padding: 7px 18px; border-radius: 999px;
            font-size: 1.08rem; font-weight: 750; color: var(--text-strong);
            background:
              linear-gradient(180deg, rgba(88,156,255,.35), rgba(44,104,173,.35)) padding-box;
            border: 1px solid rgba(160,200,255,.35);
            box-shadow:
              0 8px 18px rgba(6,14,26,.45),
              inset 0 1px 0 rgba(255,255,255,.25);
          }

          @media (max-width: 680px) {
            .brand-inner { padding: 12px 14px; gap: 14px; }
            .logo-chip { height: 56px; width: 56px; border-radius: 14px; }
            .logo-chip .plate { height: 46px; width: 46px; border-radius: 10px; }
            .logo-chip img { height: 34px; }
            .brand-hero-title { font-size: 1.35rem; text-shadow:none; }
            .brand-hero-pill { display:none; }
          }

          @media (prefers-color-scheme: light) {
            .fixed-brand-banner {
              background:
                radial-gradient(120% 140% at 50% -50%, rgba(77,163,255,.12), transparent 60%),
                linear-gradient(180deg, #f6f9ff, #eef4ff);
              border-bottom: 1px solid rgba(36,78,125,.18);
            }
            .brand-hero-title { color:#0b1422; text-shadow:none; }
            .brand-hero-pill { color:#0b1422; }
            .logo-chip .plate { background: linear-gradient(180deg,#ffffff,#f3f7ff); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # scroll effect
    st.markdown(
        """
        <script>
          window.addEventListener('scroll', () => {
            const b = document.querySelector('.fixed-brand-banner');
            if (!b) return;
            if (window.scrollY > 2) b.classList.add('scrolled'); else b.classList.remove('scrolled');
          });
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Build logo HTML (with plate for contrast)
    if ce_logo:
        left_logo_html = f"<span class='logo-chip'><span class='plate'><img src='{ce_logo}' alt='{product_name} logo'/></span></span>"
    else:
        left_logo_html = "<span class='logo-chip'><span class='plate'>👁️‍🗨️</span></span>"

    if craftifai_logo:
        inner = f"<span class='plate'><img src='{craftifai_logo}' alt='{company_name} logo'/></span>"
        if company_link:
            right_logo_html = f"<a class='logo-chip' href='{company_link}' target='_blank' rel='noopener noreferrer'>{inner}</a>"
        else:
            right_logo_html = f"<span class='logo-chip'>{inner}</span>"
    else:
        right_logo_html = f"<span style='opacity:.95;color:var(--text-soft);font-weight:800;letter-spacing:.3px'>{company_name}</span>"

    st.markdown(
        f"""
        <div class="fixed-brand-banner" role="banner" aria-label="{product_name} header">
          <div class="brand-inner">
            {left_logo_html}
            <div class="brand-center">
              <div class="brand-hero-title">{product_name}</div>
              <div class="brand-hero-pill">{tagline}</div>
            </div>
            {right_logo_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
def verify_and_restore_state():
    """
    Verifies critical state variables and attempts to restore them if missing.
    Call this function at the beginning of each page to ensure consistent state.
    """
    # List of expected state keys for proper flow
    critical_keys = ["selected_area", "selected_domain", "selected_solution"]
    
    # Get current page without using script_run_ctx (which is not available)
    import inspect
    import os
    try:
        # Get the caller's filename
        frame = inspect.stack()[1]
        current_path = frame.filename
        current_page = os.path.basename(current_path)
    except:
        current_page = ""
        
    is_dashboard = "Dashboard" in current_page or "7_" in current_page
    is_monitor = "Monitor" in current_page or "6_" in current_page
    is_crowd_solution_page = any([
        "crowd" in current_page.lower(),
        "crowd_monitoring" in str(st.session_state.get("feeds_meta", {})).lower(),
        "pilgrim" in str(st.session_state.get("feeds_meta", {})).lower()
    ])
    
    # Auto-restore for dashboard and monitor pages
    if "selected_solution" not in st.session_state and (is_dashboard or is_monitor):
        if is_crowd_solution_page or "model_settings" in st.session_state:
            # We can confidently restore the Pilgrim Crowd Monitoring solution
            st.session_state["selected_solution"] = "Pilgrim Crowd Monitoring"
            
            # Also restore area/domain if needed
            if "selected_area" not in st.session_state:
                st.session_state["selected_area"] = "Pilgrim Site"
            if "selected_domain" not in st.session_state:
                st.session_state["selected_domain"] = "Crowd Management"
    
    # General fallback restoration for any page
    elif not all(k in st.session_state for k in critical_keys) and any(k in st.session_state for k in critical_keys):
        # Try to restore missing state from available information
        if "selected_solution" not in st.session_state:
            # Try to restore from related state
            if "selected_domain" in st.session_state and "selected_area" in st.session_state:
                domain = st.session_state["selected_domain"]
                if domain in ["Crowd Management", "Event Management"]:
                    st.session_state["selected_solution"] = "Pilgrim Crowd Monitoring"
    
    # Initialize monitoring flag if not present
    if "monitoring_active" not in st.session_state:
        st.session_state["monitoring_active"] = False
        
    # Initialize current_feed_id if missing but needed
    if "current_feed_id" not in st.session_state and "feeds_meta" in st.session_state:
        feeds = st.session_state.get("feeds_meta", {})
        if feeds:
            # Set the first feed as default
            st.session_state["current_feed_id"] = next(iter(feeds))
            
def ensure_cross_tab_data_persistence():
    """
    Helps ensure data persistence between tabs by writing key session state 
    to browser local storage and saving important metrics to CSV files.
    This helps when opening the dashboard in a new tab.
    """
    # Check if we have the enhanced data sharing module available
    try:
        from solutions.crowd_monitoring.data_sharing import ensure_dashboard_data_sharing
        
        # If we have metrics and a feed ID, write them to the shared data file
        if "metrics" in st.session_state and "current_feed_id" in st.session_state:
            feed_id = st.session_state.get("current_feed_id")
            metrics = st.session_state.get("metrics", {})
            ensure_dashboard_data_sharing(feed_id, metrics)
    except ImportError:
        # Fall back to basic persistence if module not available
        pass
    
    # List of keys we want to persist between tabs
    persistent_keys = ["monitoring_active", "current_feed_id", "metrics", "selected_solution", 
                       "selected_area", "selected_domain"]
    
    # Generate JavaScript to store these values
    js_code = """
    <script>
    // Store key metrics in localStorage when this page loads
    document.addEventListener('DOMContentLoaded', function() {
    """
    
    # Add code to store each key
    for key in persistent_keys:
        if key in st.session_state:
            # Convert the value to JSON string for storage
            value = str(st.session_state.get(key, ""))
            js_code += f"""
            try {{
                localStorage.setItem('crafteye_{key}', '{value}');
            }} catch (e) {{
                console.log('Error storing {key}:', e);
            }}
            """
    
    # Special case for Pilgrim Crowd Monitoring solution
    if st.session_state.get("selected_solution") == "Pilgrim Crowd Monitoring":
        js_code += """
        localStorage.setItem('crafteye_has_pilgrim_solution', 'true');
        """
    
    # Add code to retrieve values if they're missing
    js_code += """
    // Add listener to restore values when Streamlit connects
    window.parent.addEventListener('streamlit:render', function(event) {
        // On dashboard page, ensure we have solution state
        const path = window.location.pathname;
        if (path.includes('Dashboard') || path.includes('7_')) {
            if (localStorage.getItem('crafteye_has_pilgrim_solution') === 'true') {
                // We can add buttons or send messages to restore state
                console.log("Dashboard detected - solution state can be restored from localStorage");
            }
        }
    });
    });
    </script>
    """
    
    # Add a help message for restoring state when needed
    import inspect
    import os
    try:
        # Get the caller's filename
        frame = inspect.stack()[1]
        current_path = frame.filename
        current_page = os.path.basename(current_path)
    except:
        current_page = ""
        
    if ("Dashboard" in current_page or "7_" in current_page or "Monitor" in current_page or "6_" in current_page):
        if not st.session_state.get("selected_solution"):
            js_code += """
            <script>
            // If we're on Dashboard/Monitor without solution state, add helper
            if (localStorage.getItem('crafteye_has_pilgrim_solution') === 'true') {
                // Add code to inject a helper message/button
                console.log("Missing solution state detected - helper could be shown");
            }
            </script>
            """
    
    st.markdown(js_code, unsafe_allow_html=True)

def render_solution_context():
    """
    Display the selected solution as a context banner across all pages.
    This creates consistent visibility of which solution the user is working with.
    Also ensures state consistency across page navigation.
    """
    # First verify and restore state if needed
    verify_and_restore_state()
    
    # Extra check for the monitor and dashboard pages
    import inspect
    import os
    try:
        # Get the caller's filename
        frame = inspect.stack()[1]
        current_path = frame.filename
        current_page = os.path.basename(current_path)
    except:
        current_page = ""
        
    if ("Dashboard" in current_page or "7_" in current_page or 
        "Monitor" in current_page or "6_" in current_page):
        # Special handling - if we've gone directly to the dashboard or monitor,
        # and we're missing solution state but have other state indicators
        if not st.session_state.get("selected_solution") and (
            "feeds_meta" in st.session_state or 
            "model_settings" in st.session_state or
            "monitoring_active" in st.session_state
        ):
            st.session_state["selected_solution"] = "Pilgrim Crowd Monitoring"
            st.session_state["selected_area"] = "Pilgrim Site"  
            st.session_state["selected_domain"] = "Crowd Management"
            
            # Avoid showing warning on monitor/dashboard pages
            if "crowd_monitoring" in current_page:
                pass  # Silently restore
            else:
                st.info("Solution context restored to 'Pilgrim Crowd Monitoring'")
    
    selected_solution = st.session_state.get("selected_solution")
    
    if selected_solution:
        st.markdown(
            f"""
            <div style="background-color: rgba(45, 139, 255, 0.1); 
                        border: 1px solid rgba(45, 139, 255, 0.3);
                        border-radius: 8px;
                        padding: 8px 16px;
                        margin-bottom: 20px;
                        display: flex;
                        align-items: center;">
                <span style="font-size: 18px; margin-right: 10px;">📋</span>
                <div>
                    <div style="font-weight: 600;">Selected Solution</div>
                    <div>{selected_solution}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
