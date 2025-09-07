"""
Cross-tab navigation helper for the CraftEye platform.
Helps with navigating between Monitor and Dashboard pages with proper URL params.
"""
import streamlit as st
import time
import os
from typing import Optional, Dict, Any, List

def get_base_url() -> str:
    """Try to determine base URL for navigation"""
    # Check if we're in development or deployment mode
    is_dev = os.environ.get("STREAMLIT_ENV", "").lower() == "development"
    
    if is_dev:
        return ""  # In development, relative paths work best
    else:
        # In production, depends on server configuration
        # Try to get from query parameters or config
        return st.session_state.get("base_url", "")

def navigate_to_dashboard(feed_id: Optional[str] = None) -> bool:
    """
    Navigate to the dashboard page using multiple fallback methods.
    Returns True if navigation was successful using primary method.
    """
    navigation_success = False
    fid = feed_id or st.session_state.get("current_feed_id")
    
    # Store feed ID in session state for cross-tab access
    if fid:
        st.session_state["current_feed_id"] = fid
    
    # Try multiple navigation methods in order of preference
    
    # Method 1: Modern Streamlit st.switch_page with query params
    try:
        if fid:
            st.switch_page("pages/7_Dashboard.py?feed_id=" + fid)
        else:
            st.switch_page("pages/7_Dashboard.py")
        navigation_success = True
        return navigation_success
    except Exception:
        pass
    
    # Method 2: Try alternative st.switch_page formats
    try:
        if fid:
            st.switch_page("/pages/7_Dashboard.py?feed_id=" + fid)
        else:
            st.switch_page("/pages/7_Dashboard.py")
        navigation_success = True
        return navigation_success
    except Exception:
        pass
    
    # Method 3: Direct dashboard.py in main directory (using our new file)
    try:
        if fid:
            st.switch_page("Dashboard.py?feed_id=" + fid)
        else:
            st.switch_page("Dashboard.py")
        navigation_success = True
        return navigation_success
    except Exception:
        pass
        
    # Method 4: Newer Streamlit versions with page_link
    try:
        if fid:
            st.page_link("7_Dashboard.py", query_params={"feed_id": fid}, use_container_width=True)
        else:
            st.page_link("7_Dashboard.py", use_container_width=True)
        navigation_success = True
        return navigation_success
    except Exception:
        pass
    
    # Method 5: Last resort - HTML refresh with multiple URL formats
    urls_to_try = [
        f"/7_Dashboard{f'?feed_id={fid}' if fid else ''}",
        f"./7_Dashboard{f'?feed_id={fid}' if fid else ''}",
        f"pages/7_Dashboard.py{f'?feed_id={fid}' if fid else ''}",
        f"Dashboard.py{f'?feed_id={fid}' if fid else ''}"
    ]
    
    for url in urls_to_try:
        st.markdown(f"""
            <meta http-equiv="refresh" content="0;URL='{url}'">
        """, unsafe_allow_html=True)
    
    return navigation_success

def navigate_to_monitor(feed_id: Optional[str] = None) -> bool:
    """
    Navigate to the monitor page using multiple fallback methods.
    Returns True if navigation was successful using primary method.
    """
    navigation_success = False
    fid = feed_id or st.session_state.get("current_feed_id")
    
    # Store feed ID in session state for cross-tab access
    if fid:
        st.session_state["current_feed_id"] = fid
    
    # Try multiple navigation methods in order of preference
    
    # Method 1: Modern Streamlit st.switch_page with query params
    try:
        if fid:
            st.switch_page("pages/6_Monitor.py?feed_id=" + fid)
        else:
            st.switch_page("pages/6_Monitor.py")
        navigation_success = True
        return navigation_success
    except Exception:
        pass
    
    # Method 2: Try alternative st.switch_page formats
    try:
        if fid:
            st.switch_page("/pages/6_Monitor.py?feed_id=" + fid)
        else:
            st.switch_page("/pages/6_Monitor.py")
        navigation_success = True
        return navigation_success
    except Exception:
        pass
    
    # Method 3: Newer Streamlit versions with page_link
    try:
        if fid:
            st.page_link("6_Monitor.py", query_params={"feed_id": fid}, use_container_width=True)
        else:
            st.page_link("6_Monitor.py", use_container_width=True)
        navigation_success = True
        return navigation_success
    except Exception:
        pass
    
    # Method 4: Last resort - HTML refresh with multiple URL formats
    urls_to_try = [
        f"/6_Monitor{f'?feed_id={fid}' if fid else ''}",
        f"./6_Monitor{f'?feed_id={fid}' if fid else ''}",
        f"pages/6_Monitor.py{f'?feed_id={fid}' if fid else ''}"
    ]
    
    for url in urls_to_try:
        st.markdown(f"""
            <meta http-equiv="refresh" content="0;URL='{url}'">
        """, unsafe_allow_html=True)
    
    return navigation_success
