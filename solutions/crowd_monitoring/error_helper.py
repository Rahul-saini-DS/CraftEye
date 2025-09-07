"""Error handling helper module."""

import streamlit as st
import traceback

def user_friendly_error_message(ui, e):
    """
    Show user-friendly error message instead of raw error text.
    
    Args:
        ui: UI dictionary with 'processing_status' element
        e: The exception that was caught
    """
    error_msg = str(e)
    
    # Check for common UI component errors
    if "'trend_chart'" in error_msg or "'zone_metrics'" in error_msg or "'alert_log'" in error_msg:
        ui['processing_status'].warning("⚠️ Analytics display needs refresh. Please reload the page.")
    else:
        ui['processing_status'].error("⚠️ Display update issue. Please try refreshing the page.")
    
    # Show technical details only in debug mode
    if st.session_state.get("debug_mode", False):
        ui['processing_status'].error(f"Technical details: {error_msg}\n{traceback.format_exc()}")
