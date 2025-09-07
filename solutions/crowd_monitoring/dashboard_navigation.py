import streamlit as st

def get_dashboard_url(feed_id=None):
    """
    Get the correct URL for the dashboard based on the application structure.
    
    Args:
        feed_id: Optional feed ID to include as a parameter
        
    Returns:
        URL string for the dashboard
    """
    # Try to determine the correct URL format
    dashboard_base = "/7_Dashboard"  # Default format
    
    # Add feed_id as query parameter if provided
    if feed_id:
        dashboard_base += f"?feed_id={feed_id}"
        
    return dashboard_base

def open_dashboard_in_new_tab(feed_id=None):
    """
    Generate JavaScript to open the dashboard in a new tab with fallback options.
    
    Args:
        feed_id: Optional feed ID to include as a parameter
        
    Returns:
        HTML/JavaScript code to include in the page
    """
    # Create a list of possible dashboard URLs to try
    dashboard_urls = [
        f"/7_Dashboard{f'?feed_id={feed_id}' if feed_id else ''}",
        f"7_Dashboard{f'?feed_id={feed_id}' if feed_id else ''}",
        f"/Dashboard{f'?feed_id={feed_id}' if feed_id else ''}",
        f"Dashboard{f'?feed_id={feed_id}' if feed_id else ''}",
    ]
    
    # JavaScript function to try opening each URL
    js_code = f"""
    <script>
    function openDashboardInNewTab() {{
        var urls = {dashboard_urls};
        var opened = false;
        
        // Try each URL until one works
        for (var i = 0; i < urls.length; i++) {{
            try {{
                window.open(urls[i], '_blank');
                opened = true;
                break;
            }} catch (e) {{
                console.log('Failed to open: ' + urls[i]);
            }}
        }}
        
        if (!opened) {{
            alert('Could not open dashboard. Please try the Same Tab button instead.');
        }}
    }}
    </script>
    <button 
        onclick="openDashboardInNewTab()" 
        style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer; width: 100%; font-size: 16px;"
    >
        📊 Open Dashboard in New Tab
    </button>
    """
    
    return js_code
