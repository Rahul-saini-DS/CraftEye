import os
import json
import time
from typing import List, Dict, Any
import streamlit as st

# Path for zone analytics JSON files
ZONE_DATA_DIR = os.path.join("data", "zone_analytics")

def ensure_zone_data_dir() -> None:
    """Ensure the zone data directory exists."""
    os.makedirs(ZONE_DATA_DIR, exist_ok=True)

def get_zone_json_path(feed_id: str) -> str:
    """
    Get path to the zone analytics JSON file for a specific feed.
    
    Args:
        feed_id: Camera feed ID
        
    Returns:
        Path to the JSON file
    """
    # Sanitize feed_id to ensure it's a valid filename
    safe_feed_id = "".join(c if c.isalnum() else "_" for c in str(feed_id))
    return os.path.join(ZONE_DATA_DIR, f"{safe_feed_id}_zones.json")

def write_zone_data(feed_id: str, zone_counts: List[int], zone_densities: List[float]) -> None:
    """
    Write zone analytics data to a JSON file.
    
    Args:
        feed_id: Camera feed ID
        zone_counts: List of people counts per zone
        zone_densities: List of density values per zone
    """
    ensure_zone_data_dir()
    json_path = get_zone_json_path(feed_id)
    
    # Prepare data structure
    zone_data = {
        "timestamp": time.time(),
        "feed_id": feed_id,
        "names": ["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
        "counts": zone_counts,
        "densities": zone_densities,
        "highest_zone": zone_counts.index(max(zone_counts)) if max(zone_counts) > 0 else 0
    }
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(zone_data, f)
    except Exception as e:
        st.error(f"Failed to write zone data for feed '{feed_id}': {e}")

def read_zone_data(feed_id: str) -> Dict[str, Any]:
    """
    Read zone analytics data for a feed.
    
    Args:
        feed_id: Camera feed ID
        
    Returns:
        Dictionary with zone data or empty dict if not found
    """
    json_path = get_zone_json_path(feed_id)
    if not os.path.exists(json_path):
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to read zone data for feed '{feed_id}': {e}")
        return {}
