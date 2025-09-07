import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st


# ---------------------------
# Paths & setup
# ---------------------------

DATA_DIR = os.path.join("data", "crowd")


def ensure_data_dir() -> None:
    """
    Ensure the crowd data directory exists.
    """
    os.makedirs(DATA_DIR, exist_ok=True)


def _sanitize_feed_id(feed_id: str) -> str:
    """
    Sanitize a feed_id so it is safe to use as a filename.
    Keeps alphanumerics and replaces everything else with '_'.
    """
    return "".join(c if c.isalnum() else "_" for c in str(feed_id))


def get_csv_path(feed_id: str) -> str:
    """
    Get the path to the CSV file for a specific feed.

    Args:
        feed_id: Camera feed ID

    Returns:
        Path to the CSV file
    """
    safe_feed_id = _sanitize_feed_id(feed_id)
    return os.path.join(DATA_DIR, f"{safe_feed_id}.csv")


# ---------------------------
# Write helpers
# ---------------------------

def write_crowd_data_row(feed_id: str, count: int, density: float, density_level: str) -> None:
    """
    Append a row of crowd data to the feed's CSV file.

    Args:
        feed_id: Camera feed ID
        count: Number of people detected
        density: Crowd density value (people per m²)
        density_level: Categorical density level (Low, Moderate, High, Very High, etc.)
    """
    ensure_data_dir()
    csv_path = get_csv_path(feed_id)

    # Detect if we need to write a header
    file_exists = os.path.exists(csv_path)
    timestamp = datetime.utcnow().isoformat()

    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "count", "density", "density_level", "feed_id"])
            writer.writerow([timestamp, int(count), float(density), str(density_level), str(feed_id)])
    except Exception as e:
        # Don't crash the app; surface a visible error once
        st.error(f"Failed to write crowd data for feed '{feed_id}': {e}")


# ---------------------------
# Read & maintenance (optional helpers)
# ---------------------------

def read_recent_rows(feed_id: str, max_rows: int = 1000) -> List[Dict[str, Any]]:
    """
    Read up to `max_rows` most recent rows for a feed.
    Useful for dashboard charts. Not used by callers today, but safe to import.

    Returns:
        List of dict rows (possibly empty on error/missing file).
    """
    csv_path = get_csv_path(feed_id)
    if not os.path.exists(csv_path):
        return []

    rows: List[Dict[str, Any]] = []
    try:
        # Efficient tail-read: read all but slice only last max_rows
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            rows = all_rows[-max_rows:] if len(all_rows) > max_rows else all_rows
    except Exception as e:
        st.error(f"Failed to read crowd data for feed '{feed_id}': {e}")
        return []

    return rows


def prune_csv_rows(feed_id: str, max_rows: int = 200_000) -> Optional[int]:
    """
    Keep only the last `max_rows` rows in the feed CSV to limit disk usage.
    Returns the number of rows kept (or None on error).
    """
    csv_path = get_csv_path(feed_id)
    if not os.path.exists(csv_path):
        return 0

    try:
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        if not reader:
            return 0

        header, body = reader[0], reader[1:]
        if len(body) <= max_rows:
            return len(body)

        trimmed = body[-max_rows:]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(trimmed)
        return len(trimmed)
    except Exception as e:
        st.error(f"Failed to prune CSV for feed '{feed_id}': {e}")
        return None


# ---------------------------
# Density classification
# ---------------------------

def classify_density_level(density: float) -> str:
    """
    Classify density into a categorical level using thresholds in session state
    (if available) or reasonable defaults (persons per square meter).

    Args:
        density: Crowd density value (people per square meter)

    Returns:
        Categorical density level string
    """
    # Prefer thresholds configured in session state
    if "crowd_monitoring" in st.session_state:
        cm = st.session_state.crowd_monitoring
        minimal = cm["density_thresholds"].get("minimal", 0.1)
        low = cm["density_thresholds"].get("low", 0.2)
        moderate = cm["density_thresholds"].get("moderate", 0.5)
        high = cm["density_thresholds"].get("high", 0.8)
        very_high = cm["density_thresholds"].get("very_high", 1.2)
        extreme = cm["density_thresholds"].get("extreme", 2.0)
    else:
        # Defaults (aligned with your monitor_bridge initialization)
        minimal = 0.1     # very sparse
        low = 0.2         # comfortable
        moderate = 0.5    # getting busy
        high = 0.8        # crowded
        very_high = 1.2   # dense
        extreme = 2.0     # very dense (beyond this = critical)

    if density < minimal:
        return "Minimal"
    elif density < low:
        return "Low"
    elif density < moderate:
        return "Moderate"
    elif density < high:
        return "High"
    elif density < very_high:
        return "Very High"
    elif density < extreme:
        return "Extreme"
    else:
        return "Critical"
