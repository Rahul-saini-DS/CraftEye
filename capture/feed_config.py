"""Feed configuration persistence module for multi-camera setup.

Handles saving/loading feed configurations to/from JSON files.
Adds a backward-compatible task schema for "footfall" (line-crossing).
"""

import json
import uuid
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

DEFAULT_CONFIG_PATH = "feeds.json"

# -------- Task defaults (backward compatible) --------
DEFAULT_DETECT_TASK: Dict[str, Any] = {
    "type": "detect",
    "model": "models/yolov8n.pt",   # keep lightweight default
    "person_class": 0               # COCO 'person'
}

DEFAULT_FOOTFALL_TASK: Dict[str, Any] = {
    "type": "footfall",
    "model": "models/yolov8n.pt",   # detect model (NOT seg) for line-crossing
    "person_class": 0,              # COCO 'person'
    # Virtual line (x1, y1, x2, y2) in frame coordinates. Defaults to a horizontal mid-line.
    "line": [100, 240, 540, 240],
    # Direction rule used by your counter (you can interpret as you like in Monitor/pipeline)
    # Common options to standardize downstream logic:
    #   "up_is_enter", "down_is_enter", "left_is_enter", "right_is_enter"
    "direction": "up_is_enter",
    # Debounce settings (pixels of centroid travel & time between events)
    "min_travel_px": 12,
    "min_time_between_counts_ms": 500,
    # Logging target for dashboard
    "csv_path": "seconds_and_counts.csv"
}


# ----------------- ID & IO helpers -------------------

def generate_feed_id() -> str:
    """Generate a short unique ID for a new feed."""
    return str(uuid.uuid4())[:8]


def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------- Task sanitation helpers -------------

def _normalize_path(p: Optional[str]) -> Optional[str]:
    """Normalize paths for cross-platform use (avoid backslashes in JSON)."""
    if not p:
        return p
    return p.replace("\\", "/")

def _ensure_detect_defaults(task: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**DEFAULT_DETECT_TASK, **(task or {})}
    merged["model"] = _normalize_path(merged.get("model"))
    return merged

def _ensure_footfall_defaults(task: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**DEFAULT_FOOTFALL_TASK, **(task or {})}
    merged["model"] = _normalize_path(merged.get("model"))
    # Validate line shape
    line = merged.get("line")
    if (not isinstance(line, list)) or len(line) != 4:
        merged["line"] = DEFAULT_FOOTFALL_TASK["line"][:]  # safe fallback
    # Validate direction
    if merged.get("direction") not in {
        "up_is_enter", "down_is_enter", "left_is_enter", "right_is_enter"
    }:
        merged["direction"] = DEFAULT_FOOTFALL_TASK["direction"]
    # Clamp numeric guards
    merged["min_travel_px"] = max(0, int(merged.get("min_travel_px", 12)))
    merged["min_time_between_counts_ms"] = max(0, int(merged.get("min_time_between_counts_ms", 500)))
    # CSV path
    merged["csv_path"] = _normalize_path(merged.get("csv_path") or DEFAULT_FOOTFALL_TASK["csv_path"])
    return merged

def _ensure_task_defaults(task: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a sanitized task dict with safe defaults based on type."""
    if not isinstance(task, dict):
        return _ensure_detect_defaults(DEFAULT_DETECT_TASK)
    t = task.get("type", "detect")
    if t == "footfall":
        return _ensure_footfall_defaults(task)
    # fall back to detect for unknown types
    return _ensure_detect_defaults(task)


# ----------------- Public CRUD functions -------------

def load_feeds(config_path: str = DEFAULT_CONFIG_PATH) -> List[Dict[str, Any]]:
    """Load feed configurations from JSON file.

    Returns an empty list if file is missing/invalid.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return []
    try:
        feeds = _read_json(config_file)
        # Backward-compat: ensure each feed has a well-formed task
        for f in feeds:
            f["task"] = _ensure_task_defaults(f.get("task"))
            # Normalize common fields
            if "source" in f and isinstance(f["source"], str):
                f["source"] = f["source"].strip()
            if "type" not in f:
                f["type"] = "webcam"
            if "resolution" not in f:
                f["resolution"] = [640, 480]
            if "fps_cap" not in f:
                f["fps_cap"] = 15
        return feeds
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Warning: Failed to load feeds from {config_path}")
        return []

def save_feeds(feeds: List[Dict[str, Any]], config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """Save feed configurations to JSON file."""
    try:
        # Sanitize before writing
        safe = []
        for f in feeds:
            ff = dict(f)
            ff["task"] = _ensure_task_defaults(ff.get("task"))
            # normalize model path for JSON
            ff["task"]["model"] = _normalize_path(ff["task"].get("model"))
            safe.append(ff)
        _write_json(Path(config_path), safe)
        return True
    except Exception as e:
        print(f"Error saving feeds: {e}")
        return False

def add_feed(feed_config: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Add a new feed to the configuration file."""
    feeds = load_feeds(config_path)
    # Generate ID if not provided
    if "id" not in feed_config:
        feed_config["id"] = generate_feed_id()
    # Ensure task schema
    feed_config["task"] = _ensure_task_defaults(feed_config.get("task"))
    feeds.append(feed_config)
    save_feeds(feeds, config_path)
    return feed_config["id"]

def update_feed(feed_id: str, feed_config: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """Update an existing feed in the configuration file."""
    feeds = load_feeds(config_path)
    for i, feed in enumerate(feeds):
        if feed.get("id") == feed_id:
            updated = {**feed, **feed_config}
            updated["task"] = _ensure_task_defaults(updated.get("task"))
            feeds[i] = updated
            save_feeds(feeds, config_path)
            return True
    return False

def remove_feed(feed_id: str, config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """Remove a feed from the configuration file."""
    feeds = load_feeds(config_path)
    for i, feed in enumerate(feeds):
        if feed.get("id") == feed_id:
            del feeds[i]
            save_feeds(feeds, config_path)
            return True
    return False


# -------------- CameraManager conversion --------------

def convert_to_camera_manager_config(feed_json: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a feed JSON to CameraManager config format.

    Passes through a sanitized 'task' dict so downstream (Monitor/pipeline)
    can initialize FootfallCounter per feed without hardcoding.
    """
    task = _ensure_task_defaults(feed_json.get("task"))
    return {
        "id": feed_json.get("id", generate_feed_id()),
        "source": feed_json.get("source", "0"),
        "type": feed_json.get("type", "webcam"),
        "resolution": feed_json.get("resolution", [640, 480]),
        "fps_cap": feed_json.get("fps_cap", 15),
        "task": task,
    }


# ---------------- Convenience helpers ----------------

def make_footfall_task(
    line: Tuple[int, int, int, int],
    direction: str = "up_is_enter",
    csv_path: str = "seconds_and_counts.csv",
    model: str = "models/yolov8n.pt",
    person_class: int = 0,
    min_travel_px: int = 12,
    min_time_between_counts_ms: int = 500,
) -> Dict[str, Any]:
    """Create a valid 'footfall' task dict (helps your UI code)."""
    return _ensure_footfall_defaults({
        "type": "footfall",
        "model": model,
        "person_class": int(person_class),
        "line": list(line),
        "direction": direction,
        "min_travel_px": int(min_travel_px),
        "min_time_between_counts_ms": int(min_time_between_counts_ms),
        "csv_path": csv_path,
    })


def set_feed_footfall(
    feed_id: str,
    line: Tuple[int, int, int, int],
    direction: str = "up_is_enter",
    csv_path: str = "seconds_and_counts.csv",
    model: str = "models/yolov8n.pt",
    person_class: int = 0,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> bool:
    """Update a feed to use the 'footfall' task with given parameters."""
    feeds = load_feeds(config_path)
    for i, f in enumerate(feeds):
        if f.get("id") == feed_id:
            f["task"] = make_footfall_task(
                line=line,
                direction=direction,
                csv_path=csv_path,
                model=model,
                person_class=person_class,
            )
            return save_feeds(feeds, config_path)
    return False
