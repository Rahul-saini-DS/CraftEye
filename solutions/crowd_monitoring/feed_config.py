"""
feed_config.py
Persistence helpers for camera feed configurations and per-feed task settings.

This module is intentionally light on dependencies and safe on Windows.
It writes JSON atomically and validates/migrates task_settings so that
Crowd Monitoring pages (Tasks / Monitor / Dashboard) read consistent data.

Public API expected by the rest of the app:
- save_feeds(feeds: list[dict]) -> None
- load_feeds() -> list[dict]
- get_feeds_file() -> str
"""

from __future__ import annotations
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

# -------- Paths --------

def _project_root() -> str:
    # Put data/ one level above this file's directory by default (…/capture/ -> project root)
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))

_DATA_DIR = os.environ.get("CRAFT_EYE_DATA_DIR", os.path.join(_project_root(), "data"))
_FEEDS_FILE = os.path.join(_DATA_DIR, "feeds.json")
_BACKUP_FILE = os.path.join(_DATA_DIR, "feeds.backup.json")

def get_feeds_file() -> str:
    """Return absolute path to the feeds.json file."""
    return _FEEDS_FILE

def _ensure_dirs() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)

# -------- Schema helpers --------

def _default_task_settings() -> Dict[str, Any]:
    return {
        "location_label": "",
        "install_coords": {},          # {"lat": float, "lon": float}
        "intended_tasks": [],          # ["density","flow"]
        "calibration": {},             # {"homography_matrix": [...], "area_m2": float}
        "density": {
            "enabled": True,
            "roi": [],
            "conf": 0.35,
            "heatmap_grid": 32,
            "agg_window_sec": 10,
            "person_class": 0
        },
        "flow": {
            "enabled": True,
            "lines": [((100, 240), (540, 240))],
            "direction": "up_is_enter",
            "debounce_ms": 500,
            "min_track_len": 5
        }
    }

def _migrate_task_settings(ts: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Bring an existing task_settings dict up to date with expected keys.
    Never removes custom keys; only fills missing defaults.
    """
    base = _default_task_settings()
    ts = ts or {}

    # Top-level simple fields
    for k in ("location_label", "install_coords", "intended_tasks", "calibration"):
        if k not in ts:
            ts[k] = base[k]

    # density block
    dens_in = ts.get("density", {})
    dens_def = base["density"].copy()
    dens_def.update(dens_in or {})
    ts["density"] = dens_def

    # flow block
    flow_in = ts.get("flow", {})
    flow_def = base["flow"].copy()
    flow_def.update(flow_in or {})
    ts["flow"] = flow_def

    return ts

def _migrate_feed(feed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure minimal fields exist on each feed dict and migrate task_settings.
    """
    out = dict(feed)  # shallow copy
    out.setdefault("id", feed.get("id"))
    out.setdefault("source", feed.get("source"))
    out.setdefault("type", feed.get("type", "video"))
    out.setdefault("resolution", feed.get("resolution", [640, 480]))
    out.setdefault("fps_cap", feed.get("fps_cap", 30))
    out.setdefault("name", feed.get("name", f"Camera {out.get('id', 'unknown')}"))
    out.setdefault("model_settings", feed.get("model_settings", {}))
    out["task_settings"] = _migrate_task_settings(feed.get("task_settings"))
    return out

# -------- IO helpers --------

def _atomic_write_json(path: str, data: Any) -> None:
    """
    Write JSON atomically: write to temp file, fsync, then replace.
    Creates a .backup.json copy of any existing file before replacing.
    """
    _ensure_dirs()
    dirname = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix="feeds_", suffix=".json", dir=dirname)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        # Create/refresh a backup of the last good file (best-effort)
        if os.path.exists(path):
            try:
                shutil.copy2(path, _BACKUP_FILE)
            except Exception:
                pass
        # Replace atomically
        os.replace(tmp, path)
    except Exception:
        # Clean temp if something goes wrong
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

def _safe_read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------- Public API --------

def save_feeds(feeds: List[Dict[str, Any]]) -> None:
    """
    Persist the list of feed configs to data/feeds.json.

    Each feed dict may include:
      - id (str)
      - source (str)
      - type (str)
      - resolution (list[int,int])
      - fps_cap (int)
      - name (str)
      - model_settings (dict)
      - task_settings (dict)
        - location_label, install_coords, intended_tasks, calibration
        - density {enabled, roi, conf, heatmap_grid, agg_window_sec, person_class}
        - flow {enabled, lines, direction, debounce_ms, min_track_len}

    Unknown keys are preserved.
    """
    if not isinstance(feeds, list):
        raise TypeError("save_feeds expects a list of feed dicts")

    # Migrate/validate
    migrated: List[Dict[str, Any]] = []
    for feed in feeds:
        if not isinstance(feed, dict):
            continue
        migrated.append(_migrate_feed(feed))

    # Write atomically
    _atomic_write_json(_FEEDS_FILE, migrated)

def load_feeds() -> List[Dict[str, Any]]:
    """
    Load feeds from data/feeds.json (or empty list if missing).
    Returns migrated/validated data so callers always get current schema.
    """
    raw = _safe_read_json(_FEEDS_FILE)
    if not raw:
        return []
    if not isinstance(raw, list):
        # If corrupted, try backup; else return empty
        backup = _safe_read_json(_BACKUP_FILE)
        if isinstance(backup, list):
            return [_migrate_feed(f) for f in backup]
        return []
    return [_migrate_feed(f) for f in raw]

# -------- Optional: convenience for session hydration (not required) --------

def to_feeds_meta(feeds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert persisted feeds list into a feeds_meta-like mapping:
    { feed_id: { name, model_settings, task_settings } }
    Does not touch Streamlit session; caller decides.
    """
    meta: Dict[str, Dict[str, Any]] = {}
    for f in feeds:
        fid = f.get("id")
        if not fid:
            continue
        meta[fid] = {
            "name": f.get("name", f"Camera {fid}"),
            "model_settings": f.get("model_settings", {}),
            "task_settings": _migrate_task_settings(f.get("task_settings")),
        }
    return meta

