import os
import sys
import traceback
import streamlit as st

try:
    # Ultralytics >=8
    from ultralytics import YOLO
except Exception as _e:
    YOLO = None


def _cache_key(model_path: str, device: str, half: bool) -> str:
    """
    Deterministic cache key to avoid collisions between CPU/GPU or FP32/FP16 loads.
    """
    return f"{os.path.abspath(model_path)}|{str(device).lower()}|{'fp16' if half else 'fp32'}"


def _move_to_device(model, device: str):
    """
    Move the underlying model to the given device, handling both Ultralytics wrappers and raw torch modules.
    """
    # Preferred API on Ultralytics models
    try:
        if hasattr(model, "to"):
            model.to(device)  # Ultralytics wrapper forwards to the internal torch model
            return
    except Exception:
        pass

    # Fallback: try underlying torch nn.Module
    try:
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)  # type: ignore[attr-defined]
            return
    except Exception:
        pass


def _set_half_precision(model, device: str, half: bool):
    """
    Enable half precision if requested and supported (CUDA only).
    """
    if not half:
        return
    if not str(device).lower().startswith("cuda"):
        # Half precision on CPU is either unsupported or slower; skip silently.
        return

    # Try wrapper first
    try:
        if hasattr(model, "model") and hasattr(model.model, "half"):
            model.model.half()  # type: ignore[attr-defined]
            return
    except Exception:
        pass

    # Some builds expose .half() at the top level
    try:
        if hasattr(model, "half"):
            model.half()  # type: ignore[attr-defined]
            return
    except Exception:
        pass


def _warmup(model, imgsz: int = 640):
    """
    Optional warm-up to compile kernels / allocate buffers, reducing first-frame latency.
    """
    try:
        import torch
        import numpy as np

        # Create a dummy input that matches expected input size
        dummy = (torch.zeros(1, 3, imgsz, imgsz) if str(next(model.parameters()).device).startswith("cuda")
                 else torch.zeros(1, 3, imgsz, imgsz))
        # If Ultralytics model supports .predict, invoke once
        if hasattr(model, "predict"):
            _ = model.predict(dummy, verbose=False)
        else:
            # Fallback to directly calling the module (rare)
            _ = model(dummy)
    except Exception:
        # Warm-up is best-effort; swallow errors
        pass


@st.cache_resource(show_spinner=False, persist=False)
def _load_cached_model_internal(cache_key: str, model_path: str, device: str, half: bool):
    """
    Internal cached loader. The first parameter (cache_key) ensures Streamlit
    separates cache entries by (model_path, device, half).
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics is not available. Please install with `pip install ultralytics`.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    model = YOLO(model_path)

    # Move to device
    _move_to_device(model, device)

    # Set FP16 if appropriate
    _set_half_precision(model, device, half)

    # Light warm-up (won’t run again for the same cache entry)
    try:
        _warmup(model, imgsz=640)
    except Exception:
        pass

    return model


def load_cached_model(model_path: str, device: str = "cpu", half: bool = False):
    """
    Load and cache a YOLO model for reuse.

    Args:
        model_path: Path to the YOLO model (e.g., 'models/yolo11n.pt').
        device: Device to load the model on ("cpu", "cuda", "cuda:0", etc.).
        half: Use half precision (FP16) when on CUDA.

    Returns:
        Ultralytics YOLO model instance, or None if loading failed.
    """
    try:
        key = _cache_key(model_path, device, half)
        return _load_cached_model_internal(key, model_path, device, half)
    except FileNotFoundError as e:
        st.error(str(e))
    except RuntimeError as e:
        st.error(str(e))
    except Exception as e:
        # Log full traceback to the terminal for debugging, show concise msg in UI
        traceback.print_exc(file=sys.stderr)
        st.error(f"Error loading model '{model_path}' on '{device}': {e}")
    return None

