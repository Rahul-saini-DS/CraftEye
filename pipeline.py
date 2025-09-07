"""
Universal YOLO inference pipeline
Works with PyTorch (.pt), ONNX (.onnx), TensorRT (.engine), OpenVINO (.xml directory)
Optimized for both CPU and GPU using Ultralytics best practices

Enhancements:
- Footfall advanced params: gate zone, min travel px, debounce (ms)
- Restrict inference to person class (COCO class 0) for speed/accuracy in footfall mode
- Optional feed_id context passed to FootfallCounter.update for per-feed duration logs
- Optional auto-snap of counting line to full frame width/height (respects UI endpoints unless enabled)
"""
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO


class Inference:
    """
    Universal YOLO inference class using Ultralytics.
    Supports .pt, .onnx, .engine, .torchscript, and OpenVINO IR formats with a unified API.
    Default input size 640x640 (adjustable via imgsz).
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.45,
        device: str = "auto",
        half: bool = False,
        task: Optional[str] = None,
        footfall_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self.task = task

        # Context that can be set by callers (e.g., Monitor) to tag logs
        self.feed_id: Optional[str] = None

        # Keep original cfg + autosnap flag
        self.footfall_cfg: Optional[Dict[str, Any]] = dict(footfall_cfg) if isinstance(footfall_cfg, dict) else None
        self._footfall_autosnapped: bool = False

        # Detect device type and batch capability based on model format
        self.device_type = self._detect_device_type(device)
        self.supports_batching = self._check_batch_support()

        # Performance tracking
        self.inference_times: List[float] = []
        self.total_frames = 0

        # Load the model once during initialization
        self.model = None
        self._load_model()

        # ---- Footfall (optional) ----
        self.footfall_counter = None
        self.tracker_yaml = "bytetrack.yaml"  # default tracker for stable IDs
        try:
            use_footfall = False

            # Defaults; will be overridden if footfall_cfg provided
            line = [0, self.imgsz // 2, self.imgsz, self.imgsz // 2]
            csv_path = "seconds_and_counts.csv"
            direction = "up_is_enter"
            min_gap = 500  # ms debounce between counts for the same track
            enable_gate = True
            gate_width_px = 8
            min_travel_px = 6

            cfg = self.footfall_cfg
            if isinstance(cfg, dict) and cfg.get("type") == "footfall":
                use_footfall = True
                line = cfg.get("line", line)
                csv_path = cfg.get("csv_path", csv_path)
                direction = cfg.get("direction", direction)
                min_gap = int(cfg.get("min_time_between_counts_ms", min_gap))
                enable_gate = bool(cfg.get("enable_gate", enable_gate))
                gate_width_px = int(cfg.get("gate_width_px", gate_width_px))
                min_travel_px = int(cfg.get("min_travel_px", min_travel_px))
                self.tracker_yaml = cfg.get("tracker", self.tracker_yaml)
                # Optional: infer feed_id from the csv filename if present (e.g., ".../<feed_id>.csv")
                try:
                    fname = Path(csv_path).stem
                    if fname and fname != "seconds_and_counts":
                        self.feed_id = fname
                except Exception:
                    pass
            elif (
                "model_settings" in st.session_state
                and "footfall" in st.session_state.model_settings.get("task_types", [])
            ):
                use_footfall = True  # global toggle

            if use_footfall:
                from retail_analytics import FootfallCounter  # your footfall module

                self.footfall_counter = FootfallCounter(
                    line_start=(int(line[0]), int(line[1])),
                    line_end=(int(line[2]), int(line[3])),
                    csv_path=csv_path,
                    direction=direction,
                    min_time_between_counts_ms=min_gap,
                    enable_gate=enable_gate,
                    gate_width_px=gate_width_px,
                    min_travel_px=min_travel_px,
                )
        except Exception as e:
            # FootfallCounter might not be available or initialization could fail
            print(f"⚠️ Footfall initialization skipped: {e}")

    # ---------------- Device & Model Helper Methods ----------------
    def _detect_device_type(self, device: str) -> str:
        """Classify the device string into a type category (cuda, mps, cpu, openvino, etc.)."""
        device_str = str(device).lower()
        if device_str.startswith("intel:"):
            return "openvino"
        elif device_str.startswith("cuda") or device_str.isdigit():
            return "cuda"
        elif device_str == "mps":
            return "mps"
        elif device_str == "cpu":
            return "cpu"
        else:
            return "unknown"

    def _check_batch_support(self) -> bool:
        """Determine if the model format supports batch predictions (True for PyTorch .pt)."""
        model_path_lower = self.model_path.lower()
        if model_path_lower.endswith(".pt"):
            return True  # Ultralytics PyTorch models support batching
        elif model_path_lower.endswith(".onnx"):
            return False
        elif model_path_lower.endswith(".engine"):
            return False
        elif "openvino" in model_path_lower or model_path_lower.endswith(".xml"):
            return False
        else:
            return False

    def _get_model_info(self) -> Dict[str, Any]:
        """Return human-readable info about the model (format, task type, file name, size)."""
        model_path_lower = self.model_path.lower()
        if model_path_lower.endswith(".pt"):
            format_type = "PyTorch"
        elif model_path_lower.endswith(".onnx"):
            format_type = "ONNX"
        elif model_path_lower.endswith(".engine"):
            format_type = "TensorRT"
        elif model_path_lower.endswith(".torchscript"):
            format_type = "TorchScript"
        elif "openvino" in model_path_lower or model_path_lower.endswith(".xml"):
            format_type = "OpenVINO"
        else:
            format_type = "Unknown"

        # Infer task type from filename if possible
        filename = Path(self.model_path).stem.lower()
        if "seg" in filename:
            task_type = "segmentation"
        elif "pose" in filename:
            task_type = "pose estimation"
        elif "cls" in filename or "classify" in filename:
            task_type = "classification"
        else:
            task_type = "object detection"

        return {
            "format": format_type,
            "task": task_type,
            "filename": Path(self.model_path).name,
            "size_mb": round(Path(self.model_path).stat().st_size / (1024 * 1024), 1)
            if Path(self.model_path).exists()
            else 0.0,
        }

    def _load_model(self):
        """Load the YOLO model using Ultralytics, placing it on the appropriate device."""
        try:
            print(f"⏳ Loading model: {self.model_path}")
            if self.task:
                # If a specific task (e.g., "segment", "classify") is forced
                self.model = YOLO(self.model_path, task=self.task)
                print(f"🎯 Using explicit task mode: {self.task}")
            else:
                # YOLO will auto-detect if this is detect/segment/pose based on model weights
                self.model = YOLO(self.model_path)
                print("🔍 Auto-detecting model task type...")
            model_info = self._get_model_info()
            print("✅ Model loaded successfully:")
            print(f"   📁 Path: {self.model_path}")
            print(f"   🏷️ Format: {model_info['format']}")
            print(f"   🎯 Task: {model_info['task']}")
            print(f"   📏 Input size: {self.imgsz}x{self.imgsz}")

            # Resolve 'auto' device to actual available hardware
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            
            # Only enable half-precision for CUDA devices
            if self.device_type == "cuda" and torch.cuda.is_available():
                self.half = True
            else:
                # Disable half for CPU, MPS, OpenVINO or other backends
                self.half = False
                
            # Move model to device if applicable (only PyTorch .pt; others handle internally)
            if self.device_type == "openvino":
                print(f"⚡ Using OpenVINO device: {self.device}")
                # Ultralytics handles OV runtime internally
            elif self.device_type == "cuda":
                if torch.cuda.is_available():
                    device_id = (
                        0
                        if str(self.device) == "cuda"
                        else int(str(self.device).split(":")[1])
                        if ":" in str(self.device)
                        else int(self.device)
                    )
                    gpu_name = torch.cuda.get_device_name(device_id)
                    mem_gb = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                    print(f"🚀 Using CUDA GPU {device_id}: {gpu_name} ({mem_gb:.1f} GB)")
                    if not self.model_path.lower().endswith((".onnx", ".engine")):
                        # For Ultralytics, model.model is torch.nn.Module; .to() on YOLO wrapper is not universal
                        try:
                            self.model.model.to(device_id)
                        except Exception:
                            pass
                else:
                    print("⚠️ CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
                    self.device_type = "cpu"
            elif self.device_type == "mps":
                print("🍎 Using Apple MPS (Metal Performance Shaders) device")
            else:
                print(f"🔧 Using device: {self.device}")
                self.device = "cpu"
                self.device_type = "cpu"

            # Warm up the model with a few dummy inferences for consistent latency
            self._warmup()
            print(f"✅ Model ready on device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _warmup(self, num_warmup: int = 3):
        """Run a few dummy predictions to warm up the model (improve first-run latency)."""
        print("🔥 Warming up model...")
        
        # For larger image sizes (>=1024), do fewer warmup runs to save time
        if self.imgsz >= 1024:
            num_warmup = 1
            print(f"   Using single warmup for large imgsz ({self.imgsz})")
            
        # Check if already warmed up
        if getattr(self, "_warmed", False):
            print("   Model already warmed up, skipping")
            return
            
        dummy_frame = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for i in range(num_warmup):
            try:
                _ = self.model.predict(
                    dummy_frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                    show=False,
                    save=False,
                )
                print(f"   Warmup {i+1}/{num_warmup} OK")
            except Exception as e:
                print(f"   Warmup {i+1}/{num_warmup} failed: {e}")
                
        # Mark as warmed
        self._warmed = True
        
        # Reset performance stats after warmup
        self.inference_times.clear()
        self.total_frames = 0
        print("✅ Warmup complete")

    # ---------- Line autosnap (respects UI unless auto_full_width True) ----------
    def _autosnap_line_to_frame(self, frame_rgb: np.ndarray):
        """
        Adjust the configured counting line ONCE to match the first frame's dimensions.
        Behaviour:
          - If `auto_full_width` is True, snap to a full-width (horizontal) or full-height (vertical) line.
          - Otherwise, **preserve the user-specified endpoints** and simply clamp/optionally scale to fit.
        """
        if not self.footfall_counter:
            return
        try:
            h, w = frame_rgb.shape[:2]
        except Exception:
            return

        # Resolve configuration and preferences
        cfg = self.footfall_cfg if isinstance(self.footfall_cfg, dict) else {}
        auto_full = bool(cfg.get("auto_full_width", False))
        bottom_off = int(cfg.get("bottom_offset_px", 0))
        base_wh = cfg.get("base_resolution") or cfg.get("base_wh")

        # Prefer the original config line; fallback to current counter line
        cfg_line = None
        if isinstance(cfg, dict):
            cfg_line = cfg.get("line")
        if cfg_line is None:
            try:
                ax, ay = self.footfall_counter.a
                bx, by = self.footfall_counter.b
                cfg_line = [ax, ay, bx, by]
            except Exception:
                return

        x1, y1, x2, y2 = [int(v) for v in cfg_line]
        horizontal_guess = abs(y2 - y1) <= abs(x2 - x1)

        if auto_full:
            # Full-width (horizontal) or full-height (vertical) snap
            if horizontal_guess:
                y = max(0, min(h - 1, (h - 1 - bottom_off) if bottom_off > 0 else y1))
                self.footfall_counter.set_line((0, y), (w - 1, y))
            else:
                x = max(0, min(w - 1, x1))
                self.footfall_counter.set_line((x, 0), (x, h - 1))
            return

        # Otherwise: preserve endpoints; optionally scale from a base resolution if supplied
        if isinstance(base_wh, (list, tuple)) and len(base_wh) == 2 and base_wh[0] and base_wh[1]:
            sx = float(w) / float(base_wh[0])
            sy = float(h) / float(base_wh[1])
            x1 = int(round(x1 * sx)); y1 = int(round(y1 * sy))
            x2 = int(round(x2 * sx)); y2 = int(round(y2 * sy))

        # Clamp to frame bounds
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        self.footfall_counter.set_line((x1, y1), (x2, y2))

    # ---------------- Inference APIs ----------------
    def predict_batch(
        self,
        frames_rgb: List[np.ndarray],
        warmup: bool = False,
        stream: bool = False,
        save: bool = False,
        project: str = "runs/predict",
        name: str = "exp",
    ) -> List[Any]:
        """
        Run inference on a batch of frames (RGB format). Returns a list of YOLO Results (one per frame).
        If footfall tracking is enabled, uses `.track()` per frame to get consistent IDs.
        """
        if not frames_rgb:
            return []

        # Auto-snap the counting line to this frame’s dimensions (do this only once)
        if self.footfall_counter and not getattr(self, "_footfall_autosnapped", False):
            try:
                self._autosnap_line_to_frame(frames_rgb[0])
            finally:
                self._footfall_autosnapped = True

        start_time = time.time()
        use_tracking = self.footfall_counter is not None

        results_list: List[Any] = []

        # Get person class from config (default 0)
        person_class = 0
        if isinstance(self.footfall_cfg, dict):
            person_class = int(self.footfall_cfg.get("person_class", 0))
        
        # Get class filtering from config or use person class only for footfall
        classes_filter = None
        if isinstance(self.footfall_cfg, dict) and "classes" in self.footfall_cfg:
            classes_filter = self.footfall_cfg.get("classes")
            if not classes_filter:  # Empty list means "all classes"
                classes_filter = None
        elif use_tracking:  # Default for footfall: use person class only
            classes_filter = [person_class]
            
        # Tracker config
        tracker_cfg = self.tracker_yaml
        if isinstance(self.footfall_cfg, dict) and "tracker" in self.footfall_cfg:
            tracker_cfg = self.footfall_cfg.get("tracker") or self.tracker_yaml
        
        # (1) Tracking path for footfall (persist IDs); restrict to person class for speed/accuracy
        if use_tracking:
            for i, frame in enumerate(frames_rgb):
                try:
                    res = self.model.track(
                        frame,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        half=self.half,
                        tracker=tracker_cfg,
                        persist=True,   # keep track of IDs across frames
                        classes=classes_filter,
                        verbose=False,
                        show=False,
                    )
                    # Ultralytics track returns a list
                    results_list.append(res[0] if isinstance(res, list) else res)
                except Exception as e:
                    print(f"⚠️ Tracking failed for frame {i}: {e}")
                    results_list.append(None)
        else:
            # (2) No tracking: use batch predict if supported
            if self.supports_batching and len(frames_rgb) > 1:
                try:
                    results = self.model.predict(
                        frames_rgb,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        half=self.half,
                        classes=classes_filter,
                        verbose=False,
                        stream=stream,
                        show=False,
                        save=save,
                        project=project,
                        name=name,
                        exist_ok=True,
                    )
                    if stream:
                        return results
                    results_list = results if isinstance(results, list) else [results]
                except Exception as batch_err:
                    if not warmup:
                        # Only print the warning once per Inference instance
                        if not getattr(self, "_batch_fallback_warned", False):
                            print(f"⚠️ Batch inference failed: {batch_err}. Falling back to frame-by-frame processing.")
                            self._batch_fallback_warned = True
            if not results_list:  # Fallback to processing frames individually
                for i, frame in enumerate(frames_rgb):
                    try:
                        res = self.model.predict(
                            frame,
                            imgsz=self.imgsz,
                            conf=self.conf,
                            iou=self.iou,
                            device=self.device,
                            half=self.half,
                            classes=classes_filter,  # Use configured classes
                            verbose=False,
                            stream=False,
                            show=False,
                            save=False,
                        )
                        results_list.append(res[0] if isinstance(res, list) else res)
                    except Exception as e:
                        print(f"⚠️ Frame {i} processing failed: {e}")
                        results_list.append(None)

        # (3) Post-inference: if footfall counter is active, update counts and (optionally) overlay line
        if self.footfall_counter is not None:
            for idx, res in enumerate(results_list):
                if res is None or not hasattr(res, "boxes"):
                    continue
                detections = []
                try:
                    for box in res.boxes:
                        # Only consider person class for footfall counting
                        cls_id = int(box.cls[0].item()) if hasattr(box, "cls") else 0
                        # Get person class from config (default 0) 
                        person_class = 0
                        if isinstance(self.footfall_cfg, dict):
                            person_class = int(self.footfall_cfg.get("person_class", 0))
                        if cls_id != person_class:
                            continue
                        xyxy = box.xyxy[0].detach().cpu().numpy()  # [x1,y1,x2,y2]
                        conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.0
                        track_id = None
                        if hasattr(box, "id") and box.id is not None:
                            tid = box.id
                            try:
                                track_id = int(tid[0].item()) if hasattr(tid, "__len__") else int(tid.item())
                            except Exception:
                                track_id = None
                        detections.append({"bbox": xyxy, "track_id": track_id, "confidence": conf})

                    # Update object count metric if Streamlit session is present
                    try:
                        if 'metrics' in st.session_state:
                            st.session_state.metrics['objects_detected'] = len(detections)
                    except Exception:
                        pass

                    # Update footfall with optional feed context
                    footfall_metrics = self.footfall_counter.update(
                        detections, feed_id=getattr(self, "feed_id", None)
                    )
                    if isinstance(footfall_metrics, dict) and "metrics" in st.session_state:
                        st.session_state.metrics.update(
                            {
                                "people_entered": footfall_metrics.get("entered", 0),
                                "people_exited": footfall_metrics.get("exited", 0),
                                "current_occupancy": footfall_metrics.get("occupancy", 0),
                            }
                        )

                    # Throttled timeline logging to CSV (if FootfallCounter exposes it)
                    try:
                        if hasattr(self.footfall_counter, "log_if_needed"):
                            self.footfall_counter.log_if_needed(feed_id=getattr(self, "feed_id", None))
                    except Exception:
                        pass

                    # If only one frame and draw function exists, overlay the counting line on the visualization
                    if len(frames_rgb) == 1 and hasattr(self.footfall_counter, "draw_counter_line"):
                        try:
                            frames_rgb[0] = self.footfall_counter.draw_counter_line(frames_rgb[0])
                        except Exception:
                            pass
                except Exception as proc_err:
                    print(f"⚠️ Footfall processing error: {proc_err}")

        # (4) Timing and performance metrics
        if self.device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure all GPU ops finished
        if not warmup:
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
            self.total_frames += len(frames_rgb)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)  # keep sliding window of recent timings

        return results_list

    def predict_single(self, frame_rgb: np.ndarray) -> Any:
        """
        Convenience method to run prediction on a single image (RGB).
        Returns a single YOLO Result or None.
        """
        if frame_rgb is None:
            print("⚠️ Received None frame in predict_single")
            return None
        if frame_rgb.size == 0 or frame_rgb.shape[0] == 0 or frame_rgb.shape[1] == 0:
            print(f"⚠️ Received invalid frame shape: {frame_rgb.shape}")
            return None
        try:
            results = self.predict_batch([frame_rgb])
            return results[0] if results else None
        except Exception as e:
            print(f"⚠️ Error in predict_single: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, float]:
        """Return average inference time (ms), average FPS, and total frames processed."""
        if not self.inference_times:
            return {
                "avg_inference_time_ms": 0.0,
                "avg_fps": 0.0,
                "total_frames": self.total_frames,
                "total_batches": 0,
                "last_inference_time_ms": 0.0,
            }
        total_time = sum(self.inference_times)
        avg_time = total_time / len(self.inference_times)
        avg_fps = (self.total_frames / total_time) if total_time > 0 else 0.0
        last_time = self.inference_times[-1]
        return {
                "avg_inference_time_ms": avg_time * 1000.0,
                "avg_fps": avg_fps,
                "total_frames": self.total_frames,
                "total_batches": len(self.inference_times),
                "last_inference_time_ms": last_time * 1000.0,
        }

    def get_last_inference_time_ms(self) -> float:
        """Return the inference time (ms) for the most recent prediction batch."""
        return (self.inference_times[-1] * 1000.0) if self.inference_times else 0.0

    def update_thresholds(self, conf: Optional[float] = None, iou: Optional[float] = None):
        """Dynamically update the confidence or IoU thresholds for inference."""
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        print(f"📊 Updated thresholds: confidence={self.conf}, iou={self.iou}")

    def verify_model_usage(self) -> Dict[str, Any]:
        """Return a dictionary with the current model/device configuration for debugging."""
        info = self._get_model_info()
        verification = {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "model_info": info,
            "device": self.device,
            "device_type": self.device_type,
            "supports_batching": self.supports_batching,
            "input_size": f"{self.imgsz}x{self.imgsz}",
            "confidence_threshold": self.conf,
            "iou_threshold": self.iou,
            "half_precision": self.half,
        }
        print("🔍 Model Verification:")
        print(f"   ✅ Model: {info['filename']} ({info['format']})")
        print(f"   🎯 Task: {info['task']}")
        print(f"   🔧 Device: {self.device} ({self.device_type})")
        print(f"   📏 Input size: {self.imgsz}x{self.imgsz}")
        print(f"   ⚙️ Batch support: {self.supports_batching}")
        return verification

    def set_feed_id(self, feed_id: Optional[str]):
        """Optional helper to tag detections/logs with a feed identifier."""
        self.feed_id = str(feed_id) if feed_id is not None else None

    def cleanup(self):
        """Free the loaded model and GPU memory (if any)."""
        if self.model is not None:
            del self.model
            self.model = None
        if "cuda" in str(self.device):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()
        print("🧹 Inference pipeline cleaned up")


def discover_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """
    Scan the models directory for supported model files and return a list of model info dicts.
    Supports .pt, .onnx, .engine, .torchscript, and OpenVINO (dir with .xml & .bin).
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    models = []
    supported_exts = [".pt", ".onnx", ".engine", ".torchscript"]
    # Direct model files
    for model_file in models_path.iterdir():
        if model_file.suffix.lower() in supported_exts:
            name = model_file.stem
            if "seg" in name.lower():
                task = "segmentation"
            elif "pose" in name.lower():
                task = "pose"
            elif "cls" in name.lower() or "classify" in name.lower():
                task = "classification"
            else:
                task = "detection"
            models.append(
                {
                    "name": model_file.stem,
                    "path": str(model_file),
                    "format": model_file.suffix[1:],
                    "task": task,
                    "size": model_file.stat().st_size,
                }
            )
    # OpenVINO IR models (stored in subdirectories with .xml and .bin files)
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            xml_files = list(model_dir.glob("*.xml"))
            bin_files = list(model_dir.glob("*.bin"))
            if xml_files and bin_files:
                name = model_dir.name
                if "seg" in name.lower():
                    task = "segmentation"
                elif "pose" in name.lower():
                    task = "pose"
                elif "cls" in name.lower() or "classify" in name.lower():
                    task = "classification"
                else:
                    task = "detection"
                total_size = sum(f.stat().st_size for f in (xml_files + bin_files))
                models.append(
                    {
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "format": "openvino",
                        "task": task,
                        "size": total_size,
                    }
                )
    return sorted(models, key=lambda m: m["name"].lower())


def get_device_config(prefer_gpu: bool = True) -> Dict[str, Any]:
    """
    Choose the optimal device for inference based on availability.
    Returns a dict with 'device', 'half', 'imgsz', etc.
    """
    if prefer_gpu and torch.cuda.is_available():
        # Use first CUDA GPU
        device_id = 0
        device_name = torch.cuda.get_device_name(device_id)
        total_mem_gb = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        print(f"🚀 Using CUDA GPU {device_id}: {device_name} ({total_mem_gb:.1f} GB)")
        return {
            "device": f"cuda:{device_id}",
            "half": True,  # use half precision on CUDA for speed
            "imgsz": 640,
            "batch_optimal": True,
            "device_type": "cuda",
            "device_name": device_name,
        }
    elif prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🍎 Using Apple MPS (Metal Performance Shaders)")
        return {
            "device": "mps",
            "half": False,
            "imgsz": 640,
            "batch_optimal": True,
            "device_type": "mps",
            "device_name": "Apple GPU",
        }
    elif prefer_gpu:
        # Try Intel GPU via OpenVINO (if installed)
        try:
            import openvino  # noqa: F401

            print("⚡ OpenVINO available - using Intel GPU (OpenVINO)")
            return {
                "device": "intel:gpu",
                "half": False,
                "imgsz": 640,
                "batch_optimal": False,
                "device_type": "openvino_gpu",
                "device_name": "Intel GPU (OpenVINO)",
            }
        except ImportError:
            pass
    # Default to CPU (with OpenVINO if available)
    try:
        import openvino  # noqa: F401

        print("🔧 Using CPU with OpenVINO acceleration (if supported)")
        return {
            "device": "intel:cpu",
            "half": False,
            "imgsz": 640,
            "batch_optimal": False,
            "device_type": "openvino_cpu",
            "device_name": "Intel CPU (OpenVINO)",
        }
    except ImportError:
        print("🔧 Using standard CPU")
        return {
            "device": "cpu",
            "half": False,
            "imgsz": 640,
            "batch_optimal": False,
            "device_type": "cpu",
            "device_name": "Standard CPU",
        }
