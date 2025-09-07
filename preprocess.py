"""
Universal preprocessing module for YOLO inference pipeline
Model-agnostic frame preparation for YOLOv8/YOLOv11, PyTorch/ONNX/TensorRT/OpenVINO
Optimized for Windows with DirectShow backend
"""
from __future__ import annotations

import os
import time
import tempfile
from typing import BinaryIO, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np


# ---------------- Camera utilities ----------------

def open_cam(idx: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Universal camera opener optimized for Windows (DirectShow).
    Falls back to default backend on non-Windows.
    """
    # Set FFMPEG options for better RTSP stability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|buffer_size;1048576"
    
    if os.name == "nt":
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows stability
    else:
        cap = cv2.VideoCapture(idx)

    if not cap or not cap.isOpened():
        print(f"❌ Failed to open camera {idx}")
        return None

    # Optimized settings for real-time processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Minimize latency with small buffer
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print(f"✅ Camera {idx} opened successfully (640x480@30fps)")
    return cap


def detect_webcams(max_indices: int = 5) -> List[int]:
    """Detect available webcam indices by probing the first N indices."""
    available: List[int] = []
    for i in range(max_indices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


# ---------------- Frame preprocessing ----------------

def make_views_rgb(frame_bgr: np.ndarray, target: Tuple[int, int] = (640, 640)) -> List[np.ndarray]:
    """
    BGR frame → 4 RGB views at 640x640 for YOLO (0°, 90°, 180°, 270°).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, target)
    return [
        resized,                                                    # 0°
        cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE),              # 90°
        cv2.rotate(resized, cv2.ROTATE_180),                       # 180°
        cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE),       # 270°
    ]


# ---------------- Video-file utilities ----------------

class VideoReader:
    """
    Safe wrapper for video-file playback with OpenCV.

    - Accepts a path or an in-memory file (e.g., Streamlit UploadedFile).
    - read_once(): grab one frame for rerun-driven apps (Streamlit pattern).
    - frames(): generator that sleeps ~1/fps (useful for scripts).
    - Capable of looped playback when loop=True.
    - Always call .release() when done to close and cleanup temp files.
    """
    def __init__(self, src: Union[str, BinaryIO], loop: bool = False):
        self.loop = loop
        self.tmp_path: Optional[str] = None
        self.path = self._ensure_path(src)

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            self._cleanup_tmp()
            raise RuntimeError(f"Could not open video: {self.path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Handle corrupted/unknown FPS values gracefully
        self.fps = fps if 0.0 < fps < 121.0 else 25.0
        self.delay = 1.0 / self.fps

    def _ensure_path(self, src: Union[str, BinaryIO]) -> str:
        if isinstance(src, str):
            return src
        # buffer-like object (e.g., UploadedFile)
        self.tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(self.tmp_path, "wb") as f:
            f.write(src.read())
        return self.tmp_path

    def frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            yield frame
            # Give UI some air (slightly under native fps for safety)
            time.sleep(self.delay * 0.85)

    def read_once(self) -> Optional[np.ndarray]:
        """Read one frame (for apps that re-run each iteration)."""
        ok, frame = self.cap.read()
        if ok:
            return frame
        if self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            return frame if ok else None
        return None

    def _cleanup_tmp(self):
        if self.tmp_path and os.path.exists(self.tmp_path):
            try:
                os.remove(self.tmp_path)
            except Exception:
                pass

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        finally:
            self._cleanup_tmp()


def open_rtsp_stream(url: str) -> Optional[cv2.VideoCapture]:
    """
    Open an RTSP stream with optimized settings for stability.
    
    Args:
        url: RTSP URL to connect to
        
    Returns:
        VideoCapture object or None if connection failed
    """
    # Set FFMPEG options for RTSP stability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|buffer_size;1048576"
    
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"❌ Failed to open RTSP stream: {url}")
            return None
            
        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        print(f"✅ RTSP stream opened: {url}")
        return cap
    except Exception as e:
        print(f"❌ Error opening RTSP stream: {url}, {str(e)}")
        return None

def open_video_capture(src: Union[str, BinaryIO]) -> Tuple[cv2.VideoCapture, Optional[str]]:
    """
    Open a video from path or buffer and return (cap, tmp_path).

    Caller should:
      - use cap.read() each iteration,
      - cap.release() on stop,
      - if tmp_path is not None, os.remove(tmp_path) after release.
    """
    tmp_path: Optional[str] = None
    path: str

    if isinstance(src, str):
        path = src
    else:
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(tmp_path, "wb") as f:
            f.write(src.read())
        path = tmp_path

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Could not open video: {path}")

    return cap, tmp_path


# ---------------- Multi-angle webcam capture (kept) ----------------

class MultiAngleWebcamCapture:
    """
    Optimized multi-angle webcam capture producing 4 views @640x640
    """
    def __init__(self, webcam_index: int = 0, target_size: Tuple[int, int] = (640, 640)):
        self.webcam_index = webcam_index
        self.target_size = (640, 640)
        self.cap: Optional[cv2.VideoCapture] = None
        self.angle_names = ["Front (0°)", "Right (90°)", "Rear (180°)", "Left (270°)"]
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

    def open(self) -> bool:
        self.cap = open_cam(self.webcam_index)
        return self.cap is not None

    def capture_multi_angle_frames(self) -> Tuple[bool, List[np.ndarray], List[str]]:
        if self.cap is None:
            return False, [], []
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, [], []
        views = make_views_rgb(frame, self.target_size)
        self.frame_count += 1
        current_time = time.time()
        if self.frame_count % 30 == 0:
            elapsed = current_time - self.last_fps_time
            self.fps = 30.0 / elapsed if elapsed > 0 else 0.0
            self.last_fps_time = current_time
        return True, views, self.angle_names

    def get_fps(self) -> float:
        return self.fps

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()


# ---------------- Self-test ----------------

if __name__ == "__main__":
    print("🔍 Detecting available webcams...")
    cams = detect_webcams()
    print(f"Available webcams: {cams}")

    if cams:
        capture = MultiAngleWebcamCapture(cams[0], target_size=(640, 640))
        if capture.open():
            print("📹 Testing multi-angle capture...")
            for i in range(5):
                ok, views, names = capture.capture_multi_angle_frames()
                if ok:
                    print(f"Frame {i+1}: {len(views)} views at {views[0].shape}, FPS {capture.get_fps():.1f}")
                else:
                    print(f"Frame {i+1}: Failed to capture")
                time.sleep(0.1)
            capture.release()
            print("✅ Test completed")
    else:
        print("❌ No webcams detected")


