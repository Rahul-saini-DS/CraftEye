"""
Universal postprocessing module for YOLO inference results.
Uses Ultralytics built-in methods for visualization and structured outputs.
"""
import cv2
import numpy as np
import json
from typing import List, Dict, Any, Optional

def draw_results(results: List[Any], save: bool = False, project: str = "runs/predict", name: str = "exp") -> List[np.ndarray]:
    """
    Draw bounding boxes on images using Ultralytics built-in plotting/saving.
    Args:
        results: List of Ultralytics YOLO `Results` objects (one per image/frame).
        save: If True, save annotated images to `project/name` directory.
        project: Directory for saving images if save=True.
        name: Experiment name (subfolder) for saving images.
    Returns:
        List of annotated frames in BGR format (ready for display with OpenCV/Streamlit).
    """
    # If saving is requested, use Ultralytics .save() on each result
    if save:
        for res in results:
            if res is not None:
                try:
                    res.save(dir=f"{project}/{name}")
                except Exception as e:
                    print(f"⚠️ Could not save result: {e}")
    # Use Ultralytics .plot() to get annotated image (returns numpy array in BGR)
    annotated_frames: List[np.ndarray] = []
    for res in results:
        if res is None:
            continue
        try:
            annotated = res.plot()  # Ultralytics plot returns BGR image array:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
        except Exception as e:
            print(f"⚠️ Plotting failed for a result: {e}")
            continue
        annotated_frames.append(annotated)
    return annotated_frames

def summarize_results(results: List[Any], want_ultralytics_json: bool = True, task_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Convert YOLO results into structured data (bounding boxes, classes, etc.).
    Args:
        results: List of YOLO `Results` objects.
        want_ultralytics_json: If True, use Ultralytics' built-in serialization if available.
        task_types: (Optional) list of task types (e.g. ["detect", "footfall"]) to provide context.
    Returns:
        List of dictionaries, one per result, with detection details.
    """
    summaries: List[Dict[str, Any]] = []
    for idx, res in enumerate(results):
        if res is None:
            summaries.append({
                'source_index': idx,
                'detections': [],
                'total_detections': 0
            })
            continue

        summary: Dict[str, Any] = {
            'source_index': idx,
            'detections': [],
            'total_detections': 0,
            'task_type': 'unknown'
        }

        # Detection or Segmentation tasks (look for .boxes attribute with detections)
        if hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
            summary['task_type'] = 'detection'
            summary['total_detections'] = len(res.boxes)
            try:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                class_ids = res.boxes.cls.cpu().numpy()
            except Exception:
                # Fallback: iterate boxes if direct tensor extraction fails
                boxes_xyxy = [box.xyxy[0].cpu().numpy() for box in res.boxes]
                confs = [float(box.conf[0]) if hasattr(box, "conf") else 0.0 for box in res.boxes]
                class_ids = [int(box.cls[0]) if hasattr(box, "cls") else -1 for box in res.boxes]
            # Map class IDs to names if available
            names_map = res.names if hasattr(res, 'names') else {}
            for j in range(len(boxes_xyxy)):
                cls_id = int(class_ids[j]) if class_ids is not None else -1
                class_name = names_map.get(cls_id, f"class_{cls_id}")
                # Get image dimensions for normalization
                img_h, img_w = 0, 0
                if hasattr(res, 'orig_shape') and res.orig_shape is not None:
                    img_h, img_w = res.orig_shape[:2]
                
                # Extract box coordinates
                box_coords = boxes_xyxy[j].tolist()
                
                # Create normalized box coordinates (resolution-agnostic)
                norm_box = None
                if img_w > 0 and img_h > 0:
                    x1, y1, x2, y2 = box_coords
                    norm_box = [x1/img_w, y1/img_h, x2/img_w, y2/img_h]
                    
                det = {
                    'bbox': box_coords,
                    'confidence': float(confs[j]) if confs is not None else 0.0,
                    'class_id': cls_id,
                    'name': class_name
                }
                
                # Add normalized box if available
                if norm_box:
                    det['bbox_norm'] = norm_box
                summary['detections'].append(det)
            # If segmentation masks present
            if hasattr(res, 'masks') and res.masks is not None:
                summary['task_type'] = 'segmentation'
                summary['masks_count'] = len(res.masks)

        # Pose estimation tasks (look for .keypoints)
        elif hasattr(res, 'keypoints') and res.keypoints is not None and len(res.keypoints) > 0:
            summary['task_type'] = 'pose'
            summary['total_detections'] = len(res.keypoints)
            keypoints_data = res.keypoints.numpy() if hasattr(res.keypoints, 'numpy') else res.keypoints
            summary['keypoints'] = keypoints_data.tolist() if hasattr(keypoints_data, 'tolist') else keypoints_data

        # Classification tasks (look for .probs or .cls)
        elif hasattr(res, 'probs') and res.probs is not None:
            summary['task_type'] = 'classification'
            # Ultralytics classification results: .probs is a tensor of class probabilities
            probs = res.probs.softmax(0) if hasattr(res.probs, "softmax") else res.probs
            topk_vals, topk_idxs = probs.topk(5)  # take top-5 predictions as an example
            summary['total_detections'] = 1
            summary['predictions'] = [
                {
                    'class_id': int(idx),
                    'confidence': float(val),
                    'name': res.names.get(int(idx), f"class_{int(idx)}") if hasattr(res, 'names') else f"class_{int(idx)}"
                } for val, idx in zip(topk_vals, topk_idxs)
            ]

        # Append summary for this result
        summaries.append(summary)
    return summaries

