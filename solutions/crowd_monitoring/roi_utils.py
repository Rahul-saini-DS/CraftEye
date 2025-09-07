import copy
import cv2
import numpy as np

# ---------------------------
# Geometry helpers
# ---------------------------

def _polygon_contour(roi_polygon):
    """
    Convert Python list of (x, y) to an OpenCV contour of shape (N,1,2), int32.
    Returns None if roi is invalid.
    """
    if not roi_polygon or len(roi_polygon) < 3:
        return None
    pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
    return pts


def _foot_point_from_xyxy(x1, y1, x2, y2):
    """
    Return the bottom-center (foot) point for a bbox.
    """
    cx = (int(x1) + int(x2)) // 2
    cy = int(y2)
    return cx, cy


def _indices_inside_roi(boxes, roi_polygon):
    """
    Compute indices of detection boxes whose FOOT POINT lies inside the ROI polygon.
    Works with Ultralytics Boxes' xyxy attribute.
    """
    contour = _polygon_contour(roi_polygon)
    if contour is None:
        return []

    indices = []
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None or len(xyxy) == 0:  # empty or missing
        return []

    # Ensure numpy array for fast access (works on torch tensors as well)
    try:
        arr = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
    except Exception:
        arr = np.asarray(xyxy)

    for i, b in enumerate(arr):
        x1, y1, x2, y2 = b[:4]
        fx, fy = _foot_point_from_xyxy(x1, y1, x2, y2)
        # pointPolygonTest: >0 inside, =0 on edge, <0 outside
        inside = cv2.pointPolygonTest(contour, (float(fx), float(fy)), False) >= 0
        if inside:
            indices.append(i)

    return indices


def _slice_boxes_inplace(dst_boxes, src_boxes, keep_idx):
    """
    Slice Ultralytics Boxes-like attributes in-place.
    We only modify attributes that exist on the src object.
    """
    # Helper to slice attribute if present
    def _slice_attr(obj, name):
        if hasattr(obj, name) and getattr(obj, name) is not None:
            val = getattr(obj, name)
            try:
                setattr(obj, name, val[keep_idx])
            except Exception:
                # fall back to numpy slicing
                arr = np.asarray(val)
                setattr(obj, name, arr[keep_idx])

    # Common Boxes attributes
    for field in ("xyxy", "xywh", "conf", "cls", "id", "data"):
        _slice_attr(dst_boxes, field)


# ---------------------------
# Public API
# ---------------------------

def filter_detections_by_roi(results_or_boxes, roi_polygon):
    """
    Filter YOLO detections to only include those whose FOOT POINT lies inside the ROI.

    This function is flexible:
      - If you pass a Ultralytics Results object (with .boxes), it returns a *new* Results
        with .boxes filtered.
      - If you pass a Ultralytics Boxes object, it returns a *new* Boxes filtered.
      - If the ROI is invalid or there are no boxes, the original object is returned unchanged.

    Args:
        results_or_boxes: Ultralytics Results or Boxes
        roi_polygon: List[(x, y)] ROI polygon vertices

    Returns:
        Same type as input (Results or Boxes), filtered to ROI.
    """
    if not roi_polygon or len(roi_polygon) < 3 or results_or_boxes is None:
        return results_or_boxes

    # Case 1: input is a Results-like object with .boxes
    if hasattr(results_or_boxes, "boxes"):
        src_results = results_or_boxes
        boxes = getattr(src_results, "boxes", None)
        if boxes is None or len(getattr(boxes, "xyxy", [])) == 0:
            return results_or_boxes

        keep_idx = _indices_inside_roi(boxes, roi_polygon)
        # If nothing to keep, return an empty copy
        dst_results = copy.deepcopy(src_results)
        dst_boxes = copy.deepcopy(boxes)

        if len(keep_idx) == 0:
            # empty everything
            for field in ("xyxy", "xywh", "conf", "cls", "id", "data"):
                if hasattr(dst_boxes, field) and getattr(dst_boxes, field) is not None:
                    val = getattr(dst_boxes, field)
                    try:
                        setattr(dst_boxes, field, val[:0])
                    except Exception:
                        setattr(dst_boxes, field, np.asarray(val)[:0])
            dst_results.boxes = dst_boxes
            return dst_results

        # Slice in-place on the copy
        _slice_boxes_inplace(dst_boxes, boxes, keep_idx)
        dst_results.boxes = dst_boxes
        return dst_results

    # Case 2: input is a Boxes-like object
    boxes = results_or_boxes
    if not hasattr(boxes, "xyxy") or len(getattr(boxes, "xyxy", [])) == 0:
        return results_or_boxes

    keep_idx = _indices_inside_roi(boxes, roi_polygon)
    dst_boxes = copy.deepcopy(boxes)

    if len(keep_idx) == 0:
        for field in ("xyxy", "xywh", "conf", "cls", "id", "data"):
            if hasattr(dst_boxes, field) and getattr(dst_boxes, field) is not None:
                val = getattr(dst_boxes, field)
                try:
                    setattr(dst_boxes, field, val[:0])
                except Exception:
                    setattr(dst_boxes, field, np.asarray(val)[:0])
        return dst_boxes

    _slice_boxes_inplace(dst_boxes, boxes, keep_idx)
    return dst_boxes


def draw_roi_polygon(frame, roi_polygon, color=(0, 255, 0), thickness=2, fill=False, alpha=0.25, label=None):
    """
    Draw the ROI polygon on the frame.

    Args:
        frame: BGR image (H, W, 3)
        roi_polygon: list[(x, y)] polygon points
        color: BGR outline color
        thickness: line thickness
        fill: if True, draw a translucent fill
        alpha: fill opacity (0..1)
        label: optional text label to put near the first vertex

    Returns:
        New frame with the ROI drawn (original is not modified).
    """
    if frame is None or not roi_polygon or len(roi_polygon) < 3:
        return frame

    out = frame.copy()
    pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))

    if fill:
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    # Outline
    cv2.polylines(out, [pts], True, color, thickness)

    # Optional label
    if label:
        x, y = roi_polygon[0]
        cv2.putText(out, str(label), (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)

    return out

