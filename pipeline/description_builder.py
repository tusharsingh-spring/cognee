"""Composite description builder — dense, timestamped CCTV forensics."""

import time
from typing import Dict, List, Tuple


def build_person_description(
    track_id: int,
    vlm_caption: str,
    yolo_objects: List[str],
    fast_action: Dict,
    face_id: str,
    reid_id: str,
    history: List[Dict],
    frame_shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    frame_time: float,
) -> str:
    x1, y1, x2, y2 = bbox
    h, w = frame_shape
    bbox_h = y2 - y1
    pct = round(bbox_h / h * 100)

    action = fast_action.get("action", "standing")
    movement = fast_action.get("movement_px", 0)

    time_str = time.strftime("%H:%M:%S", time.localtime(frame_time))

    parts = [f"[{time_str}] Person_{track_id}"]

    if vlm_caption and len(vlm_caption) > 20:
        parts.append(f" | Appears to be: {vlm_caption.strip().rstrip('.')}")

    parts.append(f" | Action: {action} ({movement:.0f}px {pct}% frame)")

    obj_filtered = [o for o in yolo_objects if o != "person"]
    if obj_filtered:
        parts.append(f" | Objects: {', '.join(obj_filtered[:6])}")

    if face_id:
        parts.append(f" | Face: {face_id}")
    if reid_id:
        parts.append(f" | ReID: {reid_id}")

    if history:
        dur = history[-1].get("timestamp", 0) - history[0].get("timestamp", 0) if len(history) > 1 else 0
        if dur > 0:
            parts.append(f" | Duration: {dur:.0f}s")

    return "".join(parts)
