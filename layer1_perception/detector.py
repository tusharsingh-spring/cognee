"""YOLOv11 ONNX detection + ByteTrack tracking.

Full 80-class COCO detection with person tracking on triggered frames.
Person crop extraction with padding for downstream models.

Model cascade order: yolo11n → yolo11s → yolov9n → yolov9s → yolov8n → yolov8s → auto-download
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    OBJECT_CLASSES_OF_INTEREST,
    OBJECT_CONFIDENCE,
    TRACK_PERSIST,
    TRACKER_CONFIG,
    VLM_CROP_PADDING,
    YOLO_CONFIDENCE,
    YOLO_IMG_SIZE,
    YOLO_IOU,
    YOLO_MODEL,
    YOLO_PERSON_CLASS,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

MODEL_CASCADE = [
    "yolo11n.pt", "yolo11s.pt",
    "yolov9n.pt", "yolov9s.pt",
    "yolov8n.pt", "yolov8s.pt",
]


class PersonDetector:
    def __init__(self) -> None:
        self.model = None
        self.model_name: str = ""
        self.model_version: str = ""
        self._loaded = False
        self._frame_count = 0
        self._load_model()

    def _load_model(self) -> None:
        from ultralytics import YOLO

        candidates = [YOLO_MODEL] + MODEL_CASCADE
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                logger.info(f"[DETECTOR] Trying model: {candidate}")
                self.model = YOLO(candidate)
                self.model_name = candidate
                if "yolo11" in candidate.lower():
                    self.model_version = "yolo11"
                elif "yolov9" in candidate.lower():
                    self.model_version = "yolov9"
                elif "yolov8" in candidate.lower():
                    self.model_version = "yolov8"
                self._loaded = True
                logger.info(f"[DETECTOR] Loaded {candidate} ({self.model_version})")
                return
            except Exception as e:
                logger.warning(f"[DETECTOR] Failed to load {candidate}: {e}")

        try:
            logger.info("[DETECTOR] Auto-downloading yolo11n.pt...")
            self.model = YOLO("yolo11n.pt")
            self.model_name = "yolo11n.pt"
            self.model_version = "yolo11"
            self._loaded = True
        except Exception as e:
            logger.error(f"[DETECTOR] All models failed: {e}")
            self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.model is not None

    def detect_and_track(self, frame: np.ndarray) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Detect persons with tracking. Returns (person_list, annotated_frame)."""
        if not self.is_ready:
            return [], None

        prober = profiler.get("detection")
        prober.start()

        self._frame_count += 1

        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                imgsz=YOLO_IMG_SIZE,
                classes=[YOLO_PERSON_CLASS],
                tracker=TRACKER_CONFIG,
                verbose=False,
                device="cpu",
            )
        except Exception as e:
            logger.debug(f"[DETECTOR] Track failed: {e}")
            try:
                results = self.model(
                    frame,
                    conf=YOLO_CONFIDENCE,
                    iou=YOLO_IOU,
                    imgsz=YOLO_IMG_SIZE,
                    classes=[YOLO_PERSON_CLASS],
                    verbose=False,
                    device="cpu",
                )
            except Exception as e2:
                logger.error(f"[DETECTOR] Detection failed: {e2}")
                prober.stop()
                return [], None

        persons = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                h, w = frame.shape[:2]
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == YOLO_PERSON_CLASS:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        tid = int(box.id[0]) if box.id is not None else -1

                        crop = self._extract_crop(frame, x1, y1, x2, y2)

                        persons.append({
                            "track_id": tid,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf,
                            "crop": crop,
                            "class_name": "person",
                        })

        prober.stop()
        return persons, None

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect all COCO objects (not just persons)."""
        if not self.is_ready:
            return []

        prober = profiler.get("objects")
        prober.start()

        try:
            results = self.model(
                frame,
                conf=OBJECT_CONFIDENCE,
                iou=YOLO_IOU,
                imgsz=YOLO_IMG_SIZE,
                verbose=False,
                device="cpu",
            )
        except Exception as e:
            logger.debug(f"[DETECTOR] Object detection failed: {e}")
            prober.stop()
            return []

        objects = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                h, w = frame.shape[:2]
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")

                    if cls_id != YOLO_PERSON_CLASS and conf >= OBJECT_CONFIDENCE:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        objects.append({
                            "class_id": cls_id,
                            "name": name,
                            "confidence": round(float(conf), 3),
                            "bbox": (x1, y1, x2, y2),
                        })

        prober.stop()
        return objects

    def get_person_object_proximity(
        self, person_bbox: Tuple[int, int, int, int], objects: List[Dict]
    ) -> List[Dict]:
        """Find objects near a person bbox."""
        px1, py1, px2, py2 = person_bbox
        nearby = []
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj["bbox"]
            iou = self._bbox_iou((px1, py1, px2, py2), (ox1, oy1, ox2, oy2))
            if iou > 0.01 or self._distance((px1, py1, px2, py2), (ox1, oy1, ox2, oy2)) < 100:
                nearby.append({**obj, "proximity": round(iou, 3)})
        return sorted(nearby, key=lambda x: x["proximity"], reverse=True)[:10]

    def _extract_crop(self, frame, x1, y1, x2, y2) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        pad_w = int((x2 - x1) * VLM_CROP_PADDING)
        pad_h = int((y2 - y1) * VLM_CROP_PADDING)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame[cy1:cy2, cx1:cx2].copy()

    def _bbox_iou(self, b1, b2) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _distance(self, b1, b2) -> float:
        c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
        c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
