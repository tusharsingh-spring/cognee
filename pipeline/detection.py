"""Person detection + tracking using YOLOv11n with built-in ByteTrack. Falls back to YOLOv9/v8."""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
    YOLO_IOU,
    YOLO_MODEL,
    YOLO_PERSON_CLASS,
    TRACK_PERSIST,
    TRACKER_CONFIG,
    VLM_CROP_PADDING,
    VLM_MAX_SIZE,
    DETECT_ALL_OBJECTS,
    OBJECT_CONFIDENCE,
    OBJECT_CLASSES_OF_INTEREST,
    MODEL_DIR,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)

DetectedPerson = Dict[str, object]
DetectedObject = Dict[str, object]

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

YOLO_MODEL_CANDIDATES = [
    "yolo11n.pt",
    "yolov9n.pt",
    "yolov8n.pt",
    "yolo11s.pt",
    "yolov9s.pt",
    "yolov8s.pt",
]

ALL_CLASSES = set(range(80))
YOLO_PERSON = 0


class PersonDetector:
    def __init__(self) -> None:
        from ultralytics import YOLO

        self._model_name = None
        self._model_version = "unknown"
        self._device = "cpu"

        model_path = self._resolve_model()
        logger.info(f"[DETECT] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self._device)
        self.model.overrides["imgsz"] = YOLO_IMAGE_SIZE
        self.model.overrides["verbose"] = False
        logger.info(f"[DETECT] YOLO forced imgsz={YOLO_IMAGE_SIZE}")

        bn = os.path.basename(str(model_path)).lower()
        if "yolo11" in bn:
            self._model_version = "YOLOv11"
        elif "yolov9" in bn:
            self._model_version = "YOLOv9"
        elif "yolov8" in bn:
            self._model_version = "YOLOv8"

        self._tracked_persons: Dict[int, float] = {}
        self._id_counter = 0
        self._last_objects: List[DetectedObject] = []
        self._total_frames = 0

    def _resolve_model(self) -> str:
        for candidate in YOLO_MODEL_CANDIDATES:
            cand_path = MODEL_DIR / candidate
            if cand_path.exists():
                self._model_name = candidate
                logger.info(f"[DETECT] Found model: {cand_path}")
                return str(cand_path)

            if os.path.exists(candidate):
                self._model_name = candidate
                return candidate

        if os.path.exists(YOLO_MODEL):
            self._model_name = os.path.basename(YOLO_MODEL)
            return YOLO_MODEL

        self._model_name = "yolo11n.pt"
        logger.info("[DETECT] Model not found locally, Ultralytics will auto-download yolo11n.pt")
        return "yolo11n.pt"

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_name(self) -> str:
        return self._model_name or "unknown"

    def detect_and_track(
        self, frame: np.ndarray
    ) -> Tuple[List[DetectedPerson], np.ndarray]:
        prober = profiler.get("detection")
        prober.start()
        self._total_frames += 1

        inference_size = YOLO_IMAGE_SIZE
        small_frame = cv2.resize(frame, (inference_size, inference_size),
                                 interpolation=cv2.INTER_LINEAR)

        results = self.model.track(
            small_frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            classes=[YOLO_PERSON_CLASS],
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False,
            device=self._device,
        )

        if results is None or len(results) == 0:
            prober.stop()
            return [], frame

        result = results[0]
        annotated = result.plot(conf=True, labels=True, boxes=True) if hasattr(result, "plot") else frame
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            prober.stop()
            return [], annotated

        persons: List[DetectedPerson] = []
        new_ids: List[int] = []
        h, w = frame.shape[:2]
        inference_size = YOLO_IMAGE_SIZE
        scale_x = w / inference_size
        scale_y = h / inference_size

        for i in range(len(boxes)):
            conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
            cls_id = int(boxes.cls[i]) if boxes.cls is not None else -1

            if cls_id != YOLO_PERSON_CLASS:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy()
            x1 = int(xyxy[0] * scale_x)
            y1 = int(xyxy[1] * scale_y)
            x2 = int(xyxy[2] * scale_x)
            y2 = int(xyxy[3] * scale_y)

            track_id = (
                int(boxes.id[i].item())
                if boxes.id is not None and i < len(boxes.id)
                else -1
            )

            pad_w = int((x2 - x1) * VLM_CROP_PADDING)
            pad_h = int((y2 - y1) * VLM_CROP_PADDING)
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(w, x2 + pad_w)
            cy2 = min(h, y2 + pad_h)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crop = cv2.resize(crop, (VLM_MAX_SIZE, VLM_MAX_SIZE))

            person: DetectedPerson = {
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "crop": crop,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "model_version": self._model_version,
            }
            persons.append(person)

            if track_id not in self._tracked_persons:
                new_ids.append(track_id)
                logger.info(f"[DETECT] NEW Person ID:{track_id} (conf={conf:.2f})")
            self._tracked_persons[track_id] = cv2.getTickCount() / cv2.getTickFrequency()

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        expired = [
            tid
            for tid, ts in self._tracked_persons.items()
            if current_time - ts > TRACK_PERSIST
        ]
        for tid in expired:
            del self._tracked_persons[tid]
            logger.info(f"[DETECT] Person ID:{tid} left scene")

        if persons:
            ids = [p["track_id"] for p in persons]
            logger.debug(f"[DETECT] {len(persons)} persons | IDs: {ids}")

        prober.stop()
        return persons, annotated

    def detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        if not DETECT_ALL_OBJECTS:
            return []

        h, w = frame.shape[:2]
        inference_size = YOLO_IMAGE_SIZE
        scale_x = w / inference_size
        scale_y = h / inference_size
        small_frame = cv2.resize(frame, (inference_size, inference_size),
                                 interpolation=cv2.INTER_LINEAR)

        results = self.model(
            small_frame,
            conf=OBJECT_CONFIDENCE,
            iou=YOLO_IOU,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False,
            device=self._device,
        )

        if results is None or len(results) == 0:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        objects: List[DetectedObject] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i]) if boxes.cls is not None else -1
            conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0

            if cls_id == YOLO_PERSON_CLASS:
                continue

            if cls_id not in ALL_CLASSES:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy()
            x1 = int(xyxy[0] * scale_x)
            y1 = int(xyxy[1] * scale_y)
            x2 = int(xyxy[2] * scale_x)
            y2 = int(xyxy[3] * scale_y)
            name = COCO_CLASSES.get(cls_id, f"cls_{cls_id}")

            objects.append({
                "class_id": cls_id,
                "name": name,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            })

        self._last_objects = objects
        return objects

    def get_person_object_proximity(
        self, person_bbox: Tuple[int, int, int, int], objects: List[DetectedObject], threshold: int = 150
    ) -> List[Dict]:
        px1, py1, px2, py2 = person_bbox
        pc = ((px1 + px2) // 2, (py1 + py2) // 2)
        nearby = []

        for obj in objects:
            ox1, oy1, ox2, oy2 = obj["bbox"]
            oc = obj["center"]
            dist = np.sqrt((pc[0] - oc[0]) ** 2 + (pc[1] - oc[1]) ** 2)

            overlap_x = max(0, min(px2, ox2) - max(px1, ox1))
            overlap_y = max(0, min(py2, oy2) - max(py1, oy1))
            overlap = (overlap_x * overlap_y) / ((px2 - px1) * (py2 - py1) + 1e-6)

            if dist < threshold or overlap > 0.1:
                nearby.append({
                    "object": obj["name"],
                    "class_id": obj["class_id"],
                    "distance": float(dist),
                    "overlap": float(overlap),
                    "confidence": obj["confidence"],
                })

        return nearby

    @property
    def tracked_ids(self) -> List[int]:
        return list(self._tracked_persons.keys())

    @property
    def last_objects(self) -> List[DetectedObject]:
        return self._last_objects
