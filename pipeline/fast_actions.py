"""Simple movement classifier — no hallucination, just facts from bbox tracking."""

from collections import deque
from typing import Dict, Tuple


class FastActionDetector:
    def __init__(self, history_frames: int = 20) -> None:
        self._history: Dict[int, deque] = {}
        self._len = history_frames

    def classify(
        self,
        track_id: int,
        crop,
        bbox: Tuple[int, int, int, int],
        frame_time: float,
    ) -> Dict:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h = y2 - y1

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self._len)
            self._history[track_id].append({"cx": cx, "cy": cy, "h": h})
            return {"action": "standing", "confidence": 1.0, "changed": True,
                    "movement_px": 0.0, "hand_to_face": False, "torso_stable": False,
                    "upper_motion_ratio": 0.0, "bbox_change_px": 0.0}

        prev = self._history[track_id][-1]
        dx = cx - prev["cx"]
        dy = cy - prev["cy"]
        movement = (dx * dx + dy * dy) ** 0.5
        self._history[track_id].append({"cx": cx, "cy": cy, "h": h})

        changed = False
        if movement > 50:
            action = "walking fast"
            changed = True
        elif movement > 20:
            action = "walking"
            changed = True
        elif movement > 8:
            action = "moving"
            changed = True
        else:
            history_list = list(self._history[track_id])
            if len(history_list) > 15:
                recent_moves = sum(
                    abs(history_list[i]["cx"] - history_list[i - 1]["cx"]) +
                    abs(history_list[i]["cy"] - history_list[i - 1]["cy"])
                    for i in range(-15, 0)
                )
                action = "stationary" if recent_moves < 20 else "active"
            else:
                action = "standing"
            changed = len(history_list) <= 2

        return {"action": action, "confidence": min(1.0, movement / 30),
                "changed": changed, "movement_px": round(movement, 1),
                "hand_to_face": False, "torso_stable": False,
                "upper_motion_ratio": 0.0, "bbox_change_px": 0.0}
