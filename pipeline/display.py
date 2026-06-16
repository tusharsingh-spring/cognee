"""Video display with all on-screen overlays: bboxes, captions, VQA, VSS matches."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    DISPLAY_BOX_COLOR,
    DISPLAY_FONT_SCALE,
    DISPLAY_OVERLAY_ALPHA,
    DISPLAY_TEXT_COLOR,
    DISPLAY_WINDOW,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class DisplayManager:
    def __init__(self) -> None:
        self._window_created = False
        self._overlay_panel = None

    def render(
        self,
        frame: np.ndarray,
        persons: List[Dict],
        captions: Dict[int, str],
        vqa_answers: Dict[int, List[Dict]],
        vss_matches: Dict[int, List[Tuple[int, float]]],
        stats: Dict[str, object],
    ) -> np.ndarray:
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        progress = stats.get("project_progress", {})
        if progress:
            overall = progress.get("overall", 0)
            phase = progress.get("phase", "")
            bar_w = 300
            bar_h = 12
            bar_x = 10
            bar_y = 40
            filled_w = int(bar_w * overall / 100)

            bar_overlay = canvas.copy()
            cv2.rectangle(bar_overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            if filled_w > 0:
                if overall < 40:
                    bar_color = (50, 50, 220)
                elif overall < 70:
                    bar_color = (50, 200, 220)
                else:
                    bar_color = (50, 220, 50)
                cv2.rectangle(bar_overlay, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
            cv2.rectangle(bar_overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
            canvas = cv2.addWeighted(bar_overlay, 0.8, canvas, 0.2, 0)

            cv2.putText(
                canvas,
                f"PROJECT: {overall:.0f}% | {phase}",
                (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

            cats = progress.get("categories", [])
            cat_x = bar_x + bar_w + 20
            cat_y = bar_y
            for i, cat in enumerate(cats[:5]):
                pct = cat.get("progress", 0)
                short = cat["category"].split(":")[0].strip() if ":" in cat["category"] else cat["category"][:12]
                color = (100, 255, 100) if pct >= 90 else (100, 200, 255) if pct >= 50 else (100, 100, 255)
                cv2.putText(
                    canvas,
                    f"{short}: {pct:.0f}%",
                    (cat_x, cat_y + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                )

        overlay = canvas.copy()

        for person in persons:
            tid = person["track_id"]
            x1, y1, x2, y2 = person["bbox"]

            cv2.rectangle(overlay, (x1, y1), (x2, y2), DISPLAY_BOX_COLOR, 2)

            label_y = y1 - 10 if y1 > 20 else y1 + 20

            cv2.putText(
                overlay,
                f"ID:{tid}",
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                DISPLAY_FONT_SCALE,
                DISPLAY_BOX_COLOR,
                2,
            )

            y_offset = label_y - 25

            if tid in captions and captions[tid]:
                lines = self._wrap_text(captions[tid], 40)
                for line in lines[:2]:
                    cv2.putText(
                        overlay,
                        line,
                        (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        DISPLAY_FONT_SCALE * 0.9,
                        (255, 255, 0),
                        2,
                    )
                    y_offset -= 20

            if tid in vqa_answers:
                for ans in vqa_answers[tid][:2]:
                    text = f"Q:{ans.get('question','?')[:15]} A:{ans.get('answer','')[:30]}"
                    cv2.putText(
                        overlay,
                        text,
                        (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        DISPLAY_FONT_SCALE * 0.7,
                        (255, 200, 100),
                        1,
                    )
                    y_offset -= 18

            if tid in vss_matches:
                for match_tid, sim in vss_matches[tid][:1]:
                    text = f"Similar: ID:{match_tid} ({sim:.2f})"
                    cv2.putText(
                        overlay,
                        text,
                        (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        DISPLAY_FONT_SCALE * 0.7,
                        (0, 255, 255),
                        1,
                    )
                    y_offset -= 18

            face_ids = stats.get("face_ids", {})
            if tid in face_ids:
                cv2.putText(
                    overlay,
                    f"Face: {face_ids[tid][:12]}",
                    (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    DISPLAY_FONT_SCALE * 0.7,
                    (200, 100, 255),
                    1,
                )
                y_offset -= 18

            reid_ids = stats.get("reid_ids", {})
            if tid in reid_ids:
                cv2.putText(
                    overlay,
                    f"ID: {reid_ids[tid][:16]}",
                    (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    DISPLAY_FONT_SCALE * 0.7,
                    (255, 150, 150),
                    1,
                )
                y_offset -= 18

        canvas = cv2.addWeighted(overlay, DISPLAY_OVERLAY_ALPHA, canvas, 1 - DISPLAY_OVERLAY_ALPHA, 0)

        objects = stats.get("objects", [])
        if objects:
            obj_y = h - 50
            obj_text = "OBJ: " + ", ".join(o["name"] for o in objects[:8])
            cv2.putText(
                canvas,
                obj_text[:120],
                (10, obj_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

        action_summary = stats.get("action_summary", {})
        if action_summary:
            act_text = (
                f"ACTIONS: {action_summary.get('unique_persons',0)}p/{action_summary.get('total_actions',0)}a | "
                f"{action_summary.get('sequences_detected',0)} seq"
            )
            cv2.putText(
                canvas,
                act_text[:120],
                (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 255, 200),
                1,
            )

        fps = stats.get("fps", 0)
        person_count = stats.get("person_count", 0)
        alert_count = stats.get("alert_count", 0)

        cv2.putText(
            canvas,
            f"FPS: {fps:.1f} | Persons: {person_count} | Alerts: {alert_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return canvas

    def show(self, frame: np.ndarray) -> bool:
        if not self._window_created:
            cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
            self._window_created = True

        cv2.imshow(DISPLAY_WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()
        self._window_created = False

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> List[str]:
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current = (current + " " + word).strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines
