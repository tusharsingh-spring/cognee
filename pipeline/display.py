"""Multi-channel display overlay: YOLO boxes+conf, pose skeletons, depth heatmap,
optical flow, SAM2 masks, hand keypoints, gaze arrows, action labels, contact states,
and a side panel with per-person metadata."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    DISPLAY_BOX_COLOR,
    DISPLAY_FONT_SCALE,
    DISPLAY_OVERLAY_ALPHA,
    DISPLAY_TEXT_COLOR,
    DISPLAY_WINDOW,
    SEG_ALPHA,
)
from pipeline.pose_estimator import COCO_COLORS, SKELETON_EDGES
from utils.logger import get_logger

logger = get_logger(__name__)

PANEL_WIDTH = 280
COLORS = {
    "person1": (0, 255, 0),
    "person2": (255, 0, 0),
    "person3": (0, 0, 255),
    "person4": (255, 255, 0),
    "person5": (255, 0, 255),
    "person6": (0, 255, 255),
    "contact": (0, 140, 255),
    "gaze": (0, 255, 255),
    "depth": (255, 255, 255),
    "flow": (200, 200, 0),
    "action": (200, 255, 200),
    "hand": (255, 200, 0),
    "text_ok": (0, 255, 0),
    "text_warn": (0, 200, 255),
    "text_alert": (0, 0, 255),
    "panel_bg": (30, 30, 30),
    "panel_border": (70, 70, 70),
    "model_active": (100, 255, 100),
    "model_inactive": (80, 80, 80),
}

PERSON_COLORS = [
    COLORS["person1"], COLORS["person2"], COLORS["person3"],
    COLORS["person4"], COLORS["person5"], COLORS["person6"],
]


class DisplayManager:
    def __init__(self) -> None:
        self._window_created = False
        self._show_depth = False
        self._show_flow = False
        self._show_masks = True
        self._toggle_keys = {
            ord("d"): "_show_depth",
            ord("f"): "_show_flow",
            ord("m"): "_show_masks",
        }

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

        h, w = frame.shape[:2]
        panel_w = PANEL_WIDTH
        total_w = w + panel_w + 4

        canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
        main_canvas = frame.copy()

        masks = stats.get("seg_masks") or {}
        if masks:
            main_canvas = self._overlay_masks(main_canvas, masks)

        depth_map = stats.get("depth_map")
        if depth_map is not None and self._show_depth:
            main_canvas = self._overlay_depth_heatmap(main_canvas, depth_map)

        flow_map = stats.get("flow_map")
        if flow_map is not None and self._show_flow:
            main_canvas = self._overlay_flow(main_canvas, flow_map)

        poses = stats.get("poses") or {}
        self._draw_pose_skeletons(main_canvas, persons, poses)

        hand_data = stats.get("hand_data") or {}
        self._draw_hands(main_canvas, hand_data)

        gaze_data = stats.get("gaze_data") or {}
        self._draw_gaze(main_canvas, gaze_data)

        for i, person in enumerate(persons):
            tid = person["track_id"]
            bbox = person["bbox"]
            conf = person.get("confidence", 0.0)
            x1, y1, x2, y2 = bbox

            color = PERSON_COLORS[i % len(PERSON_COLORS)]
            cv2.rectangle(main_canvas, (x1, y1), (x2, y2), color, 2)

            label_parts = [f"ID:{tid}"]
            if conf > 0:
                label_parts.append(f"{conf:.0%}")
            label = " | ".join(label_parts)

            label_y = y1 - 10 if y1 > 25 else y1 + 20
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(main_canvas, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 4), color, -1)
            cv2.putText(main_canvas, label, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            action_results = stats.get("action_results") or {}
            action = action_results.get(tid, {})
            action_label = action.get("action", "")
            action_conf = action.get("confidence", 0.0)
            if action_label:
                act_text = f"{action_label}"
                if action_conf:
                    act_text += f" ({action_conf:.0%})"
                cv2.putText(main_canvas, act_text, (x1, y2 + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["action"], 1)

            contact_data = stats.get("contact_data") or {}
            for (pa, pb), cd in contact_data.items():
                if cd.get("contact") and (tid == pa or tid == pb):
                    other = pb if tid == pa else pa
                    contact_text = f"CONTACT: ID:{other}"
                    cv2.putText(main_canvas, contact_text, (x1, y2 + 36),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["contact"], 1)
                    break

            gaze_data = stats.get("gaze_data") or {}
            gd = gaze_data.get(tid)
            if gd:
                gaze_targets = stats.get("gaze_targets") or {}
                target = gaze_targets.get(tid)
                gdir = gd.get("direction", "center")
                gaze_str = f"Gaze: {gdir}"
                if target is not None:
                    gaze_str += f" -> P{target}"
                cv2.putText(main_canvas, gaze_str, (x1, y2 + 54),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["gaze"], 1)

        canvas[:, :w] = main_canvas

        self._draw_panel(canvas, w + 2, persons, captions, stats)

        cv2.rectangle(canvas, (w, 0), (w + 2, h), (50, 50, 50), -1)

        objects = stats.get("objects", [])
        self._draw_bottom_bar(canvas, objects, stats)

        fps = stats.get("fps", 0)
        person_count = stats.get("person_count", 0)
        gate_stats = stats.get("gate_stats") or {}
        gate_str = f"G:{gate_stats.get('forwarded',0)}/{gate_stats.get('total',0)} ({gate_stats.get('skip_rate',0):.0f}%)" if gate_stats else "G:off"

        cv2.putText(canvas, f"FPS: {fps:.1f} | P: {person_count} | {gate_str} | YOLO: {stats.get('yolo_model','?')}",
                   (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return canvas

    def _draw_panel(
        self, canvas: np.ndarray, px: int, persons: List[Dict],
        captions: Dict[int, str], stats: Dict[str, object],
    ) -> None:
        h = canvas.shape[0]
        cv2.rectangle(canvas, (px, 0), (canvas.shape[1] - 1, h), COLORS["panel_bg"], -1)

        y = 8
        cv2.putText(canvas, "CHANNELS", (px + 8, y + 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y += 22

        model_status = stats.get("model_status") or {}
        model_names = [
            ("YOLO", "detection"), ("Pose", "pose"), ("Depth", "depth"),
            ("Flow", "flow"), ("Contact", "contact"), ("Seg", "seg"),
            ("Hands", "hand"), ("Gaze", "gaze"), ("Action", "action"),
            ("VLM", "vlm"), ("Causal", "causal"),
        ]
        col1_x = px + 8
        col2_x = px + 148
        for i, (label, key) in enumerate(model_names):
            active = model_status.get(key, False)
            col = COLORS["model_active"] if active else COLORS["model_inactive"]
            cx = col1_x if i < 6 else col2_x
            cy = y + (i % 6) * 16
            cv2.circle(canvas, (cx + 4, cy + 4), 4, col, -1)
            cv2.putText(canvas, label, (cx + 12, cy + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

        y += 6 * 16 + 10
        cv2.line(canvas, (px + 4, y), (canvas.shape[1] - 4, y), COLORS["panel_border"], 1)
        y += 6

        for i, person in enumerate(persons):
            tid = person["track_id"]
            bbox = person["bbox"]
            conf = person.get("confidence", 0.0)
            color = PERSON_COLORS[i % len(PERSON_COLORS)]

            cv2.putText(canvas, f"PERSON {tid}", (px + 8, y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(canvas, f"conf={conf:.0%}  box={bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}px",
                       (px + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            y += 30

            depth_data = stats.get("depth_per_person") or {}
            dd = depth_data.get(tid)
            if dd:
                cv2.putText(canvas, f"depth: {dd.get('torso_depth',0):.3f} (mean={dd.get('mean_depth',0):.3f})",
                           (px + 12, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["depth"], 1)
                y += 14

            flow_data = stats.get("flow_per_person") or {}
            fd = flow_data.get(tid)
            if fd:
                cv2.putText(canvas, f"flow: {fd.get('mean_magnitude',0):.2f}  max={fd.get('max_magnitude',0):.2f}",
                           (px + 12, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["flow"], 1)
                y += 14

            action_results = stats.get("action_results") or {}
            ar = action_results.get(tid)
            if ar:
                cv2.putText(canvas, f"action: {ar.get('action','?')} ({ar.get('confidence',0):.0%})",
                           (px + 12, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["action"], 1)
                y += 14

            hand_data = stats.get("hand_data") or {}
            hd = hand_data.get(tid)
            if hd:
                for hand in hd:
                    hand_str = f"{hand.get('handedness','?')}: {'grip' if hand.get('is_grip') else ('open' if hand.get('is_open') else 'neutral')}"
                    cv2.putText(canvas, hand_str, (px + 12, y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["hand"], 1)
                    y += 14

            gaze_data = stats.get("gaze_data") or {}
            gd = gaze_data.get(tid)
            if gd:
                cv2.putText(canvas, f"gaze: {gd.get('direction','?')} ({gd.get('yaw',0):.0f} deg)",
                           (px + 12, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["gaze"], 1)
                y += 14

            if tid in captions and captions[tid]:
                caption = captions[tid]
                available_h = h - y - 20
                max_chars = max(10, int(available_h / 12) * 30)
                short_caption = caption[:max_chars]
                lines = self._wrap_text(short_caption, 30)
                for line in lines[:6]:
                    cv2.putText(canvas, line, (px + 8, y + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)
                    y += 12

            y += 6
            cv2.line(canvas, (px + 4, y), (canvas.shape[1] - 4, y), COLORS["panel_border"], 1)
            y += 4

    def _draw_bottom_bar(
        self, canvas: np.ndarray, objects: List[Dict], stats: Dict[str, object]
    ) -> None:
        h = canvas.shape[0]
        objects = objects or []
        causal_summary = stats.get("causal_summary") or {}
        active_contacts = causal_summary.get("total_contact_events", 0)
        n_vars = causal_summary.get("unique_variables", 0)

        obj_str = f"OBJ: {', '.join(o['name'] for o in objects[:6])}" if objects else ""
        causal_str = f"  |  Contacts: {active_contacts}  |  Vars: {n_vars}" if n_vars > 0 else ""
        combined = (obj_str + causal_str)[:150]

        if combined:
            cv2.putText(canvas, combined, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    def _draw_pose_skeletons(
        self, canvas: np.ndarray, persons: List[Dict], poses: Dict[int, np.ndarray]
    ) -> None:
        for person in persons:
            tid = person["track_id"]
            kpts = poses.get(tid)
            if kpts is None:
                continue

            color = PERSON_COLORS[list(poses.keys()).index(tid) % len(PERSON_COLORS)] if tid in poses else (0, 255, 0)
            for ja, jb in SKELETON_EDGES:
                if ja >= len(kpts) or jb >= len(kpts):
                    continue
                if kpts[ja, 2] > 0.3 and kpts[jb, 2] > 0.3:
                    pt1 = (int(kpts[ja, 0]), int(kpts[ja, 1]))
                    pt2 = (int(kpts[jb, 0]), int(kpts[jb, 1]))
                    cv2.line(canvas, pt1, pt2, color, 2)

            for ji in range(min(len(kpts), 17)):
                if kpts[ji, 2] > 0.3:
                    pt = (int(kpts[ji, 0]), int(kpts[ji, 1]))
                    cv2.circle(canvas, pt, 3, COCO_COLORS[ji % len(COCO_COLORS)], -1)

    def _draw_hands(self, canvas: np.ndarray, hand_data: Dict[int, List[Dict]]) -> None:
        for tid, hands in hand_data.items():
            for hand in hands:
                kpts = hand.get("keypoints", [])
                if len(kpts) < 21:
                    continue
                pts = [(int(k[0]), int(k[1])) for k in kpts]

                connections = [
                    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                    (13,17),(0,17),(17,18),(18,19),(19,20),
                ]
                for ci, cj in connections:
                    if ci < len(pts) and cj < len(pts):
                        cv2.line(canvas, pts[ci], pts[cj], COLORS["hand"], 1)
                for pt in pts:
                    cv2.circle(canvas, pt, 2, COLORS["hand"], -1)

    def _draw_gaze(self, canvas: np.ndarray, gaze_data: Dict[int, Dict]) -> None:
        for tid, gd in gaze_data.items():
            ec = gd.get("eyes_center")
            if ec is None:
                continue
            yaw = gd.get("yaw", 0)
            gaze_len = 35
            dx = int(gaze_len * np.sin(np.radians(yaw)))
            end = (ec[0] + dx, ec[1])
            cv2.circle(canvas, ec, 3, COLORS["gaze"], -1)
            cv2.arrowedLine(canvas, ec, end, COLORS["gaze"], 2, tipLength=0.3)

    def _overlay_masks(self, canvas: np.ndarray, masks: Dict[int, np.ndarray]) -> np.ndarray:
        if not masks:
            return canvas
        colors_list = PERSON_COLORS
        overlay = canvas.copy()
        for i, (tid, mask) in enumerate(masks.items()):
            if mask.shape[:2] != canvas.shape[:2]:
                mask_resized = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]))
            else:
                mask_resized = mask
            color = colors_list[i % len(colors_list)]
            overlay[mask_resized > 0] = color
        return cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0)

    def _overlay_depth_heatmap(self, canvas: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        if depth_map.shape[:2] != canvas.shape[:2]:
            depth_map = cv2.resize(depth_map, (canvas.shape[1], canvas.shape[0]))
        heatmap = (depth_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
        return cv2.addWeighted(canvas, 0.6, heatmap, 0.4, 0)

    def _overlay_flow(self, canvas: np.ndarray, flow_map: np.ndarray) -> np.ndarray:
        if flow_map is None:
            return canvas
        h, w = canvas.shape[:2]
        fh, fw = flow_map.shape[:2]
        if fh != h or fw != w:
            flow_map = cv2.resize(flow_map, (w, h))
        mag, ang = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 200
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(canvas, 0.7, flow_bgr, 0.3, 0)

    def show(self, frame: np.ndarray) -> bool:
        if not self._window_created:
            cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DISPLAY_WINDOW, 1440, 720)
            self._window_created = True

        cv2.imshow(DISPLAY_WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in self._toggle_keys:
            attr_name = self._toggle_keys[key]
            setattr(self, attr_name, not getattr(self, attr_name, False))
            state = getattr(self, attr_name)
            label_map = {"_show_depth": "Depth overlay", "_show_flow": "Flow overlay", "_show_masks": "Masks overlay"}
            logger.info(f"[Display] {label_map.get(attr_name, attr_name)}: {state}")

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
