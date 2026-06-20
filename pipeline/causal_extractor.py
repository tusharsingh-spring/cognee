"""Causal variable extraction: organize CV outputs as PCMCI+-ready indexed time series."""

import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    CAUSAL_ENABLED,
    CAUSAL_OUTPUT_DIR,
    CAUSAL_WINDOW_SECONDS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class CausalExtractor:
    def __init__(self) -> None:
        self._enabled = CAUSAL_ENABLED
        self._output_dir = CAUSAL_OUTPUT_DIR
        self._window = CAUSAL_WINDOW_SECONDS

        self._time_series: Dict[int, List[Dict]] = defaultdict(list)
        self._contact_series: List[Dict] = []
        self._scene_series: List[Dict] = []
        self._variable_names: set = set()
        self._frame_counter = 0

        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def extract(
        self,
        frame_idx: int,
        timestamp: float,
        track_ids: List[int],
        poses: Dict[int, np.ndarray],
        depth_stats: Dict[int, Dict[str, float]],
        flow_stats: Dict[int, Optional[Dict[str, float]]],
        contact_data: Dict[Tuple[int, int], Dict],
        hand_data: Dict[int, List[Dict]],
        gaze_data: Dict[int, Dict],
        action_results: Dict[int, Dict],
        objects: List[Dict],
    ) -> Dict[int, Dict]:
        if not self._enabled:
            return {}

        self._frame_counter += 1
        frame_vars: Dict[int, Dict] = {}

        for tid in track_ids:
            person_vars = {
                "track_id": tid,
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "variables": {},
            }

            kpts = poses.get(tid)
            if kpts is not None and len(kpts) >= 17:
                pv = person_vars["variables"]
                pv["nose_x"] = round(float(kpts[0, 0]), 1)
                pv["nose_y"] = round(float(kpts[0, 1]), 1)
                pv["nose_conf"] = round(float(kpts[0, 2]), 3)

                shoulder_mid_x = float(np.mean([kpts[5, 0], kpts[6, 0]])) if kpts[5, 2] > 0.3 and kpts[6, 2] > 0.3 else 0.0
                shoulder_mid_y = float(np.mean([kpts[5, 1], kpts[6, 1]])) if kpts[5, 2] > 0.3 and kpts[6, 2] > 0.3 else 0.0
                pv["torso_x"] = round(shoulder_mid_x, 1)
                pv["torso_y"] = round(shoulder_mid_y, 1)

                lw = kpts[9]
                rw = kpts[10]
                if lw[2] > 0.3 and rw[2] > 0.3:
                    pv["wrist_distance"] = round(float(np.linalg.norm(lw[:2] - rw[:2])), 1)

                if tid in poses and self._has_prev_pose(tid):
                    prev = self._get_prev_pose(tid)
                    dt = timestamp - self._get_prev_pose_time(tid)
                    if dt > 0:
                        vel = np.linalg.norm(kpts[5:7, :2] - prev[5:7, :2], axis=1).mean() / dt
                        pv["torso_velocity"] = round(float(vel), 2)

                self._store_pose(tid, kpts, timestamp)

            d_stats = depth_stats.get(tid)
            if d_stats:
                person_vars["variables"]["depth_mean"] = round(d_stats.get("mean_depth", 0.0), 4)
                person_vars["variables"]["depth_torso"] = round(d_stats.get("torso_depth", 0.0), 4)

            f_stats = flow_stats.get(tid)
            if f_stats:
                person_vars["variables"]["flow_magnitude"] = round(f_stats.get("mean_magnitude", 0.0), 2)

            action = action_results.get(tid)
            if action:
                person_vars["variables"]["action"] = action.get("action", "unknown")
                person_vars["variables"]["action_conf"] = round(action.get("confidence", 0.0), 3)

            hands = hand_data.get(tid)
            if hands:
                person_vars["variables"]["hands_detected"] = len(hands)
                grip = any(h.get("is_grip", False) for h in hands)
                person_vars["variables"]["is_gripping"] = grip

            gaze = gaze_data.get(tid)
            if gaze:
                person_vars["variables"]["gaze_yaw"] = round(gaze.get("yaw", 0.0), 1)
                person_vars["variables"]["gaze_direction"] = gaze.get("direction", "center")

            self._variable_names.update(person_vars["variables"].keys())
            self._time_series[tid].append(person_vars)
            frame_vars[tid] = person_vars

        for (ta, tb), cd in contact_data.items():
            contact_event = {
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "person_a": ta,
                "person_b": tb,
                "contact": cd.get("contact", False),
                "score": cd.get("score", 0.0),
                "evidence": cd.get("evidence", []),
                "pixel_dist": cd.get("pixel_dist", 0),
            }
            self._contact_series.append(contact_event)

            if ta in frame_vars:
                frame_vars[ta]["variables"][f"contact_person_{tb}"] = cd.get("contact", False)
                frame_vars[ta]["variables"][f"contact_score_{tb}"] = cd.get("score", 0.0)
            if tb in frame_vars:
                frame_vars[tb]["variables"][f"contact_person_{ta}"] = cd.get("contact", False)
                frame_vars[tb]["variables"][f"contact_score_{ta}"] = cd.get("score", 0.0)

        scene_vars = {
            "frame": frame_idx,
            "timestamp": round(timestamp, 3),
            "num_persons": len(track_ids),
            "objects_detected": [o.get("name", "unknown") for o in objects[:10]],
            "active_contacts": sum(1 for cd in contact_data.values() if cd.get("contact", False)),
        }
        self._scene_series.append(scene_vars)

        return frame_vars

    def _has_prev_pose(self, tid: int) -> bool:
        series = self._time_series.get(tid, [])
        return len(series) >= 2

    def _get_prev_pose(self, tid: int) -> np.ndarray:
        series = self._time_series.get(tid, [])
        prev = series[-2]
        v = prev.get("variables", {})
        kpts = np.zeros((17, 2), dtype=np.float32)
        for i in range(17):
            kpts[i, 0] = v.get(f"kpt_{i}_x", 0.0)
            kpts[i, 1] = v.get(f"kpt_{i}_y", 0.0)
        return kpts

    def _get_prev_pose_time(self, tid: int) -> float:
        series = self._time_series.get(tid, [])
        if len(series) >= 2:
            return float(series[-2].get("timestamp", 0))
        return 0.0

    def _store_pose(self, tid: int, kpts: np.ndarray, timestamp: float) -> None:
        pass

    def get_time_series(self, tid: int, window: Optional[float] = None) -> List[Dict]:
        series = self._time_series.get(tid, [])
        if not series or window is None:
            return series

        cutoff = series[-1]["timestamp"] - window
        return [e for e in series if e["timestamp"] >= cutoff]

    def get_contact_series(self) -> List[Dict]:
        return self._contact_series

    def get_variable_names(self) -> List[str]:
        return sorted(self._variable_names)

    def export_csv(self, output_path: Optional[str] = None) -> str:
        import csv

        path = output_path or str(self._output_dir / f"causal_{int(time.time())}.csv")
        all_vars = sorted(self._variable_names)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["track_id", "frame", "timestamp"] + all_vars
            writer.writerow(header)

            for tid in sorted(self._time_series.keys()):
                for entry in self._time_series[tid]:
                    row = [entry["track_id"], entry["frame"], entry["timestamp"]]
                    row += [entry["variables"].get(v, "") for v in all_vars]
                    writer.writerow(row)

        logger.info(f"[Causal] Exported {sum(len(v) for v in self._time_series.values())} rows to {path}")
        return path

    def export_json(self, output_path: Optional[str] = None) -> str:
        path = output_path or str(self._output_dir / f"causal_{int(time.time())}.json")

        export_data = {
            "variable_names": sorted(self._variable_names),
            "time_series": {str(k): v for k, v in self._time_series.items()},
            "contact_series": self._contact_series,
            "scene_series": self._scene_series,
            "total_frames": self._frame_counter,
        }

        with open(path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"[Causal] Exported JSON to {path}")
        return path

    def get_summary(self) -> Dict:
        return {
            "total_frames": self._frame_counter,
            "total_persons_tracked": len(self._time_series),
            "total_contact_events": len([c for c in self._contact_series if c.get("contact")]),
            "unique_variables": len(self._variable_names),
            "variable_names": sorted(self._variable_names),
        }
