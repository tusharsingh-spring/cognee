"""Contact detection derived from pose keypoints, depth, and optical flow."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    CONTACT_DISTANCE_THRESHOLD_MM,
    CONTACT_ENABLED,
    CONTACT_FLOW_THRESHOLD,
    CONTACT_IOU_OVERLAP,
)
from utils.logger import get_logger

logger = get_logger(__name__)

CONTACT_JOINT_PAIRS = [
    (9, 9), (10, 10),
    (9, 10), (10, 9),
    (7, 7), (8, 8),
    (5, 5), (6, 6),
]


class ContactDetector:
    def __init__(self) -> None:
        self._enabled = CONTACT_ENABLED
        self._dist_threshold = CONTACT_DISTANCE_THRESHOLD_MM
        self._flow_threshold = CONTACT_FLOW_THRESHOLD
        self._iou_overlap = CONTACT_IOU_OVERLAP
        self._contact_history: Dict[Tuple[int, int], List[Dict]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(
        self,
        persons: List[Dict],
        poses: Dict[int, np.ndarray],
        depth_stats: Dict[int, Dict[str, float]],
        flow_stats: Dict[int, Optional[Dict[str, float]]],
        flow_boundaries: Optional[np.ndarray] = None,
    ) -> Dict[Tuple[int, int], Dict]:
        if not self._enabled:
            return {}

        results: Dict[Tuple[int, int], Dict] = {}
        n = len(persons)

        for i in range(n):
            for j in range(i + 1, n):
                pa = persons[i]
                pb = persons[j]
                tid_a = pa["track_id"]
                tid_b = pb["track_id"]

                kpts_a = poses.get(tid_a)
                kpts_b = poses.get(tid_b)
                depth_a = depth_stats.get(tid_a)
                depth_b = depth_stats.get(tid_b)
                flow_a = flow_stats.get(tid_a)
                flow_b = flow_stats.get(tid_b)

                contact_data = self._evaluate_pair(
                    tid_a, tid_b, pa["bbox"], pb["bbox"],
                    kpts_a, kpts_b, depth_a, depth_b, flow_a, flow_b,
                    flow_boundaries,
                )

                if contact_data is not None:
                    pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))
                    results[pair_key] = contact_data

                    if pair_key not in self._contact_history:
                        self._contact_history[pair_key] = []
                    self._contact_history[pair_key].append(contact_data)

        return results

    def _evaluate_pair(
        self,
        tid_a: int, tid_b: int,
        bbox_a: Tuple[int, int, int, int],
        bbox_b: Tuple[int, int, int, int],
        kpts_a: Optional[np.ndarray],
        kpts_b: Optional[np.ndarray],
        depth_a: Optional[Dict[str, float]],
        depth_b: Optional[Dict[str, float]],
        flow_a: Optional[Dict[str, float]],
        flow_b: Optional[Dict[str, float]],
        flow_boundaries: Optional[np.ndarray],
    ) -> Optional[Dict]:
        contact_score = 0.0
        evidence = []

        iou_val = self._bbox_iou(bbox_a, bbox_b)
        if iou_val > 0:
            contact_score += min(iou_val / self._iou_overlap, 1.0) * 0.3
            evidence.append(("bbox_overlap", round(iou_val, 3)))

        pixel_dist = self._bbox_center_distance(bbox_a, bbox_b)
        if kpts_a is not None and kpts_b is not None:
            joint_min_dist, contacting_joints = self._min_joint_distance(kpts_a, kpts_b)
            if joint_min_dist is not None:
                max_dim = max(bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0])
                if max_dim > 0:
                    norm_dist = joint_min_dist / max_dim
                    if norm_dist < 0.25:
                        contact_score += 0.5 * (1.0 - norm_dist / 0.25)
                        evidence.append(("joint_proximity", round(joint_min_dist, 1), contacting_joints))

        if depth_a is not None and depth_b is not None:
            z_a = depth_a.get("torso_depth", depth_a.get("mean_depth", 0))
            z_b = depth_b.get("torso_depth", depth_b.get("mean_depth", 0))
            if z_a > 0 and z_b > 0:
                depth_diff = abs(z_a - z_b)
                if depth_diff < 0.15:
                    contact_score += 0.3 * (1.0 - depth_diff / 0.15)
                    evidence.append(("depth_plane", round(depth_diff, 4)))

        if flow_a is not None and flow_b is not None:
            mag_a = flow_a.get("mean_magnitude", 0)
            mag_b = flow_b.get("mean_magnitude", 0)
            if mag_a > self._flow_threshold and mag_b > self._flow_threshold:
                contact_score += 0.2
                evidence.append(("flow_motion", round(max(mag_a, mag_b), 2)))

        contact_score = min(contact_score, 1.0)

        is_contact = contact_score > 0.45
        result = {
            "contact": is_contact,
            "score": round(contact_score, 3),
            "pair": (tid_a, tid_b),
            "evidence": evidence,
            "pixel_dist": round(pixel_dist, 1),
            "iou": round(iou_val, 3),
        }

        if is_contact:
            return result
        elif contact_score > 0.25:
            result["contact"] = False
            return result

        return None

    def _bbox_iou(self, a: Tuple, b: Tuple) -> float:
        xa = max(a[0], b[0])
        ya = max(a[1], b[1])
        xb = min(a[2], b[2])
        yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _bbox_center_distance(self, a: Tuple, b: Tuple) -> float:
        ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        return float(np.sqrt((ca[0] - cb[0])**2 + (ca[1] - cb[1])**2))

    def _min_joint_distance(
        self, kpts_a: np.ndarray, kpts_b: np.ndarray
    ) -> Tuple[Optional[float], List[Tuple[int, int, float]]]:
        min_dist = float("inf")
        contacting = []
        for ja, jb in CONTACT_JOINT_PAIRS:
            if ja >= len(kpts_a) or jb >= len(kpts_b):
                continue
            ca = kpts_a[ja, 2]
            cb = kpts_b[jb, 2]
            if ca < 0.3 or cb < 0.3:
                continue
            d = float(np.linalg.norm(kpts_a[ja, :2] - kpts_b[jb, :2]))
            if d < min_dist:
                min_dist = d
            if d < 50:
                contacting.append((ja, jb, round(d, 1)))
        if min_dist == float("inf"):
            return None, []
        return min_dist, contacting

    def get_contact_history(self, pair: Optional[Tuple[int, int]] = None) -> Dict:
        if pair:
            return {
                "pair": pair,
                "events": self._contact_history.get(pair, []),
                "count": len(self._contact_history.get(pair, [])),
            }
        return {
            str(k): {"count": len(v)} for k, v in self._contact_history.items()
        }

    def get_active_contacts(self) -> Dict[Tuple[int, int], bool]:
        active = {}
        for pair, history in self._contact_history.items():
            if history and history[-1].get("contact", False):
                active[pair] = True
        return active
