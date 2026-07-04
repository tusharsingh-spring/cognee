"""Contact detection between persons — pure logic, no ML.

Multi-modal scoring using: bbox IoU, pose joint proximity, depth similarity,
optical flow correlation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    CONTACT_DISTANCE_THRESHOLD_MM,
    CONTACT_ENABLED,
    CONTACT_FLOW_THRESHOLD,
    CONTACT_IOU_OVERLAP,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)


class ContactDetector:
    def __init__(self) -> None:
        self.enabled = CONTACT_ENABLED
        self._active_contacts: Dict = {}
        self._contact_persistence: Dict = {}

    def detect(
        self,
        persons: List[Dict],
        poses: Dict[int, np.ndarray],
        depth_per_person: Dict[int, Dict],
        flow_per_person: Dict[int, Optional[Dict]],
        seg_masks: Optional[Dict[int, np.ndarray]],
    ) -> Dict:
        if not self.enabled or len(persons) < 2:
            return {}

        prober = profiler.get("contact")
        prober.start()

        results = {}
        for i, p1 in enumerate(persons):
            for p2 in persons[i + 1:]:
                tid_a, tid_b = p1["track_id"], p2["track_id"]
                pair = self._evaluate_pair(
                    tid_a, p1["bbox"], tid_b, p2["bbox"],
                    poses.get(tid_a), poses.get(tid_b),
                    depth_per_person.get(tid_a), depth_per_person.get(tid_b),
                    flow_per_person.get(tid_a), flow_per_person.get(tid_b),
                )
                key = (min(tid_a, tid_b), max(tid_a, tid_b))
                results[key] = pair

                if pair.get("contact"):
                    self._contact_persistence[key] = self._contact_persistence.get(key, 0) + 1
                    self._active_contacts[key] = pair
                else:
                    self._contact_persistence[key] = max(0, self._contact_persistence.get(key, 0) - 1)
                    if self._contact_persistence.get(key, 0) <= 0:
                        self._active_contacts.pop(key, None)

        prober.stop()
        return results

    def _evaluate_pair(
        self,
        tid_a: int, bbox_a: Tuple,
        tid_b: int, bbox_b: Tuple,
        kpts_a: Optional[np.ndarray],
        kpts_b: Optional[np.ndarray],
        depth_a: Optional[Dict],
        depth_b: Optional[Dict],
        flow_a: Optional[Dict],
        flow_b: Optional[Dict],
    ) -> Dict:
        evidence = []
        score = 0.0

        iou = self._bbox_iou(bbox_a, bbox_b)
        iou_score = min(1.0, iou / (CONTACT_IOU_OVERLAP + 1e-8) * 2)
        if iou > CONTACT_IOU_OVERLAP:
            evidence.append("iou")
            score += iou_score * 0.3

        joint_dist = 999.0
        if kpts_a is not None and kpts_b is not None:
            joint_dist = self._min_joint_distance(kpts_a, kpts_b)
            if joint_dist < 80:
                evidence.append("joint_proximity")
                score += (1.0 - joint_dist / 80) * 0.35

        depth_diff = 999.0
        if depth_a is not None and depth_b is not None:
            depth_diff = abs(depth_a.get("torso_depth", 0) - depth_b.get("torso_depth", 0))
            if depth_diff < CONTACT_DISTANCE_THRESHOLD_MM:
                evidence.append("depth_similar")
                score += 0.2 * (1.0 - depth_diff / CONTACT_DISTANCE_THRESHOLD_MM)

        flow_corr = 0.0
        if flow_a is not None and flow_b is not None:
            flow_diff = abs(flow_a.get("mean_magnitude", 0) - flow_b.get("mean_magnitude", 0))
            if flow_diff < CONTACT_FLOW_THRESHOLD:
                evidence.append("flow_correlated")
                score += 0.15

        contact = score > 0.3 and len(evidence) >= 1

        return {
            "person_a": tid_a,
            "person_b": tid_b,
            "contact": contact,
            "score": round(min(score, 1.0), 3),
            "evidence": evidence,
            "iou_score": round(iou_score, 3),
            "joint_distance_px": round(joint_dist, 1),
            "depth_diff_mm": round(depth_diff, 1),
            "flow_correlation": round(flow_corr, 3),
        }

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

    def _min_joint_distance(self, k1, k2) -> float:
        valid1 = k1[:, 2] > 0.3
        valid2 = k2[:, 2] > 0.3
        if not valid1.any() or not valid2.any():
            return 999.0
        d1 = k1[valid1, :2]
        d2 = k2[valid2, :2]
        if len(d1) == 0 or len(d2) == 0:
            return 999.0
        dists = np.linalg.norm(d1[:, None] - d2[None, :], axis=2)
        return float(dists.min())

    def get_active_contacts(self) -> Dict:
        return dict(self._active_contacts)

    def get_contact_persistence(self, tid_a: int, tid_b: int) -> int:
        key = (min(tid_a, tid_b), max(tid_a, tid_b))
        return self._contact_persistence.get(key, 0)
