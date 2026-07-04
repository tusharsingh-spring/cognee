"""Layer 1 Perception Pipeline — orchestrates all perception models in parallel.

Runs: detection → pose, action, gaze, flow, depth, contact, hands, segmentation
(all in parallel via ThreadPoolExecutor). Outputs a PerceptionPacket.

This is the single source of truth for what's happening in the frame.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    ACTION_ENABLED,
    CONTACT_ENABLED,
    DEPTH_ENABLED,
    FLOW_ENABLED,
    GAZE_ENABLED,
    HAND_ENABLED,
    POSE_ENABLED,
    SEG_ENABLED,
)
from layer1_perception.action_stgcn import ActionRecognizer
from layer1_perception.contact import ContactDetector
from layer1_perception.depth import DepthEstimator
from layer1_perception.detector import PersonDetector
from layer1_perception.flow import OpticalFlowEstimator
from layer1_perception.gaze import GazeEstimator
from layer1_perception.hand_tracker import HandTracker
from layer1_perception.perception_schema import (
    ActionResult,
    ContactInfo,
    DepthInfo,
    FlowInfo,
    GazeResult,
    HandInfo,
    ObjectEntry,
    PerceptionPacket,
    PersonEntry,
    PoseResult,
)
from layer1_perception.pose import PoseEstimator
from layer1_perception.segmentation import Segmenter
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)

MAX_WORKERS = 6


class PerceptionPipeline:
    """Runs all Layer 1 models in parallel and merges results into PerceptionPacket."""

    def __init__(self) -> None:
        self.detector = PersonDetector()
        self.pose = PoseEstimator()
        self.action = ActionRecognizer()
        self.gaze = GazeEstimator()
        self.flow = OpticalFlowEstimator()
        self.depth = DepthEstimator()
        self.contact = ContactDetector()
        self.hand = HandTracker()
        self.seg = Segmenter()

        self._prev_frame: Optional[np.ndarray] = None
        self._frame_number: int = 0
        self._object_cache: List[ObjectEntry] = []
        self._object_cache_ttl: int = 5

        self._status = {
            "detection": self.detector.is_ready,
            "pose": self.pose.is_ready,
            "action": self.action.is_ready,
            "gaze": self.gaze.is_ready,
            "flow": self.flow.is_ready,
            "depth": self.depth.is_ready,
            "contact": self.contact.enabled,
            "hand": self.hand.is_ready,
            "seg": self.seg.is_ready,
        }

        ready_count = sum(1 for v in self._status.values() if v)
        logger.info(f"[PIPELINE] Layer 1 ready: {ready_count}/{len(self._status)} models")
        for name, status in self._status.items():
            logger.info(f"  {name}: {'READY' if status else 'OFF'}")

    @property
    def is_ready(self) -> bool:
        return self.detector.is_ready

    @property
    def model_status(self) -> Dict[str, bool]:
        return dict(self._status)

    def process(self, frame: np.ndarray) -> PerceptionPacket:
        """Main entry point — process one frame through all perception models.

        Returns a PerceptionPacket with all structured data.
        """
        self._frame_number += 1
        frame_time = time.time()
        h, w = frame.shape[:2]

        pipeline_prober = profiler.get("layer1_total")
        pipeline_prober.start()

        packet = PerceptionPacket(
            timestamp=frame_time,
            frame_number=self._frame_number,
            frame_width=w,
            frame_height=h,
        )

        # ── Detection + Tracking (sequential — others depend on it) ──
        persons_raw, _ = self.detector.detect_and_track(frame)
        persons_raw = [p for p in persons_raw if p["track_id"] >= 0]

        # ── Object detection (cached, every 5th frame) ──
        if self._frame_number % 5 == 0:
            objects_raw = self.detector.detect_objects(frame)
            self._object_cache = objects_raw

        packet.persons = [
            PersonEntry(track_id=p["track_id"], bbox=p["bbox"],
                       confidence=p["confidence"], class_name=p.get("class_name", "person"))
            for p in persons_raw
        ]
        packet.objects = [
            ObjectEntry(class_id=o["class_id"], name=o["name"],
                       confidence=o["confidence"], bbox=o["bbox"])
            for o in self._object_cache
        ]

        model_errors = {}

        if not persons_raw:
            pipeline_prober.stop()
            return packet

        # ── Run all models in parallel ──
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            if POSE_ENABLED and self.pose.is_ready:
                futures["pose"] = pool.submit(self.pose.estimate, frame, persons_raw)

            if HAND_ENABLED and self.hand.is_ready:
                futures["hand"] = pool.submit(self.hand.track, frame, persons_raw, True)

            if GAZE_ENABLED and self.gaze.is_ready:
                futures["gaze"] = pool.submit(self.gaze.estimate, frame, persons_raw, True)

            if DEPTH_ENABLED and self.depth.is_ready:
                futures["depth"] = pool.submit(self.depth.estimate, frame, True)

            if FLOW_ENABLED and self.flow.is_ready and self._prev_frame is not None:
                futures["flow"] = pool.submit(self.flow.compute, self._prev_frame, frame, True)

            if SEG_ENABLED and self.seg.is_ready:
                futures["seg"] = pool.submit(self.seg.segment, frame, persons_raw, True)

            for key, future in futures.items():
                try:
                    result = future.result(timeout=10.0)
                except Exception as e:
                    model_errors[key] = str(e)
                    continue

                if key == "pose" and result:
                    for tid, kpts_arr in result.items():
                        packet.poses[tid] = PoseResult.from_ndarray(tid, kpts_arr)

                elif key == "hand" and result:
                    for tid, hands_list in result.items():
                        packet.hands[tid] = [
                            HandInfo(
                                track_id=tid,
                                handedness=h.get("handedness", "unknown"),
                                state=h.get("state", "neutral"),
                                landmarks=h.get("landmarks", []),
                            )
                            for h in hands_list
                        ]

                elif key == "gaze" and result:
                    for tid, gd in result.items():
                        packet.gaze[tid] = GazeResult(
                            track_id=tid,
                            yaw=gd.get("yaw", 0),
                            pitch=gd.get("pitch", 0),
                            direction=gd.get("direction", "center"),
                        )

                elif key == "depth" and result is not None:
                    depth_map = result
                    for person in persons_raw:
                        tid = person["track_id"]
                        dd = self.depth.get_person_depth(depth_map, person["bbox"])
                        if dd is not None:
                            packet.depth[tid] = DepthInfo(
                                track_id=tid,
                                mean_depth=dd.get("mean_depth", 0),
                                torso_depth=dd.get("torso_depth", 0),
                                min_depth=dd.get("min_depth", 0),
                                max_depth=dd.get("max_depth", 0),
                            )

                elif key == "flow" and result is not None:
                    flow_map = result
                    for person in persons_raw:
                        tid = person["track_id"]
                        fs = self.flow.get_person_flow_stats(flow_map, person["bbox"])
                        if fs:
                            packet.flow[tid] = FlowInfo(
                                track_id=tid,
                                mean_magnitude=fs.get("mean_magnitude", 0),
                                max_magnitude=fs.get("max_magnitude", 0),
                                direction_degrees=fs.get("direction_degrees", 0),
                            )

        # ── Action recognition (uses pose sequences) ──
        if ACTION_ENABLED and self.action.is_ready:
            for person in persons_raw:
                tid = person["track_id"]
                kpts_arr = packet.poses.get(tid)
                if kpts_arr is not None:
                    kpts_np = np.array(kpts_arr.keypoints)
                    result = self.action.update(tid, kpts_np, frame_time)
                    if result:
                        packet.actions[tid] = ActionResult(
                            track_id=tid,
                            action=result.get("action", "standing"),
                            confidence=result.get("confidence", 0.5),
                            top3=result.get("top3", []),
                            source=result.get("source", "stgcn"),
                            timestamp=frame_time,
                        )

        # ── Gaze target computation ──
        if GAZE_ENABLED and self.gaze.is_ready and packet.gaze and len(packet.persons) >= 2:
            person_bboxes = {p.track_id: p.bbox for p in packet.persons}
            for tid, gd in packet.gaze.items():
                other_bboxes = {ot: ob for ot, ob in person_bboxes.items() if ot != tid}
                gaze_dict = {"direction": gd.direction, "yaw": gd.yaw, "pitch": gd.pitch}
                target = self.gaze.compute_gaze_target(gaze_dict, other_bboxes)
                if target is not None:
                    gd.target_person_id = target

        # ── Contact detection ──
        if CONTACT_ENABLED and self.contact.enabled and len(packet.persons) >= 2:
            poses_arr = {tid: np.array(k.keypoints) for tid, k in packet.poses.items()}
            depth_dict = {
                tid: {"torso_depth": d.torso_depth, "mean_depth": d.mean_depth}
                for tid, d in packet.depth.items()
            }
            flow_dict = {
                tid: {"mean_magnitude": f.mean_magnitude, "max_magnitude": f.max_magnitude}
                for tid, f in packet.flow.items()
            }
            contact_results = self.contact.detect(
                persons_raw, poses_arr, depth_dict, flow_dict, None
            )
            for (a, b), cd in contact_results.items():
                packet.contacts.append(ContactInfo(
                    person_a=a,
                    person_b=b,
                    contact=cd.get("contact", False),
                    score=cd.get("score", 0),
                    evidence=cd.get("evidence", []),
                    iou_score=cd.get("iou_score", 0),
                    joint_distance_px=cd.get("joint_distance_px", 999),
                    depth_diff_mm=cd.get("depth_diff_mm", 999),
                    flow_correlation=cd.get("flow_correlation", 0),
                ))

        packet.model_errors = model_errors
        pipeline_prober.stop()

        self._prev_frame = frame.copy()
        return packet

    def get_current_state(self) -> Dict:
        """Quick snapshot of current scene for the chat UI."""
        return {
            "frame": self._frame_number,
            "model_status": self._status,
            "detector_model": self.detector.model_version if self.detector.is_ready else "none",
            "pose_backend": self.pose._backend if hasattr(self.pose, "_backend") else "none",
        }
