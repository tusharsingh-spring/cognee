"""
ARGUS - Video Intelligence System
Main entry point that orchestrates dual pipeline layers:
  CV Pipeline: YOLO + Pose + Depth + Flow + Contact + Seg + Hands + Gaze + Action
  VLM Pipeline: Florence-2 dense captioning + VQA
  Knowledge: Graph + Vector DB + SQLite + Causal extraction
  Output: Display + Alerts + LLM Chat + Dashboard
"""

import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    DENSE_CAPTIONING,
    DENSE_FRAME_INTERVAL,
    INPUT_VID_DIR,
    SUMMARY_INTERVAL,
    VIDEOS_DIR,
    SUPPORTED_VIDEO_FORMATS,
    VLM_REFRESH_INTERVAL,
    VLM_TURBO_MODE,
    GATE_ENABLED,
    POSE_ENABLED,
    DEPTH_ENABLED,
    FLOW_ENABLED,
    CONTACT_ENABLED,
    SEG_ENABLED,
    HAND_ENABLED,
    GAZE_ENABLED,
    ACTION_RECOG_ENABLED,
    CAUSAL_ENABLED,
    AUDIO_ENABLED,
)
from pipeline.capture import MotionCapture
from pipeline.frame_gate import FrameGate
from pipeline.detection import PersonDetector
from pipeline.pose_estimator import PoseEstimator
from pipeline.depth_estimator import DepthEstimator
from pipeline.optical_flow import OpticalFlowEstimator
from pipeline.contact_detector import ContactDetector
from pipeline.segmentation import Segmenter
from pipeline.hand_tracker import HandTracker
from pipeline.gaze_estimator import GazeEstimator
from pipeline.action_recognizer import ActionRecognizer
from pipeline.causal_extractor import CausalExtractor
from pipeline.description_builder import build_person_description
from pipeline.fast_actions import FastActionDetector
from pipeline.face_recognition import FaceRecognizer
from pipeline.reid_handler import ReIDHandler
from pipeline.action_engine import TemporalActionEngine
from pipeline.scene_analyzer import SceneAnalyzer
from pipeline.llm_aggregator import LLMAggregator
from pipeline.vlm_engine import VLMEngine
from pipeline.vlm_trigger import VLMTriggerManager
from pipeline.vqa_handler import VQAHandler
from pipeline.vss_handler import VSSHandler
from pipeline.display import DisplayManager
from knowledge.graph_store import KnowledgeGraph
from knowledge.vector_store import VectorStore
from knowledge.sqlite_store import SQLiteStore
from knowledge.summary_engine import SummaryEngine
from knowledge.project_tracker import tracker as project_tracker
from knowledge.session_manager import SessionManager
from knowledge.cctv_qa import CCTVQA
from knowledge.groq_chat import GroqChatBot
from alerts.alert_engine import AlertEngine
from alerts.webhook import WebhookNotifier
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger("ARGUS")


class ARGUS:
    def __init__(self, video_file: Optional[str] = None, turbo: bool = False) -> None:
        logger.info("=" * 60)
        logger.info("  ARGUS Video Intelligence System Starting" + (" [TURBO]" if turbo else ""))
        logger.info("=" * 60)

        self._turbo = turbo
        self.capture = MotionCapture(source=video_file)
        self.frame_gate = FrameGate()
        self.detector = PersonDetector()
        self.fast_actions = FastActionDetector(history_frames=30)
        self.face_recognizer = FaceRecognizer()
        self.reid = ReIDHandler()
        self.action_engine = TemporalActionEngine()
        self.scene_analyzer = SceneAnalyzer()
        self.llm_aggregator = LLMAggregator()
        self.cctv_qa = CCTVQA()
        self.groq_chat = GroqChatBot()
        self.vlm = VLMEngine()
        self.vlm_trigger = VLMTriggerManager()
        self.vqa = VQAHandler(self.vlm)
        self.vss = VSSHandler()
        self.display = DisplayManager()
        self.graph = KnowledgeGraph()
        self.vector_store = VectorStore()
        self.sqlite = SQLiteStore()
        self.summary_engine = SummaryEngine()
        self.session_manager = SessionManager()
        self.alerts = AlertEngine()
        self.webhook = WebhookNotifier()

        self.pose_estimator = PoseEstimator()
        self.depth_estimator = DepthEstimator()
        self.flow_estimator = OpticalFlowEstimator()
        self.contact_detector = ContactDetector()
        self.segmentation = Segmenter()
        self.hand_tracker = HandTracker()
        self.gaze_estimator = GazeEstimator()
        self.action_recognizer = ActionRecognizer()
        self.causal_extractor = CausalExtractor()

        self._captions: Dict[int, str] = {}
        self._vqa_answers: Dict[int, List[Dict]] = {}
        self._vss_matches: Dict[int, List] = {}
        self._face_ids: Dict[int, str] = {}
        self._reid_ids: Dict[int, str] = {}
        self._objects: List[Dict] = []
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_skip = 0
        self._running = False
        self._stopped = False

        self.graph.load()
        self.face_recognizer.load()
        self.reid.load()
        self.session_manager.start_session()

        project_tracker.mark_active("camera")
        project_tracker.mark_active("mog2")
        project_tracker.mark_active("yolo")
        project_tracker.mark_active("bytetrack")
        project_tracker.mark_active("crop")
        project_tracker.mark_active("graph")
        project_tracker.mark_active("sqlite")
        project_tracker.mark_active("alerts")
        project_tracker.mark_active("summary")
        if GATE_ENABLED:
            project_tracker.mark_active("frame_gate")
        if POSE_ENABLED:
            project_tracker.mark_active("pose")
        if DEPTH_ENABLED:
            project_tracker.mark_active("depth")
        if FLOW_ENABLED:
            project_tracker.mark_active("flow")
        if CONTACT_ENABLED:
            project_tracker.mark_active("contact")
        if SEG_ENABLED:
            project_tracker.mark_active("seg")
        if HAND_ENABLED:
            project_tracker.mark_active("hand")
        if GAZE_ENABLED:
            project_tracker.mark_active("gaze")
        if ACTION_RECOG_ENABLED:
            project_tracker.mark_active("action_recog")
        if CAUSAL_ENABLED:
            project_tracker.mark_active("causal")

    def start(self) -> None:
        logger.info("[ARGUS] Starting main loop...")
        self.vlm.start_worker()
        project_tracker.mark_active("florence")
        project_tracker.mark_active("queue")
        self.vss.load()
        project_tracker.mark_active("chromadb")
        project_tracker.mark_active("vss")
        if self.face_recognizer.is_available:
            project_tracker.mark_active("face")
        if self.reid.is_ready:
            project_tracker.mark_active("reid")
        project_tracker.log_progress()

        logger.info("[ARGUS] CV pipeline modules:")
        logger.info(f"  Frame Gate: {self.frame_gate.enabled}")
        logger.info(f"  Pose: {self.pose_estimator.is_ready}")
        logger.info(f"  Depth: {self.depth_estimator.is_ready}")
        logger.info(f"  Flow: {self.flow_estimator.is_ready}")
        logger.info(f"  Contact: {self.contact_detector.enabled}")
        logger.info(f"  Segmentation: {self.segmentation.is_ready}")
        logger.info(f"  Hands: {self.hand_tracker.is_ready}")
        logger.info(f"  Gaze: {self.gaze_estimator.is_ready}")
        logger.info(f"  Action: {self.action_recognizer.is_ready}")
        logger.info(f"  Causal: {self.causal_extractor.enabled}")
        logger.info(f"  YOLO: {self.detector.model_version} ({self.detector.model_name})")

        self._qa_thread = threading.Thread(target=self._qa_console_loop, daemon=True)
        self._qa_thread.start()
        logger.info("[QA] Console Q&A ready - type questions below")

        self._running = True
        self._run_loop()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        logger.info("[ARGUS] Shutting down...")
        self.vlm.stop_worker()
        self.capture.release()
        self.display.close()
        self.graph.save()
        self.sqlite.close()
        self.session_manager.end_session()
        logger.info("[ARGUS] Shutdown complete")

    def _qa_console_loop(self) -> None:
        import select
        logger.info("")
        logger.info("=" * 50)
        logger.info("  Type your questions below (press Enter to send)")
        logger.info("  Examples: 'who is here?', 'what is person_1 doing?'")
        logger.info("  'summary', 'timeline', 'objects', 'how many people?'")
        logger.info("=" * 50)
        logger.info("")

        while self._running:
            try:
                if select.select([sys.stdin], [], [], 1.0)[0]:
                    question = sys.stdin.readline().strip()
                    if question:
                        if question.lower() in ("q", "quit", "exit"):
                            self._running = False
                            break
                        self._answer_question(question)
            except (OSError, ValueError):
                pass

    def _answer_question(self, question: str) -> None:
        q = question.lower().strip()
        result = self.cctv_qa.answer(
            question, self.graph, self.vector_store, self.sqlite,
            self.action_engine, self.session_manager, self.reid
        )

        context = self.scene_analyzer.gather_context_for_llm(
            self.action_engine, self.graph, self.reid, self.session_manager
        )
        agg_result = self.llm_aggregator.answer_question(question, context)

        base_answer = result.get("answer", agg_result.get("answer", "No answer found"))

        captured_info = []
        if self._captions:
            if "how many" in q or "count" in q or "total" in q:
                captured_info.append(f"{len(self._captions)} person(s) detected")
            for tid, caption in sorted(self._captions.items()):
                if f"person_{tid}" in q.replace(" ", "") or str(tid) in q or "all" in q or "every" in q:
                    captured_info.append(f"Person_{tid}: {caption[:300]}")
                elif "who" in q or "describe" in q or "what happened" in q:
                    captured_info.append(f"Person_{tid}: {caption[:200]}")

        if captured_info:
            logger.info("=" * 60)
            for line in captured_info:
                logger.info(f"  {line}")
            logger.info("=" * 60)

        logger.info("")
        logger.info(f"Q: {question}")
        logger.info(f"A: {base_answer[:500]}")
        if result.get("evidence"):
            ev_count = len(result["evidence"]) if isinstance(result["evidence"], list) else 1
            logger.info(f"   Evidence: {ev_count} items (type: {result.get('type', 'unknown')})")
        logger.info("")

        if not captured_info and not base_answer:
            self._dump_final_report()

    def _run_loop(self) -> None:
        while self._running:
            self._frame_skip += 1
            if not hasattr(self, "_vid_frame"):
                self._vid_frame = 0
            self._vid_frame += 1
            frame_prober = profiler.get("frame_total")
            frame_prober.start()

            ret, frame, motion_boxes = self.capture.read()
            if not ret:
                if self.capture._is_video_file:
                    logger.info("[ARGUS] Video file ended, generating final report...")
                    self._dump_final_report()
                    self.causal_extractor.export_json()
                    self.causal_extractor.export_csv()
                    self._running = False
                    break
                logger.warning("[ARGUS] Camera read failed, retrying...")
                time.sleep(0.5)
                continue

            h, w = frame.shape[:2]
            frame_time = time.time()

            should_process = True
            gate_reason = "gate_disabled"

            if GATE_ENABLED:
                prev_ids = self._prev_gate_ids if hasattr(self, "_prev_gate_ids") else set()
                pose_preview = self._get_pose_preview_for_gate(prev_ids)
                contact_preview = self._get_contact_preview_for_gate()
                should_process, gate_reason = self.frame_gate.decide(
                    frame, prev_ids, motion_boxes, pose_preview, contact_preview
                )
                if not should_process:
                    self._prev_frame = frame
                    frame_prober.stop()
                    continue

            if not hasattr(self, "_pipeline_timer"):
                self._pipeline_timer = time.time()
            pipeline_elapsed = time.time() - self._pipeline_timer
            if pipeline_elapsed > 0.5:
                frame_to_skip = int(pipeline_elapsed / 0.1)
                self._vid_frame += frame_to_skip
            self._pipeline_timer = time.time()

            persons, _ = self.detector.detect_and_track(frame)
            persons = [p for p in persons if p["track_id"] >= 0]
            current_ids = {p["track_id"] for p in persons}

            if GATE_ENABLED:
                self._prev_gate_ids = current_ids

            objects = self.detector.detect_objects(frame) if self._frame_skip % 5 == 0 else self._objects
            self._objects = objects

            should_process = True

            poses: Dict[int, np.ndarray] = {}
            depth_map = None
            depth_per_person: Dict[int, Dict] = {}
            flow = None
            flow_per_person: Dict[int, Optional[Dict]] = {}
            contact_data: Dict = {}
            seg_masks: Dict[int, np.ndarray] = {}
            hand_data: Dict[int, List[Dict]] = {}
            gaze_data: Dict[int, Dict] = {}
            action_results: Dict[int, Dict] = {}
            gaze_targets: Dict[int, Optional[int]] = {}

            futures = {}
            with ThreadPoolExecutor(max_workers=6) as pool:
                if POSE_ENABLED and persons:
                    futures["pose"] = pool.submit(self.pose_estimator.estimate, frame, persons)
                if HAND_ENABLED and persons:
                    futures["hand"] = pool.submit(self.hand_tracker.track, frame, persons, should_process)
                if GAZE_ENABLED and persons:
                    futures["gaze"] = pool.submit(self.gaze_estimator.estimate, frame, persons, should_process)
                if DEPTH_ENABLED:
                    futures["depth"] = pool.submit(self.depth_estimator.estimate, frame, should_process)
                if FLOW_ENABLED and self._prev_frame is not None:
                    futures["flow"] = pool.submit(self.flow_estimator.compute, self._prev_frame, frame, should_process)
                if SEG_ENABLED and persons:
                    futures["seg"] = pool.submit(self.segmentation.segment, frame, persons, should_process)

                for key, future in futures.items():
                    try:
                        result = future.result(timeout=5.0)
                    except Exception:
                        continue

                    if key == "pose" and result:
                        poses = result
                        self._set_pose_preview_for_gate(poses)
                    elif key == "hand":
                        hand_data = result
                    elif key == "gaze":
                        gaze_data = result
                    elif key == "depth" and result is not None:
                        depth_map = result
                        for person in persons:
                            dd = self.depth_estimator.get_person_depth(depth_map, person["bbox"])
                            if dd is not None:
                                depth_per_person[person["track_id"]] = dd
                    elif key == "flow" and result is not None:
                        flow = result
                        for person in persons:
                            fs = self.flow_estimator.get_person_flow_stats(flow, person["bbox"])
                            flow_per_person[person["track_id"]] = fs
                    elif key == "seg":
                        seg_masks = result

            if CONTACT_ENABLED and len(persons) >= 2:
                contact_data = self.contact_detector.detect(
                    persons, poses, depth_per_person, flow_per_person, None
                )

            if ACTION_RECOG_ENABLED and persons:
                for person in persons:
                    tid = person["track_id"]
                    kpts = poses.get(tid)
                    if kpts is not None:
                        result = self.action_recognizer.update(tid, kpts, frame_time)
                        if result:
                            action_results[tid] = result

            if GAZE_ENABLED and gaze_data:
                other_bboxes = {}
                for p in persons:
                    if p["track_id"] not in gaze_data:
                        other_bboxes[p["track_id"]] = p["bbox"]
                for tid, gd in gaze_data.items():
                    target = self.gaze_estimator.compute_gaze_target(
                        gd, {ot: ob for ot, ob in other_bboxes.items() if ot != tid}
                    )
                    gaze_targets[tid] = target

            if CAUSAL_ENABLED:
                self.causal_extractor.extract(
                    self._vid_frame, frame_time, list(current_ids),
                    poses, depth_per_person, flow_per_person, contact_data,
                    hand_data, gaze_data, action_results, objects,
                )

            active_ids = set(self.vlm_trigger.get_active_ids())
            for old_id in active_ids - current_ids:
                self.vlm_trigger.unregister_person(old_id)
                if old_id in self._captions:
                    self.action_engine.log_action(old_id, "left", self._captions.get(old_id, ""))

            for person in persons:
                tid = person["track_id"]
                crop = person["crop"]
                bbox = person["bbox"]

                self.vlm_trigger.register_person(tid)

                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                prev_bbox = getattr(self, "_prev_bboxes", {}).get(tid)
                movement_delta = 0.0
                if prev_bbox:
                    px1, py1, px2, py2 = prev_bbox
                    prev_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                    movement_delta = float(np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2))
                if not hasattr(self, "_prev_bboxes"):
                    self._prev_bboxes: Dict[int, Tuple] = {}
                self._prev_bboxes[tid] = bbox

                fast_result = {"action": action_results.get(tid, {}).get("action", "standing"),
                              "confidence": action_results.get(tid, {}).get("confidence", 0.5),
                              "changed": tid not in getattr(self, "_person_seen", set()),
                              "movement_px": movement_delta}

                if not hasattr(self, "_person_seen"):
                    self._person_seen = set()
                if tid not in self._person_seen:
                    self._person_seen.add(tid)
                    if fast_result["changed"]:
                        fast_result["changed"] = True

                if not self._turbo:
                    had_vlm = tid in getattr(self, "_person_vlm_done", set())
                    had_scene = tid in getattr(self, "_person_scene_done", set())
                    had_od = tid in getattr(self, "_person_od_done", set())
                    if not hasattr(self, "_person_vlm_done"):
                        self._person_vlm_done = set()
                    if not hasattr(self, "_person_scene_done"):
                        self._person_scene_done = set()
                    if not hasattr(self, "_person_od_done"):
                        self._person_od_done = set()
                    if not hasattr(self, "_person_vlm_times"):
                        self._person_vlm_times: Dict[int, float] = {}

                    if not had_vlm:
                        cached_age = self.vlm.get_cache_age(tid, "dense")
                        if cached_age > 3.0:
                            self.vlm.submit_scene_dense(tid, frame, bbox,
                                [o["name"] for o in self._objects[:8]] if self._objects else [], task="dense")
                            self._person_vlm_done.add(tid)
                            self._person_vlm_times[tid] = frame_time

                    elif not had_od and frame_time - self._person_vlm_times.get(tid, 0) > 4.0:
                        cached_age = self.vlm.get_cache_age(tid, "od")
                        if cached_age > 8.0:
                            self.vlm.submit_scene_dense(tid, frame, bbox,
                                [o["name"] for o in self._objects[:8]] if self._objects else [], task="od")
                            self._person_od_done.add(tid)

                    elif not had_scene and frame_time - self._person_vlm_times.get(tid, 0) > 8.0:
                        cached_age = self.vlm.get_cache_age(tid, "scene")
                        if cached_age > 15.0:
                            self.vlm.submit_scene_dense(tid, frame, bbox,
                                [o["name"] for o in self._objects[:8]] if self._objects else [], task="scene")
                            self._person_scene_done.add(tid)

                dense_result = self.vlm.get_result(tid, "dense") if not self._turbo else ""
                scene_result = self.vlm.get_result(tid, "scene") if not self._turbo else ""
                od_result = self.vlm.get_result(tid, "od") if not self._turbo else ""

                vlm_combined = self.vlm.get_combined_result(tid) if not self._turbo else {}

                if not hasattr(self, "_person_vlm_data"):
                    self._person_vlm_data: Dict[int, str] = {}
                if dense_result and len(dense_result) > 30 and "QA>" not in dense_result:
                    self._person_vlm_data[tid] = dense_result
                elif scene_result and len(scene_result) > 30 and "QA>" not in scene_result:
                    if tid not in self._person_vlm_data:
                        self._person_vlm_data[tid] = scene_result

                vlm_text = self._person_vlm_data.get(tid, "")
                od_text = od_result if od_result and "QA>" not in od_result else ""
                obj_names = [o["name"] for o in self._objects[:10]] if self._objects else []
                action_label = fast_result["action"]
                timeline = self.action_engine.get_person_timeline(tid)
                face = self._face_ids.get(tid, "")
                reid = self._reid_ids.get(tid, "")

                cv_context_parts = []
                action_info = action_results.get(tid, {})
                if action_info:
                    act = action_info.get("action", "")
                    conf = action_info.get("confidence", 0)
                    src = action_info.get("source", "")
                    if act:
                        cv_context_parts.append(f"CV Action: {act} (conf={conf:.2f}, src={src})")

                if tid in poses and poses[tid] is not None:
                    kp = poses[tid]
                    cv_context_parts.append(f"CV Pose: {len(kp)} joints tracked")

                dp = depth_per_person.get(tid, {})
                if dp:
                    cv_context_parts.append(f"CV Depth: torso={dp.get('torso_depth',0):.3f} mean={dp.get('mean_depth',0):.3f}")

                hd = hand_data.get(tid, [])
                for h in hd:
                    hand_state = "gripping" if h.get("is_grip") else ("open palm" if h.get("is_open") else "neutral")
                    cv_context_parts.append(f"CV Hand ({h.get('handedness','?')}): {hand_state}")

                gd = gaze_data.get(tid, {})
                if gd:
                    gtarget = gaze_targets.get(tid)
                    gaze_str = f"CV Gaze: {gd.get('direction','?')} ({gd.get('yaw',0)} deg)"
                    if gtarget is not None:
                        gaze_str += f" targeting Person_{gtarget}"
                    cv_context_parts.append(gaze_str)

                for (pa, pb), cd in contact_data.items():
                    if cd.get("contact") and (tid == pa or tid == pb):
                        other = pb if tid == pa else pa
                        cv_context_parts.append(f"CV Contact: with Person_{other} (score={cd.get('score',0):.2f})")

                fp = flow_per_person.get(tid)
                if fp:
                    cv_context_parts.append(f"CV Flow: mean={fp.get('mean_magnitude',0):.2f} max={fp.get('max_magnitude',0):.2f}")

                cv_context = "; ".join(cv_context_parts) if cv_context_parts else "no CV data"

                parts = []
                parts.append(f"=== Person_{tid} Analysis ===")
                if vlm_text and len(vlm_text) > 20:
                    parts.append(f"[VLM Visual Description]: {vlm_text}")
                if od_text and len(od_text) > 20:
                    parts.append(f"[VLM Objects Detected]: {od_text}")
                if cv_context_parts:
                    parts.append(f"[CV Pipeline Data]: {cv_context}")
                parts.append(f"[YOLO Scene Objects]: {', '.join(obj_names) if obj_names else 'none'}")
                if face:
                    parts.append(f"[Face ID]: {face}")
                if reid:
                    parts.append(f"[Re-ID]: {reid}")
                parts.append(f"[Action Timeline]: {' -> '.join(a['action_type'] for a in timeline[-8:] if a.get('action_type')) or 'no history'}")

                grounded_caption = "\n".join(parts)

                if vlm_text and len(vlm_text) > 50:
                    expanded = self._llm_expand_caption(tid, vlm_text, cv_context, obj_names, action_label)
                    if expanded and len(expanded) > len(grounded_caption):
                        grounded_caption = (
                            f"=== Person_{tid} Rich Analysis (CV + VLM + LLM) ===\n"
                            f"{expanded}\n"
                            f"[Raw VLM]: {vlm_text[:200]}\n"
                            f"[CV Data]: {cv_context}"
                        )
                        self._person_vlm_data[tid] = expanded

                prev = self._captions.get(tid, "")
                is_new = not prev
                vlm_clean = vlm_text.strip().rstrip(".") if vlm_text else ""
                changed_vlm = vlm_clean and (vlm_clean not in prev)
                new_better = (
                    is_new
                    or (vlm_clean and not any(marker in prev for marker in ("Appears to be:", "Description:")))
                    or changed_vlm
                )

                if new_better and (is_new or changed_vlm or action_label not in ("standing", "active")):
                    self._captions[tid] = grounded_caption
                    if is_new:
                        logger.info(f"[NEW]  Person_{tid} | {grounded_caption}")
                    elif changed_vlm:
                        logger.info(f"[VLM]  Person_{tid} | {vlm_text[:300]}")

                    self.action_engine.log_action(
                        tid, action_label, grounded_caption, bbox,
                        {"event": "caption", "vlm": vlm_text,
                         "objects": obj_names, "action": action_label}
                    )
                    embedding = self.vss.store(tid, grounded_caption)
                    if embedding is not None and embedding.size > 0:
                        meta = {
                            "action": action_label,
                            "time": frame_time,
                            "pose": f"{poses.get(tid).shape if tid in poses else 'none'}",
                            "depth": str(depth_per_person.get(tid, {})),
                            "gaze": str(gaze_data.get(tid, {})),
                            "contact": str(contact_data.get((min(tid, ot), max(tid, ot)), {})) if contact_data else "none",
                        }
                        if obj_names:
                            meta["objects"] = ", ".join(obj_names[:10])
                        try:
                            self.vector_store.store(tid, embedding.tolist(), grounded_caption, meta)
                            self.vector_store.store_frame(
                                self._vid_frame, frame_time,
                                {tid: {
                                    "action": action_label,
                                    "contact": str(contact_data) if contact_data else "",
                                    "gaze": str(gaze_data.get(tid, {}).get("direction", "")) if gaze_data else "",
                                    "depth": str(depth_per_person.get(tid, {})) if depth_per_person else "",
                                    "hand": str(hand_data.get(tid, [])) if hand_data else "",
                                }},
                                grounded_caption,
                            )
                        except Exception:
                            pass
                    self.graph.add_person_node(tid, grounded_caption, {"objects": obj_names})
                    self.graph.parse_caption_for_graph(tid, grounded_caption)
                    self.sqlite.log_event("caption", tid, {"caption": grounded_caption, "time": frame_time})
                    self.sqlite.upsert_node(f"Person_{tid}", "Person", grounded_caption[:500])

                    threat_words = ["knife", "weapon", "gun", "steal", "theft", "robbery", "fight",
                                    "break-in", "force entry", "lock", "suspicious behavior",
                                    "threat", "violen", "attack", "baseball bat", "scissors",
                                    "crouching", "hiding", "running away"]
                    all_text = grounded_caption.lower()
                    hits = [w for w in threat_words if w in all_text]
                    if hits:
                        alert_key = f"{tid}:{','.join(sorted(hits))}"
                        if alert_key not in getattr(self, "_fired_alerts", set()):
                            if not hasattr(self, "_fired_alerts"):
                                self._fired_alerts = set()
                            self._fired_alerts.add(alert_key)
                            alert_msg = f"ALERT ({','.join(hits)}): {grounded_caption[:300]}"
                            logger.warning(f"[ALERT] {alert_msg}")
                            self.webhook.send({"alert": alert_msg, "track_id": tid})
                            self.sqlite.log_event("alert", tid, {"alert": alert_msg, "threats": hits, "time": frame_time})

                    sim_matches = self.vss.search_similar(tid, grounded_caption)
                    if sim_matches:
                        self._vss_matches[tid] = sim_matches

                self.action_engine.log_action(
                    tid, fast_result["action"],
                    f"{fast_result['action']} (move={fast_result['movement_px']}px)",
                    bbox,
                    {
                        "frame_time": frame_time,
                        "movement_px": fast_result["movement_px"],
                        "confidence": fast_result["confidence"],
                        "changed": fast_result["changed"],
                    }
                )

                if crop is not None and crop.size > 0:
                    if self.face_recognizer.is_available and self._frame_skip % 3 == 0:
                        face_results = self.face_recognizer.process_person(tid, crop)
                        if face_results:
                            for fr in face_results:
                                self._face_ids[tid] = fr.get("face_id", "")
                                self.session_manager.add_identity(fr.get("face_id", ""))

                    if self.reid.is_ready and self._frame_skip % 5 == 0:
                        reid_result = self.reid.match_identity(tid, crop)
                        if reid_result:
                            self._reid_ids[tid] = reid_result["global_id"]
                            self.session_manager.add_identity(reid_result["global_id"])

                else:
                    pass

            if not self._turbo and self._vid_frame % 50 == 0:
                if not hasattr(self, "_scene_vlm_time") or frame_time - self._scene_vlm_time > 20.0:
                    self._scene_vlm_time = frame_time
                    self.vlm.submit_full_scene("full_scene", frame)

            for (ta, tb), cd in contact_data.items():
                if cd.get("contact"):
                    self.alerts.evaluate_interaction(ta, tb)
                    self.action_engine.log_action(
                        ta, "contact",
                        f"Contact with Person_{tb} (score={cd.get('score', 0):.2f})",
                        persons[0]["bbox"] if persons else (0, 0, 0, 0),
                        {"with": tb, "contact_data": cd}
                    )
                    self.graph.add_edge(f"Person_{ta}", f"Person_{tb}", "CONTACT_WITH")

            for obj in objects:
                self.graph.add_object_node(obj["name"])
                self.graph.add_edge(f"Obj_{obj['name']}", f"Obj_{obj['name']}", "DETECTED_IN_SCENE")

            model_status = {
                "detection": True,
                "pose": self.pose_estimator.is_ready,
                "depth": self.depth_estimator.is_ready,
                "flow": self.flow_estimator.is_ready,
                "contact": self.contact_detector.enabled,
                "seg": self.segmentation.is_ready,
                "hand": self.hand_tracker.is_ready,
                "gaze": self.gaze_estimator.is_ready,
                "action": self.action_recognizer.is_ready,
                "vlm": not self._turbo,
                "causal": self.causal_extractor.enabled,
            }

            stats = {
                "fps": frame_prober.avg_ms,
                "person_count": len(persons),
                "alert_count": self.alerts.alert_count,
                "project_progress": project_tracker.get_status_display(),
                "face_ids": self._face_ids,
                "reid_ids": self._reid_ids,
                "objects": self._objects,
                "action_summary": self.action_engine.get_scene_summary(60),
                "poses": poses,
                "depth_map": depth_map,
                "depth_per_person": depth_per_person,
                "flow_map": flow,
                "flow_per_person": flow_per_person,
                "contact_data": contact_data,
                "seg_masks": seg_masks,
                "hand_data": hand_data,
                "gaze_data": gaze_data,
                "gaze_targets": gaze_targets,
                "action_results": action_results,
                "model_status": model_status,
                "gate_stats": self.frame_gate.get_stats(),
                "yolo_model": f"{self.detector.model_version}",
                "causal_summary": self.causal_extractor.get_summary(),
            }

            rendered = self.display.render(
                frame, persons, self._captions, self._vqa_answers, self._vss_matches, stats
            )

            self.summary_engine.update({
                "current_persons": len(persons),
                "total_alerts": self.alerts.alert_count,
            })

            self.session_manager.update_stat("total_persons", len(persons))

            if self.summary_engine.should_summarize(SUMMARY_INTERVAL):
                self.graph.purge_old()
                gs = self.graph.get_stats()
                self.summary_engine.generate(gs)
                project_tracker.log_progress()
                session_summary = self.session_manager.get_current_summary()
                logger.info(
                    f"[SESSION] ID: {session_summary.get('session_id','?')} | "
                    f"Identities: {session_summary.get('stats',{}).get('unique_identities',0)} | "
                    f"Duration: {session_summary.get('duration',0):.0f}s"
                )

            if not getattr(self, "_headless", False):
                if not self.display.show(rendered):
                    logger.info("[ARGUS] User pressed 'q'")
                    break

            self._prev_frame = frame
            frame_prober.stop()

    def _llm_expand_caption(
        self, tid: int, vlm_text: str, cv_context: str, obj_names: list, action: str
    ) -> str:
        if not self.groq_chat._available:
            return ""
        try:
            prompt = (
                f"You are analyzing CCTV footage. Expand this into a detailed 200-300 word paragraph describing "
                f"everything about Person_{tid}.\n\n"
                f"VLM visual description: {vlm_text}\n"
                f"CV pipeline data: {cv_context}\n"
                f"Detected objects: {', '.join(obj_names) if obj_names else 'none'}\n"
                f"Current action: {action}\n\n"
                f"Describe: full appearance, clothing details, exact posture, what they are doing, "
                f"objects they hold or interact with, their body language, facial expression, "
                f"gaze direction, hand movements, position relative to scene elements, "
                f"any contact with other people, movement patterns, and environmental context. "
                f"Use ALL the CV and VLM data provided. Write 200-300 words as a single flowing paragraph."
            )
            result = self.groq_chat._gather_and_ask_direct(prompt)
            if result and len(result) > 100:
                return result
        except Exception:
            pass
        return ""

    def _dump_final_report(self) -> None:
        logger.info("=" * 60)
        logger.info("  FINAL CCTV ANALYSIS REPORT")
        logger.info("=" * 60)
        logger.info(f"  Total persons detected: {len(self._captions)}")
        logger.info(f"  Total actions logged: {len(self.action_engine.actions)}")
        logger.info(f"  YOLO model: {self.detector.model_version} ({self.detector.model_name})")
        logger.info(f"  Frame gate: {self.frame_gate.get_stats()}")

        causal_summary = self.causal_extractor.get_summary()
        logger.info(f"  Causal variables: {causal_summary.get('unique_variables', 0)}")
        logger.info(f"  Contact events: {causal_summary.get('total_contact_events', 0)}")
        logger.info(f"  Total frames processed: {causal_summary.get('total_frames', 0)}")

        for tid, caption in sorted(self._captions.items()):
            logger.info(f"  --- Person_{tid} ---")
            logger.info(f"    {caption}")
            face = self._face_ids.get(tid, "")
            reid = self._reid_ids.get(tid, "")
            if self._person_vlm_data.get(tid):
                logger.info(f"    VLM: {self._person_vlm_data[tid]}")
        if not self._captions:
            logger.info("  No persons detected in this video.")
        logger.info("=" * 60)

    def run_headless(self) -> None:
        self._headless = True
        logger.info("[ARGUS] Starting headless mode...")
        self.vlm.start_worker()
        self.vss.load()
        self._running = True
        self._run_loop()
        self._interactive_qa()

    def _interactive_qa(self) -> None:
        if not self._captions:
            logger.info("[QA] No persons detected — nothing to query.")
            return
        logger.info("")
        logger.info("=" * 50)
        logger.info("  Interactive Q&A — ask about the video")
        logger.info("  Commands: report | persons | objects | alerts | actions N | q")
        logger.info("  Or just ask natural questions (uses Groq AI)")
        logger.info("=" * 50)
        logger.info("")

        while True:
            try:
                question = input(">>> ").strip()
                if not question:
                    continue
                if question.lower() in ("q", "quit", "exit"):
                    self._dump_final_report()
                    break
                if question.lower() == "report":
                    self._dump_final_report()
                    continue
                if question.lower() == "persons":
                    for tid in sorted(self._captions):
                        tl = self.action_engine.get_person_timeline(tid)
                        acts = [a["action_type"] for a in tl[-5:] if a.get("action_type")]
                        logger.info(f"  Person_{tid}: {' -> '.join(acts) if acts else 'no actions'}")
                    continue
                if question.lower() == "objects":
                    objs = [o["name"] for o in self._objects[:15]] if self._objects else []
                    logger.info(f"  Objects: {', '.join(set(objs))}" if objs else "  No objects recorded")
                    continue
                if question.lower() == "alerts":
                    alerts = self.sqlite.get_recent_events(limit=50)
                    alert_events = [e for e in alerts if e.get("event_type") == "alert"]
                    if alert_events:
                        for a in alert_events[:10]:
                            logger.info(f"  ALERT: {a}")
                    else:
                        logger.info("  No alerts triggered")
                    continue
                if question.lower().startswith("actions "):
                    try:
                        tid = int(question.split()[1])
                        tl = self.action_engine.get_person_timeline(tid)
                        for a in tl[-20:]:
                            logger.info(f"  {a.get('time_str','?')} | {a.get('action_type','?')}")
                    except (IndexError, ValueError):
                        logger.info("  Usage: actions <track_id>")
                    continue

                self._chat_answer(question)
            except (EOFError, KeyboardInterrupt):
                break

    def _get_pose_preview_for_gate(self, current_ids: set) -> Optional[Dict[int, np.ndarray]]:
        if not hasattr(self, "_last_gate_poses"):
            self._last_gate_poses: Dict[int, np.ndarray] = {}
        return self._last_gate_poses if self._last_gate_poses else None

    def _set_pose_preview_for_gate(self, poses: Dict[int, np.ndarray]) -> None:
        self._last_gate_poses = {k: v.copy() for k, v in poses.items()}

    def _get_contact_preview_for_gate(self) -> Optional[Dict]:
        return self.contact_detector.get_active_contacts() if CONTACT_ENABLED else None

    def _chat_answer(self, question: str) -> None:
        result = self.groq_chat.ask(
            question=question,
            graph_store=self.graph,
            vector_store=self.vector_store,
            sqlite_store=self.sqlite,
            action_engine=self.action_engine,
            captions=self._captions,
            objects=self._objects,
            causal_extractor=self.causal_extractor if CAUSAL_ENABLED else None,
            contact_detector=self.contact_detector if CONTACT_ENABLED else None,
        )
        logger.info("")
        logger.info(f"Q: {question}")
        if result.get("source") == "groq":
            logger.info(f"A: {result['answer']}")
        elif result.get("source") == "fallback":
            if result.get("answer"):
                logger.info(f"A: {result['answer']}")
            else:
                logger.info("A: Groq not configured. Set $env:GROQ_API_KEY for AI chat.")
                if result.get("context"):
                    logger.info(f"   Context: {result['context'][:500]}")
        logger.info("")


def _discover_videos() -> List[Path]:
    videos: List[Path] = []
    for base_dir in (VIDEOS_DIR, INPUT_VID_DIR):
        if base_dir.exists():
            for ext in SUPPORTED_VIDEO_FORMATS:
                videos.extend(base_dir.glob(f"*{ext}"))
                videos.extend(base_dir.glob(f"*{ext.upper()}"))
    return sorted(set(videos))


def _resolve_video_path(raw: str) -> Optional[str]:
    if raw == "list":
        videos = _discover_videos()
        if not videos:
            print("No videos found in data/videos/ or input_vid/")
            print(f"Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}")
            print("Drop videos into data/videos/ or input_vid/ and run again.")
        else:
            print(f"Found {len(videos)} video(s):")
            for v in videos:
                size_mb = v.stat().st_size / (1024 * 1024)
                tag = "[input_vid]" if INPUT_VID_DIR in v.parents else "[data]"
                print(f"  {tag} {v.name}  ({size_mb:.1f} MB)")
        return None

    if raw == "auto":
        videos = _discover_videos()
        if not videos:
            print("No videos found in data/videos/ or input_vid/")
            return None
        picked = videos[0]
        tag = "[input_vid]" if INPUT_VID_DIR in picked.parents else "[data]"
        print(f"Auto-selected: {tag} {picked.name}")
        return str(picked)

    if raw == "all":
        videos = _discover_videos()
        if not videos:
            print("No videos found to process")
            return None
        return "<<ALL>>"

    video_path = Path(raw)
    if video_path.is_file():
        return str(video_path.resolve())

    resolved = VIDEOS_DIR / raw
    if resolved.is_file():
        return str(resolved.resolve())

    resolved = INPUT_VID_DIR / raw
    if resolved.is_file():
        return str(resolved.resolve())

    print(f"Video file not found: {raw}")
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ARGUS Video Intelligence System")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display window",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        metavar="PATH|auto|list",
        help="Use a video file as source instead of live camera. "
             "PATH=path to video, auto=pick first from data/videos/, "
             "list=show available videos",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Ultra-fast mode: skip VLM, use only instant heuristic detection. "
             "Best for short action clips (<30s).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop all videos continuously (requires --video all).",
    )
    args = parser.parse_args()

    if args.dashboard:
        import subprocess
        dashboard_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        return

    turbo_mode = args.turbo or VLM_TURBO_MODE

    video_files: List[str] = []
    if args.video is not None:
        resolved = _resolve_video_path(args.video)
        if resolved is None:
            sys.exit(1)
        if resolved == "<<ALL>>":
            video_files = [str(p) for p in _discover_videos() if not p.name.startswith(".")]
        else:
            video_files = [resolved]
    else:
        video_files = [None]

    print(f"Loaded {len(video_files)} video(s) to process")
    if len(video_files) > 1:
        print("  " + "\n  ".join(Path(v).name for v in video_files))

    _run_video_list(video_files, turbo_mode, args.headless, args.loop)


def _run_video_list(
    video_files: List[Optional[str]],
    turbo_mode: bool,
    headless: bool,
    loop: bool,
) -> None:
    first = True
    while first or loop:
        first = False
        for vf in video_files:
            name = Path(vf).name if vf else "(live camera)"
            logger.info(f"[ARGUS] Processing: {name}")
            argus = ARGUS(video_file=vf, turbo=turbo_mode)

            def signal_handler(sig, frame):
                argus._running = False

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            try:
                if headless:
                    argus.run_headless()
                else:
                    argus.start()
            except KeyboardInterrupt:
                argus.stop()
                return
            except Exception as e:
                logger.error(f"[ARGUS] Fatal error: {e}", exc_info=True)
            finally:
                argus.stop()

            if not loop:
                break  # process only one video in non-loop mode
        if not loop:
            break


if __name__ == "__main__":
    main()
