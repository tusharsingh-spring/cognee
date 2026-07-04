"""
ARGUS V3 — 3-Layer Video Intelligence System

LAYER 1: PERCEPTION — YOLOv11x + Pose + Action + Gaze + Depth + Flow + Contact + Hands + Seg
LAYER 2: VLM (Florence-2) — Visual context that structured models miss
LAYER 3: LOCAL LLM — Narrative, intent, anomaly, notification decision

KNOWLEDGE: Cognee Graph RAG + ChromaDB Vector Store + SQLite Event Log
UI: OpenCV Display + Streamlit Chat + Streamlit Dashboard
"""

import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    ACTION_RECOG_ENABLED,
    AUDIO_ENABLED,
    CAUSAL_ENABLED,
    CONTACT_ENABLED,
    DEPTH_ENABLED,
    DISPLAY_ENABLED,
    FLOW_ENABLED,
    GAZE_ENABLED,
    GATE_ENABLED,
    HAND_ENABLED,
    INPUT_VID_DIR,
    POSE_ENABLED,
    SEG_ENABLED,
    SUMMARY_INTERVAL,
    SUPPORTED_VIDEO_FORMATS,
    VIDEOS_DIR,
    VLM_ENABLED,
    VLM_TURBO_MODE,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger("ARGUS")


class ARGUS:
    """3-Layer Intelligence Pipeline: Perception → VLM → LLM → Knowledge Graph"""

    def __init__(self, video_file: Optional[str] = None, turbo: bool = False) -> None:
        logger.info("=" * 60)
        logger.info("  ARGUS V3 — 3-Layer Video Intelligence System")
        if turbo:
            logger.info("  [TURBO MODE] — VLM + LLM disabled, heuristic-only")
        logger.info("=" * 60)

        self._turbo = turbo
        self._running = False
        self._stopped = False

        # ── LAYER 1: Perception ──
        from layer1_perception.perception_pipeline import PerceptionPipeline
        self.perception = PerceptionPipeline()

        from layer1_perception.gating import YOLOv8nGate
        self.gate = YOLOv8nGate()

        from pipeline.capture import MotionCapture
        self.capture = MotionCapture(source=video_file)

        from pipeline.display import DisplayManager
        self.display = DisplayManager()

        from pipeline.face_recognition import FaceRecognizer
        self.face_recognizer = FaceRecognizer()

        from pipeline.reid_handler import ReIDHandler
        self.reid = ReIDHandler()

        from pipeline.fast_actions import FastActionDetector
        self.fast_actions = FastActionDetector(history_frames=30)

        from pipeline.action_engine import TemporalActionEngine
        self.action_engine = TemporalActionEngine()

        from pipeline.causal_extractor import CausalExtractor
        self.causal_extractor = CausalExtractor()

        # ── LAYER 2: VLM (on important frames only) ──
        self._vlm = None
        self._vlm_trigger = None
        if VLM_ENABLED:
            from layer2_vlm.vlm_engine import VLMEngine
            self._vlm = VLMEngine()
            from layer2_vlm.vlm_trigger import VLMTriggerManager
            self._vlm_trigger = VLMTriggerManager()

        # ── LAYER 3: LLM (DISABLED in pipeline — Groq reserved for chatbot UI) ──
        self._llm = None
        self._llm_reason_counter = 0

        # ── KNOWLEDGE STORES ──
        from graph_rag.cognee_bridge import CogneeBridge
        self.cognee = CogneeBridge()

        from storage.vector_store import VectorStore
        self.vector_store = VectorStore()

        from storage.sqlite_store import SQLiteStore
        self.sqlite = SQLiteStore()

        from notifications.alert_engine import AlertEngine
        self.alerts = AlertEngine()

        from notifications.webhook import WebhookNotifier
        self.webhook = WebhookNotifier()

        from knowledge.summary_engine import SummaryEngine
        self.summary_engine = SummaryEngine()

        from knowledge.session_manager import SessionManager
        self.session_manager = SessionManager()

        from knowledge.cctv_qa import CCTVQA
        self.cctv_qa = CCTVQA()

        # ── State ──
        self._captions: Dict[int, str] = {}
        self._vlm_data: Dict[int, str] = {}
        self._face_ids: Dict[int, str] = {}
        self._reid_ids: Dict[int, str] = {}
        self._prev_frame: Optional[np.ndarray] = None
        self._vid_frame: int = 0
        self._headless: bool = False
        self._frame_time: float = 0.0
        self._person_seen: set = set()
        self._person_vlm_times: Dict[int, float] = {}
        self._last_llm_reason: float = 0.0

        self.face_recognizer.load()
        self.reid.load()
        self.session_manager.start_session()

    def start(self) -> None:
        logger.info("[ARGUS] Starting 3-layer pipeline...")

        if self._vlm is not None:
            try:
                self._vlm.start_worker()
                logger.info("[VLM] Layer 2 worker started")
            except Exception as e:
                logger.warning(f"[VLM] Failed to start: {e} — continuing without VLM")
                self._vlm = None

        logger.info(f"[PERCEPTION] Layer 1: {self.perception.model_status}")
        logger.info(f"[COGNEE] Graph RAG: {self.cognee.get_stats()}")

        self._running = True
        self._run_loop()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        logger.info("[ARGUS] Shutting down...")
        if self._vlm is not None:
            self._vlm.stop_worker()
        self.capture.release()
        self.display.close()
        self.cognee.save()
        self.sqlite.close()
        self.session_manager.end_session()
        logger.info("[ARGUS] Shutdown complete")

    def _run_loop(self) -> None:
        while self._running:
            frame_prober = profiler.get("frame_total")
            frame_prober.start()

            ret, frame, motion_boxes = self.capture.read()
            if not ret:
                if self.capture._is_video_file:
                    logger.info("[ARGUS] Video ended, generating report...")
                    self._dump_final_report()
                    self._running = False
                    break
                time.sleep(0.5)
                continue

            self._vid_frame += 1
            self._frame_time = time.time()
            h, w = frame.shape[:2]

            # ── GATE: YOLOv8n skips idle frames ──
            should_process = not GATE_ENABLED
            if GATE_ENABLED:
                should_process = self.gate.should_process(frame)[0]

            if not should_process:
                self._prev_frame = frame
                if DISPLAY_ENABLED and not self._headless:
                    if not self.display.show(frame):
                        break
                frame_prober.stop()
                continue

            # ── LAYER 1: PERCEPTION (all models in parallel) ──
            packet = self.perception.process(frame)

            # ── Feed Cognee Graph RAG ──
            self.cognee.ingest_perception(packet, frame_time=self._frame_time)

            # ── Face Recognition + Re-ID ──
            for person in packet.persons:
                tid = person.track_id
                p_info_raw = None
                bbox = (int(person.bbox[0]), int(person.bbox[1]), int(person.bbox[2]), int(person.bbox[3]))
                x1, y1, x2, y2 = bbox
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                if crop.size > 0 and self._vid_frame % 3 == 0:
                    if self.face_recognizer.is_available:
                        try:
                            face_results = self.face_recognizer.process_person(tid, crop)
                            if face_results:
                                for fr in face_results:
                                    self._face_ids[tid] = fr.get("face_id", "")
                                    self.session_manager.add_identity(fr.get("face_id", ""))
                        except Exception:
                            pass

                if crop.size > 0 and self._vid_frame % 5 == 0:
                    if self.reid.is_ready:
                        try:
                            reid_result = self.reid.match_identity(tid, crop)
                            if reid_result:
                                self._reid_ids[tid] = reid_result["global_id"]
                                self.session_manager.add_identity(reid_result["global_id"])
                        except Exception:
                            pass

            # ── Build CV context string per person ──
            cv_contexts: Dict[int, str] = {}
            for person in packet.persons:
                tid = person.track_id
                parts = []
                if tid in packet.actions:
                    a = packet.actions[tid]
                    parts.append(f"action: {a.action} (conf={a.confidence:.2f})")
                if tid in packet.poses:
                    parts.append(f"pose: {packet.poses[tid].visible_count}/17 joints")
                if tid in packet.gaze:
                    parts.append(f"gaze: {packet.gaze[tid].direction}")
                if tid in packet.depth:
                    parts.append(f"depth: {packet.depth[tid].torso_depth:.2f}")
                if tid in packet.flow:
                    parts.append(f"flow: {packet.flow[tid].mean_magnitude:.1f}px")
                cv_contexts[tid] = "; ".join(parts) if parts else "no data"

            # ── LAYER 2: VLM (submit per-person with VLMTriggerManager gating) ──
            if self._vlm is not None:
                have_contact = any(c.contact for c in packet.contacts if hasattr(packet, 'contacts'))
                for person in packet.persons:
                    tid = person.track_id
                    bbox = (int(person.bbox[0]), int(person.bbox[1]), int(person.bbox[2]), int(person.bbox[3]))
                    x1, y1, x2, y2 = bbox

                    pad = int((x2 - x1) * 0.1)
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    crop = frame[cy1:cy2, cx1:cx2]

                    if crop.size == 0:
                        continue

                    is_new = tid not in self._person_seen
                    current_action = packet.actions[tid].action if tid in packet.actions else "standing"
                    self._vlm_trigger.register_person(tid)

                    should, reason = self._vlm_trigger.should_trigger(
                        tid, current_action, is_new_person=is_new,
                        has_contact=have_contact,
                    )
                    if not should:
                        continue

                    per_context = cv_contexts.get(tid, "")
                    obj_names = [o.name for o in packet.objects[:8]] if packet.objects else []
                    if obj_names:
                        per_context += f"; objects seen: {', '.join(obj_names)}"

                    submitted = self._vlm.submit_with_perception(tid, crop, per_context, task="dense")
                    if submitted:
                        logger.info(f"[VLM] Triggered for Person_{tid} reason={reason}")

                if self._vid_frame % 50 == 0:
                    self._vlm.submit_full_scene("full_scene", frame, packet.to_context_string())

            # ── Build captions & store events ──
            for person in packet.persons:
                tid = person.track_id
                bbox = (int(person.bbox[0]), int(person.bbox[1]), int(person.bbox[2]), int(person.bbox[3]))

                if tid not in self._person_seen:
                    self._person_seen.add(tid)
                    logger.info(f"[NEW] Person_{tid} appeared (conf={person.confidence:.2f})")

                vlm_text = self._vlm_data.get(tid, "")
                if self._vlm is not None:
                    fresh = self._vlm.get_result(tid, "dense") or ""
                    if fresh:
                        vlm_text = fresh
                        self._vlm_data[tid] = fresh

                obj_names = [o.name for o in packet.objects[:10]] if packet.objects else []
                action = packet.actions[tid].action if tid in packet.actions else "standing"
                action_conf = packet.actions[tid].confidence if tid in packet.actions else 0
                cv_ctx = cv_contexts.get(tid, "no data")
                face_id = self._face_ids.get(tid, "")
                reid = self._reid_ids.get(tid, "")

                caption_parts = [f"=== Person_{tid} Analysis ==="]
                caption_parts.append(f"[Perception]: {cv_ctx}")
                if vlm_text and len(vlm_text) > 10:
                    caption_parts.append(f"[VLM]: {vlm_text[:300]}")
                if obj_names:
                    caption_parts.append(f"[Objects]: {', '.join(obj_names)}")
                if face_id:
                    caption_parts.append(f"[Face]: {face_id}")
                if reid:
                    caption_parts.append(f"[Re-ID]: {reid}")

                caption = "\n".join(caption_parts)

                is_new = tid not in self._captions
                changed = vlm_text and (vlm_text not in self._captions.get(tid, ""))
                if is_new or changed:
                    self._captions[tid] = caption
                    self.action_engine.log_action(tid, action, caption, bbox, {
                        "vlm": vlm_text, "objects": obj_names, "frame_time": self._frame_time
                    })
                else:
                    self.action_engine.log_action(tid, action, cv_ctx, bbox, {"frame_time": self._frame_time})

                self.cognee.ingest_vlm_output(tid, vlm_text or "", frame_time=self._frame_time)

                if caption:
                    try:
                        self.vector_store.store_event("perception", {"track_id": tid, "action": action}, caption)
                        self.sqlite.log_event("frame", tid, {"caption": caption[:500], "time": self._frame_time})
                        self.sqlite.upsert_node(f"Person_{tid}", "Person", caption[:500])
                    except Exception as e:
                        logger.debug(f"[STORE] Error: {e}")

                # Threat detection
                threat_words = ["knife", "weapon", "gun", "steal", "theft", "robbery", "fight",
                                "break-in", "force entry", "suspicious behavior", "threat",
                                "violen", "attack", "baseball bat", "scissors", "crouching",
                                "hiding", "running away"]
                all_text = caption.lower()
                hits = [w for w in threat_words if w in all_text]
                if hits:
                    alert_key = f"{tid}:{','.join(sorted(hits))}"
                    if not hasattr(self, "_fired_alerts"):
                        self._fired_alerts = set()
                    if alert_key not in self._fired_alerts:
                        self._fired_alerts.add(alert_key)
                        alert_msg = f"ALERT ({','.join(hits)}): Person_{tid}"
                        logger.warning(f"[ALERT] {alert_msg}")
                        self.webhook.send({"alert": alert_msg, "track_id": tid})
                        self.sqlite.log_event("alert", tid, {"alert": alert_msg, "threats": hits})

            # ── Contact handling ──
            for contact in packet.contacts:
                if contact.contact:
                    self.alerts.evaluate_interaction(contact.person_a, contact.person_b)
                    self.action_engine.log_action(
                        contact.person_a, "contact",
                        f"Contact with Person_{contact.person_b} (score={contact.score:.2f})",
                        (0, 0, 0, 0),
                        {"with": contact.person_b}
                    )

            # ── Causal extraction ──
            if CAUSAL_ENABLED and packet.persons:
                person_ids = [p.track_id for p in packet.persons]
                poses_dict = {tid: np.array(k.keypoints) for tid, k in packet.poses.items()}
                depth_dict = {tid: {"torso_depth": d.torso_depth} for tid, d in packet.depth.items()}
                flow_dict = {tid: {"mean_magnitude": f.mean_magnitude} for tid, f in packet.flow.items()}
                contact_dict = {(c.person_a, c.person_b): {"contact": c.contact, "score": c.score} for c in packet.contacts}
                gaze_dict = {tid: {"direction": g.direction} for tid, g in packet.gaze.items()}
                hand_dict = {tid: [{"handedness": h.handedness} for h in hl] for tid, hl in packet.hands.items()}
                action_dict = {tid: {"action": a.action} for tid, a in packet.actions.items()}
                objects_raw = [{"class_id": o.class_id, "name": o.name} for o in packet.objects]

                self.causal_extractor.extract(
                    self._vid_frame, self._frame_time, list(set(person_ids)),
                    poses_dict, depth_dict, flow_dict, contact_dict,
                    hand_dict, gaze_dict, action_dict, objects_raw,
                )

            # ── Summary ──
            self.summary_engine.update({"current_persons": len(packet.persons), "total_alerts": self.alerts.alert_count})
            self.session_manager.update_stat("total_persons", len(packet.persons))

            if self.summary_engine.should_summarize(SUMMARY_INTERVAL):
                self.cognee.detect_patterns()
                self.cognee.purge_old()
                gs = self.cognee.get_stats()
                self.summary_engine.generate(gs)

            # ── Display ──
            if DISPLAY_ENABLED and not self._headless:
                stats = {
                    "fps": frame_prober.avg_ms,
                    "person_count": len(packet.persons),
                    "alert_count": self.alerts.alert_count,
                    "model_status": self.perception.model_status,
                    "gate_stats": {"frame_count": self.gate._frame_count} if hasattr(self, "gate") else {},
                }
                rendered = self.display.render(frame, [], self._captions, {}, {}, stats)
                if not self.display.show(rendered):
                    logger.info("[ARGUS] User pressed 'q'")
                    break

            self._prev_frame = frame.copy()
            frame_prober.stop()

    def _gather_short_term_memory(self, tid: int) -> str:
        events = self.sqlite.get_recent_events(limit=10)
        lines = []
        for e in events:
            lines.append(f"[{e.get('timestamp', '?')}] {e.get('event_type', '?')}: track={e.get('track_id', '?')}")
        return "\n".join(lines) if lines else "none"

    def _gather_medium_term_memory(self) -> str:
        events = self.cognee.get_recent_events(10)
        lines = []
        for e in events:
            if isinstance(e, dict):
                etype = e.get("type", e.get("event_type", "?"))
                lines.append(f"[{e.get('timestamp', '?')}]: {etype}")
            else:
                lines.append(str(e)[:150])
        return "\n".join(lines) if lines else "none"

    def _gather_long_term_baselines(self) -> str:
        parts = []
        try:
            events = self.sqlite.get_recent_events(limit=100)
            if events:
                actions = [e for e in events if e.get("event_type") == "frame"]
                alerts = [e for e in events if e.get("event_type") == "alert"]
                persons_seen = set()
                for e in events:
                    tid = e.get("track_id")
                    if tid:
                        persons_seen.add(tid)
                parts.append(f"{len(persons_seen)} unique persons seen in this session")
                parts.append(f"{len(actions)} frames logged, {len(alerts)} alerts")
                if events:
                    first_ts = events[-1].get("timestamp", 0)
                    last_ts = events[0].get("timestamp", 0)
                    if first_ts and last_ts and last_ts > first_ts:
                        duration = last_ts - first_ts
                        parts.append(f"Session duration: {duration:.0f}s")
        except Exception:
            pass
        try:
            periodics = self.summary_engine._last_summary
            if periodics:
                parts.append(f"Last periodic summary: {str(periodics)[:300]}")
        except Exception:
            pass
        return "\n".join(parts) if parts else "no baseline established yet"

    def _gather_identity_context(self, tid: int) -> str:
        parts = []
        face_id = self._face_ids.get(tid)
        reid = self._reid_ids.get(tid)
        if face_id:
            parts.append(f"Face: {face_id}")
        if reid:
            parts.append(f"Re-ID: {reid}")
        history = self.cognee.get_person_history(tid)
        if history.get("actions"):
            parts.append(f"Past actions: {len(history['actions'])}")
        elif history.get("events"):
            parts.append(f"Past events: {len(history['events'])}")
        return "; ".join(parts) if parts else "unknown"

    def _handle_llm_reasoning(self, tid: int, reasoning: dict, frame_time: float = 0.0) -> None:
        narrative = reasoning.get("narrative", "")
        anomaly = reasoning.get("anomaly_score", 0.0)
        notify = reasoning.get("notify", False)
        urgency = reasoning.get("urgency", "none")
        notification_text = reasoning.get("notification_text", "")
        tags = reasoning.get("store_tags", [])
        intent = reasoning.get("intent", "")

        self.cognee.ingest_llm_output(tid, reasoning, frame_time=frame_time)

        logger.info(f"[LLM] Person_{tid} narrative: {narrative[:200]}")
        if anomaly > 0.5:
            logger.info(f"[LLM] Person_{tid} anomaly={anomaly:.2f}, notify={notify}, urgency={urgency}")

        if notify and notification_text:
            self.webhook.send({
                "alert": notification_text,
                "track_id": tid,
                "urgency": urgency,
                "anomaly_score": anomaly,
                "intent": intent,
            })
            self.sqlite.log_event("llm_alert", tid, reasoning)

        if tags:
            for tag in tags:
                self.cognee.log_event("tag", tid, {"tag": tag, "timestamp": frame_time}, frame_time=frame_time)

    def _dump_final_report(self) -> None:
        logger.info("=" * 60)
        logger.info("  FINAL CCTV ANALYSIS REPORT")
        logger.info("=" * 60)
        logger.info(f"  Total persons: {len(self._captions)}")
        logger.info(f"  Actions logged: {len(self.action_engine.actions)}")
        logger.info(f"  Cognee graph: {self.cognee.get_stats()}")

        for tid, caption in sorted(self._captions.items()):
            logger.info(f"  --- Person_{tid} ---")
            logger.info(f"    {caption[:400]}")

        self.causal_extractor.export_json()
        logger.info("=" * 60)


# ── Entry point ──

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
        else:
            print(f"Found {len(videos)} video(s):")
            for v in videos:
                size_mb = v.stat().st_size / (1024 * 1024)
                print(f"  {v.name} ({size_mb:.1f} MB)")
        return None
    if raw == "auto":
        videos = _discover_videos()
        return str(videos[0]) if videos else None
    if raw == "all":
        return "<<ALL>>"
    video_path = Path(raw)
    if video_path.is_file():
        return str(video_path.resolve())
    for d in (VIDEOS_DIR, INPUT_VID_DIR):
        resolved = d / raw
        if resolved.is_file():
            return str(resolved.resolve())
    return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ARGUS V3 — 3-Layer Video Intelligence")
    parser.add_argument("--headless", action="store_true", help="Run without display window")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--chat", action="store_true", help="Launch Streamlit chat UI")
    parser.add_argument("--video", type=str, metavar="PATH|auto|list", help="Video source")
    parser.add_argument("--turbo", action="store_true", help="Skip VLM + LLM, heuristic only")
    parser.add_argument("--webcam", action="store_true", help="Use local webcam (camera index 0)")
    parser.add_argument("--loop", action="store_true", help="Loop videos continuously")
    args = parser.parse_args()

    if args.dashboard:
        import subprocess
        dashboard_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        return

    if args.chat:
        import subprocess
        chat_path = Path(__file__).resolve().parent / "chat_ui" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(chat_path)])
        return

    turbo_mode = args.turbo or VLM_TURBO_MODE

    video_files: List[str] = []
    if args.webcam:
        video_files = ["0"]  # Force local webcam (overrides .env CAMERA_URL)
    elif args.video is not None:
        resolved = _resolve_video_path(args.video)
        if resolved is None:
            sys.exit(1)
        if resolved == "<<ALL>>":
            video_files = [str(p) for p in _discover_videos() if not p.name.startswith(".")]
        else:
            video_files = [resolved]
    else:
        video_files = [None]

    print(f"Processing {len(video_files)} video(s)")
    _run_video_list(video_files, turbo_mode, args.headless, args.loop)


def _run_video_list(video_files, turbo_mode, headless, loop):
    first = True
    while first or loop:
        first = False
        for vf in video_files:
            name = Path(vf).name if vf else "(live camera)"
            logger.info(f"[ARGUS] Processing: {name}")
            argus = ARGUS(video_file=vf, turbo=turbo_mode)
            if headless:
                argus._headless = True
            try:
                argus.start()
            except KeyboardInterrupt:
                argus.stop()
                return
            except Exception as e:
                logger.error(f"[ARGUS] Error: {e}", exc_info=True)
            finally:
                argus.stop()
        if not loop:
            break


if __name__ == "__main__":
    main()
