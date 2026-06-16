"""
ARGUS - Video Intelligence System
Main entry point that orchestrates all pipeline layers:
  1. Motion Detection (MOG2)
  2. Person Detection + Object Detection + Tracking (YOLOv8n + ByteTrack)
  3. Face Recognition + Person Re-Identification
  4. VLM Inference (Florence-2-large)
  5. Knowledge Graph + Vector Store + Graph RAG
  6. Alerts + Display + Session Management
"""

import signal
import sys
import threading
import time
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
)
from pipeline.capture import MotionCapture
from pipeline.detection import PersonDetector
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

        self._captions: Dict[int, str] = {}
        self._vqa_answers: Dict[int, List[Dict]] = {}
        self._vss_matches: Dict[int, List] = {}
        self._face_ids: Dict[int, str] = {}
        self._reid_ids: Dict[int, str] = {}
        self._objects: List[Dict] = []
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
                    self._running = False
                    break
                logger.warning("[ARGUS] Camera read failed, retrying...")
                time.sleep(0.5)
                continue

            h, w = frame.shape[:2]
            canvas = frame.copy()

            if self.capture._is_video_file and self._vid_frame % DENSE_FRAME_INTERVAL != 0:
                continue

            persons, annotated = self.detector.detect_and_track(frame)
            persons = [p for p in persons if p["track_id"] >= 0]
            objects = self.detector.detect_objects(frame) if self._frame_skip % 5 == 0 else self._objects
            self._objects = objects
            current_ids = {p["track_id"] for p in persons}
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
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_time = time.time()
                movement_delta = 0.0

                prev_bbox = getattr(self, "_prev_bboxes", {}).get(tid)
                if prev_bbox:
                    px1, py1, px2, py2 = prev_bbox
                    prev_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                    movement_delta = float(np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2))
                if not hasattr(self, "_prev_bboxes"):
                    self._prev_bboxes: Dict[int, Tuple] = {}
                self._prev_bboxes[tid] = bbox

                fast_result = {"action": "standing", "confidence": 0.5, "changed": False}
                if crop is not None and crop.size > 0:
                    fast_result = self.fast_actions.classify(tid, crop, bbox, frame_time)

                    is_new = tid not in getattr(self, "_person_seen", set())
                    if is_new:
                        if not hasattr(self, "_person_seen"):
                            self._person_seen = set()
                        self._person_seen.add(tid)
                        fast_result["changed"] = True

                    if False:
                        pass

                    if not self._turbo:
                        had_vlm = tid in getattr(self, "_person_vlm_done", set())
                        had_scene = tid in getattr(self, "_person_scene_done", set())
                        if not hasattr(self, "_person_vlm_done"):
                            self._person_vlm_done = set()
                        if not hasattr(self, "_person_scene_done"):
                            self._person_scene_done = set()
                        if not hasattr(self, "_person_vlm_times"):
                            self._person_vlm_times: Dict[int, float] = {}

                        if not had_vlm:
                            cached_age = self.vlm.get_cache_age(tid, "dense")
                            if cached_age > 3.0:
                                self.vlm.submit_scene_dense(tid, frame, bbox,
                                    [o["name"] for o in self._objects[:8]] if self._objects else [], task="dense")
                                self._person_vlm_done.add(tid)
                                self._person_vlm_times[tid] = frame_time
                                logger.info(f"[VLM] Person description request for Person_{tid}")

                        elif not had_scene and frame_time - self._person_vlm_times.get(tid, 0) > 8.0:
                            cached_age = self.vlm.get_cache_age(tid, "scene")
                            if cached_age > 10.0:
                                self.vlm.submit_scene_dense(tid, frame, bbox,
                                    [o["name"] for o in self._objects[:8]] if self._objects else [], task="scene")
                                self._person_scene_done.add(tid)
                                logger.info(f"[VLM] Scene context request for Person_{tid}")

                    dense_result = self.vlm.get_result(tid, "dense") if not self._turbo else ""
                    scene_result = self.vlm.get_result(tid, "scene") if not self._turbo else ""

                    if not hasattr(self, "_person_vlm_data"):
                        self._person_vlm_data: Dict[int, str] = {}

                    if dense_result and len(dense_result) > 30 and "QA>" not in dense_result:
                        self._person_vlm_data[tid] = dense_result
                    elif scene_result and len(scene_result) > 30 and "QA>" not in scene_result:
                        if tid not in self._person_vlm_data:
                            self._person_vlm_data[tid] = scene_result

                    vlm_text = self._person_vlm_data.get(tid, "")

                    obj_names = [o["name"] for o in self._objects[:10]] if self._objects else []
                    action = fast_result["action"]
                    timeline = self.action_engine.get_person_timeline(tid)
                    face = self._face_ids.get(tid, "")
                    reid = self._reid_ids.get(tid, "")

                    grounded_caption = build_person_description(
                        track_id=tid,
                        vlm_caption=vlm_text,
                        yolo_objects=obj_names,
                        fast_action=fast_result,
                        face_id=face,
                        reid_id=reid,
                        history=timeline,
                        frame_shape=(h, w),
                        bbox=bbox,
                        frame_time=frame_time,
                    )

                    prev = self._captions.get(tid, "")
                    is_new = not prev
                    vlm_clean = vlm_text.strip().rstrip(".") if vlm_text else ""
                    changed_vlm = vlm_clean and (vlm_clean not in prev)
                    new_better = (
                        is_new
                        or (vlm_clean and not any(marker in prev for marker in ("Appears to be:", "Description:")))
                        or changed_vlm
                    )

                    if new_better and (is_new or changed_vlm or fast_result["action"] not in ("standing", "active")):
                        self._captions[tid] = grounded_caption
                        if is_new:
                            logger.info(f"[NEW]  Person_{tid} | {grounded_caption}")
                        elif changed_vlm:
                            logger.info(f"[VLM]  Person_{tid} | {vlm_text[:300]}")


                        self.action_engine.log_action(
                            tid, action, grounded_caption, bbox,
                            {"event": "caption", "vlm": vlm_text,
                             "objects": obj_names, "action": action}
                        )
                        embedding = self.vss.store(tid, grounded_caption)
                        if embedding is not None and embedding.size > 0:
                            meta = {"action": action, "time": frame_time}
                            if obj_names:
                                meta["objects"] = ", ".join(obj_names[:10])
                            try:
                                self.vector_store.store(tid, embedding.tolist(), grounded_caption, meta)
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

                if False:
                    pass

                self.action_engine.log_action(
                    tid, fast_result["action"],
                    f"{fast_result['action']} (move={fast_result['movement_px']}px)",
                    bbox,
                    {
                        "frame_time": frame_time,
                        "movement_px": fast_result["movement_px"],
                        "hand_to_face": fast_result["hand_to_face"],
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

                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                for other in persons:
                    if other["track_id"] <= tid:
                        continue
                    ox1, oy1, ox2, oy2 = other["bbox"]
                    oc = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
                    dist = np.sqrt((center[0] - oc[0]) ** 2 + (center[1] - oc[1]) ** 2)
                    if dist < 200:
                        alert = self.alerts.evaluate_interaction(tid, other["track_id"])
                        if alert:
                            self.webhook.send(alert)
                            self.sqlite.log_event("interaction", tid, alert.get("data", {}))
                            self.graph.add_edge(
                                f"Person_{tid}",
                                f"Person_{other['track_id']}",
                                "INTERACTING_WITH",
                            )
                            self.action_engine.log_action(
                                tid, "interacting",
                                f"Interacting with Person_{other['track_id']}",
                                bbox, {"with": other["track_id"]}
                            )

            for obj in objects:
                obj_node = f"Obj_{obj['name']}"
                self.graph.add_object_node(obj["name"])
                self.graph.add_edge(
                    f"Obj_{obj['name']}",
                    f"Obj_{obj['name']}",
                    "DETECTED_IN_SCENE",
                )

            stats = {
                "fps": frame_prober.avg_ms,
                "person_count": len(persons),
                "alert_count": self.alerts.alert_count,
                "project_progress": project_tracker.get_status_display(),
                "face_ids": self._face_ids,
                "reid_ids": self._reid_ids,
                "objects": self._objects,
                "action_summary": self.action_engine.get_scene_summary(60),
            }

            rendered = self.display.render(
                canvas, persons, self._captions, self._vqa_answers, self._vss_matches, stats
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

                if self._frame_skip % 60 == 0:
                    pass

            if not getattr(self, "_headless", False):
                if not self.display.show(rendered):
                    logger.info("[ARGUS] User pressed 'q'")
                    break

            frame_prober.stop()

    def _dump_final_report(self) -> None:
        logger.info("=" * 60)
        logger.info("  FINAL CCTV ANALYSIS REPORT")
        logger.info("=" * 60)
        logger.info(f"  Total persons detected: {len(self._captions)}")
        logger.info(f"  Total actions logged: {len(self.action_engine.actions)}")
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

    def _chat_answer(self, question: str) -> None:
        result = self.groq_chat.ask(
            question=question,
            graph_store=self.graph,
            vector_store=self.vector_store,
            sqlite_store=self.sqlite,
            action_engine=self.action_engine,
            captions=self._captions,
            objects=self._objects,
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
