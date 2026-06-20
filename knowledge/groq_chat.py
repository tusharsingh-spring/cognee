"""Graph RAG + Vector RAG chat bot powered by Groq LLM API.
Answers natural language questions about CCTV footage using stored data."""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

CHAT_SYSTEM_PROMPT = """You are an AI CCTV security analyst with access to a dual CV+VLM video intelligence pipeline. Answer questions using ONLY the provided context data. The context includes:

- Person descriptions from VLM dense captioning (Florence-2)
- Pose keypoint data (17 COCO joints per person)
- Optical flow motion data (velocity magnitude per person)
- Monocular depth estimates (per-person 3D positioning)
- Contact detection (person-person contact events with scores)
- Hand tracking (grip/open detection, 21 keypoints per hand)
- Gaze estimation (gaze direction vector, gaze-to-person targets)
- Action recognition (standing/walking/reaching/grabbing/falling etc.)
- YOLO object detections (80 COCO classes with confidence scores)
- Segmentation masks (pixel-accurate per-person boundaries)
- Temporal event timeline (chronological ordered event log)
- Identity matches (face recognition, re-identification)

Rules:
1. ONLY use information present in the context. Do NOT invent or guess.
2. If the context doesn't contain enough information, say so clearly.
3. Be specific - mention Person IDs, timestamps, actions, and descriptions.
4. For temporal queries ("what happened before/after", "sequence"), use the timeline order.
5. For causal queries, reference contact events, gaze targets, and action transitions.
6. For spatial queries, reference depth estimates and relative positions.
7. If threat keywords appear, mention them explicitly.
8. Keep answers concise but informative - 3-5 sentences unless asked for detail."""

CHAT_SYSTEM_PROMPT_TIMELINE = """You are an AI CCTV security analyst with a complete chronological event timeline.
The context is organized as an ordered sequence of events with timestamps.
When answering, ALWAYS reference the timestamps and explain what happened in order.
For causal questions, trace the chain: who did what, when, to whom, and what followed.
Cross-reference CV data (pose, depth, contact, gaze) with VLM descriptions."""

TIMELINE_CONTEXT_TEMPLATE = """
=== EVENT TIMELINE (chronological) ===
{timeline}
=== CURRENT CAPTIONS ===
{captions}
=== OBJECTS DETECTED ===
{objects}
=== GRAPH SUMMARY ===
{graph}
=== VECTOR SEARCH RESULTS ===
{vector_results}
"""

DENSE_CONTEXT_TEMPLATE = """
=== PERSON CAPTIONS ===
{captions}
=== OBJECTS DETECTED ===
{objects}
=== GRAPH SUMMARY ===
{graph}
=== SCENE SUMMARY ===
{scene}
=== VECTOR SEARCH RESULTS ===
{vector_results}
=== RECENT EVENTS ===
{events}
"""


class GroqChatBot:
    def __init__(self) -> None:
        self._history: List[Dict] = []
        self._client = None

    @property
    def _available(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY", ""))

    def _get_client(self):
        if self._client is None and self._available:
            try:
                from groq import Groq
                self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logger.info(f"[CHAT] Groq client ready (model={GROQ_MODEL})")
            except ImportError:
                logger.warning("[CHAT] groq package not installed. Run: pip install groq")
            except Exception as e:
                logger.warning(f"[CHAT] Groq init failed: {e}")
                self._client = None
        return self._client

    def ask(
        self,
        question: str,
        graph_store,
        vector_store,
        sqlite_store,
        action_engine,
        captions: Dict[int, str],
        objects: List[Dict],
        causal_extractor=None,
        contact_detector=None,
    ) -> Dict:
        is_temporal = any(w in question.lower() for w in [
            "timeline", "sequence", "before", "after", "when", "order",
            "first", "then", "contact", "touch", "reach", "grab", "look",
            "gaze", "looked", "proximity", "distance",
        ])

        if is_temporal and causal_extractor is not None:
            context, system_prompt = self._gather_timeline_context(
                question, graph_store, vector_store, sqlite_store,
                action_engine, captions, objects, causal_extractor, contact_detector,
            )
            sp = system_prompt
        else:
            context = self._gather_context(
                question, graph_store, vector_store, sqlite_store,
                action_engine, captions, objects,
            )
            sp = CHAT_SYSTEM_PROMPT

        if not self._available:
            return self._fallback_answer(question, context)

        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": f"Context data:\n{context}\n\nQuestion: {question}"},
        ]

        try:
            client = self._get_client()
            if client is None:
                return self._fallback_answer(question, context)

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            logger.info(f"[CHAT] Groq answered ({len(answer)} chars)")

            self._history.append({"q": question, "a": answer, "t": time.time()})
            if len(self._history) > 50:
                self._history = self._history[-50:]

            return {"answer": answer, "context": context, "source": "groq"}

        except Exception as e:
            logger.warning(f"[CHAT] Groq API error: {e}")
            return self._fallback_answer(question, context)

    def _gather_timeline_context(
        self,
        question: str,
        graph_store,
        vector_store,
        sqlite_store,
        action_engine,
        captions: Dict[int, str],
        objects: List[Dict],
        causal_extractor,
        contact_detector,
    ) -> Tuple[str, str]:
        parts = []

        if causal_extractor is not None:
            try:
                cs = causal_extractor.get_contact_series()
                if cs:
                    timeline_lines = []
                    for event in cs[-30:]:
                        ts = event.get("timestamp", 0)
                        pa = event.get("person_a", "?")
                        pb = event.get("person_b", "?")
                        contact = "CONTACT" if event.get("contact") else "proximity"
                        score = event.get("score", 0)
                        evidence = event.get("evidence", [])
                        ev_str = ", ".join(f"{e[0]}={e[1]}" for e in evidence[:3])
                        timeline_lines.append(
                            f"[T={ts:.1f}s] P{pa}<->P{pb}: {contact} (score={score:.2f}) [{ev_str}]"
                        )
                    parts.append("=== CONTACT TIMELINE ===")
                    parts.extend(timeline_lines[-30:])
            except Exception:
                pass

        try:
            timeline_data = []
            for tid in sorted(captions.keys()):
                tl = action_engine.get_person_timeline(tid)
                for entry in tl[-20:]:
                    ts = entry.get("timestamp", entry.get("frame_time", 0))
                    if isinstance(ts, (int, float)):
                        timeline_data.append((ts, tid, entry))
            timeline_data.sort(key=lambda x: x[0])
            lines = []
            for ts, tid, entry in timeline_data[-50:]:
                action = entry.get("action_type", "?")
                desc = entry.get("description", "")[:120]
                lines.append(f"[T={ts:.1f}s] Person_{tid}: {action} | {desc}")
            if lines:
                parts.append("=== ACTION TIMELINE ===")
                parts.extend(lines)
        except Exception:
            pass

        caption_lines = []
        for tid, caption in sorted(captions.items()):
            caption_lines.append(f"Person_{tid}: {caption[:300]}")
            timeline = action_engine.get_person_timeline(tid)
            actions = [a["action_type"] for a in timeline[-10:] if a.get("action_type")]
            if actions:
                caption_lines.append(f"  Actions: {' -> '.join(actions[-8:])}")

        obj_names = [o["name"] for o in objects[:15]] if objects else []
        obj_str = ", ".join(set(obj_names)) if obj_names else "none"

        graph_str = "no graph data"
        try:
            gs = graph_store.get_stats()
            graph_str = str(gs)[:300]
        except Exception:
            pass

        vec_str = "no vector results"
        if vector_store and captions:
            try:
                results = vector_store.search_by_text(question, n_results=3)
                vec_str = str(results)[:500]
            except Exception:
                pass

        context = TIMELINE_CONTEXT_TEMPLATE.format(
            timeline="\n".join(parts),
            captions="\n".join(caption_lines[:20]),
            objects=obj_str,
            graph=graph_str,
            vector_results=vec_str,
        )
        return context, CHAT_SYSTEM_PROMPT_TIMELINE

    def _gather_context(
        self,
        question: str,
        graph_store,
        vector_store,
        sqlite_store,
        action_engine,
        captions: Dict[int, str],
        objects: List[Dict],
    ) -> str:
        parts = []

        for tid, caption in sorted(captions.items()):
            parts.append(f"Person_{tid}: {caption}")
            timeline = action_engine.get_person_timeline(tid)
            actions = [a["action_type"] for a in timeline[-10:] if a.get("action_type")]
            if actions:
                parts.append(f"  Actions: {' -> '.join(actions[-8:])}")

        if objects:
            obj_names = [o["name"] for o in objects[:15]]
            parts.append(f"Detected objects: {', '.join(set(obj_names))}")

        try:
            graph_stats = graph_store.get_stats()
            parts.append(f"Knowledge graph: {graph_stats}")
        except Exception:
            pass

        try:
            scene = action_engine.get_scene_summary(300)
            parts.append(f"Scene summary (5min): {json.dumps(scene)[:300]}")
        except Exception:
            pass

        if vector_store and captions:
            try:
                results = vector_store.search_by_text(question, n_results=3)
                if results:
                    parts.append(f"Vector matches: {results}")
            except Exception:
                pass

        try:
            events = sqlite_store.get_recent_events(limit=10)
            if events:
                parts.append(f"Recent events: {events}")
        except Exception:
            pass

        return "\n".join(parts)

    def _fallback_answer(self, question: str, context: str) -> Dict:
        q = question.lower()

        if "person" in q or "who" in q:
            import re
            ids = re.findall(r"\d+", q)
            if ids:
                parts = []
                for line in context.split("\n"):
                    if f"Person_{ids[0]}" in line or f"Person {ids[0]}" in line:
                        parts.append(line)
                return {"answer": "\n".join(parts) if parts else "No specific data for that person.",
                        "source": "fallback"}

        return {"answer": "Groq API not configured. Set GROQ_API_KEY env var for AI chat."
                          "\n\nAvailable data:\n" + context[:2000],
                "source": "fallback", "context": context}

    def _gather_and_ask_direct(self, prompt: str) -> str:
        if not self._available:
            return ""
        try:
            client = self._get_client()
            if client is None:
                return ""
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.debug(f"[CHAT] Direct ask failed: {e}")
            return ""
