"""Graph RAG + Vector RAG chat bot powered by Groq LLM API.
Answers natural language questions about CCTV footage using stored data."""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

CHAT_SYSTEM_PROMPT = """You are an AI CCTV security analyst. Answer questions about surveillance 
footage using ONLY the provided context data. The context includes:

- Person descriptions from computer vision + VLM analysis
- Detected actions and movement patterns
- Object detections (YOLO)
- Identity matches (face recognition, re-identification)
- Timestamps and durations

Rules:
1. ONLY use information present in the context. Do NOT invent or guess.
2. If the context doesn't contain enough information, say so clearly.
3. Be specific - mention Person IDs, timestamps, actions, and descriptions.
4. For questions about "what happened", provide a chronological summary.
5. For questions about specific persons, describe their appearance, actions, and timeline.
6. If threat keywords appear, mention them explicitly.
7. Keep answers concise but informative - 3-5 sentences unless asked for detail."""


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
    ) -> Dict:
        context = self._gather_context(
            question, graph_store, vector_store, sqlite_store,
            action_engine, captions, objects
        )

        if not self._available:
            return self._fallback_answer(question, context)

        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
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
        q = question.lower()

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
