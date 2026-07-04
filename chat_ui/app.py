"""Streamlit Chat UI — full-context CCTV chatbot.

Allows users to chat with their CCTV camera. Retrieves context from:
- Cognee knowledge graph (graph RAG) — entity + relationship lookups
- ChromaDB vector store (semantic search) — similar past events
- SQLite event log (temporal queries) — recent event timeline
- Current perception state (real-time) — live model outputs

Powered by the local LLM (or Groq fallback).
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

st.set_page_config(
    page_title="ARGUS V3 - CCTV Chat",
    page_icon="📹",
    layout="wide",
)

st.title("ARGUS V3 — CCTV Intelligence Chat")
st.markdown("*Chat with your CCTV camera — full context from perception + VLM + knowledge graph + vector search*")
st.markdown("---")


class ChatUI:
    def __init__(self) -> None:
        self._init_session()
        self._load_components()

    def _init_session(self) -> None:
        defaults = {
            "messages": [],
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def _load_components(self) -> None:
        self._cognee = None
        self._vector_store = None
        self._sqlite = None
        self._llm = None

        try:
            from graph_rag.cognee_bridge import CogneeBridge
            self._cognee = CogneeBridge(readonly=True)
        except Exception:
            pass

        try:
            from storage.vector_store import VectorStore
            self._vector_store = VectorStore()
        except Exception:
            pass

        try:
            from storage.sqlite_store import SQLiteStore
            self._sqlite = SQLiteStore()
        except Exception:
            pass

        try:
            from layer3_llm.llm_engine import LLMEngine
            self._llm = LLMEngine()
        except Exception:
            pass

    def render(self) -> None:
        col1, col2 = st.columns([2, 1])

        with col1:
            self._render_chat()
        with col2:
            self._render_context_panel()

    def _render_chat(self) -> None:
        st.subheader("Ask your CCTV anything")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for src in msg["sources"]:
                            st.caption(f"[{src['type']}] {src['preview'][:200]}")

        prompt = st.chat_input("Ask about what's happening...")
        if prompt:
            st.session_state.messages.append({
                "role": "user", "content": prompt, "timestamp": time.strftime("%H:%M:%S")
            })
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge graph + vector store..."):
                    answer, sources = self._generate_answer(prompt)
                st.write(answer)
                if sources:
                    with st.expander("Sources used"):
                        for src in sources:
                            st.caption(f"[{src['type']}] {src['preview'][:200]}")
            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sources": sources, "timestamp": time.strftime("%H:%M:%S"),
            })
            st.rerun()

    def _render_context_panel(self) -> None:
        st.subheader("Live Context")

        tabs = st.tabs(["Knowledge Graph", "Timeline", "Suggestions"])

        # Auto-refresh: pull fresh stats on every render
        kg_stats = None
        if self._cognee is not None:
            try:
                kg_stats = self._cognee.get_stats()
            except Exception:
                pass

        with tabs[0]:
            col_a, col_b, col_c = st.columns(3)
            nodes_count = kg_stats.get("total_nodes", 0) if kg_stats else 0
            edges_count = kg_stats.get("total_edges", 0) if kg_stats else 0
            events_count = kg_stats.get("events_indexed", 0) if kg_stats else 0
            col_a.metric("Nodes", nodes_count)
            col_b.metric("Edges", edges_count)
            col_c.metric("Events", events_count)
            if kg_stats and kg_stats.get("node_types"):
                with st.expander("Node Types"):
                    for nt, count in kg_stats["node_types"].items():
                        st.text(f"  {nt}: {count}")
            if nodes_count == 0 and events_count == 0:
                st.info("No knowledge graph data yet. Start: `python run.py --webcam`")

        with tabs[1]:
            if self._sqlite is not None:
                try:
                    sql_events = self._sqlite.get_recent_events(limit=20)
                except Exception:
                    sql_events = []
            else:
                sql_events = []
            cognee_events = []
            if self._cognee is not None:
                try:
                    cognee_events = self._cognee.get_recent_events(10)
                except Exception:
                    pass

            if cognee_events:
                st.caption("From Cognee (knowledge graph)")
                for evt in cognee_events[:10]:
                    ts = evt.get("timestamp", 0) if isinstance(evt, dict) else 0
                    if ts:
                        time_str = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
                        etype = evt.get("type", evt.get("event_type", "?")) if isinstance(evt, dict) else "?"
                        st.text(f"[{time_str}] {etype}")
            elif sql_events:
                st.caption("From SQLite (event log)")
                for e in sql_events[:20]:
                    ts = e.get("timestamp", 0)
                    if ts:
                        time_str = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
                        st.text(f"[{time_str}] {e.get('event_type', '?')}: track={e.get('track_id', '?')}")
            else:
                st.info("No events yet. Pipeline writes every 3s.")

        with tabs[2]:
            st.write("**Try asking**")
            suggestions = [
                "What is Person_1 doing?",
                "Has anyone been near the laptop?",
                "Show all interactions in the last hour",
                "Is there anything unusual today?",
                "Give me a summary of today's activity",
                "Who has visited most often today?",
                "What objects were detected?",
                "Any suspicious behavior?",
            ]
            for s in suggestions:
                if st.button(s, key=f"sugg_{s}"):
                    st.session_state.messages.append({
                        "role": "user", "content": s, "timestamp": time.strftime("%H:%M:%S")
                    })
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            answer, sources = self._generate_answer(s)
                        st.write(answer)
                        if sources:
                            with st.expander("Sources"):
                                for src in sources:
                                    st.caption(f"[{src['type']}] {src['preview'][:200]}")
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "sources": sources, "timestamp": time.strftime("%H:%M:%S"),
                    })
                    st.rerun()

        # Data refreshes live on each interaction (chat message, button click)
        st.caption("Stats update on each interaction")

    def _generate_answer(self, question: str) -> tuple:
        context_parts = []
        sources = []

        # ── Graph RAG: Cognee knowledge graph ──
        if self._cognee is not None:
            try:
                graph_results = self._cognee.retrieve_context(question, top_k=10)
                if graph_results:
                    ctx = ["=== Knowledge Graph (Cognee) ==="]
                    for r in graph_results[:8]:
                        ctx.append(f"{r['node']} [{r['type']}]: {r['label'][:200]}")
                        if r.get("neighbors"):
                            for n in r["neighbors"][:3]:
                                ctx.append(f"  -> {n.get('target', n.get('source', ''))} ({n.get('relation', '')})")
                    context_parts.append("\n".join(ctx))
                    sources.append({"type": "graph_rag", "preview": graph_results[0].get("label", "")})
            except Exception:
                pass

        # ── Vector Semantic Search ──
        if self._vector_store is not None:
            try:
                vec_results = self._vector_store.search_by_text(question, n_results=5)
                if vec_results:
                    ctx = ["=== Semantic Matches ==="]
                    for r in vec_results:
                        doc = r.get("document", "")[:250]
                        if doc:
                            ctx.append(f"[{r.get('id', '?')}] {doc}")
                    context_parts.append("\n".join(ctx))
                    if vec_results:
                        sources.append({"type": "vector_search", "preview": vec_results[0].get("document", "")[:200]})
            except Exception:
                pass

        # ── SQLite Timeline ──
        if self._sqlite is not None:
            try:
                events = self._sqlite.get_recent_events(limit=20)
                if events:
                    ctx = ["=== Recent Events (SQLite) ==="]
                    for e in events[:15]:
                        ts = e.get("timestamp", 0)
                        time_str = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S") if ts else "?"
                        ctx.append(f"[{time_str}] {e.get('event_type','?')}: track={e.get('track_id','?')}")
                    context_parts.append("\n".join(ctx))
            except Exception:
                pass

        # ── Local LLM / Groq ──
        context = "\n\n".join(context_parts) if context_parts else "No context available yet."

        if self._llm is not None and self._llm.available:
            try:
                answer = self._llm.answer_question(question, context)
                return answer, sources
            except Exception:
                pass

        # Fallback when no LLM available
        if context_parts:
            return (f"Based on available context:\n\n{context[:3000]}\n\n"
                    f"(LLM not available — showing raw context. Start pipeline first: `python main.py --video auto`)",
                    sources)

        return ("No data available yet. The system needs to process video first. "
                "Start the pipeline with: `python main.py --video auto`", sources)


def main():
    ui = ChatUI()
    ui.render()


if __name__ == "__main__":
    main()
