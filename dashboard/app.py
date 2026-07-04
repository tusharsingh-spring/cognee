"""Streamlit dashboard for ARGUS V3 Video Intelligence."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

st.set_page_config(
    page_title="ARGUS V3 Dashboard",
    page_icon="📹",
    layout="wide",
)

st.title("ARGUS V3 — Video Intelligence System")
st.markdown("*3-Layer Architecture: Perception → VLM → LLM with Cognee Graph RAG*")
st.markdown("---")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "db" / "events.db"


def get_cognee():
    try:
        from graph_rag.cognee_bridge import CogneeBridge
        return CogneeBridge()
    except Exception:
        return None


def get_vector_store():
    try:
        from storage.vector_store import VectorStore
        return VectorStore()
    except Exception:
        return None


def get_sqlite():
    try:
        from storage.sqlite_store import SQLiteStore
        return SQLiteStore()
    except Exception:
        return None


def load_events() -> List[Dict]:
    if not DB_PATH.exists():
        return []
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT event_type, track_id, data, timestamp FROM events ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return [
        {
            "event_type": r[0],
            "track_id": r[1],
            "data": json.loads(r[2]) if r[2] else {},
            "timestamp": r[3],
        }
        for r in rows
    ]


# ── Top Metrics ──
cognee = get_cognee()
vs = get_vector_store()
sqlite = get_sqlite()

stats = cognee.get_stats() if cognee else {}
events = load_events()
event_types = {}
for e in events:
    event_types[e["event_type"]] = event_types.get(e["event_type"], 0) + 1

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Graph Nodes", stats.get("total_nodes", 0))
with col2:
    st.metric("Graph Edges", stats.get("total_edges", 0))
with col3:
    st.metric("Events Logged", stats.get("events_indexed", 0))
with col4:
    alert_count = event_types.get("alert", 0) + event_types.get("llm_alert", 0)
    st.metric("Alerts", alert_count)

st.markdown("---")

# ── System Status ──
st.subheader("System Status")
status_cols = st.columns(5)
checks = {
    "Cognee Graph RAG": cognee is not None,
    "ChromaDB Vector": vs is not None,
    "SQLite Events": sqlite is not None,
    "Graph Backup": (DATA_DIR / "graph_backup.json").exists(),
    "Database": DB_PATH.exists(),
}
for i, (name, ok) in enumerate(checks.items()):
    with status_cols[i]:
        st.metric(name, "Active" if ok else "Inactive")

st.markdown("---")

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["Knowledge Graph", "Timeline", "Graph Query", "Summary"])

with tab1:
    st.subheader("Knowledge Graph (Cognee)")
    if cognee and stats.get("total_nodes", 0) > 0:
        st.write(f"**Nodes**: {stats['total_nodes']} | **Edges**: {stats['total_edges']} | **Events**: {stats['events_indexed']}")

        if stats.get("node_types"):
            st.write("**Node Types**")
            for nt, count in sorted(stats["node_types"].items()):
                st.text(f"  {nt}: {count}")

        with st.expander("Recent Events (Graph RAG)"):
            recent = cognee.get_recent_events(20)
            for r in recent:
                ts = r.get("timestamp", 0)
                time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"
                st.text(f"[{time_str}] {r.get('type', '?')}: {r.get('label', '')[:150]}")

        with st.expander("Person Histories"):
            if hasattr(cognee, 'graph') and cognee.graph:
                for node in list(cognee.graph.nodes(data=True)):
                    node_id = node[0]
                    node_data = node[1]
                    if node_data.get("type") == "Person":
                        tid = int(node_id.split("_")[-1]) if "_" in node_id else None
                        if tid is not None:
                            hist = cognee.get_person_history(tid)
                            actions = hist.get("actions", [])
                            interactions = hist.get("interactions", [])
                            st.text(f"{node_id}: {len(actions)} actions, {len(interactions)} interactions")
            else:
                stats = cognee.get_stats()
                if stats.get("events_indexed", 0) > 0:
                    st.text(f"Person tracking available via search. {stats['events_indexed']} events indexed in cognee.")
                    query_tid = st.number_input("Track ID to search", min_value=0, value=1, step=1, key="person_tid")
                    if st.button("Search Person", key="search_person"):
                        hist = cognee.get_person_history(int(query_tid))
                        actions = hist.get("actions", [])
                        interactions = hist.get("interactions", [])
                        st.text(f"Person_{query_tid}: {len(actions)} actions, {len(interactions)} interactions")
                        for a in actions[:5]:
                            st.text(f"  - {str(a.get('answer', a))[:200]}")
                else:
                    st.info("No graph data yet. Run the pipeline first.")
    else:
        st.info("Knowledge graph is empty. Run the pipeline to populate it: `python main.py --video auto`")

with tab2:
    st.subheader("Event Timeline")
    if events:
        for evt in events[:50]:
            ts = evt.get("timestamp", 0)
            time_str = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S") if ts else "?"
            data_preview = str(evt.get("data", {}))[:100]
            st.text(f"[{time_str}] {evt['event_type']} | Track: {evt['track_id']} | {data_preview}")
    else:
        st.info("No events recorded yet. Run the pipeline first.")

with tab3:
    st.subheader("Graph Query")
    query = st.text_input("Search knowledge graph", placeholder="e.g. Person_1, laptop, scene...")
    if query and cognee:
        results = cognee.retrieve_context(query, top_k=15)
        if results:
            st.write(f"**Found {len(results)} matches**")
            for r in results[:20]:
                score = r.get("score", 0)
                neighbors = r.get("neighbors", [])
                n_preview = ", ".join(
                    n.get("target", n.get("source", "")) for n in neighbors[:3]
                )
                st.text(f"[score={score:.1f}] {r['node']} [{r['type']}]: {r['label'][:200]}")
                if neighbors:
                    st.text(f"  Connected to: {n_preview}")
        else:
            st.info("No matches found in knowledge graph.")

with tab4:
    st.subheader("System Summary")
    if events:
        st.write("**Event Type Distribution**")
        st.json(event_types)

        if cognee:
            daily_patterns = cognee.get_daily_patterns()
            if daily_patterns:
                st.write("**Daily Patterns**")
                for p in daily_patterns[:5]:
                    st.text(p.get("label", "")[:200])
            else:
                st.info("No daily patterns yet (requires more data).")

        from collections import Counter
        track_events = Counter(e["track_id"] for e in events if e["track_id"] is not None)
        if track_events:
            st.write("**Most Active Persons**")
            for tid, count in track_events.most_common(5):
                st.text(f"  Person_{tid}: {count} events")
    else:
        st.info("No data available yet.")

st.markdown("---")
st.caption("ARGUS V3 — 3-Layer Video Intelligence System | Real-time monitoring")
