"""Streamlit dashboard for ARGUS Video Intelligence."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

st.set_page_config(
    page_title="ARGUS Dashboard",
    page_icon="📹",
    layout="wide",
)

st.title("ARGUS Video Intelligence System")
st.markdown("---")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
GRAPH_BACKUP = DATA_DIR / "graph_backup.json"
DB_PATH = DATA_DIR / "db" / "events.db"


def compute_project_progress() -> dict:
    progress = {
        "overall": 0.0,
        "phase": "Initializing",
        "categories": [
            {"category": "Layer 1: Motion Detection", "progress": 0, "active": 0, "total": 2},
            {"category": "Layer 2: Detection + Tracking", "progress": 0, "active": 0, "total": 3},
            {"category": "Layer 3: VLM Engine", "progress": 0, "active": 0, "total": 4},
            {"category": "Layer 4: Knowledge + Memory", "progress": 0, "active": 0, "total": 4},
            {"category": "Layer 5: Alerts + Dashboard", "progress": 0, "active": 0, "total": 4},
        ],
    }
    components = {
        "Layer 1: Motion Detection": [("camera", 50), ("mog2", 50)],
        "Layer 2: Detection + Tracking": [("yolo", 33.3), ("bytetrack", 33.3), ("crop", 33.3)],
        "Layer 3: VLM Engine": [("florence", 25), ("caption", 25), ("vqa", 25), ("queue", 25)],
        "Layer 4: Knowledge + Memory": [("graph", 25), ("chromadb", 25), ("sqlite", 25), ("vss", 25)],
        "Layer 5: Alerts + Dashboard": [("alerts", 33.3), ("webhook", 33.3), ("dashboard", 33.3), ("summary", 0)],
    }
    checks = {
        "camera": GRAPH_BACKUP.exists() or DB_PATH.exists(),
        "mog2": GRAPH_BACKUP.exists() or DB_PATH.exists(),
        "yolo": GRAPH_BACKUP.exists() or DB_PATH.exists(),
        "bytetrack": GRAPH_BACKUP.exists() or DB_PATH.exists(),
        "crop": GRAPH_BACKUP.exists() or DB_PATH.exists(),
        "florence": False,
        "caption": DB_PATH.exists(),
        "vqa": DB_PATH.exists(),
        "queue": False,
        "graph": GRAPH_BACKUP.exists(),
        "chromadb": (DATA_DIR / "chroma").exists(),
        "sqlite": DB_PATH.exists(),
        "vss": DB_PATH.exists(),
        "alerts": DB_PATH.exists(),
        "webhook": False,
        "dashboard": True,
        "summary": DB_PATH.exists(),
    }

    cat_progress = []
    total_weight = 0
    earned_weight = 0
    for cat_name, items in components.items():
        cat_total = sum(w for _, w in items)
        cat_earned = 0
        active = 0
        for check, weight in items:
            total_weight += weight
            cat_earned += weight if checks.get(check, False) else 0
            if checks.get(check, False):
                active += 1
        cat_pct = (cat_earned / cat_total * 100) if cat_total > 0 else 0
        cat_progress.append({
            "category": cat_name,
            "progress": round(cat_pct, 1),
            "active": active,
            "total": len(items),
        })
        earned_weight += cat_earned

    overall = (earned_weight / total_weight * 100) if total_weight > 0 else 0
    if overall >= 95:
        phase = "Production Ready"
    elif overall >= 70:
        phase = "Operational"
    elif overall >= 40:
        phase = "Partial"
    elif overall >= 10:
        phase = "Warming Up"
    else:
        phase = "Initializing"

    progress["overall"] = round(overall, 1)
    progress["phase"] = phase
    progress["categories"] = cat_progress
    return progress


def load_graph_data() -> Optional[Dict]:
    if not GRAPH_BACKUP.exists():
        return None
    with open(GRAPH_BACKUP) as f:
        return json.load(f)


def load_events() -> List[Dict]:
    if not DB_PATH.exists():
        return []
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT event_type, track_id, data, timestamp FROM events ORDER BY timestamp DESC LIMIT 50"
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


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Persons Detected", st.session_state.get("person_count", 0))
with col2:
    st.metric("Interactions", st.session_state.get("interaction_count", 0))
with col3:
    st.metric("Objects Identified", st.session_state.get("object_count", 0))
with col4:
    st.metric("Alerts", st.session_state.get("alert_count", 0))

st.markdown("---")

progress_data = compute_project_progress()
st.subheader(f"Project Progress: {progress_data['overall']:.1f}% - {progress_data['phase']}")
st.progress(progress_data["overall"] / 100.0)

pcols = st.columns(5)
for i, cat in enumerate(progress_data["categories"]):
    with pcols[i]:
        st.metric(
            cat["category"].split(":")[0].strip(),
            f"{cat['progress']:.0f}%",
            f"{cat['active']}/{cat['total']} active",
        )

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Knowledge Graph", "Timeline", "Query", "Q&A"])

with tab1:
    st.subheader("Knowledge Graph")
    graph_data = load_graph_data()
    if graph_data:
        st.write(f"Nodes: {len(graph_data.get('nodes', []))} | Edges: {len(graph_data.get('links', []))}")

        with st.expander("View Nodes"):
            for node in graph_data.get("nodes", []):
                st.text(f"{node.get('id', '?')} [{node.get('type', '?')}] - {node.get('label', '')}")
    else:
        st.info("No graph data available yet. Data is saved periodically when the pipeline is running.")

with tab2:
    st.subheader("Event Timeline")
    events = load_events()
    if events:
        for evt in events:
            ts = time.strftime("%H:%M:%S", time.localtime(evt["timestamp"]))
            st.text(f"[{ts}] {evt['event_type']} | Track ID: {evt['track_id']}")
    else:
        st.info("No events recorded yet.")

with tab3:
    st.subheader("Graph Query")
    query = st.text_input("Search nodes or edges")
    if query:
        graph_data = load_graph_data()
        if graph_data:
            matches = []
            for node in graph_data.get("nodes", []):
                if query.lower() in str(node).lower():
                    matches.append(node)
            st.write(f"Found {len(matches)} matching nodes")
            for m in matches[:20]:
                st.text(f"{m.get('id', '?')} [{m.get('type', '?')}]")

with tab4:
    st.subheader("Ask Questions About CCTV Data")
    st.caption("Query recorded events, persons, objects, and interactions")

    qa_query = st.text_input("Ask a question:", key="qa_input",
        placeholder="e.g. 'who was detected?', 'what objects were seen?', 'show all interactions'")

    if qa_query:
        q = qa_query.lower()
        events = load_events()
        graph_data = load_graph_data()

        if "who" in q or "person" in q:
            persons = [e for e in events if e["event_type"] in ("caption", "vlm_request")]
            if persons:
                st.write(f"Found {len(persons)} person-related events:")
                for p in persons[:10]:
                    ts = time.strftime("%H:%M:%S", time.localtime(p["timestamp"]))
                    st.text(f"[{ts}] Person_{p['track_id']}: {p.get('data', {}).get('caption', '')[:100]}")
            else:
                st.info("No person events found.")

        elif "object" in q or "what" in q:
            if graph_data:
                obj_nodes = [n for n in graph_data.get("nodes", []) if n.get("type") == "Object"]
                if obj_nodes:
                    st.write(f"Found {len(obj_nodes)} objects:")
                    for n in obj_nodes[:20]:
                        st.text(f"  {n.get('id','?')}: {n.get('label','')}")
                else:
                    st.info("No objects detected yet.")
            else:
                st.info("No data available.")

        elif "interact" in q:
            int_events = [e for e in events if e["event_type"] == "interaction"]
            if int_events:
                st.write(f"Found {len(int_events)} interactions:")
                for ie in int_events[:15]:
                    ts = time.strftime("%H:%M:%S", time.localtime(ie["timestamp"]))
                    data = ie.get("data", {})
                    st.text(f"[{ts}] Person_{data.get('person_a','?')} ↔ Person_{data.get('person_b','?')}")
            else:
                st.info("No interactions recorded.")

        elif "alert" in q:
            alert_events = [e for e in events if e["event_type"] == "alert"]
            if alert_events:
                st.write(f"Found {len(alert_events)} alerts:")
                for ae in alert_events[:10]:
                    ts = time.strftime("%H:%M:%S", time.localtime(ae["timestamp"]))
                    st.text(f"[{ts}] {ae.get('data', {}).get('alert', '')}")
            else:
                st.info("No alerts.")

        elif "summary" in q or "stats" in q:
            types = {}
            for e in events:
                types[e["event_type"]] = types.get(e["event_type"], 0) + 1
            st.write("Event Summary:")
            st.json(types)

            if graph_data:
                node_types = {}
                for n in graph_data.get("nodes", []):
                    node_types[n.get("type", "?")] = node_types.get(n.get("type", "?"), 0) + 1
                st.write("Graph Nodes:")
                st.json(node_types)
                st.write(f"Edges: {len(graph_data.get('links', []))}")

        else:
            all_text = ""
            for e in events:
                all_text += str(e.get("data", {})) + " "
            if graph_data:
                for n in graph_data.get("nodes", []):
                    all_text += str(n.get("label", "")) + " "

            words = q.split()
            matches = []
            for word in words:
                if len(word) > 2 and word in all_text.lower():
                    matches.append(word)

            if matches:
                st.write(f"Keywords found in data: {', '.join(matches)}")
                related = [e for e in events if any(w in str(e.get("data", {})).lower() for w in matches)]
                for r in related[:10]:
                    ts = time.strftime("%H:%M:%S", time.localtime(r["timestamp"]))
                    st.text(f"[{ts}] {r['event_type']}: {str(r.get('data', {}))[:120]}")
            else:
                st.info("No matches found. Try asking about: persons, objects, interactions, alerts, summary")

st.markdown("---")
st.caption("ARGUS - Video Intelligence System | Real-time monitoring dashboard")
