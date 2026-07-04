"""Cognee Graph RAG integration — knowledge graph construction and retrieval.

Cognee ingests perception packets, VLM outputs, and LLM reasoning to build
a rich temporal knowledge graph. Provides graph RAG retrieval for the chat LLM.

Runs entirely local using NetworkX backend.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from config.settings import COGNEE_ENABLED, DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

NODE_PERSON = "Person"
NODE_OBJECT = "Object"
NODE_EVENT = "Event"
NODE_SCENE = "Scene"
NODE_ACTION = "Action"
NODE_DAILY_PATTERN = "DailyPattern"

EDGE_INTERACTS = "INTERACTS_WITH"
EDGE_HOLDS = "HOLDS"
EDGE_NEAR = "NEAR"
EDGE_LOOKS_AT = "LOOKS_AT"
EDGE_PERFORMS = "PERFORMS"
EDGE_OCCURS_IN = "OCCURS_IN"
EDGE_PART_OF = "PART_OF"
EDGE_SIMILAR_TO = "SIMILAR_TO"


class CogneeManager:
    """Manages knowledge graph construction, retrieval, and cognee integration."""

    def __init__(self) -> None:
        self.enabled = COGNEE_ENABLED
        self.graph = nx.DiGraph()
        self._lock = threading.Lock()
        self._save_path = DATA_DIR / "graph_backup.json"
        self._event_count = 0
        self._last_save = 0
        self._save_interval = 50

        if self._save_path.is_file():
            self.load()

        logger.info(f"[COGNEE] Knowledge graph ready ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")

    # ── Ingestion ──

    def ingest_perception(self, packet) -> None:
        """Ingest a PerceptionPacket into the knowledge graph."""
        if not self.enabled:
            return

        scene_node = f"Scene_{int(packet.timestamp)}"
        self._add_node(scene_node, NODE_SCENE, f"Frame {packet.frame_number}", {
            "timestamp": packet.timestamp,
            "frame": packet.frame_number,
            "width": packet.frame_width,
            "height": packet.frame_height,
        })

        for person in packet.persons:
            p_node = f"Person_{person.track_id}"
            self._add_node(p_node, NODE_PERSON, f"Person {person.track_id}", {
                "confidence": person.confidence,
                "last_seen": packet.timestamp,
            })
            self._add_edge(p_node, scene_node, EDGE_OCCURS_IN)

            if person.track_id in packet.actions:
                action = packet.actions[person.track_id]
                a_node = f"Action_{action.action}_{person.track_id}_{int(packet.timestamp)}"
                self._add_node(a_node, NODE_ACTION, action.action, {
                    "confidence": action.confidence,
                    "source": action.source,
                    "timestamp": packet.timestamp,
                })
                self._add_edge(p_node, a_node, EDGE_PERFORMS)
                self._add_edge(a_node, scene_node, EDGE_OCCURS_IN)

            if person.track_id in packet.gaze:
                gaze = packet.gaze[person.track_id]
                if gaze.target_person_id is not None:
                    target_node = f"Person_{gaze.target_person_id}"
                    if self.graph.has_node(target_node):
                        self._add_edge(p_node, target_node, EDGE_LOOKS_AT, {
                            "direction": gaze.direction,
                            "timestamp": packet.timestamp,
                        })

        for obj in packet.objects:
            o_node = f"Obj_{obj.name}_{int(packet.timestamp)}"
            self._add_node(o_node, NODE_OBJECT, obj.name, {
                "confidence": obj.confidence,
                "timestamp": packet.timestamp,
            })
            self._add_edge(o_node, scene_node, EDGE_OCCURS_IN)

            for person in packet.persons:
                p_node = f"Person_{person.track_id}"
                obj_cx = (obj.bbox[0] + obj.bbox[2]) / 2
                obj_cy = (obj.bbox[1] + obj.bbox[3]) / 2
                p_cx = (person.bbox[0] + person.bbox[2]) / 2
                p_cy = (person.bbox[1] + person.bbox[3]) / 2
                dist = ((obj_cx - p_cx) ** 2 + (obj_cy - p_cy) ** 2) ** 0.5
                if dist < 150:
                    self._add_edge(p_node, o_node, EDGE_NEAR, {"distance": round(dist, 1)})

        for contact in packet.contacts:
            if contact.contact:
                a_node = f"Person_{contact.person_a}"
                b_node = f"Person_{contact.person_b}"
                self._add_edge(a_node, b_node, EDGE_INTERACTS, {
                    "score": contact.score,
                    "evidence": contact.evidence,
                    "timestamp": packet.timestamp,
                })

        self._event_count += 1
        self._maybe_save()

    def ingest_vlm_output(self, track_id: int, vlm_text: str, task: str = "dense") -> None:
        if not self.enabled:
            return
        p_node = f"Person_{track_id}"
        self._update_node_property(p_node, f"vlm_{task}", vlm_text[:500])
        self._update_node_property(p_node, "vlm_timestamp", time.time())

    def ingest_llm_output(self, track_id: int, reasoning: dict) -> None:
        if not self.enabled:
            return
        p_node = f"Person_{track_id}"
        e_node = f"Event_{int(time.time())}_{track_id}"
        self._add_node(e_node, NODE_EVENT, reasoning.get("narrative", "")[:200], {
            "narrative": reasoning.get("narrative", ""),
            "intent": reasoning.get("intent", ""),
            "anomaly_score": reasoning.get("anomaly_score", 0),
            "is_normal": reasoning.get("is_normal", True),
            "urgency": reasoning.get("urgency", "none"),
            "timestamp": time.time(),
        })
        self._add_edge(p_node, e_node, EDGE_PART_OF)

    def ingest_chat_qa(self, question: str, answer: str) -> None:
        if not self.enabled:
            return
        qa_node = f"QA_{int(time.time())}"
        self._add_node(qa_node, "QA", question[:150], {
            "question": question,
            "answer": answer[:500],
            "timestamp": time.time(),
        })

    # ── Retrieval ──

    def retrieve_context(self, query: str, top_k: int = 15) -> List[Dict]:
        if not self.enabled:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []

        with self._lock:
            for node, data in self.graph.nodes(data=True):
                label = data.get("label", "")
                label_lower = label.lower()
                score = 0.0

                if query_lower in label_lower:
                    score += 5.0

                for word in query_words:
                    if len(word) > 2 and word in label_lower:
                        score += 1.5

                if score > 0:
                    neighbors = []
                    for _, target, edata in self.graph.out_edges(node, data=True):
                        neighbors.append({"target": target, "relation": edata.get("relation", "")})
                    for source, _, edata in self.graph.in_edges(node, data=True):
                        neighbors.append({"source": source, "relation": edata.get("relation", "")})

                    results.append({
                        "node": node,
                        "type": data.get("type", ""),
                        "label": label,
                        "score": score,
                        "neighbors": neighbors,
                        "properties": data,
                        "timestamp": data.get("timestamp", 0),
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        """Get last N scene/event nodes ordered by timestamp."""
        scenes = [
            (node, data) for node, data in self.graph.nodes(data=True)
            if data.get("type") in (NODE_SCENE, NODE_EVENT)
        ]
        scenes.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
        return [{"node": node, **data} for node, data in scenes[:n]]

    def get_person_history(self, track_id: int) -> Dict:
        """Get complete history for a person from the graph."""
        p_node = f"Person_{track_id}"
        if p_node not in self.graph:
            return {"person": None, "actions": [], "interactions": [], "objects_near": []}

        person_data = dict(self.graph.nodes[p_node])

        actions = []
        interactions = []
        objects_near = []

        for _, target, edata in self.graph.out_edges(p_node, data=True):
            rel = edata.get("relation", "")
            target_data = self.graph.nodes.get(target, {})
            if rel == EDGE_PERFORMS:
                actions.append({"target": target, "label": target_data.get("label", ""), **edata})
            elif rel == EDGE_LOOKS_AT:
                interactions.append({"type": "gaze", "target": target, **edata})
            elif rel == EDGE_NEAR:
                objects_near.append({"target": target, "label": target_data.get("label", ""), **edata})

        for source, _, edata in self.graph.in_edges(p_node, data=True):
            rel = edata.get("relation", "")
            if rel == EDGE_LOOKS_AT:
                interactions.append({"type": "gazed_by", "source": source, **edata})
            elif rel == EDGE_INTERACTS:
                interactions.append({"type": "contact", "source": source, **edata})

        return {
            "person": person_data,
            "actions": actions[-20:],
            "interactions": interactions[-10:],
            "objects_near": objects_near[-10:],
        }

    def get_daily_patterns(self) -> List[Dict]:
        """Extract daily behavioral patterns from graph."""
        patterns = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == NODE_DAILY_PATTERN:
                patterns.append({"node": node, **data})
        return sorted(patterns, key=lambda x: x.get("timestamp", 0), reverse=True)

    def detect_patterns(self) -> None:
        """Run community/pattern detection on the graph."""
        if self.graph.number_of_nodes() < 10:
            return
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph.to_undirected())
            for node, community_id in partition.items():
                self.graph.nodes[node]["community"] = int(community_id)
            logger.debug(f"[COGNEE] Community detection found {len(set(partition.values()))} communities")
        except ImportError:
            pass

    # ── Graph Operations ──

    def _add_node(self, node_id: str, node_type: str, label: str, properties: Optional[Dict] = None) -> None:
        with self._lock:
            if node_id not in self.graph:
                self.graph.add_node(node_id, type=node_type, label=label, properties=properties or {}, timestamp=time.time())
            else:
                self.graph.nodes[node_id]["label"] = label
                self.graph.nodes[node_id]["timestamp"] = time.time()
                if properties:
                    self.graph.nodes[node_id]["properties"].update(properties)

    def _update_node_property(self, node_id: str, key: str, value) -> None:
        with self._lock:
            if node_id in self.graph:
                self.graph.nodes[node_id][key] = value

    def _add_edge(self, source: str, target: str, relation: str, properties: Optional[Dict] = None) -> None:
        with self._lock:
            if not self.graph.has_node(source) or not self.graph.has_node(target):
                return
            self.graph.add_edge(source, target, relation=relation, properties=properties or {}, timestamp=time.time())

    def _maybe_save(self) -> None:
        if self._event_count - self._last_save >= self._save_interval:
            self.save()
            self._last_save = self._event_count

    def save(self) -> None:
        try:
            data = nx.node_link_data(self.graph)
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._save_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[COGNEE] Save error: {e}")

    def load(self) -> None:
        if not self._save_path.is_file():
            return
        try:
            with open(self._save_path) as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
            logger.info("[COGNEE] Loaded from backup")
        except Exception as e:
            logger.warning(f"[COGNEE] Load error: {e}")

    def get_stats(self) -> Dict:
        types = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "Unknown")
            types[t] = types.get(t, 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": types,
            "events_indexed": self._event_count,
        }

    def purge_old(self, max_age: float = 3600.0) -> None:
        now = time.time()
        old = [n for n, d in self.graph.nodes(data=True) if now - d.get("timestamp", 0) > max_age]
        self.graph.remove_nodes_from(old)
        if old:
            logger.debug(f"[COGNEE] Purged {len(old)} old nodes")
