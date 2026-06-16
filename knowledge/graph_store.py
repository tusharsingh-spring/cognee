"""NetworkX knowledge graph with semantic connections."""

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from config.settings import GRAPH_PURGE_AGE, GRAPH_SAVE_INTERVAL, DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

NODE_PERSON = "Person"
NODE_OBJECT = "Object"
NODE_ACTION = "Action"
NODE_LOCATION = "Location"

EDGE_HOLDING = "HOLDING"
EDGE_INTERACTING = "INTERACTING_WITH"
EDGE_LOCATED = "LOCATED_AT"
EDGE_PERFORMING = "PERFORMING"


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._event_count = 0
        self._last_save = 0
        self._save_path = DATA_DIR / "graph_backup.json"

    def add_person_node(self, track_id: int, caption: str, meta: Optional[Dict] = None) -> None:
        node_name = f"Person_{track_id}"
        if node_name not in self.graph:
            self.graph.add_node(
                node_name,
                type=NODE_PERSON,
                label=caption[:100],
                properties=meta or {},
                timestamp=time.time(),
            )
            logger.debug(f"[GRAPH] Added node: {node_name}")
        else:
            self.graph.nodes[node_name]["label"] = caption[:100]
            self.graph.nodes[node_name]["timestamp"] = time.time()
            if meta:
                self.graph.nodes[node_name]["properties"].update(meta)
        self._event_count += 1
        self._maybe_save()

    def add_object_node(self, object_name: str, meta: Optional[Dict] = None) -> str:
        node_name = f"Obj_{object_name}"
        if node_name not in self.graph:
            self.graph.add_node(
                node_name,
                type=NODE_OBJECT,
                label=object_name,
                properties=meta or {},
                timestamp=time.time(),
            )
            logger.debug(f"[GRAPH] Added object: {node_name}")
        self._event_count += 1
        self._maybe_save()
        return node_name

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        meta: Optional[Dict] = None,
    ) -> None:
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return
        self.graph.add_edge(
            source,
            target,
            relation=relation,
            properties=meta or {},
            timestamp=time.time(),
        )
        logger.debug(f"[GRAPH] Edge added: {source} -[{relation}]-> {target}")
        self._event_count += 1
        self._maybe_save()

    def parse_caption_for_graph(self, track_id: int, caption: str) -> None:
        node_name = f"Person_{track_id}"
        caption_lower = caption.lower()

        holding_keywords = ["holding", "carrying", "grasping", "holding a", "carrying a"]
        for kw in holding_keywords:
            if kw in caption_lower:
                idx = caption_lower.index(kw)
                obj_part = caption[idx + len(kw):].strip().split(",")[0].split(".")[0].strip()
                if obj_part and len(obj_part) < 30:
                    obj_node = f"Obj_{obj_part}"
                    self.add_object_node(obj_part)
                    self.add_edge(node_name, obj_node, EDGE_HOLDING)

        action_keywords = ["walking", "running", "sitting", "standing", "reading", "talking", "eating", "drinking"]
        for kw in action_keywords:
            if kw in caption_lower:
                action_node = f"Action_{kw}"
                self.graph.add_node(
                    action_node, type=NODE_ACTION, label=kw, timestamp=time.time()
                )
                self.add_edge(node_name, action_node, EDGE_PERFORMING)

        location_keywords = ["in a", "on a", "at a", "in the", "on the", "at the", "near"]
        for kw in location_keywords:
            if kw in caption_lower:
                idx = caption_lower.index(kw)
                loc_part = caption[idx + len(kw):].strip().split(",")[0].split(".")[0].strip()
                if loc_part and len(loc_part) < 30:
                    loc_node = f"Loc_{loc_part}"
                    self.graph.add_node(
                        loc_node, type=NODE_LOCATION, label=loc_part, timestamp=time.time()
                    )
                    self.add_edge(node_name, loc_node, EDGE_LOCATED)

    def query_neighbors(self, track_id: int) -> List[Tuple[str, str, str]]:
        node_name = f"Person_{track_id}"
        if node_name not in self.graph:
            return []
        results = []
        for _, target, data in self.graph.out_edges(node_name, data=True):
            results.append((node_name, target, data.get("relation", "")))
        for source, _, data in self.graph.in_edges(node_name, data=True):
            results.append((source, node_name, data.get("relation", "")))
        return results

    def query_by_relation(self, relation: str) -> List[Tuple[str, str]]:
        results = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == relation:
                results.append((u, v))
        return results

    def rag_search(self, query: str, top_k: int = 5) -> List[Dict]:
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for node, data in self.graph.nodes(data=True):
            label = data.get("label", "")
            label_lower = label.lower()
            score = 0

            if query_lower in label_lower:
                score += 5.0

            for word in query_words:
                if len(word) > 2 and word in label_lower:
                    score += 1.0

            if score > 0:
                neighbors = []
                for _, target, edata in self.graph.out_edges(node, data=True):
                    neighbors.append({
                        "target": target,
                        "relation": edata.get("relation", ""),
                        "timestamp": edata.get("timestamp", 0),
                    })
                for source, _, edata in self.graph.in_edges(node, data=True):
                    neighbors.append({
                        "source": source,
                        "relation": edata.get("relation", ""),
                        "timestamp": edata.get("timestamp", 0),
                    })

                results.append({
                    "node": node,
                    "type": data.get("type", ""),
                    "label": label,
                    "score": score,
                    "neighbors": neighbors,
                    "timestamp": data.get("timestamp", 0),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_connected_nodes(self, node_name: str, depth: int = 2) -> Dict:
        result = {
            "node": node_name,
            "connections": [],
            "subgraph_nodes": [],
            "subgraph_edges": [],
        }

        if node_name not in self.graph:
            return result

        visited = {node_name}
        frontier = [node_name]
        all_nodes = set()
        all_edges = set()

        for _ in range(depth):
            next_frontier = []
            for n in frontier:
                for _, target in self.graph.out_edges(n):
                    all_edges.add((n, target))
                    all_nodes.add(target)
                    if target not in visited:
                        visited.add(target)
                        next_frontier.append(target)
                for source, _ in self.graph.in_edges(n):
                    all_edges.add((source, n))
                    all_nodes.add(source)
                    if source not in visited:
                        visited.add(source)
                        next_frontier.append(source)
            frontier = next_frontier

        result["subgraph_nodes"] = [
            {"id": n, "type": self.graph.nodes[n].get("type", ""),
             "label": self.graph.nodes[n].get("label", "")}
            for n in all_nodes
        ]
        result["subgraph_edges"] = [
            {"source": s, "target": t, "relation": self.graph.edges[s, t].get("relation", "")}
            for s, t in all_edges
        ]

        for _, target, data in self.graph.out_edges(node_name, data=True):
            result["connections"].append({
                "direction": "out",
                "target": target,
                "relation": data.get("relation", ""),
            })
        for source, _, data in self.graph.in_edges(node_name, data=True):
            result["connections"].append({
                "direction": "in",
                "source": source,
                "relation": data.get("relation", ""),
            })

        return result

    def to_json(self) -> str:
        import json
        data = nx.node_link_data(self.graph)
        return json.dumps(data, indent=2)

    def purge_old(self) -> None:
        now = time.time()
        old_nodes = [
            n
            for n, d in self.graph.nodes(data=True)
            if now - d.get("timestamp", 0) > GRAPH_PURGE_AGE
        ]
        for n in old_nodes:
            self.graph.remove_node(n)
        if old_nodes:
            logger.debug(f"[GRAPH] Purged {len(old_nodes)} old nodes")

    def get_stats(self) -> Dict[str, int]:
        types = {"Person": 0, "Object": 0, "Action": 0, "Location": 0}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "")
            if t in types:
                types[t] += 1
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("relation", "UNKNOWN")
            edge_types[r] = edge_types.get(r, 0) + 1
        return {
            "person_nodes": types["Person"],
            "object_nodes": types["Object"],
            "action_nodes": types["Action"],
            "location_nodes": types["Location"],
            "total_edges": self.graph.number_of_edges(),
        }

    def _maybe_save(self) -> None:
        if self._event_count - self._last_save >= GRAPH_SAVE_INTERVAL:
            self.save()
            self._last_save = self._event_count

    def save(self) -> None:
        try:
            data = nx.node_link_data(self.graph)
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._save_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("[GRAPH] Saved to disk")
        except Exception as e:
            logger.error(f"[GRAPH] Save error: {e}")

    def load(self) -> None:
        if not self._save_path.exists():
            return
        try:
            with open(self._save_path) as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
            logger.info("[GRAPH] Loaded from disk")
        except Exception as e:
            logger.warning(f"[GRAPH] Load error: {e}")
