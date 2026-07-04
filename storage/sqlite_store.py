"""SQLite event store — persistent structured event log."""

import json
import sqlite3
import time
from typing import Dict, List, Optional

from config.settings import SQLITE_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteStore:
    def __init__(self) -> None:
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                track_id INTEGER,
                data TEXT,
                timestamp REAL DEFAULT (strftime('%s', 'now'))
            );
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT,
                properties TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                properties TEXT,
                timestamp REAL DEFAULT (strftime('%s', 'now'))
            );
            CREATE TABLE IF NOT EXISTS daily_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                summary TEXT,
                stats TEXT,
                created_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_track ON events(track_id);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)
        self._conn.commit()
        logger.info("[SQLITE] Database ready")

    def log_event(self, event_type: str, track_id: Optional[int], data: Optional[dict] = None) -> int:
        cursor = self._conn.execute(
            "INSERT INTO events (event_type, track_id, data, timestamp) VALUES (?, ?, ?, ?)",
            (event_type, track_id, json.dumps(data or {}), time.time()),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_recent_events(self, limit: int = 50, event_type: Optional[str] = None) -> List[Dict]:
        if event_type:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_events_since(self, since_ts: float) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE timestamp > ? ORDER BY timestamp ASC", (since_ts,)
        ).fetchall()
        return [dict(r) for r in rows]

    def upsert_node(self, node_id: str, node_type: str, label: str, properties: Optional[dict] = None) -> None:
        self._conn.execute(
            """INSERT INTO nodes (id, node_type, label, properties) VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET label=excluded.label, properties=excluded.properties""",
            (node_id, node_type, label, json.dumps(properties or {})),
        )
        self._conn.commit()

    def add_edge(self, source: str, target: str, relation: str, properties: Optional[dict] = None) -> None:
        self._conn.execute(
            "INSERT INTO edges (source, target, relation, properties) VALUES (?, ?, ?, ?)",
            (source, target, relation, json.dumps(properties or {})),
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> Optional[Dict]:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return dict(row) if row else None

    def search_nodes(self, query: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE label LIKE ? OR id LIKE ?", (f"%{query}%", f"%{query}%")
        ).fetchall()
        return [dict(r) for r in rows]

    def get_graph_context(self, node_id: str, depth: int = 1) -> Dict:
        node = self.get_node(node_id)
        if not node:
            return {"node": None, "edges": []}

        edges = []
        frontier = [node_id]
        visited = {node_id}

        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                rows_out = self._conn.execute(
                    "SELECT * FROM edges WHERE source = ?", (nid,)
                ).fetchall()
                for r in rows_out:
                    edges.append({"source": r["source"], "target": r["target"],
                                  "relation": r["relation"], "direction": "out"})
                    if r["target"] not in visited:
                        visited.add(r["target"])
                        next_frontier.append(r["target"])

                rows_in = self._conn.execute(
                    "SELECT * FROM edges WHERE target = ?", (nid,)
                ).fetchall()
                for r in rows_in:
                    edges.append({"source": r["source"], "target": r["target"],
                                  "relation": r["relation"], "direction": "in"})
                    if r["source"] not in visited:
                        visited.add(r["source"])
                        next_frontier.append(r["source"])
            frontier = next_frontier

        return {"node": dict(node), "edges": edges}

    def save_daily_summary(self, date_str: str, summary: str, stats: dict) -> None:
        self._conn.execute(
            "INSERT INTO daily_summaries (date, summary, stats, created_at) VALUES (?, ?, ?, ?)",
            (date_str, summary, json.dumps(stats), time.time()),
        )
        self._conn.commit()

    def get_daily_summaries(self, limit: int = 7) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM daily_summaries ORDER BY date DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
