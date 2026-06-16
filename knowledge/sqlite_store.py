"""SQLite event persistence for timeline queries."""

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from config.settings import SQLITE_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    track_id INTEGER,
    data TEXT,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    label TEXT,
    properties TEXT,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    properties TEXT,
    timestamp REAL NOT NULL,
    PRIMARY KEY (source, target, relation)
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_track ON events(track_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
CREATE INDEX IF NOT EXISTS idx_edges_ts ON edges(timestamp);
"""


class SQLiteStore:
    def __init__(self) -> None:
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        logger.info(f"[SQLITE] Connected: {SQLITE_PATH}")

    def log_event(
        self,
        event_type: str,
        track_id: Optional[int] = None,
        data: Optional[Dict] = None,
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO events (event_type, track_id, data, timestamp) VALUES (?, ?, ?, ?)",
            (event_type, track_id, json.dumps(data or {}), time.time()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def upsert_node(
        self, node_id: str, node_type: str, label: str, properties: Optional[Dict] = None
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO nodes (id, node_type, label, properties, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (node_id, node_type, label, json.dumps(properties or {}), time.time()),
        )
        self.conn.commit()

    def upsert_edge(
        self,
        source: str,
        target: str,
        relation: str,
        properties: Optional[Dict] = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO edges (source, target, relation, properties, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (source, target, relation, json.dumps(properties or {}), time.time()),
        )
        self.conn.commit()

    def get_recent_events(
        self, event_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        if event_type:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            {
                "id": r[0],
                "event_type": r[1],
                "track_id": r[2],
                "data": json.loads(r[3]) if r[3] else {},
                "timestamp": r[4],
            }
            for r in rows
        ]

    def get_timeline(self, track_id: int, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE track_id = ? ORDER BY timestamp DESC LIMIT ?",
            (track_id, limit),
        ).fetchall()
        return [
            {
                "id": r[0],
                "event_type": r[1],
                "track_id": r[2],
                "data": json.loads(r[3]) if r[3] else {},
                "timestamp": r[4],
            }
            for r in rows
        ]

    def get_node(self, node_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row:
            return {
                "id": row[0],
                "node_type": row[1],
                "label": row[2],
                "properties": json.loads(row[3]) if row[3] else {},
                "timestamp": row[4],
            }
        return None

    def close(self) -> None:
        self.conn.close()
