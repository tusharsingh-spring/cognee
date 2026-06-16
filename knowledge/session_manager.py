"""Session Manager - tracks sessions, provides Graph RAG queries across sessions."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import DATA_DIR, SESSION_ENABLED, SESSION_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class Session:
    def __init__(self, session_id: str = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.stats: Dict[str, Any] = {
            "total_persons": 0,
            "unique_identities": 0,
            "total_objects": 0,
            "total_interactions": 0,
            "total_alerts": 0,
            "face_detections": 0,
            "vlm_captions": 0,
            "reid_matches": 0,
        }
        self.events: List[Dict] = []
        self.identities_seen: set = set()

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time or time.time()) - self.start_time,
            "start_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "stats": self.stats,
            "identities_seen": list(self.identities_seen),
            "event_count": len(self.events),
        }


class SessionManager:
    def __init__(self) -> None:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Session] = {}
        self.current_session: Optional[Session] = None
        self._index_path = SESSION_DIR / "session_index.json"
        self._load_index()

    def start_session(self) -> str:
        session = Session()
        self.sessions[session.session_id] = session
        self.current_session = session
        logger.info(f"[SESSION] Started session: {session.session_id}")
        return session.session_id

    def end_session(self) -> Optional[Dict]:
        if self.current_session is None:
            return None
        self.current_session.end_time = time.time()
        self._save_session(self.current_session)
        self._update_index(self.current_session)
        sid = self.current_session.session_id
        summary = self.current_session.to_dict()
        logger.info(f"[SESSION] Ended session: {sid} (duration: {summary['duration']:.0f}s)")
        self.current_session = None
        return summary

    def log_event(self, event: Dict) -> None:
        if self.current_session is None:
            return
        self.current_session.events.append({
            "timestamp": time.time(),
            **event,
        })
        if len(self.current_session.events) > 10000:
            self.current_session.events = self.current_session.events[-5000:]

    def update_stat(self, key: str, value: Any) -> None:
        if self.current_session is None:
            return
        self.current_session.stats[key] = value

    def add_identity(self, gid: str) -> None:
        if self.current_session is None:
            return
        self.current_session.identities_seen.add(gid)
        self.current_session.stats["unique_identities"] = len(self.current_session.identities_seen)

    def get_current_summary(self) -> Dict:
        if self.current_session is None:
            return {"active": False}
        return self.current_session.to_dict()

    def get_session(self, session_id: str) -> Optional[Dict]:
        filepath = SESSION_DIR / f"{session_id}.json"
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return None

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        files = sorted(SESSION_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        sessions = []
        for fp in files[:limit]:
            if fp.name == "session_index.json":
                continue
            try:
                with open(fp) as f:
                    sessions.append(json.load(f))
            except Exception:
                pass
        return sessions

    def rag_query(self, query: Dict[str, Any]) -> List[Dict]:
        results = []
        for session_file in SESSION_DIR.glob("*.json"):
            if session_file.name == "session_index.json":
                continue
            try:
                with open(session_file) as f:
                    session = json.load(f)
                score = self._score_session(session, query)
                if score > 0:
                    results.append({"session": session, "score": score})
            except Exception:
                pass

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _score_session(self, session: Dict, query: Dict) -> float:
        score = 0.0
        sid = query.get("session_id")
        if sid and session.get("session_id") == sid:
            score += 10.0

        gid = query.get("global_id")
        if gid and gid in session.get("identities_seen", []):
            score += 5.0

        min_duration = query.get("min_duration", 0)
        if session.get("duration", 0) >= min_duration:
            score += 1.0

        return score

    def _save_session(self, session: Session) -> None:
        filepath = SESSION_DIR / f"{session.session_id}.json"
        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def _update_index(self, session: Session) -> None:
        index = self._load_index_raw()
        index.append({
            "session_id": session.session_id,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "duration": (session.end_time or time.time()) - session.start_time,
            "identities": len(session.identities_seen),
            "events": len(session.events),
        })
        index = index[-100:]
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _load_index(self) -> None:
        index = self._load_index_raw()
        logger.info(f"[SESSION] Loaded session index: {len(index)} past sessions")

    def _load_index_raw(self) -> List[Dict]:
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []
