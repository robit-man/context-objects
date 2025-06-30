import threading
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

DB_PATH = Path("group_registry.db")

@dataclass
class GroupRegistry:
    path: Path = DB_PATH

    def __post_init__(self):
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS groups(
                name    TEXT PRIMARY KEY,
                chat_id INTEGER NOT NULL
            )
        """)
        self.conn.commit()
        self._lock = threading.Lock()

    def add(self, name: str, chat_id: int):
        grp = (name or "").strip()
        if not grp:
            return
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO groups(name,chat_id)
                VALUES(?,?)
                ON CONFLICT(name) DO UPDATE SET chat_id=excluded.chat_id
                """,
                (grp, chat_id)
            )
            self.conn.commit()

    def id_for(self, name: str) -> Optional[int]:
        grp = name.strip()
        with self._lock:
            cur = self.conn.execute(
                "SELECT chat_id FROM groups WHERE name=?", (grp,)
            )
            row = cur.fetchone()
        return row[0] if row else None

    def list_all(self) -> List[Dict[str,int]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT name,chat_id FROM groups ORDER BY name"
            )
            rows = cur.fetchall()
        return [{"name": r[0], "chat_id": r[1]} for r in rows]

# global instance
_GREG = GroupRegistry()
