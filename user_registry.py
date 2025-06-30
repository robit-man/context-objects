# user_registry.py
import threading
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

DB_PATH = Path("user_registry.db")

@dataclass
class UserRegistry:
    path: Path = DB_PATH

    def __post_init__(self):
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users(
                username TEXT PRIMARY KEY,
                user_id  INTEGER NOT NULL
            )
        """)
        self.conn.commit()
        self._lock = threading.Lock()

    def add(self, username: str, user_id: int):
        uname = username.lstrip("@").lower()
        if not uname:
            return
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO users(username,user_id)
                VALUES(?,?)
                ON CONFLICT(username) DO UPDATE SET user_id=excluded.user_id
                """,
                (uname, user_id)
            )
            self.conn.commit()

    def id_for(self, username: str) -> Optional[int]:
        uname = username.lstrip("@").lower()
        with self._lock:
            cur = self.conn.execute(
                "SELECT user_id FROM users WHERE username=?", (uname,)
            )
            row = cur.fetchone()
        return row[0] if row else None

    def list_all(self) -> List[Dict[str,int]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT username,user_id FROM users ORDER BY username"
            )
            rows = cur.fetchall()
        return [{"username":r[0],"id":r[1]} for r in rows]

# the one global instance
_REG = UserRegistry()
