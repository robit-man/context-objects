# context.py

import os
import uuid
import json
import math
import logging
import sqlite3
import threading
import contextlib
import numpy as np
from threading import Lock
from json import JSONDecodeError
import math, time, json, collections
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

if os.name == "nt":
    @contextlib.contextmanager
    def _locked(f, exclusive: bool):
        """
        No-op file lock on Windows. We rely on the in-process threading.Lock
        to serialize JSONL writes, and avoid msvcrt.locking permission errors.
        """
        yield f
else:                              # POSIX ─ use fcntl
    import fcntl

    @contextlib.contextmanager
    def _locked(f, exclusive: bool):
        lock = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(f.fileno(), lock)
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# ─ Utility: default clock ────────────────────────────────────────────────────────
def default_clock() -> datetime:
    return datetime.utcnow()


# ─ MemoryTrace ───────────────────────────────────────────────────────────────────
@dataclass
class MemoryTrace:
    """
    Records each recall occurrence of this context object,
    following the 'neurons that fire together wire together' principle.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: default_clock().strftime("%Y%m%dT%H%M%SZ"))
    coactivated_with: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─ ContextObject ────────────────────────────────────────────────────────────────
@dataclass
class ContextObject:
    """
    A universal, schema-versioned context object for chaining, retrieval,
    and self-improvement in an agent pipeline.
    Now domain-agnostic: holds segments, stages, or arbitrary artifacts
    (tool code, schemas, prompts, policies, plain knowledge).
    """

    # ─ Required (no default) ────────────────────────────────────────────────────
    domain: str           # e.g. "segment", "stage", or "artifact"
    component: str        # e.g. "tool_code", "schema", "prompt", "policy", "knowledge"
    semantic_label: str   # e.g. "select_tools", "user_prompt", "db_schema"

    # ─ Defaults & Optionals ─────────────────────────────────────────────────────
    schema_version: int = field(init=False, default=1)
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: Optional[str] = None
    timestamp: str = field(default_factory=lambda: default_clock().strftime("%Y%m%dT%H%M%SZ"))

    # Core references
    segment_ids: List[str] = field(default_factory=list)
    stage_id: Optional[str] = None
    references: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # Content & summaries
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Retrieval & similarity
    embedding: Optional[List[float]] = None
    retrieval_score: Optional[float] = None
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

    # Provenance & transformation
    provenance: Dict[str, Any] = field(default_factory=dict)

    # Graph & memory tier
    graph_node_id: Optional[str] = None
    memory_tier: Optional[str] = None

    # Associative memory
    memory_traces: List[MemoryTrace]        = field(default_factory=list)
    association_strengths: Dict[str, float] = field(default_factory=dict)
    recall_stats: Dict[str, Any]            = field(default_factory=lambda: {"count": 0, "last_recalled": None})
    firing_rate: Optional[float]            = None

    # Additional metadata & policies
    metadata: Dict[str, Any] = field(default_factory=dict)
    pinned: bool              = False
    last_accessed: Optional[str] = None
    expires_at: Optional[str]     = None
    acl: Dict[str, Any]         = field(default_factory=dict)
    batch_id: Optional[str]     = None
    dirty: bool                 = True

    def _ts_seconds(self) -> float:
        """Return timestamp (or last_accessed) in epoch seconds."""
        ts = self.last_accessed or self.timestamp
        return datetime.strptime(ts.rstrip("Z"), "%Y%m%dT%H%M%S").timestamp()


    def __post_init__(self):
        # create the lock here (not a dataclass field)
        self._lock = threading.Lock()

        # Normalize domain to one of the allowed buckets
        if self.domain not in {"segment", "stage", "artifact"}:
            self.domain = "artifact"

        # Validate required fields
        if not all([self.domain, self.component, self.semantic_label]):
            raise ValueError("domain, component and semantic_label are required")

        # Normalize timestamp if needed
        if isinstance(self.timestamp, datetime):
            self.timestamp = self.timestamp.strftime("%Y%m%dT%H%M%SZ")

        # Initialize last_accessed
        if not self.last_accessed:
            self.last_accessed = self.timestamp

        # Default memory tier by domain
        tier_map = {"segment": "STM", "stage": "LTM", "artifact": "WM"}
        self.memory_tier = tier_map.get(self.domain, "WM")

        # Setup logger
        self._logger = logging.getLogger(__name__)

    def __getstate__(self):
        """
        Exclude non-picklable items (like threading.Lock) during pickling.
        """
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_logger", None)
        return state

    def __setstate__(self, state):
        """
        Restore state and recreate lock and logger after unpickling.
        """
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def __repr__(self):
        return f"<ContextObject {self.domain}/{self.component}/{self.semantic_label}@{self.timestamp}>"

    # ─ Core Helpers ────────────────────────────────────────────────
    def touch(self):
        """Mark accessed just now."""
        self.last_accessed = default_clock().strftime("%Y%m%dT%H%M%SZ")
        self.dirty = True

    def set_expiration(self, ttl_seconds: int):
        """Expire after TTL seconds."""
        exp = default_clock() + timedelta(seconds=ttl_seconds)
        self.expires_at = exp.strftime("%Y%m%dT%H%M%SZ")
        self.dirty = True

    def compute_embedding(
        self,
        default_embedder: Callable[[str], List[float]],
        component_embedder: Optional[Dict[str, Callable[[str], List[float]]]] = None
    ):
        """
        Generate embedding from summary via provided embedder(s).
        If a component_embedder map is given and contains this.component,
        that function is used; otherwise default_embedder is used.
        """
        if not self.summary:
            return
        fn = default_embedder
        if component_embedder and self.component in component_embedder:
            fn = component_embedder[self.component]
        self.embedding = fn(self.summary)
        self.dirty = True

    def log_context(self, level=logging.INFO):
        """Emit full context JSON to logs."""
        self._logger.log(level, json.dumps(self.to_dict()))

    def record_recall(
        self,
        stage_id: Optional[str],
        coactivated_with: Optional[List[str]] = None,
        retrieval_score: Optional[float] = None
    ):
        """
        Register a recall event: adds a MemoryTrace, updates stats & associations.
        Thread-safe.
        """
        with self._lock:
            mt = MemoryTrace(
                stage_id=stage_id,
                coactivated_with=coactivated_with or [],
                metadata={"retrieval_score": retrieval_score}
            )
            self.memory_traces.append(mt)

            # update recall stats
            stats = self.recall_stats
            stats["count"] += 1
            stats["last_recalled"] = mt.timestamp

            # simplistic firing_rate
            self.firing_rate = 1.0 / stats["count"] if stats["count"] else None

            # association strengthening
            for other in mt.coactivated_with:
                self.association_strengths[other] = self.association_strengths.get(other, 0.0) + 1.0

            # mark accessed
            self.touch()

    # ─ Serialization ──────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["memory_traces"] = [asdict(mt) for mt in self.memory_traces]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    


    @classmethod
    def make_narrative(
        cls,
        entry: str,
        tags: Optional[List[str]] = None,
        **metadata
    ) -> "ContextObject":
        """
        Create *one* narrative row per unique `entry`.
        If the most-recent narrative row already has an identical summary,
        reuse it instead of creating a duplicate.
        """
        repo: ContextRepository = ContextRepository.instance()  # singleton accessor

        # latest narrative row, if any
        rows = repo.query(lambda c: c.component == "narrative")
        rows.sort(key=lambda c: c.timestamp, reverse=True)
        latest = rows[0] if rows else None

        if latest and latest.summary == entry:
            # ensure tags/metadata are up-to-date, touch timestamp once
            if tags:
                for t in tags:
                    if t not in latest.tags:
                        latest.tags.append(t)
            latest.metadata.update(metadata)
            latest.touch()
            repo.save(latest)
            return latest

        # otherwise insert a fresh row
        obj = cls(
            domain="artifact",
            component="narrative",
            semantic_label="self_narrative",
        )
        obj.summary = entry
        obj.metadata.update(metadata)
        obj.tags = tags or ["narrative"]
        return obj

    
    @classmethod
    def make_success(cls, description: str, refs: List[str] = None) -> "ContextObject":
        """Log that an action or plan succeeded."""
        obj = cls(domain="stage", component="success", semantic_label="success")
        obj.summary = description
        obj.references = refs or []
        obj.tags = ["success"]
        return obj

    @classmethod
    def make_failure(cls, description: str, refs: List[str] = None) -> "ContextObject":
        """Log that an action or plan failed."""
        obj = cls(domain="stage", component="failure", semantic_label="failure")
        obj.summary = description
        obj.references = refs or []
        obj.tags = ["failure"]
        return obj
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextObject":
        mts = [MemoryTrace(**mt) for mt in data.get("memory_traces", [])]
        obj = cls(
            domain=data["domain"],
            component=data["component"],
            semantic_label=data["semantic_label"],
            version=data.get("version"),
            timestamp=data.get("timestamp", default_clock().strftime("%Y%m%dT%H%M%SZ"))
        )
        # Overwrite defaults with saved values
        obj.context_id             = data.get("context_id", obj.context_id)
        obj.segment_ids            = data.get("segment_ids", [])
        obj.stage_id               = data.get("stage_id")
        obj.references             = data.get("references", [])
        obj.children               = data.get("children", [])
        obj.summary                = data.get("summary")
        obj.tags                   = data.get("tags", [])
        obj.embedding              = data.get("embedding")
        obj.retrieval_score        = data.get("retrieval_score")
        obj.retrieval_metadata     = data.get("retrieval_metadata", {})
        obj.provenance             = data.get("provenance", {})
        obj.graph_node_id          = data.get("graph_node_id")
        obj.memory_tier            = data.get("memory_tier", obj.memory_tier)
        obj.memory_traces          = mts
        obj.association_strengths  = data.get("association_strengths", {})
        obj.recall_stats           = data.get("recall_stats", {"count": 0, "last_recalled": None})
        obj.firing_rate            = data.get("firing_rate")
        obj.metadata               = data.get("metadata", {})
        obj.pinned                 = data.get("pinned", False)
        obj.last_accessed          = data.get("last_accessed", obj.last_accessed)
        obj.expires_at             = data.get("expires_at")
        obj.acl                    = data.get("acl", {})
        obj.batch_id               = data.get("batch_id")
        obj.dirty                  = data.get("dirty", True)
        return obj

    @staticmethod
    def from_json(s: str) -> "ContextObject":
        return ContextObject.from_dict(json.loads(s))

    # ─ Factory Methods ───────────────────────────────────────────
    @classmethod
    def make_segment(
        cls,
        semantic_label: str,
        content_refs: List[str],
        tags: Optional[List[str]] = None,
        **metadata
    ) -> "ContextObject":
        obj = cls(domain="segment", component="segment", semantic_label=semantic_label)
        obj.segment_ids = content_refs
        obj.tags = tags or ["segment"]
        obj.metadata.update(metadata)
        return obj

    @classmethod
    def make_stage(
        cls,
        stage_name: str,
        input_refs: List[str],
        output: Any,
        **metadata
    ) -> "ContextObject":
        obj = cls(domain="stage", component=stage_name, semantic_label=stage_name)
        obj.references = input_refs
        obj.tags = [stage_name]
        obj.metadata.update(metadata)
        obj.metadata["output"] = output
        return obj

    @classmethod
    def make_tool_code(
        cls,
        label: str,
        code: str,
        tags: Optional[List[str]] = None,
        **metadata
    ) -> "ContextObject":
        obj = cls(domain="artifact", component="tool_code", semantic_label=label)
        obj.summary = code[:120] + "…" if len(code) > 120 else code
        obj.tags = tags or ["code"]
        obj.metadata.update(metadata)
        obj.metadata["code"] = code
        return obj

    @classmethod
    def make_schema(
        cls,
        label: str,
        schema_def: str,
        tags: Optional[List[str]] = None,
        **metadata
    ) -> "ContextObject":
        obj = cls(domain="artifact", component="schema", semantic_label=label)
        obj.summary = schema_def[:120] + "…" if len(schema_def) > 120 else schema_def
        obj.tags = tags or ["schema"]
        obj.metadata.update(metadata)
        obj.metadata["schema"] = schema_def
        return obj

    @classmethod
    def make_prompt(
        cls,
        label: str,
        prompt_text: str,
        tags: Optional[List[str]] = None,
        **metadata
    ) -> "ContextObject":
        obj = cls(domain="artifact", component="prompt", semantic_label=label)
        obj.summary = prompt_text[:120] + "…" if len(prompt_text) > 120 else prompt_text
        obj.tags = tags or ["prompt"]
        obj.metadata.update(metadata)
        obj.metadata["prompt"] = prompt_text
        return obj

    @classmethod
    def make_policy(
        cls,
        label: str,
        policy_text: str,
        tags: Optional[List[str]] = None,
        **metadata
    ) -> "ContextObject":
        obj = cls(domain="artifact", component="policy", semantic_label=label)
        obj.summary = policy_text[:120] + "…" if len(policy_text) > 120 else policy_text
        obj.tags = tags or ["policy"]
        obj.metadata.update(metadata)
        obj.metadata["policy"] = policy_text
        return obj

    @classmethod
    def make_knowledge(
        cls,
        label: str,
        content: str,
        tags: Optional[List[str]] = None,
        **metadata
    ) -> "ContextObject":
        obj = cls(domain="artifact", component="knowledge", semantic_label=label)
        obj.summary = content[:120] + "…" if len(content) > 120 else content
        obj.tags = tags or ["knowledge"]
        obj.metadata.update(metadata)
        obj.metadata["content"] = content
        return obj


def sanitize_jsonl(path: str):
    """
    Reads 'path' under shared lock, drops any corrupted JSON lines,
    logs them into 'path.corrupt', and atomically rewrites with only valid ones.
    """
    if not os.path.exists(path):
        return

    corrupt_path = path + ".corrupt"
    good_lines = []
    bad_entries = []

    # 1) Read & classify under shared lock
    with open(path, "r+", encoding="utf-8") as f, _locked(f, exclusive=False):
        for idx, line in enumerate(f, start=1):
            try:
                json.loads(line)
                good_lines.append(line)
            except JSONDecodeError as e:
                logging.warning(f"sanitize_jsonl: dropping invalid JSON at line {idx} in {path}: {e}")
                bad_entries.append((idx, line.rstrip("\n")))

        # 2) If we found bad lines, log them out
        if bad_entries:
            with open(corrupt_path, "a", encoding="utf-8") as cf:
                for idx, text in bad_entries:
                    cf.write(f"{datetime.utcnow().isoformat()} LINE {idx}: {text}\n")

        # 3) Rewrite the JSONL in-place
        f.seek(0)
        f.truncate()
        f.writelines(good_lines)
        f.flush()
        os.fsync(f.fileno())

        
class JSONLContextRepository:
    _singleton = None

    def __init__(self, path: str):
        # 0) Repair any pre‐existing corruption
        sanitize_jsonl(path)

        # 1) Ensure directory exists
        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)

        # 2) Initialize file and lock
        self.path = path
        self._lock = threading.Lock()
        open(self.path, "a").close()  # create file if missing

        # 3) Register singleton
        JSONLContextRepository._singleton = self

    def get(self, context_id: str) -> ContextObject:
        """
        Look up a single context; if a JSON error is encountered,
        attempt one repair pass then retry.
        """
        tried_sanitize = False

        while True:
            with open(self.path, "r", encoding="utf-8") as f, _locked(f, exclusive=False):
                for lineno, line in enumerate(f, start=1):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.warning(
                            f"JSONLContextRepository.get: parse error on line {lineno}: {e}"
                        )
                        if not tried_sanitize:
                            sanitize_jsonl(self.path)
                            tried_sanitize = True
                            break  # abort this read, retry after repair
                        else:
                            continue  # skip this line on second pass
                    if data.get("context_id") == context_id:
                        return ContextObject.from_dict(data)
                else:
                    # finished file without finding ID
                    break

            # if we repaired once already, don't loop again
            if tried_sanitize:
                break

        raise KeyError(f"Context {context_id} not found")

    def save(self, ctx: ContextObject) -> None:
        """
        Append a dirty context object to JSONL under exclusive lock.
        """
        if not ctx.dirty:
            return
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f, _locked(f, exclusive=True):
                f.write(ctx.to_json() + "\n")
                f.flush()
                os.fsync(f.fileno())
            ctx.dirty = False

    def delete(self, context_id: str) -> None:
        """
        Remove all entries matching context_id, skipping any corrupted lines.
        """
        # First ensure file is clean
        sanitize_jsonl(self.path)

        kept = []
        with self._lock:
            with open(self.path, "r+", encoding="utf-8") as f, _locked(f, exclusive=True):
                for lineno, line in enumerate(f, start=1):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logging.warning(
                            f"JSONLContextRepository.delete: skipping bad line {lineno}"
                        )
                        continue
                    if data.get("context_id") != context_id:
                        kept.append(data)
                f.seek(0)
                f.truncate()
                for entry in kept:
                    f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def query(self, filter_fn) -> list[ContextObject]:
        """
        Iterate all contexts, skipping any bad lines; attempt one repair pass if needed.
        """
        results = []
        tried_sanitize = False

        while True:
            with open(self.path, "r", encoding="utf-8") as f, _locked(f, exclusive=False):
                bad_line = False
                for lineno, line in enumerate(f, start=1):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.warning(
                            f"JSONLContextRepository.query: parse error on line {lineno}: {e}"
                        )
                        if not tried_sanitize:
                            sanitize_jsonl(self.path)
                            tried_sanitize = True
                            bad_line = True
                            break  # restart after repair
                        else:
                            continue
                    ctx = ContextObject.from_dict(data)
                    if filter_fn(ctx):
                        results.append(ctx)

                if bad_line:
                    # we repaired; retry the query on clean file
                    continue
                # no bad line or already sanitized
                break

        return results

    @classmethod
    def instance(cls) -> "JSONLContextRepository":
        if cls._singleton is None:
            raise RuntimeError("ContextRepository not initialised")
        return cls._singleton
# ──────────────────────────────────────────────────────────────────────────────
# SQLite-backed archive for long-term storage
# ──────────────────────────────────────────────────────────────────────────────
class SQLiteContextRepository:
    def __init__(self, db_path="context.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = Lock()
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("""
          CREATE TABLE IF NOT EXISTS contexts (
            context_id     TEXT PRIMARY KEY,
            timestamp      TEXT,
            last_accessed  TEXT,
            json_blob      TEXT
          )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON contexts(last_accessed)")
        self.conn.commit()

    def save(self, ctx):
        blob = ctx.to_json()
        now  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        with self._lock:
            self.conn.execute("""
              INSERT INTO contexts(context_id,timestamp,last_accessed,json_blob)
              VALUES(?,?,?,?)
              ON CONFLICT(context_id) DO UPDATE SET
                json_blob     = excluded.json_blob,
                last_accessed = excluded.last_accessed
            """, (ctx.context_id, ctx.timestamp, now, blob))
            self.conn.commit()
        ctx.dirty = False

    def get(self, cid):
        cur = self.conn.cursor()
        cur.execute("SELECT json_blob FROM contexts WHERE context_id=?", (cid,))
        row = cur.fetchone()
        if not row:
            raise KeyError(cid)
        from context import ContextObject
        return ContextObject.from_json(row[0])

    def delete(self, cid):
        with self._lock:
            self.conn.execute("DELETE FROM contexts WHERE context_id=?", (cid,))
            self.conn.commit()

    def query(self, filter_fn):
        from context import ContextObject
        out = []
        for (blob,) in self.conn.execute("SELECT json_blob FROM contexts"):
            obj = ContextObject.from_json(blob)
            if filter_fn(obj):
                out.append(obj)
        return out

# ──────────────────────────────────────────────────────────────────────────────
# Hybrid: keep recent in JSONL, archive older into SQLite
# ──────────────────────────────────────────────────────────────────────────────
class HybridContextRepository:
    _singleton: "HybridContextRepository" = None

    def __init__(
        self,
        jsonl_path: str = "context.jsonl",
        sqlite_path: str = "context.db",
        archive_max_mb: float = 10.0,   # max JSONL size in megabytes
    ):
        self.json_repo = JSONLContextRepository(jsonl_path)
        self.sql_repo  = SQLiteContextRepository(sqlite_path)
        self._max_bytes = int(archive_max_mb * 1024 * 1024)
        HybridContextRepository._singleton = self                 # register singleton

    def save(self, ctx: ContextObject) -> None:
        # always append the new object
        self.json_repo.save(ctx)
        # then prune by size if needed
        self._archive_by_size()

    def get(self, cid: str) -> ContextObject:
        try:
            return self.json_repo.get(cid)
        except KeyError:
            return self.sql_repo.get(cid)

    def delete(self, cid: str) -> None:
        self.json_repo.delete(cid)
        try:
            self.sql_repo.delete(cid)
        except KeyError:
            pass

    def query(self, filter_fn):
        seen = set()
        out  = []
        for repo in (self.json_repo, self.sql_repo):
            for ctx in repo.query(filter_fn):
                if ctx.context_id not in seen:
                    seen.add(ctx.context_id)
                    out.append(ctx)
        return out

    def _archive_by_size(self):
        path = self.json_repo.path
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        if size <= self._max_bytes:
            return

        # read all lines
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # parse lines, skip non-archiveable objects
        entries: list[tuple[int, datetime, ContextObject]] = []
        for idx, line in enumerate(lines):
            try:
                obj = ContextObject.from_json(line)
            except Exception:
                continue

            # <<< NEW: skip tool-code artifacts entirely >>>
            if obj.domain == "artifact" and obj.component == "tool_code":
                continue

            # never archive schema artifacts either
            if obj.domain == "artifact" and obj.component == "schema":
                continue

            # track (index, timestamp, object)
            ts = datetime.strptime(obj.timestamp, "%Y%m%dT%H%M%SZ")
            entries.append((idx, ts, obj))

        # sort oldest first
        entries.sort(key=lambda x: x[1])

        to_remove = set()
        # remove oldest until under size limit
        for idx, ts, obj in entries:
            # archive into SQLite
            self.sql_repo.save(obj)
            to_remove.add(idx)

            # estimate remaining size
            rem = sum(
                len(l.encode("utf-8"))
                for i, l in enumerate(lines)
                if i not in to_remove
            )
            if rem <= self._max_bytes:
                break

        # rewrite JSONL without archived lines
        with open(path, "w", encoding="utf-8") as f:
            for i, l in enumerate(lines):
                if i not in to_remove:
                    f.write(l)


    @classmethod
    def instance(cls) -> "ContextRepository":
        """
        Return the one live repository registered in __init__.
        Raises RuntimeError if called before the first repo is created.
        """
        if cls._singleton is None:
            raise RuntimeError("ContextRepository has not been initialised yet")
        
        return cls._singleton

# ╔══════════════════════════════════════════════════════════════╗
# ║            H O L O G R A P H I C   M E M O R Y               ║
# ╚══════════════════════════════════════════════════════════════╝
# --- internal parameters (tweak freely) -------------------------
_HMR_SIM_THRESH   = 0.35        # cosine similarity edge cut-off
_HMR_TAG_W        = 0.6         # weight on shared-tag edges
_HMR_REF_W        = 1.0         # explicit reference edge weight
_HMR_SIM_W        = 0.4         # multiplier on sim edges
_HMR_DECAY_SECS   = 60 * 60 * 24   # temporal proximity half-life


def _ts_seconds(self) -> float:
    """Return timestamp (or last_accessed) in epoch seconds."""
    ts = self.last_accessed or self.timestamp
    return datetime.strptime(ts.rstrip("Z"), "%Y%m%dT%H%M%S").timestamp()


ContextRepository = HybridContextRepository

# ─ MemoryManager / Service Layer ──────────────────────────────────────────────
class MemoryManager:
    """
    High-level service for associative recall, reinforcement, pruning,
    and spreading-activation (“thought chains”).
    """

    _graph: Dict[str, Dict[str, float]] = {}   # ← NEW: shared holographic graph

    def __init__(self, repo: ContextRepository):
        self.repo = repo

    def recall(
        self,
        seed_ids: List[str],
        k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> List[ContextObject]:
        weights = weights or {"assoc": 1.0, "recency": 1.0}
        now = default_clock()

        # 1) one‐hop candidate scoring
        scores: Dict[str, float] = {}
        for sid in seed_ids:
            seed = self.repo.get(sid)
            for oid, strength in seed.association_strengths.items():
                base = strength * weights["assoc"]
                other = self.repo.get(oid)
                last = datetime.strptime(other.last_accessed, "%Y%m%dT%H%M%SZ")
                age = (now - last).total_seconds()
                base += weights["recency"] / (1.0 + age)
                scores[oid] = scores.get(oid, 0.0) + base

        # 2) pick top‐k
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results: List[ContextObject] = []

        for cid, score in top:
            ctx = self.repo.get(cid)

            # a) stamp retrieval_score & metadata
            ctx.retrieval_score = score
            ctx.retrieval_metadata = {"seed_ids": seed_ids}

            # b) record the recall event
            ctx.record_recall(
                stage_id="recall",
                coactivated_with=seed_ids,
                retrieval_score=score
            )

            # c) save the updated object
            self.repo.save(ctx)

            results.append(ctx)

        return results

    def spread_activation(
        self,
        seed_ids: List[str],
        hops: int = 3,
        decay: float = 0.5,
        assoc_weight: float = 1.0,
        recency_weight: float = 1.0,
    ) -> Dict[str, float]:
        """
        Perform spreading-activation from seed_ids over N hops.

        - hops: max graph distance
        - decay: per-hop multiplier (0 < decay ≤ 1)
        - assoc_weight: scales edge strengths
        - recency_weight: bonus per node based on recency
        Returns a map {context_id: activation_score}.
        """
        now = default_clock()
        # Initialize activations
        activation: Dict[str, float] = {cid: 1.0 for cid in seed_ids}

        for hop in range(1, hops + 1):
            new_act: Dict[str, float] = {}
            for cid, act in list(activation.items()):
                try:
                    ctx = self.repo.get(cid)
                except KeyError:
                    continue
                for neigh, strength in ctx.association_strengths.items():
                    inc = act * strength * assoc_weight * (decay ** (hop - 1))
                    new_act[neigh] = new_act.get(neigh, 0.0) + inc
            # Merge new activations
            for cid, inc in new_act.items():
                activation[cid] = activation.get(cid, 0.0) + inc

        # Recency bonus
        for cid in list(activation.keys()):
            try:
                ctx = self.repo.get(cid)
                last = datetime.strptime(ctx.last_accessed, "%Y%m%dT%H%M%SZ")
                age = (now - last).total_seconds()
                activation[cid] += recency_weight / (1.0 + age)
            except Exception:
                continue

        return activation

    def decay_and_promote(
        self,
        half_life: float = 86_400.0,     # seconds in a day
        promote_threshold: int = 3
    ) -> None:
        import math

        now = default_clock()
        for row in list(self.repo.query(lambda c: True)):
            try:
                ctx = self.repo.get(row.context_id)
            except KeyError:
                continue

            last_ts = datetime.strptime(ctx.last_accessed, "%Y%m%dT%H%M%SZ")
            delta = (now - last_ts).total_seconds()

            new_strengths: Dict[str, float] = {}
            for oid, strength in ctx.association_strengths.items():
                try:
                    self.repo.get(oid)
                except KeyError:
                    continue
                decayed = strength * math.exp(-delta / half_life)
                if decayed > 1e-6:
                    new_strengths[oid] = decayed

            should_promote = ctx.recall_stats.get("count", 0) >= promote_threshold
            if new_strengths != ctx.association_strengths or (should_promote and ctx.memory_tier!="LTM"):
                ctx.association_strengths = new_strengths
                if should_promote:
                    ctx.memory_tier = "LTM"
                ctx.touch()
                self.repo.save(ctx)

    def reinforce(self, context_id: str, coactivated: List[str]) -> None:
        """
        Strengthen edges between `context_id` and every ID in `coactivated`
        while skipping dangling references.
        """
        try:
            ctx = self.repo.get(context_id)
        except KeyError:
            return

        valid_refs: List[str] = []
        for rid in coactivated:
            try:
                self.repo.get(rid)
                valid_refs.append(rid)
            except KeyError:
                continue

        if not valid_refs:
            return

        ctx.record_recall(stage_id=None, coactivated_with=valid_refs)
        self.repo.save(ctx)

    def prune(self, ttl_hours: int) -> None:
        cutoff = default_clock() - timedelta(hours=ttl_hours)
        def stale(c: ContextObject) -> bool:
            la = datetime.strptime(c.last_accessed, "%Y%m%dT%H%M%SZ")
            return la < cutoff and not c.pinned
        for ctx in self.repo.query(stale):
            self.repo.delete(ctx.context_id)

    def _add_edge(self, src: str, dst: str, w: float) -> None:
        if src == dst:
            return
        self._graph.setdefault(src, {})
        self._graph[src][dst] = self._graph[src].get(dst, 0.0) + w

    # ------------- 1️⃣  register_relationships ----------------------
    def register_relationships(self,
                               ctx: ContextObject,
                               embed_fn: Callable[[str], np.ndarray]) -> None:
        """
        Call once after saving a new/updated ContextObject.
        • Skips re-registering relationships for the same ctx.
        • Uses an in-memory cache for embeddings.
        • Limits similarity scans to the last N items.
        """
        import math
        # Keep track of which contexts we've already processed
        if not hasattr(self, "_registered_ctxs"):
            self._registered_ctxs: set[str] = set()
        cid = ctx.context_id
        if cid in self._registered_ctxs:
            return
        self._registered_ctxs.add(cid)

        # ---------- explicit references ----------
        for rid in ctx.references:
            self._add_edge(cid, rid, _HMR_REF_W)
            self._add_edge(rid, cid, _HMR_REF_W)

        # ---------- shared tags ----------
        for tag in ctx.tags:
            tag_node = f"tag::{tag}"
            self._add_edge(cid, tag_node, _HMR_TAG_W)
            self._add_edge(tag_node, cid, _HMR_TAG_W)

        # ---------- semantic similarity ----------
        # initialize embedding cache if missing
        if not hasattr(self, "_embed_cache"):
            self._embed_cache: Dict[str, np.ndarray] = {}
        def _get_vec(text: str) -> np.ndarray:
            if text not in self._embed_cache:
                self._embed_cache[text] = embed_fn(text)
            return self._embed_cache[text]

        try:
            v1 = _get_vec(ctx.summary or "")
            # only compare to the most recent M items
            recents = self.repo.query(lambda _: True)[-200:]
            for other in recents:
                if other.context_id == cid or not other.summary:
                    continue
                # skip if we've already embedded this other
                txt = other.summary
                v2 = _get_vec(txt)
                sim = float(np.dot(v1, v2) /
                            (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                if sim >= _HMR_SIM_THRESH:
                    w = _HMR_SIM_W * sim
                    self._add_edge(cid, other.context_id, w)
                    self._add_edge(other.context_id, cid, w)
        except Exception:
            pass

        # ---------- temporal proximity (<10 min) ----------
        now = ctx._ts_seconds()
        # limit scan to contexts added in the last window
        window = 600  # seconds
        candidates = [
            c for c in self.repo.query(lambda c: True)
            if abs(now - c._ts_seconds()) <= window and c.context_id != cid
        ]
        for other in candidates:
            age = abs(now - other._ts_seconds())
            w = math.exp(-age / _HMR_DECAY_SECS)
            self._add_edge(cid, other.context_id, w)
            self._add_edge(other.context_id, cid, w)

    # ------------- 2️⃣  holographic_recall --------------------------
    def holographic_recall(self,
                        cue_ids: List[str] | None = None,
                        cue_text: str | None = None,
                        hops: int = 2,
                        top_n: int = 10,
                        embed_fn: Callable[[str], np.ndarray] | None = None
                        ) -> List[ContextObject]:
        """
        Fuse cue_ids &/or cue_text into a single excitation vector,
        run multi-hop spreading activation over _graph, return top_n ContextObjects.
        """
        cue_ids = cue_ids or []
        activation: Dict[str, float] = collections.Counter({cid: 1.0 for cid in cue_ids})

        # text cue → similarity edges once
        if cue_text and embed_fn:
            qv = embed_fn(cue_text)
            for c in self.repo.query(lambda _: True):
                if not c.summary:
                    continue
                vv  = embed_fn(c.summary)
                sim = float(np.dot(qv, vv) /
                            (np.linalg.norm(qv) * np.linalg.norm(vv) + 1e-9))
                if sim >= _HMR_SIM_THRESH:
                    activation[c.context_id] += _HMR_SIM_W * sim

        # hop propagation
        frontier = dict(activation)
        for _ in range(hops):
            new_frontier = collections.Counter()
            for nid, act in frontier.items():
                for nbr, w in self._graph.get(nid, {}).items():
                    new_frontier[nbr] += act * w
            for k, v in new_frontier.items():
                activation[k] += v
            frontier = new_frontier

        # rank & materialise
        ranked = sorted(activation.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        out    = []
        for cid, score in ranked:
            try:
                obj = self.repo.get(cid)
                obj.retrieval_score = score
                out.append(obj)
            except KeyError:
                continue
        return out



# ─ Graph Interface Layer ───────────────────────────────────────────────────────
class ContextGraph:
    """
    In-memory directed graph with weighted edges for context associations.
    """
    def __init__(self):
        # map: from_id → { to_id → weight }
        self.adj: Dict[str, Dict[str, float]] = {}

    def add_node(self, ctx: ContextObject) -> None:
        self.adj.setdefault(ctx.context_id, {})

    def add_edge(self, from_id: str, to_id: str, weight: float = 1.0) -> None:
        self.adj.setdefault(from_id, {})
        self.adj[from_id][to_id] = self.adj[from_id].get(to_id, 0.0) + weight

    def neighbors(self, context_id: str) -> List[str]:
        return list(self.adj.get(context_id, {}).keys())

    def neighbors_with_weights(self, context_id: str) -> Dict[str, float]:
        return dict(self.adj.get(context_id, {}))
