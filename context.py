# context.py

import uuid
import json
import math
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
import os
import contextlib
import sqlite3
from threading import Lock

if os.name == "nt":                # Windows ─ use msvcrt
    import msvcrt

    @contextlib.contextmanager
    def _locked(f, exclusive: bool):
        mode = msvcrt.LK_LOCK if exclusive else msvcrt.LK_NBLCK
        try:
            msvcrt.locking(f.fileno(), mode, 1)
            yield f
        finally:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

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

class JSONLContextRepository:
    _singleton = None

    def __init__(self, path: str):
        import threading
        from context import _locked

        # ensure the directory exists (allow current dir if no dirname)
        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)

        self.path = path
        self._lock = threading.Lock()
        # create the file if missing
        open(self.path, "a").close()
        JSONLContextRepository._singleton = self

    def get(self, context_id: str):
        import json
        from context import ContextObject, _locked
        with open(self.path, "r", encoding="utf-8") as f, _locked(f, exclusive=False):
            for line in f:
                data = json.loads(line)
                if data["context_id"] == context_id:
                    return ContextObject.from_dict(data)
        raise KeyError(f"Context {context_id} not found")

    def save(self, ctx):
        import os
        from context import _locked
        if not ctx.dirty:
            return
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f, _locked(f, exclusive=True):
                f.write(ctx.to_json() + "\n")
                f.flush(); os.fsync(f.fileno())
            ctx.dirty = False

    def delete(self, context_id: str):
        import json
        from context import _locked
        with self._lock:
            with open(self.path, "r+", encoding="utf-8") as f, _locked(f, exclusive=True):
                entries = [json.loads(l) for l in f if json.loads(l)["context_id"] != context_id]
                f.seek(0); f.truncate()
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
                f.flush(); os.fsync(f.fileno())

    def query(self, filter_fn):
        import json
        from context import ContextObject, _locked
        results = []
        with open(self.path, "r", encoding="utf-8") as f, _locked(f, exclusive=False):
            for line in f:
                ctx = ContextObject.from_dict(json.loads(line))
                if filter_fn(ctx):
                    results.append(ctx)
        return results

    @classmethod
    def instance(cls):
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
        entries = []
        for idx, line in enumerate(lines):
            try:
                obj = ContextObject.from_json(line)
            except Exception:
                continue
            # never archive schema artifacts
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
            rem = 0
            for i, l in enumerate(lines):
                if i not in to_remove:
                    rem += len(l.encode("utf-8"))
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

    
ContextRepository = HybridContextRepository

# ─ MemoryManager / Service Layer ──────────────────────────────────────────────
class MemoryManager:
    """
    High-level service for associative recall, reinforcement, and pruning.
    """
    def __init__(self, repo: ContextRepository):
        self.repo = repo

    def recall(
        self,
        seed_ids: List[str],
        k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> List[ContextObject]:
        weights = weights or {"assoc": 1.0, "recency": 1.0}
        candidates: Dict[str, float] = {}
        now = default_clock()

        for sid in seed_ids:
            seed = self.repo.get(sid)
            for other_id, strength in seed.association_strengths.items():
                score = strength * weights["assoc"]
                other = self.repo.get(other_id)
                last = datetime.strptime(other.last_accessed, "%Y%m%dT%H%M%SZ")
                age = (now - last).total_seconds()
                score += weights["recency"] / (1.0 + age)
                candidates[other_id] = candidates.get(other_id, 0.0) + score

        sorted_ids = sorted(candidates, key=lambda x: candidates[x], reverse=True)
        return [self.repo.get(cid) for cid in sorted_ids[:k]]
    

    def decay_and_promote(
        self,
        half_life: float = 86_400.0,     # seconds in a day
        promote_threshold: int = 3
    ) -> None:
        import math
        from datetime import datetime

        now = default_clock()

        # snapshot to avoid mutation-during-iteration
        for row in list(self.repo.query(lambda c: True)):
            try:
                ctx = self.repo.get(row.context_id)
            except KeyError:
                continue

            # compute the decayed strengths
            last_ts = datetime.strptime(ctx.last_accessed, "%Y%m%dT%H%M%SZ")
            delta = (now - last_ts).total_seconds()
            new_strengths = {}
            for oid, strength in ctx.association_strengths.items():
                try:
                    self.repo.get(oid)
                except KeyError:
                    continue
                decayed = strength * math.exp(-delta / half_life)
                if decayed > 1e-6:
                    new_strengths[oid] = decayed

            # figure out if we need to promote
            should_promote = ctx.recall_stats.get("count", 0) >= promote_threshold
            promoted = (ctx.memory_tier != "LTM") and should_promote

            # only update & save if anything really changed
            if new_strengths != ctx.association_strengths or promoted:
                ctx.association_strengths = new_strengths
                if should_promote:
                    ctx.memory_tier = "LTM"
                ctx.touch()
                self.repo.save(ctx)



    # ──────────────────────────────────────────────────────────────────────────
    # CURRENT (replace everything in this method)
    # ──────────────────────────────────────────────────────────────────────────
    def reinforce(self, context_id: str, coactivated: List[str]) -> None:
        """
        Strengthen edges between `context_id` and every ID in `coactivated`
        while **skipping dangling references safely**.
        """
        # ensure the target still exists
        try:
            ctx = self.repo.get(context_id)
        except KeyError:
            return  # target row was deleted → nothing to do

        # keep only co-activated IDs that still resolve
        valid_refs: List[str] = []
        for rid in coactivated:
            try:
                self.repo.get(rid)
                valid_refs.append(rid)
            except KeyError:
                continue  # silently drop dangling ref

        if not valid_refs:
            return  # nothing valid to link

        ctx.record_recall(stage_id=None, coactivated_with=valid_refs)
        self.repo.save(ctx)



    def prune(self, ttl_hours: int) -> None:
        cutoff = default_clock() - timedelta(hours=ttl_hours)
        def stale(c: ContextObject) -> bool:
            la = datetime.strptime(c.last_accessed, "%Y%m%dT%H%M%SZ")
            return la < cutoff and not c.pinned
        for ctx in self.repo.query(stale):
            self.repo.delete(ctx.context_id)


# ─ Graph Interface Layer ───────────────────────────────────────────────────────
class ContextGraph:
    """
    Simple in-memory directed graph for context-object relations.
    """
    def __init__(self):
        self.adj: Dict[str, List[str]] = {}

    def add_node(self, ctx: ContextObject) -> None:
        self.adj.setdefault(ctx.context_id, [])

    def add_edge(self, from_id: str, to_id: str, label: Optional[str] = None) -> None:
        self.adj.setdefault(from_id, []).append(to_id)

    def neighbors(self, context_id: str) -> List[str]:
        return self.adj.get(context_id, [])
