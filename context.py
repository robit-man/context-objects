# context.py  — core context model + repositories

import os
import uuid
import json
import logging
import sqlite3
import threading
import contextlib
import collections
from pathlib import Path
from threading import Lock
from json import JSONDecodeError
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Cross-platform file locking for JSONL (Windows lock optional to avoid flakiness)
# ──────────────────────────────────────────────────────────────────────────────
if os.name == "nt":
    try:
        import msvcrt  # noqa: F401

        @contextlib.contextmanager
        def _locked(f, exclusive: bool):
            """
            Best-effort Windows lock. Disabled by default due to common AV/permission
            conflicts. Enable by setting CTX_ENABLE_WIN_LOCK=1.
            """
            if not os.environ.get("CTX_ENABLE_WIN_LOCK"):
                yield f
                return
            # msvcrt.locking requires a byte range; we lock 1 byte at start.
            mode = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
            try:
                msvcrt.locking(f.fileno(), mode, 1)
                yield f
            finally:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
    except Exception:
        @contextlib.contextmanager
        def _locked(f, exclusive: bool):
            yield f
else:  # POSIX ─ use fcntl
    import fcntl  # type: ignore

    @contextlib.contextmanager
    def _locked(f, exclusive: bool):
        lock = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(f.fileno(), lock)
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ─ Utilities ──────────────────────────────────────────────────────────────────
_TS_FMT = "%Y%m%dT%H%M%SZ"



_TS_FMT      = "%Y%m%dT%H%M%SZ"
_TS_FMT_NOZ  = "%Y%m%dT%H%M%S"

def default_clock() -> datetime:
    # always UTC-aware
    return datetime.now(timezone.utc)

def _fmt_ts(dt: datetime | None = None) -> str:
    # normalize to UTC and emit canonical Z-suffixed form
    dt = dt or default_clock()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(_TS_FMT)

def _parse_ts(ts: str) -> datetime:
    # tolerant reader: accept with-Z and without-Z, plus ISO-ish fallbacks
    s = (ts or "").strip()
    if not s:
        return default_clock()
    try:
        return datetime.strptime(s, _TS_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.strptime(s, _TS_FMT_NOZ).replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # last-ditch fallbacks for ISO-like strings
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            continue
    # If truly malformed, surface the error
    raise ValueError(f"Unsupported timestamp format: {ts!r}")


# ─ MemoryTrace ─────────────────────────────────────────────────────────────────
@dataclass
class MemoryTrace:
    """
    Records each recall occurrence of this context object,
    following the 'neurons that fire together wire together' principle.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: _fmt_ts(default_clock()))
    coactivated_with: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─ ContextObject ───────────────────────────────────────────────────────────────
@dataclass
class ContextObject:
    """
    A universal, schema-versioned context object for chaining, retrieval,
    and self-improvement in an agent pipeline.
    Domain-agnostic: holds segments, stages, or arbitrary artifacts
    (tool code, schemas, prompts, policies, knowledge).
    """

    # ─ Required ─
    domain: str           # "segment" | "stage" | "artifact"
    component: str        # "tool_code" | "schema" | "prompt" | "policy" | "knowledge" | ...
    semantic_label: str   # e.g. "select_tools", "user_prompt", "db_schema"

    # ─ Defaults & Optionals ─
    schema_version: int = field(init=False, default=1)
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: Optional[str] = None
    timestamp: str = field(default_factory=lambda: _fmt_ts(default_clock()))

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
    retrieval_score: Optional[float] = None      # static similarity score

    # Reinforcement-Learning signals
    value_estimate: Optional[float]  = None      # Q-value / critic estimate
    outcome_reward: Optional[float]  = None      # immediate scalar reward
    advantage:      Optional[float]  = None      # reward − baseline
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

    # ─ Internals (not dataclass fields) ─ set in __post_init__
    # _lock: threading.Lock
    # _logger: logging.Logger

    # ─ Helpers ─
    def _ts_seconds(self) -> float:
        """Return timestamp (or last_accessed) in epoch seconds."""
        ts = self.last_accessed or self.timestamp
        return _parse_ts(ts).timestamp()

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
            self.timestamp = _fmt_ts(self.timestamp)

        # Initialize last_accessed
        if not self.last_accessed:
            self.last_accessed = self.timestamp

        # Default memory tier by domain
        tier_map = {"segment": "STM", "stage": "LTM", "artifact": "WM"}
        self.memory_tier = self.memory_tier or tier_map.get(self.domain, "WM")

        # Setup logger
        self._logger = logging.getLogger(__name__)

    def __getstate__(self):
        """Exclude non-picklable items during pickling."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_logger", None)
        return state

    def __setstate__(self, state):
        """Restore state and recreate lock and logger after unpickling."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def __repr__(self):
        return f"<ContextObject {self.domain}/{self.component}/{self.semantic_label}@{self.timestamp}>"

    # ─ Core Helpers ─
    def touch(self):
        """Mark accessed just now."""
        self.last_accessed = _fmt_ts(default_clock())
        self.dirty = True

    def set_expiration(self, ttl_seconds: int):
        """Expire after TTL seconds."""
        exp = default_clock() + timedelta(seconds=ttl_seconds)
        self.expires_at = _fmt_ts(exp)
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
        try:
            fn = component_embedder.get(self.component) if component_embedder else None
            fn = fn or default_embedder
            self.embedding = fn(self.summary)
            self.dirty = True
        except Exception as e:
            self._logger.warning("compute_embedding failed: %s", e)

    def log_context(self, level=logging.INFO):
        """Emit full context JSON to logs."""
        try:
            self._logger.log(level, json.dumps(self.to_dict(), ensure_ascii=False))
        except Exception:
            # Fallback to repr if unserializable
            self._logger.log(level, repr(self))

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
            stats["count"] = int(stats.get("count", 0)) + 1
            stats["last_recalled"] = mt.timestamp

            # simplistic firing_rate
            self.firing_rate = 1.0 / stats["count"] if stats["count"] else None

            # association strengthening
            for other in mt.coactivated_with:
                self.association_strengths[other] = self.association_strengths.get(other, 0.0) + 1.0

            # mark accessed
            self.touch()

    # ─ Serialization ─
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["memory_traces"] = [asdict(mt) for mt in self.memory_traces]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def make_performance(
        cls,
        reward: float,
        stage_ids: List[str],
        metrics: Dict[str, Any] | None = None,
    ) -> "ContextObject":
        """
        Factory for a stage_performance object that carries
        the scalar `reward` and any extra metrics.
        """
        obj = cls(domain="stage",
                  component="stage_performance",
                  semantic_label="stage_performance")
        obj.outcome_reward = reward
        obj.metadata.update(metrics or {})
        obj.references = list(stage_ids or [])
        obj.tags = ["performance"]
        return obj

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
        repo: "ContextRepository" = ContextRepository.instance()  # singleton accessor

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
    def make_success(cls, description: str, refs: Optional[List[str]] = None) -> "ContextObject":
        """Log that an action or plan succeeded."""
        obj = cls(domain="stage", component="success", semantic_label="success")
        obj.summary = description
        obj.references = list(refs or [])
        obj.tags = ["success"]
        return obj

    @classmethod
    def make_failure(cls, description: str, refs: Optional[List[str]] = None) -> "ContextObject":
        """Log that an action or plan failed."""
        obj = cls(domain="stage", component="failure", semantic_label="failure")
        obj.summary = description
        obj.references = list(refs or [])
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
            timestamp=data.get("timestamp", _fmt_ts(default_clock())),
        )
        # Overwrite defaults with saved values
        obj.context_id             = data.get("context_id", obj.context_id)
        obj.segment_ids            = list(data.get("segment_ids", []))
        obj.stage_id               = data.get("stage_id")
        obj.references             = list(data.get("references", []))
        obj.children               = list(data.get("children", []))
        obj.summary                = data.get("summary")
        obj.tags                   = list(data.get("tags", []))
        obj.embedding              = data.get("embedding")
        obj.retrieval_score        = data.get("retrieval_score")
        obj.retrieval_metadata     = dict(data.get("retrieval_metadata", {}))
        obj.provenance             = dict(data.get("provenance", {}))
        obj.graph_node_id          = data.get("graph_node_id")
        obj.memory_tier            = data.get("memory_tier", obj.memory_tier)
        obj.memory_traces          = mts
        obj.association_strengths  = dict(data.get("association_strengths", {}))
        obj.recall_stats           = dict(data.get("recall_stats", {"count": 0, "last_recalled": None}))
        obj.firing_rate            = data.get("firing_rate")
        obj.metadata               = dict(data.get("metadata", {}))
        obj.pinned                 = bool(data.get("pinned", False))
        obj.last_accessed          = data.get("last_accessed", obj.last_accessed)
        obj.expires_at             = data.get("expires_at")
        obj.acl                    = dict(data.get("acl", {}))
        obj.batch_id               = data.get("batch_id")
        obj.dirty                  = bool(data.get("dirty", True))
        # RL fields (previously omitted in some versions)
        obj.value_estimate         = data.get("value_estimate")
        obj.outcome_reward         = data.get("outcome_reward")
        obj.advantage              = data.get("advantage")
        return obj

    @staticmethod
    def from_json(s: str) -> "ContextObject":
        return ContextObject.from_dict(json.loads(s))

    # ─ Factory Methods ─
    @classmethod
    def make_segment(
        cls,
        semantic_label: str,
        content_refs: List[str],
        tags: Optional[List[str]] = None,
        **metadata
    ) -> "ContextObject":
        obj = cls(domain="segment", component="segment", semantic_label=semantic_label)
        obj.segment_ids = list(content_refs or [])
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
        obj.references = list(input_refs or [])
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
        obj.summary = (code[:120] + "…") if len(code) > 120 else code
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
        obj.summary = (schema_def[:120] + "…") if len(schema_def) > 120 else schema_def
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
        obj.summary = (prompt_text[:120] + "…") if len(prompt_text) > 120 else prompt_text
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
        obj.summary = (policy_text[:120] + "…") if len(policy_text) > 120 else policy_text
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
        obj.summary = (content[:120] + "…") if len(content) > 120 else content
        obj.tags = tags or ["knowledge"]
        obj.metadata.update(metadata)
        obj.metadata["content"] = content
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# JSONL maintenance
# ──────────────────────────────────────────────────────────────────────────────
def sanitize_jsonl(path: str) -> None:
    """
    Reads 'path' under shared lock, drops any corrupted JSON lines,
    logs them into 'path.corrupt', and—only if invalid lines are found—
    atomically rewrites with the remaining valid ones.
    """
    if not os.path.exists(path):
        return

    corrupt_path = path + ".corrupt"
    good_lines: List[str] = []
    bad_entries: List[tuple[int, str]] = []

    # 1) Read & classify under shared lock
    with open(path, "r+", encoding="utf-8") as f, _locked(f, exclusive=False):
        for idx, line in enumerate(f, start=1):
            try:
                json.loads(line)
                good_lines.append(line)
            except JSONDecodeError as e:
                logging.warning("sanitize_jsonl: dropping invalid JSON at line %d in %s: %s", idx, path, e)
                bad_entries.append((idx, line.rstrip("\n")))

        # 2) If no bad entries, leave file as-is
        if not bad_entries:
            return

        # 3) Log all bad lines
        with open(corrupt_path, "a", encoding="utf-8") as cf:
            now = datetime.utcnow().isoformat()
            for idx, text in bad_entries:
                cf.write(f"{now} LINE {idx}: {text}\n")

        # 4) Rewrite only when there were bad lines
        f.seek(0)
        f.truncate()
        f.writelines(good_lines)
        f.flush()
        os.fsync(f.fileno())


# ──────────────────────────────────────────────────────────────────────────────
# Repositories
# ──────────────────────────────────────────────────────────────────────────────
class JSONLContextRepository:
    _singleton: "JSONLContextRepository" = None

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
                        logging.warning("JSONLContextRepository.get: parse error on line %d: %s", lineno, e)
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
        """Append a dirty context object to JSONL under exclusive lock."""
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

        kept: List[Dict[str, Any]] = []
        with self._lock:
            with open(self.path, "r+", encoding="utf-8") as f, _locked(f, exclusive=True):
                for lineno, line in enumerate(f, start=1):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logging.warning("JSONLContextRepository.delete: skipping bad line %d", lineno)
                        continue
                    if data.get("context_id") != context_id:
                        kept.append(data)
                f.seek(0)
                f.truncate()
                for entry in kept:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def query(self, filter_fn: Callable[[ContextObject], bool]) -> List[ContextObject]:
        """
        Iterate all contexts, skipping any bad lines; attempt one repair pass if needed.
        """
        results: List[ContextObject] = []
        tried_sanitize = False

        while True:
            with open(self.path, "r", encoding="utf-8") as f, _locked(f, exclusive=False):
                bad_line = False
                for lineno, line in enumerate(f, start=1):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.warning("JSONLContextRepository.query: parse error on line %d: %s", lineno, e)
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
    def __init__(self, db_path: str = "context.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = Lock()
        self._init_schema()

    def _init_schema(self) -> None:
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

    def save(self, ctx: ContextObject) -> None:
        blob = ctx.to_json()
        now  = _fmt_ts(default_clock())
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

    def get(self, cid: str) -> ContextObject:
        cur = self.conn.cursor()
        cur.execute("SELECT json_blob FROM contexts WHERE context_id=?", (cid,))
        row = cur.fetchone()
        if not row:
            raise KeyError(cid)
        return ContextObject.from_json(row[0])

    def delete(self, cid: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM contexts WHERE context_id=?", (cid,))
            self.conn.commit()

    def query(self, filter_fn: Callable[[ContextObject], bool]) -> List[ContextObject]:
        out: List[ContextObject] = []
        for (blob,) in self.conn.execute("SELECT json_blob FROM contexts"):
            obj = ContextObject.from_json(blob)
            if filter_fn(obj):
                out.append(obj)
        return out

    def count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM contexts")
        return cur.fetchone()[0]


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid: JSONL + SQLite (dual-write) with size-based archiving
# ──────────────────────────────────────────────────────────────────────────────
class HybridContextRepository:
    _singleton: "HybridContextRepository" = None

    def __init__(
        self,
        jsonl_path: str = "context.jsonl",
        sqlite_path: str = "context.db",
        archive_max_mb: float = 10.0,   # max JSONL size in megabytes
        *,
        dual_write: bool = True,        # write every object to SQLite immediately
        verbose: bool = True,           # print row deltas & archive actions
    ):
        # ─── ensure our subfolder exists ───────────────────────────────
        base_dir = Path("context_repos")
        base_dir.mkdir(exist_ok=True)

        # ─── force both paths into that folder ─────────────────────────
        jsonl_filename   = Path(jsonl_path).name
        sqlite_filename  = Path(sqlite_path).name
        jsonl_full_path  = str(base_dir / jsonl_filename)
        sqlite_full_path = str(base_dir / sqlite_filename)

        # ─── wire up underlying repositories ────────────────────────────
        self.json_repo  = JSONLContextRepository(jsonl_full_path)
        self.sql_repo   = SQLiteContextRepository(sqlite_full_path)
        self._max_bytes = int(archive_max_mb * 1024 * 1024)
        self._dual_write = bool(dual_write)
        self._verbose = bool(verbose)

        HybridContextRepository._singleton = self  # register singleton

    # ──────────────────────────────────────────────────────────────────
    # Core ops
    # ──────────────────────────────────────────────────────────────────
    def save(self, ctx: ContextObject) -> None:
        # 1) append to JSONL (source of truth for most-recent)
        self.json_repo.save(ctx)

        # 2) mirror into SQLite so DB grows every save
        if self._dual_write:
            before = self._safe_count()
            self.sql_repo.save(ctx)
            after = self._safe_count()
            if self._verbose and after is not None and before is not None:
                print(f"[HybridRepo] SQLite rows: {before} → {after} (+{after - before})")

        # 3) prune JSONL by size if needed (moves oldest non-artifacts to SQLite)
        self._archive_by_size()

    def get(self, cid: str) -> ContextObject:
        try:
            return self.json_repo.get(cid)
        except KeyError:
            return self.sql_repo.get(cid)

    def delete(self, cid: str) -> None:
        # remove from JSONL
        self.json_repo.delete(cid)
        # best-effort remove from SQLite
        try:
            self.sql_repo.delete(cid)
        except KeyError:
            pass

    def query(self, filter_fn: Callable[[ContextObject], bool]) -> List[ContextObject]:
        seen: set[str] = set()
        out: List[ContextObject] = []
        for repo in (self.json_repo, self.sql_repo):
            for ctx in repo.query(filter_fn):
                if ctx.context_id not in seen:
                    seen.add(ctx.context_id)
                    out.append(ctx)
        return out

    # ──────────────────────────────────────────────────────────────────
    # Archiving
    # ──────────────────────────────────────────────────────────────────
    def force_archive(self) -> int:
        """Force a single offload pass regardless of file size. Returns # rows moved."""
        return self._archive_by_size(force=True)

    def _archive_by_size(self, force: bool = False) -> int:
        path = self.json_repo.path
        try:
            size = os.path.getsize(path)
        except OSError:
            if self._verbose:
                logging.info("[HybridRepo] archive: JSONL path missing")
            return 0

        if not force and size <= self._max_bytes:
            return 0

        # Ensure JSONL is clean before archiving
        sanitize_jsonl(path)

        # Load JSONL lines
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            logging.warning("[HybridRepo] archive: cannot read JSONL: %s", e)
            return 0

        archived = 0
        entries: List[tuple[int, datetime, ContextObject]] = []

        for idx, line in enumerate(lines):
            try:
                obj = ContextObject.from_json(line)
            except Exception as e:
                logging.warning("[HybridRepo] archive: skip unparsable line %d: %s", idx, e)
                continue

            # Skip prompt & schema artifacts (keep them in JSONL)
            if obj.domain == "artifact" and obj.component in ("prompt", "schema"):
                continue

            # Parse timestamp; skip bad rows
            try:
                ts = _parse_ts(obj.timestamp)
            except Exception as e:
                logging.warning("[HybridRepo] archive: bad timestamp line %d: %s", idx, e)
                continue

            entries.append((idx, ts, obj))

        if not entries:
            if self._verbose:
                logging.info("[HybridRepo] archive: no eligible entries (jsonl=%d bytes)", size)
            return 0

        # Oldest first
        entries.sort(key=lambda x: x[1])

        to_remove: set[int] = set()
        for idx, _ts, obj in entries:
            try:
                self.sql_repo.save(obj)
                archived += 1
                to_remove.add(idx)
            except Exception as e:
                logging.error("[HybridRepo] archive: SQLite save failed for %s: %s", obj.context_id, e)

            if not force:
                remaining_bytes = self._remaining_bytes_after_removal(lines, to_remove)
                if remaining_bytes <= self._max_bytes:
                    break

        if to_remove:
            # Rewrite JSONL without archived lines
            try:
                with open(path, "w", encoding="utf-8") as f:
                    for i, l in enumerate(lines):
                        if i not in to_remove:
                            f.write(l)
            except OSError as e:
                logging.error("[HybridRepo] archive: failed to rewrite JSONL: %s", e)

        if self._verbose:
            try:
                now = os.path.getsize(path) if os.path.exists(path) else 0
            except OSError:
                now = -1
            logging.info("[HybridRepo] archive: moved %d rows to SQLite; JSONL now %d bytes", archived, now)

        return archived

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────
    def _safe_count(self) -> Optional[int]:
        try:
            return self.sql_repo.count()  # optional convenience method
        except Exception:
            return None

    @staticmethod
    def _remaining_bytes_after_removal(lines: List[str], to_remove: set[int]) -> int:
        # Sum encoded byte lengths of lines not marked for removal
        return sum(len(l.encode("utf-8")) for i, l in enumerate(lines) if i not in to_remove)

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
_HMR_DECAY_SECS   = 60 * 60 * 24   # temporal proximity half-life (seconds)

ContextRepository = HybridContextRepository

# ─ MemoryManager / Service Layer ──────────────────────────────────────────────
class MemoryManager:
    """
    High-level service for associative recall, reinforcement, pruning,
    spreading-activation (“thought chains”) and consolidation.
    """

    _graph: Dict[str, Dict[str, float]] = {}
    _graph_path: str = "context_repos/holo_graph.json"

    def __init__(self, repo: ContextRepository):
        import json, os, threading
        self.repo = repo
        self._graph_lock = threading.Lock()
        # lazy-load persisted graph once
        if not MemoryManager._graph and os.path.exists(self._graph_path):
            try:
                with open(self._graph_path, "r", encoding="utf-8") as f:
                    MemoryManager._graph = json.load(f)
            except Exception:
                MemoryManager._graph = {}

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _half_life_factor(delta_seconds: float, half_life_seconds: float) -> float:
        """Return multiplicative decay factor using true half-life math."""
        if half_life_seconds <= 0:
            return 0.0
        # 0.5 ** (dt / half_life)
        return 0.5 ** (max(delta_seconds, 0.0) / float(half_life_seconds))

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        return datetime.strptime(ts, "%Y%m%dT%H%M%SZ")

    def _scope_filter(self, allowed_user: str | None, allowed_conv: str | None):
        """Restrict repo scans to a user and/or conversation when provided."""
        def _f(c: ContextObject) -> bool:
            uid = (c.metadata or {}).get("user_id")
            cid = (c.metadata or {}).get("conversation_id")
            ok_user = (allowed_user is None) or (uid == allowed_user)
            ok_conv = (allowed_conv is None) or (cid == allowed_conv)
            return ok_user and ok_conv
        return _f

    # ──────────────────────────────────────────────────────────────
    # Graph maintenance
    # ──────────────────────────────────────────────────────────────
    def decay_graph_edges(self, half_life_secs: float = 86_400.0, min_w: float = 1e-6) -> None:
        """Exponential decay on holographic edges; drop tiny weights."""
        import time
        now = time.time()
        # cache last decay time
        if not hasattr(self, "_last_graph_decay_ts"):
            self._last_graph_decay_ts = now
            return
        dt = max(now - self._last_graph_decay_ts, 0.0)
        if dt < 60.0:   # throttle
            return
        factor = self._half_life_factor(dt, half_life_secs)
        with self._graph_lock:
            for u, nbrs in list(self._graph.items()):
                for v, w in list(nbrs.items()):
                    w2 = w * factor
                    if w2 <= min_w:
                        del nbrs[v]
                    else:
                        nbrs[v] = w2
                if not nbrs:
                    del self._graph[u]
        self._last_graph_decay_ts = now
        self._save_graph()

    def start_episode(self, title: str, meta: Dict[str, Any] | None = None) -> ContextObject:
        from context import ContextObject
        epi = ContextObject(
            domain="stage",
            component="episode",
            semantic_label="episode",
        )
        epi.summary = title
        epi.tags = ["episode"]
        epi.metadata.update(meta or {})
        self.repo.save(epi)
        # remember current open episode id
        self._current_episode_id = epi.context_id
        return epi

    def add_to_episode(self, ctx: ContextObject) -> None:
        """Link a ctx into the open episode (if any) with strong edges."""
        epi_id = getattr(self, "_current_episode_id", None)
        if not epi_id:
            return
        self._add_edge(epi_id, ctx.context_id, 1.5)
        self._add_edge(ctx.context_id, epi_id, 1.0)
        self._save_graph()

    def end_episode(self) -> None:
        if hasattr(self, "_current_episode_id"):
            delattr(self, "_current_episode_id")

    def consolidate_stm_to_ltm(self, promote_threshold: int = 3) -> None:
        """Promote frequently-recalled items to LTM tier (and pin them)."""
        for ctx in self.repo.query(lambda c: True):
            cnt = (ctx.recall_stats or {}).get("count", 0)
            if cnt >= promote_threshold and ctx.memory_tier != "LTM":
                ctx.memory_tier = "LTM"
                ctx.pinned = True
                ctx.touch()
                self.repo.save(ctx)

    def _save_graph(self) -> None:
        """Persist holographic graph to disk (best-effort)."""
        import json, os, tempfile, shutil
        with self._graph_lock:
            try:
                os.makedirs(os.path.dirname(self._graph_path), exist_ok=True)
                tmp = self._graph_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._graph, f)
                shutil.move(tmp, self._graph_path)
            except Exception:
                pass

    def vector_search(
        self,
        cue_text: str,
        top_k: int = 10,
        *,
        allowed_user: str | None = None,
        allowed_conv: str | None = None,
        embed_fn: Callable[[str], np.ndarray],
        reembed: bool = False,
        use_hybrid_rank: bool = True,
    ) -> list[ContextObject]:
        """
        Pure vector search (optionally hybridized with recency + assoc).
        - Persists embeddings lazily on each ContextObject.
        - Scopes to user/conv if provided.
        """
        if not cue_text:
            return []

        # Build query vector (unit)
        q = np.asarray(embed_fn(cue_text), dtype=np.float32).reshape(-1)
        q /= (np.linalg.norm(q) + 1e-9)

        # Candidate pool: in-scope only
        filt = self._scope_filter(allowed_user, allowed_conv)
        cands = self.repo.query(filt)

        now = default_clock()

        def _get_unit_embed(c: ContextObject) -> np.ndarray | None:
            # prefer persisted
            if not reembed and c.embedding:
                v = np.asarray(c.embedding, dtype=np.float32).reshape(-1)
            else:
                txt = (c.summary or "").strip()
                if not txt:
                    return None
                v = np.asarray(embed_fn(txt), dtype=np.float32).reshape(-1)
                # persist for future use
                c.embedding = v.tolist()
                c.touch()
                self.repo.save(c)
            n = float(np.linalg.norm(v) + 1e-9)
            return v / n

        rows = []
        for c in cands:
            u = _get_unit_embed(c)
            if u is None:
                continue
            sim = float(np.dot(q, u))  # cosine (unit vectors)
            # Optional hybrid re-rank
            if use_hybrid_rank:
                # Recency term
                last = _parse_ts(c.last_accessed or c.timestamp)
                age = (now - last).total_seconds()
                rec = 1.0 / (1.0 + age)
                # Assoc prior: how connected is this node (degree/strength)
                assoc = sum((c.association_strengths or {}).values())
                # Normalize assoc lightly
                assoc = np.log1p(assoc)
                score = 0.55 * sim + 0.25 * rec + 0.20 * assoc
            else:
                score = sim
            rows.append((c, score, sim))

        # rank
        rows.sort(key=lambda t: t[1], reverse=True)
        pool = rows[: max(top_k * 4, top_k)]

        # MMR (reuse your alpha)
        alpha = 0.75
        selected: list[tuple[ContextObject, float]] = []
        vec_cache: dict[str, np.ndarray] = {}

        def _vec(c: ContextObject) -> np.ndarray | None:
            v = vec_cache.get(c.context_id)
            if v is not None:
                return v
            emb = np.asarray(c.embedding, dtype=np.float32).reshape(-1) if c.embedding else None
            if emb is None or emb.size == 0:
                return None
            u = emb / (np.linalg.norm(emb) + 1e-9)
            vec_cache[c.context_id] = u
            return u

        cands = pool[:]
        while cands and len(selected) < top_k:
            best = None
            best_score = -1e9
            for c, score, sim in cands:
                if not selected:
                    mmr = score
                else:
                    v = _vec(c)
                    if v is None:
                        novelty = 1.0
                    else:
                        sims = []
                        for cs, _s in selected:
                            sv = _vec(cs)
                            if sv is not None:
                                sims.append(float(np.dot(v, sv)))
                        novelty = 1.0 - max(sims) if sims else 1.0
                    mmr = alpha * score + (1.0 - alpha) * novelty
                if mmr > best_score:
                    best_score = mmr
                    best = (c, score)
            selected.append(best)
            cands = [(c, s, sim) for (c, s, sim) in cands if c.context_id != best[0].context_id]

        # finalize
        out: list[ContextObject] = []
        for c, score in selected:
            c.retrieval_score = float(score)
            out.append(c)
        return out


    # ──────────────────────────────────────────────────────────────
    # Lightweight recall (one hop via association_strengths)
    # ──────────────────────────────────────────────────────────────
    def recall(
        self,
        seed_ids: List[str],
        k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> List[ContextObject]:
        weights = weights or {"assoc": 1.0, "recency": 1.0}
        now = default_clock()

        if not seed_ids:
            return []

        # infer scope from the first seed (best-effort)
        owner_user = owner_conv = None
        try:
            first = self.repo.get(seed_ids[0])
            owner_user = (first.metadata or {}).get("user_id")
            owner_conv = (first.metadata or {}).get("conversation_id")
        except Exception:
            pass

        def _in_scope(cid: str) -> bool:
            if owner_user is None and owner_conv is None:
                return True
            try:
                c = self.repo.get(cid)
                return self._scope_filter(owner_user, owner_conv)(c)
            except Exception:
                return False

        # 1) one-hop candidate scoring (scoped)
        scores: Dict[str, float] = {}
        for sid in seed_ids:
            try:
                seed = self.repo.get(sid)
            except KeyError:
                continue
            for oid, strength in (seed.association_strengths or {}).items():
                if not _in_scope(oid):
                    continue
                try:
                    other = self.repo.get(oid)
                except KeyError:
                    continue
                base = strength * weights["assoc"]
                last = self._parse_ts(other.last_accessed)
                age  = (now - last).total_seconds()
                base += weights["recency"] / (1.0 + age)
                scores[oid] = scores.get(oid, 0.0) + base

        # 2) top-k, stamp, record
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results: List[ContextObject] = []
        for cid, score in top:
            try:
                ctx = self.repo.get(cid)
            except KeyError:
                continue
            ctx.retrieval_score = score
            ctx.retrieval_metadata = {"seed_ids": seed_ids}
            ctx.record_recall(stage_id="recall", coactivated_with=seed_ids, retrieval_score=score)
            self.repo.save(ctx)
            results.append(ctx)
        return results

    # ──────────────────────────────────────────────────────────────
    # Multi-hop activation over association_strengths
    # ──────────────────────────────────────────────────────────────
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
            for cid2, inc in new_act.items():
                activation[cid2] = activation.get(cid2, 0.0) + inc

        # Recency bonus
        for cid in list(activation.keys()):
            try:
                ctx = self.repo.get(cid)
                last = self._parse_ts(ctx.last_accessed)
                age = (now - last).total_seconds()
                activation[cid] += recency_weight / (1.0 + age)
            except Exception:
                continue

        return activation

    # ──────────────────────────────────────────────────────────────
    # Decay association strengths in objects + promotion to LTM
    # ──────────────────────────────────────────────────────────────
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

            last_ts = self._parse_ts(ctx.last_accessed)
            delta = (now - last_ts).total_seconds()

            new_strengths: Dict[str, float] = {}
            for oid, strength in ctx.association_strengths.items():
                try:
                    self.repo.get(oid)
                except KeyError:
                    continue
                decayed = strength * self._half_life_factor(delta, half_life)
                if decayed > 1e-6:
                    new_strengths[oid] = decayed

            should_promote = ctx.recall_stats.get("count", 0) >= promote_threshold
            if new_strengths != ctx.association_strengths or (should_promote and ctx.memory_tier != "LTM"):
                ctx.association_strengths = new_strengths
                if should_promote:
                    ctx.memory_tier = "LTM"
                ctx.touch()
                self.repo.save(ctx)

    # ──────────────────────────────────────────────────────────────
    # Reinforcement (clique strengthening among coactivated)
    # ──────────────────────────────────────────────────────────────
    def reinforce(self, context_id: str, coactivated: List[str]) -> None:
        """Strengthen symmetric edges among all coactivated items + context_id."""
        try:
            base = self.repo.get(context_id)
        except KeyError:
            return

        valid = []
        for rid in coactivated:
            try:
                self.repo.get(rid)
                valid.append(rid)
            except KeyError:
                continue
        if not valid:
            return

        all_ids = [context_id] + valid
        # clique reinforcement
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                a, b = all_ids[i], all_ids[j]
                self._add_edge(a, b, 0.5)
                self._add_edge(b, a, 0.5)

        base.record_recall(stage_id="reinforce", coactivated_with=valid)
        self.repo.save(base)
        self._save_graph()

    # ──────────────────────────────────────────────────────────────
    # Prune stale, unpinned contexts
    # ──────────────────────────────────────────────────────────────
    def prune(self, ttl_hours: int) -> None:
        cutoff = default_clock() - timedelta(hours=ttl_hours)
        def stale(c: ContextObject) -> bool:
            la = self._parse_ts(c.last_accessed)
            return la < cutoff and not c.pinned
        for ctx in self.repo.query(stale):
            self.repo.delete(ctx.context_id)

    # ──────────────────────────────────────────────────────────────
    # Edge ops (thread-safe)
    # ──────────────────────────────────────────────────────────────
    def _add_edge(self, src: str, dst: str, w: float) -> None:
        if src == dst or w == 0.0:
            return
        with self._graph_lock:
            self._graph.setdefault(src, {})
            self._graph[src][dst] = self._graph[src].get(dst, 0.0) + w

    # ------------- 1️⃣  register_relationships ----------------------
    def register_relationships(
        self,
        ctx: ContextObject,
        embed_fn: Callable[[str], np.ndarray],
    ) -> None:
        """
        Call once after saving a new/updated ContextObject.
        • Skips re-registering relationships for the same ctx.
        • Uses an in-memory cache for embeddings.
        • Limits similarity scans to the last N items (scoped).
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
        MAX_TAG_NEIGHBORS = 200
        for tag in ctx.tags:
            tag_node = f"tag::{tag}"
            self._add_edge(cid, tag_node, _HMR_TAG_W)
            self._add_edge(tag_node, cid, _HMR_TAG_W)
            # trim oversized tag neighborhoods
            with self._graph_lock:
                nbrs = self._graph.get(tag_node, {})
                if len(nbrs) > MAX_TAG_NEIGHBORS:
                    keep = dict(sorted(nbrs.items(), key=lambda kv: kv[1], reverse=True)[:MAX_TAG_NEIGHBORS])
                    self._graph[tag_node] = keep

        # ---------- semantic similarity ----------
        # initialize embedding cache if missing
        if not hasattr(self, "_embed_cache"):
            # cache stores (unit_vector, original_norm)
            self._embed_cache: Dict[str, tuple[np.ndarray, float]] = {}

        def _unit(text: str) -> np.ndarray:
            if text in self._embed_cache:
                u, _ = self._embed_cache[text]
                return u
            raw = embed_fn(text)
            v = np.asarray(raw, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(v) + 1e-9)
            u = v / n
            self._embed_cache[text] = (u, n)
            return u

        try:
            base_text = (ctx.summary or "").strip()
            if base_text:
                v1 = _unit(base_text)
                # restrict recents to same user/conversation when available
                allowed_user = (ctx.metadata or {}).get("user_id")
                allowed_conv = (ctx.metadata or {}).get("conversation_id")
                # Stable recency ordering (by last_accessed then timestamp)
                recents_all = self.repo.query(self._scope_filter(allowed_user, allowed_conv))
                recents_sorted = sorted(
                    recents_all,
                    key=lambda c: (c.last_accessed or c.timestamp),
                )[-200:]
                for other in recents_sorted:
                    if other.context_id == cid:
                        continue
                    txt = (other.summary or "").strip()
                    if not txt:
                        continue
                    v2 = _unit(txt)
                    sim = float(np.dot(v1, v2))  # already unit vectors
                    if sim >= _HMR_SIM_THRESH:
                        w = _HMR_SIM_W * sim
                        self._add_edge(cid, other.context_id, w)
                        self._add_edge(other.context_id, cid, w)
        except Exception:
            pass

        # ---------- temporal proximity (<10 min) ----------
        now_sec = ctx._ts_seconds()
        window = 600  # seconds
        allowed_user = (ctx.metadata or {}).get("user_id")
        allowed_conv = (ctx.metadata or {}).get("conversation_id")
        candidates = [
            c for c in self.repo.query(self._scope_filter(allowed_user, allowed_conv))
            if abs(now_sec - c._ts_seconds()) <= window and c.context_id != cid
        ]
        for other in candidates:
            age = abs(now_sec - other._ts_seconds())
            w = self._half_life_factor(age, _HMR_DECAY_SECS)
            self._add_edge(cid, other.context_id, w)
            self._add_edge(other.context_id, cid, w)

        # Persist graph after all mutations
        self._save_graph()

    # ------------- 2️⃣  holographic_recall --------------------------
    def holographic_recall(
        self,
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
        self.decay_graph_edges()
        self.consolidate_stm_to_ltm()

        cue_ids = cue_ids or []
        activation: Dict[str, float] = collections.Counter({cid: 1.0 for cid in cue_ids})

        # infer scope from the first cue (if available)
        owner_user = None
        owner_conv = None
        if cue_ids:
            try:
                first = self.repo.get(cue_ids[0])
                owner_user = (first.metadata or {}).get("user_id")
                owner_conv = (first.metadata or {}).get("conversation_id")
            except Exception:
                pass

        # text cue → similarity edges once, scoped to user/conv
        if cue_text and embed_fn:
            def _unit(text: str) -> np.ndarray:
                v = np.asarray(embed_fn(text), dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(v) + 1e-9)
                return v / n
            qv = _unit(cue_text)
            for c in self.repo.query(self._scope_filter(owner_user, owner_conv)):
                if not c.summary:
                    continue
                vv = _unit(c.summary)
                sim = float(np.dot(qv, vv))  # unit vectors
                if sim >= _HMR_SIM_THRESH:
                    activation[c.context_id] += _HMR_SIM_W * sim

        # hop propagation (optionally scoped to user/conv)
        def _in_scope(cid: str) -> bool:
            if owner_user is None and owner_conv is None:
                return True
            try:
                c = self.repo.get(cid)
                return self._scope_filter(owner_user, owner_conv)(c)
            except Exception:
                return False

        # Snapshot graph under lock to avoid concurrent modification during traversal
        with self._graph_lock:
            graph_snapshot = {k: dict(v) for k, v in self._graph.items()}

        frontier = dict(activation)
        for _ in range(hops):
            new_frontier = collections.Counter()
            for nid, act in frontier.items():
                for nbr, w in graph_snapshot.get(nid, {}).items():
                    if not _in_scope(nbr):
                        continue
                    new_frontier[nbr] += act * w
            for k, v in new_frontier.items():
                activation[k] = activation.get(k, 0.0) + v
            frontier = new_frontier

        # scope filter BEFORE ranking/MMR
        if owner_user is not None or owner_conv is not None:
            activation = {k: v for k, v in activation.items() if _in_scope(k)}

        # candidate pool for MMR (oversample for diversity)
        cands = sorted(activation.items(), key=lambda kv: kv[1], reverse=True)[: max(top_n * 4, top_n)]
        selected: list[tuple[str, float]] = []
        alpha = 0.75  # relevance vs novelty

        # unit-vector helper for MMR; safe when embed_fn is None
        def _unit_from_cid(cid: str):
            if not embed_fn:
                return None
            try:
                obj = self.repo.get(cid)
                txt = (obj.summary or "").strip()
                if not txt:
                    return None
                v = np.asarray(embed_fn(txt), dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(v) + 1e-9)
                return v / n
            except Exception:
                return None

        vecs = {cid: _unit_from_cid(cid) for cid, _ in cands}

        # MMR selection (single pass)
        while cands and len(selected) < top_n:
            best = None
            best_score = -1e9
            for cid, rel in cands:
                v = vecs.get(cid)
                if not selected or v is None:
                    novelty = 1.0
                else:
                    sims = []
                    for scid, _ in selected:
                        sv = vecs.get(scid)
                        if sv is None or v is None:
                            continue
                        sims.append(float(np.dot(v, sv)))  # unit vectors
                    max_sim = max(sims) if sims else 0.0
                    novelty = 1.0 - max(0.0, max_sim)
                mmr = alpha * float(rel) + (1.0 - alpha) * novelty
                if mmr > best_score:
                    best_score = mmr
                    best = (cid, rel)
            selected.append(best)
            cands = [(cid, r) for (cid, r) in cands if cid != best[0]]

        # materialize selected set
        out = []
        for cid, score in selected:
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
