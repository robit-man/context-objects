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
# JSONL maintenance (robust to non-UTF-8 and truncated lines)
# ──────────────────────────────────────────────────────────────────────────────
def sanitize_jsonl(path: str) -> None:
    """
    Robust JSONL sanitizer:
      • Reads file in **binary** to avoid global UTF-8 decode failures.
      • Decodes each line as UTF-8 (strict). Undecodable lines are logged & dropped.
      • Drops lines that decode but are not valid JSON.
      • Writes back only the valid lines (UTF-8), atomically.
      • Appends all dropped lines (with reasons) to '<path>.corrupt'.

    No-op if file does not exist.
    """
    if not os.path.exists(path):
        return

    corrupt_path = path + ".corrupt"
    good_text_lines: List[str] = []
    bad_records: List[str] = []

    # Read & classify under shared lock, but in BINARY mode to survive bad bytes
    with open(path, "rb") as f, _locked(f, exclusive=False):
        lineno = 0
        while True:
            raw = f.readline()
            if not raw:
                break
            lineno += 1

            # Trim a single trailing newline/carriage-return for decode/parse; we will re-add '\n' on write
            raw_stripped = raw.rstrip(b"\r\n")

            # 1) UTF-8 decode (strict). If it fails, log & skip.
            try:
                text = raw_stripped.decode("utf-8")
            except UnicodeDecodeError as e:
                # keep a short hex preview to avoid dumping megabytes
                preview = raw[:64].hex()
                bad_records.append(
                    f"{datetime.utcnow().isoformat()}Z LINE {lineno}: <NON-UTF8> "
                    f"{e}; first64hex={preview}"
                )
                continue

            # 2) Remove BOM if present on first line
            if lineno == 1 and text.startswith("\ufeff"):
                text = text.lstrip("\ufeff")

            # 3) Validate JSON
            try:
                json.loads(text)
                good_text_lines.append(text + "\n")  # normalize newline
            except JSONDecodeError as e:
                # Include a compact preview of the offending text
                preview = (text[:120] + "…") if len(text) > 120 else text
                bad_records.append(
                    f"{datetime.utcnow().isoformat()}Z LINE {lineno}: <BAD-JSON> {e}; preview={preview!r}"
                )
                continue

    # If nothing was bad, do nothing
    if not bad_records:
        return

    # Append all bad records to .corrupt (text, UTF-8)
    try:
        with open(corrupt_path, "a", encoding="utf-8") as cf:
            for rec in bad_records:
                cf.write(rec + "\n")
    except Exception:
        # best-effort only
        pass

    # Atomically rewrite the JSONL with only the valid lines
    dirpath = os.path.dirname(path) or "."
    tmp_path = os.path.join(dirpath, f".{os.path.basename(path)}.tmp")

    # Use exclusive lock during rewrite to avoid readers seeing mid-write content
    with open(path, "rb+") as f, _locked(f, exclusive=True):
        try:
            with open(tmp_path, "wb") as tf:
                for line in good_text_lines:
                    tf.write(line.encode("utf-8"))
                tf.flush()
                os.fsync(tf.fileno())
            # Move over original
            os.replace(tmp_path, path)
            # fsync directory for durability on some filesystems
            try:
                dir_fd = os.open(dirpath, os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass
        finally:
            # If something failed mid-way, clean up temp file
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Repositories  (FAST: async batched SQLite + hysteresis bulk archiver)
# ──────────────────────────────────────────────────────────────────────────────
import queue
import time

class JSONLContextRepository:
    _singleton: "JSONLContextRepository" = None

    def __init__(self, path: str):
        sanitize_jsonl(path)
        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        # ensure file exists
        open(self.path, "a").close()
        JSONLContextRepository._singleton = self

    def get(self, context_id: str) -> ContextObject:
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
                            break
                        else:
                            continue
                    if data.get("context_id") == context_id:
                        return ContextObject.from_dict(data)
                else:
                    break
            if tried_sanitize:
                break
        raise KeyError(f"Context {context_id} not found")

    def save(self, ctx: ContextObject) -> None:
        """Append a dirty context object to JSONL. Line-buffered, no fsync by default."""
        if not ctx.dirty:
            return
        with self._lock:
            # line buffered IO; avoid fsync on every write unless explicitly requested
            with open(self.path, "a", encoding="utf-8", buffering=1) as f, _locked(f, exclusive=True):
                f.write(ctx.to_json() + "\n")
                if os.getenv("CTX_STRICT_FSYNC", "0").lower() in ("1", "true", "yes"):
                    f.flush()
                    os.fsync(f.fileno())
            ctx.dirty = False

    def delete(self, context_id: str) -> None:
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
                if os.getenv("CTX_STRICT_FSYNC", "0").lower() in ("1", "true", "yes"):
                    f.flush()
                    os.fsync(f.fileno())

    def query(self, filter_fn: Callable[[ContextObject], bool]) -> List[ContextObject]:
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
                            break
                        else:
                            continue
                    ctx = ContextObject.from_dict(data)
                    if filter_fn(ctx):
                        results.append(ctx)
                if bad_line:
                    continue
                break
        return results

    @classmethod
    def instance(cls) -> "JSONLContextRepository":
        if cls._singleton is None:
            raise RuntimeError("ContextRepository not initialised")
        return cls._singleton


class SQLiteContextRepository:
    """
    Single-writer async pipeline to SQLite using WAL, generous busy_timeout,
    batched commits, and robust retry against 'database is locked'.
    """
    def __init__(self, db_path: str = "context.db",
                 *, async_writes: bool = True,
                 busy_timeout_ms: int | None = None,
                 ipc_lock: bool | None = None):
        import sqlite3, os, threading, time

        # Connection with long timeout (complements PRAGMA busy_timeout)
        connect_timeout_s = float(os.getenv("CTX_SQL_CONNECT_TIMEOUT_S", "60"))
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=connect_timeout_s,
            isolation_level=None,  # autocommit; we'll manage BEGIN/COMMIT
        )

        # WAL + performance pragmas
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms or int(os.getenv('CTX_SQL_BUSY_TIMEOUT_MS','60000')))};")
        self.conn.execute("PRAGMA wal_autocheckpoint=2000;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA mmap_size=268435456;")
        self.conn.execute("PRAGMA cache_size=-40000;")
        self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL;")
        self.conn.execute("PRAGMA max_page_count=1048576;")
        self.conn.execute("PRAGMA journal_size_limit=52428800;")
        self._init_schema()
        self._lock = Lock()

        # Optional inter-process write lock (POSIX only). Toggle via CTX_SQL_IPC_LOCK=1
        env_ipc = os.getenv("CTX_SQL_IPC_LOCK")
        self._ipc_lock_enabled = (ipc_lock if ipc_lock is not None else (env_ipc and env_ipc.lower() in ("1","true","yes")))
        self._lockfile_fd = None
        self._lockfile_path = f"{db_path}.wlock"
        if self._ipc_lock_enabled and os.name != "nt":
            try:
                self._lockfile_fd = os.open(self._lockfile_path, os.O_CREAT | os.O_RDWR, 0o644)
            except Exception:
                self._ipc_lock_enabled = False  # graceful disable if fs doesn't allow

        # Async writer
        import queue
        self._async = bool(async_writes)
        self._q: "queue.Queue[ContextObject | list[ContextObject] | None]" = queue.Queue(maxsize=10000)
        self._writer_thread = None
        self._last_maint = time.monotonic()
        self._maint_interval_s = float(os.getenv("CTX_SQL_MAINT_INTERVAL_S", "30"))
        self._checkpoint_mode = os.getenv("CTX_SQL_CHECKPOINT_MODE", "PASSIVE").upper()  # PASSIVE|NONE
        if self._async:
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="SQLiteWriter")
            self._writer_thread.start()

    # --------------- schema ----------------
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

    # --------------- public helpers ----------------
    def save_sync(self, ctx: "ContextObject") -> None:
        self._save_many_tx([ctx])

    def flush(self, timeout: float = 2.0) -> None:
        if not self._async:
            return
        import time
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout and not self._q.empty():
            time.sleep(0.01)

    # --------------- writer loop ----------------
    def _writer_loop(self):
        import time, queue
        BATCH_MAX = int(os.getenv("CTX_SQL_BATCH_MAX", "500"))
        COMMIT_MS = int(os.getenv("CTX_SQL_COMMIT_MS", "300"))
        while True:
            batch: list["ContextObject"] = []
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                item = None

            # lightweight maintenance (never TRUNCATE; that causes global locks)
            if item is None:
                now = time.monotonic()
                if now - self._last_maint > self._maint_interval_s:
                    try:
                        if self._checkpoint_mode != "NONE":
                            # PASSIVE won't fight with other writers
                            self.conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                        # keep vacuum optional (off by default)
                        if os.getenv("CTX_SQL_ENABLE_VACUUM", "0").lower() in ("1","true","yes"):
                            self.conn.execute("PRAGMA incremental_vacuum(2000);")
                        self.conn.commit()
                    except Exception:
                        pass
                    self._last_maint = now
                continue

            # coalesce into a batch
            if isinstance(item, list):
                batch.extend(item)
            else:
                batch.append(item)

            t_start = time.monotonic()
            while len(batch) < BATCH_MAX and (time.monotonic() - t_start) * 1000 < COMMIT_MS:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    break
                if isinstance(nxt, list):
                    batch.extend(nxt)
                else:
                    batch.append(nxt)

            if batch:
                self._save_many_tx(batch)

    # --------------- txn with retry ----------------
    def _ipc_write_lock(self):
        """Context manager for optional cross-process lock (POSIX only)."""
        import contextlib, os
        @contextlib.contextmanager
        def _noop():
            yield
        if not self._ipc_lock_enabled or os.name == "nt" or self._lockfile_fd is None:
            return _noop()
        import fcntl  # POSIX
        @contextlib.contextmanager
        def _flock():
            try:
                fcntl.flock(self._lockfile_fd, fcntl.LOCK_EX)
                yield
            finally:
                try:
                    fcntl.flock(self._lockfile_fd, fcntl.LOCK_UN)
                except Exception:
                    pass
        return _flock()

    def _save_many_tx(self, ctxs: list["ContextObject"]) -> None:
        import time, random, sqlite3, logging
        if not ctxs:
            return

        # Prepare rows first, outside the critical section
        rows = []
        now = _fmt_ts(default_clock())
        for c in ctxs:
            rows.append((c.context_id, c.timestamp, now, c.to_json()))
            c.dirty = False

        max_retries = int(os.getenv("CTX_SQL_MAX_RETRIES", "12"))
        base_sleep = float(os.getenv("CTX_SQL_RETRY_BASE_MS", "20")) / 1000.0
        max_sleep  = float(os.getenv("CTX_SQL_RETRY_MAX_MS", "1500")) / 1000.0

        attempt = 0
        while True:
            try:
                with self._lock:
                    with self._ipc_write_lock():
                        cur = self.conn.cursor()
                        # Acquire write lock early
                        cur.execute("BEGIN IMMEDIATE;")
                        cur.executemany(
                            """
                            INSERT INTO contexts(context_id,timestamp,last_accessed,json_blob)
                            VALUES(?,?,?,?)
                            ON CONFLICT(context_id) DO UPDATE SET
                              json_blob     = excluded.json_blob,
                              last_accessed = excluded.last_accessed
                            """,
                            rows
                        )
                        cur.execute("COMMIT;")
                # success
                break
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                # Roll back the txn if opened
                try:
                    self.conn.execute("ROLLBACK;")
                except Exception:
                    pass
                if "locked" in msg or "busy" in msg:
                    if attempt < max_retries:
                        # exponential backoff with jitter
                        sleep_s = min(max_sleep, base_sleep * (1.7 ** attempt)) + random.random() * 0.05
                        time.sleep(sleep_s)
                        attempt += 1
                        continue
                    else:
                        logging.warning(
                            "SQLite save retried %d× due to 'database is locked/busy'; giving up on batch of %d rows",
                            attempt, len(rows)
                        )
                        break
                else:
                    logging.error("SQLite batch save failed: %s", e)
                    break
            except Exception as e:
                try:
                    self.conn.execute("ROLLBACK;")
                except Exception:
                    pass
                logging.error("SQLite batch save failed: %s", e)
                break

        # Optional: print throttled row count (quiet)
        try:
            nowt = time.monotonic()
            if nowt - getattr(self, "_log_t0", 0.0) > 2.0:
                self._log_t0 = nowt
                cur = self.conn.cursor()
                cur.execute("SELECT COUNT(*) FROM contexts")
                rc = cur.fetchone()[0]
                if abs(rc - getattr(self, "_last_row_count_print", 0)) >= 50:
                    # keep as print to match your previous behavior but much less often
                    print(f"[HybridRepo] SQLite rows ≈ {rc}")
                    self._last_row_count_print = rc
        except Exception:
            pass

    # --------------- public API ----------------
    def save(self, ctx: "ContextObject") -> None:
        if self._async:
            try:
                self._q.put_nowait(ctx)
            except Exception:
                # queue full → fallback to sync (best effort)
                self._save_many_tx([ctx])
        else:
            self._save_many_tx([ctx])
        ctx.dirty = False

    def enqueue_many(self, ctxs: list["ContextObject"]) -> None:
        if not ctxs:
            return
        if self._async:
            try:
                self._q.put_nowait(ctxs)
            except Exception:
                self._save_many_tx(ctxs)
        else:
            self._save_many_tx(ctxs)

    def get(self, cid: str) -> "ContextObject":
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

    def query(self, filter_fn: Callable[["ContextObject"], bool]) -> List["ContextObject"]:
        out: List["ContextObject"] = []
        for (blob,) in self.conn.execute("SELECT json_blob FROM contexts"):
            obj = ContextObject.from_json(blob)
            if filter_fn(obj):
                out.append(obj)
        return out

    def count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM contexts")
        return cur.fetchone()[0]



class HybridContextRepository:
    _singleton: "HybridContextRepository" = None

    def __init__(
        self,
        jsonl_path: str = "context.jsonl",
        sqlite_path: str = "context.db",
        archive_max_mb: float = 10.0,   # JSONL size cap before archiving
        *,
        dual_write: bool = True,        # mirror each save directly to SQLite
        verbose: bool = True,
    ):
        base_dir = Path("context_repos")
        base_dir.mkdir(exist_ok=True)
        jsonl_full_path  = str(base_dir / Path(jsonl_path).name)
        sqlite_full_path = str(base_dir / Path(sqlite_path).name)

        # env overrides
        env_dual = os.getenv("CTX_DUAL_WRITE")
        if env_dual is not None:
            dual_write = env_dual.lower() not in ("0", "false", "no")
        archive_max_mb = float(os.getenv("CTX_JSONL_MAX_MB", archive_max_mb))

        self.json_repo  = JSONLContextRepository(jsonl_full_path)
        self.sql_repo   = SQLiteContextRepository(sqlite_full_path, async_writes=True)
        self._max_bytes = int(archive_max_mb * 1024 * 1024)
        self._dual_write = bool(dual_write)
        self._verbose = bool(verbose)

        # archiver knobs (hysteresis)
        self._hi = float(os.getenv("CTX_ARCHIVE_HI", "1.15"))   # trigger when > 115% of cap
        self._lo = float(os.getenv("CTX_ARCHIVE_LO", "0.70"))   # reduce to 70% of cap
        self._arch_evt = threading.Event()
        self._arch_lock = threading.Lock()
        self._arch_thread = threading.Thread(target=self._archiver_loop, daemon=True, name="JSONLArchiver")
        self._arch_thread.start()

        # log throttling
        self._last_count_log_ts = 0.0
        self._last_count = None

        HybridContextRepository._singleton = self

    # ──────────────────────────────────────────────────────────────────
    # Core ops
    # ──────────────────────────────────────────────────────────────────
    def save(self, ctx: ContextObject) -> None:
        # JSONL append (cheap)
        self.json_repo.save(ctx)

        # optional mirror to SQLite
        if self._dual_write:
            self.sql_repo.save(ctx)

        # schedule archiving only when beyond HI watermark (avoid per-save work)
        try:
            size = os.path.getsize(self.json_repo.path)
            if size > int(self._max_bytes * self._hi):
                self._arch_evt.set()
        except OSError:
            pass

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
    # Archiver (bulk offload with hysteresis)
    # ──────────────────────────────────────────────────────────────────
    def _archiver_loop(self):
        while True:
            self._arch_evt.wait()
            self._arch_evt.clear()
            # only one archiver pass at a time
            if not self._arch_lock.acquire(blocking=False):
                continue
            try:
                # keep offloading until we are under LO watermark
                while True:
                    moved = self._archive_bulk_pass()
                    if moved <= 0:
                        break
                    try:
                        now_size = os.path.getsize(self.json_repo.path)
                    except OSError:
                        break
                    if now_size <= int(self._max_bytes * self._lo):
                        break
            finally:
                self._arch_lock.release()

    def _archive_bulk_pass(self) -> int:
        path = self.json_repo.path
        try:
            size = os.path.getsize(path)
        except OSError:
            return 0

        if size <= int(self._max_bytes * self._hi):
            return 0

        sanitize_jsonl(path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            logging.warning("[HybridRepo] archiver: cannot read JSONL: %s", e)
            return 0

        entries: list[tuple[int, datetime, ContextObject]] = []
        for idx, line in enumerate(lines):
            try:
                obj = ContextObject.from_json(line)
            except Exception:
                continue
            # keep prompt/schema artifacts hot in JSONL
            if obj.domain == "artifact" and obj.component in ("prompt", "schema"):
                continue
            try:
                ts = _parse_ts(obj.timestamp)
            except Exception:
                continue
            entries.append((idx, ts, obj))

        if not entries:
            return 0

        # Oldest first, move a big chunk
        entries.sort(key=lambda x: x[1])
        target_bytes = int(self._max_bytes * self._lo)
        to_remove: set[int] = set()
        moved_objs: list[ContextObject] = []
        moved_bytes = 0

        # Move at least 10% of current lines to avoid thrash
        min_move = max(100, int(0.10 * len(entries)))
        i = 0
        remaining_bytes = sum(len(l.encode("utf-8")) for l in lines)

        while i < len(entries) and (remaining_bytes - moved_bytes) > target_bytes or len(to_remove) < min_move:
            idx, _ts, obj = entries[i]
            to_remove.add(idx)
            moved_objs.append(obj)
            moved_bytes += len(lines[idx].encode("utf-8"))
            i += 1

        # Ship to SQLite (if dual_write==False this is the first time; else it's harmless upsert)
        try:
            self.sql_repo.enqueue_many(moved_objs)
        except Exception as e:
            logging.error("[HybridRepo] archiver: enqueue_many failed: %s", e)

        # Rewrite JSONL without those lines (atomic in-place)
        try:
            with self.json_repo._lock:
                with open(path, "r+", encoding="utf-8") as f, _locked(f, exclusive=True):
                    f.seek(0)
                    f.truncate()
                    for j, l in enumerate(lines):
                        if j not in to_remove:
                            f.write(l)
                    if os.getenv("CTX_STRICT_FSYNC", "0").lower() in ("1", "true", "yes"):
                        f.flush()
                        os.fsync(f.fileno())
        except OSError as e:
            logging.error("[HybridRepo] archiver: rewrite failed: %s", e)
            return 0

        moved = len(to_remove)
        if self._verbose:
            try:
                now = os.path.getsize(path)
            except OSError:
                now = -1
            logging.info("[HybridRepo] archive: moved %d rows to SQLite; JSONL now %d bytes", moved, now)
        return moved

    # utilities
    def force_archive(self) -> int:
        """Force an immediate bulk pass (use sparingly)."""
        return self._archive_bulk_pass()

    def _safe_count(self) -> Optional[int]:
        try:
            return self.sql_repo.count()
        except Exception:
            return None

    @staticmethod
    def _remaining_bytes_after_removal(lines: List[str], to_remove: set[int]) -> int:
        return sum(len(l.encode("utf-8")) for i, l in enumerate(lines) if i not in to_remove)

    @classmethod
    def instance(cls) -> "ContextRepository":
        if cls._singleton is None:
            raise RuntimeError("ContextRepository has not been initialised yet")
        return cls._singleton



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
        # 1) append to JSONL
        self.json_repo.save(ctx)

        # 2) mirror to SQLite (async)
        if self._dual_write:
            self.sql_repo.save(ctx)

        # 🔇 Remove the per-save COUNT(*) print — it races with async writes and spams logs.
        # If you want visibility, throttle it and don't expect a delta after async enqueue.
        if os.getenv("CTX_VERBOSE_ROWCOUNT", "0") in ("1","true","yes"):
            now = time.monotonic()
            if now - getattr(self, "_last_count_log_ts", 0.0) > 10.0:  # once every 10s max
                setattr(self, "_last_count_log_ts", now)
                rc = self._safe_count()
                if rc is not None:
                    logging.info("[HybridRepo] SQLite rows ≈ %d", rc)

        # 3) archive JSONL by size
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

import os
import io
import time
import math
import sqlite3
import threading
import collections
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np

# ─ MemoryManager / Service Layer ──────────────────────────────────────────────
class MemoryManager:
    """
    High-level service for associative recall, reinforcement, pruning,
    spreading-activation (“thought chains”) and consolidation.

    This implementation persists:
      • Embeddings       → table embeddings(context_id, dim, vec BLOB)
      • Graph edges      → table edges(src, dst, w, updated_at)
      • Light node cache → table nodes(context_id, user_id, conversation_id,
                                       summary, last_accessed, timestamp)

    Notes:
      • Uses WAL + useful PRAGMAs for speed/safety.
      • Vector search computes dot products in NumPy over a capped candidate set.
      • Graph decay executes in-SQL (update + delete).
    """

    # ── legacy attrs (kept for compatibility; no longer used) ─────────────
    _graph: Dict[str, Dict[str, float]] = {}
    _graph_path: str = "context_repos/holo_graph.json"

    def __init__(self, repo: ContextRepository):
        self.repo = repo
        self._graph_lock = threading.Lock()   # kept for API compatibility
        self._db_lock = threading.Lock()
        self._registered_ctxs: set[str] = set()
        self._embed_cache: Dict[str, tuple[np.ndarray, float]] = {}

        # choose DB path close to the repo
        base_dir = "context_repos"
        try:
            # HybridContextRepository exposes sqlite_path/jsonl_path
            if getattr(repo, "sqlite_path", None):
                base_dir = os.path.dirname(os.path.abspath(repo.sqlite_path))
            elif getattr(repo, "jsonl_path", None):
                base_dir = os.path.dirname(os.path.abspath(repo.jsonl_path))
        except Exception:
            pass

        os.makedirs(base_dir, exist_ok=True)
        # one holo DB per repo (best-effort name)
        db_name = "holo_graph"
        try:
            # bake in a short repo-id for isolation
            rid = os.path.basename(getattr(repo, "sqlite_path", "global")).split(".")[0]
            if rid:
                db_name = f"holo_{rid}"
        except Exception:
            pass
        self._db_path = os.path.join(base_dir, f"{db_name}.db")

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=5.0)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA mmap_size=268435456;")  # 256MB
        self._ensure_schema()

    # ──────────────────────────────────────────────────────────────
    # DB schema & helpers
    # ──────────────────────────────────────────────────────────────
    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    context_id TEXT PRIMARY KEY,
                    dim        INTEGER NOT NULL,
                    vec        BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS edges (
                    src        TEXT NOT NULL,
                    dst        TEXT NOT NULL,
                    w          REAL NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (src, dst)
                );
                CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
                CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);

                CREATE TABLE IF NOT EXISTS nodes (
                    context_id      TEXT PRIMARY KEY,
                    user_id         TEXT,
                    conversation_id TEXT,
                    summary         TEXT,
                    last_accessed   TEXT,
                    timestamp       TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_conv_time
                    ON nodes(conversation_id, last_accessed);
                CREATE INDEX IF NOT EXISTS idx_nodes_user_time
                    ON nodes(user_id, last_accessed);
                """
            )

    @staticmethod
    def _half_life_factor(delta_seconds: float, half_life_seconds: float) -> float:
        if half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (max(delta_seconds, 0.0) / float(half_life_seconds))

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        # Stored format: "%Y%m%dT%H%M%SZ"
        return datetime.strptime(ts, "%Y%m%dT%H%M%SZ")

    def _scope_filter(self, allowed_user: str | None, allowed_conv: str | None):
        def _f(c: ContextObject) -> bool:
            uid = (c.metadata or {}).get("user_id")
            cid = (c.metadata or {}).get("conversation_id")
            ok_user = (allowed_user is None) or (uid == allowed_user)
            ok_conv = (allowed_conv is None) or (cid == allowed_conv)
            return ok_user and ok_conv
        return _f

    # nodes table mirrors essential metadata for fast selection
    def _upsert_node(self, ctx: 'ContextObject') -> None:
        try:
            uid = (ctx.metadata or {}).get("user_id")
            cid = (ctx.metadata or {}).get("conversation_id")
            with self._conn:
                self._conn.execute(
                    """INSERT INTO nodes(context_id, user_id, conversation_id,
                                         summary, last_accessed, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(context_id) DO UPDATE SET
                         user_id=excluded.user_id,
                         conversation_id=excluded.conversation_id,
                         summary=excluded.summary,
                         last_accessed=excluded.last_accessed,
                         timestamp=excluded.timestamp
                    """,
                    (
                        ctx.context_id, uid, cid,
                        (ctx.summary or ""),
                        ctx.last_accessed,
                        ctx.timestamp,
                    )
                )
        except Exception:
            pass

    def _get_recent_candidates(self, user: str | None, conv: str | None, limit: int) -> List[str]:
        where = []
        args: List[Any] = []
        if user is not None:
            where.append("user_id = ?")
            args.append(user)
        if conv is not None:
            where.append("conversation_id = ?")
            args.append(conv)
        wh = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            SELECT context_id
            FROM nodes
            {wh}
            ORDER BY COALESCE(last_accessed, timestamp) ASC
        """
        # pull last N
        ids: List[str] = []
        try:
            cur = self._conn.execute(sql, args)
            rows = cur.fetchall()
            if rows:
                ids = [r[0] for r in rows[-limit:]]
        except Exception:
            pass
        return ids

    def _get_embeddings(self, ids: List[str]) -> Dict[str, np.ndarray]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        out: Dict[str, np.ndarray] = {}
        try:
            cur = self._conn.execute(
                f"SELECT context_id, dim, vec FROM embeddings WHERE context_id IN ({placeholders})",
                ids
            )
            for cid, dim, blob in cur.fetchall():
                # vec stored as raw float32 bytes
                v = np.frombuffer(blob, dtype=np.float32, count=dim)
                out[cid] = v
        except Exception:
            pass
        return out

    def _persist_embedding(self, ctx_id: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        blob = vec.tobytes(order="C")
        dim = int(vec.shape[0])
        with self._conn:
            self._conn.execute(
                """INSERT INTO embeddings(context_id, dim, vec)
                   VALUES (?, ?, ?)
                   ON CONFLICT(context_id) DO UPDATE SET
                     dim=excluded.dim, vec=excluded.vec
                """,
                (ctx_id, dim, blob)
            )

    def _sum_assoc_strength(self, ids: List[str]) -> Dict[str, float]:
        """Return total incident edge weight for each id (degree strength)."""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        out: Dict[str, float] = {i: 0.0 for i in ids}
        try:
            # outgoing
            cur = self._conn.execute(
                f"SELECT src, SUM(w) FROM edges WHERE src IN ({placeholders}) GROUP BY src", ids
            )
            for src, s in cur.fetchall():
                out[src] = out.get(src, 0.0) + float(s or 0.0)
            # incoming
            cur = self._conn.execute(
                f"SELECT dst, SUM(w) FROM edges WHERE dst IN ({placeholders}) GROUP BY dst", ids
            )
            for dst, s in cur.fetchall():
                out[dst] = out.get(dst, 0.0) + float(s or 0.0)
        except Exception:
            pass
        return out

    # ──────────────────────────────────────────────────────────────
    # Graph maintenance (SQLite)
    # ──────────────────────────────────────────────────────────────
    def decay_graph_edges(self, half_life_secs: float = 86_400.0, min_w: float = 1e-6) -> None:
        now = time.time()
        if not hasattr(self, "_last_graph_decay_ts"):
            self._last_graph_decay_ts = now
            return
        dt = max(now - self._last_graph_decay_ts, 0.0)
        if dt < 60.0:
            return
        factor = self._half_life_factor(dt, half_life_secs)
        with self._conn:
            try:
                self._conn.execute("UPDATE edges SET w = w * ?", (factor,))
                self._conn.execute("DELETE FROM edges WHERE w <= ?", (min_w,))
            except Exception:
                pass
        self._last_graph_decay_ts = now

    def start_episode(self, title: str, meta: Dict[str, Any] | None = None) -> 'ContextObject':
        from context import ContextObject
        epi = ContextObject(domain="stage", component="episode", semantic_label="episode")
        epi.summary = title
        epi.tags = ["episode"]
        epi.metadata.update(meta or {})
        self.repo.save(epi)
        self._current_episode_id = epi.context_id
        # cache in nodes
        self._upsert_node(epi)
        return epi

    def add_to_episode(self, ctx: 'ContextObject') -> None:
        eid = getattr(self, "_current_episode_id", None)
        if not eid:
            return
        self._add_edge(eid, ctx.context_id, 1.5)
        self._add_edge(ctx.context_id, eid, 1.0)

    def end_episode(self) -> None:
        if hasattr(self, "_current_episode_id"):
            delattr(self, "_current_episode_id")

    def consolidate_stm_to_ltm(self, promote_threshold: int = 3) -> None:
        for ctx in self.repo.query(lambda c: True):
            cnt = (ctx.recall_stats or {}).get("count", 0)
            if cnt >= promote_threshold and ctx.memory_tier != "LTM":
                ctx.memory_tier = "LTM"
                ctx.pinned = True
                ctx.touch()
                self.repo.save(ctx)
                # keep nodes up-to-date
                self._upsert_node(ctx)

    # JSON graph saver is a no-op now (kept for API compatibility)
    def _save_graph(self) -> None:
        pass

    # ──────────────────────────────────────────────────────────────
    # Vector search (SQLite-backed embeddings)
    # ──────────────────────────────────────────────────────────────
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
    ) -> list['ContextObject']:
        if not cue_text:
            return []

        # query vector (unit)
        q = np.asarray(embed_fn(cue_text), dtype=np.float32).reshape(-1)
        q /= (np.linalg.norm(q) + 1e-9)

        # candidate pool: most-recent in scope (from nodes table)
        MAX_CANDS = int(os.getenv("CTX_VECTOR_MAX_CANDS", "2000"))
        cand_ids = self._get_recent_candidates(allowed_user, allowed_conv, MAX_CANDS)

        # ensure nodes table has content for fresh rows (best-effort)
        if not cand_ids:
            # fallback: scan repo (cap)
            filt = self._scope_filter(allowed_user, allowed_conv)
            recents = sorted(
                self.repo.query(filt),
                key=lambda c: (c.last_accessed or c.timestamp)
            )[-MAX_CANDS:]
            cand_ids = [c.context_id for c in recents]
            for c in recents:
                self._upsert_node(c)

        # fetch embeddings for candidates; compute & persist missing if reembed or absent
        emb_map = self._get_embeddings(cand_ids)
        missing = [cid for cid in cand_ids if (cid not in emb_map) or reembed]

        if missing:
            # compute embeddings for the missing in one pass using repo
            id_to_ctx: Dict[str, 'ContextObject'] = {}
            for cid in missing:
                try:
                    id_to_ctx[cid] = self.repo.get(cid)
                except KeyError:
                    continue
            for cid, ctx in id_to_ctx.items():
                txt = (ctx.summary or "").strip()
                if not txt:
                    continue
                v = np.asarray(embed_fn(txt), dtype=np.float32).reshape(-1)
                self._persist_embedding(cid, v)
                emb_map[cid] = v

        # build matrix for fast cosine = dot on unit vectors
        rows: List[Tuple[str, float, float]] = []  # (id, hybrid_score, pure_sim)
        if not emb_map:
            return []

        # align vectors
        ids = [cid for cid in cand_ids if cid in emb_map]
        V = np.stack([emb_map[cid] for cid in ids], axis=0).astype(np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        U = V / norms
        sims = (U @ q).astype(np.float32)  # cosine

        # optional hybrid terms
        rec_map: Dict[str, float] = {}
        assoc_map: Dict[str, float] = {}
        if use_hybrid_rank:
            # fetch node times for recency
            placeholders = ",".join("?" for _ in ids)
            try:
                cur = self._conn.execute(
                    f"SELECT context_id, COALESCE(last_accessed, timestamp) FROM nodes WHERE context_id IN ({placeholders})",
                    ids
                )
                now = default_clock()
                for cid, ts in cur.fetchall():
                    try:
                        last = self._parse_ts(ts)
                        age = (now - last).total_seconds()
                        rec_map[cid] = 1.0 / (1.0 + age)
                    except Exception:
                        rec_map[cid] = 0.0
            except Exception:
                pass
            assoc_map = self._sum_assoc_strength(ids)

        for i, cid in enumerate(ids):
            sim = float(sims[i])
            if use_hybrid_rank:
                rec = rec_map.get(cid, 0.0)
                assoc = math.log1p(assoc_map.get(cid, 0.0))
                score = 0.55 * sim + 0.25 * rec + 0.20 * assoc
            else:
                score = sim
            rows.append((cid, score, sim))

        rows.sort(key=lambda t: t[1], reverse=True)
        pool = rows[: max(top_k * 4, top_k)]

        # MMR for diversity
        alpha = 0.75
        selected: List[Tuple[str, float]] = []
        vec_cache: Dict[str, np.ndarray] = {cid: (U[ids.index(cid)]) for cid, _, _ in pool}

        cands = pool[:]
        while cands and len(selected) < top_k:
            best = None
            best_score = -1e9
            for cid, score, _sim in cands:
                if not selected:
                    mmr = score
                else:
                    v = vec_cache.get(cid)
                    sims2 = []
                    for scid, _ in selected:
                        sv = vec_cache.get(scid)
                        if sv is not None and v is not None:
                            sims2.append(float(np.dot(v, sv)))
                    novelty = 1.0 - (max(sims2) if sims2 else 0.0)
                    mmr = alpha * score + (1.0 - alpha) * novelty
                if mmr > best_score:
                    best_score = mmr
                    best = (cid, score)
            selected.append(best)
            cands = [(cid, s, sim) for (cid, s, sim) in cands if cid != best[0]]

        # materialize
        out: List['ContextObject'] = []
        for cid, score in selected:
            try:
                obj = self.repo.get(cid)
                obj.retrieval_score = float(score)
                out.append(obj)
            except KeyError:
                continue
        return out

    # ──────────────────────────────────────────────────────────────
    # Lightweight recall (one hop via association_strengths)
    # ──────────────────────────────────────────────────────────────
    def recall(
        self,
        seed_ids: List[str],
        k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> List['ContextObject']:
        weights = weights or {"assoc": 1.0, "recency": 1.0}
        now = default_clock()
        if not seed_ids:
            return []

        # infer scope
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

        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results: List['ContextObject'] = []
        for cid, score in top:
            try:
                ctx = self.repo.get(cid)
            except KeyError:
                continue
            ctx.retrieval_score = score
            ctx.retrieval_metadata = {"seed_ids": seed_ids}
            ctx.record_recall(stage_id="recall", coactivated_with=seed_ids, retrieval_score=score)
            self.repo.save(ctx)
            self._upsert_node(ctx)
            results.append(ctx)
        return results

    # ──────────────────────────────────────────────────────────────
    # Multi-hop activation over persisted edges (SQLite)
    # ──────────────────────────────────────────────────────────────
    def spread_activation(
        self,
        seed_ids: List[str],
        hops: int = 3,
        decay: float = 0.5,
        assoc_weight: float = 1.0,
        recency_weight: float = 1.0,
    ) -> Dict[str, float]:
        now = default_clock()
        if not seed_ids:
            return {}
        activation: Dict[str, float] = {cid: 1.0 for cid in seed_ids}

        # one hop at a time (pull neighbors from SQL)
        frontier = dict(activation)
        for hop in range(1, hops + 1):
            new_frontier: Dict[str, float] = {}
            ids = list(frontier.keys())
            if not ids:
                break
            placeholders = ",".join("?" for _ in ids)
            try:
                cur = self._conn.execute(
                    f"SELECT src, dst, w FROM edges WHERE src IN ({placeholders})",
                    ids
                )
                for src, dst, w in cur.fetchall():
                    act = frontier.get(src, 0.0)
                    inc = act * float(w) * assoc_weight * (decay ** (hop - 1))
                    new_frontier[dst] = new_frontier.get(dst, 0.0) + inc
            except Exception:
                pass
            for k, v in new_frontier.items():
                activation[k] = activation.get(k, 0.0) + v
            frontier = new_frontier

        # recency bonus (from nodes)
        if activation:
            placeholders = ",".join("?" for _ in activation.keys())
            try:
                cur = self._conn.execute(
                    f"SELECT context_id, COALESCE(last_accessed, timestamp) FROM nodes WHERE context_id IN ({placeholders})",
                    list(activation.keys())
                )
                for cid, ts in cur.fetchall():
                    try:
                        last = self._parse_ts(ts)
                        age = (now - last).total_seconds()
                        activation[cid] += recency_weight / (1.0 + age)
                    except Exception:
                        pass
            except Exception:
                pass

        return activation

    def decay_and_promote(
        self,
        half_life: float = 86_400.0,
        promote_threshold: int = 3
    ) -> None:
        now = default_clock()
        for row in list(self.repo.query(lambda c: True)):
            try:
                ctx = self.repo.get(row.context_id)
            except KeyError:
                continue

            last_ts = self._parse_ts(ctx.last_accessed)
            delta = (now - last_ts).total_seconds()

            new_strengths: Dict[str, float] = {}
            for oid, strength in (ctx.association_strengths or {}).items():
                try:
                    self.repo.get(oid)
                except KeyError:
                    continue
                decayed = strength * self._half_life_factor(delta, half_life)
                if decayed > 1e-6:
                    new_strengths[oid] = decayed

            should_promote = (ctx.recall_stats or {}).get("count", 0) >= promote_threshold
            if new_strengths != (ctx.association_strengths or {}) or (should_promote and ctx.memory_tier != "LTM"):
                ctx.association_strengths = new_strengths
                if should_promote:
                    ctx.memory_tier = "LTM"
                ctx.touch()
                self.repo.save(ctx)
                self._upsert_node(ctx)

    def reinforce(self, context_id: str, coactivated: List[str]) -> None:
        try:
            self.repo.get(context_id)
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
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                a, b = all_ids[i], all_ids[j]
                self._add_edge(a, b, 0.5)
                self._add_edge(b, a, 0.5)

        try:
            base = self.repo.get(context_id)
            base.record_recall(stage_id="reinforce", coactivated_with=valid)
            self.repo.save(base)
            self._upsert_node(base)
        except Exception:
            pass

    def prune(self, ttl_hours: int) -> None:
        cutoff = default_clock() - timedelta(hours=ttl_hours)
        def stale(c: 'ContextObject') -> bool:
            la = self._parse_ts(c.last_accessed)
            return la < cutoff and not c.pinned
        for ctx in self.repo.query(stale):
            self.repo.delete(ctx.context_id)
            # clean up persisted state
            try:
                with self._conn:
                    self._conn.execute("DELETE FROM nodes WHERE context_id=?", (ctx.context_id,))
                    self._conn.execute("DELETE FROM embeddings WHERE context_id=?", (ctx.context_id,))
                    self._conn.execute("DELETE FROM edges WHERE src=? OR dst=?", (ctx.context_id, ctx.context_id))
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────
    # Edge ops (persisted in SQLite)
    # ──────────────────────────────────────────────────────────────
    def _add_edge(self, src: str, dst: str, w: float) -> None:
        if src == dst or w == 0.0:
            return
        now = int(time.time())
        with self._conn:
            self._conn.execute(
                """INSERT INTO edges(src, dst, w, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(src, dst) DO UPDATE SET
                     w = edges.w + excluded.w,
                     updated_at = excluded.updated_at
                """,
                (src, dst, float(w), now)
            )

    # ------------- 1️⃣  register_relationships ----------------------
    def register_relationships(
        self,
        ctx: 'ContextObject',
        embed_fn: Callable[[str], np.ndarray],
    ) -> None:
        """
        Call once after saving a new/updated ContextObject.
        • Skips re-registering relationships for the same ctx.
        • Maintains node cache + persisted edges + embeddings.
        """
        cid = ctx.context_id
        if cid in self._registered_ctxs:
            return
        self._registered_ctxs.add(cid)

        # keep node row fresh
        self._upsert_node(ctx)

        # ---------- explicit references ----------
        for rid in ctx.references:
            self._add_edge(cid, rid, _HMR_REF_W)
            self._add_edge(rid, cid, _HMR_REF_W)

        # ---------- shared tags ----------
        MAX_TAG_NEIGHBORS = 200
        for tag in (ctx.tags or []):
            tag_node = f"tag::{tag}"
            self._add_edge(cid, tag_node, _HMR_TAG_W)
            self._add_edge(tag_node, cid, _HMR_TAG_W)
            # nothing else to trim here — edges table handles it

        # ---------- semantic similarity (last 200 in scope) ----------
        try:
            base_text = (ctx.summary or "").strip()
            if base_text:
                # persist this embedding now (so future queries are fast)
                v1 = np.asarray(embed_fn(base_text), dtype=np.float32).reshape(-1)
                self._persist_embedding(cid, v1)

                allowed_user = (ctx.metadata or {}).get("user_id")
                allowed_conv = (ctx.metadata or {}).get("conversation_id")

                recent_ids = self._get_recent_candidates(allowed_user, allowed_conv, 200)
                # include this ctx in nodes if not already
                if cid not in recent_ids:
                    recent_ids.append(cid)

                emb_map = self._get_embeddings(recent_ids)
                # compute similarities
                u1 = v1 / (np.linalg.norm(v1) + 1e-9)
                for oid, v2 in emb_map.items():
                    if oid == cid:
                        continue
                    u2 = v2 / (np.linalg.norm(v2) + 1e-9)
                    sim = float(np.dot(u1, u2))
                    if sim >= _HMR_SIM_THRESH:
                        w = _HMR_SIM_W * sim
                        self._add_edge(cid, oid, w)
                        self._add_edge(oid, cid, w)
        except Exception:
            pass

        # ---------- temporal proximity (<10 min) ----------
        now_sec = ctx._ts_seconds()
        window = 600  # seconds
        allowed_user = (ctx.metadata or {}).get("user_id")
        allowed_conv = (ctx.metadata or {}).get("conversation_id")

        # fetch candidates within ~window using nodes cache
        ids = self._get_recent_candidates(allowed_user, allowed_conv, 500)
        for oid in ids:
            if oid == cid:
                continue
            try:
                other = self.repo.get(oid)
            except KeyError:
                continue
            age = abs(now_sec - other._ts_seconds())
            if age <= window:
                w = self._half_life_factor(age, _HMR_DECAY_SECS)
                self._add_edge(cid, oid, w)
                self._add_edge(oid, cid, w)

    # ------------- 2️⃣  holographic_recall --------------------------
    def holographic_recall(
        self,
        cue_ids: List[str] | None = None,
        cue_text: str | None = None,
        hops: int = 2,
        top_n: int = 10,
        embed_fn: Callable[[str], np.ndarray] | None = None
    ) -> List['ContextObject']:
        """
        Fuse cue_ids &/or cue_text into a single excitation vector,
        run multi-hop spreading activation over persisted edges, return top_n.
        """
        self.decay_graph_edges()
        self.consolidate_stm_to_ltm()

        cue_ids = cue_ids or []
        activation: Dict[str, float] = collections.Counter({cid: 1.0 for cid in cue_ids})

        # infer scope from first cue
        owner_user = owner_conv = None
        if cue_ids:
            try:
                first = self.repo.get(cue_ids[0])
                owner_user = (first.metadata or {}).get("user_id")
                owner_conv = (first.metadata or {}).get("conversation_id")
            except Exception:
                pass

        # text cue → immediate sim activation
        if cue_text and embed_fn:
            q = np.asarray(embed_fn(cue_text), dtype=np.float32).reshape(-1)
            q /= (np.linalg.norm(q) + 1e-9)
            # candidates = last N nodes in scope
            ids = self._get_recent_candidates(owner_user, owner_conv, 2000)
            emb_map = self._get_embeddings(ids)
            for cid, v in emb_map.items():
                u = v / (np.linalg.norm(v) + 1e-9)
                sim = float(np.dot(q, u))
                if sim >= _HMR_SIM_THRESH:
                    activation[cid] += _HMR_SIM_W * sim

        def _in_scope(cid: str) -> bool:
            if owner_user is None and owner_conv is None:
                return True
            try:
                c = self.repo.get(cid)
                return self._scope_filter(owner_user, owner_conv)(c)
            except Exception:
                return False

        # spread over edges
        frontier = dict(activation)
        for _ in range(hops):
            new_frontier: Dict[str, float] = collections.Counter()
            if not frontier:
                break
            ids = list(frontier.keys())
            placeholders = ",".join("?" for _ in ids)
            try:
                cur = self._conn.execute(
                    f"SELECT src, dst, w FROM edges WHERE src IN ({placeholders})",
                    ids
                )
                for src, dst, w in cur.fetchall():
                    if not _in_scope(dst):
                        continue
                    new_frontier[dst] += frontier.get(src, 0.0) * float(w)
            except Exception:
                pass
            for k, v in new_frontier.items():
                activation[k] = activation.get(k, 0.0) + v
            frontier = new_frontier

        # scope filter BEFORE ranking/MMR
        if owner_user is not None or owner_conv is not None:
            activation = {k: v for k, v in activation.items() if _in_scope(k)}

        # candidate pool
        cands = sorted(activation.items(), key=lambda kv: kv[1], reverse=True)[: max(top_n * 4, top_n)]

        # MMR (use embeddings if available)
        vecs: Dict[str, Optional[np.ndarray]] = {}
        if embed_fn and cands:
            emb_map = self._get_embeddings([cid for cid, _ in cands])
            for cid, _ in cands:
                v = emb_map.get(cid)
                if v is None:
                    try:
                        obj = self.repo.get(cid)
                        txt = (obj.summary or "").strip()
                        if txt:
                            v = np.asarray(embed_fn(txt), dtype=np.float32).reshape(-1)
                            self._persist_embedding(cid, v)
                    except Exception:
                        v = None
                vecs[cid] = None if v is None else (v / (np.linalg.norm(v) + 1e-9))

        alpha = 0.75
        selected: List[Tuple[str, float]] = []
        cand_list = cands[:]
        while cand_list and len(selected) < top_n:
            best = None
            best_score = -1e9
            for cid, rel in cand_list:
                v = vecs.get(cid)
                if not selected or v is None:
                    novelty = 1.0
                else:
                    sims = []
                    for scid, _ in selected:
                        sv = vecs.get(scid)
                        if sv is not None:
                            sims.append(float(np.dot(v, sv)))
                    novelty = 1.0 - (max(sims) if sims else 0.0)
                mmr = alpha * float(rel) + (1.0 - alpha) * novelty
                if mmr > best_score:
                    best_score = mmr
                    best = (cid, float(rel))
            selected.append(best)
            cand_list = [(cid, r) for (cid, r) in cand_list if cid != best[0]]

        out: List['ContextObject'] = []
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
