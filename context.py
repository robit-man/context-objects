# context.py

import uuid
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

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
    """

    # Schema version
    schema_version: int = field(init=False, default=1)

    # Identification
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str           # e.g. "segment", "stage", "comparator"
    component: str        # e.g. "instruction", "tool_chain", "and"
    semantic_label: str   # e.g. "select_tools"
    version: Optional[str] = None
    timestamp: str = field(default_factory=lambda: default_clock().strftime("%Y%m%dT%H%M%SZ"))

    # Core references
    segment_ids: List[str] = field(default_factory=list)
    stage_id:    Optional[str] = None
    references:  List[str] = field(default_factory=list)
    children:    List[str] = field(default_factory=list)

    # Content & summaries
    summary: Optional[str] = None
    tags:    List[str] = field(default_factory=list)

    # Retrieval & similarity
    embedding:        Optional[List[float]] = None
    retrieval_score:  Optional[float] = None
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

    # Provenance & transformation
    provenance: Dict[str, Any] = field(default_factory=dict)

    # Graph & memory tier
    graph_node_id: Optional[str] = None
    memory_tier:   Optional[str] = None

    # Associative memory
    memory_traces: List[MemoryTrace] = field(default_factory=list)
    association_strengths: Dict[str, float] = field(default_factory=dict)
    recall_stats: Dict[str, Any] = field(default_factory=lambda: {"count":0, "last_recalled":None})
    firing_rate: Optional[float] = None

    # Additional metadata & policies
    metadata: Dict[str, Any] = field(default_factory=dict)
    pinned: bool = False
    last_accessed: Optional[str] = None
    expires_at: Optional[str] = None
    acl: Dict[str, Any] = field(default_factory=dict)
    batch_id: Optional[str] = None
    dirty: bool = True

    # Internal lock for thread-safe updates
    _lock: threading.Lock = field(init=False, repr=False, compare=False, default_factory=threading.Lock)

    def __post_init__(self):
        # 1) Mandatory fields
        if not all([self.domain, self.component, self.semantic_label]):
            raise ValueError("domain, component and semantic_label are required")
        # 2) Normalize timestamp
        try:
            # if it's a datetime, convert; else assume correct string
            if isinstance(self.timestamp, datetime):
                self.timestamp = self.timestamp.strftime("%Y%m%dT%H%M%SZ")
        except Exception:
            pass
        # 3) Initialize last_accessed
        if not self.last_accessed:
            self.last_accessed = self.timestamp
        # 4) Default memory tier by domain
        if not self.memory_tier:
            self.memory_tier = {"segment":"STM","stage":"LTM"}.get(self.domain, "WM")
        # 5) Setup logger
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

    def compute_embedding(self, embedder: Callable[[str], List[float]]):
        """Generate embedding from summary via provided embedder."""
        if self.summary:
            self.embedding = embedder(self.summary)
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
            mt = MemoryTrace(stage_id=stage_id,
                             coactivated_with=coactivated_with or [],
                             metadata={"retrieval_score": retrieval_score})
            self.memory_traces.append(mt)

            # update recall stats
            stats = self.recall_stats
            stats["count"] += 1
            stats["last_recalled"] = mt.timestamp

            # simplistic firing_rate
            self.firing_rate = 1.0 / stats["count"]

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
    def from_dict(cls, data: Dict[str, Any]) -> "ContextObject":
        mts = [MemoryTrace(**mt) for mt in data.get("memory_traces", [])]
        obj = cls(
            context_id=data["context_id"],
            domain=data["domain"],
            component=data["component"],
            semantic_label=data["semantic_label"],
            version=data.get("version"),
            timestamp=data["timestamp"],
            segment_ids=data.get("segment_ids", []),
            stage_id=data.get("stage_id"),
            references=data.get("references", []),
            children=data.get("children", []),
            summary=data.get("summary"),
            tags=data.get("tags", []),
            embedding=data.get("embedding"),
            retrieval_score=data.get("retrieval_score"),
            retrieval_metadata=data.get("retrieval_metadata", {}),
            provenance=data.get("provenance", {}),
            graph_node_id=data.get("graph_node_id"),
            memory_tier=data.get("memory_tier"),
            memory_traces=mts,
            association_strengths=data.get("association_strengths", {}),
            recall_stats=data.get("recall_stats", {"count":0,"last_recalled":None}),
            firing_rate=data.get("firing_rate"),
            metadata=data.get("metadata", {}),
            pinned=data.get("pinned", False),
            last_accessed=data.get("last_accessed"),
            expires_at=data.get("expires_at"),
            acl=data.get("acl", {}),
            batch_id=data.get("batch_id"),
            dirty=data.get("dirty", True)
        )
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
        return cls(
            domain="segment",
            component="generic",
            semantic_label=semantic_label,
            segment_ids=content_refs,
            tags=tags or [],
            metadata=metadata
        )

    @classmethod
    def make_stage(
        cls,
        stage_name: str,
        input_refs: List[str],
        output: Any,
        **metadata
    ) -> "ContextObject":
        obj = cls(
            domain="stage",
            component=stage_name,
            semantic_label=stage_name,
            metadata=metadata
        )
        obj.references = input_refs
        obj.tags = [stage_name]
        obj.metadata["output"] = output
        return obj

# ─ Builder ─────────────────────────────────────────────────────────────────────
class ContextObjectBuilder:
    def __init__(self, semantic_label: str):
        self._fields: Dict[str, Any] = {"semantic_label": semantic_label}
    def domain(self, d: str):   self._fields["domain"] = d; return self
    def component(self, c: str):self._fields["component"] = c; return self
    def version(self, v: str):  self._fields["version"] = v; return self
    def tags(self, t: List[str]): self._fields["tags"] = t; return self
    def metadata(self, m: Dict[str,Any]): self._fields["metadata"] = m; return self
    def build(self) -> ContextObject:
        return ContextObject(**self._fields)

# ─ Repository / DAO Layer ──────────────────────────────────────────────────────
class ContextRepository:
    """
    JSONL-backed repository for ContextObjects.
    """
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        open(self.path, "a").close()

    def get(self, context_id: str) -> ContextObject:
        with open(self.path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if obj["context_id"] == context_id:
                    return ContextObject.from_dict(obj)
        raise KeyError(f"Context {context_id} not found")

    def save(self, ctx: ContextObject) -> None:
        with self._lock:
            with open(self.path, "a") as f:
                f.write(ctx.to_json() + "\n")
            ctx.dirty = False

    def delete(self, context_id: str) -> None:
        with self._lock:
            objs = []
            with open(self.path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj["context_id"] != context_id:
                        objs.append(obj)
            with open(self.path, "w") as f:
                for o in objs:
                    f.write(json.dumps(o) + "\n")

    def query(self, filter_func: Callable[[ContextObject], bool]) -> List[ContextObject]:
        results: List[ContextObject] = []
        with open(self.path, "r") as f:
            for line in f:
                ctx = ContextObject.from_dict(json.loads(line))
                if filter_func(ctx):
                    results.append(ctx)
        return results

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
        weights = weights or {"assoc":1.0, "recency":1.0}
        candidates: Dict[str, float] = {}
        now = default_clock()

        for sid in seed_ids:
            seed = self.repo.get(sid)
            # collect associations
            for other_id, strength in seed.association_strengths.items():
                score = strength * weights["assoc"]
                other = self.repo.get(other_id)
                # apply recency boost
                last = datetime.strptime(other.last_accessed, "%Y%m%dT%H%M%SZ")
                age = (now - last).total_seconds()
                score += weights["recency"] / (1.0 + age)
                candidates[other_id] = candidates.get(other_id, 0.0) + score

        # return top-k
        sorted_ids = sorted(candidates, key=lambda x: candidates[x], reverse=True)
        return [self.repo.get(cid) for cid in sorted_ids[:k]]

    def reinforce(self, context_id: str, coactivated: List[str]) -> None:
        ctx = self.repo.get(context_id)
        ctx.record_recall(stage_id=None, coactivated_with=coactivated)
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
