# grand_integrator.py

import heapq
from typing import List, Dict, Any
from datetime import datetime, timedelta
import re
from datetime import datetime

_ISO_SHORT = re.compile(r"^\d{8}T\d{6}$")        # 20250711T203918

def _parse_ts(ts: str) -> datetime:
    """Accept either full ISO-8601 or compact YYYYMMDDTHHMMSS[Z] format."""
    ts = ts.rstrip("Z")
    if _ISO_SHORT.match(ts):
        # convert 20250711T203918 → 2025-07-11T20:39:18
        ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:]}"
    return datetime.fromisoformat(ts)

class GrandIntegrator:
    """
    Maintains a dynamic knowledge-graph of ContextObjects.
    Nodes are keyed by ContextObject.context_id. Edges are implicit
    (by ingestion order or semantic similarity). Supports:
      • ingesting new context nodes
      • expanding around focus nodes
      • contracting to a manageable window by TTL and max_nodes
      • explaining the current graph state
    """

    def __init__(self, repo, memory_manager, config: Dict[str, Any]):
        """
        repo:        a ContextRepository instance
        memory_manager: a MemoryManager instance
        config:      dict with keys 'max_nodes', 'ttl_days', 'expand_k'
        """
        self.repo = repo
        self.memman = memory_manager
        self.config = config

        self.max_nodes = config.get("max_nodes", 50)
        self.ttl_days  = config.get("ttl_days", 30)
        self.expand_k  = config.get("expand_k", 5)

        # internal storage
        # nodes: cid -> { obj, timestamp, tags, type }
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # min-heap of (timestamp, cid) for TTL pruning
        self._time_heap: List[tuple[datetime, str]] = []
        self.embedder = config.get("embedder")  # Callable[[str], np.ndarray]
        self._vec_cache: Dict[str, Any] = {}    # cid → np.array


    def ingest(self, ctx_objs: List[Any]) -> None:
        """
        Add a list of ContextObject instances into the graph.
        """
        now = datetime.utcnow()
        for c in ctx_objs:
            cid = c.context_id
            ts = getattr(c, "timestamp", now)
            if isinstance(ts, str):
                ts = _parse_ts(ts)
            meta = {
                "edges": set(),  # populated externally (e.g., by memory manager)
                "obj": c,
                "timestamp": ts,
                "tags": set(getattr(c, "tags", [])),
                "type": getattr(c, "semantic_label", c.component),
            }
            self.nodes[cid] = meta
            heapq.heappush(self._time_heap, (ts, cid))
        self._prune_by_ttl()

    def expand(self, focus_ids: List[str], budget: int = None) -> List[Any]:
        """
        Return up to `budget` nodes most semantically similar to focus nodes.
        Fallbacks to recency if no embedder is configured.
        """
        budget = budget or self.expand_k
        if not self.embedder:
            # fallback to recency
            return super().expand(focus_ids, budget)

        # build focus vector
        focus_vecs = [
            self._get_vector(cid)
            for cid in focus_ids if cid in self.nodes
        ]
        if not focus_vecs:
            return []

        import numpy as np
        focus_mean = np.mean(focus_vecs, axis=0)

        # compute similarity for all non-focus nodes
        scores = []
        for cid, meta in self.nodes.items():
            if cid in focus_ids:
                continue
            vec = self._get_vector(cid)
            sim = float(np.dot(vec, focus_mean))  # assumes normalized embeddings
            scores.append((sim, cid))

        topk = heapq.nlargest(budget, scores, key=lambda x: x[0])
        return [self.nodes[cid]["obj"] for _, cid in topk]

    def _get_vector(self, cid: str):
        if cid in self._vec_cache:
            return self._vec_cache[cid]
        obj = self.nodes[cid]["obj"]
        vec = self.embedder(obj.summary or "")
        self._vec_cache[cid] = vec
        return vec
    
    def get_slice(self, kind: str, max_items: int = 10) -> List[Any]:
        """
        Return a slice of nodes matching `kind` = semantic_label/component.
        """
        matches = [
            meta["obj"] for meta in self.nodes.values()
            if getattr(meta["obj"], "semantic_label", "") == kind
        ]
        return sorted(matches, key=lambda c: c.timestamp)[-max_items:]

    def contract(self, keep_ids: List[str], max_nodes: int = None) -> List[Any]:
        max_nodes = max_nodes or self.max_nodes
        keep_set = set(keep_ids)

        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda kv: (
                kv[0] not in keep_set,                           # prioritize keep_ids
                -len(kv[1]["tags"]),                             # richer nodes preferred
                -kv[1]["timestamp"].timestamp()
            )
        )

        result = []
        seen = set()
        for cid, meta in sorted_nodes:
            if len(result) >= max_nodes:
                break
            if cid not in seen:
                result.append(meta["obj"])
                seen.add(cid)

        return result


    def explain(self) -> str:
        """
        Returns a human-readable summary of the graph’s state:
          • total nodes
          • TTL cutoff
          • max_nodes cap
          • five most recent node IDs & labels
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(days=self.ttl_days)
        total = len(self.nodes)
        recent_meta = sorted(
            self.nodes.values(),
            key=lambda m: m["timestamp"],
            reverse=True
        )[:5]

        lines = [
            f"GrandIntegrator: {total} nodes in graph",
            f"- TTL cutoff: {self.ttl_days} days (keep ts >= {cutoff.isoformat()}Z)",
            f"- Max nodes cap: {self.max_nodes}",
            "- 5 most recent nodes:",
        ]
        for meta in recent_meta:
            ts = meta["timestamp"].isoformat() + "Z"
            cid = meta["obj"].context_id
            lbl = getattr(meta["obj"], "semantic_label", meta["obj"].component)
            lines.append(f"   • {cid} ({lbl}) @ {ts}")
        return "\n".join(lines)
    
    def related_to(self, cid: str) -> List[Any]:
        """
        Return nodes explicitly linked to a given node by ID.
        """
        if cid not in self.nodes:
            return []
        edges = self.nodes[cid].get("edges", set())
        return [self.nodes[eid]["obj"] for eid in edges if eid in self.nodes]
    
    def _prune_by_ttl(self) -> None:
        """
        Remove any nodes older than TTL (self.ttl_days).
        """
        cutoff = datetime.utcnow() - timedelta(days=self.ttl_days)
        while self._time_heap and self._time_heap[0][0] < cutoff:
            ts, cid = heapq.heappop(self._time_heap)
            # only delete if timestamp still matches
            if cid in self.nodes and self.nodes[cid]["timestamp"] == ts:
                del self.nodes[cid]
