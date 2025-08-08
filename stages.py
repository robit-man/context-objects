# stages.py

import json
import re
import os
import ast
import inspect
import asyncio
import importlib
import random
import math
import hashlib
import logging
import difflib
import shutil
import threading
import concurrent.futures
import io
import contextlib
import textwrap
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import numpy as np
from ollama import chat, embed
from tools import Tools
from context import (
    ContextObject,
    default_clock,
    HybridContextRepository,
    MemoryManager,
)

@dataclass
class TurnState:
    turn_id: str
    plan_id: str
    graph: dict            # the raw DAG from planner
    pending: set[str]      # node-ids still to run
    done: set[str] = field(default_factory=set)
    results: dict[str, Any] = field(default_factory=dict)
    attempts_left: dict[str, int] = field(default_factory=dict)

# if your stage refers to other parts of your assembler you may need to
# import them too; adjust as needed.
# ——— NEW helper ———

def _utc_iso() -> str:
    """UTC timestamp ending with 'Z' (e.g. 2025-07-07T18:04:31.123456Z)."""
    return datetime.utcnow().isoformat() + "Z"

def _stamp(ctx, state):
    ctx.metadata.setdefault("conversation_id", state["conversation_id"])
    ctx.metadata.setdefault("user_id",         state["user_id"])

def squash_narrative(lines, keep_last=5, max_len=160):
    """
    lines : iterable[str]   (chronological order)
    Returns a compact list with at most `keep_last` recent items plus a
    single summary for the elided middle.
    """
    total = len(lines)
    if total <= keep_last:
        pool = lines
    else:
        pool = ["… (%d earlier entries)" % (total - keep_last)] + list(lines[-keep_last:])

    out = []
    for ln in pool:
        # cut at first sentence end ≤ max_len
        sent_end = re.search(r"[.!?]\s", ln)
        clip_at  = sent_end.end() if sent_end and sent_end.end() <= max_len else max_len
        out.append(textwrap.shorten(ln, clip_at, placeholder="…"))
    return out

def trim_snip(snip, max_words=100, ctx_id=""):
    words = snip.split()
    if len(words) <= max_words:
        return snip
    # find nearest sentence end before the cut
    cut = " ".join(words[:max_words])
    m   = re.search(r"[.!?]\s[^.!?]*?$", cut)
    cut = cut[:m.end()] if m else cut
    return f"{cut} … [extended {len(snip)-len(cut)} chars, search {ctx_id}]"

# -----------------------------------------------------------------------
# Pretty debug snapshot for _stage10_assemble_and_infer
# -----------------------------------------------------------------------
def _dump_ai_state(state: dict[str, Any]) -> None:
    from textwrap import shorten

    # ── helpers ─────────────────────────────────────────────────────────
    def _crop(val: str, ln: int = 60) -> str:
        return shorten(str(val), width=ln, placeholder="…")

    # Core scalars -------------------------------------------------------
    plan_out  = _crop(state.get("plan_output", "(none)"), 120)
    tc_ctx_id = getattr(state.get("tc_ctx"), "context_id", None)
    wm_ids    = state.get("wm_ids", [])
    rec_ids   = state.get("recent_ids", [])

    # Tool-ctx table -----------------------------------------------------
    tool_rows = []
    for t in state.get("tool_ctxs", []):
        stage  = t.stage_id
        cid    = t.context_id
        output = _crop(t.metadata.get("output", "(no output)"))
        tool_rows.append(f"│ {stage:<24} │ {cid:<36} │ {output}")

    tool_table = (
        "│ stage_id                 │ context_id                          │ output\n"
        "├" + "─"*26 + "┼" + "─"*38 + "┼" + "─"*62 + "\n"
        + "\n".join(tool_rows or ["│ (none)                    │ (n/a)                               │"])
    )

    # Final block --------------------------------------------------------
    block = f"""
╔═══════════════════════  ASSEMBLE & INFER  ═══════════════════════╗
║ State keys : {_crop(list(state.keys()), 100)}{' '*(53-len(_crop(list(state.keys()),100)))}║
║ plan_out   : {plan_out:<53}║
║ tc_ctx_id  : {tc_ctx_id or '(none)':<53}║
║ wm_ids     : {_crop(wm_ids, 100):<53}║
║ recent_ids : {_crop(rec_ids, 100):<53}║
╟───────────────────────────── TOOL CTXS ──────────────────────────╢
{tool_table}
╚══════════════════════════════════════════════════════════════════╝
"""
    print(block.strip("\n"))

@dataclass
class TurnStateModel:
    turn_id: str = ""
    plan_id: str = ""
    graph: dict | None = None           # {"nodes":[{"id","tool","args","after":[]}], "meta":{...}}
    pending_nodes: set = field(default_factory=set)
    completed_nodes: set = field(default_factory=set)
    tool_ctx_ids: list[str] = field(default_factory=list)
    budgets: dict = field(default_factory=lambda: {"tokens": 128_000, "time": 60, "calls": 20})
    last_results: dict = field(default_factory=dict)   # node_id -> raw output

def _ensure_turn_state(state: dict) -> TurnStateModel:
    """Attach a TurnStateModel into state['turn'] and return it."""
    ts = state.get("turn")
    if isinstance(ts, TurnStateModel):
        return ts
    ts = TurnStateModel()
    # create a new turn_id each user message if missing
    if not ts.turn_id:
        ts.turn_id = f"turn_{uuid.uuid4().hex[:8]}"
    state["turn"] = ts
    return ts

def _ensure_ids(ctx, conv_id, user_id):
    ctx.metadata.setdefault("conversation_id", conv_id)
    ctx.metadata.setdefault("user_id",         user_id)

def _stage1_record_input(self, user_text: str, state: Dict[str, Any]) -> ContextObject:
    turn = _ensure_turn_state(state)

    ctx = ContextObject.make_segment("user_input", [], tags=["user_input"])
    ctx.summary  = user_text
    ctx.stage_id = "user_input"

    # 🔑 inject routing + turn metadata
    ctx.metadata.update({
        "conversation_id": state["conversation_id"],
        "user_id":         state["user_id"],
        "turn_id":         turn.turn_id,
    })

    ctx.touch()
    self.repo.save(ctx)

    # keep for downstream
    state["user_ctx"] = ctx
    state["user_text"] = user_text
    return ctx

def _stage2_load_system_prompts(self) -> List[ContextObject]:
    """
    Load each static system-prompt ContextObject (never evicted)
    and return them as a list, in label order.
    """
    # 1) make sure we’ve seeded/updated the on-disk prompt artifacts
    self._seed_static_prompts()

    # 2) for each prompt label we know about, grab the newest ContextObject
    prompts: List[ContextObject] = []
    for label in self.system_prompts.keys():
        # find all saved prompts with this semantic_label
        candidates = sorted(
            self.repo.query(lambda c: c.semantic_label == label),
            key=lambda c: c.timestamp,
            reverse=True
        )
        if not candidates:
            # (shouldn’t happen, since seed just inserted it)
            continue
        prompts.append(candidates[0])

    # 3) Persist & index them so retrieval/integrator never prunes them
    self._persist_and_index(prompts)

    return prompts

# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Retrieve & Merge Context (token+recency aware, MMR dedupe, safer)
# ──────────────────────────────────────────────────────────────────────────────
def _stage3_retrieve_and_merge_context(
    self,
    user_text: str,
    user_ctx: "ContextObject | None",
    sys_ctx: "ContextObject | List[ContextObject] | None",
    extra_ctx: List["ContextObject"] | None = None,
    recall_ids: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Retrieve & merge context for downstream stages with:
      • RL-gated semantic/associative recalls
      • Recency-aware scoring
      • Working-memory slice (last N turns)
      • Robust dedup (content-hash + id)
      • Token-budget aware trimming hooks
      • MMR-style diversity for semantic/assoc/tool snippets

    Returns (backwards-compatible keys):
        merged, merged_ids, wm_ids, history, tools, semantic, assoc
    Also stashes the same into `state` if available as self._last_state.
    """
    from datetime import datetime, timezone
    import hashlib

    # ─── Helper shorthands ────────────────────────────────────────────
    def _ensure_list(x):
        if x is None:
            return []
        return x if isinstance(x, list) else [x]

    def _to_dt(ts) -> datetime:
        try:
            if isinstance(ts, datetime):
                return ts
            # support "...Z"
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    def _prefix(ctx):
        """Label summary as User: or Assistant: if not already."""
        txt = (getattr(ctx, "summary", None) or "").lstrip()
        if not txt:
            return
        if txt.startswith(("User:", "Assistant:")):
            return
        role = "Assistant" if ctx.semantic_label == "assistant" else "User"
        ctx.summary = f"{role}: {txt}"

    def _safe_text(ctx):
        return (getattr(ctx, "summary", None) or (getattr(ctx, "metadata", {}) or {}).get("text") or "").strip()

    def _hash_text(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _mmr_rank(items, get_vec, get_id, top_k):
        """
        Simple MMR with cosine via `self.embed_text`. Fallback: lexical Jaccard.
        items: list[Any]
        get_vec(item) -> vector-like or None
        get_id(item)  -> stable id str
        """
        import numpy as np

        # Precompute embeddings (best-effort)
        vecs, ids = [], []
        for it in items:
            try:
                v = self.embed_text(get_vec(it))
            except Exception:
                v = None
            vecs.append(v)
            ids.append(get_id(it))

        chosen, chosen_idx = [], []
        if not items:
            return []

        # pick the most recent/similar first by a simple proxy: order as-is
        for _ in range(min(top_k, len(items))):
            best_j, best_score = -1, -1e9
            for j, it in enumerate(items):
                if j in chosen_idx:
                    continue
                # relevance proxy: existence of vec
                rel = 1.0 if vecs[j] is not None else 0.5

                # diversity penalty: max sim to chosen
                div_pen = 0.0
                if chosen_idx and vecs[j] is not None:
                    for k in chosen_idx:
                        if vecs[k] is not None:
                            a = np.array(vecs[j], dtype=float).reshape(-1)
                            b = np.array(vecs[k], dtype=float).reshape(-1)
                            den = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
                            div_pen = max(div_pen, float(np.dot(a, b) / den))
                score = rel - 0.4 * div_pen
                if score > best_score:
                    best_score, best_j = score, j
            if best_j >= 0:
                chosen.append(items[best_j])
                chosen_idx.append(best_j)
        return chosen

    # ─── 1) Flatten inputs & identify conversation/user ───────────────
    user_list  = _ensure_list(user_ctx)
    sys_list   = _ensure_list(sys_ctx)
    extra_list = extra_ctx or []
    recall_ids = recall_ids or []

    if not user_list:
        return {"merged": [], "merged_ids": [], "wm_ids": [], "history": [],
                "tools": [], "semantic": [], "assoc": []}

    primary = user_list[0]
    meta = getattr(primary, "metadata", {}) or {}
    conv_id = meta.get("conversationid") or meta.get("conversation_id")
    user_id = meta.get("user_id")

    # ─── 2) Gather raw conversation segments ──────────────────────────
    segs = [
        c for c in self.repo.query(lambda c:
            c.domain == "segment"
            and c.semantic_label in ("user_input", "assistant")
            and ((c.metadata or {}).get("conversationid") == conv_id
                 or (c.metadata or {}).get("conversation_id") == conv_id)
            and (c.metadata or {}).get("user_id") in (user_id, None)
        )
    ]
    # include extra_ctx and explicit recall_ids
    seen_ids = {c.context_id for c in segs}
    for c in extra_list:
        if c.context_id not in seen_ids:
            segs.append(c); seen_ids.add(c.context_id)
    for rid in recall_ids:
        try:
            c = self.repo.get(rid)
            if c.context_id not in seen_ids:
                segs.append(c); seen_ids.add(c.context_id)
        except KeyError:
            pass
    segs.sort(key=lambda c: _to_dt(c.timestamp))

    # ─── 3) Working memory slice ──────────────────────────────────────
    WM_TURNS = int(getattr(self, "max_history_items", 20))
    history_slice = segs[-WM_TURNS:]
    wm_ids        = [c.context_id for c in history_slice]

    # ─── 4) RL gating signal (activation of WM ids) ───────────────────
    rf = 0.0
    try:
        if wm_ids and getattr(self, "memman", None):
            activation_map = self.memman.spread_activation(
                seed_ids=wm_ids, hops=2, decay=0.6,
                assoc_weight=1.0, recency_weight=0.5
            )
            top_vals = sorted(activation_map.values(), reverse=True)[: len(wm_ids)]
            if top_vals:
                rf = sum(top_vals) / len(top_vals)
    except Exception:
        rf = 0.0

    # ─── 5) Semantic retrieval (RL-gated) ─────────────────────────────
    semantic = []
    can_semantic = getattr(self, "rl", None) is None or self.rl.should_run("semantic_retrieval", rf)
    if can_semantic and getattr(self, "engine", None):
        try:
            cands = self.engine.query(
                stage_id="semantic_retrieval",
                similarity_to=user_text,
                top_k=max(1, int(getattr(self, "max_semantic_items", 10)) * 3)
            )
        except Exception:
            cands = []
        # Diversity: MMR
        semantic = _mmr_rank(cands, get_vec=lambda c: c.summary or "", get_id=lambda c: c.context_id,
                             top_k=int(getattr(self, "max_semantic_items", 10)))

    # ─── 6) Associative recall (RL-gated) ─────────────────────────────
    assoc = []
    can_assoc = getattr(self, "rl", None) is None or self.rl.should_run("memory_retrieval", rf)
    if can_assoc and wm_ids and getattr(self, "memman", None):
        try:
            scores = self.memman.spread_activation(
                seed_ids=wm_ids, hops=3, decay=0.7,
                assoc_weight=1.0, recency_weight=0.5
            )
            top_ids = sorted(scores, key=scores.get, reverse=True)[: int(getattr(self, "max_memory_items", 10))]
            for cid in top_ids:
                try:
                    assoc.append(self.repo.get(cid))
                except KeyError:
                    pass
        except Exception:
            assoc = []

    # ─── 7) Recent tool outputs (MMR-limited) ────────────────────────
    tools = [
        c for c in self.repo.query(lambda c:
            c.component == "tool_output"
            and ((c.metadata or {}).get("conversationid") == conv_id
                 or (c.metadata or {}).get("conversation_id") == conv_id)
        )
    ]
    tools.sort(key=lambda c: _to_dt(c.timestamp))
    # keep last K, but encourage diversity
    max_tools = int(getattr(self, "max_tool_outputs", 10))
    tools = _mmr_rank(tools[-(max_tools*3):], get_vec=lambda c: str((c.metadata or {}).get("output") or ""),
                      get_id=lambda c: c.context_id, top_k=max_tools)

    # ─── 8) Prefix role labels (for all) ─────────────────────────────
    for c in sys_list + user_list + history_slice + semantic + assoc + tools:
        try:
            _prefix(c)
        except Exception:
            pass

    # ─── 9) Merge & dedupe (id + content-hash) ───────────────────────
    merged = []
    seen = set()
    seen_hashes = set()

    def _add(lst):
        for c in lst:
            if c.context_id in seen:
                continue
            txt = _safe_text(c)
            h = _hash_text(txt) if txt else None
            # allow same id once; content duplicates get collapsed
            if h and h in seen_hashes:
                continue
            merged.append(c)
            seen.add(c.context_id)
            if h:
                seen_hashes.add(h)

    _add(sys_list)
    _add(user_list)
    _add(history_slice)
    _add(semantic)
    _add(assoc)
    _add(tools)

    # ensure latest user turn is last
    last_u = user_list[-1]
    if last_u.context_id in {c.context_id for c in merged}:
        merged = [c for c in merged if c.context_id != last_u.context_id] + [last_u]

    merged_ids = [c.context_id for c in merged]

    # ─── 10) Debug banner ─────────────────────────────────────────────
    self._print_stage_context("retrieve_and_merge_context", {
        "merged_ids":   merged_ids[:12],
        "wm_ids":       wm_ids,
        "semantic_ids": [c.context_id for c in semantic],
        "assoc_ids":    [c.context_id for c in assoc],
        "tool_ids":     [c.context_id for c in tools],
        "recall_feat":  round(rf, 3),
    })

    out = {
        "merged":     merged,
        "merged_ids": merged_ids,
        "wm_ids":     wm_ids,
        "history":    history_slice,
        "tools":      tools,
        "semantic":   semantic,
        "assoc":      assoc,
    }

    # stash for later stages if caller doesn't
    try:
        if getattr(self, "_last_state", None) is not None:
            self._last_state.update(out)
    except Exception:
        pass

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Intent Clarification (strict JSON, budget-aware, richer fields)
# ──────────────────────────────────────────────────────────────────────────────
def _stage4_intent_clarification(
    self,
    user_text: str,
    state: Dict[str, Any],
    *,
    on_token: Callable[[str], None] | None = None,
) -> "ContextObject":
    """
    Clarifier prompt now:
      • Uses post-tool block when available; else last 8 turns
      • Includes (up to) last 3 tool outputs (truncated)
      • Adds short semantic / associative / tool reference snippets
      • Enforces STRICT JSON with robust extraction
      • Produces keys: {keywords, notes, debug_notes, intents?, constraints?, red_flags?}
    """
    import json, textwrap, re, hashlib
    from context import ContextObject

    # ---------- 0) Guards ----------
    state          = state or {}
    merged         = state.get("merged", [])
    tool_ctxs      = state.get("tools", []) or state.get("tool_ctxs", [])
    semantic_ctxs  = state.get("semantic", [])
    assoc_ctxs     = state.get("assoc", [])
    tool_refs      = state.get("tools", [])

    # keep dialog ContextObjects in chronological order
    hist = [c for c in merged if c.semantic_label in ("user_input", "assistant")]
    hist.sort(key=lambda c: c.timestamp)

    # ---------- 1) Build “recent dialogue” ----------
    last_tool_ts = max((c.timestamp for c in tool_ctxs), default=None)
    dialogue: list[str] = []
    for c in hist:
        if last_tool_ts and c.timestamp <= last_tool_ts:
            continue
        role  = "User" if c.semantic_label == "user_input" else "Assistant"
        text  = c.summary or (c.metadata.get("text", "") if isinstance(c.metadata, dict) else "")
        dialogue.append(f"{role}: {text}")

    if not dialogue:
        for c in hist[-8:]:
            role = "User" if c.semantic_label == "user_input" else "Assistant"
            dialogue.append(f"{role}: {c.summary or (c.metadata.get('text', '') if isinstance(c.metadata, dict) else '')}")

    dialog_block = "\n".join(dialogue)[-1500:] or "(none)"

    # ---------- 2) Previous block (last 8 before current) ----------
    prev_lines: list[str] = []
    if len(hist) >= 2:
        for c in hist[-9:-1]:
            role = "User" if c.semantic_label == "user_input" else "Assistant"
            prev_lines.append(f"{role}: {c.summary or (c.metadata.get('text','') if isinstance(c.metadata, dict) else '')}")
    prev_block = "\n".join(prev_lines) if prev_lines else "(none)"

    # ---------- 3) Last 3 tool outputs ----------
    def _tool_preview(tc):
        payload = (tc.metadata.get("output") if isinstance(tc.metadata, dict) else None) or \
                  (tc.metadata.get("exception") if isinstance(tc.metadata, dict) else "") or ""
        try:
            blob = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        except Exception:
            blob = repr(payload)
        return (blob[:950] + " …") if len(blob) > 950 else blob

    tools_block = "\n".join(f"[{tc.stage_id}] {_tool_preview(tc)}" for tc in sorted(tool_ctxs, key=lambda c: c.timestamp)[-3:]) or "(none)"

    # ---------- 4) Short context snippets ----------
    def _first_n(ctxs, n=3):
        out = []
        for c in ctxs[:n]:
            short = (c.summary or "")[:120].replace("\n", " ")
            out.append(f"• {short}  (id={c.context_id[:8]})")
        return out

    semantic_block = "\n".join(_first_n(semantic_ctxs)) or "(none)"
    assoc_block    = "\n".join(_first_n(assoc_ctxs))    or "(none)"
    tools_block2   = "\n".join(_first_n(tool_refs))      or "(none)"

    # ---------- 5) System/context ----------
    clar_sys = self.clarifier_prompt
    full_ctx = textwrap.dedent(f"""
        ### Recent Dialogue ###
        {dialog_block}

        ### Previous User / Assistant Turns ###
        {prev_block}

        ### Recent Tool Outputs ###
        {tools_block}

        ### Retrieved Semantic Context ###
        {semantic_block}

        ### Retrieved Associative Context ###
        {assoc_block}

        ### Tool Reference Context ###
        {tools_block2}

        ### Current User Query ###
        {user_text}
    """).strip()

    MAX_PROMPT_CHARS = 4096
    if len(full_ctx) > MAX_PROMPT_CHARS:
        full_ctx = full_ctx[-MAX_PROMPT_CHARS:]

    # ---------- 6) Call Clarifier (STRICT JSON) ----------
    msgs = [
        {"role": "system", "content": clar_sys},
        {"role": "system", "content": (
            "Return ONLY JSON with keys: keywords(list), notes(str), debug_notes(list, optional), "
            "intents(list, optional), constraints(list, optional), red_flags(list, optional)."
        )},
        {"role": "system", "content": full_ctx},
        {"role": "user",   "content": user_text},
    ]
    out = self._stream_and_capture(
        self.primary_model,
        msgs,
        tag="[Clarifier]",
        images=state.get("images"),
        on_token=None,
    ).strip()

    # ---------- 7) Parse / repair JSON ----------
    def _extract_json_blob(s: str) -> str | None:
        # try fenced
        m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
        if m:
            return m.group(1)
        # try first {...} block
        m = re.search(r"(\{.*\})", s, flags=re.S)
        if m:
            return m.group(1)
        return None

    blob = _extract_json_blob(out) or out
    try:
        clar = json.loads(blob)
        if not isinstance(clar, dict):
            raise ValueError("not a dict")
    except Exception:
        clar = {"keywords": [], "notes": out, "debug_notes": dialogue[-8:]}

    # enforce required keys & types
    clar.setdefault("keywords", [])
    clar.setdefault("notes", "")
    clar.setdefault("debug_notes", dialogue[-8:])
    for k in ("intents", "constraints", "red_flags"):
        if k in clar and not isinstance(clar[k], list):
            clar[k] = [clar[k]]

    # ---------- 8) Persist Clarifier Context ----------
    input_refs = [state["user_ctx"].context_id] if state.get("user_ctx") else []
    clar_ctx = ContextObject.make_stage(
        "intent_clarification",
        input_refs=input_refs,
        output=clar,
    )
    clar_ctx.metadata.update(clar)
    clar_ctx.stage_id       = "intent_clarification"
    clar_ctx.semantic_label = "intent_clarification"
    clar_ctx.tags.append("clarifier")

    # propagate conversation/user ids if available
    if state.get("user_ctx"):
        try:
            clar_ctx.metadata.update(
                {
                    "conversation_id": state["user_ctx"].metadata["conversation_id"],
                    "user_id": state["user_ctx"].metadata["user_id"],
                }
            )
        except Exception:
            pass

    # topic fingerprint (useful for grouping downstream)
    try:
        clar_ctx.metadata["topic_id"] = hashlib.md5((clar_ctx.metadata.get("notes","")[:512]).encode("utf-8")).hexdigest()[:10]
    except Exception:
        pass

    clar_ctx.summary = (clar.get("notes") or "")[:250]
    clar_ctx.touch()
    self.repo.save(clar_ctx)

    # Optional: register relationships
    try:
        self.memman.register_relationships(clar_ctx, self.embed_text)
    except Exception:
        pass

    return clar_ctx


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — External Knowledge (MMR, recency boost, structured metadata)
# ──────────────────────────────────────────────────────────────────────────────
def _stage5_external_knowledge(
    self,
    clar_ctx: "ContextObject",
    state: Dict[str, Any] | None = None,
) -> "ContextObject":
    """
    Build ranked “external knowledge” for the planner with:
      • Multi-source signal fusion (dialogue/tool/semantic/assoc/fresh engine)
      • Recency boost + similarity
      • MMR to enforce diversity
      • Structured metadata (while keeping summary as newline-joined snippets)
    """
    import json, time, math
    from datetime import datetime, timezone
    from context import ContextObject

    state = state or {}

    # Tunables
    MAX_SNIPPETS        = 12
    MAX_PER_CATEGORY    = 6
    SIM_TOP_K           = max(3, int(getattr(self, "top_k", 3)))
    HALF_LIFE_DAYS      = 3.0
    NOW_TS              = time.time()

    # -------- embedding wrappers (accept both shapes) --------------
    def _embed_vec(text: str):
        try:
            v = self.embed_text(text or "")
            # normalize to 1D
            import numpy as np
            a = np.asarray(v, dtype=float)
            return a.reshape(-1)
        except Exception:
            return None

    def _cos(a, b) -> float:
        import numpy as np
        if a is None or b is None:
            return 0.0
        den = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / den)

    def _recency_boost(ctx) -> float:
        ts = getattr(ctx, "timestamp", None)
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            elif isinstance(ts, datetime):
                ts = ts.timestamp()
        except Exception:
            ts = NOW_TS
        age_days = max((NOW_TS - float(ts)) / 86400.0, 0.0)
        return 0.5 ** (age_days / HALF_LIFE_DAYS)

    # -------- collect candidates -----------------------------------
    scored = []  # (score, text, ctx, source)
    seen_texts = set()

    def _add(text: str, ctx, source: str, base_score: float):
        text = (text or "").strip()
        if not text:
            return
        line = text.replace("\n", " ")
        if line in seen_texts:
            return
        seen_texts.add(line)
        scored.append((base_score, line, ctx, source))

    # 1) recent dialogue
    for c in reversed(state.get("history", [])[-MAX_PER_CATEGORY:]):
        txt = c.summary or (c.metadata.get("text", "") if isinstance(c.metadata, dict) else "")
        _add(f"(DIALOGUE) {txt}", c, "dialogue", 1.0)

    # 2) recent tool outputs
    for c in reversed(state.get("tools", [])[-MAX_PER_CATEGORY:]):
        payload = (c.metadata.get("output") if isinstance(c.metadata, dict) else None) or \
                  (c.metadata.get("exception") if isinstance(c.metadata, dict) else "") or ""
        try:
            blob = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        except Exception:
            blob = repr(payload)
        _add(f"(TOOL {c.stage_id}) {blob[:300]}", c, "tool", 1.0)

    # 3) semantic & assoc recalls (use retrieval_score if present)
    for key, lbl in (("semantic", "SEM"), ("assoc", "ASSOC")):
        for c in state.get(key, [])[:MAX_PER_CATEGORY]:
            sim = getattr(c, "retrieval_score", None) or (c.metadata.get("retrieval_score") if isinstance(c.metadata, dict) else None) or 0.7
            txt = c.summary or (c.metadata.get("content", "") if isinstance(c.metadata, dict) else "")
            _add(f"({lbl}) {txt}", c, key, float(sim))

    # 4) holographic associative recall (lightweight; may overlap with assoc)
    try:
        seed_ids = [clar_ctx.context_id] + [c.context_id for c in state.get("history", [])[-2:]]
        hm_hits = self.memman.holographic_recall(
            cue_ids=seed_ids,
            cue_text=clar_ctx.summary or "",
            hops=2,
            top_n=MAX_PER_CATEGORY,
            embed_fn=lambda t: self.embed_text(t),
        )
    except Exception:
        hm_hits = []

    for h in hm_hits:
        assoc = getattr(h, "retrieval_score", None) or (h.metadata.get("retrieval_score") if isinstance(h.metadata, dict) else None) or 0.5
        txt = h.summary or (h.metadata.get("content", "") if isinstance(h.metadata, dict) else "")
        # add a recency boost
        score = 0.55 * float(assoc) + 0.25 * _recency_boost(h) + 0.20 * float(assoc)
        _add(f"(HMM) {txt}", h, "hmm", score)

    # 5) fresh similarity hits from engine (per keyword)
    kws = (clar_ctx.metadata.get("keywords") if isinstance(clar_ctx.metadata, dict) else None) or []
    if not kws and clar_ctx.summary:
        kws = [clar_ctx.summary]
    for kw in kws[:6]:
        try:
            hits = self.engine.query(similarity_to=kw, stage_id="external_knowledge_query", top_k=SIM_TOP_K)
        except Exception:
            hits = []
        for h in hits:
            txt = h.summary or (h.metadata.get("content", "") if isinstance(h.metadata, dict) else "")
            sim = getattr(h, "retrieval_score", None) or (h.metadata.get("retrieval_score") if isinstance(h.metadata, dict) else None) or 0.0
            score = 0.55 * float(sim) + 0.25 * _recency_boost(h)
            _add(f"(FRESH) {txt}", h, "fresh", score)

    # -------- MMR rerank & select top K -----------------------------
    scored.sort(key=lambda t: t[0], reverse=True)

    # prepare vectors for MMR on the text itself
    items = [{"score": s, "text": txt, "ctx": c, "src": src, "vec": _embed_vec(txt)} for (s, txt, c, src) in scored]

    chosen = []
    chosen_idx = []
    import numpy as np
    while len(chosen) < min(MAX_SNIPPETS, len(items)):
        best_j, best_obj, best_obj_score = -1, None, -1e9
        for j, it in enumerate(items):
            if j in chosen_idx:
                continue
            rel = float(it["score"])
            div_pen = 0.0
            for k in chosen_idx:
                a, b = it["vec"], items[k]["vec"]
                div_pen = max(div_pen, _cos(a, b))
            score = rel - 0.35 * div_pen
            if score > best_obj_score:
                best_j, best_obj, best_obj_score = j, it, score
        if best_j < 0:
            break
        chosen_idx.append(best_j)
        chosen.append(best_obj)

    # -------- Persist context --------------------------------------
    lines = [x["text"] for x in chosen]
    meta_snips = [
        {
            "text": x["text"],
            "source": x["src"],
            "score": x["score"],
            "ctx_id": getattr(x["ctx"], "context_id", None),
            "stage_id": getattr(x["ctx"], "stage_id", None),
        }
        for x in chosen
    ]

    ext = ContextObject.make_stage(
        "external_knowledge_retrieval",
        input_refs=[clar_ctx.context_id],
        output={"snippets": lines, "meta": meta_snips},
    )
    ext.stage_id       = "external_knowledge_retrieval"
    ext.semantic_label = "external_knowledge"
    ext.tags.append("external")
    ext.summary = "\n".join(lines)[:1024]
    ext.metadata["snippets"] = lines
    ext.metadata["meta"] = meta_snips

    ext.touch()
    self.repo.save(ext)

    try:
        self.memman.register_relationships(ext, self.embed_text)
    except Exception:
        pass

    # debug
    self._print_stage_context("external_knowledge_retrieval", {
        "chosen_snippets": lines,
        "total_candidates": len(scored),
    })

    # expose to downstream
    if state is not None:
        state["knowledge_snippets"] = lines

    return ext


def _stage5b_build_planning_kg(self, clar_ctx, know_ctx, tools_list, state):
    """
    Build a lightweight planning knowledge graph that:
      • Adds tool nodes + parameter nodes (from JSON schema)
      • Extracts literal required/optional param names per tool
      • Links concepts (clarifier keywords + knowledge snippets) to tools via embedding affinity
      • Exposes a compact summary of top tool candidates

    Output `kg_ctx.metadata["arg_catalog"]` exposes literal arg names for stage 7.
    """
    import json, re
    import numpy as np
    from context import ContextObject

    nodes, edges = {}, []

    # tool + param nodes (literal names)
    arg_catalog = {}
    for t in tools_list or []:
        tname = t["name"]
        tn = f"tool:{tname}"
        nodes[tn] = {"type": "tool"}
        params = (t.get("schema", {}) or {}).get("parameters", {}) or {}
        props  = (params.get("properties", {}) or {})
        req    = list(params.get("required", []) or [])
        arg_catalog[tname] = {
            "required": list(req),
            "optional": [k for k in props.keys() if k not in req],
            "properties": list(props.keys()),
        }
        for p in props:
            pn = f"param:{tname}.{p}"
            nodes[pn] = {"type": "param"}
            edges.append((tn, "has_param", pn))

    # concepts from clarifier + snippets
    kws = (clar_ctx.metadata.get("keywords") or [])
    if know_ctx and (getattr(know_ctx, "summary", None) or ""):
        kws += re.findall(r"\b[A-Za-z][\w-]{2,}\b", know_ctx.summary or "")
    # de-dup preserve order
    seen_kw = set()
    uniq_kws = []
    for k in kws:
        if k not in seen_kw:
            uniq_kws.append(k); seen_kw.add(k)

    for kw in uniq_kws[:80]:
        cn = f"concept:{kw}"
        nodes[cn] = {"type": "concept"}

    # embedding helpers
    def emb(x):
        try:
            return np.array(self.embed_text(x or ""), dtype=float).reshape(-1)
        except Exception:
            return None

    def cos(a, b):
        if a is None or b is None:
            return 0.0
        den = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / den)

    # pre-embed tool descriptions
    tool_desc = {t['name']: (t.get('description') or t['name']) for t in tools_list or []}
    t_vecs  = {name: emb(desc) for name, desc in tool_desc.items()}
    kw_vecs = {kw: emb(kw) for kw in uniq_kws[:80]}

    affinities = []
    for kw, kv in kw_vecs.items():
        for name, tv in t_vecs.items():
            score = cos(kv, tv)
            if score >= 0.22:
                affinities.append((f"concept:{kw}", "affinity", f"tool:{name}", score))

    affinities.sort(key=lambda x: x[3], reverse=True)
    top_pairs = affinities[: min(150, len(affinities))]

    kg = {
        "nodes": nodes,
        "edges": edges + [(a, b, c) for a, b, c, _ in top_pairs],
        "top_tool_candidates": sorted(
            {c.split(':', 1)[1] for _, _, c, _ in top_pairs},
            key=lambda n: max(s for a, b, c, s in top_pairs if c.endswith(n)),
            reverse=True
        )[:10]
    }

    kg_ctx = ContextObject.make_knowledge("planning_kg", kg, tags=["planning", "kg"])
    kg_ctx.summary = json.dumps({"top_tool_candidates": kg["top_tool_candidates"]}, ensure_ascii=False)[:512]
    kg_ctx.metadata["arg_catalog"] = arg_catalog
    self.repo.save(kg_ctx)

    try:
        self.memman.register_relationships(kg_ctx, self.embed_text)
    except Exception:
        pass

    state["planning_kg"] = kg
    state["arg_catalog"] = arg_catalog
    return kg_ctx


def _stage6_prepare_tools(self) -> list[dict]:
    """
    Return a list of tool specs for planning:
      [{ "name": str, "schema": dict, "description": str }, ...]
    Tries active tool_schema first, then falls back to legacy_tool_schema
    to survive transient demotions during auto-reload.
    """
    import json

    def _parse_schema(meta_schema):
        # meta_schema may be already a dict or a compact JSON string
        if isinstance(meta_schema, dict):
            return meta_schema
        try:
            return json.loads(meta_schema)
        except Exception:
            return {}

    # Prefer active tools; fall back to legacy when none are active
    rows = self.repo.query(lambda c: c.component == "schema" and "tool_schema" in (c.tags or []))
    if not rows:
        rows = self.repo.query(lambda c: c.component == "schema" and "legacy_tool_schema" in (c.tags or []))

    tools: list[dict] = []
    seen: set[str] = set()
    for c in rows:
        name = (c.semantic_label or "").strip()
        if not name or name in seen:
            continue
        sch = _parse_schema((c.metadata or {}).get("schema", "{}"))
        desc = (sch.get("description") or "").strip() if isinstance(sch, dict) else ""
        tools.append({"name": name, "schema": sch, "description": desc})
        seen.add(name)

    tools.sort(key=lambda t: t["name"])
    return tools


def _stage7_planning_summary(
    self,
    clar_ctx: ContextObject,
    know_ctx: ContextObject,
    tools_list: List[Dict[str, Any]],
    user_text: str,
    state: Dict[str, Any],
) -> Tuple[ContextObject, str]:
    """
    Stage 7 — Planner (DAG) with dynamic schema/signature grounding,
    strict arg-name selection, success/failed recall, and retry seeding.
    """
    import json, re, hashlib, datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ──────────────────────────────────────────────────────────────────
    # Normalize required contexts (guard against None)
    # ──────────────────────────────────────────────────────────────────
    def _ensure_ctx(ctx, stage_id: str, label: str) -> ContextObject:
        if ctx is not None:
            return ctx
        dummy = ContextObject.make_stage(stage_id, [], {"summary": ""})
        dummy.component = stage_id
        dummy.semantic_label = label
        try:
            _stamp(dummy, state)  # safe if available
        except Exception:
            pass
        dummy.touch()
        self.repo.save(dummy)
        return dummy

    clar_ctx = _ensure_ctx(clar_ctx, "intent_clarification_dummy", "intent_clarification")
    know_ctx = _ensure_ctx(know_ctx, "external_knowledge_dummy", "external_knowledge")

    # Precompute safe summaries (avoid repeated attribute access)
    _clar_summary = (clar_ctx.summary or "").strip()
    _know_summary = (know_ctx.summary or "").strip()

    # ──────────────────────────────────────────────────────────────────
    # Helpers (local to stage)
    # ──────────────────────────────────────────────────────────────────
    def _clean_json_block(text: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text or "", flags=re.S)
        if m:
            return m.group(1)
        m2 = re.search(r"(\{.*\})", text or "", flags=re.S)
        return (m2.group(1) if m2 else (text or "")).strip()

    def _first_sentence(desc: str) -> str:
        head = (desc or "").split(".", 1)[0]
        return head + ("." if head and not head.endswith(".") else "")

    # Canonicalize a tool call string from metadata safely (no external helpers)
    def _canon_call_str(meta: dict) -> str:
        """
        Best-effort canonical "tool(k=v,...)" from stored metadata.
        Accepts meta with keys: tool_name, args/tool_input/arguments, or raw call string.
        """
        call = (meta or {}).get("call") or (meta or {}).get("tool_call") or ""
        if isinstance(call, str) and "(" in call and ")" in call:
            return call.strip()

        name = (meta or {}).get("tool_name") or (meta or {}).get("name") or "tool"
        args = (
            (meta or {}).get("args")
            or (meta or {}).get("tool_input")
            or (meta or {}).get("arguments")
            or {}
        )
        # Stable key order
        try:
            items = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in sorted(args.items()))
        except Exception:
            items = ", ".join(f"{k}={repr(v)}" for k, v in sorted(args.items()))
        return f"{name}({items})"

    # ──────────────────────────────────────────────────────────────────
    # Recall helpers: successes, failures, mined-missing, and scoring
    # ──────────────────────────────────────────────────────────────────
    def _shorten(s: str, n: int = 240) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else (s[: n - 1] + "…")

    def _recency_score(ts: str, days: int = 14) -> float:
        """Newer → closer to 1.0 (simple linear over `days`)."""
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            t   = datetime.fromisoformat(ts.replace("Z","+00:00"))
            age = max(0.0, (now - t).total_seconds() / 86400.0)
            return max(0.0, 1.0 - (age / float(days)))
        except Exception:
            return 0.5

    def _lex_sim(a: str, b: str) -> float:
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, (a or "")[:500], (b or "")[:500]).ratio()
        except Exception:
            return 0.0

    def _score_example(ctx, seed: str) -> float:
        return 0.6 * _recency_score(getattr(ctx, "timestamp", "") or "") + 0.4 * _lex_sim(seed, getattr(ctx, "summary", "") or "")

    def _load_success_examples(tool_names: List[str], seed_text: str, top_k: int = 3) -> Dict[str, List[str]]:
        """
        Mine prior successful tool executions to show concrete, working calls per tool.
        Returns {tool_name: ["tool(arg=…)", ...]} ranked by recency+affinity.
        """
        if not tool_names:
            return {}
        names = set(tool_names)

        rows = self.repo.query(lambda c:
            c.component == "tool_output"
            and isinstance(c.metadata, dict)
            and (c.metadata or {}).get("ok") is True
            and ((c.metadata or {}).get("tool_name") in names)
        )

        by_tool: Dict[str, List[Tuple[float, str]]] = {}
        for tctx in rows:
            meta = tctx.metadata or {}
            tname = meta.get("tool_name")
            if not tname:
                continue
            call_str = _canon_call_str(meta)
            if not call_str:
                continue
            score = _score_example(tctx, seed_text)
            by_tool.setdefault(tname, []).append((score, call_str))

        out: Dict[str, List[str]] = {}
        for nm, arr in by_tool.items():
            arr.sort(key=lambda x: x[0], reverse=True)
            seen = set()
            kept = []
            for _, cs in arr:
                if cs not in seen:
                    seen.add(cs)
                    kept.append(_shorten(cs, 220))
                if len(kept) >= top_k:
                    break
            if kept:
                out[nm] = kept
        return out

    def _aggregate_failures(tool_names: List[str], top_k: int = 3) -> Dict[str, List[str]]:
        """
        Collect common failure messages per tool from prior tool_output exceptions
        and retry critiques marked not-ok. Returns {tool_name: ["msg", ...]}.
        """
        if not tool_names:
            return {}
        from collections import Counter
        names = set(tool_names)
        fails: Dict[str, Counter] = {}

        err_rows = self.repo.query(lambda c:
            c.component == "tool_output"
            and isinstance(c.metadata, dict)
            and ((c.metadata or {}).get("tool_name") in names)
        )
        for r in err_rows:
            meta = r.metadata or {}
            tname = meta.get("tool_name")
            if not tname:
                continue
            exc = meta.get("exception") or (r.summary or "")
            if not exc:
                continue
            msg = _shorten(str(exc), 200)
            if not msg:
                continue
            fails.setdefault(tname, Counter())
            fails[tname][msg] += 1

        crit_rows = self.repo.query(lambda c:
            c.component == "analysis"
            and c.semantic_label == "tool_retry_critique"
            and isinstance(c.metadata, dict)
            and (c.metadata or {}).get("tool_name") in names
            and (c.metadata or {}).get("status") in ("failed","rejected","bad")
        )
        for r in crit_rows:
            tname = (r.metadata or {}).get("tool_name")
            msg = _shorten(r.summary or (r.metadata or {}).get("hint") or "", 200)
            if not tname or not msg:
                continue
            fails.setdefault(tname, Counter())
            fails[tname][msg] += 1

        out: Dict[str, List[str]] = {}
        for nm, counter in fails.items():
            if not counter:
                continue
            out[nm] = [k for k, _ in counter.most_common(top_k)]
        return out

    # ──────────────────────────────────────────────────────────────────
    # Dynamic argument exposure: schema + signature (no hardcoded aliasing)
    # ──────────────────────────────────────────────────────────────────
    def _schema_req_props(schema: dict) -> tuple[list[str], dict]:
        params = (schema or {}).get("parameters", {}) or {}
        req    = list(params.get("required", []) or [])
        props  = dict(params.get("properties", {}) or {})
        return req, props

    def _get_tool_callable(tool_name: str):
        """Try to find a Python callable for a tool by name from known registries."""
        # 1) scan tools_list for embedded callable
        for t in tools_list or []:
            if t.get("name") == tool_name:
                for key in ("callable", "fn", "func", "impl", "handler", "object"):
                    if key in t and callable(t[key]):
                        return t[key]
        # 2) try self.tools_registry (common pattern)
        try:
            reg = getattr(self, "tools_registry", None)
            if isinstance(reg, dict) and callable(reg.get(tool_name)):
                return reg.get(tool_name)
        except Exception:
            pass
        # 3) try self.tools or self.<name>
        try:
            mod = getattr(self, "tools", None)
            if mod is not None:
                cand = getattr(mod, tool_name, None)
                if callable(cand):
                    return cand
        except Exception:
            pass
        try:
            cand = getattr(self, tool_name, None)
            if callable(cand):
                return cand
        except Exception:
            pass
        return None

    def _introspect_tool_signature(tool_name: str) -> tuple[list[str], list[str], bool]:
        """
        Returns (required_from_sig, optional_from_sig, accepts_var_kw).
        Uses the literal Python signature (excluding 'self', *args).
        """
        import inspect as _inspect
        fn = _get_tool_callable(tool_name)
        if fn is None:
            return [], [], False
        # unwrap bound/descriptor if needed
        if hasattr(fn, "__func__"):
            fn = fn.__func__
        try:
            sig = _inspect.signature(fn)
        except Exception:
            return [], [], False

        req, opt, vkw = [], [], False
        for p in sig.parameters.values():
            if p.name == "self":
                continue
            if p.kind == p.VAR_POSITIONAL:
                continue
            if p.kind == p.VAR_KEYWORD:
                vkw = True
                continue
            if p.default is _inspect._empty:
                req.append(p.name)
            else:
                opt.append(p.name)
        return req, opt, vkw

    def _allowed_keys_for_tool(tool_name: str, schema: dict) -> dict:
        """
        Canonical, *literal* names the planner/refiner must use:
        union of JSON-schema properties and Python signature names.
        """
        req_s, props = _schema_req_props(schema)
        req_sig, opt_sig, vkw = _introspect_tool_signature(tool_name)

        required = sorted(set(req_s) | set(req_sig))
        optional = sorted((set(props.keys()) | set(opt_sig)) - set(required))
        allowed  = required + [n for n in optional if n not in required]

        return {
            "required": required,
            "optional": optional,
            "allowed":  allowed,
            "accepts_varkw": bool(vkw),
            "properties": props,
        }

    def _history_param_usage(tool_name: str) -> dict[str,int]:
        """
        Mine *successful* past calls for the exact keyword names used.
        Looks at tool_output rows with ok=True and parses call strings.
        """
        import ast as _ast, re as _re
        counts: dict[str,int] = {}
        rows = self.repo.query(lambda c:
            c.component == "tool_output"
            and (c.metadata or {}).get("tool_name") == tool_name
            and bool((c.metadata or {}).get("ok"))
        )
        for r in rows:
            call = (r.metadata or {}).get("tool_call") or (r.metadata or {}).get("call") or ""
            call = call.strip()
            if not call:
                continue
            try:
                tree = _ast.parse(call)
                node = tree.body[0].value  # type: ignore
                for kw in getattr(node, "keywords", []):
                    if getattr(kw, "arg", None):
                        counts[kw.arg] = counts.get(kw.arg, 0) + 1
            except Exception:
                for m in _re.finditer(r"([A-Za-z_]\w*)\s*=", call):
                    k = m.group(1)
                    counts[k] = counts.get(k, 0) + 1
        return counts

    def _mine_missing_from_errors(tool_names: list[str]) -> dict[str, list[str]]:
        """
        Parse previous exceptions to detect missing parameter names.
        Returns {tool_name: ['param', ...]}.
        """
        if not tool_names:
            return {}
        names = set(tool_names)
        rows  = self.repo.query(lambda c: c.component == "tool_output")
        out: dict[str, list[str]] = {}
        import re as _re
        pats = [
            r"missing\s+\d+\s+required\s+positional\s+argument[s]?:\s*'([^']+)'",
            r"missing\s+1\s+required\s+keyword-only\s+argument:\s*'([^']+)'",
            r"required\s+property\s+'([^']+)'",
            r"field\s+required\s*[: ]\s*'([^']+)'",
            r"KeyError:\s*'([^']+)'",
        ]
        for r in sorted(rows, key=lambda c: c.timestamp, reverse=True):
            meta  = r.metadata or {}
            tname = meta.get("tool_name") or r.semantic_label or ""
            if tname not in names:
                continue
            exc = f"{meta.get('exception','')} {r.summary or ''}"
            found = set()
            for p in pats:
                for m in _re.finditer(p, exc):
                    if m and m.group(1):
                        found.add(m.group(1).strip())
            if found:
                cur = out.setdefault(tname, [])
                for f in found:
                    if f not in cur:
                        cur.append(f)
        return out

    def _normalize_args_dynamic(tool_name: str, args: dict, allowed_catalog: dict) -> tuple[dict, list[str]]:
        """
        Rename stray keys into canonical allowed names via *fuzzy* match,
        but only map into REQUIRED names; drop everything else.
        No hardcoded alias tables.
        """
        from difflib import SequenceMatcher as _SM
        info    = allowed_catalog.get(tool_name, {})
        allowed = set(info.get("allowed", []))
        required = set(info.get("required", []))
        props   = info.get("properties", {}) or {}

        out: dict = {}
        notes: list[str] = []

        for k, v in (args or {}).items():
            if k in allowed:
                out[k] = v
                continue
            # dynamic candidate: nearest allowed name by similarity
            best, score = None, 0.0
            for cand in allowed:
                s = _SM(None, k.lower(), cand.lower()).ratio()
                if s > score:
                    best, score = cand, s
            # only map if target is REQUIRED and similarity high
            if best and (best in required) and score >= 0.86:
                out[best] = v
                notes.append(f"renamed '{k}' → '{best}' (sim={score:.2f})")
            else:
                notes.append(f"dropped unknown arg '{k}'")

        # strip 'kwargs' unless schema explicitly allows it
        if "kwargs" in out and "kwargs" not in props:
            out.pop("kwargs", None)
            notes.append("dropped 'kwargs'")

        return out, notes

    # Graph validation (IDs, deps, cycles, tools)
    def _validate_graph(graph: dict, tools_list: list[dict]) -> list[str]:
        from collections import deque, defaultdict
        errs: list[str] = []
        nodes = graph.get("nodes") or []
        ids = [n.get("id") for n in nodes if n.get("id")]
        if len(ids) != len(set(ids)):
            errs.append("duplicate node ids")
        idset = set(ids)
        # after refs exist
        for n in nodes:
            for dep in (n.get("after") or []):
                if dep not in idset:
                    errs.append(f"node {n.get('id')} after-> {dep} missing")
        # unknown tools
        known = {t["name"] for t in tools_list}
        for n in nodes:
            if n.get("tool") not in known:
                errs.append(f"unknown tool '{n.get('tool')}' in node {n.get('id')}")
        # acyclicity (Kahn)
        indeg = defaultdict(int)
        g = defaultdict(list)
        for n in nodes:
            for dep in (n.get("after") or []):
                g[dep].append(n["id"]); indeg[n["id"]] += 1
        q = deque([i for i in ids if indeg[i]==0])
        seen = 0
        while q:
            u = q.popleft(); seen += 1
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        if seen != len(ids):
            errs.append("cycle detected")
        return errs

    # Implicit dependency injection via placeholders
    _PL_RE = re.compile(
        r"""
        \[            # opening [
        \s*<?\s*    # optional leading ‘<’
        (?P<id>[A-Za-z0-9_-]+)
        \s*>?\s*    # optional trailing ‘>’
        \.output    # literal
        (?:\.[A-Za-z0-9_.-]+)?   # optional attribute access
        \]            # closing ]
        |
        \{\{(?P<alias>[A-Za-z0-9_-]+)\}\}   # {{alias}}
        """,
        re.X,
    )
    def _inject_implicit_deps(graph: dict) -> None:
        nodes = graph.get("nodes") or []
        idset = {n["id"] for n in nodes if n.get("id")}
        alias2id = {(n.get("alias") or n["id"]): n["id"] for n in nodes if n.get("id")}
        for n in nodes:
            for _, v in (n.get("args") or {}).items():
                s = v if isinstance(v, str) else None
                if not s:
                    continue
                for m in _PL_RE.finditer(s):
                    ref = m.group(1) or m.group(3)
                    target = alias2id.get(ref) or (ref if ref in idset else None)
                    if target and target != n["id"]:
                        n.setdefault("after", [])
                        if target not in n["after"]:
                            n["after"].append(target)

    # Retry-hint helpers ------------------------------------------------
    def _load_retry_hints(tool_names: List[str]) -> Dict[str, List[str]]:
        """Return {tool_name: [hint_line, ...]} for confirmed/success retry critiques."""
        if not tool_names:
            return {}
        names = set(tool_names)
        rows = self.repo.query(lambda c:
            c.component == "analysis"
            and c.semantic_label == "tool_retry_critique"
            and isinstance((c.metadata or {}).get("tool_name"), str)
            and (c.metadata.get("status") in ("confirmed","success","refined"))
            and c.metadata.get("tool_name") in names
        )
        hints: Dict[str, List[str]] = {}
        for r in rows:
            tname = r.metadata.get("tool_name")
            text  = r.metadata.get("hint") or r.summary or ""
            if not text:
                req  = r.metadata.get("schema_required") or []
                ex   = r.metadata.get("filled_params") or {}
                text = f"When calling {tname}, include required {req}. Example keys used previously: {list(ex.keys())}."
            hints.setdefault(tname, [])
            if text not in hints[tname]:
                hints[tname].append(text)
        for k in list(hints.keys()):
            hints[k] = hints[k][:3]
        return hints

    def _persist_retry_candidate(tool_name: str, required: List[str], props: Dict[str, Any],
                                 filled: Dict[str, Any], turn_id: str, plan_id: str) -> None:
        """Save a candidate retry-critique row to be promoted later on success."""
        from context import ContextObject as _CO
        arg_list = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (filled or {}).items())
        req_list = ", ".join(required) if required else "(none)"
        hint = f"When using {tool_name}, supply required [{req_list}]. Example: {tool_name}({arg_list})"
        meta = {
            "tool_name": tool_name,
            "status": "refined",
            "schema_required": list(required or []),
            "schema_properties": list((props or {}).keys()),
            "filled_params": dict(filled or {}),
            "plan_id": plan_id,
            "turn_id": turn_id,
            "hint": hint,
        }
        crit = _CO.make_stage("tool_retry_critique", [], meta)
        crit.component = "analysis"
        crit.semantic_label = "tool_retry_critique"
        crit.tags = (crit.tags or []) + ["tool_retry", "candidate"]
        crit.summary = hint[:250]
        try:
            _stamp(crit, state)
        except Exception:
            pass
        crit.touch(); self.repo.save(crit)

    # ──────────────────────────────────────────────────────────────────
    # 0) Turn state + diagnostics
    # ──────────────────────────────────────────────────────────────────
    turn = _ensure_turn_state(state)

    incoming = {
        "turn_id":         turn.turn_id,
        "user_text":       user_text,
        "clarifier_notes": _clar_summary,
        "knowledge_snips": len((_know_summary or "").splitlines()),
        "merged_len":      len(state.get("merged", [])),
        "history_len":     len(state.get("history", [])),
        "available_tools": len(tools_list),
        "semantic_len":    len(state.get("semantic", [])),
        "assoc_len":       len(state.get("assoc", [])),
    }
    banner = "=" * 20 + " PLANNER INPUT " + "=" * 20
    print("\n" + banner)
    for k, v in incoming.items():
        print(f"{k:>15}: {v}")
    print("=" * len(banner) + "\n")
    self._print_stage_context("planning_summary_tools", {
        "tool_names": [t["name"] for t in tools_list]
    })
    self._print_stage_context("planning_summary_incoming", incoming)

    # ──────────────────────────────────────────────────────────────────
    # 1) Conversation so far (compact)
    # ──────────────────────────────────────────────────────────────────
    merged = state.get("merged", [])
    N = min(10, len(merged))
    convo_lines = [f"- {c.summary}" for c in merged[-N:]]
    convo_block = ("Conversation so far:\n" + "\n".join(convo_lines)) if convo_lines else ""

    # ──────────────────────────────────────────────────────────────────
    # 2) Critique + planner prompt
    # ──────────────────────────────────────────────────────────────────
    critique_rows = sorted(
        self.repo.query(lambda c: c.component == "analysis" and c.semantic_label == "plan_critique"),
        key=lambda c: c.timestamp,
    )
    critique_ids = [c.context_id for c in critique_rows]

    prompt_rows = sorted(
        self.repo.query(lambda c: c.component == "artifact" and c.semantic_label == "planning_prompt"),
        key=lambda c: c.timestamp,
        reverse=True,
    )
    raw_prompt = (prompt_rows[0].summary if prompt_rows else self._get_prompt("planning_prompt"))
    first_two  = ".".join(raw_prompt.split(".", 2)[:2]) + "."

    tool_lines = "\n".join(
        f"- **{t['name']}**: {_first_sentence(t.get('description',''))}"
        for t in tools_list
    ) or "(none)"

    base_system = f"{first_two}\n\nAvailable tools:\n{tool_lines}"
    replan_system_base = (
        "You MUST choose argument names only from the Allowed lists below. Do not invent synonyms.\n\n"
        "Your last plan may be incomplete—**OUTPUT ONLY** the JSON, no extra text.\n\n"
        f"Available tools:\n{tool_lines}"
    )

    print("\n" + "=" * 20 + " PLANNER PROMPT " + "=" * 20)
    print(base_system)

    # ──────────────────────────────────────────────────────────────────
    # 3) Build USER message to planner
    # ──────────────────────────────────────────────────────────────────
    original_snips = (_know_summary or "").splitlines()

    def build_user(snips: List[str]) -> str:
        blocks = [
            convo_block,
            f"User question:\n{user_text}",
            f"Clarified intent:\n{clar_ctx.metadata.get('notes') or clar_ctx.summary or '(none)'}",
            "Snippets:\n" + ("\n".join(snips) if snips else "(none)"),
        ]
        if state.get("plan_errors"):
            err_lines = "\n".join(f"- {e}" for e in state["plan_errors"])
            blocks.append("Previous validation errors:\n" + err_lines)
        recent = []
        for c in state.get("tools", [])[-5:]:
            ts = getattr(c, "timestamp", "")[:19].replace("T", " ")
            recent.append(f"- [{ts}] {c.stage_id}: {c.summary}")
        if recent:
            blocks.append("Recent tool outputs:\n" + "\n".join(recent))
        if state.get("plan_output_prev"):
            blocks.append("Previous plan:\n" + state["plan_output_prev"])
        if state.get("draft"):
            blocks.append("Assistant draft:\n" + state["draft"])
        return "\n\n".join([b for b in blocks if b])

    full_user = build_user(original_snips)

    # ──────────────────────────────────────────────────────────────────
    # 4) Planner passes: initial, then schema+hint replan
    # ──────────────────────────────────────────────────────────────────
    valid_tool_names = {t["name"] for t in tools_list}
    schema_map_all = {t["name"]: (t.get("schema") or {}) for t in tools_list}

    def _run_planner(sys_p: str, user_p: str, tag: str) -> dict:
        raw = self._stream_and_capture(
            self.secondary_model,
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            tag=tag,
            images=state.get("images"),
        ).strip()

        cleaned = _clean_json_block(raw)
        try:
            cand = json.loads(cleaned)
        except Exception:
            cand = None

        # normalize into {"tasks":[...]} or {"graph":{...}}
        if isinstance(cand, dict) and ("graph" in cand or "tasks" in cand):
            plan_obj = cand
        elif isinstance(cand, dict) and "tool_calls" in cand:
            def _name_args(tc):
                if isinstance(tc, str):
                    return tc, {}
                name = (
                    tc.get("call")
                    or tc.get("tool_call")
                    or tc.get("tool")
                    or tc.get("tool_name")
                    or tc.get("name")
                )
                args = (
                    tc.get("tool_input")
                    or tc.get("arguments")
                    or tc.get("args")
                    or {}
                )
                return name, args

            tasks = []
            for tc in cand["tool_calls"]:
                name, inp = _name_args(tc)
                if not name:
                    continue
                subs = (tc.get("subtasks") if isinstance(tc, dict) else None) or []
                tasks.append({"call": name, "tool_input": inp, "subtasks": subs})
            plan_obj = {"tasks": tasks}
        elif isinstance(cand, dict):
            plan_obj = {"tasks": [cand]}
        else:
            calls = re.findall(r"\b[A-Za-z_]\w*\([^)]*\)", cleaned or raw)
            plan_obj = {"tasks": [{"call": c, "tool_input": {}, "subtasks": []} for c in calls]}

        # prune unknown tools immediately (lets us replan cleanly)
        if isinstance(plan_obj, dict) and isinstance(plan_obj.get("tasks"), list):
            plan_obj["tasks"] = [t for t in plan_obj["tasks"] if t.get("call") in valid_tool_names]
        return plan_obj if isinstance(plan_obj, dict) else {"tasks": []}

    # First pass (no schemas)
    first_plan = _run_planner(base_system, full_user, tag="[Planner]")

    # Prepare second pass extras based on *selected* tools from first pass
    def _extract_calls(plan_obj: dict) -> List[str]:
        names = []
        if not isinstance(plan_obj, dict):
            return names
        if "graph" in plan_obj and isinstance(plan_obj["graph"], dict):
            for n in (plan_obj["graph"].get("nodes") or []):
                nm = (n or {}).get("tool")
                if nm:
                    names.append(nm)
        elif "tasks" in plan_obj and isinstance(plan_obj["tasks"], list):
            for t in plan_obj["tasks"]:
                nm = (t or {}).get("call")
                if nm:
                    names.append(nm)
        return names

    selected_names_1 = [n for n in _extract_calls(first_plan) if n in valid_tool_names]
    selected_schema_catalog = {n: schema_map_all.get(n, {}) for n in dict.fromkeys(selected_names_1)}
    retry_hints_map = _load_retry_hints(selected_names_1)

    # NEW: build allowed catalogs and recall bundles
    allowed_catalog = {n: _allowed_keys_for_tool(n, schema_map_all.get(n, {})) for n in selected_names_1}
    history_usage   = {n: _history_param_usage(n) for n in selected_names_1}
    mined_missing   = _mine_missing_from_errors(selected_names_1)

    # NEW: mine prior successes/failures tailored to this ask
    _seed = f"{user_text}\n{clar_ctx.metadata.get('notes') or clar_ctx.summary or ''}"
    success_examples = _load_success_examples(selected_names_1, seed_text=_seed)
    fail_patterns    = _aggregate_failures(selected_names_1)

    # Build replan system with schemas + strict key lists + hints + successes + failures
    replan_system = replan_system_base
    if selected_schema_catalog:
        replan_system += "\n\n[Selected Tool Schemas]\n" + json.dumps(selected_schema_catalog, ensure_ascii=False)

    if allowed_catalog:
        replan_system += "\n\n[Allowed Argument Keys]\n" + json.dumps(
            {n: {"required": v["required"], "optional": v["optional"]} for n, v in allowed_catalog.items()},
            ensure_ascii=False
        )
        replan_system += (
            "\n\nSTRICT RULES:\n"
            "• For each tool, you MUST use argument names only from its Allowed list.\n"
            "• Prefer REQUIRED names exactly; do NOT invent synonyms.\n"
            "• If a value is unknown, leave the key absent rather than inventing a key.\n"
        )

    if retry_hints_map:
        lines = []
        for nm, hints in retry_hints_map.items():
            for h in hints:
                lines.append(f"- ({nm}) {h}")
        if lines:
            replan_system += "\n\n[Retry Hints]\n" + "\n".join(lines[:12])

    if history_usage:
        replan_system += "\n\n[Historically Used Keys]\n" + json.dumps(history_usage, ensure_ascii=False)

    if mined_missing:
        replan_system += "\n\n[Recently Missing Keys]\n" + json.dumps(mined_missing, ensure_ascii=False)

    if success_examples:
        ex_lines = []
        for nm, exs in success_examples.items():
            for ex in exs:
                ex_lines.append(f"- ({nm}) {ex}")
        if ex_lines:
            replan_system += "\n\n[Success Examples]\n" + "\n".join(ex_lines[:12])

    if fail_patterns:
        fp_lines = []
        for nm, msgs in fail_patterns.items():
            for msg in msgs:
                fp_lines.append(f"- ({nm}) {msg}")
        if fp_lines:
            replan_system += "\n\n[Common Failure Patterns]\n" + "\n".join(fp_lines[:12])

    # Second pass (schema + strict keys + hints)
    half_snips = original_snips[: max(1, len(original_snips)//2)]
    second_plan = _run_planner(replan_system, build_user(half_snips), tag="[PlannerReplanSchemas]")

    # Choose better of the two: prefer second if valid & non-empty; else fallback
    def _is_valid_plan_obj(obj: dict) -> bool:
        if not isinstance(obj, dict):
            return False
        if "graph" in obj and isinstance(obj["graph"], dict):
            nodes = obj["graph"].get("nodes")
            return isinstance(nodes, list) and any(isinstance(n, dict) and n.get("tool") for n in nodes)
        if "tasks" in obj and isinstance(obj["tasks"], list):
            return any(isinstance(t, dict) and t.get("call") for t in obj["tasks"])
        return False

    plan_obj: dict = second_plan if _is_valid_plan_obj(second_plan) else first_plan
    if not _is_valid_plan_obj(plan_obj):
        tiny_snips = original_snips[:1]
        third_plan = _run_planner(replan_system, build_user(tiny_snips), tag="[PlannerReplanMin]")
        if _is_valid_plan_obj(third_plan):
            plan_obj = third_plan

    # ──────────────────────────────────────────────────────────────────
    # 5) Refine each chosen tool with schema (fill only missing)
    #    + strict normalization + persist retry-candidates
    # ──────────────────────────────────────────────────────────────────
    schema_map = schema_map_all  # already built

    def refine_single_tool(task: dict) -> dict:
        name   = task.get("call")
        schema = schema_map.get(name, {}) or {}
        req, props = _schema_req_props(schema)

        refined = {
            "call":       name,
            "tool_input": dict(task.get("tool_input", {}) or {}),
            "subtasks":   list(task.get("subtasks", []) or []),
        }

        # Normalize using dynamic allowed-catalog (schema + signature + history)
        norm_args, norm_notes = _normalize_args_dynamic(name, refined["tool_input"], allowed_catalog)
        refined["tool_input"] = norm_args

        # Compute truly-missing required keys after normalization
        missing = [k for k in req if k not in refined["tool_input"]]

        # If nothing missing, persist normalization note and return
        if not missing or not name:
            if norm_notes:
                _persist_retry_candidate(
                    tool_name=name, required=list(req), props=props, filled={},
                    turn_id=getattr(turn, "turn_id", ""), plan_id=getattr(turn, "plan_id", "")
                )
            return refined

        # Ask LLM to fill only the missing required keys, with STRICT constraints
        success_ex = (success_examples.get(name) or [])[:3]
        fail_ex    = (fail_patterns.get(name) or [])[:3]
        hist_names = history_usage.get(name, {})
        allowed_info = allowed_catalog.get(name, {})
        recent_missing = mined_missing.get(name, [])

        prompt = {
            "instruction": (
                "Fill values for the MISSING REQUIRED keys ONLY. "
                "Use EXACT key names from 'allowed.required'. "
                "Do NOT invent new keys; leave anything unknown absent."
            ),
            "call":        {"call": name, "tool_input": refined["tool_input"]},
            "missing":     missing,
            "schema":      schema,
            "allowed":     {"required": allowed_info.get("required", []), "optional": allowed_info.get("optional", [])},
            "history_keys": hist_names,
            "recent_missing": recent_missing,
            "success_examples": success_ex,
            "avoid_patterns":  fail_ex,
            "user_text":   user_text,
            "clar_notes":  (clar_ctx.metadata.get("notes") or clar_ctx.summary or ""),
        }

        out = self._stream_and_capture(
            self.secondary_model,
            [
                {"role": "system", "content":
                    "Return ONLY JSON: {\"call\":{\"tool_input\":{...}}}. "
                    "Keys MUST come from 'allowed.required'/'allowed.optional'. "
                    "If you cannot confidently fill a required value, omit it."
                },
                {"role": "user",   "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            tag=f"[PlannerRefine_{name}]",
            images=state.get("images"),
        ).strip()

        filled_now = {}
        try:
            cand = json.loads(_clean_json_block(out)).get("call", {})
            ti   = cand.get("tool_input", {}) or {}
            # accept only keys that exist in props (schema is the ground truth)
            for k, v in ti.items():
                if k in props:
                    refined["tool_input"][k] = v
                    filled_now[k] = v
        except Exception:
            pass

        _persist_retry_candidate(
            tool_name=name,
            required=list(req),
            props=props,
            filled=dict(filled_now),
            turn_id=getattr(turn, "turn_id", ""),
            plan_id=getattr(turn, "plan_id", ""),
        )

        return refined

    tasks_in = plan_obj.get("tasks", [])
    if isinstance(tasks_in, list) and tasks_in:
        with ThreadPoolExecutor(max_workers=len(tasks_in) or 1) as pool:
            futures = [pool.submit(refine_single_tool, t) for t in tasks_in]
            refined_list = [f.result() for f in as_completed(futures)]
        # preserve input order
        call2ref = {t.get("call"): t for t in refined_list if isinstance(t, dict)}
        plan_obj["tasks"] = [call2ref.get(t.get("call"), t) for t in tasks_in]
    else:
        plan_obj["tasks"] = []

    # ──────────────────────────────────────────────────────────────────
    # 6) Ensure DAG graph exists; convert tasks → graph if needed
    # ──────────────────────────────────────────────────────────────────
    def _flatten_with_edges(task: dict, parent_id: str | None, idx_seed: List[int], acc_nodes: list):
        """Produce node list with 'after' edges; depth-first numbering."""
        idx_seed[0] += 1
        node_id = task.get("id") or f"n{idx_seed[0]}"
        node = {
            "id":    node_id,
            "tool":  task.get("call"),
            "args":  task.get("tool_input", {}) or {},
            "after": [parent_id] if parent_id else [],
        }
        # Preserve alias/retry/timeout if already in task
        for k in ("alias", "retries", "timeout_s"):
            if k in task:
                node[k] = task[k]
        acc_nodes.append(node)
        for sub in (task.get("subtasks") or []):
            _flatten_with_edges(sub, node_id, idx_seed, acc_nodes)

    if "graph" not in plan_obj:
        nodes: list = []
        counter = [0]
        for t in (plan_obj.get("tasks") or []):
            if not isinstance(t, dict) or not t.get("call"):
                continue
            _flatten_with_edges(t, None, counter, nodes)

        # If no explicit dependencies and multiple top-level tasks, chain them
        if nodes:
            tops = [n for n in nodes if not n.get("after")]
            if len(tops) > 1:
                for i in range(1, len(tops)):
                    tops[i].setdefault("after", []).append(tops[i-1]["id"])

        plan_obj = {
            "graph": {
                "nodes": nodes,
                "meta": {
                    "goal": clar_ctx.summary or user_text,
                    "created_by": "planner",
                },
            }
        }

    # Normalize graph + meta defaults (parallelism/retries/timeout)
    graph = plan_obj.get("graph", {"nodes": [], "meta": {}})
    graph.setdefault("nodes", [])
    meta = graph.setdefault("meta", {})
    default_parallelism = int(getattr(turn, "budgets", {}).get("parallelism", 2))
    meta.setdefault("parallelism", default_parallelism)
    meta.setdefault("retries", 0)
    meta.setdefault("timeout_s", 0.0)

    # Ensure per-node fields and inject implicit deps from placeholders
    for n in graph["nodes"]:
        n.setdefault("args", {})
        n.setdefault("after", [])
        n.setdefault("retries", meta.get("retries", 0))
        n.setdefault("timeout_s", meta.get("timeout_s", 0.0))

    _inject_implicit_deps(graph)
    errs = _validate_graph(graph, tools_list)
    if errs:
        state["plan_errors"] = errs
        err_ctx = ContextObject.make_failure(
            description="plan graph validation errors",
            refs=[clar_ctx.context_id, know_ctx.context_id],
        )
        err_ctx.summary = "; ".join(errs)
        _stamp(err_ctx := err_ctx, state)
        err_ctx.touch(); self.repo.save(err_ctx)

    # Recompute a stable plan signature from the (possibly adjusted) graph
    plan_json_sorted = json.dumps(graph, ensure_ascii=False, sort_keys=True)
    plan_sig = hashlib.md5(plan_json_sorted.encode("utf-8")).hexdigest()[:8]

    # ──────────────────────────────────────────────────────────────────
    # 7) Initialize TurnState for this turn
    # ──────────────────────────────────────────────────────────────────
    turn.plan_id         = f"plan_{plan_sig}"
    turn.graph           = graph
    turn.completed_nodes = set()
    turn.pending_nodes   = {n["id"] for n in graph.get("nodes", []) if n.get("id")}
    turn.tool_ctx_ids    = []
    # budgets kept from turn.budgets or state override

    # ──────────────────────────────────────────────────────────────────
    # 8) Persist artefacts & plan_tracker
    # ──────────────────────────────────────────────────────────────────
    plan_json_out = json.dumps({"graph": graph}, ensure_ascii=False)

    # Planning summary ctx
    ctx = ContextObject.make_stage(
        "planning_summary",
        clar_ctx.references + know_ctx.references + critique_ids,
        {"graph": graph, "plan_id": turn.plan_id, "turn_id": turn.turn_id},
    )
    ctx.stage_id = f"planning_summary_{plan_sig}"
    ctx.summary  = plan_json_out
    ctx.metadata.update({"plan_id": turn.plan_id, "turn_id": turn.turn_id})
    _stamp(ctx, state)
    ctx.touch(); self.repo.save(ctx)

    # Success/failure signal
    succ_cls = ContextObject.make_success if graph.get("nodes") and not state.get("plan_errors") else ContextObject.make_failure
    succ_msg = (
        f"Planner → {len(graph.get('nodes', []))} DAG node(s)"
        if graph.get("nodes") else "Planner → empty graph"
    )
    if state.get("plan_errors"):
        succ_msg += f" (validation errors: {len(state['plan_errors'])})"
    succ = succ_cls(succ_msg, refs=[ctx.context_id])
    succ.stage_id = f"planning_summary_signal_{plan_sig}"
    _stamp(succ, state)
    succ.touch(); self.repo.save(succ)

    # Plan tracker
    tracker = ContextObject.make_stage(
        "plan_tracker",
        [ctx.context_id],
        {
            "plan_id":       turn.plan_id,
            "turn_id":       turn.turn_id,
            "total_nodes":   len(graph.get("nodes", [])),
            "pending_nodes": list(turn.pending_nodes),
            "completed":     [],
            "attempts":      0,
            "status":        "in_progress" if not state.get("plan_errors") else "needs_fix",
            "started_at":    datetime.datetime.utcnow().isoformat() + "Z",
            # executor hints copied here for convenience
            "parallelism":   meta.get("parallelism"),
            "default_retries": int(meta.get("retries", 0)),
            "default_timeout_s": float(meta.get("timeout_s", 0.0)),
            "validation_errors": state.get("plan_errors", []),
        },
    )
    tracker.semantic_label = plan_sig
    tracker.stage_id       = f"plan_tracker_{plan_sig}"
    tracker.summary        = "initialized plan tracker"
    _stamp(tracker, state)
    tracker.touch(); self.repo.save(tracker)

    # ──────────────────────────────────────────────────────────────────
    # 9) Expose to downstream
    # ──────────────────────────────────────────────────────────────────
    state["plan_ctx"]         = ctx
    state["plan_output"]      = {"graph": graph}
    state["tools_list"]       = tools_list
    state["tc_ctx"]           = None
    state["plan_output_prev"] = json.dumps(first_plan, ensure_ascii=False)

    # Expose recall bundle for later stages (retries/reflection/executor)
    state["retry_bundle"] = {
        "hints":       retry_hints_map,
        "success":     success_examples,
        "failures":    fail_patterns,
        "allowed":     allowed_catalog,
        "hist_keys":   history_usage,
        "miss_keys":   mined_missing,
    }

    return ctx, plan_json_out


def _stage7b_plan_validation(
    self,
    plan_ctx: ContextObject,
    plan_output: str,
    tools_list: List[Dict[str, str]],
    state: Dict[str, Any]
) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """
    DAG-aware plan validation & light repair.

    • Accepts either {"graph":{...}} or legacy {"tasks":[...]} and normalizes to a graph.
    • Injects implicit deps from placeholders like [n1.output.foo] or {{alias}}.
    • Validates: unique node ids, after-refs exist, acyclic, tool exists.
    • Checks required parameters against tool schemas; up to 3 LLM repair passes
      to fill ONLY truly missing required args (preserving everything else).
    • Persists a 'plan_validation' context with results and errors.

    Returns:
        (fixed_calls_for_display, errors_by_node, fixed_calls_for_display)

    NOTE: In DAG execution we don't *need* the call strings, but they are useful
    for UX / audit trails and are consumed by downstream tooling that expects them.
    """
    import json, re, inspect, importlib

    from context import ContextObject
    from tools import Tools

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def _clean_json_block(text: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text or "", flags=re.S)
        if m:
            return m.group(1)
        m2 = re.search(r"(\{.*\})", text or "", flags=re.S)
        return (m2.group(1) if m2 else (text or "")).strip()

    def _flatten_with_edges(task: dict, parent_id: str | None, idx_seed: list[int], acc_nodes: list):
        idx_seed[0] += 1
        node_id = task.get("id") or f"n{idx_seed[0]}"
        node = {
            "id":    node_id,
            "tool":  task.get("call"),
            "args":  task.get("tool_input", {}) or {},
            "after": [parent_id] if parent_id else [],
        }
        for k in ("alias", "retries", "timeout_s"):
            if k in task:
                node[k] = task[k]
        acc_nodes.append(node)
        for sub in task.get("subtasks", []) or []:
            _flatten_with_edges(sub, node_id, idx_seed, acc_nodes)

    # prefer class-level helpers if present (as recommended), else local fallbacks
    _validator = getattr(self, "_validate_graph", None)
    _injector  = getattr(self, "_inject_implicit_deps", None)

    def _validate_graph_local(graph: dict, tools_list: list[dict]) -> list[str]:
        from collections import deque, defaultdict
        errs = []
        nodes = graph.get("nodes") or []
        ids = [n.get("id") for n in nodes if n.get("id")]
        if len(ids) != len(set(ids)):
            errs.append("duplicate node ids")
        idset = set(ids)
        for n in nodes:
            for dep in (n.get("after") or []):
                if dep not in idset:
                    errs.append(f"node {n.get('id')} after-> {dep} missing")
        known = {t["name"] for t in tools_list}
        for n in nodes:
            if n.get("tool") not in known:
                errs.append(f"unknown tool '{n.get('tool')}' in node {n.get('id')}")
        indeg = defaultdict(int); g = defaultdict(list)
        for n in nodes:
            for dep in (n.get("after") or []):
                g[dep].append(n["id"]); indeg[n["id"]] += 1
        q = deque([i for i in ids if indeg[i]==0]); seen = 0
        while q:
            u = q.popleft(); seen += 1
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        if seen != len(ids):
            errs.append("cycle detected")
        return errs

    _PL_RE = re.compile(r"\[([A-Za-z0-9_-]+)\.output(?:\.([A-Za-z0-9_.-]+))?\]|\{\{([A-Za-z0-9_-]+)\}\}")
    def _inject_implicit_deps_local(graph: dict) -> None:
        nodes = graph.get("nodes") or []
        idset = {n["id"] for n in nodes if n.get("id")}
        alias2id = {(n.get("alias") or n["id"]): n["id"] for n in nodes if n.get("id")}
        for n in nodes:
            for _, v in (n.get("args") or {}).items():
                s = v if isinstance(v, str) else None
                if not s: continue
                for m in _PL_RE.finditer(s):
                    ref = m.group(1) or m.group(3)
                    target = alias2id.get(ref) or (ref if ref in idset else None)
                    if target and target != n["id"]:
                        n.setdefault("after", [])
                        if target not in n["after"]:
                            n["after"].append(target)

    def _ensure_graph(plan_any: str | dict) -> dict:
        """Return {'graph': {'nodes': [...], 'meta': {...}}} from various shapes."""
        plan_obj: dict = {}
        if isinstance(plan_any, dict):
            plan_obj = plan_any
        else:
            try:
                plan_obj = json.loads(_clean_json_block(plan_any))
            except Exception:
                plan_obj = {}

        # Already a graph
        if "graph" in plan_obj and isinstance(plan_obj["graph"], dict):
            g = plan_obj["graph"]
            g.setdefault("nodes", []); g.setdefault("meta", {})
            for n in g["nodes"]:
                n.setdefault("args", {}); n.setdefault("after", [])
            return {"graph": g}

        # Helper to convert a single call object to a task dict
        def _to_task(tc: dict | str) -> dict:
            if isinstance(tc, str):
                return {"call": tc, "tool_input": {}, "subtasks": []}
            name = (
                tc.get("call")
                or tc.get("tool_call")
                or tc.get("tool")
                or tc.get("tool_name")
                or tc.get("name")
            )
            args = (
                tc.get("tool_input")
                or tc.get("arguments")
                or tc.get("args")
                or {}
            )
            subs = tc.get("subtasks", []) or []
            return {"call": name, "tool_input": args, "subtasks": subs}

        # Accept legacy shapes
        tasks = []
        if isinstance(plan_obj.get("tasks"), list):
            tasks = [ _to_task(t) for t in plan_obj["tasks"] ]
        elif isinstance(plan_obj.get("tool_calls"), list):
            tasks = [ _to_task(t) for t in plan_obj["tool_calls"] ]

        # Flatten tasks -> nodes (depth-first), chaining top-level if needed
        def _flatten_with_edges(task: dict, parent_id: str | None, idx_seed: list[int], acc_nodes: list):
            idx_seed[0] += 1
            node_id = task.get("id") or f"n{idx_seed[0]}"
            node = {
                "id":    node_id,
                "tool":  task.get("call"),
                "args":  task.get("tool_input", {}) or {},
                "after": [parent_id] if parent_id else [],
            }
            for k in ("alias", "retries", "timeout_s"):
                if k in task:
                    node[k] = task[k]
            acc_nodes.append(node)
            for sub in task.get("subtasks", []) or []:
                _flatten_with_edges(sub, node_id, idx_seed, acc_nodes)

        nodes: list[dict] = []
        counter = [0]
        for t in tasks or []:
            # Skip nameless tasks (prevents None())
            if not t.get("call"):
                continue
            _flatten_with_edges(t, None, counter, nodes)

        # If multiple roots without deps, chain to preserve order
        tops = [n for n in nodes if not n.get("after")]
        if len(tops) > 1:
            for i in range(1, len(tops)):
                tops[i].setdefault("after", []).append(tops[i-1]["id"])

        return {"graph": {"nodes": nodes, "meta": plan_obj.get("meta", {}) or {}}}

    # ──────────────────────────────────────────────────────────────────
    # 1) Normalize plan → graph
    # ──────────────────────────────────────────────────────────────────
    plan_norm = _ensure_graph(plan_output)
    graph     = plan_norm["graph"]
    graph.setdefault("meta", {})
    graph["meta"].setdefault("retries", 0)
    graph["meta"].setdefault("timeout_s", 0.0)

    # Ensure node defaults
    for n in graph.get("nodes", []):
        n.setdefault("args", {})
        n.setdefault("after", [])
        n.setdefault("retries", graph["meta"].get("retries", 0))
        n.setdefault("timeout_s", graph["meta"].get("timeout_s", 0.0))

    # Inject implicit deps then validate
    (_injector or _inject_implicit_deps_local)(graph)
    errs = (_validator or _validate_graph_local)(graph, tools_list)
    state["plan_errors"] = errs[:]  # copy

    # ──────────────────────────────────────────────────────────────────
    # 2) Schema collection and docstring enrichment
    # ──────────────────────────────────────────────────────────────────
    # Load all known tool schemas from repo
    try:
        all_schema_ctxs = {
            json.loads(c.metadata["schema"])["name"]: c
            for c in self.repo.query(lambda c: c.component == "schema" and "tool_schema" in (c.tags or []))
        }
        all_schemas = {k: json.loads(v.metadata["schema"]) for k, v in all_schema_ctxs.items()}
    except Exception:
        all_schema_ctxs, all_schemas = {}, {}

    # Only for tools used in this graph
    used_tools = sorted({n.get("tool") for n in graph.get("nodes", []) if n.get("tool")})
    schemas_for_prompt = {name: all_schemas[name] for name in used_tools if name in all_schemas}

    # Enrich description with full docstrings where available
    for name, schema in schemas_for_prompt.items():
        doc = None
        if hasattr(Tools, name):
            doc = inspect.getdoc(getattr(Tools, name))
        else:
            try:
                mod = importlib.import_module("tools")
                if hasattr(mod, name):
                    doc = inspect.getdoc(getattr(mod, name))
            except ImportError:
                pass
        if doc:
            schema["description"] = doc

    # ──────────────────────────────────────────────────────────────────
    # 3) Up to 3 repair passes for missing required args per node
    # ──────────────────────────────────────────────────────────────────
    missing_by_node: dict[str, list[str]] = {}

    def _scan_missing():
        missing_by_node.clear()
        for n in graph.get("nodes", []):
            name = n.get("tool"); schema = schemas_for_prompt.get(name)
            if not schema:
                continue
            req = set(schema.get("parameters", {}).get("required", []) or [])
            found = set((n.get("args") or {}).keys())
            miss = list(req - found)
            if miss:
                missing_by_node[n["id"]] = sorted(miss)

    _scan_missing()
    for _ in range(3):
        if not missing_by_node:
            break
        prompt = {
            "description": "Some DAG nodes are missing required tool parameters. "
                           "Fill only the truly missing keys in each node's args. Do NOT invent extra keys.",
            "missing_by_node": missing_by_node,
            "graph": graph,
            "schemas": schemas_for_prompt,
        }
        repair_raw = self._stream_and_capture(
            self.secondary_model,
            [
                {"role":"system","content": "Return ONLY JSON in one of these forms:\n"
                                            "1) {\"graph\": {\"nodes\":[...]}}\n"
                                            "2) {\"nodes\":[...]}\n"
                                            "3) {\"repairs\": {\"<node_id>\": {\"key\": <value>, ...}, ...}}"},
                {"role":"user","content": json.dumps(prompt, ensure_ascii=False)}
            ],
            tag="[PlanFix_DAG]",
            images=state.get("images", None)
        ).strip()

        # Accept several shapes
        applied = False
        try:
            rj = json.loads(_clean_json_block(repair_raw))
            if isinstance(rj, dict):
                if "graph" in rj and isinstance(rj["graph"], dict) and isinstance(rj["graph"].get("nodes"), list):
                    # adopt full nodes (preserve meta)
                    graph["nodes"] = rj["graph"]["nodes"]
                    applied = True
                elif "nodes" in rj and isinstance(rj["nodes"], list):
                    graph["nodes"] = rj["nodes"]
                    applied = True
                elif "repairs" in rj and isinstance(rj["repairs"], dict):
                    nid2node = {n["id"]: n for n in graph.get("nodes", []) if n.get("id")}
                    for nid, patch in rj["repairs"].items():
                        if nid in nid2node and isinstance(patch, dict):
                            nid2node[nid].setdefault("args", {}).update(patch)
                    applied = True
        except Exception:
            applied = False

        if applied:
            # Re-ensure defaults and implicit deps after repair
            for n in graph.get("nodes", []):
                n.setdefault("args", {})
                n.setdefault("after", [])
                n.setdefault("retries", graph["meta"].get("retries", 0))
                n.setdefault("timeout_s", graph["meta"].get("timeout_s", 0.0))
            (_injector or _inject_implicit_deps_local)(graph)
            _scan_missing()
        else:
            break

    # Merge validation errors with any remaining missing args
    err_pairs: list[Tuple[str, str]] = []
    for e in errs:
        err_pairs.append(("graph", e))
    for nid, miss in missing_by_node.items():
        err_pairs.append((nid, f"missing required: {', '.join(miss)}"))

    # ──────────────────────────────────────────────────────────────────
    # 4) Build display call-strings (for audit / confirmation UI)
    # ──────────────────────────────────────────────────────────────────
    def _json_args(args: dict) -> str:
        if not args:
            return ""
        return ",".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())

    fixed_calls: list[str] = []
    for n in graph.get("nodes", []):
        name = (n.get("tool") or "").split("(", 1)[0]
        arg_s = _json_args(n.get("args", {}))
        fixed_calls.append(f"{name}({arg_s})" if arg_s else f"{name}()")

    # ──────────────────────────────────────────────────────────────────
    # 5) Persist validation context
    # ──────────────────────────────────────────────────────────────────
    meta = {
        "valid_calls": fixed_calls,
        "errors": err_pairs,
        "graph_nodes": len(graph.get("nodes", [])),
        "repaired_nodes": [nid for nid, _ in err_pairs if nid != "graph"],  # nodes still with issues will appear here
        "missing_by_node": missing_by_node,
    }
    pv_ctx = ContextObject.make_stage("plan_validation", plan_ctx.references, meta)
    pv_ctx.stage_id = "plan_validation"
    pv_ctx.summary  = "OK" if not err_pairs else f"Issues: {len(err_pairs)}"
    self._persist_and_index([pv_ctx])
    self._print_stage_context("plan_validation", meta)

    # Expose (possibly updated) normalized plan to downstream as text
    state["plan_output"] = {"graph": graph}

    return fixed_calls, err_pairs, fixed_calls


# ================================
#  High-Observability Orchestrator (additive)
# ================================
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json, os, re, uuid, hashlib
from context import ContextObject

# ---------- Standard artifact envelope ----------

@dataclass
class Artifact:
    kind: str                 # "text" | "list" | "json" | "table" | "file" | "blob" | "none"
    data: Any
    mime: str = ""
    uri: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "data": self.data,
            "mime": self.mime,
            "uri": self.uri,
            "meta": self.meta,
        }

def _coerce_artifact(val: Any, *, default_kind: str = "json") -> Artifact:
    if val is None:
        return Artifact("none", None)
    if isinstance(val, Artifact):
        return val
    if isinstance(val, str):
        # If it smells like a path that exists, call it a file
        if os.path.exists(val) and (os.path.isfile(val) or os.path.isdir(val)):
            return Artifact("file", {"path": val}, uri=val, mime="", meta={"exists": True})
        return Artifact("text", val, mime="text/plain; charset=utf-8")
    if isinstance(val, (list, tuple)):
        return Artifact("list", list(val), mime="application/json")
    if isinstance(val, dict):
        return Artifact("json", val, mime="application/json")
    if isinstance(val, (bytes, bytearray)):
        return Artifact("blob", f"<{len(val)} bytes>", mime="application/octet-stream")
    # Fallback: repr
    return Artifact(default_kind, json.loads(json.dumps(val, default=str)), mime="application/json")

# ---------- Lightweight ArtifactBus for node-to-node passing ----------

class ArtifactBus:
    def __init__(self) -> None:
        self._store: Dict[str, Artifact] = {}

    def put(self, key: str, art: Artifact) -> None:
        self._store[key] = art

    def get(self, key: str, path: Optional[str] = None) -> Any:
        art = self._store.get(key)
        if not art:
            return None
        if not path:
            return art.data
        # simple dotted-attr / dict path
        cur = art.data
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

# ---------- Orchestration events (observability) ----------

def _emit_event(self, kind: str, data: Dict[str, Any]) -> None:
    """
    Persist a lightweight orchestration event row.
    """
    try:
        evt = ContextObject.make_stage("orchestration_event", [], {"event": kind, "data": data})
        evt.stage_id = f"evt_{kind}"
        evt.summary = f"{kind}: {str(data)[:220]}"
        # inherit conv/user if we have it
        try:
            evt.metadata["conversation_id"] = data.get("conversation_id") or getattr(self, "_active_conversation_id", None)
            evt.metadata["user_id"] = data.get("user_id") or getattr(self, "current_user_id", None)
        except Exception:
            pass
        evt.touch()
        self.repo.save(evt)
    except Exception:
        pass

# ---------- Strategy templates / expander for abstract 'action:*' nodes ----------

class StrategyTemplates:
    """
    Expands abstract nodes like:
       {"id":"task1","tool":"action:web.research","args":{"query":"...", "folder":"./out","filename":"notes.md"}}
    into a concrete DAG using existing tools discovered from schemas.

    It dynamically resolves tool names from your active tool schemas, preferring
    the first match among candidate name lists.
    """
    WEB_SEARCH_CANDIDATES   = ["web_search", "search_web", "internet_search", "bing_search", "ddg_search", "duckduckgo_search", "search"]
    REFINE_LIST_CANDIDATES  = ["refine_list", "summarize_list", "list_summarize", "summarize_text", "llm_summarize", "extract_top"]
    FILE_WRITE_CANDIDATES   = ["write_file", "file_write", "save_text", "fs_write", "persist_text", "save_file"]

    def __init__(self, host) -> None:
        self.host = host

    def _resolve_tool(self, tools_list: List[Dict[str, Any]], candidates: List[str]) -> Optional[str]:
        names = {t["name"]: t for t in tools_list or []}
        # direct match first
        for c in candidates:
            if c in names:
                return c
        # relaxed contains / fuzzy
        lowered = {k.lower(): k for k in names.keys()}
        for c in candidates:
            if c.lower() in lowered:
                return lowered[c.lower()]
        # fallback: substring scan
        for k in names.keys():
            if any(c.lower() in k.lower() for c in candidates):
                return k
        return None

    @staticmethod
    def _pick_key(props: Dict[str, Any], *candidates: str, default: Optional[str] = None) -> Optional[str]:
        pset = set(props.keys())
        for c in candidates:
            if c in pset:
                return c
        # relaxed fallbacks
        for c in candidates:
            for p in pset:
                if p.lower() == c.lower():
                    return p
        return default

    def expand_node(self, node: Dict[str, Any], tools_list: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        tool = (node.get("tool") or "").strip()
        if not tool.startswith("action:"):
            return None

        action = tool.split(":", 1)[1]
        base_id = node["id"]
        args    = node.get("args", {}) or {}

        # Resolve concrete tools
        web_t   = self._resolve_tool(tools_list, self.WEB_SEARCH_CANDIDATES)
        ref_t   = self._resolve_tool(tools_list, self.REFINE_LIST_CANDIDATES)
        file_t  = self._resolve_tool(tools_list, self.FILE_WRITE_CANDIDATES)

        if action == "web.research" and web_t and ref_t and file_t:
            # Pull schemas to adapt arg names to what your tools expect
            schema_map = {t["name"]: (t.get("schema") or {}) for t in tools_list}
            web_props  = (schema_map.get(web_t, {}).get("parameters", {}) or {}).get("properties", {}) or {}
            ref_props  = (schema_map.get(ref_t, {}).get("parameters", {}) or {}).get("properties", {}) or {}
            file_props = (schema_map.get(file_t, {}).get("parameters", {}) or {}).get("properties", {}) or {}

            # Pick canonical keys per tool
            web_query_key  = self._pick_key(web_props, "query", "q", "search", default="query")
            web_k_key      = self._pick_key(web_props, "top_k", "k", "limit", "n", default="top_k")

            ref_in_key     = self._pick_key(ref_props, "items", "results", "texts", "input", "text", default="items")
            ref_goal_key   = self._pick_key(ref_props, "goal", "instruction", "prompt", default="goal")

            file_path_key  = self._pick_key(file_props, "path", "filepath", "file_path", default="path")
            file_text_key  = self._pick_key(file_props, "text", "content", "data", default="text")

            # Build the micro-DAG
            n1 = {
                "id": f"{base_id}.search",
                "tool": web_t,
                "args": {
                    web_query_key: args.get("query") or args.get("q") or "",
                    web_k_key: int(args.get("top_k", args.get("k", 10))),
                },
                "after": list(node.get("after", []) or []),
            }
            n2 = {
                "id": f"{base_id}.refine",
                "tool": ref_t,
                "args": {
                    ref_in_key: f"[{n1['id']}.output]",  # pass raw list/results forward
                    ref_goal_key: args.get("goal") or "Condense, deduplicate, and produce a clean bullet list with links.",
                },
                "after": [n1["id"]],
            }
            target_folder = args.get("folder") or args.get("dir") or "./out"
            target_name   = args.get("filename") or "notes.md"
            target_path   = os.path.join(target_folder, target_name)
            n3 = {
                "id": f"{base_id}.persist",
                "tool": file_t,
                "args": {
                    file_path_key: target_path,
                    file_text_key: f"[{n2['id']}.output]",
                },
                "after": [n2["id"]],
            }
            return [n1, n2, n3]

        # Unknown action or missing tools -> leave as-is (no expansion)
        return None

def _stage7c_expand_strategy_graph(self, graph: Dict[str, Any], tools_list: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk a graph produced by the planner and expand any abstract 'action:*' nodes
    into concrete tool nodes using StrategyTemplates. Maintains dependencies.
    """
    tmpl = StrategyTemplates(self)
    nodes: List[Dict[str, Any]] = list(graph.get("nodes", []))
    if not nodes:
        return graph

    id_to_idx = {n["id"]: i for i, n in enumerate(nodes) if n.get("id")}
    expanded: List[Dict[str, Any]] = []
    replaced: Dict[str, str] = {}  # original_id -> last_subnode_id

    for n in nodes:
        out = tmpl.expand_node(n, tools_list)
        if not out:
            expanded.append(n)
            continue
        # Remember last subnode to reroute downstream deps
        last = out[-1]["id"]
        replaced[n["id"]] = last
        expanded.extend(out)

    # If nothing expanded, return original
    if not replaced:
        return {**graph, "nodes": nodes}

    # Reroute any 'after' edges that pointed at originals -> last subnode of expansion
    for n in expanded:
        af = n.get("after", []) or []
        new_after = []
        for dep in af:
            new_after.append(replaced.get(dep, dep))
        n["after"] = list(dict.fromkeys(new_after))  # de-dupe preserve order

    out_graph = {**graph, "nodes": expanded}
    # Persist an event row for observability
    try:
        _emit_event(self, "plan_expanded", {
            "replaced": replaced,
            "total_nodes": len(nodes),
            "expanded_nodes": sum(1 for _ in replaced),
        })
    except Exception:
        pass
    return out_graph

# ---------- Critic / Planner / Executor adapters (plug into existing pipeline) ----------

class HighObsCritic:
    def __init__(self, host) -> None:
        self.host = host

    def preflight(self, graph: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """
        Validate the graph against tool availability + required params.
        Reuses your Stage 7b validator for consistency.
        """
        try:
            # Make a lightweight plan_ctx to satisfy stage signature
            dummy_plan_ctx = ContextObject.make_stage("preflight_plan_ctx", [], {"graph": graph})
            fixed, errs, _ = self.host._stage7b_plan_validation(
                plan_ctx=dummy_plan_ctx,
                plan_output={"graph": graph},
                tools_list=context.get("tools", []),
                state=self.host._last_state or {}
            )
            return [f"{nid}: {msg}" for (nid, msg) in errs]
        except Exception as e:
            return [f"preflight_error: {e}"]

    def postcheck(self, tool_ctxs: List["ContextObject"], graph: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """
        Optional postconditions: if a persist node exists, ensure it succeeded, etc.
        """
        issues: List[str] = []
        tmap = {c.metadata.get("node_id"): c for c in tool_ctxs if isinstance(c.metadata, dict)}
        for n in graph.get("nodes", []):
            if not n.get("id"):
                continue
            if ".persist" in n["id"]:
                tc = tmap.get(n["id"])
                if not tc or (tc.metadata.get("exception") is not None):
                    issues.append(f"persist_failed:{n['id']}")
        return issues

class HighObsPlannerCore:
    """
    Uses your existing Stage 7 planner, then runs expansion templates.
    """
    def __init__(self, host) -> None:
        self.host = host

    def propose(self, context: Dict[str, Any], k: int = 1) -> List[Dict[str, Any]]:
        state = self.host._last_state or {}
        clar  = context.get("clar")
        know  = context.get("knowledge")
        tools = context.get("tools") or []
        user_text = context.get("user_text", "")

        # Run your Stage 7 planner (returns ctx + JSON string)
        plan_ctx, plan_json = self.host._stage7_planning_summary(
            clar_ctx=clar, know_ctx=know, tools_list=tools, user_text=user_text, state=state
        )
        try:
            obj = json.loads(plan_json)
        except Exception:
            obj = {"graph": (context.get("graph") or {"nodes": [], "meta": {}})}

        # Expand abstract actions into concrete tool nodes
        graph0 = obj.get("graph") or {"nodes": [], "meta": {}}
        graph1 = _stage7c_expand_strategy_graph(self.host, graph0, tools, state)

        _emit_event(self.host, "plan_proposed", {
            "nodes": len(graph1.get("nodes", [])),
            "from_expansion": True,
        })

        # We return one best plan; if more are requested, duplicate (score/policy can refine later)
        return [graph1] + [graph1 for _ in range(max(0, (k or 1) - 1))]

class HighObsExecutorCore:
    """
    Wraps your Stage 9 DAG executor, emits events, and assembles a reply via your Stage 10.
    """
    def __init__(self, host) -> None:
        self.host = host

    def run(self, graph: Dict[str, Any], context: Dict[str, Any], critic: Optional[HighObsCritic] = None, tool_registry=None) -> Dict[str, Any]:
        state = self.host._last_state or {}
        user_text = context.get("user_text", "")
        clar      = context.get("clar") or {}
        issues_pre = critic.preflight(graph, context) if critic else []
        if issues_pre:
            _emit_event(self.host, "preflight_issues", {"count": len(issues_pre), "examples": issues_pre[:3]})

        plan_json = json.dumps({"graph": graph}, ensure_ascii=False)
        # Execute
        tool_ctxs = self.host._stage9_invoke_with_retries(
            raw_calls=[],
            plan_output=plan_json,
            selected_schemas=[],
            user_text=user_text,
            clar_metadata=getattr(clar, "metadata", {}) if clar else {},
            state=state,
        )
        _emit_event(self.host, "execution_finished", {"tool_ctxs": len(tool_ctxs)})

        # Postcheck
        issues_post = critic.postcheck(tool_ctxs, graph, context) if critic else []
        if issues_post:
            _emit_event(self.host, "postcheck_issues", {"count": len(issues_post), "examples": issues_post[:3]})

        # Build reply with your Stage 10/10b polish if available
        reply = self.host._stage10_assemble_and_infer(user_text=user_text, state=state)
        try:
            polished = self.host._stage10b_response_critique_and_safety(
                draft=reply, user_text=user_text, tool_ctxs=tool_ctxs, state=state
            )
            reply = polished or reply
        except Exception:
            pass

        # Memory writeback
        try:
            self.host._stage11_memory_writeback(final_answer=reply, tool_ctxs=tool_ctxs)
        except Exception:
            pass

        return {
            "reply": reply,
            "tool_ctxs": tool_ctxs,
            "graph": graph,
            "repairs": 0,
            "issues": issues_pre + issues_post,
        }

# ---------- Auto-instantiation helper (call once during assembler boot) ----------

def _ensure_orchestrator_attached(self) -> None:
    """
    Attach planner_core / executor_core / critic if missing.
    Safe to call repeatedly.
    """
    if not getattr(self, "planner_core", None):
        self.planner_core = HighObsPlannerCore(self)
    if not getattr(self, "executor_core", None):
        self.executor_core = HighObsExecutorCore(self)
    if not getattr(self, "critic", None):
        self.critic = HighObsCritic(self)



def _stage8_orchestrate(
    self,
    user_text: str,
    state: Dict[str, Any],
) -> Tuple[str, List["ContextObject"]]:
    """
    Top-level orchestrator: plan → validate → chain → execute (DAG) → reflect/replan → finalize.

    This **upgrades** existing Stage 8 to:
      • Always generate a DAG via Stage 7 (planner)
      • Validate/repair via Stage 7b (schema-aware)
      • Summarize calls + collect schema ctxs via Stage 8 (tool chaining)
      • Execute with Stage 9 (DAG executor), reflecting with Stage 9b
      • Finalize with Stage 10/10b and write-back with Stage 11
    """
    # Ensure IDs for stamping
    state.setdefault("conversation_id", getattr(self, "_active_conversation_id", uuid.uuid4().hex))
    state.setdefault("user_id", getattr(self, "current_user_id", "anon"))

    # ─── 1) Planner → DAG ───────────────────────────────────────────
    clar_ctx = state["clar_ctx"]
    know_ctx = state["know_ctx"]
    tools    = state.get("tools_list", []) or []

    plan_ctx, plan_json = self._stage7_planning_summary(
        clar_ctx   = clar_ctx,
        know_ctx   = know_ctx,
        tools_list = tools,
        user_text  = user_text,
        state      = state,
    )
    state["plan_ctx"]    = plan_ctx
    state["plan_output"] = plan_json

    # ─── 2) Validate/repair + tool-chaining (for schema refs) ───────
    try:
        _fixed_calls, _errors, _again = self._stage7b_plan_validation(
            plan_ctx      = plan_ctx,
            plan_output   = state["plan_output"],
            tools_list    = tools,
            state         = state,
        )
        state["fixed_calls"] = _fixed_calls
    except Exception:
        pass

    try:
        tc_ctx, calls, selected_schemas = self._stage8_tool_chaining(
            plan_ctx    = plan_ctx,
            plan_output = state["plan_output"],
            tools_list  = tools,
            state       = state,
        )
        state["tc_ctx"]  = tc_ctx
        state["schemas"] = selected_schemas
        _ = self._stage8_5_user_confirmation(calls, user_text)  # auto-approve; keeps legacy semantics
    except Exception:
        state["schemas"] = state.get("schemas", [])

    # ─── 3) Execute DAG → reflect/replan until done ─────────────────
    all_tool_ctxs: List["ContextObject"] = []

    while True:
        # 3a) Execute ready nodes
        tcs = self._stage9_invoke_with_retries(
            raw_calls        = [],  # ignored for DAG mode
            plan_output      = state["plan_output"],
            selected_schemas = state.get("schemas", []),
            user_text        = user_text,
            clar_metadata    = clar_ctx.metadata,
            state            = state,
        )
        all_tool_ctxs.extend(tcs)

        # done?
        turn = state["turn"]
        if not getattr(turn, "pending_nodes", None):
            break

        # 3b) Reflection (optional replan)
        replan = self._stage9b_reflection_and_replan(
            tool_ctxs     = tcs,
            plan_output   = state["plan_output"],
            user_text     = user_text,
            clar_metadata = clar_ctx.metadata,
            state         = state,
        )
        if replan is None:      # OK/no changes
            continue

        # swap in new plan JSON (normalized)
        state["plan_output_prev"] = state["plan_output"]
        state["plan_output"]      = replan

    # ─── 4) Final answer assembly & polish ───────────────────────────
    reply = self._stage10_assemble_and_infer(user_text=user_text, state=state)
    polished = self._stage10b_response_critique_and_safety(
        draft     = reply,
        user_text = user_text,
        tool_ctxs = all_tool_ctxs,
        state     = state,
    )
    if polished:
        reply = polished

    # ─── 5) Memory write-back ────────────────────────────────────────
    self._stage11_memory_writeback(final_answer=reply, tool_ctxs=all_tool_ctxs)

    return reply, all_tool_ctxs


def _stage8_tool_chaining(
    self,
    plan_ctx: ContextObject,
    plan_output: str | dict,
    tools_list: List[Dict[str, Any]],
    state: Dict[str, Any],
    *,
    on_token: Callable[[str], None] | None = None,
) -> Tuple[ContextObject, List[str], List[ContextObject]]:
    """
    DAG-aware tool chaining summary.

    • If a DAG graph is present, we emit a *display* list of call-strings in node order
      (no placeholder substitution here; executor resolves at runtime).
    • Collect and return the schema ContextObjects for the referenced tools.
    • Persist a 'tool_chaining' context with the final list.
    """
    import json
    from context import ContextObject

    # Normalize plan → graph
    def _clean_json_block(text: str) -> str:
        import re
        m = re.search(r"```json\s*(\{.*?\})\s*```", text or "", flags=re.S)
        if m: return m.group(1)
        m2 = re.search(r"(\{.*\})", text or "", flags=re.S)
        return (m2.group(1) if m2 else (text or "")).strip()

    if isinstance(plan_output, str):
        try:
            plan = json.loads(_clean_json_block(plan_output))
        except Exception:
            plan = {}
    else:
        plan = plan_output or {}

    graph = plan.get("graph")
    calls: List[str] = []

    if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
        nodes = graph["nodes"]
        def _json_args(args: dict) -> str:
            if not args: return ""
            return ",".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
        for n in nodes:
            name = (n.get("tool") or "").split("(", 1)[0]
            arg_s = _json_args(n.get("args", {}))
            calls.append(f"{name}({arg_s})" if arg_s else f"{name}()")
    else:
        # Legacy fallback: try to render tasks
        tasks = (plan.get("tasks") or []) if isinstance(plan.get("tasks"), list) else []
        for t in tasks:
            name = t.get("call") or ""
            kwargs = t.get("tool_input", {}) or {}
            if kwargs:
                arg_s = ",".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in kwargs.items())
                calls.append(f"{name}({arg_s})")
            else:
                calls.append(f"{name}()")

    # Collect schema ctxs for referenced tools
    try:
        all_schema_ctxs = {
            json.loads(c.metadata["schema"])["name"]: c
            for c in self.repo.query(lambda c: c.component == "schema" and "tool_schema" in (c.tags or []))
        }
    except Exception:
        all_schema_ctxs = {}

    tool_names = [c.split("(", 1)[0] for c in calls]
    selected_schemas = [all_schema_ctxs[n] for n in tool_names if n in all_schema_ctxs]

    # Persist summary
    ctx_refs = plan_ctx.references + [s.context_id for s in selected_schemas]
    tc_ctx = ContextObject.make_stage(
        "tool_chaining",
        ctx_refs,
        {"tool_calls": calls}
    )
    tc_ctx.stage_id = "tool_chaining"
    tc_ctx.summary  = json.dumps(calls, ensure_ascii=False)
    tc_ctx.touch(); self.repo.save(tc_ctx)

    return tc_ctx, calls, selected_schemas


def _stage8_5_user_confirmation(
    self,
    calls: list[Any],
    user_text: str
) -> list[str]:
    """
    Surface the to-be-invoked calls for user approval (DAG-friendly).

    • Accepts strings or function-call dicts and renders `tool(arg=...)`.
    • For DAGs, this list is informational; the executor schedules by dependencies.
    """
    import json

    def _obj2str(item) -> str:
        if isinstance(item, dict):
            name = item.get("name") or item.get("tool_name") or item.get("call")
            if not name:
                return str(item).strip()
            args_blob = item.get("arguments", item.get("parameters", {})) or {}
            if isinstance(args_blob, dict) and args_blob:
                arg_str = ",".join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in args_blob.items())
                return f"{name}({arg_str})"
            return f"{name}()"
        return str(item).strip()

    confirmed = [_obj2str(c) for c in calls]

    self._print_stage_context("user_confirmation", {"calls": confirmed})

    ctx = ContextObject.make_stage("user_confirmation", [], {"confirmed_calls": confirmed})
    ctx.stage_id = "user_confirmation"
    ctx.summary  = f"Auto-approved: {confirmed}"
    self._persist_and_index([ctx])

    return confirmed


def _stage9b_reflection_and_replan(
    self,
    tool_ctxs: List["ContextObject"],
    plan_output: Any,  # may be str or dict
    user_text: str,
    clar_metadata: Dict[str, Any],
    state: Dict[str, Any],
    max_tokens: int = 128000,
) -> Optional[str]:
    """
    DAG-aware reflection + (optional) re-planning stage.
    Returns
        • None   → keep existing graph
        • "OK"   → same as None (graph already satisfied intent)
        • <json> → new plan (normalised {'graph': {...}})
    """
    import json, re, hashlib, datetime
    from typing import Any, Dict

    # ────────────────────────── helper fns ────────────────────────────
    def _clean_json_block(text: Any) -> str:
        """Accept str or dict and return a JSON text to load from."""
        if isinstance(text, (dict, list)):
            try:
                return json.dumps(text, ensure_ascii=False)
            except Exception:
                return "{}"
        s = text or ""
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                s = ""
        m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
        if m:
            return m.group(1)
        m2 = re.search(r"(\{.*\})", s, flags=re.S)
        return (m2.group(1) if m2 else s).strip()

    def _pretty(obj: Any) -> str:
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
        except Exception:
            return repr(obj)

    _sig = lambda g: hashlib.md5(
        json.dumps(g or {}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    # convert {"tasks":[...]} or bare task into {'graph':{...}}
    def _as_graph(plan_any: Any) -> Dict[str, Any]:
        from copy import deepcopy

        def _flatten(task, parent, idx, acc):
            idx[0] += 1
            nid = task.get("id") or f"n{idx[0]}"
            acc.append(
                {
                    "id": nid,
                    "tool": task.get("call"),
                    "args": task.get("tool_input", {}) or {},
                    "after": [parent] if parent else [],
                }
            )
            for sub in task.get("subtasks", []) or []:
                _flatten(sub, nid, idx, acc)

        # Already a graph dict
        if isinstance(plan_any, dict) and "graph" in plan_any:
            g = deepcopy(plan_any["graph"])
            g.setdefault("nodes", [])
            g.setdefault("meta", {})
            for n in g["nodes"]:
                n.setdefault("args", {})
                n.setdefault("after", [])
            return {"graph": g}

        # Plan as dict with tasks
        if isinstance(plan_any, dict) and "tasks" in plan_any:
            nodes, counter = [], [0]
            for t in plan_any["tasks"] or []:
                _flatten(t, None, counter, nodes)
            return {"graph": {"nodes": nodes, "meta": plan_any.get("meta", {}) or {}}}

        # If it’s text, try to parse
        if isinstance(plan_any, str) and plan_any.strip():
            try:
                obj = json.loads(_clean_json_block(plan_any))
                return _as_graph(obj)
            except Exception:
                pass

        # Bare single task dict
        if isinstance(plan_any, dict) and "call" in plan_any:
            return _as_graph({"tasks": [plan_any]})

        return {"graph": {"nodes": [], "meta": {}}}

    # deps-satisfied util
    def _deps_ok(n: dict, done: set[str]) -> bool:
        return all(d in done for d in (n.get("after") or []))

    # ───────────────────── current turn / plan state ──────────────────
    turn = _ensure_turn_state(state)

    # Robustly normalize incoming plan_output (dict or str) → graph
    try:
        if isinstance(plan_output, dict):
            plan_obj_in = plan_output
        elif isinstance(plan_output, str):
            plan_obj_in = json.loads(_clean_json_block(plan_output)) if plan_output.strip() else {}
        else:
            plan_obj_in = {}
    except Exception:
        plan_obj_in = {}

    cur_graph = _as_graph(plan_obj_in).get("graph")
    cur_sig   = _sig(cur_graph)

    # safe tracker lookup (works for list or iterator)
    trackers = self.repo.query(
        lambda c: c.component == "plan_tracker"
        and (
            c.metadata.get("plan_id") == getattr(turn, "plan_id", None)
            or c.semantic_label == cur_sig
        )
    )
    try:
        tracker = trackers[0] if isinstance(trackers, list) else next(iter(trackers))
    except Exception:
        tracker = None

    completed = set()
    pending   = set()
    errs_by   = {}

    if tracker:
        completed = set(tracker.metadata.get("completed") or [])
        pending   = set(tracker.metadata.get("pending_nodes") or [])
        errs_by   = dict(tracker.metadata.get("errors_by_node") or {})
    else:
        completed = set(getattr(turn, "completed_nodes", set()) or [])
        pending   = set(getattr(turn, "pending_nodes", set()) or [])

    ready = [
        n["id"]
        for n in cur_graph.get("nodes") or []
        if n["id"] in pending and _deps_ok(n, completed)
    ]

    # ───────────────────── assemble reflection prompt ─────────────────
    clar_notes = (clar_metadata or {}).get("notes", "")
    clar_keywords = (clar_metadata or {}).get("keywords", [])

    parts = [
        f"=== TURN ID ===\n{getattr(turn,'turn_id','')}",
        f"=== USER ===\n{user_text}",
        f"=== CLARIFIER NOTES ===\n{clar_notes or '(none)'}",
        f"=== EXEC STATUS ===\n{_pretty(dict(total=len(cur_graph.get('nodes',[])),completed=sorted(completed),pending=sorted(pending),ready_now=sorted(ready),errors_by_node=errs_by))}",
    ]
    if clar_keywords:
        parts.append("=== CLARIFIER KEYWORDS ===\n" + ", ".join(clar_keywords))

    for tc in tool_ctxs:
        oid = (tc.metadata or {}).get("node_id")
        tnm = (tc.metadata or {}).get("tool_name")
        payload = (tc.metadata or {}).get("output_short") or (tc.metadata or {}).get("output_full") or (tc.metadata or {})
        parts.append(f"=== TOOL OUTPUT [node={oid} tool={tnm}] ===\n{_pretty(payload)[:800]}")

    parts.append("=== ORIGINAL GRAPH ===\n" + _pretty(cur_graph))

    prompt_user = "\n\n".join(parts) + (
        "\n\nDid these tool outputs fully satisfy the user request?\n"
        "• If YES, reply exactly: OK\n"
        "• Otherwise return ONLY the corrected plan as JSON.\n"
        "Accepted formats:\n"
        "  {\"graph\": {\"nodes\": [...], \"meta\": {...}}}\n"
        "  — or —\n"
        "  {\"tasks\": [...]} (will be auto-converted)"
    )

    # ───────────────────── call model for reflection ──────────────────
    sys_msg = self._get_prompt("reflection_prompt")
    reply_raw = self._stream_and_capture(
        self.secondary_model,
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt_user}],
        tag="[Reflection]",
    )

    # ───────────────— quick exits: OK / same graph —───────────────────
    if isinstance(reply_raw, str) and re.fullmatch(r"(?i)(ok|okay)[.!]?", reply_raw.strip()):
        return None

    # If model returned dict already, great; otherwise parse text
    if isinstance(reply_raw, dict):
        new_plan = reply_raw
    else:
        try:
            new_plan = json.loads(_clean_json_block(reply_raw or ""))
        except Exception:
            # unparsable – bubble raw string upwards (executor/upper layer decides)
            return reply_raw if isinstance(reply_raw, str) else None

    new_norm  = _as_graph(new_plan)
    new_graph = new_norm["graph"]
    if _sig(new_graph) == cur_sig:
        return None  # identical -> nothing to change

    # --------------- update TurnState & tracker for new graph ----------
    new_sig = _sig(new_graph)
    turn.plan_id = f"plan_{new_sig}"
    turn.graph   = new_graph
    turn.pending_nodes = {n["id"] for n in new_graph.get("nodes", [])}
    turn.completed_nodes &= turn.pending_nodes  # keep only still-existing ones

    tracker_new = ContextObject.make_stage(
        "plan_tracker",
        [],
        dict(
            plan_id=turn.plan_id,
            turn_id=turn.turn_id,
            total_nodes=len(new_graph.get("nodes", [])),
            pending_nodes=sorted(list(turn.pending_nodes)),
            completed=sorted(list(turn.completed_nodes)),
            errors_by_node={},
            attempts=0,
            status="in_progress",
            started_at=datetime.datetime.utcnow().isoformat() + "Z",
        ),
    )
    tracker_new.semantic_label = new_sig
    tracker_new.stage_id = f"plan_tracker_{new_sig}"
    tracker_new.summary = "tracker (from reflection)"
    tracker_new.touch()
    self.repo.save(tracker_new)

    # return normalised JSON string
    return json.dumps({"graph": new_graph}, ensure_ascii=False)



def _stage9_invoke_with_retries(
    self,
    raw_calls: List[str],
    plan_output: str,
    selected_schemas: List[ContextObject],
    user_text: str,
    clar_metadata: Dict[str, Any],
    state: Dict[str, Any],
) -> List[ContextObject]:
    """
    DAG executor: runs *ready* nodes, retries failures with back-off, and
    persists a tool_output ContextObject for every execution.

    Returns only the tool outputs produced **during this call** so that the
    caller can surface “IMMEDIATE” results without re-querying the repo.
    """
    # ------------------------------------------------------------------ imports
    import json, re, hashlib, datetime, time
    from typing import Any, Dict, List
    from tools import Tools                           # local tool runner
    Tools.repo = self.repo                            # bind repo

    # ------------------------------------------------------------------ helpers
    def _smart_truncate(txt: str, max_len: int = 4_000) -> str:
        if txt is None or len(txt) <= max_len:
            return txt or ""
        head, tail = txt[: max_len // 2], txt[-max_len // 2 :]
        return f"{head}\n… ⟪{len(txt)-max_len} chars elided⟫ …\n{tail}"

    def _validate(res: dict) -> tuple[bool, str]:
        exc = res.get("exception")
        return exc is None, (str(exc) if exc else "")

    normalize_key = lambda k: re.sub(r"\W+", "", str(k)).lower()

    def _clean_json_block(txt: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.S)
        if m:
            return m.group(1)
        m2 = re.search(r"(\{.*\})", txt, flags=re.S)
        return (m2.group(1) if m2 else (txt or "")).strip()

    # ------------------------------------------------------------------ PLAN / TURN
    turn = _ensure_turn_state(state)
    # ────────────────────────────────────────────────────────────────
    # 1)  Normalise the incoming plan → graph obj
    # ────────────────────────────────────────────────────────────────
    try:
        plan_obj = json.loads(_clean_json_block(plan_output)) if plan_output else {}
    except Exception:
        plan_obj = {}
    graph_obj: Dict[str, Any] = plan_obj.get("graph") or getattr(turn, "graph", {}) or {}
    nodes: List[Dict[str, Any]] = graph_obj.get("nodes") or []
    if not isinstance(nodes, list):
        nodes = []

    # quick id → node map
    graph_index: Dict[str, Dict[str, Any]] = {n.get("id"): n for n in nodes if n.get("id")}

    # ────────────────────────────────────────────────────────────────
    # 2)  SCHEMAS (for citations)
    # ────────────────────────────────────────────────────────────────
    schema_map: Dict[str, ContextObject] = {}
    for s in selected_schemas or []:
        try:
            nm = json.loads(s.metadata["schema"])["name"]
            schema_map[nm] = s
        except Exception:
            continue

    # ────────────────────────────────────────────────────────────────
    # 3)  PLAN-TRACKER row bootstrap / update
    # ────────────────────────────────────────────────────────────────
    plan_sig_src = json.dumps(graph_obj, ensure_ascii=False, sort_keys=True)
    plan_sig = hashlib.md5(plan_sig_src.encode("utf-8")).hexdigest()[:8]

    tracker = next(
        (
            c
            for c in self.repo.query(
                lambda c: c.component == "plan_tracker"
                and (
                    c.metadata.get("plan_id") == getattr(turn, "plan_id", None)
                    or c.semantic_label == plan_sig
                )
            )
        ),
        None,
    )

    if not tracker:
        tracker = ContextObject.make_stage(
            "plan_tracker",
            [],
            dict(
                plan_id=getattr(turn, "plan_id", f"plan_{plan_sig}"),
                turn_id=getattr(turn, "turn_id", ""),
                total_nodes=len(nodes),
                pending_nodes=[n.get("id") for n in nodes],
                completed=[],
                errors_by_node={},
                attempts=0,
                status="in_progress",
                started_at=datetime.datetime.utcnow().isoformat() + "Z",
            ),
        )
        tracker.semantic_label = plan_sig
        tracker.stage_id = f"plan_tracker_{plan_sig}"
    tracker.metadata["attempts"] = tracker.metadata.get("attempts", 0) + 1
    tracker.metadata["last_attempt_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    tracker.touch(); self.repo.save(tracker)

    # ────────────────────────────────────────────────────────────────
    # 4)  TURN bookkeeping
    # ────────────────────────────────────────────────────────────────
    if not getattr(turn, "pending_nodes", None):
        turn.pending_nodes = {nid for nid in graph_index}
    if not getattr(turn, "completed_nodes", None):
        turn.completed_nodes = set()
    turn.done = turn.completed_nodes                          # handy alias

    # results & retry counters
    turn.results = getattr(turn, "results", {})
    turn.attempts_left = getattr(turn, "attempts_left", {})
    for nid in graph_index:
        turn.attempts_left.setdefault(nid, 3)                 # default retries / node

    # ────────────────────────────────────────────────────────────────
    # 5)  Placeholder memory + helpers
    # ────────────────────────────────────────────────────────────────
    last_results: Dict[str, Any] = {}

    def _record_result(node_id: str, tool_name: str, alias: str | None, value: Any):
        for k in filter(None, [node_id, tool_name, alias]):
            last_results[k] = value
            last_results[normalize_key(k)] = value

    # — regex for [n1.output] or [<n1>.output] and for {{alias}} —
    _BRACKET_RE = re.compile(
        r"\[\s*<?\s*([A-Za-z0-9_-]+)\s*>?\s*\.output(?:\.[A-Za-z0-9_.-]+)?\s*\]"
    )
    _ALIAS_RE = re.compile(r"\{\{([A-Za-z0-9_-]+)\}\}")

    def _subst_in_str(s: str) -> Any:
        if not isinstance(s, str):
            return s

        # stand-alone replacements
        m = _BRACKET_RE.fullmatch(s)
        if m:
            key = m.group(1)
            return last_results.get(key) or last_results.get(normalize_key(key))
        m = _ALIAS_RE.fullmatch(s)
        if m:
            key = m.group(1)
            return last_results.get(key) or last_results.get(normalize_key(key))

        # inline replacements
        def _rep(mo):
            key = mo.group(1)
            val = last_results.get(key) or last_results.get(normalize_key(key)) or ""
            return json.dumps(val, ensure_ascii=False)

        s = _BRACKET_RE.sub(_rep, s)
        s = _ALIAS_RE.sub(_rep, s)
        return s

    def _subst(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _subst(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_subst(v) for v in obj]
        if isinstance(obj, str):
            return _subst_in_str(obj)
        return obj

    # ────────────────────────────────────────────────────────────────
    # 6)  Ready-node helpers
    # ────────────────────────────────────────────────────────────────
    def _deps_satisfied(nid: str) -> bool:
        deps = graph_index.get(nid, {}).get("after", []) or []
        return all(d in turn.done for d in deps)

    def _next_ready_nodes() -> List[Dict[str, Any]]:
        return [
            n
            for n in nodes
            if n.get("id") in turn.pending_nodes and _deps_satisfied(n["id"])
        ]

    # ────────────────────────────────────────────────────────────────
    # 7)  Budgets & execution scaffolding
    # ────────────────────────────────────────────────────────────────
    budgets = getattr(turn, "budgets", {}) or state.get("budgets", {}) or {}
    calls_budget  = int(budgets.get("calls", 1_000_000))
    time_budget_s = float(budgets.get("time", 1e9))
    start_time = time.time()

    tool_ctxs: List[ContextObject] = []
    pending_backoff: Dict[str, float] = {}

    plateau_guard = 0
    while True:
        if calls_budget <= 0 or (time.time() - start_time) > time_budget_s:
            break

        ready = [
            n
            for n in _next_ready_nodes()
            if pending_backoff.get(n["id"], 0.0) <= time.time()
        ]

        if not ready:
            plateau_guard += 1
            if plateau_guard >= 2 or not turn.pending_nodes:
                break
            time.sleep(0.05)
            continue
        plateau_guard = 0

        for node in ready:
            if calls_budget <= 0 or (time.time() - start_time) > time_budget_s:
                break

            nid, tname = node["id"], node["tool"]
            raw_args = node.get("args") or {}

            args_resolved = _subst(raw_args)

            # canonical call string
            if args_resolved:
                try:
                    arg_s = ",".join(
                        f"{k}={json.dumps(v, ensure_ascii=False)}"
                        for k, v in args_resolved.items()
                    )
                except Exception:
                    arg_s = ",".join(f"{k}={repr(v)}" for k, v in args_resolved.items())
                call_str = f"{tname}({arg_s})"
            else:
                call_str = f"{tname}()"

            res = Tools.run_tool_once(call_str)
            ok, err_msg = _validate(res)
            raw_out = res.get("output")

            try:
                pretty = json.dumps(raw_out, ensure_ascii=False, indent=2)
            except Exception:
                pretty = repr(raw_out)
            short = _smart_truncate(pretty)

            refs = []
            if (sc := schema_map.get(tname)):
                refs = [sc.context_id]

            meta = dict(
                turn_id=getattr(turn, "turn_id", ""),
                plan_id=getattr(turn, "plan_id", f"plan_{plan_sig}"),
                node_id=nid,
                tool_call=call_str,
                tool_name=tname,
                args=args_resolved,
                output_full=pretty,
                output_short=short,
                output=raw_out,
                exception=res.get("exception"),
                conversation_id=state.get("conversation_id"),
                user_id=state.get("user_id"),
            )
            ctx = ContextObject.make_stage("tool_output", refs, meta)
            ctx.stage_id = f"tool_output_{tname}"
            ctx.summary = ("ERROR: " + err_msg) if not ok else short
            ctx.touch()
            self.repo.save(ctx)

            tool_ctxs.append(ctx)
            turn.tool_ctx_ids.append(ctx.context_id)

            # ---- progress bookkeeping
            calls_budget -= 1
            tracker.metadata.setdefault("node_status", {})[nid] = ok

            alias = node.get("alias")
            if ok:
                _record_result(nid, tname, alias, raw_out)
                turn.results[nid] = raw_out
                turn.results[tname] = raw_out
                turn.completed_nodes.add(nid)
                turn.pending_nodes.discard(nid)
                tracker.metadata.setdefault("completed", []).append(nid)
                pending_backoff.pop(nid, None)
            else:
                tracker.metadata.setdefault("errors_by_node", {})[nid] = err_msg
                turn.attempts_left[nid] -= 1
                if turn.attempts_left[nid] <= 0:
                    # give up, propagate None so deps can continue
                    _record_result(nid, tname, alias, None)
                    turn.results[nid] = None
                    turn.results[tname] = None
                    turn.completed_nodes.add(nid)
                    turn.pending_nodes.discard(nid)
                else:
                    # schedule retry
                    fail_cnt = node.get("_fail_cnt", 0) + 1
                    node["_fail_cnt"] = fail_cnt
                    pending_backoff[nid] = time.time() + (2 ** fail_cnt)

            tracker.touch()
            self.repo.save(tracker)

        # -------- loop exit checks
        if not turn.pending_nodes:
            break
        if not _next_ready_nodes():
            break

    # ------------------------------------------------------------------ tracker final status
    if not turn.pending_nodes:
        tracker.metadata["status"] = "success"
    elif calls_budget <= 0:
        tracker.metadata["status"] = "budget_exhausted"
    else:
        tracker.metadata["status"] = "partial"
    tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    tracker.touch()
    self.repo.save(tracker)

    # ------------------------------------------------------------------ return / ingest
    state["tool_ctxs"] = tool_ctxs
    if tool_ctxs:
        self.integrator.ingest(tool_ctxs)
        state.setdefault("merged", []).extend(tool_ctxs)

    return tool_ctxs


def _await_if_needed(obj):
    """Return result synchronously whether obj is a coroutine or not.
       Safe to call from a worker thread (no running loop there)."""
    if inspect.isawaitable(obj):
        return asyncio.run(obj)   # we're inside a to_thread worker, so no active loop
    return obj


def _stage10_assemble_and_infer(self, user_text: str, state: dict[str, Any]) -> str:
    import json, pprint
    from collections import OrderedDict
    from datetime import datetime

    def _as_dt(ts: str) -> datetime:
        try:    return datetime.fromisoformat((ts or "").rstrip("Z"))
        except: return datetime.min

    # ─── (0) System: final-inference prompt ──────────────────────────
    final_sys = self._get_prompt("final_inference_prompt")

    # ─── (1) Clarified intent (compact) ─────────────────────────────
    clar_notes = ""
    clar_ctx = state.get("clar_ctx")
    if clar_ctx:
        clar_notes = (clar_ctx.metadata.get("notes") or clar_ctx.summary or "").strip()
    clarifier_block = "[Clarified intent]\n" + (clar_notes or "(none)")

    # ─── (2) Latest user question ───────────────────────────────────
    latest_user_block = "[Latest user question]\n" + (user_text or "")

    # ─── (3) Conversation since last tool output (fallback: last 2) ─
    merged = state.get("merged", [])
    segments = [c for c in merged if c.domain == "segment" and c.semantic_label in ("user_input","assistant")]

    tool_ctxs = state.get("tool_ctxs", []) or []
    last_tool_ts = max((_as_dt(getattr(c, "timestamp", "")) for c in tool_ctxs), default=None)
    if last_tool_ts:
        scoped = [c for c in segments if _as_dt(getattr(c,"timestamp","")) >= last_tool_ts]
    else:
        scoped = segments[-2:]  # keep it tiny if we have no tool run

    convo_lines = []
    for c in scoped:
        role = "User" if c.semantic_label == "user_input" else "Assistant"
        src  = f"{c.component}/{c.semantic_label or c.stage_id}"
        text = c.summary or ""
        convo_lines.append(f"[{src}] {role}: {text}")

    conversation_block = "[Conversation (current turn)]\n" + "\n".join(convo_lines)

    # ─── (4) Plan (normalized) ──────────────────────────────────────
    raw_plan = state.get("plan_output", "(no plan)")
    if not isinstance(raw_plan, str):
        try:    raw_plan = json.dumps(raw_plan, ensure_ascii=False, indent=2)
        except: raw_plan = pprint.pformat(raw_plan, compact=True)
    plan_block = "[Plan]\n" + raw_plan

    # ─── (5) Tool outputs (IMMEDIATE only, already in state) ────────
    tool_ctxs.sort(key=lambda c: getattr(c, "timestamp", ""))
    tool_blocks = []
    for tc in tool_ctxs:
        meta = tc.metadata or {}
        data = meta.get("output", meta.get("output_full", meta))
        try:    payload = json.dumps(data, ensure_ascii=False, indent=2)
        except: payload = pprint.pformat(data, compact=True)
        call_name = meta.get("tool_call", tc.stage_id)
        ts        = getattr(tc, "timestamp", "")
        tool_blocks.append(f"--- {tc.stage_id} ({call_name}) @ {ts} ---")
        tool_blocks.append(payload)
    tools_block = "[Tool outputs]\n" + "\n\n".join(tool_blocks) if tool_blocks else ""

    # IMPORTANT: **DROP** Narrative entirely to avoid cross-turn bleed
    # (was: narrative_block = "[Narrative] ...")

    # ─── (6) Assemble messages ───────────────────────────────────────
    msgs = [
        {"role": "system", "content": final_sys},
        {"role": "system", "content": clarifier_block},
        {"role": "user",   "content": latest_user_block},
    ]
    if conversation_block:
        msgs.append({"role":"system","content": conversation_block})
    msgs.append({"role":"system","content": plan_block})
    if tools_block:
        msgs.append({"role":"system","content": tools_block})

    # ─── (7) Debug (trimmed) ─────────────────────────────────────────
    try:
        exact_prompt = self._gemma_format(msgs)
    except:
        import json as _json
        exact_prompt = _json.dumps(msgs, ensure_ascii=False, indent=2)

    turn = _ensure_turn_state(state)
    debug = OrderedDict([
        ("turn_id",           getattr(turn, "turn_id", "")),
        ("plan_id",           getattr(turn, "plan_id", "")),
        ("assembled_prompt_text", exact_prompt),
    ])
    self._print_stage_context("assemble_and_infer", debug)

    # ─── (8) Call the model ─────────────────────────────────────────
    raw_reply = _await_if_needed(
        self._stream_and_capture(self.primary_model, msgs, tag="[Assistant]", images=state.get("images"))
    )
    reply = (raw_reply or "").strip()

    # ─── (9) Persist reply (only this turn’s refs) ───────────────────
    refs = [c.context_id for c in scoped] + [c.context_id for c in tool_ctxs]
    resp_ctx = ContextObject.make_stage("final_inference", refs, {"text": reply})
    resp_ctx.stage_id = "final_inference"
    resp_ctx.summary  = reply

    # ─── NEW: compute embedding‐based relevance between user & reply ──
    from numpy import dot
    from numpy.linalg import norm
    uvec = self.embed_text(user_text)
    rvec = self.embed_text(reply)
    sim = float(dot(uvec, rvec) / (norm(uvec) * norm(rvec) + 1e-9))
    resp_ctx.metadata["relevance_score"] = sim

    self._persist_and_index([resp_ctx])

    seg = ContextObject.make_segment("assistant", [resp_ctx.context_id], tags=["assistant"])
    seg.summary  = reply
    seg.stage_id = "assistant"
    seg.touch(); self.repo.save(seg)

    state["draft"]         = reply
    state["assistant_ctx"] = resp_ctx
    return reply



def _stage10b_response_critique_and_safety(
    self,
    draft: str,
    user_text: str,
    tool_ctxs: list["ContextObject"],
    state: dict[str, Any],
) -> str:
    import json, difflib, pprint

    # Helper: robust stringify for any object (dicts, lists, etc.)
    def _to_str(obj) -> str:
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            try:
                return pprint.pformat(obj, compact=True)
            except Exception:
                return str(obj)

    if not draft:
        return draft

    # ─── 1) Build blocks ───────────────────────────────────────────
    user_block  = "[Latest user question]\n" + (user_text or "")
    draft_block = "[Draft response]\n"       + _to_str(draft)

    # Merge snippets as before
    merged = state.get("merged", [])
    merged_texts = "\n\n".join(f"[{c.stage_id}] {c.summary}" for c in merged) or "(none)"
    merged_block = "[Merged context snippets]\n" + merged_texts

    # Plan block (fix: handle dict or str safely)
    plan_any  = state.get("plan_output", "(no plan)")
    plan_txt  = _to_str(plan_any)
    plan_block = "[Plan executed]\n" + plan_txt

    # ─── 2) Immediate tool‐outputs only ────────────────────────────
    outputs = []
    for tc in (tool_ctxs or []):
        raw = tc.metadata.get("output", tc.metadata)
        if isinstance(raw, dict) and "results" in raw:
            frag = "\n".join(
                f"{r.get('timestamp','')} {r.get('role','')}: {r.get('content','')}"
                for r in raw["results"]
            )
        else:
            try:
                frag = json.dumps(raw, indent=2, ensure_ascii=False)
            except Exception:
                frag = pprint.pformat(raw, compact=True)
        outputs.append(f"[{tc.stage_id}]\n{frag}")
    tools_block = "[Tool outputs]\n" + "\n\n".join(outputs) if outputs else ""

    # ─── 3) Relevance Extraction ────────────────────────────────────
    extractor_sys = self._get_prompt("extractor_sys_prompt")
    extractor_msgs = [
        {"role":"system", "content": extractor_sys},
        {"role":"system", "content": user_block},
        {"role":"system", "content": draft_block},
        {"role":"system", "content": plan_block},
        {"role":"system", "content": merged_block},
    ]
    if tools_block:
        extractor_msgs.append({"role":"system","content":tools_block})

    bullets = self._stream_and_capture(
        self.secondary_model,
        extractor_msgs,
        tag="[RelevExtract]",
        images=state.get("images"),
    ).strip()

    sum_ctx = ContextObject.make_stage("relevance_summary", [], {"summary": bullets})
    sum_ctx.stage_id = "relevance_summary"; sum_ctx.summary = bullets
    self._persist_and_index([sum_ctx])

    # ─── 4) Polishing / Safety Critique ─────────────────────────────
    editor_sys = self._get_prompt("editor_sys_prompt")
    editor_msgs = [
        {"role":"system","content":editor_sys},
        {"role":"system","content": user_block},
        {"role":"system","content": draft_block},
        {"role":"system","content":"[Relevance bullets]\n"+bullets},
    ]
    polished = self._stream_and_capture(
        self.secondary_model,
        editor_msgs,
        tag="[Polisher]",
        images=state.get("images"),
    ).strip()

    if polished == (draft.strip() if isinstance(draft, str) else _to_str(draft).strip()):
        return polished

    # ─── 5) diff & dynamic_patch (unchanged) ────────────────────────
    orig_lines = draft.splitlines() if isinstance(draft, str) else _to_str(draft).splitlines()
    diff = difflib.unified_diff(orig_lines, polished.splitlines(), lineterm="", n=1)
    diff_summary = "; ".join(ln for ln in diff if ln.startswith(("+ ", "- "))) or "(format refined)"

    rows = sorted(
        self.repo.query(lambda c: c.component=="policy" and c.semantic_label=="dynamic_prompt_patch"),
        key=lambda c: c.timestamp, reverse=True
    )
    patch = rows[0] if rows else ContextObject.make_policy("dynamic_prompt_patch", diff_summary, tags=["dynamic_prompt"])
    if patch.summary != diff_summary:
        patch.summary = diff_summary
        patch.metadata["policy"] = diff_summary
        patch.touch(); self.repo.save(patch)

    # ─── 6) Persist polished & critique ─────────────────────────────
    resp_ctx = ContextObject.make_stage("response_critique", [sum_ctx.context_id], {"text": polished})
    resp_ctx.stage_id = "response_critique"; resp_ctx.summary = polished
    self._persist_and_index([resp_ctx])

    critique_ctx = ContextObject.make_stage(
        "plan_critique",
        [resp_ctx.context_id] + [tc.context_id for tc in (tool_ctxs or [])],
        {"critique": polished, "diff": diff_summary},
    )
    critique_ctx.component      = "analysis"
    critique_ctx.semantic_label = "plan_critique"
    self._persist_and_index([critique_ctx])

    return polished





def _stage11_memory_writeback(
    self,
    final_answer: str,
    tool_ctxs: list[ContextObject],
) -> None:
    """
    Long-term memory write-back that never balloons context.jsonl.

    • `auto_memory` → *singleton* (insert once, then update in-place)
    • narrative     → one new row per turn (intended)
    • every object is persisted exactly ONCE
    """

    turn = None
    try:
        # not fatal if missing
        turn = _ensure_turn_state(getattr(self, "_state", {}))
    except Exception:
        pass

    # ── 1)  Up-sert the single `auto_memory` row ────────────────────────
    mem_candidates = self.repo.query(
        lambda c: c.domain == "artifact"
        and c.component == "knowledge"
        and c.semantic_label == "auto_memory"
    )
    mem = mem_candidates[0] if mem_candidates else None

    if mem is None:                             # first run  → INSERT
        mem = ContextObject.make_knowledge(
            label   = "auto_memory",
            content = final_answer,
            tags    = ["memory_writeback"],
        )
    else:                                       # later runs → UPDATE (if text changed)
        if mem.metadata.get("content") != final_answer:
            mem.metadata["content"] = final_answer
            mem.summary             = final_answer

    mem.touch()                                 # refresh timestamp / last_accessed

    # IMPORTANT:  call reinforce **before** the single save below.
    # MemoryManager mutates mem in-place but does NOT append a new row,
    # so persisting once afterwards keeps the file tidy.
    # ── Guard against dangling refs ────────────────────────────────
    valid_refs = []
    for c in tool_ctxs:
        try:
            # verify the object still exists (and is persisted)
            self.repo.get(c.context_id)
            valid_refs.append(c.context_id)
        except KeyError:
            # skip IDs that were deduped, pruned, or never saved
            continue

    self.memman.reinforce(mem.context_id, valid_refs)

    if turn:
        mem.metadata.setdefault("turn_ids", []).append(turn.turn_id)
        mem.metadata.setdefault("plan_ids", []).append(turn.plan_id)

    # One narrative row per *unique* answer – duplicates are skipped
    narr = ContextObject.make_narrative(
        f"At {default_clock().strftime('%Y-%m-%d %H:%M:%SZ')}, "
        f"I handled the user’s request and generated: "
        f"{final_answer[:200]}…"
    )
    # make_narrative() already touches & saves when it reuses a row;
    # only save when we truly inserted a new one
    if narr.context_id not in {c.context_id for c in self.repo.query(lambda c: c.component == "narrative")}:
        narr.touch()
        self.repo.save(narr)


# ──────────────────────────────────────────────────────────────────
def _stage12_performance_rating(self, state: dict[str, Any]) -> None:
    """
    Compute a scalar reward (−1 … +1) for this turn,
    • persist it via ContextObject.make_performance()
    • copy the reward into every ContextObject touched this turn
      (so MemoryManager can later use outcome_reward for pruning /
       promotion decisions).
    """
    import time
    from context import ContextObject

    # --- 1) crude heuristic reward -----------------------------------
    err_penalty  = -0.4 if state.get("errors") else 0.0
    tool_penalty = -0.2 if any(tc.metadata.get("exception") for tc in state.get("tool_ctxs", [])) else 0.0
    speed_bonus  = +0.2 if state.get("provisional_sent") else 0.0

    # ─── NEW: boost reward if reply was highly relevant ───────────
    perf_objs = self.repo.query(lambda c: c.component=="stage" and c.semantic_label=="final_inference")
    last_resp = max(perf_objs, key=lambda c: c.timestamp, default=None)
    rel_bonus = 0.0
    if last_resp and isinstance(last_resp.metadata.get("relevance_score"), float):
        rel = last_resp.metadata["relevance_score"]
        # scale so if sim > .8 give up to +0.2
        rel_bonus = max(0.0, (rel - 0.8)) * 1.0

    reward = max(-1.0, min(1.0, 1.0 + err_penalty + tool_penalty + speed_bonus + rel_bonus))

    # --- 2) persist stage-performance object -------------------------
    perf = ContextObject.make_performance(
        reward = reward,
        stage_ids = list(state.get("stages_run", [])),
        metrics = {
            "latency_ms": round((time.time() - state.get("start_ts", time.time())) * 1000),
            "errors": state.get("errors", []),
        },
    )
    self.repo.save(perf)
    self.memman.register_relationships(perf, self.embed_text)

    # --- 3) propagate reward into all objects written this turn ------
    touched_ids = (
        state.get("merged_ids", []) +
        [c.context_id for c in state.get("tool_ctxs", [])] +
        [perf.context_id]
    )
    for cid in touched_ids:
        try:
            obj = self.repo.get(cid)
            if obj.outcome_reward is None:        # don’t overwrite later passes
                obj.outcome_reward = reward
                obj.touch()
                self.repo.save(obj)
        except KeyError:
            continue

    # --- 4) tell the RL controllers ----------------------------------
    self.rl.update(list(state.get("stages_run", [])), reward)
    self.curiosity_rl.update(state.get("curiosity_used", []), reward)

    # --- 5) record our reward in the rolling metacognitive context ----
    try:
        lines = (self.metacog_ctx.summary or "").splitlines()
        lines.append(f"stage12 reward={reward:+.2f}")
        self.metacog_ctx.summary = "\n".join(lines)
        self.metacog_ctx.touch()
        self.repo.save(self.metacog_ctx)
    except Exception:
        pass



def _stage_generate_narrative(self, state: Dict[str, Any]) -> ContextObject:
    """
    Build a running narrative of this conversation turn by turn,
    link it to all the context objects we’ve touched so far,
    and store the narrative’s ContextObject ID for future reference.
    """
    #── only once ────────────────────────────────────────────────────
    if getattr(self, "_narrative_emitted", False):
        return self._narrative_cache

    # gather all the IDs of contexts created/used this turn
    used_ids = []
    for key in ("user_ctx","sys_ctx","clar_ctx","know_ctx","plan_ctx","tc_ctx"):
        if key in state:
            used_ids.append(state[key].context_id)
    used_ids += [c.context_id for c in state.get("tool_ctxs",[])]
    # de-dupe
    used_ids = list(dict.fromkeys(used_ids))

    # assemble human narrative
    from datetime import datetime
    lines = [
        f"{datetime.utcnow():%Y-%m-%d %H:%M:%SZ}:",
        f"• User asked: {state['user_text']!r}",
        f"• Clarified into: {state['clar_ctx'].summary!r}",
    ]
    if "plan_output" in state:
        lines.append(f"• Planner proposed: {state['plan_output']}")
    if "final" in state:
        lines.append(f"• Assistant replied: {state['final']!r}")

    narrative_text = "\n".join(lines)

    # upsert the single narrative_context keeper
    nc = self._get_or_make_singleton(
        label="narrative_context",
        component="stage",
        tags=["narrative"]
    )
    nc.metadata.setdefault("history_ids", []).extend(used_ids)
    nc.metadata["history_text"] = (
        (nc.metadata.get("history_text","") + "\n\n" + narrative_text)
        .strip()
    )
    nc.summary    = nc.metadata["history_text"]
    nc.references = nc.metadata["history_ids"]

    nc.touch()
    self.repo.save(nc)

    # mark done and cache
    self._narrative_emitted = True
    self._narrative_cache   = nc
    return nc

    
def _stage_prune_context_store(self, state: Dict[str, Any]) -> str:
    """
    Remove *only* ephemeral contexts older than `context_ttl_days`
    or beyond a hard cap, leaving static prompts/schemas untouched.
    Returns a one-line summary for status_cb.
    """
    from datetime import datetime, timedelta

    cutoff = default_clock() - timedelta(days=self.context_ttl_days)
    EPHEMERAL = {
        "segment", "tool_output", "knowledge", "narrative", "stage_performance"
    }

    deleted = 0
    # 1) Delete by age
    for ctx in self.repo.query(lambda c: c.component in EPHEMERAL):
        ts_raw = ctx.timestamp.rstrip("Z") if isinstance(ctx.timestamp, str) else None
        try:
            ts = datetime.fromisoformat(ts_raw) if ts_raw else None
        except Exception:
            continue
        if ts and ts < cutoff:
            try:
                self.repo.delete(ctx.context_id)
                deleted += 1
            except KeyError:
                pass

    # 2) Hard cap
    all_ephe = [
        c for c in self.repo.query(lambda c: c.component in EPHEMERAL)
    ]
    all_ephe.sort(
        key=lambda c: datetime.fromisoformat(
            (c.timestamp.rstrip("Z") if isinstance(c.timestamp, str) else default_clock().isoformat())
        ),
        reverse=True
    )
    cap = self.cfg.get("max_total_context", 1000)
    for old_ctx in all_ephe[cap:]:
        try:
            self.repo.delete(old_ctx.context_id)
            deleted += 1
        except KeyError:
            pass

    return f"pruned {deleted} items (ttl={self.context_ttl_days}d, cap={cap})"

def _stage_narrative_mull(self, state: Dict[str, Any]) -> str:
    """
    Async “self-talk” that:
        1. Gathers narrative, prompts, architecture.
        2. Pulls last-turn stage metrics & tool failures.
        3. Asks the LLM for ≤3 improvement items (diagnosis + questions + patches + mini-plans).
        4. Records Q&A, applies prompt patches, executes any plan_calls via normal pipeline.
    """
    import threading, io, contextlib, json, textwrap, datetime

    def _arch_dump() -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.dump_architecture()
        return buf.getvalue()

    def _collect_metrics() -> str:
        # fetch all stage_performance objects from repo
        perf_rows = list(self.repo.query(lambda c: c.component=="stage_performance"))
        data = [
            {
                "stage": r.metadata["stage"],
                "duration": round(r.metadata["duration"],3),
                "error": r.metadata["error"]
            }
            for r in perf_rows[-20:]  # last 20 entries
        ]
        return json.dumps(data, indent=2)

    def _collect_tool_failures() -> str:
        failures = []
        for ctx in state.get("tool_ctxs", []):
            if ctx.output.get("result", "").startswith("ERROR"):
                failures.append({
                    "call": ctx.metadata.get("call"),
                    "error": ctx.output["result"]
                })
        return json.dumps(failures, indent=2)

    def _worker():
        try:
            # 1) narrative + prompts + arch
            narr = self._load_narrative_context()
            full_narr = narr.metadata.get("history_text", narr.summary or "")

            prompts = self.repo.query(
                lambda c: c.component in ("prompt","policy") and "dynamic_prompt" not in c.tags
            )
            prompts.sort(key=lambda c: c.timestamp)
            prompt_block = "\n".join(
                f"- {textwrap.shorten(p.summary or '', 80)}"
                for p in prompts
            ) or "(none)"

            arch = _arch_dump()
            metrics = _collect_metrics()
            fails  = _collect_tool_failures()

            # 2) assemble meta-prompt
            meta = (
                self._get_prompt("narrative_mull_prompt")
                + "\n\n### Narrative ###\n" + full_narr
                + "\n\n### Prompts ###\n" + prompt_block
                + "\n\n### Architecture ###\n" + arch
                + "\n\n### Recent Stage Metrics ###\n" + metrics
                + "\n\n### Tool Failures ###\n" + fails
            )

            raw = self._stream_and_capture(
                self.primary_model,
                [{"role":"system","content": meta}],
                tag="[NarrativeMull]"
            ).strip()
            data = json.loads(raw)
            issues = data.get("issues", [])
            if not isinstance(issues, list):
                return

        except Exception:
            return  # abort silently

        # 3) process each issue
        for idx, item in enumerate(issues, 1):
            if not isinstance(item, dict):
                continue
            area      = item.get("area", f"area_{idx}")
            diag      = item.get("diagnosis","").strip()
            q_text    = item.get("question","").strip()
            patch     = item.get("prompt_patch","").strip()
            plan_calls= item.get("plan_calls", [])

            # record question
            q_ctx = ContextObject.make_stage(
                "narrative_question",
                [narr.context_id],
                {"area": area, "diagnosis": diag, "question": q_text}
            )
            q_ctx.component="narrative"; q_ctx.tags.append("narrative")
            #q_ctx.touch(); self.repo.save(q_ctx)
            self._persist_and_index([q_ctx])

            # get answer
            answer = ""
            if q_text:
                answer = self._stream_and_capture(
                    self.primary_model,
                    [
                        {"role":"system","content":
                            "You are the same meta-reasoner; answer only from the given data, be concise."},
                        {"role":"user","content": q_text}
                    ],
                    tag=f"[NarrativeAnswer_{idx}]",
                    images=state.get("images")
                ).strip()

            # record answer
            a_ctx = ContextObject.make_stage(
                "narrative_answer",
                [q_ctx.context_id],
                {"answer": answer}
            )
            a_ctx.component="narrative"; a_ctx.tags.append("narrative")
            #a_ctx.touch(); self.repo.save(a_ctx)
            self._persist_and_index([a_ctx])

            # apply prompt patch
            if patch:
                txt = (
                    f"// {datetime.datetime.utcnow().isoformat()}Z\n"
                    f"Issue:{area}\nDiag:{diag}\nQ:{q_text}\nA:{answer}\nPATCH:{patch}\n"
                )
                dyn = ContextObject.make_policy(
                    "dynamic_prompt_patch", policy_text=txt, tags=["dynamic_prompt"]
                )
                dyn.touch(); self.repo.save(dyn)

            # run plan_calls via real pipeline
            if plan_calls:
                try:
                    # Build a linear mini-DAG from plan_calls in order
                    nodes = []
                    prev_id = None
                    for i, c in enumerate(plan_calls, 1):
                        name = c.split("(",1)[0]
                        args_str = c.split("(",1)[1][:-1] if "(" in c else ""
                        # naive args parse: keep it simple, executor will JSON-load where possible
                        # or leave as string and Tools.run_tool_once() will accept it.
                        node_id = f"m{i}"
                        node = {"id": node_id, "tool": name, "args": {}, "after": ([prev_id] if prev_id else [])}
                        if args_str.strip():
                            # VERY SIMPLE parse: k=v pairs split on commas at top-level
                            # (safe enough for your existing call style)
                            kvs = [p for p in args_str.split(",") if p.strip()]
                            parsed = {}
                            for kv in kvs:
                                k, _, v = kv.partition("=")
                                k = k.strip()
                                v = v.strip()
                                try:
                                    parsed[k] = json.loads(v)
                                except Exception:
                                    parsed[k] = v
                            node["args"] = parsed
                        nodes.append(node)
                        prev_id = node_id

                    graph = {"nodes": nodes, "meta": {"goal": q_text or "(narrative mull)"}}
                    # Initialize TurnState for this mini-run
                    turn = _ensure_turn_state(state)
                    plan_sig = hashlib.md5(json.dumps(graph, sort_keys=True).encode("utf-8")).hexdigest()[:8]
                    turn.plan_id = f"plan_{plan_sig}"
                    turn.graph = graph
                    turn.pending_nodes = {n["id"] for n in nodes}
                    turn.completed_nodes = set()
                    turn.tool_ctx_ids = []

                    # Persist a lightweight internal plan ctx
                    p_ctx = ContextObject.make_stage("internal_plan_dag", [a_ctx.context_id], {"graph": graph})
                    _stamp(p_ctx, state)
                    p_ctx.touch(); self.repo.save(p_ctx)

                    # Collect schemas for referenced tools
                    tools = self._stage6_prepare_tools()
                    name_to_schema_ctx = {
                        json.loads(c.metadata["schema"])["name"]: c
                        for c in self.repo.query(lambda c: c.component == "schema" and "tool_schema" in (c.tags or []))
                    }
                    selected_schema_ctxs = [name_to_schema_ctx[n["tool"]] for n in nodes if n["tool"] in name_to_schema_ctxs]

                    # Execute DAG immediately
                    self._stage9_invoke_with_retries(
                        raw_calls=[],  # ignored in DAG mode
                        plan_output=json.dumps({"graph": graph}, ensure_ascii=False),
                        selected_schemas=selected_schema_ctxs,
                        user_text="(self-review DAG)",
                        clar_metadata={},
                        state=state,
                    )
                except Exception as e:
                    err = ContextObject.make_failure(f"narrative_mull DAG error: {e}", refs=[a_ctx.context_id])
                    _stamp(err, state)
                    err.touch(); self.repo.save(err)

    # start thread
    threading.Thread(target=_worker, daemon=True).start()
    return "(narrative mull dispatched)"