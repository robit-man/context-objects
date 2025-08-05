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

def _stage3_retrieve_and_merge_context(
    self,
    user_text: str,
    user_ctx: "ContextObject | None",
    sys_ctx: "ContextObject | List[ContextObject] | None",
    extra_ctx: List["ContextObject"] | None = None,
    recall_ids: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Retrieve & merge context for downstream stages, using RL gating
    and past relevance to bias what comes back.
    """
    from datetime import datetime, timedelta

    # ─── Helpers ────────────────────────────────────────────────────────
    def _ensure_list(x):
        if x is None:
            return []
        return x if isinstance(x, list) else [x]

    def _to_dt(ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts.rstrip("Z"))
        except Exception:
            return datetime.min

    def _prefix(ctx):
        """Label summary as User: or Assistant: if not already."""
        if not getattr(ctx, "summary", None):
            return
        txt = ctx.summary.lstrip()
        if txt.startswith(("User:", "Assistant:")):
            return
        role = "Assistant" if ctx.semantic_label == "assistant" else "User"
        ctx.summary = f"{role}: {ctx.summary}"

    # ─── 1️⃣ Flatten inputs & conversation metadata ────────────────────
    user_list  = _ensure_list(user_ctx)
    sys_list   = _ensure_list(sys_ctx)
    extra_list = extra_ctx or []
    recall_ids = recall_ids or []

    if not user_list:
        return {"merged": [], "merged_ids": [], "wm_ids": [], "history": [],
                "tools": [], "semantic": [], "assoc": []}

    primary = user_list[0]
    conv_id = primary.metadata.get("conversationid") or primary.metadata.get("conversation_id")
    user_id = primary.metadata.get("user_id")

    # ─── 2️⃣ Gather raw conversation segments ──────────────────────────
    segs = [
        c for c in self.repo.query(lambda c:
            c.domain=="segment"
            and c.semantic_label in ("user_input","assistant")
            and (c.metadata.get("conversationid")==conv_id
                 or c.metadata.get("conversation_id")==conv_id)
            and c.metadata.get("user_id") in (user_id, None)
        )
    ]
    # include extra_ctx and explicit recall_ids
    seen = {c.context_id for c in segs}
    for c in extra_list:
        if c.context_id not in seen:
            segs.append(c); seen.add(c.context_id)
    for rid in recall_ids:
        try:
            c = self.repo.get(rid)
            if c.context_id not in seen:
                segs.append(c); seen.add(c.context_id)
        except KeyError:
            pass
    segs.sort(key=lambda c: _to_dt(c.timestamp))

    # ─── 3️⃣ Working memory slice ─────────────────────────────────────
    WM_TURNS     = getattr(self, "max_history_items", 20)
    history_slice = segs[-WM_TURNS:]
    wm_ids        = [c.context_id for c in history_slice]

    # ─── Compute recall feature for RL gating ─────────────────────────
    rf = 0.0
    if wm_ids:
        activation_map = self.memman.spread_activation(
            seed_ids=wm_ids, hops=2, decay=0.6,
            assoc_weight=1.0, recency_weight=0.5
        )
        top_vals = sorted(activation_map.values(), reverse=True)[: len(wm_ids)]
        if top_vals:
            rf = sum(top_vals) / len(top_vals)

    # ─── 4️⃣ Semantic retrieval (RL-gated, relevance‐biased) ─────────
    semantic = []
    if self.rl.should_run("semantic_retrieval", rf):
        # fetch a few more candidates
        candidates = self.engine.query(
            stage_id="semantic_retrieval",
            similarity_to=user_text,
            top_k=getattr(self, "max_semantic_items", 10) * 2
        )
        now = datetime.utcnow()
        ttl = getattr(self, "context_ttl_days", 7)
        # scoring: combine past relevance_score and recency
        def _score(c):
            rel = float(c.metadata.get("relevance_score", 0.0) or 0.0)
            age_days = (now - _to_dt(c.timestamp)).total_seconds() / 86400
            recency = max(0.0, 1.0 - age_days / ttl)
            return rel * 0.7 + recency * 0.3
        candidates.sort(key=_score, reverse=True)
        semantic = candidates[: getattr(self, "max_semantic_items", 10) ]

    # ─── 5️⃣ Associative (memory) recall (RL-gated) ───────────────────
    assoc = []
    if self.rl.should_run("memory_retrieval", rf) and wm_ids:
        scores = self.memman.spread_activation(
            seed_ids=wm_ids, hops=3, decay=0.7,
            assoc_weight=1.0, recency_weight=0.5
        )
        top_ids = sorted(scores, key=scores.get, reverse=True)[: getattr(self, "max_memory_items", 10)]
        for cid in top_ids:
            try:
                assoc.append(self.repo.get(cid))
            except KeyError:
                pass

    # ─── 6️⃣ Recent tool outputs ──────────────────────────────────────
    tools = [
        c for c in self.repo.query(lambda c:
            c.component=="tool_output"
            and (c.metadata.get("conversationid")==conv_id
                 or c.metadata.get("conversation_id")==conv_id)
        )
    ]
    tools.sort(key=lambda c: _to_dt(c.timestamp))
    tools = tools[- getattr(self, "max_tool_outputs", 10):]

    # ─── 7️⃣ Prefix role labels ───────────────────────────────────────
    for c in sys_list + user_list + history_slice + semantic + assoc + tools:
        _prefix(c)

    # ─── 8️⃣ Merge in order & dedupe ─────────────────────────────────
    merged = []
    seen = set()
    def _add(lst):
        for c in lst:
            if c.context_id not in seen:
                merged.append(c)
                seen.add(c.context_id)
    _add(sys_list)
    _add(user_list)
    _add(history_slice)
    _add(semantic)
    _add(assoc)
    _add(tools)
    # ensure latest user turn is last
    last_u = user_list[-1]
    if last_u.context_id in seen:
        merged = [c for c in merged if c.context_id != last_u.context_id] + [last_u]

    merged_ids = [c.context_id for c in merged]

    # ─── 9️⃣ Debug banner ─────────────────────────────────────────────
    self._print_stage_context("retrieve_and_merge_context", {
        "merged_ids":   merged_ids[:12],
        "wm_ids":       wm_ids,
        "semantic_ids": [c.context_id for c in semantic],
        "assoc_ids":    [c.context_id for c in assoc],
        "tool_ids":     [c.context_id for c in tools],
        "recall_feat":  round(rf, 3),
    })

    # ─── 🔟 Return ─────────────────────────────────────────────────────
    return {
        "merged":     merged,
        "merged_ids": merged_ids,
        "wm_ids":     wm_ids,
        "history":    history_slice,
        "tools":      tools,
        "semantic":   semantic,
        "assoc":      assoc,
    }





def _stage4_intent_clarification(
    self,
    user_text: str,
    state: Dict[str, Any],
    *,
    on_token: Callable[[str],None] | None = None,
    ) -> "ContextObject":
    """
    Ask the Clarifier model to restate / expand the user's intent.

    Prompt includes:
      • All post-tool dialogue (otherwise last 8 turns)
      • The 8 turns *preceding* the current message (contextual glue)
      • Last 3 tool outputs (truncated)
      • Short semantic / associative / tool context snippets

    Returned JSON must contain:
        { "keywords": [], "notes": "", "debug_notes": [] }
    """
    import json, textwrap
    from context import ContextObject

    # ------------------------------------------------------------------ #
    # 0) Guards & shorthands                                             #
    # ------------------------------------------------------------------ #
    state          = state or {}
    merged         = state.get("merged", [])
    tool_ctxs      = state.get("tool_ctxs", [])
    semantic_ctxs  = state.get("semantic", [])
    assoc_ctxs     = state.get("assoc", [])
    tool_refs      = state.get("tools", [])

    # keep dialog ContextObjects in chronological order
    hist = [
        c for c in merged
        if c.semantic_label in ("user_input", "assistant")
    ]
    hist.sort(key=lambda c: c.timestamp)

    # ------------------------------------------------------------------ #
    # 1) Build “recent dialogue” (post-tool or fallback)                 #
    # ------------------------------------------------------------------ #
    last_tool_ts = max((tc.timestamp for tc in tool_ctxs), default=None)
    dialogue: list[str] = []

    for c in hist:
        if last_tool_ts and c.timestamp <= last_tool_ts:
            # skip dialogue that happened *before* the last tool run
            continue
        role  = "User" if c.semantic_label == "user_input" else "Assistant"
        text  = c.summary or c.metadata.get("text", "")
        dialogue.append(f"{role}: {text}")

    # Fallback → last 8 turns if post-tool block ended up empty
    if not dialogue:
        for c in hist[-8:]:
            role = "User" if c.semantic_label == "user_input" else "Assistant"
            dialogue.append(f"{role}: {c.summary or c.metadata.get('text', '')}")

    # Hard truncate dialog block
    dialog_block = "\n".join(dialogue)[-1500:] or "(none)"

    # ------------------------------------------------------------------ #
    # 2) Previous-turn snippet (8 lines before the current message)      #
    # ------------------------------------------------------------------ #
    prev_lines: list[str] = []
    if len(hist) >= 2:                     # guarantee at least one earlier turn
        for c in hist[-9:-1]:
            role = "User" if c.semantic_label == "user_input" else "Assistant"
            prev_lines.append(f"{role}: {c.summary or c.metadata.get('text', '')}")
    prev_block = "\n".join(prev_lines) if prev_lines else "(none)"

    # ------------------------------------------------------------------ #
    # 3) Last 3 tool outputs                                             #
    # ------------------------------------------------------------------ #
    tool_lines: list[str] = []
    for tc in sorted(tool_ctxs, key=lambda c: c.timestamp)[-3:]:
        payload = tc.metadata.get("output") or tc.metadata.get("exception") or ""
        try:
            blob = (
                payload
                if isinstance(payload, str)
                else json.dumps(payload, ensure_ascii=False)
            )
        except Exception:
            blob = repr(payload)
        if len(blob) > 950:
            blob = blob[:950] + " …"
        tool_lines.append(f"[{tc.stage_id}] {blob}")
    tools_block = "\n".join(tool_lines) if tool_lines else "(none)"

    # ------------------------------------------------------------------ #
    # 4) Semantic / associative / tool reference snippets                #
    # ------------------------------------------------------------------ #
    def _first_n(ctxs, n=3):
        out = []
        for c in ctxs[:n]:
            short = (c.summary or "")[:120].replace("\n", " ")
            out.append(f"• {short}  (id={c.context_id[:8]})")
        return out

    semantic_block = "\n".join(_first_n(semantic_ctxs)) or "(none)"
    assoc_block    = "\n".join(_first_n(assoc_ctxs))    or "(none)"
    tools_block2   = "\n".join(_first_n(tool_refs))      or "(none)"

    # ------------------------------------------------------------------ #
    # 5) Assemble full system/context prompt                             #
    # ------------------------------------------------------------------ #
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

    # cap entire context to 4 kB to protect model window
    MAX_PROMPT_CHARS = 4096
    if len(full_ctx) > MAX_PROMPT_CHARS:
        full_ctx = full_ctx[-MAX_PROMPT_CHARS:]

    # ------------------------------------------------------------------ #
    # 6) Call the Clarifier model                                        #
    # ------------------------------------------------------------------ #
    msgs = [
        {"role": "system", "content": clar_sys},
        {"role": "system", "content": full_ctx},
        {"role": "user",   "content": user_text},
    ]
    out = self._stream_and_capture(
        self.primary_model,                       # ← use primary model
        msgs,
        tag="[Clarifier]",
        images=state.get("images"),
    ).strip()

    # ------------------------------------------------------------------ #
    # 7) Parse / repair JSON                                             #
    # ------------------------------------------------------------------ #
    def _as_json(raw: str) -> dict | None:
        try:
            data = json.loads(raw)
            if (
                isinstance(data, dict)
                and "keywords" in data
                and "notes"    in data
            ):
                return data
        except Exception:
            pass
        return None

    clar = _as_json(out)


    # Final fallback: wrap raw text
    if clar is None:
        clar = {
            "keywords": [],
            "notes": out,
            "debug_notes": dialogue[-8:],
        }

    # guarantee debug_notes
    clar.setdefault("debug_notes", dialogue[-8:])

    # ------------------------------------------------------------------ #
    # 8) Persist Clarifier Context                                       #
    # ------------------------------------------------------------------ #
    input_refs = [state["user_ctx"].context_id] if state.get("user_ctx") else []
    clar_ctx = ContextObject.make_stage(
        "intent_clarification",
        input_refs=input_refs,
        output=clar,
    )
    clar_ctx.metadata.update(clar)            # keep keywords / notes
    clar_ctx.stage_id       = "intent_clarification"
    clar_ctx.semantic_label = "intent_clarification"
    clar_ctx.tags.append("clarifier")

    # propagate conversation / user ids if available
    if state.get("user_ctx"):
        clar_ctx.metadata.update(
            {
                "conversation_id": state["user_ctx"].metadata["conversation_id"],
                "user_id": state["user_ctx"].metadata["user_id"],
            }
        )

    clar_ctx.summary = clar.get("notes", "")[:250]

    clar_ctx.touch()
    self.repo.save(clar_ctx)
    # embed for later retrieval
    #self.memman.register_relationships(clar_ctx, self.embed_text)

    return clar_ctx

# ──────────────────────────────────────────────────────────────────
# _stage5_external_knowledge   (upgraded)
# ──────────────────────────────────────────────────────────────────
def _stage5_external_knowledge(
    self,
    clar_ctx: "ContextObject",
    state: Dict[str, Any] | None = None,
) -> "ContextObject":
    """
    Build a ranked “external knowledge” ContextObject for the planner.

    Signal sources (in trust order):

      • Recent dialogue turns            (last 6)
      • Recent tool outputs              (last 6)
      • Semantic recalls                 (saved in state["semantic"])
      • Associative holographic recall   (MemoryManager.holographic_recall)
      • Fresh similarity hits            (engine.query, recency‑boosted)

    Score  =  0.55 · similarity   +   0.25 · recency_boost   +   0.20 · assoc
    (dialogue / tool snippets keep max score)

    Top `MAX_SNIPPETS` unique snippets are kept and persisted.
    """
    import json, math, time
    from datetime import datetime, timezone
    from context import ContextObject

    # ─── tunables ───────────────────────────────────────────────────
    MAX_SNIPPETS        = 12
    MAX_PER_CATEGORY    = 6
    SIM_TOP_K           = max(3, getattr(self, "top_k", 3))
    HALF_LIFE_DAYS      = 3.0
    NOW_TS              = time.time()

    state = state or {}

    # ---------- helpers --------------------------------------------
    def _recency_boost(ctx) -> float:
        ts = ctx.timestamp
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0
        age_days = max((NOW_TS - ts) / 86400.0, 0.0)
        return 0.5 ** (age_days / HALF_LIFE_DAYS)

    def _label_trim(text: str, lbl: str, limit: int = 220) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0] + " …"
        return f"({lbl}) {text}"

    scored: list[tuple[float, str, ContextObject]] = []
    seen_texts: dict[str, float] = {}   # text → best_score

    # ---------- 1) recent dialogue ---------------------------------
    for c in reversed(state.get("history", [])[-MAX_PER_CATEGORY:]):
        txt = c.summary or c.metadata.get("text", "")
        s   = _label_trim(txt, "USER" if c.semantic_label == "user_input" else "ASSIST")
        score = 1.0
        scored.append((score, s, c))
        seen_texts[s] = score

    # ---------- 2) recent tool outputs -----------------------------
    for c in reversed(state.get("tools", [])[-MAX_PER_CATEGORY:]):
        payload = c.metadata.get("output") or c.metadata.get("exception") or ""
        if not isinstance(payload, str):
            try:    payload = json.dumps(payload, ensure_ascii=False)[:300]
            except: payload = repr(payload)[:300]
        s = _label_trim(payload, f"TOOL:{c.stage_id}")
        score = 1.0
        scored.append((score, s, c))
        seen_texts[s] = score

    # ---------- 3) semantic & assoc recalls from previous stages ---
    for lbl, key in (("SEM", "semantic"), ("ASSOC", "assoc")):
        for c in state.get(key, [])[:MAX_PER_CATEGORY]:
            txt = c.summary or ""
            s   = _label_trim(txt, lbl)
            if s in seen_texts:
                continue
            sim = c.retrieval_score or 0.7
            scored.append((sim, s, c))
            seen_texts[s] = sim

    # ---------- 4) holographic associative recall ------------------
    seed_ids = [clar_ctx.context_id] + [c.context_id for c in state.get("history", [])[-2:]]
    assoc_hits = self.memman.holographic_recall(
        cue_ids=seed_ids,
        cue_text=clar_ctx.summary or "",
        hops=2,
        top_n=MAX_PER_CATEGORY,
        embed_fn=self.embed_text
    )
    for h in assoc_hits:
        txt = h.summary or h.metadata.get("content", "")
        s   = _label_trim(txt, "HMM")
        if s in seen_texts:
            continue
        assoc = h.retrieval_score or 0.5
        rec   = _recency_boost(h)
        score = 0.20 * assoc + 0.25 * rec + 0.55 * assoc  # assoc doubles as similarity proxy
        scored.append((score, s, h))
        seen_texts[s] = score

    # ---------- 5) fresh similarity hits (recency‑boosted) ---------
    kws = clar_ctx.metadata.get("keywords") or []
    if not kws and clar_ctx.summary:
        kws = [clar_ctx.summary]

    for kw in kws:
        for h in self.engine.query(similarity_to=kw,
                                   stage_id="external_knowledge_query",
                                   top_k=SIM_TOP_K):
            txt = (h.summary or h.metadata.get("content", "")).strip()
            s   = _label_trim(txt, "FRESH")
            if s in seen_texts:
                continue
            sim  = h.retrieval_score or 0.0
            rec  = _recency_boost(h)
            score = 0.55 * sim + 0.25 * rec + 0.20 * 0.0   # no assoc for engine hits
            scored.append((score, s, h))
            seen_texts[s] = score

    # ---------- 6) final ranking & de‑dup --------------------------
    scored.sort(key=lambda t: t[0], reverse=True)
    uniq_lines = []
    added = set()
    for _, txt, _ in scored:
        if txt not in added:
            uniq_lines.append(txt)
            added.add(txt)
        if len(uniq_lines) >= MAX_SNIPPETS:
            break

    # ---------- 7) persist ContextObject ---------------------------
    ext_ctx = ContextObject.make_stage(
        "external_knowledge_retrieval",
        input_refs=[clar_ctx.context_id],
        output={"snippets": uniq_lines},
    )
    ext_ctx.stage_id       = "external_knowledge_retrieval"
    ext_ctx.semantic_label = "external_knowledge"
    ext_ctx.tags.append("external")
    ext_ctx.summary = "\n".join(uniq_lines)[:1024]
    ext_ctx.touch()
    self.repo.save(ext_ctx)
    self.memman.register_relationships(ext_ctx, self.embed_text)

    # ---------- 8) debug print -------------------------------------
    self._print_stage_context(
        "external_knowledge_retrieval",
        {"chosen_snippets": uniq_lines, "total_candidates": len(scored)},
    )

    # expose snippets to downstream stages
    if state is not None:
        state["knowledge_snippets"] = uniq_lines

    return ext_ctx

def _stage5b_build_planning_kg(self, clar_ctx, know_ctx, tools_list, state):
    import json
    nodes, edges = {}, []
    # tool + param nodes
    for t in tools_list:
        tn = f"tool:{t['name']}"; nodes[tn] = {"type":"tool"}
        props = (t.get("schema",{}).get("parameters",{}).get("properties",{}) or {})
        for p in props:
            pn = f"param:{t['name']}.{p}"; nodes[pn] = {"type":"param"}
            edges.append((tn, "has_param", pn))
    # concepts from clarifier + snippets
    kws = (clar_ctx.metadata.get("keywords") or []) + re.findall(r"\b[A-Za-z][\w-]{2,}\b", know_ctx.summary or "")
    kws = list(dict.fromkeys(kws))[:50]
    for kw in kws:
        cn = f"concept:{kw}"; nodes[cn] = {"type":"concept"}

    # simple affinity: embedding cosine between kw and tool/param descriptions
    def emb(x): return self.embed_text(x or "")
    tool_desc = {t['name']: (t['description'] or "") for t in tools_list}
    kw_vecs = {kw: emb(kw) for kw in kws}
    t_vecs  = {name: emb(desc) for name,desc in tool_desc.items()}
    def cos(a,b): 
        import numpy as np
        a=np.array(a); b=np.array(b)
        den = (np.linalg.norm(a)*np.linalg.norm(b)) or 1.0
        return float(np.dot(a,b)/den)
    affinities = []
    for kw in kws:
        for name in tool_desc:
            score = cos(kw_vecs[kw], t_vecs[name])
            if score >= 0.25:
                affinities.append((f"concept:{kw}", "affinity", f"tool:{name}", score))
    affinities.sort(key=lambda x: x[3], reverse=True)
    top_pairs = affinities[: min(100, len(affinities))]

    kg = {"nodes": nodes, "edges": edges + [(a,b,c) for a,b,c,_ in top_pairs],
          "top_tool_candidates": sorted(
              {c.split(':',1)[1] for _,_,c,_ in top_pairs}, key=lambda n: max(s for a,b,c,s in top_pairs if c.endswith(n)), reverse=True)[:10]
    }

    kg_ctx = ContextObject.make_knowledge("planning_kg", kg, tags=["planning","kg"])
    kg_ctx.summary = json.dumps({"top_tool_candidates": kg["top_tool_candidates"]}, ensure_ascii=False)
    self.repo.save(kg_ctx)
    self.memman.register_relationships(kg_ctx, self.embed_text)
    state["planning_kg"] = kg
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
    Updated for DAG + TurnState (+ robustness + schema/hint replan):
    ...
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

    def _is_valid_plan_obj(obj: dict) -> bool:
        if not isinstance(obj, dict):
            return False
        if "graph" in obj and isinstance(obj["graph"], dict):
            nodes = obj["graph"].get("nodes")
            return isinstance(nodes, list) and any(isinstance(n, dict) and n.get("tool") for n in nodes)
        if "tasks" in obj and isinstance(obj["tasks"], list):
            return any(isinstance(t, dict) and t.get("call") for t in obj["tasks"])
        return False

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
            and (c.metadata.get("status") in ("confirmed","success","refined"))  # prefer confirmed/success; allow refined
            and c.metadata.get("tool_name") in names
        )
        hints: Dict[str, List[str]] = {}
        for r in rows:
            tname = r.metadata.get("tool_name")
            text  = r.metadata.get("hint") or r.summary or ""
            if not text:
                # synthesize from keys if missing
                req  = r.metadata.get("schema_required") or []
                ex   = r.metadata.get("filled_params") or {}
                text = f"When calling {tname}, include required {req}. Example keys used previously: {list(ex.keys())}."
            hints.setdefault(tname, [])
            if text not in hints[tname]:
                hints[tname].append(text)
        # keep up to 3 per tool
        for k in list(hints.keys()):
            hints[k] = hints[k][:3]
        return hints

    def _persist_retry_candidate(tool_name: str, required: List[str], props: Dict[str, Any],
                                 filled: Dict[str, Any], turn_id: str, plan_id: str) -> None:
        """Save a candidate retry-critique row to be promoted later on success."""
        from context import ContextObject as _CO
        # Make a short, reusable hint text
        arg_list = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in filled.items()) if filled else ""
        req_list = ", ".join(required) if required else "(none)"
        hint = f"When using {tool_name}, supply required [{req_list}]. Example: {tool_name}({arg_list})"
        meta = {
            "tool_name": tool_name,
            "status": "refined",  # candidate; can be promoted to 'confirmed' later
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
        # propagate ids if available
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
        "Your last plan may be incomplete—**OUTPUT ONLY** the JSON, no extra text.\n\n"
        f"Available tools:\n{tool_lines}"
    )

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
                    continue  # skip nameless entries; avoids None()
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
    selected_names_1 = _extract_calls(first_plan)
    selected_names_1 = [n for n in selected_names_1 if n in valid_tool_names]
    schema_map_all = {t["name"]: (t.get("schema") or {}) for t in tools_list}
    selected_schema_catalog = {n: schema_map_all.get(n, {}) for n in dict.fromkeys(selected_names_1)}
    retry_hints_map = _load_retry_hints(selected_names_1)

    # Build replan system with schemas + hints (only selected tools)
    replan_system = replan_system_base
    if selected_schema_catalog:
        replan_system += "\n\n[Selected Tool Schemas]\n" + json.dumps(selected_schema_catalog, ensure_ascii=False)
    if retry_hints_map:
        # compact, per-tool bullet list
        lines = []
        for nm, hints in retry_hints_map.items():
            for h in hints:
                lines.append(f"- ({nm}) {h}")
        replan_system += "\n\n[Retry Hints]\n" + "\n".join(lines[:12])

    # Second pass (schema + hints). Use fewer snippets to make room.
    half_snips = original_snips[: max(1, len(original_snips)//2)]
    second_plan = _run_planner(replan_system, build_user(half_snips), tag="[PlannerReplanSchemas]")

    # Choose better of the two: prefer second if valid & non-empty; else fallback
    plan_obj: dict = second_plan if _is_valid_plan_obj(second_plan) else first_plan
    if not _is_valid_plan_obj(plan_obj):
        # one more minimal attempt with aggressive truncation (still includes schemas/hints if any)
        tiny_snips = original_snips[:1]
        third_plan = _run_planner(replan_system, build_user(tiny_snips), tag="[PlannerReplanMin]")
        if _is_valid_plan_obj(third_plan):
            plan_obj = third_plan

    # ──────────────────────────────────────────────────────────────────
    # 5) Refine each chosen tool with schema (fill only missing)
    #    + persist candidate retry-critique records
    # ──────────────────────────────────────────────────────────────────
    schema_map = schema_map_all  # already built

    def refine_single_tool(task: dict) -> dict:
        name   = task.get("call")
        schema = schema_map.get(name, {}) or {}
        required = (schema.get("parameters", {}) or {}).get("required", []) or []
        props    = (schema.get("parameters", {}) or {}).get("properties", {}) or {}

        refined = {
            "call":       name,
            "tool_input": dict(task.get("tool_input", {}) or {}),
            "subtasks":   list(task.get("subtasks", []) or []),
        }

        # Only attempt to fix missing requireds (no extra fields invented)
        missing = [p for p in required if p not in refined["tool_input"]]
        if not missing or not schema or not name:
            return refined

        prompt = {
            "description": "Fill only the truly missing required parameters for this tool call. Do not add extra keys.",
            "missing":     missing,
            "schema":      schema,
            "call":        refined,
            "user_text":   user_text,
            "clar_notes":  (clar_ctx.metadata.get("notes") or clar_ctx.summary or ""),
        }
        out = self._stream_and_capture(
            self.secondary_model,
            [
                {"role": "system", "content": "Return ONLY JSON {\"call\":{...}} with missing params filled. No extra keys."},
                {"role": "user",   "content": json.dumps(prompt, ensure_ascii=False)}
            ],
            tag=f"[PlannerRefine_{name}]",
            images=state.get("images"),
        ).strip()
        filled_now = {}
        try:
            cand = json.loads(_clean_json_block(out)).get("call", {})
            ti   = cand.get("tool_input", {}) or {}
            # accept only keys that exist in props
            for k, v in ti.items():
                if k in props:
                    refined["tool_input"][k] = v
                    filled_now[k] = v
        except Exception:
            pass

        # Persist a candidate retry-critique for this tool if we actually filled anything
        if filled_now:
            _persist_retry_candidate(
                tool_name=name,
                required=list(required),
                props=props,
                filled=filled_now,
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
    # Provide executor hints; executor will honor these if present
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
        # keep optional 'alias' if present

    _inject_implicit_deps(graph)                # <── NEW
    errs = _validate_graph(graph, tools_list)   # <── NEW
    if errs:
        # Surface to subsequent stages/prompts and persist a small marker
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
    state["plan_ctx"]        = ctx
    state["plan_output"]     = {"graph": graph}  # keep as object for DAG stages
    state["tools_list"]      = tools_list
    state["tc_ctx"]          = None
    state["plan_output_prev"] = json.dumps(first_plan, ensure_ascii=False)  # for diagnostics / reflection

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


def _stage8_orchestrate(
    self,
    user_text: str,
    state: Dict[str, Any],
) -> Tuple[str, List[ContextObject]]:
    """
    Top-level orchestrator: plan → execute → (maybe replan) → finalize.
    Returns (assistant_reply, all_tool_outputs)
    """

    # ─── 0) Make sure conversation_id & user_id exist for stamping ───
    state.setdefault(
        "conversation_id",
        getattr(self, "_active_conversation_id", uuid.uuid4().hex)
    )
    state.setdefault(
        "user_id",
        getattr(self, "current_user_id", "anon")
    )

    # ─── 1) Planner: from clarifier + knowledge → graph
    clar_ctx = state["clar_ctx"]
    know_ctx = state["know_ctx"]
    tools    = state.get("tools_list", [])

    plan_ctx, plan_json = self._stage7_planning_summary(
        clar_ctx   = clar_ctx,
        know_ctx   = know_ctx,
        tools_list = tools,
        user_text  = user_text,
        state      = state,
    )
    state["plan_ctx"]    = plan_ctx
    state["plan_output"] = plan_json

    all_tool_ctxs: List[ContextObject] = []

    # ─── 2) Execute + reflect/replan until done ───────────────────────
    while True:
        # 2a) Run any ready DAG nodes
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

        # 2b) Reflection & maybe re-plan
        replan = self._stage9b_reflection_and_replan(
            tool_ctxs    = tcs,
            plan_output  = state["plan_output"],
            user_text    = user_text,
            clar_metadata= clar_ctx.metadata,
            state        = state,
        )
        # None means “OK” — keep the same graph
        if replan is None:
            continue

        # otherwise swap in the new plan
        state["plan_output_prev"] = state["plan_output"]
        state["plan_output"]      = replan

    # ─── 3) Final answer assembly ─────────────────────────────────────
    reply = self._stage10_assemble_and_infer(user_text=user_text, state=state)

    # ─── 4) Optional safety / polish ──────────────────────────────────
    polished = self._stage10b_response_critique_and_safety(
        draft     = reply,
        user_text = user_text,
        tool_ctxs = all_tool_ctxs,
        state     = state,
    )
    if polished:
        reply = polished

    # ─── 5) Memory write-back ─────────────────────────────────────────
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
    tool_ctxs: List[ContextObject],
    plan_output: str,
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
    def _clean_json_block(text: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if m:
            return m.group(1)
        m2 = re.search(r"(\{.*\})", text, flags=re.S)
        return (m2.group(1) if m2 else (text or "")).strip()

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
        # (same normaliser as before – omitted here for brevity)
        from copy import deepcopy
        def _flatten(task, parent, idx, acc):
            idx[0] += 1
            nid = task.get("id") or f"n{idx[0]}"
            acc.append(
                {"id": nid, "tool": task.get("call"), "args": task.get("tool_input", {}), "after": [parent] if parent else []}
            )
            for sub in task.get("subtasks", []) or []:
                _flatten(sub, nid, idx, acc)

        if isinstance(plan_any, dict) and "graph" in plan_any:
            g = deepcopy(plan_any["graph"])
            g.setdefault("nodes", [])
            g.setdefault("meta", {})
            return {"graph": g}

        if isinstance(plan_any, dict) and "tasks" in plan_any:
            nodes, counter = [], [0]
            for t in plan_any["tasks"]:
                _flatten(t, None, counter, nodes)
            return {"graph": {"nodes": nodes, "meta": plan_any.get("meta", {})}}

        # bare single task
        if isinstance(plan_any, dict) and "call" in plan_any:
            return _as_graph({"tasks": [plan_any]})

        return {"graph": {"nodes": [], "meta": {}}}

    # deps-satisfied util
    def _deps_ok(n: dict, done: set[str]) -> bool:
        return all(d in done for d in (n.get("after") or []))

    # ───────────────────── current turn / plan state ──────────────────
    turn = _ensure_turn_state(state)

    cur_graph = _as_graph(json.loads(_clean_json_block(plan_output)) if plan_output else {}).get("graph")
    cur_sig   = _sig(cur_graph)

    # safe tracker lookup (works whether repo.query returns list or iterator)
    trackers = self.repo.query(
        lambda c: c.component == "plan_tracker"
        and (
            c.metadata.get("plan_id") == getattr(turn, "plan_id", None)
            or c.semantic_label == cur_sig
        )
    )
    if isinstance(trackers, list):
        tracker = trackers[0] if trackers else None
    else:                                 # generator / iterator
        tracker = next(trackers, None)

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
        oid = tc.metadata.get("node_id")
        tnm = tc.metadata.get("tool_name")
        payload = tc.metadata.get("output_short") or tc.metadata.get("output_full")
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
    ).strip()

    # ───────────────— quick exits: OK / same graph —───────────────────
    if re.fullmatch(r"(?i)(ok|okay)[.!]?", reply_raw):
        return None

    try:
        new_plan = json.loads(_clean_json_block(reply_raw))
    except Exception:
        # unparsable – bubble raw string upwards
        return reply_raw

    new_norm = _as_graph(new_plan)
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

    if not draft:
        return draft

    # ─── 1) Build blocks ───────────────────────────────────────────
    # Use only the original user_text here:
    user_block  = "[Latest user question]\n" + user_text
    draft_block = "[Draft response]\n"       + draft

    # Merge snippets as before
    merged = state.get("merged", [])
    merged_texts = "\n\n".join(f"[{c.stage_id}] {c.summary}" for c in merged) or "(none)"
    merged_block = "[Merged context snippets]\n" + merged_texts

    # Plan block
    plan_txt = state.get("plan_output", "(no plan)")
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
            except:
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

    if polished == draft.strip():
        return polished

    # ─── 5) diff & dynamic_patch (unchanged) ────────────────────────
    diff = difflib.unified_diff(draft.splitlines(), polished.splitlines(), lineterm="", n=1)
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