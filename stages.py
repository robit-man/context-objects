# stages.py

import json
import re
import os
import ast
import inspect
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

import numpy as np
from ollama import chat, embed
from tools import Tools
from context import (
    ContextObject,
    default_clock,
    HybridContextRepository,
    MemoryManager,
)
# if your stage refers to other parts of your assembler you may need to
# import them too; adjust as needed.
# ——— NEW helper ———

def _utc_iso() -> str:
    """UTC timestamp ending with 'Z' (e.g. 2025-07-07T18:04:31.123456Z)."""
    from datetime import datetime
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
import uuid
def _ensure_ids(ctx, conv_id, user_id):
    ctx.metadata.setdefault("conversation_id", conv_id)
    ctx.metadata.setdefault("user_id",         user_id)

def _stage1_record_input(self, user_text: str, state: Dict[str, Any]) -> ContextObject:
    ctx = ContextObject.make_segment("user_input", [], tags=["user_input"])
    ctx.summary = user_text
    ctx.stage_id = "user_input"

    # 🔑  inject routing metadata so Stage 3 can find it later
    ctx.metadata.update({
        "conversation_id": state["conversation_id"],
        "user_id":         state["user_id"],
    })

    ctx.touch()
    self.repo.save(ctx)
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
    Build the conversation memory window for downstream stages.

    Returns:
      merged        →  List[ContextObject]  (entire ordered context window)
      merged_ids    →  List[str]            (ids of ^)
      wm_ids        →  List[str]            (ids of working-memory slice)
      history       →  last 8 user/assistant ContextObjects
      tools         →  recent tool-output ContextObjects
      semantic      →  semantic recall ContextObjects
      assoc         →  associative recall ContextObjects
    """
    from datetime import datetime, timedelta
    now = datetime.utcnow()

    # If there's no user context, nothing to build
    if user_ctx is None:
        return {
            "merged":       [],
            "merged_ids":   [],
            "wm_ids":       [],
            "history":      [],
            "tools":        [],
            "semantic":     [],
            "assoc":        [],
        }

    # Helpers
    def _to_dt(ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts.rstrip("Z"))
        except Exception:
            return now

    def _ensure_list(x):
        if x is None:
            return []
        return x if isinstance(x, list) else [x]

    # Flatten inputs
    user_list  = _ensure_list(user_ctx)
    sys_list   = _ensure_list(sys_ctx)
    extra_list = extra_ctx or []

    # Pull conversation & user IDs from the first user_ctx
    primary = user_list[0]
    conv_id = (
        primary.metadata.get("conversationid")
        or primary.metadata.get("conversation_id")
    )
    user_id = primary.metadata.get("user_id")

    # 1. Load raw conversation segments
    segs = [
        c for c in self.repo.query(lambda c:
            c.metadata.get("conversationid") == conv_id
            or c.metadata.get("conversation_id") == conv_id
        )
        if c.domain == "segment"
        and c.metadata.get("user_id") in (user_id, None)
        and c.semantic_label in ("user_input", "assistant")
    ]

    # 1a. Prepend extra_ctx
    seen_ids = {c.context_id for c in segs}
    for c in extra_list:
        if c.context_id not in seen_ids:
            segs.append(c)
            seen_ids.add(c.context_id)

    # 1b. Include explicit recall_ids
    if recall_ids:
        for rid in recall_ids:
            try:
                c = self.repo.get(rid)
                if c.context_id not in seen_ids:
                    segs.append(c)
                    seen_ids.add(c.context_id)
            except KeyError:
                pass

    segs.sort(key=lambda c: _to_dt(c.timestamp))

    # 2. Working-memory & short-term slices
    WM_TURNS   = 20
    ST_MINUTES = 120
    wm_slice   = segs[-WM_TURNS:]
    cutoff     = now - timedelta(minutes=ST_MINUTES)
    st_slice   = [c for c in segs if _to_dt(c.timestamp) >= cutoff]

    # 3. Semantic & associative recall
    semantic, assoc = [], []
    if self.rl.should_run("semantic_retrieval", 0.0):
        hits = self.engine.query(
            stage_id="semantic_retrieval",
            similarity_to=user_text,
            exclude_tags=self.STAGES + ["tool_schema", "tool_output"],
            top_k=self.top_k,
        )
        semantic = [
            h for h in hits
            if (h.metadata.get("conversationid") == conv_id
                or h.metadata.get("conversation_id") == conv_id)
        ]

    if self.rl.should_run("memory_retrieval", 0.0):
        seeds  = [primary.context_id]
        scores = self.memman.spread_activation(seeds, hops=3, decay=0.7)
        for cid in sorted(scores, key=scores.get, reverse=True)[: self.top_k]:
            try:
                c = self.repo.get(cid)
                if (c.metadata.get("conversationid") == conv_id
                    or c.metadata.get("conversation_id") == conv_id):
                    c.retrieval_score = scores[cid]
                    assoc.append(c)
            except KeyError:
                pass

    # 4. Recent tool outputs
    recent_tools = [
        c for c in self.repo.query(lambda c:
            c.component == "tool_output"
            and ((c.metadata.get("conversationid") == conv_id)
                 or (c.metadata.get("conversation_id") == conv_id))
        )
    ]
    recent_tools.sort(key=lambda c: _to_dt(c.timestamp))
    recent_tools = recent_tools[-self.top_k:]

    # 5. Prefix summaries once
    def _prefix(ctx):
        if not getattr(ctx, "summary", None):
            return
        if ctx.summary.lstrip().startswith(("User:", "Assistant:")):
            return
        role = "Assistant" if ctx.semantic_label == "assistant" else "User"
        ctx.summary = f"{role}: {ctx.summary or ''}"

    for c in sys_list + user_list + segs + semantic + assoc + recent_tools:
        _prefix(c)

    # 6. Assemble merged list, de-duped
    merged, seen = [], set()
    def _add(objs):
        for c in objs:
            cid = c.context_id
            if cid not in seen:
                merged.append(c)
                seen.add(cid)

    _add(sys_list)
    _add(user_list)
    _add(wm_slice)
    _add(st_slice)
    _add(semantic)
    _add(assoc)
    _add(recent_tools)

    # 7. Build return dict
    result = {
        "merged":      merged,
        "merged_ids":  [c.context_id for c in merged],
        "wm_ids":      [c.context_id for c in wm_slice],
        "history":     merged[-8:],
        "tools":       recent_tools,
        "semantic":    semantic,
        "assoc":       assoc,
    }

    # 8. Debug logging
    self._print_stage_context("retrieve_and_merge_context", {
        "total_segments": len(segs),
        "merged_ids":     result["merged_ids"][:12],
        "wm_ids":         result["wm_ids"],
        "semantic_ids":   [c.context_id for c in semantic],
        "assoc_ids":      [c.context_id for c in assoc],
        "tool_ids":       [c.context_id for c in recent_tools],
    })

    return result




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


def _stage5_external_knowledge(
    self,
    clar_ctx: "ContextObject",
    state: Dict[str, Any] | None = None,
) -> "ContextObject":
    """
    Build a compact “external knowledge” context for the planner from:
      • recent dialogue/history
      • recent tool outputs
      • semantic recalls
      • associative recalls
      • fresh engine.query hits based on clarifier keywords
    """
    import json, textwrap
    from datetime import datetime
    from context import ContextObject

    state = state or {}

    # ------------------------------------------------------------------ #
    # 1) Clarifier keywords → fallback to summary if empty               #
    # ------------------------------------------------------------------ #
    kws = clar_ctx.metadata.get("keywords") or []
    if not kws and clar_ctx.summary:
        kws = [clar_ctx.summary]

    # ------------------------------------------------------------------ #
    # 2) Fresh similarity hits                                           #
    # ------------------------------------------------------------------ #
    TOP_K = max(3, self.top_k)
    fresh_hits: list[tuple["ContextObject", str]] = []
    seen_ids: set[str] = set()

    for kw in kws:
        hits = self.engine.query(
            similarity_to=kw,
            stage_id="external_knowledge_query",
            top_k=TOP_K,
        )
        for h in hits:
            if h.context_id in seen_ids:
                continue
            seen_ids.add(h.context_id)
            txt = (h.summary or h.metadata.get("content", "")).strip()
            fresh_hits.append((h, txt))
            if len(fresh_hits) >= TOP_K:
                break
        if len(fresh_hits) >= TOP_K:
            break

    # ------------------------------------------------------------------ #
    # 3) Helper to label + truncate                                      #
    # ------------------------------------------------------------------ #
    def _label_and_trim(text: str, lbl: str, limit: int = 200) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) > limit:
            text = text[: limit].rsplit(" ", 1)[0] + " …"
        return f"({lbl}) {text}"

    lines: list[str] = []

    # a) dialogue / history
    for c in state.get("history", []):
        role = "USER" if c.semantic_label == "user_input" else "ASSIST"
        lines.append(_label_and_trim(c.summary or c.metadata.get("text", ""), role))

    # b) recent tool outputs
    for c in state.get("tools", []):
        payload = c.metadata.get("output") or c.metadata.get("exception") or ""
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload, ensure_ascii=False)
            except Exception:
                payload = repr(payload)
        lines.append(_label_and_trim(payload, f"TOOL:{c.stage_id}"))

    # c) semantic recalls
    for c in state.get("semantic", []):
        lines.append(_label_and_trim(c.summary or "", "SEM"))

    # d) associative recalls
    for c in state.get("assoc", []):
        lines.append(_label_and_trim(c.summary or "", "ASSOC"))

    # e) fresh similarity hits
    for ctx, txt in fresh_hits:
        lines.append(_label_and_trim(txt, "FRESH"))

    # Hard cap: keep the most recent ~4·TOP_K snippets
    lines = lines[: TOP_K * 4]

    # ------------------------------------------------------------------ #
    # 4) Persist ContextObject                                           #
    # ------------------------------------------------------------------ #
    ext_ctx = ContextObject.make_stage(
        "external_knowledge_retrieval",
        input_refs=[clar_ctx.context_id],
        output={"snippets": lines},
    )
    ext_ctx.stage_id       = "external_knowledge_retrieval"
    ext_ctx.semantic_label = "external_knowledge"
    ext_ctx.tags.append("external")
    ext_ctx.summary = "\n".join(lines)[:1024]  # store a concise summary

    ext_ctx.touch()
    self.repo.save(ext_ctx)
    # 🔑  **register embedding so future similarity queries can find it**
    #self.memman.register_relationships(ext_ctx, self.embed_text)

    # ------------------------------------------------------------------ #
    # 5) Optional debug print                                            #
    # ------------------------------------------------------------------ #
    self._print_stage_context(
        "external_knowledge_retrieval",
        {"snippets": lines, "total": len(lines)},
    )

    return ext_ctx




def _stage6_prepare_tools(self) -> List[Dict[str, Any]]:
    """
    Return a de-duplicated, lexicographically sorted list of:
      { "name": "<tool_name>",
        "description": "<one-line truncated desc>",
        "schema": <full JSON-RPC schema dict>
      }
    for every tool_schema in the repo.
    """
    import json, textwrap

    # 1) Load every schema context object tagged "tool_schema"
    rows = self.repo.query(
        lambda c: c.component == "schema" and "tool_schema" in c.tags
    )

    # 2) Keep only the newest per tool name
    buckets: Dict[str, Any] = {}
    for ctx in rows:
        try:
            blob = json.loads(ctx.metadata["schema"])
            name = blob["name"]
        except Exception:
            continue
        if name not in buckets or ctx.timestamp > buckets[name].timestamp:
            buckets[name] = ctx

    # 3) Build the list, sorted by tool name, including truncated description + full schema
    tool_defs: List[Dict[str, Any]] = []
    for name in sorted(buckets):
        blob = json.loads(buckets[name].metadata["schema"])
        full_desc = blob.get("description", "").split("\n", 1)[0]
        short_desc = textwrap.shorten(full_desc, width=60, placeholder="…")
        tool_defs.append({
            "name":        name,
            "description": short_desc,
            "schema":      blob,
        })

    return tool_defs


def _stage7_planning_summary(
    self,
    clar_ctx: ContextObject,
    know_ctx: ContextObject,
    tools_list: List[Dict[str, Any]],
    user_text: str,
    state: Dict[str, Any],
) -> Tuple[ContextObject, str]:
    """
    1) Load the latest planning_prompt artifact (or fallback to config)
    2) Inject a “Conversation so far” block from merged context
    3) Inject truncated tool list (name + first sentence of description)
    4) Run up to 3 JSON-only planning passes, halving snippets each retry
    5) For each selected tool, refine the call against its schema
    6) Persist artefacts & plan tracker; return (ctx, raw-JSON)
    """

    import json, re, hashlib, datetime, textwrap
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ------------------------------------------------------------------ #
    # 0️⃣  Diagnostic print                                             #
    # ------------------------------------------------------------------ #
    incoming = {
        "user_text":       user_text,
        "clarifier_notes": clar_ctx.summary,
        "knowledge_snips": len((know_ctx.summary or "").splitlines()),
        "merged_len":      len(state.get("merged", [])),
        "history_len":     len(state.get("history",   [])),
        "tools_len":       len(state.get("tools",     [])),
        "semantic_len":    len(state.get("semantic",  [])),
        "assoc_len":       len(state.get("assoc",     [])),
    }
    banner = "=" * 20 + " PLANNER INPUT " + "=" * 20
    print("\n" + banner)
    for k, v in incoming.items():
        print(f"{k:>15}: {v}")
    print("=" * len(banner) + "\n")
    self._print_stage_context("planning_summary_incoming", incoming)

    # ------------------------------------------------------------------ #
    # 1️⃣  Build “Conversation so far” from merged contexts             #
    # ------------------------------------------------------------------ #
    merged = state.get("merged", [])
    N = min(10, len(merged))
    convo_lines = [f"- {c.summary}" for c in merged[-N:]]
    convo_block = "Conversation so far:\n" + "\n".join(convo_lines) if convo_lines else ""

    # ------------------------------------------------------------------ #
    # Helper utilities                                                  #
    # ------------------------------------------------------------------ #
    def _clean_json_block(text: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if m:
            return m.group(1)
        m2 = re.search(r"(\{.*\})", text, flags=re.S)
        return (m2.group(1) if m2 else text).strip()

    def _first_sentence(desc: str) -> str:
        head = desc.split(".", 1)[0]
        return head + ("." if not head.endswith(".") else "")

    # ------------------------------------------------------------------ #
    # 2️⃣  Fetch critique & prompt                                       #
    # ------------------------------------------------------------------ #
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
    raw_prompt = prompt_rows[0].summary if prompt_rows else self._get_prompt("planning_prompt")
    first_two  = ".".join(raw_prompt.split(".", 2)[:2]) + "."

    tool_lines = "\n".join(
        f"- **{t['name']}**: {_first_sentence(t['description'])}"
        for t in tools_list
    ) or "(none)"

    base_system   = f"{first_two}\n\nAvailable tools:\n{tool_lines}"
    replan_system = (
        "Your last plan was invalid—**OUTPUT ONLY** the JSON, no extra text.\n\n"
        f"Available tools:\n{tool_lines}"
    )

    # ------------------------------------------------------------------ #
    # 3️⃣  Build USER message                                           #
    # ------------------------------------------------------------------ #
    original_snips = (know_ctx.summary or "").splitlines()

    def build_user(snips):
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
        return "\n\n".join(blocks)

    full_user = build_user(original_snips)

    # ─── 4) High‐level planning loop ────────────────────────────────
    last_calls = None
    plan_obj   = None
    raw_json   = None

    for attempt in range(1, 4):
        if attempt == 1:
            sys_p, user_p, tag = base_system, full_user, "[Planner]"
        else:
            keep = max(1, len(original_snips) // (2 ** (attempt - 1)))
            sys_p, user_p, tag = replan_system, build_user(original_snips[:keep]), "[PlannerReplan]"

        raw = self._stream_and_capture(
            self.secondary_model,
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            tag=tag,
            images=state.get("images"),
        ).strip()
        raw_json = _clean_json_block(raw)

        try:
            cand = json.loads(raw_json)
        except Exception:
            cand = None

        # ─ Normalization: ALWAYS end up with {"tasks":[…]} ───────────────
        if isinstance(cand, dict) and "tasks" in cand and isinstance(cand["tasks"], list):
            plan_obj = cand

        elif isinstance(cand, dict) and "tool_calls" in cand:
            # wrap old shape into new tasks schema
            tasks = []
            for tc in cand["tool_calls"]:
                if isinstance(tc, str):
                    tasks.append({"call": tc, "tool_input": {}, "subtasks": []})
                else:
                    name    = tc.get("call") or tc.get("tool_call")
                    inp     = tc.get("tool_input", {}) or {}
                    subs    = tc.get("subtasks", []) or []
                    tasks.append({"call": name, "tool_input": inp, "subtasks": subs})
            plan_obj = {"tasks": tasks}

        else:
            # fallback: regex‐extract foo(...) calls into minimal tasks
            calls = re.findall(r"\b[A-Za-z_]\w*\([^)]*\)", raw_json or raw)
            plan_obj = {"tasks": [{"call": c, "tool_input": {}, "subtasks": []} for c in calls]}

        # ensure structure
        if not isinstance(plan_obj, dict) or "tasks" not in plan_obj or not isinstance(plan_obj["tasks"], list):
            plan_obj = {"tasks": []}

        # validate tool names
        valid = {t["name"] for t in tools_list}
        if any(t.get("call") not in valid for t in plan_obj["tasks"]):
            continue

        calls_now = [t.get("call") for t in plan_obj["tasks"]]
        if calls_now == last_calls:
            continue
        last_calls = calls_now
        break

    # ─── 5) Refine each tool call ────────────────────────────────────
    schema_map = {t["name"]: t["schema"] for t in tools_list}

    def refine_single_tool(task: dict) -> dict:
        name     = task["call"]
        schema   = schema_map[name]
        required = schema["parameters"].get("required", [])
        props    = schema["parameters"].get("properties", {})
        schema_json = json.dumps(schema, indent=2)

        def make_base():
            return {"call": name, "tool_input": task.get("tool_input", {}) or {}, "subtasks": task.get("subtasks", []) or []}

        refined = make_base()
        critic  = ""
        errors  = []

        for retry in range(3):
            missing = [p for p in required if p not in refined["tool_input"]]
            if missing:
                errors.append(f"⚠️ Missing required parameters: {missing}")

            parts = []
            if critic:
                parts.append(f"💡 Critic: {critic}")
            if errors:
                parts.append("=== ERRORS ===\n" + "\n".join(errors))
            parts.extend([
                "=== CURRENT CALL ===\n```json\n" + json.dumps({"tasks":[refined]}, indent=2) + "\n```",
                "Output **only** a JSON object `{ \"tasks\": [...] }` matching the schema exactly.",
                "Required parameters:\n" + "\n".join(f"- `{p}`: {props[p]['type']}" for p in required),
                "=== SCHEMA ===\n```json\n" + schema_json + "\n```",
            ])
            sys_msg  = "\n\n".join(parts)
            user_msg = json.dumps({"tasks":[refined]})
            out = self._stream_and_capture(
                self.secondary_model,
                [{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
                tag=f"[PlannerRefine_{name}]_retry{retry}",
                images=state.get("images"),
            )
            clean = _clean_json_block(out)
            try:
                refined = json.loads(clean)["tasks"][0]
            except:
                pass

            missing = [p for p in required if p not in refined["tool_input"]]
            if not missing:
                return refined

            critic = self._stream_and_capture(
                self.secondary_model,
                [
                    {"role":"system","content":"In one sentence, tell the planner exactly what to fix to conform to the schema."},
                    {"role":"user","content":"SCHEMA:\n"+schema_json+"\n\nCALL:\n"+json.dumps(refined)},
                ],
                tag=f"[ToolCritic_{name}]",
                images=state.get("images"),
            ).strip()

        return refined

    tasks_in = plan_obj["tasks"]
    with ThreadPoolExecutor(max_workers=len(tasks_in) or 1) as pool:
        futures = [pool.submit(refine_single_tool, t) for t in tasks_in]
        refined = [f.result() for f in as_completed(futures)]
    order_map = {t["call"]: t for t in refined}
    plan_obj["tasks"] = [order_map.get(t["call"], t) for t in tasks_in]

    # record validation
    invalid = []
    for t in plan_obj["tasks"]:
        call = t["call"]
        if call not in schema_map:
            invalid.append(f"unknown tool '{call}'")
            continue
        miss = [p for p in schema_map[call]["parameters"].get("required", []) if p not in (t["tool_input"] or {})]
        if miss:
            invalid.append(f"tool '{call}' missing {miss}")
    if invalid:
        state.setdefault("errors", []).append(("plan_validation","; ".join(invalid)))
        state["plan_errors"]      = invalid
        state["plan_output_prev"] = json.dumps(plan_obj)

    # ─── 6) Flatten → call_strings ──────────────────────────────────
    def _flatten(t):
        out = [t]
        for s in t.get("subtasks", []):
            out.extend(_flatten(s))
        return out

    flat, calls = [], []
    for t in plan_obj["tasks"]:
        flat.extend(_flatten(t))
    for t in flat:
        inp = t.get("tool_input", {}) or {}
        if inp:
            arg_s = ",".join(f"{k}={json.dumps(v)}" for k,v in inp.items())
            calls.append(f"{t['call']}({arg_s})")
        else:
            calls.append(f"{t['call']}()")

    state["plan_calls"]  = calls
    state["valid_calls"] = calls
    state["fixed_calls"] = calls

    # ─── 7) Persist artefacts & tracker ────────────────────────────
    plan_json = json.dumps(plan_obj)
    plan_sig  = hashlib.md5(plan_json.encode()).hexdigest()[:8]

    ctx = ContextObject.make_stage(
        "planning_summary",
        clar_ctx.references + know_ctx.references + critique_ids,
        {"plan": plan_obj, "attempt": attempt, "plan_id": plan_sig},
    )
    ctx.stage_id = f"planning_summary_{plan_sig}"
    ctx.summary  = plan_json
    ctx.touch()
    self.repo.save(ctx)

    succ_cls = ContextObject.make_success if calls else ContextObject.make_failure
    succ_msg = f"Planner → {len(calls)} task(s)" if calls else "Planner → empty plan"
    succ = succ_cls(succ_msg, refs=[ctx.context_id])
    succ.stage_id = f"planning_summary_signal_{plan_sig}"
    succ.touch()
    self.repo.save(succ)

    tracker = ContextObject.make_stage(
        "plan_tracker",
        [ctx.context_id],
        {
            "plan_id":     plan_sig,
            "plan_calls":  calls,
            "total_calls": len(calls),
            "succeeded":   0,
            "attempts":    0,
            "status":      "in_progress",
            "started_at":  datetime.datetime.utcnow().isoformat() + "Z",
        },
    )
    tracker.semantic_label = plan_sig
    tracker.stage_id       = f"plan_tracker_{plan_sig}"
    tracker.summary        = "initialized plan tracker"
    tracker.touch()
    self.repo.save(tracker)

    state["plan_ctx"]      = ctx
    state["plan_output"]   = plan_json
    state["tools_list"]    = tools_list
    state["tc_ctx"]        = None

    return ctx, plan_json


def _stage7b_plan_validation(
    self,
    plan_ctx: ContextObject,
    plan_output: str,
    tools_list: List[Dict[str, str]],
    state: Dict[str, Any]
) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """
    Robust JSON-based plan validation. Keeps original tool_input intact,
    asks the LLM to fill in only truly missing parameters, and then
    reconstructs call-strings with full, exact arguments.

    Now *honors* the tools_list you passed in and injects full docstrings
    for just those tools you actually plan to call.
    """
    import json, re, inspect, importlib

    # A) Load all known tool schemas from your repo
    all_schemas = {
        json.loads(c.metadata["schema"])["name"]: json.loads(c.metadata["schema"])
        for c in self.repo.query(
            lambda c: c.component == "schema" and "tool_schema" in c.tags
        )
    }

    # B) Parse the planner’s JSON output
    try:
        plan_obj = json.loads(plan_output)
        tasks    = plan_obj.get("tasks", [])
    except Exception:
        raise RuntimeError("Planner output was not valid JSON; cannot validate tool_input parameters")

    # C) Figure out which tools the plan actually calls
    selected_calls = [t.get("call") for t in tasks if isinstance(t.get("call"), str)]
    available     = {t["name"] for t in tools_list}
    selected      = [name for name in selected_calls if name in available]

    # D) Build a filtered schema map for *just* those selected tools
    schemas_for_prompt = {
        name: all_schemas[name]
        for name in selected
        if name in all_schemas
    }

    # E) Inject full docstring into each filtered schema
    for name, schema in schemas_for_prompt.items():
        doc = None
        # 1) try Tools.<name>
        if hasattr(Tools, name):
            doc = inspect.getdoc(getattr(Tools, name))
        else:
            # 2) try top-level tools module
            try:
                mod = importlib.import_module("tools")
                if hasattr(mod, name):
                    doc = inspect.getdoc(getattr(mod, name))
            except ImportError:
                pass

        if doc:
            schema["description"] = doc  # overwrite truncated desc

    # F) Up to 3 repair passes: fill any missing required params
    missing = {}
    for _ in range(3):
        missing.clear()
        for idx, task in enumerate(tasks):
            name       = task.get("call")
            schema     = schemas_for_prompt.get(name)
            tool_input = task.get("tool_input", {}) or {}
            if not schema:
                continue
            required = set(schema["parameters"].get("required", []))
            found    = set(tool_input.keys())
            miss     = list(required - found)
            if miss:
                missing[idx] = miss

        if not missing:
            break

        prompt = {
            "description": "Some tool calls are missing required arguments.",
            "missing":     missing,
            "plan":        plan_obj,
            "schemas":     schemas_for_prompt
        }
        repair = self._stream_and_capture(
            self.secondary_model,
            [
                {"role": "system", "content":
                    "Return ONLY a JSON {'tasks':[...]} with each task’s tool_input now complete."},
                {"role": "user",   "content": json.dumps(prompt)},
            ],
            tag="[PlanFix]",
            images=state.get("images", None)
        ).strip()

        try:
            plan_obj = json.loads(repair)
            tasks    = plan_obj.get("tasks", [])
        except Exception:
            break

    # G) Re-serialize every task into a real call string
    fixed_calls = []
    for task in tasks:
        name = task["call"].split("(", 1)[0]            # strip if planner already added ()
        ti   = task.get("tool_input", {}) or {}
        if ti:
            args = ",".join(
                f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in ti.items()
            )
            fixed_calls.append(f"{name}({args})")
        else:
            fixed_calls.append(f"{name}()")


    # H) Persist the validation step
    meta = {
        "valid":       fixed_calls,
        "errors":      [],  
        "fixed_calls": fixed_calls
    }
    pv_ctx = ContextObject.make_stage("plan_validation", plan_ctx.references, meta)
    pv_ctx.stage_id = "plan_validation"
    pv_ctx.summary  = "OK" if not missing else f"Repaired {len(missing)} task(s)"
    #pv_ctx.touch(); self.repo.save(pv_ctx)
    self._persist_and_index([pv_ctx])
    self._print_stage_context("plan_validation", meta)

    return fixed_calls, [], fixed_calls

def _stage8_tool_chaining(
    self,
    plan_ctx: ContextObject,
    plan_output: str,
    tools_list: List[Dict[str, Any]],
    state: Dict[str, Any],
    *,
    on_token: Callable[[str], None] | None = None,
) -> Tuple[ContextObject, List[str], List[ContextObject]]:
    """
    1) Parse the plan JSON into call-strings.
    2) Substitute any tool-output placeholders from `state["last_tool_outputs"]`.
    3) Validate against known schemas.
    4) Emit a ContextObject ("tool_chaining") that records the final list.
    """
    import json, re
    from context import ContextObject
    from tools import Tools

    # Ensure Tools.repo is set
    Tools.repo = self.repo

    # Load schemas once
    try:
        all_schemas = {
            json.loads(c.metadata["schema"])["name"]: c
            for c in self.repo.query(
                lambda c: c.component == "schema" and "tool_schema" in c.tags
            )
        }
    except Exception:
        all_schemas = {}

    # Step A: Extract raw call objects from JSON plan
    calls: List[str] = []
    try:
        plan = json.loads(plan_output)
        def _collect(tasks):
            out = []
            for t in tasks or []:
                out.append(t)
                out.extend(_collect(t.get("subtasks", [])))
            return out

        for t in _collect(plan.get("tasks", [])):
            name = t.get("call", "")
            if not name:
                continue
            # JSON-encode any arguments
            kwargs = t.get("tool_input", {}) or {}
            if kwargs:
                arg_s = ",".join(
                    f"{k}={json.dumps(v, ensure_ascii=False)}"
                    for k,v in kwargs.items()
                )
                calls.append(f"{name}({arg_s})")
            else:
                calls.append(f"{name}()")
    except Exception:
        # Fallback regex
        salvage = set()
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(([^)]*)\)", plan_output or ""):
            fn, raw = m.group(1), m.group(2).strip()
            salvage.add(f"{fn}({raw})" if raw else f"{fn}()")
        calls = sorted(salvage)

    # Dedupe
    calls = list(dict.fromkeys(calls))

    # Step B: Placeholder substitution
    # state["last_tool_outputs"] is a dict: { "tool_name": <python obj> }
    last_out = state.get("last_tool_outputs", {})
    def _substitute( call_str: str ) -> str:
        # matches [tool.method.output] or {{tool}}
        def _replace(match):
            key = match.group(1) or match.group(2)
            val = last_out.get(key)
            return json.dumps(val, ensure_ascii=False) if val is not None else match.group(0)
        # square bracket style
        call_str = re.sub(r"\[([A-Za-z_]\w*)\.output\]", _replace, call_str)
        # moustache style
        call_str = re.sub(r"\{\{([A-Za-z_]\w*)\}\}", _replace, call_str)
        return call_str

    calls = [_substitute(c) for c in calls]

    # Step C: Schema validation / repair loop
    broken = []
    final_calls = []
    for call_str in calls:
        tool_name = call_str.split("(",1)[0]
        schema_ctx = all_schemas.get(tool_name)
        if not schema_ctx:
            # unknown tool → drop it & record error
            broken.append(f"Unknown tool `{tool_name}`")
            continue
        final_calls.append(call_str)

    # If anything broke, bubble it up
    if broken:
        raise RuntimeError(f"Tool chaining failed: {broken}")

    # Step D: Pick schema ContextObjects
    selected_schemas = []
    for c in final_calls:
        nm = c.split("(",1)[0]
        ctx = all_schemas.get(nm)
        if ctx:
            selected_schemas.append(ctx)

    # Step E: Save the stage context
    ctx_refs = plan_ctx.references + [s.context_id for s in selected_schemas]
    tc_ctx = ContextObject.make_stage(
        "tool_chaining",
        ctx_refs,
        {"tool_calls": final_calls}
    )
    tc_ctx.stage_id = "tool_chaining"
    tc_ctx.summary  = json.dumps(final_calls, ensure_ascii=False)
    tc_ctx.touch()
    self.repo.save(tc_ctx)

    return tc_ctx, final_calls, selected_schemas

def _stage8_5_user_confirmation(
    self,
    calls: list[Any],
    user_text: str
) -> list[str]:
    """
    Surface the to-be-invoked calls for user approval.
    Auto-converts ANY function-call object Gemma might emit into the
    canonical string form `tool_name(arg1=...,arg2=...)`.
    """
    import json

    def _obj2str(item) -> str:
        # Gemma-style or OpenAI function_call object
        if isinstance(item, dict):
            name = item.get("name") or item.get("tool_name") or item.get("call")
            if not name:
                return str(item).strip()

            # accept either "arguments" or "parameters"
            args_blob = item.get("arguments", item.get("parameters", {})) or {}
            if isinstance(args_blob, dict) and args_blob:
                arg_str = ",".join(
                    f'{k}={json.dumps(v, ensure_ascii=False)}'
                    for k, v in args_blob.items()
                )
                return f"{name}({arg_str})"
            return f"{name}()"

        # already a string
        return str(item).strip()

    confirmed: list[str] = [_obj2str(c) for c in calls]

    # diagnostics
    self._print_stage_context("user_confirmation", {"calls": confirmed})

    ctx = ContextObject.make_stage(
        "user_confirmation", [], {"confirmed_calls": confirmed}
    )
    ctx.stage_id = "user_confirmation"
    ctx.summary  = f"Auto-approved: {confirmed}"
    #ctx.touch(); self.repo.save(ctx)
    self._persist_and_index([ctx])

    return confirmed


def _stage9b_reflection_and_replan(
    self,
    tool_ctxs: List[ContextObject],
    plan_output: str,
    user_text: str,
    clar_metadata: Dict[str, Any],
    state: Dict[str, Any],
    max_tokens: int = 128000
) -> Optional[str]:
    """
    Reflect on the full context (user question, clarifier notes/keywords,
    **all** tool outputs, original plan).  If everything satisfied the intent,
    return None; otherwise return only the corrected JSON plan.
    """
    import json, re

    # 1) Gather clarifier info
    clar_notes = clar_metadata.get("notes", "")
    clar_keywords = clar_metadata.get("keywords", [])

    # 2) Build the full context blob
    parts = [
        f"=== USER QUESTION ===\n{user_text}",
        f"=== CLARIFIER NOTES ===\n{clar_notes}"
    ]
    if clar_keywords:
        parts.append(f"=== CLARIFIER KEYWORDS ===\n{', '.join(clar_keywords)}")

    # 3) Append every single tool output, unfiltered
    for c in tool_ctxs:
        payload = c.metadata.get("output_short") or c.metadata.get("output_full")

        try:
            blob = json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception:
            blob = repr(payload)
        parts.append(f"=== TOOL OUTPUT [{c.stage_id}] ===\n{blob}")

    # 4) Finally, the original plan
    parts.append(f"=== ORIGINAL PLAN ===\n{plan_output}")

    context_blob = "\n\n".join(parts)

    # 5) Reflection prompt
    system_msg = self._get_prompt("reflection_prompt")
    user_payload = (
        context_blob
        + "\n\nDid these tool outputs satisfy the original intent? "
            "If yes, reply OK.  If not, return only the corrected JSON plan."
    )

    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_payload},
    ]
    resp = self._stream_and_capture(self.secondary_model, msgs, tag="[Reflection]").strip()

    # ── NEW GUARD: identical-equals treat as OK
    try:
        new_plan = json.loads(resp)
        old_plan = json.loads(plan_output)
        if new_plan == old_plan:
            ok_ctx = ContextObject.make_success(
                description="Reflection confirmed plan (identical echo)",
                refs=[c.context_id for c in tool_ctxs]
            )
            #ok_ctx.touch(); self.repo.save(ok_ctx)
            self._persist_and_index([ok_ctx])
            return None
    except:
        pass

    # ── If literally "OK", keep original
    if re.fullmatch(r"(?i)(ok|okay)[.!]?", resp):
        ok_ctx = ContextObject.make_success(
            description="Reflection confirmed plan satisfied intent",
            refs=[c.context_id for c in tool_ctxs]
        )
        #ok_ctx.touch(); self.repo.save(ok_ctx)
        self._persist_and_index([ok_ctx])
        return None

    # ── Else record replan
    fail_ctx = ContextObject.make_failure(
        description="Reflection triggered replan",
        refs=[c.context_id for c in tool_ctxs]
    )
    #fail_ctx.touch(); self.repo.save(fail_ctx)
    self._persist_and_index([fail_ctx])

    repl = ContextObject.make_stage(
        "reflection_and_replan",
        [c.context_id for c in tool_ctxs],
        {"replan": resp}
    )
    repl.stage_id = "reflection_and_replan"
    repl.summary  = resp
    repl.touch(); self.repo.save(repl)

    return resp


def _stage9_invoke_with_retries(
    self,
    raw_calls: List[str],
    plan_output: str,
    selected_schemas: List[ContextObject],
    user_text: str,
    clar_metadata: Dict[str, Any],
    state: Dict[str, Any]
) -> List[ContextObject]:
    import json, re, hashlib, datetime, logging
    from tools import Tools
    # --- utilities -------------------------------------------------
    def _smart_truncate(text: str, max_len: int = 4000) -> str:
        if len(text) <= max_len:
            return text
        head = text[: max_len // 2]
        tail = text[-max_len // 2 :]
        return head + f"\n… ⟪{len(text)-max_len} chars elided⟫ …\n" + tail

    plan_sig = hashlib.md5(plan_output.encode("utf-8")).hexdigest()[:8]
    tracker = next(
        (c for c in self.repo.query(
            lambda c: c.component == "plan_tracker" and c.semantic_label == plan_sig
        )), None
    )

    # initialize or refresh tracker
    if not tracker:
        tracker = ContextObject.make_stage(
            "plan_tracker", [], {
                "plan_id":       plan_sig,
                "plan_calls":    raw_calls.copy(),
                "total_calls":   len(raw_calls),
                "succeeded":     0,
                "attempts":      0,
                "call_status_map": {},
                "errors_by_call": {},
                "status":        "in_progress",
                "started_at":    datetime.datetime.utcnow().isoformat() + "Z"
            }
        )
        tracker.semantic_label = plan_sig
        tracker.stage_id       = f"plan_tracker_{plan_sig}"
        tracker.summary        = "initialized plan tracker"
        tracker.touch(); self.repo.save(tracker)
    else:
        meta = tracker.metadata
        meta.setdefault("plan_calls", raw_calls.copy())
        meta.setdefault("total_calls", len(raw_calls))
        meta.setdefault("succeeded", 0)
        meta.setdefault("call_status_map", {})
        meta.setdefault("errors_by_call", {})
        meta.setdefault("status", "in_progress")
        tracker.touch(); self.repo.save(tracker)

    tracker.metadata["attempts"] = tracker.metadata.get("attempts", 0) + 1
    tracker.metadata["last_attempt_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    tracker.touch(); self.repo.save(tracker)

    # normalize and helper funcs
    def _norm(calls):
        out = []
        for c in calls:
            if isinstance(c, dict) and "tool_call" in c:
                out.append(c["tool_call"])
            elif isinstance(c, str):
                out.append(c)
        return out

    def _validate(res):
        exc = res.get("exception")
        return (exc is None, exc or "")

    def normalize_key(k):
        return re.sub(r"\W+", "", k).lower()

    all_calls = _norm(raw_calls)
    call_status = tracker.metadata.setdefault("call_status_map", {})
    pending = [c for c in all_calls if not call_status.get(c, False)]
    last_results = {}
    tool_ctxs = []

    # ─── fast‐path: nothing to run ─────────────────────────────
    if not pending:
        tracker.metadata["status"] = "success"
        tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        tracker.touch(); self.repo.save(tracker)

        existing = []
        for call in all_calls:
            matches = [
                c for c in self.repo.query(
                    lambda c: c.component == "tool_output"
                              and c.metadata.get("tool_call") == call
                )
            ]
            if matches:
                matches.sort(key=lambda c: c.timestamp, reverse=True)
                existing.append(matches[0])

        state["tool_ctxs"] = existing
        self.integrator.ingest(existing)
        state["merged"].extend(existing)
        return existing

    # ─── retry loop ────────────────────────────────────────────
    max_retries, prev = 10, None
    for _ in range(max_retries):
        if prev is not None and pending == prev:
            logging.warning("plateau, giving up retries")
            break
        prev = pending.copy()

        import random; random.shuffle(pending)
        for original in list(pending):
            call_str = original

            # placeholder chaining
            for ph in re.findall(r"\[([^\]]+)\]", call_str):
                key = normalize_key(ph[:-7]) if ph.endswith(".output") else normalize_key(ph)
                if key in last_results:
                    call_str = call_str.replace(f"[{ph}]", repr(last_results[key]))

            # alias chaining
            for ph in re.findall(r"\{\{([^}]+)\}\}", call_str):
                phn = normalize_key(ph)
                if phn in last_results:
                    call_str = call_str.replace(f"{{{{{ph}}}}}", repr(last_results[phn]))

            # nested zero-arg calls
            for inner in re.findall(r"\b([A-Za-z_]\w*)\(\)", call_str):
                if inner not in last_results:
                    r_i = Tools.run_tool_once(f"{inner}()")
                    ok_i, err_i = _validate(r_i)
                    last_results[inner] = r_i.get("output")
                    call_str = re.sub(rf"\b{inner}\(\)", repr(last_results[inner]), call_str)
                    # persist nested result
                    try:
                        sch_i = next(
                            s for s in selected_schemas
                            if json.loads(s.metadata["schema"])["name"] == inner
                        )
                        ctx_i = ContextObject.make_stage("tool_output", [sch_i.context_id], r_i)
                        ctx_i.stage_id = f"tool_output_nested_{inner}"
                        ctx_i.summary = str(r_i.get("output")) if ok_i else f"ERROR: {err_i}"
                        ctx_i.metadata.update(r_i)
                        ctx_i.touch(); self.repo.save(ctx_i)
                        tool_ctxs.append(ctx_i)
                    except StopIteration:
                        pass

            # ─── invoke main call ─────────────────────────────────
            res = Tools.run_tool_once(call_str)
            ok, err = _validate(res)
            raw_out = res.get("output")

            # pretty/short formatting
            try:
                pretty = json.dumps(raw_out, ensure_ascii=False, indent=2)
            except:
                pretty = repr(raw_out)
            short = _smart_truncate(pretty)

            # find schema ref
            name = original.split("(", 1)[0]
            refs = []
            for s in selected_schemas:
                if json.loads(s.metadata["schema"])["name"] == name:
                    refs = [s.context_id]
                    break

            # persist final tool_output
            meta = {
                "tool_call":    original,
                "output_full":  pretty,
                "output_short": short,
                "output":       raw_out,
                "exception":    res.get("exception"),
            }
            ctx = ContextObject.make_stage("tool_output", refs, meta)
            ctx.stage_id = f"tool_output_{name}"
            ctx.summary  = ("ERROR: " + str(err)) if not ok else repr(raw_out)
            ctx.touch(); self.repo.save(ctx)
            tool_ctxs.append(ctx)

            # track success/failure
            call_status[original] = ok
            tracker.metadata["succeeded"] = tracker.metadata.get("succeeded", 0) + int(ok)
            if not ok:
                tracker.metadata.setdefault("errors_by_call", {})[original] = err
            tracker.touch(); self.repo.save(tracker)

            if ok:
                pending.remove(original)

        if not pending:
            # ─── on successful retry ────────────────────────────
            tracker.metadata["status"] = "success"
            tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            self.repo.save(tracker)

            existing = []
            for call in all_calls:
                matches = [
                    c for c in self.repo.query(
                        lambda c: c.component == "tool_output"
                                  and c.metadata.get("tool_call") == call
                    )
                ]
                if matches:
                    matches.sort(key=lambda c: c.timestamp, reverse=True)
                    existing.append(matches[0])

            state["tool_ctxs"] = existing
            self.integrator.ingest(existing)    # ★ ensure integrator knows them
            state["merged"].extend(existing)    # ★ and Stage10 sees them
            return existing

        # if still pending, abort retry loop
        break

    # ─── final fail ───────────────────────────────────────────────
    tracker.metadata["status"] = "failed"
    tracker.metadata["last_errors"] = list(tracker.metadata.get("errors_by_call", {}).values())
    tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    tracker.touch(); self.repo.save(tracker)

    state["tool_ctxs"] = tool_ctxs
    self.integrator.ingest(tool_ctxs)
    state["merged"].extend(tool_ctxs)
    return tool_ctxs


def _stage10_assemble_and_infer(self, user_text: str, state: Dict[str, Any]) -> str:
    import json
    from collections import OrderedDict
    from context import ContextObject

    # ─── 1) Conversation snippets ────────────────────────────────────
    MAX_CTX_OBJS = 32
    merged = state.get("merged", [])
    recent = merged[-MAX_CTX_OBJS:]
    convo_lines = [c.summary or "" for c in recent]
    conversation_block = (
        "[Conversation so far]\n" + "\n".join(convo_lines)
        if convo_lines else ""
    )

    # ─── 2) Tool outputs ─────────────────────────────────────────────
    tool_ctxs = state.get("tool_ctxs", []) or []
    tool_blocks = []
    for tc in tool_ctxs:
        meta = tc.metadata.copy()
        if "tool_call" in meta:
            meta["tool_call"] = meta["tool_call"]
        try:
            payload = json.dumps(meta, ensure_ascii=False, indent=2)
        except Exception:
            payload = repr(meta)
        call_name = meta.get("tool_call", tc.stage_id)
        ts = getattr(tc, "timestamp", "")
        header = f"--- {tc.stage_id} ({call_name}) @ {ts} ---"
        tool_blocks.append(header)
        tool_blocks.append(payload)

    tools_block = (
        "[Tool outputs]\n" + "\n\n".join(tool_blocks)
        if tool_blocks else ""
    )

    # ─── 3) Narrative ────────────────────────────────────────────────
    narr_ctx = self._load_narrative_context()
    narrative_block = "[Narrative]\n" + (narr_ctx.summary or "")

    # ─── 4) Recent history bullets ──────────────────────────────────
    recent_hist = merged[-5:]
    bullets = [f"- {c.semantic_label}: {c.summary}" for c in recent_hist]
    recent_hist_block = (
        "[Recent History]\n" + "\n".join(bullets)
        if bullets else ""
    )

    # ─── 5) Plan & user question ────────────────────────────────────
    plan_txt = state.get("plan_output", "(no plan)")
    plan_block = "[Plan]\n" + plan_txt
    user_block = "[User question]\n" + user_text

    # ─── 6) Compose the exact messages for the LLM ──────────────────
    final_sys = self._get_prompt("final_inference_prompt")
    msgs = [
        {"role": "system", "content": final_sys},
        {"role": "system", "content": narrative_block},
    ]
    if conversation_block:
        msgs.append({"role": "system", "content": conversation_block})
    msgs.append({"role": "system", "content": self.inference_prompt})
    if recent_hist_block:
        msgs.append({"role": "system", "content": recent_hist_block})
    msgs.append({"role": "system", "content": plan_block})
    msgs.append({"role": "system", "content": user_block})
    msgs.append({"role": "user",   "content": user_text})
    if tools_block:
        msgs.append({"role": "system", "content": tools_block})

    # ─── 7) DEBUG PRINT: show all blocks *and* the assembled msgs ────
    debug_payload = OrderedDict([
        ("conversation_block", conversation_block),
        ("tools_block",        tools_block),
        ("narrative_block",    narrative_block),
        ("recent_hist_block",  recent_hist_block),
        ("plan_block",         plan_block),
        ("user_block",         user_block),
        ("assembled_msgs",     json.dumps(msgs, ensure_ascii=False, indent=2)),
        ("merged_ids",         [c.context_id for c in recent]),
        ("tool_ctx_ids",       [tc.context_id for tc in tool_ctxs]),
    ])
    self._print_stage_context("assemble_and_infer", debug_payload)

    # ─── 8) Call the model ───────────────────────────────────────────
    reply = self._stream_and_capture(
        self.primary_model,
        msgs,
        tag="[Assistant]",
        images=state.get("images"),
    ).strip()

    # ─── 9) Persist assistant reply ─────────────────────────────────
    refs = debug_payload["merged_ids"] + debug_payload["tool_ctx_ids"]
    resp_ctx = ContextObject.make_stage("final_inference", refs, {"text": reply})
    resp_ctx.stage_id = "final_inference"
    resp_ctx.summary  = reply
    self._persist_and_index([resp_ctx])

    seg = ContextObject.make_segment("assistant",
                                     [resp_ctx.context_id],
                                     tags=["assistant"])
    seg.summary   = reply
    seg.stage_id  = "assistant"
    seg.touch(); self.repo.save(seg)

    return reply


def _stage10b_response_critique_and_safety(
    self,
    draft: str,
    user_text: str,
    tool_ctxs: list["ContextObject"],
    state: dict[str, Any],
) -> str:
    import json, difflib
    from context import ContextObject

    # Pull in full merged context (including prior user/assistant turns)
    merged_ctxs = state.get("merged", [])
    merged_texts = "\n\n".join(f"[{c.stage_id}] {c.summary}" for c in merged_ctxs)

    # Retrieve the last executed plan
    plan_text = getattr(self, "_last_plan_output", "(no plan)")

    # 1) Serialize every tool output clearly
    outputs = []
    for c in tool_ctxs:
        raw = c.metadata.get("output")
        if isinstance(raw, dict) and "results" in raw:
            fragment = "\n".join(f"{r.get('timestamp','')} {r.get('role','')}: {r.get('content','')}"
                                  for r in raw["results"])
        else:
            fragment = json.dumps(raw, indent=2, ensure_ascii=False)
        outputs.append(f"[{c.stage_id}]\n{fragment}")

    # 2️⃣  Relevance Extraction
    extractor_sys = self._get_prompt("extractor_sys_prompt")
    extractor_user = "\n\n".join([
        f"USER QUESTION:\n{user_text}",
        f"PLAN EXECUTED:\n{plan_text}",
        "MERGED CONTEXT SNIPPETS:\n" + (merged_texts or "(none)"),
        "RAW TOOL OUTPUTS:\n" + ("\n\n".join(outputs) or "(none)"),
        "DRAFT RESPONSE:\n" + draft,
        "Your goal: identify exactly the facts, data points, or insights that should shape the revised response."
    ])
    bullets = self._stream_and_capture(
        self.secondary_model,
        [
            {"role": "system", "content": extractor_sys},
            {"role": "user",   "content": extractor_user},
        ],
        tag="[RelevExtract]",
        images=state.get("images"),
    ).strip()

    # Persist relevance summary
    sum_ctx = ContextObject.make_stage("relevance_summary", [], {"summary": bullets})
    sum_ctx.stage_id = "relevance_summary"
    sum_ctx.summary  = bullets
    self._persist_and_index([sum_ctx])

    # 3️⃣  Polishing / Safety Critique

    editor_sys = self._get_prompt("editor_sys_prompt")
    editor_user = "\n\n".join([
        f"USER QUESTION:\n{user_text}",
        "RELEVANCE BULLETS:\n" + bullets,
        "CURRENT DRAFT:\n" + draft
    ])
    polished = self._stream_and_capture(
        self.secondary_model,
        [
            {"role": "system", "content": editor_sys},
            {"role": "user",   "content": editor_user},
        ],
        tag="[Polisher]",
        images=state.get("images"),
    ).strip()

    # If nothing changed, return original draft
    if polished == draft.strip():
        return polished

    # 4️⃣  Compute diff & update dynamic patch
    diff = difflib.unified_diff(draft.splitlines(), polished.splitlines(), lineterm="", n=1)
    diff_summary = "; ".join(ln for ln in diff if ln.startswith(("+ ", "- "))) or "(format refined)"

    patch_rows = self.repo.query(
        lambda c: c.component == "policy" and c.semantic_label == "dynamic_prompt_patch"
    )
    patch_rows.sort(key=lambda c: c.timestamp, reverse=True)
    dynamic_patch = patch_rows[0] if patch_rows else ContextObject.make_policy(
        "dynamic_prompt_patch", diff_summary, tags=["dynamic_prompt"]
    )
    if dynamic_patch.summary != diff_summary:
        dynamic_patch.summary = diff_summary
        dynamic_patch.metadata["policy"] = diff_summary
        dynamic_patch.touch(); self.repo.save(dynamic_patch)

    # 5️⃣  Persist polished reply and critique contexts
    resp_ctx = ContextObject.make_stage("response_critique", [sum_ctx.context_id], {"text": polished})
    resp_ctx.stage_id = "response_critique"
    resp_ctx.summary  = polished
    self._persist_and_index([resp_ctx])

    critique_ctx = ContextObject.make_stage(
        "plan_critique",
        [resp_ctx.context_id] + [c.context_id for c in tool_ctxs],
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
                    plan_obj = {"tasks":[
                        {"call":c.split("(",1)[0], "tool_input":{}, "subtasks": []}
                        for c in plan_calls
                    ]}
                    pj = json.dumps(plan_obj)
                    p_ctx = ContextObject.make_stage("internal_plan",[a_ctx.context_id],{"plan":plan_obj})
                    #p_ctx.touch(); self.repo.save(p_ctx)
                    self._persist_and_index([p_ctx])

                    tools = self._stage6_prepare_tools()
                    fixed,_,_ = self._stage7b_plan_validation(p_ctx,pj,tools)
                    tc, calls, schemas = self._stage8_tool_chaining(p_ctx,"\n".join(fixed),tools)
                    self._stage9_invoke_with_retries(
                        raw_calls=calls, plan_output="\n".join(fixed),
                        selected_schemas=schemas,
                        user_text="(self-review)", clar_metadata={}
                    )
                except Exception as e:
                    err = ContextObject.make_failure(
                        f"narrative_mull plan error: {e}", refs=[a_ctx.context_id]
                    )
                    err.touch(); self.repo.save(err)

    # start thread
    threading.Thread(target=_worker, daemon=True).start()
    return "(narrative mull dispatched)"
