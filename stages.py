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



def _stage1_record_input(self, user_text: str) -> ContextObject:
    # record the new user input
    ctx = ContextObject.make_segment("user_input", [], tags=["user_input"])
    ctx.summary = user_text
    ctx.stage_id = "user_input"
    ctx.touch()
    self.repo.save(ctx)
    return ctx

def _stage2_load_system_prompts(self) -> ContextObject:
    prom = self._get_prompt("system_prompt")   # or load from repo/artifact
    ctx = ContextObject.make_stage(
        "system_prompt",
        [], {"prompt": prom}
    )
    ctx.stage_id = "system_prompt"
    ctx.summary  = prom
    ctx.touch(); self.repo.save(ctx)
    return ctx

def _stage3_retrieve_and_merge_context(
    self,
    user_text: str,
    user_ctx: ContextObject,
    sys_ctx: ContextObject,
    extra_ctx: List[ContextObject] = None,
    recall_ids: List[str] = None
) -> Dict[str, Any]:
    """
    Merge per-conversation context (including extra_ctx), rewrite each ctx.summary as:
       "<sender>: <original summary>"
    so downstream stages know who said what.
    """
    from datetime import datetime, timedelta

    conv_id = user_ctx.metadata.get("conversation_id")
    user_id = user_ctx.metadata.get("user_id")
    now = datetime.utcnow()

    def _to_dt(ts):
        try:
            return datetime.fromisoformat(ts.rstrip("Z"))
        except:
            return now

    # 1) Load all raw user/assistant segments for this convo
    all_segments = [
        c for c in self.repo.query(lambda c:
            c.metadata.get("conversation_id") == conv_id
            and c.domain == "segment"
            and c.metadata.get("user_id") in (user_id, None)
            and c.semantic_label in ("user_input", "assistant")
        )
    ]

    # 1a) Prepend any extra_ctx the caller provided
    if extra_ctx:
        # combine and dedupe by context_id
        combined = list(extra_ctx) + all_segments
        seen_ids = set()
        deduped = []
        for seg in combined:
            if seg.context_id not in seen_ids:
                deduped.append(seg)
                seen_ids.add(seg.context_id)
        all_segments = deduped

    # 1b) Re-sort everything chronologically
    all_segments.sort(key=lambda c: _to_dt(c.timestamp))

    # 2) Working memory: last N turns
    WM_TURNS   = 20    # keep more turns
    segments_wm = all_segments[-WM_TURNS:]

    # 3) Short-term memory: last M minutes
    ST_MINUTES = 120   # wider time window
    cutoff = now - timedelta(minutes=ST_MINUTES)
    segments_st = [c for c in all_segments if _to_dt(c.timestamp) >= cutoff]

    # 4) Semantic retrieval (as before)
    semantic = []
    if self.rl.should_run("semantic_retrieval", 0.0):
        hits = self.engine.query(
            stage_id="semantic_retrieval",
            similarity_to=user_text,
            include_tags=[],
            exclude_tags=self.STAGES + ["tool_schema", "tool_output"],
            top_k=self.top_k
        )
        semantic = [c for c in hits
                    if c.metadata.get("conversation_id") == conv_id]

    # 5) Associative recall (as before)
    assoc = []
    if self.rl.should_run("memory_retrieval", 0.0):
        seeds = [user_ctx.context_id]
        scores = self.memman.spread_activation(
            seed_ids=seeds, hops=3, decay=0.7,
            assoc_weight=1.0, recency_weight=1.0
        )
        for cid in sorted(scores, key=scores.get, reverse=True)[: self.top_k]:
            try:
                cobj = self.repo.get(cid)
                if cobj.metadata.get("conversation_id") == conv_id:
                    cobj.retrieval_score = scores[cid]
                    assoc.append(cobj)
            except KeyError:
                pass

    # 6) Recent tool outputs (as before)
    recent_tools = [
        c for c in self.repo.query(lambda c:
            c.component == "tool_output"
            and c.metadata.get("conversation_id") == conv_id
        )
    ]
    recent_tools.sort(key=lambda c: _to_dt(c.timestamp))
    recent_tools = recent_tools[-self.top_k:]

    # 7) Prefix each summary with its sender
    def _apply_sender(ctx: ContextObject):
        sender = ctx.metadata.get("sender")
        if not sender:
            sender = "Assistant" if ctx.semantic_label=="assistant" else "User"
        text = ctx.summary or ""
        ctx.summary = f"{sender}: {text}"

    # collect *all* candidate contexts
    all_buckets = [sys_ctx, user_ctx] + segments_wm + segments_st + semantic + assoc + recent_tools
    for ctx in all_buckets:
        _apply_sender(ctx)

    # 8) Merge in priority order, de-duping by (stage_id, summary)
    merged, seen = [], set()
    def _add(bucket):
        for c in bucket:
            key = (c.stage_id, c.summary)
            if key not in seen:
                merged.append(c)
                seen.add(key)

    _add([sys_ctx])
    _add([user_ctx])
    _add(segments_wm)
    _add(segments_st)
    _add(semantic)
    _add(assoc)
    _add(recent_tools)

    # 9) Build return values
    wm_ids     = [c.context_id for c in segments_wm]
    merged_ids = [c.context_id for c in merged]

    return {
        "merged":     merged,
        "merged_ids": merged_ids,
        "wm_ids":     wm_ids,
    }


def _stage4_intent_clarification(self, user_text: str, state: Dict[str, Any]) -> ContextObject:
    import json, textwrap
    from context import ContextObject

    # 1) Fetch the last 4 user/assistant segments from the store
    conv_id = state["user_ctx"].metadata["conversation_id"]
    segs = [
        c for c in self.repo.query(lambda c:
            c.domain == "segment"
            and c.metadata.get("conversation_id") == conv_id
            and c.semantic_label in ("user_input", "assistant")
        )
    ]
    # sort by timestamp, keep only the last 4
    segs.sort(key=lambda c: c.timestamp)
    last_dialog = segs[-4:]

    # 2) Build a simple "Recent Dialogue" block
    dialog_lines = []
    for c in last_dialog:
        role = "User" if c.semantic_label == "user_input" else "Assistant"
        text = c.summary or c.metadata.get("text", "")
        dialog_lines.append(f"{role}: {text}")
    recent_dialog = "\n".join(dialog_lines) or "(no recent dialog)"

    # 3) Build the clarifier prompt
    clarifier_sys = self.clarifier_prompt
    full_context_block = textwrap.dedent(f"""
        ### Recent Dialogue ###
        {recent_dialog}

        ### Current User Query ###
        {user_text}
    """).strip()

    msgs = [
        {"role": "system",  "content": clarifier_sys},
        {"role": "system",  "content": full_context_block},
        {"role": "user",    "content": user_text},
    ]

    out = self._stream_and_capture(
        self.secondary_model,
        msgs,
        tag="[Clarifier]",
        images=state.get("images")
    ).strip()

    # 4) Enforce JSON shape (unchanged)
    for attempt in (1, 2):
        try:
            clar = json.loads(out)
            if isinstance(clar, dict) and "keywords" in clar and "notes" in clar:
                break
        except Exception:
            clar = None

        if attempt == 1:
            retry_sys = (
                "⚠️ Please output ONLY a JSON object with keys "
                "\"keywords\" (array) and \"notes\" (string or array)."
            )
            out = self._stream_and_capture(
                self.secondary_model,
                [
                    {"role": "system", "content": retry_sys},
                    {"role": "user",   "content": out}
                ],
                tag="[ClarifierRetry]",
                images=state.get("images")
            ).strip()
        else:
            clar = {"keywords": [], "notes": out}

    # 5) Persist the Clarifier ContextObject
    clar_ctx = ContextObject.make_stage(
        "intent_clarification",
        input_refs=[state["user_ctx"].context_id],
        output=clar
    )
    clar_ctx.stage_id       = "intent_clarification"
    clar_ctx.semantic_label = "clarifier_question"
    clar_ctx.tags.append("clarifier_question")
    clar_ctx.metadata.update({
        "conversation_id": state["user_ctx"].metadata["conversation_id"],
        "user_id":         state["user_ctx"].metadata["user_id"],
    })
    clar_ctx.summary = clar.get("notes", "")
    clar_ctx.touch()
    self.repo.save(clar_ctx)
    #self.memman.register_relationships(clar_ctx, embed_text)

    return clar_ctx




def _stage5_external_knowledge(
    self,
    clar_ctx: ContextObject,
    state: Dict[str, Any] | None = None,
) -> ContextObject:
    """
    Build a compact “external knowledge” context for the planner using:
      • the last few user/assistant turns (history)
      • recent tool outputs (tools)
      • semantic recalls (semantic)
      • associative recalls (assoc)
      • and fresh engine.query hits on the clarifier keywords
    """
    import json
    from datetime import datetime
    from context import ContextObject

    # 1) Gather clarifier keywords
    kws = clar_ctx.metadata.get("keywords") or []
    if not kws and clar_ctx.summary:
        kws = [clar_ctx.summary]

    # 2) Pull fresh hits for each keyword
    top_k = max(3, self.top_k)
    fresh_hits = []
    for kw in kws:
        hits = self.engine.query(
            similarity_to=kw,
            stage_id="external_knowledge_query",
            top_k=top_k
        )
        fresh_hits.extend(hits)
    # de-dupe by context_id
    seen = set()
    fresh_snips = []
    for h in fresh_hits:
        if h.context_id not in seen:
            seen.add(h.context_id)
            snippet = h.summary or h.metadata.get("content","")
            fresh_snips.append((h, snippet))
        if len(fresh_snips) >= top_k:
            break

    # 3) Build labeled snippet lines
    lines = []
    def _label_and_trim(ctx, label):
        txt = (ctx.summary or ctx.metadata.get("content","")).strip()
        if len(txt) > 200:
            txt = txt[:200].rsplit(" ",1)[0] + " …"
        return f"({label}) {txt}"

    # a) last few turns
    for c in state.get("history", []):
        role = "USER" if c.semantic_label=="user_input" else "ASSIST"
        lines.append(_label_and_trim(c, role))

    # b) tool outputs
    for c in state.get("tools", []):
        out = c.metadata.get("output")
        out_str = json.dumps(out, ensure_ascii=False) if not isinstance(out, str) else out
        lines.append(f"(TOOL:{c.stage_id}) {out_str}")

    # c) semantic recalls
    for c in state.get("semantic", []):
        lines.append(_label_and_trim(c, "SEM"))

    # d) associative recalls
    for c in state.get("assoc", []):
        lines.append(_label_and_trim(c, "ASSOC"))

    # e) fresh hits on clarifier keywords
    for ctx, txt in fresh_snips:
        lines.append(_label_and_trim(ctx, "FRESH"))

    # cap total lines
    lines = lines[: top_k * 4]

    # 4) Persist as one external_knowledge_retrieval ContextObject
    combined = "\n".join(lines)
    ext_ctx = ContextObject.make_stage(
        "external_knowledge_retrieval",
        clar_ctx.references,
        {"snippets": lines},
    )
    ext_ctx.stage_id = "external_knowledge_retrieval"
    ext_ctx.summary  = combined
    ext_ctx.touch(); self.repo.save(ext_ctx)

    # (optional) log for debugging
    self._print_stage_context("external_knowledge_retrieval", {
        "lines": lines,
        "count": len(lines),
    })

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
    2) Inject truncated tool list (name + first sentence of description)
    3) Run up to 3 JSON-only planning passes, halving snippets each retry
    4) Take the raw plan, find which tools were selected
    5) Inject full schemas for only those tools and do a second pass to fill missing args
       and *explicitly* enforce correct key names per the schema
    6) Serialize to plan_json (preserving tool_input keys) and persist
    """
    import json, re, hashlib, datetime, textwrap

    def _clean_json_block(text: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if m:
            return m.group(1)
        m2 = re.search(r"(\{.*\})", text, flags=re.S)
        return (m2.group(1) if m2 else text).strip()

    def _first_sentence(desc: str) -> str:
        parts = desc.split(".", 1)
        return parts[0] + "." if len(parts) > 1 else desc

    # 0) collect any prior plan_critique IDs
    critique_rows = self.repo.query(
        lambda c: c.component == "analysis" and c.semantic_label == "plan_critique"
    )
    critique_rows.sort(key=lambda c: c.timestamp)
    critique_ids = [c.context_id for c in critique_rows]

    # 1) load the two-sentence planning prompt
    prompt_rows = self.repo.query(
        lambda c: c.component == "artifact" and c.semantic_label == "planning_prompt"
    )
    prompt_rows.sort(key=lambda c: c.timestamp, reverse=True)
    raw_prompt = (
        prompt_rows[0].summary
        if prompt_rows
        else self._get_prompt("planning_prompt")
    )
    first_two = ".".join(raw_prompt.split(".", 2)[:2]) + "."

    # 2) build truncated tool list
    tool_lines = "\n".join(
        f"- **{t['name']}**: {_first_sentence(t['description'])}"
        for t in tools_list
    ) or "(none)"
    base_system = f"{first_two}\n\nAvailable tools:\n{tool_lines}"
    replan_system = (
        "Your last plan was invalid—**OUTPUT ONLY** the JSON, no extra text.\n\n"
        f"Available tools:\n{tool_lines}"
    )

    # 3) prepare user block
    original_snips = (know_ctx.summary or "").splitlines()

    def build_user(snips):
        return "\n\n".join([
            f"User question:\n{user_text}",
            f"Clarified intent:\n{clar_ctx.summary}",
            "Snippets:\n" + ("\n".join(snips) if snips else "(none)")
        ])

    full_user = build_user(original_snips)

    # 4) run up to 3 planner passes
    last_calls = None
    plan_obj = None
    for attempt in range(1, 4):
        if attempt == 1:
            sys_p, user_p, tag = base_system, full_user, "[Planner]"
        else:
            keep = max(1, len(original_snips) // (2 ** (attempt - 1)))
            sys_p, user_p, tag = (
                replan_system,
                build_user(original_snips[:keep]),
                "[PlannerReplan]",
            )

        raw = self._stream_and_capture(
            self.secondary_model,
            [{"role": "system", "content": sys_p},
             {"role": "user",   "content": user_p}],
            tag=tag,
            images=state.get("images"),
        )
        cleaned = _clean_json_block(raw)
        try:
            cand = json.loads(cleaned)
            assert isinstance(cand, dict) and "tasks" in cand
            plan_obj = cand
        except:
            calls = re.findall(r"\b[A-Za-z_]\w*\([^)]*\)", raw)
            plan_obj = {
                "tasks": [
                    {"call": c, "tool_input": {}, "subtasks": []}
                    for c in calls
                ]
            }

        valid = {t["name"] for t in tools_list}
        if any(t["call"] not in valid for t in plan_obj["tasks"]):
            continue
        if [t["call"] for t in plan_obj["tasks"]] == last_calls:
            continue
        last_calls = [t["call"] for t in plan_obj["tasks"]]
        break

    # 5) REFINE pass: inline entire JSON schema for every selected tool
    selected = {t["call"] for t in plan_obj["tasks"]}
    selected_tools = [t for t in tools_list if t["name"] in selected]

    if selected_tools:
        # build full‐schema blocks
        schema_blocks: List[str] = []
        for tool in selected_tools:
            full_schema = tool["schema"]  # includes description, params, etc.
            schema_json = json.dumps(full_schema, indent=2)
            schema_blocks.append(
                f"**{tool['name']}** full JSON schema:\n```json\n{schema_json}\n```"
            )

        # list required params only (to force renaming/filling)
        param_list: List[str] = []
        for tool in selected_tools:
            props = tool["schema"]["parameters"]["properties"]
            required = tool["schema"]["parameters"].get("required", [])
            for p in required:
                param_list.append(f"- `{p}`: {props[p].get('type')}")

        param_block = "\n".join(param_list) or "(no required params)"

        refine_system = (
            "Now that you've selected your tool(s), please *use exactly* the following parameter names:\n"
            f"{param_block}\n\n"
            "Fill in any missing `tool_input` entries and *rename* any incorrect keys.\n\n"
            "Here are the **entire** JSON schemas for your selected tools:\n\n"
            + "\n\n".join(schema_blocks)
        )
        refine_user = json.dumps({"tasks": plan_obj["tasks"]})

        # log it for visibility
        print("\n── REFINE_SYSTEM START ──")
        print(refine_system)
        print("── REFINE_SYSTEM END ──\n")

        raw2 = self._stream_and_capture(
            self.secondary_model,
            [{"role": "system", "content": refine_system},
             {"role": "user",   "content": refine_user}],
            tag="[PlannerRefine]",
            images=state.get("images"),
        )
        cleaned2 = _clean_json_block(raw2)
        try:
            cand2 = json.loads(cleaned2)
            assert isinstance(cand2, dict) and "tasks" in cand2
            plan_obj = cand2
        except:
            pass

    # 6) flatten + build call strings
    def _flatten(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = [task]
        for sub in task.get("subtasks", []):
            out.extend(_flatten(sub))
        return out

    flat: List[Dict[str, Any]] = []
    for t in plan_obj.get("tasks", []):
        flat.extend(_flatten(t))

    call_strings: List[str] = []
    for task in flat:
        name = task["call"]
        params = task.get("tool_input", {}) or {}
        if params:
            args = ",".join(
                f"{k}={json.dumps(v, ensure_ascii=False)}"
                for k, v in params.items()
            )
            call_strings.append(f"{name}({args})")
        else:
            call_strings.append(f"{name}()")

    plan_json = json.dumps(plan_obj, ensure_ascii=False)
    plan_sig = hashlib.md5(plan_json.encode("utf-8")).hexdigest()[:8]

    # 7) persist everything
    ctx = ContextObject.make_stage(
        "planning_summary",
        clar_ctx.references + know_ctx.references + critique_ids,
        {"plan": plan_obj, "attempt": attempt, "plan_id": plan_sig},
    )
    ctx.stage_id = f"planning_summary_{plan_sig}"
    ctx.summary = plan_json
    ctx.touch()
    self.repo.save(ctx)

    succ_cls = ContextObject.make_success if call_strings else ContextObject.make_failure
    succ_msg = (
        f"Planner → {len(call_strings)} task(s)"
        if call_strings
        else "Planner → empty plan"
    )
    succ = succ_cls(succ_msg, refs=[ctx.context_id])
    succ.stage_id = f"planning_summary_signal_{plan_sig}"
    succ.touch()
    self.repo.save(succ)

    tracker = ContextObject.make_stage(
        "plan_tracker",
        [ctx.context_id],
        {
            "plan_id":     plan_sig,
            "total_calls": len(call_strings),
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
    pv_ctx.touch(); self.repo.save(pv_ctx)
    self._print_stage_context("plan_validation", meta)

    return fixed_calls, [], fixed_calls


def _stage8_tool_chaining(
    self,
    plan_ctx: ContextObject,
    plan_output: str,
    tools_list: List[Dict[str, Any]],
    state: Dict[str, Any]
) -> Tuple[ContextObject, List[str], List[ContextObject]]:
    import json, re, inspect
    from tools import Tools

    # Ensure Tools can always reach the current repo
    Tools.repo = self.repo

    # 0) load schemas from the repo
    schema_map = {
        json.loads(s.metadata["schema"])["name"]: json.loads(s.metadata["schema"])
        for s in self.repo.query(lambda c: c.component=="schema" and "tool_schema" in c.tags)
    }

    # A) extract flat list of calls from the JSON plan (or fallback via regex)
    calls: List[str] = []
    try:
        plan = json.loads(plan_output)
        def flatten(tasks):
            out = []
            for t in tasks:
                out.append(t)
                out.extend(flatten(t.get("subtasks", [])))
            return out

        for t in flatten(plan.get("tasks", [])):
            name   = t["call"].split("(", 1)[0]
            params = t.get("tool_input", {}) or {}
            if params:
                arg_s = ",".join(
                    f"{k}={json.dumps(v, ensure_ascii=False)}"
                    for k, v in params.items()
                )
                calls.append(f"{name}({arg_s})")
            else:
                calls.append(f"{name}()")
    except Exception:
        salvage = set()
        for m in re.finditer(r'\b([A-Za-z_]\w*)\s*\(([^)]*)\)', plan_output):
            nm, raw = m.group(1), m.group(2).strip()
            salvage.add(f"{nm}({raw})" if raw else f"{nm}()")
        calls.extend(sorted(salvage))

    # B) pick matching schema ContextObjects
    wanted = {c.split("(",1)[0] for c in calls}
    selected_schemas = [
        s for s in self.repo.query(lambda c: c.component=="schema" and "tool_schema" in c.tags)
        if json.loads(s.metadata["schema"])["name"] in wanted
    ]

    # C) actually invoke each tool, injecting assembler only when needed
    tool_ctxs: List[ContextObject] = []
    confirmed: List[str]  = []
    for call_str in calls:
        name, arg_blob = call_str.split("(",1)
        arg_blob = arg_blob.rstrip(")")

        # parse kwargs from the call string
        try:
            kwargs = json.loads("{" + arg_blob + "}")
        except Exception:
            kwargs = {}
            for part in arg_blob.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        kwargs[k] = json.loads(v)
                    except Exception:
                        kwargs[k] = v.strip('"\'')

        func = getattr(Tools, name)
        sig  = inspect.signature(func)

        # only inject assembler if the tool actually declares it
        if "assembler" in sig.parameters:
            invoke_kwargs = {"assembler": self, **kwargs}
        else:
            invoke_kwargs = kwargs

        try:
            output = func(**invoke_kwargs)
            exception = None
        except Exception as e:
            output, exception = None, str(e)

        # verbose logging
        print(f"[ToolInvocation] {name} called with {invoke_kwargs!r} → "
              f"output={output!r}, exception={exception!r}")

        # persist the tool_output context
        sch_ctx = next(
            (s for s in selected_schemas
             if json.loads(s.metadata["schema"])["name"] == name),
            None
        )
        refs = [sch_ctx.context_id] if sch_ctx else []
        ctx = ContextObject.make_stage(
            "tool_output",
            refs,
            {"tool_call": call_str, "output": output, "exception": exception}
        )
        ctx.stage_id = f"tool_output_{name}"
        ctx.summary  = repr(output) if exception is None else f"ERROR: {exception}"
        ctx.touch()
        self.repo.save(ctx)

        tool_ctxs.append(ctx)
        confirmed.append(call_str)

    # D) record the chaining itself
    tc_ctx = ContextObject.make_stage(
        "tool_chaining",
        plan_ctx.references + [s.context_id for s in selected_schemas],
        {"tool_calls": confirmed}
    )
    tc_ctx.stage_id = "tool_chaining"
    tc_ctx.summary  = json.dumps(confirmed, ensure_ascii=False)
    tc_ctx.touch()
    self.repo.save(tc_ctx)

    # make tool_ctxs available downstream
    state["tool_ctxs"] = tool_ctxs

    return tc_ctx, confirmed, tool_ctxs


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
    ctx.touch(); self.repo.save(ctx)

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
        payload = c.metadata.get("output")
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
            ok_ctx.touch(); self.repo.save(ok_ctx)
            return None
    except:
        pass

    # ── If literally "OK", keep original
    if re.fullmatch(r"(?i)(ok|okay)[.!]?", resp):
        ok_ctx = ContextObject.make_success(
            description="Reflection confirmed plan satisfied intent",
            refs=[c.context_id for c in tool_ctxs]
        )
        ok_ctx.touch(); self.repo.save(ok_ctx)
        return None

    # ── Else record replan
    fail_ctx = ContextObject.make_failure(
        description="Reflection triggered replan",
        refs=[c.context_id for c in tool_ctxs]
    )
    fail_ctx.touch(); self.repo.save(fail_ctx)

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

    plan_sig = hashlib.md5(plan_output.encode("utf-8")).hexdigest()[:8]
    tracker = next(
        (c for c in self.repo.query(
            lambda c: c.component=="plan_tracker" and c.semantic_label==plan_sig
        )), None
    )
    if not tracker:
        tracker = ContextObject.make_stage(
            "plan_tracker", [], {
                "plan_id":plan_sig,
                "plan_calls":raw_calls.copy(),
                "total_calls":len(raw_calls),
                "succeeded":0,
                "attempts":0,
                "call_status_map":{},
                "errors_by_call":{},
                "status":"in_progress",
                "started_at":datetime.datetime.utcnow().isoformat()+"Z"
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

    tracker.metadata["attempts"] += 1
    tracker.metadata["last_attempt_at"] = datetime.datetime.utcnow().isoformat()+"Z"
    tracker.touch(); self.repo.save(tracker)

    def _norm(calls):
        out=[]
        for c in calls:
            if isinstance(c,dict) and "tool_call" in c: out.append(c["tool_call"])
            elif isinstance(c,str): out.append(c)
        return out

    def _validate(res):
        exc = res.get("exception")
        return (exc is None, exc or "")

    def normalize_key(k): return re.sub(r"\W+","",k).lower()

    all_calls   = _norm(raw_calls)
    call_status = tracker.metadata["call_status_map"]
    pending     = [c for c in all_calls if not call_status.get(c,False)]
    last_results = {}
    tool_ctxs = []

    # fast‐path if nothing to run
    if not pending:
        tracker.metadata["status"] = "success"
        tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat()+"Z"
        tracker.touch(); self.repo.save(tracker)

        existing=[]
        for call in all_calls:
            matches = [
                c for c in self.repo.query(
                    lambda c: c.component=="tool_output"
                              and c.metadata.get("tool_call")==call
                )
            ]
            if matches:
                matches.sort(key=lambda c: c.timestamp,reverse=True)
                existing.append(matches[0])

        state["tool_ctxs"] = existing
        return existing

    # retry loop
    max_retries, prev = 10, None
    for _ in range(max_retries):
        if prev is not None and pending==prev:
            logging.warning("plateau, giving up retries")
            break
        prev = pending.copy()

        import random; random.shuffle(pending)
        for original in list(pending):
            call_str = original

            # placeholder chaining
            for ph in re.findall(r"\[([^\]]+)\]",call_str):
                if ph.endswith(".output"):
                    key=normalize_key(ph[:-7])
                    if key in last_results:
                        call_str=call_str.replace(f"[{ph}]",repr(last_results[key]))
                else:
                    phn=normalize_key(ph)
                    m=next((k for k in last_results if phn in normalize_key(k)),None)
                    if m: call_str=call_str.replace(f"[{ph}]",repr(last_results[m]))

            # alias chaining
            for ph in re.findall(r"\{\{([^}]+)\}\}",call_str):
                phn=normalize_key(ph)
                m=next((k for k in last_results if phn in normalize_key(k)),None)
                if m: call_str=call_str.replace(f"{{{{{ph}}}}}",repr(last_results[m]))

            # nested zero-arg
            for inner in re.findall(r"\B([A-Za-z_]\w*)\(\)",call_str):
                nested=f"{inner}()"
                if nested in call_str and inner not in last_results:
                    r_i=Tools.run_tool_once(nested)
                    ok_i,err_i=_validate(r_i)
                    last_results[inner]=r_i.get("output")
                    call_str=re.sub(rf"\b{inner}\(\)",repr(last_results[inner]),call_str)
                    # persist nested
                    try:
                        sch_i=next(
                            s for s in selected_schemas
                            if json.loads(s.metadata["schema"])["name"]==inner
                        )
                        ctx_i=ContextObject.make_stage("tool_output",[sch_i.context_id],r_i)
                        ctx_i.stage_id=f"tool_output_nested_{inner}"
                        ctx_i.summary=str(r_i.get("output")) if ok_i else f"ERROR: {err_i}"
                        ctx_i.metadata.update(r_i)
                        ctx_i.touch(); self.repo.save(ctx_i)
                        tool_ctxs.append(ctx_i)
                    except StopIteration:
                        pass

            # main call
            res = Tools.run_tool_once(call_str)
            ok,err = _validate(res)
            key = normalize_key(call_str.split("(",1)[0])
            last_results[key] = res.get("output")

            # persist
            try:
                nm=original.split("(",1)[0]
                sch=next(
                    s for s in selected_schemas
                    if json.loads(s.metadata["schema"])["name"]==nm
                )
                refs=[sch.context_id]
            except StopIteration:
                refs=[]
            ctx=ContextObject.make_stage("tool_output",refs,res)
            ctx.metadata["tool_call"]=original
            ctx.metadata.update(res)
            ctx.stage_id=f"tool_output_{nm}"
            ctx.summary=str(res.get("output")) if ok else f"ERROR: {err}"
            ctx.touch(); self.repo.save(ctx)
            tool_ctxs.append(ctx)

            # record
            call_status[original]=ok
            tracker.metadata["succeeded"]+=int(ok)
            if ok: pending.remove(original)
            else:
                tracker.metadata["errors_by_call"][original]=err
            tracker.touch(); self.repo.save(tracker)

        if not pending:
            tracker.metadata["status"]="success"
            tracker.metadata["completed_at"]=datetime.datetime.utcnow().isoformat()+"Z"
            self.repo.save(tracker)
            existing=[]
            for call in all_calls:
                matches=[
                    c for c in self.repo.query(
                        lambda c: c.component=="tool_output"
                                  and c.metadata.get("tool_call")==call
                    )
                ]
                if matches:
                    matches.sort(key=lambda c: c.timestamp,reverse=True)
                    existing.append(matches[0])
            state["tool_ctxs"]=existing
            return existing

        # ask LLM to repair pending...
        break  # for brevity, assume no further retries in this drop-in

    # final fail
    tracker.metadata["status"]="failed"
    tracker.metadata["last_errors"] = list(tracker.metadata.get("errors_by_call",{}).values())
    tracker.metadata["completed_at"]=datetime.datetime.utcnow().isoformat()+"Z"
    tracker.touch(); self.repo.save(tracker)

    state["tool_ctxs"] = tool_ctxs
    return tool_ctxs


def _stage10_assemble_and_infer(self, user_text: str, state: Dict[str, Any]) -> str:
    import json, textwrap
    from context import ContextObject

    MAX_CTX_OBJS = 20
    MAX_CHARS_PER_OBJ = 380

    def snip(txt: str) -> str:
        if not txt:
            return ""
        if len(txt) <= MAX_CHARS_PER_OBJ:
            return txt
        head = txt[:MAX_CHARS_PER_OBJ]
        stop = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
        return (head[:stop+1] if stop != -1 else head).rstrip() + " …"

    # ─── 1) Pull in the merged conversation contexts ─────────────────
    merged_ctxs = state.get("merged", [])
    # Only keep the last MAX_CTX_OBJS for inlining
    recent_merged = merged_ctxs[-MAX_CTX_OBJS:]

    # Build “Conversation so far” block
    convo_lines = [snip(c.summary or "") for c in recent_merged]
    conversation_block = ""
    if convo_lines:
        conversation_block = "[Conversation so far]\n" + "\n".join(convo_lines)

    # ─── 2) Gather & unpack any tool outputs ─────────────────────────
    tool_blocks = []
    for tc in state.get("tool_ctxs", []):
        raw = tc.metadata.get("output")
        if isinstance(raw, dict) and "results" in raw:
            for r in raw["results"]:
                ts   = r.get("timestamp", "")
                role = r.get("role", "")
                content = r.get("content", "")
                tool_blocks.append(f"{ts} {role}: {snip(content)}")
        else:
            out = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
            tool_blocks.append(f"{tc.stage_id}: {snip(out)}")

    tools_block = ""
    if tool_blocks:
        tools_block = "\n\n[Tool outputs]\n" + "\n".join(tool_blocks)

    # ─── 3) Debug print of raw tool outputs ───────────────────────────
    print("🮕🮖🮖 TOOL OUTPUTS CAPTURED 🮖🮖\n")
    for tc in state.get("tool_ctxs", []):
        print(f"🡶 {tc.stage_id} 🡷")
        print(f"{tc.metadata.get('output')!r}\n")
    print("🮕🮖🮖🮖 END TOOL OUTPUTS 🮖🮖🮖🮖\n\n")

    # ─── 4) Logging context snapshot ─────────────────────────────────
    included_ids = [c.context_id for c in recent_merged]
    self._print_stage_context("assemble_and_infer", {
        "user_question": [user_text],
        "plan":         [state.get("plan_output", "(no plan)")],
        "included_ids": included_ids,
        "snippets_count": len(convo_lines) + len(tool_blocks),
    })

    # ─── 5) Build the LLM messages ───────────────────────────────────
    final_sys = self._get_prompt("final_inference_prompt")
    plan_txt   = state.get("plan_output", "(no plan)")
    self._last_plan_output = plan_txt

    msgs = [
        {"role": "system", "content": final_sys},
    ]
    if conversation_block:
        msgs.append({"role": "system", "content": conversation_block})
    if tools_block:
        msgs.append({"role": "system", "content": tools_block})

    # Then the standard inference prompt + user query
    msgs += [
        {"role": "system", "content": self.inference_prompt},
        {"role": "user",   "content": user_text},
    ]

    # ─── 6) Call the model ────────────────────────────────────────────
    reply = self._stream_and_capture(
        self.primary_model,
        msgs,
        tag="[Assistant]",
        images=state.get("images")
    ).strip()

    # ─── 7) Persist the assistant’s reply ────────────────────────────
    refs = included_ids + [tc.context_id for tc in state.get("tool_ctxs", [])]
    resp_ctx = ContextObject.make_stage("final_inference", refs, {"text": reply})
    resp_ctx.stage_id = "final_inference"
    resp_ctx.summary  = reply
    resp_ctx.touch(); self.repo.save(resp_ctx)

    seg = ContextObject.make_segment("assistant", [resp_ctx.context_id], tags=["assistant"])
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

    # Gather the last plan
    plan_text = getattr(self, "_last_plan_output", "(no plan)")

    # 1) Unpack all tool outputs fully
    outputs = []
    for c in tool_ctxs:
        raw = c.metadata.get("output")
        if isinstance(raw, dict) and "results" in raw:
            blob = "\n".join(
                f"{r.get('timestamp','')} {r.get('role','')}: {r.get('content','')}"
                for r in raw["results"]
            )
        else:
            blob = json.dumps(raw, indent=2, ensure_ascii=False)
        outputs.append(f"[{c.stage_id}]\n{blob}")

    # 2️⃣  Relevance Extractor
    extractor_sys = (
        "You are a relevance extractor.\n"
        "Return ONLY the information that directly helps answer the user.\n"
        "Strict format: at most 8 lines, each starting with the bullet '• '."
    )
    extractor_user = "\n\n".join((
        f"USER QUESTION:\n{user_text}",
        f"PLAN EXECUTED:\n{plan_text}",
        "RAW DRAFT:\n" + draft,
        "RAW TOOL OUTPUTS:\n" + "\n\n".join(outputs),
    ))
    bullets = self._stream_and_capture(
        self.secondary_model,
        [
            {"role": "system", "content": extractor_sys},
            {"role": "user",   "content": extractor_user},
        ],
        tag="[RelevExtract]",
        images=state.get("images"),
    ).strip()

    sum_ctx = ContextObject.make_stage("relevance_summary", [], {"summary": bullets})
    sum_ctx.stage_id = "relevance_summary"
    sum_ctx.summary  = bullets
    sum_ctx.touch(); self.repo.save(sum_ctx)

    # 3️⃣  Polisher / Critic
    editor_sys = (
        "You are an expert editor.\n"
        "Use ONLY the bullet list to fix or extend the draft so it answers the "
        "user fully and directly. Do NOT invent extra content or apologies."
    )
    editor_user = "\n\n".join((
        f"USER QUESTION:\n{user_text}",
        "RELEVANT BULLETS:\n" + bullets,
        "CURRENT DRAFT:\n" + draft,
    ))
    polished = self._stream_and_capture(
        self.secondary_model,
        [
            {"role": "system", "content": editor_sys},
            {"role": "user",   "content": editor_user},
        ],
        tag="[Polisher]",
        images=state.get("images"),
    ).strip()

    if polished == draft.strip():
        return polished

    # 4️⃣  Diff & dynamic-prompt patch
    diff = difflib.unified_diff(draft.splitlines(), polished.splitlines(), lineterm="", n=1)
    diff_summary = "; ".join(ln for ln in diff if ln.startswith(("+ ", "- "))) or "(minor re-formatting)"

    patch_rows = self.repo.query(
        lambda c: c.component == "policy" and c.semantic_label == "dynamic_prompt_patch"
    )
    patch_rows.sort(key=lambda c: c.timestamp, reverse=True)
    patch = patch_rows[0] if patch_rows else ContextObject.make_policy(
        "dynamic_prompt_patch", "", tags=["dynamic_prompt"]
    )
    if patch.summary != diff_summary:
        patch.summary = diff_summary
        patch.metadata["policy"] = diff_summary
        patch.touch(); self.repo.save(patch)

    # 5️⃣  Persist polished reply & critique
    resp_ctx = ContextObject.make_stage("response_critique", [sum_ctx.context_id], {"text": polished})
    resp_ctx.stage_id = "response_critique"
    resp_ctx.summary  = polished
    resp_ctx.touch(); self.repo.save(resp_ctx)

    critique_ctx = ContextObject.make_stage(
        "plan_critique",
        [resp_ctx.context_id] + [c.context_id for c in tool_ctxs],
        {"critique": polished, "diff": diff_summary},
    )
    critique_ctx.component      = "analysis"
    critique_ctx.semantic_label = "plan_critique"
    critique_ctx.touch(); self.repo.save(critique_ctx)

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
            q_ctx.touch(); self.repo.save(q_ctx)

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
            a_ctx.touch(); self.repo.save(a_ctx)

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
                    p_ctx.touch(); self.repo.save(p_ctx)

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
