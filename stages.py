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


def _stage1_record_input(self, user_text: str) -> ContextObject:
    # record the new user input
    ctx = ContextObject.make_segment("user_input", [], tags=["user_input"])
    ctx.summary = user_text
    ctx.stage_id = "user_input"
    ctx.touch()
    self.repo.save(ctx)
    return ctx

def _stage2_load_system_prompts(self) -> ContextObject:
    return self._load_system_prompts()

def _stage3_retrieve_and_merge_context(
    self,
    user_text: str,
    user_ctx: ContextObject,
    sys_ctx: ContextObject,
    extra_ctx: List[ContextObject] = None,
    recall_ids: List[str] = None
) -> Dict[str, Any]:
    """
    Merge system prompt, narrative (episodic memory), working memory (short-term),
    semantic retrieval (long-term memory), associative recall (spreading activation),
    and recent tool outputs — using human-inspired capacities and decay.
    """
    import re
    from datetime import datetime as _dt, timezone, timedelta
    import concurrent.futures

    # ——— Helpers —————————————————————————————————————————————————————————————
    now = default_clock()

    def _to_dt(ts):
        """
        Parse ISO timestamp or datetime, and always return a naïve UTC datetime.
        """
        if isinstance(ts, str):
            try:
                # parse as aware UTC, then drop tzinfo
                dt = _dt.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception as e:
                # fallback to now if parse fails
                print(f"⚠️ [timestamp parse error] {ts!r} → {e}")
                return now
        if isinstance(ts, _dt):
            # drop any tzinfo if present
            return ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts
        return now

    def _dedupe(objs):
        seen = set()
        unique = []
        for c in objs:
            if c.summary not in seen:
                unique.append(c)
                seen.add(c.summary)
        return unique

    # Human-style capacity limits
    WM_CAPACITY = 5       # working memory slots
    ST_CAPACITY = 10      # short-term history slots
    LT_K = self.top_k     # long-term retrieval slots

    # ——— Step A: decide context window size & tags —————————————————————
    with concurrent.futures.ThreadPoolExecutor() as ex:
        f_count = ex.submit(self.decision_callback, user_text,
                            ["5","10","20","50"],
                            "How many recent turns to keep? {arg1}, {arg2}, {arg3}, or {arg4}.",
                            history_size=0, var_names=["arg1","arg2","arg3","arg4"])
        f_window = ex.submit(self.decision_callback, user_text,
                                ["5m","30m","1h","6h","1d"],
                                "Time window for memory: {arg1}, {arg2}, {arg3}, {arg4}, or {arg5}.",
                                history_size=0, var_names=["arg1","arg2","arg3","arg4","arg5"])
        f_tags = ex.submit(self.decision_callback, user_text,
                            ["important","task","decision_point","question"],
                            "Salient tags to boost: {arg1}, {arg2}, {arg3}, or {arg4}.",
                            history_size=0, var_names=["arg1","arg2","arg3","arg4"])
    # parse count
    try:
        n = int(re.search(r"\d+", f_count.result()).group())
        ST_CAPACITY = min(max(n, 1), 50)
    except:
        pass
    # parse window
    try:
        w = f_window.result()
        num, unit = re.match(r"(\d+)([mhd])", w).groups()
        delta = {"m":60, "h":3600, "d":86400}[unit]
        time_cutoff = now - timedelta(seconds=int(num)*delta)
    except:
        time_cutoff = now - timedelta(minutes=5)
    # parse tags
    try:
        sim_tags = [t.strip() for t in f_tags.result().split(",") if t.strip()]
    except:
        sim_tags = ["important", "decision_point"]

    # ——— Step B: episodic memory (keep just the most recent narrative event) —
    all_narr = sorted(
        [c for c in self.repo.query(lambda c: c.component=="narrative")],
        key=lambda c: _to_dt(c.timestamp), reverse=True
    )
    narr_ctx = all_narr[0] if all_narr else self._load_narrative_context()

    # ——— Step C: working memory ——————————————————————————————————————
    segments = sorted(
        [c for c in self.repo.query(
            lambda c: c.domain=="segment" and c.semantic_label in ("user_input","assistant")
        )],
        key=lambda c: _to_dt(c.timestamp)
    )[-WM_CAPACITY:]
    inferences = sorted(
        [c for c in self.repo.query(lambda c: c.semantic_label=="final_inference")],
        key=lambda c: _to_dt(c.timestamp)
    )[-WM_CAPACITY:]
    segments = _dedupe(segments)
    inferences = _dedupe(inferences)

    # ——— Step D: short-term memory (recent conversation) ——————————————————
    recent = sorted(
        [c for c in self.repo.query(
            lambda c: c.domain=="segment"
                        and c.semantic_label in ("user_input","assistant")
                        and _to_dt(c.timestamp) >= time_cutoff
        )],
        key=lambda c: _to_dt(c.timestamp)
    )[-ST_CAPACITY:]
    if extra_ctx:
        extra_ctx = _dedupe(extra_ctx)
        recent.extend(extra_ctx)
    recent = _dedupe(recent)

    # ——— Step E: semantic (long-term) retrieval ——————————————————————
    if self.rl.should_run("semantic_retrieval", 0.0):
        tr = ( (now - timedelta(seconds=(now - time_cutoff).total_seconds())).strftime("%Y%m%dT%H%M%SZ"),
                now.strftime("%Y%m%dT%H%M%SZ") )
        semantic = self.engine.query(
            stage_id="recent_retrieval",
            time_range=tr,
            similarity_to=user_text,
            include_tags=sim_tags,
            exclude_tags=self.STAGES + ["tool_schema","tool_output","assistant","system_prompt"],
            top_k=LT_K
        )
    else:
        semantic = []

    # ——— Step F: associative recall (spreading activation with decay) —————
    assoc = []
    if self.rl.should_run("memory_retrieval", 0.0):
        seeds = [user_ctx.context_id, narr_ctx.context_id]
        scores = self.memman.spread_activation(
            seed_ids=seeds,
            hops=3, decay=0.7,
            assoc_weight=1.0, recency_weight=1.0
        )
        for cid in sorted(scores, key=scores.get, reverse=True)[:LT_K]:
            try:
                cobj = self.repo.get(cid)
                cobj.retrieval_score = scores[cid]
                assoc.append(cobj)
            except KeyError:
                continue

    # ——— Step G: recent tool outputs ——————————————————————————————
    tools = sorted(
        [c for c in self.repo.query(lambda c: c.component=="tool_output")],
        key=lambda c: _to_dt(c.timestamp)
    )
    recent_tools = tools[-LT_K:] if self.rl.should_run("tool_output_retrieval", 0.0) else []

    # ——— Step H: merge in human-inspired priority order —————————————
    merged = []
    seen = set()
    def _add(bucket):
        for c in bucket:
            if c.summary not in seen:
                merged.append(c)
                seen.add(c.summary)

    _add([sys_ctx])
    _add([user_ctx])
    _add(segments + inferences)
    _add(recent)
    _add(semantic)
    _add(assoc)
    _add(recent_tools)

    merged_ids = [c.context_id for c in merged]
    wm_ids     = [c.context_id for c in segments + inferences]

    return {
        "narrative_ctx": narr_ctx,
        "history":       recent,
        "recent":        semantic,
        "assoc":         assoc,
        "recent_ids":    merged_ids,
        "wm_ids":        wm_ids,
    }

def _stage4_intent_clarification(self, user_text: str, state: Dict[str, Any]) -> ContextObject:
    import json, re

    # 1) Build the context block, now including the narrative up front
    pieces = [
        state['sys_ctx'],
        state['narrative_ctx'],       # ← inject running narrative
    ] + state['history'] + state['recent'] + state['assoc']

    block = "\n".join(f"[{c.semantic_label}] {c.summary}" for c in pieces)

    # 2) Ask the clarifier for valid JSON
    clarifier_system = self.clarifier_prompt  # “Output only valid JSON…”
    msgs = [
        {"role": "system", "content": clarifier_system},
        {"role": "system", "content": f"Context:\n{block}"},
        {"role": "user",   "content": user_text},
    ]

    out = self._stream_and_capture(self.secondary_model, msgs, tag="[Clarifier]", images=state.get("images"))
    # retry once on JSON parse failure
    for attempt in (1, 2):
        try:
            clar = json.loads(out)
            break
        except json.JSONDecodeError:
            if attempt == 1:
                retry_sys = (
                    "⚠️ Your last response wasn’t valid JSON.  "
                    "Please output only JSON with keys `keywords` and `notes`."
                )
                out = self._stream_and_capture(
                    self.secondary_model,
                    [
                        {"role": "system", "content": retry_sys},
                        {"role": "user",   "content": out}
                    ],
                    tag="[ClarifierRetry]",
                    images=state.get("images")
                )
            else:
                # give up: wrap raw text into notes
                clar = {"keywords": [], "notes": out}

    # 3) Build the ContextObject, referencing narrative_ctx as well
    input_refs = [
        state['user_ctx'].context_id,
        state['sys_ctx'].context_id,
        state['narrative_ctx'].context_id,
    ] + state['recent_ids']

    ctx = ContextObject.make_stage(
        "intent_clarification",
        input_refs,
        clar
    )
    ctx.stage_id = "intent_clarification"
    ctx.summary  = clar.get("notes", "")
    ctx.touch()
    self.repo.save(ctx)
    return ctx

def _stage5_external_knowledge(self, clar_ctx: ContextObject, state: Dict[str,Any] = None) -> ContextObject:
    import json
    import concurrent.futures
    from datetime import datetime, timedelta

    now = default_clock()

    # ── 0) Prune stale/overflow contexts ──────────────────────────────────
    prune_summary = self._stage_prune_context_store({})

    # ── 1) In parallel, gather raw ContextObjects ────────────────────────
    with concurrent.futures.ThreadPoolExecutor() as exec:
        fut_seg = exec.submit(lambda: sorted(
            [c for c in self.repo.query(
                lambda c: c.domain=="segment" and c.semantic_label in ("user_input","assistant")
            )],
            key=lambda c: c.timestamp, reverse=True
        )[: self.top_k])

        fut_inf = exec.submit(lambda: sorted(
            [c for c in self.repo.query(
                lambda c: c.semantic_label=="final_inference"
            )],
            key=lambda c: c.timestamp, reverse=True
        )[: self.top_k])

        fut_sim = exec.submit(lambda: [
            h for kw in clar_ctx.metadata.get("keywords", [])
            for h in self.engine.query(
                stage_id="external_knowledge_retrieval",
                similarity_to=kw,
                top_k=self.top_k
            )
        ])

        fut_loc = exec.submit(lambda: [
            c for c in self.repo.query(
                lambda c: (
                    c.semantic_label in ("user_input","assistant","final_inference")
                    and datetime.fromisoformat(c.timestamp.rstrip("Z")) >= now - timedelta(hours=1)
                    and any(kw.lower() in (c.summary or "").lower() for kw in clar_ctx.metadata.get("keywords", []))
                )
            )
        ])

    segs = fut_seg.result(timeout=2.0) if fut_seg else []
    infs = fut_inf.result(timeout=2.0) if fut_inf else []
    sims = fut_sim.result(timeout=2.0) if fut_sim else []
    locs = fut_loc.result(timeout=2.0) if fut_loc else []

    # ── 2) snippet‐truncation helper ───────────────────────────────────────
    MAX_SNIP_LEN = 200
    def truncate(txt: str) -> str:
        if len(txt) > MAX_SNIP_LEN:
            return txt[:MAX_SNIP_LEN].rstrip() + "…"
        return txt

    # ── 3) Build working memory entries (truncated) ───────────────────────
    working_memory = []
    for c in reversed(segs):
        working_memory.append(f"(WM)[{c.semantic_label}] {truncate(c.summary)}")
    for c in reversed(infs):
        working_memory.append(f"(WM)[{c.semantic_label}] {truncate(c.summary)}")

    # ── 4) Build similarity snippets ──────────────────────────────────────
    sim_snips = [
        f"(EXT)[{','.join(h.tags)}] {truncate(h.summary)}"
        for h in sims
    ]

    # ── 5) Build local‐match snippets ─────────────────────────────────────
    loc_snips = []
    for c in locs:
        tag = "SEG" if c.domain=="segment" else "INF"
        loc_snips.append(f"(LOC-{tag})[{c.semantic_label}] {truncate(c.summary)}")

    # ── 6) Fallback to last 5 segments if nothing collected ───────────────
    if not (working_memory or sim_snips or loc_snips):
        fallback = sorted(
            [c for c in self.repo.query(
                lambda c: c.domain=="segment" and c.semantic_label in ("user_input","assistant")
            )],
            key=lambda c: c.timestamp, reverse=True
        )[:5]
        for c in reversed(fallback):
            working_memory.append(f"(FB)[{c.semantic_label}] {truncate(c.summary)}")

    # ── 7) Combine, dedupe, and cap total snippet count ───────────────────
    all_snips = working_memory + sim_snips + loc_snips
    seen = set(); unique = []
    for s in all_snips:
        if s not in seen:
            seen.add(s); unique.append(s)
            if len(unique) >= self.top_k * 3:  # avoid runaway length
                break

    # ── 8) Persist and return as a ContextObject ─────────────────────────
    summary_text = "\n".join(unique) or "(none)"
    ctx = ContextObject.make_stage(
        "external_knowledge_retrieval",
        clar_ctx.references,
        {"snippets": unique}
    )
    ctx.stage_id = "external_knowledge_retrieval"
    ctx.summary  = summary_text
    ctx.touch()
    self.repo.save(ctx)

    # ── 9) Debug print ────────────────────────────────────────────────────
    self._print_stage_context("external_knowledge_retrieval", {
        "pruned":         prune_summary,
        "working_memory": working_memory or ["(none)"],
        "similarity":     sim_snips     or ["(none)"],
        "local_matches":  loc_snips     or ["(none)"],
        "total_snips":    len(unique),
    })

    return ctx

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
    2) Inject truncated tool list (name + short desc)
    3) Run up to 3 JSON-only planning passes, halving snippets each retry
    4) Take the raw plan, find which tools were selected
    5) Inject full schemas for only those tools and do one final “fill in missing args” pass
    6) Serialize to plan_json and persist
    """
    import json, re, hashlib, datetime, textwrap

    def _clean_json_block(text: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if m:
            return m.group(1)
        m2 = re.search(r"(\{.*\})", text, flags=re.S)
        return (m2.group(1) if m2 else text).strip()

    # 0) load any prior plan critiques
    critique_rows = self.repo.query(
        lambda c: c.component == "analysis" and c.semantic_label == "plan_critique"
    )
    critique_rows.sort(key=lambda c: c.timestamp)
    critique_ids = [c.context_id for c in critique_rows]

    # 1) fetch the “planning_prompt” artifact
    prompt_rows = self.repo.query(
        lambda c: c.component == "artifact" and c.semantic_label == "planning_prompt"
    )
    prompt_rows.sort(key=lambda c: c.timestamp, reverse=True)
    system_base = prompt_rows[0].summary if prompt_rows else self._get_prompt("planning_prompt")

    # 2) assemble truncated tool list
    tool_lines = "\n".join(
        f"- **{t['name']}**: {t['description']}"
        for t in tools_list
    ) or "(none)"
    base_system = "\n\n".join([
        system_base.rstrip(),
        "Available tools:",
        tool_lines
    ])
    replan_system = "\n\n".join([
        "Your last plan was invalid—**OUTPUT ONLY** the JSON, no extra text.",
        "Available tools:",
        tool_lines
    ])

    # 3) prepare user block
    original_snips = (know_ctx.summary or "").splitlines()
    def build_user(snips):
        return "\n\n".join([
            f"User question:\n{user_text}",
            f"Clarified intent:\n{clar_ctx.summary}",
            "Snippets:\n" + ("\n".join(snips) if snips else "(none)")
        ])
    full_user = build_user(original_snips)

    # 4) up to 3 planning passes
    last_calls = None
    plan_obj   = None
    for attempt in range(1, 4):
        if attempt == 1:
            sys_p, user_p, tag = base_system, full_user, "[Planner]"
        else:
            keep = max(1, len(original_snips)//(2**(attempt-1)))
            sys_p, user_p, tag = replan_system, build_user(original_snips[:keep]), "[PlannerReplan]"

        raw = self._stream_and_capture(
            self.secondary_model,
            [{"role":"system","content":sys_p},
             {"role":"user",  "content":user_p}],
            tag=tag,
            images=state.get("images")
        )
        cleaned = _clean_json_block(raw)
        try:
            cand = json.loads(cleaned)
            assert isinstance(cand, dict) and "tasks" in cand
            plan_obj = cand
        except:
            calls = re.findall(r"\b[A-Za-z_]\w*\([^)]*\)", raw)
            plan_obj = {"tasks":[{"call":c,"tool_input":{},"subtasks":[]} for c in calls]}

        # reject unknown tools
        valid   = {t["name"] for t in tools_list}
        unknown = [t["call"] for t in plan_obj["tasks"] if t["call"] not in valid]
        if unknown:
            self._print_stage_context(
                f"planning_summary:unknown_tools(attempt={attempt})",
                {"unknown":unknown,"allowed":sorted(valid)}
            )
            continue

        # plateau guard
        this_calls = [t["call"] for t in plan_obj["tasks"]]
        if last_calls is not None and this_calls == last_calls:
            self._print_stage_context(
                f"planning_summary:plateaued(attempt={attempt})",
                {"calls":this_calls}
            )
        last_calls = this_calls
        break

    # 5) now automatically refine by presenting full schemas for selected tools
    selected = {t["call"] for t in plan_obj.get("tasks", [])}
    schema_lines = []
    for t in tools_list:
        if t["name"] in selected:
            schema_lines.append(
                f"**{t['name']}**\n```json\n"
                + json.dumps(t["schema"], indent=2)
                + "\n```"
            )
    if schema_lines:
        refine_system = "Fill in all missing tool_input arguments using the schemas below:\n\n" + "\n\n".join(schema_lines)
        refine_user   = json.dumps({"tasks": plan_obj["tasks"]})
        raw2 = self._stream_and_capture(
            self.secondary_model,
            [{"role":"system","content":refine_system},
             {"role":"user",  "content":refine_user}],
            tag="[PlannerRefine]",
            images=state.get("images")
        )
        cleaned2 = _clean_json_block(raw2)
        try:
            cand2 = json.loads(cleaned2)
            assert isinstance(cand2, dict) and "tasks" in cand2
            plan_obj = cand2
        except:
            pass

    # 6) flatten & serialize
    def _flatten(task):
        out = [task]
        for sub in task.get("subtasks", []):
            out.extend(_flatten(sub))
        return out

    flat = []
    for t in plan_obj.get("tasks", []):
        flat.extend(_flatten(t))

    call_strings = []
    for task in flat:
        name   = task["call"]
        params = task.get("tool_input", {}) or {}
        if params:
            args = ",".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in params.items())
            call_strings.append(f"{name}({args})")
        else:
            call_strings.append(f"{name}()")

    plan_json = json.dumps({"tasks":[{"call":s,"subtasks":[]} for s in call_strings]})
    plan_sig  = hashlib.md5(plan_json.encode("utf-8")).hexdigest()[:8]

    # 7) persist planning_summary + signal + tracker
    ctx = ContextObject.make_stage(
        "planning_summary",
        clar_ctx.references + know_ctx.references + critique_ids,
        {"plan":plan_obj, "attempt":attempt, "plan_id":plan_sig}
    )
    ctx.stage_id = f"planning_summary_{plan_sig}"
    ctx.summary  = plan_json
    ctx.touch(); self.repo.save(ctx)

    succ_cls = ContextObject.make_success if call_strings else ContextObject.make_failure
    succ_msg = f"Planner → {len(call_strings)} task(s)" if call_strings else "Planner → empty plan"
    succ = succ_cls(succ_msg, refs=[ctx.context_id])
    succ.stage_id = f"planning_summary_signal_{plan_sig}"
    succ.touch(); self.repo.save(succ)

    tracker = ContextObject.make_stage(
        "plan_tracker",
        [ctx.context_id],
        {
            "plan_id":     plan_sig,
            "total_calls": len(call_strings),
            "succeeded":   0,
            "attempts":    0,
            "status":      "in_progress",
            "started_at":  datetime.datetime.utcnow().isoformat() + "Z"
        }
    )
    tracker.semantic_label = plan_sig
    tracker.stage_id       = f"plan_tracker_{plan_sig}"
    tracker.summary        = "initialized plan tracker"
    tracker.touch(); self.repo.save(tracker)

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
    tools_list: List[Dict[str, str]],
    state: Dict[str, Any]
) -> Tuple[ContextObject, List[str], List[ContextObject]]:
    import json, re

    # ── A) JSON-first extraction of calls ──────────────────────────────
    calls: List[str] = []
    try:
        plan = json.loads(plan_output)
        def _flatten(tasks):
            out = []
            for t in tasks:
                out.append(t)
                out.extend(_flatten(t.get("subtasks", [])))
            return out

        flat = _flatten(plan.get("tasks", []))
        for t in flat:
            name = t["call"]
            inp  = t.get("tool_input", {}) or {}
            if inp:
                args = ",".join(
                    f'{k}={json.dumps(v, ensure_ascii=False)}'
                    for k, v in inp.items()
                )
                calls.append(f"{name}({args})")
            else:
                calls.append(f"{name}()")

    except Exception:
        # only if JSON really is broken, fallback to regex
        parsed = Tools.parse_tool_call(plan_output)
        if parsed:
            raw = [parsed]
        else:
            raw = re.findall(r'\b[A-Za-z_]\w*\([^)]*\)', plan_output)
        seen = set()
        for c in raw:
            if c not in seen:
                seen.add(c)
                calls.append(c)

    # ── B) Load matching schemas & build docs blob ──────────────────
    all_schemas = self.repo.query(
        lambda c: c.component=="schema" and "tool_schema" in c.tags
    )
    selected_schemas, docs_blob_parts = [], []
    wanted = {c.split("(")[0] for c in calls}
    for sch_obj in all_schemas:
        schema = json.loads(sch_obj.metadata["schema"])
        if schema["name"] in wanted:
            selected_schemas.append(sch_obj)
            docs_blob_parts.append(
                f"**{schema['name']}**\n```json\n"
                + json.dumps(schema, indent=2)
                + "\n```"
            )
    docs_blob = "\n\n".join(docs_blob_parts)

    # ── C) Confirm with the LLM ────────────────────────────────────
    system = self._get_prompt("toolchain_prompt") + docs_blob
    self._print_stage_context("tool_chaining", {
        "plan":    [plan_output[:200]],
        "schemas": docs_blob_parts
    })
    out = self._stream_and_capture(
        self.secondary_model,
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": json.dumps({"tool_calls": calls})},
        ],
        tag="[ToolChain]"
    )

    # ── D) Parse back the confirmed list, then normalise ───────────────
    raw_confirmed = calls
    try:
        blob = re.search(r'\{.*"tool_calls".*?\}', out, flags=re.S).group(0)
        parsed2 = json.loads(blob)
        if isinstance(parsed2.get("tool_calls"), list):
            raw_confirmed = parsed2["tool_calls"]
    except Exception:
        pass

    # --- normalise any dict objects to strings ------------------------
    def _obj2str(item):
        if isinstance(item, dict):
            name = item.get("name") or item.get("tool_name") or item.get("call")
            args = item.get("arguments", item.get("parameters", {})) or {}
            if args:
                arg_s = ",".join(f'{k}={json.dumps(v, ensure_ascii=False)}'
                                 for k, v in args.items())
                return f"{name}({arg_s})"
            return f"{name}()"
        return str(item).strip()

    confirmed = [_obj2str(c) for c in raw_confirmed]

    # ── E) Persist & return ────────────────────────────────────────────
    tc_ctx = ContextObject.make_stage(
        "tool_chaining",
        plan_ctx.references + [s.context_id for s in selected_schemas],
        {"tool_calls": confirmed, "tool_docs": docs_blob}
    )
    tc_ctx.stage_id = "tool_chaining"
    tc_ctx.summary  = json.dumps(confirmed)
    tc_ctx.touch(); self.repo.save(tc_ctx)

    return tc_ctx, confirmed, selected_schemas

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
    selected_schemas: List["ContextObject"],
    user_text: str,
    clar_metadata: Dict[str, Any],
    state: Dict[str, Any]
) -> List["ContextObject"]:
    import json, re, hashlib, datetime, logging
    from typing import Tuple, Any, Dict, List

    # ── 0) Tracker init ────────────────────────────────────────────────
    plan_sig = hashlib.md5(plan_output.encode("utf-8")).hexdigest()[:8]
    tracker = next(
        (c for c in self.repo.query(
            lambda c: c.component == "plan_tracker" and c.semantic_label == plan_sig
        )),
        None
    )
    if not tracker:
        tracker = ContextObject.make_stage(
            "plan_tracker", [], {
                "plan_id":           plan_sig,
                "plan_calls":        raw_calls.copy(),
                "total_calls":       len(raw_calls),
                "succeeded":         0,
                "attempts":          0,
                "call_status_map":   {},
                "errors_by_call":    {},
                "status":            "in_progress",
                "started_at":        datetime.datetime.utcnow().isoformat() + "Z"
            }
        )
        tracker.semantic_label = plan_sig
        tracker.stage_id       = f"plan_tracker_{plan_sig}"
        tracker.summary        = "initialized plan tracker"
        tracker.touch(); self.repo.save(tracker)
    else:
        meta = tracker.metadata
        meta.setdefault("plan_calls",      raw_calls.copy())
        meta.setdefault("total_calls",     len(raw_calls))
        meta.setdefault("succeeded",       0)
        meta.setdefault("attempts",        0)
        meta.setdefault("call_status_map", {})
        meta.setdefault("errors_by_call",  {})
        meta.setdefault("status",          "in_progress")
        meta.setdefault("started_at",      datetime.datetime.utcnow().isoformat() + "Z")
        tracker.touch(); self.repo.save(tracker)

    tracker.metadata["attempts"] += 1
    tracker.metadata["last_attempt_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    tracker.touch()
    self.repo.save(tracker)

    # — if we've already succeeded, gather and return the previous outputs —
    if tracker.metadata.get("status") == "success":
        existing: List[ContextObject] = []
        # all_calls was normalized earlier from raw_calls
        for call in all_calls:
            matches = [
                c for c in self.repo.query(
                    lambda c: c.component == "tool_output"
                                and c.metadata.get("tool_call") == call
                )
            ]
            if matches:
                # pick the most recent one
                matches.sort(key=lambda c: c.timestamp, reverse=True)
                existing.append(matches[0])
        return existing

    # ── Helpers ─────────────────────────────────────────────────────────
    def _norm(calls: List[Any]) -> List[str]:
        out = []
        for c in calls:
            if isinstance(c, dict) and "tool_call" in c:
                out.append(c["tool_call"])
            elif isinstance(c, str):
                out.append(c)
        return out

    def _validate(res: Dict[str, Any]) -> Tuple[bool, str]:
        exc = res.get("exception")
        return (exc is None, exc or "")

    def normalize_key(k: str) -> str:
        return re.sub(r"\W+", "", k).lower()

    # ── 1) Initialise pending calls in original order ────────────────
    all_calls   = _norm(raw_calls)
    call_status = tracker.metadata["call_status_map"]
    pending     = [c for c in all_calls if not call_status.get(c, False)]
    last_results: Dict[str, Any] = {}
    tool_ctxs: List[ContextObject] = []

    # ── 1b) If nothing to run, return previously-saved outputs ────────
    if not pending:
        tracker.metadata["status"] = "success"
        tracker.metadata["completed_at"] = _utc_iso()
        tracker.touch(); self.repo.save(tracker)

        existing: List[ContextObject] = []
        for call in all_calls:
            matches = [
                c for c in self.repo.query(
                    lambda c:
                        c.component == "tool_output"
                        and c.metadata.get("tool_call") == call
                )
            ]
            if matches:
                matches.sort(key=lambda c: c.timestamp, reverse=True)
                existing.append(matches[0])
        return existing

    # ── 2) Retry loop over only pending calls ────────────────────────────
    max_retries = 10
    prev_pending = None
    for attempt in range(1, max_retries + 1):
        errors: List[Tuple[str, str]] = []

        # plateau detection
        if prev_pending is not None and pending == prev_pending:
            logging.warning(f"[ToolChainRetry] plateau on attempt {attempt}, giving up")
            break
        prev_pending = pending.copy()

        import random
        random.shuffle(pending)

        for original in list(pending):
            call_str = original

            # 1) [alias from tool_name] placeholders
            for ph in re.findall(r"\[([^\]]+)\]", call_str):
                if " from " in ph:
                    alias, toolname = ph.split(" from ", 1)
                    tool_key = normalize_key(toolname)
                    match = tool_key if tool_key in last_results else None
                else:
                    phn = normalize_key(ph)
                    match = next(
                        (k for k in last_results
                            if phn in normalize_key(k)
                            or normalize_key(k) in phn),
                        None
                    )
                if match:
                    call_str = call_str.replace(f"[{ph}]", repr(last_results[match]))

            # 2) {{alias}} placeholders
            for ph in re.findall(r"\{\{([^}]+)\}\}", call_str):
                phn = normalize_key(ph)
                match = next(
                    (k for k in last_results
                        if phn in normalize_key(k)
                        or normalize_key(k) in phn),
                    None
                )
                if match:
                    call_str = call_str.replace(f"{{{{{ph}}}}}", repr(last_results[match]))

            # 3) inline nested zero-arg calls if embedded
            for inner in re.findall(r"\B([A-Za-z_]\w*)\(\)", call_str):
                nested = f"{inner}()"
                if nested in call_str and inner not in last_results:
                    r_i = Tools.run_tool_once(nested)
                    ok_i, err_i = _validate(r_i)
                    out_i = r_i.get("output")
                    last_results[inner] = out_i
                    call_str = re.sub(
                        rf"\b{re.escape(inner)}\(\)",
                        repr(out_i),
                        call_str
                    )
                    try:
                        sch_i = next(
                            s for s in selected_schemas
                            if json.loads(s.metadata["schema"])["name"] == inner
                        )
                        ctx_i = ContextObject.make_stage(
                            "tool_output", [sch_i.context_id], r_i
                        )
                        ctx_i.stage_id = f"tool_output_nested_{inner}"
                        ctx_i.summary = str(out_i) if ok_i else f"ERROR: {err_i}"
                        ctx_i.metadata.update(r_i)
                        ctx_i.touch(); self.repo.save(ctx_i)
                        tool_ctxs.append(ctx_i)
                    except StopIteration:
                        pass

            # 4) main invocation
            res = Tools.run_tool_once(call_str)
            ok, err = _validate(res)
            tool_key = normalize_key(call_str.split("(", 1)[0])
            last_results[tool_key] = res.get("output")

            # persist output
            try:
                name = original.split("(", 1)[0]
                sch = next(
                    s for s in selected_schemas
                    if json.loads(s.metadata["schema"])["name"] == name
                )
                refs = [sch.context_id]
            except StopIteration:
                refs = []
            ctx = ContextObject.make_stage("tool_output", refs, res)
            ctx.metadata["tool_call"] = original
            ctx.metadata.update(res)
            ctx.stage_id = f"tool_output_{name}"
            ctx.summary = str(res.get("output")) if ok else f"ERROR: {err}"
            ctx.touch();
            self.repo.save(ctx)
            tool_ctxs.append(ctx)

            # record success/failure
            call_status[original] = ok
            tracker.metadata["succeeded"] += int(ok)
            if ok:
                pending.remove(original)
            else:
                tracker.metadata["errors_by_call"][original] = err
                errors.append((original, err))
            tracker.touch(); self.repo.save(tracker)

        # if all succeeded, finish
        if not pending:
            tracker.metadata["status"] = "success"
            tracker.metadata["completed_at"] = _utc_iso()
            self.repo.save(tracker)
            existing = []
            for call in all_calls:
                matches = [c for c in self.repo.query(
                    lambda c: c.component=="tool_output"
                                and c.metadata.get("tool_call")==call
                )]
                if matches:
                    matches.sort(key=lambda c: c.timestamp, reverse=True)
                    existing.append(matches[0])
            return existing

        # otherwise, ask LLM to repair remaining calls
        retry_sys = (
            self._get_prompt("toolchain_retry_prompt")
            + "\n\nOriginal question:\n" + user_text
            + "\n\nPlan:\n" + plan_output
            + "\n\nPending calls:\n" + "\n".join(pending)
            + "\n\nErrors so far:\n"
            + "\n".join(f"- {c}: {e}" for c, e in errors)
            + "\n\n(Feel free to reorder or adjust these calls.)"
        )
        retry_msgs = [
            {"role": "system", "content": retry_sys},
            {"role": "user",   "content": json.dumps({"tool_calls": pending})},
        ]
        out = self._stream_and_capture(
            self.secondary_model, retry_msgs, tag="[ToolChainRetry]", images=state.get("images")
        ).strip()
        try:
            pending = json.loads(out)["tool_calls"]
        except:
            parsed = Tools.parse_tool_call(out)
            pending = _norm(parsed if isinstance(parsed, list) else [parsed] or pending)

    # ── 3) After max_retries or exit, mark failure ───────────────────────
    if pending:
        tracker.metadata["status"] = "failed"
        tracker.metadata["errors_by_call"].update({c: "unresolved" for c in pending})
        tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        tracker.touch(); self.repo.save(tracker)

    tracker.metadata["status"]      = "failed"
    tracker.metadata["last_errors"] = [e for _, e in errors]
    tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    tracker.touch(); self.repo.save(tracker)
    return tool_ctxs

def _stage10_assemble_and_infer(self, user_text: str, state: Dict[str, Any]) -> str:
    """
    Assemble every bit of context we have, run one final LLM pass, and
    persist the answer — even if earlier tool stages failed or were skipped.
    Also record the assistant’s reply as a segment for future context.
    """
    import json
    from context import ContextObject  # adjust import as needed

    print(
        "[assemble_and_infer] tool_ctxs = "
        + ", ".join(f"{t.stage_id}:{t.metadata.get('output')!r}" for t in state.get("tool_ctxs", [])),
        "DEBUG"
    )
    print("[DEBUG] → entering assemble_and_infer with state keys:", list(state.keys()))
    print("[DEBUG]    plan_output =", repr(state.get("plan_output")))
    print("[DEBUG]    tc_ctx      =", getattr(state.get("tc_ctx"), "context_id", None))
    print("[DEBUG]    tool_ctxs   =", [t.context_id for t in state.get("tool_ctxs", [])])
    print("[DEBUG]    wm_ids      =", state.get("wm_ids"))
    print("[DEBUG]    recent_ids  =", state.get("recent_ids"))

    # ─── 1) Collect all relevant context-IDs ──────────────────────────────────
    refs: list[str] = []
    def _maybe_add(key: str):
        ctx = state.get(key)
        if ctx:
            refs.append(ctx.context_id)

    _maybe_add("user_ctx")
    _maybe_add("sys_ctx")
    _maybe_add("tc_ctx")    # include the tool_chaining context
    refs.extend(state.get("wm_ids", []))
    refs.extend(state.get("recent_ids", []))
    _maybe_add("clar_ctx")
    _maybe_add("know_ctx")
    _maybe_add("plan_ctx")
    for t in state.get("tool_ctxs", []):
        refs.append(t.context_id)

    # de-duplicate while preserving order
    seen, ordered = set(), []
    for cid in refs:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    # ─── 2) Load the ContextObjects, skipping missing ones ───────────────────
    ctxs = []
    for cid in ordered:
        try:
            ctx = self.repo.get(cid)
        except KeyError:
            print(f"[DEBUG]    missing context {cid}")
            continue
        print(f"[DEBUG]    loaded {cid} → {ctx.summary!r}")
        ctxs.append(ctx)
    ctxs.sort(key=lambda c: c.timestamp)

    # ─── 3) Inline into one big “knowledge sheet” ──────────────────────────
    interm_parts: list[str] = []
    for c in ctxs:
        if c.component == "tool_output":
            out = c.metadata.get("output")
            try:
                blob = json.dumps(out, indent=2, ensure_ascii=False)
            except Exception:
                blob = repr(out)
            interm_parts.append(f"[{c.stage_id} (tool output)]\n{blob}")
        elif c.semantic_label == "tool_chaining":
            docs = c.metadata.get("tool_docs", "")
            interm_parts.append(f"[tool_schemas]\n{docs}")
        else:
            interm_parts.append(f"[{c.stage_id}] {c.summary}")
    interm = "\n\n".join(interm_parts)

    self._print_stage_context("assemble_and_infer", {
        "user_question":    [user_text],
        "plan":             [state.get("plan_output", "(no plan)")],
        "inference_prompt": [self.inference_prompt],
        "inlined_context":  interm_parts,
    })

    # ─── 4) Final LLM call ──────────────────────────────────────────────────
    final_sys = self._get_prompt("final_inference_prompt")
    plan_text = state.get("plan_output", "(no plan)")
    self._last_plan_output = plan_text

    msgs = [
        {"role": "system", "content": final_sys},
        {"role": "system", "content": f"User question:\n{user_text}"},
        {"role": "system", "content": f"Plan:\n{plan_text}"},
        {"role": "system", "content": self.inference_prompt},
        {"role": "system", "content": interm},
        {"role": "user",   "content": user_text},
    ]
    print("[DEBUG]    refs to load:", ordered)
    reply = self._stream_and_capture(
        self.primary_model, msgs, tag="[Assistant]", images=state.get("images")
    ).strip()

    # ─── 5) Persist the answer ──────────────────────────────────────────────
    ci_refs = []
    if state.get("tc_ctx"):
        ci_refs.append(state["tc_ctx"].context_id)
    ci_refs.extend(t.context_id for t in state.get("tool_ctxs", []))

    resp_ctx = ContextObject.make_stage(
        stage_name="final_inference",
        input_refs=ci_refs,
        output={"text": reply},
    )
    resp_ctx.stage_id = "final_inference"
    resp_ctx.summary  = reply
    resp_ctx.touch()
    self.repo.save(resp_ctx)

    # ─── 6) Also record this reply as assistant segment ───────────────────
    seg = ContextObject.make_segment(
        semantic_label="assistant",
        content_refs=[resp_ctx.context_id],
        tags=["assistant"]
    )
    seg.summary  = reply
    seg.stage_id = "assistant"
    seg.touch()
    self.repo.save(seg)

    return reply

def _stage10b_response_critique_and_safety(
    self,
    draft: str,
    user_text: str,
    tool_ctxs: List[ContextObject],
    state: Dict[str, Any]
) -> str:
    import json, difflib, datetime

    # Recover the plan
    plan_text = getattr(self, "_last_plan_output", "(no plan)")

    # 1) Collect all tool outputs
    outputs = []
    for c in tool_ctxs:
        out = c.metadata.get("output")
        try:
            blob = json.dumps(out, indent=2, ensure_ascii=False)
        except:
            blob = repr(out)
        outputs.append(f"[{c.stage_id}]\n{blob}")

    # 2) Build richer critic prompt
    critic_sys = self._get_prompt("critic_prompt")
    prompt_user = "\n\n".join([
        f"User asked:\n{user_text}",
        f"Plan executed:\n{plan_text}",
        f"Your draft:\n{draft}",
        "Here are the raw tool results:\n" + "\n\n".join(outputs),
    ])

    self._print_stage_context("response_critique", {
        "system_directives": [critic_sys],
        "user_payload":      [prompt_user],
    })

    # 3) Invoke LLM
    msgs = [
        {"role": "system", "content": critic_sys},
        {"role": "user",   "content": prompt_user},
    ]
    polished = self._stream_and_capture(
        self.secondary_model,
        msgs,
        tag="[Critic]",
        images=state.get("images")
    ).strip()
    
    # 4) If unchanged, just return it
    if polished == draft.strip():
        return polished

    # 5) Compute diff summary
    diff = difflib.unified_diff(
        draft.splitlines(), polished.splitlines(),
        lineterm="", n=1
    )
    diff_summary = "; ".join(
        ln for ln in diff if ln.startswith(("+ ", "- "))
    ) or "(minor re-formatting)"

    # 6) Upsert dynamic_prompt_patch (as before)
    patch_rows = self.repo.query(
        lambda c: c.component == "policy" and c.semantic_label == "dynamic_prompt_patch"
    )
    patch_rows.sort(key=lambda c: c.timestamp, reverse=True)
    if patch_rows:
        patch = patch_rows[0]
        for extra in patch_rows[1:]:
            self.repo.delete(extra.context_id)
    else:
        patch = ContextObject.make_policy(
            "dynamic_prompt_patch", "", tags=["dynamic_prompt"]
        )
    if patch.summary != diff_summary:
        patch.summary = diff_summary
        patch.metadata["policy"] = diff_summary
        patch.touch(); self.repo.save(patch)
        self._print_stage_context("dynamic_patch_written", {"patch": diff_summary})

    # 7) Persist the polished reply
    resp_ctx = ContextObject.make_stage(
        "response_critique",
        [],  # no extra refs
        {"text": polished}
    )
    resp_ctx.stage_id = "response_critique"
    resp_ctx.summary  = polished
    resp_ctx.touch(); self.repo.save(resp_ctx)

    # 8) Create a standalone plan_critique that will feed into the next planner run
    critique_ctx = ContextObject.make_stage(
        "plan_critique",
        [resp_ctx.context_id] + [c.context_id for c in tool_ctxs],
        {"critique": polished, "diff": diff_summary}
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

    # gather all the IDs of contexts created/used this turn
    used_ids = []
    for key in ("user_ctx","sys_ctx","clar_ctx","know_ctx","plan_ctx","tc_ctx"):
        if key in state:
            used_ids.append(state[key].context_id)
    used_ids += [c.context_id for c in state.get("tool_ctxs",[])]
    used_ids = list(dict.fromkeys(used_ids))  # dedupe

    # assemble human narrative
    lines = [
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}:",
        f"• User asked: {state['user_text']!r}",
        f"• Clarified into: {state['clar_ctx'].summary!r}",
    ]
    if "plan_output" in state:
        lines.append(f"• Planner proposed: {state['plan_output']}")
    if "final" in state:
        lines.append(f"• Assistant replied: {state['final']!r}")

    narrative_text = "\n".join(lines)

    # either create or update the singleton narrative_context
    nc = self._get_or_make_singleton(
        label="narrative_context",
        component="stage",
        tags=["narrative"]
    )
    nc.metadata.setdefault("history_ids", []).extend(used_ids)
    nc.metadata["history_text"] = (nc.metadata.get("history_text","") + "\n\n" + narrative_text).strip()
    nc.summary = nc.metadata["history_text"]
    nc.references = nc.metadata["history_ids"]
    nc.touch()
    self.repo.save(nc)
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
