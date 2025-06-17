#!/usr/bin/env python3
"""
assembler.py — Stage-driven pipeline with full observability and
dynamic, chronological context windows per stage.
"""

import json
import logging
import re
from datetime import timedelta
from typing import List, Optional, Tuple, Callable, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat

from context import ContextObject, ContextRepository, MemoryManager, default_clock
from tools import TOOL_SCHEMAS, Tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] assembler: %(message)s",
)
logger = logging.getLogger("assembler")


class ContextQueryEngine:
    """
    Retrieval with time, tags, domain/component filters,
    regex & embedding similarity.
    Records recalls per stage.
    """
    def __init__(self, repo: ContextRepository, embedder: Callable[[str], np.ndarray]):
        self.repo = repo
        self.embedder = embedder
        self._cache: Dict[str, np.ndarray] = {}

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.embedder(text)
        return self._cache[text]

    def query(
        self,
        *,
        stage_id: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None,
        tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        domain: Optional[List[str]] = None,
        component: Optional[List[str]] = None,
        similarity_to: Optional[str] = None,
        summary_regex: Optional[str] = None,
        top_k: int = 5
    ) -> List[ContextObject]:
        # fetch all
        ctxs = self.repo.query(lambda c: True)

        # time window
        if time_range:
            start, end = time_range
            ctxs = [c for c in ctxs if start <= c.timestamp <= end]

        # tags include/exclude
        if tags:
            ctxs = [c for c in ctxs if set(tags) & set(c.tags)]
        if exclude_tags:
            ctxs = [c for c in ctxs if not (set(exclude_tags) & set(c.tags))]

        # domain/component
        if domain:
            ctxs = [c for c in ctxs if c.domain in domain]
        if component:
            ctxs = [c for c in ctxs if c.component in component]

        # regex on summary
        if summary_regex:
            pat = re.compile(summary_regex, re.I)
            ctxs = [c for c in ctxs if c.summary and pat.search(c.summary)]

        # similarity ranking
        if similarity_to:
            qv = self._vec(similarity_to)
            scored = []
            for c in ctxs:
                if not c.summary:
                    continue
                vv = self._vec(c.summary)
                sim = float(np.dot(qv, vv) /
                            (np.linalg.norm(qv) * np.linalg.norm(vv)))
                scored.append((c, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            ctxs = [c for c, _ in scored]

        out = ctxs[:top_k]
        for c in out:
            c.record_recall(stage_id=stage_id, coactivated_with=[])
            self.repo.save(c)
        return out


class Assembler:
    STAGES = [
        "recent_retrieval",
        "intent_clarification",
        "external_knowledge_retrieval",
        "planning_summary",
        "tool_chaining",
        "assemble_prompt",
        "final_inference",
    ]

    def __init__(
        self,
        context_path: str = "context.jsonl",
        config_path: str = "config.json",
        lookback_minutes: int = 60,
        top_k: int = 5
    ):
        # load or init config
        try:
            self.cfg = json.load(open(config_path))
        except FileNotFoundError:
            self.cfg = {}
        self.primary_model   = self.cfg.get("primary_model",   "gemma3:4b")
        self.secondary_model = self.cfg.get("secondary_model", self.primary_model)
        self.lookback        = self.cfg.get("lookback_minutes", lookback_minutes)
        self.top_k           = self.cfg.get("top_k",            top_k)
        self.hist_k          = self.cfg.get("history_turns",    5)

        self.clarifier_prompt = self.cfg.get(
            "clarifier_prompt",
            "You are Clarifier. Expand the user’s intent into a JSON object with "
            "two keys: 'keywords' (an array of concise keywords) and 'notes' (a brief explanation). "
            "Output only valid JSON."
        )
        self.assembler_prompt = self.cfg.get(
            "assembler_prompt",
            "You are Assembler. Distill context into a concise summary."
        )
        self.inference_prompt = self.cfg.get(
            "inference_prompt",
            "You are a helpful, context-aware assistant. Use all provided snippets and tool outputs."
        )

        # persist defaults
        defaults = {
            "primary_model":    self.primary_model,
            "secondary_model":  self.secondary_model,
            "lookback_minutes": self.lookback,
            "top_k":            self.top_k,
            "history_turns":    self.hist_k
        }
        if defaults != self.cfg:
            json.dump(defaults, open(config_path, "w"), indent=2)

        # init context store + memory
        self.repo   = ContextRepository(context_path)
        self.memman = MemoryManager(self.repo)

        # embedding engine
        stm = SentenceTransformer("all-MiniLM-L6-v2")
        self.engine = ContextQueryEngine(
            self.repo,
            lambda t: stm.encode(t, normalize_embeddings=True)
        )

        # seed missing schemas only once
        existing = {
            c.semantic_label
            for c in self.repo.query(
                lambda x: x.component=="schema" and "tool_schema" in x.tags
            )
        }
        for name,schema in TOOL_SCHEMAS.items():
            if name in existing: continue
            obj = ContextObject.make_schema(
                label=name,
                schema_def=json.dumps(schema),
                tags=["artifact","tool_schema"]
            )
            obj.touch(); self.repo.save(obj)

        # seed system prompts
        for nm,txt in [
            ("clarifier_prompt", self.clarifier_prompt),
            ("assembler_prompt", self.assembler_prompt),
            ("inference_prompt", self.inference_prompt),
        ]:
            p = ContextObject.make_prompt(label=nm, prompt_text=txt,
                                          tags=["artifact","prompt"])
            p.touch(); self.repo.save(p)

        logger.info("Assembler initialized.")

    def _load_system_prompts(self) -> ContextObject:
        prompts = self.repo.query(
            lambda c: c.domain=="artifact" and c.component in ("prompt","policy")
        )[:self.top_k]
        block = "\n".join(
            f"{c.semantic_label}: {c.metadata.get('prompt') or c.metadata.get('policy')}"
            for c in prompts
        ) or "(none)"
        ctx = ContextObject.make_stage("system_prompts",
                                       [c.context_id for c in prompts],
                                       {"prompts": block})
        ctx.stage_id="system_prompts"; ctx.summary=block
        ctx.touch(); self.repo.save(ctx)
        return ctx

    def _get_history(self) -> List[ContextObject]:
        segs = self.repo.query(lambda c: c.domain=="segment"
                               and c.component in ("user_input","assistant"))
        segs.sort(key=lambda c: c.timestamp)
        return segs[-self.hist_k:]

    def _print_stage_context(self, name:str, sections:Dict[str,List[Any]]):
        print(f"\n>>> [Stage: {name}] Context window:")
        for title,lines in sections.items():
            print(f"  -- {title}:")
            for ln in lines:
                print(f"     {ln}")

    def _save_stage(self, ctx: ContextObject, stage: str):
        ctx.stage_id = stage
        ctx.summary = (ctx.references and
                       (ctx.metadata.get("plan") or ctx.metadata.get("tool_call"))) or ctx.summary
        ctx.touch()
        self.repo.save(ctx)

    # helpers to keep main loop DRY
    def _stream_and_capture(self, model, messages, tag=""):
        out = ""
        print(f"{tag} ", end="", flush=True)
        for part in chat(model=model, messages=messages, stream=True):
            chunk = part["message"]["content"]
            print(chunk, end="", flush=True)
            out += chunk
        print()
        return out

    def run_with_meta_context(self, user_text: str) -> str:
        # ── Stage 1: record user input
        user_ctx = ContextObject.make_segment("user_input", [], tags=["user_input"])
        user_ctx.summary, user_ctx.stage_id = user_text, "user_input"
        user_ctx.touch(); self.repo.save(user_ctx)

        # ── Stage 2: load system prompts
        sys_ctx = self._load_system_prompts()

        # ── Stage 3: retrieve relevant past context
        now, past = default_clock(), default_clock() - timedelta(minutes=self.lookback)
        tr = (past.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%dT%H%M%SZ"))
        history = self._get_history()
        recent = self.engine.query(
            stage_id="recent_retrieval",
            time_range=tr,
            similarity_to=user_text,
            exclude_tags=self.STAGES + ["tool_schema","tool_output","assistant","system_prompt"],
            top_k=self.top_k
        )
        assoc = self.memman.recall([user_ctx.context_id], k=self.top_k)
        for c in assoc:
            c.record_recall(stage_id="recent_retrieval", coactivated_with=[user_ctx.context_id])
            self.repo.save(c)

        # merge without duplicates
        seen = {c.context_id for c in history}
        merged = history + [c for c in recent if c.context_id not in seen]
        seen |= {c.context_id for c in merged}
        merged += [c for c in assoc if c.context_id not in seen]
        recent_ids = [c.context_id for c in merged]

        self._print_stage_context("recent_retrieval", {
            "system_prompts": [sys_ctx.summary],
            "history":        [f"{c.semantic_label}: {c.summary}" for c in history],
            "retrieved":      [f"{c.semantic_label}: {c.summary}" for c in recent],
            "associative":    [f"{c.semantic_label}: {c.summary}" for c in assoc],
        })
        stage_refs = {"recent_retrieval": recent_ids}

        # ── Stage 4: intent clarification
        clar_block = "\n".join(f"[{c.semantic_label}] {c.summary}" for c in [sys_ctx] + merged)
        clar_msgs = [
            {"role":"system", "content": self.clarifier_prompt},
            {"role":"system", "content": f"Context:\n{clar_block}"},
            {"role":"user",   "content": user_text},
        ]
        self._print_stage_context("intent_clarification", {
            "system":  [self.clarifier_prompt],
            "context": clar_block.split("\n"),
            "user":    [user_text],
        })
        clar_output = self._stream_and_capture(self.secondary_model, clar_msgs, tag="[Clarifier]")
        try:
            clar_json = json.loads(clar_output)
        except:
            clar_json = {"keywords": [], "notes": clar_output}

        clar_ctx = ContextObject.make_stage(
            "intent_clarification",
            [user_ctx.context_id, sys_ctx.context_id] + recent_ids,
            clar_json
        )
        clar_ctx.stage_id = "intent_clarification"
        clar_ctx.summary = clar_json.get("notes", clar_output)
        clar_ctx.touch(); self.repo.save(clar_ctx)
        stage_refs["intent_clarification"] = clar_ctx.context_id

        # ── Stage 5: external knowledge retrieval
        snippets = []
        for kw in clar_json.get("keywords", []):
            hits = self.engine.query(
                stage_id="external_knowledge_retrieval",
                similarity_to=kw,
                top_k=self.top_k
            )
            snippets += [h.summary for h in hits]
        know_ctx = ContextObject.make_stage(
            "external_knowledge_retrieval",
            clar_ctx.references,
            {"snippets": snippets}
        )
        know_ctx.stage_id = "external_knowledge_retrieval"
        know_ctx.summary = "\n".join(snippets) if snippets else "(none)"
        know_ctx.touch(); self.repo.save(know_ctx)
        self._print_stage_context("external_knowledge_retrieval", {"snippets": [know_ctx.summary]})
        stage_refs["external_knowledge_retrieval"] = know_ctx.context_id
        # ── Stage 6: planning summary (with unpacked tools list) ─────────
        # 1) pull out all tool_schema contexts
        schema_ctxs = self.repo.query(
            lambda c: c.component=="schema" and "tool_schema" in c.tags
        )
        tools_list = []
        for c in schema_ctxs:
            data = json.loads(c.metadata["schema"])
            tools_list.append({
                "name":        data["name"],
                # only first line for planning overview
                "description": data.get("description","").split("\n",1)[0]
            })

        # 2) build a plain-text bullet list for planning
        tools_text = "\n".join(f"- **{t['name']}**: {t['description']}"
                               for t in tools_list)

        planning_system = (
            "Available tools:\n"
            f"{tools_text}\n\n"
            "Devise a concise plan. If you intend to call one or more tools, "
            "include exactly `tool_name(arg1=..., arg2=...)` in your plan for each, "
            "choosing only from the above list."
        )
        plan_input = clar_ctx.summary + "\n\nSnippets:\n" + know_ctx.summary
        self._print_stage_context("planning_summary", {
            "tools": [tools_text],
            "input": [plan_input[:200]],
        })
        plan_msgs = [
            {"role":"system","content":planning_system},
            {"role":"user",  "content":plan_input},
        ]
        plan_output = self._stream_and_capture(
            self.secondary_model, plan_msgs, tag="[Planner]"
        )

        plan_ctx = ContextObject.make_stage(
            "planning_summary",
            know_ctx.references,
            {"plan": plan_output}
        )
        plan_ctx.stage_id = "planning_summary"
        plan_ctx.summary  = plan_output
        plan_ctx.touch(); self.repo.save(plan_ctx)
        stage_refs["planning_summary"] = plan_ctx.context_id

        # ── Stage 7: initial tool chaining (multi-call JSON) ─────────────
        tc_system = (
            "Available tools:\n" f"{tools_text}\n\n"
            "Decide which tools to call (in order). **Reply only** with one‐line JSON:\n"
            '  {"tool_calls": ["tool1(arg1=...,arg2=...)", "tool2(...)"]}\n'
            "or\n"
            '  {"tool_calls": []}'
        )
        self._print_stage_context("tool_chaining", {
            "tools": [tools_text],
            "plan":  [plan_output[:200]],
        })
        tc_msgs = [
            {"role":"system","content":tc_system},
            {"role":"user",  "content":plan_output},
        ]
        chain_out = self._stream_and_capture(
            self.secondary_model, tc_msgs, tag="[ToolChain]"
        )

        # parse the raw tool calls list
        try:
            tc_json   = json.loads(chain_out.strip())
            raw_calls = tc_json.get("tool_calls", [])
        except json.JSONDecodeError:
            parsed   = Tools.parse_tool_call(chain_out)
            if isinstance(parsed, list):
                raw_calls = parsed
            elif parsed:
                raw_calls = [parsed]
            else:
                raw_calls = []

        # save chaining stage
        tc_ctx = ContextObject.make_stage(
            "tool_chaining",
            plan_ctx.references,
            {"tool_calls": raw_calls}
        )
        tc_ctx.stage_id = "tool_chaining"
        tc_ctx.summary  = json.dumps(raw_calls)
        tc_ctx.touch(); self.repo.save(tc_ctx)
        stage_refs["tool_chaining"] = tc_ctx.context_id

        # ── Stage 7b: enrich chaining with full docstrings ─────────────
        if raw_calls:
            # select only the schemas for the calls
            sel_names = {call.split("(")[0] for call in raw_calls}
            selected = [
                json.loads(c.metadata["schema"])
                for c in schema_ctxs
                if json.loads(c.metadata["schema"])["name"] in sel_names
            ]
            # build full descriptions
            details = "\n\n".join(
                f"**{s['name']}**\n{s.get('description','(no docs)')}"
                for s in selected
            )
            tc_system_full = (
                "You have chosen these tools and their full docs:\n\n"
                f"{details}\n\n"
                "Now confirm your calls or adjust them. **Reply only** with JSON:\n"
                '{"tool_calls": ["tool1(arg1=...,arg2=...)", ...]}'
            )
            self._print_stage_context("tool_chaining_details", {
                "selected_tools": details.split("\n"),
            })
            tc_msgs2 = [
                {"role":"system","content":tc_system_full},
                {"role":"user",  "content": json.dumps({"tool_calls": raw_calls})},
            ]
            chain_out = self._stream_and_capture(
                self.secondary_model, tc_msgs2, tag="[ToolChainDetails]"
            )
            # re-parse in case of adjustment
            try:
                tc_json2  = json.loads(chain_out.strip())
                raw_calls = tc_json2.get("tool_calls", raw_calls)
            except:
                pass

        # ── Stage 8: invoke each tool in series, capture results/errors ──
        tool_ctxs, errors = [], []
        for idx, call_str in enumerate(raw_calls):
            print(f"[ToolInvocation] Running: {call_str}", flush=True)
            try:
                result = Tools.run_tool_once(call_str)
                exc    = None
            except Exception as e:
                result = None
                exc    = str(e)
            print(f"[ToolInvocation] Result: {{'output': {result!r}, 'exception': {exc!r}}}",
                  flush=True)

            # find the schema context for this tool
            name = call_str.split("(")[0]
            sch_ctx = next(
                c for c in schema_ctxs
                if json.loads(c.metadata["schema"])["name"] == name
            )

            out_ctx = ContextObject.make_stage(
                "tool_output",
                [sch_ctx.context_id],
                {"call": call_str, "output": result, "exception": exc}
            )
            out_ctx.stage_id = f"tool_output_{idx}"
            out_ctx.summary = (
                result if isinstance(result, str)
                else json.dumps(result) if result is not None
                else f"ERROR: {exc}"
            )
            out_ctx.touch(); self.repo.save(out_ctx)
            tool_ctxs.append(out_ctx)

            if exc:
                errors.append(f"{call_str} → {exc}")

        stage_refs["tool_invocation"] = [c.context_id for c in tool_ctxs]

        # ── Stage 9: retry chaining if any errors ─────────────────────────
        if errors:
            err_block = "\n".join(errors)
            retry_system = (
                "The previous tool invocations failed:\n"
                f"{err_block}\n\n"
                "Please correct your tool_calls. **Reply only** with JSON:\n"
                '  {"tool_calls": ["tool1(...)", ...]}'
            )
            self._print_stage_context("tool_chaining_retry", {"errors": errors})
            retry_msgs = [
                {"role": "system", "content": retry_system},
                {"role": "user",   "content": plan_output},
            ]
            retry_out = self._stream_and_capture(self.secondary_model, retry_msgs, tag="[ToolChainRetry]")

            try:
                retry_json = json.loads(retry_out.strip())
                raw_calls  = retry_json.get("tool_calls", [])
            except:
                parsed = Tools.parse_tool_call(retry_out)
                raw_calls = parsed if isinstance(parsed, list) else ([parsed] if parsed else [])

            # invoke retry calls same as above
            for idx, call_str in enumerate(raw_calls):
                print(f"[ToolInvocationRetry] Running: {call_str}", flush=True)
                try:
                    result = Tools.run_tool_once(call_str)
                    exc = None
                except Exception as e:
                    result = None
                    exc = str(e)
                print(f"[ToolInvocationRetry] Result: {{'output': {result!r}, 'exception': {exc!r}}}", flush=True)

                name = call_str.split("(")[0]
                sch_ctx = next(
                    c for c in schema_ctxs
                    if json.loads(c.metadata["schema"])["name"] == name
                )
                out_ctx = ContextObject.make_stage(
                    "tool_output",
                    [sch_ctx.context_id],
                    {"call": call_str, "output": result, "exception": exc}
                )
                out_ctx.stage_id = f"tool_output_retry_{idx}"
                out_ctx.summary = (
                    result if isinstance(result, str)
                    else json.dumps(result) if result is not None
                    else f"ERROR: {exc}"
                )
                out_ctx.touch(); self.repo.save(out_ctx)
                tool_ctxs.append(out_ctx)

            stage_refs["tool_invocation"] = [c.context_id for c in tool_ctxs]
        # ── Stage 7: initial tool chaining (multi-call JSON) ────────────
        tc_system = (
            "Available tools (name + one-line desc):\n"
            f"{tools_text}\n\n"
            "Decide which tools to call (in order). **Reply only** with JSON:\n"
            '  {"tool_calls": ["tool1(arg1=...,arg2=...)", "tool2(...)"]}\n'
            "or\n"
            '  {"tool_calls": []}'
        )
        self._print_stage_context("tool_chaining", {
            "tools": [tools_text],
            "plan":  [plan_output[:200]],
        })
        chain_out = self._stream_and_capture(
            self.secondary_model,
            [{"role":"system","content":tc_system},
             {"role":"user","content":plan_output}],
            tag="[ToolChain]"
        )

        try:
            tc_json   = json.loads(chain_out.strip())
            raw_calls = tc_json.get("tool_calls", [])
        except:
            parsed    = Tools.parse_tool_call(chain_out)
            raw_calls = parsed if isinstance(parsed, list) else ([parsed] if parsed else [])

        tc_ctx = ContextObject.make_stage(
            "tool_chaining", plan_ctx.references,
            {"tool_calls": raw_calls}
        )
        tc_ctx.stage_id = "tool_chaining"
        tc_ctx.summary  = json.dumps(raw_calls)
        tc_ctx.touch(); self.repo.save(tc_ctx)
        stage_refs["tool_chaining"] = tc_ctx.context_id

        # ── Stage 7b: enrich with full docstrings ───────────────────────
        if raw_calls:
            sel   = {c.split("(")[0] for c in raw_calls}
            docs  = []
            for c in schema_ctxs:
                schema = json.loads(c.metadata["schema"])
                if schema["name"] in sel:
                    docs.append(f"**{schema['name']}**\n{schema.get('description','(no docs)')}")
            full_doc = "\n\n".join(docs)
            tc_system2 = (
                "You have chosen these tools with full docs:\n\n"
                f"{full_doc}\n\n"
                "Now confirm or adjust your calls. **Reply only** with JSON:\n"
                '{"tool_calls":["tool1(arg1=...,arg2=...)", ...]}'
            )
            self._print_stage_context("tool_chaining_details", {
                "selected_tools": full_doc.split("\n")
            })
            chain_out2 = self._stream_and_capture(
                self.secondary_model,
                [{"role":"system","content":tc_system2},
                 {"role":"user","content":json.dumps({"tool_calls":raw_calls})}],
                tag="[ToolChainDetails]"
            )
            try:
                tc_json2  = json.loads(chain_out2.strip())
                raw_calls = tc_json2.get("tool_calls", raw_calls)
            except:
                pass

        # ── Stage 8: invoke each tool in series, capture results/errors ──
        tool_ctxs, results, errors = [], [], []
        for idx, call_str in enumerate(raw_calls):
            print(f"[ToolInvocation] Running: {call_str}", flush=True)
            try:
                out = Tools.run_tool_once(call_str)
                exc = None
            except Exception as e:
                out, exc = None, str(e)
            print(f"[ToolInvocation] Result: {{'output':{out!r},'exception':{exc!r}}}", flush=True)

            # record
            name = call_str.split("(")[0]
            sch  = next(c for c in schema_ctxs
                        if json.loads(c.metadata["schema"])["name"]==name)
            ctx  = ContextObject.make_stage(
                "tool_output",[sch.context_id],
                {"call":call_str,"output":out,"exception":exc}
            )
            ctx.stage_id = f"tool_output_{idx}"
            ctx.summary  = (
                out if isinstance(out,str)
                else json.dumps(out) if out is not None
                else f"ERROR: {exc}"
            )
            ctx.touch(); self.repo.save(ctx)
            tool_ctxs.append(ctx)

            results.append((call_str,out))
            if exc or "malformed node" in (exc or ""):
                errors.append(f"{call_str} → {exc}")

        stage_refs["tool_invocation"] = [c.context_id for c in tool_ctxs]

        # ── Stage 9: retry if any parsing/AST errors ────────────────────
        if errors:
            # Build a rich retry prompt that includes:
            #  - the error lines
            #  - the raw successful outputs (so model can interpolate)
            err_block = "\n".join(errors)
            res_block = "\n".join(f"{c}: {o}" for c,o in results if o is not None)
            retry_sys = (
                "Some calls failed or had parse errors:\n"
                f"{err_block}\n\n"
                "Successful outputs so far:\n"
                f"{res_block}\n\n"
                "Please correct your tool_calls (e.g. interpolate the location value). **Reply only** with JSON:\n"
                '{"tool_calls":["tool1(arg1=...,arg2=...)", ...]}'
            )
            self._print_stage_context("tool_chaining_retry", {
                "errors": errors,
                "results": results
            })
            retry_out = self._stream_and_capture(
                self.secondary_model,
                [{"role":"system","content":retry_sys},
                 {"role":"user","content":json.dumps({"tool_calls":raw_calls})}],
                tag="[ToolChainRetry]"
            )
            try:
                rj        = json.loads(retry_out.strip())
                raw_calls = rj.get("tool_calls",[])
            except:
                parsed   = Tools.parse_tool_call(retry_out)
                raw_calls = parsed if isinstance(parsed,list) else ([parsed] if parsed else [])

            retry_ctx = ContextObject.make_stage(
                "tool_chaining_retry", plan_ctx.references,
                {"tool_calls":raw_calls}
            )
            retry_ctx.stage_id, retry_ctx.summary = "tool_chaining_retry", json.dumps(raw_calls)
            retry_ctx.touch(); self.repo.save(retry_ctx)
            stage_refs["tool_chaining_retry"] = retry_ctx.context_id

            # Invoke retry calls once more
            retry_results, retry_errors = [], []
            for idx, call_str in enumerate(raw_calls):
                print(f"[Retry_ToolInvocation] Running: {call_str}", flush=True)
                try:
                    out = Tools.run_tool_once(call_str)
                    exc = None
                except Exception as e:
                    out, exc = None, str(e)
                print(f"[Retry_ToolInvocation] Result: {{'output':{out!r},'exception':{exc!r}}}", flush=True)

                name = call_str.split("(")[0]
                sch  = next(c for c in schema_ctxs
                            if json.loads(c.metadata["schema"])["name"]==name)
                ctx  = ContextObject.make_stage(
                    "tool_output",[sch.context_id],
                    {"call":call_str,"output":out,"exception":exc}
                )
                ctx.stage_id = f"tool_output_retry_{idx}"
                ctx.summary  = (
                    out if isinstance(out,str)
                    else json.dumps(out) if out is not None
                    else f"ERROR: {exc}"
                )
                ctx.touch(); self.repo.save(ctx)
                tool_ctxs.append(ctx)

                retry_results.append((call_str,out))
                if exc:
                    retry_errors.append(f"{call_str} → {exc}")

            # if still errors, we give up retrying
            if retry_errors:
                self._print_stage_context("tool_chaining_failed", {
                    "errors": retry_errors
                })

            stage_refs["tool_invocation"] = [c.context_id for c in tool_ctxs]

        # ── Stage 10: assemble intermediate context ─────────────────────
        refs = [user_ctx.context_id, sys_ctx.context_id] + recent_ids
        for v in stage_refs.values():
            refs += v if isinstance(v,list) else [v]
        refs += [t.context_id for t in tool_ctxs]
        seen,all_refs = set(),[]
        for r in refs:
            if r not in seen:
                seen.add(r); all_refs.append(r)
        ctxs = [self.repo.get(cid) for cid in all_refs]
        ctxs.sort(key=lambda c:c.timestamp)
        interm = "\n".join(f"[{c.semantic_label}] {c.summary}" for c in ctxs)
        # ── Stage 11: final inference ───────────────────────────────────
        final_system = (
            "You are a helpful assistant. Use ONLY the provided context below—"
            "including the user’s original input, the plan, the exact tool calls made, "
            "and each tool’s outputs—to answer the user’s query. "
            "Do NOT invent new actions or call any further tools. "
            "If the context fully answers the question, respond directly. "
            "If not, admit that you cannot answer without more information."
        )
        # ── Stage 11: final inference ───────────────────────────────────
        final_msgs = [
            {"role": "system",  "content": final_system},
            {"role":"system","content":self.inference_prompt},
            {"role":"system","content":interm},
            {"role":"user",  "content":user_text},
        ]
        reply = self._stream_and_capture(self.primary_model, final_msgs, tag="[Assistant]")

        resp_ctx = ContextObject.make_stage(
            "final_inference", [tc_ctx.context_id], {"text":reply}
        )
        resp_ctx.stage_id, resp_ctx.summary = "final_inference", reply
        resp_ctx.touch(); self.repo.save(resp_ctx)

        return reply




if __name__ == "__main__":
    asm = Assembler()
    print("Assembler ready. Type your message, Ctrl-C to quit.")
    try:
        while True:
            msg = input(">> ").strip()
            if msg:
                print(asm.run_with_meta_context(msg))
    except KeyboardInterrupt:
        print("\nGoodbye.")
