#!/usr/bin/env python3
"""
assembler.py — Stage-driven pipeline with full observability and
dynamic, chronological context windows per stage.
"""

import json
import logging
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable

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
        context_path:     str = "context.jsonl",
        config_path:      str = "config.json",
        lookback_minutes: int = 60,
        top_k:            int = 5,
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

        # system‐and‐stage prompts
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

        # seed missing schemas
        existing = {
            c.semantic_label
            for c in self.repo.query(
                lambda x: x.component=="schema" and "tool_schema" in x.tags
            )
        }
        for name, schema in TOOL_SCHEMAS.items():
            if name in existing: continue
            obj = ContextObject.make_schema(
                label=name,
                schema_def=json.dumps(schema),
                tags=["artifact","tool_schema"]
            )
            obj.touch()
            self.repo.save(obj)

        # seed system prompts
        for nm, txt in [
            ("clarifier_prompt", self.clarifier_prompt),
            ("assembler_prompt", self.assembler_prompt),
            ("inference_prompt", self.inference_prompt),
        ]:
            p = ContextObject.make_prompt(label=nm, prompt_text=txt,
                                          tags=["artifact","prompt"])
            p.touch()
            self.repo.save(p)

        logger.info("Assembler initialized.")

    def _ensure_str(self, x: Any) -> str:
        """
        Coerce non-string into JSON/text so strip()/json.loads() never fails.
        """
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x)
        except:
            return str(x)

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
        ctx.stage_id = "system_prompts"
        ctx.summary  = block
        ctx.touch()
        self.repo.save(ctx)
        return ctx

    def _get_history(self) -> List[ContextObject]:
        segs = self.repo.query(lambda c: c.domain=="segment"
                               and c.component in ("user_input","assistant"))
        segs.sort(key=lambda c: c.timestamp)
        return segs[-self.hist_k:]

    def _print_stage_context(self, name: str, sections: Dict[str, Any]):
        print(f"\n>>> [Stage: {name}] Context window:")
        for title, lines in sections.items():
            print(f"  -- {title}:")
            # if it's a raw string, split on real newlines only
            if isinstance(lines, str):
                for ln in lines.splitlines():
                    print(f"     {ln}")
            # if it's a list of items, print each
            elif isinstance(lines, list):
                for ln in lines:
                    print(f"     {ln}")
            # fallback (e.g. single int or dict)
            else:
                print(f"     {lines}")

    def _save_stage(self, ctx: ContextObject, stage: str):
        ctx.stage_id = stage
        ctx.summary = (ctx.references and
                       (ctx.metadata.get("plan") or ctx.metadata.get("tool_call"))) or ctx.summary
        ctx.touch()
        self.repo.save(ctx)

    def _stream_and_capture(self, model: str, messages: List[Dict[str,Any]], tag: str="") -> str:
        out = ""
        print(f"{tag} ", end="", flush=True)
        for part in chat(model=model, messages=messages, stream=True):
            chunk = part["message"]["content"]
            print(chunk, end="", flush=True)
            out += chunk
        print()
        return out



    def run_with_meta_context(self, user_text: str) -> str:
        state: Dict[str, Any] = {}

        # Stage 1: record input
        state['user_ctx'] = self._stage1_record_input(user_text)

        # Stage 2: load system prompts
        state['sys_ctx']  = self._stage2_load_system_prompts()

        # Stage 3: retrieve & merge context
        ctx3 = self._stage3_retrieve_and_merge_context(
            user_text, state['user_ctx'], state['sys_ctx']
        )
        state.update(ctx3)

        # Stage 4: clarify intent
        state['clar_ctx'] = self._stage4_intent_clarification(user_text, state)

        # Stage 5: external knowledge
        state['know_ctx'] = self._stage5_external_knowledge(state['clar_ctx'])

        # Stage 6: prepare tools & plan
        tools_list = self._stage6_prepare_tools()
        state['plan_ctx'], state['plan_output'] = self._stage7_planning_summary(
            state['clar_ctx'], state['know_ctx'], tools_list
        )

        # Stage 7b: static plan validation *and* fix
        state['valid_calls'], state['plan_errors'], state['fixed_calls'] = self._stage7b_plan_validation(
            state['plan_ctx'],
            state['plan_output'],
            tools_list
        )
        if state['plan_errors']:
            # instead of bailing out, we could loop or escalate—but for now,
            # just surface the error
            raise RuntimeError(f"Plan validation failed: {state['plan_errors']}")

        # Build a single‐string “plan” from the fixed calls
        plan_cmds = "\n".join(state['fixed_calls'])

        # Stage 7: tool chaining (drive from fixed_calls, not original plan_output)
        state['tc_ctx'], raw_calls, selected_schemas = self._stage8_tool_chaining(
            state['plan_ctx'],
            plan_cmds,
            tools_list
        )

        # Stage 8.5: (optional) user confirmation
        state['confirmed_calls'] = self._stage8_5_user_confirmation(
            raw_calls, user_text
        )

        # Stage 8: invoke with retries (again, pass plan_cmds)
        state['tool_ctxs'] = self._stage9_invoke_with_retries(
            state['confirmed_calls'],
            plan_cmds,
            selected_schemas,
            user_text,
            state['clar_ctx'].metadata
        )

        # Stage 9b: reflection & possible replan (pass plan_cmds)
        replan = self._stage9b_reflection_and_replan(
            state['tool_ctxs'],
            plan_cmds,
            user_text,
            state['clar_ctx'].metadata
        )
        if replan:
            # one‐shot replan loop (you can repeat the above pattern)
            plan_cmds = replan
            state['valid_calls'], state['plan_errors'], state['fixed_calls'] = self._stage7b_plan_validation(
                state['plan_ctx'], plan_cmds, tools_list
            )
            state['tc_ctx'], raw_calls, selected_schemas = self._stage8_tool_chaining(
                state['plan_ctx'], plan_cmds, tools_list
            )
            state['confirmed_calls'] = self._stage8_5_user_confirmation(raw_calls, user_text)
            state['tool_ctxs'] = self._stage9_invoke_with_retries(
                state['confirmed_calls'],
                plan_cmds,
                selected_schemas,
                user_text,
                state['clar_ctx'].metadata
            )

        # Stage 10: assemble draft answer
        draft = self._stage10_assemble_and_infer(user_text, state)

        # Stage 10b: critique & safety polish
        final = self._stage10b_response_critique_and_safety(draft, user_text)

        # Stage 11: memory write-back
        self._stage11_memory_writeback(final, state['tool_ctxs'])

        return final




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
        self, user_text: str, user_ctx: ContextObject, sys_ctx: ContextObject
    ) -> Dict[str, Any]:
        # window for recent-retrieval
        now, past = default_clock(), default_clock() - timedelta(minutes=self.lookback)
        tr = (past.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%dT%H%M%SZ"))

        # 1) grab last 20 user+assistant turns
        all_segments = self.repo.query(
            lambda c: c.domain=="segment" and c.component in ("user_input","assistant")
        )
        all_segments.sort(key=lambda c: c.timestamp)
        history = all_segments[-40:]  # up to 20 user + 20 assistant

        # ensure the very last assistant reply is included
        last_final = [
            c for c in all_segments
            if c.stage_id=="final_inference"
        ]
        if last_final:
            history.append(last_final[-1])

        # 2) semantic recent retrieval
        recent = self.engine.query(
            stage_id="recent_retrieval",
            time_range=tr,
            similarity_to=user_text,
            exclude_tags=self.STAGES + ["tool_schema","tool_output","assistant","system_prompt"],
            top_k=self.top_k
        )

        # 3) associative memory
        assoc = self.memman.recall([user_ctx.context_id], k=self.top_k)
        for c in assoc:
            c.record_recall(stage_id="recent_retrieval", coactivated_with=[user_ctx.context_id])
            self.repo.save(c)

        # merge, preserving order: history → recent → assoc
        seen = {c.context_id for c in history}
        merged = list(history)
        for c in recent:
            if c.context_id not in seen:
                merged.append(c)
                seen.add(c.context_id)
        for c in assoc:
            if c.context_id not in seen:
                merged.append(c)

        recent_ids = [c.context_id for c in merged]

        # print for debugging
        self._print_stage_context("recent_retrieval", {
            "system_prompts": [sys_ctx.summary],
            "history":        [f"{c.semantic_label}: {c.summary}" for c in history],
            "retrieved":      [f"{c.semantic_label}: {c.summary}" for c in recent],
            "associative":    [f"{c.semantic_label}: {c.summary}" for c in assoc],
        })

        return {
            "history":    history,
            "recent":     recent,
            "assoc":      assoc,
            "recent_ids": recent_ids
        }

    def _stage4_intent_clarification(
        self, user_text: str, state: Dict[str, Any]
    ) -> ContextObject:
        # assemble full context block
        pieces = [state['sys_ctx']] + state['history'] + state['recent'] + state['assoc']
        block = "\n".join(f"[{c.semantic_label}] {c.summary}" for c in pieces)

        msgs = [
            {"role": "system",  "content": self.clarifier_prompt},
            {"role": "system",  "content": f"Context:\n{block}"},
            {"role": "user",    "content": user_text},
        ]
        self._print_stage_context("intent_clarification", {
            "system":  [self.clarifier_prompt],
            "context": block.split("\n"),
            "user":    [user_text],
        })
        out = self._stream_and_capture(self.secondary_model, msgs, tag="[Clarifier]")
        try:
            clar = json.loads(out)
        except:
            clar = {"keywords": [], "notes": out}

        ctx = ContextObject.make_stage(
            "intent_clarification",
            [state['user_ctx'].context_id, state['sys_ctx'].context_id] + state['recent_ids'],
            clar
        )
        ctx.stage_id = "intent_clarification"
        ctx.summary  = clar.get("notes", out)
        ctx.touch()
        self.repo.save(ctx)
        return ctx

    def _stage5_external_knowledge(self, clar_ctx: ContextObject) -> ContextObject:
        snippets: List[str] = []
        for kw in clar_ctx.metadata.get("keywords", []):
            hits = self.engine.query(
                stage_id="external_knowledge_retrieval",
                similarity_to=kw,
                top_k=self.top_k
            )
            snippets.extend(h.summary for h in hits)

        ctx = ContextObject.make_stage(
            "external_knowledge_retrieval",
            clar_ctx.references,
            {"snippets": snippets}
        )
        ctx.stage_id = "external_knowledge_retrieval"
        ctx.summary  = "\n".join(snippets) or "(none)"
        ctx.touch()
        self.repo.save(ctx)
        self._print_stage_context("external_knowledge_retrieval", {
            "snippets": snippets or ["(none)"]
        })
        return ctx

    def _stage6_prepare_tools(self) -> List[Dict[str, str]]:
        schema_ctxs = self.repo.query(
            lambda c: c.component=="schema" and "tool_schema" in c.tags
        )
        return [
            {
                "name": data["name"],
                "description": data.get("description","").split("\n",1)[0]
            }
            for c in schema_ctxs
            for data in [json.loads(c.metadata["schema"])]
        ]

    def _stage7_planning_summary(
        self, clar_ctx: ContextObject, know_ctx: ContextObject, tools_list: List[Dict[str, str]]
    ) -> Tuple[ContextObject, str]:
        import inspect

        def _can_be_called_with_no_args(tool_name: str) -> bool:
            fn = getattr(Tools, tool_name, None)
            if not callable(fn):
                return False
            sig = inspect.signature(fn)
            # if any positional-or-keyword param has no default, it needs args
            return all(
                param.default is not param.empty or
                param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                for param in sig.parameters.values()
            )

        # only keep zero-arg tools (fallback to first top_k if none)
        zero_arg = [t for t in tools_list if _can_be_called_with_no_args(t["name"])]
        if not zero_arg:
            zero_arg = tools_list[: self.top_k]
        tools_list = zero_arg

        # build prompt
        tools_text = "\n".join(f"- **{t['name']}**: {t['description']}" for t in tools_list)
        system = (
            "Available tools:\n" + tools_text +
            "\n\nDevise a concise plan. If you intend to call tools, "
            "include `tool_name(arg1=..., arg2=...)`."
        )
        inp = clar_ctx.summary + "\n\nSnippets:\n" + know_ctx.summary

        self._print_stage_context("planning_summary", {
            "tools": [tools_text],
            "input": [inp[:200]]
        })
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": inp}
        ]
        plan = self._stream_and_capture(self.secondary_model, msgs, tag="[Planner]")

        ctx = ContextObject.make_stage(
            "planning_summary", know_ctx.references, {"plan": plan}
        )
        ctx.stage_id = "planning_summary"
        ctx.summary  = plan
        ctx.touch()
        self.repo.save(ctx)
        return ctx, plan
    
    def _stage7b_plan_validation(
        self,
        plan_ctx: ContextObject,
        plan_output: str,
        tools_list: List[Dict[str, str]]
    ) -> Tuple[List[str], List[Tuple[str,str]], List[str]]:
        """
        1) Statically verify each intended call in plan_output against its schema.
        2) If any are missing required params, loop up to max_attempts:
           - In all but the last attempt, include only the schema docs.
           - On the last attempt, also fetch and include the actual tool source code
             to give the LLM maximum context for filling in missing args.
        3) If the model hallucinates a non‐existent tool, automatically remap it
           to the closest real tool name before validating.
        Returns (valid_calls, errors, fixed_calls)
        """
        import json, re, inspect, difflib

        # load all tool schemas
        all_schs = {
            json.loads(c.metadata["schema"])["name"]: json.loads(c.metadata["schema"])
            for c in self.repo.query(
                lambda c: c.component=="schema" and "tool_schema" in c.tags
            )
        }

        # extract the initial calls
        raw_calls = re.findall(r'\b[A-Za-z_]\w*\([^)]*\)', plan_output)
        fixed_calls = list(raw_calls)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            valid: List[str] = []
            errors: List[Tuple[str,str]] = []
            missing: Dict[str,List[str]] = {}

            # A) validate (and remap hallucinations) in one pass
            for idx, call in enumerate(list(fixed_calls)):
                name, args_str = call.split("(", 1)
                args_str = args_str.rstrip(")")
                # 1) if tool name doesn't exist, try to auto‐remap
                if name not in all_schs:
                    best = difflib.get_close_matches(name, all_schs.keys(), n=1)
                    if best:
                        real = best[0]
                        new_call = f"{real}({args_str})"
                        fixed_calls[idx] = new_call
                        name = real
                        sch  = all_schs[real]
                    else:
                        errors.append((call, "no schema found"))
                        continue
                else:
                    sch = all_schs[name]

                # 2) parse kwargs and find missing
                kwargs: Dict[str,str] = {}
                for part in filter(None, args_str.split(",")):
                    if "=" not in part: continue
                    k, v = part.split("=", 1)
                    kwargs[k.strip()] = v.strip()

                req = set(sch.get("parameters", {}).get("required", []))
                miss = sorted(req - set(kwargs.keys()))
                if miss:
                    errors.append((fixed_calls[idx], f"missing required: {miss}"))
                    missing[fixed_calls[idx]] = miss
                else:
                    valid.append(fixed_calls[idx])

            # B) if nothing is missing, we’re done
            if not missing:
                break

            # C) build docs for failures (schema, and on last attempt full source)
            docs_lines: List[str] = []
            for call, miss in missing.items():
                tool = call.split("(",1)[0]
                sch  = all_schs.get(tool, {})
                desc   = sch.get("description", "(no description)")
                params = sch.get("parameters", {})
                docs_lines.append(
                    f"• **{tool}**: {desc}\n"
                    f"  Parameters:\n{json.dumps(params, indent=2)}"
                )
                if attempt == max_attempts:
                    try:
                        src = inspect.getsource(getattr(Tools, tool))
                    except Exception:
                        src = "(source unavailable)"
                    docs_lines.append(f"```python\n{src}\n```")

            # print to your console for debugging
            self._print_stage_context("plan_validation_docs", {
                "tool_docs": docs_lines
            })

            docs_blob  = "\n\n".join(docs_lines)
            miss_lines = "\n".join(f"{c} → missing {m}" for c, m in missing.items())

            # D) assemble the fix prompt
            if attempt < max_attempts:
                prompt = (
                    "Some tool calls are missing required arguments:\n"
                    f"{miss_lines}\n\n"
                    "Here are the schemas for those tools:\n\n"
                    f"{docs_blob}\n\n"
                    "**Return only** JSON of the form:\n"
                    '{"fixed_calls": ["tool1(arg=...)", ...]}'
                )
            else:
                prompt = (
                    "Final attempt: some calls remain invalid:\n"
                    f"{miss_lines}\n\n"
                    "Below are both the schema definitions AND the full source code.\n"
                    f"{docs_blob}\n\n"
                    "**Return only** JSON:\n"
                    '{"fixed_calls": ["tool1(arg=...)", ...]}'
                )

            out = self._stream_and_capture(
                self.secondary_model,
                [{"role": "system", "content": prompt}],
                tag="[PlanFix]"
            ).strip()

            # parse LLM’s fixed_calls
            try:
                fixed_calls = json.loads(out)["fixed_calls"]
            except Exception:
                parsed = Tools.parse_tool_call(out)
                if isinstance(parsed, list):
                    fixed_calls = parsed
                elif parsed:
                    fixed_calls = [parsed]

        # E) persist final results
        meta = {"valid": valid, "errors": errors, "fixed_calls": fixed_calls}
        ctx = ContextObject.make_stage("plan_validation", plan_ctx.references, meta)
        ctx.stage_id = "plan_validation"
        ctx.summary  = f"Valid: {valid}" if not errors else f"Errors: {errors}"
        ctx.touch(); self.repo.save(ctx)
        self._print_stage_context("plan_validation", meta)

        return valid, errors, fixed_calls


    def _stage8_tool_chaining(
        self,
        plan_ctx: ContextObject,
        plan_output: str,
        tools_list: List[Dict[str, str]]
    ) -> Tuple[ContextObject, List[str], List[ContextObject]]:
        import json, re

        # ── A) Extract raw calls from the plan ──────────────────────────────
        text = plan_output
        parsed = Tools.parse_tool_call(text)
        if isinstance(parsed, list):
            raw = parsed
        elif parsed:
            raw = [parsed]
        else:
            # fallback to regex
            raw = re.findall(r'\b[A-Za-z_]\w*\([^)]*\)', text)

        # ── B) Dedupe & preserve order ─────────────────────────────────────
        seen = set()
        calls: List[str] = []
        for c in raw:
            if isinstance(c, str) and c not in seen:
                seen.add(c)
                calls.append(c)

        # ── C) Retrieve only the schemas that match these calls ────────────
        all_schemas = self.repo.query(
            lambda c: c.component=="schema" and "tool_schema" in c.tags
        )
        selected_schemas: List[ContextObject] = []
        full_schemas: List[Dict[str,Any]] = []
        for sch_obj in all_schemas:
            schema = json.loads(sch_obj.metadata["schema"])
            if schema["name"] in {call.split("(")[0] for call in calls}:
                selected_schemas.append(sch_obj)
                full_schemas.append(schema)

        # ── D) Build a docs_blob with the full JSON schema for each tool ───
        docs_blob_parts = []
        for schema in full_schemas:
            docs_blob_parts.append(
                f"**{schema['name']}**\n```json\n"
                + json.dumps(schema, indent=2)
                + "\n```"
            )
        docs_blob = "\n\n".join(docs_blob_parts)

        # ── E) Prompt the LLM, passing only these full schemas ─────────────
        system = (
            "You have the following tools (full JSON schemas shown).  "
            "Please confirm or adjust your tool calls.  Reply **ONLY** with JSON:\n"
            '{"tool_calls": ["tool1(arg1=...,arg2=...)", ...]}\n\n'
            f"{docs_blob}"
        )
        self._print_stage_context("tool_chaining", {
            "plan":   [plan_output[:200]],
            "schemas": docs_blob_parts
        })
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": json.dumps({"tool_calls": calls})},
        ]
        out = self._stream_and_capture(self.secondary_model, msgs, tag="[ToolChain]")

        # ── F) Parse the confirmed calls ───────────────────────────────────
        try:
            parsed2 = json.loads(out.strip())
            confirmed = parsed2.get("tool_calls", calls)
        except:
            parsed2 = Tools.parse_tool_call(out)
            confirmed = (
                parsed2 if isinstance(parsed2, list)
                else [parsed2] if parsed2
                else calls
            )

        # ── G) Persist the final chaining stage, including the docs ─────────
        tc_ctx = ContextObject.make_stage(
            "tool_chaining",
            plan_ctx.references + [s.context_id for s in selected_schemas],
            {
                "tool_calls": confirmed,
                "tool_docs": docs_blob
            }
        )
        tc_ctx.stage_id = "tool_chaining"
        tc_ctx.summary  = json.dumps(confirmed)
        tc_ctx.touch()
        self.repo.save(tc_ctx)

        return tc_ctx, confirmed, selected_schemas


    def _stage8_5_user_confirmation(
        self,
        calls: List[str],
        user_text: str
    ) -> List[str]:
        """
        (Optional) Surface the to-be-invoked calls for user approval.
        Here it auto-approves, but hook in your UI/CLI as needed.
        """
        self._print_stage_context("user_confirmation", {"calls": calls})
        confirmed = calls  # replace with real confirmation logic
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
        clar_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        After tool execution, ask: did outputs meet intent?
        If not, return a new plan; otherwise return None.
        """
        outputs = [f"[{c.stage_id}] {c.summary}" for c in tool_ctxs]
        prompt = (
            "We executed these calls:\n"
            + "\n".join(outputs) + "\n\n"     # ← real newlines here
            + "\n\nOriginal plan:\n" + plan_output
            + "\n\nDid these outputs satisfy the original intent?"
              " If not, provide a corrected plan text only; otherwise reply 'OK'."
        )
        self._print_stage_context("reflection_and_replan", {"tool_outputs": outputs})
        resp = self._stream_and_capture(
            self.secondary_model,
            [{"role":"user","content": prompt}],
            tag="[Reflection]"
        ).strip()
        if resp.upper() == "OK":
            return None

        ctx = ContextObject.make_stage(
            "reflection_and_replan",
            [c.context_id for c in tool_ctxs],
            {"replan": resp}
        )
        ctx.stage_id = "reflection_and_replan"
        ctx.summary  = resp
        ctx.touch(); self.repo.save(ctx)
        return resp


    def _stage9_invoke_with_retries(
        self,
        raw_calls: List[str],
        plan_output: str,
        selected_schemas: List[ContextObject],
        user_text: str,
        clar_metadata: Dict[str, Any]
    ) -> List[ContextObject]:
        """
        1) Normalize & execute each call in raw_calls in order.
        2) Run nested inner calls first, capture outputs to ${tool}_output.
        3) Substitute any ${tool}_output variables.
        4) Persist each invocation (output/exception) in context.
        5) On failures, retry up to 10×, prompting only for the failed calls,
           including original query, clarification notes, plan, extracted vars,
           and full schemas for the failed tools.
        Returns list of all tool_output ContextObjects (including nested).
        """
        import json, re
        from typing import Tuple, Any, Dict

        def _norm(calls):
            out = []
            for c in calls:
                if isinstance(c, dict) and "tool_call" in c:
                    out.append(c["tool_call"])
                elif isinstance(c, str):
                    out.append(c)
            return out

        def validate(res: Dict[str, Any]) -> Tuple[bool, str]:
            exc = res.get("exception")
            return (exc is None, exc or "")

        # ── INITIALIZE ─────────────────────────────────────────────────────────
        raw_calls = _norm(raw_calls)
        call_status: Dict[str, bool] = {c: False for c in raw_calls}
        tool_ctxs: List[ContextObject] = []
        last_results: Dict[str, Any] = {}
        max_retries = 10

        # ── RETRY LOOP ──────────────────────────────────────────────────────────
        for attempt in range(1, max_retries + 1):
            errors: List[Tuple[str, str]] = []
            print(f"\n>>> [Attempt {attempt}/{max_retries}] Executing tool_calls", flush=True)

            for original_call in list(raw_calls):
                # skip if already succeeded
                if call_status[original_call]:
                    continue

                # 1) Substitute any ${tool}_output variables
                call_str = original_call
                for name, val in last_results.items():
                    call_str = call_str.replace(f"${{{name}_output}}", repr(val))

                # 2) Detect & run any nested inner calls first
                inner_calls = re.findall(r'\b([A-Za-z_]\w*)\(\)', call_str)
                for inner in inner_calls:
                    inner_call = f"{inner}()"
                    if inner not in last_results:
                        print(f"[NestedInvocation] Running inner: {inner_call}", flush=True)
                        try:
                            iout = Tools.run_tool_once(inner_call)
                            iblock = {"output": iout, "exception": None}
                        except Exception as e:
                            iblock = {"output": None, "exception": str(e)}
                        ok_i, err_i = validate(iblock)
                        print(f"[NestedInvocation] {'OK' if ok_i else 'ERROR: '+err_i}", flush=True)
                        last_results[inner] = iblock["output"]

                        # persist nested result
                        inner_sch = next(
                            sch for sch in selected_schemas
                            if json.loads(sch.metadata["schema"])["name"] == inner
                        )
                        in_ctx = ContextObject.make_stage(
                            "tool_output",
                            [inner_sch.context_id],
                            iblock
                        )
                        in_ctx.metadata.update(iblock)
                        in_ctx.stage_id = f"tool_output_{attempt}_inner_{inner}"
                        in_ctx.summary  = (
                            str(iblock["output"])
                            if ok_i
                            else f"ERROR: {err_i.splitlines()[-1]}"
                        )
                        in_ctx.touch(); self.repo.save(in_ctx)
                        tool_ctxs.append(in_ctx)

                # 3) Execute the primary call
                print(f"[ToolInvocation] Running: {call_str}", flush=True)
                try:
                    out = Tools.run_tool_once(call_str)
                    block = {"output": out, "exception": None}
                except Exception as e:
                    block = {"output": None, "exception": str(e)}

                ok, err = validate(block)
                if ok:
                    print(f"[ToolInvocation] OK → {block['output']!r}", flush=True)
                else:
                    print(f"[ToolInvocation] ERROR → {err.splitlines()[-1]}", flush=True)

                # capture for future substitution
                tool_name = original_call.split("(", 1)[0]
                last_results[tool_name] = block["output"]

                # persist this invocation
                sch = next(
                    sch for sch in selected_schemas
                    if json.loads(sch.metadata["schema"])["name"] == tool_name
                )
                out_ctx = ContextObject.make_stage(
                    "tool_output",
                    [sch.context_id],
                    block
                )
                out_ctx.metadata.update(block)
                out_ctx.stage_id = f"tool_output_{attempt}_{tool_name}"
                out_ctx.summary  = (
                    str(block["output"])
                    if ok
                    else f"ERROR: {err.splitlines()[-1]}"
                )
                out_ctx.touch(); self.repo.save(out_ctx)
                tool_ctxs.append(out_ctx)

                call_status[original_call] = ok
                if not ok:
                    errors.append((original_call, err))

            # if all succeeded, break
            if not errors:
                break

            # ── PREPARE RETRY PROMPT ───────────────────────────────────────────────
            err_lines    = [f"{c} → {e.splitlines()[-1]}" for c,e in errors]
            failed_calls = [c for c,_ in errors]

            # only schemas for the failures
            docs: List[str] = []
            for sch in selected_schemas:
                data = json.loads(sch.metadata["schema"])
                if data["name"] in {f.split("(")[0] for f in failed_calls}:
                    params = json.dumps(data.get("parameters", {}), indent=2)
                    desc   = data.get("description","(no docs)")
                    docs.append(
                        f"• **{data['name']}**\n"
                        f"  {desc}\n"
                        f"  parameters:\n{params}"
                    )

            # extracted-variable block
            vars_block = "\n".join(f"{n}_output = {v!r}" for n,v in last_results.items())

            retry_sys = (
                "Some tool calls failed:\n"
                + "\n".join(err_lines)
                + "\n\n**Original query:**\n" + user_text
                + "\n\n**Clarification notes:**\n" + clar_metadata.get("notes","(none)")
                + "\n\n**Plan:**\n"     + plan_output
                + (f"\n\n**Available variables:**\n{vars_block}" if vars_block else "")
                + "\n\n**Schemas for failing tools:**\n"
                + "\n\n".join(docs)
                + "\n\nPlease correct **only** the failing calls. "
                  "Reply **ONLY** with one-line JSON:\n"
                  '{"tool_calls":["fixed_call1(...)", ...]}'
            )
            self._print_stage_context("tool_chaining_retry", {
                "errors":    err_lines,
                "variables": vars_block.split("\n") if vars_block else [],
                "fail_docs": docs,
            })

            retry_msgs = [
                {"role":"system", "content": retry_sys},
                {"role":"user",   "content": json.dumps({"tool_calls": failed_calls})}
            ]
            retry_out = self._stream_and_capture(self.secondary_model, retry_msgs, tag="[ToolChainRetry]")

            # parse corrected calls
            try:
                fixed = json.loads(retry_out.strip())["tool_calls"]
            except:
                parsed = Tools.parse_tool_call(retry_out)
                fixed = parsed if isinstance(parsed, list) else ([parsed] if parsed else failed_calls)

            # merge corrections back into raw_calls (preserving order)
            fixed = _norm(fixed)
            if not fixed:
                fixed = failed_calls

            new_raw = []
            fix_iter = iter(fixed)
            for c in raw_calls:
                if c in failed_calls:
                    new_raw.append(next(fix_iter, c))
                else:
                    new_raw.append(c)
            raw_calls = new_raw
            # reset statuses for next attempt
            call_status = {c: False for c in raw_calls}

        return tool_ctxs

    def _stage10b_response_critique_and_safety(
        self,
        draft: str,
        user_text: str
    ) -> str:
        """
        Final‐pass LLM check: remove hallucinations, enforce policy,
        and polish tone/format. Returns the corrected answer.
        """
        system = (
            "You are a final-pass assistant. Given a draft answer and the original user"
            " question, identify any unsupported statements, correct them, ensure policy"
            " compliance, and polish the tone. Return only the fixed answer."
        )
        self._print_stage_context("response_critique", {"draft": draft})
        msgs = [
            {"role":"system","content": system},
            {"role":"user",  "content": f"Draft:\n{draft}\n\nUser asked: {user_text}"}
        ]
        polished = self._stream_and_capture(self.secondary_model, msgs, tag="[Critic]")
        ctx = ContextObject.make_stage(
            "response_critique", [], {"text": polished}
        )
        ctx.stage_id = "response_critique"
        ctx.summary  = polished
        ctx.touch(); self.repo.save(ctx)
        return polished

    def _stage10_assemble_and_infer(self, user_text: str, state: Dict[str, Any]) -> str:
        # collect all context IDs, including tool outputs
        refs = [state['user_ctx'].context_id,
                state['sys_ctx'].context_id] + state['recent_ids']
        refs += [state['clar_ctx'].context_id,
                 state['know_ctx'].context_id,
                 state['plan_ctx'].context_id]
        refs += [t.context_id for t in state['tool_ctxs']]

        # dedupe & fetch
        seen, allr = set(), []
        for cid in refs:
            if cid not in seen:
                seen.add(cid)
                allr.append(cid)
        ctxs = [self.repo.get(cid) for cid in allr]
        ctxs.sort(key=lambda c: c.timestamp)

        # **Use real newlines** so the LLM sees each tool output separately
        interm = "\n".join(f"[{c.semantic_label}] {c.summary}" for c in ctxs)

        final_sys = (
            "You will receive quite a bit of context, including tool outputs, "
            "assemble it into a clean and concise relevant answer. "
            "Do not hallucinate."
        )
        msgs = [
            {"role": "system", "content": final_sys},
            {"role": "system", "content": self.inference_prompt},
            {"role": "system", "content": interm},
            {"role": "user",   "content": user_text},
        ]

        reply = self._stream_and_capture(self.primary_model, msgs, tag="[Assistant]")

        # persist final
        resp_ctx = ContextObject.make_stage(
            "final_inference", [state['tc_ctx'].context_id], {"text": reply}
        )
        resp_ctx.stage_id = "final_inference"
        resp_ctx.summary  = reply
        resp_ctx.touch()
        self.repo.save(resp_ctx)

        return reply


    def _stage11_memory_writeback(
        self,
        final_answer: str,
        tool_ctxs: List[ContextObject]
    ) -> None:
        """
        Persist the final answer into long-term memory as a knowledge artifact.
        """
        # Create a new knowledge ContextObject
        mem = ContextObject.make_knowledge(
            label="auto_memory",
            content=final_answer,
            tags=["memory_writeback"]
        )
        mem.touch()
        self.repo.save(mem)

        # Also record that we co-activated this memory with all tool outputs
        ctx_ids = [c.context_id for c in tool_ctxs]
        self.memman.reinforce(mem.context_id, ctx_ids)

        # Log for debugging
        meta = {"stored_memory_id": mem.context_id, "coactivated_with": ctx_ids}
        self._print_stage_context("memory_writeback", meta)
        
if __name__ == "__main__":
    asm = Assembler(
        context_path="context.jsonl",
        config_path="config.json",
        top_k=5,
    )
    print("Assembler ready. Type your message, Ctrl-C to quit.")
    try:
        while True:
            msg = input(">> ").strip()
            if msg:
                print(asm.run_with_meta_context(msg))
    except KeyboardInterrupt:
        print("\nGoodbye.")
