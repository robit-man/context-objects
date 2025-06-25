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

    def _print_stage_context(self, name: str, sections: Dict[str, List[Any]]):
        print(f"\n>>> [Stage: {name}] Context window:")
        for title, lines in sections.items():
            print(f"  -- {title}:")
            for ln in lines:
                print(f"     {ln}")

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

        # Stage 6: prepare tools & plan summary
        tools_list = self._stage6_prepare_tools()
        state['plan_ctx'], state['plan_output'] = self._stage7_planning_summary(
            state['clar_ctx'], state['know_ctx'], tools_list
        )

        # Stage 7: tool chaining → parse tool_calls
        state['tc_ctx'], raw_calls, selected_schemas = self._stage8_tool_chaining(
            state['plan_ctx'], state['plan_output'], tools_list
        )

        # Stage 8: invoke with retries
        state['tool_ctxs'] = self._stage9_invoke_with_retries(
            raw_calls, state['plan_output'], selected_schemas
        )

        # Stage 9: assemble intermediate & final inference
        return self._stage10_assemble_and_infer(user_text, state)


    def _stage1_record_input(self, user_text: str) -> ContextObject:
        ctx = ContextObject.make_segment("user_input", [], tags=["user_input"])
        ctx.summary, ctx.stage_id = user_text, "user_input"
        ctx.touch()
        self.repo.save(ctx)
        return ctx

    def _stage2_load_system_prompts(self) -> ContextObject:
        return self._load_system_prompts()

    def _stage3_retrieve_and_merge_context(
        self, user_text: str, user_ctx: ContextObject, sys_ctx: ContextObject
    ) -> Dict[str, Any]:
        now, past = default_clock(), default_clock() - timedelta(minutes=self.lookback)
        tr = (past.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%dT%H%M%SZ"))
        history = self._get_history()
        recent  = self.engine.query(
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
        return {
            "history": history,
            "recent": recent,
            "assoc": assoc,
            "recent_ids": recent_ids
        }

    def _stage4_intent_clarification(
        self, user_text: str, state: Dict[str, Any]
    ) -> ContextObject:
        block = "\n".join(
            f"[{c.semantic_label}] {c.summary}"
            for c in [state['sys_ctx']] + state['history'] + state['recent'] + state['assoc']
        )
        msgs = [
            {"role": "system", "content": self.clarifier_prompt},
            {"role": "system", "content": f"Context:\n{block}"},
            {"role": "user",   "content": user_text},
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
        snippets = []
        for kw in clar_ctx.metadata.get("keywords", []):
            hits = self.engine.query(
                stage_id="external_knowledge_retrieval",
                similarity_to=kw,
                top_k=self.top_k
            )
            snippets += [h.summary for h in hits]
        ctx = ContextObject.make_stage(
            "external_knowledge_retrieval",
            clar_ctx.references,
            {"snippets": snippets}
        )
        ctx.stage_id = "external_knowledge_retrieval"
        ctx.summary  = "\n".join(snippets) or "(none)"
        ctx.touch()
        self.repo.save(ctx)
        self._print_stage_context("external_knowledge_retrieval", {"snippets": [ctx.summary]})
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
        tools_text = "\n".join(f"- **{t['name']}**: {t['description']}" for t in tools_list)
        system = (
            "Available tools:\n" + tools_text +
            "\n\nDevise a concise plan. If you intend to call tools, "
            "include `tool_name(arg1=..., arg2=...)`."
        )
        inp = clar_ctx.summary + "\n\nSnippets:\n" + know_ctx.summary
        self._print_stage_context("planning_summary", {
            "tools": [tools_text], "input": [inp[:200]]
        })
        msgs = [{"role":"system","content":system}, {"role":"user","content":inp}]
        plan = self._stream_and_capture(self.secondary_model, msgs, tag="[Planner]")

        ctx = ContextObject.make_stage(
            "planning_summary", know_ctx.references, {"plan": plan}
        )
        ctx.stage_id = "planning_summary"
        ctx.summary  = plan
        ctx.touch()
        self.repo.save(ctx)
        return ctx, plan

    def _stage8_tool_chaining(
        self,
        plan_ctx: ContextObject,
        plan_output: str,
        tools_list: List[Dict[str, str]]
    ) -> Tuple[ContextObject, List[str], List[ContextObject]]:
        import inspect

        # Step A: initial one-line JSON prompt
        tools_text = "\n".join(f"- **{t['name']}**: {t['description']}" for t in tools_list)
        system_short = (
            "Available tools:\n" +
            tools_text +
            "\n\nReply ONLY with one-line JSON:\n"
            '{"tool_calls": ["tool1(arg1=...,arg2=...)", ...]}'
        )
        self._print_stage_context("tool_chaining", {
            "tools": [tools_text],
            "plan":  [plan_output[:200]],
        })
        msgs_short = [
            {"role": "system", "content": system_short},
            {"role": "user",   "content": plan_output},
        ]
        out_short = self._stream_and_capture(self.secondary_model, msgs_short, tag="[ToolChain]")

        # Step B: parse JSON or fallback to Tools.parse_tool_call
        try:
            raw_calls = json.loads(out_short.strip())["tool_calls"]
        except:
            parsed = Tools.parse_tool_call(out_short)
            raw_calls = parsed if isinstance(parsed, list) else ([parsed] if parsed else [])

        # Step C: regex-extract any calls embedded in the plan text
        plan_regex = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)', plan_output)
        for call in plan_regex:
            if call not in raw_calls:
                raw_calls.append(call)

        # Step D: ensure every tool name mentioned by name in the plan is invoked
        for t in tools_list:
            name = t["name"]
            if name in plan_output and not any(rc.startswith(f"{name}(") for rc in raw_calls):
                raw_calls.append(f"{name}()")

        # Step E: save the chaining ContextObject
        tc_ctx = ContextObject.make_stage(
            "tool_chaining",
            plan_ctx.references,
            {"tool_calls": raw_calls}
        )
        tc_ctx.stage_id = "tool_chaining"
        tc_ctx.summary  = json.dumps(raw_calls)
        tc_ctx.touch()
        self.repo.save(tc_ctx)

        # Step F: select only the schemas for the chosen calls
        all_schemas = self.repo.query(
            lambda c: c.component=="schema" and "tool_schema" in c.tags
        )
        sel_names = {c.split("(")[0] for c in raw_calls if isinstance(c, str)}
        selected_schemas = [
            sch for sch in all_schemas
            if json.loads(sch.metadata["schema"])["name"] in sel_names
        ]

        return tc_ctx, raw_calls, selected_schemas

    def _stage9_invoke_with_retries(
        self,
        raw_calls: List[str],
        plan_output: str,
        selected_schemas: List[ContextObject]
    ) -> List[ContextObject]:
        """
        Executes raw_calls with up to 10 retries.
        On errors, prints the full retry context so the LLM can correct calls.
        """
        from typing import Tuple, Dict

        def validate(block: Dict[str, Any]) -> Tuple[bool, str]:
            exc = block.get("exception")
            return (exc is None, exc or "")

        # Normalize any dict-wrapped calls
        def _norm(calls):
            out = []
            for c in calls:
                if isinstance(c, dict) and "tool_call" in c:
                    out.append(c["tool_call"])
                elif isinstance(c, str):
                    out.append(c)
            return out

        raw_calls = _norm(raw_calls)
        call_status = {c: None for c in raw_calls}
        tool_ctxs: List[ContextObject] = []
        max_retries = 10

        for attempt in range(1, max_retries + 1):
            errors: List[Tuple[str,str]] = []
            print(f"\n>>> [Attempt {attempt}/{max_retries}] Executing tool_calls", flush=True)

            # invoke pending
            for call_str in list(raw_calls):
                if call_status.get(call_str) is True:
                    continue

                print(f"[ToolInvocation] Running: {call_str}", flush=True)
                try:
                    raw = Tools.run_tool_once(call_str)
                    if isinstance(raw, dict) and "exception" in raw:
                        block = {"output": raw.get("output"), "exception": raw["exception"]}
                    else:
                        block = {"output": raw, "exception": None}
                except Exception as e:
                    block = {"output": None, "exception": str(e)}

                ok, err = validate(block)
                print(f"[ToolInvocation] {'OK' if ok else 'ERROR: ' + err}", flush=True)

                # persist output or error
                name = call_str.split("(")[0]
                sch_ctx = next(
                    s for s in selected_schemas
                    if json.loads(s.metadata["schema"])["name"] == name
                )
                out_ctx = ContextObject.make_stage(
                    "tool_output", [sch_ctx.context_id], block
                )
                out_ctx.stage_id = f"tool_output_{attempt}"
                out_ctx.summary  = block["output"] if ok else f"ERROR: {err}"
                out_ctx.touch()
                self.repo.save(out_ctx)
                tool_ctxs.append(out_ctx)

                call_status[call_str] = ok
                if not ok:
                    errors.append((call_str, err))

            if not errors:
                break

            # prepare retry context
            err_lines    = [f"{c} → {e}" for c,e in errors]
            failed_calls = [c for c,_ in errors]

            docs = []
            for sch in selected_schemas:
                data = json.loads(sch.metadata["schema"])
                if data["name"] in {fc.split("(")[0] for fc in failed_calls}:
                    docs.append(f"**{data['name']}**: {data.get('description','(no docs)')}")

            analysis_system = (
                "The following tool calls failed:\n"
                + "\n".join(err_lines)
                + "\n\nOriginal plan:\n" + plan_output
                + "\n\nTool signatures & docs:\n" + "\n\n".join(docs)
                + "\n\nReply ONLY with JSON {\"tool_calls\": [ ... ]} of corrected calls."
            )

            self._print_stage_context("tool_chaining_retry", {
                "errors": err_lines,
                "docs":   docs,
                "prompt": [analysis_system]
            })

            retry_out = self._stream_and_capture(
                self.secondary_model,
                [
                    {"role":"system","content":analysis_system},
                    {"role":"user","content":json.dumps({"tool_calls":failed_calls})}
                ],
                tag="[ToolChainRetry]"
            )

            try:
                fixed = json.loads(retry_out.strip())["tool_calls"]
            except:
                parsed = Tools.parse_tool_call(retry_out)
                fixed = parsed if isinstance(parsed,list) else ([parsed] if parsed else [])

            fixed = _norm(fixed)
            if not fixed:
                fixed = [c for c,ok in call_status.items() if ok is not True]

            # merge corrected calls
            new_raw = []
            fix_iter = iter(fixed)
            for orig in raw_calls:
                new_raw.append(next(fix_iter, orig) if orig in failed_calls else orig)
            raw_calls = new_raw
            call_status = {c: call_status.get(c) for c in raw_calls}

        return tool_ctxs

    def _stage10_assemble_and_infer(self, user_text: str, state: Dict[str, Any]) -> str:
        refs = [state['user_ctx'].context_id, state['sys_ctx'].context_id] + state['recent_ids']
        refs += [
            state['clar_ctx'].context_id,
            state['know_ctx'].context_id,
            state['plan_ctx'].context_id
        ]
        refs += [t.context_id for t in state['tool_ctxs']]
        seen, allr = set(), []
        for r in refs:
            if r not in seen:
                seen.add(r)
                allr.append(r)
        ctxs = [self.repo.get(cid) for cid in allr]
        ctxs.sort(key=lambda c: c.timestamp)
        interm = "\\n".join(f"[{c.semantic_label}] {c.summary}" for c in ctxs)

        final_sys = (
            "Use ONLY the provided context—query, plan, calls, outputs—to answer. "
            "Do NOT invent new actions."
        )
        msgs = [
            {"role":"system","content": final_sys},
            {"role":"system","content": self.inference_prompt},
            {"role":"system","content": interm},
            {"role":"user","content": user_text},
        ]
        reply = self._stream_and_capture(self.primary_model, msgs, tag="[Assistant]")

        resp_ctx = ContextObject.make_stage(
            "final_inference", [state['tc_ctx'].context_id], {"text": reply}
        )
        resp_ctx.stage_id = "final_inference"
        resp_ctx.summary  = reply
        resp_ctx.touch()
        self.repo.save(resp_ctx)

        return reply


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
