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

    # ── Helpers ────────────────────────────────────────────────────────────
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

        # ── Stage 2: system prompts
        sys_ctx = self._load_system_prompts()

        # ── Stage 3: recent + associative retrieval
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
        stage_refs = {"recent_retrieval": recent_ids}

        # ── Stage 4: intent clarification
        clar_block = "\n".join(f"[{c.semantic_label}] {c.summary}" for c in [sys_ctx] + merged)
        clar_msgs = [
            {"role":"system","content": self.clarifier_prompt},
            {"role":"system","content": f"Context:\n{clar_block}"},
            {"role":"user",  "content": user_text},
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
        know_ctx.summary = "\n".join(snippets) or "(none)"
        know_ctx.touch(); self.repo.save(know_ctx)
        self._print_stage_context("external_knowledge_retrieval", {
            "snippets": [know_ctx.summary]
        })
        stage_refs["external_knowledge_retrieval"] = know_ctx.context_id

        # ── Stage 6: planning summary (first‐line tool list)
        schema_ctxs = self.repo.query(
            lambda c: c.component=="schema" and "tool_schema" in c.tags
        )
        tools_list = []
        for c in schema_ctxs:
            data = json.loads(c.metadata["schema"])
            tools_list.append({
                "name": data["name"],
                "description": data.get("description","").split("\n",1)[0]
            })
        tools_text = "\n".join(
            f"- **{t['name']}**: {t['description']}"
            for t in tools_list
        )
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
            {"role":"system","content": planning_system},
            {"role":"user",  "content": plan_input},
        ]
        plan_output = self._stream_and_capture(
            self.secondary_model, plan_msgs, tag="[Planner]"
        )
        plan_ctx = ContextObject.make_stage(
            "planning_summary",
            know_ctx.references,
            {"plan": plan_output}
        )
        plan_ctx.stage_id, plan_ctx.summary = "planning_summary", plan_output
        plan_ctx.touch(); self.repo.save(plan_ctx)
        stage_refs["planning_summary"] = plan_ctx.context_id

        # ── Stage 7: initial tool chaining (multi-call JSON) ─────────────
        tc_system = (
            "Available tools (name + one-line description):\n"
            f"{tools_text}\n\n"
            "Reply **ONLY** with one-line JSON:\n"
            '{"tool_calls":["tool1(arg1=...,arg2=...)", "tool2(...)"]}\n'
            "or\n"
            '{"tool_calls":[]}'
        )
        self._print_stage_context("tool_chaining", {
            "tools": [tools_text],
            "plan":  [plan_output[:200]],
        })
        chain_out = self._stream_and_capture(
            self.secondary_model,
            [{"role":"system","content":tc_system},
             {"role":"user",  "content":plan_output}],
            tag="[ToolChain]"
        )
        try:
            raw_calls = json.loads(chain_out.strip())["tool_calls"]
        except:
            parsed = Tools.parse_tool_call(chain_out)
            raw_calls = parsed if isinstance(parsed,list) else ([parsed] if parsed else [])

        tc_ctx = ContextObject.make_stage(
            "tool_chaining", plan_ctx.references, {"tool_calls": raw_calls}
        )
        tc_ctx.stage_id, tc_ctx.summary = "tool_chaining", json.dumps(raw_calls)
        tc_ctx.touch(); self.repo.save(tc_ctx)
        stage_refs["tool_chaining"] = tc_ctx.context_id

        # ── Stages 8–9: invoke → validate → intermediate fix-prompt → retry ───
        def validate(block: Dict[str,Any]) -> Tuple[bool,str]:
            """
            Returns (ok, error_message).
            Only an actual exception should be considered a failure.
            Treat None output as valid.
            """
            exc = block.get("exception")
            if exc:
                return False, exc
            return True, ""


        max_retries = 10
        tool_ctxs = []
        call_status = {call: None for call in raw_calls}  # None=pending, True=ok, False=error
        for attempt in range(1, max_retries+1):
            print(f"\n>>> [Attempt {attempt}/{max_retries}] Executing tool_calls", flush=True)
            errors = []

            # execute only pending calls
            for call_str in raw_calls:
                if call_status[call_str] is True:
                    continue  # already succeeded

                print(f"[ToolInvocation] Running: {call_str}", flush=True)
                try:
                    result = Tools.run_tool_once(call_str)
                    block  = {"output": result, "exception": None}
                except Exception as e:
                    block  = {"output": None, "exception": str(e)}

                ok, err = validate(block)
                status = "OK" if ok else f"ERROR: {err}"
                print(f"[ToolInvocation] {status}", flush=True)

                # persist this tool_output
                name = call_str.split("(")[0]
                sch_ctx = next(
                    c for c in schema_ctxs
                    if json.loads(c.metadata["schema"])["name"] == name
                )
                out_ctx = ContextObject.make_stage(
                    "tool_output", [sch_ctx.context_id], block
                )
                out_ctx.stage_id = f"tool_output_{attempt}"
                out_ctx.summary = block["output"] if ok else f"ERROR: {err}"
                out_ctx.touch(); self.repo.save(out_ctx)
                tool_ctxs.append(out_ctx)

                call_status[call_str] = ok
                if not ok:
                    errors.append((call_str, err))

            # if no errors remain, stop retrying
            if not errors:
                break

            # prepare retry prompt: only for calls that failed
            err_lines = [f"{c} → {e}" for c,e in errors]
            failed_calls = [c for c,_ in errors]

            # gather full docs for failed tools
            docs = []
            for c in schema_ctxs:
                schema = json.loads(c.metadata["schema"])
                if schema["name"] in {fc.split("(")[0] for fc in failed_calls}:
                    docs.append(f"**{schema['name']}**\n{schema.get('description','(no docs)')}")

            retry_system = (
                "Some tool calls failed:\n"
                + "\n".join(err_lines)
                + "\n\nOriginal plan:\n"
                + plan_output
                + "\n\nFull docstrings for the failed tools:\n\n"
                + "\n\n".join(docs)
                + "\n\nPlease correct **only** the failing calls. "
                  "Reply **ONLY** with JSON:\n"
                '{"tool_calls": ["call1(...)", ...]}'
            )
            self._print_stage_context("tool_chaining_retry", {
                "errors": err_lines,
            })
            retry_out = self._stream_and_capture(
                self.secondary_model,
                [{"role":"system","content":retry_system},
                 {"role":"user","content": json.dumps({"tool_calls": failed_calls})}],
                tag="[ToolChainRetry]"
            )

            # parse the corrected calls
            try:
                fixed = json.loads(retry_out.strip())["tool_calls"]
            except:
                parsed = Tools.parse_tool_call(retry_out)
                fixed = parsed if isinstance(parsed,list) else ([parsed] if parsed else [])

            # merge back into raw_calls, preserving successes
            new_raw = []
            for call in raw_calls:
                if call in failed_calls:
                    # take next corrected if available, else keep old to try again
                    candidate = fixed.pop(0) if fixed else call
                    new_raw.append(candidate)
                else:
                    new_raw.append(call)
            raw_calls = new_raw

        stage_refs["tool_invocation"] = [c.context_id for c in tool_ctxs]

        # ── Stage 10: assemble intermediate context ────────────────────────
        refs = [user_ctx.context_id, sys_ctx.context_id] + recent_ids
        for v in stage_refs.values():
            refs += v if isinstance(v,list) else [v]
        refs += [t.context_id for t in tool_ctxs]
        seen, all_refs = set(), []
        for r in refs:
            if r not in seen:
                seen.add(r); all_refs.append(r)
        all_ctxs = [self.repo.get(cid) for cid in all_refs]
        all_ctxs.sort(key=lambda c:c.timestamp)
        interm = "\n".join(f"[{c.semantic_label}] {c.summary}" for c in all_ctxs)

        # ── Stage 11: final inference (tool-free) ─────────────────────────
        final_sys = (
            "Use ONLY the provided context—original query, plan, exact tool calls & their outputs—to answer. "
            "Do NOT invent new actions. If the context suffices, answer; otherwise admit you cannot."
        )
        reply = self._stream_and_capture(
            self.primary_model,
            [{"role":"system","content":final_sys},
             {"role":"system","content":interm},
             {"role":"user","content":user_text}],
            tag="[Assistant]"
        )

        resp_ctx = ContextObject.make_stage(
            "final_inference",[tc_ctx.context_id],{"text": reply}
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
