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
from dataclasses import dataclass, field
import numpy as np
import difflib
from ollama import chat, embed
from tts_service import TTSManager
from audio_service import AudioService
from context import ContextObject, ContextRepository, MemoryManager, default_clock
from tools import TOOL_SCHEMAS, Tools
import inspect
logger = logging.getLogger("assembler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] assembler: %(message)s",
)

@dataclass
class TaskNode:
    call: str
    parent: Optional["TaskNode"] = None
    children: List["TaskNode"] = field(default_factory=list)
    context_ids: List[str] = field(default_factory=list)
    completed: bool = False
    errors: List[str] = field(default_factory=list)

class TaskExecutor:
    """
    Executes a TaskNode tree in DFS order:
      1) Validate the node.call via Assembler._stage7b_plan_validation
      2) Chain & confirm tool calls (stages 8 & 8.5)
      3) Invoke with retries (stage 9)
      4) Reflect & possibly replan (stage 9b)
      5) Recurse into children
      6) Mark node completed
    Accumulates all resulting ContextObject IDs into each node's context_ids.
    """
    def __init__(self, asm: "Assembler", user_text: str, clar_metadata: Dict[str,Any]):
        self.asm = asm
        self.user_text = user_text
        self.clar_metadata = clar_metadata

    def execute(self, node: TaskNode) -> None:
        import json

        # 1) Static validation / fix — always pull the real plan_ctx from the node itself
        plan_ctx_id = node.context_ids[0]
        plan_ctx_obj = self.asm.repo.get(plan_ctx_id)

        # reuse the planning-validation with a single-call plan
        _, errors, fixed = self.asm._stage7b_plan_validation(
            plan_ctx_obj,
            node.call,
            self.asm.tools_list
        )
        if errors:
            node.errors = [err for (_, err) in errors]
        calls = fixed or [node.call]

        # 2) Tool chaining
        tc_ctx, raw_calls, schemas = self.asm._stage8_tool_chaining(
            plan_ctx_obj,
            "\n".join(calls),
            self.asm.tools_list
        )
        node.context_ids.append(tc_ctx.context_id)

        # 3) User confirmation
        confirmed = self.asm._stage8_5_user_confirmation(raw_calls, self.user_text)

        # 4) Invoke with retries
        tool_ctxs = self.asm._stage9_invoke_with_retries(
            confirmed,
            "\n".join(calls),
            schemas,
            self.user_text,
            self.clar_metadata
        )
        for t in tool_ctxs:
            node.context_ids.append(t.context_id)

            # ——— record per-tool success/failure —————————————
            if t.metadata.get("exception") is None:
                succ = ContextObject.make_success(
                    f"Tool `{t.metadata.get('tool_name', t.semantic_label)}` succeeded",
                    refs=[t.context_id]
                )
                succ.touch()
                self.asm.repo.save(succ)
                self.asm.memman.reinforce(succ.context_id, [t.context_id])
            else:
                fail = ContextObject.make_failure(
                    f"Tool `{t.metadata.get('tool_name', t.semantic_label)}` failed: {t.metadata.get('exception')}",
                    refs=[t.context_id]
                )
                fail.touch()
                self.asm.repo.save(fail)
                self.asm.memman.reinforce(fail.context_id, [t.context_id])

        # 5) Reflection & replan
        replan = self.asm._stage9b_reflection_and_replan(
            [self.asm.repo.get(cid) for cid in node.context_ids],
            "\n".join(calls),
            self.user_text,
            self.clar_metadata
        )

        # ——— record reflection outcome —————————————
        if replan is None:
            succ = ContextObject.make_success(
                "Reflection validated original plan (OK)",
                refs=node.context_ids
            )
            succ.touch()
            self.asm.repo.save(succ)
            self.asm.memman.reinforce(succ.context_id, node.context_ids)
        else:
            fail = ContextObject.make_failure(
                "Reflection triggered plan adjustment",
                refs=node.context_ids
            )
            fail.touch()
            self.asm.repo.save(fail)
            self.asm.memman.reinforce(fail.context_id, node.context_ids)

            # if there's a new plan, expand into subtasks
            try:
                tree = json.loads(replan)
                node.children = self.asm._parse_task_tree(tree, parent=node)
            except Exception:
                pass

        # 6) Recurse into children
        for child in node.children:
            self.execute(child)

        # ——— at end of this node, log overall task success/failure ————
        if not node.errors and replan is None:
            overall = ContextObject.make_success(
                f"Task `{node.call}` completed successfully",
                refs=node.context_ids
            )
            overall.touch()
            self.asm.repo.save(overall)
            self.asm.memman.reinforce(overall.context_id, node.context_ids)
        else:
            overall = ContextObject.make_failure(
                f"Task `{node.call}` failed or was replanned",
                refs=node.context_ids
            )
            overall.touch()
            self.asm.repo.save(overall)
            self.asm.memman.reinforce(overall.context_id, node.context_ids)

        node.completed = True


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

    """
    Fractal Assembler with recursive TaskExecutor.
    """
    def __init__(
        self,
        context_path:     str = "context.jsonl",
        config_path:      str = "config.json",
        lookback_minutes: int = 60,
        top_k:            int = 10,
        tts_manager:      TTSManager | None = None,
    ):
        # load or init config
        try:
            self.cfg = json.load(open(config_path))
        except FileNotFoundError:
            self.cfg = {}

        # models & lookback
        self.primary_model   = self.cfg.get("primary_model",   "gemma3:4b")
        self.secondary_model = self.cfg.get("secondary_model", self.primary_model)
        self.lookback        = self.cfg.get("lookback_minutes", lookback_minutes)
        self.top_k           = self.cfg.get("top_k",            top_k)
        self.hist_k          = self.cfg.get("history_turns",    5)

        # system‐and‐stage prompts
        self.clarifier_prompt = self.cfg.get(
            "clarifier_prompt",
            "You are Clarifier. Expand the user’s intent into a JSON object with "
            "two keys: 'keywords' (an array of concise keywords) and 'notes' "
            "You must stay away from security or privacy relatec concerns, as those are handled downstream, "
            "and remain extremely objective in your choices, as the result of your analysis aids in calling of available tools."
            "(a brief explanation). Output only valid JSON."
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

        @staticmethod
        def embed_text(text):
            """
            Embed into a 1-D numpy array of shape (768,).
            """
            try:
                #log_message("Embedding text for context.", "PROCESS")
                response = embed(model="nomic-embed-text", input=text)
                vec = np.array(response["embeddings"], dtype=float)
                # ensure 1-D
                vec = vec.flatten()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                #log_message("Text embedding computed and normalized.", "SUCCESS")
                return vec
            except Exception as e:
                #log_message("Error during text embedding: " + str(e), "ERROR")
                return np.zeros(768, dtype=float)


        self.engine = ContextQueryEngine(
           self.repo,
           lambda t: embed_text(t)  # Use your custom embedding function
        )

                
        self._seed_tool_schemas()
        self._seed_static_prompts()
        self.tts = tts_manager
        
    def _seed_tool_schemas(self) -> None:
        """
        Guarantee exactly one ContextObject per tool in TOOL_SCHEMAS:
         - INSERT if missing
         - UPDATE if JSON differs
         - DEDUPE extras
        """
        import json

        # 1) Bucket existing schema rows by tool-name
        buckets: Dict[str, List[ContextObject]] = {}
        for ctx in self.repo.query(lambda c: c.component == "schema" and "tool_schema" in c.tags):
            try:
                name = json.loads(ctx.metadata["schema"])["name"]
                buckets.setdefault(name, []).append(ctx)
            except Exception:
                continue

        # 2) Upsert each canonical schema
        for name, canonical in TOOL_SCHEMAS.items():
            canonical_blob = json.dumps(canonical, sort_keys=True)
            rows = buckets.get(name, [])

            # A) No existing row → insert
            if not rows:
                new_ctx = ContextObject.make_schema(
                    label=name,
                    schema_def=canonical_blob,
                    tags=["artifact", "tool_schema"],
                )
                new_ctx.touch()
                self.repo.save(new_ctx)
                continue

            # B) Keep only the newest, delete duplicates
            rows.sort(key=lambda c: c.timestamp, reverse=True)
            keeper = rows[0]
            for dup in rows[1:]:
                self.repo.delete(dup.context_id)

            # C) Check for changes
            changed = False
            existing_blob = json.dumps(json.loads(keeper.metadata["schema"]), sort_keys=True)
            if existing_blob != canonical_blob:
                keeper.metadata["schema"] = canonical_blob
                changed = True

            if "tool_schema" not in keeper.tags:
                keeper.tags.append("tool_schema")
                changed = True

            # D) Only persist if we actually changed something
            if changed:
                keeper.touch()
                self.repo.save(keeper)

    def _seed_static_prompts(self) -> None:
        """
        Guarantee exactly one ContextObject for each static system prompt:
         - INSERT if missing
         - UPDATE if text differs
         - DEDUPE extras
        """
        static = {
            "clarifier_prompt": self.clarifier_prompt,
            "assembler_prompt": self.assembler_prompt,
            "inference_prompt": self.inference_prompt,
        }

        # Bucket rows by semantic_label
        buckets: Dict[str, List[ContextObject]] = {}
        for ctx in self.repo.query(lambda c: c.component == "prompt"):
            buckets.setdefault(ctx.semantic_label, []).append(ctx)

        for label, desired_text in static.items():
            rows = buckets.get(label, [])

            # A) No existing row → insert
            if not rows:
                new_ctx = ContextObject.make_prompt(
                    label=label,
                    prompt_text=desired_text,
                    tags=["artifact", "prompt"],
                )
                new_ctx.touch()
                self.repo.save(new_ctx)
                continue

            # B) Keep only the newest, delete duplicates
            rows.sort(key=lambda c: c.timestamp, reverse=True)
            keeper = rows[0]
            for dup in rows[1:]:
                self.repo.delete(dup.context_id)

            # C) Check for changes
            changed = False
            if keeper.metadata.get("prompt") != desired_text:
                keeper.metadata["prompt"] = desired_text
                changed = True

            if "prompt" not in keeper.tags:
                keeper.tags.append("prompt")
                changed = True

            # D) Only persist if we actually changed something
            if changed:
                keeper.touch()
                self.repo.save(keeper)



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
        
    def _get_or_make_singleton(
        self,
        *,
        label: str,
        component: str,
        tags: list[str] | None = None,
    ) -> ContextObject:
        """
        Return the one-and-only ContextObject with `semantic_label == label`
        and `component == component`.

        - If none exists → create it.
        - If >1 exist    → keep the newest, delete the extras.
        - Always make sure the supplied `tags` are present on the keeper.
        """
        tags = tags or []
        # grab *all* candidates
        rows = self.repo.query(
            lambda c: c.semantic_label == label and c.component == component
        )

        if not rows:                       # ---- INSERT ----
            ctx = ContextObject.make_stage(label, [], {})
            ctx.component = component
            ctx.tags = list(tags)
            ctx.touch()
            self.repo.save(ctx)
            return ctx

        # ---- DEDUPE ----  (rows[0] is newest because jsonl is append-only)
        rows.sort(key=lambda c: c.timestamp, reverse=True)
        keeper, *dups = rows
        for extra in dups:
            self.repo.delete(extra.context_id)

        # ensure tags are present
        for t in tags:
            if t not in keeper.tags:
                keeper.tags.append(t)
        return keeper


    def _load_narrative_context(self) -> ContextObject:
        keeper = self._get_or_make_singleton(
            label="narrative_context",
            component="stage",
            tags=["narrative"],
        )

        narr_objs = self.repo.query(lambda c: c.component == "narrative")
        narr_objs.sort(key=lambda c: c.timestamp)
        keeper.metadata["narrative"] = "\n".join(n.summary for n in narr_objs)
        keeper.summary = keeper.metadata["narrative"] or "(no narrative yet)"
        keeper.references = [n.context_id for n in narr_objs]
        keeper.touch()
        self.repo.save(keeper)
        return keeper
    
    def _load_system_prompts(self) -> ContextObject:
        keeper = self._get_or_make_singleton(
            label="system_prompts",
            component="stage",
            tags=["system"],
        )

        narr_ctx = self._load_narrative_context()
        static_objs = self.repo.query(
            lambda c: c.domain == "artifact" and c.component in ("prompt", "policy")
        )[: self.top_k]
        dyn = self.repo.query(
            lambda c: c.component == "policy" and "dynamic_prompt" in c.tags
        )[-5:]

        # build the text *fresh* each turn
        block = "---\n**My Narrative So Far:**\n" + narr_ctx.summary + "\n---\n"
        block += "\n".join(
            f"{c.semantic_label}: {c.metadata.get('prompt') or c.metadata.get('policy')}"
            for c in static_objs
        )
        block += "\n\n# Learned Prompt Adjustments:\n"
        block += "\n".join(f"* {c.summary}" for c in dyn) or "(none)"

        keeper.metadata["prompts"] = block
        keeper.summary = block
        keeper.references = (
            [narr_ctx.context_id]
            + [c.context_id for c in static_objs]
            + [c.context_id for c in dyn]
        )
        keeper.touch()
        self.repo.save(keeper)
        return keeper


    def _get_history(self) -> List[ContextObject]:
        segs = self.repo.query(
            lambda c: c.domain=="segment"
            and c.component in ("user_input","assistant")
        )
        segs.sort(key=lambda c: c.timestamp)
        return segs[-self.hist_k:]

    def _print_stage_context(self, name: str, sections: Dict[str, Any]):
        print(f"\n>>> [Stage: {name}] Context window:")
        for title, lines in sections.items():
            print(f"  -- {title}:")
            if isinstance(lines, str):
                for ln in lines.splitlines():
                    print(f"     {ln}")
            elif isinstance(lines, list):
                for ln in lines:
                    print(f"     {ln}")
            else:
                print(f"     {lines}")

    def _save_stage(self, ctx: ContextObject, stage: str):
        ctx.stage_id = stage
        ctx.summary = (
            (ctx.references and
             (ctx.metadata.get("plan") or ctx.metadata.get("tool_call")))
            or ctx.summary
        )
        ctx.touch()
        self.repo.save(ctx)

    def _stream_and_capture(
        self,
        model: str,
        messages: List[Dict[str,Any]],
        tag: str = ""
    ) -> str:
        out = ""
        print(f"{tag} ", end="", flush=True)
        for part in chat(model=model, messages=messages, stream=True):
            chunk = part["message"]["content"]
            print(chunk, end="", flush=True)
            out += chunk
        print()
        return out

    def _parse_task_tree(
        self,
        tree: Dict[str,Any],
        parent: Optional[TaskNode] = None
    ) -> List[TaskNode]:
        """
        Given JSON of shape {"tasks":[{"call":str,"subtasks":[...]}...]},
        return a list of TaskNode with proper parent links.
        """
        nodes: List[TaskNode] = []
        for t in tree.get("tasks", []):
            node = TaskNode(call=t["call"], parent=parent)
            node.children = self._parse_task_tree(
                {"tasks": t.get("subtasks", [])},
                parent=node
            )
            nodes.append(node)
        return nodes
    
    def _generate_system_prompt(self, purpose, schema=None, variables={}):
        system_meta = (
            "Generate a clear and concise instruction set for an agent to perform a task.\n"
            "First, review the purpose of the system prompt:\n"
            f"{purpose}\n\n"
        )
        
        if schema is not None:
            system_meta += (
                "Now, based on the purpose, we have a schema which will be used to perform regex and capture outputs. Creatively inform the system prompt you are about to generate based on the expected model outputs:\n"
                f"{schema}\n\n"
            )
        
        # Inject additional variables
        for var_name, var_value in variables.items():
            system_meta += (
                f"Additionally, consider the following variable: {var_name}.\n"
                f"The value of {var_name} is: {var_value}\n\n"
            )
        
        system_meta += (
            "Generate a clear instruction set based on the above, with no additional introduction or explanation. Output only the exact system message and nothing more."
        )
        
        system = self._stream_and_capture(
            self.primary_model,
            [{"role": "system", "content": system_meta}],
            tag="[Refine]"
        ).strip()
        
        return system


            
    def run_with_meta_context(self, user_text: str) -> str:
        import re, json

        print("\n=== run_with_meta_context ===")
        print("User input:", repr(user_text))

        # ——— Stage 0: handle pending yes/no confirmation ——————————
        print("\n--- Stage 0: Pending Confirmation Check ---")
        if getattr(self, "_awaiting_confirmation", False):
            print("⚠️  _awaiting_confirmation is True")
            print("Pending plan:", repr(self._pending_plan))
            ans = user_text.strip().lower()
            print("Normalized answer:", repr(ans))

            refine_prompt = (
                "you must determine if the user, in reply to the pending plan, is requesting a revision or confirming it as adequate.\n"
                "First, you must review the plan:\n"
                f"{self._pending_plan}\n\n"
                "Now here is the users reply to that plan:\n"
                f"{ans}\n\n"
                "Based on the users reply, if they are requesting a reconsideration, or denial of the plan, you must only output [replan], or [no] as the sole word you reply with and NOTHING MORE."
                "If the users response is favorable and appears to desire to run the plan, respond ONLY with [yes], or [sure] as the sole word you reply with and NOTHING MORE."
            )
            print("Refine prompt >>>\n", refine_prompt)
            analysis = self._stream_and_capture(
                self.primary_model,
                [{"role": "system", "content": refine_prompt}],
                tag="[Refine]"
            ).strip()
            print("Refine analysis result:", repr(analysis))

            # YES branch
            if re.search(r"\b(yes|y|sure|please do)\b", analysis, re.I):
                print("✅  Stage 0 → YES branch")
                confirm_prompt = (
                    "You are an assistant. The user agreed to proceed with this plan:\n"
                    f"{self._pending_plan}\n\n"
                    "Summarize the next steps in one or two sentences."
                )
                print("Confirm prompt >>>\n", confirm_prompt)
                summary = self._stream_and_capture(
                    self.primary_model,
                    [{"role": "system", "content": confirm_prompt}],
                    tag="[Confirm]"
                ).strip()
                print("Confirm summary:", summary)
                self.tts.enqueue(summary)

                # restore saved state & tool-chain
                state         = self._pending_state
                plan_str      = self._pending_plan
                fixed_calls   = self._pending_fixed_calls
                tc_ctx, raw_calls, schemas = self._pending_tool_chain
                print("Restored state keys:", list(state.keys()))
                print("Restored fixed_calls:", fixed_calls)
                print("Restored raw_calls    :", raw_calls)
                print("Restored schemas      :", [s.context_id for s in schemas])

                # clear pending flags
                for attr in ("_awaiting_confirmation","_pending_state","_pending_plan","_pending_fixed_calls","_pending_tool_chain"):
                    if hasattr(self, attr):
                        delattr(self, attr)
                print("Cleared all _pending_* attributes")

                # ——— Stage 9: invoke with retries ——————————————
                print("\n--- Stage 9: Invoke Tools with Retries ---")
                print("Invoking:", raw_calls)
                tool_ctxs = self._stage9_invoke_with_retries(
                    raw_calls, plan_str, schemas, user_text, state['clar_ctx'].metadata
                )
                print("→ tool_ctxs IDs:", [t.context_id for t in tool_ctxs])

                # ——— Stage 9b: reflection & replan ——————————————
                print("\n--- Stage 9b: Reflection & Replan ---")
                replan = self._stage9b_reflection_and_replan(
                    tool_ctxs, plan_str, user_text, state['clar_ctx'].metadata
                )
                print("Reflection outcome:", repr(replan))

                # optional internal replan loop
                if replan and replan.strip().startswith("{"):
                    print("↻  Replanning based on reflection")
                    try:
                        plan_obj = json.loads(replan)
                        fixed_calls = [t["call"] for t in plan_obj.get("tasks", [])]
                        plan_str    = "\n".join(fixed_calls)
                    except json.JSONDecodeError:
                        pass

                    print("New plan_str:", plan_str)
                    tc_ctx, raw_calls, schemas = self._stage8_tool_chaining(
                        state['plan_ctx'], plan_str, state['tools_list']
                    )
                    print("Re-chained raw_calls:", raw_calls)
                    tool_ctxs = self._stage9_invoke_with_retries(
                        raw_calls, plan_str, schemas, user_text, state['clar_ctx'].metadata
                    )
                    print("Re-invoked tool_ctxs:", [t.context_id for t in tool_ctxs])

                # bundle into state for assembly
                state['tc_ctx']    = tc_ctx
                state['tool_ctxs'] = tool_ctxs

                # ——— Stage 10: assemble & infer ——————————————
                print("\n--- Stage 10: Assemble & Infer ---")
                print("State keys:", list(state.keys()))
                draft = self._stage10_assemble_and_infer(user_text, state)
                print("Draft answer:", draft)

                # ——— Stage 10b: critique & safety ——————————————
                print("\n--- Stage 10b: Critique & Safety ---")
                final = self._stage10b_response_critique_and_safety(draft, user_text, tool_ctxs)
                print("Final answer:", final)

                # ——— Stage 11: memory writeback & decay ——————————————
                print("\n--- Stage 11: Memory Writeback & Decay ---")
                self._stage11_memory_writeback(final, tool_ctxs)
                self.memman.decay_and_promote()
                self.tts.enqueue(final[:200])
                return final

            # NO branch
            else:
                print("❌  Stage 0 → NO branch")
                refine_plan_prompt = (
                    "Based on the user's response, refine the current plan to address their concerns.\n"
                    "Old plan:\n" f"{self._pending_plan}\n\n"
                    "User feedback:\n" f"{ans}\n"
                )
                print("RefinePlan prompt >>>\n", refine_plan_prompt)
                new_plan = self._stream_and_capture(
                    self.primary_model,
                    [{"role": "system", "content": refine_plan_prompt}],
                    tag="[RefinePlan]"
                ).strip()
                print("New pending plan:", new_plan)
                self._pending_plan = new_plan
                self.tts.enqueue("I've adjusted the plan based on your feedback.  Here's the new approach:")
                
                # ask for confirmation on refined plan
                ask_confirm = (
                    "Here is the refined plan:\n"
                    f"{new_plan}\n\n"
                    "Does this look good to run? (yes/no)"
                )
                print("AskConfirm prompt >>>\n", ask_confirm)
                question = self._stream_and_capture(
                    self.primary_model,
                    [{"role": "system", "content": ask_confirm}],
                    tag="[AskConfirm]"
                ).strip()
                print("Confirmation question:", question)
                self.tts.enqueue(question)
                return question

        # ——— Stage 1: record input ——————————————
        print("\n--- Stage 1: Record Input ---")
        print("User text:", user_text)
        state: Dict[str, Any] = {}
        state['user_ctx'] = self._stage1_record_input(user_text)
        print("user_ctx.summary:", state['user_ctx'].summary)

        # ——— Stages 2–6: prompts, merge, clarify, knowledge, tools ——
        print("\n--- Stage 2: Load System Prompts ---")
        state['sys_ctx'] = self._stage2_load_system_prompts()
        print("sys_ctx.summary[:200]:", state['sys_ctx'].summary[:200], "…")

        print("\n--- Stage 3: Retrieve & Merge Context ---")
        ctx3 = self._stage3_retrieve_and_merge_context(
            user_text, state['user_ctx'], state['sys_ctx'], extra_ctx=state.get("chat_history", [])
        )
        state.update(ctx3)
        print("ctx3 keys:", list(ctx3.keys()))
        print("recent_ids:", ctx3.get("recent_ids"))

        print("\n--- Stage 4: Intent Clarification ---")
        state['clar_ctx'] = self._stage4_intent_clarification(user_text, state)
        print("clar_ctx.summary:", state['clar_ctx'].summary)

        print("\n--- Stage 5: External Knowledge Retrieval ---")
        state['know_ctx'] = self._stage5_external_knowledge(state['clar_ctx'])
        print("know_ctx.summary[:200]:", state['know_ctx'].summary[:200], "…")

        print("\n--- Stage 6: Prepare Tools ---")
        state['tools_list'] = self._stage6_prepare_tools()
        print("tools_list:", state['tools_list'])

        # ——— Stage 7: planning summary ——————————————
        print("\n--- Stage 7: Planning Summary ---")
        state['plan_ctx'], state['plan_output'] = self._stage7_planning_summary(
            state['clar_ctx'], state['know_ctx'], state['tools_list'], user_text
        )
        print("plan_output:", state['plan_output'])

        # ——— Stage 7b: plan validation ——————————————
        print("\n--- Stage 7b: Plan Validation ---")
        valid, errors, fixed_calls = self._stage7b_plan_validation(
            state['plan_ctx'], state['plan_output'], state['tools_list']
        )
        print("valid:", valid, "errors:", errors, "fixed_calls:", fixed_calls)
        if errors:
            raise RuntimeError(f"Plan validation failed: {errors}")

        plan_str = "\n".join(fixed_calls)
        print("plan_str:", repr(plan_str))

        # ——— Stage 8 & 8.5: chain & stash for confirmation —————————
        if fixed_calls:
            print("\n--- Stage 8: Tool Chaining ---")
            tc_ctx, raw_calls, schemas = self._stage8_tool_chaining(
                state['plan_ctx'], plan_str, state['tools_list']
            )
            print("Chained raw_calls:", raw_calls)
            print("Chained schemas IDs:", [s.context_id for s in schemas])

            ask_confirm = (
                "You are an assistant. Here is the plan I intend to run:\n"
                f"{plan_str}\n\n"
                "Ask the user if they think the plan is ready to go, and explain the plan briefly so they understand the steps that will take place. Very brief and concise! Make sure you correctly label the tools and their order by exact name in the plan."
            )
            print("AskConfirm prompt >>>\n", ask_confirm)
            question = self._stream_and_capture(
                self.primary_model,
                [{"role": "system", "content": ask_confirm}],
                tag="[AskConfirm]"
            ).strip()
            print("Confirmation question:", question)
            self.tts.enqueue(question)

            self._awaiting_confirmation   = True
            self._pending_state            = state
            self._pending_plan             = plan_str
            self._pending_fixed_calls      = fixed_calls
            self._pending_tool_chain       = (tc_ctx, raw_calls, schemas)
            return question

        # ——— Stage 8–9–9b & straight-through execution ——————————————
        print("\n--- No tools to confirm: straight-through execution ---")
        print("Re-entering Stage 8:")
        tc_ctx, raw_calls, schemas = self._stage8_tool_chaining(
            state['plan_ctx'], plan_str, state['tools_list']
        )
        print("RAW CALLS:", raw_calls)
        print("TC CONTEXT:", tc_ctx.context_id)
        print("SCHEMAS:", [s.context_id for s in schemas])

        print("\n--- Stage 9: Invoke Tools with Retries ---")
        tool_ctxs = self._stage9_invoke_with_retries(
            raw_calls, plan_str, schemas, user_text, state['clar_ctx'].metadata
        )
        print("tool_ctxs IDs:", [t.context_id for t in tool_ctxs])

        print("\n--- Stage 9b: Reflection & Replan ---")
        replan = self._stage9b_reflection_and_replan(
            tool_ctxs, plan_str, user_text, state['clar_ctx'].metadata
        )
        print("Reflection result:", repr(replan))

        if replan and replan.strip().startswith("{"):
            print("↻  Replan detected, repeating Stages 8 & 9")
            try:
                plan_obj   = json.loads(replan)
                fixed_calls= [t["call"] for t in plan_obj.get("tasks",[])]
                plan_str   = "\n".join(fixed_calls)
            except json.JSONDecodeError:
                pass
            tc_ctx, raw_calls, schemas = self._stage8_tool_chaining(
                state['plan_ctx'], plan_str, state['tools_list']
            )
            print("Re-chained raw_calls:", raw_calls)
            tool_ctxs = self._stage9_invoke_with_retries(
                raw_calls, plan_str, schemas, user_text, state['clar_ctx'].metadata
            )
            print("Re-invoked tool_ctxs:", [t.context_id for t in tool_ctxs])

        # ——— Stage 10+: assemble, critique, memory, decay ——————————————
        state['tc_ctx']    = tc_ctx
        state['tool_ctxs'] = tool_ctxs

        print("\n--- Stage 10: Assemble & Infer ---")
        draft = self._stage10_assemble_and_infer(user_text, state)
        print("Draft answer:", draft)

        print("\n--- Stage 10b: Critique & Safety ---")
        final = self._stage10b_response_critique_and_safety(draft, user_text, tool_ctxs)
        print("Final answer:", final)

        print("\n--- Stage 11: Memory Writeback & Decay ---")
        self._stage11_memory_writeback(final, tool_ctxs)
        self.memman.decay_and_promote()

        self.tts.enqueue(final)
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
        self,
        user_text: str,
        user_ctx: ContextObject,
        sys_ctx: ContextObject,
        extra_ctx: List[ContextObject] = None
    ) -> Dict[str, Any]:
        """
        Merge together:
          1) the last raw segments (user/assistant),
          2) our tool-driven chat_history,
          3) semantic retrieval,
          4) associative memory,
          5) recent tool outputs.
        """
        from datetime import timedelta
        now, past = default_clock(), default_clock() - timedelta(minutes=self.lookback)
        tr = (past.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%dT%H%M%SZ"))

        # 1) raw user+assistant up to hist_k turns
        all_segs = self.repo.query(lambda c:
            c.domain=="segment" and c.component in ("user_input","assistant")
        )
        all_segs.sort(key=lambda c: c.timestamp)
        history = all_segs[-self.hist_k*2:]  # e.g. last 10 user + 10 assistant

        # 1.5) fold in any explicit chat_history objs
        if extra_ctx:
            history.extend(extra_ctx)

        # 2) semantic recent retrieval (exclude everything except mind-map snippets)
        recent = self.engine.query(
            stage_id="recent_retrieval",
            time_range=tr,
            similarity_to=user_text,
            exclude_tags=self.STAGES
                         + ["tool_schema","tool_output","assistant","system_prompt"],
            top_k=self.top_k
        )

        # 3) associative memory (first pass)
        assoc1 = self.memman.recall([user_ctx.context_id], k=self.top_k)
        for c in assoc1:
            c.record_recall(stage_id="recent_retrieval",
                            coactivated_with=[user_ctx.context_id])
            self.repo.save(c)

        # 4) include recent tool outputs
        tool_outs = self.repo.query(lambda c: c.component=="tool_output")
        tool_outs.sort(key=lambda c: c.timestamp)
        recent_tools = tool_outs[-self.top_k:]

        # merge everything, preserving chronology but deduping
        merged = []
        seen = set()
        for bucket in (history, recent, assoc1, recent_tools):
            for c in bucket:
                if c.context_id not in seen:
                    merged.append(c)
                    seen.add(c.context_id)

        recent_ids = [c.context_id for c in merged]

        # debug print
        self._print_stage_context("recent_retrieval", {
            "system_prompts": [sys_ctx.summary],
            "history":        [f"{c.semantic_label}: {c.summary}" for c in history],
            "semantic":       [f"{c.semantic_label}: {c.summary}" for c in recent],
            "memory":         [f"{c.semantic_label}: {c.summary}" for c in assoc1],
            "tool_outputs":   [f"{c.semantic_label}: {c.summary}" for c in recent_tools],
        })

        return {
            "history":    history,
            "recent":     recent,
            "assoc":      assoc1,
            "recent_ids": recent_ids
        }

    # ── Stage 4: Intent Clarification ─────────────────────────────────────
    def _stage4_intent_clarification(self, user_text: str, state: Dict[str, Any]) -> ContextObject:
        import json, re

        # 1) Build the context block
        pieces = [state['sys_ctx']] + state['history'] + state['recent'] + state['assoc']
        block = "\n".join(f"[{c.semantic_label}] {c.summary}" for c in pieces)

        # 2) Force valid JSON out of the clarifier
        clarifier_system = self.clarifier_prompt  # “Output only valid JSON…”
        msgs = [
            {"role": "system", "content": clarifier_system},
            {"role": "system", "content": f"Context:\n{block}"},
            {"role": "user",   "content": user_text},
        ]

        out = self._stream_and_capture(self.secondary_model, msgs, tag="[Clarifier]")
        # retry once if it isn’t JSON
        for attempt in (1, 2):
            try:
                clar = json.loads(out)
                break
            except:
                if attempt == 1:
                    retry_sys = "⚠️ Your last response wasn’t valid JSON.  Please output only JSON with keys `keywords` and `notes`."
                    out = self._stream_and_capture(
                        self.secondary_model,
                        [
                            {"role": "system", "content": retry_sys},
                            {"role": "user",   "content": out}
                        ],
                        tag="[ClarifierRetry]"
                    )
                else:
                    # give up: empty intent
                    clar = {"keywords": [], "notes": out}

        # 3) Build the ContextObject (summary = notes)
        ctx = ContextObject.make_stage(
            "intent_clarification",
            [state['user_ctx'].context_id, state['sys_ctx'].context_id] + state['recent_ids'],
            clar
        )
        ctx.stage_id = "intent_clarification"
        ctx.summary  = clar.get("notes", "")
        ctx.touch(); self.repo.save(ctx)
        return ctx

    def _stage5_external_knowledge(self, clar_ctx: ContextObject) -> ContextObject:
        import json

        snippets: List[str] = []

        # ——— 1) Original external Web/snippet lookup ——————————
        for kw in clar_ctx.metadata.get("keywords", []):
            hits = self.engine.query(
                stage_id="external_knowledge_retrieval",
                similarity_to=kw,
                top_k=self.top_k
            )
            snippets.extend(h.summary for h in hits)

        # ——— 2) Local memory lookup via your new context_query tool ——
        # Build a time-window covering, say, the last hour:
        now = default_clock()
        start = (now - timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ")
        end   = now.strftime("%Y%m%dT%H%M%SZ")

        for kw in clar_ctx.metadata.get("keywords", []):
            raw = Tools.context_query(
                time_range=[start, end],
                query=kw,
                top_k=self.top_k
            )
            try:
                results = json.loads(raw).get("results", [])
                for r in results:
                    # r["summary"] holds the ContextObject.summary
                    snippets.append(f"(MEM) {r.get('summary','')}")
                # avoid duplicates
                snippets = list(dict.fromkeys(snippets))
            except json.JSONDecodeError:
                # if something goes wrong, skip local-memory for this keyword
                continue

        # ——— 3) Persist and return as before ————————————————
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
        """
        Return a de-duplicated list of tool schemas:

        * Only the most-recent row per tool-name is used.
        * Output order is stable (lexicographic by tool name).
        """
        import json

        # 1️⃣  Fetch **all** schema rows
        rows = self.repo.query(
            lambda c: c.component == "schema" and "tool_schema" in c.tags
        )

        # 2️⃣  Bucket rows by tool name, keep the newest per bucket
        buckets: dict[str, ContextObject] = {}
        for r in rows:
            try:
                name = json.loads(r.metadata["schema"])["name"]
            except Exception:
                continue
            if name not in buckets or r.timestamp > buckets[name].timestamp:
                buckets[name] = r

        # 3️⃣  Build the final list, sorted for reproducibility
        tool_defs: list[dict[str, str]] = []
        for name in sorted(buckets):
            data = json.loads(buckets[name].metadata["schema"])
            tool_defs.append({
                "name":        data["name"],
                "description": data.get("description", "").split("\n", 1)[0],
            })

        return tool_defs
    
    
    # ── Stage 7: Planning Summary with Tool-Existence Check ────────────────────────────────────
    def _stage7_planning_summary(
        self,
        clar_ctx: ContextObject,
        know_ctx: ContextObject,
        tools_list: List[Dict[str, str]],
        user_text: str,
    ) -> Tuple[ContextObject, str]:
        import json, re, inspect

        # —— helper to strip markdown fences —— 
        def _clean(text: str) -> str:
            t = text.strip()
            # first try to pull out a ```json { ... } ``` block
            m = re.search(r"```json\s*(\{.*?\})\s*```", t, flags=re.S)
            if m:
                return m.group(1).strip()
            # otherwise remove any leading/trailing ``` fences
            t = re.sub(r"^```(?:json)?\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
            return t.strip()

        # —— build your base prompts —— 
        tools_md = "\n".join(f"- **{t['name']}**: {t['description']}"
                             for t in tools_list)
        base_system = (
            "You are the Planner.  Emit **only** a JSON object matching:\n\n"
            "{ \"tasks\": [ { \"call\": \"tool_name\", "
            "\"tool_input\": { /* any named parameters */ }, \"subtasks\": [] }, … ] }\n\n"
            "If you cannot, just list the tool calls.  Available tools:\n"
            f"{tools_md}"
        )
        base_user = (
            f"User question:\n{user_text}\n\n"
            f"Clarified intent:\n{clar_ctx.summary}\n\n"
            f"Snippets:\n{know_ctx.summary or '(none)'}"
        )

        self._print_stage_context("planning_summary:init", {
            "system": [base_system],
            "input":  [base_user],
        })

        plan_obj = None
        cleaned   = ""
        raw       = ""

        # —— up to 3 attempts to eliminate unknown tools —— 
        for attempt in range(1, 4):
            tag = "[Planner]" if attempt == 1 else "[PlannerReplan]"
            if attempt == 1:
                msgs = [
                    {"role": "system", "content": base_system},
                    {"role": "user",   "content": base_user},
                ]
            else:
                msgs = [
                    {"role": "system", "content":
                        "Your previous plan included unknown tools.  "
                        "Please replan using only valid tools."},
                    {"role": "user",   "content": cleaned},
                ]

            raw = self._stream_and_capture(self.secondary_model, msgs, tag=tag)
            cleaned = _clean(raw)

            # try parsing JSON
            try:
                cand = json.loads(cleaned)
                if isinstance(cand, dict) and "tasks" in cand:
                    plan_obj = cand
                else:
                    plan_obj = None
            except json.JSONDecodeError:
                plan_obj = None

            # fallback to regex-only extraction
            if plan_obj is None:
                calls = re.findall(r'\b[A-Za-z_]\w*\([^)]*\)', raw)
                plan_obj = {
                    "tasks": [
                        {"call": c, "tool_input": {}, "subtasks": []}
                        for c in calls
                    ]
                }

            # check against our actual list of tools
            available = {t["name"] for t in tools_list}
            unknown   = [
                task["call"]
                for task in plan_obj["tasks"]
                if task["call"] not in available
            ]
            if not unknown:
                # success: no unknown calls
                break

            # log unknown/tools for next iteration
            self._print_stage_context(
                f"planning_summary:unknown_tools(attempt={attempt})", {
                    "unknown":   unknown,
                    "available": sorted(available),
                }
            )
        else:
            # after 3 failed replans, give up and proceed with last plan_obj
            self._print_stage_context(
                "planning_summary:gave_up_replanning", {"final_plan": plan_obj}
            )

        # —— build the final call strings —— 
        call_strings: List[str] = []
        for task in plan_obj["tasks"]:
            name   = task.get("call")
            params = task.get("tool_input", {}) or task.get("params", {}) or {}
            if params:
                arg_list = [
                    f"{k}={json.dumps(v, ensure_ascii=False)}"
                    for k, v in params.items()
                ]
                call_strings.append(f"{name}({','.join(arg_list)})")
            else:
                call_strings.append(f"{name}()")

        # —— persist the plan context —— 
        ctx = ContextObject.make_stage(
            "planning_summary",
            clar_ctx.references + know_ctx.references,
            {"plan": plan_obj}
        )
        ctx.stage_id = "planning_summary"
        ctx.summary  = json.dumps({
            "tasks": [{"call": s, "subtasks": []} for s in call_strings]
        })
        ctx.touch()
        self.repo.save(ctx)

        # —— RL signals —— 
        if call_strings:
            ctx_s = ContextObject.make_success(
                f"Planner → {len(call_strings)} task(s)",
                refs=[ctx.context_id]
            )
        else:
            ctx_s = ContextObject.make_failure(
                "Planner → empty plan",
                refs=[ctx.context_id]
            )
        ctx_s.touch()
        self.repo.save(ctx_s)

        # —— return for downstream chaining —— 
        return ctx, json.dumps({
            "tasks": [{"call": s, "subtasks": []} for s in call_strings]
        })


    def _stage7b_plan_validation(
        self,
        plan_ctx: ContextObject,
        plan_output: str,
        tools_list: List[Dict[str, str]],
    ) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        """
        Robust JSON-based plan validation. Keeps original tool_input intact,
        asks the LLM to fill in only truly missing parameters, and then
        reconstructs call-strings with full, exact arguments.
        """
        import json, re

        # A) Load all known tool schemas
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
            # fallback to regex-only if the planner wasn’t JSON
            calls = re.findall(r'\b[A-Za-z_]\w*\([^)]*\)', plan_output)
            return calls, [], calls

        # C) Up to 3 repair passes: fill any missing required params
        for _ in range(3):
            missing = {}
            for idx, task in enumerate(tasks):
                name       = task.get("call")
                schema     = all_schemas.get(name)
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

            # build a prompt for the LLM to repair only the missing fields
            prompt = {
                "description": "Some tool calls are missing required arguments.",
                "missing":     missing,
                "plan":        plan_obj,
                "schemas":     {n: all_schemas[n] for n in all_schemas}
            }
            repair = self._stream_and_capture(
                self.secondary_model,
                [
                  {"role":"system","content":
                    "Return ONLY a JSON `{'tasks':[…]}` with each task’s `tool_input` now complete."},
                  {"role":"user","content": json.dumps(prompt)}
                ],
                tag="[PlanFix]"
            ).strip()

            try:
                plan_obj = json.loads(repair)
                tasks    = plan_obj.get("tasks", [])
            except:
                break

        # D) Re-serialize every task into a real, exact call string
        fixed_calls = []
        for task in tasks:
            name = task["call"]
            ti   = task.get("tool_input", {}) or {}
            args = ",".join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k,v in ti.items())
            fixed_calls.append(f"{name}({args})")

        # E) Persist the validation step
        meta = {
            "valid":       fixed_calls,
            "errors":      [],  # we never hard-fail
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
        tools_list: List[Dict[str, str]]
    ) -> Tuple[ContextObject, List[str], List[ContextObject]]:
        import json, re

        # ── A) Extract raw calls from the plan ──────────────────────────────
        text = plan_output
        parsed = Tools.parse_tool_call(text)

        print("PARSED TOOL CALL")
        print(parsed)
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
        clar_metadata: Dict[str, Any],
        max_tokens: int = 128000   # no longer used for trimming
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
        system_msg = (
            "You are the Reflection agent.  Please review **all** of the following "
            "context, including the user question, clarifier notes, every tool output, "
            "and the original plan.  Decide whether the plan execution satisfied the user's intent.  "
            "If yes, reply exactly `OK`.  Otherwise, reply **only** with the corrected JSON plan."
        )
        user_payload = (
            context_blob
            + "\n\n"
            + "Did these tool outputs satisfy the original intent? "
              "If yes, reply OK.  If not, return only the corrected JSON plan."
        )

        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_payload},
        ]
        resp = self._stream_and_capture(self.secondary_model, msgs, tag="[Reflection]").strip()

        # 6) If it’s literally “OK”, we keep the original outputs
        if re.fullmatch(r"(?i)(ok|okay)[.!]?", resp):
            ok_ctx = ContextObject.make_success(
                description="Reflection confirmed plan satisfied intent",
                refs=[c.context_id for c in tool_ctxs]
            )
            ok_ctx.touch()
            self.repo.save(ok_ctx)
            return None

        # 7) Otherwise record the new plan
        fail_ctx = ContextObject.make_failure(
            description="Reflection triggered replan",
            refs=[c.context_id for c in tool_ctxs]
        )
        fail_ctx.touch()
        self.repo.save(fail_ctx)

        repl = ContextObject.make_stage(
            "reflection_and_replan",
            [c.context_id for c in tool_ctxs],
            {"replan": resp}
        )
        repl.stage_id = "reflection_and_replan"
        repl.summary  = resp  # full JSON plan
        repl.touch()
        self.repo.save(repl)

        return resp
    
    def _stage9_invoke_with_retries(
        self,
        raw_calls: List[str],
        plan_output: str,
        selected_schemas: List[ContextObject],
        user_text: str,
        clar_metadata: Dict[str, Any]
    ) -> List[ContextObject]:
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

        def _validate(res: Dict[str, Any]) -> Tuple[bool, str]:
            exc = res.get("exception")
            return (exc is None, exc or "")

        # ── INITIALIZE ─────────────────────────────────────────────────────────
        raw_calls = _norm(raw_calls)
        call_status: Dict[str, bool] = {c: False for c in raw_calls}
        tool_ctxs: List[ContextObject] = []
        last_results: Dict[str, Any] = {}
        max_retries = 10

        print("\n--- Stage 9: invoke_with_retries ---")
        print("Initial raw_calls:", raw_calls)
        print("Plan output:\n", plan_output)

        # Helper for matching [PLACEHOLDER]
        def normalize_key(k: str) -> str:
            return re.sub(r'\W+', '', k).lower()

        # ── RETRY LOOP ──────────────────────────────────────────────────────────
        for attempt in range(1, max_retries + 1):
            errors: List[Tuple[str, str]] = []
            print(f"\n>>> Attempt {attempt}/{max_retries}")

            for original_call in list(raw_calls):
                if call_status[original_call]:
                    continue

                # — Step A: placeholder substitution [Foo] → last_results["Foo"]
                call_str = original_call
                for ph in re.findall(r'\[([^\]]+)\]', call_str):
                    norm = normalize_key(ph)
                    match = next(
                        (k for k in last_results if normalize_key(k).find(norm) != -1),
                        None
                    )
                    if match:
                        val = last_results[match]
                        print(f" Substituting [{ph}] with {val!r} from '{match}'")
                        call_str = call_str.replace(f"[{ph}]", repr(val))
                    else:
                        print(f" ⚠️ No upstream value for placeholder [{ph}]")

                # — Step B: nested inner calls and ${tool}_output
                # 1) substitute ${tool}_output
                for name, val in last_results.items():
                    call_str = call_str.replace(f"${{{name}_output}}", repr(val))

                # 2) detect bare nested calls foo() inside our call_str
                for inner in re.findall(r'\b([A-Za-z_]\w*)\(\)', call_str):
                    if inner not in last_results:
                        inner_call = f"{inner}()"
                        print(f"[NestedInvocation] Running inner: {inner_call}")
                        res_i = Tools.run_tool_once(inner_call)
                        ok_i, err_i = _validate(res_i)
                        last_results[inner] = res_i["output"]
                        print(f"[NestedInvocation] {'OK' if ok_i else 'ERROR:' + err_i}")

                        # persist nested result
                        sch_i = next(
                            sch for sch in selected_schemas
                            if json.loads(sch.metadata["schema"])["name"] == inner
                        )
                        in_ctx = ContextObject.make_stage(
                            "tool_output",
                            [sch_i.context_id],
                            res_i
                        )
                        in_ctx.stage_id = f"tool_output_{attempt}_inner_{inner}"
                        in_ctx.summary  = (
                            str(res_i["output"])
                            if ok_i else f"ERROR: {err_i}"
                        )
                        in_ctx.metadata.update(res_i)
                        in_ctx.touch(); self.repo.save(in_ctx)
                        tool_ctxs.append(in_ctx)

                # — Step C: primary call
                print(f"[ToolInvocation] Running: {call_str}")
                res = Tools.run_tool_once(call_str)
                ok, err = _validate(res)
                status = "OK" if ok else f"ERROR: {err.splitlines()[-1]}"
                print(f"[ToolInvocation] {status}")

                # capture for future substitution
                tool_name = original_call.split("(", 1)[0]
                last_results[tool_name] = res["output"]

                # persist this invocation
                sch = next(
                    sch for sch in selected_schemas
                    if json.loads(sch.metadata["schema"])["name"] == tool_name
                )
                out_ctx = ContextObject.make_stage(
                    "tool_output",
                    [sch.context_id],
                    res
                )
                out_ctx.stage_id = f"tool_output_{attempt}_{tool_name}"
                out_ctx.summary  = (
                    str(res["output"]) if ok else f"ERROR: {err.splitlines()[-1]}"
                )
                out_ctx.metadata.update(res)
                out_ctx.touch(); self.repo.save(out_ctx)
                tool_ctxs.append(out_ctx)

                call_status[original_call] = ok
                if not ok:
                    errors.append((original_call, err))

            if not errors:
                print("All calls succeeded.")
                break

            # ── PREPARE RETRY LLM PROMPT ─────────────────────────────────────────
            err_lines   = [f"{c} → {e.splitlines()[-1]}" for c,e in errors]
            failed_calls = [c for c,_ in errors]
            print("Failures:", err_lines)

            # collect schemas for the failures
            docs = []
            for sch in selected_schemas:
                data = json.loads(sch.metadata["schema"])
                if data["name"] in {c.split("(")[0] for c in failed_calls}:
                    docs.append(f"• **{data['name']}** parameters:\n"
                                + json.dumps(data.get("parameters", {}), indent=2))

            # available variables block
            vars_block = "\n".join(f"{k}_output = {v!r}" for k,v in last_results.items())

            retry_sys = (
                "Some tool calls failed:\n"
                + "\n".join(err_lines)
                + "\n\n**Original query:**\n"    + user_text
                + "\n\n**Clarification notes:**\n" + clar_metadata.get("notes","(none)")
                + "\n\n**Plan:**\n"              + plan_output
                + (f"\n\n**Available variables:**\n{vars_block}" if vars_block else "")
                + "\n\n**Schemas for failing tools:**\n"
                + "\n\n".join(docs)
                + "\n\nPlease return ONLY JSON: {\"tool_calls\":[\"fixed_call1(...)\", …]}"
            )
            self._print_stage_context("tool_chaining_retry", {
                "errors":    err_lines,
                "variables": vars_block.split("\n") if vars_block else [],
                "fail_docs": docs,
            })

            retry_msgs = [
                {"role":"system",  "content": retry_sys},
                {"role":"user",    "content": json.dumps({"tool_calls": failed_calls})}
            ]
            retry_out = self._stream_and_capture(
                self.secondary_model, retry_msgs, tag="[ToolChainRetry]"
            ).strip()

            # parse corrected calls
            try:
                fixed = json.loads(retry_out)["tool_calls"]
            except:
                parsed = Tools.parse_tool_call(retry_out)
                fixed = parsed if isinstance(parsed, list) else ([parsed] if parsed else failed_calls)

            fixed = _norm(fixed) or failed_calls

            # merge corrections back
            new_raw = []
            fix_iter = iter(fixed)
            for c in raw_calls:
                if c in failed_calls:
                    new_raw.append(next(fix_iter, c))
                else:
                    new_raw.append(c)
            raw_calls = new_raw
            call_status = {c: False for c in raw_calls}

        return tool_ctxs

    def _stage10_assemble_and_infer(self, user_text: str, state: Dict[str, Any]) -> str:
        import json

        # 1) Gather all context IDs including tool outputs
        refs = [
            state['user_ctx'].context_id,
            state['sys_ctx'].context_id,
            *state['recent_ids'],
            state['clar_ctx'].context_id,
            state['know_ctx'].context_id,
            state['plan_ctx'].context_id,
            *[t.context_id for t in state['tool_ctxs']],
        ]

        # 2) De-dup & load them, sorted chronologically
        seen, ordered = set(), []
        for cid in refs:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        ctxs = [self.repo.get(cid) for cid in ordered]
        ctxs.sort(key=lambda c: c.timestamp)

        print(f"[assemble_and_infer] total context objects: {len(ctxs)}")

        # 3) Build a rich inlined context block
        interm_parts = []
        for c in ctxs:
            if c.semantic_label == "tool_output":
                out = c.metadata.get("output")
                try:
                    blob = json.dumps(out, indent=2, ensure_ascii=False)
                except Exception:
                    blob = repr(out)
                interm_parts.append(f"[{c.stage_id} (tool output)]\n{blob}")
            else:
                interm_parts.append(f"[{c.stage_id}] {c.summary}")
        interm = "\n\n".join(interm_parts)

        # Debug dump
        self._print_stage_context("assemble_and_infer", {
            "user_question":   [user_text],
            "plan":            [state.get('plan_output', '(no plan)')],
            "inference_prompt": [self.inference_prompt],
            "inlined_context": interm_parts,
        })

        # 4) Build LLM prompt
        final_sys = (
            "You are the Assembler.  Combine the user question, the plan, "
            "and all provided context/tool outputs into one concise, factual answer.  "
            "Do NOT hallucinate or invent new details."
        )
        plan_text = state.get('plan_output', '(no plan)')
        # stash for critique stage
        self._last_plan_output = plan_text

        msgs = [
            {"role": "system", "content": final_sys},
            {"role": "system", "content": f"User question:\n{user_text}"},
            {"role": "system", "content": f"Plan:\n{plan_text}"},
            {"role": "system", "content": self.inference_prompt},
            {"role": "system", "content": interm},
            {"role": "user",   "content": user_text},
        ]

        reply = self._stream_and_capture(self.primary_model, msgs, tag="[Assistant]").strip()

        # 5) Persist final answer
        resp_ctx = ContextObject.make_stage(
            "final_inference",
            [state['tc_ctx'].context_id] + [t.context_id for t in state['tool_ctxs']],
            {"text": reply}
        )
        resp_ctx.stage_id = "final_inference"
        resp_ctx.summary  = reply
        resp_ctx.touch(); self.repo.save(resp_ctx)

        return reply


    def _stage10b_response_critique_and_safety(
        self,
        draft: str,
        user_text: str,
        tool_ctxs: List[ContextObject]
    ) -> str:
        import json, difflib

        # Recover the plan
        plan_text = getattr(self, "_last_plan_output", "(no plan)")

        # 1) Collect all tool outputs
        outputs = []
        for c in tool_ctxs:
            out = c.metadata.get("output")
            try:
                blob = json.dumps(out, indent=2, ensure_ascii=False)
            except Exception:
                blob = repr(out)
            outputs.append(f"[{c.stage_id}]\n{blob}")

        print(f"[response_critique] tool output chunks: {len(outputs)}, draft length: {len(draft)} chars")

        # 2) Build richer critic prompt
        critic_sys = (
            "You are the final-pass assistant.  You will get:\n"
            "  • The original user question\n"
            "  • The plan that was run\n"
            "  • The draft answer\n"
            "  • All raw tool outputs\n\n"
            "Your task:\n"
            "  1) Remove any unsupported claims or hallucinations.\n"
            "  2) Ground everything in the tool outputs and context.\n"
            "  3) Observe policy/tone.\n"
            "  4) Return ONLY the polished, concise answer."
        )
        prompt_user = "\n\n".join([
            f"User question:\n{user_text}",
            f"Plan:\n{plan_text}",
            f"Draft answer:\n{draft}",
            "Tool outputs:\n" + "\n\n".join(outputs),
        ])

        # Debug dump
        self._print_stage_context("response_critique", {
            "system_directives": [critic_sys],
            "user_payload":      [prompt_user],
        })

        # 3) Invoke LLM
        msgs = [
            {"role": "system", "content": critic_sys},
            {"role": "user",   "content": prompt_user},
        ]
        polished = self._stream_and_capture(self.secondary_model, msgs, tag="[Critic]").strip()

        # 4) If unchanged, return draft
        if polished == draft.strip():
            return polished

        # 5) Compute one-line diff
        diff = difflib.unified_diff(
            draft.splitlines(), polished.splitlines(),
            lineterm="", n=1
        )
        diff_summary = "; ".join(
            ln for ln in diff if ln.startswith(("+ ", "- "))
        ) or "(minor re-formatting)"

        # 6) Upsert dynamic_prompt_patch
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

        # 7) Persist polished reply
        ctx = ContextObject.make_stage(
            "response_critique",
            [],  # no extra refs
            {"text": polished}
        )
        ctx.stage_id = "response_critique"
        ctx.summary = polished
        ctx.touch(); self.repo.save(ctx)

        return polished


    # ----------------------------------------------------------------------
    # 2.  STAGE-11  –  memory write-back (singleton style, no duplicates)
    # ----------------------------------------------------------------------
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
