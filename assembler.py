#!/usr/bin/env python3
"""
assembler.py — Stage-driven pipeline with full observability and
dynamic, chronological context windows per stage.
"""
import json
import logging
import re
import os, json, random, math, textwrap
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
from datetime import datetime
from collections import deque
import inspect
import threading
import ast, json, re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
def _canon(call: str) -> str:
    """Return a canonical signature for a tool call (idempotent)."""
    name, _ = call.split("(", 1)
    tree = ast.parse(call.strip())
    node = tree.body[0].value                     # type: ignore[arg-type]
    pos = [ast.get_source_segment(call, a).strip() for a in node.args]
    kw  = {k.arg: ast.get_source_segment(call, k.value).strip()
           for k in node.keywords
           if ast.get_source_segment(call, k.value).strip() not in ("''", '""', 'None')}
    sig = name.strip() + "("
    sig += ",".join(pos)
    if kw:
        sig += "," if pos else ""
        sig += ",".join(f"{k}={v}" for k, v in sorted(kw.items()))
    sig += ")"
    return sig

@lru_cache(maxsize=None)
def _done_calls(repo) -> set[str]:
    """Any *successful* canonical signatures stored in the context log."""
    done: set[str] = set()
    for obj in repo.query(lambda c: c.component == "tool_output"):
        # success is recorded via metadata["ok"] we add below
        if obj.metadata.get("ok"):
            done.add(obj.metadata["tool_call"])
    return done

class RLController:
    """
    Multi-armed bandit with baseline + recall bias.
    Q[s]: estimated reward for stage s
    R_bar: global baseline
    Each optional stage also has a gamma parameter that
    amplifies the signal from context-recall frequency.
    """
    def __init__(self,
                 stages: List[str],
                 alpha: float = 0.1,
                 beta:  float = 0.01,
                 gamma: float = 0.1,
                 path:  str   = "rl.weights"):
        self.alpha = alpha     # LR for Q
        self.beta  = beta      # LR for baseline
        self.gamma = gamma     # weight on recall_feature
        self.path  = path

        if os.path.exists(path):
            data = json.load(open(path))
        else:
            data = {}

        self.Q     = {s: data.get("Q",{}).get(s, 0.0) for s in stages}
        self.N     = {s: data.get("N",{}).get(s, 0)   for s in stages}
        self.R_bar = data.get("R_bar", 0.0)

    def probability(self, stage: str, recall_feat: float = 0.0) -> float:
        # advantage plus recall_bias
        adv = self.Q[stage] - self.R_bar + self.gamma * recall_feat
        return 1.0 / (1.0 + math.exp(-adv))

    def should_run(self, stage: str, recall_feat: float = 0.0) -> bool:
        return random.random() < self.probability(stage, recall_feat)

    def update(self, included: List[str], reward: float):
        # update baseline
        self.R_bar += self.beta * (reward - self.R_bar)
        # update each included stage
        for s in included:
            self.N[s] += 1
            lr = self.alpha / math.sqrt(self.N[s])
            self.Q[s] += lr * (reward - self.Q[s])
        self.save()

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "Q":     self.Q,
                "N":     self.N,
                "R_bar": self.R_bar
            }, f, indent=2)
        os.replace(tmp, self.path)

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

    def __init__(
        self,
        context_path:     str = "context.jsonl",
        config_path:      str = "config.json",
        lookback_minutes: int = 60,
        top_k:            int = 10,
        tts_manager:      TTSManager | None = None,
    ):
        # — load or init config —
        try:
            self.cfg = json.load(open(config_path))
        except FileNotFoundError:
            self.cfg = {}

        # — models & lookback settings —
        self.primary_model   = self.cfg.get("primary_model",   "gemma3:4b")
        self.secondary_model = self.cfg.get("secondary_model", self.primary_model)
        self.lookback        = self.cfg.get("lookback_minutes", lookback_minutes)
        self.top_k           = self.cfg.get("top_k",            top_k)
        self.hist_k          = self.cfg.get("history_turns",    5)

        # — system & stage prompts —
        self.clarifier_prompt = self.cfg.get(
            "clarifier_prompt",
            "You are Clarifier. Expand the user’s intent into a JSON object with "
            "two keys: 'keywords' (an array of concise keywords) and 'notes'. "
            "Your notes must be highly detailed and context aware. Do not provide judgements beyond scope of exact context present, or make assumptions beyond what is absolutely known. "
            "Output only valid JSON."
        )
        self.assembler_prompt = self.cfg.get(
            "assembler_prompt",
            "You are Assembler. Distill context into a concise summary, but do not omit implied content which is needed for effective evaluation."
        )
        self.inference_prompt = self.cfg.get(
            "inference_prompt",
            "You are helpful and highly detailed in your context-aware assessment. Use all provided snippets and tool outputs to inform your reply, abide by internal instruction present and distill coherent and verbose responses based on contextual understanding and intention."
        )
        self.planning_prompt = self.cfg.get(
            "planning_prompt",
            "You are the Planner.  Emit **only** a JSON object matching:\n\n"
            "{ \"tasks\": [ { \"call\": \"tool_name\", \"tool_input\": { /* named params */ }, \"subtasks\": [] }, … ] }\n\n"
            "If you cannot, just list the tool calls.  Available tools:\n"
        )
        self.toolchain_prompt = self.cfg.get(
            "toolchain_prompt",
            "You have these tools (full JSON schemas shown).\n"
            "I will send you exactly one JSON object with key \"tool_calls\".\n"
            "YOUR ONLY JOB is to return back that same JSON object (modifying only calls that violate the schema).\n"
            "When building each call, you **must** inline any placeholder of the form [alias from full_function_name] or {{alias}} by substituting the actual value returned by that tool in previous steps.\n"
            "Do NOT add, remove, or simulate any outputs or internal state.\n"
            "Reply with exactly one JSON object and nothing else:\n\n"
            "{\"tool_calls\": [\"tool1(arg1=...,arg2=...)\", ...]}\n\n"
        )
        self.reflection_prompt = self.cfg.get(
            "reflection_prompt",
            "You are the Reflection agent.  Please review **all** of the following "
            "context, including the user question, clarifier notes, every tool output, "
            "and the original plan.  Decide whether the plan execution satisfied the user's intent.  "
            "If yes, reply exactly `OK`.  Otherwise, reply **only** with the corrected JSON plan."
        )
        self.toolchain_retry_prompt = self.cfg.get(
            "toolchain_retry_prompt",
            "Some tool calls failed.  Return ONLY JSON {\"tool_calls\":[\"fixed_call(...)\", …]}."
        )
        self.final_inference_prompt = self.cfg.get(
            "final_inference_prompt",
            "You are the Assembler.  Combine the user question, the plan, and all "
            "provided context/tool outputs into one concise, factual answer.  "
            "Do NOT hallucinate or invent new details."
        )
        self.critic_prompt = self.cfg.get(
            "critic_prompt",
            "You have an array of content at your disposal which must be analyzed and responded to accordingly.  "
            "You’ll see:\n"
            "  • Their original question\n"
            "  • The plan of tool calls we ran\n"
            "  • Your initial draft reply\n"
            "  • All raw tool outputs\n\n"
            "Make it engaging and human-sounding—feel free to vary phrasing and inject warmth—"
            "but be accurate and grounded in those outputs.  Return just the final polished answer."
        )
        self.narrative_mull_prompt = self.cfg.get(
            "narrative_mull_prompt",
            "You are an autonomous meta-reasoner reflecting on your own "
            "narrative and system prompts.\n\n"
            "List up to three improvement areas.  For each, supply:\n"
            '  { "area": "<short-name>", "question": "<self-reflection question>" }\n'
            "Return ONLY valid JSON of the form:\n"
            '{ "issues": [ … ] }'
        )


        # — persist defaults back to config file if they changed —
        defaults = {
            "primary_model":    self.primary_model,
            "secondary_model":  self.secondary_model,
            "lookback_minutes": self.lookback,
            "top_k":            self.top_k,
            "history_turns":    self.hist_k
        }
        if defaults != self.cfg:
            json.dump(defaults, open(config_path, "w"), indent=2)

        # — init context store & memory manager —
        self.repo   = ContextRepository(context_path)
        self.memman = MemoryManager(self.repo)

        # — setup embedding engine —
        @staticmethod
        def embed_text(text: str) -> np.ndarray:
            try:
                resp = embed(model="nomic-embed-text", input=text)
                vec  = np.array(resp["embeddings"], dtype=float).flatten()
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except:
                return np.zeros(768, dtype=float)

        self.engine = ContextQueryEngine(self.repo, lambda t: embed_text(t))

        # — seed all schemas & static prompts —
        self._seed_tool_schemas()
        self._seed_static_prompts()

        # — text-to-speech manager —
        self.tts = tts_manager
        self._chat_contexts: set[int] = set()
        self._telegram_bot = None  # will be set by telegram_input

        # — auto-discover any _stage_<name>() methods as “optional” —
        all_methods = {name for name, _ in inspect.getmembers(self, inspect.ismethod)}

        # include core STAGES + all three meta-stages
        discovered = [
            s for s in self.STAGES
                + ["curiosity_probe", "system_prompt_refine", "narrative_mull"]
            if f"_stage_{s}" in all_methods
        ]

        # allow config override, else use discovered
        self._optional_stages = self.cfg.get("rl_optional", discovered)

        # on-disk RL controller now knows about your meta-stages too
        self.rl = RLController(
            stages=self._optional_stages,
            alpha=self.cfg.get("rl_alpha", 0.05),
            path=self.cfg.get("rl_weights_path", "weights.rl")
        )

        # — seed & load “curiosity” templates from the repo —
        self.curiosity_templates = self.repo.query(
            lambda c: c.component=="policy"
                      and c.semantic_label.startswith("curiosity_template")
        )
        if not self.curiosity_templates:
            defaults: dict[str, str] = {
                "curiosity_template_missing_notes": (
                    "I’m not quite sure what you meant by: «{snippet}». "
                    "Could you clarify?"
                ),
                "curiosity_template_missing_date": (
                    "You mentioned a date but didn’t specify which one—"
                    "what date are you thinking of?"
                ),
                "curiosity_template_auto_mull": (
                    "I’m reflecting on your request. Here’s something I’m still "
                    "unsure about: «{snippet}». Thoughts?"
                ),
            }
            for label, text in defaults.items():
                tmpl = ContextObject.make_policy(
                    label=label,
                    policy_text=text,
                    tags=["dynamic_prompt","curiosity_template"]
                )
                tmpl.touch(); self.repo.save(tmpl)
                self.curiosity_templates.append(tmpl)

        # auto‐generate “requires X” templates if missing
        for name, fn in inspect.getmembers(self, inspect.ismethod):
            if name.startswith("_stage_"):
                doc = fn.__doc__ or ""
                for hint in re.findall(r"requires\s+(\w+)", doc, flags=re.I):
                    label = f"curiosity_require_{hint.lower()}"
                    if not any(t.semantic_label == label for t in self.curiosity_templates):
                        text = (
                            f"It looks like stage `{name}` requires `{hint}`—"
                            " could you clarify?"
                        )
                        tmpl = ContextObject.make_policy(
                            label=label,
                            policy_text=text,
                            tags=["dynamic_prompt","curiosity_template"]
                        )
                        tmpl.touch()
                        self.repo.save(tmpl)
                        self.curiosity_templates.append(tmpl)

        # — RLController for curiosity-template selection —
        self.curiosity_rl = RLController(
            stages=[t.semantic_label for t in self.curiosity_templates],
            alpha=self.cfg.get("curiosity_alpha", 0.1),
            path=self.cfg.get("curiosity_weights_path", "curiosity_weights.rl")
        )


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
            "clarifier_prompt":       self.clarifier_prompt,
            "assembler_prompt":       self.assembler_prompt,
            "inference_prompt":       self.inference_prompt,
            "planning_prompt":        self.planning_prompt,
            "toolchain_prompt":       self.toolchain_prompt,
            "reflection_prompt":      self.reflection_prompt,
            "toolchain_retry_prompt": self.toolchain_retry_prompt,
            "final_inference_prompt": self.final_inference_prompt,
            "critic_prompt":          self.critic_prompt,
            "narrative_mull_prompt":  self.narrative_mull_prompt,
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


    def _should_ask_confirmation(self, state: Dict[str, Any]) -> bool:
        """
        Ask the LLM if we need to show the plan to the user before running.
        Returns True if it replies 'yes', False otherwise.
        """
        import re, json

        calls = state.get("fixed_calls", [])
        # build a one-line summary of the recent context
        ctx_summ = " | ".join(
            f"{c.stage_id or c.semantic_label}: {c.summary[:40].replace(chr(10), ' ')}"
            for c in [
                state.get("user_ctx"),
                state.get("clar_ctx"),
                state.get("know_ctx"),
            ]
            if c
        )
        prompt = {
            "plan": calls,
            "context_summary": ctx_summ
        }
        system = (
            "You are a meta‐reasoner.  Given the plan (list of tool calls) "
            "and a brief context summary, decide whether you need explicit user "
            "confirmation before running the plan.  Answer only 'yes' or 'no'."
        )
        out = self._stream_and_capture(
            self.primary_model,
            [
                {"role":"system",  "content": system},
                {"role":"user",    "content": json.dumps(prompt)}
            ],
            tag="[ConfirmCheck]"
        ).strip()
        return bool(re.search(r"\byes\b", out, re.I))
   

    # helper: resume after user says yes/no
    def _handle_confirmation(self, reply: str) -> str:
        import re
        ans = reply.strip().lower()
        # YES
        if re.search(r"\b(yes|y|sure|go ahead)\b", ans):
            st = self._pending_state
            queue = self._pending_queue
            # clear flags
            del self._awaiting_confirmation
            del self._pending_state
            del self._pending_queue
            # continue where we left off
            return self.run_with_meta_context(st["user_text"])
        # NO → abort or replan
        else:
            # clear flags
            del self._awaiting_confirmation
            st = self._pending_state
            queue = self._pending_queue
            # for simplicity, force replanning
            return self.run_with_meta_context(st["user_text"])
        

    # ────────────────────────────────────────────────────────────────────
    # NEW: Called from telegram_input to register incoming chats
    def register_chat(self, chat_id: int, user_text: str):
        """Remember which Telegram chat issued this request."""
        self._chat_contexts.add(chat_id)

    # ────────────────────────────────────────────────────────────────────
    # NEW: Proactive “appiphany” ping
    def _maybe_appiphany(self, chat_id: int):
        """
        If our pipeline thinks there’s a high-value insight to share,
        ping the user in text + voice.
        """
        # Example condition: no errors this turn + at least one curiosity probe
        if not getattr(self, "_last_errors", False) and getattr(self, "curiosity_used", []):
            text = "💡 I just made an insight that might help you!"
            # send text
            self._telegram_bot.send_message(chat_id=chat_id, text=text)
            # enqueue voice
            self.tts.enqueue(text)
            try:
                ogg = self.tts.wait_for_latest_ogg(timeout=1.0)
                with open(ogg, "rb") as vf:
                    self._telegram_bot.send_voice(chat_id=chat_id, voice=vf)
            except Exception:
                    pass

    def dump_architecture(self):
        import inspect, json

        arch = {
            "stages":               self.STAGES,
            "optional_stages":      self._optional_stages,
            "curiosity_templates":  [t.semantic_label for t in self.curiosity_templates],
            "rl_weights":           {"Q": self.rl.Q, "R_bar": self.rl.R_bar},
            "curiosity_weights":    {"Q": self.curiosity_rl.Q, "R_bar": self.curiosity_rl.R_bar},
            "system_prompts":       list(self.system_prompts),
            "stage_methods":        {}
        }
        for s in self.STAGES + ["curiosity_probe", "system_prompt_refine", "narrative_mull"]:
            fn = getattr(self, f"_stage_{s}", None)
            if fn:
                arch["stage_methods"][s] = {
                    "signature": str(inspect.signature(fn)),
                    "doc":       fn.__doc__,
                }

        print(json.dumps(arch, indent=2))


    def _stage_curiosity_probe(self, state: Dict[str,Any]) -> List[str]:
        """
        Identify gaps in clarified intent, auto-mull or explicit follow-ups via RL,
        ask the LLM for answers, record Q&A as ContextObjects, return answers.
        """
        from typing import Tuple, List
        probes: List[str] = []
        clar = state.get("clar_ctx")
        if clar is None:
            return probes

        # 1) build recall feature for gating
        recall_ids = state.get("recent_ids", [])
        counts = []
        for cid in recall_ids:
            try:
                counts.append(self.repo.get(cid).metadata.get("recall_stats", {}).get("count", 0))
            except:
                pass
        rf = sum(counts) / len(counts) if counts else 0.0

        # 2) detect explicit gaps
        gaps: List[Tuple[str,str]] = []
        if not clar.metadata.get("notes"):
            gaps.append(("missing_notes", clar.summary[:50]))
        plan_out = state.get("plan_output", "")
        if "date(" in plan_out and not any(
            kw.lower().startswith("date") for kw in clar.metadata.get("keywords", [])
        ):
            gaps.append(("missing_date", "plan mentions a date"))

        # 3) if no explicit gaps, maybe auto-mull
        if not gaps:
            gaps.append(("auto_mull", "self-reflection"))

        # 4) for each gap, pick template, ask LLM, record
        for gap_name, snippet in gaps:
            candidates = [t for t in self.curiosity_templates if gap_name in t.semantic_label]
            if not candidates:
                continue
            picked = max(
                candidates,
                key=lambda t: self.curiosity_rl.probability(t.semantic_label, rf)
            )
            prompt = picked.metadata.get("policy", picked.summary).format(snippet=snippet)

            # record question
            q_ctx = ContextObject.make_stage(
                f"curiosity_question_{gap_name}",
                [clar.context_id],
                {"question": prompt},
                session_id=getattr(self, "session_id", None)
            )
            q_ctx.component = "curiosity"
            q_ctx.semantic_label = "question"
            q_ctx.tags.append("curiosity")
            q_ctx.touch(); self.repo.save(q_ctx)

            # ask LLM
            reply = self._stream_and_capture(
                self.primary_model,
                [
                    {"role":"system","content":"Please answer this follow-up question:"},
                    {"role":"user",  "content":prompt}
                ],
                tag=f"[CuriosityAnswer_{gap_name}]"
            ).strip()

            # record answer
            a_ctx = ContextObject.make_stage(
                f"curiosity_answer_{gap_name}",
                [q_ctx.context_id],
                {"answer": reply},
                session_id=getattr(self, "session_id", None)
            )
            a_ctx.component = "curiosity"
            a_ctx.semantic_label = "answer"
            a_ctx.tags.append("curiosity")
            a_ctx.touch(); self.repo.save(a_ctx)

            state.setdefault("curiosity_used", []).append(picked.semantic_label)
            probes.append(reply)

        return probes
    
    def _get_prompt(self, label: str) -> str:
        ctx = next(c for c in self.repo.query(lambda c:
            c.semantic_label == label and c.component == "prompt"
        ))
        return ctx.metadata["prompt"]
    
    # ---------------------------------------------------------------------
    #  Self-tuning of system / policy prompts
    # ---------------------------------------------------------------------
    def _stage_system_prompt_refine(self, state: Dict[str, Any]) -> str | None:
        """
        RL-gated self-mutation of prompts & policies.

        • Chooses ONE tiny change (add / remove) per invocation.
        • Fully defensive: never crashes on missing context rows or bad JSON.
        """
        import json, textwrap

        # 1️  Recall-feature for RL gate
        recall_ids = state.get("recent_ids", [])
        counts: list[int] = []
        for cid in recall_ids:
            try:
                cobj = self.repo.get(cid)
                counts.append(cobj.metadata.get("recall_stats", {}).get("count", 0))
            except KeyError:
                continue  # archived / pruned
        rf = (sum(counts) / len(counts)) if counts else 0.0

        # 2️  RL gate – exit early if not chosen
        if not self.rl.should_run("system_prompt_refine", rf):
            return None

        # 3️  Snapshot **current** prompts / policies   (skip dynamic patches)
        rows = self.repo.query(
            lambda c: c.component in ("prompt", "policy") and "dynamic_prompt" not in c.tags
        )
        rows.sort(key=lambda c: c.timestamp)
        current_prompts = [
            (c.metadata.get("prompt") or c.metadata.get("policy") or "") for c in rows
        ]

        # 4️  Build meta-prompt for the LLM
        metrics = {
            "errors":          len(state["errors"]),
            "curiosity_used":  state["curiosity_used"][-5:],
            "recall_mean_cnt": rf,
        }
        prompt_block = "\n".join(
            f"- {textwrap.shorten(p, 60)}" for p in current_prompts
        ) or "(none)"

        refine_prompt = (
            "You are a self-optimising agent.\n\n"
            "### Active Prompts / Policies ###\n"
            f"{prompt_block}\n\n"
            "### Recent Metrics ###\n"
            f"{json.dumps(metrics, indent=2)}\n\n"
            "Propose exactly **one** minimal change and return ONLY valid JSON:\n"
            '{"action":"add","prompt":"<text>"}  or  '
            '{"action":"remove","prompt":"<substring>"}'
        )

        # 5️  Ask the model – be tolerant of malformed output
        try:
            raw = self._stream_and_capture(
                self.primary_model,
                [{"role": "system", "content": refine_prompt}],
                tag="[SysPromptRefine]",
            ).strip()
            plan = json.loads(raw)
            if not isinstance(plan, dict):
                return None
            action = plan.get("action")
            text = (plan.get("prompt") or "").strip()
        except Exception:
            return None  # bad JSON → abort

        # 6️  Apply the change and update static prompts in the repo
        if action == "add" and text:
            patch = ContextObject.make_policy(
                label=f"dynamic_prompt_{len(text)}",
                policy_text=text,
                tags=["dynamic_prompt"],
            )
            patch.touch()
            self.repo.save(patch)

            # bake this change into the static prompt records
            self._seed_static_prompts()

        elif action == "remove" and text:
            for row in rows:
                blob = row.metadata.get("prompt") or row.metadata.get("policy") or ""
                if text in blob:
                    try:
                        self.repo.delete(row.context_id)
                    except KeyError:
                        pass  # already archived

            # reflect removals in the static prompt records
            self._seed_static_prompts()

        else:
            return None  # no-op

        # 7️  Log the refinement decision
        refine_ctx = ContextObject.make_stage(
            "system_prompt_refine",
            [cid for cid in recall_ids if self.repo_exists(cid)],
            {"action": action, "text": text},
        )
        refine_ctx.component = "patch"
        refine_ctx.touch()
        self.repo.save(refine_ctx)

        return f"{action}:{text or '(none)'}"



    # Helper used above ---------------------------------------------------
    def repo_exists(self, cid: str) -> bool:
        """Return True iff the context-id still resolves in the repository."""
        try:
            self.repo.get(cid)
            return True
        except KeyError:
            return False


    def run_with_meta_context(
        self,
        user_text: str,
        status_cb: Callable[[str, Any], None] = lambda *a: None,
        chat_id:  Optional[int]               = None,
        msg_id:   Optional[int]               = None,
    ) -> str:
        """
        End-to-end orchestrator.

        • Returns immediately on empty input.
        • Catches any unexpected errors and returns a simple apology.
        """
        # ① Short-circuit empty or whitespace-only inputs
        if not user_text or not user_text.strip():
            return ""

        try:

            # ── helper: crude complexity metric ──────────────────────────────────
            def _complexity(q: str, notes: str, kws: list[str]) -> float:
                wc = min(len(q.split()), 50) / 50.0
                nb = min(len(notes) / 200.0, 1.0)
                kb = min(len(kws) / 5.0, 1.0)
                pd = sum(q.count(c) for c in "?,;!") / max(1, len(q))
                pb = min(pd * 5.0, 1.0)
                return 0.4 * wc + 0.2 * nb + 0.2 * kb + 0.2 * pb

            # ── bootstrap core stages ───────────────────────────────────────────
            queue = deque([
                "record_input",
                "load_system_prompts",
                "retrieve_and_merge_context",
                "intent_clarification",
            ])

            state = {
                "user_text":      user_text,
                "errors":         [],
                "curiosity_used": [],
                "rl_included":    [],
                "recent_ids":     [],
            }
            answer: str | None = None
            self._last_errors = False

            if chat_id is not None:
                self._chat_contexts.add(chat_id)

            # ── event loop ─────────────────────────────────────────────────────
            while queue:
                stage = queue.popleft()
                start = datetime.utcnow()
                summary, ok = "", False

                try:
                    status_cb(stage, "…")

                    # ── meta stages (fire-and-forget) ────────────────────────
                    if stage == "curiosity_probe" and hasattr(self, "_stage_curiosity_probe"):
                        threading.Thread(target=self._stage_curiosity_probe,
                                        args=(state.copy(),), daemon=True).start()
                        summary, ok = "(probe dispatched)", True
                    elif stage == "system_prompt_refine" and hasattr(self, "_stage_system_prompt_refine"):
                        threading.Thread(target=self._stage_system_prompt_refine,
                                        args=(state.copy(),), daemon=True).start()
                        summary, ok = "(refine dispatched)", True
                    elif stage == "narrative_mull" and hasattr(self, "_stage_narrative_mull"):
                        threading.Thread(target=self._stage_narrative_mull,
                                        args=(state.copy(),), daemon=True).start()
                        summary, ok = "(mull dispatched)", True

                    # ── core pipeline stages ──────────────────────────────────
                    elif stage == "record_input":
                        ctx = self._stage1_record_input(user_text)
                        state["user_ctx"] = ctx
                        state["recent_ids"].append(ctx.context_id)
                        summary, ok = ctx.summary, True

                    elif stage == "load_system_prompts":
                        ctx = self._stage2_load_system_prompts()
                        state["sys_ctx"] = ctx
                        state["recent_ids"].append(ctx.context_id)
                        summary, ok = "(prompts loaded)", True

                    elif stage == "retrieve_and_merge_context":
                        extra = self._get_history()
                        out = self._stage3_retrieve_and_merge_context(
                            user_text, state["user_ctx"], state["sys_ctx"], extra_ctx=extra
                        )
                        state.update(out)
                        summary, ok = "(context merged)", True

                    elif stage == "intent_clarification":
                        ctx = self._stage4_intent_clarification(user_text, state)
                        state["clar_ctx"] = ctx
                        state["recent_ids"].append(ctx.context_id)
                        summary = ctx.summary
                        comp = _complexity(
                            user_text,
                            ctx.metadata.get("notes", ""),
                            ctx.metadata.get("keywords", []),
                        )
                        state["complexity"] = comp
                        queue.extend([
                            "external_knowledge", "prepare_tools", "planning_summary",
                            "plan_validation", "tool_chaining", "invoke_with_retries",
                            "reflection_and_replan", "assemble_and_infer",
                            "response_critique", "memory_writeback",
                        ])
                        ok = True

                    elif stage == "external_knowledge":
                        ctx = self._stage5_external_knowledge(state["clar_ctx"])
                        state["know_ctx"] = ctx
                        state["recent_ids"].append(ctx.context_id)
                        summary, ok = "(knowledge)", True

                    elif stage == "prepare_tools":
                        lst = self._stage6_prepare_tools()
                        state["tools_list"] = lst
                        summary, ok = f"{len(lst)} tools", True

                    elif stage == "planning_summary":
                        ctx, plan_json = self._stage7_planning_summary(
                            state["clar_ctx"], state["know_ctx"],
                            state["tools_list"], user_text
                        )
                        state["plan_ctx"], state["plan_output"] = ctx, plan_json
                        state["recent_ids"].append(ctx.context_id)
                        summary, ok = "plan ready", True

                    elif stage == "plan_validation":
                        _, errs, fixed = self._stage7b_plan_validation(
                            state["plan_ctx"], state["plan_output"], state["tools_list"]
                        )
                        if errs:
                            raise RuntimeError(f"plan-validation: {errs}")
                        state["fixed_calls"] = fixed
                        queue.extend([
                            "tool_chaining", "invoke_with_retries", "reflection_and_replan",
                            "assemble_and_infer", "response_critique", "memory_writeback",
                        ])
                        summary, ok = f"{len(fixed)} calls", True
                    elif stage == "tool_chaining":
                        tc_ctx, confirmed_calls, schemas = self._stage8_tool_chaining(
                            state["plan_ctx"],
                            "\n".join(state["fixed_calls"]),
                            state["tools_list"]
                        )
                        state.update({
                        "tc_ctx": tc_ctx,
                        "raw_calls": confirmed_calls,
                        "schemas": schemas,
                        # also store the exact chainer input so we can re-use it:
                        "chainer_input": "\n".join(state["fixed_calls"])
                        })

                    elif stage == "invoke_with_retries":
                        tctxs = self._stage9_invoke_with_retries(
                            raw_calls=state["raw_calls"],
                            plan_output=state["chainer_input"],      # must be the same text you gave to stage8
                            selected_schemas=state["schemas"],
                            user_text=state["user_text"],
                            clar_metadata=state["clar_ctx"].metadata
                        )
                        state["tool_ctxs"] = tctxs
                    elif stage == "reflection_and_replan":
                        rp = self._stage9b_reflection_and_replan(
                            state["tool_ctxs"], state["plan_output"],
                            user_text, state["clar_ctx"].metadata, state["plan_ctx"]
                        )
                        if rp:
                            queue.extend(["planning_summary", "plan_validation"])
                        summary, ok = f"replan={bool(rp)}", True

                    elif stage == "assemble_and_infer":
                        draft = self._stage10_assemble_and_infer(user_text, state)
                        state["draft"] = draft
                        summary, ok = "(drafted)", True

                    elif stage == "response_critique":
                        final = self._stage10b_response_critique_and_safety(
                            state["draft"], user_text, state.get("tool_ctxs", [])
                        )
                        state["final"] = final
                        summary, ok = "(polished)", True

                    elif stage == "memory_writeback":
                        # no-op here: defer output until after loop
                        summary, ok = "(memory queued)", True

                except Exception as e:
                    state["errors"].append((stage, str(e)))
                    summary, ok = f"error:{e}", False

                finally:
                    dur = (datetime.utcnow() - start).total_seconds()
                    if "user_ctx" in state:
                        perf = ContextObject.make_stage(
                            "stage_performance",
                            [state["user_ctx"].context_id],
                            {"stage": stage, "duration": dur, "error": not ok},
                        )
                        perf.touch(); self.repo.save(perf)
                    status_cb(stage, summary)
                    self._last_errors |= not ok

            # ── post-pipeline async meta-jobs ────────────────────────────────
            def _maybe_async(tag: str, fn: Callable):
                if hasattr(self, fn.__name__) and self.rl.should_run(tag, 0.0):
                    threading.Thread(target=fn, args=(state.copy(),), daemon=True).start()
                    state["rl_included"].append(tag)

            _maybe_async("curiosity_probe",      self._stage_curiosity_probe)
            _maybe_async("system_prompt_refine", self._stage_system_prompt_refine)
            _maybe_async("narrative_mull",       self._stage_narrative_mull)

            # send any “appiphany” pings
            for cid in list(self._chat_contexts):
                try: self._maybe_appiphany(cid)
                except: pass

            # RL reward update
            self.rl.update(state["rl_included"], 1.0 if not state["errors"] else 0.0)

            # ── ensure we always emit something via status_cb ──────────────────
            final_text = (answer or state.get("final", "")).strip()
            if not final_text:
                final_text = "Sorry, I couldn’t process that. Could you try rephrasing?"
            # send as a single chunk
            status_cb("output_1/1", final_text)

            return final_text

        except Exception:
            apology = "Sorry, something went wrong. Please try again."
            status_cb("output_1/1", apology)
            return apology




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
        from datetime import timedelta

        snippets: List[str] = []

        # ——— 1) External Web/snippet lookup ———————————————————
        for kw in clar_ctx.metadata.get("keywords", []):
            hits = self.engine.query(
                stage_id="external_knowledge_retrieval",
                similarity_to=kw,
                top_k=self.top_k
            )
            snippets.extend(h.summary for h in hits)

        # ——— 2) Local memory lookup ——————————————————————
        now = default_clock()
        start = (now - timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ")
        for kw in clar_ctx.metadata.get("keywords", []):
            raw = Tools.context_query(
                time_range=[start, now.strftime("%Y%m%dT%H%M%SZ")],
                query=kw,
                top_k=self.top_k
            )
            try:
                results = json.loads(raw).get("results", [])
                for r in results:
                    snippets.append(f"(MEM) {r.get('summary','')}")
                snippets = list(dict.fromkeys(snippets))
            except json.JSONDecodeError:
                continue

        # ——— 3) Fallback to recent chat if no snippets found —————
        if not snippets:
            recent = self._get_history()[-5:]
            for ctx in recent:
                snippets.append(f"(HIST) {ctx.summary}")
            snippets = list(dict.fromkeys(snippets))

        # ——— 4) Persist and return ———————————————————————
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

        # 1️  Fetch **all** schema rows
        rows = self.repo.query(
            lambda c: c.component == "schema" and "tool_schema" in c.tags
        )

        # 2️  Bucket rows by tool name, keep the newest per bucket
        buckets: dict[str, ContextObject] = {}
        for r in rows:
            try:
                name = json.loads(r.metadata["schema"])["name"]
            except Exception:
                continue
            if name not in buckets or r.timestamp > buckets[name].timestamp:
                buckets[name] = r

        # 3️  Build the final list, sorted for reproducibility
        tool_defs: list[dict[str, str]] = []
        for name in sorted(buckets):
            data = json.loads(buckets[name].metadata["schema"])
            tool_defs.append({
                "name":        data["name"],
                "description": data.get("description", "").split("\n", 1)[0],
            })

        return tool_defs
    
    # ── Stage 7: Planning Summary with Tool-Existence Check & Plan Tracking ────────────────────
    def _stage7_planning_summary(
        self,
        clar_ctx: ContextObject,
        know_ctx: ContextObject,
        tools_list: List[Dict[str, str]],
        user_text: str,
    ) -> Tuple[ContextObject, str]:
        import json, re, hashlib, datetime
        from typing import Any, Dict, List, Tuple

        # strip markdown fences and pull out a bare JSON object
        def _clean_json_block(text: str) -> str:
            m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
            if m:
                return m.group(1)
            m2 = re.search(r"(\{.*\})", text, flags=re.S)
            return m2.group(1) if m2 else text.strip()

        # build a markdown list of available tools
        tools_md = "\n".join(f"- **{t['name']}**: {t['description']}"
                             for t in tools_list)

        # inject it directly into the system prompt
        base_system = self._get_prompt("planning_prompt") + "\n" + tools_md

        # assemble the user prompt
        base_user = (
            f"User question:\n{user_text}\n\n"
            f"Clarified intent:\n{clar_ctx.summary}\n\n"
            f"Snippets:\n{know_ctx.summary or '(none)'}"
        )

        last_calls = None
        plan_obj   = None
        cleaned    = ""

        # up to 3 attempts to get a valid plan
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
                        "Your previous plan used unknown or invalid tools.  "
                        "Please replan using only valid tools."},
                    {"role": "user",   "content": cleaned},
                ]

            raw     = self._stream_and_capture(self.secondary_model, msgs, tag=tag)
            cleaned = _clean_json_block(raw)

            # parse JSON or fallback to regex extraction
            try:
                cand = json.loads(cleaned)
                if isinstance(cand, dict) and "tasks" in cand:
                    plan_obj = cand
                else:
                    raise ValueError
            except Exception:
                calls = re.findall(r'\b[A-Za-z_]\w*\([^)]*\)', raw)
                plan_obj = {"tasks": [{"call": c, "tool_input": {}, "subtasks": []}
                                      for c in calls]}

            # drop unknown calls
            available = {t["name"] for t in tools_list}
            unknown   = [t["call"] for t in plan_obj["tasks"] if t["call"] not in available]
            if unknown:
                self._print_stage_context(
                    f"planning_summary:unknown_tools(attempt={attempt})",
                    {"unknown": unknown, "available": sorted(available)}
                )
                continue

            # plateau guard (same plan twice → stop)
            this_calls = [t["call"] for t in plan_obj["tasks"]]
            if last_calls is not None and this_calls == last_calls:
                self._print_stage_context(
                    f"planning_summary:plateaued(attempt={attempt})",
                    {"calls": this_calls}
                )
                break
            last_calls = this_calls
            break

        # ── Flatten tasks + subtasks ──────────────────────────────────────────
        def _flatten(task: Dict[str, Any]) -> List[Dict[str, Any]]:
            out = [task]
            for sub in task.get("subtasks", []):
                out.extend(_flatten(sub))
            return out

        flat_tasks: List[Dict[str, Any]] = []
        for t in (plan_obj or {}).get("tasks", []):
            flat_tasks.extend(_flatten(t))

        # build call-strings from the flattened list
        call_strings: List[str] = []
        for task in flat_tasks:
            name   = task["call"]
            params = task.get("tool_input", {}) or {}
            if params:
                args = ",".join(f'{k}={json.dumps(v, ensure_ascii=False)}'
                                for k, v in params.items())
                call_strings.append(f"{name}({args})")
            else:
                call_strings.append(f"{name}()")

        # serialize plan and compute a short plan_id
        plan_json = json.dumps({"tasks": [{"call": s, "subtasks": []}
                                          for s in call_strings]})
        plan_sig  = hashlib.md5(plan_json.encode("utf-8")).hexdigest()[:8]

        # persist planning_summary context
        ctx = ContextObject.make_stage(
            "planning_summary",
            clar_ctx.references + know_ctx.references,
            {"plan": plan_obj, "attempt": attempt, "plan_id": plan_sig}
        )
        ctx.stage_id = f"planning_summary_{plan_sig}"
        ctx.summary  = plan_json
        ctx.touch(); self.repo.save(ctx)

        # RL signal for the planning step
        if call_strings:
            succ = ContextObject.make_success(
                f"Planner → {len(call_strings)} task(s)",
                refs=[ctx.context_id]
            )
        else:
            succ = ContextObject.make_failure(
                "Planner → empty plan",
                refs=[ctx.context_id]
            )
        succ.stage_id = f"planning_summary_signal_{plan_sig}"
        succ.touch(); self.repo.save(succ)

        # create initial plan-tracker context
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
            #calls = re.findall(r'\b[A-Za-z_]\w*\([^)]*\)', plan_output)
            #return calls, [], calls
            raise RuntimeError("Planner output was not valid JSON; cannot validate tool_input parameters")

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

        # ── D) Parse back the confirmed list ────────────────────────────
        confirmed = calls
        try:
            blob = re.search(r'\{.*"tool_calls".*\}', out, flags=re.S).group(0)
            parsed2 = json.loads(blob)
            if isinstance(parsed2.get("tool_calls"), list):
                confirmed = parsed2["tool_calls"]
        except Exception:
            pass

        # ── E) Persist & return ─────────────────────────────────────────
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
    ) -> List["ContextObject"]:
        import json, re, hashlib, datetime
        from typing import Tuple, Any, Dict, List

        # ── 0) Tracker init ────────────────────────────────────────────────
        plan_sig = hashlib.md5(plan_output.encode("utf-8")).hexdigest()[:8]
        tracker = next(
            (c for c in self.repo.query(
                lambda c: c.component=="plan_tracker" and c.semantic_label==plan_sig
            )),
            None
        )
        if not tracker:
            tracker = ContextObject.make_stage(
                "plan_tracker", [], {
                    "plan_id":        plan_sig,
                    "plan_calls":     raw_calls.copy(),
                    "total_calls":    len(raw_calls),
                    "succeeded":      0,
                    "attempts":       0,
                    "call_status_map":{},
                    "errors_by_call": {},
                    "status":         "in_progress",
                    "started_at":     datetime.datetime.utcnow().isoformat()+"Z"
                }
            )
            tracker.semantic_label = plan_sig
            tracker.stage_id       = f"plan_tracker_{plan_sig}"
            tracker.summary        = "initialized plan tracker"
            tracker.touch(); self.repo.save(tracker)
        else:
            meta = tracker.metadata
            meta.setdefault("plan_calls",     raw_calls.copy())
            meta.setdefault("total_calls",    len(raw_calls))
            meta.setdefault("succeeded",      0)
            meta.setdefault("attempts",       0)
            meta.setdefault("call_status_map",{})
            meta.setdefault("errors_by_call", {})
            meta.setdefault("status",         "in_progress")
            meta.setdefault("started_at",     datetime.datetime.utcnow().isoformat()+"Z")
            tracker.touch(); self.repo.save(tracker)

        tracker.metadata["attempts"] += 1
        tracker.metadata["last_attempt_at"] = datetime.datetime.utcnow().isoformat()+"Z"
        tracker.touch(); self.repo.save(tracker)
        if tracker.metadata.get("status") == "success":
            return []

        # ── Helpers ─────────────────────────────────────────────────────────
        def _norm(calls: List[Any]) -> List[str]:
            out = []
            for c in calls:
                if isinstance(c, dict) and "tool_call" in c:
                    out.append(c["tool_call"])
                elif isinstance(c, str):
                    out.append(c)
            return out

        def _validate(res: Dict[str,Any]) -> Tuple[bool,str]:
            exc = res.get("exception")
            return (exc is None, exc or "")

        def normalize_key(k: str) -> str:
            return re.sub(r"\W+","",k).lower()

        # ── 1) Initialise pending calls in original order ────────────────
        all_calls   = _norm(raw_calls)
        call_status = tracker.metadata["call_status_map"]
        pending     = [c for c in all_calls if not call_status.get(c, False)]
        last_results: Dict[str,Any] = {}
        tool_ctxs: List[ContextObject] = []

        if not pending:
            tracker.metadata["status"]       = "success"
            tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat()+"Z"
            tracker.touch(); self.repo.save(tracker)
            return []

        # ── 2) Retry loop over only pending calls ────────────────────────────
        max_retries = 10
        for attempt in range(1, max_retries+1):
            errors: List[Tuple[str,str]] = []

            for original in list(pending):
                call_str = original

                # 1) [alias from tool_name] style placeholders
                for ph in re.findall(r"\[([^\]]+)\]", call_str):
                    # does it look like "[alias from tool]"?
                    if " from " in ph:
                        alias, toolname = ph.split(" from ", 1)
                        alias_key   = normalize_key(alias)
                        tool_key    = normalize_key(toolname)
                        # find the exact last_results entry for that tool
                        match = tool_key if tool_key in last_results else None

                    else:
                        # fallback: match placeholder substring to any key
                        phn = normalize_key(ph)
                        match = next(
                            (k for k in last_results
                            if phn in normalize_key(k)
                            or normalize_key(k) in phn),
                            None
                        )
                    if match:
                        value = last_results[match]
                        call_str = call_str.replace(f"[{ph}]", repr(value))


                # 2) {{alias}} style placeholders
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


                # 3) inline nested zero-arg calls only if embedded (not top‐level)
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
                        # persist nested output
                        try:
                            sch_i = next(
                                s for s in selected_schemas
                                if json.loads(s.metadata["schema"])["name"] == inner
                            )
                            ctx_i = ContextObject.make_stage(
                                "tool_output", [sch_i.context_id], r_i
                            )
                            ctx_i.stage_id = f"tool_output_nested_{inner}"
                            ctx_i.summary  = str(out_i) if ok_i else f"ERROR: {err_i}"
                            ctx_i.metadata.update(r_i)
                            ctx_i.touch(); self.repo.save(ctx_i)
                            tool_ctxs.append(ctx_i)
                        except StopIteration:
                            pass

                # 4) main invocation
                res = Tools.run_tool_once(call_str)
                ok, err = _validate(res)
                tool_key = normalize_key(call_str.split("(",1)[0])
                last_results[tool_key] = res["output"]

                # persist output
                try:
                    name = original.split("(",1)[0]
                    sch  = next(
                        s for s in selected_schemas
                        if json.loads(s.metadata["schema"])["name"] == name
                    )
                    refs = [sch.context_id]
                except StopIteration:
                    refs = []
                ctx = ContextObject.make_stage("tool_output", refs, res)
                ctx.stage_id = f"tool_output_{name}"
                ctx.summary  = str(res.get("output")) if ok else f"ERROR: {err}"
                ctx.metadata.update(res)
                ctx.touch(); self.repo.save(ctx)
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
                tracker.metadata["status"]       = "success"
                tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat()+"Z"
                tracker.touch(); self.repo.save(tracker)
                return tool_ctxs

            # otherwise replan only the remaining pending calls
            retry_sys = (
                self._get_prompt("toolchain_retry_prompt")
                + "\n\nOriginal question:\n" + user_text
                + "\n\nPlan:\n" + plan_output
                + "\n\nPending calls:\n" + "\n".join(pending)
            )
            retry_msgs = [
                {"role":"system","content":retry_sys},
                {"role":"user",  "content":json.dumps({"tool_calls":pending})},
            ]
            out = self._stream_and_capture(
                self.secondary_model, retry_msgs, tag="[ToolChainRetry]"
            ).strip()

            try:
                pending = json.loads(out)["tool_calls"]
            except:
                parsed = Tools.parse_tool_call(out)
                pending = _norm(parsed if isinstance(parsed, list) else [parsed] or pending)

        # ── 3) Give up after max_retries ────────────────────────────────────
        tracker.metadata["status"]       = "failed"
        tracker.metadata["last_errors"]  = [e for _,e in errors]
        
        tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat()+"Z"
        tracker.touch(); self.repo.save(tracker)
        return tool_ctxs


    def _stage10_assemble_and_infer(self, user_text: str, state: Dict[str, Any]) -> str:
        """
        Assemble every bit of context we have, run one final LLM pass, and
        persist the answer — even if earlier tool stages failed or were skipped.
        """
        import json

        # ──────────────────────────────────────────────────────────────────────
        # 1) Collect *all* relevant context-IDs, but only if they exist
        # ──────────────────────────────────────────────────────────────────────
        refs: list[str] = []

        def _maybe_add(key: str):
            obj = state.get(key)
            if obj:
                refs.append(obj.context_id)

        _maybe_add("user_ctx")
        _maybe_add("sys_ctx")
        refs.extend(state.get("recent_ids", []))
        _maybe_add("clar_ctx")
        _maybe_add("know_ctx")
        _maybe_add("plan_ctx")

        # tool outputs may be missing
        for t in state.get("tool_ctxs", []):
            refs.append(t.context_id)

        # de-dup while keeping original order
        seen, ordered = set(), []
        for cid in refs:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)

        # ──────────────────────────────────────────────────────────────────────
        # 2) Load the ContextObjects, skipping any that were deleted / archived
        # ──────────────────────────────────────────────────────────────────────
        ctxs = []
        for cid in ordered:
            try:
                ctxs.append(self.repo.get(cid))
            except KeyError:
                continue          # stale pointer – ignore

        ctxs.sort(key=lambda c: c.timestamp)
        print(f"[assemble_and_infer] total context objects: {len(ctxs)}")

        # ──────────────────────────────────────────────────────────────────────
        # 3) Inline every bit of context into one big “knowledge sheet”
        # ──────────────────────────────────────────────────────────────────────
        interm_parts: list[str] = []
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

        self._print_stage_context("assemble_and_infer", {
            "user_question":    [user_text],
            "plan":             [state.get("plan_output", "(no plan)")],
            "inference_prompt": [self.inference_prompt],
            "inlined_context":  interm_parts,
        })

        # ──────────────────────────────────────────────────────────────────────
        # 4) Final LLM call
        # ──────────────────────────────────────────────────────────────────────
        final_sys = self._get_prompt("final_inference_prompt")

        plan_text = state.get("plan_output", "(no plan)")
        self._last_plan_output = plan_text        # for the critic stage

        msgs = [
            {"role": "system", "content": final_sys},
            {"role": "system", "content": f"User question:\n{user_text}"},
            {"role": "system", "content": f"Plan:\n{plan_text}"},
            {"role": "system", "content": self.inference_prompt},
            {"role": "system", "content": interm},
            {"role": "user",   "content": user_text},
        ]
        reply = self._stream_and_capture(
            self.primary_model, msgs, tag="[Assistant]"
        ).strip()

        # ──────────────────────────────────────────────────────────────────────
        # 5) Persist the answer (even if tc_ctx/tool_ctxs missing)
        # ──────────────────────────────────────────────────────────────────────
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
        critic_sys = self._get_prompt("critic_prompt")


        prompt_user = "\n\n".join([
            f"User asked:\n{user_text}",
            f"Plan executed:\n{plan_text}",
            f"Your draft:\n{draft}",
            "Here are the raw tool results:\n" + "\n\n".join(outputs),
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

    def _stage_generate_narrative(self, state: Dict[str, Any]) -> ContextObject:
        """
        Build a running narrative of this conversation turn by turn,
        link it to all the context objects we’ve touched so far,
        and store the narrative’s ContextObject ID for future reference.
        """
        from datetime import datetime

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

    # ---------------------------------------------------------------------
    #  Async self-reflection stage (robust, non-blocking)
    # ---------------------------------------------------------------------
    def _stage_narrative_mull(self, state: Dict[str, Any]) -> str:
        """
        Fire-and-forget background task that:
        1) Reads the running narrative & current prompts
        2) Asks the LLM for ≤3 improvement areas + questions
        3) Stores each Q-and-A as ContextObjects
        4) Emits a dynamic_prompt_patch for every answer

        Never blocks the user-facing pipeline.
        """
        import threading

        def _worker():
            import json, textwrap
            from datetime import datetime

            # 1️  Get the full narrative text
            narr_ctx = self._load_narrative_context()                 # singleton
            full_narr = narr_ctx.metadata.get("narrative", narr_ctx.summary or "")

            # 2️  Snapshot active prompts / policies (skip previous patches)
            prompt_rows = self.repo.query(
                lambda c: c.component in ("prompt", "policy") and "dynamic_prompt" not in c.tags
            )
            prompt_rows.sort(key=lambda c: c.timestamp)
            prompt_block = "\n".join(
                f"{textwrap.shorten(c.summary or '', width=120, placeholder='…')}"
                for c in prompt_rows
            ) or "(none)"

            # 3️  Build the meta-prompt
            meta_prompt = (
                self._get_prompt("narrative_mull_prompt")
                + "\n\n### Narrative ###\n"    + full_narr
                + "\n\n### Active Prompts & Policies ###\n" + prompt_block
            )

            # 4️  Call the LLM
            try:
                raw  = self._stream_and_capture(
                    self.primary_model,
                    [{"role": "system", "content": meta_prompt}],
                    tag="[NarrativeMull]"
                ).strip()
                data = json.loads(raw)
                issues = data.get("issues", [])
                if not isinstance(issues, list):
                    return
            except Exception:
                return   # silently bail on bad JSON / API failure

            # 5️  Persist each Q & A pair
            for idx, item in enumerate(issues, 1):
                if not isinstance(item, dict):
                    continue
                area     = item.get("area", f"area_{idx}")
                question = (item.get("question") or "").strip()
                if not question:
                    continue

                # QUESTION object
                q_ctx = ContextObject.make_stage(
                    stage_name="narrative_question",
                    input_refs=[narr_ctx.context_id],
                    output={"question": question},
                )
                q_ctx.component      = "narrative"
                q_ctx.semantic_label = f"narrative_question_{idx}"
                q_ctx.tags.append("narrative")
                q_ctx.touch(); self.repo.save(q_ctx)

                # ANSWER from the model
                answer = self._stream_and_capture(
                    self.primary_model,
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are the same meta-reasoner.  Answer the following "
                                "self-reflection question **only** from the narrative & prompts."
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                    tag=f"[NarrativeAnswer_{idx}]"
                ).strip()

                # ANSWER object
                a_ctx = ContextObject.make_stage(
                    stage_name="narrative_answer",
                    input_refs=[q_ctx.context_id],
                    output={"answer": answer},
                )
                a_ctx.component      = "narrative"
                a_ctx.semantic_label = f"narrative_answer_{idx}"
                a_ctx.tags.append("narrative")
                a_ctx.touch(); self.repo.save(a_ctx)

                # Dynamic prompt patch
                patch_text = (
                    f"// {datetime.utcnow().isoformat(timespec='seconds')}Z\n"
                    f"Issue: {area}\nQ: {question}\nA: {answer}\n"
                )
                patch = ContextObject.make_policy(
                    label="dynamic_prompt_patch",
                    policy_text=patch_text,
                    tags=["dynamic_prompt"],
                )
                patch.touch(); self.repo.save(patch)

        # Launch the mull on a daemon thread and return immediately
        threading.Thread(target=_worker, daemon=True).start()
        return "(narrative mull dispatched)"


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
