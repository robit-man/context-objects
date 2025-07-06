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
from context import ContextObject, ContextRepository, HybridContextRepository, MemoryManager, default_clock
from tools import TOOL_SCHEMAS, Tools
from datetime import datetime
from collections import deque
import inspect
from threading import Lock, Thread, Event
from queue import Queue, Empty
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
                 path:  str   = "weights.rl"):
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
        tts_manager:      Any | None    = None,
        engine:           Any | None    = None,
        rl_controller:    Any | None    = None,
        repo:             ContextRepository | None = None,
    ):
        # 1) Remember your store paths
        self.context_path = context_path
        self.config_path  = config_path

        # — load or init config —
        try:
            self.cfg = json.load(open(config_path))
        except FileNotFoundError:
            self.cfg = {}

        # New pruning & window parameters
        self.context_ttl_days   = self.cfg.get("context_ttl_days",    7)
        self.max_history_items  = self.cfg.get("max_history_items",  10)
        self.max_semantic_items = self.cfg.get("max_semantic_items", 10)
        self.max_memory_items   = self.cfg.get("max_memory_items",   10)
        self.max_tool_outputs   = self.cfg.get("max_tool_outputs",   10)

        # Models & lookback
        self.primary_model   = self.cfg.get("primary_model",   "gemma3:4b")
        self.secondary_model = self.cfg.get("secondary_model", self.primary_model)
        self.decision_model = self.cfg.get("decision_model", self.secondary_model)
        self.lookback        = self.cfg.get("lookback_minutes", lookback_minutes)
        self.top_k           = self.cfg.get("top_k",            top_k)
        self.hist_k          = self.cfg.get("history_turns",    5)

        # — system & stage prompts —
        self.clarifier_prompt = self.cfg.get(
            "clarifier_prompt",
            "You are Clarifier. Expand the user’s intent into a JSON object with "
            "two keys: 'keywords' (an array of concise keywords) and 'notes'. "
            "Notes should produce NO value judgements or claims, and should only expand on what the user said."
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
            "If you cannot, just list the tool calls. Only return exact objects from the list of Available tools:\n"
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
            "Alright, detective, here’s your raw evidence:\n"
            "  • The user’s question\n"
            "  • The plan you executed, flagged with successes and failures\n"
            "  • Your first draft—warts and all\n"
            "  • Every raw tool output, including error dumps\n\n"
            "Now, dissect each failure:\n"
            "  - Pinpoint the exact call and its error message\n"
            "  - Explain how that slip-up skewed your response\n"
            "  - Propose a no-nonsense fix or fallback—no excuses\n\n"
            "Then, grudgingly piece together a final answer that:\n"
            "  • Owns up to any gaps caused by failures\n"
            "  • Delivers razor-sharp accuracy and brevity\n"
            "  • Milk every success for all it’s worth\n\n"
            "Return only the final answer text."
        )
        self.narrative_mull_prompt = self.cfg.get(
            "narrative_mull_prompt",
            "You are an autonomous meta-reasoner performing deep introspection on your own pipeline execution.  "
            "You will be provided with:\n"
            "  • The rolling narrative so far (conversation history + assistant actions)\n"
            "  • The current system prompts and any dynamic prompt patches\n"
            "  • The pipeline architecture (STAGES, optional_stages, RL weights)\n"
            "  • Recent tool outputs, including errors and exceptions\n\n"
            "Your task:\n"
            "  1. Identify up to three distinct improvement areas.\n"
            "  2. For each area, produce a JSON object with these keys:\n"
            "     - \"area\":    a brief identifier (e.g. \"prompt_clarity\", \"error_handling\")\n"
            "     - \"question\":a focused self-reflection question to probe why the issue occurred\n"
            "     - \"recommendation\": a concise, actionable suggestion to address it\n"
            "     - \"plan_calls\": optional array of tool calls (e.g. [\"toolX(param=…)\"]) if you can automate a fix\n\n"
            "Return **only** valid JSON in this exact shape:\n"
            "{\n"
            "  \"issues\": [\n"
            "    {\n"
            "      \"area\": \"<short-name>\",\n"
            "      \"question\": \"<self-reflection question>\",\n"
            "      \"recommendation\": \"<concise suggestion>\",\n"
            "      \"plan_calls\": [\"toolA(arg=…)\", …]\n"
            "    },\n"
            "    …\n"
            "  ]\n"
            "}"
        )
        defaults = {
            "primary_model":    self.primary_model,
            "secondary_model":  self.secondary_model,
            "decision_model":  self.decision_model,
            "lookback_minutes": self.lookback,
            "top_k":            self.top_k,
            "history_turns":    self.hist_k,
        }
        if any(defaults[k] != self.cfg.get(k) for k in defaults):
            json.dump({**self.cfg, **defaults}, open(self.config_path, "w"), indent=2)

        # — init context store & memory manager —
        if repo is not None:
            # use the injected per-chat repository
            self.repo = repo
        else:
            # fallback: shard JSONL + SQLite alongside context_path
            sqlite_path = context_path.replace(".jsonl", ".db")
            self.repo = HybridContextRepository(
                jsonl_path=context_path,
                sqlite_path=sqlite_path,
                archive_max_mb=self.cfg.get("archive_max_mb", 10.0),
            )
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
        
        from context import sanitize_jsonl
        sanitize_jsonl(self.context_path)
        self._seed_tool_schemas()
        self._seed_static_prompts()

        # — text-to-speech manager —
        self.tts = tts_manager
        self._chat_contexts: set[int] = set()
        self._telegram_bot = None

        # Self-review background thread control
        import threading
        self._stop_self_review    = threading.Event()
        self._self_review_thread  = None

        # — auto-discover any _stage_<name>() methods as “optional” —
        all_methods = {name for name, _ in inspect.getmembers(self, inspect.ismethod)}
        discovered = [
            s for s in self.STAGES
                + ["curiosity_probe", "system_prompt_refine", "narrative_mull"]
            if f"_stage_{s}" in all_methods
        ]
        self._optional_stages = self.cfg.get("rl_optional", discovered)

        self.rl = rl_controller or RLController(
            stages=[
                "curiosity_probe",
                "system_prompt_refine",
                "narrative_mull",
                "prune_context_store",
                "semantic_retrieval",
                "memory_retrieval",
                "tool_output_retrieval",
            ],
            alpha=self.cfg.get("rl_alpha", 0.1),
            beta= self.cfg.get("rl_beta",  0.01),
            gamma=self.cfg.get("rl_gamma", 0.1),
            path=self.cfg.get("rl_path", "weights.rl"),
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
                #print("Embedding text for context.", "PROCESS")
                response = embed(model="nomic-embed-text", input=text)
                vec = np.array(response["embeddings"], dtype=float)
                # ensure 1-D
                vec = vec.flatten()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                #print("Text embedding computed and normalized.", "SUCCESS")
                return vec
            except Exception as e:
                #print("Error during text embedding: " + str(e), "ERROR")
                return np.zeros(768, dtype=float)


        self.engine = ContextQueryEngine(
           self.repo,
           lambda t: embed_text(t)  # Use your custom embedding function
        )



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
        # ── ADD THIS LINE ────────────────────────────────────────────
        self.system_prompts = {
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
        # ─────────────────────────────────────────────────────────────

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
        # coerce None→"" so join never fails
        keeper.metadata["narrative"] = "\n".join((n.summary or "") for n in narr_objs)
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
        print("--- Stage Context Dump -----------------------------------------------------")
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
        print("--- Stage Context Dump End -------------------------------------------------")

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
        from datetime import datetime

        arch = {
            "stages":               self.STAGES,
            "optional_stages":      self._optional_stages,
            "curiosity_templates":  [t.semantic_label for t in self.curiosity_templates],
            "rl_weights":           {"Q": self.rl.Q, "R_bar": self.rl.R_bar},
            "curiosity_weights":    {"Q": self.curiosity_rl.Q, "R_bar": self.curiosity_rl.R_bar},
            # now output the full mapping of prompt names → text
            "system_prompts":       self.system_prompts,
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
        from datetime import datetime

        probes: List[str] = []
        clar = state.get("clar_ctx")
        if clar is None:
            return probes

        # 1) Compute cascade-activation–based recall feature
        recall_ids = state.get("recent_ids", [])
        if recall_ids:
            activation_map = self.memman.spread_activation(
                seed_ids=recall_ids,
                hops=2,
                decay=0.6,
                assoc_weight=1.0,
                recency_weight=0.5
            )
            # take mean of top-N activations
            top_vals = sorted(activation_map.values(), reverse=True)[: len(recall_ids)]
            rf = sum(top_vals) / len(top_vals) if top_vals else 0.0
        else:
            rf = 0.0

        # 2) Detect explicit gaps
        gaps: List[Tuple[str,str]] = []
        if not clar.metadata.get("notes"):
            gaps.append(("missing_notes", clar.summary[:50]))
        plan_out = state.get("plan_output", "")
        if "date(" in plan_out and not any(
            kw.lower().startswith("date") for kw in clar.metadata.get("keywords", [])
        ):
            gaps.append(("missing_date", "plan mentions a date"))

        # 3) If no explicit gaps, auto-mull
        if not gaps:
            gaps.append(("auto_mull", "self-reflection"))

        # 4) For each gap, pick a template, probe LLM, record Q&A
        for gap_name, snippet in gaps:
            # choose best template by RL probability
            candidates = [
                t for t in self.curiosity_templates
                if gap_name in t.semantic_label
            ]
            if not candidates:
                continue
            picked = max(
                candidates,
                key=lambda t: self.curiosity_rl.probability(t.semantic_label, rf)
            )
            prompt = picked.metadata.get("policy", picked.summary).format(snippet=snippet)

            # 4a) Record question ContextObject
            q_ctx = ContextObject.make_stage(
                f"curiosity_question_{gap_name}",
                [clar.context_id],
                {"question": prompt}
            )
            q_ctx.component        = "curiosity"
            q_ctx.semantic_label   = "question"
            q_ctx.tags.append("curiosity")
            # annotate retrieval metrics
            score = activation_map.get(picked.context_id, 0.0)
            q_ctx.retrieval_score    = score
            q_ctx.retrieval_metadata = {"template": picked.semantic_label}
            # record reinforcement: clar -> question
            self.memman.reinforce(clar.context_id, [q_ctx.context_id])
            q_ctx.touch()
            self.repo.save(q_ctx)

            # 4b) Ask the LLM
            reply = self._stream_and_capture(
                self.primary_model,
                [
                    {"role":"system","content":"Please answer this follow-up question:"},
                    {"role":"user",  "content":prompt}
                ],
                tag=f"[CuriosityAnswer_{gap_name}]"
            ).strip()

            # 4c) Record answer ContextObject
            a_ctx = ContextObject.make_stage(
                f"curiosity_answer_{gap_name}",
                [q_ctx.context_id],
                {"answer": reply}
            )
            a_ctx.component        = "curiosity"
            a_ctx.semantic_label   = "answer"
            a_ctx.tags.append("curiosity")
            # annotate retrieval metrics
            a_score = activation_map.get(q_ctx.context_id, 0.0)
            a_ctx.retrieval_score    = a_score
            a_ctx.retrieval_metadata = {"question_id": q_ctx.context_id}
            # record reinforcement: question -> answer
            self.memman.reinforce(q_ctx.context_id, [a_ctx.context_id])
            a_ctx.touch()
            self.repo.save(a_ctx)

            # track which template you used and collect the reply
            state.setdefault("curiosity_used", []).append(picked.semantic_label)
            probes.append(reply)

        return probes

    
    def _get_prompt(self, label: str) -> str:
        ctx = next(c for c in self.repo.query(lambda c:
            c.semantic_label == label and c.component == "prompt"
        ))
        return ctx.metadata["prompt"]
    
    def _stage_system_prompt_refine(self, state: Dict[str, Any]) -> str | None:
        """
        RL-gated self-mutation of prompts & policies, with full visibility
        into narrative, architecture, tool outcomes—and now a window of past
        evaluation events.
        """
        import json, textwrap, os, shutil
        from datetime import datetime
        import io, contextlib

        # — Helpers to pull in extra context —
        def _arch_dump() -> str:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.dump_architecture()
            return buf.getvalue()

        # 1) Compute RL recall feature via spreading activation
        recall_ids = state.get("recent_ids", [])
        activation_map: Dict[str, float] = {}
        if recall_ids:
            activation_map = self.memman.spread_activation(
                seed_ids=recall_ids,
                hops=2,
                decay=0.7,
                assoc_weight=1.0,
                recency_weight=0.5
            )
            top_vals = sorted(activation_map.values(), reverse=True)[: len(recall_ids)]
            rf = sum(top_vals) / len(top_vals)
        else:
            rf = 0.0

        # 2) RL-gate: maybe skip based on richer feature
        if not self.rl.should_run("system_prompt_refine", rf):
            return None

        # 3) Snapshot static prompts/policies
        rows = list(self.repo.query(
            lambda c: c.component in ("prompt", "policy") and "dynamic_prompt" not in c.tags
        ))
        rows.sort(key=lambda c: c.timestamp)

        # 3A) Annotate each with its activation score
        for ctx in rows:
            score = activation_map.get(ctx.context_id, 0.0)
            ctx.retrieval_score    = score
            ctx.retrieval_metadata = {"seed_ids": recall_ids}
            ctx.record_recall(
                stage_id="system_prompt_refine",
                coactivated_with=recall_ids,
                retrieval_score=score
            )
            self.repo.save(ctx)

        prompt_block = "\n".join(
            f"- {textwrap.shorten(c.metadata.get('prompt', c.metadata.get('policy','')), 80)}"
            for c in rows
        ) or "(none)"

        # ── 3B) Pull in last 10 evaluation events ────────────────────────
        eval_rows = list(self.repo.query(
            lambda c: c.component == "stage_performance"
        ))
        eval_rows.sort(key=lambda c: c.timestamp)
        recent_evals = eval_rows[-10:]
        eval_block = "\n".join(
            f"[{e.timestamp}] { (e.summary or '').replace(chr(10), ' ') }"
            for e in recent_evals
        ) or "(no prior evaluations)"

        # 4A) Metrics & diagnostics
        metrics = {
            "errors":          len(state.get("errors", [])),
            "curiosity_used":  state.get("curiosity_used", [])[-5:],
            "recall_mean":     rf,
        }
        rl_snapshot = {
            stage: round(self.rl.Q.get(stage, 0.0), 3)
            for stage in ("curiosity_probe", "system_prompt_refine", "narrative_mull")
        }
        diagnostics = {
            "rl_Q":           rl_snapshot,
            "rl_R_bar":       round(self.rl.R_bar, 3),
            "repo_total":     sum(1 for _ in self.repo.query(lambda _: True)),
            "repo_ephemeral": sum(
                1 for c in self.repo.query(lambda c: c.component in {
                    "segment", "tool_output", "narrative", "knowledge", "stage_performance"
                })
            ),
        }

        # 4B) Running narrative
        narr_ctx = self._load_narrative_context()
        full_narr = narr_ctx.metadata.get("history_text", narr_ctx.summary or "(empty)")

        # 4C) Last round of tool contexts
        tool_ctxs = state.get("tool_ctxs", [])
        tools_summary = json.dumps([
            {
                "call":   t.metadata.get("call", "<unknown>"),
                "result": (t.metadata.get("output") or {}).get("result", "<no result>")
                         if isinstance(t.metadata.get("output"), dict)
                         else t.metadata.get("output", "<no result>"),
                "error":  (t.metadata.get("output") or {}).get("error", False)
                         if isinstance(t.metadata.get("output"), dict)
                         else False
            }
            for t in tool_ctxs
        ], indent=2)

        # 5) Build the refine prompt (now including eval block)—
        arch = _arch_dump()
        refine_prompt = (
            "You are a self-optimising agent, reflecting on your entire run.\n\n"
            "### Active System Prompts & Policies ###\n"
            f"{prompt_block}\n\n"
            "### Recent Evaluation History ###\n"
            f"{eval_block}\n\n"
            "### Running Narrative History ###\n"
            f"{textwrap.shorten(full_narr, width=2000, placeholder='…')}\n\n"
            "### Architecture Snapshot ###\n"
            f"{textwrap.shorten(arch, width=2000, placeholder='…')}\n\n"
            "### Recent Tool Activity ###\n"
            f"{tools_summary}\n\n"
            "### Metrics & Diagnostics ###\n"
            f"{json.dumps(metrics, indent=2)}\n"
            f"{json.dumps(diagnostics, indent=2)}\n\n"
            "Propose **exactly one** minimal change and return ONLY JSON:\n"
            '  {"action":"add","prompt":"<text>"}\n'
            'OR\n'
            '  {"action":"remove","prompt":"<substring>"}\n\n'
            "Your change should be small, targeted, and improve performance."
        )

        # 6) Invoke the LLM
        try:
            raw = self._stream_and_capture(
                self.primary_model,
                [{"role": "system", "content": refine_prompt}],
                tag="[SysPromptRefine]"
            ).strip()
            plan = json.loads(raw)
        except Exception:
            return None
        if not isinstance(plan, dict):
            return None

        action = plan.get("action")
        text   = (plan.get("prompt") or "").strip()

        # 7) Backup & apply (unchanged)
        backup = self.context_path + ".bak"
        try:
            shutil.copy(self.context_path, backup)
        except Exception:
            return None

        try:
            if action == "add" and text:
                patch = ContextObject.make_policy(
                    label=f"dynamic_prompt_add_{len(text)}",
                    policy_text=text,
                    tags=["dynamic_prompt"],
                )
                patch.touch()
                self.repo.save(patch)

            elif action == "remove" and text:
                for row in rows:
                    blob = row.metadata.get("prompt") or row.metadata.get("policy") or ""
                    if text in blob:
                        self.repo.delete(row.context_id)
            else:
                os.remove(backup)
                return None

            self._seed_static_prompts()

        except Exception:
            shutil.move(backup, self.context_path)
            return None

        # 8) Clean up & record
        try:
            os.remove(backup)
        except:
            pass

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
                

    def decision_callback(
        self,
        user_text: str,
        options: List[str],
        system_template: str,
        history_size: int,
        var_names: List[str]
    ) -> str:
        # 1) Build the mapping correctly with “:”
        mapping = {vn: opt for vn, opt in zip(var_names, options)}

        # 2) Format the system prompt
        system_msg = system_template.format(**mapping)

        # 3) Gather recent conversation
        segs = sorted(
            [c for c in self.repo.query(lambda c: c.domain=="segment"
                                       and c.semantic_label in ("user_input","assistant"))],
            key=lambda c: c.timestamp
        )[-history_size:]
        convo = "\n".join(
            ("User: " if c.semantic_label=="user_input" else "Assistant: ")
            + c.summary
            for c in segs
        )
        user_msg = f"=== Recent Conversation ===\n{convo}\n\n=== New Message ===\n{user_text}"

        # 4) Call the model
        resp = ""
        for part in chat(
            model=self.decision_model,
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user",  "content":user_msg}
            ],
            stream=True
        ):
            resp += part["message"]["content"]

        # 5) Record Q&A
        q = ContextObject.make_stage("decision_question", [], {"prompt": system_msg+"\n\n"+user_msg})
        q.component = "decision"; q.semantic_label = "question"; q.touch(); self.repo.save(q)
        a = ContextObject.make_stage("decision_answer", [q.context_id], {"answer": resp.strip()})
        a.component = "decision"; a.semantic_label = "answer"; a.touch(); self.repo.save(a)

        # 6) Return the first matching option
        for opt in options:
            if re.search(rf"\b{re.escape(opt)}\b", resp, re.I):
                return opt
        return resp.strip()
    
    def filter_callback(self, user_text: str) -> bool:
        choice = self.decision_callback(
            user_text=user_text,
            options=["YES","NO"],
            system_template=(
                "You are an attentive meta-reasoner.  Decide whether to respond now.  "
                "Answer exactly {arg1} or {arg2}."
            ),
            history_size=8,
            var_names=["arg1","arg2"]
        )
        return choice.upper() == "YES"


    def tools_callback(self, user_text: str) -> bool:
        choice = self.decision_callback(
            user_text=user_text,
            options=["TOOLS","NO_TOOLS"],
            system_template=(
                "You are a lightweight judge.  Decide if external tools or searches are required.  "
                "Answer exactly {arg1} or {arg2}."
            ),
            history_size=6,
            var_names=["arg1","arg2"]
        )
        return choice.upper() == "TOOLS"
    
    def run_with_meta_context(
        self,
        user_text: str,
        status_cb: Callable[[str, Any], None] = lambda *a: None,
    ) -> str:
        from context import sanitize_jsonl
        from datetime import datetime
        import logging

        # sanitize JSONL store
        sanitize_jsonl(self.context_path)

        # short-circuit empty
        if not user_text or not user_text.strip():
            status_cb("output", "")
            return ""

        state: Dict[str, Any] = {
            "user_text":  user_text,
            "errors":     [],
            "tool_ctxs":  [],
            "recent_ids": [],
        }

        def _record_perf(stage: str, summary: str, ok: bool, ctx: ContextObject = None):
            duration = (datetime.utcnow() - t0).total_seconds()
            refs = [state["user_ctx"].context_id] if "user_ctx" in state else []
            perf = ContextObject.make_stage(
                "stage_performance",
                refs,
                {"stage": stage, "duration": duration, "error": not ok}
            )
            perf.touch()
            self.repo.save(perf)
            status_cb(stage, summary)

        # ─── Stage 0: decision_to_respond ────────────────────────────────
        t0 = datetime.utcnow()
        try:
            should = self.filter_callback(user_text)
            state["should_respond"] = should
            _record_perf("decision_to_respond", str(should), True)
            if not should:
                status_cb("output", "")
                return ""
        except Exception as e:
            _record_perf("decision_to_respond", str(e), False)
            # proceed anyway if this fails

        # ─── Stage 0.5: decide if we need tool calls / planning ───────────
        t0 = datetime.utcnow()
        try:
            use_tools = self.tools_callback(user_text)
            state["use_tools"] = use_tools
            _record_perf("decide_tool_usage", str(use_tools), True)
        except Exception as e:
            state["use_tools"] = True        # default to full pipeline
            _record_perf("decide_tool_usage", str(e), False)

        # ─── Fast‐path when tools not needed ──────────────────────────────
        if not state["use_tools"]:
            # build a minimal context for a one‐shot reply
            recent = self.repo.query(lambda c: c.domain=="segment")[-10:]
            history = [{"role":"user" if c.semantic_label=="user_input" else "assistant",
                        "content":c.summary} for c in recent]
            # append the new user_text
            history.append({"role":"user","content":user_text})
            # one‐shot inference
            from ollama import chat
            reply = ""
            for chunk in chat(model=self.primary_model, messages=history, stream=True):
                reply += chunk["message"]["content"]
                status_cb("fast_reply", reply)
            status_cb("output", reply)
            return reply

        # ─── Stage 1: record_input ───────────────────────────────────────
        t0 = datetime.utcnow()
        try:
            ctx1 = self._stage1_record_input(user_text)
            state["user_ctx"] = ctx1
            _record_perf("record_input", ctx1.summary, True, ctx1)
        except Exception as e:
            _record_perf("record_input", str(e), False)
            state["errors"].append(("record_input", str(e)))

        # ─── Stage 2: load_system_prompts ───────────────────────────────
        t0 = datetime.utcnow()
        try:
            ctx2 = self._stage2_load_system_prompts()
            state["sys_ctx"] = ctx2
            _record_perf("load_system_prompts", "(loaded)", True, ctx2)
        except Exception as e:
            _record_perf("load_system_prompts", str(e), False)
            state["errors"].append(("load_system_prompts", str(e)))

        # ─── Stage 3: retrieve_and_merge_context ────────────────────────
        t0 = datetime.utcnow()
        try:
            extra = self._get_history()
            out3 = self._stage3_retrieve_and_merge_context(
                user_text, state["user_ctx"], state["sys_ctx"], extra_ctx=extra
            )
            state.update(out3)
            _record_perf("retrieve_and_merge_context", "(merged)", True)
        except Exception as e:
            _record_perf("retrieve_and_merge_context", str(e), False)
            state["errors"].append(("retrieve_and_merge_context", str(e)))

        # ─── Stage 4: intent_clarification ──────────────────────────────
        t0 = datetime.utcnow()
        try:
            ctx4 = self._stage4_intent_clarification(user_text, state)
            state["clar_ctx"] = ctx4
            _record_perf("intent_clarification", ctx4.summary, True, ctx4)
        except Exception as e:
            _record_perf("intent_clarification", str(e), False)
            state["errors"].append(("intent_clarification", str(e)))
            # ── Minimal fallback so clar_ctx always exists ────────────
            dummy = ContextObject.make_stage(
                "intent_clarification_failed",
                [ state["user_ctx"].context_id ] if "user_ctx" in state else [],
                {"summary": ""}
            )
            dummy.touch()
            self.repo.save(dummy)
            state["clar_ctx"] = dummy
            
        # ─── Stage 5: external_knowledge ────────────────────────────────
        t0 = datetime.utcnow()
        try:
            ctx5 = self._stage5_external_knowledge(state["clar_ctx"])
            state["know_ctx"] = ctx5
            _record_perf("external_knowledge", "(snippets)", True, ctx5)
        except Exception as e:
            _record_perf("external_knowledge", str(e), False)
            state["errors"].append(("external_knowledge", str(e)))
            dummy = ContextObject.make_stage(
                "external_knowledge_retrieval",
                state["clar_ctx"].references,
                {"snippets": []}
            )
            dummy.stage_id = "external_knowledge_retrieval"
            dummy.summary  = "(no snippets)"
            dummy.touch(); self.repo.save(dummy)
            state["know_ctx"] = dummy

        # ─── Stage 6: prepare_tools ────────────────────────────────────
        t0 = datetime.utcnow()
        try:
            tools = self._stage6_prepare_tools()
            state["tools_list"] = tools
            _record_perf("prepare_tools", f"{len(tools)} tools", True)
        except Exception as e:
            _record_perf("prepare_tools", str(e), False)
            state["errors"].append(("prepare_tools", str(e)))

        # ─── Stage 7: planning_summary ──────────────────────────────────
        t0 = datetime.utcnow()
        try:
            logging.debug("Planning with tools: %s", [t["name"] for t in state.get("tools_list", [])])
            ctx7, plan_out = self._stage7_planning_summary(
                state.get("clar_ctx"),
                state.get("know_ctx"),
                state.get("tools_list", []),
                user_text
            )
            state["plan_ctx"]    = ctx7
            state["plan_output"] = plan_out
            _record_perf("planning_summary", "(planned)", True, ctx7)
        except Exception as e:
            _record_perf("planning_summary", str(e), False)
            state["errors"].append(("planning_summary", str(e)))

        # ─── Stage 7b: plan_validation ────────────────────────────────
        t0 = datetime.utcnow()
        try:
            _, _, fixed = self._stage7b_plan_validation(
                state["plan_ctx"], state["plan_output"], state.get("tools_list", [])
            )
            state["fixed_calls"] = fixed
            _record_perf("plan_validation", f"{len(fixed)} calls", True)
        except Exception as e:
            _record_perf("plan_validation", str(e), False)
            state["errors"].append(("plan_validation", str(e)))

        # ─── Stage 8: tool_chaining ────────────────────────────────────
        t0 = datetime.utcnow()
        try:
            tc_ctx, raw_calls, schemas = self._stage8_tool_chaining(
                state["plan_ctx"],
                "\n".join(state.get("fixed_calls", [])),
                state.get("tools_list", [])
            )
            state.update({"tc_ctx": tc_ctx, "raw_calls": raw_calls, "schemas": schemas})
            _record_perf("tool_chaining", f"{len(raw_calls)} calls", True, tc_ctx)
        except Exception as e:
            _record_perf("tool_chaining", str(e), False)
            state["errors"].append(("tool_chaining", str(e)))

        # ─── Stage 8.5: user_confirmation ──────────────────────────────
        t0 = datetime.utcnow()
        try:
            confirmed = self._stage8_5_user_confirmation(state.get("raw_calls", []), user_text)
            state["confirmed_calls"] = confirmed
            _record_perf("user_confirmation", "(auto-approved)", True)
        except Exception as e:
            _record_perf("user_confirmation", str(e), False)
            state["errors"].append(("user_confirmation", str(e)))

        # ─── Stage 9: invoke_with_retries ─────────────────────────────
        t0 = datetime.utcnow()
        try:
            tcs = self._stage9_invoke_with_retries(
                state.get("confirmed_calls", []),
                state.get("plan_output", ""),
                state.get("schemas", []),
                user_text,
                state["clar_ctx"].metadata
            )
            state["tool_ctxs"] = tcs
            _record_perf("invoke_with_retries", f"{len(tcs)} runs", True)
        except Exception as e:
            _record_perf("invoke_with_retries", str(e), False)
            state["errors"].append(("invoke_with_retries", str(e)))

        # ─── Stage 9b: reflection_and_replan ──────────────────────────
        t0 = datetime.utcnow()
        try:
            rp = self._stage9b_reflection_and_replan(
                state["tool_ctxs"],
                state.get("plan_output", ""),
                user_text,
                state["clar_ctx"].metadata
            )
            state["replan"] = rp
            _record_perf("reflection_and_replan", f"replan={bool(rp)}", True)
        except Exception as e:
            _record_perf("reflection_and_replan", str(e), False)
            state["errors"].append(("reflection_and_replan", str(e)))

        # ─── Stage 10: assemble_and_infer ─────────────────────────────
        t0 = datetime.utcnow()
        try:
            draft = self._stage10_assemble_and_infer(user_text, state)
            state["draft"] = draft
            _record_perf("assemble_and_infer", "(drafted)", True)
        except Exception as e:
            _record_perf("assemble_and_infer", str(e), False)
            state["errors"].append(("assemble_and_infer", str(e)))

        # ─── Stage 10b: response_critique_and_safety ───────────────────
        t0 = datetime.utcnow()
        try:
            final = self._stage10b_response_critique_and_safety(
                state.get("draft", ""), user_text, state.get("tool_ctxs", [])
            )
            state["final"] = final
            _record_perf("response_critique", "(polished)", True)
        except Exception as e:
            _record_perf("response_critique", str(e), False)
            state["errors"].append(("response_critique", str(e)))

        # ─── Stage 11: memory_writeback ───────────────────────────────
        t0 = datetime.utcnow()
        try:
            self._stage11_memory_writeback(state.get("final", ""), state.get("tool_ctxs", []))
            _record_perf("memory_writeback", "(queued)", True)
        except Exception as e:
            _record_perf("memory_writeback", str(e), False)
            state["errors"].append(("memory_writeback", str(e)))

        # one-off self-review & narrative-mull
        try:    self._stage_system_prompt_refine(state)
        except: pass
        try:    self._stage_narrative_mull(state)
        except: pass

        # final output
        out = state.get("final", "").strip()
        status_cb("output", out)
        return out


    def _background_self_review(self):
        import time
        while not self._stop_self_review.is_set():
            try:
                # make a copy so we don’t mutate the live state
                state = getattr(self, "_last_state", {}).copy()
                # run the two meta-stages continuously
                self._stage_system_prompt_refine(state)
                self._stage_narrative_mull(state)
            except Exception:
                pass
            # wait before next self-talk cycle
            time.sleep(10)


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
        Merge system prompts, narrative, dynamic chat history,
        semantic/memory/tool contexts — with three user‐driven decisions for:
          • how many history entries to include
          • how far back in time to look
          • which tags to use for similarity search
        Falls back to at least the last 5 minutes, last 10 interactions,
        and default high-priority tags if decisions fail.
        """
        import re
        from datetime import timedelta
        import concurrent.futures

        now = default_clock()

        # ── Step A: dynamically decide context parameters via decision_callback ──
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut_count = executor.submit(
                self.decision_callback,
                user_text,
                ["5", "10", "20", "50"],
                "Choose how many recent chat entries to include: {arg1}, {arg2}, {arg3}, or {arg4}.",
                timeout=2.0,
                arg_names=["arg1","arg2","arg3","arg4"]
            )
            fut_window = executor.submit(
                self.decision_callback,
                user_text,
                ["5 minutes", "30 minutes", "1 hour", "6 hours", "1 day"],
                "Choose a time window for context: {arg1}, {arg2}, {arg3}, {arg4}, or {arg5}.",
                timeout=2.0,
                arg_names=["arg1","arg2","arg3","arg4","arg5"]
            )
            fut_tags = executor.submit(
                self.decision_callback,
                user_text,
                ["important", "decision_point", "task", "question"],
                "Choose relevance tags (comma-separated) from: {arg1}, {arg2}, {arg3}, {arg4}.",
                timeout=2.0,
                arg_names=["arg1","arg2","arg3","arg4"]
            )

            # parse number of entries, fallback to 10
            try:
                cnt = fut_count.result()
                num_entries = max(1, int(re.search(r"\d+", cnt).group()))
            except Exception:
                num_entries = 10

            # parse time window, fallback to 5 minutes
            try:
                window = fut_window.result()
                m = re.match(r"(\d+)\s*(minute|hour|day)s?", window)
                if m:
                    val, unit = int(m.group(1)), m.group(2)
                    delta = {
                        "minute": timedelta(minutes=val),
                        "hour":   timedelta(hours=val),
                        "day":    timedelta(days=val),
                    }[unit]
                else:
                    delta = timedelta(minutes=5)
            except Exception:
                delta = timedelta(minutes=5)

            # parse similarity tags, fallback to high-priority defaults
            try:
                tags = fut_tags.result()
                sim_tags = [t.strip() for t in tags.split(",") if t.strip()]
            except Exception:
                sim_tags = ["important", "decision_point"]

        # ── Step B: compute recall feature for RL gating ─────────────────────
        rf = 0.0
        if recall_ids:
            counts = [
                self.repo.get(cid).recall_stats.get("count", 0)
                for cid in recall_ids
                if cid and self.repo.get(cid)
            ]
            rf = sum(counts) / len(counts) if counts else 0.0

        # ── Step C: load & slice recent narrative entries ────────────────────
        full_narr = self._load_narrative_context()
        all_narr = sorted(
            [c for c in self.repo.query(lambda c: c.component=="narrative")],
            key=lambda c: c.timestamp, reverse=True
        )[:5]
        all_narr.reverse()
        full_narr.summary = "\n".join(n.summary for n in all_narr)
        narr_ctx = full_narr

        # ── Step D: harvest working memory (raw segments & final inferences) ─
        WM = 5
        raw_segments = sorted(
            [c for c in self.repo.query(
                lambda c: c.domain=="segment"
                          and c.semantic_label in ("user_input","assistant")
            )],
            key=lambda c: c.timestamp
        )[-WM:]
        inferences = sorted(
            [c for c in self.repo.query(
                lambda c: c.semantic_label=="final_inference"
            )],
            key=lambda c: c.timestamp
        )[-WM:]
        # extract summaries
        wm_segments = [c.summary for c in raw_segments]
        wm_infers   = [c.summary for c in inferences]

        # ── Step E: time-+-count-filtered chat history ───────────────────────
        cutoff = now - delta
        recent_segs = sorted(
            [c for c in self.repo.query(
                lambda c: c.domain=="segment"
                          and c.semantic_label in ("user_input","assistant")
                          and c.timestamp >= cutoff
            )],
            key=lambda c: c.timestamp
        )
        history = recent_segs[-num_entries:]
        if extra_ctx:
            history.extend(extra_ctx)

        # ── Step F: semantic retrieval using sim_tags ───────────────────────
        tr = (
            (now - delta).strftime("%Y%m%dT%H%M%SZ"),
            now.strftime("%Y%m%dT%H%M%SZ")
        )
        if self.rl.should_run("semantic_retrieval", rf):
            recent = self.engine.query(
                stage_id="recent_retrieval",
                time_range=tr,
                similarity_to=user_text,
                include_tags=sim_tags,
                exclude_tags=self.STAGES + ["tool_schema","tool_output","assistant","system_prompt"],
                top_k=self.top_k
            )
        else:
            recent = []

        # ── Step G: associative memory via spread-activation ───────────────
        if self.rl.should_run("memory_retrieval", rf):
            seeds = [user_ctx.context_id, narr_ctx.context_id]
            scores = self.memman.spread_activation(
                seed_ids=seeds,
                hops=3, decay=0.6,
                assoc_weight=1.0, recency_weight=1.0
            )
            top_ids = sorted(scores, key=scores.get, reverse=True)[:self.top_k]
            assoc = []
            for cid in top_ids:
                try:
                    cobj = self.repo.get(cid)
                    cobj.retrieval_score = scores[cid]
                    cobj.retrieval_metadata = {"seed_ids": seeds}
                    cobj.record_recall(
                        stage_id="memory_retrieval",
                        coactivated_with=seeds,
                        retrieval_score=scores[cid]
                    )
                    self.repo.save(cobj)
                    assoc.append(cobj)
                except KeyError:
                    continue
        else:
            assoc = []

        # ── Step H: recent tool outputs ────────────────────────────────────
        tools = sorted(
            [c for c in self.repo.query(lambda c: c.component=="tool_output")],
            key=lambda c: c.timestamp
        )
        recent_tools = tools[-self.top_k:] if self.rl.should_run("tool_output_retrieval", rf) else []

        # ── Step I: interleave all buckets into merged list ────────────────
        merged: List[ContextObject] = []
        seen = set()
        # 1) system prompt
        for c in (sys_ctx,):
            merged.append(c); seen.add(c.context_id)
        # 2) narrative
        if narr_ctx.context_id not in seen:
            merged.append(narr_ctx); seen.add(narr_ctx.context_id)
        # 3) working memory segments
        for c in raw_segments:
            if c.context_id not in seen:
                merged.append(c); seen.add(c.context_id)
        # 4) working memory inferences
        for c in inferences:
            if c.context_id not in seen:
                merged.append(c); seen.add(c.context_id)
        # 5) filtered history
        for c in history:
            if c.context_id not in seen:
                merged.append(c); seen.add(c.context_id)
        # 6) semantic retrieval
        for c in recent:
            if c.context_id not in seen:
                merged.append(c); seen.add(c.context_id)
        # 7) associative memory
        for c in assoc:
            if c.context_id not in seen:
                merged.append(c); seen.add(c.context_id)
        # 8) tool outputs
        for c in recent_tools:
            if c.context_id not in seen:
                merged.append(c); seen.add(c.context_id)

        merged_ids = [c.context_id for c in merged]

        # ── Step J: debug print of all context buckets ─────────────────────
        self._print_stage_context("context_merge", {
            "system":    [sys_ctx.summary],
            "narrative": [n.summary for n in all_narr],
            "working_segments": wm_segments,
            "working_inferences": wm_infers,
            "history":   [f"{c.semantic_label}: {c.summary}" for c in history],
            "semantic":  [f"[Tags:{','.join(c.tags)}] {c.summary}" for c in recent],
            "memory":    [f"[Score:{getattr(c,'retrieval_score',0):.2f}] {c.summary}" for c in assoc],
            "tool_outputs": [c.summary for c in recent_tools],
        })

        return {
            "narrative_ctx": narr_ctx,
            "history":       history,
            "recent":        recent,
            "assoc":         assoc,
            "recent_ids":    merged_ids,
        }



    # ── Stage 4: Intent Clarification ─────────────────────────────────────
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

        out = self._stream_and_capture(self.secondary_model, msgs, tag="[Clarifier]")
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
                        tag="[ClarifierRetry]"
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
    def _stage5_external_knowledge(self, clar_ctx: ContextObject) -> ContextObject:
        import json
        import concurrent.futures
        from datetime import datetime, timedelta

        now = default_clock()

        # ── 0) Prune stale/overflow contexts ──────────────────────────────────
        prune_summary = self._stage_prune_context_store({})

        # ── 1) In parallel, gather:
        #    A) recent segments (user_input/assistant)
        #    B) recent final_inference stages
        #    C) similarity hits on clarifier keywords
        #    D) local summary‐match hits
        with concurrent.futures.ThreadPoolExecutor() as exec:
            fut_seg = exec.submit(
                lambda: sorted(
                    [c for c in self.repo.query(
                        lambda c: c.domain=="segment" and c.semantic_label in ("user_input","assistant")
                    )],
                    key=lambda c: c.timestamp, reverse=True
                )[: self.top_k]
            )
            fut_inf = exec.submit(
                lambda: sorted(
                    [c for c in self.repo.query(
                        lambda c: c.semantic_label=="final_inference"
                    )],
                    key=lambda c: c.timestamp, reverse=True
                )[: self.top_k]
            )
            fut_sim = exec.submit(
                lambda: [
                    h for kw in clar_ctx.metadata.get("keywords", [])
                    for h in self.engine.query(
                        stage_id="external_knowledge_retrieval",
                        similarity_to=kw,
                        top_k=self.top_k
                    )
                ]
            )
            fut_loc = exec.submit(
                lambda: [
                    c for c in self.repo.query(
                        lambda c: (
                            c.semantic_label in ("user_input","assistant","final_inference")
                            and datetime.fromisoformat(c.timestamp.rstrip("Z")) >= now - timedelta(hours=1)
                            and any(kw.lower() in (c.summary or "").lower() for kw in clar_ctx.metadata.get("keywords", []))
                        )
                    )
                ]
            )

        segs = fut_seg.result(timeout=2.0) if fut_seg else []
        infs = fut_inf.result(timeout=2.0) if fut_inf else []
        sims = fut_sim.result(timeout=2.0) if fut_sim else []
        locs = fut_loc.result(timeout=2.0) if fut_loc else []

        # ── 2) Build working memory entries
        working_memory = []
        for c in reversed(segs):
            working_memory.append(f"(WM) [{c.semantic_label}] {c.summary}")
        for c in reversed(infs):
            working_memory.append(f"(WM) [{c.semantic_label}] {c.summary}")

        # ── 3) Build similarity snippets
        sim_snips = [f"(EXT) [{','.join(h.tags)}] {h.summary}" for h in sims]

        # ── 4) Build local‐match snippets
        loc_snips = []
        for c in locs:
            tag = "SEG" if c.domain=="segment" else "INF"
            loc_snips.append(f"(LOC-{tag}) [{c.semantic_label}] {c.summary}")

        # ── 5) Fallback to last 5 segments if nothing collected
        if not (working_memory or sim_snips or loc_snips):
            fallback = sorted(
                [c for c in self.repo.query(
                    lambda c: c.domain=="segment" and c.semantic_label in ("user_input","assistant")
                )],
                key=lambda c: c.timestamp, reverse=True
            )[:5]
            for c in reversed(fallback):
                working_memory.append(f"(FB) [{c.semantic_label}] {c.summary}")

        # ── 6) Combine and dedupe, preserving order
        all_snips = working_memory + sim_snips + loc_snips
        seen = set(); unique = []
        for s in all_snips:
            if s not in seen:
                seen.add(s); unique.append(s)

        # ── 7) Persist and return
        ctx = ContextObject.make_stage(
            "external_knowledge_retrieval",
            clar_ctx.references,
            {"snippets": unique}
        )
        ctx.stage_id = "external_knowledge_retrieval"
        ctx.summary  = "\n".join(unique) or "(none)"
        ctx.touch()
        self.repo.save(ctx)

        # ── 8) Debug print
        self._print_stage_context("external_knowledge_retrieval", {
            "pruned":         prune_summary,
            "working_memory": working_memory or ["(none)"],
            "similarity":     sim_snips or ["(none)"],
            "local_matches":  loc_snips or ["(none)"],
        })

        return ctx


    # ────────────────────────────────────────────────────────────────────────────────
    # Stage 6: Gather & Dedupe Tool Schemas
    # ────────────────────────────────────────────────────────────────────────────────
    def _stage6_prepare_tools(self) -> List[Dict[str, str]]:
        """
        Return a de-duplicated, lexicographically sorted list of:
        { "name": "<tool_name>", "description": "<one-line desc>" }
        for every tool_schema in the repo.
        """
        import json

        # 1) Load every schema context object tagged "tool_schema"
        rows = self.repo.query(
            lambda c: c.component == "schema" and "tool_schema" in c.tags
        )

        # 2) Keep only the newest per tool name
        buckets: dict[str, ContextObject] = {}
        for ctx in rows:
            try:
                blob = json.loads(ctx.metadata["schema"])
                name = blob["name"]
            except Exception:
                continue
            if name not in buckets or ctx.timestamp > buckets[name].timestamp:
                buckets[name] = ctx

        # 3) Build the list, sorted by tool name
        tool_defs: list[dict[str, str]] = []
        for name in sorted(buckets):
            blob = json.loads(buckets[name].metadata["schema"])
            desc = blob.get("description", "").split("\n", 1)[0]
            tool_defs.append({"name": name, "description": desc})

        return tool_defs


    # ────────────────────────────────────────────────────────────────────────────────
    # Stage 7: Planning Summary with Tool-List Injection & Strict Validation
    # ────────────────────────────────────────────────────────────────────────────────
    def _stage7_planning_summary(
        self,
        clar_ctx: ContextObject,
        know_ctx: ContextObject,
        tools_list: List[Dict[str, str]],
        user_text: str,
    ) -> Tuple[ContextObject, str]:
        import json, re, hashlib, datetime
        from typing import Any, Dict, List, Tuple

        # Helper to strip out a bare JSON block
        def _clean_json_block(text: str) -> str:
            m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
            if m:
                return m.group(1)
            m2 = re.search(r"(\{.*\})", text, flags=re.S)
            return m2.group(1) if m2 else text.strip()

        # 0) Load prior plan critiques for optional feedback (unchanged)
        critique_rows = self.repo.query(
            lambda c: c.component == "analysis" and c.semantic_label == "plan_critique"
        )
        critique_rows.sort(key=lambda c: c.timestamp)
        critique_block = (
            "\n".join(f"• {c.metadata.get('critique','')}" for c in critique_rows)
            if critique_rows else "(no prior critiques)"
        )
        critique_ids = [c.context_id for c in critique_rows]

        # 1) Build the strict system prompt + injected tool list
        #    planning_prompt ends with "Available tools:\n"
        tool_lines = "\n".join(f"- **{t['name']}**: {t['description']}"
                                for t in tools_list) or "(none)"
        base_system = (
            self._get_prompt("planning_prompt")
            + tool_lines
        )

        # 2) Build the user prompt
        base_user = (
            f"User question:\n{user_text}\n\n"
            f"Clarified intent:\n{clar_ctx.summary}\n\n"
            f"Snippets:\n{know_ctx.summary or '(none)'}"
        )

        last_calls = None
        plan_obj = None
        cleaned = ""

        # 3) Up to 3 passes to produce valid JSON & exact tool names
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
                        "Your previous plan used invalid or non-exact tool names.  "
                        "Please use only the EXACT names listed and output only the required JSON."},
                    {"role": "user",   "content": cleaned},
                ]

            raw = self._stream_and_capture(self.secondary_model, msgs, tag=tag)
            cleaned = _clean_json_block(raw)

            # Attempt to parse
            try:
                cand = json.loads(cleaned)
                assert isinstance(cand, dict) and "tasks" in cand
                plan_obj = cand
            except:
                # Fallback: regex extract any calls (won't match exact names, but let validation catch)
                calls = re.findall(r'\b[A-Za-z_]\w*\([^)]*\)', raw)
                plan_obj = {"tasks": [
                    {"call": c, "tool_input": {}, "subtasks": []} for c in calls
                ]}

            # 4) Reject any non-exact tool names
            valid_names = {t["name"] for t in tools_list}
            unknown = [t["call"] for t in plan_obj["tasks"] if t["call"] not in valid_names]
            if unknown:
                self._print_stage_context(
                    f"planning_summary:unknown_tools(attempt={attempt})",
                    {"unknown": unknown, "allowed": sorted(valid_names)}
                )
                continue

            # 5) Plateau guard
            this_calls = [t["call"] for t in plan_obj["tasks"]]
            if last_calls is not None and this_calls == last_calls:
                self._print_stage_context(
                    f"planning_summary:plateaued(attempt={attempt})",
                    {"calls": this_calls}
                )
                break
            last_calls = this_calls
            break

        # 6) Flatten subtasks → list of calls
        def _flatten(task: Dict[str, Any]) -> List[Dict[str, Any]]:
            out = [task]
            for sub in task.get("subtasks", []):
                out.extend(_flatten(sub))
            return out

        flat = []
        for t in plan_obj.get("tasks", []):
            flat.extend(_flatten(t))

        call_strings = []
        for task in flat:
            name = task["call"]
            params = task.get("tool_input", {}) or {}
            if params:
                args = ",".join(f'{k}={json.dumps(v, ensure_ascii=False)}'
                                for k, v in params.items())
                call_strings.append(f"{name}({args})")
            else:
                call_strings.append(f"{name}()")

        # 7) Serialize final plan & compute short ID
        plan_json = json.dumps({"tasks": [
            {"call": s, "subtasks": []} for s in call_strings
        ]})
        plan_sig = hashlib.md5(plan_json.encode("utf-8")).hexdigest()[:8]

        # 8) Persist planning_summary context
        ctx = ContextObject.make_stage(
            "planning_summary",
            clar_ctx.references + know_ctx.references + critique_ids,
            {"plan": plan_obj, "attempt": attempt, "plan_id": plan_sig}
        )
        ctx.stage_id = f"planning_summary_{plan_sig}"
        ctx.summary = plan_json
        ctx.touch(); self.repo.save(ctx)

        # Signal success/failure for RL
        if call_strings:
            succ = ContextObject.make_success(
                f"Planner → {len(call_strings)} task(s)", refs=[ctx.context_id]
            )
        else:
            succ = ContextObject.make_failure(
                "Planner → empty plan", refs=[ctx.context_id]
            )
        succ.stage_id = f"planning_summary_signal_{plan_sig}"
        succ.touch(); self.repo.save(succ)

        # 9) Initialize plan tracker
        tracker = ContextObject.make_stage(
            "plan_tracker",
            [ctx.context_id],
            {
                "plan_id": plan_sig,
                "total_calls": len(call_strings),
                "succeeded": 0,
                "attempts": 0,
                "status": "in_progress",
                "started_at": datetime.datetime.utcnow().isoformat() + "Z"
            }
        )
        tracker.semantic_label = plan_sig
        tracker.stage_id = f"plan_tracker_{plan_sig}"
        tracker.summary = "initialized plan tracker"
        tracker.touch(); self.repo.save(tracker)

        return ctx, plan_json


    # ── Stage 7b: Plan Validation with Full-Docstring Injection ────────────────────────────────────────────────────
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
                tag="[PlanFix]"
            ).strip()

            try:
                plan_obj = json.loads(repair)
                tasks    = plan_obj.get("tasks", [])
            except Exception:
                break

        # G) Re-serialize every task into a real call string
        fixed_calls = []
        for task in tasks:
            name = task["call"]
            ti   = task.get("tool_input", {}) or {}
            args = ",".join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in ti.items())
            fixed_calls.append(f"{name}({args})")

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
        prev_pending = None
        for attempt in range(1, max_retries + 1):
            errors: List[Tuple[str, str]] = []

            # plateau detection: if the LLM returns the exact same list twice, give up
            if prev_pending is not None and pending == prev_pending:
                logging.warning(f"[ToolChainRetry] plateau on attempt {attempt}, giving up")
                break
            prev_pending = pending.copy()

            # shuffle the order to introduce stochasticity
            random.shuffle(pending)

            for original in list(pending):
                call_str = original

                # 1) [alias from tool_name] style placeholders
                for ph in re.findall(r"\[([^\]]+)\]", call_str):
                    if " from " in ph:
                        alias, toolname = ph.split(" from ", 1)
                        alias_key = normalize_key(alias)
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

                # 3) inline nested zero-arg calls only if embedded (not top-level)
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
                last_results[tool_key] = res["output"]

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
                ctx.stage_id = f"tool_output_{name}"
                ctx.summary = str(res.get("output")) if ok else f"ERROR: {err}"
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

            # if everything succeeded, finish
            if not pending:
                tracker.metadata["status"] = "success"
                tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
                tracker.touch(); self.repo.save(tracker)
                return tool_ctxs

            # otherwise ask the LLM to repair only the remaining calls, and show it the errors
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
                self.secondary_model, retry_msgs, tag="[ToolChainRetry]"
            ).strip()
            try:
                pending = json.loads(out)["tool_calls"]
            except:
                parsed = Tools.parse_tool_call(out)
                pending = _norm(parsed if isinstance(parsed, list) else [parsed] or pending)

        # ── 3) If we exit the loop with some calls still pending, mark failure ──
        if pending:
            tracker.metadata["status"] = "failed"
            tracker.metadata["errors_by_call"].update({c: "unresolved" for c in pending})
            tracker.metadata["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            tracker.touch(); self.repo.save(tracker)

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
        Also record the assistant’s reply as a segment for future context.
        """
        import json
        from context import ContextObject  # adjust import as needed

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
        # include any tool calls
        for t in state.get("tool_ctxs", []):
            refs.append(t.context_id)

        # de-dup while preserving order
        seen, ordered = set(), []
        for cid in refs:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)

        # ──────────────────────────────────────────────────────────────────────
        # 2) Load the ContextObjects, skipping stale ones
        # ──────────────────────────────────────────────────────────────────────
        ctxs = []
        for cid in ordered:
            try:
                ctxs.append(self.repo.get(cid))
            except KeyError:
                continue
        ctxs.sort(key=lambda c: c.timestamp)

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
        self._last_plan_output = plan_text  # for critic stage

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

        # ──────────────────────────────────────────────────────────────────────
        # 6) **NEW**: also record this reply as an assistant segment for next turn
        # ──────────────────────────────────────────────────────────────────────
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



    # ── Stage 10b: Response Critique & Safety ─────────────────────────────────────────────
    def _stage10b_response_critique_and_safety(
        self,
        draft: str,
        user_text: str,
        tool_ctxs: List[ContextObject]
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
        polished = self._stream_and_capture(self.secondary_model, msgs, tag="[Critic]").strip()

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
                        tag=f"[NarrativeAnswer_{idx}]"
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
