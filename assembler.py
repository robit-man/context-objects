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
logger = logging.getLogger("assembler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] assembler: %(message)s",
)

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

    """
    Fractal Assembler with recursive TaskExecutor and on-disk RL for stage selection
    plus an RL-driven “curiosity” probe system for gap‐closing questions.
    """
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
        self._chat_contexts: dict[int, str] = {}
        # will be set by telegram_input
        self._telegram_bot = None

        # — auto-discover any _stage_<name>() methods as “optional” —
        all_methods = {name for name, _ in inspect.getmembers(self, inspect.ismethod)}
        discovered = [
            s for s in self.STAGES + ["curiosity_probe"]
            if f"_stage_{s}" in all_methods
        ]

        # allow config override, else use discovered
        self._optional_stages = self.cfg.get("rl_optional", discovered)

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
            # insert two example probe templates if none exist
            defaults = {
                "curiosity_template_missing_notes":
                  "I’m not quite sure what you meant by: «{snippet}». Can you clarify?",
                "curiosity_template_missing_date":
                  "You mentioned a date but didn’t specify which one—what date are you thinking of?"
            }
            for label, text in defaults.items():
                tmpl = ContextObject.make_policy(
                    label=label,
                    policy_text=text,
                    tags=["dynamic_prompt","curiosity_template"]
                )
                tmpl.touch(); self.repo.save(tmpl)
                self.curiosity_templates.append(tmpl)

        for name, fn in inspect.getmembers(self, inspect.ismethod):
            if name.startswith("_stage_"):
                doc = fn.__doc__ or ""
                for hint in re.findall(r"requires\s+(\w+)", doc, flags=re.I):
                    label = f"curiosity_require_{hint.lower()}"
                    # only create each probe once
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

        # — RLController for curiosity‐template selection —
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
            

    def dump_architecture(self):
            import inspect, json

            arch = {
                "stages":               self.STAGES,
                "optional_stages":      self._optional_stages,
                "curiosity_templates":  [t.semantic_label for t in self.curiosity_templates],
                "rl_weights":           self.rl.weights,
                "curiosity_weights":    self.curiosity_rl.weights,
                "stage_methods":        {}
            }
            for s in self.STAGES + ["curiosity_probe"]:
                fn = getattr(self, f"_stage_{s}", None)
                if fn:
                    arch["stage_methods"][s] = {
                        "signature": str(inspect.signature(fn)),
                        "doc":       fn.__doc__,
                    }

            print(json.dumps(arch, indent=2))
    
    def _stage_curiosity_probe(self, state: Dict[str,Any]) -> List[str]:
        probes = []
        clar = state["clar_ctx"]
        # find “gaps” you care about
        gaps = []
        if not clar.metadata.get("notes"):
            gaps.append(("missing_notes", clar.summary[:50]))
        if "date(" in state.get("plan_output","") and not any(
            kw.lower().startswith("date") for kw in clar.metadata.get("keywords",[])
        ):
            gaps.append(("missing_date", "plan mentions a date"))

        # for each gap, pick the best template by RL
        for gap_name, snippet in gaps:
            # sample among all templates whose label contains our gap
            choices = [t for t in self.curiosity_templates if gap_name in t.semantic_label]
            if not choices:
                continue
            # pick one by its sigmoid(weight)
            picked = max(choices,
                        key=lambda t: self.curiosity_rl.probability(t.semantic_label)
            )
            prompt = picked.metadata["policy"].format(snippet=snippet)
            probes.append(prompt)

            # record that we “ran” this template
            state.setdefault("curiosity_used", []).append(picked.semantic_label)

            # save the question as a ContextObject so we can slot in the answer later
            ctx = ContextObject.make_stage(
                "curiosity",
                [clar.context_id],
                {"question": prompt},
                session_id=self.session_id
            )
            ctx.component = "curiosity"
            ctx.semantic_label = "question"
            ctx.tags.append("curiosity")
            ctx.touch(); self.repo.save(ctx)

        return probes


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
        self._chat_contexts[chat_id] = user_text

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
            
    def run_with_meta_context(
        self,
        user_text: str,
        status_cb: Callable[[str, Any], None] = lambda *a: None,
        chat_id:  Optional[int]               = None,
        msg_id:   Optional[int]               = None,
    ) -> str:
        """
        Orchestrate all stages.  Each stage calls status_cb(stage, summary)
        so you can stream updates to Telegram.  Returns the full assistant reply.
        """

        import json, textwrap
        from collections import deque
        from datetime import datetime

        # — Helper: your provided complexity metric —
        def _compute_query_complexity(user_text: str, clar_notes: str, keywords: list[str]) -> float:
            wc         = min(len(user_text.split()), 50) / 50.0
            note_bonus = min(len(clar_notes) / 200.0, 1.0)
            kw_bonus   = min(len(keywords)     /   5.0, 1.0)
            punc_count = sum(user_text.count(c) for c in "?,;!")
            punc_density = punc_count / max(1, len(user_text))
            punc_bonus   = min(punc_density * 5.0, 1.0)
            return 0.4 * wc + 0.2 * note_bonus + 0.2 * kw_bonus + 0.2 * punc_bonus

        # 1) Base pipeline
        queue_stages = deque([
            "record_input",
            "load_system_prompts",
            "retrieve_and_merge_context",
            "intent_clarification",
            "external_knowledge",
            "prepare_tools",
            "planning_summary",
            "plan_validation",
        ])

        # 2) State bag (including RL tracking)
        state = {
            "user_text":      user_text,
            "errors":         [],
            "curiosity_used": [],
            "rl_included":    [],   # which optional stages we actually ran
        }
        result = None
        self._last_errors = False

        # 1-b) RL-driven optional stages, with recall-stat bias
        for opt in self._optional_stages:
            if not hasattr(self, f"_stage_{opt}"):
                continue
            # compute a simple recall feature: average recall count of last contexts
            recall_ids = state.get("recent_ids", [])
            counts = []
            for cid in recall_ids:
                try:
                    ctx = self.repo.get(cid)
                    counts.append(ctx.metadata.get("recall_stats", {}).get("count", 0))
                except KeyError:
                    continue
            recall_feat = sum(counts) / len(counts) if counts else 0.0

            if self.rl.should_run(opt, recall_feat):
                queue_stages.append(opt)
                state["rl_included"].append(opt)

        # 3) Track chat for appiphany
        if chat_id is not None:
            self._chat_contexts.add(chat_id)

        # 4) Main loop
        while queue_stages:
            stage   = queue_stages.popleft()
            start_t = datetime.utcnow()
            ran_ok  = False
            summary = ""

            # notify “starting…”
            try:
                status_cb(stage, "…")
            except:
                pass

            try:
                if stage == "record_input":
                    ctx = self._stage1_record_input(state["user_text"])
                    state["user_ctx"] = ctx
                    summary = ctx.summary

                elif stage == "load_system_prompts":
                    ctx = self._stage2_load_system_prompts()
                    state["sys_ctx"] = ctx
                    summary = "(loaded prompts)"

                elif stage == "retrieve_and_merge_context":
                    recent = self._get_history()
                    out    = self._stage3_retrieve_and_merge_context(
                        state["user_text"], state["user_ctx"], state["sys_ctx"],
                        extra_ctx=recent
                    )
                    state.update(out)
                    summary = out.get("know_ctx", state["sys_ctx"]).summary

                elif stage == "intent_clarification":
                    ctx = self._stage4_intent_clarification(state["user_text"], state)
                    state["clar_ctx"] = ctx
                    summary = ctx.summary

                    # compute & emit complexity
                    notes      = ctx.metadata.get("notes", "")
                    keywords   = ctx.metadata.get("keywords", [])
                    complexity = _compute_query_complexity(state["user_text"], notes, keywords)
                    state["complexity"] = complexity
                    status_cb("complexity", f"{complexity:.2f}")

                    # branch by complexity
                    if complexity < 0.3:
                        queue_stages.extend([
                            "assemble_and_infer",
                            "memory_writeback",
                        ])
                    else:
                        queue_stages.extend([
                            "external_knowledge",
                            "prepare_tools",
                            "planning_summary",
                            "plan_validation",
                            "tool_chaining",
                            "invoke_with_retries",
                            "reflection_and_replan",
                            "assemble_and_infer",
                            "response_critique",
                            "memory_writeback",
                        ])

                elif stage == "curiosity_probe":
                    notes = state["clar_ctx"].metadata.get("notes", "")
                    summary = notes[:60] + ("…" if len(notes) > 60 else "")
                    if len(notes.strip()) < 20:
                        for tmpl in self.curiosity_templates:
                            if self.curiosity_rl.should_run(tmpl.semantic_label):
                                q        = tmpl.metadata.get("policy", tmpl.summary)\
                                             .format(snippet=notes or state["clar_ctx"].summary)
                                followup = self._stream_and_capture(
                                    self.primary_model,
                                    [{"role":"system","content":q}],
                                    tag="[Curiosity]"
                                ).strip()
                                updated  = notes + "\n" + followup
                                c        = state["clar_ctx"]
                                c.metadata["notes"] = updated
                                c.summary = updated
                                c.touch(); self.repo.save(c)
                                state["curiosity_used"].append(tmpl.semantic_label)
                                summary = "probe→" + followup[:60]
                                break

                elif stage == "external_knowledge":
                    ctx = self._stage5_external_knowledge(state["clar_ctx"])
                    state["know_ctx"] = ctx
                    summary = "(external retrieved)"

                elif stage == "prepare_tools":
                    lst = self._stage6_prepare_tools()
                    state["tools_list"] = lst
                    summary = f"{len(lst)} tools"

                elif stage == "planning_summary":
                    ctx, plan_str = self._stage7_planning_summary(
                        state["clar_ctx"], state["know_ctx"],
                        state["tools_list"], state["user_text"]
                    )
                    state["plan_ctx"], state["plan_output"] = ctx, plan_str
                    summary = plan_str.strip().replace("\n", " ")[:60] + "…"
                    try:
                        tree = json.loads(plan_str)
                        if any(t.get("subtasks") for t in tree.get("tasks", [])):
                            roots    = self._parse_task_tree(tree)
                            executor = TaskExecutor(
                                        self, user_text,
                                        state["clar_ctx"].metadata
                                    )
                            for node in roots:
                                executor.execute(node)
                    except:
                        pass

                elif stage == "plan_validation":
                    valid, errors, fixed = self._stage7b_plan_validation(
                        state["plan_ctx"], state["plan_output"], state["tools_list"]
                    )
                    if errors:
                        raise RuntimeError(f"Plan validation errors: {errors}")
                    state["fixed_calls"] = fixed
                    summary = f"{len(fixed)} calls"
                    queue_stages.clear()
                    queue_stages.extend([
                        "tool_chaining",
                        "invoke_with_retries",
                        "reflection_and_replan",
                        "assemble_and_infer",
                        "response_critique",
                        "memory_writeback",
                    ])

                elif stage == "tool_chaining":
                    tc, raw, schemas = self._stage8_tool_chaining(
                        state["plan_ctx"],
                        "\n".join(state["fixed_calls"]),
                        state["tools_list"],
                    )
                    state.update(tc_ctx=tc, raw_calls=raw, schemas=schemas)
                    summary = "chained"

                elif stage == "invoke_with_retries":
                    tctxs = self._stage9_invoke_with_retries(
                        state["raw_calls"], state["plan_output"],
                        state["schemas"], state["user_text"],
                        state["clar_ctx"].metadata,
                    )
                    state["tool_ctxs"] = tctxs
                    summary = f"{len(tctxs)} runs"

                elif stage == "reflection_and_replan":
                    replan = self._stage9b_reflection_and_replan(
                        state["tool_ctxs"], state["plan_output"],
                        state["user_text"], state["clar_ctx"].metadata,
                        state["plan_ctx"]
                    )
                    summary = f"replan={bool(replan)}"
                    if replan:
                        queue_stages.clear()
                        queue_stages.extend([
                            "planning_summary",
                            "plan_validation",
                        ])

                elif stage == "assemble_and_infer":
                    d = self._stage10_assemble_and_infer(
                                state["user_text"], state
                            )
                    state["draft"] = d
                    summary = d.strip().replace("\n"," ")[:60] + "…"

                elif stage == "response_critique":
                    f = self._stage10b_response_critique_and_safety(
                            state["draft"], state["user_text"],
                            state.get("tool_ctxs", [])
                        )
                    state["final"] = f
                    summary = f.strip().replace("\n"," ")[:60] + "…"

                elif stage == "memory_writeback":
                    final = state.get("final","") or ""
                    try:
                        self._stage11_memory_writeback(final, state.get("tool_ctxs", []))
                        self.memman.decay_and_promote()
                    except Exception as e:
                        logger.warning(f"Memory writeback error (continuing): {e}")

                    chunks = textwrap.wrap(final, width=3800, replace_whitespace=False)
                    for idx, chunk in enumerate(chunks, 1):
                        status_cb(f"output_{idx}/{len(chunks)}", chunk)
                    result = final
                    ran_ok  = True
                    summary = f"{len(chunks)} chunk(s)"

                else:
                    fn = getattr(self, f"_stage_{stage}", None)
                    if callable(fn):
                        out = fn(state)
                        summary = str(out).replace("\n"," ")[:60] + "…"
                    ran_ok = True

                if stage != "memory_writeback":
                    ran_ok = True

            except Exception as e:
                state["errors"].append((stage, str(e)))
                summary = f"error:{str(e)[:40]}"
                logger.warning(f"Stage `{stage}` failed: {e}")

            finally:
                dur = (datetime.utcnow() - start_t).total_seconds()
                perf = ContextObject.make_stage(
                    "stage_performance",
                    input_refs=[state["user_ctx"].context_id],
                    output={"stage": stage, "duration": dur, "error": not ran_ok}
                )
                perf.touch(); self.repo.save(perf)
                self.memman.reinforce(perf.context_id, state.get("recent_ids", []))
                try:
                    status_cb(stage, summary)
                except:
                    pass
                self._last_errors |= (not ran_ok) or bool(state["errors"])

        # 5) update narrative
        try:
            self._stage_generate_narrative(state)
        except Exception:
            pass

        # 6) fallback if writeback never fired
        if result is None and "final" in state:
            result = state["final"] or ""

        # 6) RL update: reward=1 if no errors, else 0
        reward = 1.0 if not state["errors"] else 0.0
        self.rl.update(state["rl_included"], reward)

        # 7) spontaneous ping
        for cid in list(self._chat_contexts):
            try:
                self._maybe_appiphany(cid)
            except:
                pass

        # 8) return
        return result



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

        # ── A) Extract raw calls from the plan ──────────────────────────────
        text = plan_output
        parsed = Tools.parse_tool_call(text)
        print("PARSED TOOL CALL")
        print(parsed)
        if parsed:
            raw = [parsed]
        else:
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
        wanted = {call.split("(")[0] for call in calls}
        for sch_obj in all_schemas:
            schema = json.loads(sch_obj.metadata["schema"])
            if schema["name"] in wanted:
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
            "You have these tools (full JSON schemas shown).\n"
            "I will send you exactly one JSON object with key \"tool_calls\".\n"
            "YOUR ONLY JOB is to return back that same JSON object (modifying only calls that violate the schema).\n"
            "Do NOT add, remove, or simulate any outputs or internal state.\n"
            "Reply with exactly one JSON object and nothing else:\n\n"
            '{"tool_calls": ["tool1(arg1=...,arg2=...)", ...]}\n\n'
            f"{docs_blob}"
        )
        self._print_stage_context("tool_chaining", {
            "plan":    [plan_output[:200]],
            "schemas": docs_blob_parts
        })
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": json.dumps({"tool_calls": calls})},
        ]
        out = self._stream_and_capture(self.secondary_model, msgs, tag="[ToolChain]")

        # ── F) Parse the confirmed calls, fallback if anything goes wrong ──
        confirmed = calls
        try:
            # extract the JSON blob
            blob = re.search(r'\{.*"tool_calls".*\}', out, flags=re.S).group(0)
            parsed2 = json.loads(blob)
            if isinstance(parsed2.get("tool_calls"), list):
                confirmed = parsed2["tool_calls"]
        except Exception:
            # model deviated: stick with our original calls
            confirmed = calls

        # ── G) Persist the final chaining stage, including the docs ─────────
        tc_ctx = ContextObject.make_stage(
            "tool_chaining",
            plan_ctx.references + [s.context_id for s in selected_schemas],
            {
                "tool_calls": confirmed,
                "tool_docs":  docs_blob
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
