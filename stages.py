import json
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

from context import ContextObject, ContextRepository, MemoryManager
from tools import Tools, TOOL_SCHEMAS

logger = logging.getLogger(__name__)


class StageFactory:
    """
    Builds and persists ContextObjects for each pipeline stage,
    including system prompts, tool schemas, inputs, timing, and outputs.
    """

    # ——— The full ordered list of stage names ———
    PIPELINE_STAGES = [
        "self_repair", "first_error_patch", "state_reflection",
        "prompt_optimization", "rl_experimentation",
        "context_analysis", "intent_clarification",
        "external_knowledge_retrieval", "planning_summary",
        "goal_alignment", "drive_generation", "internal_adversarial",
        "creative_promotion", "tool_self_improvement",
        "define_criteria", "task_decomposition", "plan_validation",
        "execute_actions", "confidence_tracking", "verify_results",
        "plan_completion_check", "checkpoint",
        "task_management", "subtask_management", "execute_tasks",
        "tool_chaining", "assemble_prompt", "self_review",
        "final_inference", "user_feedback", "output_review",
        "chain_of_thought", "meta_reflection", "adversarial_loop",
        "notification_audit", "flow_health_check"
    ]

    def __init__(self, repo: ContextRepository):
        self.repo = repo
        self.memman = MemoryManager(repo)

        # 1) ensure our global tools registry is created once
        self.tools_registry = self._ensure_tools_registry()

        # 2) seed missing tool schemas exactly once
        #    collect what’s already in the repo
        existing = {
            ctx.semantic_label
            for ctx in self.repo.query(
                lambda c: c.component == "schema" and "tool_schema" in c.tags
            )
        }
        for tool_name, schema in TOOL_SCHEMAS.items():
            if tool_name not in existing:
                ctx = ContextObject.make_schema(
                    label=tool_name,
                    schema_def=json.dumps(schema),
                    tags=["artifact", "tool_schema"]
                )
                ctx.touch()
                self.repo.save(ctx)

    def make_user_input(self, user_text: str) -> ContextObject:
        """
        Persist the raw user message as a ContextObject.
        Returns that ContextObject for use as a reference in stages.
        """
        ctx = ContextObject(
            domain="user",
            component="message",
            semantic_label="user_message",
            summary=user_text,
            tags=["user_input"]
        )
        ctx.touch()
        self.repo.save(ctx)
        return ctx

    def _ensure_tools_registry(self) -> ContextObject:
        """
        One‐time: store a JSON blob of all available tools.
        """
        existing = self.repo.query(lambda c: c.component == "tools_registry")
        if existing:
            return existing[0]

        tool_meta = json.loads(Tools.list_tools(detail=True))
        ctx = ContextObject(
            domain="artifact",
            component="tools_registry",
            semantic_label="tools_registry",
            summary="Registry of available tools and metadata",
            tags=["artifact", "tools_registry"],
            metadata={"tools": tool_meta}
        )
        ctx.touch()
        self.repo.save(ctx)
        return ctx

    def ensure_tool_schema(self, tool_name: str) -> ContextObject:
        """
        Persist the JSON‐schema for a given tool, if not already stored.
        (Useful if you add new tools at runtime.)
        """
        found = self.repo.query(
            lambda c: c.component == "schema"
                      and c.semantic_label == tool_name
                      and "tool_schema" in c.tags
        )
        if found:
            return found[0]

        schema = TOOL_SCHEMAS.get(tool_name)
        if not schema:
            raise KeyError(f"No TOOL_SCHEMAS entry for '{tool_name}'")

        ctx = ContextObject.make_schema(
            label=tool_name,
            schema_def=json.dumps(schema),
            tags=["artifact", "tool_schema"]
        )
        ctx.touch()
        self.repo.save(ctx)
        return ctx

    @contextmanager
    def stage(
        self,
        name: str,
        system_prompt: str,
        inputs: List[ContextObject],
        user_input: Optional[ContextObject] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        start = datetime.utcnow()

        # 1) persist the system prompt
        sp = ContextObject(
            domain="stage",
            component="system_prompt",
            semantic_label=f"{name}_system_prompt",
            summary=system_prompt,
            tags=["system_prompt", name]
        )
        sp.touch()
        self.repo.save(sp)
        logger.debug(f"[{name}] saved system_prompt id={sp.context_id!r}")

        # 2) log inputs & user_input
        logger.debug(f"[{name}] inputs:")
        for idx, inp in enumerate(inputs, 1):
            logger.debug(
                f"    {idx}. [{inp.component}:{inp.semantic_label}] "
                f"id={inp.context_id!r}"
            )
        if user_input:
            logger.debug(
                f"    user_input [{user_input.component}:"
                f"{user_input.semantic_label}] id={user_input.context_id!r}"
            )

        # 3) hand back a mutable dict for the caller to stash outputs
        ctx_data: Dict[str, Any] = {"outputs": None}
        yield ctx_data

        # 4) immediately log whatever the caller wrote
        logger.debug(f"[{name}] raw outputs: {ctx_data['outputs']!r}")

        # 5) record duration
        end = datetime.utcnow()
        duration = (end - start).total_seconds()

        # 6) build reference list
        refs = [self.tools_registry.context_id, sp.context_id]
        refs += [i.context_id for i in inputs]
        if user_input:
            refs.append(user_input.context_id)

        # 7) truncate & stringify outputs for summary
        summary = ""
        if ctx_data.get("outputs") is not None:
            summary = str(ctx_data["outputs"])[:512]

        # 8) save the stage ContextObject
        stage_ctx = ContextObject(
            domain="stage",
            component=name,
            semantic_label=name,
            summary=summary,
            tags=[name],
            references=refs,
            metadata={**(extra_metadata or {}), "duration": duration}
        )
        stage_ctx.touch()
        self.repo.save(stage_ctx)
        logger.debug(f"[{name}] stage_ctx.id={stage_ctx.context_id!r} saved.")

    def _expand(self, stage: str, ctx: Any) -> List[str]:
        """
        Dynamic next‐stage routing logic (drawn from ChatManager._expand).
        """
        # 1️⃣ on failure → self_repair
        if getattr(ctx, "last_failure", None):
            return ["self_repair"]

        # 2️⃣ forced injections
        if getattr(ctx, "_forced_next", None):
            return [ctx._forced_next.pop(0)]

        # 3️⃣ linear flow shortcuts
        if stage == "start":
            return ["context_analysis"]
        if stage == "context_analysis":
            return ["task_decomposition"]
        if stage == "task_decomposition":
            return ["tool_chaining"]

        # 4️⃣ after tool_chaining
        if stage == "tool_chaining":
            if getattr(ctx, "tool_summaries", []):
                return ["assemble_prompt", "final_inference"]
            # else fallback to RL‐selected remainder
            base = getattr(self, "config", {}).get("system", "")
            all_names = self.PIPELINE_STAGES.copy()
            prompt, flow = self.rl_manager.select(base, all_names)
            ctx.pipeline_sequence = flow.copy()
            ctx._pipeline_initialised = True
            self.config["system"] = prompt
            return [ctx.pipeline_sequence.pop(0)] if ctx.pipeline_sequence else []

        # 5️⃣ after final_inference → output_review
        if stage == "final_inference":
            return ["output_review"]

        # 6️⃣ normal dequeue
        seq = getattr(ctx, "pipeline_sequence", [])
        if seq:
            return [seq.pop(0)]

        # 7️⃣ done
        return []
