# assembler.py

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat, ResponseError

from context import ContextObject, ContextRepository, MemoryManager

# ────────────────────────────────────────────────────────────────────────────────
# Logging configuration for observability
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Default system prompts
# ────────────────────────────────────────────────────────────────────────────────
DEFAULT_CLARIFICATION_PROMPT = """You are Clarifier. Given the user's latest input and a set of relevant context snippets,
identify the user's underlying intention and expand it into one or more search keywords.
Also provide a brief explanation of your reasoning. 
Respond in valid JSON with exactly two keys:
{
  "keywords": ["keyword1", "keyword2", ...],
  "notes": "your explanation here"
}"""

DEFAULT_ASSEMBLER_PROMPT = (
    "You are ContextAssembler. Your job is to take the clarified intent and "
    "the relevant past context snippets, then distill or transform that information "
    "into a concise summary for the main assistant."
)

DEFAULT_INFERENCE_PROMPT = (
    "You are a helpful, context-aware assistant. Use the provided context snippets "
    "to answer the user's query accurately."
)

# ────────────────────────────────────────────────────────────────────────────────
# ContextQueryEngine: filter contexts by time, tags, and similarity
# ────────────────────────────────────────────────────────────────────────────────
class ContextQueryEngine:
    def __init__(self, store: ContextRepository, embedder):
        self.store = store
        self.embedder = embedder

    def query_segments(
        self,
        time_range: Optional[Tuple[str, str]] = None,
        tags: Optional[List[str]] = None,
        similarity_to: Optional[str] = None,
        top_k: int = 5
    ) -> List[ContextObject]:
        all_ctx = self.store.query(lambda c: True)
        if time_range:
            start, end = time_range
            all_ctx = [c for c in all_ctx if start <= c.timestamp <= end]
        if tags:
            all_ctx = [
                c for c in all_ctx
                if any(t in c.tags for t in tags) or c.domain in tags
            ]
        if similarity_to:
            qv = self.embedder(similarity_to)
            sims = []
            for c in all_ctx:
                txt = c.summary or ""
                if not txt:
                    continue
                v = self.embedder(txt)
                sim = float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v)))
                sims.append((c, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in sims[:top_k]]
        return all_ctx[:top_k]

# ────────────────────────────────────────────────────────────────────────────────
# Assembler: three-stage pipeline + dynamic prompt updates + logging
# ────────────────────────────────────────────────────────────────────────────────
class Assembler:
    def __init__(
        self,
        context_path: str = "context.jsonl",
        config_path:  str = "config.json",
        lookback_minutes: int = 60,
        top_k: int = 5
    ):
        # Load or create config
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            cfg = {}
        self.config_path = config_path
        self.config = cfg

        # Models (fallback to "gemma3:4b")
        self.primary_model   = cfg.get("primary_model") or "gemma3:4b"
        self.secondary_model = cfg.get("secondary_model") or self.primary_model

        # Prompts (fallback to defaults)
        self.clarifier_prompt = cfg.get("clarifier_prompt", DEFAULT_CLARIFICATION_PROMPT)
        self.assembler_prompt = cfg.get("assembler_prompt", DEFAULT_ASSEMBLER_PROMPT)
        self.inference_prompt = cfg.get("inference_prompt", DEFAULT_INFERENCE_PROMPT)

        # Persist defaults back to config.json
        updated = False
        for key, val in [
            ("primary_model", self.primary_model),
            ("secondary_model", self.secondary_model),
            ("clarifier_prompt", self.clarifier_prompt),
            ("assembler_prompt", self.assembler_prompt),
            ("inference_prompt", self.inference_prompt),
        ]:
            if cfg.get(key) != val:
                cfg[key] = val
                updated = True
        if updated:
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)

        # Context store & memory manager
        self.repo    = ContextRepository(context_path)
        self.mem_mgr = MemoryManager(self.repo)

        # Embedder & query engine
        stm = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedder = lambda txt: stm.encode(txt, normalize_embeddings=True)
        self.engine   = ContextQueryEngine(self.repo, self.embedder)

        # Runtime parameters
        self.lookback = lookback_minutes
        self.top_k    = top_k

        logger.info(f"Assembler initialized: primary_model={self.primary_model}, secondary_model={self.secondary_model}")

    def update_clarifier_prompt(self, new_prompt: str) -> None:
        logger.info("Updating clarifier prompt")
        self.clarifier_prompt = new_prompt
        self.config["clarifier_prompt"] = new_prompt
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        ctx = ContextObject(
            domain="stage",
            component="clarifier",
            semantic_label="clarifier_prompt_update",
            summary=new_prompt,
            tags=["clarifier", "config"]
        )
        ctx.touch()
        self.repo.save(ctx)
        logger.info("Clarifier prompt updated and saved")

    def assemble_meta_context(
        self,
        user_text: str
    ) -> Tuple[ContextObject, List[ContextObject]]:
        # Record user input
        user_ctx = ContextObject.make_segment("user_input", [], tags=["user_input"])
        user_ctx.summary = user_text
        user_ctx.touch()
        self.repo.save(user_ctx)
        logger.info(f"User input recorded: {user_ctx.context_id}")

        # Time window
        now = datetime.utcnow()
        past = now - timedelta(minutes=self.lookback)
        trange = (past.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%dT%H%M%SZ"))

        # Retrieve relevant contexts
        relevant = self.engine.query_segments(
            time_range=trange,
            tags=["user_input", "assistant", "assembler"],
            similarity_to=user_text,
            top_k=self.top_k
        )
        logger.info(f"Retrieved {len(relevant)} relevant contexts")
        return user_ctx, relevant

    def run_clarification(
        self,
        user_ctx: ContextObject,
        relevant_ctxs: List[ContextObject]
    ) -> Tuple[ContextObject, List[ContextObject]]:
        """
        Clarify user intention, produce search keywords, then fetch
        extra contexts based on those keywords.
        """
        snippets = [
            f"[{c.semantic_label}] {c.summary}"
            for c in relevant_ctxs if c.summary
        ]
        ctx_block = "\n".join(snippets) or "(no prior context)"

        logger.info("Clarification stage inputs:")
        logger.info(f"  Context:\n{ctx_block}")
        logger.info(f"  User input: {user_ctx.summary}")

        messages = [
            {"role": "system",  "content": self.clarifier_prompt},
            {"role": "system",  "content": f"Context:\n{ctx_block}"},
            {"role": "user",    "content": user_ctx.summary}
        ]

        try:
            resp = chat(
                model=self.secondary_model,
                messages=messages,
                stream=False
            )
        except ResponseError as e:
            logger.error(f"Clarification inference failed: {e}")
            raise

        content = resp.message.content.strip()
        logger.info(f"Clarification output raw: {content}")

        try:
            clar = json.loads(content)
            keywords = clar.get("keywords", [])
            notes    = clar.get("notes", "")
        except Exception as e:
            raise ValueError(f"Failed to parse clarification JSON: {e}\nOutput was:\n{content}")

        clar_ctx = ContextObject(
            domain="stage",
            component="clarifier",
            semantic_label="clarification",
            summary=notes,
            tags=["clarifier"],
            references=[user_ctx.context_id] + [c.context_id for c in relevant_ctxs]
        )
        clar_ctx.touch()
        self.repo.save(clar_ctx)
        logger.info(f"Saved clarification context: {clar_ctx.context_id}")

        # Fetch extra contexts based on keywords
        extra_ctxs = []
        seen = {user_ctx.context_id, clar_ctx.context_id} | {c.context_id for c in relevant_ctxs}
        for kw in keywords:
            sims = self.engine.query_segments(similarity_to=kw, top_k=self.top_k)
            for c in sims:
                if c.context_id not in seen:
                    extra_ctxs.append(c)
                    seen.add(c.context_id)
        logger.info(f"Fetched {len(extra_ctxs)} extra contexts via clarification keywords")

        return clar_ctx, extra_ctxs

    def run_secondary_inference(
        self,
        user_ctx: ContextObject,
        context_slice: List[ContextObject]
    ) -> ContextObject:
        """
        Distill the clarified context slice for the main assistant.
        """
        snippets = [
            f"[{c.semantic_label}] {c.summary}" for c in context_slice if c.summary
        ]
        ctx_block = "\n".join(snippets) or "(no context)"

        logger.info("Assembler stage inputs:")
        logger.info(f"  Prompt: {self.assembler_prompt}")
        logger.info(f"  Context:\n{ctx_block}")
        logger.info(f"  User input: {user_ctx.summary}")

        messages = [
            {"role": "system", "content": self.assembler_prompt},
            {"role": "system", "content": f"Context:\n{ctx_block}"},
            {"role": "user",   "content": user_ctx.summary}
        ]

        try:
            resp = chat(
                model=self.secondary_model,
                messages=messages,
                stream=False
            )
        except ResponseError as e:
            logger.error(f"Assembler inference failed: {e}")
            raise

        distilled = resp.message.content.strip()
        logger.info(f"Assembler output: {distilled}")

        asm_ctx = ContextObject(
            domain="stage",
            component="assembler",
            semantic_label="assemble_context",
            summary=distilled,
            tags=["assembler"],
            references=[user_ctx.context_id] + [c.context_id for c in context_slice]
        )
        asm_ctx.touch()
        self.repo.save(asm_ctx)
        logger.info(f"Saved assembler context: {asm_ctx.context_id}")

        return asm_ctx

    def run_with_meta_context(self, user_text: str) -> str:
        """
        Full pipeline:
          1) Record & retrieve (assemble_meta_context)
          2) Clarification → clar_ctx + extra_ctxs
          3) Assembler → assembler_ctx
          4) Primary inference (streaming)
          5) Record assistant reply
        """
        # Stage 1
        user_ctx, relevant = self.assemble_meta_context(user_text)

        # Stage 2
        clar_ctx, extra = self.run_clarification(user_ctx, relevant)

        # Stage 3
        assembler_ctx = self.run_secondary_inference(user_ctx, relevant + extra + [clar_ctx])

        # Stage 4: primary inference
        all_ctx = relevant + extra + [clar_ctx, assembler_ctx]
        snippets = [f"[{c.semantic_label}] {c.summary}" for c in all_ctx if c.summary]
        ctx_block = "\n".join(snippets)

        logger.info("Primary inference inputs:")
        logger.info(f"  Prompt: {self.inference_prompt}")
        logger.info(f"  Context:\n{ctx_block}")
        logger.info(f"  User input: {user_text}")

        messages = [
            {"role": "system",  "content": self.inference_prompt},
            {"role": "system",  "content": f"Context:\n{ctx_block}"},
            {"role": "user",    "content": user_text}
        ]

        stream = chat(model=self.primary_model, messages=messages, stream=True)

        reply = ""
        print("[Assistant] ", end="", flush=True)
        try:
            for chunk in stream:
                part = chunk["message"]["content"]
                print(part, end="", flush=True)
                reply += part
            print()
        except ResponseError as e:
            logger.error(f"Primary inference failed: {e}")
            raise

        # Stage 5: record assistant reply
        asst_ctx = ContextObject(
            domain="stage",
            component="assistant",
            semantic_label="response",
            summary=reply,
            tags=["assistant"],
            references=[user_ctx.context_id, clar_ctx.context_id, assembler_ctx.context_id]
                      + [c.context_id for c in relevant]
                      + [c.context_id for c in extra]
        )
        asst_ctx.touch()
        self.repo.save(asst_ctx)
        logger.info(f"Saved assistant response context: {asst_ctx.context_id}")

        return reply

# ────────────────────────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asm = Assembler()
    print("Assembler ready. Type your message, or Ctrl-C to exit.")
    try:
        while True:
            text = input(">> ").strip()
            if text:
                asm.run_with_meta_context(text)
    except KeyboardInterrupt:
        print("\nGoodbye.")
