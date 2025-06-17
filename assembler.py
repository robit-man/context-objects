# assembler.py

import json
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from context import ContextObject, ContextRepository, MemoryManager

# ────────────────────────────────────────────────────────────────────────────────
# System prompt: teaches the model how to ingest and emit ContextObject JSON
# ────────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MetaContextManager. You know about the Python class ContextObject
(importable from context.py) whose JSON schema includes at least these fields:
  - context_id (string UUID)
  - domain (e.g. "segment", "stage")
  - component (string)
  - semantic_label (string)
  - timestamp (ISO YYYYMMDDTHHMMSSZ)
  - references (list of context_id strings)
  - summary (string)
  - tags (list of strings)
Optional fields include embedding, retrieval_metadata, memory_traces, etc.

Below you will receive an array existing_contexts of such ContextObject JSON objects,
and a user_input string. After processing, respond with strictly valid JSON:
{
  "new_context": { ...ContextObject JSON... },
  "response": "<your natural-language reply>"
}
Do NOT add any other keys. Ensure JSON parses cleanly.
"""

# ────────────────────────────────────────────────────────────────────────────────
# Query Engine: filters stored contexts by time, tags, and semantic similarity
# ────────────────────────────────────────────────────────────────────────────────
class ContextQueryEngine:
    def __init__(self,
                 store: ContextRepository,
                 embedder: Callable[[str], List[float]]):
        """
        store: ContextRepository instance
        embedder: function mapping text to normalized embedding vector
        """
        self.store = store
        self.embedder = embedder

    def query_segments(
        self,
        time_range: Optional[Tuple[str, str]] = None,
        tags: Optional[List[str]] = None,
        similarity_to: Optional[str] = None,
        top_k: int = 5
    ) -> List[ContextObject]:
        # 1) retrieve all contexts
        all_ctx = self.store.query(lambda c: True)

        # 2) time filter
        if time_range:
            start, end = time_range
            all_ctx = [
                c for c in all_ctx
                if c.timestamp >= start and c.timestamp <= end
            ]

        # 3) tag filter
        if tags:
            all_ctx = [
                c for c in all_ctx
                if any(t in c.tags for t in tags) or c.domain in tags
            ]

        # 4) semantic similarity filter
        if similarity_to:
            q_vec = self.embedder(similarity_to)
            sims: List[Tuple[ContextObject, float]] = []
            for c in all_ctx:
                text = c.summary or ""
                if not text:
                    continue
                v = self.embedder(text)
                sim = float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v)))
                sims.append((c, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in sims[:top_k]]

        return all_ctx[:top_k]

# ────────────────────────────────────────────────────────────────────────────────
# Assembler: orchestrates meta-context assembly and Ollama invocation
# ────────────────────────────────────────────────────────────────────────────────
class Assembler:
    def __init__(
        self,
        context_path: str = "context.jsonl",
        config_path:  str = "config.json",
        lookback_minutes: int = 60,
        top_k: int = 5
    ):
        # load or create config
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"{config_path} not found; run main.py to generate it.")

        self.primary_model   = cfg["primary_model"]
        self.secondary_model = cfg.get("secondary_model")

        # context and memory
        self.repo    = ContextRepository(context_path)
        self.mem_mgr = MemoryManager(self.repo)

        # embedder & query engine
        stm = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedder = lambda txt: stm.encode(txt, normalize_embeddings=True)
        self.engine   = ContextQueryEngine(self.repo, self.embedder)

        # parameters
        self.lookback = lookback_minutes
        self.top_k     = top_k
        self.system_prompt = SYSTEM_PROMPT

    def assemble_meta_context(
        self,
        user_text: str
    ) -> Tuple[ContextObject, List[ContextObject]]:
        """
        1) Record user input as a ContextObject segment.
        2) Query past contexts by time, tag, similarity.
        Returns (user_ctx, list_of_relevant_ctxs).
        """
        # record user turn
        user_ctx = ContextObject.make_segment(
            semantic_label="user_input",
            content_refs=[],
            tags=["user_input"]
        )
        user_ctx.summary = user_text
        user_ctx.touch()
        self.repo.save(user_ctx)

        # time window
        now = datetime.utcnow()
        past = now - timedelta(minutes=self.lookback)
        trange = (past.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%dT%H%M%SZ"))

        # query relevant contexts
        rel_ctxs = self.engine.query_segments(
            time_range=trange,
            tags=["user_input", "assistant"],
            similarity_to=user_text,
            top_k=self.top_k
        )
        return user_ctx, rel_ctxs

    def run_with_meta_context(self, user_text: str) -> str:
        """
        Executes one inference:
          - Assembles meta-context
          - Calls Ollama gemma3:4b with system prompt + payload
          - Parses response JSON
          - Records new ContextObject
          - Returns the 'response' text
        """
        user_ctx, rel_ctxs = self.assemble_meta_context(user_text)

        payload = {
            "existing_contexts": [c.to_dict() for c in rel_ctxs],
            "user_input": user_text
        }
        cmd = [
            "ollama", "run", self.primary_model,
            "--system-prompt", self.system_prompt,
            "--prompt", json.dumps(payload)
        ]
        raw = subprocess.check_output(cmd, text=True).strip()
        result = json.loads(raw)

        # hydrate and store new context
        new_ctx = ContextObject.from_dict(result["new_context"])
        new_ctx.touch()
        self.repo.save(new_ctx)

        return result["response"]

# ────────────────────────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asm = Assembler()
    print("Meta-context assembler ready. Type input and press Enter.")
    try:
        while True:
            text = input(">> ").strip()
            if not text:
                continue
            reply = asm.run_with_meta_context(text)
            print(reply)
    except KeyboardInterrupt:
        print("\nGoodbye.")
