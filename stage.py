# stage.py
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from context import ContextObject
from ollama import chat

class Stage(ABC):
    """
    Base class for a single pipeline stage.
    Subclasses must define:
      - name: the stage_id / semantic_label
      - model: which LLM to call (None for non-LLM stages)
      - assemble(state): List of messages to send, or [] if no LLM
      - process(raw, state): turn raw LLM or parser output → metadata dict
    """
    name: str
    model: Optional[str] = None

    def __init__(self, repo, printer, default_model: str):
        self.repo = repo
        self.print_context = printer
        self.default_model = default_model

    @abstractmethod
    def assemble(self, state: Dict[str, Any]) -> List[Dict[str,Any]]:
        ...

    @abstractmethod
    def process(self, raw: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def run(self, state: Dict[str,Any]) -> ContextObject:
        # 1) assemble messages
        msgs = self.assemble(state)

        # 2) call LLM if needed
        if msgs:
            model = self.model or self.default_model
            raw = self._stream_and_capture(model, msgs, tag=f"[{self.name}]")
        else:
            raw = None

        # 3) process
        meta = self.process(raw, state)

        # 4) persist as ContextObject
        ctx = ContextObject.make_stage(
            self.name,
            state.get("refs", []),
            meta
        )
        ctx.stage_id = self.name
        # summary = JSON if dict/list, else str
        ctx.summary = json.dumps(meta) if isinstance(meta, (dict, list)) else str(meta)
        ctx.touch()
        self.repo.save(ctx)

        # 5) record ref for next stages
        state["refs"].append(ctx.context_id)
        state[self.name] = ctx
        state.update(meta)
        return ctx

    def _stream_and_capture(self, model: str, messages: List[Dict[str,Any]], tag: str="") -> str:
        out = ""
        print(f"{tag} ", end="", flush=True)
        for part in chat(model=model, messages=messages, stream=True):
            chunk = part["message"]["content"]
            print(chunk, end="", flush=True)
            out += chunk
        print()
        return out
