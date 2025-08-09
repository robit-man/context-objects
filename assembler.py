#!/usr/bin/env python3
"""
assembler.py — Stage-driven pipeline with full observability and
dynamic, chronological context windows per stage.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import ast
import os
import re
import sys
import io
import math
import json
import uuid
import time
import base64
import random
import shutil
import inspect
import asyncio
import hashlib
import tempfile
import threading
import traceback
import contextlib
import textwrap
from pathlib import Path
from types import MethodType
from collections import deque
from functools import lru_cache
from difflib import SequenceMatcher
from rl import RLManager, RewardConfig
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import requests
from ollama import chat, embed
from ollama._types import ResponseError as _OllamaError

# ── Project-local ─────────────────────────────────────────────────────────────
import stages
import tools
from tools import Tools, TOOL_SCHEMAS, _thread_local
from context import (
    ContextObject,
    ContextRepository,
    HybridContextRepository,
    MemoryManager,
    default_clock,
    sanitize_jsonl,
)
from grand_integrator import GrandIntegrator


# ──────────────────────────────────────────────────────────────────────────────
def _canon(call: str) -> str:
    """Return a canonical signature for a tool call (idempotent)."""
    s = (call or "").strip()
    if not s or "(" not in s or not s.endswith(")"):
        return s
    name, _ = s.split("(", 1)
    try:
        tree = ast.parse(s)
        node = tree.body[0].value  # type: ignore[attr-defined]
    except Exception:
        return s
    pos = [ast.get_source_segment(s, a).strip() for a in getattr(node, "args", [])]
    kw = {
        k.arg: ast.get_source_segment(s, k.value).strip()
        for k in getattr(node, "keywords", [])
        if ast.get_source_segment(s, k.value).strip() not in ("''", '""', 'None')
    }
    sig = name.strip() + "("
    sig += ",".join(pos)
    if kw:
        sig += "," if pos else ""
        sig += ",".join(f"{k}={v}" for k, v in sorted(kw.items()))
    sig += ")"
    return sig


# ────────────────────────────────────────────────────────────────
# 1) Safe-call wrappers
# ────────────────────────────────────────────────────────────────
def _safe_call(func: Callable, *args, **kwargs):
    """
    Call func but drop any args/kwargs its signature doesn’t accept.
    """
    try:
        return func(*args, **kwargs)
    except TypeError:
        sig = inspect.signature(func)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        max_pos = sum(
            1
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
        trimmed = args[:max_pos]
        return func(*trimmed, **allowed)


async def _to_thread_safe(func: Callable, *args, **kwargs):
    """asyncio.to_thread wrapper around _safe_call."""
    return await asyncio.to_thread(_safe_call, func, *args, **kwargs)


@lru_cache(maxsize=None)
def _done_calls(repo) -> set[str]:
    """
    Any *successful* canonical signatures stored in the context log.
    NOTE: relies on the object being hashable for lru_cache. Default Python
    objects are hashable by id unless __eq__ is overridden.
    """
    done: set[str] = set()
    try:
        for obj in repo.query(lambda c: c.component == "tool_output"):
            if (obj.metadata or {}).get("ok"):
                call_sig = (obj.metadata or {}).get("tool_call")
                if call_sig:
                    done.add(str(call_sig))
    except Exception:
        pass
    return done


# ────────────────────────────────────────────────────────────────
# 2) Embedding utilities (thread-safe, non-blocking)  ────────────
# ────────────────────────────────────────────────────────────────
_EMBED_CACHE: dict[str, np.ndarray] = {}
_CACHE_LOCK = threading.Lock()
_ZERO = np.zeros(768, dtype=float)
_embed_executor = ThreadPoolExecutor(max_workers=4)


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    try:
        norm = float(np.linalg.norm(vec))
        return vec / (norm or 1.0)
    except Exception:
        return _ZERO


def embed_text(text: str) -> np.ndarray:
    """
    Non-blocking embed with a shared ThreadPoolExecutor.
    Returns a zero vector immediately; the cache fills asynchronously.
    """
    key = text if isinstance(text, str) else str(text)
    with _CACHE_LOCK:
        if key in _EMBED_CACHE:
            return _EMBED_CACHE[key]

    def _worker(t: str):
        vec = _ZERO
        try:
            resp = embed(model="nomic-embed-text", input=t)
            # Ollama may return "embedding" for str input or "embeddings" for list input
            if isinstance(resp, dict):
                if "embedding" in resp and isinstance(resp["embedding"], list):
                    vec = np.array(resp["embedding"], dtype=float)
                elif "embeddings" in resp and isinstance(resp["embeddings"], list):
                    first = resp["embeddings"][0] if resp["embeddings"] else []
                    vec = np.array(first, dtype=float)
            vec = _normalize_vec(vec)
        except Exception:
            vec = _ZERO
        with _CACHE_LOCK:
            _EMBED_CACHE[t] = vec

    _embed_executor.submit(_worker, key)
    return _ZERO


# ────────────────────────────────────────────────────────────────
# 3) RL controller  ──────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
class RLController:
    """
    Multi-armed bandit with baseline + recall bias.
    Q[s]: estimated reward for stage s
    R_bar: global baseline
    Each optional stage also has a gamma parameter that
    amplifies the signal from context-recall frequency.
    """

    def __init__(
        self,
        stages: List[str],
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 0.1,
        path: str = "weights.rl",
    ):
        self.alpha = alpha  # LR for Q
        self.beta = beta  # LR for baseline
        self.gamma = gamma  # weight on recall_feature
        self.path = path

        data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        self.Q = {s: data.get("Q", {}).get(s, 0.0) for s in stages}
        self.N = {s: data.get("N", {}).get(s, 0) for s in stages}
        self.R_bar = float(data.get("R_bar", 0.0))

    def probability(self, stage: str, recall_feat: float = 0.0) -> float:
        adv = self.Q.get(stage, 0.0) - self.R_bar + self.gamma * recall_feat
        try:
            return 1.0 / (1.0 + math.exp(-adv))
        except OverflowError:
            return 0.0 if adv < 0 else 1.0

    def should_run(self, stage: str, recall_feat: float = 0.0) -> bool:
        return random.random() < self.probability(stage, recall_feat)

    def update(self, included: List[str], reward: float):
        self.R_bar += self.beta * (reward - self.R_bar)
        for s in included:
            self.N[s] = self.N.get(s, 0) + 1
            lr = self.alpha / math.sqrt(self.N[s])
            self.Q[s] = self.Q.get(s, 0.0) + lr * (reward - self.Q.get(s, 0.0))
        self.save()

    def save(self):
        tmp = f"{self.path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {"Q": self.Q, "N": self.N, "R_bar": self.R_bar},
                    f,
                    indent=2,
                )
            os.replace(tmp, self.path)
        except Exception:
            # best effort; ignore save errors
            pass


# ────────────────────────────────────────────────────────────────
# 4) Task graph plumbing  ────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
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

    def __init__(self, asm: "Assembler", user_text: str, clar_metadata: Dict[str, Any]):
        self.asm = asm
        self.user_text = user_text
        self.clar_metadata = clar_metadata
        self.tools_list = getattr(asm, "tools_list", [])
        self.memman = asm.memman

    def execute(self, node: TaskNode) -> None:
        # 1) Static validation / fix
        plan_ctx_id = node.context_ids[0]
        plan_ctx_obj = self.asm.repo.get(plan_ctx_id)

        _, errors, fixed = self.asm._stage7b_plan_validation(
            plan_ctx_obj,
            node.call,
            self.tools_list,
        )
        if errors:
            node.errors = [err for (_, err) in errors]

        calls = fixed or [node.call]

        # 2) Tool chaining (stage 8)
        tc_ctx, raw_calls, schemas = self.asm._stage8_tool_chaining(
            plan_ctx_obj, "\n".join(calls), self.tools_list
        )
        node.context_ids.append(tc_ctx.context_id)

        # 3) User confirmation (stage 8.5)
        confirmed = self.asm._stage8_5_user_confirmation(raw_calls, self.user_text)

        # 4) Invoke with retries (stage 9)
        tool_ctxs = self.asm._stage9_invoke_with_retries(
            confirmed,
            "\n".join(calls),
            schemas,
            self.user_text,
            self.clar_metadata,
        )
        for t in tool_ctxs:
            node.context_ids.append(t.context_id)

            if (t.metadata or {}).get("exception") is None:
                succ = ContextObject.make_success(
                    f"Tool `{(t.metadata or {}).get('tool_name', t.semantic_label)}` succeeded",
                    refs=[t.context_id],
                )
                succ.touch()
                self.asm.repo.save(succ)
                self.memman.register_relationships(succ, self.asm.embed_text)
                self.memman.reinforce(succ.context_id, [t.context_id])

                # Promote 'refined' retry candidates for this tool
                for crit in self.asm.repo.query(
                    lambda c: c.component == "analysis"
                    and c.semantic_label == "tool_retry_critique"
                    and (c.metadata or {}).get("status") == "refined"
                    and (c.metadata or {}).get("tool_name")
                    == (t.metadata or {}).get("tool_name")
                ):
                    crit.metadata["status"] = "confirmed"
                    crit.touch()
                    self.asm.repo.save(crit)
            else:
                fail = ContextObject.make_failure(
                    f"Tool `{(t.metadata or {}).get('tool_name', t.semantic_label)}` failed: {(t.metadata or {}).get('exception')}",
                    refs=[t.context_id],
                )
                fail.touch()
                self.asm.repo.save(fail)
                self.memman.reinforce(fail.context_id, [t.context_id])

        # 5) Reflection & replan (stage 9b)
        all_ctx_objs = [self.asm.repo.get(cid) for cid in node.context_ids]
        replan = self.asm._stage9b_reflection_and_replan(
            all_ctx_objs, "\n".join(calls), self.user_text, self.clar_metadata
        )

        if replan is None:
            succ = ContextObject.make_success(
                "Reflection validated original plan (OK)", refs=node.context_ids
            )
            succ.touch()
            self.asm.repo.save(succ)
            self.memman.reinforce(succ.context_id, node.context_ids)
        else:
            fail = ContextObject.make_failure(
                "Reflection triggered plan adjustment", refs=node.context_ids
            )
            fail.touch()
            self.asm.repo.save(fail)
            self.memman.reinforce(fail.context_id, node.context_ids)

            try:
                tree = json.loads(replan)
                node.children = self.asm._parse_task_tree(tree, parent=node)
            except Exception:
                pass

        # 6) Recurse into children
        for child in node.children:
            self.execute(child)

        # 7) Mark node overall success/failure
        if not node.errors and replan is None:
            overall = ContextObject.make_success(
                f"Task `{node.call}` completed successfully", refs=node.context_ids
            )
        else:
            overall = ContextObject.make_failure(
                f"Task `{node.call}` failed or was replanned", refs=node.context_ids
            )

        overall.touch()
        self.asm.repo.save(overall)
        self.memman.reinforce(overall.context_id, node.context_ids)
        node.completed = True


# ────────────────────────────────────────────────────────────────
# 5) TTS helpers  ────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
def _speak_now(self, text: str, status_cb):
    """
    Immediate, non-streamed utterance. Kills any current TTS, bypasses the live
    stream dedupe, and says `text` right now.
    """
    txt = (text or "").strip()
    if not txt:
        return
    # Stop anything already talking
    if getattr(self, "tts_bridge", None):
        self.tts_bridge.stop(hard=True)
    elif getattr(self, "tts_player", None):
        try:
            self.tts_player.stop()
        except Exception:
            pass

    status_cb("tts_immediate", txt)
    try:
        if getattr(self, "tts_bridge", None):
            self.tts_bridge.say(txt)
        else:
            self.tts_player.enqueue(txt)  # fallback to your raw player
    except Exception:
        pass


class _LiveTTSBridge:
    """
    Ultra-low-latency TTS streamer.

    feed(token)  -> buffer & auto-flush on punctuation or timeout
    say(text)    -> immediate full sentence (deduped)
    stop(hard)   -> clear buffers and optionally stop device
    flush(force) -> push whatever is buffered

    Use one instance per turn (or call .reset(turn_id)).
    """

    def __init__(self, tts_player, status_cb=None, min_ms=120, max_ms=700, punct=r"[.!?…]\s*$"):
        self.tts_player = tts_player
        self.status_cb = status_cb or (lambda *_: None)
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.punct_re = re.compile(punct)
        self.buf: List[str] = []
        self.last_flush = 0.0
        self.lock = threading.Lock()
        self.spoken_hash: set[str] = set()
        self.turn_id: Optional[str] = None
        self._paused_cb = None
        self._time = time
        self._hashlib = hashlib

    def new_turn(self, turn_id: str):
        self.spoken_hash.clear()
        self.turn_id = turn_id

    # ─── helpers ─────────────────────────────────────────────────────
    def _hash(self, txt: str) -> str:
        base = f"{getattr(self, 'turn_id', '')}:{txt}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _pause_asr(self):
        svc = getattr(self.tts_player, "audio_service", None)
        if not svc:
            return
        self._paused_cb = getattr(svc, "on_transcription", None)
        try:
            svc.on_transcription = lambda *_: None
        except Exception:
            pass

    def _resume_asr(self):
        svc = getattr(self.tts_player, "audio_service", None)
        if svc and self._paused_cb is not None:
            try:
                svc.on_transcription = self._paused_cb
            except Exception:
                pass
        self._paused_cb = None

    def _speak(self, text: str):
        text = text.strip()
        if not text:
            return
        h = self._hash(text)
        if h in self.spoken_hash:
            return
        self.spoken_hash.add(h)
        self.status_cb("tts_chunk", text)
        try:
            self._pause_asr()
            self.tts_player.enqueue(text)
        finally:
            self._resume_asr()

    # ─── public API ──────────────────────────────────────────────────
    def reset(self, turn_id: str):
        """Call at start of each turn."""
        with self.lock:
            self.buf.clear()
            self.spoken_hash.clear()
            self.last_flush = 0.0
            self.turn_id = turn_id

    def feed(self, chunk: str):
        if not chunk:
            return
        now = self._time.time() * 1000
        with self.lock:
            self.buf.append(chunk)
            if self.punct_re.search(chunk) or (now - self.last_flush) > self.max_ms:
                self._flush_locked(force=True)
            elif (now - self.last_flush) >= self.min_ms:
                self._flush_locked(force=False)

    def flush(self, force=False):
        with self.lock:
            self._flush_locked(force)

    def _flush_locked(self, force=False):
        if not self.buf:
            return
        joined = "".join(self.buf).strip()
        if not joined:
            self.buf.clear()
            return

        now = self._time.time() * 1000
        if force or self.punct_re.search(joined) or (now - self.last_flush) >= self.min_ms:
            self._speak(joined)
            self.buf.clear()
            self.last_flush = now

    def say(self, text: str):
        """Immediate sentence."""
        self.flush(force=True)
        self._speak(text)

    def stop(self, hard=False):
        """Clear buffers; if hard, also stop device output."""
        with self.lock:
            self.buf.clear()
            self.last_flush = 0.0
            self.spoken_hash.clear()
        self.status_cb("tts_stop", hard)
        if hard:
            try:
                self.tts_player.stop()
            except Exception:
                pass


# ────────────────────────────────────────────────────────────────
# 6) Context query engine  ───────────────────────────────────────
# ────────────────────────────────────────────────────────────────
class ContextQueryEngine:
    """
    Retrieval with time, tags, domain/component filters, regex & embedding similarity.
    Records recalls & registers associative edges.
    """

    def __init__(
        self,
        repo: ContextRepository,
        embedder: Callable[[str], np.ndarray],
        memman: MemoryManager,
    ):
        self.repo = repo
        self.embedder = embedder
        self.memman = memman
        self._cache: Dict[str, np.ndarray] = {}

    def _vec(self, text: Any) -> np.ndarray:
        """Coerce any input into a string key so we can safely cache lookups."""
        key = str(text)
        if key not in self._cache:
            self._cache[key] = self.embedder(key)
        return self._cache[key]

    def query(
        self,
        *,
        stage_id: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None,
        tags: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        domain: Optional[List[str]] = None,
        component: Optional[List[str]] = None,
        similarity_to: Optional[str] = None,
        summary_regex: Optional[str] = None,
        top_k: int = 5,
    ) -> List[ContextObject]:
        # 1) fetch and filter...
        ctxs = list(self.repo.query(lambda c: True))
        if time_range:
            start, end = time_range
            ctxs = [c for c in ctxs if start <= c.timestamp <= end]
        real_include = include_tags if include_tags is not None else tags
        if real_include:
            ctxs = [c for c in ctxs if set(real_include) & set(c.tags)]
        if exclude_tags:
            ctxs = [c for c in ctxs if not (set(exclude_tags) & set(c.tags))]
        if domain:
            ctxs = [c for c in ctxs if c.domain in domain]
        if component:
            ctxs = [c for c in ctxs if c.component in component]
        if summary_regex:
            pat = re.compile(summary_regex, re.I)
            ctxs = [c for c in ctxs if (c.summary and pat.search(c.summary))]

        # 2) similarity sort
        if similarity_to:
            qv = self._vec(similarity_to)
            scored: List[Tuple[ContextObject, float]] = []
            for c in ctxs:
                if not c.summary:
                    continue
                vv = self._vec(c.summary)
                denom = (np.linalg.norm(qv) * np.linalg.norm(vv)) or 1.0
                sim = float(np.dot(qv, vv) / denom)
                scored.append((c, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            ctxs = [c for c, _ in scored]

        # 3) take top_k, record & register
        out = ctxs[:top_k]
        for c in out:
            try:
                c.record_recall(stage_id=stage_id, coactivated_with=[])
                self.repo.save(c)
                self.memman.register_relationships(c, self.embedder)
            except Exception:
                pass

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
        "performance_rating",
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
        
        for name, func in inspect.getmembers(stages, inspect.isfunction):
            if name.startswith("_stage"):
                setattr(self, name, MethodType(func, self))

        self.current_user_id: str = "anon"

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

        self._task_poll_event  = threading.Event()   # existing
        self._task_wake_event  = threading.Event()   # ← NEW: kick to recalc sleep
        self._task_lock_file   = None                # ← NEW: poller singleton lock handle

        # — system & stage prompts —
        self.clarifier_prompt = self.cfg.get(
            "clarifier_prompt",
            # ── UPDATED clarifier instruction: pull in prior turns if relevant ────
            "You are Clarifier.  Expand the user’s intent into a JSON object with two keys:\n"
            "  • 'keywords' (an array of concise keywords)\n"
            "  • 'notes' (a short narrative expansion of what the user wants, "
            "drawing *only* on the user’s latest message AND any immediately preceding "
            "conversation turns that clarify or disambiguate that message)\n"
            "Additionally:\n"
            "- Under a key called 'debug_notes', include the last 3 turns of raw "
            "conversation (both user and assistant) even if they seem redundant, "
            "so we can diagnose mis‐clarifications. DO NOT HALLUCINATE, YOUR MODEL KNOWLEDGE SHOULD NOT BE RELIED UPON AND IS OUTDATED, NECESITATING TOOL USE TO GET RELEVANT UP TO DATE INFORMATION ON ANYTHING!\n"
            "- Notes should produce NO value judgments or claims, and should only "
            "expand what the user actually said.\n"
            "- Ignore irrelevant errors or tool outputs that do not bear on the "
            "user’s expressed intent.\n"
            "Output only valid JSON."
        )
        self.assembler_prompt = self.cfg.get(
            "assembler_prompt",
            "Distill context into a concise summary, but do not omit implied content which is needed for effective evaluation. Dont repeat this instruction in your response!"
        )
        self.inference_prompt = self.cfg.get(
            "inference_prompt",
            "Use all provided snippets and tool outputs to inform your reply, abide by internal instruction present and distill coherent and verbose responses based on contextual understanding and intention. Dont repeat this instruction in your response!"
        )

        # ──────────────────────────────────────────────────────────────────────
        #  PROMPT DEFAULTS  (updated for richer placeholder syntax)
        # ──────────────────────────────────────────────────────────────────────
        self.planning_prompt = self.cfg.get(
            "planning_prompt",
            # ── DAG PLANNER PROMPT ────────────────────────────────────────────
            "You are the Planner. Output **only** valid JSON for a small task-graph (DAG).\n"
            "Follow this schema exactly (no extra prose):\n"
            "{\n"
            "  \"graph\": {\n"
            "    \"nodes\": [\n"
            "      {\n"
            "        \"id\": \"t1\",                 // unique id per node\n"
            "        \"tool\": \"<tool_name>\",     // MUST match an available tool name exactly\n"
            "        \"args\": { /* named params matching the tool schema */ },\n"
            "        \"after\": []                  // ids of nodes that must complete before this one\n"
            "      }\n"
            "      // ≤ 6 nodes total; ≤ 3 runnable in any single layer\n"
            "    ],\n"
            "    \"meta\": {\n"
            "      \"goal\": \"<1–2 sentence summary of the user’s request>\",\n"
            "      \"constraints\": [ /* optional constraints */ ]\n"
            "    }\n"
            "  }\n"
            "}\n"
            "\n"
            "Rules:\n"
            "• Use ONLY tools from the provided *Available tools* list (names must match exactly).\n"
            "• Keep the graph minimal: prefer 1–3 nodes; max 6; ≤ 3 runnable in parallel per layer.\n"
            "• To reference prior-node output inside args you may write **any** of:\n"
            "      \"[tX.output]\"                     // full output object\n"
            "      \"[tX.output.some.path]\"           // specific sub-field via dot-path\n"
            "      \"[ <tX>.output ]\"                // same as first (angle-brackets allowed)\n"
            "      \"{{tX}}\"                          // shorthand = full output\n"
            "  (Whitespace inside […] is ignored.)\n"
            "• Do NOT invent argument keys; use exactly the keys/types from the tool’s schema.\n"
            "• No cycles; every id in \"after\" must exist.\n"
            "• If a single tool suffices, return a graph with one node only.\n"
            "\n"
            "Example (illustrative):\n"
            "{\n"
            "  \"graph\": {\n"
            "    \"nodes\": [\n"
            "      {\"id\":\"t1\",\"tool\":\"web_search\",\"args\":{\"query\":\"kayfabe definition\"},\"after\":[]},\n"
            "      {\"id\":\"t2\",\"tool\":\"summarize\",\"args\":{\"text\":\"[t1.output]\",\"length\":\"short\"},\"after\":[\"t1\"]}\n"
            "    ],\n"
            "    \"meta\": {\"goal\":\"Answer the user’s question concisely\"}\n"
            "  }\n"
            "}\n"
            "\n"
            "Return **ONLY** that JSON object—no markdown fences, no commentary."
            # ──────────────────────────────────────────────────────────────────
        )

        self.planning_prompt_select = self.cfg.get(
            "planning_prompt_select",
            # ── PASS 1: TOOL SELECTION ────────────────────────────────────────
            "You are the Planner (Selection Phase).\n"
            "Choose which tools are most appropriate to satisfy the user's request.\n"
            "Return ONLY valid JSON of the form:\n"
            "{ \"tools\": [\"<tool_name>\", ...] }\n"
            "\n"
            "Rules:\n"
            "• Use ONLY tool names from the *Available tools* list (names must match exactly).\n"
            "• Prefer the smallest sufficient set (often 1–3 tools).\n"
            "• Do NOT include arguments or prose; just the list.\n"
        )
        
        self.planning_prompt_fill = self.cfg.get(
            "planning_prompt_fill",
            # ── DAG PLANNER PROMPT (Filling Phase) ─────────────────────────────
            "You are the Planner (Filling Phase). Output **only** valid JSON for a small task-graph (DAG).\n"
            "Follow this schema exactly (no extra prose):\n"
            "{\n"
            "  \"graph\": {\n"
            "    \"nodes\": [\n"
            "      {\n"
            "        \"id\": \"t1\",                 // unique id per node\n"
            "        \"tool\": \"<tool_name>\",     // MUST match an available tool name exactly\n"
            "        \"args\": { /* named params matching the tool schema */ },\n"
            "        \"after\": []                  // ids of nodes that must complete before this one\n"
            "      }\n"
            "      // ≤ 6 nodes total; ≤ 3 runnable in any single layer\n"
            "    ],\n"
            "    \"meta\": {\n"
            "      \"goal\": \"<1–2 sentence summary of the user’s request>\",\n"
            "      \"constraints\": [ /* optional constraints */ ]\n"
            "    }\n"
            "  }\n"
            "}\n"
            "\n"
            "Rules:\n"
            "• Use ONLY tools from the provided *Selected Tools* list (names must match exactly).\n"
            "• Fill each node’s \"args\" using the tool’s docstring/description and its JSON schema (types/enums/ranges/defaults).\n"
            "• Keep the graph minimal: prefer 1–3 nodes; max 6; ≤ 3 runnable in parallel per layer.\n"
            "• To reference prior-node output inside args you may write **any** of:\n"
            "      \"[tX.output]\"                     // full output object\n"
            "      \"[tX.output.some.path]\"           // specific sub-field via dot-path\n"
            "      \"[ <tX>.output ]\"                // same as first (angle-brackets allowed)\n"
            "      \"{{tX}}\"                          // shorthand = full output\n"
            "  (Whitespace inside […] is ignored.)\n"
            "• Do NOT invent argument keys; use exactly the keys/types from the tool’s schema.\n"
            "• No cycles; every id in \"after\" must exist.\n"
            "• If a single tool suffices, return a graph with one node only.\n"
            "\n"
            "Example (illustrative):\n"
            "{\n"
            "  \"graph\": {\n"
            "    \"nodes\": [\n"
            "      {\"id\":\"t1\",\"tool\":\"web_search\",\"args\":{\"query\":\"kayfabe definition\"},\"after\":[]},\n"
            "      {\"id\":\"t2\",\"tool\":\"summarize\",\"args\":{\"text\":\"[t1.output]\",\"length\":\"short\"},\"after\":[\"t1\"]}\n"
            "    ],\n"
            "    \"meta\": {\"goal\":\"Answer the user’s question concisely\"}\n"
            "  }\n"
            "}\n"
            "\n"
            "Return **ONLY** that JSON object—no markdown fences, no commentary."
            # ──────────────────────────────────────────────────────────────────
        )

        self.toolchain_prompt = self.cfg.get(
            "toolchain_prompt",
            # ── TOOLCHAIN / EXECUTOR PROMPT ───────────────────────────────────
            "You will receive a single JSON object describing the next runnable DAG layer:\n"
            "{\n"
            "  \"ready_nodes\": [\n"
            "    { \"id\": \"t2\", \"tool\": \"<tool_name>\", \"args\": { /* may contain placeholders */ } },\n"
            "    { \"id\": \"t3\", \"tool\": \"<tool_name>\", \"args\": { /* … */ } }\n"
            "  ],\n"
            "  \"schemas\": { \"<tool_name>\": { /* JSON schema for that tool */ }, ... },\n"
            "  \"last_results\": { \"t1\": <raw output object>, \"tX\": <…> }\n"
            "}\n"
            "\n"
            "Your job:\n"
            "1) **Resolve placeholders** in each node’s args using last_results:\n"
            "     • \"[tX.output]\"  or \"[ <tX>.output ]\"  → substitute the full output\n"
            "     • dotted paths like \"[tX.output.text]\"   → substitute that sub-field\n"
            "     • \"{{tX}}\"                              → same as full output\n"
            "     • Whitespace inside […] is ignored.\n"
            "2) **Validate** args against the tool’s JSON schema:\n"
            "   • Drop unknown keys; keep types correct; do **NOT** invent new keys.\n"
            "   • If a required arg is still missing and no safe default exists, leave it blank (the executor may re-plan).\n"
            "3) Output **exactly** one JSON object (no prose) with parallel call-strings aligned to ready_nodes order:\n"
            "{\n"
            "  \"tool_calls\": [ \"toolA(arg1=...,arg2=...)\" , \"toolB(arg=...)\" ],\n"
            "  \"node_ids\":   [ \"t2\", \"t3\" ]\n"
            "}\n"
            "\n"
            "Do **not** add commentary, markdown, or extra fields. Return ONLY that JSON."
            # ──────────────────────────────────────────────────────────────────
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
            # ──────────────────────────────────────────────────────────────────
            "You will receive exactly two sections:\n"
            "1) [User query]  –  The user’s original question only, unchanged.\n"
            "2) [Tool outputs]  –  A list of each tool call and its JSON result from this session.\n\n"
            "• Do NOT include any assistant drafts, system logs, or previous responses in either section.\n"
            "• In your own words, summarize the user’s intent clearly and concisely.\n"
            "• Integrate every relevant fact from the provided plan and tool outputs;\n"
            "  do NOT invent, soften, or omit any detail.\n"
            "• Keep your answer human-readable and focused solely on the user’s original question.\n"
            "Ignore any empty sections.  Rely exclusively on the supplied tool outputs for factual data.\n"
            "Do not repeat the user’s phrasing verbatim nor include extraneous information.\n"
            "YOU DO NOT PRODUCE JSON OBJECTS, YOU PRODUCE HUMAN READABLE TEXT RESPONSES.\n"
            # ──────────────────────────────────────────────────────────────────
        )
        self.critic_prompt = self.cfg.get(
            "critic_prompt",
            # ──────────────────────────────────────────────────────────────────
            "You are the Assembler’s uncompromising Critic.  You will receive:\n"
            "• [User query] – The user’s exact question only.\n"
            "• [Plan]       – The JSON plan executed.\n"
            "• [Draft]      – The assistant’s initial draft response.\n"
            "• [Tool outputs] – Each tool call’s raw JSON result from this run.\n\n"
            "Critique the draft *only* using these inputs, ensuring you do not mistake any draft text for the user’s query.\n"
            "For each gap or failure:\n"
            "  1. Identify which tool call malfunctioned or lacked data.\n"
            "  2. Explain precisely how that gap affected the draft’s accuracy or completeness.\n"
            "  3. Propose a concrete correction or alternative approach.\n\n"
            "Then produce **only** the final, fully integrated answer that:\n"
            "• Corrects all identified issues.\n"
            "• Incorporates every valid tool output.\n"
            "• Exactly satisfies the user’s original intent.\n"
            "Return only that polished answer text—no JSON, no analysis, no extra commentary.\n"
            # ──────────────────────────────────────────────────────────────────
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
        self.editor_sys_prompt = self.cfg.get(
            "editor_sys_prompt",
            "You are an expert editor focused on completeness and clarity.\n"
            "Given the user’s question, the plan, merged context, tool outputs, and the draft:\n"
            "• Integrate any missing data points or corrections from the relevance bullets.\n"
            "• Improve structure, coherence, and ensure the answer fully satisfies the original intent.\n"
            "• Do NOT invent new facts; rely only on the provided context and tool outputs.\n"
            "• Return exactly the revised answer text, with no JSON or extra commentary."
        )
        self.extractor_sys_prompt = self.cfg.get(
            "extractor_sys_prompt",
            # ──────────────────────────────────────────────────────────────────
            "You are a Relevance Extractor.  Your task is to parse the entire context "
            "(user question, planning summary, merged knowledge snippets, and raw tool outputs) "
            "and produce a concise, bulleted list of exactly the facts, data points, "
            "or insights that must appear in the final answer.  Focus only on content "
            "directly tied to the user’s explicit intent; omit any irrelevant or redundant "
            "information.  Return **only** the bullet list, one bullet per line, with no "
            "additional commentary or JSON wrappers."
            # ──────────────────────────────────────────────────────────────────
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
            self.repo = repo
            self.context_path = self.repo.json_repo.path
        else:

            # ensure our storage directory exists
            base = Path("context_repos")
            base.mkdir(parents=True, exist_ok=True)

            # build per-chat filenames under that dir
            filename     = Path(context_path).name
            jsonl_file   = base / filename
            sqlite_file  = base / filename.replace(".jsonl", ".db")

            # create the Hybrid repo
            self.repo = HybridContextRepository(
                jsonl_path=str(jsonl_file),
                sqlite_path=str(sqlite_file),
                archive_max_mb=self.cfg.get("archive_max_mb", 10.0),
            )

            # remember the actual on‑disk JSONL path for later pruning
            self.context_path = str(jsonl_file)

        try:
            sanitize_jsonl(self.repo.json_repo.path)
        except Exception:
            pass
            
        tools.repo = self.repo            # for module-level tools
        tools.Tools.repo = self.repo      # for any methods on the Tools class

        self.memman = MemoryManager(self.repo)

        # RL / bandits (tools, prompt variants, knobs)
        self.bandit = RLManager(
            repo=self.repo,
            d_tool=self.cfg.get("linucb_d", 5),
            alpha=self.cfg.get("linucb_alpha", 0.75),
            reward_cfg=RewardConfig(
                latency_budget_ms=self.cfg.get("latency_budget_ms", 60_000)  # 60s default
            )
        )
        # per-turn prompt bookkeeping
        self._prompt_variants_cache: dict[str, dict[str, str]] = {}
        self._prompts_used_current: dict[str, str] = {}


        self._task_poll_event = threading.Event()
        self._task_bg_thread  = None
        # start a lightweight poller (2s) to launch due tasks and keep countdown fresh
        try:
            self._task_start_background(poll_seconds=self.cfg.get("task_poll_seconds", 2.0))
        except Exception:
            pass

        self.embed_text = embed_text          # ← ADD THIS LINE
        
        integrator_config = {
            # maximum number of nodes to keep in the graph at once
            "max_nodes": self.cfg.get("max_total_context", 50),
            # how many days before a context node expires
            "ttl_days": self.cfg.get("context_ttl_days", 30),
            # how many hops (or edges) to expand around your focus
            "expand_k": self.cfg.get("integrator_expand_k", 5),
        }

        # instantiate once, so it persists across turns
        self.integrator = GrandIntegrator(
            repo=self.repo,
            memory_manager=self.memman,
            config=integrator_config
        )
        # Metacognitive context keeper (rolling self-state echo)
        self.metacog_ctx = self._get_or_make_singleton(
            label="metacog_context",
            component="stage",
            tags=["metacognition"]
        )
        
        self._prompts_ready_evt = threading.Event()

        self._seed_tool_schemas()
        self._seed_static_prompts()
        self._prompts_ready_evt.set()

        self.tts_live_stages = set(
            self.cfg.get("tts_live_stages", [
            ])
        )

        # — text-to-speech manager —
        self.tts = tts_manager

        # TTS bridge placeholder (built once, reused per turn)
        self.tts_bridge = _LiveTTSBridge(self.tts, status_cb=lambda *_: None) if self.tts else None

        self._chat_contexts: set[int] = set()
        self._telegram_bot = None

        # Self-review background thread control
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
                self.memman.register_relationships(tmpl, embed_text)

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
                        self.memman.register_relationships(tmpl, embed_text)
                        self.curiosity_templates.append(tmpl)

        # — RLController for curiosity-template selection —
        self.curiosity_rl = RLController(
            stages=[t.semantic_label for t in self.curiosity_templates],
            alpha=self.cfg.get("curiosity_alpha", 0.1),
            path=self.cfg.get("curiosity_weights_path", "curiosity_weights.rl")
        )
        self.engine = ContextQueryEngine(
            repo=self.repo,
            embedder=embed_text,
            memman=self.memman
        )
        self._orig_stage10 = getattr(self, "_stage10_assemble_and_infer", None)
        self._stage10_assemble_and_infer = MethodType(Assembler._final_inference_override, self)



    # thread-safe cache
    _EMBED_CACHE: dict[str, np.ndarray] = {}
    _CACHE_LOCK = threading.Lock()
    _ZERO = np.zeros(768, dtype=float)


    def _task__soonest_due_seconds(self) -> float:
        """Return seconds until the soonest scheduled task (∞ if none)."""
        best = float("inf")
        for t in self._task_list():
            m = t.metadata or {}
            if m.get("status") != "scheduled":
                continue
            s = self._task__sec_until(m.get("due_at", ""))
            if s < best:
                best = s
        return best

    def _task__kick(self):
        """Wake the poller to recompute its sleep immediately."""
        try:
            self._task_wake_event.set()
        except Exception:
            pass

    def _task__acquire_singleton_lock(self) -> bool:
        """
        Best-effort cross-process lock so only one poller runs.
        Others skip starting the background thread.
        """
        try:
            lock_path = os.path.join(os.path.dirname(self.context_path) or ".", ".task_poller.lock")
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            fh = open(lock_path, "a+b")
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    self._task_lock_file = fh
                    return True
                except OSError:
                    fh.close()
                    return False
            else:
                import fcntl
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._task_lock_file = fh
                    return True
                except BlockingIOError:
                    fh.close()
                    return False
        except Exception:
            return True  # fail-open: better to have a poller than none

    def _task__release_singleton_lock(self):
        try:
            fh = getattr(self, "_task_lock_file", None)
            if not fh:
                return
            if os.name == "nt":
                import msvcrt
                try:
                    fh.seek(0); msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                finally:
                    fh.close()
            else:
                import fcntl
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()
            self._task_lock_file = None
        except Exception:
            pass



    async def _emit_provisional(
        self,
        user_text: str,
        state: dict,
        status_cb: Callable[[str, Any], None],
        on_token: Callable[[str], None] | None,
    ) -> str:
        """
        Fire a super-fast draft answer from already merged context (no tools).
        Streams tokens to TTS immediately.
        """
        # Build a minimal prompt from what we already have
        merged_txt = "\n".join(
            (c.summary or "")[:400] for c in state.get("merged", [])[:15]
        )
        clar_notes = (state.get("clar_ctx") and state["clar_ctx"].metadata.get("notes", "")) or ""
        sys = (
            "You are the FastResponder. Give a 1–3 sentence helpful answer NOW, "
            "based ONLY on what you see. Say you'll refine after tools if needed."
        )
        usr = (
            f"User said: {user_text}\n\n"
            f"Clarified intent: {clar_notes}\n\n"
            f"Relevant snippets:\n{merged_txt}"
        )

        # Stream with token callback → feeds TTS bridge
        provisional = await self._stream_and_capture_async(
            self.primary_model,
            [{"role":"system","content":sys},{"role":"user","content":usr}],
            tag="[Provisional]",
            on_token=on_token
        )

        provisional = provisional.strip()
        if provisional:
            status_cb("provisional_answer", provisional)
            # queue to TTS file pipeline as well so Telegram pump sees it
            if getattr(self, "tts", None):
                try: self.tts.enqueue(provisional)
                except Exception: pass
        return provisional

    def _hash8(self, text: str) -> str:
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8]

    def _gather_dynamic_patches(self) -> list[str]:
        """
        Collect small dynamic policy snippets (created by system_prompt_refine).
        We treat the concatenation as a *variant patch* on top of a slot’s base prompt.
        """
        patches = []
        try:
            for row in self.repo.query(lambda c: c.component == "policy" and "dynamic_prompt" in (c.tags or [])):
                txt = (row.metadata.get("policy") or row.summary or "").strip()
                if txt:
                    patches.append(txt)
        except Exception:
            pass
        return patches

    def _build_prompt_variants(self, slot: str, base_text: str) -> dict[str, str]:
        """
        Build 2–3 variants for a prompt slot from (a) base, (b) base+all patches,
        (c) an ultra-concise nudge. Keys are stable variant_ids.
        """
        variants: dict[str, str] = {}

        # (A) baseline
        vA = base_text.strip()
        variants[f"{slot}:base:{self._hash8(vA)}"] = vA

        # (B) baseline + dynamic patches (if any)
        patches = self._gather_dynamic_patches()
        if patches:
            vB = (vA + "\n\n" + "\n".join(patches)).strip()
            variants[f"{slot}:dyn:{self._hash8(vB)}"] = vB

        # (C) a shortness/style nudge (cheap exploration)
        vC = (vA + "\n\n" + "STYLE OVERRIDE: Prefer fewer words; tighten phrasing.").strip()
        variants[f"{slot}:tight:{self._hash8(vC)}"] = vC

        return variants

    def _choose_prompt_text(self, slot: str, base_text: str) -> tuple[str, str]:
        """
        Return (chosen_text, variant_id) using Thompson sampling w/ canary ramping.
        Caches the variant texts for this turn; also records self._prompts_used_current.
        """
        variants = self._build_prompt_variants(slot, base_text)
        self._prompt_variants_cache.setdefault(slot, {}).update(variants)

        chosen_id = self.bandit.choose_prompt_variant(slot, list(variants.keys()))
        if chosen_id not in variants:
            # fallback to baseline
            chosen_id = next(iter(variants.keys()))
        chosen_text = variants[chosen_id]
        self._prompts_used_current[slot] = chosen_id
        return chosen_text, chosen_id

    
    def _prune_jsonl_duplicates(self) -> None:
        """
        Logically dedupe the repo JSONL:
        - For 'prompt': keep newest per semantic_label.
        - For 'schema': keep newest per semantic_label (fallback to parsed schema.name).
        - For everything else: keep newest per context_id.
        Malformed lines -> *.corrupt

        Also canonicalizes metadata["schema"] to eliminate whitespace-induced mismatches.
        """

        path         = self.repo.json_repo.path
        corrupt_path = path + ".corrupt"

        total = bad = 0
        keep_by_key: dict[str, dict] = {}

        def _canon_schema_str(s: str) -> str:
            try:
                obj = json.loads(s)
                return json.dumps(obj, sort_keys=True, separators=(',', ':'))
            except Exception:
                return s  # leave as-is if not valid JSON

        def _logical_key(obj: dict) -> str | None:
            comp = obj.get("component", "")
            cid  = obj.get("context_id", "")
            ts   = obj.get("timestamp", "")
            if not isinstance(cid, str) or not isinstance(ts, str):
                return None

            if comp == "prompt":
                label = (obj.get("semantic_label") or "").strip()
                return f"prompt::{label}" if label else None

            if comp == "schema":
                label = (obj.get("semantic_label") or "").strip()
                if not label:
                    # fallback: parse schema.name from metadata
                    try:
                        blob = json.loads(obj.get("metadata", {}).get("schema", "{}"))
                        name = (blob.get("name") or "").strip()
                        if name:
                            label = name
                    except Exception:
                        pass
                return f"schema::{label}" if label else f"cid::{cid}"

            # everything else: dedupe by context_id
            return f"cid::{cid}"

        with open(path, "r", encoding="utf8") as infile, \
            open(corrupt_path, "a", encoding="utf8") as badf:
            for line in infile:
                total += 1
                try:
                    obj = json.loads(line)
                    key = _logical_key(obj)
                    if not key:
                        raise ValueError("no logical key")
                except Exception:
                    bad += 1
                    badf.write(line)
                    continue

                # Canonicalize embedded schema JSON strings
                if obj.get("component") == "schema":
                    meta = obj.get("metadata") or {}
                    sch  = meta.get("schema")
                    if isinstance(sch, str):
                        meta["schema"] = _canon_schema_str(sch)
                        obj["metadata"] = meta

                prev = keep_by_key.get(key)
                if prev is None or obj["timestamp"] > prev["timestamp"]:
                    keep_by_key[key] = obj

        survivors = sorted(keep_by_key.values(), key=lambda o: o["timestamp"])

        tmp_dir = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
        with os.fdopen(fd, "w", encoding="utf8") as out:
            for o in survivors:
                out.write(json.dumps(o, separators=(',', ':')) + "\n")

        # POSIX/Windows: atomic-ish replace
        os.replace(tmp_path, path)

        print(f"[prune_jsonl_duplicates] {total} read, {len(survivors)} kept, {bad} malformed → wrote {path}")




    def _seed_tool_schemas(self) -> None:
        """
        Idempotent tool-schema seeding (no external lock).
        - Key on semantic_label (fallback to parsed schema.name).
        - Keep newest per label; delete older dups.
        - Canonicalize JSON before compare/store to avoid whitespace diffs.
        - Demote non-canonical tools to legacy.
        - If duplicates or updates happened, sanitize + logical prune.
        """

        def canon(obj) -> str:
            return json.dumps(obj, sort_keys=True, separators=(',', ':'))

        changed_or_dupes = False

        # 1) Build canonical set
        try:
            Tools.generate_all_tool_schemas()
        except Exception:
            return
        canonical = {name: schema for name, schema in TOOL_SCHEMAS.items()}
        if not canonical:
            return

        # 2) Read all schema rows (active + legacy so we can normalize)
        rows = list(self.repo.query(
            lambda c: c.component == "schema" and any(t in (c.tags or []) for t in ("tool_schema", "legacy_tool_schema"))
        ))

        def label_for(ctx) -> str:
            lbl = (ctx.semantic_label or "").strip()
            if lbl:
                return lbl
            # fallback: extract name from metadata.schema
            try:
                blob = json.loads(ctx.metadata.get("schema", "{}"))
                name = (blob.get("name") or "").strip()
                return name or f"__missing__::{ctx.context_id}"
            except Exception:
                return f"__missing__::{ctx.context_id}"

        # 3) Bucket and dedupe: keep newest per label
        buckets: dict[str, list[ContextObject]] = {}
        for ctx in rows:
            buckets.setdefault(label_for(ctx), []).append(ctx)

        keepers: dict[str, ContextObject] = {}
        for lbl, lst in buckets.items():
            if len(lst) > 1:
                lst.sort(key=lambda c: c.timestamp, reverse=True)
                keeper, dups = lst[0], lst[1:]
                for d in dups:
                    try:
                        self.repo.delete(d.context_id)
                        changed_or_dupes = True
                    except Exception:
                        pass
                keepers[lbl] = keeper
            else:
                keepers[lbl] = lst[0]

        present_labels = { (ctx.semantic_label or label_for(ctx)).strip(): ctx for ctx in keepers.values() }

        # 4) Upsert/normalize canonical
        for name, want in canonical.items():
            want_json = canon(want)
            cur = present_labels.get(name)
            if cur is None:
                # INSERT
                sc = ContextObject.make_schema(
                    label=name,
                    schema_def=want_json,
                    tags=["artifact", "tool_schema"],
                )
                sc.semantic_label = name
                sc.touch()
                self.repo.save(sc)
                changed_or_dupes = True
                present_labels[name] = sc
                continue

            # UPDATE only if content or tags differ
            try:
                have_json = canon(json.loads(cur.metadata.get("schema", "{}")))
            except Exception:
                have_json = ""
            need_update = (have_json != want_json)
            tags = set(cur.tags or [])
            if "tool_schema" not in tags:
                # promote legacy back to active if it's canonical
                tags.discard("legacy_tool_schema")
                tags.add("tool_schema")
                need_update = True
            if (cur.semantic_label or "").strip() != name:
                cur.semantic_label = name
                need_update = True
            if need_update:
                cur.metadata["schema"] = want_json
                cur.tags = sorted(tags | {"artifact"})
                cur.touch()
                self.repo.save(cur)
                changed_or_dupes = True

        # 5) Demote any non-canonical active schemas to legacy
        canonical_names = set(canonical.keys())
        for lbl, ctx in list(present_labels.items()):
            if lbl and lbl not in canonical_names:
                tags = set(ctx.tags or [])
                if "tool_schema" in tags:
                    tags.remove("tool_schema")
                    tags.add("legacy_tool_schema")
                    ctx.tags = sorted(tags)
                    ctx.touch()
                    self.repo.save(ctx)
                    changed_or_dupes = True

        # 6) Sanitize + logical prune only if something changed/dupes existed
        if changed_or_dupes:
            jsonl_path = self.repo.json_repo.path
            try:
                sanitize_jsonl(jsonl_path)
            finally:
                # prune does canonicalization of schema strings too
                self._prune_jsonl_duplicates()
                

    def _seed_static_prompts(self) -> None:
        """
        Idempotent seeding of static system prompts with a cross-process lock.

        • Fast-path: if we’ve already seeded in this process, do nothing.
        • Pre-flight: if _all_ semantic_labels are already in the repo, do nothing.
        • Otherwise: acquire lock, bucket, insert missing, delete duplicates.
        """

        # ─── FAST-PATH GUARD ───────────────────────────────────────────
        if getattr(self, "_static_prompts_seeded", False):
            return

        # ---- small cross-process lock (POSIX flock / Windows msvcrt) ----
        class _Lock:
            def __init__(self, path: str, timeout: float = 30.0, poll: float = 0.1):
                self.path, self.timeout, self.poll = path, timeout, poll
                self._fh = None
            def __enter__(self):
                lock_dir = os.path.dirname(self.path) or "."
                os.makedirs(lock_dir, exist_ok=True)
                self._fh = open(self.path, "a+b")
                start = time.time()
                if os.name == "nt":
                    import msvcrt
                    while True:
                        try:
                            msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                            break
                        except OSError:
                            if time.time() - start > self.timeout:
                                self._fh.close(); raise TimeoutError(f"Timeout acquiring lock {self.path}")
                            time.sleep(self.poll)
                else:
                    import fcntl
                    while True:
                        try:
                            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.time() - start > self.timeout:
                                self._fh.close(); raise TimeoutError(f"Timeout acquiring lock {self.path}")
                            time.sleep(self.poll)
                return self
            def __exit__(self, *_):
                try:
                    if os.name == "nt":
                        import msvcrt
                        try:
                            self._fh.seek(0); msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                        finally:
                            self._fh.close()
                    else:
                        import fcntl
                        try:
                            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                        finally:
                            self._fh.close()
                finally:
                    pass  # keep the lock file

        # 1) Canonical source-of-truth dict (we do not overwrite existing text)
        self.system_prompts = {
            "clarifier_prompt":        self.clarifier_prompt,
            "assembler_prompt":        self.assembler_prompt,
            "inference_prompt":        self.inference_prompt,
            "planning_prompt":         self.planning_prompt,
            "planning_prompt_select":  self.planning_prompt_select,
            "planning_prompt_fill":   self.planning_prompt_fill,
            "toolchain_prompt":        self.toolchain_prompt,
            "reflection_prompt":       self.reflection_prompt,
            "toolchain_retry_prompt":  self.toolchain_retry_prompt,
            "final_inference_prompt":  self.final_inference_prompt,
            "critic_prompt":           self.critic_prompt,
            "narrative_mull_prompt":   self.narrative_mull_prompt,
            "extractor_sys_prompt":    self.extractor_sys_prompt,
            "editor_sys_prompt":       self.editor_sys_prompt,
        }
        static_labels = set(self.system_prompts.keys())

        # ─── PRE-FLIGHT CHECK ──────────────────────────────────────────
        existing_labels = {
            (c.semantic_label or "").strip()
            for c in self.repo.query(lambda c: c.component == "prompt")
        }
        if existing_labels.issuperset(static_labels):
            # nothing to do
            self._static_prompts_seeded = True
            return

        # 2) Bucket existing by normalized semantic_label
        buckets: dict[str, list[ContextObject]] = {}
        for ctx in self.repo.query(lambda c: c.component == "prompt"):
            lbl = (ctx.semantic_label or "").strip()
            if lbl:
                buckets.setdefault(lbl, []).append(ctx)

        lock_path = os.path.join(os.path.dirname(self.repo.json_repo.path) or ".", ".seed_static_prompts.lock")
        changed_or_dupes = False

        with _Lock(lock_path):
            # 3) Ensure existence per label; dedupe extras; never touch existing content
            for label, desired_text in self.system_prompts.items():
                lbl = label.strip()
                rows = buckets.get(lbl, [])
                if not rows:
                    # missing → insert once
                    new_ctx = ContextObject.make_prompt(
                        label=lbl,
                        prompt_text=desired_text,
                        tags=["artifact", "prompt"],
                    )
                    new_ctx.touch()
                    self.repo.save(new_ctx)
                    buckets.setdefault(lbl, []).append(new_ctx)
                    changed_or_dupes = True
                elif len(rows) > 1:
                    # dedupe extras
                    rows.sort(key=lambda c: c.timestamp, reverse=True)
                    keeper, dups = rows[0], rows[1:]
                    for dup in dups:
                        try:
                            self.repo.delete(dup.context_id)
                            changed_or_dupes = True
                        except Exception:
                            pass
                    buckets[lbl] = [keeper]

        # 4) Clean only if something changed or dupes were found
        if changed_or_dupes:
            jsonl_path = self.repo.json_repo.path
            try:
                sanitize_jsonl(jsonl_path)
            finally:
                self._prune_jsonl_duplicates()

        # mark that we’ve done this once
        self._static_prompts_seeded = True



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
            self.memman.register_relationships(ctx, embed_text)

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
        """
        Build (or fetch) the singleton narrative_context exactly once per turn,
        then dedupe and purge any vestigial narrative entries before
        reassembling the keeper’s narrative.
        """
        # If we've already built it this turn, return the cached keeper.
        if getattr(self, "_narrative_loaded", False):
            return self._narrative_cache  # type: ignore[attr-defined]

        # Mark as built for this turn
        self._narrative_loaded = True

        # 1) get or create the one true keeper
        keeper = self._get_or_make_singleton(
            label="narrative_context",
            component="stage",
            tags=["narrative"],
        )

        # 2) fetch all raw narrative entries (exclude the keeper itself)
        raw = [
            c for c in self.repo.query(lambda c: c.component == "narrative")
            if c.context_id != keeper.context_id
        ]
        raw.sort(key=lambda c: c.timestamp)

        # 3) dedupe by summary text, collect duplicates for deletion
        seen: set[str] = set()
        unique: list[ContextObject] = []
        duplicates: list[ContextObject] = []
        for entry in raw:
            text = entry.summary or ""
            if text in seen:
                duplicates.append(entry)
            else:
                seen.add(text)
                unique.append(entry)

        # 4) purge any duplicate ContextObjects from the repo
        for dup in duplicates:
            self.repo.delete(dup.context_id)

        # 5) stitch together the keeper’s metadata from the deduped list
        narrative_text = "\n".join(n.summary or "" for n in unique)
        keeper.metadata["narrative"] = narrative_text
        keeper.summary = narrative_text or "(no narrative yet)"
        keeper.references = [n.context_id for n in unique]

        # 6) persist and re-embed
        keeper.touch()
        self.repo.save(keeper)
        self.memman.register_relationships(keeper, embed_text)

        # cache it so subsequent calls in this turn are no-ops
        self._narrative_cache = keeper  # type: ignore[attr-defined]
        return keeper

    
    def _load_arbitrary_context(
        self,
        semantic_label: str = "narrative_context",
        component: str = "stage",
        tags: list[str] | None = None,
    ) -> ContextObject:
        # normalize tags and ensure we always include at least 'narrative'
        tags = list({*(tags or []), "narrative"})

        # get or create our singleton keeper
        keeper = self._get_or_make_singleton(
            label=semantic_label,
            component=component,
            tags=tags,
        )

        # pull all contexts of the requested component *and* matching any of our tags
        ctx_objs = self.repo.query(
            lambda c: c.component == component and any(t in c.tags for t in tags)
        )
        # sort chronologically
        ctx_objs.sort(key=lambda c: c.timestamp)

        # concatenate their summaries
        joined = "\n".join((c.summary or "") for c in ctx_objs)

        # write it back into metadata under our semantic_label key
        keeper.metadata[semantic_label] = joined
        keeper.summary = joined or f"(no {semantic_label} yet)"
        keeper.references = [c.context_id for c in ctx_objs]

        keeper.touch()
        self.repo.save(keeper)

        # re-embed the fresh blob so similarity searches reflect the update
        self.memman.register_relationships(keeper, embed_text)
        return keeper


    def _get_history(self) -> List[ContextObject]:
        segs = self.repo.query(
            lambda c: c.domain=="segment"
            and c.component in ("user_input","assistant")
        )
        segs.sort(key=lambda c: c.timestamp)
        return segs[-self.hist_k:]

    def _print_stage_context(self, name: str, sections: Dict[str, Any]):
        """
        Pretty-prints the stage-debug context.

        ── Features ───────────────────────────────────────────────────────────
        • Console width auto-detected (fallback 120 columns).
        • BEGIN / END banners use a █ ▓ ▒ ░ gradient.
        • Every subsection is isolated inside a boxed block:
            ▛▀▀ START … ▀▀▜
            ▌  …content…  ▐
            ▙▄▄ END   … ▄▄▟
        • All lines are wrapped and padded to fit neatly inside the box.
        """

        # ── 1) Console dimensions ────────────────────────────────────────────
        W = max(60, shutil.get_terminal_size(fallback=(120, 20)).columns)
        INNER = W - 4                       # room for "▌ " … " ▐"

        # ── 2) Gradient helpers for main banners ─────────────────────────────
        SHADES = ['█', '▓', '▒', '░']       # heavy → light

        def _gradient(n: int, rev: bool = False) -> str:
            if n <= 0:
                return ''
            seq = SHADES[::-1] if rev else SHADES
            steps = len(seq) - 1
            return ''.join(seq[round(i * steps / (n - 1))] for i in range(n))

        def _main_banner(text: str, tag: str) -> str:
            label = f"[{tag}: {text}]"
            if len(label) >= W:
                return label[:W]
            remain = W - len(label)
            left = _gradient(remain // 2, rev=False)
            right = _gradient(remain - len(left), rev=True)
            return left + label + right

        # ── 3) Box helpers for subsections ───────────────────────────────────
        # Corners: ▛ ▜  (top)   ▙ ▟ (bottom)   verticals: ▌ ▐
        def _top_box(label: str) -> str:
            lbl = f" START {label} "
            fill = max(0, W - len(lbl) - 2)
            left, right = fill // 2, fill - (fill // 2)
            return "▛" + "▀" * left + lbl + "▀" * right + "▜"

        def _bot_box(label: str) -> str:
            lbl = f" END   {label} "
            fill = max(0, W - len(lbl) - 2)
            left, right = fill // 2, fill - (fill // 2)
            return "▙" + "▄" * left + lbl + "▄" * right + "▟"

        def _boxed_lines(raw: Any) -> None:
            # Convert raw → list[str]
            if isinstance(raw, str):
                lines = raw.splitlines() or ["(empty)"]
            elif isinstance(raw, list):
                lines = [str(x) for x in (raw or ["(empty)"])]
            else:                      # pretty-print dicts / objects
                try:
                    lines = json.dumps(raw, ensure_ascii=False, indent=2).splitlines()
                except Exception:
                    lines = textwrap.dedent(repr(raw)).splitlines()

            for ln in lines:
                for seg in textwrap.wrap(ln, width=INNER) or ['']:
                    print(f"▌ {seg.ljust(INNER)} ▐")

        # ── 4) Print everything ──────────────────────────────────────────────
        print("\n" + _main_banner(name, "BEGIN"))
        for title, content in sections.items():
            print(_top_box(title))
            _boxed_lines(content)
            print(_bot_box(title) + "\n")
        print(_main_banner(name, "END") + "\n")


    def _save_stage(self, ctx: ContextObject, stage: str):
        ctx.stage_id = stage
        ctx.summary = (
            (ctx.references and
             (ctx.metadata.get("plan") or ctx.metadata.get("tool_call")))
            or ctx.summary
        )
        ctx.touch()
        self.repo.save(ctx)
        self.memman.register_relationships(ctx, embed_text)


    def _persist_and_index(self, ctxs: list[ContextObject]):
        for ctx in ctxs:
            ctx.touch()
            self.repo.save(ctx)
        # one bulk ingest is cheaper than N singles
        self.integrator.ingest(ctxs)


    def _estimate_tokens(self, text: str) -> int:
        """
        Very rough token estimator (~4 chars/token). Safe for budgets.
        """
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    def _compute_budgets(self, state: dict) -> dict:
        """
        Decide how big the window should be based on clarifier breadth/depth
        and whether tools were used. The values are small enough to keep the
        final prompt readable, big enough to be useful.
        """
        clar = state.get("clar_ctx")
        kw = (clar and clar.metadata.get("keywords")) or []
        notes = (clar and clar.metadata.get("notes")) or ""
        note_words = len(notes.split())
        tool_layers = len(state.get("tools_list") or [])
        broad = (len(kw) >= 6) or (note_words > 60) or (tool_layers >= 8)

        # Defaults intentionally conservative
        base = {
            "history":        3,
            "semantic":       4,
            "memory":         3,
            "know_snippets":  6,
            "tools":          6,
            "max_tokens": 1800
        }

        if broad:
            base.update({
                "history":        5,
                "semantic":       8,
                "memory":         6,
                "know_snippets": 10,
                "tools":          8,
                "max_tokens": 2600
            })
        # Micro/simple episode
        if note_words < 15 and len(kw) <= 3:
            base.update({
                "history":        2,
                "semantic":       2,
                "memory":         2,
                "know_snippets":  4,
                "tools":          4,
                "max_tokens": 1200
            })
        return base

    def _pick_ctx(self, ctxs: list, n: int) -> list:
        """
        Pick up to n context objects prioritizing:
        1) higher retrieval_score if present
        2) more recent timestamps
        """
        if not ctxs or n <= 0:
            return []
        def _score(c):
            rs = float(getattr(c, "retrieval_score", 0.0) or 0.0)
            ts = getattr(c, "timestamp", "") or ""
            return (rs, ts)
        return sorted(ctxs, key=_score, reverse=True)[:n]

    def _collect_tool_successes(self, tool_ctxs: list) -> list[dict]:
        """
        Normalize tool outputs into a compact, consistent shape, success-only.
        """
        out = []
        for t in tool_ctxs or []:
            meta = t.metadata or {}
            if meta.get("exception") is not None:
                continue
            tool_name = meta.get("tool_name", t.semantic_label or "tool")
            call      = meta.get("call") or meta.get("tool_call") or ""
            output    = meta.get("output")
            # Avoid huge blobs; light truncation happens later
            out.append({
                "id": t.context_id,
                "tool": tool_name,
                "call": call,
                "result": output
            })
        return out

    def _curate_final_window(self, state: dict, budgets: dict) -> dict:
        """
        Build a compact, de-duplicated window across the most useful buckets.
        Returns:
        {
            "history": [str...],
            "semantic": [str...],
            "memory": [str...],
            "knowledge": [str...],
            "tools": [ {tool, call, result}... ],
            "token_estimate": int
        }
        """
        window = {"history": [], "semantic": [], "memory": [], "knowledge": [], "tools": []}
        tokens = 0

        # History (recent turns already in state["history"])
        hist = self._pick_ctx(state.get("history", []), budgets["history"])
        for h in hist:
            s = (h.summary or "").strip()
            if not s:
                continue
            window["history"].append(s)
            tokens += self._estimate_tokens(s)

        # Semantic & memory (if present in state)
        sem = self._pick_ctx(state.get("semantic", []), budgets["semantic"])
        mem = self._pick_ctx(state.get("assoc", []), budgets["memory"])

        for c in sem:
            s = (c.summary or "").strip()
            if not s:
                continue
            window["semantic"].append(s)
            tokens += self._estimate_tokens(s)

        for c in mem:
            s = (c.summary or "").strip()
            if not s:
                continue
            window["memory"].append(s)
            tokens += self._estimate_tokens(s)

        # Knowledge snippets (already flattened by upstream stages)
        kn = (state.get("know_snippets") or [])[:budgets["know_snippets"]]
        for s in kn:
            s = (s or "").strip()
            if not s:
                continue
            window["knowledge"].append(s)
            tokens += self._estimate_tokens(s)

        # Tools (as compact JSON-like rows; truncation below)
        tools_norm = self._collect_tool_successes(state.get("tool_ctxs", []))[:budgets["tools"]]
        window["tools"] = tools_norm

        window["token_estimate"] = tokens
        return window

    def _compose_dynamic_system_prompt(self, state: dict, budgets: dict) -> str:
        """
        Patch the final system prompt with clarifier-derived intent,
        answer style hints, and explicit instruction to respect
        the curated window.
        """
        base = self._get_prompt("final_inference_prompt").strip() or (
            "You will receive two sections and must answer only from them."
        )
        clar = state.get("clar_ctx")
        kws  = (clar and clar.metadata.get("keywords")) or []
        notes = (clar and clar.metadata.get("notes")) or ""
        style = []
        if len(kws) <= 3:
            style.append("Be concise.")
        if len(kws) >= 6:
            style.append("Prefer structure: short paragraphs or bullets.")
        if "compare" in " ".join(kws).lower():
            style.append("Include a comparison where relevant.")

        patch = "\n".join([
            "— CURATED CONTEXT RULES —",
            f"- Final token budget ~{budgets['max_tokens']} (soft).",
            "- Use the curated context window only; do not speculate outside it.",
            "- Integrate tool results exactly as provided; do not invent fields.",
            ("- " + " ".join(style)) if style else "",
            ("- Intent notes: " + notes) if notes else "",
            ("- Focus keywords: " + ", ".join(kws[:8])) if kws else "",
        ]).strip()

        return base + "\n\n" + patch

    def _summarize_tools_for_prompt(self, tools: list[dict], max_per_tool_chars: int = 1000) -> list[dict]:
        """
        Trim oversized tool outputs to keep the prompt lean, preserving keys.
        """
        out = []
        for row in tools:
            tool = row.get("tool", "tool")
            call = row.get("call", "")
            result = row.get("result")
            # Normalize result → compact JSON-safe structure with trimming
            if isinstance(result, str):
                trimmed = (result[:max_per_tool_chars] + "…") if len(result) > max_per_tool_chars else result
            elif isinstance(result, (int, float, bool)) or result is None:
                trimmed = result
            else:
                try:
                    blob = json.dumps(result, ensure_ascii=False)
                except Exception:
                    blob = str(result)
                trimmed = (blob[:max_per_tool_chars] + "…") if len(blob) > max_per_tool_chars else blob

            out.append({
                "tool": tool,
                "call": call,
                "result": trimmed
            })
        return out

    def _final_inference_override(self, user_text: str, state: dict) -> str:
        """
        Replacement for _stage10_assemble_and_infer:
        - Computes budgets from the clarifier
        - Builds a compact, ordered context window
        - Patches the final system prompt dynamically
        - Streams the final answer using ONLY curated inputs
        """
        try:
            budgets = self._compute_budgets(state)
            window  = self._curate_final_window(state, budgets)
            sysmsg  = self._compose_dynamic_system_prompt(state, budgets)

            # Build user payload with strict sections and curated window
            # Section 1 — user query
            sec1 = f"[User query]\n{user_text.strip()}\n"

            # (Optional) Section 1.5 — compressed context window to guide reasoning
            # Keep it small & readable; this *supplements* but does not replace tools.
            lines = []
            if window["history"]:
                lines.append("• History: " + " | ".join(window["history"][:3]))
            if window["semantic"]:
                lines.append("• Similar context: " + " | ".join(window["semantic"][:3]))
            if window["memory"]:
                lines.append("• Memory: " + " | ".join(window["memory"][:3]))
            if window["knowledge"]:
                lines.append("• Knowledge: " + " | ".join(window["knowledge"][:4]))
            ctx_window_text = "\n".join(lines)

            # Section 2 — tool outputs (trimmed)
            tools_compact = self._summarize_tools_for_prompt(window["tools"])
            sec2 = "[Tool outputs]\n" + json.dumps(tools_compact, ensure_ascii=False, indent=2)

            user_payload = sec1
            if ctx_window_text:
                user_payload += "\n[Context window]\n" + ctx_window_text + "\n"
            user_payload += "\n" + sec2

            # Stream final answer
            out = self._stream_and_capture(
                self.primary_model,
                [{"role": "system", "content": sysmsg},
                {"role": "user",   "content": user_payload}],
                tag="[FinalOverride]"
            ).strip()

            # Stash for upstream callers that read state
            state["final_window"] = window
            state["final_system_prompt"] = sysmsg
            state["final"] = out
            return out
        except Exception:
            # Fall back to original stage10 if bound and working
            if getattr(self, "_orig_stage10", None):
                try:
                    return self._orig_stage10(user_text, state)
                except Exception:
                    pass
            # Last-resort: minimal answer from merged context
            fallback = " ".join((getattr(c, "summary", "") or "") for c in (state.get("merged") or [])[:3])
            sysmsg = "Answer succinctly using only the provided material."
            payload = f"[User query]\n{user_text}\n\n[Tool outputs]\n[]\n\n[Context]\n{fallback}"
            return self._stream_and_capture(
                self.primary_model,
                [{"role": "system", "content": sysmsg},
                {"role": "user",   "content": payload}],
                tag="[FinalFallback]"
            ).strip()

    # ————————————————————————————————————————————————————————————
    # Gemma-3 prompt builder
    def _gemma_format(self, messages: list[dict[str, str]]) -> str:
        """
        Collapse an OpenAI-style messages array into Gemma-3’s two-role
        format.  Any `system` messages become the “instructions” section,
        and the *last* `user` message is treated as the question.
        """
        # 1) split streams
        sys_parts  = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [m["content"] for m in messages if m["role"] == "user"]
        if not user_parts:
            raise ValueError("Gemma formatter needs at least one user message")

        # 2) build canonical block
        block  = "<start_of_turn>user\n"
        if sys_parts:
            block += "# ——— SYSTEM INSTRUCTIONS ———\n" + "\n".join(sys_parts) + "\n"
        block += "# ——— USER QUESTION ———\n" + user_parts[-1]   # keep only newest
        block += "<end_of_turn>\n<start_of_turn>model\n"
        return block
        

    def _extract_image_b64(self, text: str, *, max_bytes: int = 8 * 1024 * 1024) -> list[str]:
        """
        Scan *text* for image-like tokens and return a list of base-64 strings
        ready for Ollama’s  `images=[ … ]` parameter.

        Recognised forms
        ─────────────────
          • HTTP/HTTPS URLs ending in .jpg/.jpeg/.png/.bmp/.gif/.webp
          • Absolute/relative POSIX paths   (/foo/bar.png,  ./pic.jpg,  ../x.webp)
          • Windows-style paths             (C:\\images\\cat.jpeg)
          • Home-relative paths             (~/Downloads/photo.png)

        Safety guards
        ─────────────
          • Any item > *max_bytes* is skipped.
          • Network fetches use streaming + 5 s timeout.
        """
        # full list of accepted extensions
        exts = r"(?:jpg|jpeg|png|bmp|gif|webp)"

        pattern = rf"""
            (?P<url>https?://\S+?\.{exts}) |               # remote
            (?P<path>
                (?:~|\.{1,2}|[A-Za-z]:)?[^\s"'<>|]+\.{exts} # local
            )
        """

        imgs_b64: list[str] = []
        for m in re.finditer(pattern, text, re.IGNORECASE | re.VERBOSE):
            loc = m.group().strip()

            try:
                # ── Remote URL ──────────────────────────────────────────────
                if loc.lower().startswith(("http://", "https://")):
                    with requests.get(loc, timeout=5, stream=True) as resp:
                        resp.raise_for_status()
                        data = resp.raw.read(max_bytes + 1)
                        if len(data) > max_bytes:
                            continue  # too large
                # ── Local file path ─────────────────────────────────────────
                else:
                    p = Path(loc).expanduser().resolve()
                    if not p.is_file() or p.stat().st_size > max_bytes:
                        continue
                    data = p.read_bytes()

                imgs_b64.append(base64.b64encode(data).decode("ascii"))

            except Exception:
                # swallow any fetch/IO error
                continue

        return imgs_b64
    
    def _b64_from_paths(self, paths: List[str], *, max_bytes: int = 8 * 1024 * 1024) -> List[str]:
        """
        Given absolute file paths, load and base-64-encode each image
        (skipping any > max_bytes).  Returns the unique, ordered list.
        """
        out, seen = [], set()
        for p in paths:
            try:
                if p in seen or not os.path.isfile(p) or os.path.getsize(p) > max_bytes:
                    continue
                with open(p, "rb") as fh:
                    out.append(base64.b64encode(fh.read()).decode("ascii"))
                    seen.add(p)
            except Exception:
                continue
        return out
        
    def _stream_and_capture(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        tag: str = "",
        max_image_bytes: int = 8 * 1024 * 1024,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
        _is_fallback: bool = False,
    ) -> str:
        """
        Stream a response with:
        • automatic image-inlining
        • token‐level callback
        • run‐away guards (token, line, pattern, multi-token, fuzzy‐phrase, n-gram, erosion)
        • crash resilience + retry loop
        • optional fallback to secondary model
        """
        import re
        import time
        import requests
        from pathlib import Path
        from collections import deque
        from difflib import SequenceMatcher
        # assume chat and _OllamaError are available in scope

        # ── tweakables ─────────────────────────────────────────────────────
        TOKEN_WINDOW               = 2000

        TOKEN_REPEAT_LIMIT         = 400
        LINE_REPEAT_LIMIT          = 40

        # back-ref repeat detector (MUCH LOOSER)
        PATTERN_MIN_LEN            = 24      # NEW: ignore tiny patterns
        PATTERN_MAX_LEN            = 240     # ↑ allow longer repeated chunks
        PATTERN_REPEAT_THRESHOLD   = 20      # ↑ require many repeats

        # multi-token loop detector
        SEQ_MIN, SEQ_MAX           = 2, 50
        SEQ_REPEAT_LIMIT           = 18

        # fuzzy phrase repetition
        CHAR_WINDOW                = 600
        CHUNK_MIN, CHUNK_MAX       = 8, 48
        PHRASE_REPEAT_LIMIT        = 12
        FUZZY_SIM_THRESH           = 0.93

        # fuzzy token n-gram repetition
        NGRAM_TOKEN_WINDOW         = 120
        NGRAM_MIN, NGRAM_MAX       = 5, 20
        NGRAM_REPEAT_LIMIT         = 30
        NGRAM_FUZZY_SIM_THRESH     = 0.97

        # erosion / left-shift detector
        EROSION_CHAR_WINDOW        = 200
        EROSION_SLICE_LEN          = 14
        EROSION_STEPS_CHECK        = 10
        EROSION_SIM_THRESH         = 0.97
        EROSION_MIN_TRIGGERS       = 10

        MAX_ATTEMPTS               = 5
        SESSION_TIMEOUT_SEC        = 10 * 60
        GUARD_DELAY_SEC            = 5
        # ──────────────────────────────────────────────────────────────────

        # regex for verbatim repeated‐pattern (MUCH LESS SENSITIVE)
        # - requires chunk length >= PATTERN_MIN_LEN
        # - allows optional surrounding whitespace between repeats
        # - demands PATTERN_REPEAT_THRESHOLD occurrences
        pat_regex = re.compile(
            rf'(.{{{PATTERN_MIN_LEN},{PATTERN_MAX_LEN}}}?)(?:\s*\1\s*){{{PATTERN_REPEAT_THRESHOLD-1},}}',
            re.DOTALL
        )

        # ── inline images ─────────────────────────────────────────────────
        path_pat = re.compile(
            r"(?P<path>(?:~|\.{1,2}|[A-Za-z]:)?[^\s\"'<>|]+\."
            r"(?:jpg|jpeg|png|bmp|gif|webp))",
            re.IGNORECASE,
        )
        imgs_data = images or []
        if not imgs_data:
            for loc in {p for m in messages for p in path_pat.findall(m["content"])}:
                try:
                    if loc.lower().startswith(("http://", "https://")):
                        r = requests.get(loc, timeout=5); r.raise_for_status()
                        imgs_data.append(r.content)
                    else:
                        p = Path(loc).expanduser().resolve()
                        if p.is_file() and p.stat().st_size <= max_image_bytes:
                            imgs_data.append(p.read_bytes())
                except Exception:
                    pass
        if imgs_data:
            messages[-1]["images"] = imgs_data

        # ── guard helpers ────────────────────────────────────────────────
        def token_guard(tokens: deque[str]) -> bool:
            return (
                len(tokens) >= TOKEN_REPEAT_LIMIT
                and len(set(list(tokens)[-TOKEN_REPEAT_LIMIT:])) == 1
            )

        def line_guard(lines: deque[str]) -> bool:
            return (
                len(lines) >= LINE_REPEAT_LIMIT
                and len(set(list(lines)[-LINE_REPEAT_LIMIT:])) == 1
            )

        def multi_token_guard(tokens: deque[str]) -> bool:
            arr = list(tokens); n = len(arr)
            for L in range(SEQ_MIN, min(SEQ_MAX, max(SEQ_MIN, n // SEQ_REPEAT_LIMIT)) + 1):
                seq = arr[-L:]
                ok = True
                for r in range(2, SEQ_REPEAT_LIMIT + 1):
                    start = -r * L
                    end = start + L
                    if arr[start:end] != seq:
                        ok = False
                        break
                if ok:
                    return True
            return False

        def _norm_chars(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "")).strip().lower()

        def _sim(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()

        def fuzzy_phrase_guard(full_text: str) -> bool:
            tail = _norm_chars(full_text[-CHAR_WINDOW:])
            if len(tail) < CHUNK_MIN * PHRASE_REPEAT_LIMIT:
                return False
            candidates = []
            step = max(1, (CHUNK_MAX - CHUNK_MIN) // 4)
            for L in range(CHUNK_MIN, CHUNK_MAX + 1, step):
                if len(tail) >= L:
                    candidates.append(tail[-L:])
            for cand in candidates:
                if not cand.strip():
                    continue
                count = 1
                pos = len(tail) - len(cand)
                min_gap = int(len(cand) * 0.8)
                max_gap = int(len(cand) * 1.2)
                while pos - min_gap >= 0 and count < PHRASE_REPEAT_LIMIT + 3:
                    found = False
                    probes = (max(0, pos-max_gap), (max(0, pos-max_gap)+max(0, pos-min_gap))//2, max(0, pos-min_gap))
                    for st in probes:
                        seg = tail[st:st+len(cand)]
                        if seg and _sim(cand, seg) >= FUZZY_SIM_THRESH:
                            count += 1
                            pos = st
                            found = True
                            break
                    if not found:
                        break
                if count >= PHRASE_REPEAT_LIMIT:
                    return True
            return False

        def ngram_guard_fuzzy(tokens: deque[str]) -> bool:
            toks = list(tokens)[-NGRAM_TOKEN_WINDOW:]
            if len(toks) < NGRAM_MIN * NGRAM_REPEAT_LIMIT:
                return False
            joined = [t.lower() for t in toks]
            for n in range(NGRAM_MIN, NGRAM_MAX + 1):
                if len(joined) < n * NGRAM_REPEAT_LIMIT:
                    break
                pattern = " ".join(joined[-n:])
                cnt = 1
                for i in range(len(joined) - 2*n, -1, -n):
                    seg = " ".join(joined[i:i+n])
                    if _sim(pattern, seg) >= NGRAM_FUZZY_SIM_THRESH:
                        cnt += 1
                        if cnt >= NGRAM_REPEAT_LIMIT:
                            return True
                if cnt < NGRAM_REPEAT_LIMIT:
                    cnt2 = 1
                    for i in range(len(joined) - n - 1, -1, -1):
                        seg = " ".join(joined[i:i+n])
                        if _sim(pattern, seg) >= NGRAM_FUZZY_SIM_THRESH:
                            cnt2 += 1
                            if cnt2 >= NGRAM_REPEAT_LIMIT:
                                return True
            return False

        def erosion_guard(full_text: str) -> bool:
            tail = _norm_chars(full_text[-EROSION_CHAR_WINDOW:])
            if len(tail) < (EROSION_SLICE_LEN + EROSION_STEPS_CHECK):
                return False
            triggers = 0
            base_end = len(tail)
            for k in range(EROSION_STEPS_CHECK):
                end_a = base_end - k
                start_a = max(0, end_a - EROSION_SLICE_LEN)
                end_b = end_a - 1
                start_b = max(0, end_b - EROSION_SLICE_LEN)
                a = tail[start_a:end_a]
                b = tail[start_b:end_b]
                if _sim(a, b) >= EROSION_SIM_THRESH:
                    triggers += 1
                else:
                    if triggers > 0:
                        break
            return triggers >= EROSION_MIN_TRIGGERS

        session_start = time.time()

        def one_pass() -> tuple[str, bool]:
            buf_tokens = deque(maxlen=TOKEN_WINDOW)
            buf_lines  = deque(maxlen=LINE_REPEAT_LIMIT)
            chunks: list[str] = []
            inside_json = False
            first_output = None

            print(f"{tag} ", end="", flush=True)
            try:
                stream_iter = chat(model=model, messages=messages, stream=True)
            except _OllamaError as e:
                print(f"\n[Ollama crash before start] {e}")
                return "", True

            try:
                for part in stream_iter:
                    if getattr(self, "_abort_inference", False):
                        print("\n[Interrupted] aborting generation.")
                        return "".join(chunks), False
                    if time.time() - session_start > SESSION_TIMEOUT_SEC:
                        print("\n[Timeout guard] session expired → aborting pass.")
                        return "".join(chunks), True

                    chunk = part["message"]["content"]
                    st = chunk.strip()
                    if st.startswith("```json"):
                        inside_json = True; continue
                    if inside_json and st.startswith("```"):
                        inside_json = False; continue
                    if inside_json:
                        continue

                    if first_output is None and chunk:
                        first_output = time.time()

                    print(chunk, end="", flush=True)
                    chunks.append(chunk)
                    if on_token:
                        try: on_token(chunk)
                        except: pass

                    for tok in chunk.split():
                        buf_tokens.append(tok)
                    for ln in chunk.splitlines():
                        if ln.strip():
                            buf_lines.append(ln.strip())

                    if first_output and (time.time() - first_output) > GUARD_DELAY_SEC:
                        if token_guard(buf_tokens):
                            print("\n[Run-away guard] token repeat → aborting pass.")
                            return "".join(chunks), True
                        if line_guard(buf_lines):
                            print("\n[Run-away guard] line repeat → aborting pass.")
                            return "".join(chunks), True
                        if multi_token_guard(buf_tokens):
                            print("\n[Run-away guard] multi-token loop → aborting pass.")
                            return "".join(chunks), True

                        full_now = "".join(chunks)

                        # Only run the regex when the buffer is *long enough* to plausibly
                        # contain that many repeats of a minimum-length pattern.
                        if len(full_now) >= (PATTERN_MIN_LEN * PATTERN_REPEAT_THRESHOLD):
                            if pat_regex.search(full_now) or full_now.count("```json") > 1:
                                print("\n[Run-away guard] pattern repetition → aborting pass.")
                                return full_now, True

                        if fuzzy_phrase_guard(full_now):
                            print("\n[Run-away guard] fuzzy phrase repetition → aborting pass.")
                            return full_now, True
                        if ngram_guard_fuzzy(buf_tokens):
                            print("\n[Run-away guard] n-gram repetition → aborting pass.")
                            return "".join(chunks), True
                        if erosion_guard(full_now):
                            print("\n[Run-away guard] erosion (left-shift) loop → aborting pass.")
                            return full_now, True

            except _OllamaError as e:
                print(f"\n[Ollama crash] {e}")
                return "".join(chunks), True

            print()
            return "".join(chunks), False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            text, retry = one_pass()
            if not retry:
                return text
            print(f"[Guard/crash] restart ({attempt}/{MAX_ATTEMPTS}) …")
            time.sleep(0.1)

        if (
            model == getattr(self, "primary_model", "")
            and not _is_fallback
            and getattr(self, "secondary_model", None)
        ):
            print("[Fallback] primary model kept failing → switching to secondary.")
            return self._stream_and_capture(
                self.secondary_model,
                messages,
                tag=tag + "(fallback)",
                max_image_bytes=max_image_bytes,
                images=images,
                on_token=on_token,
                _is_fallback=True,
            )

        print(f"[Run-away guard] giving up after {MAX_ATTEMPTS} attempts.")
        return ""

    # ─── Task scheduling & countdown (repo-native, no extra files) ───────────────

    def _task__reschedule_from_rrule(self, task_ctx) -> bool:
        """
        If task_ctx has an RFC 5545 RRULE, compute the next occurrence and
        update the task in-place: due_at, status=scheduled, remaining_seconds, summary.
        Returns True if it rescheduled, False otherwise.
        """
        try:
            rrule_str = (task_ctx.metadata or {}).get("rrule")
            if not rrule_str:
                return False

            from dateutil.rrule import rrulestr  # pip install python-dateutil
            # Base the rule on the last due_at; fall back to now if missing
            last_due = (task_ctx.metadata or {}).get("due_at") or self._task__fmt_due(self._task__now_utc())
            dtstart = datetime.fromisoformat(str(last_due).replace("Z", ""))

            rule = rrulestr(rrule_str, dtstart=dtstart)
            nxt = rule.after(self._task__now_utc(), inc=False)
            if not nxt:
                return False

            task_ctx.metadata["due_at"] = self._task__fmt_due(nxt)
            task_ctx.metadata["status"] = "scheduled"
            task_ctx.metadata["remaining_seconds"] = self._task__sec_until(task_ctx.metadata["due_at"])
            title = task_ctx.metadata.get("title", "Scheduled task")
            task_ctx.summary = f"[scheduled] {title}  (T-{int(task_ctx.metadata['remaining_seconds'])}s)"

            task_ctx.touch(); self.repo.save(task_ctx)
            self.memman.register_relationships(task_ctx, embed_text)
            return True
        except Exception:
            return False


    # ─── Task scheduling & countdown (repo-native, no extra files) ───────────────
    def _task__fmt_due(self, dt: datetime) -> str:
        """Repo-friendly UTC timestamp."""
        from context import _fmt_ts
        return _fmt_ts(dt.replace(tzinfo=None))

    def _task__now_utc(self) -> datetime:
        from context import default_clock
        return default_clock()

    def _task__sec_until(self, due_at_s: str) -> float:
        """Seconds until due_at_s (repo format %Y%m%dT%H%M%SZ)."""
        try:
            # repo parser expects trailing Z; be lenient if missing
            ts = due_at_s if str(due_at_s).endswith("Z") else f"{due_at_s}Z"
            return max((datetime.strptime(ts, "%Y%m%dT%H%M%SZ") - self._task__now_utc()).total_seconds(), 0.0)
        except Exception:
            return 0.0

    def _task__to_ctx(self, *, title: str, due_at: str, payload_text: str,
                    conversation_id: str | None, user_id: str | None,
                    status: str = "scheduled", **extra) -> ContextObject:
        """Create a ContextObject (artifact/task)."""
        from context import ContextObject
        ctx = ContextObject(
            domain="artifact",
            component="task",
            semantic_label="scheduled_task",
        )
        ctx.summary = f"[{status}] {title}"
        ctx.tags = ["task", "scheduled"]
        ctx.metadata.update({
            "model": "scheduled_task/v1",
            "task_id": ctx.context_id,
            "title": title,
            "due_at": due_at,
            "payload_text": payload_text,
            "status": status,
            "created_at": self._task__fmt_due(self._task__now_utc()),
            "remaining_seconds": self._task__sec_until(due_at),
            "conversation_id": conversation_id,
            "user_id": user_id,
            **(extra or {})
        })
        # show a T-… hint in summary for quick scanning
        rem = ctx.metadata.get("remaining_seconds")
        if rem is not None:
            ctx.summary = f"{ctx.summary}  (T-{int(rem)}s)"
        return ctx



    def _task_schedule(self, *, title: str, due_in_seconds: int | None = None,
                    due_at_utc: str | None = None, payload_text: str,
                    conversation_id: str | None, user_id: str | None,
                    rrule: str | None = None, allow_update: bool = False) -> ContextObject:
        """
        Upsert a task. If allow_update and a very similar title exists in-scope,
        update it instead of creating a new one (fills missing due time, etc.).

        Also: after creating/updating a task, we 'kick' the background poller so it
        recalculates sleep and can start near the exact due time.
        """
        # --- Resolve/normalize due_at in repo format ---
        if not due_at_utc:
            if due_in_seconds is None:
                raise ValueError("Provide due_in_seconds or due_at_utc")
            try:
                # clamp negatives to 'now'
                secs = max(0, int(due_in_seconds))
            except Exception:
                secs = 0
            due_at = self._task__now_utc() + timedelta(seconds=secs)
            due_at_utc = self._task__fmt_due(due_at)
        else:
            # normalize to repo format (ensure trailing Z removed since _fmt_ts adds it)
            try:
                iso = str(due_at_utc).replace("Z", "")
                dt = datetime.fromisoformat(iso) if "T" in iso else datetime.fromisoformat(iso + "T00:00:00")
                due_at_utc = self._task__fmt_due(dt)
            except Exception:
                # fallback: keep as-is; countdown may be zero
                pass

        # --- Dedup/update in-scope (conversation_id + user_id) ---
        existing = self._task_list(conversation_id=conversation_id, user_id=user_id)
        keeper = None
        if allow_update:
            # simple fuzzy match on title
            tnorm = (title or "").strip().lower()
            for t in existing:
                en = (t.metadata.get("title", "") or t.summary or "").strip().lower()
                if en and (en == tnorm or tnorm in en or en in tnorm):
                    keeper = t
                    break

        if keeper:
            meta = keeper.metadata or {}
            # Fill missing fields only (preserve original semantics)
            if not meta.get("due_at") and due_at_utc:
                meta["due_at"] = due_at_utc
            if not meta.get("payload_text"):
                meta["payload_text"] = payload_text
            # Add rrule if newly supplied
            if rrule and not meta.get("rrule"):
                meta["rrule"] = rrule
            # Ensure scope metadata is present
            if conversation_id and not meta.get("conversation_id"):
                meta["conversation_id"] = conversation_id
            if user_id and not meta.get("user_id"):
                meta["user_id"] = user_id
            # Standard scheduling fields
            meta["title"] = title or meta.get("title", "Scheduled task")
            meta["status"] = "scheduled"
            if meta.get("due_at"):
                meta["remaining_seconds"] = self._task__sec_until(meta["due_at"])
            else:
                meta["remaining_seconds"] = 0
            keeper.summary = f"[{meta['status']}] {meta['title']}  (T-{int(meta['remaining_seconds'])}s)"
            keeper.metadata = meta
            keeper.touch(); self.repo.save(keeper)
            self.memman.register_relationships(keeper, embed_text)

            # Wake poller to re-schedule sleep immediately
            try:
                self._task__kick()
            except Exception:
                pass
            return keeper

        # --- Create a brand-new task ---
        ctx = self._task__to_ctx(
            title=title,
            due_at=due_at_utc,
            payload_text=payload_text,
            conversation_id=conversation_id,
            user_id=user_id,
            status="scheduled",
            rrule=rrule,
        )
        ctx.touch(); self.repo.save(ctx)
        self.memman.register_relationships(ctx, embed_text)

        # Wake poller so it aligns sleep with this new due time
        try:
            self._task__kick()
        except Exception:
            pass

        return ctx


    def _task_list(self, *, conversation_id: str | None = None, user_id: str | None = None) -> list[ContextObject]:
        rows = self.repo.query(lambda c:
            c.domain=="artifact" and c.component=="task" and
            ((conversation_id is None) or (c.metadata or {}).get("conversation_id")==conversation_id) and
            ((user_id is None) or (c.metadata or {}).get("user_id")==user_id)
        )
        rows.sort(key=lambda c: (c.metadata or {}).get("due_at", c.timestamp))
        return rows

    def _task_inject_countdown_context(self, state: dict) -> list[ContextObject]:
        """
        Update per-task remaining_seconds and emit a 'task_countdown' stage
        visible to planner when something is due within 1 hour.
        """
        from context import ContextObject
        conv = state.get("conversation_id")
        uid  = state.get("user_id")
        soon: list[ContextObject] = []
        for t in self._task_list(conversation_id=conv, user_id=uid):
            meta = t.metadata or {}
            if meta.get("status") != "scheduled":
                continue
            meta["remaining_seconds"] = self._task__sec_until(meta.get("due_at",""))
            t.metadata = meta
            t.summary  = f"[{meta.get('status')}] {meta.get('title','task')}  (T-{int(meta['remaining_seconds'])}s)"
            t.touch(); self.repo.save(t)
            if meta["remaining_seconds"] <= 3600 and meta["remaining_seconds"] > 0:
                soon.append(t)

        if not soon:
            return []

        lines = [
            f"• { (s.metadata.get('title') or s.summary) } — T-{int(s.metadata.get('remaining_seconds',0))}s (due {s.metadata.get('due_at')})"
            for s in sorted(soon, key=lambda x: x.metadata.get("remaining_seconds", 0))
        ]
        msg = "Upcoming tasks:\n" + "\n".join(lines)
        sc = ContextObject.make_stage(
            "task_countdown",
            input_refs=[state.get("user_ctx", None) and state["user_ctx"].context_id or ""],
            output={"text": msg, "items": [s.metadata for s in soon]},
        )
        sc.summary = msg[:250]
        sc.metadata.update({"conversation_id": conv, "user_id": uid, "count": len(soon)})
        sc.touch(); self.repo.save(sc)
        self.memman.register_relationships(sc, embed_text)

        state.setdefault("merged", []).append(sc)
        state.setdefault("merged_ids", []).append(sc.context_id)
        return [sc]

    def _task__due_now(self) -> list[ContextObject]:
        """Return all scheduled tasks whose due time has arrived (any scope)."""
        out = []
        for t in self._task_list():
            meta = t.metadata or {}
            if meta.get("status") != "scheduled":
                continue
            if self._task__sec_until(meta.get("due_at","")) <= 0:
                out.append(t)
        return out

    def _launch_scheduled(self, task_ctx: ContextObject):
        """
        Launch a new repo-assembler session seeded with the task's payload.

        Transitions:
        scheduled -> running (set by _task_tick_and_launch) -> completed
        OR, if metadata.rrule is present -> rescheduled to next due_at
        On exception -> failed

        Side effects:
        - Updates last_run_at, last_completed_at / last_failed_at
        - Increments runs_count / failures_count
        - Calls _task__reschedule_from_rrule on success for recurring tasks
        - Calls _task__kick() after (re)scheduling to self-start soon-due items
        """
        async def _go():
            # Re-fetch the latest copy to avoid stale metadata
            try:
                t = self.repo.get(task_ctx.context_id)
            except Exception:
                t = task_ctx

            meta = t.metadata or {}

            # Idempotency guard: only proceed if task is currently 'running'
            if meta.get("status") != "running":
                return

            # Mark this attempt
            meta["last_run_at"] = self._task__fmt_due(self._task__now_utc())
            meta["runs_count"] = int(meta.get("runs_count", 0)) + 1
            t.metadata = meta
            t.touch(); self.repo.save(t)

            # Prepare payload
            payload = meta.get("payload_text") or t.summary or ""

            # Open episode (best-effort)
            try:
                self.memman.start_episode(
                    f"Task: {meta.get('title','task')}",
                    {
                        "task_id": meta.get("task_id"),
                        "due_at": meta.get("due_at"),
                        "conversation_id": meta.get("conversation_id"),
                        "user_id": meta.get("user_id"),
                    }
                )
            except Exception:
                pass

            try:
                # Run the task workload
                await self.run_with_meta_context(
                    payload, skip_quick_phases=True, direct_plan=True, images=None
                )

                # Success path
                try:
                    fresh = self.repo.get(t.context_id)
                except Exception:
                    fresh = t
                fmeta = fresh.metadata or {}
                fmeta["last_completed_at"] = self._task__fmt_due(self._task__now_utc())

                # If recurring, try to roll forward; else mark completed
                if fmeta.get("rrule"):
                    ok = False
                    try:
                        ok = self._task__reschedule_from_rrule(fresh)
                    except Exception:
                        ok = False
                    if not ok:
                        fmeta["status"] = "completed"
                        fresh.summary = f"[completed] {fmeta.get('title','task')}"
                        fresh.metadata = fmeta
                        fresh.touch(); self.repo.save(fresh)
                    # Nudge the poller so newly-rescheduled near-due tasks launch promptly
                    try:
                        self._task__kick()
                    except Exception:
                        pass
                else:
                    fmeta["status"] = "completed"
                    fresh.summary = f"[completed] {fmeta.get('title','task')}"
                    fresh.metadata = fmeta
                    fresh.touch(); self.repo.save(fresh)

            except Exception:
                # Failure path
                try:
                    fresh = self.repo.get(t.context_id)
                except Exception:
                    fresh = t
                fmeta = fresh.metadata or {}
                fmeta["status"] = "failed"
                fmeta["last_failed_at"] = self._task__fmt_due(self._task__now_utc())
                fmeta["failures_count"] = int(fmeta.get("failures_count", 0)) + 1
                fresh.metadata = fmeta
                fresh.summary = f"[failed] {fmeta.get('title','task')}"
                fresh.touch(); self.repo.save(fresh)
            finally:
                try:
                    self.memman.end_episode()
                except Exception:
                    pass

        # fire-and-forget
        try:
            import asyncio
            asyncio.create_task(_go())
        except RuntimeError:
            # no running loop → run in a thread
            import threading, asyncio as _asyncio
            threading.Thread(target=lambda: _asyncio.run(_go()), daemon=True).start()


    def _task_tick_and_launch(self) -> int:
        """One poll tick: refresh countdowns and launch all due tasks. Returns count launched."""
        launched = 0
        # refresh countdowns globally
        for t in self._task_list():
            meta = t.metadata or {}
            if meta.get("status") != "scheduled":
                continue
            meta["remaining_seconds"] = self._task__sec_until(meta.get("due_at",""))
            t.metadata = meta
            t.summary  = f"[{meta.get('status')}] {meta.get('title','task')}  (T-{int(meta['remaining_seconds'])}s)"
            t.touch(); self.repo.save(t)
        # launch due
        for t in self._task__due_now():
            try:
                t.metadata["status"] = "running"
                t.metadata["last_run_at"] = self._task__fmt_due(self._task__now_utc())
                t.summary = f"[running] {t.metadata.get('title','task')}"
                t.touch(); self.repo.save(t)
                self._launch_scheduled(t)
                launched += 1
            except Exception:
                try:
                    t.metadata["status"] = "failed"
                    t.touch(); self.repo.save(t)
                except Exception:
                    pass
        return launched

    def _task_start_background(self, poll_seconds: float = 2.0):
        """Start background poller that launches due tasks with adaptive sleeping."""
        # Singleton guard across processes
        if not self._task__acquire_singleton_lock():
            # another process owns the poller
            return

        if self._task_bg_thread and self._task_bg_thread.is_alive():
            return

        self._task_poll_event.clear()
        self._task_wake_event.clear()

        # Tunables: how early to wake and max idle when nothing is coming soon
        start_early_s   = float(self.cfg.get("task_start_early_s", 0.75))   # start up to 0.75s early
        idle_cap_s      = float(self.cfg.get("task_idle_cap_seconds", 60.0))# max sleep if no tasks due soon
        hard_min_sleep  = 0.10                                              # don't busy spin

        def _run():
            # 1) Immediate catch-up on boot
            try:
                self._task_tick_and_launch()
            except Exception:
                pass

            # 2) Loop with adaptive sleep or wake kicks
            while not self._task_poll_event.is_set():
                try:
                    # Launch any tasks that just became due
                    self._task_tick_and_launch()
                except Exception:
                    pass

                # Compute how long to sleep until the next due time
                try:
                    soonest = self._task__soonest_due_seconds()
                except Exception:
                    soonest = float("inf")

                if soonest == float("inf"):
                    timeout = idle_cap_s
                else:
                    # Wake a bit early so launch jitter is <~1s
                    timeout = max(hard_min_sleep, soonest - start_early_s)
                    # Don't sleep *too* long even if the next due is far away
                    timeout = min(timeout, idle_cap_s)

                # Wait for either a kick (new task scheduled) or the timeout
                kicked = self._task_wake_event.wait(timeout)
                if kicked:
                    # Clear and loop immediately to recompute timing
                    try:
                        self._task_wake_event.clear()
                    except Exception:
                        pass

            # loop ending → releasing singleton lock
            try:
                self._task__release_singleton_lock()
            except Exception:
                pass

        self._task_bg_thread = threading.Thread(target=_run, name="TaskPoller", daemon=True)
        self._task_bg_thread.start()



    def _task_stop_background(self):
        try:
            if self._task_bg_thread:
                self._task_poll_event.set()
                # also wake immediately so the thread can exit now, not after timeout
                try: self._task_wake_event.set()
                except Exception: pass
                self._task_bg_thread.join(timeout=1.5)
                self._task_bg_thread = None
        finally:
            try:
                self._task__release_singleton_lock()
            except Exception:
                pass


    def _stage_task_detect_and_schedule(self, *, user_text: str, state: dict, allow_update: bool = False) -> ContextObject | None:
        """
        Lightweight detector that decides whether the user's message implies a
        future task (e.g., “remind me at 5pm …”, “run the sync tonight”).
        If so, schedules it and returns the created/updated task ContextObject.
        """
        # 1) Ask the small model to produce a strict JSON object
        sys = (
            "You classify scheduling intents. Return ONLY JSON with keys:\n"
            "{\n"
            '  "should_schedule": true|false,\n'
            '  "title": "<short title or empty>",\n'
            '  "payload_text": "<what to run or empty>",\n'
            '  "due_in_seconds": <integer or null>,\n'
            '  "due_at_utc": "<YYYY-MM-DDTHH:MM:SSZ or null>"\n'
            "}\n"
            "Rules:\n"
            "• If the user clearly asked for a future action/reminder/run, should_schedule=true.\n"
            "• Prefer due_at_utc (UTC) if a specific date/time is given, else due_in_seconds.\n"
            "• payload_text should be what the agent should execute at that time.\n"
            "• No prose. No markdown. Return valid JSON only."
        )
        # Include clarifier notes if available
        hint = ""
        try:
            clar = state.get("clar_ctx")
            if clar and clar.metadata.get("notes"):
                hint = f"\n\n[Intent notes]\n{clar.metadata.get('notes')}"
        except Exception:
            pass

        raw = self._stream_and_capture(
            self.decision_model,
            [{"role":"system","content":sys},
            {"role":"user","content":(user_text or "").strip() + hint}],
            tag="[TaskDetect]"
        ).strip()

        # 2) Parse JSON; fallback to no-op on failure
        try:
            data = json.loads(raw)
        except Exception:
            return None
        if not isinstance(data, dict) or not data.get("should_schedule"):
            return None

        title   = (data.get("title") or "").strip() or "Scheduled task"
        payload = (data.get("payload_text") or user_text or "").strip()
        due_in  = data.get("due_in_seconds")
        due_at  = data.get("due_at_utc")

        # 3) Create/update the task in-scope and surface a small confirmation stage
        tctx = self._task_schedule(
            title=title,
            due_in_seconds=int(due_in) if isinstance(due_in, (int, float)) else None,
            due_at_utc=due_at if isinstance(due_at, str) and due_at else None,
            payload_text=payload,
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            allow_update=allow_update
        )

        # short confirmation (stage)
        try:
            from context import ContextObject
            conf = ContextObject.make_stage(
                "task_scheduled",
                [state.get("user_ctx", None) and state["user_ctx"].context_id or ""],
                {"title": title, "due_at": tctx.metadata.get("due_at"), "payload": payload}
            )
            conf.summary = f"Scheduled: {title} @ {tctx.metadata.get('due_at')}"
            conf.metadata.update({"conversation_id": state.get("conversation_id"), "user_id": state.get("user_id")})
            conf.touch(); self.repo.save(conf)
            self.memman.register_relationships(conf, embed_text)
            state.setdefault("merged", []).append(conf)
            state.setdefault("merged_ids", []).append(conf.context_id)
        except Exception:
            pass

        return tctx


    from self_state import SelfState

    def _load_self_state(self) -> SelfState:
        rows = sorted(self.repo.query(lambda c: c.component=="self_state"), key=lambda c: c.timestamp, reverse=True)
        if rows:
            try:
                return SelfState(rows[0].metadata.get("data", {}))
            except Exception:
                pass
        return SelfState()

    def _save_self_state(self, ss: SelfState):
        from context import ContextObject
        ctx = ContextObject.make_stage("self_state", [], {"data": ss.data})
        ctx.stage_id = "self_state"; ctx.summary = "self_state update"
        ctx.touch(); self.repo.save(ctx)

    # convenience
    def param(self, key, default):
        ss = getattr(self, "_self_state", None) or self._load_self_state()
        self._self_state = ss
        return ss.param(key, default)



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
        
    def _generate_system_prompt(
        self,
        purpose: str,
        schema: dict | str | None = None,
        variables: dict | None = None,
        *,
        label: str | None = None,
        save: bool = True,
        refine_with_llm: bool = False,
    ) -> str:
        """
        Build a strong, deterministic system prompt from inputs.
        - purpose:   short paragraph describing the agent's job
        - schema:    JSON schema (dict or JSON string) describing EXPECTED OUTPUT fields
        - variables: freeform knobs/context (e.g. audience="exec", tone="formal")
        - label:     semantic label for saving to repo
        - save:      persist prompt to repo as a prompt artifact
        - refine_with_llm: optional one-pass polish using primary_model
        """
        import json, hashlib, textwrap

        variables = variables or {}

        # Parse schema if needed
        schema_obj: dict | None = None
        if isinstance(schema, str):
            try:
                schema_obj = json.loads(schema)
            except Exception:
                schema_obj = None
        elif isinstance(schema, dict):
            schema_obj = schema

        def _schema_bullets(s: dict) -> list[str]:
            bullets = []
            if not isinstance(s, dict):
                return bullets
            props = (s.get("properties") or {}) if isinstance(s.get("properties"), dict) else {}
            req   = set(s.get("required") or [])
            for name, meta in props.items():
                t = meta.get("type", "any")
                desc = (meta.get("description") or "").strip()
                oneof = meta.get("enum")
                rng = []
                if "minimum" in meta: rng.append(f"≥ {meta['minimum']}")
                if "maximum" in meta: rng.append(f"≤ {meta['maximum']}")
                extras = []
                if oneof: extras.append(f"one of {oneof}")
                if rng:   extras.append(", ".join(rng))
                tail = f" ({'; '.join(extras)})" if extras else ""
                need = "REQUIRED" if name in req else "optional"
                bullets.append(f"- `{name}` ({t}, {need}){tail}: {desc}")
            return bullets

        # Render variables
        var_lines = [f"- {k}: {v}" for k, v in variables.items() if v is not None]

        # Render schema bullets
        sch_lines = _schema_bullets(schema_obj) if schema_obj else []

        # Build deterministic base system message
        parts = [
            "ROLE: You are a precise, reliable agent. Follow instructions exactly. Do not invent facts.",
            "",
            "OBJECTIVE:",
            textwrap.dedent(purpose).strip(),
            "",
            "OUTPUT CONTRACT:",
            ("Adhere strictly to the following field contract:" if sch_lines else "No fixed field contract was provided."),
        ]
        if sch_lines:
            parts.extend(sch_lines)
            parts += [
                "",
                "CONSTRAINTS:",
                "- If a field cannot be derived, set it to null (do not guess).",
                "- Preserve types exactly as specified.",
                "- Do not include extra fields.",
            ]
        else:
            parts += [
                "",
                "CONSTRAINTS:",
                "- Keep outputs unambiguous, minimal, and directly actionable.",
            ]

        if var_lines:
            parts += ["", "CONTEXT / KNOBS:"] + var_lines

        parts += [
            "",
            "STYLE:",
            "- Be concise; avoid hedging or filler.",
            "- Prefer bullet points for multi-item guidance.",
            "- Use active voice.",
            "",
            "FAILURE MODE:",
            "- If the request is underspecified, ask 1–2 crisp clarifying questions.",
            "- If a step is impossible, state exactly why and suggest the closest safe alternative.",
        ]

        system_text = "\n".join(parts).strip()

        # Optional one-pass polish (keeps it short)
        if refine_with_llm:
            try:
                prompt = (
                    "Tighten the following system prompt without changing meaning. "
                    "Keep all constraints and field contracts. Return only the revised text.\n\n"
                    + system_text
                )
                system_text = self._stream_and_capture(
                    self.primary_model,
                    [{"role": "system", "content": prompt}],
                    tag="[PromptPolish]"
                ).strip() or system_text
            except Exception:
                pass

        # Save to repo (idempotent-ish)
        if save:
            from context import ContextObject
            # Derive a stable label if not supplied
            h = hashlib.sha1(system_text.encode("utf-8")).hexdigest()[:10]
            label = label or f"generated_system_prompt_{h}"
            ctx = ContextObject.make_prompt(label=label, prompt_text=system_text, tags=["artifact","prompt","generated"])
            ctx.touch(); self.repo.save(ctx)
            self.memman.register_relationships(ctx, self.embed_text)

        return system_text



    def _should_ask_confirmation(self, state: Dict[str, Any]) -> bool:
        """
        Decide whether to show the plan to the user before running.
        Heuristics → LLM fallback. Records a 'confirm_decision' stage node.
        """
        import re, json, time
        from context import ContextObject

        calls: list[str] = list(state.get("fixed_calls", []) or [])
        ctx_summ = " | ".join(
            f"{(c.stage_id or c.semantic_label)}: {(c.summary or '')[:80].replace(chr(10),' ')}"
            for c in [state.get("user_ctx"), state.get("clar_ctx"), state.get("know_ctx")] if c
        )

        # Heuristic signals
        risky_keywords = re.compile(
            r"\b(delete|remove|truncate|drop|post|publish|tweet|send|email|sms|text|call|"
            r"buy|purchase|charge|wire|transfer|subscribe|deploy|production|prod)\b",
            re.I
        )
        money_re = re.compile(r"\b(\$|usd|eur|gbp)\s*\d|\bamount\s*=\s*\d", re.I)
        pii_re   = re.compile(r"\b(ssn|passport|credit\s*card|cvc|cvv)\b", re.I)

        risk_score = 0
        reasons: list[str] = []

        # 1) Lots of tool calls → conservative
        if len(calls) >= 4:
            risk_score += 1; reasons.append("many_calls")

        # 2) Keyword risks
        joined = " | ".join(calls)
        if risky_keywords.search(joined):
            risk_score += 2; reasons.append("risky_verbs")
        if money_re.search(joined):
            risk_score += 2; reasons.append("money_like")
        if pii_re.search(joined):
            risk_score += 2; reasons.append("pii_like")

        # 3) Tool schema flags (x_requires_confirmation)
        try:
            from tools import TOOL_SCHEMAS
            for call in calls:
                name = call.split("(", 1)[0].strip()
                sch = TOOL_SCHEMAS.get(name) or {}
                if sch.get("x_requires_confirmation") or sch.get("x_side_effects"):
                    risk_score += 3; reasons.append(f"schema_flag:{name}")
        except Exception:
            pass

        # 4) Suspicious HTTP verbs in args
        if re.search(r"method\s*=\s*['\"]?(POST|DELETE|PUT)['\"]?", joined, re.I):
            risk_score += 2; reasons.append("http_mutation")

        # 5) Scheduling / automation (creating future triggers)
        if re.search(r"(schedule|remind|cron|rrule|due_at)", joined, re.I):
            risk_score += 1; reasons.append("scheduling")

        # 6) Domain check (rough) — unknown external hostnames in calls
        if re.search(r"https?://[^/\s]+", joined, re.I):
            risk_score += 1; reasons.append("external_domains")

        # Decide from heuristics
        heuristic_yes = risk_score >= 3

        # LLM fallback only if undecided
        llm_yes = False
        if not heuristic_yes:
            try:
                prompt = {"plan": calls, "context_summary": ctx_summ}
                system = (
                    "Decide if explicit user confirmation is required before running this plan.\n"
                    "Say only 'yes' or 'no'.\n"
                    "Heuristics: ask for confirmation if actions are destructive, irreversible, spend money, "
                    "send messages/emails, touch production, or create future automations."
                )
                out = self._stream_and_capture(
                    self.primary_model,
                    [{"role":"system","content":system},{"role":"user","content":json.dumps(prompt)}],
                    tag="[ConfirmCheck]"
                ).strip().lower()
                llm_yes = bool(re.search(r"\byes\b", out))
                if not re.search(r"\b(?:yes|no)\b", out):
                    # default conservative if model drifted
                    llm_yes = True
            except Exception:
                # fail safe → ask
                llm_yes = True

        need = bool(heuristic_yes or llm_yes)

        # Record decision
        try:
            node = ContextObject.make_stage(
                "confirm_decision",
                [state.get("user_ctx", None) and state["user_ctx"].context_id or ""],
                {"need_confirmation": need, "risk_score": risk_score, "reasons": reasons, "calls": calls}
            )
            node.summary = f"confirm={need} (risk={risk_score}; reasons={','.join(reasons) or '-'})"
            node.touch(); self.repo.save(node)
            self.memman.register_relationships(node, self.embed_text)
        except Exception:
            pass

        try:
            state["asked_confirmation"] = bool(need)
        except Exception:
            pass

        return need


    async def _handle_confirmation_async(self, reply: str) -> str:
        """
        Handle user's yes/no after a confirmation prompt.
        Continues or replans by calling run_with_meta_context(...).
        """
        ans = (reply or "").strip().lower()
        yes = bool(re.search(r"\b(yes|y|sure|go ahead|ok|okay|proceed|confirm)\b", ans))

        st = getattr(self, "_pending_state", None)
        if not st or "user_text" not in st:
            return "No pending plan to confirm."

        # clear flags safely
        for attr in ("_awaiting_confirmation","_pending_state","_pending_queue"):
            if hasattr(self, attr):
                try: delattr(self, attr)
                except Exception: setattr(self, attr, None)

        if yes:
            return await self.run_with_meta_context(st["user_text"])
        else:
            # simple replan: nudge clarifier to note refusal
            rej_text = st["user_text"] + "\n\n(User did NOT approve previous plan. Replan without side-effects.)"
            return await self.run_with_meta_context(rej_text)

    def _handle_confirmation(self, reply: str) -> str:
        """
        Sync wrapper for environments without an event loop.
        Spawns/awaits as needed.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # schedule task and return a short ack
            loop.create_task(self._handle_confirmation_async(reply))
            return "Got it — continuing in the background."
        else:
            return asyncio.run(self._handle_confirmation_async(reply))


    def register_chat(self, chat_id: int, user_text: str):
        """
        Remember the Telegram chat and bind/refresh a conversation id.
        """
        import time, uuid
        self._chat_contexts.add(chat_id)
        if not hasattr(self, "_chat_index"):
            self._chat_index = {}  # chat_id -> {conversation_id, last_seen, last_text}

        rec = self._chat_index.get(chat_id) or {}
        rec["conversation_id"] = rec.get("conversation_id") or getattr(self, "_active_conversation_id", uuid.uuid4().hex)
        rec["last_seen"] = int(time.time())
        rec["last_text"] = (user_text or "")[:200]
        self._chat_index[chat_id] = rec

        # Make this the active conversation for the next turn
        self._active_conversation_id = rec["conversation_id"]

    def _maybe_appiphany(self, chat_id: int) -> bool:
        """
        If there's a high-value insight, ping the user in text (and voice if supported).
        Rate-limited to once per 2 minutes.
        """
        import time
        if not getattr(self, "curiosity_used", []):  # needs at least one curiosity probe
            return False
        if getattr(self, "_last_errors", False):
            return False

        now = time.time()
        last = getattr(self, "_last_appiphany_at", 0.0)
        if now - last < 120:  # 2 min cooldown
            return False

        bot = getattr(self, "_telegram_bot", None)
        if not bot:
            return False

        text = "💡 I just found an insight that might help—want to hear it?"
        try:
            bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            return False

        # Voice is optional; guard hard
        try:
            if getattr(self, "tts", None):
                self.tts.enqueue(text)
                ogg = None
                try:
                    ogg = self.tts.wait_for_latest_ogg(timeout=1.0)
                except Exception:
                    ogg = None
                if ogg:
                    with open(ogg, "rb") as vf:
                        bot.send_voice(chat_id=chat_id, voice=vf)
        except Exception:
            pass

        self._last_appiphany_at = now
        return True


    def dump_architecture(self, save: bool = True) -> str:
        """
        Return a JSON string with the current architecture snapshot and optionally save it.
        """
        import json, inspect, textwrap
        arch = {
            "stages":               self.STAGES,
            "optional_stages":      self._optional_stages,
            "curiosity_templates":  [t.semantic_label for t in getattr(self, "curiosity_templates", [])],
            "rl_weights":           {"Q": self.rl.Q, "R_bar": self.rl.R_bar},
            "curiosity_weights":    {"Q": self.curiosity_rl.Q, "R_bar": self.curiosity_rl.R_bar},
            "system_prompts":       getattr(self, "system_prompts", {}),
            "stage_methods":        {}
        }

        for s in self.STAGES + ["curiosity_probe", "system_prompt_refine", "narrative_mull"]:
            fn = getattr(self, f"_stage_{s}", None)
            if fn:
                arch["stage_methods"][s] = {
                    "signature": str(inspect.signature(fn)),
                    "doc": (fn.__doc__ or "").strip(),
                }

        text = json.dumps(arch, indent=2)

        if save:
            from context import ContextObject
            ctx = ContextObject.make_stage("architecture_dump", [], {"json": arch})
            ctx.summary = "architecture_dump"
            ctx.touch(); self.repo.save(ctx)
            self.memman.register_relationships(ctx, self.embed_text)

        print(text)
        return text


    def _stage_curiosity_probe(self, state: Dict[str,Any]) -> List[str]:
        """
        Identify gaps in clarified intent, auto-mull or explicit follow-ups via RL,
        ask the LLM for answers, record Q&A as ContextObjects, return answers.
        """
        from context import ContextObject
        import re

        probes: List[str] = []
        clar = state.get("clar_ctx")
        if clar is None:
            return probes

        max_probes = int(self.cfg.get("curiosity_max_probes", 2))

        # 1) Cascade activation feature (guarded)
        activation_map: Dict[str, float] = {}
        recall_ids = state.get("recent_ids", [])
        if recall_ids:
            try:
                activation_map = self.memman.spread_activation(
                    seed_ids=recall_ids, hops=2, decay=0.6, assoc_weight=1.0, recency_weight=0.5
                ) or {}
            except Exception:
                activation_map = {}
        top_vals = sorted(activation_map.values(), reverse=True)[: len(recall_ids) or 1]
        rf = (sum(top_vals) / len(top_vals)) if top_vals else 0.0

        # 2) Detect explicit gaps
        gaps: List[Tuple[str,str]] = []
        if not (clar.metadata or {}).get("notes"):
            gaps.append(("missing_notes", (clar.summary or "")[:80]))
        plan_out = state.get("plan_output", "") or ""
        if ("date(" in plan_out) and not any((kw or "").lower().startswith("date") for kw in (clar.metadata or {}).get("keywords", [])):
            gaps.append(("missing_date", "plan mentions a date"))

        # 3) If no explicit gaps, auto-mull
        if not gaps:
            gaps.append(("auto_mull", "self-reflection"))

        used = 0
        for gap_name, snippet in gaps:
            if used >= max_probes:
                break

            # choose the best template by RL probability
            candidates = [t for t in getattr(self, "curiosity_templates", []) if gap_name in (t.semantic_label or "")]
            prompt = None
            picked = None
            if candidates:
                picked = max(candidates, key=lambda t: self.curiosity_rl.probability(t.semantic_label, rf))
                tmpl = (picked.metadata.get("policy") or picked.summary or "").strip()
                if tmpl:
                    prompt = tmpl.format(snippet=snippet)

            # fallback generic prompt
            if not prompt:
                prompt = f"Clarify this gap ({gap_name}): «{snippet}». Provide one concise answer."

            # 4a) record question node
            q_ctx = ContextObject.make_stage(f"curiosity_question_{gap_name}", [clar.context_id], {"question": prompt})
            q_ctx.component = "curiosity"; q_ctx.semantic_label = "question"; q_ctx.tags.append("curiosity")
            score = activation_map.get(getattr(picked, "context_id", ""), 0.0) if picked else 0.0
            q_ctx.retrieval_score = score
            q_ctx.retrieval_metadata = {"template": getattr(picked, "semantic_label", "fallback")}
            self.memman.reinforce(clar.context_id, [q_ctx.context_id])
            q_ctx.touch(); self.repo.save(q_ctx)
            self.memman.register_relationships(q_ctx, self.embed_text)

            # 4b) ask model
            try:
                reply = self._stream_and_capture(
                    self.primary_model,
                    [{"role":"system","content":"Answer the follow-up question succinctly and concretely."},
                    {"role":"user","content":prompt}],
                    tag=f"[CuriosityAnswer_{gap_name}]"
                ).strip()
            except Exception:
                reply = ""

            # 4c) record answer
            a_ctx = ContextObject.make_stage(f"curiosity_answer_{gap_name}", [q_ctx.context_id], {"answer": reply})
            a_ctx.component = "curiosity"; a_ctx.semantic_label = "answer"; a_ctx.tags.append("curiosity")
            a_ctx.retrieval_score = activation_map.get(q_ctx.context_id, 0.0)
            a_ctx.retrieval_metadata = {"question_id": q_ctx.context_id}
            self.memman.reinforce(q_ctx.context_id, [a_ctx.context_id])
            a_ctx.touch(); self.repo.save(a_ctx)
            self.memman.register_relationships(a_ctx, self.embed_text)

            state.setdefault("curiosity_used", []).append(getattr(picked, "semantic_label", f"fallback_{gap_name}"))
            probes.append(reply)
            used += 1

        return probes


    
    def _get_prompt(self, label: str) -> str:
        # Ensure initial seeding completed (non-blocking if already set)
        try:
            self._await_prompts_ready(0.0)
            self._ensure_prompts_present()
        except Exception:
            pass

        # Prefer repo copy
        rows = [c for c in self.repo.query(lambda c:
            c.component == "prompt" and (c.semantic_label or "").strip() == label
        )]
        if rows:
            rows.sort(key=lambda c: c.timestamp, reverse=True)
            meta = rows[0].metadata or {}
            val = meta.get("prompt")
            if isinstance(val, str) and val.strip():
                return val

        # Fallback to the in-memory canonical prompt text
        base = getattr(self, "system_prompts", {}).get(label, "")

        # NEW: route through RL prompt variants (slot == label)
        try:
            chosen_text, _vid = self._choose_prompt_text(label, base)
            return chosen_text
        except Exception:
            return base    


    def _stage_system_prompt_refine(self, state: Dict[str, Any]) -> str | None:
        """
        RL-gated self-mutation of prompts & policies, with full visibility
        into narrative, architecture, tool outcomes—and now a window of past
        evaluation events.
        """
        

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
            self.memman.register_relationships(ctx, embed_text)


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
                self.memman.register_relationships(patch, embed_text)


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
        self.memman.register_relationships(refine_ctx, embed_text)


        return f"{action}:{text or '(none)'}"



    # Helper used above ---------------------------------------------------
    def repo_exists(self, cid: str) -> bool:
        """Return True iff the context-id still resolves in the repository."""
        try:
            self.repo.get(cid)
            return True
        except KeyError:
            return False
                    

    # ─────────────────────────────────────────────────────────────────────────────
    #  Decision & utility callbacks
    # ─────────────────────────────────────────────────────────────────────────────

    def decision_callback(
        self,
        user_text: str,
        options: List[str],
        system_template: str,
        history_size: int,
        context_type: str,
        var_names: List[str],
        record: bool = True
    ) -> str:
        """
        Ask `self.decision_model` to choose exactly one item from `options`.
        Returns the model's full response: a one-sentence justification and,
        on a new line, exactly one option token.

        Upgrades:
        • Strict options parsing with up to 3 attempts (guard against drift).
        • Optional policy_manager "priors" to bias decisions (if available).
        • Context includes a compact narrative + recent turns.
        """
        import re
        from context import ContextObject

        options = [str(o).strip() for o in options if str(o).strip()]
        assert options, "decision_callback: options required"

        # 0) Optional priors from a policy manager
        priors = {}
        pm = getattr(self, "policy_manager", None)
        if pm and hasattr(pm, "decision_priors"):
            try:
                priors = pm.decision_priors(user_text=user_text, options=options) or {}
            except Exception:
                priors = {}

        # 1) Build mapping & primary system prompt
        mapping    = {vn: opt for vn, opt in zip(var_names, options)}
        system_msg = system_template.format(**mapping)

        # 2) Load narrative
        narr_ctx  = self._load_arbitrary_context(semantic_label=context_type)
        narrative = narr_ctx.summary or "(no narrative yet)"

        # 3) Recent turns
        segs = sorted(
            [c for c in self.repo.query(
                lambda c: c.domain=="segment" and c.semantic_label in ("user_input","assistant")
            )],
            key=lambda c:c.timestamp
        )[-history_size:]
        snippet = "\n".join(
            f"{'User' if c.semantic_label=='user_input' else 'Assistant'}: {c.summary}"
            for c in segs
        )

        if snippet:
            context_block = (
                "### Narrative So Far ###\n" f"{narrative}\n\n"
                "### Recent Turns ###\n"   f"{snippet}"
            )
        else:
            context_block = "### Narrative So Far ###\n" f"{narrative}"

        # 4) Second system prompt with justification instruction + priors
        priors_txt = ""
        if priors:
            pairs = ", ".join(f"{k}:{priors.get(k):.2f}" for k in options if k in priors)
            priors_txt = f"\nUse these option priors as a *soft* bias: {pairs}"

        system_msg_2 = (
            "Now, based on the above, obey the ruleset below.\n"
            "When you answer, **first** write a **one-sentence justification**, "
            "**then** on a **new line** write exactly one of: "
            + ", ".join(options)
            + "\n\nRuleset: "
            + system_template.format(**mapping)
            + priors_txt
        )

        # 5) Debug dump
        self._print_stage_context("decision_callback", {
            "narrative":      narrative,
            "recent_turns":   snippet or "(none)",
            "options":        ", ".join(options),
            "system_prompt":  system_msg,
            "ruleset_prompt": system_msg_2,
            "user_text":      user_text,
            "priors":         priors or {},
        })

        # 6) Build user message
        user_msg = f"{context_block}\n\nNEW MESSAGE:\n{user_text}"

        # 7) Invoke model up to 3 attempts until we see a standalone option token
        attempt = 0
        prompt_user = user_msg
        while attempt < 3:
            full_resp = self._stream_and_capture(
                model=self.decision_model,
                messages=[
                    {"role":"system","content":system_msg},
                    {"role":"user",  "content":prompt_user},
                    {"role":"system","content":system_msg_2},
                ],
                tag="[Decision]"
            ).strip()

            # record Q&A if desired
            if record:
                q_name = "decision_question" if attempt==0 else "decision_feedback_question"
                q_ctx = ContextObject.make_stage(q_name, [narr_ctx.context_id], {
                    "prompt_system": system_msg,
                    "prompt_user":   prompt_user
                })
                q_ctx.component="decision"; q_ctx.semantic_label="question"; q_ctx.tags.append("decision")
                q_ctx.touch(); self.repo.save(q_ctx)

                a_name = "decision_answer" if attempt==0 else "decision_feedback_answer"
                a_ctx = ContextObject.make_stage(a_name, [q_ctx.context_id], {"answer": full_resp})
                a_ctx.component="decision"; a_ctx.semantic_label="answer"; a_ctx.tags.append("decision")
                a_ctx.touch(); self.repo.save(a_ctx)

            # strict standalone token match (begin/end or newline boundaries)
            m = re.search(rf"(?:^|\n)\s*({'|'.join(map(re.escape, options))})\s*(?:$|\n)", full_resp, re.I)
            if m:
                return full_resp

            prompt_user = (
                "I didn’t see one of the required options on its own line.\n"
                f"Previous: {full_resp}\n\n"
                "Please answer with exactly one of, on a new line: "
                + ", ".join(options)
            )
            attempt += 1

        # Last resort: extract any option occurrence
        m2 = re.search(rf"\b({'|'.join(map(re.escape, options))})\b", full_resp, re.I)
        if m2:
            return full_resp + f"\n{m2.group(1)}"
        return full_resp


    def filter_callback(self, user_text: str) -> tuple[bool,str]:
        """
        Returns (should_respond, full_response_with_justification)
        """
        resp = self.decision_callback(
            user_text=user_text,
            options=["YES","NO"],
            system_template=(
                "Decide whether to respond.\n"
                "Prefer YES unless clear spam, duplicate, or empty.\n"
                "Answer exactly {arg1} or {arg2}."
            ),
            context_type="narrative_context",
            history_size=4,
            var_names=["arg1","arg2"],
            record=False
        )
        import re
        m = re.search(r"\b(YES|NO)\b", resp, re.I)
        decision = (m.group(1).upper() if m else "YES")
        return (decision=="YES", resp)


    def tools_callback(self, user_text: str) -> tuple[bool,str]:
        """
        Returns (use_tools, full_response_with_justification)
        Bias: use tools by default unless trivial small-talk/short answer.
        """
        resp = self.decision_callback(
            user_text=user_text,
            options=["TOOLS","NO_TOOLS"],
            system_template=(
                "Choose whether to call tools. Prefer TOOLS unless the task is a trivial, "
                "low-risk, short conversational reply.\n"
                "Answer exactly {arg1} or {arg2}."
            ),
            context_type="narrative_context",
            history_size=4,
            var_names=["arg1","arg2"],
            record=False
        )
        import re
        m = re.search(r"\b(TOOLS|NO_TOOLS)\b", resp, re.I)
        decision = (m.group(1).upper() if m else "TOOLS")
        return (decision=="TOOLS", resp)


    # ─────────────────────────────────────────────────────────────────────────────
    #  I/O helpers
    # ─────────────────────────────────────────────────────────────────────────────

    async def _stream_and_capture_async(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        tag: str = "",
        max_image_bytes: int = 8 * 1024 * 1024,
        images: list[bytes] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """
        Async wrapper around `_stream_and_capture` that simply runs the blocking
        function in a worker thread.  All keyword-only args are forwarded as
        **keywords**, preventing the positional‐argument crash.
        """
        import asyncio
        return await asyncio.to_thread(
            self._stream_and_capture,
            model,
            messages,
            tag=tag,
            max_image_bytes=max_image_bytes,
            images=images,
            on_token=on_token,
        )


    async def _assemble_and_infer(
        self,
        user_text: str,
        state: dict,
        status_cb: Callable[[str, Any], None]
    ) -> str:
        """
        Runs your sync `_stage10_assemble_and_infer(user_text, state)` safely.
        """
        return (await _to_thread_safe(self._stage10_assemble_and_infer, user_text, state) or "").strip()


    async def _invoke_single_tool(
        self,
        call: str,
        state: dict,
        status_cb: Callable[[str, Any], None]
    ):
        """
        Invoke one tool in parallel; drops into thread and reports via status_cb.
        """
        try:
            ctx = await _to_thread_safe(self._stage9_invoke_tool, call, state)
            if ctx is not None:
                status_cb("tool_output", {call: ctx.metadata.get("output", ctx.metadata)})
            return ctx
        except Exception as e:
            status_cb("tool_error", f"{call}: {e}")
            return None


    async def _bootstrap_for_quick_take(self, user_text: str) -> dict:
        """
        Lightweight bootstrap to enable an *informed* quick-take without waiting
        for the full retrieval stack. Uses integrator quick-contract and preserves
        mandatory state keys.
        """
        import uuid
        from context import ContextObject

        boot_state: dict[str, Any] = {
            "errors":        [],
            "recent_ids":    [],
            "tool_ctxs":     [],
            "images":        [],
            "fixed_calls":   [],
            "provisional_sent": False,
            "user_text":     user_text.strip(),
            "conversation_id": getattr(self, "_active_conversation_id", uuid.uuid4().hex),
            "user_id": getattr(self, "current_user_id", "anon"),
        }

        # Stage 1 – record raw input
        try:
            boot_state["user_ctx"] = await asyncio.to_thread(self._stage1_record_input, user_text, boot_state)
        except Exception as e:
            boot_state["errors"].append(("record_input", str(e)))
            dummy = ContextObject.make_stage("record_input_failed", [], {"summary": user_text[:120]})
            dummy.touch(); self.repo.save(dummy)
            boot_state["user_ctx"] = dummy

        # Integrator quick ingest + contract
        try:
            await asyncio.to_thread(self.integrator.ingest, [boot_state["user_ctx"]])
            quick = await asyncio.to_thread(self.integrator.contract, keep_ids=[boot_state["user_ctx"].context_id])
            boot_state["merged"] = quick
        except Exception as e:
            boot_state["errors"].append(("integrator_quick", str(e)))
            boot_state["merged"] = [boot_state["user_ctx"]]

        return boot_state


    def _await_prompts_ready(self, timeout: float = 5.0) -> None:
        """
        Block until initial prompt seeding is done (or timeout).
        No-op if the event is missing (backward compat).
        """
        evt = getattr(self, "_prompts_ready_evt", None)
        if evt is not None:
            evt.wait(timeout)


    def _ensure_prompts_present(self) -> None:
        """
        Ensure canonical prompts exist in the repo (idempotent).
        """
        try:
            have = {(c.semantic_label or "").strip() for c in self.repo.query(lambda c: c.component == "prompt")}
            need = set(getattr(self, "system_prompts", {}).keys())
            if need and not need.issubset(have):
                self._seed_static_prompts()
        except Exception:
            pass


    # ─────────────────────────────────────────────────────────────────────────────
    #  Orchestrator (Quick-Take + Planner) — with direct-plan bypass
    # ─────────────────────────────────────────────────────────────────────────────
    async def run_with_meta_context(
        self,
        user_text: str,
        status_cb: Callable[[str, Any], None] | None = None,
        *,
        images: List[str] | None = None,
        on_token: Callable[[str], None] | None = None,
        skip_quick_phases: bool = False,
        direct_plan: bool = False,  # bypasses QT/clarifier/etc. and feeds user_text into Stage 7
    ) -> str:
        """
        Two-phase orchestrator (non-destructive). Supports a direct Stage-7 bypass:

        1) Quick-Take  – fast one-liner with ranked prior U↔A pairs (disabled when direct_plan=True)
        2) Planner     – full pipeline:
                        5b(KG) → 7(planner) → 7b(validate) → 8(chain)
                        → 9(DAG exec) → 9b(reflect/replan) → 10/10b(finalize) → 11(memory)

        When direct_plan=True:
        • Skip Quick-Take and clarifier
        • Seed planner directly with raw user_text (state['planner_seed_text'])
        • Attempt explicit Stage-7 calls if available; else hint _stage8_orchestrate via state flags
        """
        import json, traceback, uuid, textwrap, inspect, time
        from datetime import datetime, timezone
        import numpy as np

        # ── status callback ───────────────────────────────────────────────
        if status_cb is None:
            status_cb = lambda *_a, **_k: None

        # Attach high-observability pieces if present (no-op if already set)
        try:
            if hasattr(self, "_ensure_orchestrator_attached"):
                self._ensure_orchestrator_attached()
        except Exception:
            pass

        # ---------- helper: ranked interleaved context via Tools.context_query ----------
        def _ranked_pairs_via_tools(
            state: dict,
            *,
            top_pairs: int = 5,
            pool: int = 80,
            window: str | None = None,   # e.g. "72 hours"
        ) -> tuple[list[str], list[str]]:
            """
            Use Tools.context_query to fetch recent user_input segments and final_inference stages,
            interleave them into U→A pairs, then rank by cosine(sim(user_text, pair)) with a small
            recency boost. Returns (display_lines, flattened_pair_ids).
            """
            # resolve Tools class
            _Tools = None
            try:
                _Tools = Tools  # already imported in module scope?
            except NameError:
                try:
                    from tools import Tools as _Tools  # best effort
                except Exception:
                    _Tools = None
            if _Tools is None:
                return [], []

            def _to_dt(ts):
                try:
                    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except Exception:
                    return datetime.min.replace(tzinfo=timezone.utc)

            # knobs
            window = window or self.cfg.get("quick_take_window", "72 hours")
            top_k_each = max(pool, top_pairs * 6)

            # scope by conversation tag if available (falls back if empty)
            tag_filter = None
            cid = state.get("conversation_id")
            if cid:
                tag_filter = [cid]

            def _q(**kw):
                return _Tools.context_query(assembler=self, **kw)

            # Pull final_inference (Assistant)
            def _pull_infers(use_tags: bool):
                return _q(
                    window=window,
                    domain=["stage"],
                    component=["final_inference"],
                    semantic_label=["final_inference"],
                    similarity_to=state.get("user_text") or "",
                    top_k=top_k_each,
                    tags=(tag_filter if use_tags else None),
                )

            # Pull user_input (User)
            def _pull_user_inputs(use_tags: bool):
                return _q(
                    window=window,
                    domain=["segment"],
                    component=["segment"],
                    semantic_label=["user_input"],
                    similarity_to=state.get("user_text") or "",
                    top_k=top_k_each,
                    tags=(tag_filter if use_tags else None),
                )

            # try with tags first, then retry without tags if nothing comes back
            try:
                j_infers = json.loads(_pull_infers(use_tags=True))
                j_users  = json.loads(_pull_user_inputs(use_tags=True))
                infers = j_infers.get("results", [])
                users  = j_users.get("results", [])
                if not infers or not users:
                    j_infers = json.loads(_pull_infers(use_tags=False))
                    j_users  = json.loads(_pull_user_inputs(use_tags=False))
                    infers = j_infers.get("results", [])
                    users  = j_users.get("results", [])
            except Exception:
                return [], []

            if not infers or not users:
                return [], []

            users_sorted  = sorted(users,  key=lambda r: _to_dt(r.get("timestamp")))
            infers_sorted = sorted(infers, key=lambda r: _to_dt(r.get("timestamp")))

            # pair: for each User, pick the next FinalInference at/after the user timestamp,
            # else fall back to the nearest prior inference.
            pairs = []
            j = 0
            for u in users_sorted[-(pool * 3):]:
                uts = _to_dt(u.get("timestamp"))
                while j < len(infers_sorted) and _to_dt(infers_sorted[j].get("timestamp")) < uts:
                    j += 1
                a = infers_sorted[j] if j < len(infers_sorted) else (infers_sorted[-1] if infers_sorted else None)
                if not a:
                    continue
                u_txt = (u.get("summary") or "").strip()
                a_txt = (a.get("summary") or "").strip()
                if not u_txt and not a_txt:
                    continue
                pairs.append((
                    u_txt, a_txt,
                    uts, _to_dt(a.get("timestamp")),
                    u.get("context_id"), a.get("context_id")
                ))

            if not pairs:
                return [], []

            # score: cosine(sim(user_text, "U\nA")) + recency boost
            def _embed(t: str) -> np.ndarray | None:
                try:
                    v = self.embed_text(t or "")
                    return np.asarray(v, dtype=float).reshape(-1)
                except Exception:
                    return None

            uvec = _embed(state.get("user_text") or "")
            now = datetime.now(timezone.utc)

            def _recency_boost(ts):
                age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
                return 0.5 ** (age_days / 3.0)

            scored = []
            for u_txt, a_txt, uts, ats, ucid, acid in pairs:
                pair_text = f"{u_txt}\n{a_txt}"
                pvec = _embed(pair_text)
                if uvec is None or pvec is None:
                    cos = 0.0
                else:
                    den = float(np.linalg.norm(uvec) * np.linalg.norm(pvec)) or 1.0
                    cos = float(np.dot(uvec, pvec) / den)
                rec = _recency_boost(max(uts, ats))
                score = 0.72 * cos + 0.28 * rec

                u_short = textwrap.shorten(u_txt.replace("\n", " "), width=180, placeholder="…")
                a_short = textwrap.shorten(a_txt.replace("\n", " "), width=180, placeholder="…")
                scored.append((score, f"U: {u_short} || A: {a_short}", (ucid, acid)))

            scored.sort(key=lambda t: t[0], reverse=True)
            lines = [s for _, s, _ in scored[:top_pairs]]
            pair_ids = [i for _, _, ids in scored[:top_pairs] for i in ids if i]
            return lines, pair_ids

        # ── shared state bootstrap (WITH TIME CONTEXT) ───────────────────────
        state = getattr(self, "_last_state", {}) or {}
        state["start_ts"] = state.get("start_ts") or datetime.now(timezone.utc).timestamp()
        state.setdefault("stages_run", [])
        state["images"] = images or []
        state.setdefault("conversation_id", getattr(self, "_active_conversation_id", uuid.uuid4().hex))
        state.setdefault("user_id", getattr(self, "current_user_id", "anon"))

        # Build time context (human + machine) and inject everywhere
        local_now = datetime.now().astimezone()
        utc_now   = datetime.now(timezone.utc)

        human_local = local_now.strftime("%A, %B %d, %Y %I:%M %p %Z (UTC%z)")
        human_utc   = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")
        time_banner = (
            f"To aid in making chronologically informed decisions, the current time is {human_local}. "
            f"UTC now: {human_utc}. Always compute time deltas relative to this moment."
        )

        # Stash rich time fields in state for every stage to consume
        state["system_time_prompt"]   = time_banner
        state["now_local_human"]      = human_local
        state["now_utc_human"]        = human_utc
        state["now_local_iso"]        = local_now.isoformat()
        state["now_utc_iso"]          = utc_now.isoformat().replace("+00:00", "Z")
        state["now_epoch"]            = utc_now.timestamp()
        try:
            off = local_now.utcoffset() or (utc_now.utcoffset() or 0)
            state["now_tz_offset_minutes"] = int(off.total_seconds() // 60)
        except Exception:
            state["now_tz_offset_minutes"] = 0
        state["now_tz_name"]          = local_now.tzname() or "Local"

        # Create a first-class ContextObject for time and persist it so merges/planner see it
        time_ctx = None
        try:
            from context import ContextObject
            time_ctx = ContextObject.make_stage(
                "system_time",
                [],
                {
                    "text": time_banner,
                    "now_local_iso": state["now_local_iso"],
                    "now_utc_iso": state["now_utc_iso"],
                    "now_epoch": state["now_epoch"],
                    "tz_name": state["now_tz_name"],
                    "tz_offset_minutes": state["now_tz_offset_minutes"],
                },
            )
            time_ctx.stage_id = "system_time"
            time_ctx.summary = time_banner[:250]
            time_ctx.metadata.update({
                "conversation_id": state["conversation_id"],
                "user_id": state["user_id"],
            })
            time_ctx.touch()
            self.repo.save(time_ctx)
        except Exception:
            time_ctx = None  # non-fatal

        # ── Stage 1: record input ────────────────────────────────────────────
        try:
            user_ctx = self._stage1_record_input(user_text, state)
            state["user_ctx"] = user_ctx
            state["user_text"] = user_text
            status_cb("input_recorded", {"ctx_id": getattr(user_ctx, "context_id", None), "now": state["now_local_human"]})
        except Exception as e:
            status_cb("error_stage1", {"error": f"{type(e).__name__}: {e}"})

        # ── Stage 2: system prompts (policies, etc.) + time injection ────────
        try:
            sys_ctxs = self._stage2_load_system_prompts()
            if time_ctx:
                try:
                    sys_ctxs = list(sys_ctxs) if isinstance(sys_ctxs, (list, tuple)) else [sys_ctxs]
                    sys_ctxs.append(time_ctx)
                except Exception:
                    pass
            state["sys_ctx"] = sys_ctxs
            state["stages_run"].append("stage2_load_system_prompts")
        except Exception as e:
            sys_ctxs = []
            status_cb("error_stage2", {"error": f"{type(e).__name__}: {e}"})

        state["extra_system_messages"] = [state["system_time_prompt"]]

        try:
            self._task_inject_countdown_context(state)
            self._stage_task_detect_and_schedule(user_text=user_text, state=state)
        except Exception:
            pass

        do_quick = (not skip_quick_phases) and (not direct_plan)

        # ── Stage 3: retrieve & merge context (skip on direct_plan) ──────────
        if not direct_plan:
            try:
                merged_out = self._stage3_retrieve_and_merge_context(
                    user_text=user_text,
                    user_ctx=user_ctx,
                    sys_ctx=sys_ctxs,
                    extra_ctx=[time_ctx] if time_ctx else None,
                    recall_ids=None,
                )
                state.update(merged_out or {})
                state["stages_run"].append("stage3_retrieve_and_merge_context")
                state.setdefault("merged", state.get("merged", []))
                state.setdefault("merged_ids", state.get("merged_ids", [c.context_id for c in state["merged"] if hasattr(c, "context_id")]))
                status_cb("context_merged", {"merged": len(state.get("merged_ids", [])), "now": state["now_local_human"]})
            except Exception as e:
                state.setdefault("merged", [])
                state.setdefault("merged_ids", [])
                status_cb("error_stage3", {"error": f"{type(e).__name__}: {e}"})
        else:
            state.setdefault("merged", [])
            state.setdefault("merged_ids", [])
            if time_ctx:
                try:
                    state["merged"].append(time_ctx)
                    state["merged_ids"].append(time_ctx.context_id)
                except Exception:
                    pass

        # ── Quick-Take (optional) — never when direct_plan=True ─────────────
        if do_quick:
            try:
                status_cb("quick_take_begin", {"now": state["now_local_human"]})
                pairs, pair_ids = _ranked_pairs_via_tools(
                    state,
                    top_pairs=6,
                    pool=80,
                    window=self.cfg.get("quick_take_window", "72 hours"),
                )
                state["quick_take_pairs"] = pairs
                state["quick_take_pair_ids"] = pair_ids

                qt_sys = self._get_prompt("quick_take_prompt")
                qt_guard = getattr(self, "_get_prompt", lambda *_: None)("quick_take_policy") or (
                    "SYSTEM / QuickTake Guard: Respond in one short line. "
                    "If the request conflicts with safety or policy, reply with a brief refusal like \"Can't help with that.\" "
                    "No disclaimers, hotlines, or explanations."
                )
                qt_ctx = "\n".join(pairs) if pairs else "(no prior context)"
                msgs = [
                    {"role": "system", "content": state["system_time_prompt"]},
                    {"role": "system", "content": qt_sys},
                    {"role": "system", "content": qt_guard},
                    {"role": "system", "content": "Ranked prior U↔A pairs:\n" + qt_ctx},
                    {"role": "user",   "content": user_text},
                ]
                kwargs = {"tag": "[QuickTake]", "images": state.get("images")}
                if "on_token" in getattr(self._stream_and_capture, "__code__", type("", (), {"co_varnames": ()})).co_varnames:
                    kwargs["on_token"] = on_token
                qt_raw = self._stream_and_capture(self.primary_model, msgs, **kwargs)
                quick = await qt_raw if inspect.isawaitable(qt_raw) else (qt_raw or "")
                quick = (quick or "").strip()
                if quick:
                    state["provisional_sent"] = True
                    try:
                        from context import ContextObject
                        prov = ContextObject.make_stage(
                            "quick_take",
                            [state["user_ctx"].context_id],
                            {"text": quick, "pairs": pairs, "now": state["now_local_human"]}
                        )
                        prov.stage_id = "quick_take"; prov.summary = quick[:250]
                        prov.metadata.update({"conversation_id": state["conversation_id"], "user_id": state["user_id"]})
                        prov.touch(); self.repo.save(prov)
                        state.setdefault("merged", []).append(prov)
                        state.setdefault("merged_ids", []).append(prov.context_id)
                    except Exception:
                        pass
                    status_cb("quick_take_done", {"preview": quick[:220]})
            except Exception as e:
                status_cb("quick_take_error", {"error": f"{type(e).__name__}: {e}"})

        # ── Stage 4: intent clarification (skip on direct_plan) ──────────────
        clar_ctx = None
        if not direct_plan:
            try:
                clar_ctx = self._stage4_intent_clarification(
                    user_text=user_text,
                    state=state,
                    on_token=None,  # never stream clarifier (prevents TTS on JSON)
                )
                state["clar_ctx"] = clar_ctx
                state["stages_run"].append("stage4_intent_clarification")
                status_cb("clarified", {"topic": (getattr(clar_ctx, "summary", "") or "")[:140], "now": state["now_local_human"]})
            except Exception as e:
                clar_ctx = None
                status_cb("error_stage4", {"error": f"{type(e).__name__}: {e}"})

            # Re-attempt scheduling with better structure if we have it
            try:
                if clar_ctx:
                    self._stage_task_detect_and_schedule(
                        user_text=(clar_ctx.metadata.get("notes") or user_text),
                        state=state,
                        allow_update=True,
                    )
            except Exception:
                pass

        # ── Stage 5: external knowledge (skip on direct_plan) ────────────────
        know_ctx = None
        if not direct_plan:
            try:
                know_ctx = self._stage5_external_knowledge(clar_ctx, state)
                state["know_ctx"] = know_ctx
                state["stages_run"].append("stage5_external_knowledge")
                status_cb("knowledge_built", {"snippets": len(state.get("knowledge_snippets", [])), "now": state["now_local_human"]})
            except Exception as e:
                know_ctx = None
                status_cb("error_stage5", {"error": f"{type(e).__name__}: {e}"})

        # ── Stage 5b: planning KG (tool/arg catalog) ─────────────────────────
        try:
            tools_list = self._stage6_prepare_tools()
            state["tools_list"] = tools_list

            def _tool_name(obj):
                if isinstance(obj, str):
                    return obj
                if isinstance(obj, dict):
                    return obj.get("name") or obj.get("tool") or obj.get("semantic_label") or obj.get("id") or "unknown"
                return str(obj)

            def _tool_features_map(tool_names: list[str]) -> dict[str, list[float]]:
                """
                Build LinUCB features:
                [affinity, success_rate, 1-arg_err, 1-norm_latency, 1.0]
                from recent repo observations + cheap text affinity to the clarifier notes.
                """
                names = list(tool_names)
                feats: dict[str, list[float]] = {}
                clar_txt = ""
                try:
                    clar = state.get("clar_ctx")
                    clar_txt = ((clar and clar.metadata.get("notes")) or (state.get("user_text") or ""))[:400]
                except Exception:
                    pass

                # vectorize once
                try:
                    qv = self.embed_text(clar_txt)
                except Exception:
                    qv = np.zeros(768)

                # historical latency mins for normalization
                latencies: dict[str, list[float]] = {n: [] for n in names}
                succs: dict[str, list[int]] = {n: [] for n in names}
                argerrs: dict[str, list[int]] = {n: [] for n in names}

                rows = list(self.repo.query(lambda c: c.component == "tool_output"))
                rows = rows[-500:]  # recent window
                for r in rows:
                    meta = r.metadata or {}
                    tn = meta.get("tool_name") or _tool_name(meta)
                    if tn not in latencies:
                        continue
                    if meta.get("exception") is None:
                        succs[tn].append(1)
                    else:
                        succs[tn].append(0)
                        if "argument" in str(meta.get("exception", "")).lower():
                            argerrs[tn].append(1)
                        else:
                            argerrs[tn].append(0)
                    if "latency_ms" in meta:
                        latencies[tn].append(float(meta["latency_ms"]))

                # global latency normalization
                all_lats = [x for arr in latencies.values() for x in arr]
                hi = (np.percentile(all_lats, 90) if all_lats else 1.0) or 1.0

                for nm in names:
                    # affinity: crude cosine between clar text and the tool name
                    try:
                        tv = self.embed_text(nm)
                        denom = (np.linalg.norm(qv) * np.linalg.norm(tv)) or 1.0
                        affinity = float(np.dot(qv, tv) / denom)
                    except Exception:
                        affinity = 0.0

                    sr = (sum(succs[nm]) / max(1, len(succs[nm]))) if succs[nm] else 0.5
                    arg_bad = (sum(argerrs[nm]) / max(1, len(argerrs[nm]))) if argerrs[nm] else 0.2
                    lat = np.median(latencies[nm]) if latencies[nm] else hi
                    inv_lat = 1.0 - min(1.0, float(lat) / float(hi or 1.0))

                    feats[nm] = [affinity, sr, 1.0 - arg_bad, inv_lat, 1.0]
                return feats

            # compute ranked order (fallback to identity if bandit missing)
            names = [_tool_name(t) for t in tools_list]
            feats_map = _tool_features_map(names)
            try:
                ranked_names = self.bandit.rank_tools(names, feats_map)
            except Exception:
                ranked_names = names

            name_to_objs: dict[str, list] = {}
            for t in tools_list:
                name_to_objs.setdefault(_tool_name(t), []).append(t)

            tools_list_ranked: list = []
            for n in ranked_names:
                tools_list_ranked.extend(name_to_objs.pop(n, []))
            for remaining in name_to_objs.values():
                tools_list_ranked.extend(remaining)

            state["tools_list"] = tools_list_ranked
            self.tools_list = tools_list_ranked

            self._stage5b_build_planning_kg(
                clar_ctx if not direct_plan else None,
                know_ctx if not direct_plan else None,
                tools_list_ranked,
                state
            )
            state["stages_run"].append("stage5b_build_planning_kg")
        except Exception as e:
            status_cb("error_stage5b", {"error": f"{type(e).__name__}: {e}"})

        # === Bandit knobs (planner/executor) — choose once per run ===
        if hasattr(self, "bandit"):
            idx_t, temp = self.bandit.choose_knob("planner_temperature", [0.2, 0.4, 0.6, 0.8])
            idx_r, retries = self.bandit.choose_knob("executor_retries", [1, 2, 3])

            state.setdefault("selected_knobs", {})
            state["selected_knobs"]["planner_temperature"] = (idx_t, temp)
            state["selected_knobs"]["executor_retries"]     = (idx_r, retries)

        # ── DIRECT PLAN BYPASS INTO STAGE 7 ───────────────────────────────────
        if direct_plan:
            state["planner_mode"] = "direct_from_user"
            state["planner_seed_text"] = user_text
            state.setdefault("planner_context_ids", [])
            if state.get("user_ctx") and hasattr(state["user_ctx"], "context_id"):
                state["planner_context_ids"].append(state["user_ctx"].context_id)
            if time_ctx and time_ctx.context_id not in state["planner_context_ids"]:
                state["planner_context_ids"].append(time_ctx.context_id)

            state["planner_context_pack"] = {
                "seed_text": user_text,
                "merged_ids": state.get("merged_ids", []),
                "tools_list": state.get("tools_list", []),
                "conversation_id": state.get("conversation_id"),
                "system_time_prompt": state["system_time_prompt"],
                "now_utc_iso": state["now_utc_iso"],
                "now_local_iso": state["now_local_iso"],
                "now_epoch": state["now_epoch"],
            }

            try:
                status_cb("planner_direct_begin", {"mode": "direct_from_user", "now": state["now_local_human"]})

                if hasattr(self, "_stage7_planning_summary"):
                    plan_ctx = self._stage7_planning_summary(
                        user_text=user_text,
                        state=state,
                        seed_text=user_text,
                        bypass_context=True,
                    )
                    state["plan_ctx"] = plan_ctx
                    state["stages_run"].append("stage7_planning_summary")

                    if hasattr(self, "_stage7b_plan_validation"):
                        val_ctx = self._stage7b_plan_validation(plan_ctx, state)
                        state["plan_validation_ctx"] = val_ctx
                        state["stages_run"].append("stage7b_plan_validation")

                    final, tool_ctxs = self._stage8_orchestrate(user_text=user_text, state=state)
                    state["tool_ctxs"] = tool_ctxs
                    state["final"] = final
                    state["stages_run"].extend([
                        "stage8_tool_chaining", "stage9_invoke_with_retries",
                        "stage9b_reflection_and_replan", "stage10_assemble_and_infer",
                        "stage10b_response_critique_and_safety", "stage11_memory_writeback"
                    ])
                    status_cb("planner_direct_done", {"tools": len(tool_ctxs), "now": state["now_local_human"]})
                else:
                    state["bypass_to_stage7"] = True
                    final, tool_ctxs = self._stage8_orchestrate(user_text=user_text, state=state)
                    state["tool_ctxs"] = tool_ctxs
                    state["final"] = final
                    state["stages_run"].extend([
                        "stage7_planning_summary", "stage7b_plan_validation",
                        "stage8_tool_chaining", "stage9_invoke_with_retries",
                        "stage9b_reflection_and_replan", "stage10_assemble_and_infer",
                        "stage10b_response_critique_and_safety", "stage11_memory_writeback"
                    ])
                    status_cb("planner_direct_done", {"tools": len(tool_ctxs), "now": state["now_local_human"]})

            except Exception as e:
                status_cb("error_pipeline", {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
                state["final"] = (state.get("draft") or user_text or "").strip()

        else:
            # ── Normal Planner+Executor pipeline via Stage 8 orchestrator ─────
            try:
                status_cb("planner_begin", {"now": state["now_local_human"]})
                final, tool_ctxs = self._stage8_orchestrate(user_text=user_text, state=state)
                state["tool_ctxs"] = tool_ctxs
                state["final"] = final
                state["stages_run"].extend([
                    "stage7_planning_summary", "stage7b_plan_validation",
                    "stage8_tool_chaining", "stage9_invoke_with_retries",
                    "stage9b_reflection_and_replan", "stage10_assemble_and_infer",
                    "stage10b_response_critique_and_safety", "stage11_memory_writeback"
                ])
                status_cb("planner_done", {"tools": len(tool_ctxs), "now": state["now_local_human"]})
            except Exception as e:
                status_cb("error_pipeline", {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
                state["final"] = (state.get("draft") or user_text or "").strip()

        # ── Optional: Stage 12 performance rating ────────────────────────────
        try:
            self._stage12_performance_rating(state)
        except Exception:
            pass

        # ensure _last_state is kept up to date
        try:
            self._last_state = state
        except Exception:
            pass

        # -------- RL: register turn & persist --------
        try:
            # latency
            try:
                start_ts = state.get("start_ts")
                if start_ts:
                    state["latency_ms"] = int((time.time() - start_ts) * 1000)
            except Exception:
                pass

            # reflection status
            state.setdefault("reflection_status", "OK" if not state.get("errors") else "NEEDS_FIX")
            state.setdefault("critic_needed", False)
            state.setdefault("user_feedback", state.get("user_feedback", 0))  # -1/0/+1 if you collect explicit thumbs

            # per-tool updates
            tool_updates = []
            seen = set()
            for t in (state.get("tool_ctxs") or []):
                meta = t.metadata or {}
                name = meta.get("tool_name") or meta.get("call", "").split("(")[0]
                if not name or name in seen:
                    continue
                seen.add(name)
                succ = 1.0 if meta.get("exception") is None else 0.0
                arg_ok = 1.0 if ("argument" not in str(meta.get("exception","")).lower()) else 0.0
                inv_lat = 1.0
                if "latency_ms" in meta:
                    inv_lat = max(0.0, min(1.0, 1.0 - float(meta["latency_ms"]) / float(self.cfg.get("latency_budget_ms", 60_000) or 60_000)))
                feats = [0.0, succ, arg_ok, inv_lat, 1.0]
                tool_updates.append((name, feats))

            selected_knobs = state.get("selected_knobs", {})
            prompts_used = dict(getattr(self, "_prompts_used_current", {}) or {})

            self.bandit.register_turn(
                tool_updates=tool_updates,
                selected_knobs=selected_knobs,
                prompts_used=prompts_used,
                state=state,
            )
            self.bandit.finalize_turn()
        except Exception:
            pass

        return state.get("final", "")
