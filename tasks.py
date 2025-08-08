# tasks.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import asyncio
import threading
import uuid

from context import ContextObject, HybridContextRepository, default_clock, _fmt_ts, ContextRepository

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _parse_utc(ts: str) -> datetime:
    # All our repo timestamps are UTC “YYYYmmddTHHMMSSZ”
    # Be generous and accept missing trailing Z.
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1]
    return datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

def _now_utc() -> datetime:
    # default_clock() returns naive UTC; standardize as aware UTC for math
    return default_clock().replace(tzinfo=timezone.utc)

def _sec_until(ts: str) -> float:
    try:
        return max((_parse_utc(ts) - _now_utc()).total_seconds(), 0.0)
    except Exception:
        return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Task model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScheduledTask:
    """
    Persistable task. Stored as a ContextObject (artifact/task).
    """
    task_id: str
    title: str
    due_at: str                    # UTC repo format "%Y%m%dT%H%M%SZ"
    payload_text: str              # what to run (seed for planner)
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: _fmt_ts(default_clock()))
    status: str = "scheduled"      # scheduled | running | completed | failed | canceled
    rrule: Optional[str] = None    # optional RRULE string for recurrence
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None
    remaining_seconds: Optional[float] = None
    tries: int = 0
    max_retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        title: str,
        due_in: Optional[float] = None,  # seconds from now
        due_at: Optional[datetime] = None,
        *,
        payload_text: str,
        conversation_id: Optional[str],
        user_id: Optional[str],
        rrule: Optional[str] = None,
        max_retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ScheduledTask":
        if due_at is None and due_in is None:
            raise ValueError("Provide due_at or due_in")
        if due_at is None:
            due_at = _now_utc() + timedelta(seconds=float(due_in))
        due_at_s = _fmt_ts(due_at.replace(tzinfo=None))  # repo helper adds Z
        tid = str(uuid.uuid4())
        return cls(
            task_id=tid,
            title=title,
            due_at=due_at_s,
            payload_text=payload_text,
            conversation_id=conversation_id,
            user_id=user_id,
            rrule=rrule,
            max_retries=max_retries,
            metadata=dict(metadata or {}),
        )

    # ── repo <-> context object ───────────────────────────────────────
    def to_context(self) -> ContextObject:
        ctx = ContextObject(
            domain="artifact",
            component="task",
            semantic_label="scheduled_task",
        )
        ctx.context_id = self.task_id
        ctx.summary = f"[{self.status}] {self.title}"
        ctx.tags = ["task", "scheduled"]
        ctx.metadata.update(asdict(self))
        ctx.metadata["model"] = "scheduled_task/v1"
        # put countdown in summary tail for human scanning
        if self.remaining_seconds is not None:
            ctx.summary = f"{ctx.summary}  (T-{int(self.remaining_seconds)}s)"
        # scope
        if self.conversation_id:
            ctx.metadata["conversation_id"] = self.conversation_id
        if self.user_id:
            ctx.metadata["user_id"] = self.user_id
        return ctx

    @classmethod
    def from_context(cls, ctx: ContextObject) -> "ScheduledTask":
        m = dict(ctx.metadata or {})
        # backfill from known fields if missing in metadata
        m.setdefault("task_id", ctx.context_id)
        m.setdefault("title", ctx.summary or "task")
        return cls(**m)


# ──────────────────────────────────────────────────────────────────────────────
# Task Manager
# ──────────────────────────────────────────────────────────────────────────────

class TaskManager:
    """
    - Persists tasks as ContextObjects (artifact/task)
    - Emits countdown updates as context (stage/task_countdown) to keep pipeline aware
    - Polls and launches due tasks via user-provided launcher callable
    """

    def __init__(
        self,
        repo: ContextRepository,
        *,
        launcher: Callable[[ScheduledTask], None | Any],   # called when due (can be async or sync)
        countdown_stage: bool = True,
        default_poll_seconds: float = 2.0,
    ):
        self.repo = repo
        self.launcher = launcher
        self.countdown_stage = countdown_stage
        self.default_poll_seconds = float(default_poll_seconds)
        self._stop_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None

    # ── CRUD ──────────────────────────────────────────────────────────

    def schedule(
        self,
        title: str,
        *,
        due_in: Optional[float] = None,
        due_at: Optional[datetime] = None,
        payload_text: str,
        conversation_id: Optional[str],
        user_id: Optional[str],
        rrule: Optional[str] = None,
        max_retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        task = ScheduledTask.create(
            title=title,
            due_in=due_in,
            due_at=due_at,
            payload_text=payload_text,
            conversation_id=conversation_id,
            user_id=user_id,
            rrule=rrule,
            max_retries=max_retries,
            metadata=metadata,
        )
        ctx = task.to_context()
        self.repo.save(ctx)
        return task

    def cancel(self, task_id: str) -> bool:
        try:
            ctx = self.repo.get(task_id)
        except KeyError:
            return False
        t = ScheduledTask.from_context(ctx)
        if t.status not in ("scheduled", "running"):
            return False
        t.status = "canceled"
        self.repo.save(t.to_context())
        return True

    def list(self, *, conversation_id: Optional[str] = None, user_id: Optional[str] = None) -> List[ScheduledTask]:
        def _f(c: ContextObject) -> bool:
            if c.domain != "artifact" or c.component != "task":
                return False
            if conversation_id and (c.metadata or {}).get("conversation_id") != conversation_id:
                return False
            if user_id and (c.metadata or {}).get("user_id") != user_id:
                return False
            return True
        rows = self.repo.query(_f)
        rows.sort(key=lambda c: (c.metadata or {}).get("due_at", c.timestamp))
        return [ScheduledTask.from_context(c) for c in rows]

    # ── Countdown & context emission ──────────────────────────────────

    def inject_countdown_context(self, state: Dict[str, Any]) -> List[ContextObject]:
        """
        Compute remaining secs for all scheduled tasks in scope and
        emit a small stage context per “soon” task so the planner can see it.
        """
        try:
            conv = state.get("conversation_id")
            uid = state.get("user_id")
            now = _now_utc()
            soon = []
            for t in self.list(conversation_id=conv, user_id=uid):
                if t.status != "scheduled":
                    continue
                rem = _sec_until(t.due_at)
                t.remaining_seconds = rem
                # persist the task with fresh remaining_seconds in metadata
                self.repo.save(t.to_context())
                if rem <= 0:
                    continue
                if rem <= 3600:  # only surface tasks under 1h
                    soon.append(t)

            out: List[ContextObject] = []
            if self.countdown_stage and soon:
                lines = []
                for t in sorted(soon, key=lambda x: x.remaining_seconds):
                    lines.append(f"• {t.title} — T-{int(t.remaining_seconds)}s (due {t.due_at})")
                msg = "Upcoming tasks:\n" + "\n".join(lines)
                sc = ContextObject.make_stage(
                    "task_countdown",
                    input_refs=[state.get("user_ctx", None) and state["user_ctx"].context_id or ""],
                    output={"text": msg, "items": [asdict(s) for s in soon]},
                )
                sc.summary = msg[:250]
                sc.metadata.update({
                    "conversation_id": conv,
                    "user_id": uid,
                    "count": len(soon),
                })
                sc.touch()
                self.repo.save(sc)
                out.append(sc)
            return out
        except Exception:
            return []

    # ── Poll & launch ────────────────────────────────────────────────

    def _due_tasks(self) -> List[ScheduledTask]:
        due: List[ScheduledTask] = []
        for t in self.list():
            if t.status != "scheduled":
                continue
            if _sec_until(t.due_at) <= 0.0:
                due.append(t)
        return due

    def tick_and_launch(self) -> int:
        """
        One-shot poll: update countdowns and launch all due tasks.
        Returns number of tasks launched.
        """
        launched = 0
        # update countdowns (also persists tasks)
        _ = self.inject_countdown_context({"conversation_id": None, "user_id": None, "user_ctx": None})
        # launch due
        for t in self._due_tasks():
            try:
                t.status = "running"
                t.last_run_at = _fmt_ts(default_clock())
                self.repo.save(t.to_context())

                maybe_coro = self.launcher(t)
                launched += 1

                # If launcher is async, fire-and-forget here
                if asyncio.iscoroutine(maybe_coro):
                    asyncio.create_task(maybe_coro)
            except Exception:
                # mark failure (will be retried if rrule or manual)
                t.status = "failed"
                t.tries += 1
                self.repo.save(t.to_context())
        return launched

    # ── background daemon (optional) ─────────────────────────────────

    def start_background(self, poll_seconds: Optional[float] = None) -> None:
        if self._bg_thread and self._bg_thread.is_alive():
            return
        period = self.default_poll_seconds if poll_seconds is None else float(poll_seconds)

        def _run():
            while not self._stop_event.is_set():
                try:
                    self.tick_and_launch()
                except Exception:
                    pass
                self._stop_event.wait(period)

        self._stop_event.clear()
        self._bg_thread = threading.Thread(target=_run, name="TaskManagerPoll", daemon=True)
        self._bg_thread.start()

    def stop_background(self) -> None:
        if not self._bg_thread:
            return
        self._stop_event.set()
        self._bg_thread.join(timeout=2.0)
        self._bg_thread = None
