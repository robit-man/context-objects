"""
task.py

A self-contained, idempotent task/reminder scheduler with robust de-dupe
and JSONL write throttling:

• Natural-language detection (“remind me…”, “in 15m”, “tomorrow 7am”, “every weekday at 9”)
• ContextObject(stage="task") lifecycle:
    scheduled -> firing(leased) -> sent  (and back to scheduled for RRULE recurrences)
• Single background asyncio daemon; wakes on demand, sleeps otherwise
• Delivery via provided async/sync send_func(chat_id, text)

De-dup & stability layers:
1) Stable task_key at schedule time
2) Per-task delivery guard fields in metadata:
   - attempt_delivery_key / attempt_epoch  (written BEFORE send)
   - last_delivery_key                      (written AFTER successful send)
3) Claim+lease with verification, so only one worker sends
4) In-process TTL cache to mute bursts even if the repo hiccups
5) Stale-lease repair returns 'firing' to 'scheduled' safely
6) Write throttling + change fingerprinting to stop JSONL churn
7) due_utc_iso/epoch normalization to avoid 1s epoch drift storms
8) Auto-finalize overdue one-shots (no re-send) to quiet “zombie” tasks
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Tuple, Dict

# ─────────────────────────── Tunables ────────────────────────────
LEASE_TTL_SEC         = 300          # how long a "firing" lease is valid before we repair
DELIVERY_DEDUPE_TTL   = 300          # suppress re-send for same delivery_key within this window
SEND_BACKOFF_ON_FAIL  = 30           # seconds before retrying a failed send
SLEEP_DEFAULT_SEC     = 60.0         # max sleep if nothing due
QUERY_BATCH_MAX       = 2000         # soft cap; repo.query still decides
CLAIM_BACKOFF_SEC     = 5            # backoff if claim verification fails
RETRY_SKEW_SEC        = 10           # small skew on rescheduling after guard conflict

# JSONL protection
WRITE_THROTTLE_SEC    = 8            # minimum seconds between identical saves per context
DUE_EPOCH_WIGGLE_SEC  = 2            # tolerate tiny epoch drift when comparing
OVERDUE_GRACE_SEC     = 5            # how long past due before we auto-finalize one-shots


class TaskScheduler:
    def __init__(
        self,
        repo: Any,
        logger: Optional[Any] = None,
        send_func: Optional[Callable[[Any, str], Any]] = None,
        node_id: str = "local",
        *,
        sqlite_retry_ms: int = 50,
        sqlite_retry_tries: int = 6,
    ):
        """
        repo: must support repo.query(callable) -> Iterable[ContextObject] and repo.save(ContextObject)
        logger: optional; must support .info/.warning/.error/.exception
        send_func: async or sync function(chat_id, text) to deliver Telegram messages
        node_id: identifier used when claiming due tasks
        """
        self.repo = repo
        self.logger = logger or _NullLogger()
        self.send_func = send_func
        self.node_id = node_id

        self._sched_event: Optional[asyncio.Event] = None
        self._scheduler_task: Optional[asyncio.Task] = None

        self._sqlite_retry_ms = int(sqlite_retry_ms)
        self._sqlite_retry_tries = int(sqlite_retry_tries)

        # in-process delivery guard: delivery_key -> last_epoch
        self._recent_deliveries: Dict[str, int] = {}

        # write-throttle caches
        self._fp_cache: Dict[str, str] = {}
        self._fp_time: Dict[str, int] = {}

    # ─────────────────────────── Public API ────────────────────────────

    def ensure_running(self) -> None:
        """Idempotently start the background daemon."""
        loop = asyncio.get_running_loop()
        if self._sched_event is None:
            self._sched_event = asyncio.Event()
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = loop.create_task(self._scheduler_loop(), name="TaskDaemon")

    def wakeup(self) -> None:
        """Poke the daemon so it re-scans now."""
        if self._sched_event is not None:
            self._sched_event.set()

    async def stop(self) -> None:
        """Gracefully stop the daemon (optional)."""
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

    def maybe_schedule(self, user_text: str, state: dict, user_ctx: Any | None) -> Tuple[bool, str]:
        """
        Detect & schedule a reminder from free text.
        Returns (scheduled?, ack_string). Idempotent (won’t create dups).

        state must contain:
          - conversation_id
          - user_id
          - user_text (best-effort)
        user_ctx is the ContextObject for this user message (for linking).
        """
        t = (user_text or "").strip()
        lower = t.lower()
        if not t:
            return False, ""

        # Opt-in triggers or obviously timey phrases
        obvious = any(kw in lower for kw in (
            "remind me", "set a timer", "timer for", "alarm", "wake me", "nudge me", "alert me", "ping me"
        ))
        terse_timey = bool(re.search(r"\b(in\s+\d+\s*(seconds?|minutes?|hours?|days?|weeks?)\b)|(\bat\s*\d{1,2}(:\d{2})?\s*(am|pm)?\b)", lower))
        has_every = "every" in lower

        if not (obvious or terse_timey or has_every):
            return False, ""

        now_local = datetime.now().astimezone()
        parsed = _parse_time_bits(lower, now_local)
        if (parsed.get("due_local") is None) and (parsed.get("rrule") is None):
            return False, ""

        # Normalize past/now → now+1s
        if parsed.get("due_local"):
            if parsed["due_local"].astimezone(timezone.utc).timestamp() <= _now_epoch():
                parsed["due_local"] = parsed["due_local"] + timedelta(seconds=1)

        # Commit to repo with de-dup
        ok, ack = self._schedule_task_from_parse(parsed, state, user_ctx)
        if ok:
            self.wakeup()
        return ok, ack

    # ─────────────────────────── Internals ─────────────────────────────

    def _find_chat_id(self, state: dict, user_ctx: Any | None) -> Any:
        # 1) most precise: present on the message metadata (telegram_input writes it)
        try:
            mid = getattr(user_ctx, "metadata", {}) or {}
            cid = mid.get("chat_id")
            if cid is not None:
                return cid
        except Exception:
            pass

        # 2) assembler may stash _chat_contexts on state (optional)
        try:
            ctxs = state.get("_chat_contexts") or set()
            if ctxs:
                return sorted(ctxs)[-1]
        except Exception:
            pass

        # 3) admin_chat_id fallback (if present in cfg inside state)
        try:
            cfg = state.get("cfg") or {}
            admin = cfg.get("admin_chat_id")
            return admin
        except Exception:
            return None

    # ─── JSONL-protecting save helpers ─────────────────────────────────

    def _meaningful_fingerprint(self, co: Any) -> str:
        """Hash only fields that actually matter; bucket due_epoch to avoid 1s churn."""
        import json
        ctx_id = getattr(co, "context_id", None)
        md = (getattr(co, "metadata", None) or {}).copy()
        # bucket due_epoch to 5s to avoid write storms from 1s drift
        due_epoch = md.get("due_epoch")
        if isinstance(due_epoch, (int, float)):
            md["due_epoch_bucket"] = int(due_epoch) // 5
        # keep only meaningful keys
        keep = {
            "status","due_epoch_bucket","due_utc_iso","rrule",
            "lease_owner","lease_epoch","send_token",
            "attempt_delivery_key","attempt_epoch","last_delivery_key",
            "task_key","telegram_chat_id","title","tz_offset_minutes"
        }
        md_trim = {k: md.get(k) for k in sorted(keep)}
        base = {
            "context_id": ctx_id,
            "stage_id": getattr(co, "stage_id", None),
            "summary": getattr(co, "summary", None),
            "metadata": md_trim,
        }
        s = repr(base).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    def _normalize_due_fields(self, co: Any) -> bool:
        """Fix due_epoch if it disagrees with due_utc_iso (>2s). Returns True if changed."""
        md = co.metadata or {}
        iso = md.get("due_utc_iso")
        ep  = md.get("due_epoch")
        if not iso or not isinstance(ep, (int, float)):
            return False
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            good = int(dt.astimezone(timezone.utc).timestamp())
        except Exception:
            return False
        if abs(int(ep) - good) > DUE_EPOCH_WIGGLE_SEC:
            md["due_epoch"] = good
            return True
        return False

    def _finalize_stuck_overdue(self, co: Any, now_epoch: int) -> bool:
        """
        Non-RRULE, scheduled, due in the past → mark sent (no new message).
        Returns True if it changed the object.
        """
        md = co.metadata or {}
        if md.get("status") != "scheduled": return False
        if md.get("rrule"): return False
        due = md.get("due_epoch")
        if not isinstance(due, (int, float)): return False
        if int(due) <= (now_epoch - OVERDUE_GRACE_SEC):
            md["status"] = "sent"
            md["sent_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
            # make the delivery key deterministic so future scans see it as finalized
            title = md.get("title") or ""
            chat_id = md.get("telegram_chat_id")
            task_key = md.get("task_key") or _mk_task_key(title, int(due), str(chat_id) if chat_id is not None else None)
            md["last_delivery_key"] = _mk_delivery_key(task_key, int(due))
            return True
        return False

    def _safe_save(self, co: Any, *, allow_churn: bool = False) -> bool:
        """
        Save with retries, but skip if nothing meaningful changed recently.
        Set allow_churn=True only for truly new objects (e.g., initial creation).
        """
        ctx_id = getattr(co, "context_id", None)
        now = _now_epoch()
        fp = self._meaningful_fingerprint(co)

        # throttle identical writes
        if not allow_churn:
            last_fp = self._fp_cache.get(ctx_id)
            last_t  = self._fp_time.get(ctx_id, 0)
            if fp == last_fp and (now - last_t) < WRITE_THROTTLE_SEC:
                return True  # treat as success; nothing to persist

        # perform the actual save with small retries
        last_err = None
        for _ in range(self._sqlite_retry_tries):
            try:
                self.repo.save(co)
                # cache on success
                self._fp_cache[ctx_id] = fp
                self._fp_time[ctx_id] = now
                return True
            except Exception as e:
                last_err = e
                time.sleep(self._sqlite_retry_ms / 1000.0)
        self.logger.error("TaskScheduler: repo.save failed after retries: %s", last_err)
        return False

    # ─── scheduling ───────────────────────────────────────────────────

    def _schedule_task_from_parse(self, parse: dict, state: dict, user_ctx: Any | None) -> Tuple[bool, str]:
        from context import ContextObject  # local import to avoid cycles

        due_local: datetime | None = parse.get("due_local")
        rrule: str | None = parse.get("rrule")

        if not (due_local or rrule):
            return False, ""

        title = _extract_title(state.get("user_text") or "", parse.get("extracted_time_text", ""))
        chat_id = self._find_chat_id(state, user_ctx)

        # due fields
        due_utc_iso = None
        due_epoch = None
        if due_local:
            due_utc = due_local.astimezone(timezone.utc)
            due_utc_iso = due_utc.isoformat().replace("+00:00", "Z")
            due_epoch = int(due_utc.timestamp())

        # idempotency key (title + due_epoch + chat_id)
        task_key = _mk_task_key(title, due_epoch, str(chat_id) if chat_id is not None else None)

        # De-dup: exact key, else near-time same-title guard
        try:
            existing = list(self.repo.query(
                lambda c: (
                    (getattr(c, "stage_id", None) == "task" or getattr(c, "semantic_label", "") in ("task", "reminder", "scheduled_task"))
                    and isinstance(getattr(c, "metadata", {}), dict)
                    and (c.metadata.get("status") in ("scheduled", "firing", "sent"))
                )
            ))
        except Exception:
            existing = []

        for c in existing:
            md = c.metadata or {}
            if md.get("task_key") == task_key and md.get("status") in ("scheduled", "firing"):
                return True, f"Already scheduled **{title}**. I’ll message you here when it’s time."

        # small ±120s guard for visually identical titles at same time
        for c in existing:
            md = c.metadata or {}
            if (md.get("title", "").strip().lower() == title.strip().lower()
                and isinstance(md.get("due_epoch"), (int, float))
                and due_epoch is not None
                and abs(float(md["due_epoch"]) - float(due_epoch)) <= 120
                and md.get("status") in ("scheduled", "firing")):
                local_str = _fmt_local(due_local) if due_local else "(recurring)"
                rel = _rel_human(due_local - datetime.now().astimezone()) if due_local else ""
                return True, f"Already scheduled **{title}** for {local_str} ({rel}). I won’t duplicate it."

        # Create new task row
        parents = []
        try:
            if user_ctx and getattr(user_ctx, "context_id", None):
                parents.append(user_ctx.context_id)
        except Exception:
            pass

        task = ContextObject.make_stage(
            "task",
            parents,
            {
                "text": title,
                "request_text": state.get("user_text") or "",
            }
        )
        task.stage_id = "task"
        task.summary = title

        now_local = datetime.now().astimezone()
        tz_off_min = int((now_local.utcoffset() or timedelta(0)).total_seconds() // 60)

        md = task.metadata
        md.update({
            "title": title,
            "status": "scheduled",
            "task_key": task_key,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "due_utc_iso": due_utc_iso,
            "due_epoch": due_epoch,
            "rrule": rrule,
            "user_id": state.get("user_id"),
            "conversation_id": state.get("conversation_id"),
            "telegram_chat_id": chat_id,
            "tz_name": now_local.tzname() or "Local",
            "tz_offset_minutes": tz_off_min,
            "source": "task_scheduler",
            "send_token": 0,  # monotonic bump on claim
            # delivery guards (populated later):
            # "attempt_delivery_key": str
            # "attempt_epoch": int
            # "last_delivery_key": str
        })
        if due_local:
            md["due_local_iso"] = due_local.isoformat()

        task.touch()
        self._safe_save(task, allow_churn=True)

        # human ACK
        if rrule and not due_local:
            ack = f"Got it — I’ve scheduled **{title}** ({rrule}). I’ll message you here when it fires."
        else:
            local_str = _fmt_local(due_local) if due_local else "(recurring)"
            rel = _rel_human(due_local - now_local) if due_local else ""
            if rrule:
                ack = f"All set — **{title}** at {local_str} ({rel}), repeating ({rrule}). I’ll message you here when it’s time."
            else:
                ack = f"All set — **{title}** at {local_str} ({rel}). I’ll message you here when it’s time."

        try:
            self.logger.info("TaskScheduler: scheduled title=%s chat=%s due_epoch=%s rrule=%s",
                             title, chat_id, due_epoch, rrule)
        except Exception:
            pass

        return True, ack

    async def _scheduler_loop(self):
        """Single background worker; never dies (catches and logs)."""
        if self._sched_event is None:
            self._sched_event = asyncio.Event()
        event = self._sched_event

        while True:
            try:
                now_epoch = _now_epoch()
                self._prune_inproc_dedupe(now_epoch)

                # 0) repair: reclaim stale 'firing' leases
                try:
                    def _is_stale_firing(c):
                        try:
                            if getattr(c, "stage_id", None) == "task":
                                md = c.metadata or {}
                                if md.get("status") == "firing":
                                    lease_epoch = int(md.get("lease_epoch", 0) or 0)
                                    return (now_epoch - lease_epoch) > LEASE_TTL_SEC
                        except Exception:
                            return False
                        return False

                    stale = list(self.repo.query(_is_stale_firing))
                    for t in stale:
                        md = t.metadata or {}
                        md["status"] = "scheduled"
                        md["due_epoch"] = now_epoch + max(5, SEND_BACKOFF_ON_FAIL // 2)
                        md["repaired_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
                        self._safe_save(t)
                except Exception as e:
                    self.logger.warning("TaskScheduler: stale-lease repair failed: %s", e)

                # 1) gather scheduled
                def _is_sched(c):
                    try:
                        if getattr(c, "stage_id", None) == "task" or getattr(c, "semantic_label", "") in ("task","reminder","scheduled_task"):
                            md = c.metadata or {}
                            return md.get("status") == "scheduled"
                    except Exception:
                        return False
                    return False

                tasks = list(self.repo.query(_is_sched))
                if len(tasks) > QUERY_BATCH_MAX:
                    tasks = tasks[:QUERY_BATCH_MAX]

                # 1a) normalize due fields + finalize ancient one-shots; save only if changed
                for t in tasks:
                    changed = False
                    changed |= self._normalize_due_fields(t)
                    changed |= self._finalize_stuck_overdue(t, now_epoch)
                    if changed:
                        self._safe_save(t)

                # 1b) compute due_now / next_epoch
                next_epoch = None
                due_now = []

                for t in tasks:
                    md = t.metadata or {}
                    if md.get("status") != "scheduled":
                        continue

                    due_epoch = md.get("due_epoch")
                    rrule = md.get("rrule")
                    tz_offset = int(md.get("tz_offset_minutes", 0))

                    # compute initial due from RRULE if missing
                    if due_epoch is None and rrule:
                        due_epoch = _next_epoch_from_rrule(rrule, now_epoch - 1, tz_offset)
                        if due_epoch is not None:
                            md["due_epoch"] = int(due_epoch)
                            md["due_utc_iso"] = datetime.fromtimestamp(due_epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                            self._safe_save(t)

                    if isinstance(due_epoch, (int, float)) and due_epoch <= now_epoch:
                        due_now.append(t)
                    elif isinstance(due_epoch, (int, float)):
                        next_epoch = due_epoch if next_epoch is None else min(next_epoch, due_epoch)

                # 2) sleep until next due (or a wakeup)
                if not due_now:
                    timeout = SLEEP_DEFAULT_SEC
                    if next_epoch is not None:
                        timeout = max(0.0, float(next_epoch - now_epoch))
                    event.clear()
                    try:
                        await asyncio.wait_for(event.wait(), timeout=timeout)
                        continue
                    except asyncio.TimeoutError:
                        pass

                # 3) fire each due task
                for t in due_now:
                    md = t.metadata or {}
                    if md.get("status") != "scheduled":
                        continue  # someone else claimed

                    title = md.get("title") or (getattr(t, "summary", "") or "Reminder")
                    chat_id = md.get("telegram_chat_id")
                    due_epoch = int(md.get("due_epoch") or now_epoch)
                    task_key = md.get("task_key") or _mk_task_key(title, due_epoch, str(chat_id) if chat_id is not None else None)
                    delivery_key = _mk_delivery_key(task_key, due_epoch)

                    # If we ever sent (or finalized) this exact due, don't send again.
                    if md.get("last_delivery_key") == delivery_key:
                        self._mark_sent_or_reschedule(t, delivered=True)
                        continue

                    # If there was an attempt for this exact due within TTL, avoid a rapid duplicate.
                    attempt_key = md.get("attempt_delivery_key")
                    attempt_epoch = int(md.get("attempt_epoch") or 0)
                    if attempt_key == delivery_key and (now_epoch - attempt_epoch) < DELIVERY_DEDUPE_TTL:
                        # push a tiny backoff to re-evaluate later (don’t spam)
                        md["due_epoch"] = now_epoch + RETRY_SKEW_SEC
                        self._safe_save(t)
                        continue

                    # in-process TTL guard (belt & suspenders)
                    if self._inproc_guard_hit(delivery_key, now_epoch):
                        md["due_epoch"] = now_epoch + RETRY_SKEW_SEC
                        self._safe_save(t)
                        continue

                    # claim with lease + token
                    send_token_new = int(md.get("send_token", 0) or 0) + 1
                    md["status"] = "firing"
                    md["lease_owner"] = self.node_id
                    md["lease_epoch"] = now_epoch
                    md["send_token"] = send_token_new

                    if not self._safe_save(t):
                        self.logger.warning("TaskScheduler: claim save failed; skipping send for %s", getattr(t, "context_id", "?"))
                        continue

                    if not self._verify_claim(t, send_token_new):
                        # couldn't verify claim; return to scheduled soon
                        md["status"] = "scheduled"
                        md["due_epoch"] = now_epoch + CLAIM_BACKOFF_SEC
                        self._safe_save(t)
                        continue

                    # Guard the attempt BEFORE sending
                    md["attempt_delivery_key"] = delivery_key
                    md["attempt_epoch"] = now_epoch
                    if not self._safe_save(t):
                        # can't persist attempt → do not send (prevents infinite dup storms)
                        md["status"] = "scheduled"
                        md["due_epoch"] = now_epoch + CLAIM_BACKOFF_SEC
                        self._safe_save(t)
                        continue

                    # Send the reminder
                    text = f"⏰ Reminder: {title}"
                    delivered = False
                    try:
                        if self.send_func and chat_id is not None:
                            res = self.send_func(chat_id, text)
                            if asyncio.iscoroutine(res):
                                await res
                            delivered = True
                        else:
                            self.logger.info("TaskScheduler: deliver (noop) chat=%s text=%s", chat_id, text)
                            delivered = True
                    except Exception as e:
                        self.logger.warning("TaskScheduler: delivery failed: %s", e)
                        delivered = False

                    if delivered:
                        self._set_inproc_guard(delivery_key, now_epoch)
                        # mark as sent & stamp last_delivery_key
                        md["last_delivery_key"] = delivery_key
                        self._mark_sent_or_reschedule(t, delivered=True)
                    else:
                        # put back with small backoff
                        md["status"] = "scheduled"
                        md["due_epoch"] = _now_epoch() + SEND_BACKOFF_ON_FAIL
                        self._safe_save(t)

            except asyncio.CancelledError:
                break
            except Exception:
                try:
                    self.logger.exception("TaskScheduler: loop crashed:\n%s", traceback.format_exc())
                except Exception:
                    pass
                await asyncio.sleep(0.5)

    # ─────────────────────────── Claim / Guard helpers ─────────────────────────

    def _verify_claim(self, task_obj: Any, expected_token: int) -> bool:
        """
        Best-effort verify that our claim persisted, by refetching the same context_id.
        If your repo doesn't support ID lookup, this uses a tiny query filter.
        """
        ctx_id = getattr(task_obj, "context_id", None)

        def _by_id_and_firing(c):
            try:
                if getattr(c, "stage_id", None) != "task":
                    return False
                if ctx_id is not None and getattr(c, "context_id", None) != ctx_id:
                    return False
                md = c.metadata or {}
                return (
                    md.get("status") == "firing"
                    and md.get("lease_owner") == self.node_id
                    and int(md.get("send_token", 0) or 0) == expected_token
                )
            except Exception:
                return False

        try:
            matches = list(self.repo.query(_by_id_and_firing))
            return len(matches) == 1
        except Exception:
            return False

    def _mark_sent_or_reschedule(self, task_obj: Any, *, delivered: bool) -> None:
        """Mark as sent, or compute next if RRULE. Best-effort save."""
        md = task_obj.metadata or {}
        md["sent_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
        rrule = md.get("rrule")

        if delivered:
            md["status"] = "sent"

        if rrule:
            nxt = _next_epoch_from_rrule(rrule, _now_epoch(), int(md.get("tz_offset_minutes", 0)))
            if nxt:
                md["status"] = "scheduled"
                md["due_epoch"] = int(nxt)
                md["due_utc_iso"] = datetime.fromtimestamp(nxt, tz=timezone.utc).isoformat().replace("+00:00","Z")
                # clear attempt guard for next cycle
                md.pop("attempt_delivery_key", None)
                md.pop("attempt_epoch", None)
        else:
            # non-recurring: clear due to avoid any future 'due' math
            md["due_epoch"] = None
            md.pop("attempt_delivery_key", None)
            md.pop("attempt_epoch", None)

        self._safe_save(task_obj)

    def _inproc_guard_hit(self, delivery_key: str, now_epoch: int) -> bool:
        last = self._recent_deliveries.get(delivery_key)
        if last is None:
            return False
        return (now_epoch - last) < DELIVERY_DEDUPE_TTL

    def _set_inproc_guard(self, delivery_key: str, now_epoch: int) -> None:
        self._recent_deliveries[delivery_key] = now_epoch

    def _prune_inproc_dedupe(self, now_epoch: int) -> None:
        if not self._recent_deliveries:
            return
        stale = [k for k, v in self._recent_deliveries.items() if (now_epoch - v) >= DELIVERY_DEDUPE_TTL]
        for k in stale:
            self._recent_deliveries.pop(k, None)


# ─────────────────────────── Helpers (module-level) ───────────────────────────

class _NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def exception(self, *a, **k): pass

def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def _mk_task_key(title: str, due_epoch: int | None, chat_id: str | None) -> str:
    base = f"{(title or '').strip().lower()}|{due_epoch or 0}|{chat_id or ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _mk_delivery_key(task_key: str, due_epoch: int) -> str:
    base = f"{task_key}|{int(due_epoch)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _fmt_local(dt: Optional[datetime]) -> str:
    if not dt:
        return "(recurring)"
    try:
        return dt.strftime("%a %b %d, %I:%M %p %Z").lstrip("0")
    except Exception:
        return dt.isoformat()

def _rel_human(delta: timedelta) -> str:
    secs = int(max(delta.total_seconds(), 0))
    parts = []
    for unit, div in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        v, secs = divmod(secs, div)
        if v: parts.append(f"{v}{unit}")
        if len(parts) == 2: break
    return " ".join(parts) or "0s"

def _pick_time_from_words(words: str) -> Optional[tuple[int,int]]:
    w = words.lower()
    if "morning" in w:   return (9, 0)
    if "noon" in w:      return (12, 0)
    if "afternoon" in w: return (15, 0)
    if "evening" in w:   return (19, 0)
    if "night" in w:     return (21, 0)
    return None

_WEEKDAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
_WD_ABBR  = {w[:3]: w for w in _WEEKDAYS}

def _parse_time_bits(text: str, now_local: datetime) -> dict:
    """
    Return {
      "due_local": datetime | None,
      "rrule": str | None,
      "matched": List[(start,end)],
      "extracted_time_text": "...",
      "confidence": float
    }
    """
    t = text.lower()
    matched_spans = []
    due = None
    rrule = None
    conf = 0.0

    # in X units
    rel = list(re.finditer(r"\bin\s+(\d+)\s*(seconds?|minutes?|hours?|days?|weeks?)\b", t))
    if rel:
        total = timedelta()
        for m in rel:
            n = int(m.group(1)); unit = m.group(2)
            if unit.startswith("second"): total += timedelta(seconds=n)
            elif unit.startswith("minute"): total += timedelta(minutes=n)
            elif unit.startswith("hour"): total += timedelta(hours=n)
            elif unit.startswith("day"): total += timedelta(days=n)
            elif unit.startswith("week"): total += timedelta(weeks=n)
            matched_spans.append(m.span())
        if total.total_seconds() > 0:
            due = now_local + total
            conf = max(conf, 0.95)

    # absolute YYYY-M-D [HH[:MM]]
    if due is None:
        for m in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2})(?::(\d{2}))?)?\b", t):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hh = int(m.group(4) or 9); mm = int(m.group(5) or 0)
            try:
                cand = now_local.replace(year=y, month=mo, day=d, hour=hh, minute=mm, second=0, microsecond=0)
            except ValueError:
                continue
            due = cand; matched_spans.append(m.span()); conf = max(conf, 0.9)
            break

    # tomorrow
    if due is None and "tomorrow" in t:
        when = (now_local + timedelta(days=1)).replace(second=0, microsecond=0)
        tod = _pick_time_from_words(t)
        tm = re.search(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
        if tm:
            h = int(tm.group(1)); m = int(tm.group(2) or 0)
            ap = (tm.group(3) or "").lower()
            if ap == "pm" and h < 12: h += 12
            if ap == "am" and h == 12: h = 0
            when = when.replace(hour=h, minute=m); matched_spans.append(tm.span())
        elif tod:
            when = when.replace(hour=tod[0], minute=tod[1])
        else:
            when = when.replace(hour=9, minute=0)
        i = t.find("tomorrow")
        matched_spans.append((i, i+len("tomorrow")))
        due = when; conf = max(conf, 0.95)

    # next/this weekday
    if due is None:
        wd = re.search(r"\b(next|this|on)?\s*(mon|tue|wed|thu|fri|sat|sun)(?:day)?\b", t)
        if wd:
            ref = now_local
            idx_now = ref.weekday()  # Mon=0
            base = _WD_ABBR.get(wd.group(2)[:3], None)
            if base:
                idx_target = _WEEKDAYS.index(base)
                delta = (idx_target - idx_now) % 7
                if wd.group(1) == "next" and delta == 0:
                    delta = 7
                when = (ref + timedelta(days=delta)).replace(second=0, microsecond=0)
                tod = _pick_time_from_words(t)
                tm = re.search(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
                if tm:
                    h = int(tm.group(1)); m = int(tm.group(2) or 0)
                    ap = (tm.group(3) or "").lower()
                    if ap == "pm" and h < 12: h += 12
                    if ap == "am" and h == 12: h = 0
                    when = when.replace(hour=h, minute=m); matched_spans.append(tm.span())
                elif tod:
                    when = when.replace(hour=tod[0], minute=tod[1])
                else:
                    when = when.replace(hour=9, minute=0)
                due = when; matched_spans.append(wd.span()); conf = max(conf, 0.9)

    # every N units
    if rrule is None:
        m = re.search(r"\bevery\s+(\d+)\s*(seconds?|minutes?|hours?|days?|weeks?)\b", t)
        if m:
            n = int(m.group(1)); unit = m.group(2)
            freq = None
            if unit.startswith("second"): freq = "SECONDLY"
            elif unit.startswith("minute"): freq = "MINUTELY"
            elif unit.startswith("hour"):  freq = "HOURLY"
            elif unit.startswith("day"):   freq = "DAILY"
            elif unit.startswith("week"):  freq = "WEEKLY"
            if freq:
                rrule = f"FREQ={freq};INTERVAL={n}"
                conf = max(conf, 0.95)

    # every weekday / daily
    if rrule is None:
        if re.search(r"\bevery\s+weekday(s)?\b", t):
            byh, bym = (9, 0)
            tm = re.search(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
            tod = _pick_time_from_words(t)
            if tm:
                h = int(tm.group(1)); m = int(tm.group(2) or 0)
                ap = (tm.group(3) or "").lower()
                if ap == "pm" and h < 12: h += 12
                if ap == "am" and h == 12: h = 0
                byh, bym = h, m
            elif tod:
                byh, bym = tod
            rrule = f"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR={byh};BYMINUTE={bym};BYSECOND=0"
            conf = max(conf, 0.95)
        elif re.search(r"\b(daily|every\s+day)\b", t):
            byh, bym = (9, 0)
            tm = re.search(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
            tod = _pick_time_from_words(t)
            if tm:
                h = int(tm.group(1)); m = int(tm.group(2) or 0)
                ap = (tm.group(3) or "").lower()
                if ap == "pm" and h < 12: h += 12
                if ap == "am" and h == 12: h = 0
                byh, bym = h, m
            elif tod:
                byh, bym = tod
            rrule = f"FREQ=DAILY;BYHOUR={byh};BYMINUTE={bym};BYSECOND=0"
            conf = max(conf, 0.9)

    # “every mon|wed|fri … [at HH:MM]”
    if rrule is None:
        days = re.findall(r"\bevery\s+(mon|tue|wed|thu|fri|sat|sun)(?:day)?\b", t)
        if days:
            byh, bym = (9, 0)
            tm = re.search(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
            tod = _pick_time_from_words(t)
            if tm:
                h = int(tm.group(1)); m = int(tm.group(2) or 0)
                ap = (tm.group(3) or "").lower()
                if ap == "pm" and h < 12: h += 12
                if ap == "am" and h == 12: h = 0
                byh, bym = h, m
            elif tod:
                byh, bym = tod
            map2 = {"mon":"MO","tue":"TU","wed":"WE","thu":"TH","fri":"FR","sat":"SA","sun":"SU"}
            bydays = ",".join(map2[d[:3]] for d in days if d[:3] in map2)
            if bydays:
                rrule = f"FREQ=WEEKLY;BYDAY={bydays};BYHOUR={byh};BYMINUTE={bym};BYSECOND=0"
                conf = max(conf, 0.92)

    # time only: at 7pm
    if (due is None) and (rrule is None):
        tm = re.search(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
        if tm:
            h = int(tm.group(1)); m = int(tm.group(2) or 0)
            ap = (tm.group(3) or "").lower()
            if ap == "pm" and h < 12: h += 12
            if ap == "am" and h == 12: h = 0
            cand = now_local.replace(hour=h, minute=m, second=0, microsecond=0)
            if cand <= now_local:
                cand = cand + timedelta(days=1)
            due = cand
            conf = max(conf, 0.85)

    extracted = ""
    if matched_spans:
        s = sorted(matched_spans)
        merged = []
        cur = list(s[0])
        for a, b in s[1:]:
            if a <= cur[1] + 1:
                cur[1] = max(cur[1], b)
            else:
                merged.append(tuple(cur)); cur = [a, b]
        merged.append(tuple(cur))
        extracted = " ".join(t[m0:m1] for (m0, m1) in merged)

    return {
        "due_local": due,
        "rrule": rrule,
        "matched": matched_spans,
        "extracted_time_text": extracted,
        "confidence": conf,
    }

def _extract_title(text: str, extracted_time_text: str) -> str:
    t = text
    # strip matched time fragments
    for frag in set(extracted_time_text.split()):
        if frag and len(frag) > 2:
            t = re.sub(re.escape(frag), " ", t, flags=re.IGNORECASE)
    # strip scaffolding
    t = re.sub(r"^\s*(please|hey|ok(ay)?|can you|could you|set|create|make)\b[:,\s]*", " ", t, flags=re.I)
    t = re.sub(r"\b(remind\s+me\s*(to|that)?|set\s+(a\s+)?(timer|alarm)\s*(for)?|alarm\s*(for)?|nudge|ping|alert\s+me\s*(to|about)?)\b[:,\s]*", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    return (t or "Reminder")[:140]

def _next_epoch_from_rrule(rrule: str, after_epoch: int, tz_offset_minutes: int = 0) -> Optional[int]:
    if not rrule:
        return None
    parts = {}
    for p in rrule.split(";"):
        if "=" in p:
            k, v = p.split("=", 1)
            parts[k.upper()] = v
    freq = parts.get("FREQ", "").upper()
    interval = int(parts.get("INTERVAL", "1") or "1")
    byhour = int(parts.get("BYHOUR", "9") or "9")
    bymin  = int(parts.get("BYMINUTE", "0") or "0")
    bysec  = int(parts.get("BYSECOND", "0") or "0")

    after_dt_utc = datetime.fromtimestamp(after_epoch, tz=timezone.utc)
    after_local  = after_dt_utc + timedelta(minutes=tz_offset_minutes)

    def _local_to_utc_epoch(dt_local: datetime) -> int:
        dt_utc = (dt_local - timedelta(minutes=tz_offset_minutes))
        return int(dt_utc.replace(tzinfo=timezone.utc).timestamp())

    if freq == "SECONDLY":
        return _local_to_utc_epoch(after_local.replace(microsecond=0) + timedelta(seconds=interval))
    if freq == "MINUTELY":
        base = after_local.replace(second=bysec, microsecond=0)
        if base <= after_local:
            base += timedelta(minutes=interval)
        return _local_to_utc_epoch(base)
    if freq == "HOURLY":
        base = after_local.replace(minute=bymin, second=bysec, microsecond=0)
        if base <= after_local:
            base += timedelta(hours=interval)
        return _local_to_utc_epoch(base)
    if freq == "DAILY":
        base = after_local.replace(hour=byhour, minute=bymin, second=bysec, microsecond=0)
        if base <= after_local:
            base += timedelta(days=interval)
        return _local_to_utc_epoch(base)
    if freq == "WEEKLY":
        byday_str = parts.get("BYDAY", "MO,TU,WE,TH,FR,SA,SU")
        day_map = {"MO":0,"TU":1,"WE":2,"TH":3,"FR":4,"SA":5,"SU":6}
        targets = [day_map[d] for d in byday_str.split(",") if d in day_map] or list(day_map.values())
        start = after_local
        # search up to interval * 2 weeks ahead
        for _ in range(14):
            for d in sorted(targets):
                cand = start + timedelta(days=(d - start.weekday()) % 7)
                cand = cand.replace(hour=byhour, minute=bymin, second=bysec, microsecond=0)
                if cand > after_local:
                    return _local_to_utc_epoch(cand)
            start = start + timedelta(days=7*max(1, interval))
        return None
    return None
