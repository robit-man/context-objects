"""
telegram_input.py

Telegram I/O wrapper for the Assembler.

Features
────────
• Feeds user text into `asm.run_with_meta_context`
• Live “🛠️ Processing…” message that updates once per stage
• Replaces the placeholder with the full answer (chunked if >4 000 chars)
• Streams exactly ONE .ogg voice reply
• Accepts **voice notes** – they’re run through Whisper, then processed as text
• Maintains a SQLite-backed user registry for DM’ing users by username
• **Pins** each bot reply for cross-bot notification, then **un-pins** it
• Listens for **pin** events to ingest other bots’ outputs as context
"""

from __future__ import annotations
import asyncio
import base64

import os
import queue as _queue
import tempfile
import subprocess
import uuid
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from dotenv import load_dotenv
import json
from telegram import InputFile
from telegram import Update, Bot, error as tg_error
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, CommandHandler
from telegram.request import HTTPXRequest
from telegram.constants import ChatAction
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    CallbackQueryHandler,
    ConversationHandler,
    MessageHandler,
    filters as _flt,
    CommandHandler,
)
from context import HybridContextRepository, ContextObject, _locked, sanitize_jsonl
from tts_service import TTSManager
from user_registry import _REG
from group_registry import _GREG
from assembler import Assembler
import whisper

_WHISPER = whisper.load_model("base")  # load once

CAPTURE_DIR = Path(__file__).parent / "captures"
CAPTURE_DIR.mkdir(exist_ok=True, parents=True)

# basic logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO
)

# Silence all the HTTP “Request: POST … getUpdates” lines
logging.getLogger("httpx").setLevel(logging.WARNING)
# (Optionally also silence any lower‑level PTB logs)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# chat_id → dedicated Assembler instance
assemblers: dict[int, Any] = {}

# chat_id → pending inference queue
_pending_pin: dict[tuple[int,int], asyncio.Queue[tuple[str,int]]] = {}
_running_pin: dict[tuple[int,int], asyncio.Task] = {}
_pending: dict[tuple[int,int], asyncio.Queue[Tuple[str,int]]] = {}

# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def log_message(msg: str, category: str="INFO"):
    COLOR_RESET   = "\033[0m"
    COLOR_INFO    = "\033[94m"
    COLOR_SUCCESS = "\033[92m"
    COLOR_WARNING = "\033[93m"
    COLOR_ERROR   = "\033[91m"
    COLOR_PROCESS = "\033[96m"
    color = {
        "INFO":    COLOR_INFO,
        "SUCCESS": COLOR_SUCCESS,
        "WARNING": COLOR_WARNING,
        "ERROR":   COLOR_ERROR,
        "PROCESS": COLOR_PROCESS,
    }.get(category.upper(), COLOR_RESET)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}[{ts}] {category.upper()}: {msg}{COLOR_RESET}")

async def _delayed_unpin(bot, chat_id: int, message_id: int, delay: float = 2.0):
    await asyncio.sleep(delay)
    try:
        await bot.unpin_chat_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass

async def _send_long_text_async(
    bot,
    chat_id: int,
    text: str,
    *,
    chunk_size: int = 3800,
    reply_to: Optional[int] = None
) -> None:
    if not text.strip():
        return
    import re
    paras = text.split("\n\n")
    buffer, chunks = "", []
    for para in paras:
        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            if len(para) <= chunk_size:
                buffer = para
            else:
                for sent in re.split(r"(?<=[\.\?\!])\s+", para):
                    if len(buffer) + len(sent) + 1 <= chunk_size:
                        buffer = (buffer + " " + sent).strip()
                    else:
                        if buffer:
                            chunks.append(buffer)
                        buffer = ""
                        for i in range(0, len(sent), chunk_size):
                            chunks.append(sent[i : i + chunk_size])
    if buffer:
        chunks.append(buffer)

    for part in chunks:
        msg = await bot.send_message(
            chat_id=chat_id,
            text=part,
            reply_to_message_id=reply_to
        )
        try:
            await bot.pin_chat_message(
                chat_id=chat_id,
                message_id=msg.message_id,
                disable_notification=True
            )
            asyncio.create_task(
                _delayed_unpin(bot, chat_id, msg.message_id)
            )
        except Exception as e:
            logger.warning(f"pin scheduling failed for msg {msg.message_id}: {e}")

def make_per_chat_repo(chat_id: int, archive_max_mb: float = 10.0) -> HybridContextRepository:
    from pathlib import Path
    import shutil
    from context import sanitize_jsonl

    # 1) Ensure our storage directory exists
    base = Path("context_repos")
    base.mkdir(parents=True, exist_ok=True)

    # 2) Filenames for this chat
    filename_jsonl    = f"context_{chat_id}.jsonl"
    filename_db       = f"context_{chat_id}.db"
    filename_corrupt  = f"{filename_jsonl}.corrupt"

    # 3) Paths in context_repos/
    jsonl_path   = base / filename_jsonl
    sqlite_path  = base / filename_db
    corrupt_path = base / filename_corrupt

    # 4) Look in CWD for any existing files and move them in
    cwd_jsonl    = Path.cwd() / filename_jsonl
    cwd_db       = Path.cwd() / filename_db
    cwd_corrupt  = Path.cwd() / filename_corrupt

    if cwd_jsonl.exists():
        if jsonl_path.exists():
            jsonl_path.unlink()
        shutil.move(str(cwd_jsonl), str(jsonl_path))

    if cwd_db.exists():
        if sqlite_path.exists():
            sqlite_path.unlink()
        shutil.move(str(cwd_db), str(sqlite_path))

    if cwd_corrupt.exists():
        if corrupt_path.exists():
            corrupt_path.unlink()
        shutil.move(str(cwd_corrupt), str(corrupt_path))

    # 5) Initialize an empty JSONL if needed
    sanitize_jsonl(str(jsonl_path))

    # 6) Create and return the repository
    return HybridContextRepository(
        jsonl_path=str(jsonl_path),
        sqlite_path=str(sqlite_path),
        archive_max_mb=archive_max_mb,
    )

async def list_pins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /pins  →  return every pinned message we’ve logged for THIS chat,
    formatted in one (or several) triple-back-tick blocks.
    """
    chat_id = update.effective_chat.id

    chat_asm = assemblers.get(chat_id)
    if chat_asm is None:
        await update.message.reply_text("⚠️ I don’t have any history for this chat yet.")
        return

    # fetch all ContextObjects we wrote with semantic_label == "bot_pinned"
    pins = sorted(
        chat_asm.repo.query(lambda c: c.semantic_label == "telegram_pinned"),  # ← fixed
        key=lambda c: c.timestamp                                             # ← safer sort
    )
    CHUNK = 3700
    for i in range(0, len(lines := [
        f"[{p.metadata.get('telegram_message_id')}] {p.metadata.get('author')}: {p.metadata.get('text', '')}"
        for p in pins
    ]), CHUNK):
        block = "```\n" + "\n".join(lines[i:i+CHUNK]) + "\n```"
        await update.message.reply_text(block, parse_mode="Markdown")

    if not pins:
        await update.message.reply_text("No pinned messages recorded so far.")
        return

    # build one big list → chunk if >4 000 chars (Telegram hard limit)
    lines = [
        f"[{p.metadata.get('telegram_message_id')}] "
        f"{p.metadata.get('author')}: {p.metadata.get('text', '')}"
        for p in pins
    ]
    blob = "```\n" + "\n".join(lines) + "\n```"

    if len(blob) < 4000:
        await update.message.reply_text(blob, parse_mode="Markdown")
    else:
        # reuse existing helper to split into safe chunks
        await _send_long_text_async(
            context.bot,
            chat_id,
            blob,
            chunk_size=3800,
            reply_to=update.message.message_id
        )


async def set_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /debug true|false
    Only the configured admin (via /setadmin) may run this.
    Toggles config.json["debug"].
    """
    user_id = update.effective_user.id
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}

    admin_id = cfg.get("admin_chat_id")
    if user_id != admin_id:
        await update.message.reply_text("❌ You are not authorized to use /debug.")
        return

    if not context.args or context.args[0].lower() not in ("true", "false"):
        await update.message.reply_text("Usage: /debug true|false")
        return

    enabled = context.args[0].lower() == "true"
    cfg["debug"] = enabled
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    await update.message.reply_text(f"✔️ Debugging {'enabled' if enabled else 'disabled'}.")

async def set_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    # load config.json
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}

    current_admin = cfg.get("admin_chat_id")
    # if an admin is already set, only they may reassign
    if current_admin is not None and user_id != current_admin:
        await update.message.reply_text("❌ You are not authorized to set admin.")
        return

    # set new admin
    cfg["admin_chat_id"] = chat_id
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    await update.message.reply_text(f"✅ This chat ({chat_id}) is now the admin for alerts.")

async def set_dm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /dm true|false
    Only the admin (from config.json["admin_chat_id"]) may run this.
    Enables or disables DMs by setting config.json["allow_private"].
    """
    user_id = update.effective_user.id
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    admin_id = cfg.get("admin_chat_id")
    if user_id != admin_id:
        await update.message.reply_text("❌ You are not authorized to use /dm.")
        return

    if not context.args or context.args[0].lower() not in ("true", "false"):
        await update.message.reply_text("Usage: /dm true|false")
        return

    enabled = context.args[0].lower() == "true"
    cfg["allow_private"] = enabled
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    await update.message.reply_text(
        f"✔️ DMs have been {'enabled' if enabled else 'disabled'}."
    )

async def blacklist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /blacklist @username
    Only the configured admin (via /setadmin) may run this.
    Adds the target user’s Telegram ID to config.json["blacklist_user_ids"].
    """
    user = update.effective_user
    # load config.json
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}

    admin_id = cfg.get("admin_chat_id")
    # only admin can blacklist
    if user.id != admin_id:
        await update.message.reply_text("❌ You are not authorized to use /blacklist.")
        return

    # require @username argument
    if not context.args or not context.args[0].startswith("@"):
        await update.message.reply_text("Usage: /blacklist @username")
        return

    target_username = context.args[0].lstrip("@").lower()
    # resolve username → user_id
    from user_registry import _REG
    target_id = _REG.id_for(target_username)
    if not target_id:
        await update.message.reply_text(f"❌ User @{target_username} not found.")
        return

    # update blacklist_user_ids in config
    bl = set(cfg.get("blacklist_user_ids", []))
    if target_id in bl:
        await update.message.reply_text(f"⚠️ @{target_username} is already blacklisted.")
        return

    bl.add(target_id)
    cfg["blacklist_user_ids"] = sorted(bl)
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    await update.message.reply_text(f"✔️ Blacklisted @{target_username} (ID: {target_id}).")


async def config_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /config get *            → show entire config.json
    /config get <key>        → show one entry
    /config set <key> <json> → set one entry
    Only the configured admin_chat_id in config.json may run these.
    """
    user_id = update.effective_user.id

    # Load existing config (or empty dict)
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}

    admin_id = cfg.get("admin_chat_id")
    if user_id != admin_id:
        await update.message.reply_text("❌ You are not authorized to use /config.")
        return

    args = context.args or []
    if not args or args[0] not in ("get", "set"):
        await update.message.reply_text(
            "Usage:\n"
            "/config get *\n"
            "/config get <key>\n"
            "/config set <key> <json-literal>"
        )
        return

    action = args[0]

    # ── GET ───────────────────────────────────────────────────────────────
    if action == "get":
        # no further args or wildcard → full config
        if len(args) == 1 or (len(args) == 2 and args[1] in ("*", "all")):
            pretty = json.dumps(cfg, indent=2)
            # wrap in triple-backticks so Telegram preserves formatting
            await update.message.reply_text(f"```\n{pretty}\n```", parse_mode="Markdown")
            return

        # single key
        if len(args) == 2:
            key = args[1]
            val = cfg.get(key, None)
            await update.message.reply_text(
                f"```json\n{json.dumps({key: val}, indent=2)}\n```",
                parse_mode="Markdown"
            )
            return

        # too many args
        await update.message.reply_text("Usage: /config get *  OR  /config get <key>")
        return

    # ── SET ───────────────────────────────────────────────────────────────
    # args[0] == "set"
    if len(args) < 3:
        await update.message.reply_text("Usage: /config set <key> <json-literal>")
        return

    key = args[1]
    raw = " ".join(args[2:])
    try:
        new_val = json.loads(raw)
    except json.JSONDecodeError:
        new_val = raw  # treat as string

    cfg[key] = new_val
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    await update.message.reply_text(
        f"✅ Set `{key}` =\n```json\n{json.dumps(new_val, indent=2)}\n```",
        parse_mode="Markdown"
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    err = context.error
    if isinstance(err, tg_error.Conflict):
        logger.warning("Ignored Conflict error in polling")
        return
    raise err

def notify_admin(text: str):
    """
    Send a Markdown‐formatted alert to the admin_chat_id stored in config.json,
    using the BOT_TOKEN from .env. Schedules or runs the coroutine as needed.
    """
    try:
        # Load token from .env
        load_dotenv()
        token = os.getenv("BOT_TOKEN")
        if not token:
            logger.warning("notify_admin skipped: BOT_TOKEN not set in .env")
            return

        # Load admin_chat_id from config.json
        cfg_path = "config.json"
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            logger.warning("notify_admin skipped: config.json not found")
            return

        admin = cfg.get("admin_chat_id")
        if not admin:
            logger.warning("notify_admin skipped: admin_chat_id not set in config.json")
            return

        # Build coroutine
        bot = Bot(token=token)
        send_coro = bot.send_message(
            chat_id=admin,
            text=text,
            parse_mode="Markdown"
        )

        # If we're already inside an asyncio loop, schedule it;
        # otherwise run it synchronously.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # no loop -> safe to run
            asyncio.run(send_coro)
        else:
            # loop exists -> schedule
            loop.create_task(send_coro)

    except Exception as e:
        logger.error(f"notify_admin failed: {e}")

def _make_status_cb(
    loop: asyncio.AbstractEventLoop,
    bot,
    chat_id: int,
    msg_id: Optional[int],
    *,
    max_lines: int = 30,
    min_interval: float = 5,
):
    # No‐ops on Windows
    if os.name == "nt":
        def _noop_status(stage: str, output: Any):
            return
        def _noop_stop():
            return
        return _noop_status, _noop_stop

    from typing import List, Any
    global assemblers  # map of chat_id → Assembler instance
    history: List[str] = []
    last_edit_at: float = 0.0
    pending_handle = None
    disabled = False

    async def _do_edit():
        nonlocal last_edit_at, pending_handle
        if disabled or msg_id is None:
            return
        header = f"`context_{chat_id}.jsonl` updating...\n"
        body   = "\n".join(history[-max_lines:])
        text   = header + body
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=text,
                parse_mode="Markdown"
            )
        except:
            pass
        last_edit_at = loop.time()
        pending_handle = None

    def _schedule_edit():
        nonlocal pending_handle
        if disabled or pending_handle:
            return
        delay = max(min_interval - (loop.time() - last_edit_at), 0.0)
        if delay <= 0:
            asyncio.run_coroutine_threadsafe(_do_edit(), loop)
        else:
            pending_handle = loop.call_later(
                delay,
                lambda: asyncio.run_coroutine_threadsafe(_do_edit(), loop)
            )

    def status_cb(stage: str, output: Any):
        nonlocal history
        if disabled:
            return

        # 1) update the pinned‐message UI
        snippet = str(output).replace("\n", " ")
        if len(snippet) > 1000:
            snippet = snippet[:997] + "…"
        history.append(f"• *{stage}*: {snippet}")
        _schedule_edit()

        # 2) only send voice when this stage is in tts_live_stages
        chat_asm = assemblers.get(chat_id)
        live_stages = getattr(chat_asm, "tts_telegram_stages", set()) if chat_asm else set()
        if stage not in live_stages:
            return

        # 3) drain any queued .ogg files and send them immediately
        while True:
            try:
                ogg_path = chat_asm.tts._ogg_q.get_nowait()
            except _queue.Empty:
                break
            else:
                def _send(ogg=ogg_path):
                    try:
                        with open(ogg, "rb") as vf:
                            asyncio.create_task(
                                bot.send_voice(
                                    chat_id=chat_id,
                                    voice=ogg_path,            # pass the path, not an open handle
                                    reply_to_message_id=msg_id
                                )
                            )
                    except Exception:
                        pass
                loop.call_soon_threadsafe(_send)

    def stop_cb():
        nonlocal disabled, pending_handle
        disabled = True
        if pending_handle:
            pending_handle.cancel()

    return status_cb, stop_cb


async def _start_runner_for_pin(
    chat_asm, bot, chat_id: int,
    request_text: str, reply_to_id: int
):
    loop = asyncio.get_event_loop()
    status_cb, stop_status = _make_status_cb(loop, bot, chat_id, None)
    try:
        live_sink = (lambda token: chat_asm.tts.enqueue(token)) \
                    if chat_asm.tts and hasattr(chat_asm.tts, "enqueue") else None

        final = await chat_asm.run_with_meta_context(
            request_text,
            status_cb,
            on_token=live_sink,
        )
        await asyncio.to_thread(chat_asm.tts._file_q.join)
    except Exception:
        final = ""
    stop_status()
    if not (final or "").strip():
        return

    await bot.send_message(chat_id=chat_id, text=final, reply_to_message_id=reply_to_id)
    chat_asm.tts.enqueue(final)
    await asyncio.to_thread(chat_asm.tts._file_q.join)

    oggs = []
    while True:
        try:
            p = chat_asm.tts._ogg_q.get_nowait()
            if os.path.getsize(p) > 0:
                oggs.append(p)
        except _queue.Empty:
            break

    if oggs:
        if len(oggs) == 1:
            with open(oggs[0], "rb") as vf:
                await bot.send_voice(chat_id, vf, reply_to_message_id=reply_to_id)
        else:
            combined = os.path.join(tempfile.gettempdir(), f"combined_{uuid.uuid4().hex}.ogg")
            ins = sum([["-i", p] for p in oggs], [])
            streams = "".join(f"[{i}:a]" for i in range(len(oggs)))
            filt = f"{streams}concat=n={len(oggs)}:v=0:a=1,aresample=48000"
            subprocess.run(
                [
                "ffmpeg", "-y", "-loglevel", "error",
                *ins,
                "-filter_complex", filt,
                "-c:a", "libopus",
                "-b:a", "48k",
                combined
                ],
                check=True
            )
            with open(combined, "rb") as vf:
                await bot.send_voice(chat_id, vf, reply_to_message_id=reply_to_id)

async def _on_pin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle any message-pin event (from human or bot).

    • Saves a ContextObject so the Assembler can reason over the pin later.
    • Shows a one-liner ASCII block in the console for quick visibility.
    • Optionally kicks off an inference if filter_callback() says “yes”.
    """
    # ── 0) Quick sanity: must contain a pinned_message ────────────────────
    msg = update.message
    if not msg or not msg.pinned_message:
        return

    pinned   = msg.pinned_message
    chat_id  = update.effective_chat.id
    bot_id   = context.bot.id

    # Ignore if **this** bot is pinning its *own* content
    if pinned.from_user and pinned.from_user.id == bot_id:
        logger.debug("Ignoring pin of my own message %s", pinned.message_id)
        return

    # ── 1) Gather pin metadata ───────────────────────────────────────────
    pinner = (msg.from_user.username  or str(msg.from_user.id)      ).lower()
    author = (pinned.from_user.username or str(pinned.from_user.id) ).lower() \
             if pinned.from_user else "unknown"
    text   = pinned.text or pinned.caption or "<non-text content>"

    # ── 1a) VISUAL LOG for humans ────────────────────────────────────────
    pin_block = (
        "\n╔════════════════════════════════════ PIN DETECTED ═════════════════════════════╗\n"
        f"║ Chat   : {chat_id}\n"
        f"║ Pinner : {pinner}\n"
        f"║ Author : {author}\n"
        f"║ Msg ID : {pinned.message_id}\n"
        "╟───────────────────────────────────────────────────────────────────────────────╢\n"
        f"║ {text}\n"
        "╚═══════════════════════════════════════════════════════════════════════════════╝"
    )
    logger.info(pin_block)

    # ── 2) Ensure the per-chat Assembler exists ──────────────────────────
    chat_asm = assemblers.get(chat_id)
    if not chat_asm and _default_asm:
        repo = make_per_chat_repo(chat_id)
        tts  = TTSManager(
            logger=_default_asm.tts.log,
            cfg=_default_asm.cfg,
            audio_service=None
        )
        tts.set_mode("file")
        chat_asm = Assembler(
            context_path=f"context_{chat_id}.jsonl",
            config_path="config.json",
            lookback_minutes=60,
            top_k=5,
            tts_manager=tts,
            repo=repo,
        )
        assemblers[chat_id] = chat_asm
    if chat_asm:
        chat_asm._chat_contexts.add(chat_id)

    # ── 3) Persist the pin as a ContextObject ────────────────────────────
    seg = ContextObject.make_segment(
        semantic_label="telegram_pinned",
        content_refs=[],
        tags=[
            "pinned",
            f"pinner:{pinner}",
            f"author:{author}",
            f"msgid:{pinned.message_id}"
        ],
        metadata={
            "pinner":              pinner,
            "author":              author,
            "text":                text,
            "telegram_message_id": pinned.message_id,
            "pinner_is_bot":       msg.from_user.is_bot,
            "author_is_bot":       pinned.from_user.is_bot if pinned.from_user else None,
        }
    )
    seg.summary  = text
    seg.stage_id = "telegram_pinned"
    seg.touch()
    if chat_asm:
        chat_asm.repo.save(seg)

    # ── 4) Should we react to the content of the pin? ────────────────────
    if not chat_asm:
        return                               # no Assembler available

    fake_input = f"{author}: {text}"
    try:
        wants_reply = chat_asm.filter_callback(fake_input)
    except Exception:
        wants_reply = False

    if not wants_reply:
        return

    # ── 5) Queue / launch an inference run (one at a time per chat) ──────
    key   = (chat_id, bot_id)
    queue = _pending_pin.setdefault(key, asyncio.Queue())

    if (prev := _running_pin.get(key)) and not prev.done():
        await queue.put((fake_input, pinned.message_id))
    else:
        _running_pin[key] = asyncio.create_task(
            _start_runner_for_pin(
                chat_asm,
                context.bot,
                chat_id,
                fake_input,
                pinned.message_id
            )
        )

def telegram_input(asm):
    global _default_asm
    _default_asm = asm

    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing BOT_TOKEN env var")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    req = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app = ApplicationBuilder().token(token).request(req).build()

    try:
        from assembler import Assembler
    except Exception:
        tb = traceback.format_exc()
        log_message("assembler import failed, inference disabled", "ERROR")
        notify_admin(f"⚠️ *assembler.py import failed:*\n```{tb[:1500]}```")
        Assembler = None

    app.add_handler(
        MessageHandler(filters.StatusUpdate.PINNED_MESSAGE, _on_pin),
        group=0
    )    
    app.add_handler(CommandHandler("setadmin", set_admin), group=0)
    app.add_handler(CommandHandler("config", config_handler), group=0)
    app.add_handler(CommandHandler("blacklist", blacklist), group=0)
    app.add_handler(CommandHandler("dm", set_dm), group=0)
    app.add_handler(CommandHandler("debug", set_debug), group=0)    
    app.add_handler(CommandHandler("pins", list_pins), group=0)

    app.add_error_handler(error_handler)

    if Assembler:
        async def _handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """
            Unified handler for text, voice, photo, document, etc.
            • Saves incoming photos to CAPTURE_DIR
            • Creates a ContextObject segment for every message
            • Queues the request into the per-chat Assembler runner
            """
            import json, tempfile, subprocess, os, asyncio, uuid
            from datetime import datetime
            from telegram.constants import ChatAction
            from context import ContextObject
            from tts_service import TTSManager
            from user_registry import _REG
            from group_registry import _GREG

            bot       = context.bot
            chat_id   = update.effective_chat.id
            chat_type = update.effective_chat.type
            bot_name  = bot.username.lower()

            
            # ── Config flags ──────────────────────────────────────────────────
            try:
                with open("config.json", "r") as f:
                    cfg = json.load(f)
            except FileNotFoundError:
                cfg = {}
            group_only    = cfg.get("group_only", False)
            allow_private = cfg.get("allow_private", True)

            if chat_type == "private" and (group_only or not allow_private):
                return

            # ── Extract the Telegram message object ───────────────────────────
            msg = (
                update.message
                or update.edited_message
                or update.channel_post
                or update.edited_channel_post
            )
            if not msg:
                return
            ''
            # ── Who sent it? ─────────────────────────────────────────────────
            user = update.effective_user

            # ── Build metadata (needed by the photo/document handlers) ───────
            metadata = {
                "chat_id":       chat_id,
                "from_user_id":  user.id,
                "from_username": user.username,
                "message_id":    msg.message_id,
                "text":          msg.text or msg.caption or "",
            }

            # ── Identify “kind” and pull raw content / files ──────────────────
            kind, data = "other", None
            image_paths: list[str] = []

            if msg.text:
                kind, data = "text", msg.text

            elif msg.photo:
                kind = "photo"
                # only take the largest size, download and save to disk
                p = msg.photo[-1]
                file = await bot.get_file(p.file_id)
                b = await file.download_as_bytearray()
                img_bytes = bytes(b)
                # save to disk so we can reference it in context
                filename = CAPTURE_DIR / f"{chat_id}_{msg.message_id}_{p.file_unique_id}.jpg"
                with open(filename, "wb") as f_img:
                    f_img.write(img_bytes)
                image_paths.append(str(filename))
                # also stash raw bytes for inline injection
                metadata.setdefault("images", []).append(img_bytes)
                data = f"{len(image_paths)} image(s)"

            elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image"):
                kind = "photo"
                # same for image‐typed documents
                file = await bot.get_file(msg.document.file_id)
                b = await file.download_as_bytearray()
                img_bytes = bytes(b)
                filename = CAPTURE_DIR / f"{chat_id}_{msg.message_id}_{msg.document.file_unique_id}.jpg"
                with open(filename, "wb") as f_img:
                    f_img.write(img_bytes)
                image_paths.append(str(filename))
                metadata.setdefault("images", []).append(img_bytes)
                data = f"{len(image_paths)} image(s)"

            elif msg.document:
                kind, data = "document", msg.document.file_id
            elif msg.sticker:
                kind, data = "sticker", msg.sticker.file_unique_id
            elif msg.voice:
                kind, data = "voice", msg.voice.file_id
            elif msg.video:
                kind, data = "video", msg.video.file_id
            elif msg.location:
                loc  = msg.location
                kind = "location"
                data = f"{loc.latitude},{loc.longitude}"
            elif msg.poll:
                kind, data = "poll", msg.poll.question

            # ── Timestamp & logging ───────────────────────────────────────────
            now = datetime.utcnow().isoformat() + "Z"
            logger.info(
                f"[{now}] Incoming update — chat_id={chat_id} kind={kind} data={data!r}"
            )

            # ── Lazy-init per-chat Assembler & utilities ──────────────────────
            chat_asm = assemblers.get(chat_id)
            if chat_asm is None:
                import sqlite3, time

                # Retry if the SQLite DB is locked (e.g. after a hard kill)
                for attempt in range(5):
                    try:
                        repo = make_per_chat_repo(chat_id)
                        break
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e):
                            time.sleep(0.1)
                            continue
                        raise
                else:
                    # All retries failed → fall back to JSON-only store
                    from context import ContextRepository, sanitize_jsonl
                    path = f"context_{chat_id}.jsonl"
                    sanitize_jsonl(path)
                    repo = ContextRepository(jsonl_path=path)

                tts = TTSManager(logger=asm.tts.log, cfg=asm.cfg, audio_service=None)
                tts.set_mode("file")
                chat_asm = Assembler(
                    context_path=f"context_{chat_id}.jsonl",
                    config_path="config.json",
                    lookback_minutes=60,
                    top_k=5,
                    tts_manager=tts,
                    repo=repo,
                )
                assemblers[chat_id] = chat_asm

                chat_asm.tts_telegram_stages = set(chat_asm.tts_live_stages)


            chat_asm._chat_contexts.add(chat_id)

            # ── User & group registries ───────────────────────────────────────
            user = update.effective_user
            if user and user.username:
                _REG.add(user.username, user.id)
            if chat_type in ("group", "supergroup", "channel"):
                title = update.effective_chat.title or f"chat_{chat_id}"
                _GREG.add(title, chat_id)

            # ── Build tags & define metadata up-front ───────────────────────
            metadata = {
                "chat_id":       chat_id,
                "from_user_id":  user.id,
                "from_username": user.username,
                "message_id":    msg.message_id,
                "text":          msg.text or msg.caption or "",
            }

            tags = [
                "telegram_update",
                kind,
                "group" if chat_type in ("group", "supergroup") else "private",
                f"user:{user.username or user.id}",
            ]

            # ── Record reply-to info if applicable ──────────────────────────
            if msg.reply_to_message:
                reply = msg.reply_to_message
                rt_user = reply.from_user.username or str(reply.from_user.id)
                rt_text = reply.text or reply.caption or ""
                # add to metadata
                metadata["reply_to_username"]     = rt_user
                metadata["reply_to_message_text"] = rt_text
                metadata["reply_to_message_id"]   = reply.message_id
                # add tags
                tags.append(f"reply_to_user:{rt_user}")
                tags.append("reply")
                if reply.from_user.id == bot.id:
                    tags.append("reply_to_bot")

            sender = user.username or user.first_name or str(user.id)
            metadata["sender"] = sender

            # ── Create & save ContextObject segment ─────────────────────────
            seg = ContextObject.make_segment(
                semantic_label=f"tg_{kind}",
                content_refs=[],
                tags=tags,
                metadata=metadata,
            )
            # include reply chain in summary if present
            if metadata.get("reply_to_username"):
                seg.summary = (
                    f"{sender} replied to {metadata['reply_to_username']}: "
                    f"{metadata['text']}"
                )
            else:
                seg.summary = f"{sender}: {metadata['text']}"
            seg.stage_id = "telegram_update"
            seg.touch()
            chat_asm.repo.save(seg)

            # ── If we have photos, store paths & notify user ──────────────────
            if image_paths:
                metadata["image_paths"] = image_paths
                try:
                    with open("config.json","r") as _f:
                        _cfg = json.load(_f)
                except FileNotFoundError:
                    _cfg = {}
                debug_enabled = _cfg.get("debug", False)

                if debug_enabled:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "🖼️ Saved image(s) to disk:\n"
                            + "\n✅".join(f"- `{p}`" for p in image_paths)
                            + "\n\nYou can now run `/analyze_image <path>` to inspect any of these."
                        ),
                        reply_to_message_id=msg.message_id,
                        parse_mode="Markdown",
                    )

                else:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "🖼️ Processing Image..."
                        ),
                        reply_to_message_id=msg.message_id,
                        parse_mode="Markdown",
                    )


            if msg.reply_to_message:
                metadata["reply_to_message_id"] = msg.reply_to_message.message_id
                if msg.reply_to_message.from_user.id == bot.id:
                    metadata["in_reply_to_bot_message_id"] = msg.reply_to_message.message_id
           
           
            sender = user.username or user.first_name or str(user.id)
            metadata["sender"] = sender
           
            # ── Create & save ContextObject segment ───────────────────────────
            seg = ContextObject.make_segment(
                semantic_label=f"tg_{kind}",
                content_refs=[],
                tags=tags,
                metadata=metadata,
            )
            # prefix the summary with the sender’s name
            seg.summary = f"{sender}: {metadata['text']}"
            seg.stage_id = "telegram_update"
            seg.touch()
            chat_asm.repo.save(seg)

            # ── Voice-note transcription (Whisper) ────────────────────────────
            text = (msg.text or "").strip()
            if msg.voice and not text:
                try:
                    raw_ogg = tempfile.mktemp(suffix=".oga")
                    vf      = await bot.get_file(msg.voice.file_id)
                    await vf.download_to_drive(raw_ogg)
                    wav     = raw_ogg + ".wav"
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_ogg, "-ac", "1", "-ar", "16000", wav],
                        check=True,
                    )
                    result = _WHISPER.transcribe(wav, language="en")
                    text   = result.get("text", "").strip() or text
                except Exception as ex:
                    await bot.send_message(
                        chat_id, f"❌ Voice note error: {ex}", reply_to_message_id=msg.message_id
                    )
                finally:
                    for p in (locals().get("raw_ogg"), locals().get("wav")):
                        if p and os.path.exists(p):
                            try:
                                os.unlink(p)
                            except:
                                pass
            # ── Decide if we should respond / run inference ────────────────────
            sender = user.username or user.first_name or str(user.id)
            if image_paths:
                user_text = f"{sender} sent image(s): " + " ".join(image_paths)
            else:
                user_text = text or ""

            trigger_id = msg.message_id

            # Determine whether to run inference for image-only or mention events
            wants_reply = False
            try:
                wants_reply = await asyncio.to_thread(chat_asm.filter_callback, user_text)
            except Exception:
                wants_reply = False

            mention_me = any(
                ent.type == "mention"
                and msg.text
                and msg.text[ent.offset : ent.offset + ent.length].lstrip("@").lower() == bot_name
                for ent in (msg.entities or [])
            )

            do_infer = (
                chat_type == "private"
                or msg.voice
                or wants_reply
                or mention_me
                or (msg.reply_to_message and msg.reply_to_message.from_user.id == bot.id)
            )
            if not do_infer:
                return

            # ── Queue / run the Assembler for this chat/user pair ─────────────
            key   = (chat_id, user.id)
            queue = _pending.setdefault(key, asyncio.Queue())


            if (prev := running.get(key)) and not prev.done():
                await bot.send_message(
                    chat_id,
                    "⚠️ I’m still working on your previous request—I’ll handle this one next.",
                    reply_to_message_id=trigger_id,
                )
                await queue.put((user_text, trigger_id))
                return
            
            # ── Runner that stages “processing…” then delivers text+voice ─────────
            async def start_runner(request_text: str, reply_to_id: int):
                # Load debug flag
                try:
                    with open("config.json", "r") as _f:
                        _cfg = json.load(_f)
                except FileNotFoundError:
                    _cfg = {}
                debug_enabled = _cfg.get("debug", False)

                placeholder_id = None
                typing_task    = None

                # 1) Set up UI‐status editing (if debug) or a no‐op
                if debug_enabled:
                    placeholder = await bot.send_message(
                        chat_id=chat_id,
                        text="🛠️ Processing…",
                        reply_to_message_id=reply_to_id
                    )
                    placeholder_id = placeholder.message_id
                    orig_status, stop_status = _make_status_cb(
                        loop, bot, chat_id, placeholder_id
                    )
                else:
                    # no UI edits in non-debug mode
                    orig_status = lambda stage, info=None: None
                    stop_status = lambda: None

                    # still show “typing…” every few seconds
                    async def _typing_loop():
                        try:
                            while True:
                                await bot.send_chat_action(
                                    chat_id=chat_id,
                                    action=ChatAction.TYPING
                                )
                                await asyncio.sleep(4)
                        except asyncio.CancelledError:
                            return

                    typing_task = loop.create_task(_typing_loop())

                # 2) Wrap status_cb to also drain/send stage OGGs immediately
                def status_cb(stage: str, info: Any | None = None):
                    # 1) original UI/edit callback
                    orig_status(stage, info)

                    # 2) only stream OGGs for the stages we’ve opted in
                    if stage not in chat_asm.tts_telegram_stages:
                        return

                    # 3) drain all new .ogg files and send each immediately
                    while True:
                        try:
                            ogg_path = chat_asm.tts._ogg_q.get_nowait()
                        except _queue.Empty:
                            break
                        else:
                            # schedule a send_voice using the current reply_to_id
                            loop.call_soon_threadsafe(
                                lambda path=ogg_path, reply_to=reply_to_id: asyncio.create_task(
                                    bot.send_voice(
                                        chat_id=chat_id,
                                        voice=InputFile(path),
                                        reply_to_message_id=reply_to
                                    )
                                )
                            )

                async def runner():
                    nonlocal placeholder_id, typing_task
                    pre_oggs = []
                    post_oggs = []

                    # 0) Drain any stale audio queues
                    for q in (chat_asm.tts._file_q, chat_asm.tts._ogg_q):
                        while True:
                            try: q.get_nowait()
                            except _queue.Empty: break

                    # 1) Prepare image payloads
                    image_paths = metadata.get("image_paths", [])
                    images_b64 = []
                    for p in image_paths:
                        try:
                            data = Path(p).read_bytes()
                            images_b64.append(base64.b64encode(data).decode("ascii"))
                        except Exception:
                            continue

                async def runner():
                    nonlocal placeholder_id, typing_task
                    pre_oggs = []
                    post_oggs = []

                    # 0) Drain any stale audio queues so we only send fresh clips
                    for q in (chat_asm.tts._file_q, chat_asm.tts._ogg_q):
                        while True:
                            try:
                                q.get_nowait()
                            except _queue.Empty:
                                break

                    # 1) Prepare any image payloads as base64 strings
                    image_paths = metadata.get("image_paths", [])
                    images_b64 = []
                    for p in image_paths:
                        try:
                            data = Path(p).read_bytes()
                            images_b64.append(base64.b64encode(data).decode("ascii"))
                        except Exception:
                            continue

                    # 2) Run inference in file‐mode, streaming live tokens into .ogg queue
                    chat_asm.tts.set_mode("file")
                    try:
                        sink = chat_asm.tts.token_sink()
                        final = await chat_asm.run_with_meta_context(
                            request_text,
                            status_cb,
                            images=images_b64 or None,
                            on_token=sink,
                        )
                        sink(None)                              # flush synth buffer
                        await asyncio.to_thread(chat_asm.tts._file_q.join)  # wait for OGG encode
                    except Exception:
                        logger.exception("run_with_meta_context failed")
                        final = ""
                    finally:
                        stop_status()
                        if typing_task:
                            typing_task.cancel()

                    # 3) Collect any OGG clips generated *before* the text reply
                    while True:
                        try:
                            ogg = chat_asm.tts._ogg_q.get_nowait()
                        except _queue.Empty:
                            break
                        else:
                            pre_oggs.append(ogg)

                    # 5) Send or edit the assistant's text reply
                    if debug_enabled:
                        # edit the placeholder in debug mode
                        if len(final) < 4000:
                            sent = await bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=placeholder_id,
                                text=final
                            )
                            target_id = sent.message_id
                        else:
                            # delete placeholder & chunk long text
                            await bot.delete_message(chat_id=chat_id, message_id=placeholder_id)
                            await _send_long_text_async(
                                bot, chat_id, final, reply_to=reply_to_id
                            )
                            target_id = reply_to_id
                    else:
                        # normal send
                        if len(final) < 4000:
                            sent = await bot.send_message(
                                chat_id=chat_id,
                                text=final,
                                reply_to_message_id=reply_to_id
                            )
                            target_id = sent.message_id
                        else:
                            await _send_long_text_async(
                                bot, chat_id, final, reply_to=reply_to_id
                            )
                            target_id = reply_to_id

                    # 6) Collect any OGG clips generated *after* the text reply
                    while True:
                        try:
                            ogg = chat_asm.tts._ogg_q.get_nowait()
                        except _queue.Empty:
                            break
                        else:
                            post_oggs.append(ogg)

                    # 7) Send all OGG clips in chronological order
                    all_oggs = pre_oggs + post_oggs
                    if all_oggs:
                        # if exactly one clip, send it directly
                        if len(all_oggs) == 1:
                            path = all_oggs[0]
                            if os.path.getsize(path) > 0:
                                # Let Telegram open the path itself
                                await bot.send_voice(
                                    chat_id=chat_id,
                                    voice=InputFile(path),
                                    reply_to_message_id=target_id
                                )
                        else:
                            # concatenate into one OGG and send
                            combined = os.path.join(
                                tempfile.gettempdir(),
                                f"combined_{uuid.uuid4().hex}.ogg"
                            )
                            inputs  = sum([["-i", p] for p in all_oggs], [])
                            streams = "".join(f"[{i}:a]" for i in range(len(all_oggs)))
                            filt    = f"{streams}concat=n={len(all_oggs)}:v=0:a=1,aresample=48000"
                            subprocess.run(
                                [
                                    "ffmpeg", "-y", "-loglevel", "error",
                                    *inputs,
                                    "-filter_complex", filt,
                                    "-c:a", "libopus",
                                    "-b:a", "48k",
                                    combined
                                ],
                                check=True
                            )
                            with open(combined, "rb") as vf:
                                await bot.send_voice(
                                    chat_id=chat_id,
                                    voice=vf,
                                    reply_to_message_id=target_id
                                )

                    # 8) (unchanged) Photo results from search_images tool calls
                    last = getattr(chat_asm, "_last_state", {}) or {}
                    for tc in last.get("tool_ctxs", []):
                        name = tc.metadata.get("tool_call", "").split("(", 1)[0]
                        if name == "search_images" and isinstance(tc.metadata.get("output"), list):
                            for img in tc.metadata["output"]:
                                try:
                                    # local file first
                                    with open(img, "rb") as fimg:
                                        await bot.send_photo(
                                            chat_id, photo=fimg, reply_to_message_id=reply_to_id
                                        )
                                except:
                                    await bot.send_photo(
                                        chat_id, photo=img, reply_to_message_id=reply_to_id
                                    )

                    # 9) Pin / unpin the final text (debug mode only)
                    if "sent" in locals():
                        try:
                            await bot.pin_chat_message(
                                chat_id=chat_id,
                                message_id=sent.message_id,
                                disable_notification=True
                            )
                            asyncio.create_task(_delayed_unpin(bot, chat_id, sent.message_id))
                        except:
                            pass

                    # 10) Cleanup & possibly start next queued request
                    running.pop((chat_id, bot.id), None)
                    try:
                        nxt, nxt_id = _pending[(chat_id, bot.id)].get_nowait()
                    except:
                        return
                    await start_runner(nxt, nxt_id)


                running[(chat_id, bot.id)] = loop.create_task(runner())

            # ── Kick off first run ─────────────────────────────────────────────
            await start_runner(user_text, trigger_id)


        app.add_handler(
            MessageHandler(
                (filters.TEXT
                | filters.VOICE
                | filters.PHOTO
                | filters.Document.ALL)
                & ~filters.COMMAND,
                _handle
            ),
            group=1
        )

    loop.run_until_complete(app.initialize())
    loop.run_until_complete(app.start())
    loop.run_until_complete(app.updater.start_polling())
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(app.updater.stop_polling())
        loop.run_until_complete(app.stop())
        loop.run_until_complete(app.shutdown())
        loop.close()

running: dict[tuple[int,int], asyncio.Task] = {}
