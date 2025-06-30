# telegram_input.py
"""
Telegram I/O wrapper for the Assembler.

Features
────────
• Feeds user text into `asm.run_with_meta_context`
• Live “🛠️ Processing…” message that updates once per stage
• Replaces the placeholder with the full answer (chunked if >4 000 chars)
• Streams exactly ONE .ogg voice reply
• Accepts **voice notes** – they’re run through Whisper, then processed as text
• Maintains a SQLite-backed user registry for DM’ing users by username
"""

from __future__ import annotations

import asyncio
import os
import queue
import tempfile
import subprocess
import uuid
import threading
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from assembler import Assembler
from tts_service import TTSManager
from user_registry import _REG
from group_registry import _GREG

import whisper
_WHISPER = whisper.load_model("base")  # load once


# chat_id → dedicated Assembler instance
assemblers: dict[int, Assembler] = {}

# ────────────────────────────────────────────────────────────────────────
# Helper to split & send long text (async)
# ────────────────────────────────────────────────────────────────────────
async def _send_long_text_async(bot, chat_id: int, text: str, *, chunk_size: int = 3800) -> None:
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
        await bot.send_message(chat_id=chat_id, text=part)

# ────────────────────────────────────────────────────────────────────────
# Factory for per-stage status callback
# ────────────────────────────────────────────────────────────────────────
def _make_status_cb(loop: asyncio.AbstractEventLoop, bot, chat_id: int, msg_id: int,
                    *, max_lines: int = 10, min_interval: float = 5):
    history: List[str] = []
    last_edit_at: float = 0.0
    pending_handle: asyncio.TimerHandle | None = None
    disabled = False

    async def _do_edit():
        nonlocal last_edit_at, pending_handle
        if disabled: return
        text = "🛠️ Processing…\n" + "\n".join(history[-max_lines:])
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text)
        except:
            pass
        last_edit_at = loop.time()
        pending_handle = None

    def _schedule_edit():
        nonlocal pending_handle
        if disabled or pending_handle: return
        delay = max(min_interval - (loop.time() - last_edit_at), 0.0)
        if delay <= 0:
            asyncio.run_coroutine_threadsafe(_do_edit(), loop)
        else:
            pending_handle = loop.call_later(delay,
                lambda: asyncio.run_coroutine_threadsafe(_do_edit(), loop)
            )

    def status_cb(stage: str, output: Any):
        nonlocal history
        if disabled: return
        snippet = str(output).replace("\n", " ")
        if len(snippet) > 1000:
            snippet = snippet[:997] + "…"
        history.append(f"• {stage}: {snippet}")
        _schedule_edit()

    def stop_cb():
        nonlocal disabled, pending_handle
        disabled = True
        if pending_handle and not pending_handle.cancelled():
            pending_handle.cancel()
            pending_handle = None

    return status_cb, stop_cb

# ────────────────────────────────────────────────────────────────────────
# Main entry-point: launch the Telegram bot
# ────────────────────────────────────────────────────────────────────────
def telegram_input(asm: Assembler):
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing BOT_TOKEN env var")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    req = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app = ApplicationBuilder().token(token).request(req).build()

    running: dict[int, asyncio.Task] = {}
    asm._chat_contexts = set()

    async def _handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id   = update.effective_chat.id
        chat_type = update.effective_chat.type  # "private", "group", etc.
        bot_name  = context.bot.username.lower()
        msg       = update.message
        if not msg:
            return

        # record user in registry
        user = update.effective_user
        if user.username:
            _REG.add(user.username, user.id)

        # record group if in a group chat
        if chat_type in ("group", "supergroup", "channel"):
            title = update.effective_chat.title or f"chat_{chat_id}"
            _GREG.add(title, chat_id)

        # private‐only commands for both users and groups
        text = msg.text or ""
        if chat_type == "private" and text.startswith("/"):
            parts = text.split(None, 2)
            cmd = parts[0].lower()

            if cmd == "/list_users":
                users = _REG.list_all()
                await context.bot.send_message(
                    chat_id,
                    "\n".join(f"@{u['username']}: {u['id']}" for u in users) or "(none)"
                )
                return

            if cmd == "/list_groups":
                groups = _GREG.list_all()
                await context.bot.send_message(
                    chat_id,
                    "\n".join(f"{g['name']}: {g['chat_id']}" for g in groups) or "(none)"
                )
                return

            if cmd == "/dm" and len(parts) == 3:
                target, body = parts[1], parts[2]
                target_id = _REG.id_for(target)
                if not target_id:
                    await context.bot.send_message(chat_id, f"User @{target} not found.")
                else:
                    await context.bot.send_message(
                        target_id, f"DM from @{user.username}: {body}"
                    )
                    await context.bot.send_message(chat_id, f"✔️ Sent to @{target}")
                return

            if cmd == "/gm" and len(parts) == 3:
                grp, body = parts[1], parts[2]
                grp_id = _GREG.id_for(grp)
                if not grp_id:
                    await context.bot.send_message(chat_id, f"Group '{grp}' not found.")
                else:
                    await context.bot.send_message(grp_id, f"[Group DM] {body}")
                    await context.bot.send_message(chat_id, f"✔️ Sent to group '{grp}'")
                return

        # decide if we run the assembler
        addressed = (chat_type == "private") or bool(msg.voice)
        cleaned = ""
        # mentions
        if msg.text and msg.entities:
            for ent in msg.entities:
                if ent.type == "mention":
                    mention = msg.text[ent.offset:ent.offset+ent.length]
                    if mention.lstrip("@").lower() == bot_name:
                        addressed = True
                        cleaned = (msg.text[:ent.offset] + msg.text[ent.offset+ent.length:]).strip()
                        break
        # replies
        if not addressed and msg.reply_to_message:
            if msg.reply_to_message.from_user.id == context.bot.id:
                addressed = True

        if not addressed:
            return

        # plain text
        if not cleaned:
            cleaned = msg.text or ""
        cleaned = cleaned.strip()

        # voice note transcription
        if msg.voice:
            try:
                raw_ogg = tempfile.mktemp(suffix=".oga")
                vf = await context.bot.get_file(msg.voice.file_id)
                await vf.download_to_drive(raw_ogg)
                wav = raw_ogg + ".wav"
                subprocess.run(
                    ["ffmpeg","-y","-loglevel","error","-i",raw_ogg,"-ac","1","-ar","16000",wav],
                    check=True
                )
                result = _WHISPER.transcribe(wav, language="en")
                cleaned = result.get("text","").strip() or cleaned
            except Exception as ex:
                await context.bot.send_message(chat_id, f"❌ Voice note error: {ex}")
            finally:
                for p in (locals().get("raw_ogg"), locals().get("wav")):
                    if p and os.path.exists(p):
                        try: os.unlink(p)
                        except: pass

        if not cleaned:
            return

        # prefix username
        sender = user.username or user.first_name
        user_text = f"{sender}: {cleaned}"

        # per-chat assembler
        chat_asm = assemblers.get(chat_id)
        if chat_asm is None:
            tts = TTSManager(logger=asm.tts.log, cfg=asm.cfg, audio_service=None)
            tts.set_mode("file")
            chat_asm = Assembler(
                context_path=f"context_{chat_id}.jsonl",
                config_path="config.json",
                lookback_minutes=60,
                top_k=5,
                tts_manager=tts,
            )
            assemblers[chat_id] = chat_asm

        chat_asm._chat_contexts.add(chat_id)

        # cancel previous
        prev = running.get(chat_id)
        if prev and not prev.done():
            prev.cancel()

        placeholder = await context.bot.send_message(chat_id=chat_id, text="🛠️ Processing…")
        msg_id = placeholder.message_id
        status_cb, stop_status = _make_status_cb(loop, context.bot, chat_id, msg_id)

        async def runner():
            try:
                # clear any lingering TTS queues
                for q in (chat_asm.tts._file_q, chat_asm.tts._ogg_q):
                    while True:
                        try: q.get_nowait()
                        except queue.Empty: break

                chat_asm.tts.set_mode("file")
                final = await asyncio.to_thread(
                    chat_asm.run_with_meta_context,
                    user_text, status_cb, chat_id, msg_id
                )
                stop_status()

                # send text or chunks
                if final and len(final) < 4000:
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=final)
                else:
                    try:
                        await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                    except: pass
                    await _send_long_text_async(context.bot, chat_id, final or "(no response)")

                # send voice reply
                chat_asm.tts.enqueue(final or "")
                await asyncio.to_thread(chat_asm.tts._file_q.join)
                oggs = []
                while True:
                    try:
                        p = chat_asm.tts._ogg_q.get_nowait()
                        if os.path.getsize(p) > 0:
                            oggs.append(p)
                    except queue.Empty:
                        break

                if not oggs:
                    return
                if len(oggs) == 1:
                    with open(oggs[0],"rb") as vf:
                        await context.bot.send_voice(chat_id=chat_id, voice=vf)
                    return

                # combine multiple oggs
                combined = os.path.join(tempfile.gettempdir(), f"combined_{uuid.uuid4().hex}.ogg")
                ins = sum([["-i",p] for p in oggs], [])
                streams = "".join(f"[{i}:a]" for i in range(len(oggs)))
                filt = f"{streams}concat=n={len(oggs)}:v=0:a=1,aresample=48000"
                subprocess.run(
                    ["ffmpeg","-y","-loglevel","error",*ins,"-filter_complex",filt,
                     "-c:a","libopus","-b:a","48k",combined],
                    check=True
                )
                with open(combined,"rb") as vf:
                    await context.bot.send_voice(chat_id=chat_id, voice=vf)

            except asyncio.CancelledError:
                try:
                    await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_id,
                                                        text="⚠️ Previous request cancelled.")
                except: pass
            except Exception as e:
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {e}")
            finally:
                running.pop(chat_id, None)

        running[chat_id] = loop.create_task(runner())

    app.add_handler(
        MessageHandler((filters.TEXT | filters.VOICE) & ~filters.COMMAND, _handle)
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
