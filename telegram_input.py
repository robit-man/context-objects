# telegram_input.py
"""
Telegram I/O wrapper for the Assembler.

Features
────────
• Feeds user text into `asm.run_with_meta_context`
• Live “🛠 Processing…” message that updates once per stage
• Replaces the placeholder with the full answer (chunked if >4 000 chars)
• Streams exactly ONE .ogg voice reply
• Accepts **voice notes** – they’re run through Whisper, then processed as text
"""

from __future__ import annotations

import asyncio, os, queue, tempfile, subprocess, uuid
from typing import Any, Callable, List

# ── app-level imports ──────────────────────────────────────────────────
from assembler     import Assembler
from tts_service   import TTSManager

# whisper for voice-note transcription
import whisper
_WHISPER = whisper.load_model("base")                 # load once

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

# chat_id ➜ dedicated Assembler instance
assemblers: dict[int, "Assembler"] = {}


# ────────────────────────────────────────────────────────────────────────
# Helper: split & SEND long text – must be *async* and awaited
# ────────────────────────────────────────────────────────────────────────
async def _send_long_text_async(
    bot,
    chat_id: int,
    text: str,
    *,
    chunk_size: int = 3800,
) -> None:
    if not text.strip():
        return

    import re
    paras: List[str] = text.split("\n\n")
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
            else:                                   # huge paragraph – split sentences
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
# Factory for the per-stage status callback
# ────────────────────────────────────────────────────────────────────────
def _make_status_cb(
    loop: asyncio.AbstractEventLoop,
    bot,
    chat_id: int,
    msg_id: int,
    *,
    max_lines: int = 20,
    min_interval: float = 5,
):
    history: List[str] = []
    last_edit_at: float = 0.0
    pending_handle: asyncio.TimerHandle | None = None
    disabled: bool = False                      # set True by stop_cb()

    async def _do_edit() -> None:
        nonlocal last_edit_at, pending_handle
        if disabled:
            return
        text = "🛠️ Processing…\n" + "\n".join(history[-max_lines:])
        try:
            await bot.edit_message_text(chat_id=chat_id,
                                        message_id=msg_id,
                                        text=text)
        except Exception:
            pass
        finally:
            last_edit_at = loop.time()
            pending_handle = None

    def _schedule_edit() -> None:
        nonlocal pending_handle
        if disabled or pending_handle is not None:
            return
        delay = max(min_interval - (loop.time() - last_edit_at), 0.0)
        if delay == 0:
            asyncio.run_coroutine_threadsafe(_do_edit(), loop)
        else:
            pending_handle = loop.call_later(
                delay,
                lambda: asyncio.run_coroutine_threadsafe(_do_edit(), loop)
            )

    def status_cb(stage: str, output: Any) -> None:
        if disabled:
            return
        snippet = str(output).replace("\n", " ")
        if len(snippet) > 2000:
            snippet = snippet[:1997] + "…"
        history.append(f"• {stage}: {snippet}")
        _schedule_edit()

    def stop_cb() -> None:
        nonlocal disabled, pending_handle
        disabled = True
        if pending_handle is not None and not pending_handle.cancelled():
            pending_handle.cancel()
            pending_handle = None

    return status_cb, stop_cb


# ────────────────────────────────────────────────────────────────────────
# Main entry-point – launch the Telegram event loop
# ────────────────────────────────────────────────────────────────────────
def telegram_input(asm):
    """Start the Telegram bot loop."""
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing BOT_TOKEN env var")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    req = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app = ApplicationBuilder().token(token).request(req).build()

    running: dict[int, asyncio.Task] = {}
    asm._chat_contexts = set()                      # for proactive pings

    # ───────────── Telegram update handler ─────────────────────────────
    async def _handle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id

        # 1) ── TEXT directly from message
        user_text = (update.message.text or "").strip() if update.message else ""

        # 2) ── VOICE → download & Whisper
        if not user_text and update.message and update.message.voice:
            try:
                # download opus-in-OGG to temp (keep *path*, not file-handle)
                raw_ogg_path = tempfile.mktemp(suffix=".oga")
                voice_file   = await context.bot.get_file(update.message.voice.file_id)
                await voice_file.download_to_drive(raw_ogg_path)

                # convert to 16 kHz mono WAV
                wav_path = raw_ogg_path + ".wav"
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-y",
                     "-i", raw_ogg_path, "-ac", "1", "-ar", "16000", wav_path],
                    check=True
                )

                # Whisper transcription
                result    = _WHISPER.transcribe(wav_path, language="en")
                user_text = result.get("text", "").strip()

            except Exception as ex:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ Voice note error: {ex}"
                )
            finally:
                # clean up temp files
                for p in (locals().get("raw_ogg_path"), locals().get("wav_path")):
                    if p and isinstance(p, str) and os.path.exists(p):
                        try:
                            os.unlink(p)
                        except Exception:
                            pass

        if not user_text:
            return                                    # nothing to process

        # 3) ── per-chat Assembler
        chat_asm = assemblers.get(chat_id)
        if chat_asm is None:
            tts_mgr = TTSManager(
                logger        = asm.tts.log,
                cfg           = asm.cfg,
                audio_service = None
            )
            tts_mgr.set_mode("file")
            chat_asm = Assembler(
                context_path     = f"context_{chat_id}.jsonl",
                config_path      = "config.json",
                lookback_minutes = 60,
                top_k            = 5,
                tts_manager      = tts_mgr,
            )
            assemblers[chat_id] = chat_asm

        chat_asm._chat_contexts.add(chat_id)

        # cancel any previous job in this chat
        prev = running.get(chat_id)
        if prev and not prev.done():
            prev.cancel()

        placeholder = await context.bot.send_message(
            chat_id=chat_id, text="🛠️ Processing…"
        )
        msg_id = placeholder.message_id
        status_cb, stop_status = _make_status_cb(loop, context.bot, chat_id, msg_id)

        async def runner() -> None:
            try:
                for q in (chat_asm.tts._file_q, chat_asm.tts._ogg_q):
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break

                chat_asm.tts.set_mode("file")
                final = await asyncio.to_thread(
                    chat_asm.run_with_meta_context,
                    user_text,
                    status_cb,
                    chat_id,
                    msg_id,
                )
                stop_status()

                if final and len(final) < 4000:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg_id, text=final
                    )
                else:
                    try:
                        await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                    except Exception:
                        pass
                    await _send_long_text_async(context.bot, chat_id, final or "(no response)")

                chat_asm.tts.enqueue(final or "")
                await asyncio.to_thread(chat_asm.tts._file_q.join)

                ogg_paths: List[str] = []
                while True:
                    try:
                        p = chat_asm.tts._ogg_q.get_nowait()
                        if os.path.getsize(p) > 0:
                            ogg_paths.append(p)
                    except queue.Empty:
                        break

                if not ogg_paths:
                    return
                if len(ogg_paths) == 1:
                    with open(ogg_paths[0], "rb") as vf:
                        await context.bot.send_voice(chat_id=chat_id, voice=vf)
                    return

                combined = os.path.join(tempfile.gettempdir(), f"combined_{uuid.uuid4().hex}.ogg")
                ff_in = sum([["-i", p] for p in ogg_paths], [])
                streams = "".join(f"[{i}:a]" for i in range(len(ogg_paths)))
                concat_filter = f"{streams}concat=n={len(ogg_paths)}:v=0:a=1,aresample=48000"
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-y", *ff_in,
                     "-filter_complex", concat_filter,
                     "-c:a", "libopus", "-b:a", "48k", combined],
                    check=True
                )
                with open(combined, "rb") as vf:
                    await context.bot.send_voice(chat_id=chat_id, voice=vf)

            except asyncio.CancelledError:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg_id,
                        text="⚠️ Previous request cancelled."
                    )
                except Exception:
                    pass
            except Exception as e:
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {e}")
            finally:
                running.pop(chat_id, None)

        running[chat_id] = loop.create_task(runner())

    # accept plain text **and** voice messages
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
