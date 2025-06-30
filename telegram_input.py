# telegram_input.py
"""
Telegram I/O wrapper for the Assembler.

Features
────────
• Feeds user text into `asm.run_with_meta_context`
• Live “🛠 Processing…” message that updates once per stage
• Replaces the placeholder with the full answer (chunked if >4 000 chars)
• Streams exactly ONE .ogg voice reply
"""

from __future__ import annotations

import asyncio, os, queue, tempfile, subprocess, uuid
from typing import Any, Callable, List

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest


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
    """
    Break `text` into ≤chunk_size chunks (paragraph/sentence splits),
    then **await** a `send_message` for each chunk sequentially.
    """
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
            else:
                # paragraph itself is huge – split on sentences
                for sent in re.split(r"(?<=[\.\?\!])\s+", para):
                    if len(buffer) + len(sent) + 1 <= chunk_size:
                        buffer = (buffer + " " + sent).strip()
                    else:
                        if buffer:
                            chunks.append(buffer)
                        buffer = ""
                        # hard-slice any still-oversized sentence
                        for i in range(0, len(sent), chunk_size):
                            chunks.append(sent[i : i + chunk_size])
    if buffer:
        chunks.append(buffer)

    # one await per chunk
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
    max_lines: int = 6,
) -> Callable[[str, Any], None]:
    """
    Return a thread-safe `status_cb(stage, output)` that updates a rolling
    history inside the original “🛠 Processing…” message.
    """
    history: List[str] = []

    async def _do_edit(text: str) -> None:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=text,
            )
        except Exception:  # message deleted or race condition – ignore
            pass

    def status_cb(stage: str, output: Any) -> None:
        snippet = str(output).replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:397] + "…"
        history.append(f"• {stage}: {snippet}")
        text = "🛠️ Processing…\n" + "\n".join(history[-max_lines:])

        # schedule safely from ANY thread
        asyncio.run_coroutine_threadsafe(_do_edit(text), loop)

    return status_cb


# ────────────────────────────────────────────────────────────────────────
# Main entry-point – launch the Telegram event loop
# ────────────────────────────────────────────────────────────────────────
def telegram_input(asm):
    """Start the Telegram bot loop."""
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing BOT_TOKEN env var")

    # Create a dedicated asyncio loop for Telegram
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # HTTPXRequest *without* unsupported pool arguments
    req = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app = ApplicationBuilder().token(token).request(req).build()

    # Track one concurrent runner per chat to allow cancellation
    running: dict[int, asyncio.Task] = {}
    asm._chat_contexts = set()  # for possible proactive pings

    # ──────────────── Telegram update handler ──────────────────────────
    async def _handle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        asm._chat_contexts.add(chat_id)  # for _maybe_appiphany()

        # Cancel any previous unfinished task in this chat
        prev = running.get(chat_id)
        if prev and not prev.done():
            prev.cancel()

        # Send initial placeholder
        placeholder = await context.bot.send_message(
            chat_id=chat_id, text="🛠️ Processing…"
        )
        msg_id = placeholder.message_id

        # Build per-stage callback
        status_cb = _make_status_cb(loop, context.bot, chat_id, msg_id)

        # ───────────── background pipeline runner ──────────────────────
        async def runner() -> None:
            try:
                # Clear any stale TTS queues
                for q in (asm.tts._file_q, asm.tts._ogg_q):
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break

                # TTS in file mode so we get .ogg back
                asm.tts.set_mode("file")

                # Run the assembler (thread pool) with live status updates
                final = await asyncio.to_thread(
                    asm.run_with_meta_context,
                    user_text,
                    status_cb,
                    chat_id,
                    msg_id,
                )

                # ── Replace the placeholder with the answer ────────────
                if final and len(final) < 4000:
                    # short enough → single edit
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg_id, text=final
                    )
                else:
                    # delete placeholder then send chunks (async)
                    try:
                        await context.bot.delete_message(
                            chat_id=chat_id, message_id=msg_id
                        )
                    except Exception:
                        pass
                    await _send_long_text_async(
                        context.bot, chat_id, final or "(no response)"
                    )

                # ── ONE voice reply (.ogg) ────────────────────────────
                asm.tts.enqueue(final or "")
                ogg_paths: List[str] = []
                while True:
                    try:
                        path = await asyncio.to_thread(
                            asm.tts.wait_for_latest_ogg, 1.0
                        )
                        ogg_paths.append(path)
                    except queue.Empty:
                        break

                if not ogg_paths:
                    return

                if len(ogg_paths) == 1:
                    with open(ogg_paths[0], "rb") as vf:
                        await context.bot.send_voice(chat_id=chat_id, voice=vf)
                else:
                    # Concatenate multiple .ogg files
                    combined = os.path.join(
                        tempfile.gettempdir(), f"combined_{uuid.uuid4().hex}.ogg"
                    )
                    listfile = tempfile.NamedTemporaryFile("w+", delete=False)
                    for p in ogg_paths:
                        listfile.write(f"file '{p}'\n")
                    listfile.flush(), listfile.close()

                    try:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-f",
                                "concat",
                                "-safe",
                                "0",
                                "-i",
                                listfile.name,
                                "-c",
                                "copy",
                                combined,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    finally:
                        try:
                            os.unlink(listfile.name)
                        except FileNotFoundError:
                            pass

                    with open(combined, "rb") as vf:
                        await context.bot.send_voice(chat_id=chat_id, voice=vf)

            except asyncio.CancelledError:
                # superseded by a newer request
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text="⚠️ Previous request cancelled.",
                    )
                except Exception:
                    pass

            except Exception as e:
                # Fallback on any unexpected error
                await context.bot.send_message(
                    chat_id=chat_id, text=f"❌ Error: {e}"
                )

            finally:
                running.pop(chat_id, None)

        # Kick off the runner
        running[chat_id] = loop.create_task(runner())

    # Only plain text updates (ignore /commands)
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _handle)
    )

    # ─── Start the Telegram application (blocking in this thread) ──────
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
