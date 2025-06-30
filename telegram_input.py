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

import shlex, asyncio, os, queue, tempfile, subprocess, uuid
from typing import Any, Callable, List

# ── build or reuse an isolated Assembler for this chat ──────────
from assembler     import Assembler       # local import avoids cycles
from tts_service   import TTSManager

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

assemblers: dict[int, "Assembler"] = {}      # chat_id ➜ private Assembler

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
    max_lines: int = 20,
    min_interval: float = 5,
):
    """
    Returns
        status_cb(stage, output)   – pass to the assembler
        stop_cb()                  – call once when work is finished
    `stop_cb` cancels any pending edit and disables further updates,
    preventing the final message from being overwritten.
    """
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
            pass  # message deleted / race condition – ignore
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
        """Disable further edits and cancel any scheduled one."""
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


        chat_asm = assemblers.get(chat_id)
        if chat_asm is None:
            # ► fresh TTS manager (file-mode only)
            tts_mgr = TTSManager(
                logger        = asm.tts.log,      # reuse global logger
                cfg           = asm.cfg,
                audio_service = None
            )
            tts_mgr.set_mode("file")

            # ► dedicated context file per chat
            chat_asm = Assembler(
                context_path     = f"context_{chat_id}.jsonl",
                config_path      = "config.json",
                lookback_minutes = 60,
                top_k            = 5,
                tts_manager      = tts_mgr,
            )
            assemblers[chat_id] = chat_asm

        # track chat for proactive pings
        chat_asm._chat_contexts.add(chat_id)

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
        status_cb, stop_status = _make_status_cb(loop, context.bot, chat_id, msg_id)

        # ───────────── background pipeline runner ──────────────────────
        async def runner() -> None:
            try:
                # Clear any stale TTS queues
                for q in (chat_asm.tts._file_q, chat_asm.tts._ogg_q):
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break

                # TTS in file mode so we get .ogg back
                chat_asm.tts.set_mode("file")

                # Run the assembler (thread pool) with live status updates
                final = await asyncio.to_thread(
                    chat_asm.run_with_meta_context,
                    user_text,
                    status_cb,
                    chat_id,
                    msg_id,
                )

                stop_status()

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
                chat_asm.tts.enqueue(final or "")

                # wait for every chunk to finish rendering
                await asyncio.to_thread(chat_asm.tts._file_q.join)

                # drain finished .ogg paths
                ogg_paths: List[str] = []
                while True:
                    try:
                        p = chat_asm.tts._ogg_q.get_nowait()
                        if os.path.getsize(p) > 0:
                            ogg_paths.append(p)
                    except queue.Empty:
                        break

                if not ogg_paths:
                    return  # nothing to send

                if len(ogg_paths) == 1:
                    with open(ogg_paths[0], "rb") as vf:
                        await context.bot.send_voice(chat_id=chat_id, voice=vf)
                    return  # done

                # ── concatenate via filter_complex ───────────────────
                combined = os.path.join(
                    tempfile.gettempdir(), f"combined_{uuid.uuid4().hex}.ogg"
                )

                # build ffmpeg arg list:  -i file1 -i file2 ...
                ff_in_args: list[str] = []
                for p in ogg_paths:
                    ff_in_args += ["-i", p]

                # e.g. "[0:a][1:a][2:a]concat=n=3:v=0:a=1,aresample=48000"
                streams = "".join(f"[{i}:a]" for i in range(len(ogg_paths)))
                concat_filter = f"{streams}concat=n={len(ogg_paths)}:v=0:a=1,aresample=48000"

                cmd = [
                    "ffmpeg", "-y",
                    *ff_in_args,
                    "-filter_complex", concat_filter,
                    "-loglevel", "error",
                    "-c:a", "libopus", "-b:a", "48k",
                    combined,
                ]

                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    # bubble up full stderr so we see the real problem if it ever fails
                    raise RuntimeError(f"ffmpeg concat failed (exit {e.returncode}):\n{e.stderr}") from e

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
