# telegram_input.py

import os
import asyncio
import queue
import tempfile
import subprocess
import uuid
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest
from typing import Any, Callable

# ──────────────────────────────────────────────────────────────────────
def _send_long_text(bot, chat_id: int, text: str, chunk_size: int = 3800):
    """
    Break `text` into sensible ≤chunk_size chunks on paragraph/sentence
    boundaries, and send each chunk sequentially.
    """
    import re
    paras = text.split("\n\n")
    buffer = ""
    parts: list[str] = []

    for para in paras:
        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                parts.append(buffer)
                buffer = ""
            if len(para) <= chunk_size:
                buffer = para
            else:
                for sent in re.split(r'(?<=[\.\?\!])\s+', para):
                    if len(buffer) + len(sent) + 1 <= chunk_size:
                        buffer = (buffer + " " + sent).strip()
                    else:
                        if buffer:
                            parts.append(buffer)
                        buffer = ""
                        # hard-slice long sentences
                        for i in range(0, len(sent), chunk_size):
                            parts.append(sent[i : i + chunk_size])
                        buffer = ""
    if buffer:
        parts.append(buffer)

    for part in parts:
        bot.send_message(chat_id=chat_id, text=part)


def _make_status_cb(
    loop: asyncio.AbstractEventLoop,
    bot,
    chat_id: int,
    msg_id: int,
    max_lines: int = 6
) -> Callable[[str, Any], None]:
    """
    Returns status_cb(stage, output) that appends a rolling history and edits
    the original “Processing…” message in place. Thread-safe for use inside to_thread().
    """
    history: list[str] = []

    async def _do_edit(text: str):
        try:
            await bot.edit_message_text(chat_id=chat_id,
                                        message_id=msg_id,
                                        text=text)
        except Exception:
            pass  # ignore races / deleted message

    def status_cb(stage: str, output: Any):
        snippet = str(output).replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:397] + "…"
        history.append(f"• {stage}: {snippet}")
        text = "🛠️ Processing…\n" + "\n".join(history[-max_lines:])
        # Schedule on the event loop safely from any thread
        asyncio.run_coroutine_threadsafe(_do_edit(text), loop)

    return status_cb


def telegram_input(asm):
    """
    Telegram bot loop that:
      • feeds incoming text into asm.run_with_meta_context
      • live-updates one “Processing…” message per stage
      • then replaces it with the full answer (chunked if needed)
      • finally sends exactly one .ogg voice reply
    """
    load_dotenv()
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    if not BOT_TOKEN:
        raise RuntimeError("Missing BOT_TOKEN in environment")

    # Dedicated asyncio loop for Telegram
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # HTTPXRequest for robust timeouts
    req = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app = ApplicationBuilder().token(BOT_TOKEN).request(req).build()

    running_tasks: dict[int, asyncio.Task] = {}
    asm._chat_contexts = set()  # for any proactive pings

    async def _handle_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        # Track chat for “appiphany” pings
        asm._chat_contexts.add(chat_id)

        # Cancel any in-flight task for this chat
        prev = running_tasks.get(chat_id)
        if prev and not prev.done():
            prev.cancel()

        # 1) Send initial placeholder
        sent = await context.bot.send_message(chat_id=chat_id, text="🛠️ Processing…")
        msg_id = sent.message_id

        # 2) Build our status callback
        status_cb = _make_status_cb(loop, context.bot, chat_id, msg_id)

        async def runner():
            try:
                # ── Flush any leftover TTS buffers
                for q in (asm.tts._file_q, asm.tts._ogg_q):
                    try:
                        while True:
                            q.get_nowait()
                    except queue.Empty:
                        pass

                # ── Switch TTS into file output mode
                asm.tts.set_mode("file")

                # ── Run the assembler pipeline with live status updates
                final = await asyncio.to_thread(
                    asm.run_with_meta_context,
                    user_text,
                    status_cb,
                    chat_id,
                    msg_id
                )

                # ── Replace the placeholder with the final text
                if final and len(final) < 4000:
                    # Short enough: edit in place
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text=final
                    )
                else:
                    # Too long (or empty): delete placeholder and send in chunks
                    try:
                        await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                    except:
                        pass
                    _send_long_text(context.bot, chat_id, final or "(no response)")

                # ── Enqueue final for TTS and send exactly one voice reply
                asm.tts.enqueue(final or "")
                ogg_paths: list[str] = []
                while True:
                    try:
                        path = await asyncio.to_thread(asm.tts.wait_for_latest_ogg, 1.0)
                        ogg_paths.append(path)
                    except queue.Empty:
                        break

                if ogg_paths:
                    if len(ogg_paths) == 1:
                        with open(ogg_paths[0], "rb") as vf:
                            await context.bot.send_voice(chat_id=chat_id, voice=vf)
                    else:
                        # Concatenate multiple .ogg files into one
                        combined = os.path.join(tempfile.gettempdir(),
                                                f"combined_{uuid.uuid4().hex}.ogg")
                        listfile = tempfile.NamedTemporaryFile("w+", delete=False)
                        for p in ogg_paths:
                            listfile.write(f"file '{p}'\n")
                        listfile.flush()
                        listfile.close()

                        cmd1 = [
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", listfile.name, "-c", "copy", combined
                        ]
                        try:
                            subprocess.run(cmd1, check=True,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                        except subprocess.CalledProcessError:
                            # Fallback: re-encode/join
                            inputs = sum([["-i", p] for p in ogg_paths], [])
                            cmd2 = ["ffmpeg", "-y"] + inputs + [
                                "-filter_complex", f"concat=n={len(ogg_paths)}:v=0:a=1[out]",
                                "-map", "[out]", combined
                            ]
                            subprocess.run(cmd2, check=True,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                        finally:
                            try: os.unlink(listfile.name)
                            except: pass

                        with open(combined, "rb") as vf:
                            await context.bot.send_voice(chat_id=chat_id, voice=vf)

            except asyncio.CancelledError:
                # A newer request preempted this one
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text="⚠️ Previous request cancelled."
                    )
                except:
                    pass

            except Exception as e:
                # Final fallback on error
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {e}")

            finally:
                running_tasks.pop(chat_id, None)

        # Launch background runner
        running_tasks[chat_id] = loop.create_task(runner())

    # Only handle plain-text (ignore commands)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_update))

    # Start the bot (blocks this thread)
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
