# telegram_input.py

import os
import asyncio
import queue
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

def telegram_input(asm):
    """
    Start a Telegram bot loop that feeds incoming text messages
    into `asm.run_with_meta_context`, then streams back the final
    text and at most one generated .ogg voice reply (the very last one).
    """
    load_dotenv()
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    if not BOT_TOKEN:
        raise RuntimeError("Missing BOT_TOKEN in environment")

    # Create and install a dedicated asyncio loop for telegram
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Use HTTPX for robust timeouts
    req = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app = ApplicationBuilder().token(BOT_TOKEN).request(req).build()

    # Track one active request per chat so we can cancel stale ones
    running_tasks: dict[int, asyncio.Task] = {}

    async def _handle_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id   = update.effective_chat.id
        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        # Cancel any previous pending runner in this chat
        prev = running_tasks.get(chat_id)
        if prev and not prev.done():
            prev.cancel()

        # Acknowledge receipt
        sent   = await context.bot.send_message(chat_id=chat_id, text="🛠️ Processing…")
        msg_id = sent.message_id

# inside telegram_input.py, replace your runner() with:

        async def runner():
            try:
                # ── 0) Flush any leftover file‐TTS texts and OGG paths ───────
                try:
                    while True:
                        asm.tts._file_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    while True:
                        asm.tts._ogg_q.get_nowait()
                except queue.Empty:
                    pass

                # ── 1) Switch TTS into file‐output mode ─────────────────────
                asm.tts.set_mode("file")

                # ── 2) Run the assembler pipeline synchronously ────────────
                final = await asyncio.to_thread(asm.run_with_meta_context, user_text)

                # ── 3) Edit the “Processing…” message to show the final text ──
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=final or "(no response)"
                )

                # ── 4) Now *only* generate the one OGG for that final answer ──

                # Clear out any accidental enqueues during the run
                try:
                    while True:
                        asm.tts._file_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    while True:
                        asm.tts._ogg_q.get_nowait()
                except queue.Empty:
                    pass

                # Enqueue only the final text
                asm.tts.enqueue(final or "")

                # Wait for exactly one OGG
                ogg_path = asm.tts.wait_for_latest_ogg(timeout=10.0)

                # Send it back
                with open(ogg_path, "rb") as vf:
                    await context.bot.send_voice(
                        chat_id=chat_id,
                        voice=vf,
                        reply_to_message_id=msg_id
                    )

            except asyncio.CancelledError:
                # If a new request preempted this one…
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text="⚠️ Previous request cancelled."
                    )
                except:
                    pass

            except Exception as e:
                # Any other error…
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ Error: {e}"
                )

            finally:
                running_tasks.pop(chat_id, None)


        # Launch the runner as a background task
        task = loop.create_task(runner())
        running_tasks[chat_id] = task

    # Only handle plain text messages (ignore commands)
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_update)
    )

    # Start up the bot (this blocks this thread, so run in a separate thread)
    loop.run_until_complete(app.initialize())
    loop.run_until_complete(app.start())
    loop.run_until_complete(app.updater.start_polling())
    try:
        loop.run_forever()
    finally:
        # Clean shutdown
        loop.run_until_complete(app.updater.stop_polling())
        loop.run_until_complete(app.stop())
        loop.run_until_complete(app.shutdown())
        loop.close()
