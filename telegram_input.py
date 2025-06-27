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

        async def runner():
            try:
                # ── 0) Clear any stale .ogg paths from previous runs
                try:
                    while True:
                        asm.tts._ogg_q.get_nowait()
                except queue.Empty:
                    pass

                # ── 1) Switch TTS into file-output mode
                asm.tts.set_mode("file")

                # ── 2) Run the entire assembler pipeline in a thread
                final = await asyncio.to_thread(asm.run_with_meta_context, user_text)

                # ── 3) Update the message with the final text
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=final or "(no response)"
                )

                # ── 4) Wait until the file-TTS queue has been fully processed
                asm.tts._file_q.join()

                # ── 5) Now retrieve exactly one .ogg path (the most recent)
                try:
                    ogg_path = asm.tts.wait_for_latest_ogg(timeout=5.0)
                except queue.Empty:
                    return

                # ── 6) Send that single OGG back
                with open(ogg_path, "rb") as vf:
                    await context.bot.send_voice(
                        chat_id=chat_id,
                        voice=vf,
                        reply_to_message_id=msg_id
                    )

            except asyncio.CancelledError:
                # If a new request preempted this one
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text="⚠️ Previous request cancelled."
                    )
                except:
                    pass

            except Exception as e:
                # On any other error, report it
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
