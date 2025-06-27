# telegram_input.py

import os
import asyncio
import threading
import queue
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

def telegram_input(asm):
    """
    Start a Telegram bot loop that:
      • Listens for text messages,
      • Spawns one dedicated Thread per chat_id to run the assembler pipeline,
      • Streams back the final text and exactly one .ogg reply.
    """
    load_dotenv()
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    if not BOT_TOKEN:
        raise RuntimeError("Missing BOT_TOKEN in environment")

    # Create its own asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    req = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app = ApplicationBuilder().token(BOT_TOKEN).request(req).build()

    # track running threads per chat so we can cancel stale runs
    running_threads: dict[int, threading.Thread] = {}

    async def _handle_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id   = update.effective_chat.id
        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        # cancel any previous thread for this chat
        prev = running_threads.get(chat_id)
        if prev and prev.is_alive():
            # we simply drop it; the new run will overwrite
            pass

        # send initial “Processing…” message
        sent   = await context.bot.send_message(chat_id=chat_id, text="🛠️ Processing…")
        msg_id = sent.message_id

        def thread_runner():
            # 0) clear any old OGG paths
            try:
                while True:
                    asm.tts._ogg_q.get_nowait()
            except queue.Empty:
                pass

            # 1) switch TTS into file mode
            asm.tts.set_mode("file")

            # 2) run the pipeline synchronously
            try:
                final = asm.run_with_meta_context(user_text)
            except Exception as e:
                # send error message
                coro = context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ Error during processing: {e}"
                )
                asyncio.run_coroutine_threadsafe(coro, loop)
                return

            # 3) edit the original “Processing…” message with the final text
            coro = context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=final or "(no response)"
            )
            asyncio.run_coroutine_threadsafe(coro, loop)

            # 4) wait for the file‐TTS queue to finish (blocking until OGG is ready)
            asm.tts._file_q.join()

            # 5) now retrieve exactly one OGG path (the most recent)
            try:
                ogg_path = asm.tts.wait_for_latest_ogg(timeout=10.0)
            except queue.Empty:
                return

            # 6) send that single OGG back
            with open(ogg_path, "rb") as vf:
                send_voice = context.bot.send_voice(
                    chat_id=chat_id,
                    voice=vf,
                    reply_to_message_id=msg_id
                )
                asyncio.run_coroutine_threadsafe(send_voice, loop)

        # launch the thread
        t = threading.Thread(target=thread_runner, daemon=True)
        t.start()
        running_threads[chat_id] = t

    # wire up only TEXT messages
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_update)
    )

    # start the bot (this will block this thread)
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
