# tts_service.py

import threading
import queue
import subprocess
import json
import os
import uuid
import shutil
import time
import numpy as np
import re

# lock to prevent Piper/process races
tts_lock = threading.Lock()

class TTSManager:
    """
    Piper → live‐playback or file‐output TTS manager.
    Automatically splits long text into manageable chunks,
    and provides a token_sink() for buffering streaming tokens
    into full sentences (or timeout‐flushed fragments) before speaking.
    """

    def __init__(self, logger: callable, cfg: dict, audio_service=None):
        self.log            = logger
        self.config         = cfg
        self.audio_service  = audio_service
        self.volume         = cfg.get("tts_volume", 0.2)
        self.debug          = cfg.get("tts_debug", False)

        # chunk size in characters (adjust as needed)
        self.max_chunk_size = cfg.get("tts_max_chunk_size", 500)

        # Queues for live vs file modes
        self._live_q = queue.Queue()
        self._file_q = queue.Queue()
        self._ogg_q  = queue.Queue()  # holds paths to generated .ogg

        self._mode    = "live"
        self._running = True

        # Token‐buffer state
        self._token_buffer  = ""
        self._token_lock    = threading.Lock()
        self._flush_timer   = None
        self._flush_timeout = cfg.get("tts_token_flush_timeout", 1)

        # Start workers
        threading.Thread(target=self._live_worker, daemon=True).start()
        threading.Thread(target=self._file_worker, daemon=True).start()

    def set_mode(self, mode: str):
        if mode not in ("live", "file"):
            raise ValueError("TTSManager.set_mode: must be 'live' or 'file'")
        self.log(f"TTS mode → {mode}", "INFO")
        self._mode = mode
        # clear the opposite queue
        q = self._file_q if mode=="live" else self._live_q
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break

    def _flush_buffer(self):
        """Internal: emit whatever is left in the token buffer as one chunk."""
        with self._token_lock:
            text = self._token_buffer.strip()
            if text:
                self.enqueue(text)
            self._token_buffer = ""
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None

    def flush(self):
        """
        Public: immediately flush any buffered tokens (e.g. at end of stream)
        into a final sentence/fragment.
        """
        self._flush_buffer()

    def token_sink(self):
        """
        Returns a callback(token:str) that will buffer tokens into full
        sentences (splitting on . ? !), call enqueue() for each sentence,
        and then at the end you call sink(None) to flush leftovers.
        """
        sentence_end = re.compile(r'([\.!?])')

        buf = []

        def sink(token):
            # final flush
            if token is None:
                text = ''.join(buf).strip()
                if text:
                    self.enqueue(text)
                buf.clear()
                return

            buf.append(token)
            joined = ''.join(buf)
            # if we see punctuation, split there
            m = sentence_end.search(joined)
            if m:
                end = m.end()
                sentence = joined[:end].strip()
                self.enqueue(sentence)
                # carry over the rest (if any) back into buf
                remainder = joined[end:]
                buf.clear()
                if remainder:
                    buf.append(remainder)

        return sink

    def _split_text(self, text: str) -> list[str]:
        """
        1) Split on blank lines (paragraphs)
        2) Within each paragraph, split on sentence boundaries or newlines
        3) Pack sentences into chunks ≤ max_chunk_size
        4) Hard‑slice any leftover over‑long bits
        """
        paras = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
        chunks: list[str] = []
        for para in paras:
            parts = re.split(r'(?<=[\.\?\!])\s+|\n+|(?<=\u2022)\s*', para)
            buf = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # would overflow?
                if buf and len(buf) + 1 + len(part) > self.max_chunk_size:
                    chunks.append(buf)
                    buf = ""
                # fits?
                if len(part) <= self.max_chunk_size:
                    buf = (buf + " " + part).strip() if buf else part
                else:
                    # break it up
                    if buf:
                        chunks.append(buf)
                        buf = ""
                    for i in range(0, len(part), self.max_chunk_size):
                        slice_ = part[i:i+self.max_chunk_size].strip()
                        if slice_:
                            chunks.append(slice_)
            if buf:
                chunks.append(buf)
        return chunks

    def enqueue(self, text: str):
        """
        Queue up a piece of text (sentence or chunk) for TTS.
        In 'live' mode → _live_q; in 'file' mode → _file_q.
        """
        # strip emojis & markdown artifacts
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE
        )
        clean = emoji_pattern.sub('', text).strip().replace("*", "")
        if not clean:
            return

        parts = self._split_text(clean)
        total = len(parts)
        for idx, chunk in enumerate(parts, start=1):
            tag = f"{idx}/{total}"
            if self._mode == "live":
                self.log(f"Enqueue live TTS chunk {tag}: {chunk!r}", "DEBUG")
                self._live_q.put(chunk)
            else:
                self.log(f"Enqueue file TTS chunk {tag}: {chunk!r}", "DEBUG")
                self._file_q.put(chunk)

    def wait_for_latest_ogg(self, timeout: float):
        """Block up to `timeout` seconds for the next .ogg file path."""
        return self._ogg_q.get(timeout=timeout)

    def stop(self):
        """Shut everything down, cancel timers, unblock queues."""
        self._running = False
        # unblock workers
        self._live_q.put(None)
        self._file_q.put(None)
        # cancel any pending flush
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None

    # ───────────────────────────────────────────────────────────────────────
    # live‐mode worker
    # ───────────────────────────────────────────────────────────────────────
    # ───────────────────────────────────────────────────────────────────────
    # live‐mode worker
    # ───────────────────────────────────────────────────────────────────────
    def _live_worker(self):
        self.log("TTS live‐worker started.", "DEBUG")
        while self._running:
            text = self._live_q.get()
            if text is None:
                break
            try:
                self._do_live(text)
            except Exception as e:
                self.log(f"TTS live error: {e}", "ERROR")
            finally:
                self._live_q.task_done()
        self.log("TTS live‐worker exiting.", "DEBUG")


    def _do_live(self, text: str):
        """
        Speak `text` via Piper→aplay (or ffplay), but also feed
        the exact PCM into AudioService for cancellation.
        """
        import subprocess, json, time, shutil, os

        script_dir = os.path.dirname(__file__)
        piper_exe  = os.path.join(script_dir, "piper", self.config.get("piper_executable", "piper"))
        onnx_json  = os.path.join(script_dir, self.config.get("onnx_json_filename", "overwatch.onnx.json"))
        onnx_model = os.path.join(script_dir, self.config.get("onnx_model_filename", "overwatch.onnx"))

        # Helper that runs Piper and plays back via aplay/ffplay,
        # while capturing every PCM block for cancellation.
        def _speak_phrase(phrase: str):
            cmd_piper = [piper_exe, "-m", onnx_model, "--json-input", "--output_raw"]
            if self.debug:
                cmd_piper.insert(3, "--debug")

            if shutil.which("aplay"):
                cmd_play = ["aplay", "-r", "22050", "-f", "S16_LE"]
            elif shutil.which("ffplay"):
                cmd_play = ["ffplay", "-autoexit", "-nodisp", "-f", "s16le", "-ar", "22050", "-i", "pipe:0"]
            else:
                raise RuntimeError("Install 'aplay' or 'ffplay' for playback")

            payload = json.dumps({"text": phrase, "config": onnx_json, "model": onnx_model}).encode("utf-8")

            with tts_lock:
                p1 = subprocess.Popen(cmd_piper, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=(subprocess.PIPE if self.debug else subprocess.DEVNULL))
                p2 = subprocess.Popen(cmd_play, stdin=subprocess.PIPE)

            # send text → Piper
            p1.stdin.write(payload)
            p1.stdin.close()

            def _stream_audio():
                # Read raw 16‑bit PCM from Piper, cancel + play
                while True:
                    raw = p1.stdout.read(4096)
                    if not raw:
                        break

                    # 1) Decode → float32 in [-1.0,1.0]
                    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    frames = arr / 32768.0

                    # 2) Apply your volume
                    if self.volume != 1.0:
                        frames = frames * self.volume

                    # 3) Push *inverted* frames into AudioService for cancellation
                    svc = getattr(self, "audio_service", None)
                    if svc is not None:
                        try:
                            # invert here
                            svc.push_cancellation((-frames).copy())
                        except Exception:
                            pass

                    # 4) Re‑quantize → int16
                    out_arr = np.clip(frames * 32767.0, -32768, 32767).astype(np.int16)
                    out_bytes = out_arr.tobytes()

                    # 5) Write & flush so the player actually emits immediately
                    p2.stdin.write(out_bytes)
                    p2.stdin.flush()

                p2.stdin.close()


            streamer = threading.Thread(target=_stream_audio, daemon=True)
            streamer.start()
            p1.wait()
            streamer.join()
            p2.wait()

            if self.debug and p1.stderr:
                err = p1.stderr.read().decode(errors="ignore").strip()
                if err:
                    self.log(f"[Piper STDERR] {err}", "ERROR")


        # (Optional) Calibration on first call, unchanged from before:
        svc = getattr(self, "audio_service", None)
        if svc and not hasattr(svc, "_echo_profile"):
            self.log("[TTS live] Calibrating echo cancellation…", "INFO")
            cal_phrase = "calibration one two three"
            _speak_phrase(cal_phrase)
            wait_time = len(cal_phrase.split()) * 0.3 + 0.5
            time.sleep(wait_time)
            # old‐style profile capture if you still need it:
            with svc._buffer_lock:
                buf = svc._buffer.copy()
            clip_len = int(0.5 * svc.sample_rate)
            svc._echo_profile = buf[:clip_len] if len(buf) >= clip_len else buf
            self.log(f"[TTS live] Captured echo profile: {len(svc._echo_profile)} samples", "INFO")

        # Now speak the real text
        self.log(f"[TTS live] Speaking: {text!r}", "INFO")
        _speak_phrase(text)




    # ───────────────────────────────────────────────────────────────────────
    # file‐mode worker
    # ───────────────────────────────────────────────────────────────────────
    def _file_worker(self):
        self.log("TTS file‐worker started.", "DEBUG")
        while self._running:
            text = self._file_q.get()
            if text is None:
                break
            try:
                script_dir = os.path.dirname(__file__)
                piper_exe  = os.path.join(script_dir, "piper", self.config.get("piper_executable", "piper"))
                onnx_json  = os.path.join(script_dir, self.config.get("onnx_json_filename", "overwatch.onnx.json"))
                onnx_model = os.path.join(script_dir, self.config.get("onnx_model_filename",  "overwatch.onnx"))
                out_dir    = self.config.get("ogg_dir", "tts_ogg")
                os.makedirs(out_dir, exist_ok=True)

                filename = f"{uuid.uuid4().hex}.ogg"
                ogg_path = os.path.join(out_dir, filename)

                cmd_piper = [piper_exe, "-m", onnx_model, "--json-input", "--output_raw"]
                if self.debug:
                    cmd_piper.insert(3, "--debug")
                cmd_ffmpeg = [
                    "ffmpeg", "-y", "-f", "s16le", "-ar", "22050", "-ac", "1",
                    "-i", "pipe:0", "-c:a", "libopus", ogg_path
                ]

                payload = json.dumps({"text": text, "config": onnx_json, "model": onnx_model}).encode("utf-8")
                self.log(f"[TTS file] Generating OGG for: {text!r}", "INFO")

                with tts_lock:
                    p1 = subprocess.Popen(cmd_piper, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=(subprocess.PIPE if self.debug else subprocess.DEVNULL))
                    p2 = subprocess.Popen(cmd_ffmpeg, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                p1.stdin.write(payload)
                p1.stdin.close()

                while True:
                    chunk = p1.stdout.read(4096)
                    if not chunk:
                        break
                    if self.volume != 1.0:
                        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) * self.volume
                        chunk = np.clip(arr, -32768,32767).astype(np.int16).tobytes()
                    p2.stdin.write(chunk)

                p2.stdin.close()
                p1.wait(); p2.wait()

                if self.debug and p1.stderr:
                    err = p1.stderr.read().decode(errors="ignore").strip()
                    if err:
                        self.log(f"[Piper STDERR] {err}", "ERROR")

                self._ogg_q.put(ogg_path)
                ttl_seconds = int(self.config.get("ogg_ttl", 300))
                timer = threading.Timer(ttl_seconds, lambda p=ogg_path: os.path.exists(p) and os.remove(p))
                timer.daemon = True
                timer.start()

            except Exception as e:
                self.log(f"TTS file error: {e}", "ERROR")
            finally:
                self._file_q.task_done()

        self.log("TTS file‐worker exiting.", "DEBUG")
