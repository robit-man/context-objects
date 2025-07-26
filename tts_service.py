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
import math
from scipy.signal import resample_poly

# lock to prevent Piper/process races
tts_lock = threading.Lock()

class TTSManager:
    """
    Piper → live‐playback or file‐output TTS manager.
    Splits text into chunks, mutes the mic during live TTS so
    AudioService never hears its own speech, pushes resampled
    PCM for optional adaptive LMS cancellation, and supports
    interrupting in‑flight speech.
    """

    def __init__(self, logger: callable, cfg: dict, audio_service=None):
        self.log            = logger
        self.config         = cfg
        self.audio_service  = audio_service
        self.volume         = cfg.get("tts_volume", 0.2)
        self.debug          = cfg.get("tts_debug", False)

        # chunk size in characters
        self.max_chunk_size = cfg.get("tts_max_chunk_size", 500)

        # Queues for live vs file modes
        self._live_q = queue.Queue()
        self._file_q = queue.Queue()
        self._ogg_q  = queue.Queue()

        self._mode    = "live"
        self._running = True

        # token-buffer state
        self._token_buffer  = ""
        self._token_lock    = threading.Lock()
        self._flush_timer   = None
        self._flush_timeout = cfg.get("tts_token_flush_timeout", 1)

        # calibration flag
        self._calibrated = False

        # interrupt control
        self._interrupt_event = threading.Event()
        self._current_procs   = []

        # start worker threads
        threading.Thread(target=self._live_worker, daemon=True).start()
        threading.Thread(target=self._file_worker, daemon=True).start()

        # inform audio service of ourselves if provided
        if self.audio_service and hasattr(self.audio_service, 'set_tts_manager'):
            self.audio_service.set_tts_manager(self)

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

    def interrupt(self):
        """
        Immediately cut off any in‑flight Piper/aplay processes.
        """
        self.log("TTSManager: interrupt requested", "INFO")
        self._interrupt_event.set()
        for p in self._current_procs:
            try: p.terminate()
            except: pass

    def _flush_buffer(self):
        with self._token_lock:
            text = self._token_buffer.strip()
            if text:
                self.enqueue(text)
            self._token_buffer = ""
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None

    def flush(self):
        """Flush any buffered tokens immediately."""
        self._flush_buffer()

    def token_sink(self):
        """
        Returns a callback(token:str) that buffers tokens into sentences,
        enqueues them, and flushes on None.
        """
        sentence_end = re.compile(r'([\.!?])')
        buf = []
        def sink(token):
            if token is None:
                text = ''.join(buf).strip()
                if text:
                    self.enqueue(text)
                buf.clear()
                return
            buf.append(token)
            joined = ''.join(buf)
            m = sentence_end.search(joined)
            if m:
                end = m.end()
                sentence = joined[:end].strip()
                self.enqueue(sentence)
                buf.clear()
                rem = joined[end:]
                if rem:
                    buf.append(rem)
        return sink

    def _split_text(self, text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
        chunks: list[str] = []
        for para in paras:
            parts = re.split(r'(?<=[\.!?])\s+|\n+', para)
            buf = ""
            for part in parts:
                part = part.strip()
                if not part: continue
                if buf and len(buf)+1+len(part) > self.max_chunk_size:
                    chunks.append(buf); buf = ""
                if len(part) <= self.max_chunk_size:
                    buf = (buf+" "+part).strip() if buf else part
                else:
                    if buf:
                        chunks.append(buf); buf = ""
                    for i in range(0, len(part), self.max_chunk_size):
                        slice_ = part[i:i+self.max_chunk_size].strip()
                        if slice_: chunks.append(slice_)
            if buf:
                chunks.append(buf)
        return chunks

    def enqueue(self, text: str):
        """
        Queue a sentence or chunk for TTS (live vs file).
        """
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE
        )
        clean = emoji_pattern.sub('', text).strip().replace("*","")
        if not clean:
            return
        parts = self._split_text(clean)
        total = len(parts)
        for idx, chunk in enumerate(parts, start=1):
            tag = f"{idx}/{total}"
            if self._mode=="live":
                self.log(f"Enqueue live TTS chunk {tag}: {chunk!r}", "DEBUG")
                self._live_q.put(chunk)
            else:
                self.log(f"Enqueue file TTS chunk {tag}: {chunk!r}", "DEBUG")
                self._file_q.put(chunk)

    def wait_for_latest_ogg(self, timeout: float):
        """Block up to `timeout` seconds for the next .ogg filepath."""
        return self._ogg_q.get(timeout=timeout)

    def stop(self):
        """Shut down workers and timers."""
        self._running = False
        self._live_q.put(None)
        self._file_q.put(None)
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None

    # ───────────────────────────────────────────────────────────────────────────
    # Live‐mode worker
    def _live_worker(self):
        self.log("TTS live‑worker started.", "DEBUG")
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
        self.log("TTS live‑worker exiting.", "DEBUG")

    def _do_live(self, text: str):
        import subprocess, json, time, shutil, os
        script_dir = os.path.dirname(__file__)
        piper_exe  = os.path.join(script_dir, "piper", self.config.get("piper_executable","piper"))
        onnx_json  = os.path.join(script_dir, self.config.get("onnx_json_filename","overwatch.onnx.json"))
        onnx_model = os.path.join(script_dir, self.config.get("onnx_model_filename","overwatch.onnx"))

        def _speak_phrase(phrase: str):
            # clear any prior interrupt
            self._interrupt_event.clear()
            # launch Piper → output_raw
            cmd_piper = [piper_exe, "-m", onnx_model, "--json-input", "--output_raw"]
            if self.debug:
                cmd_piper.insert(3, "--debug")
            if shutil.which("aplay"):
                cmd_play = ["aplay","-r","22050","-f","S16_LE"]
            elif shutil.which("ffplay"):
                cmd_play = ["ffplay","-autoexit","-nodisp","-f","s16le","-ar","22050","-i","pipe:0"]
            else:
                raise RuntimeError("Install 'aplay' or 'ffplay'.")
            payload = json.dumps({"text": phrase,"config":onnx_json,"model":onnx_model}).encode("utf-8")

            with tts_lock:
                p1 = subprocess.Popen(cmd_piper, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=(subprocess.PIPE if self.debug else subprocess.DEVNULL))
                p2 = subprocess.Popen(cmd_play, stdin=subprocess.PIPE)
                self._current_procs = [p1, p2]

            p1.stdin.write(payload); p1.stdin.close()

            def _stream_audio():
                while True:
                    if self._interrupt_event.is_set():
                        break
                    raw = p1.stdout.read(4096)
                    if not raw:
                        break
                    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    frames = arr/32768.0 * self.volume

                    svc = getattr(self, "audio_service", None)
                    if svc and self._calibrated:
                        try:
                            src_sr, tgt_sr = 22050, svc.sample_rate
                            g = math.gcd(src_sr, tgt_sr)
                            up, down = tgt_sr//g, src_sr//g
                            frames_rs = resample_poly(frames, up, down)
                            svc.push_cancellation(frames_rs.astype(np.float32))
                        except:
                            pass

                    out = np.clip(frames*32767.0, -32768, 32767).astype(np.int16)
                    p2.stdin.write(out.tobytes()); p2.stdin.flush()

                # cleanup
                try: p2.stdin.close()
                except: pass

            streamer = threading.Thread(target=_stream_audio, daemon=True)
            streamer.start()
            p1.wait()
            streamer.join()
            p2.wait()

            # if interrupted, skip reading stderr
            if self.debug and p1.stderr and not self._interrupt_event.is_set():
                err = p1.stderr.read().decode(errors="ignore").strip()
                if err:
                    self.log(f"[Piper STDERR] {err}", "ERROR")

        svc = getattr(self, "audio_service", None)

        # actual TTS
        if svc:
            svc.mute_tts()
        self.log(f"[TTS] Speaking: {text!r}", "INFO")
        _speak_phrase(text)
        if svc:
            svc.unmute_tts()

    # ───────────────────────────────────────────────────────────────────────────
    # File‐mode worker
    def _file_worker(self):
        self.log("TTS file‑worker started.", "DEBUG")
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

        self.log("TTS file‑worker exiting.", "DEBUG")
