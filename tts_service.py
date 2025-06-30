# tts_service.py

import threading
import queue
import subprocess
import json
import os
import uuid
import numpy as np
import re

# lock to prevent Piper/process races
tts_lock = threading.Lock()

class TTSManager:
    """
    Piper → live‐playback or file‐output TTS manager.
    Automatically splits long text into manageable chunks.
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

        # Workers
        threading.Thread(target=self._live_worker, daemon=True).start()
        threading.Thread(target=self._file_worker, daemon=True).start()

    def set_mode(self, mode: str):
        if mode not in ("live", "file"):
            raise ValueError("TTSManager.set_mode: must be 'live' or 'file'")
        self.log(f"TTS mode → {mode}", "INFO")
        self._mode = mode
        # flush the opposite queue
        if mode == "live":
            while not self._file_q.empty():
                try: self._file_q.get_nowait()
                except queue.Empty: break
        else:
            while not self._live_q.empty():
                try: self._live_q.get_nowait()
                except queue.Empty: break

    def _split_text(self, text: str) -> list[str]:
        """
        1) Split on blank lines (paragraphs)
        2) Within each paragraph, split on sentence boundaries or any newline
        3) Pack sentences into chunks ≤ max_chunk_size
        4) If still too large, hard-slice into max_chunk_size windows
        """
        # normalize line endings
        import re

        # 1) Break into paragraphs
        paras = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
        chunks: list[str] = []

        for para in paras:
            # 2) Split on sentence punctuation OR on any newline/bullet
            parts = re.split(r'(?<=[\.\?\!])\s+|\n+|(?<=\u2022)\s*', para)
            # you can also include r'(?<=^[-\*\u2022]\s*)' if you want bullets
            buffer = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # if adding this part would overflow, flush buffer first
                if buffer and len(buffer) + 1 + len(part) > self.max_chunk_size:
                    chunks.append(buffer)
                    buffer = ""

                # if the single part alone fits, accumulate it
                if len(part) <= self.max_chunk_size:
                    buffer = f"{buffer} {part}".strip() if buffer else part
                else:
                    # 3) part itself is too big—even on its own—so hard-slice
                    if buffer:
                        chunks.append(buffer)
                        buffer = ""
                    for i in range(0, len(part), self.max_chunk_size):
                        slice_ = part[i : i + self.max_chunk_size].strip()
                        if slice_:
                            chunks.append(slice_)
            # at paragraph end, flush any remaining buffer
            if buffer:
                chunks.append(buffer)

        return chunks

    def enqueue(self, text: str):
        text = text.strip().replace("*", "")
        if not text:
            return

        # split once to know total
        parts = self._split_text(text)
        total = len(parts)

        for idx, chunk in enumerate(parts, start=1):
            if self._mode == "live":
                self.log(f"Enqueue live TTS chunk {idx}/{total}: {chunk!r}", "DEBUG")
                self._live_q.put(chunk)
            else:
                self.log(f"Enqueue file TTS chunk {idx}/{total}: {chunk!r}", "DEBUG")
                self._file_q.put(chunk)


    def wait_for_latest_ogg(self, timeout: float):
        """
        Block up to `timeout` seconds for the next .ogg file path.
        """
        return self._ogg_q.get(timeout=timeout)

    def stop(self):
        self._running = False
        # unblock workers
        self._live_q.put(None)
        self._file_q.put(None)

    # ───────────────────────────────────────────────────────────────────────
    # live‐mode: exactly as before
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
        if self.audio_service:
            self.audio_service.suspend()
        try:
            # build Piper + aplay commands
            script_dir = os.path.dirname(__file__)
            piper_exe  = os.path.join(script_dir, "piper", "piper")
            onnx_json  = os.path.join(
                script_dir,
                self.config.get("onnx_json", "glados_piper_medium.onnx.json")
            )
            onnx_model = os.path.join(
                script_dir,
                self.config.get("onnx_model",  "glados_piper_medium.onnx")
            )

            cmd_piper = [piper_exe, "-m", onnx_model, "--json-input", "--output_raw"]
            if self.debug:
                cmd_piper.insert(3, "--debug")
            cmd_aplay = ["aplay", "-r", "22050", "-f", "S16_LE"]

            payload = json.dumps({
                "text":   text,
                "config": onnx_json,
                "model":  onnx_model
            }).encode("utf-8")

            self.log(f"[TTS live] Speaking: {text!r}", "INFO")

            with tts_lock:
                p1 = subprocess.Popen(
                    cmd_piper,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=(subprocess.PIPE if self.debug else subprocess.DEVNULL)
                )
                p2 = subprocess.Popen(cmd_aplay, stdin=subprocess.PIPE)

            # send text to Piper
            p1.stdin.write(payload)
            p1.stdin.close()

            # stream Piper’s PCM into aplay
            def _stream_audio():
                while True:
                    chunk = p1.stdout.read(4096)
                    if not chunk:
                        break
                    if self.volume != 1.0:
                        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) * self.volume
                        chunk = np.clip(arr, -32768,32767).astype(np.int16).tobytes()
                    p2.stdin.write(chunk)
                p2.stdin.close()

            streamer = threading.Thread(target=_stream_audio, daemon=True)
            streamer.start()

            # wait for Piper to finish generating audio
            p1.wait()
            # wait for the streaming thread to drain Piper’s output into aplay
            streamer.join()
            # wait for aplay to finish playback
            p2.wait()

            if self.debug:
                err = p1.stderr.read().decode(errors="ignore").strip()
                if err:
                    self.log(f"[Piper STDERR] {err}", "ERROR")

        finally:
            if self.audio_service:
                self.audio_service.resume()

    # ───────────────────────────────────────────────────────────────────────
    # file‐mode: generate .ogg, queue file path
    # ───────────────────────────────────────────────────────────────────────
    def _file_worker(self):
        self.log("TTS file‐worker started.", "DEBUG")
        while self._running:
            text = self._file_q.get()
            if text is None:
                break

            try:
                # prepare paths
                script_dir = os.path.dirname(__file__)
                piper_exe  = os.path.join(script_dir, "piper", "piper")
                onnx_json  = os.path.join(script_dir, self.config.get("onnx_json", "glados_piper_medium.onnx.json"))
                onnx_model = os.path.join(script_dir, self.config.get("onnx_model",  "glados_piper_medium.onnx"))
                out_dir    = self.config.get("ogg_dir", "tts_ogg")
                os.makedirs(out_dir, exist_ok=True)

                filename = f"{uuid.uuid4().hex}.ogg"
                ogg_path = os.path.join(out_dir, filename)

                # Piper → raw PCM
                cmd_piper = [piper_exe, "-m", onnx_model, "--json-input", "--output_raw"]
                if self.debug:
                    cmd_piper.insert(3, "--debug")

                # FFmpeg: s16le PCM @22050 → Ogg/Opus
                cmd_ffmpeg = [
                    "ffmpeg", "-y",
                    "-f", "s16le", "-ar", "22050", "-ac", "1",
                    "-i", "pipe:0",
                    "-c:a", "libopus",
                    ogg_path
                ]

                payload = json.dumps({
                    "text":   text,
                    "config": onnx_json,
                    "model":  onnx_model
                }).encode("utf-8")

                self.log(f"[TTS file] Generating OGG for: {text!r}", "INFO")

                with tts_lock:
                    p1 = subprocess.Popen(cmd_piper, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=(subprocess.PIPE if self.debug else subprocess.DEVNULL))
                    p2 = subprocess.Popen(cmd_ffmpeg, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # feed Piper payload
                p1.stdin.write(payload)
                p1.stdin.close()

                # pipe Piper’s PCM into ffmpeg
                while True:
                    chunk = p1.stdout.read(4096)
                    if not chunk:
                        break
                    # apply volume
                    if self.volume != 1.0:
                        arr   = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) * self.volume
                        chunk = np.clip(arr, -32768,32767).astype(np.int16).tobytes()
                    p2.stdin.write(chunk)

                p2.stdin.close()
                p1.wait()
                p2.wait()

                if self.debug and p1.stderr:
                    err = p1.stderr.read().decode(errors="ignore").strip()
                    if err:
                        self.log(f"[Piper STDERR] {err}", "ERROR")

                # enqueue the result path for retrieval
                self._ogg_q.put(ogg_path)

            except Exception as e:
                self.log(f"TTS file error: {e}", "ERROR")
            finally:
                self._file_q.task_done()

        self.log("TTS file‐worker exiting.", "DEBUG")
