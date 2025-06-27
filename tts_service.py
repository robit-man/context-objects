# tts_service.py

import threading
import queue
import subprocess
import json
import os
import numpy as np

# lock to prevent Piper/process races
tts_lock = threading.Lock()

class TTSManager:
    """
    Live‐only Piper→aplay TTS manager.
    You must pass in your own logger, config dict, and (optionally) AudioService.
    """
    def __init__(self, logger: callable, cfg: dict, audio_service=None):
        self.log            = logger
        self.config         = cfg
        self.audio_service  = audio_service
        self.volume         = cfg.get("tts_volume", 0.2)
        self.debug          = cfg.get("tts_debug", False)
        self._live_q        = queue.Queue()
        self._running       = True

        threading.Thread(target=self._live_worker, daemon=True).start()

    def set_mode(self, mode: str):
        if mode != "live":
            raise ValueError("TTSManager only supports live mode")
        self.log(f"TTS mode → {mode}", "INFO")
        while not self._live_q.empty():
            try: self._live_q.get_nowait()
            except queue.Empty: break

    def enqueue(self, text: str):
        text = text.strip()
        if not text:
            return
        self.log(f"Enqueue live TTS: {text!r}", "DEBUG")
        self._live_q.put(text)

    def stop(self):
        self._running = False
        self._live_q.put(None)

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
        # suspend mic while we speak
        if self.audio_service:
            self.audio_service.suspend()

        try:
            volume     = self.volume
            script_dir = os.path.dirname(__file__)
            piper_exe  = os.path.join(script_dir, "piper", "piper")
            onnx_json  = os.path.join(script_dir, self.config.get("onnx_json", "glados_piper_medium.onnx.json"))
            onnx_model = os.path.join(script_dir, self.config.get("onnx_model",  "glados_piper_medium.onnx"))

            cmd_piper = [piper_exe, "-m", onnx_model, "--json-input", "--output_raw"]
            if self.debug:
                cmd_piper.insert(3, "--debug")
            cmd_aplay = ["aplay", "-r", "22050", "-f", "S16_LE"]

            payload = json.dumps({
                "text":   text,
                "config": onnx_json,
                "model":  onnx_model
            }).encode("utf-8")

            self.log(f"[TTS live] Playing: {text!r}", "INFO")

            with tts_lock:
                p1 = subprocess.Popen(
                    cmd_piper,
                    stdin = subprocess.PIPE,
                    stdout= subprocess.PIPE,
                    stderr= (subprocess.PIPE if self.debug else subprocess.DEVNULL)
                )
                p2 = subprocess.Popen(cmd_aplay, stdin=subprocess.PIPE)

            p1.stdin.write(payload)
            p1.stdin.close()

            def _stream_audio():
                while True:
                    chunk = p1.stdout.read(4096)
                    if not chunk:
                        break
                    if volume != 1.0:
                        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) * volume
                        chunk = np.clip(arr, -32768, 32767).astype(np.int16).tobytes()
                    p2.stdin.write(chunk)
                p2.stdin.close()

            streamer = threading.Thread(target=_stream_audio, daemon=True)
            streamer.start()

            p1.wait()
            streamer.join()

            if self.debug:
                err = p1.stderr.read().decode(errors="ignore").strip()
                if err:
                    self.log(f"[Piper STDERR] {err}", "ERROR")

        finally:
            # resume mic ASAP
            if self.audio_service:
                self.audio_service.resume()
