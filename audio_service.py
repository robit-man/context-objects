# audio_service.py

import queue
import threading
import time
import numpy as np
import difflib
import whisper
import sounddevice as sd
from sounddevice import InputStream
from scipy.signal import butter, lfilter

class AudioService:
    """
    Handles microphone capture, optional audio enhancement,
    and dual-model Whisper consensus transcription.
    """
    def __init__(
        self,
        sample_rate: int,
        rms_threshold: float,
        silence_duration: float,
        consensus_threshold: float,
        enable_denoise: bool,
        on_transcription: callable,
        logger: callable,
        cfg: dict,
        denoise_fn: callable = None,
    ):
        self.sample_rate      = sample_rate
        self.rms_threshold    = rms_threshold
        self.silence_duration = silence_duration
        self.consensus_thresh = consensus_threshold
        self.enable_denoise   = enable_denoise
        self.on_transcription = on_transcription
        self.log              = logger
        self.config           = cfg
        self._denoise_fn      = denoise_fn

        self.log("AudioService: loading Whisper models…", "INFO")
        self.model_base   = whisper.load_model("base")
        self.model_medium = whisper.load_model("medium")
        self.log("AudioService: Whisper models loaded.", "SUCCESS")

        self._audio_q  = queue.Queue()
        self._stop_evt = threading.Event()
        self._stream   = None
        self._worker   = threading.Thread(target=self._listen_loop, daemon=True)

    def start(self):
        """Begin capturing audio and running transcription loop."""
        self.log("AudioService: starting capture…", "INFO")
        self._stop_evt.clear()
        self._stream = InputStream(
            samplerate=self.sample_rate,
            blocksize=1024,
            channels=1,
            callback=self._audio_callback
        )
        self._stream.start()
        self._worker.start()

    def stop(self):
        """Stop capture and transcription."""
        self.log("AudioService: stopping capture…", "INFO")
        self._stop_evt.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._worker.join()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.log(f"AudioService: status {status}", "WARNING")
        # flatten to 1-D
        self._audio_q.put(indata[:, 0].copy())

    def _listen_loop(self):
        self.log("AudioService: listen loop started.", "DEBUG")
        while not self._stop_evt.is_set():
            buffers = []
            silence_start = None

            # accumulate until silence_duration of RMS< threshold
            while not self._stop_evt.is_set():
                buf = self._audio_q.get()
                rms = float(np.sqrt(np.mean(buf**2)))
                if rms < self.rms_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= self.silence_duration:
                        break
                else:
                    silence_start = None
                    buffers.append(buf)
                self._audio_q.task_done()

            if not buffers:
                continue

            audio_block = np.concatenate(buffers).astype(np.float32)
            if self.enable_denoise and self._denoise_fn:
                audio_block = self._denoise_fn(audio_block, self.sample_rate)

            text = self._transcribe_consensus(audio_block)
            if text:
                self.log(f"AudioService: recognized: {text!r}", "INFO")
                try:
                    self.on_transcription(text)
                except Exception as ex:
                    self.log(f"AudioService callback error: {ex}", "ERROR")

        self.log("AudioService: listen loop exiting.", "DEBUG")

    def _transcribe_consensus(self, audio: np.ndarray) -> str:
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < self.rms_threshold:
            return ""
        tb = tm = ""
        def run_b():
            nonlocal tb
            try:
                tb = self.model_base.transcribe(audio, language="en")["text"].strip()
            except Exception as e:
                self.log(f"Base transcription error: {e}", "ERROR")
        def run_m():
            nonlocal tm
            try:
                tm = self.model_medium.transcribe(audio, language="en")["text"].strip()
            except Exception as e:
                self.log(f"Medium transcription error: {e}", "ERROR")

        t1 = threading.Thread(target=run_b)
        t2 = threading.Thread(target=run_m)
        t1.start(); t2.start()
        t1.join(); t2.join()

        if not tb or not tm:
            return ""
        sim = difflib.SequenceMatcher(None, tb, tm).ratio()
        self.log(f"AudioService: transcription similarity {sim:.2f}", "DEBUG")
        return tb if sim >= self.consensus_thresh else ""
