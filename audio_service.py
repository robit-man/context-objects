# audio_service.py

import threading
import time
import numpy as np
import sounddevice as sd
from sounddevice import InputStream
from scipy.signal import resample_poly
import whisper
from difflib import SequenceMatcher

# ─── Load Whisper models once ────────────────────────────────────────────────
_MODEL_BASE  = whisper.load_model("base")
_MODEL_SMALL = whisper.load_model("small")


class AudioService:
    """
    Captures mic audio alongside the system's 'monitor' (loopback),
    performs an automated self‑calibration (ambient + test‑tone echo),
    then streams overlapping audio chunks into Whisper for
    token‑by‑token transcription with dual‑model consensus,
    while actively cancelling any TTS waveform you push in.
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
        assembler=None,
        denoise_fn: callable = None,
    ):
        # configuration
        self.sample_rate         = sample_rate
        self.rms_threshold       = rms_threshold
        self.silence_duration    = silence_duration
        self.consensus_threshold = consensus_threshold
        self.enable_denoise      = enable_denoise
        self._denoise_fn         = denoise_fn
        self.on_transcription    = on_transcription
        self.log                 = logger

        # streaming parameters
        self.stream_window       = cfg.get("stream_window", 3.0)
        self.stream_step         = cfg.get("stream_step",   0.5)
        self.delay_alpha         = cfg.get("delay_alpha", 0.1)

        # internal buffers
        self._buffer             = np.zeros(0, dtype=np.float32)
        self._buffer_lock        = threading.Lock()
        self._monitor_buffer     = np.zeros(0, dtype=np.float32)
        self._monitor_lock       = threading.Lock()
        self._cancel_buffer      = np.zeros(0, dtype=np.float32)
        self._cancel_lock        = threading.Lock()
        self._echo_profile       = None

        # dynamic cancellation delay (in samples)
        self.cancel_delay        = 0.0

        self._last_text          = ""
        self._stop_evt           = threading.Event()
        self._stream             = None
        self._monitor_stream     = None
        self._worker             = None

        # Whisper models
        self.log("AudioService: loading Whisper models…", "INFO")
        self.model_base          = _MODEL_BASE
        self.model_small         = _MODEL_SMALL
        self.log("AudioService: Whisper models ready.", "SUCCESS")


    def push_cancellation(self, cancelled_frames: np.ndarray):
        # e.g. append into a deque or array for your mic callback to consume
        with self._cancel_lock:
            self._cancel_buffer = np.concatenate((self._cancel_buffer, cancelled_frames))
            # keep only the last N samples:
            max_samps = int(self.sample_rate * self.stream_window)
            if len(self._cancel_buffer) > max_samps:
                self._cancel_buffer = self._cancel_buffer[-max_samps:]



    def _find_monitor_device(self) -> int | None:
        default = sd.default.device
        if isinstance(default, (list, tuple)) and len(default) == 2:
            _, out_idx = default
        elif hasattr(default, "output"):
            out_idx = default.output
        else:
            out_idx = None

        devs = sd.query_devices()
        # same-hostapi monitor
        if isinstance(out_idx, int) and 0 <= out_idx < len(devs):
            hostapi = devs[out_idx]["hostapi"]
            for i, d in enumerate(devs):
                if (d["hostapi"] == hostapi
                    and d["max_input_channels"] > 0
                    and "monitor" in d["name"].lower()):
                    return i
        # any “monitor”
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and "monitor" in d["name"].lower():
                return i
        return None


    def _calibrate_echo(self):
        """Play a test tone and capture its echo from the monitor buffer."""
        tone_dur = 1.0
        freq     = 440.0
        self.log("AudioService: calibrating echo via test tone…", "INFO")

        t    = np.linspace(0, tone_dur, int(self.sample_rate * tone_dur), False)
        tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        with self._monitor_lock:
            self._monitor_buffer = np.zeros(0, dtype=np.float32)

        sd.play(tone, self.sample_rate)
        sd.wait()
        time.sleep(0.2)

        with self._monitor_lock:
            prof = self._monitor_buffer.copy()

        self._echo_profile = prof
        self.log(f"AudioService: captured echo profile ({len(prof)} samples)", "INFO")


    def start(self):
        """Run startup: ambient noise → monitor → echo → mic → streaming."""
        # ambient calibration
        self.log("AudioService: calibrating ambient noise…", "INFO")
        try:
            sd.default.samplerate = self.sample_rate
            sd.default.channels   = 1
            amb = sd.rec(int(self.sample_rate * 1.0), dtype="float32")
            sd.wait()
            ambient = float(np.sqrt(np.mean(amb**2)))
            self.rms_threshold = max(self.rms_threshold, ambient * 1.5)
            self.log(f"AudioService: rms_threshold={self.rms_threshold:.6f}", "INFO")
        except Exception as e:
            self.log(f"AudioService: ambient calibration failed: {e}", "WARNING")

        # open monitor
        mon_idx = self._find_monitor_device()
        if mon_idx is not None:
            try:
                self._monitor_stream = InputStream(
                    device    = mon_idx,
                    samplerate= self.sample_rate,
                    blocksize = 1024,
                    channels  = 1,
                    callback  = self._monitor_callback
                )
                self._monitor_stream.start()
                self.log(f"AudioService: opened monitor #{mon_idx}", "INFO")
                self._calibrate_echo()
            except Exception as e:
                self.log(f"AudioService: monitor open failed: {e}", "WARNING")
        else:
            self.log("AudioService: no monitor found; TTS won't be canceled", "WARNING")

        # open mic
        self.log("AudioService: starting mic capture…", "INFO")
        self._stop_evt.clear()
        try:
            self._stream = InputStream(
                samplerate= self.sample_rate,
                blocksize = 1024,
                channels  = 1,
                callback  = self._audio_callback
            )
            self._stream.start()
        except Exception as e:
            self.log(f"AudioService: mic open failed @ {self.sample_rate} Hz: {e}", "WARNING")
            self._stream = InputStream(callback=self._audio_callback)
            self._stream.start()

        dev = sd.default.device
        sr  = getattr(self._stream, "samplerate", self.sample_rate)
        self.log(f"AudioService: mic on device {dev} @ {sr:.0f} Hz", "INFO")

        # start transcription loop
        self._worker = threading.Thread(target=self._stream_loop, daemon=True)
        self._worker.start()


    def stop(self):
        self.log("AudioService: stopping capture…", "INFO")
        self._stop_evt.set()
        if self._stream:
            self._stream.stop(); self._stream.close()
        if self._monitor_stream:
            self._monitor_stream.stop(); self._monitor_stream.close()
        if self._worker:
            self._worker.join()


    def suspend(self):
        self._stop_evt.set()


    def resume(self):
        if self._stop_evt.is_set():
            self._stop_evt.clear()
            self._worker = threading.Thread(target=self._stream_loop, daemon=True)
            self._worker.start()


    def _monitor_callback(self, indata, frames, time_info, status):
        if status:
            self.log(f"AudioService(monitor): status {status}", "WARNING")
        buf = indata[:,0].astype(np.float32)
        with self._monitor_lock:
            self._monitor_buffer = np.concatenate((self._monitor_buffer, buf))
            maxs = int(self.stream_window * self.sample_rate)
            if len(self._monitor_buffer) > maxs:
                self._monitor_buffer = self._monitor_buffer[-maxs:]


    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.log(f"AudioService: status {status}", "WARNING")
        buf = indata[:,0].astype(np.float32)

        # optional denoise
        if self.enable_denoise and self._denoise_fn:
            buf = self._denoise_fn(buf, getattr(self._stream, "samplerate", self.sample_rate))

        # subtract echo profile if set
        if self._echo_profile is not None:
            ep = self._echo_profile
            seg = ep[-len(buf):] if len(ep) >= len(buf) else np.pad(ep, (len(buf)-len(ep),0))
            buf = buf - seg

        # subtract live TTS via cancel buffer + delay
        with self._cancel_lock:
            cb = self._cancel_buffer.copy()
        d = max(0, int(round(self.cancel_delay)))
        if cb.size:
            # pad cb by d zeros in front, then take last buf‑length samples
            c_full = np.concatenate((np.zeros(d, dtype=np.float32), cb))
            if c_full.size >= len(buf):
                seg = c_full[-len(buf):]
            else:
                seg = np.pad(c_full, (len(buf)-c_full.size,0))
            buf = buf - seg

        # enqueue for background transcription
        with self._buffer_lock:
            self._buffer = np.concatenate((self._buffer, buf))
            maxs = int(self.stream_window * self.sample_rate)
            if len(self._buffer) > maxs:
                self._buffer = self._buffer[-maxs:]


    def _stream_loop(self):
        sr = getattr(self._stream, "samplerate", self.sample_rate)
        # for display
        width = 31
        center = width//2
        max_disp = int(self.sample_rate * 0.2)

        while not self._stop_evt.is_set():
            time.sleep(self.stream_step)

            # snapshot buffers
            with self._buffer_lock:
                chunk = self._buffer.copy()
            with self._cancel_lock:
                cb = self._cancel_buffer.copy()

            # alignment measurement
            if cb.size >= chunk.size and chunk.size > 0:
                seg = cb[-chunk.size:]
                corr = np.correlate(seg, chunk, mode='full')
                lag = corr.argmax() - (chunk.size - 1)
                # smooth
                self.cancel_delay = (1 - self.delay_alpha)*self.cancel_delay + self.delay_alpha*lag
                # draw meter
                disp = int(np.clip(self.cancel_delay/max_disp, -1, 1)*center)
                meter = ['-']*width
                meter[center] = '|'
                pos = center + disp
                if 0 <= pos < width:
                    meter[pos] = '^'
                print(f"\rAlign: {''.join(meter)} Δ={self.cancel_delay/self.sample_rate*1000:5.1f}ms", end="", flush=True)

            # skip if too short or quiet
            if chunk.size < sr * 0.2:
                continue
            rms_chunk = float(np.sqrt(np.mean(chunk**2)))
            if rms_chunk < self.rms_threshold:
                continue

            # resample → 16 kHz
            proc = chunk
            if sr != 16000:
                proc = resample_poly(proc, 16000, int(sr))

            # dual‑model Whisper
            try:
                tb = self.model_base.transcribe(proc, language="en", fp16=False)["text"].strip()
                tm = self.model_small.transcribe(proc, language="en", fp16=False)["text"].strip()
            except Exception as e:
                self.log(f"Whisper error: {e}", "ERROR")
                continue

            if not tb or not tm:
                continue
            sim = SequenceMatcher(None, tb, tm).ratio()
            if sim < self.consensus_threshold:
                self.log(f"Consensus {sim:.2f} < {self.consensus_threshold}", "DEBUG")
                continue

            # incremental diff
            if tb.startswith(self._last_text):
                suffix = tb[len(self._last_text):].strip()
            else:
                suffix = tb

            if suffix:
                for tok in suffix.split():
                    print()  # newline
                    print(tok, end=" ", flush=True)
                print()
                try:
                    self.on_transcription(tb)
                except Exception as cb:
                    self.log(f"AudioService callback error: {cb}", "ERROR")

            self._last_text = tb

        self.log("AudioService: stream loop exiting.", "DEBUG")
