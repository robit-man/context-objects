# audio_service.py

import time
import threading
from difflib import SequenceMatcher

import numpy as np
import sounddevice as sd
from sounddevice import InputStream
from scipy.signal import resample_poly, butter, lfilter

import torch
import whisper

from lms import StreamingLMSFilter


# ─────────────────────────────────────────────────────────────────────────────
# Process-wide Whisper cache with refcounting
# ─────────────────────────────────────────────────────────────────────────────
_WHISPER_LOCK = threading.Lock()
_WHISPER_POOL: dict[tuple[str, str], object] = {}   # (model_name, device) -> model
_WHISPER_REFS: dict[tuple[str, str], int] = {}      # refcounts


def _device_or_auto(dev: str | None) -> str:
    """
    Normalize device: "cpu", "cuda", or "auto" -> resolve to cpu/cuda.
    Default: auto (cuda if available else cpu).
    """
    if dev is None or dev == "" or dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev not in ("cpu", "cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return dev


def _get_whisper_model(name: str, device: str | None = None):
    """
    Return a process-global Whisper model instance for (name, device).
    Increments a refcount; pair with _release_whisper_model on teardown.
    """
    dev = _device_or_auto(device)
    key = (name, dev)
    with _WHISPER_LOCK:
        if key in _WHISPER_POOL:
            _WHISPER_REFS[key] += 1
            return _WHISPER_POOL[key]
        model = whisper.load_model(name, device=dev)
        _WHISPER_POOL[key] = model
        _WHISPER_REFS[key] = 1
        return model


def _release_whisper_model(name: str, device: str | None = None):
    """
    Decrement refcount; when it hits 0, move to CPU and free VRAM.
    Safe to call multiple times.
    """
    dev = _device_or_auto(device)
    key = (name, dev)
    with _WHISPER_LOCK:
        if key not in _WHISPER_POOL:
            return
        _WHISPER_REFS[key] -= 1
        if _WHISPER_REFS[key] > 0:
            return
        # Drop the model and free CUDA memory
        try:
            _WHISPER_POOL[key].to("cpu")
        except Exception:
            pass
        try:
            del _WHISPER_POOL[key]
            del _WHISPER_REFS[key]
        except Exception:
            pass
    # Vacate VRAM outside the lock
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_all_whisper_models():
    """Force-release everything (used on hot-reload)."""
    with _WHISPER_LOCK:
        keys = list(_WHISPER_POOL.keys())
    for name, dev in keys:
        _release_whisper_model(name, dev)


# ─────────────────────────────────────────────────────────────────────────────
# DSP helpers
# ─────────────────────────────────────────────────────────────────────────────
def _highpass(data: np.ndarray, fs: int, cutoff: float = 100.0) -> np.ndarray:
    """
    2nd-order Butterworth high-pass filter at `cutoff` Hz.
    Always returns float32.
    """
    b, a = butter(2, cutoff / (fs / 2), btype="high", analog=False)
    y = lfilter(b, a, data)
    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Service
# ─────────────────────────────────────────────────────────────────────────────
class AudioService:
    """
    Mic + loopback capture with echo calibration.

    VAD-driven chunking:
      • Start a chunk after ~150 ms of speech.
      • While capturing, print live words (from last 2.5 s) without duplication.
      • End chunk when silence exceeds a dynamic timeout:
            timeout = min(max(0.5, silence_duration) + 0.25*spoken_seconds, 3.0)
      • On end, transcribe the chunk (dual-model consensus) and call on_transcription().

    TTS ducking during playback to reduce feedback (attenuation + stricter VAD),
    while keeping LMS/monitor active for continued echo adaptation.
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
        assembler=None,                 # kept for compatibility
        denoise_fn: callable = None,
    ):
        # ── Config ────────────────────────────────────────────────────────────
        self.sample_rate         = sample_rate
        self.rms_threshold       = rms_threshold
        self.base_silence        = max(0.5, silence_duration)
        self.consensus_threshold = consensus_threshold
        self.enable_denoise      = enable_denoise
        self._denoise_fn         = denoise_fn
        self.on_transcription    = on_transcription
        self.log                 = logger
        self.config              = cfg or {}

        # streaming params
        self.stream_window       = float(self.config.get("stream_window", 3.0))
        self.stream_step         = float(self.config.get("stream_step",   0.2))
        self.delay_alpha         = float(self.config.get("delay_alpha",   0.1))

        # DUCKING params (reduction during TTS playback)
        self.ducking_enabled          = bool(self.config.get("ducking_enabled", True))
        self.ducking_gain             = float(self.config.get("ducking_gain", 0.08))  # ≈ −22 dB
        self.ducking_threshold_boost  = float(self.config.get("ducking_threshold_boost", 2.5))
        self.ducking_afterglow_s      = float(self.config.get("ducking_afterglow_s", 0.25))
        self.ducking_monitor_gate_db  = float(self.config.get("ducking_monitor_gate_db", -35.0))
        self._last_tts_frame_ts       = 0.0  # updated via push_cancellation()

        # buffers & locks
        self._buffer             = np.zeros(0, dtype=np.float32)
        self._buffer_lock        = threading.Lock()
        self._monitor_buffer     = np.zeros(0, dtype=np.float32)
        self._monitor_lock       = threading.Lock()
        self._echo_profile       = None

        # TTS suppression flag (set while TTS is talking)
        self._tts_muted          = threading.Event()

        # LMS echo cancellation
        self.enable_lms          = bool(self.config.get("enable_lms", False))
        taps = int(self.config.get("lms_taps", 1024))
        mu   = float(self.config.get("lms_mu", 5e-4))
        self._tts_lms           = StreamingLMSFilter(num_taps=taps, mu=mu, safe=True)
        self._lms_ref_buf       = np.zeros(0, dtype=np.float32)
        self._lms_lock          = threading.Lock()
        self._time_of_last_transcript = 0.0

        # worker threads & streams
        self._stop_evt          = threading.Event()
        self._stream            = None
        self._monitor_stream    = None
        self._worker            = None

        # ── Whisper selection (via cfg with safe defaults) ────────────────────
        self._wm_device = _device_or_auto(self.config.get("whisper_device", "auto"))
        self._wm_full   = str(self.config.get("whisper_full_model", "medium")).strip() or "medium"
        self._wm_cons   = str(self.config.get("whisper_consensus_model", "base")).strip() or "base"

        self.log(f"AudioService: loading Whisper models ({self._wm_full}, {self._wm_cons}) on {self._wm_device}", "INFO")
        # For live preview, use the *consensus* (smaller) model to keep it snappy.
        self.model_preview   = _get_whisper_model(self._wm_cons, device=self._wm_device)
        # For final decode, use full model + consensus with the smaller one.
        self.model_full      = _get_whisper_model(self._wm_full, device=self._wm_device)
        self.model_consensus = _get_whisper_model(self._wm_cons, device=self._wm_device)
        self.log("AudioService: Whisper models ready.", "SUCCESS")

        # VAD state
        self._capturing         = False
        self._chunk             = np.zeros(0, dtype=np.float32)
        self._spoken_seconds    = 0.0
        self._speech_run        = 0.0
        self._silence_run       = 0.0
        self._block_dt          = 1024.0 / float(sample_rate)

        # live-preview text
        self._last_live_text      = ""
        self._last_live_decode_ts = 0.0
        self._live_decode_interval = float(self.config.get("live_decode_interval", 0.8))
        self._live_window_seconds  = float(self.config.get("live_window_seconds", 2.5))

    # ─── TTS controls ──────────────────────────────────────────────────────────
    def mute_tts(self):
        """Call when TTS starts speaking."""
        self._tts_muted.set()

    def unmute_tts(self):
        """Call when TTS stops speaking."""
        self._tts_muted.clear()

    # ─── Feed in TTS frames for echo cancellation ─────────────────────────────
    def push_cancellation(self, cancelled_frames: np.ndarray):
        if cancelled_frames is None or cancelled_frames.size == 0:
            return
        cf = cancelled_frames.astype(np.float32)
        self._last_tts_frame_ts = time.time()

        # monitor buffer (for dB display)
        with self._monitor_lock:
            self._monitor_buffer = np.concatenate((self._monitor_buffer, cf))
            maxs = int(self.sample_rate * self.stream_window)
            if self._monitor_buffer.size > maxs:
                self._monitor_buffer = self._monitor_buffer[-maxs:]

        # LMS reference buffer
        if self.enable_lms:
            with self._lms_lock:
                self._lms_ref_buf = np.concatenate((self._lms_ref_buf, cf))
                maxs = int(self.sample_rate * self.stream_window)
                if self._lms_ref_buf.size > maxs:
                    self._lms_ref_buf = self._lms_ref_buf[-maxs:]

    # ─── Echo calibration via loopback monitor ────────────────────────────────
    def _find_monitor_device(self) -> int | None:
        default = sd.default.device
        if isinstance(default, (list, tuple)) and len(default) == 2:
            _, out_idx = default
        elif hasattr(default, "output"):
            out_idx = default.output
        else:
            out_idx = None

        devs = sd.query_devices()
        if isinstance(out_idx, int) and 0 <= out_idx < len(devs):
            hostapi = devs[out_idx]["hostapi"]
            for i, d in enumerate(devs):
                if d["hostapi"] == hostapi and d["max_input_channels"] > 0 and "monitor" in d["name"].lower():
                    return i
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and "monitor" in d["name"].lower():
                return i
        return None

    def _calibrate_echo(self):
        tone_dur = 1.0
        freq     = 440.0
        self.log("AudioService: calibrating echo via test tone…", "INFO")
        t = np.linspace(0, tone_dur, int(self.sample_rate * tone_dur), False)
        tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        with self._monitor_lock:
            self._monitor_buffer = np.zeros(0, dtype=np.float32)
        sd.play(tone, self.sample_rate); sd.wait()
        time.sleep(0.2)
        with self._monitor_lock:
            self._echo_profile = self._monitor_buffer.copy()
        self.log(f"AudioService: captured echo profile ({len(self._echo_profile)} samples).", "INFO")

    # ─── Start / Stop ─────────────────────────────────────────────────────────
    def start(self):
        # ambient noise calibration
        self.log("AudioService: calibrating ambient noise…", "INFO")
        try:
            sd.default.samplerate = self.sample_rate
            sd.default.channels   = 1
            amb = sd.rec(int(self.sample_rate * 1.0), dtype="float32"); sd.wait()
            ambient = float(np.sqrt(np.mean(amb ** 2)))
            self.rms_threshold = max(self.rms_threshold, ambient * 1.5)
            self.log(f"AudioService: rms_threshold={self.rms_threshold:.6f}", "INFO")
        except Exception as e:
            self.log(f"AudioService: ambient calibration failed: {e}", "WARNING")

        # loopback monitor
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
                self._calibrate_echo()
            except Exception as e:
                self.log(f"AudioService: monitor open failed: {e}", "WARNING")
        else:
            self.log("AudioService: no monitor found; echo calibration skipped", "WARNING")

        # mic capture
        self.log("AudioService: starting mic capture…", "INFO")
        self._stop_evt.clear()
        try:
            self._stream = InputStream(
                samplerate= self.sample_rate,
                blocksize = 1024,
                channels  = 1,
                callback  = self._audio_callback
            )
            self._block_dt = 1024.0 / float(self.sample_rate)
            self._stream.start()
        except Exception as e:
            self.log(f"AudioService: mic open failed @ {self.sample_rate}Hz: {e}", "WARNING")
            self._stream = InputStream(callback=self._audio_callback)
            self._stream.start()

        dev = sd.default.device
        sr  = getattr(self._stream, "samplerate", self.sample_rate)
        self.log(f"AudioService: mic on device {dev} @ {sr:.0f}Hz", "INFO")

        # reset LMS
        try:
            self._tts_lms.reset()
        except Exception:
            pass
        with self._lms_lock:
            self._lms_ref_buf = np.zeros(0, dtype=np.float32)

        # spawn worker
        self._worker = threading.Thread(target=self._stream_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self.log("AudioService: stopping capture…", "INFO")
        self._stop_evt.set()
        if self._stream:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
        if self._monitor_stream:
            try:
                self._monitor_stream.stop(); self._monitor_stream.close()
            except Exception:
                pass
        if self._worker:
            try:
                self._worker.join()
            except Exception:
                pass

        # Release shared Whisper models so VRAM is freed when last user goes away
        try:
            _release_whisper_model(self._wm_full, device=self._wm_device)
            _release_whisper_model(self._wm_cons, device=self._wm_device)
        except Exception:
            pass

    # ─── Callbacks ────────────────────────────────────────────────────────────
    def _monitor_callback(self, indata, frames, time_info, status):
        if status:
            self.log(f"AudioService(monitor): status {status}", "WARNING")
        buf = indata[:, 0].astype(np.float32)
        with self._monitor_lock:
            self._monitor_buffer = np.concatenate((self._monitor_buffer, buf))
            maxs = int(self.stream_window * self.sample_rate)
            if len(self._monitor_buffer) > maxs:
                self._monitor_buffer = self._monitor_buffer[-maxs:]

    def _audio_callback(self, indata, frames, time_info, status):
        # NOTE: we no longer drop frames during TTS; we attenuate instead.
        if status:
            self.log(f"AudioService: status {status}", "WARNING")

        # raw float32
        buf = indata[:, 0].astype(np.float32)

        # optional denoise (if provided)
        sr = getattr(self._stream, "samplerate", self.sample_rate)
        if self.enable_denoise and self._denoise_fn is not None:
            try:
                buf = self._denoise_fn(buf, sr).astype(np.float32)
            except Exception as e:
                self.log(f"denoise_fn error: {e}", "WARNING")

        # always apply high-pass to cut low rumble
        buf = _highpass(buf, sr, cutoff=100.0)

        # subtract static echo profile
        if self._echo_profile is not None and buf.size:
            ep = self._echo_profile
            seg = ep[-len(buf):] if len(ep) >= len(buf) else np.pad(ep, (len(buf) - len(ep), 0))
            buf = (buf - seg).astype(np.float32)

        # adaptive LMS echo cancellation
        if self.enable_lms and buf.size:
            with self._lms_lock:
                ref = self._lms_ref_buf.copy()
            if ref.size < len(buf):
                with self._monitor_lock:
                    ref = self._monitor_buffer.copy()
            if ref.size >= len(buf):
                ref_seg = ref[-len(buf):].astype(np.float32)
                try:
                    _, e = self._tts_lms.process(ref_seg, buf)
                    if e is not None and e.size == buf.size:
                        buf = e.astype(np.float32)
                except Exception as lms_err:
                    self.log(f"LMS process error: {lms_err}", "WARNING")

        # ── DUCKING: attenuate + stricter VAD during active/just-ended TTS ────
        tts_active = self._tts_muted.is_set()
        # Also consider "afterglow" and actual monitor loudness
        recent_tts = (time.time() - max(self._last_tts_frame_ts, 0.0)) <= self.ducking_afterglow_s
        with self._monitor_lock:
            count = int(sr * self.stream_step)
            mblock = (self._monitor_buffer[-count:] if len(self._monitor_buffer) >= count
                      else self._monitor_buffer)
        monitor_db = None
        if mblock.size:
            mrms = float(np.sqrt(np.mean(mblock ** 2)))
            monitor_db = 20.0 * np.log10(mrms + 1e-9)

        should_duck = (
            self.ducking_enabled
            and (tts_active or recent_tts)
            and (monitor_db is None or monitor_db >= self.ducking_monitor_gate_db)
        )

        if should_duck and buf.size:
            buf *= self.ducking_gain  # attenuate mic while TTS audio is present

        # VAD energy
        rms = float(np.sqrt(np.mean(buf ** 2))) if buf.size else 0.0
        # raise threshold briefly after a transcript to avoid double-firing
        base_mult = 1.2 if (time.time() - self._time_of_last_transcript) < 1.0 else 1.0
        duck_mult = (self.ducking_threshold_boost if should_duck else 1.0)
        thresh = self.rms_threshold * base_mult * duck_mult
        is_speech = rms >= thresh

        if is_speech:
            self._speech_run  += self._block_dt
            self._silence_run  = 0.0
        else:
            self._silence_run += self._block_dt
            self._speech_run   = 0.0

        # start chunk after ~150 ms speech
        if not self._capturing and is_speech and self._speech_run >= 0.15:
            self._capturing           = True
            self._chunk               = np.zeros(0, dtype=np.float32)
            self._spoken_seconds      = 0.0
            self._last_live_text      = ""
            self._last_live_decode_ts = 0.0
            self._time_of_last_transcript = time.time()

        # accumulate chunk
        if self._capturing:
            self._chunk = np.concatenate((self._chunk, buf))
            self._spoken_seconds += self._block_dt

        # rolling buffer for dB or other uses
        with self._buffer_lock:
            self._buffer = np.concatenate((self._buffer, buf))
            maxs = int(self.stream_window * self.sample_rate)
            if len(self._buffer) > maxs:
                self._buffer = self._buffer[-maxs:]

    # ─── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _diff_new_suffix(prev: str, curr: str) -> str:
        if not curr:
            return ""
        prev_t = prev.split()
        curr_t = curr.split()
        n = min(len(prev_t), len(curr_t))
        i = 0
        while i < n and prev_t[i] == curr_t[i]:
            i += 1
        return " ".join(curr_t[i:]).strip()

    def _transcribe_np(self, pcm: np.ndarray, full: bool = False) -> str:
        """
        Resample → cast → Whisper.
        - Live preview: uses the smaller consensus model (faster).
        - Final: uses full model + consensus check.
        """
        if pcm.size == 0:
            return ""

        # resample
        sr_in = int(self.sample_rate)
        if sr_in != 16000:
            try:
                audio_16k = resample_poly(pcm, 16000, sr_in)
            except Exception as e:
                self.log(f"Resample error: {e}", "ERROR")
                return ""
        else:
            audio_16k = pcm
        audio_16k = audio_16k.astype(np.float32)

        # live preview → consensus model for speed
        if not full:
            try:
                return self.model_preview.transcribe(
                    audio_16k, language="en", fp16=False
                )["text"].strip()
            except Exception as e:
                self.log(f"Whisper preview error: {e}", "ERROR")
                return ""

        # full transcription → full model then consensus check
        try:
            result = self.model_full.transcribe(audio_16k, language="en", fp16=False)
            text   = result.get("text", "").strip()

            # drop no-speech
            if result.get("no_speech_prob", 0.0) > 0.6 or len(text.split()) < 2:
                return ""

            # consensus with smaller model
            tm = self.model_consensus.transcribe(audio_16k, language="en", fp16=False)["text"].strip()
            if tm and text:
                ratio = SequenceMatcher(None, text, tm).ratio()
                if ratio < self.consensus_threshold:
                    self.log(f"Consensus fail ({ratio:.2f}<{self.consensus_threshold})", "INFO")
                    return ""

            return text

        except Exception as e:
            self.log(f"Whisper error: {e}", "ERROR")
            return ""

    # ─── Main loop ────────────────────────────────────────────────────────────
    def _stream_loop(self):
        sr = getattr(self._stream, "samplerate", self.sample_rate)
        max_timeout = 3.0
        while not self._stop_evt.is_set():
            now = time.time()
            time.sleep(self.stream_step)

            # Display TTS suppression level, but DO NOT skip processing anymore.
            if self._tts_muted.is_set():
                with self._monitor_lock:
                    count = int(sr * self.stream_step)
                    block = (self._monitor_buffer[-count:]
                             if len(self._monitor_buffer) >= count
                             else self._monitor_buffer)
                if block.size:
                    rms = float(np.sqrt(np.mean(block ** 2)))
                    db  = 20.0 * np.log10(rms + 1e-9)
                    print(f"\r[TTS Suppression] ~{db:5.1f}dB", end="", flush=True)
                # no 'continue' ⇒ we keep live preview / countdown running

            # live preview while capturing
            if self._capturing:
                if now - self._last_live_decode_ts >= self._live_decode_interval:
                    self._last_live_decode_ts = now
                    tail_len = int(self._live_window_seconds * self.sample_rate)
                    tail = (self._chunk[-tail_len:]
                            if len(self._chunk) > tail_len else self._chunk)
                    live_text = self._transcribe_np(tail, full=False)
                    if live_text:
                        new_suffix = self._diff_new_suffix(self._last_live_text, live_text)
                        if new_suffix:
                            for w in new_suffix.split():
                                print(w, end=" ", flush=True)
                            self._time_of_last_transcript = time.time()
                        self._last_live_text = live_text

                # countdown since last transcript
                dynamic_timeout = min(self.base_silence + 0.25 * self._spoken_seconds, max_timeout)
                elapsed = now - self._time_of_last_transcript
                remaining = max(0.0, min(dynamic_timeout - elapsed, max_timeout))
                print(f"\r⏱ {remaining:4.2f}s until send", end="", flush=True)

                if elapsed >= dynamic_timeout:
                    print()  # newline
                    final_text = self._transcribe_np(self._chunk, full=True)
                    if final_text:
                        try:
                            self.on_transcription(final_text.strip())
                        except Exception as cb_err:
                            self.log(f"AudioService callback error: {cb_err}", "ERROR")
                    # reset state
                    self._capturing           = False
                    self._chunk               = np.zeros(0, dtype=np.float32)
                    self._spoken_seconds      = 0.0
                    self._last_live_text      = ""
                    self._last_live_decode_ts = 0.0
                    self._time_of_last_transcript = time.time()

        self.log("AudioService: stream loop exiting.", "DEBUG")
