#!/usr/bin/env python3
"""
main.py

Bootstraps a virtualenv (once), installs system & Python dependencies,
ensures Piper + ONNX assets, initializes config, then enters a REPL
that uses Assembler to manage ContextObjects and invoke gemma3:4b via Ollama.
Also integrates voice I/O: continuous recording, Whisper transcription on silence,
and Piper-based TTS playback.
"""

import sys
import os
import subprocess
import platform
import shutil
import json
import time
from datetime import datetime
import re
import signal
import threading
import queue
# Import your ChatManager if you have one; here we just use Assembler directly.
# from your_chat_module import ChatManager
# from audio_utils import apply_eq_and_denoise, consensus_whisper_transcribe_helper, validate_transcription
# For simplicity, I'll inline the helper functions below.

# ──────────── HANDLE CTRL-C ────────────────────────────────────────────────
def _exit_on_sigint(signum, frame):
    print("\nInterrupted. Shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, _exit_on_sigint)

# ──────────── COLOR CODES FOR LOGGING ─────────────────────────────────────────
COLOR_RESET   = "\033[0m"
COLOR_INFO    = "\033[94m"
COLOR_SUCCESS = "\033[92m"
COLOR_WARNING = "\033[93m"
COLOR_ERROR   = "\033[91m"
COLOR_DEBUG   = "\033[95m"
COLOR_PROCESS = "\033[96m"

def log_message(message: str, category: str="INFO"):
    """Print a timestamped, colored log message."""
    cat = category.upper()
    color = {
        "INFO":    COLOR_INFO,
        "SUCCESS": COLOR_SUCCESS,
        "WARNING": COLOR_WARNING,
        "ERROR":   COLOR_ERROR,
        "DEBUG":   COLOR_DEBUG,
        "PROCESS": COLOR_PROCESS
    }.get(cat, COLOR_RESET)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}[{ts}] {cat}: {message}{COLOR_RESET}")

# ──────────── Virtualenv Bootstrap ────────────────────────────────────────────
def in_virtualenv() -> bool:
    base = getattr(sys, "base_prefix", None)
    return base is not None and sys.prefix != base

def create_and_activate_venv():
    venv_dir = os.path.join(os.getcwd(), ".venv")
    python_bin = os.path.join(venv_dir, "bin", "python")
    pip_bin    = os.path.join(venv_dir, "bin", "pip")
    if not os.path.isdir(venv_dir):
        log_message("Creating virtualenv in .venv/", "PROCESS")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        log_message("Installing pip in venv…", "PROCESS")
        subprocess.check_call([pip_bin, "install", "--upgrade", "pip"])
    log_message("Re-launching under virtualenv…", "PROCESS")
    os.execve(python_bin,
             [python_bin] + sys.argv,
             dict(os.environ, VIRTUAL_ENV=venv_dir,
                  PATH=venv_dir+"/bin:"+os.environ.get("PATH","")))

if not in_virtualenv():
    create_and_activate_venv()

# ──────────── Ensure Piper + ONNX Assets ──────────────────────────────────────
def setup_piper_and_onnx():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_folder = os.path.join(script_dir, "piper")
    piper_exe    = os.path.join(piper_folder, "piper")
    log_message(f"Checking for Piper executable at {piper_exe}", "INFO")

    os_name = platform.system()
    arch    = platform.machine().lower()
    release = ""
    if os_name == "Linux":
        if arch in ("x86_64",):
            release = "piper_linux_x86_64.tar.gz"
        elif arch in ("aarch64","arm64"):
            release = "piper_linux_aarch64.tar.gz"
        elif arch.startswith("armv7"):
            release = "piper_linux_armv7l.tar.gz"
    elif os_name == "Darwin":
        if arch in ("arm64","aarch64"):
            release = "piper_macos_aarch64.tar.gz"
        elif arch in ("x86_64","amd64"):
            release = "piper_macos_x64.tar.gz"
    elif os_name == "Windows":
        release = "piper_windows_amd64.zip"
    else:
        log_message(f"Unsupported OS: {os_name}", "ERROR")
        sys.exit(1)

    if not os.path.isfile(piper_exe):
        download_url = f"https://github.com/rhasspy/piper/releases/download/2023.11.14-2/{release}"
        archive_path = os.path.join(script_dir, release)
        log_message(f"Piper not found—downloading {release}", "PROCESS")
        try:
            subprocess.check_call(["wget","-O",archive_path,download_url])
        except Exception as e:
            log_message(f"Failed to download Piper: {e}", "ERROR")
            sys.exit(1)
        os.makedirs(piper_folder, exist_ok=True)
        if release.endswith(".tar.gz"):
            subprocess.check_call([
                "tar","-xzvf",archive_path,
                "-C",piper_folder,"--strip-components=1"
            ])
        else:
            subprocess.check_call(["unzip","-o",archive_path,"-d",piper_folder])
        log_message("Piper unpacked.", "SUCCESS")
    else:
        log_message("Piper executable present.", "SUCCESS")

    onnx_json = os.path.join(script_dir, "glados_piper_medium.onnx.json")
    onnx_model= os.path.join(script_dir, "glados_piper_medium.onnx")
    if not os.path.isfile(onnx_json):
        url = "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json"
        log_message("Downloading ONNX JSON…", "PROCESS")
        subprocess.check_call(["wget","-O",onnx_json,url])
    log_message("ONNX JSON present.", "SUCCESS")
    if not os.path.isfile(onnx_model):
        url = "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx"
        log_message("Downloading ONNX model…", "PROCESS")
        subprocess.check_call(["wget","-O",onnx_model,url])
    log_message("ONNX model present.", "SUCCESS")

setup_piper_and_onnx()

# ──────────── First-Run System & Python Deps ────────────────────────────────────
SETUP_MARKER = os.path.join(os.path.dirname(__file__), ".setup_complete")
if not os.path.exists(SETUP_MARKER):
    log_message("Installing system packages…", "PROCESS")
    if sys.platform.startswith("linux") and shutil.which("apt-get"):
        try:
            subprocess.check_call(["sudo","apt-get","update"])
            subprocess.check_call([
                "sudo","apt-get","install","-y",
                "libsqlite3-dev","ffmpeg","wget","unzip"
            ])
        except subprocess.CalledProcessError as e:
            log_message(f"apt-get failed: {e}", "ERROR")
            sys.exit(1)

    log_message("Installing Python dependencies...", "PROCESS")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "sounddevice", "numpy", "scipy",
        "openai-whisper", "ollama", "python-dotenv",
        "beautifulsoup4", "html5lib", "pywifi", "psutil",
        "num2words", "noisereduce", "denoiser",
        "pyautogui", "pillow", "opencv-python",
        "mss", "networkx", "pysqlite3", "pandas",
        "selenium", "webdriver-manager",
        "flask_cors", "flask", "tiktoken",
        "python-telegram-bot", "asyncio",
        "nest-asyncio", "sentence-transformers"
    ])
    with open(SETUP_MARKER, "w") as f:
        f.write("done")
    log_message("Dependencies installed. Restarting…", "SUCCESS")
    os.execv(sys.executable, [sys.executable] + sys.argv)
import uuid
import traceback
import numpy as np
import difflib
import sounddevice as sd
from sounddevice import InputStream
import whisper
from assembler import Assembler
from scipy.signal import butter, lfilter       # EQ enhancement

log_message("Loading Whisper models…", "INFO")
model_base   = whisper.load_model("base")
model_medium = whisper.load_model("medium")
log_message("Whisper models loaded.", "SUCCESS")

# ──────────── Load / Generate config.json ─────────────────────────────────────
CONFIG_FILE = "config.json"
default_cfg = {
    "primary_model":   "gemma3:4b",
    "secondary_model": "gemma3:4b"
}
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_cfg, f, indent=2)
    log_message(f"Created default {CONFIG_FILE}", "INFO")
with open(CONFIG_FILE) as f:
    config = json.load(f)

# ──────────── Shared Audio Queue & Settings ───────────────────────────────────
SAMPLE_RATE      = config.get("sample_rate", 16000)
RMS_THRESHOLD    = config.get("rms_threshold", 0.01)
SILENCE_DURATION = config.get("silence_duration", 2.0)
ENABLE_DENOISE   = config.get("enable_noise_reduction", False)
CONS_THRESHOLD   = config.get("consensus_threshold", 0.8)

audio_queue = queue.Queue()
session_log = open("session.log","a")

# ──────────── Whisper Consensus Helper ────────────────────────────────────────
def consensus_whisper_transcribe_helper(
    audio_array,
    language="en",
    rms_threshold=0.01,
    consensus_threshold=0.8
):
    rms = np.sqrt(np.mean(audio_array**2))
    if rms < rms_threshold:
        return ""
    text_b, text_m = "", ""
    def run_base():
        nonlocal text_b
        try:
            res = model_base.transcribe(audio_array, language=language)
            text_b = res.get("text","").strip()
        except Exception as e:
            log_message(f"Base transcription error: {e}", "ERROR")

    def run_med():
        nonlocal text_m
        try:
            res = model_medium.transcribe(audio_array, language=language)
            text_m = res.get("text","").strip()
        except Exception as e:
            log_message(f"Medium transcription error: {e}", "ERROR")

    t1 = threading.Thread(target=run_base)
    t2 = threading.Thread(target=run_med)
    t1.start(); t2.start()
    t1.join(); t2.join()

    if not text_b or not text_m:
        return ""
    sim = difflib.SequenceMatcher(None, text_b, text_m).ratio()
    log_message(f"Transcription similarity: {sim:.2f}", "DEBUG")
    return text_b if sim >= consensus_threshold else ""

def validate_transcription(text):
    if not any(ch.isalnum() for ch in text):
        log_message("Validation failed: no alphanumeric chars", "WARNING")
        return False
    if len(text.strip().split()) < 1:
        log_message("Validation failed: no words", "WARNING")
        return False
    return True

# ──────────── TTS Handler ─────────────────────────────────────────────────────
class TTSOutputHandler:
    """
    Queues text→Piper+ffmpeg→aplay in background.
    """
    def __init__(self, piper_exe, onnx_json, onnx_model,
                 out_dir="tts_output", volume=0.2, debug=False):
        self.exe = piper_exe
        self.json = onnx_json
        self.model = onnx_model
        self.dir = out_dir
        self.vol = volume
        self.dbg = debug
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.proc = None
        self.stop_evt = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        os.makedirs(self.dir, exist_ok=True)
        self.worker.start()

    def say(self, text):
        self.q.put(text)

    def flush(self):
        with self.lock:
            while not self.q.empty():
                try: self.q.get_nowait(); self.q.task_done()
                except: break
            if self.proc:
                for p in self.proc:
                    try: p.kill()
                    except: pass
                self.proc = None

    def stop(self):
        self.stop_evt.set()
        self.q.put(None)
        self.worker.join()

    def _run(self):
        log_message("TTS worker started.", "DEBUG")
        while not self.stop_evt.is_set():
            txt = self.q.get()
            if txt is None:
                break
            try:
                self._synth(txt)
            except Exception as e:
                log_message(f"TTS error: {e}\n{traceback.format_exc()}", "ERROR")
            finally:
                self.q.task_done()
        log_message("TTS worker exiting.", "DEBUG")

    def _synth(self, text):
        fname = f"{datetime.utcnow():%Y%m%dT%H%M%S}_{uuid.uuid4().hex}.ogg"
        out = os.path.join(self.dir, fname)
        payload = json.dumps({
            "text": text,
            "config": self.json,
            "model": self.model
        }).encode("utf-8")
        log_message(f"[TTS] Synthesizing → {out}", "INFO")
        cmd1 = [self.exe, "-m", self.model, "--json-input", "--output_raw"]
        if self.dbg:
            cmd1.insert(3, "--debug")
        cmd2 = ["ffmpeg","-f","s16le","-ar","22050","-ac","1","-i","pipe:0",
                "-c:a","libvorbis","-qscale:a","5", out]
        cmd3 = ["aplay","--buffer-size=777","-r","22050","-f","S16_LE"]
        with self.lock:
            p1 = subprocess.Popen(cmd1, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=(subprocess.PIPE if self.dbg else subprocess.DEVNULL))
            p2 = subprocess.Popen(cmd2, stdin=p1.stdout, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            p3 = subprocess.Popen(cmd3, stdin=subprocess.PIPE)
            self.proc = (p1, p2, p3)
        p1.stdin.write(payload); p1.stdin.close()

        def playloop():
            while True:
                ch = p1.stdout.read(4096)
                if not ch: break
                if self.vol != 1.0:
                    sm = np.frombuffer(ch, dtype=np.int16)
                    adj = (sm.astype(np.float32)*self.vol).clip(-32768,32767).astype(np.int16)
                    ch = adj.tobytes()
                p3.stdin.write(ch)
            p3.stdin.close()
            p3.wait()

        t = threading.Thread(target=playloop, daemon=True)
        t.start()
        p1.wait(); p2.wait(); t.join()
        with self.lock:
            self.proc = None
        log_message(f"[TTS] Done → {out}", "SUCCESS")

# ──────────── Initialize Assembler & REPL I/O ─────────────────────────────────
CTX_PATH = "context.jsonl"
asm = Assembler(
    context_path=CTX_PATH,
    config_path=CONFIG_FILE,
    lookback_minutes=60,
    top_k=5
)

# Instantiate TTS
script_dir = os.path.dirname(os.path.abspath(__file__))
piper_exe  = os.path.join(script_dir, "piper", "piper")
tts = TTSOutputHandler(
    piper_exe=piper_exe,
    onnx_json=config.get("onnx_json","glados_piper_medium.onnx.json"),
    onnx_model=config.get("onnx_model","glados_piper_medium.onnx"),
    volume=config.get("tts_volume",0.2),
    debug=config.get("tts_debug",False),
)

def handle_input(user_text):
    """Shared handler for typed or spoken input."""
    print(f">> {user_text}")
    # record to session log
    session_log.write(json.dumps({
        "role":"user","content":user_text,
        "timestamp":datetime.now().isoformat()
    }) + "\n")
    session_log.flush()

    # clear pending TTS
    tts.flush()

    # run assembler
    resp = asm.run_with_meta_context(user_text)
    print(resp)

    # speak response
    tts.say(resp)

# ──────────── Audio Callback to fill queue ────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    if status:
        log_message(f"Audio callback status: {status}", "WARNING")
    audio_queue.put(indata.copy())

# start microphone capture
stream = InputStream(
    callback=audio_callback,
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=1024
)
stream.start()
# ──────────── Enhanced EQ + Denoise ───────────────────────────────────────────
def apply_eq_and_denoise(
    audio: np.ndarray,
    sample_rate: int,
    lowcut: float = 300.0,
    highcut: float = 4000.0,
    eq_gain: float = 2.0,
    pre_emphasis_coef: float = 0.99,
    compress_thresh: float = 0.1,
    compress_ratio: float = 4.0
) -> np.ndarray:
    """
    1) Dynamic-range normalize
    2) Noise-reduce
    3) Pre-emphasis
    4) Band-pass EQ
    5) Simple compression
    """
    # 1) Normalize
    try:
        log_message("Enhancement: Normalizing dynamic range...", "PROCESS")
        audio = dynamic_range_normalize(
            audio, sample_rate,
            frame_ms=20, hop_ms=10,
            target_rms=0.3, smoothing_coef=0.9
        )
        log_message("Enhancement: Dynamic range normalization complete.", "SUCCESS")
    except Exception as e:
        log_message(f"Enhancement: Normalization failed: {e}", "WARNING")

    # 2) Noise reduction
    try:
        log_message("Enhancement: Reducing noise via spectral gating...", "PROCESS")
        denoised = nr.reduce_noise(
            y=audio, sr=sample_rate,
            prop_decrease=1.0, stationary=False
        )
        log_message("Enhancement: Noise reduction complete.", "SUCCESS")
    except Exception as e:
        log_message(f"Enhancement: Noise reduction failed: {e}", "WARNING")
        denoised = audio

    # 3) Pre-emphasis
    log_message("Enhancement: Applying pre-emphasis filter...", "PROCESS")
    emphasized = np.concatenate((
        denoised[:1],
        denoised[1:] - pre_emphasis_coef * denoised[:-1]
    ))

    # 4) Band-pass EQ
    log_message("Enhancement: Applying band-pass EQ...", "PROCESS")
    nyq = 0.5 * sample_rate
    b, a = butter(2, [lowcut/nyq, highcut/nyq], btype="band")
    band = lfilter(b, a, emphasized)
    eq_boosted = emphasized + (eq_gain - 1.0) * band

    # Prevent clipping
    maxv = np.max(np.abs(eq_boosted))
    if maxv > 1.0:
        eq_boosted /= maxv

    # 5) Compression
    log_message("Enhancement: Applying dynamic range compression...", "PROCESS")
    thresh = compress_thresh * np.max(np.abs(eq_boosted))
    compressed = np.copy(eq_boosted)
    mask = np.abs(eq_boosted) > thresh
    compressed[mask] = np.sign(eq_boosted[mask]) * (
        thresh + (np.abs(eq_boosted[mask]) - thresh) / compress_ratio
    )

    # Final normalize
    fm = np.max(np.abs(compressed))
    if fm > 1.0:
        compressed /= fm

    log_message("Enhancement: Audio enhancement complete.", "DEBUG")
    return compressed.astype(np.float32)

def voice_to_llm_loop():
    log_message("Voice→LLM loop started. Listening for speech…", "INFO")
    while True:
        # 1) Read one buffer
        buf = audio_queue.get()
        audio_queue.task_done()
        rms = np.sqrt(np.mean(buf**2))
        log_message(f"[CHUNK] RMS = {rms:.4f}", "DEBUG")

        # 2) If it’s still “silence”, skip
        if rms < RMS_THRESHOLD:
            continue

        # 3) We’ve hit speech — collect a small context window
        log_message("Speech detected: buffering up to 5 chunks…", "DEBUG")
        chunks = [buf.flatten()]
        for _ in range(4):  # grab up to 4 more buffers (≈5×1024/16000 ≈ 0.32 s)
            try:
                nxt = audio_queue.get(timeout=0.1)
                audio_queue.task_done()
            except queue.Empty:
                break
            chunks.append(nxt.flatten())

        audio_arr = np.concatenate(chunks, axis=0).astype(np.float32)
        log_message(f"Buffer length: {len(audio_arr)} samples (~{len(audio_arr)/SAMPLE_RATE:.2f}s)", "DEBUG")

        # 4) Optionally enhance
        if ENABLE_DENOISE:
            log_message("Enhancing audio…", "PROCESS")
            audio_arr = apply_eq_and_denoise(audio_arr, SAMPLE_RATE)
            log_message("Enhancement done.", "SUCCESS")

        # 5) Transcribe via consensus
        log_message("Transcribing via Whisper consensus…", "DEBUG")
        transcription = consensus_whisper_transcribe_helper(
            audio_arr,
            language="en",
            rms_threshold=RMS_THRESHOLD,
            consensus_threshold=CONS_THRESHOLD
        )
        log_message(f"Whisper returned: {transcription!r}", "DEBUG")

        # 6) Validate & dispatch
        if not transcription or not validate_transcription(transcription):
            log_message("Invalid or empty transcription, continuing listen loop.", "WARNING")
            continue

        user_text = transcription.strip()
        log_message(f"Recognized: {user_text!r}", "INFO")
        handle_input(user_text)
        log_message("Ready for next utterance…", "INFO")


# kick off the thread
voice_thread = threading.Thread(target=voice_to_llm_loop, daemon=True)
voice_thread.start()


# ──────────── Text REPL Override ───────────────────────────────────────────────
print(f"Using context store: {CTX_PATH}")
print(f"Primary model: {config['primary_model']}")
print("Ready. Type your message, or speak, or Ctrl-C to exit.")
while True:
    try:
        line = input(">> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        continue
    handle_input(line)

# ──────────── Cleanup on exit ────────────────────────────────────────────────
stream.stop()
stream.close()
session_log.close()
tts.stop()
print("Goodbye.")
