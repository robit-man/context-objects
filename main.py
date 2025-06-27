#!/usr/bin/env python3
"""
main.py

Bootstraps a virtualenv (once), installs system & Python dependencies,
ensures Piper + ONNX assets, initializes config, then enters a REPL
that uses Assembler to manage ContextObjects via Ollama.
Also integrates:
  • AudioService (continuous recording + Whisper consensus transcription)
  • TTSManager  (live Piper-based TTS playback)
"""

import sys
import os
import subprocess
import platform
import shutil
import json
import signal
import threading
from datetime import datetime

# ──────────── CTRL-C HANDLING ───────────────────────────────────────────────
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

def log_message(msg: str, category: str="INFO"):
    cat   = category.upper()
    color = {
      "INFO":COLOR_INFO, "SUCCESS":COLOR_SUCCESS, "WARNING":COLOR_WARNING,
      "ERROR":COLOR_ERROR, "DEBUG":COLOR_DEBUG, "PROCESS":COLOR_PROCESS
    }.get(cat, COLOR_RESET)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}[{ts}] {cat}: {msg}{COLOR_RESET}")

# ──────────── VIRTUALENV BOOTSTRAP ────────────────────────────────────────────
def in_virtualenv() -> bool:
    base = getattr(sys, "base_prefix", None)
    return base is not None and sys.prefix != base

def create_and_activate_venv():
    venv_dir   = os.path.join(os.getcwd(), ".venv")
    python_bin = os.path.join(venv_dir, "bin", "python")
    pip_bin    = os.path.join(venv_dir, "bin", "pip")
    if not os.path.isdir(venv_dir):
        log_message("Creating virtualenv in .venv/", "PROCESS")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        log_message("Upgrading pip in venv…", "PROCESS")
        subprocess.check_call([pip_bin, "install", "--upgrade", "pip"])
    log_message("Re-launching under virtualenv…", "PROCESS")
    os.execve(python_bin,
              [python_bin] + sys.argv,
              dict(os.environ, VIRTUAL_ENV=venv_dir,
                   PATH=venv_dir+"/bin:"+os.environ.get("PATH","")))

if not in_virtualenv():
    create_and_activate_venv()

# ──────────── PIPER + ONNX SETUP ─────────────────────────────────────────────
def setup_piper_and_onnx():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    piper_folder = os.path.join(script_dir, "piper")
    piper_exe    = os.path.join(piper_folder, "piper")
    log_message(f"Checking for Piper at {piper_exe}", "INFO")

    os_name = platform.system()
    arch    = platform.machine().lower()
    if os_name=="Linux":
        if arch=="x86_64":
            release="piper_linux_x86_64.tar.gz"
        elif arch in ("arm64","aarch64"):
            release="piper_linux_aarch64.tar.gz"
        else:
            release="piper_linux_armv7l.tar.gz"
    elif os_name=="Darwin":
        if arch in ("arm64","aarch64"):
            release="piper_macos_aarch64.tar.gz"
        else:
            release="piper_macos_x64.tar.gz"
    elif os_name=="Windows":
        release="piper_windows_amd64.zip"
    else:
        log_message(f"Unsupported OS: {os_name}", "ERROR")
        sys.exit(1)

    if not os.path.isfile(piper_exe):
        url     = f"https://github.com/rhasspy/piper/releases/download/2023.11.14-2/{release}"
        archive = os.path.join(script_dir, release)
        log_message(f"Downloading Piper: {release}", "PROCESS")
        subprocess.check_call(["wget","-O",archive,url])
        os.makedirs(piper_folder, exist_ok=True)
        if release.endswith(".tar.gz"):
            subprocess.check_call([
                "tar","-xzvf",archive,
                "-C",piper_folder,"--strip-components=1"
            ])
        else:
            subprocess.check_call([
                "unzip","-o",archive,
                "-d",piper_folder
            ])
        log_message("Piper unpacked.", "SUCCESS")
    else:
        log_message("Piper already present.", "SUCCESS")

    onnx_json  = os.path.join(script_dir, "glados_piper_medium.onnx.json")
    onnx_model = os.path.join(script_dir, "glados_piper_medium.onnx")
    if not os.path.isfile(onnx_json):
        log_message("Downloading ONNX JSON…", "PROCESS")
        subprocess.check_call([
            "wget","-O",onnx_json,
            "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json"
        ])
    log_message("ONNX JSON present.", "SUCCESS")
    if not os.path.isfile(onnx_model):
        log_message("Downloading ONNX model…", "PROCESS")
        subprocess.check_call([
            "wget","-O",onnx_model,
            "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx"
        ])
    log_message("ONNX model present.", "SUCCESS")

setup_piper_and_onnx()

# ──────────── FIRST-RUN DEPENDENCIES ─────────────────────────────────────────
SETUP_MARKER = os.path.join(os.path.dirname(__file__), ".setup_complete")
if not os.path.exists(SETUP_MARKER):
    log_message("Installing system & Python deps…", "PROCESS")
    if sys.platform.startswith("linux") and shutil.which("apt-get"):
        subprocess.check_call(["sudo","apt-get","update"])
        subprocess.check_call([
            "sudo","apt-get","install","-y",
            "libsqlite3-dev","ffmpeg","wget","unzip"
        ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", "pip"
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install"
    ] + [
        "sounddevice","numpy","scipy","openai-whisper","ollama",
        "python-dotenv","beautifulsoup4","html5lib","psutil",
        "noisereduce","denoiser","pillow","opencv-python",
        "mss","networkx","pandas","selenium","webdriver-manager",
        "flask_cors","flask","tiktoken","python-telegram-bot",
        "asyncio","nest-asyncio","sentence-transformers"
    ])
    with open(SETUP_MARKER,"w") as f:
        f.write("done")
    log_message("Dependencies installed. Restarting…", "SUCCESS")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ──────────── LOAD / GENERATE config.json ────────────────────────────────────
CONFIG_FILE = "config.json"
default_cfg = {"primary_model":"gemma3:4b","secondary_model":"gemma3:4b"}
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE,"w") as f:
        json.dump(default_cfg, f, indent=2)
    log_message(f"Created default {CONFIG_FILE}", "INFO")
with open(CONFIG_FILE) as f:
    config = json.load(f)


# ──────────── IMPORT HELPERS & SERVICES ──────────────────────────────────────
from assembler     import Assembler
from audio_service import AudioService
from tts_service   import TTSManager

# ──────────── INIT TTS MANAGER & ASSEMBLER ──────────────────────────────────
CTX_PATH = "context.jsonl"

# 1️⃣  Instantiate TTS (we'll hook up the mic afterward)
tts = TTSManager(
    logger         = log_message,
    cfg            = config,
    audio_service  = None,       # will assign below
)
tts.set_mode("live")

# 2️⃣  Instantiate Assembler, passing in our TTSManager
asm = Assembler(
    context_path     = CTX_PATH,
    config_path      = CONFIG_FILE,
    lookback_minutes = 60,
    top_k            = 5,
    tts_manager      = tts,
)

# 3️⃣  Open session log for transcripts
session_log = open("session.log", "a")

def handle_input(user_text: str):
    """Called for both typed and spoken inputs."""
    print(f">> {user_text}")
    session_log.write(json.dumps({
        "role":      "user",
        "content":   user_text,
        "timestamp": datetime.now().isoformat()
    }) + "\n")
    session_log.flush()

    # clear any queued TTS so we start fresh
    tts.set_mode("live")

    # run through assembler → LLM (this may already enqueue a question)
    resp = asm.run_with_meta_context(user_text)
    print(resp)

    # only enqueue when we're NOT asking the user to confirm
    #if not getattr(asm, "_awaiting_confirmation", False):
        #tts.enqueue(resp)

# ──────────── START AUDIO SERVICE ────────────────────────────────────────────
audio_svc = AudioService(
    sample_rate         = config.get("sample_rate",        16000),
    rms_threshold       = config.get("rms_threshold",      0.01),
    silence_duration    = config.get("silence_duration",   2.0),
    consensus_threshold = config.get("consensus_threshold",0.5),
    enable_denoise      = config.get("enable_noise_reduction", False),
    on_transcription    = handle_input,
    logger              = log_message,
    cfg                 = config,
)
audio_svc.start()

# now that the mic is running, give it to TTS so it can suspend/resume
tts.audio_service = audio_svc

# ──────────── TEXT REPL OVERRIDE ─────────────────────────────────────────────
print(f"Using context store: {CTX_PATH}")
print(f"Primary model: {config['primary_model']}")
print("Ready: type or speak; Ctrl-C to exit.")
while True:
    try:
        line = input(">> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if line:
        handle_input(line)

# ──────────── CLEANUP ────────────────────────────────────────────────────────
audio_svc.stop()
tts.stop()
session_log.close()
print("Goodbye.")