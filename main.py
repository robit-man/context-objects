#!/usr/bin/env python3
"""
main.py

Bootstraps a virtualenv (once), installs system & Python dependencies,
ensures Piper + ONNX assets, initializes config, then enters a REPL
that uses Assembler to manage ContextObjects and invoke gemma3:4b via Ollama.
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

def in_virtualenv() -> bool:
    """Return True if running inside a virtualenv."""
    base = getattr(sys, "base_prefix", None)
    return base is not None and sys.prefix != base

def create_and_activate_venv():
    """
    Create a .venv directory and re-launch this script inside it.
    """
    venv_dir = os.path.join(os.getcwd(), ".venv")
    python_bin = os.path.join(venv_dir, "bin", "python")
    pip_bin    = os.path.join(venv_dir, "bin", "pip")
    if not os.path.isdir(venv_dir):
        log_message("Creating virtualenv in .venv/", "PROCESS")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        log_message("Installing pip in venv…", "PROCESS")
        subprocess.check_call([pip_bin, "install", "--upgrade", "pip"])
    log_message("Re-launching under virtualenv…", "PROCESS")
    os.execve(python_bin, [python_bin] + sys.argv, dict(os.environ, VIRTUAL_ENV=venv_dir, PATH=venv_dir+"/bin:"+os.environ.get("PATH","")))

# 1) Ensure virtualenv
if not in_virtualenv():
    create_and_activate_venv()

# ──────────── Ensure Piper + ONNX Assets ──────────────────────────────────────
def setup_piper_and_onnx():
    """
    Download and unpack Piper executable and ONNX model/json if missing.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_folder = os.path.join(script_dir, "piper")
    piper_exe    = os.path.join(piper_folder, "piper")
    log_message(f"Checking for Piper executable at {piper_exe}", "INFO")

    # Determine OS and architecture
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
            subprocess.check_call(["tar","-xzvf",archive_path,"-C",piper_folder,"--strip-components=1"])
        else:
            subprocess.check_call(["unzip","-o",archive_path,"-d",piper_folder])
        log_message("Piper unpacked.", "SUCCESS")
    else:
        log_message("Piper executable present.", "SUCCESS")

    # ONNX files
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
    # 1) System-level packages
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

    # 2) Python-level dependencies
    log_message("Installing Python dependencies...", "PROCESS")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "sounddevice", "numpy", "scipy",
        "openai-whisper",   # For Whisper transcription
        "ollama",           # For Ollama Python API
        "python-dotenv",    # For environment variables
        "beautifulsoup4",   # For BS4 scraping
        "html5lib",         # Parser for BS4
        "pywifi",           # For WiFi scanning
        "psutil",           # For system utilization
        "num2words",        # For converting numbers to words
        "noisereduce",      # For noise cancellation (fallback)
        "denoiser",         # For real-time speech enhancement
        "pyautogui",        # For screen capture
        "pillow",           # For image handling
        "opencv-python",    # For image processing
        "mss",              # For screen capture
        "networkx",         # For knowledge graph operations
        "pysqlite3",        # SQLite bindings
        "pandas",           # For data manipulation
        "selenium",         # For web scraping and automation
        "webdriver-manager",# For managing web drivers
        "flask_cors",       # For CORS support in Flask
        "flask",            # For web server
        "tiktoken",         # For tokenization
        "python-telegram-bot", # For Telegram bot API
        "asyncio",          # For asynchronous operations
        "nest-asyncio",      # For asyncio compat
        "sentence-transformers"
    ])

    with open(SETUP_MARKER, "w") as f:
        f.write("done")
    log_message("Dependencies installed. Restarting…", "SUCCESS")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ──────────── Load / Generate config.json ─────────────────────────────────────
CONFIG_FILE = "config.json"
default_cfg = {
    "primary_model":   "gemma3:4b",
    "secondary_model": None
}
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_cfg, f, indent=2)
    log_message(f"Created default {CONFIG_FILE}", "INFO")
with open(CONFIG_FILE) as f:
    config = json.load(f)

# ──────────── Initialize Assembler & REPL ──────────────────────────────────────
from assembler import Assembler

CTX_PATH = "context.jsonl"
asm = Assembler(
    context_path=CTX_PATH,
    config_path=CONFIG_FILE,
    lookback_minutes=60,
    top_k=5
)

print(f"Using context store: {CTX_PATH}")
print(f"Primary model: {config['primary_model']}")
print("Ready. Type your message, or Ctrl-C to exit.")

try:
    while True:
        user_text = input(">> ").strip()
        if not user_text:
            continue
        try:
            resp = asm.run_with_meta_context(user_text)
            print(resp)
        except Exception as e:
            log_message(f"ERROR during inference: {e}", "ERROR")
except KeyboardInterrupt:
    print("\nGoodbye.")
    sys.exit(0)
