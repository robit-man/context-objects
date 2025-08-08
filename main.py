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

# ──────────── VIRTUALENV BOOTSTRAP & FIRST-RUN DEPENDENCIES ─────────────────────────


import os
import sys

if sys.platform.startswith("win"):
    # Use the SelectorEventLoop instead of the ProactorEventLoop on Windows
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import time
import json
import atexit
import shutil
import signal
import platform
import threading
import traceback
import subprocess
from datetime import datetime

if platform.system().startswith("Win"):
    # switch the console to UTF-8 so we can print arrows, en-dashes, etc.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
  
# ─── RE-LAUNCH IF NOT PYTHON 3.10+ ────────────────────────────────────────────
if sys.platform.startswith("linux"):
    # Linux: ensure Python 3.10+
    if sys.version_info < (3, 10):
        python_exec = None
        for ver in ("3.10", "3.11", "3.12"):
            exe = shutil.which(f"python{ver}")
            if exe:
                python_exec = exe
                break
        if not python_exec:
            print("PROCESS: Installing Python 3.10 via apt-get…")
            subprocess.check_call(["sudo", "apt-get", "update"])
            subprocess.check_call([
                "sudo", "apt-get", "install", "-y",
                "python3.10", "python3.10-venv", "python3.10-distutils"
            ])
            python_exec = shutil.which("python3.10")
        if python_exec:
            print(f"PROCESS: Re-launching under {os.path.basename(python_exec)}…")
            os.execv(python_exec, [python_exec] + sys.argv)
        else:
            print("ERROR: Failed to install or locate Python 3.10+ on Linux.")
            sys.exit(1)

            
elif sys.platform == "darwin":
    # macOS: ensure Python 3.10+
    if sys.version_info < (3, 10):
        python_exec = None

        # 1) Try any python3.x already on PATH
        for ver in ("3.10", "3.11", "3.12"):
            exe = shutil.which(f"python{ver}")
            if exe:
                python_exec = exe
                break

        # 2) Install via Homebrew if still missing
        if not python_exec:
            # locate brew even if not on PATH
            brew = shutil.which("brew") or (
                "/opt/homebrew/bin/brew" if os.path.exists("/opt/homebrew/bin/brew")
                else "/usr/local/bin/brew"
            )
            print(f"DEBUG: brew executable at {brew}")
            if brew and os.path.exists(brew):
                print("PROCESS: Installing Python 3.10 via Homebrew…")
                subprocess.check_call([brew, "update"])
                subprocess.check_call([brew, "install", "python@3.10"])
                # force-link so /opt/homebrew/bin/python3.10 appears
                subprocess.check_call([brew, "link", "--overwrite", "--force", "python@3.10"])

                # Now explicitly add Homebrew bins to PATH
                try:
                    prefix = subprocess.check_output([brew, "--prefix", "python@3.10"], text=True).strip()
                    brew_bins = [
                        os.path.join(prefix, "bin"),
                        os.path.join(prefix, "libexec", "bin"),
                        "/opt/homebrew/bin",
                        "/usr/local/bin",
                    ]
                    for p in brew_bins:
                        if os.path.isdir(p):
                            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                    # retry locating python3.10
                    python_exec = shutil.which("python3.10")
                except subprocess.CalledProcessError:
                    pass

        # 3) Fallback: accept generic python3 if already ≥3.10
        if not python_exec:
            candidate = shutil.which("python3")
            if candidate:
                try:
                    out = subprocess.check_output([candidate, "--version"], text=True).strip()
                    _, version = out.split()
                    major, minor, *_ = version.split(".")
                    if int(major) == 3 and int(minor) >= 10:
                        python_exec = candidate
                except Exception:
                    pass

        # 4) Re-exec under the selected interpreter or exit
        if python_exec:
            print(f"PROCESS: Re-launching under {os.path.basename(python_exec)}…")
            os.execv(python_exec, [python_exec] + sys.argv)
        else:
            print("ERROR: Failed to install or locate Python 3.10+ on macOS.")
            sys.exit(1)
elif sys.platform.startswith("win"):
    # Windows: require Python 3.10+, but no auto-relaunch
    if sys.version_info < (3, 10):
        print(
            "ERROR: Python 3.10 or later is required on Windows.\n"
            "Please download and install it from https://www.python.org/downloads/windows/"
        )
        sys.exit(1)

# CTRL-C handler
def _exit_on_sigint(signum, frame):
    print("\nInterrupted. Shutting down.")
    sys.exit(0)
signal.signal(signal.SIGINT, _exit_on_sigint)

# Logging helper
COLOR_RESET   = "\033[0m"
COLOR_INFO    = "\033[94m"
COLOR_SUCCESS = "\033[92m"
COLOR_WARNING = "\033[93m"
COLOR_ERROR   = "\033[91m"
COLOR_PROCESS = "\033[96m"

def log_message(msg: str, category: str="INFO"):
    cat = category.upper()
    color = {
        "INFO":    COLOR_INFO,
        "SUCCESS": COLOR_SUCCESS,
        "WARNING": COLOR_WARNING,
        "ERROR":   COLOR_ERROR,
        "PROCESS": COLOR_PROCESS,
    }.get(cat, COLOR_RESET)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}[{ts}] {cat}: {msg}{COLOR_RESET}")

def in_virtualenv() -> bool:
    base = getattr(sys, "base_prefix", None)
    return base is not None and sys.prefix != base

def create_and_activate_venv():
    venv_dir = os.path.join(os.getcwd(), ".venv")

    # 1) Find or install python3.10 on Debian/Ubuntu, macOS, or accept any Python >=3.10 on Windows
    py310 = shutil.which("python3.10")
    if not py310:
        if platform.system() == "Linux" and shutil.which("apt-get"):
            log_message("python3.10 not found—adding Deadsnakes PPA & installing...", "PROCESS")
            try:
                subprocess.check_call(["sudo","apt-get","update"])
                subprocess.check_call(["sudo","apt-get","install","-y","software-properties-common"])
                subprocess.check_call(["sudo","add-apt-repository","-y","ppa:deadsnakes/ppa"])
                subprocess.check_call(["sudo","apt-get","update"])
                subprocess.check_call([
                    "sudo","apt-get","install","-y",
                    "python3.10","python3.10-venv","python3.10-distutils"
                ])
                py310 = shutil.which("python3.10")
            except subprocess.CalledProcessError as e:
                log_message(f"Failed to install python3.10: {e}", "ERROR")

        elif platform.system() == "Darwin" and shutil.which("brew"):
            log_message("python3.10 not found—installing via Homebrew...", "PROCESS")
            try:
                subprocess.check_call(["brew","update"])
                subprocess.check_call(["brew","install","python@3.10"])
                py310 = shutil.which("python3.10")
            except subprocess.CalledProcessError as e:
                log_message(f"Failed to install python3.10 via Homebrew: {e}", "ERROR")

        elif platform.system().startswith("Win"):
            log_message("python3.10 not found - checking 'python' for version >=3.10...", "PROCESS")
            candidate = shutil.which("python") or shutil.which("python3")
            if candidate:
                try:
                    out = subprocess.check_output([candidate, "--version"],
                                                  stderr=subprocess.STDOUT,
                                                  text=True).strip()
                    _, version = out.split()
                    major, minor, *_ = version.split(".")
                    if int(major) == 3 and int(minor) >= 10:
                        py310 = candidate
                        log_message(f"Using {candidate} (version {version})", "PROCESS")
                    else:
                        log_message(f"{candidate} is Python {version}, which is <3.10", "WARNING")
                except Exception as e:
                    log_message(f"Failed to check {candidate} version: {e}", "ERROR")
            if not py310:
                log_message("No acceptable Python >=3.10 found on Windows—falling back to current Python", "WARNING")

    # 2) Fallback to current interpreter if still missing
    if not py310:
        log_message("python3.10 unavailable—falling back to current Python", "WARNING")
        py310 = sys.executable

    # 3) Determine python & pip paths inside the venv
    if platform.system().startswith("Win"):
        python_bin = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_bin    = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_bin = os.path.join(venv_dir, "bin", "python")
        pip_bin    = os.path.join(venv_dir, "bin", "pip")

    # 4) Create venv if needed
    if not os.path.isdir(venv_dir):
        log_message(f"Creating virtualenv in .venv/ with {os.path.basename(py310)}", "PROCESS")
        subprocess.check_call([py310, "-m", "venv", venv_dir])
        log_message("Upgrading pip in venv…", "PROCESS")
        subprocess.check_call([pip_bin, "install", "--upgrade", "pip"])

    # 5) Re-exec into the venv
    log_message("Re-launching under virtualenv…", "PROCESS")
    new_env = os.environ.copy()
    new_env["VIRTUAL_ENV"] = venv_dir
    if not platform.system().startswith("Win"):
        new_env["PATH"] = f"{venv_dir}/bin:{new_env.get('PATH','')}"

    os.execve(
        python_bin,
        [python_bin] + sys.argv,
        new_env
    )

if not in_virtualenv():
    create_and_activate_venv()


# ──────────── FIRST-RUN DEPENDENCIES ─────────────────────────────────────────

SETUP_MARKER = os.path.join(os.path.dirname(__file__), ".setup_complete")
if not os.path.exists(SETUP_MARKER):
    log_message("Installing system & Python deps…", "PROCESS")

    # Determine architecture for the correct Chromium package
    arch = platform.machine().lower()
    chromium_pkg = "chromium-browser" if arch in ("aarch64", "arm64") else "chromium"

    # System packages on Debian/Ubuntu
    if sys.platform.startswith("linux") and shutil.which("apt-get"):
        log_message("Installing system packages via apt-get...", "PROCESS")
        subprocess.check_call(["sudo", "apt-get", "update"])
        subprocess.check_call([
            "sudo", "apt-get", "install", "-y",
            "libsqlite3-dev",
            "ffmpeg",
            "wget",
            "unzip",
            chromium_pkg,
            "python3.10-venv",
            "python3.11-venv",
            "python3.12-venv",
            "curl"
        ])

    # System packages on macOS
    elif sys.platform == "darwin" and shutil.which("brew"):
        log_message("Installing system packages via Homebrew...", "PROCESS")
        subprocess.check_call(["brew", "update"])
        subprocess.check_call([
            "brew", "install",
            "sqlite3",
            "ffmpeg",
            "wget",
            "unzip",
            "portaudio",
            "python3.10-venv",
            "python3.11-venv",
            "python3.12-venv",
            "chromium",
            "curl"
        ])

    # System packages on Windows
    elif sys.platform == "win32":
        log_message("Installing system packages on Windows...", "PROCESS")
        if shutil.which("choco"):
            subprocess.check_call([
                "choco", "install", "-y",
                "sqlite",
                "ffmpeg",
                "wget",
                "unzip",
                "portaudio",
                "python3.10-venv",
                "python3.11-venv",
                "python3.12-venv",
                "chromium",
                "curl"
            ])
        else:
            log_message("Chocolatey not found; skipping system package installation on Windows", "WARNING")

    else:
        log_message("No recognized system package manager; skipping system package installation", "WARNING")


    # Python packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    deps = [
        "sounddevice","numpy","scipy","openai-whisper","ollama",
        "python-dotenv","beautifulsoup4","html5lib","psutil",
        "noisereduce","denoiser","pillow","opencv-python",
        "mss","networkx","pandas","selenium","webdriver-manager",
        "flask_cors","flask","tiktoken","python-telegram-bot",
        "nest-asyncio","sentence-transformers","telegram","num2words"
    ]
    # on Linux or Windows, also install the separate 'asyncio' package
    if sys.platform.startswith("linux"):
        deps.append("asyncio")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)

    # mark setup complete and restart under the same interpreter
    with open(SETUP_MARKER, "w") as f:
        f.write("done")
    log_message("Dependencies installed. Restarting…", "SUCCESS")
    os.execv(sys.executable, [sys.executable] + sys.argv)
    
CONFIG_FILE = "config.json"
DEFAULT_CFG = {
    # core LLM models
    "primary_model":   "gemma3:4b",
    "secondary_model": "gemma3:4b",
    "decision_model":  "gemma3:1b",

    "whisper_full_model": "medium",
    "whisper_consensus_model": "base",
    "whisper_device": "auto",

    # audio thresholds
    "sample_rate":         16000,
    "rms_threshold":       0.02,
    "silence_duration":    0.5,
    "consensus_threshold": 0.6,
    "enable_noise_reduction": True,

    # Piper release base URL & local executable name
    "piper_base_url":    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/",
    "piper_executable":  "piper",
    "piper_release_linux_x86_64": "piper_linux_x86_64.tar.gz",
    "piper_release_linux_arm64":  "piper_linux_aarch64.tar.gz",
    "piper_release_linux_armv7l": "piper_linux_armv7l.tar.gz",
    "piper_release_macos_x64":    "piper_macos_x64.tar.gz",
    "piper_release_macos_arm64":  "piper_macos_aarch64.tar.gz",
    "piper_release_windows":      "piper_windows_amd64.zip",

    # ONNX assets
    "onnx_json_filename":  "glados_piper_medium.onnx.json",
    "onnx_model_filename": "glados_piper_medium.onnx",
    "onnx_json_url":  "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json",
    "onnx_model_url": "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx",
}

try:
    import torch
    dev = config.get("whisper_device", "auto")
    if dev not in ("auto", "cpu", "cuda"):
        dev = "auto"
    if dev == "cuda" and not torch.cuda.is_available():
        log_message("whisper_device=cuda requested but CUDA not available; using cpu.", "WARNING")
        dev = "cpu"
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    config["whisper_device"] = dev

    # Defaults if missing/empty
    if not str(config.get("whisper_full_model", "")).strip():
        config["whisper_full_model"] = "medium"
    if not str(config.get("whisper_consensus_model", "")).strip():
        config["whisper_consensus_model"] = "base"
except Exception:
    pass


def _run_quiet(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except subprocess.CalledProcessError as e:
        return e.returncode, (e.output or "").strip()

def ensure_ollama_running():
    """
    Make sure the Ollama daemon is up. Prefer system service; fall back to foreground.
    """
    # Quick sanity check
    rc, _ = _run_quiet(["ollama", "list"])
    if rc == 0:
        return

    # Try systemd service (Linux)
    if shutil.which("systemctl"):
        _run_quiet(["sudo", "systemctl", "daemon-reload"])
        _run_quiet(["sudo", "systemctl", "enable", "--now", "ollama"])
        time.sleep(1.5)
        rc, _ = _run_quiet(["ollama", "list"])
        if rc == 0:
            return

    # Try launchctl (macOS, official install.sh sets a launchd plist)
    if sys.platform == "darwin" and shutil.which("launchctl"):
        _run_quiet(["launchctl", "kickstart", "-k", "gui/$(id -u)/com.ollama.ollama"])
        time.sleep(1.5)
        rc, _ = _run_quiet(["ollama", "list"])
        if rc == 0:
            return

    # Last resort: spawn a background daemon in this session
    if shutil.which("nohup"):
        # avoid blocking; discard output
        subprocess.Popen(["nohup", "ollama", "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None)
        time.sleep(1.5)

def install_or_upgrade_ollama():
    """
    Install or upgrade Ollama to latest stable for the platform.
    Safe to run repeatedly. Requires sudo on Linux for system install.
    """
    log_message("Upgrading Ollama to the latest version…", "PROCESS")

    if sys.platform.startswith("linux"):
        # Official installer handles install and upgrade
        dl_cmd = "curl -fsSL" if shutil.which("curl") else ("wget -qO-" if shutil.which("wget") else None)
        if not dl_cmd:
            subprocess.check_call(["sudo", "apt-get", "update"])
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "curl"])
            dl_cmd = "curl -fsSL"
        subprocess.check_call(["sh", "-c", f"{dl_cmd} https://ollama.com/install.sh | sh"])
        # installer usually writes/updates systemd service
        _run_quiet(["sudo", "systemctl", "daemon-reload"])
        _run_quiet(["sudo", "systemctl", "enable", "--now", "ollama"])

    elif sys.platform == "darwin":
        # Prefer Homebrew if present, else install.sh
        if shutil.which("brew"):
            subprocess.check_call(["brew", "update"])
            # brew formula name is 'ollama'
            subprocess.check_call(["brew", "upgrade", "ollama"])
            subprocess.check_call(["brew", "link", "--overwrite", "ollama"])
        else:
            dl_cmd = "curl -fsSL" if shutil.which("curl") else ("wget -qO-" if shutil.which("wget") else None)
            if not dl_cmd:
                raise RuntimeError("curl or wget required to install Ollama on macOS")
            subprocess.check_call(["sh", "-c", f"{dl_cmd} https://ollama.com/install.sh | sh"])

    elif sys.platform.startswith("win"):
        # Prefer winget; fall back to choco if available
        if shutil.which("winget"):
            # Silent upgrade if available; winget exits 0 even when already latest
            subprocess.call(["winget", "install", "-e", "--id", "Ollama.Ollama", "--silent", "--accept-package-agreements", "--accept-source-agreements"])
            subprocess.call(["winget", "upgrade", "-e", "--id", "Ollama.Ollama", "--silent", "--accept-package-agreements", "--accept-source-agreements"])
        elif shutil.which("choco"):
            subprocess.call(["choco", "upgrade", "-y", "ollama"])
        else:
            raise RuntimeError("Please install Ollama manually on Windows (winget or choco not found).")

    else:
        raise RuntimeError(f"Unsupported platform for Ollama upgrade: {sys.platform}")

    # after install/upgrade, ensure the daemon is running
    ensure_ollama_running()

def is_ollama_too_old_error(exc: Exception) -> bool:
    msg = str(exc) if exc else ""
    # match both the HTTP status and the friendly text Ollama returns
    needles = [
        "requires a newer version of Ollama",
        "412",  # HTTP 412 from API
        "pull model manifest: 412",
    ]
    return any(n in msg for n in needles)

# 1) Load existing config.json (or start empty)
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = {}

# 2) Strip any stray bot_token from config
config.pop("bot_token", None)

# ──────────── ENSURE .env and BOT_TOKEN ────────────────────────────────────
ENV_FILE = ".env"
if not os.path.exists(ENV_FILE):
    # create and prompt for BOT_TOKEN
    with open(ENV_FILE, "w") as f:
        f.write("BOT_TOKEN=\n")
    print("Please obtain a Telegram bot token from BotFather:")
    print("https://telegram.me/BotFather")
    token = input("Paste your BOT_TOKEN here: ").strip()
    with open(ENV_FILE, "w") as f:
        f.write(f"BOT_TOKEN={token}\n")

# load BOT_TOKEN from .env (do NOT write back into config.json)
from dotenv import load_dotenv
load_dotenv(ENV_FILE)
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# 3) Merge in any missing defaults (only DEFAULT_CFG keys)
updated = False
for key, val in DEFAULT_CFG.items():
    if key not in config:
        config[key] = val
        updated = True

# 4) Persist config.json if we added defaults
if updated:
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    log_message(f"Added missing defaults into {CONFIG_FILE}", "INFO")

# 5) Make BOT_TOKEN available at runtime—but never persist it on disk in config.json
config["bot_token"] = BOT_TOKEN

import shutil
import subprocess
import sys

# ─── Ensure ollama CLI is installed on every run ───────────────────────────
if shutil.which("ollama") is None:
    log_message("ollama CLI not found; installing via official script…", "PROCESS")

    # pick a downloader
    if shutil.which("curl"):
        dl_cmd = "curl -fsSL"
    elif shutil.which("wget"):
        dl_cmd = "wget -qO-"
    else:
        # install curl if we can
        if sys.platform.startswith("linux") and shutil.which("apt-get"):
            subprocess.check_call(["sudo", "apt-get", "update"])
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "curl"])
            dl_cmd = "curl -fsSL"
        elif sys.platform == "darwin" and shutil.which("brew"):
            subprocess.check_call(["brew", "install", "curl"])
            dl_cmd = "curl -fsSL"
        else:
            raise RuntimeError("Neither curl nor wget available to install ollama CLI")

    # run the official installer
    subprocess.check_call([
        "sh", "-c",
        f"{dl_cmd} https://ollama.com/install.sh | sh"
    ])

ensure_ollama_running()

# ──────────── PULL Ollama MODELS IF NEEDED ──────────────────────────────
import ollama

# 1) List locally-available specs
try:
    listing = ollama.list()
    raw_entries = listing.models
except Exception as e:
    log_message(f"Failed to fetch Ollama model list: {e}", "ERROR")
    raw_entries = []

# Extract just the model spec strings
available = [m.model for m in raw_entries]
available_set = set(available)

# Print a clean list
print("[Ollama] locally available models:")
for name in sorted(available):
    print(f"  • {name}")

# ─── Helper to render a single-line progress bar ────────────────────────────
def render_bar(completed: int, total: int, width: int = 40) -> str:
    # coerce None→0, avoid division by zero
    comp = completed or 0
    tot  = total or 0
    pct  = (comp / tot) if tot else 0.0
    filled = int(pct * width)
    bar = "█" * filled + "-" * (width - filled)
    return f"[{bar}] {pct*100:6.2f}%"

# ─── Pull any missing specs ─────────────────────────────────────────────────
for model_spec in (
    config.get("primary_model"),
    config.get("secondary_model"),
    config.get("decision_model"),
):
    if not model_spec or model_spec in available_set:
        continue

    log_message(f"Model '{model_spec}' not found locally — pulling with Ollama…", "PROCESS")
    try:
        for status in ollama.pull(model_spec, stream=True):
            # status dict may have None for completed/total until it really starts
            comp = status.get("completed")  # might be None
            tot  = status.get("total")      # might be None
            bar_line = render_bar(comp, tot)
            sys.stdout.write(f"\r[Ollama:pull {model_spec}] {bar_line}")
            sys.stdout.flush()
        print()  # newline once done
        log_message(f"Successfully pulled Ollama model '{model_spec}'.", "SUCCESS")
        available_set.add(model_spec)

    except Exception as pull_err:
        print()  # clear the bar line
        if is_ollama_too_old_error(pull_err):
            log_message("Ollama is too old for this model. Upgrading…", "WARNING")
            try:
                install_or_upgrade_ollama()
                log_message("Retrying model pull after upgrade…", "PROCESS")
                ensure_ollama_running()
                for status in ollama.pull(model_spec, stream=True):
                    comp = status.get("completed")
                    tot  = status.get("total")
                    bar_line = render_bar(comp, tot)
                    sys.stdout.write(f"\r[Ollama:pull {model_spec}] {bar_line}")
                    sys.stdout.flush()
                print()
                log_message(f"Successfully pulled Ollama model '{model_spec}' after upgrade.", "SUCCESS")
                available_set.add(model_spec)
                continue
            except Exception as e2:
                print()
                log_message(f"Retry pull failed after upgrade: {e2}", "ERROR")
        else:
            log_message(f"Error pulling Ollama model '{model_spec}': {pull_err}", "WARNING")



# ──────────── PIPER + ONNX SETUP ─────────────────────────────────────────────
def setup_piper_and_onnx():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    piper_folder = os.path.join(script_dir, "piper")
    exe_name     = config["piper_executable"]
    piper_exe    = os.path.join(piper_folder, exe_name)
    log_message(f"Checking for Piper at {piper_exe}", "INFO")

    # pick the correct archive name
    os_name = platform.system()
    arch    = platform.machine().lower()
    if os_name == "Linux":
        if arch == "x86_64":
            release = config["piper_release_linux_x86_64"]
        elif arch in ("arm64", "aarch64"):
            release = config["piper_release_linux_arm64"]
        else:
            release = config["piper_release_linux_armv7l"]
    elif os_name == "Darwin":
        if arch in ("arm64", "aarch64"):
            release = config["piper_release_macos_arm64"]
        else:
            release = config["piper_release_macos_x64"]
    elif os_name == "Windows":
        release = config["piper_release_windows"]
    else:
        log_message(f"Unsupported OS: {os_name}", "ERROR")
        sys.exit(1)

    # download utility: use wget if present, otherwise curl
    def _dl(url: str, dest: str):
        if shutil.which("wget"):
            cmd = ["wget", "-O", dest, url]
        elif shutil.which("curl"):
            cmd = ["curl", "-L", "-o", dest, url]
        else:
            log_message("Neither wget nor curl found; cannot download files.", "ERROR")
            sys.exit(1)
        subprocess.check_call(cmd)

    # download & unpack Piper if missing
    if not os.path.isfile(piper_exe):
        url     = config["piper_base_url"] + release
        archive = os.path.join(script_dir, release)
        log_message(f"Downloading Piper: {release}", "PROCESS")
        _dl(url, archive)

        os.makedirs(piper_folder, exist_ok=True)
        if release.endswith(".tar.gz"):
            subprocess.check_call(["tar", "-xzvf", archive, "-C", piper_folder, "--strip-components=1"])
        else:
            subprocess.check_call(["unzip", "-o", archive, "-d", piper_folder])
        log_message("Piper unpacked.", "SUCCESS")
    else:
        log_message("Piper executable already present.", "SUCCESS")

    # ONNX JSON
    onnx_json = os.path.join(script_dir, config["onnx_json_filename"])
    if not os.path.isfile(onnx_json):
        log_message("Downloading ONNX JSON…", "PROCESS")
        _dl(config["onnx_json_url"], onnx_json)
    else:
        log_message(f"Found ONNX JSON: {config['onnx_json_filename']}", "SUCCESS")

    # ONNX model
    onnx_model = os.path.join(script_dir, config["onnx_model_filename"])
    if not os.path.isfile(onnx_model):
        log_message("Downloading ONNX model…", "PROCESS")
        _dl(config["onnx_model_url"], onnx_model)
    else:
        log_message(f"Found ONNX model: {config['onnx_model_filename']}", "SUCCESS")

# finally, run it
setup_piper_and_onnx()

# ─────────── globals for pipelines ─────────────────────────────────────────
audio_svc = None
tts_audio = None
asm_audio = None
tts_cli   = None
asm_cli   = None
tts_tele  = None
asm_tele  = None

# ─── IMPORT CORE CLASSES ───────────────────────────────────────────────────
from audio_service import AudioService
from tts_service import TTSManager
from telegram_input import notify_admin, telegram_input
import threading, subprocess, os, sys, time, traceback

import asyncio, threading, traceback
from assembler import Assembler

CTX_PATH    = "context.jsonl"
CONFIG_FILE = "config.json"  # adjust as needed


# ─── Create one persistent event loop for the audio thread ───────────────
_audio_loop = asyncio.new_event_loop()

def start_audio_pipeline():
    global audio_svc, tts_audio, asm_audio  # ← add this

    asyncio.set_event_loop(_audio_loop)
    audio_svc = AudioService(   # ← now the global gets the instance
        sample_rate         = config["sample_rate"],
        rms_threshold       = config["rms_threshold"],
        silence_duration    = config["silence_duration"],
        consensus_threshold = config["consensus_threshold"],
        enable_denoise      = config["enable_noise_reduction"],
        on_transcription    = lambda text: None,
        logger              = log_message,
        cfg                 = config,
        denoise_fn          = None,
    )
    tts_audio = TTSManager(
        logger=log_message,
        cfg=config,
        audio_service=audio_svc
    )
    tts_audio.set_mode("live")

    asm_audio = Assembler(
        context_path     = CTX_PATH,
        config_path      = CONFIG_FILE,
        lookback_minutes = 60,
        top_k            = 5,
        tts_manager      = tts_audio,
    )

    # 3) Monkey‑patch the transcription callback to schedule on _audio_loop
    def _audio_input_cb(text: str):
        try:
            # schedule the async run (it will cancel/restart previous in-flight turn)
            future = asyncio.run_coroutine_threadsafe(
                asm_audio.run_with_meta_context(text),
                _audio_loop
            )
            # optionally handle the result (enqueue TTS) when it completes:
            def _on_done(fut):
                try:
                    answer = fut.result()
                    if answer and answer.strip():
                        tts_audio.enqueue(answer)
                except Exception as e:
                    tb = traceback.format_exc()
                    log_message(f"Audio callback error:\n{tb}", "ERROR")
                    notify_admin(f"⚠️ *Audio callback error*:\n```{tb[:1500]}```")
            future.add_done_callback(_on_done)

        except Exception:
            tb = traceback.format_exc()
            log_message(f"Audio callback scheduling failed:\n{tb}", "ERROR")
            notify_admin(f"⚠️ *Audio scheduling error*:\n```{tb[:1500]}```")

    audio_svc.on_transcription = _audio_input_cb

    # 4) Start the microphone/transcription loop
    audio_svc.start()

    # 5) Run the loop forever so cancellations & restarts work reliably
    _audio_loop.run_forever()

# ─── Kick off the audio thread once ─────────────────────────────────────────
threading.Thread(
    target=start_audio_pipeline,
    daemon=True,
    name="AudioThread"
).start()

# ─── 2) CLI PIPELINE ──────────────────────────────────────────────────────
def start_cli_pipeline():
    """
    Async-safe CLI loop:
    - awaits asm.run_with_meta_context()
    - streams tokens to TTS in live mode
    - prints stage updates
    """
    import asyncio, inspect, traceback, threading
    from tts_service import TTSManager
    from assembler import Assembler

    async def _cli_main():
        global tts_cli, asm_cli   # <-- use globals, not nonlocal

        # ----- init TTS + Assembler -----
        tts_cli = TTSManager(
            logger=log_message,
            cfg=config,
            audio_service=None
        )
        tts_cli.set_mode("live")

        asm_cli = Assembler(
            context_path     = CTX_PATH,
            config_path      = CONFIG_FILE,
            lookback_minutes = 60,
            top_k            = 5,
            tts_manager      = tts_cli,
        )

        # status callback for console
        def cli_status(stage: str, info=None):
            snippet = (str(info) if info is not None else "").replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "…"
            print(f"[{stage}] {snippet}")

        print("Ready (CLI): type your message, Ctrl-C to exit.")
        while True:
            try:
                line = await asyncio.to_thread(input, ">> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not line.strip():
                continue

            # token sink (if available) for ultra-low-latency TTS
            sink = getattr(tts_cli, "token_sink", None)
            sink = sink() if callable(sink) else None

            try:
                answer = await asm_cli.run_with_meta_context(
                    line.strip(),
                    status_cb=cli_status,
                    images=None,
                    on_token=sink
                )
                # just in case something upstream returned a coroutine
                if inspect.iscoroutine(answer):
                    answer = await answer
            except Exception:
                tb = traceback.format_exc()
                log_message(f"Run failed:\n{tb}", "ERROR")
                notify_admin(f"⚠️ *CLI turn failed*:\n```{tb[:1500]}```")
                answer = ""

            # flush pending audio chunks
            if sink:
                try: sink(None)
                except Exception: pass

            if answer and answer.strip():
                print(answer)
                try:
                    tts_cli.enqueue(answer)
                except Exception:
                    pass

        print("CLI loop exiting…")

    try:
        asyncio.run(_cli_main())
    except Exception:
        tb = traceback.format_exc()
        log_message(f"CLI pipeline startup failed:\n{tb}", "ERROR")
        notify_admin(f"⚠️ *CLI pipeline startup failed*:\n```{tb[:1500]}```")


# kick it off in a background thread (unchanged)
threading.Thread(
    target=start_cli_pipeline,
    daemon=True,
    name="CLIThread"
).start()


# ─── 3) TELEGRAM PIPELINE ─────────────────────────────────────────────────
def start_telegram_pipeline():
    global tts_tele, asm_tele
    try:
        from assembler import Assembler

        tts_tele = TTSManager(
            logger=log_message,
            cfg=config,
            audio_service=None
        )
        tts_tele.set_mode("file")

        asm_tele = Assembler(
            context_path     = CTX_PATH,
            config_path      = CONFIG_FILE,
            lookback_minutes = 60,
            top_k            = 5,
            tts_manager      = tts_tele,
        )

        telegram_input(asm_tele)

    except Exception:
        tb = traceback.format_exc()
        log_message(f"Telegram pipeline startup failed:\n{tb}", "ERROR")
        notify_admin(f"⚠️ *Telegram pipeline startup failed*:\n```{tb[:1500]}```")

threading.Thread(
    target=start_telegram_pipeline,
    daemon=True,
    name="TelegramThread"
).start()

# ─── 4) WATCHER FOR FILE CHANGES & GIT — with graceful reload ───────────────

# Single reload guard
_RELOAD_LOCK = threading.Lock()
_RELOAD_FLAG = False
_LAST_RELOAD_TS = 0.0
_RELOAD_DEBOUNCE_SEC = 0.75

def _graceful_shutdown():
    """Stop services and free GPU memory (Whisper) before reload/exit."""
    try:
        # Stop audio/TTS services if they exist
        try:
            if 'audio_svc' in globals() and audio_svc:
                audio_svc.stop()
        except Exception:
            pass
        try:
            if 'tts_audio' in globals() and tts_audio:
                tts_audio.stop()
        except Exception:
            pass
        try:
            if 'tts_cli' in globals() and tts_cli:
                tts_cli.stop()
        except Exception:
            pass
        try:
            if 'tts_tele' in globals() and tts_tele:
                tts_tele.stop()
        except Exception:
            pass

        # Stop the dedicated audio loop if we created one
        try:
            if '_audio_loop' in globals() and _audio_loop and _audio_loop.is_running():
                _audio_loop.call_soon_threadsafe(_audio_loop.stop)
        except Exception:
            pass

        # Release Whisper VRAM across the process
        try:
            from audio_service import release_all_whisper_models
            release_all_whisper_models()
        except Exception:
            # If import fails for any reason, just continue
            pass
    except Exception:
        tb = traceback.format_exc()
        log_message(f"Shutdown error:\n{tb}", "WARNING")


def trigger_reload(reason: str):
    """Idempotent reload that cleans up first, then execv's the process."""
    global _RELOAD_FLAG, _LAST_RELOAD_TS
    with _RELOAD_LOCK:
        now = time.time()
        if _RELOAD_FLAG and (now - _LAST_RELOAD_TS) < 10.0:
            return
        _RELOAD_FLAG = True
        _LAST_RELOAD_TS = now

    try:
        log_message(f"Detected change ({reason}); restarting…", "INFO")
        notify_admin(f"🔄 *Reload triggered by* `{reason}`")
    except Exception:
        pass

    # Clean up services and free GPU memory
    _graceful_shutdown()

    # Exec new process image (won't run atexit handlers)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _monitor_git_and_files(interval: float = 5.0):
    def _run(cmd):
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()

    repo_dir = os.path.dirname(os.path.abspath(__file__))

    def _watched_paths():
        """Return a set of files to watch (top-level .py + CONFIG_FILE)."""
        keep = set()
        for f in os.listdir(repo_dir):
            if f.endswith(".py") or f == CONFIG_FILE:
                keep.add(os.path.join(repo_dir, f))
        return keep

    def _snapshot(paths):
        snap = {}
        for p in paths:
            try:
                st = os.stat(p)
                snap[p] = (st.st_mtime, st.st_size)
            except OSError:
                # Missing file is a change; mark with None
                snap[p] = None
        return snap

    paths = _watched_paths()
    last_snap = _snapshot(paths)

    while True:
        try:
            # 1) File/config changes (debounced)
            curr_paths = _watched_paths()
            # handle new files
            for p in curr_paths - paths:
                last_snap[p] = None
            # handle removed files
            for p in paths - curr_paths:
                last_snap.pop(p, None)

            paths = curr_paths
            curr_snap = _snapshot(paths)

            changed = []
            for p in paths:
                if last_snap.get(p) != curr_snap.get(p):
                    changed.append(p)

            if changed:
                # debounce: wait a beat, re-check once
                time.sleep(_RELOAD_DEBOUNCE_SEC)
                curr_snap2 = _snapshot(paths)
                stable = []
                for p in changed:
                    if curr_snap2.get(p) == curr_snap.get(p):
                        stable.append(p)
                if stable:
                    reason = ", ".join(os.path.basename(p) for p in stable[:3])
                    trigger_reload(reason)
                    return  # just in case

            last_snap = curr_snap

            # 2) Git updates
            try:
                branch = _run(["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"])
                _run(["git", "-C", repo_dir, "fetch"])
                behind = int(_run(["git", "-C", repo_dir, "rev-list", f"HEAD..origin/{branch}", "--count"]))
                if behind > 0:
                    pull_out = _run(["git", "-C", repo_dir, "pull", "--ff-only"])
                    log_message(f"Git pull succeeded:\n{pull_out}", "SUCCESS")
                    notify_admin(f"🔄 *Git update*: `{branch}` +{behind} commits")
                    trigger_reload(f"git pull on {branch}")
                    return
            except subprocess.CalledProcessError as e:
                err = (e.output or "").strip()
                log_message(f"Git watcher error: {err}", "WARNING")
                notify_admin(f"⚠️ *Git watcher error*:\n```{err[:1500]}```")
            except Exception:
                tb = traceback.format_exc()
                log_message(f"Watcher exception (git):\n{tb}", "WARNING")
                notify_admin(f"⚠️ *Watcher exception (git)*:\n```{tb[:1500]}```")

        except Exception:
            tb = traceback.format_exc()
            log_message(f"Watcher loop exception:\n{tb}", "WARNING")
            try:
                notify_admin(f"⚠️ *Watcher loop exception*:\n```{tb[:1500]}```")
            except Exception:
                pass

        time.sleep(interval)


# Start the watcher thread
threading.Thread(
    target=_monitor_git_and_files,
    kwargs={"interval": 5.0},
    daemon=True,
    name="GitAndFileWatcher"
).start()


# ─── CLEANUP & WAIT ───────────────────────────────────────────────────────
def _cleanup():
    log_message("Shutting down services…", "INFO")
    _graceful_shutdown()
    log_message("Goodbye.", "INFO")

# 1) Always run cleanup on normal exit
atexit.register(_cleanup)

# 2) On SIGINT/SIGTERM, cleanup then _immediately_ kill the process
def _signal_handler(signum, frame):
    log_message(f"Received signal {signum}, exiting…", "INFO")
    _cleanup()
    os._exit(0)   # bypass Python shutdown hooks & thread.join()

signal.signal(signal.SIGINT,  _signal_handler)   # Ctrl-C
signal.signal(signal.SIGTERM, _signal_handler)   # kill, docker stop

# 3) Ignore Ctrl-Z (SIGTSTP) so you can't background it
if hasattr(signal, "SIGTSTP"):
    signal.signal(signal.SIGTSTP, lambda s, f: None)

# 4) Block here until a signal arrives
signal.pause()
