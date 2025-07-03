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
import time
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

    onnx_json  = os.path.join(script_dir, "overwatch.onnx.json")
    onnx_model = os.path.join(script_dir, "overwatch.onnx")
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
        "asyncio","nest-asyncio","sentence-transformers", "telegram", "num2words"
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


# ─── IMPORT THE CORE CLASSES ──────────────────────────────────────────────
from assembler     import Assembler
from audio_service import AudioService
from tts_service   import TTSManager
from telegram_input import telegram_input

CTX_PATH = "context.jsonl"

# ─── 1) AUDIO PIPELINE ────────────────────────────────────────────────────
audio_svc = AudioService(
    sample_rate         = config.get("sample_rate",        16000),
    rms_threshold       = config.get("rms_threshold",      0.01),
    silence_duration    = config.get("silence_duration",   0.5),
    consensus_threshold = config.get("consensus_threshold",0.3),
    enable_denoise      = config.get("enable_noise_reduction", False),
    on_transcription    = None,      # set below
    logger              = log_message,
    cfg                 = config,
)
tts_audio = TTSManager(
    logger        = log_message,
    cfg           = config,
    audio_service = audio_svc,     # live‐playback on speaker
)
tts_audio.set_mode("live")
asm_audio = Assembler(
    context_path     = CTX_PATH,
    config_path      = "config.json",
    lookback_minutes = 60,
    top_k            = 5,
    tts_manager      = tts_audio,
)

# —– **Changed**: catch assembler’s return and enqueue it for live TTS
def _audio_input_cb(text: str):
    answer = asm_audio.run_with_meta_context(text)
    if answer and answer.strip():
        tts_audio.enqueue(answer)

audio_svc.on_transcription = _audio_input_cb

# start audio in its own thread
threading.Thread(target=audio_svc.start, daemon=True).start()


# ─── 2) CLI PIPELINE ──────────────────────────────────────────────────────
def cli_loop():
    tts_cli = TTSManager(
        logger        = log_message,
        cfg           = config,
        audio_service = audio_svc   # also speak on speaker
    )
    tts_cli.set_mode("live")
    asm_cli = Assembler(
        context_path     = CTX_PATH,
        config_path      = "config.json",
        lookback_minutes = 60,
        top_k            = 5,
        tts_manager      = tts_cli,
    )

    print("Ready (CLI): type your message, Ctrl-C to exit.")
    while True:
        try:
            line = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        # 1) Run the assembler
        answer = asm_cli.run_with_meta_context(line)

        # 2) Enqueue for live TTS (so you hear it immediately)
        if answer and answer.strip():
            tts_cli.enqueue(answer)

        # (no print needed; the TTS will speak your response)
    print("CLI loop exiting…")


   
threading.Thread(target=cli_loop, daemon=True).start()


# ─── 3) FILE-WATCHER ───────────────────────────────────────────────────────
def _monitor_files(interval=1):
    paths = [
        os.path.abspath(__file__),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assembler.py")
    ]
    last_mtimes = {p: os.path.getmtime(p) for p in paths if os.path.exists(p)}
    while True:
        time.sleep(interval)
        for p in paths:
            try:
                m = os.path.getmtime(p)
                if last_mtimes.get(p) != m:
                    log_message(f"File change detected for '{os.path.basename(p)}', restarting...", "INFO")
                    # ── SHUT DOWN FLASK SERVER ───────────────────────────
                    srv = globals().get("_flask_server")
                    if srv:
                        log_message("Shutting down existing Flask server...", "INFO")
                        try:
                            srv.shutdown()
                        except Exception as e:
                            log_message(f"Error shutting down Flask server: {e}", "WARNING")
                    # ── RE-EXEC THE PROCESS ───────────────────────────────
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception:
                continue

threading.Thread(target=_monitor_files, daemon=True).start()


# ─── 4) TELEGRAM PIPELINE ─────────────────────────────────────────────────
try:
    tts_tele = TTSManager(
        logger        = log_message,
        cfg           = config,
        audio_service = None     # no speaker output
    )
    tts_tele.set_mode("file")
    asm_tele = Assembler(
        context_path     = CTX_PATH,
        config_path      = "config.json",
        lookback_minutes = 60,
        top_k            = 5,
        tts_manager      = tts_tele,
    )

    def _run_telegram():
        try:
            telegram_input(asm_tele)
        except Exception as e:
            print(f"Telegram thread error: {e}")

    threading.Thread(
        target=_run_telegram,
        daemon=True,
        name="TelegramThread"
    ).start()

except Exception as e:
    print(f"Error setting up Telegram thread: {e}")


# ─── 5) GIT‐REPO WATCHER ──────────────────────────────────────────────────────
def _monitor_git_updates(interval: float = 10.0):
    """
    Periodically fetch the remote and auto-pull if the local branch is behind.

    Args:
        interval (float, optional): Seconds between fetch+check cycles. Defaults to 10.0s.

    Behavior:
        • Runs `git rev-parse --abbrev-ref HEAD` to find the current branch.
        • Runs `git fetch` to update remote tracking refs.
        • Checks `git rev-list HEAD..origin/<branch> --count`. If > 0,
          logs and runs `git pull --ff-only`.
        • Any errors are caught and logged as warnings.

    Example:
        # Spawns in a background thread
        threading.Thread(target=_monitor_git_updates, daemon=True).start()
    """
    import subprocess
    import time

    def _run(cmd):
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()

    # determine repo root (this script’s dir)
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    while True:
        try:
            # 1) figure out the current branch name
            branch = _run(["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"])
            # 2) fetch from remote
            _run(["git", "-C", repo_dir, "fetch"])
            # 3) check if HEAD is behind origin/<branch>
            behind = int(_run([
                "git", "-C", repo_dir,
                "rev-list", "HEAD..origin/" + branch, "--count"
            ]))
            if behind > 0:
                log_message(f"Remote update detected on branch '{branch}' ({behind} new commit(s)), pulling…", "INFO")
                pull_out = _run(["git", "-C", repo_dir, "pull", "--ff-only"])
                log_message(f"Git pull succeeded:\n{pull_out}", "SUCCESS")
            # else: up-to-date; do nothing
        except subprocess.CalledProcessError as e:
            log_message(f"Git-watcher error: {e.output.strip()}", "WARNING")
        except Exception as ex:
            log_message(f"Unexpected error in git-watcher: {ex}", "WARNING")

        time.sleep(interval)

# — spawn the Git-watcher in the background ─────────────────────────────────
threading.Thread(target=_monitor_git_updates, daemon=True, name="GitWatcher").start()

# ─── WAIT FOR CTRL-C ───────────────────────────────────────────────────────
import atexit
def _cleanup():
    log_message("Shutting down services…", "INFO")
    audio_svc.stop()
    tts_audio.stop()
    # tts_cli and tts_tele threads will exit automatically
    log_message("Goodbye.", "INFO")
atexit.register(_cleanup)

threading.Event().wait()
