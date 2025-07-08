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

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import time
import json
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

    # System packages on Debian/Ubuntu
    if sys.platform.startswith("linux") and shutil.which("apt-get"):
        log_message("Installing system packages via apt-get...", "PROCESS")
        subprocess.check_call(["sudo", "apt-get", "update"])
        subprocess.check_call([
            "sudo", "apt-get", "install", "-y",
            "libsqlite3-dev", "ffmpeg", "wget", "unzip"
        ])

    # System packages on macOS
    elif sys.platform == "darwin" and shutil.which("brew"):
        log_message("Installing system packages via Homebrew...", "PROCESS")
        subprocess.check_call(["brew", "update"])
        subprocess.check_call([
            "brew", "install",
            "sqlite3", "ffmpeg", "wget", "unzip"
        ])

    # System packages on Windows
    elif sys.platform == "win32":
        log_message("Installing system packages on Windows...", "PROCESS")
        if shutil.which("choco"):
            subprocess.check_call([
                "choco", "install", "-y",
                "sqlite", "ffmpeg", "wget", "unzip"
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


# ──────────── LOAD / GENERATE config.json ────────────────────────────────────
CONFIG_FILE = "config.json"
DEFAULT_CFG = {
    # core LLM models
    "primary_model":   "gemma3:4b",
    "secondary_model": "gemma3:4b",
    "decision_model": "gemma3:1b",

    # audio thresholds
    "sample_rate":         16000,
    "rms_threshold":       0.01,
    "silence_duration":    0.5,
    "consensus_threshold": 0.3,
    "enable_noise_reduction": False,

    # Piper release base URL & local executable name
    "piper_base_url":    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/",
    "piper_executable":  "piper",  # name of the binary inside piper/
    # platform-specific Piper archives
    "piper_release_linux_x86_64": "piper_linux_x86_64.tar.gz",
    "piper_release_linux_arm64":  "piper_linux_aarch64.tar.gz",
    "piper_release_linux_armv7l": "piper_linux_armv7l.tar.gz",
    "piper_release_macos_x64":    "piper_macos_x64.tar.gz",
    "piper_release_macos_arm64":  "piper_macos_aarch64.tar.gz",
    "piper_release_windows":      "piper_windows_amd64.zip",

    # ONNX assets: point at your *local* filenames here
    "onnx_json_filename":  "glados_piper_medium.onnx.json",
    "onnx_model_filename": "glados_piper_medium.onnx",
    # …but also keep the URLs so we can download if missing…
    "onnx_json_url":  "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json",
    "onnx_model_url": "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx",
}

# load or init
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        config = json.load(f)
else:
    config = {}

# ──────────── CHECK / CREATE .env ─────────────────────────────
ENV_FILE = ".env"
if not os.path.exists(ENV_FILE):
    # 1) create .env with placeholder
    with open(ENV_FILE, "w") as f:
        f.write("BOT_TOKEN=\n")
    # 2) prompt user to obtain & enter token
    print("Please obtain a Telegram bot token from BotFather:")
    print("https://telegram.me/BotFather")
    token = input("Paste your BOT_TOKEN here: ").strip()
    # 3) write the real token back into .env
    with open(ENV_FILE, "w") as f:
        f.write(f"BOT_TOKEN={token}\n")
else:
    # load existing token if present
    with open(ENV_FILE) as f:
        for line in f:
            if line.startswith("BOT_TOKEN="):
                config["bot_token"] = line.strip().split("=", 1)[1]
                break

# fill in any missing defaults
updated = False
for k, v in DEFAULT_CFG.items():
    if k not in config:
        config[k] = v
        updated = True

if updated:
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    log_message(f"Added missing defaults into {CONFIG_FILE}", "INFO")
    
# ──────────── PULL Ollama MODELS IF NEEDED ──────────────────────────────
for model in (config.get("primary_model"), config.get("secondary_model"), config.get("decision_model")):
    # try at most twice: original pull, then (if needed) pull after upgrade
    for attempt in range(2):
        try:
            # Get the list of models already present
            existing = subprocess.check_output(
                ["ollama", "list"],
                stderr=subprocess.STDOUT,
                text=True
            )
            if model in existing:
                log_message(f"Ollama model '{model}' already present.", "INFO")
                break

            log_message(f"Model '{model}' not found locally—pulling with Ollama...", "PROCESS")

            # Run the pull and capture output so we can inspect failures
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                log_message(f"Successfully pulled Ollama model '{model}'.", "SUCCESS")
                break  # done with this model

            # ── Pull failed ────────────────────────────────────────────────
            combined_out = (result.stdout or "") + (result.stderr or "")
            needs_upgrade = (
                "requires a newer version of Ollama" in combined_out
                or "manifest: 412" in combined_out
            )

            if needs_upgrade and attempt == 0:
                log_message("Ollama is out of date; upgrading Ollama…", "PROCESS")
                try:
                    subprocess.check_call(
                        ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
                    )
                except subprocess.CalledProcessError as up_err:
                    log_message(f"Failed to upgrade Ollama: {up_err}", "ERROR")
                    break  # give up on this model
                else:
                    log_message("Upgrade complete—retrying pull…", "INFO")
                    continue  # retry pulling after upgrade
            else:
                log_message(f"Error pulling model '{model}': {combined_out.strip()}", "WARNING")
                break  # unrecoverable error; stop retrying

        except subprocess.CalledProcessError as e:
            log_message(f"Subprocess error with model '{model}': {e}", "WARNING")
            break


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

# globals for pipelines
audio_svc = None
tts_audio = None
asm_audio = None
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

import traceback

CTX_PATH = "context.jsonl"

# ─── 1) AUDIO PIPELINE ────────────────────────────────────────────────────
def start_audio_pipeline():
    global audio_svc, tts_audio, asm_audio
    try:
        from assembler import Assembler

        audio_svc = AudioService(
            sample_rate         = config["sample_rate"],
            rms_threshold       = config["rms_threshold"],
            silence_duration    = config["silence_duration"],
            consensus_threshold = config["consensus_threshold"],
            enable_denoise      = config["enable_noise_reduction"],
            on_transcription    = None,
            logger              = log_message,
            cfg                 = config,
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

        def _audio_input_cb(text: str):
            try:
                answer = asm_audio.run_with_meta_context(text)
                if answer and answer.strip():
                    tts_audio.enqueue(answer)
            except Exception:
                tb = traceback.format_exc()
                log_message(f"Audio callback error:\n{tb}", "ERROR")
                notify_admin(f"⚠️ *Audio callback error*:\n```{tb[:1500]}```")

        audio_svc.on_transcription = _audio_input_cb
        audio_svc.start()

    except Exception:
        tb = traceback.format_exc()
        log_message(f"Audio pipeline startup failed:\n{tb}", "ERROR")
        notify_admin(f"⚠️ *Audio pipeline startup failed*:\n```{tb[:1500]}```")

threading.Thread(
    target=start_audio_pipeline,
    daemon=True,
    name="AudioThread"
).start()

# ─── 2) CLI PIPELINE ──────────────────────────────────────────────────────
def start_cli_pipeline():
    global tts_cli, asm_cli
    try:
        from assembler import Assembler

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

        print("Ready (CLI): type your message, Ctrl-C to exit.")
        while True:
            try:
                line = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue

            answer = asm_cli.run_with_meta_context(line)
            if answer and answer.strip():
                tts_cli.enqueue(answer)

        print("CLI loop exiting…")

    except Exception:
        tb = traceback.format_exc()
        log_message(f"CLI pipeline startup failed:\n{tb}", "ERROR")
        notify_admin(f"⚠️ *CLI pipeline startup failed*:\n```{tb[:1500]}```")

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

        # ← previously you had notify_admin(asm_tele) by mistake
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

# ─── 4) WATCHER FOR FILE CHANGES & GIT ────────────────────────────────────
def _monitor_git_and_files(interval: float = 5.0):
    def _run(cmd):
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()

    repo_dir = os.path.dirname(os.path.abspath(__file__))

    def watched():
        return {
            os.path.join(repo_dir, f)
            for f in os.listdir(repo_dir)
            if f.endswith(".py") or f == CONFIG_FILE
        }

    last = {p: os.path.getmtime(p) for p in watched()}

    while True:
        # file/config changes
        for path, old in list(last.items()):
            try:
                new = os.path.getmtime(path)
            except OSError:
                new = None
            if new != old:
                log_message(f"Detected change in {os.path.basename(path)}; restarting…", "INFO")
                notify_admin(f"🔄 *Reload triggered by* `{os.path.basename(path)}`")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        for p in watched() - last.keys():
            last[p] = os.path.getmtime(p)

        # git updates
        try:
            branch = _run(["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"])
            _run(["git", "-C", repo_dir, "fetch"])
            behind = int(_run([
                "git", "-C", repo_dir,
                f"rev-list", f"HEAD..origin/{branch}", "--count"
            ]))
            if behind > 0:
                pull_out = _run(["git", "-C", repo_dir, "pull", "--ff-only"])
                log_message(f"Git pull succeeded:\n{pull_out}", "SUCCESS")
                notify_admin(f"🔄 *Git update*: `{branch}` +{behind} commits")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except subprocess.CalledProcessError as e:
            err = e.output.strip()
            log_message(f"Git watcher error: {err}", "WARNING")
            notify_admin(f"⚠️ *Git watcher error*:\n```{err[:1500]}```")
        except Exception:
            tb = traceback.format_exc()
            log_message(f"Watcher exception:\n{tb}", "WARNING")
            notify_admin(f"⚠️ *Watcher exception*:\n```{tb[:1500]}```")

        time.sleep(interval)

threading.Thread(
    target=_monitor_git_and_files,
    daemon=True,
    name="GitAndFileWatcher"
).start()

# ─── CLEANUP & WAIT ───────────────────────────────────────────────────────
import atexit
def _cleanup():
    log_message("Shutting down services…", "INFO")
    try: audio_svc.stop()
    except: pass
    try: tts_audio.stop()
    except: pass
    log_message("Goodbye.", "INFO")

atexit.register(_cleanup)
threading.Event().wait()