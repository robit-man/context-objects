#!/usr/bin/env python3
"""
main.py

Bootstraps a virtualenv (once), installs requirements, initializes config,
then enters a REPL that uses Assembler to manage ContextObjects and
invoke gemma3:4b via Ollama—focusing on the context pipeline/demo.
"""

import os
import sys
import subprocess
import json

# -------------------------
# 1) VENV bootstrap + restart
# -------------------------
def ensure_venv():
    """
    If not already in a virtualenv, create .venv/, install requirements,
    and re-exec under the venv's Python interpreter.
    """
    in_venv = (
        hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix
    ) or (
        hasattr(sys, "real_prefix") and sys.real_prefix != sys.prefix
    )
    if in_venv:
        return

    venv_dir = os.path.join(os.getcwd(), ".venv")
    python_bin = os.path.join(venv_dir, "bin", "python")
    pip_bin    = os.path.join(venv_dir, "bin", "pip")

    if not os.path.isdir(venv_dir):
        print("Creating virtualenv in .venv/")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        # write requirements.txt
        with open("requirements.txt", "w") as f:
            f.write("\n".join([
                "ollama",
                "sentence-transformers",
                "numpy",
                "torch"
            ]) + "\n")
        print("Installing requirements…")
        subprocess.check_call([pip_bin, "install", "-r", "requirements.txt"])

    print("Re-launching under virtualenv…")
    new_env = os.environ.copy()
    new_env["VIRTUAL_ENV"] = venv_dir
    new_env["PATH"] = os.path.join(venv_dir, "bin") + os.pathsep + new_env.get("PATH", "")
    os.execve(python_bin, [python_bin] + sys.argv, new_env)

ensure_venv()

# -------------------------
# 2) Load / generate config.json
# -------------------------
CONFIG_FILE = "config.json"
default_cfg = {
    "primary_model": "gemma3:4b",
    "secondary_model": None
}

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_cfg, f, indent=2)
    print(f"Created default {CONFIG_FILE}")

with open(CONFIG_FILE) as f:
    config = json.load(f)

# -------------------------
# 3) Import and initialize Assembler
# -------------------------
from assembler import Assembler

CTX_PATH    = "context.jsonl"
asm = Assembler(
    context_path=CTX_PATH,
    config_path=CONFIG_FILE,
    lookback_minutes=60,
    top_k=5
)

print(f"Using context store: {CTX_PATH}")
print(f"Primary model (from config): {config['primary_model']}")
print("Ready. Type your message, or Ctrl-C to exit.")

# -------------------------
# 4) REPL Loop
# -------------------------
try:
    while True:
        user_text = input(">> ").strip()
        if not user_text:
            continue
        try:
            response = asm.run_with_meta_context(user_text)
            print(response)
        except Exception as e:
            print(f"[ERROR during inference] {e}")
except KeyboardInterrupt:
    print("\nGoodbye.")
    sys.exit(0)
