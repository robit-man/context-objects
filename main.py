#!/usr/bin/env python3
"""
main.py

Bootstraps a virtualenv, installs requirements, initializes config,
pulls Ollama models, and then enters a REPL that uses Assembler
to manage ContextObjects and invoke gemma3:4b via Ollama.
"""

import os
import sys
import subprocess
import json

# -------------------------
# 1) VENV bootstrap + restart
# -------------------------
def ensure_venv():
    if not os.environ.get("VIRTUAL_ENV"):
        print("Creating virtualenv in .venv/")
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"])
        # write requirements
        with open("requirements.txt", "w") as f:
            f.write("\n".join([
                "ollama",
                "sentence-transformers",
                "numpy"
            ]) + "\n")
        print("Installing requirements…")
        pip = os.path.join(".venv", "bin", "pip")
        subprocess.check_call([pip, "install", "-r", "requirements.txt"])
        print("Re-launching under virtualenv…")
        python = os.path.join(".venv", "bin", "python")
        os.execv(python, [python] + sys.argv)

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
config = json.load(open(CONFIG_FILE))

# -------------------------
# 3) Pull Ollama models if needed
# -------------------------
def pull_model(name: str):
    if name:
        print(f"Ensuring Ollama model '{name}' is available…")
        subprocess.check_call(["ollama", "pull", name])
pull_model(config.get("primary_model"))
pull_model(config.get("secondary_model"))

# -------------------------
# 4) Import Assembler and initialize
# -------------------------
from assembler import Assembler

CTX_PATH    = "context.jsonl"
CONFIG_PATH = CONFIG_FILE

asm = Assembler(
    context_path=CTX_PATH,
    config_path=CONFIG_PATH,
    lookback_minutes=60,
    top_k=5
)

print(f"Using context store: {CTX_PATH}")
print(f"Primary model: {config['primary_model']}")
print("Ready. Type your message, or Ctrl-C to exit.")

# -------------------------
# 5) REPL Loop
# -------------------------
try:
    while True:
        user_text = input(">> ").strip()
        if not user_text:
            continue
        try:
            reply = asm.run_with_meta_context(user_text)
            print(reply)
        except Exception as e:
            print(f"[ERROR during inference] {e}")
except KeyboardInterrupt:
    print("\nGoodbye.")
    sys.exit(0)
