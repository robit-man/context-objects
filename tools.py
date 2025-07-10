
import inspect
import json
import logging
import re
from typing import Callable, Optional, Any, Dict, List, Union, Tuple
import sys, os, subprocess, platform, re, json, time, threading, queue, datetime, inspect, difflib, random, copy, statistics, ast, shutil
from datetime import datetime, timezone
from context import ContextRepository, ContextObject, default_clock
from inspect import signature, _empty
import os, json, re, difflib
from datetime import datetime, timedelta

from ollama import chat, embed
import cv2
import mss

import psutil
import traceback

import requests
from telegram import Bot
from telegram.error import TelegramError
from user_registry import _REG
from group_registry import _GREG

from selenium import webdriver
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import html, textwrap                          # (html imported only once)
import os, shutil, random, platform, json
import inspect
import asyncio
from typing import Callable, Optional, Any, Dict, List, Union, Tuple
from queue import Queue, Empty
import uuid
from pathlib import Path
from num2words import num2words
import numpy as np

import networkx as nx
import sqlite3
import pandas as pd


# ----- COLOR CODES FOR LOGGING -----
COLOR_RESET = "\033[0m"
COLOR_INFO = "\033[94m"       # Blue for general info
COLOR_SUCCESS = "\033[92m"    # Green for success messages
COLOR_WARNING = "\033[93m"    # Yellow for warnings
COLOR_ERROR = "\033[91m"      # Red for error messages
COLOR_DEBUG = "\033[95m"      # Magenta for debug messages
COLOR_PROCESS = "\033[96m"    # Cyan for process steps


# Here we set up the Flask app and CORS for cross-origin requests.
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def log_message(message, category="INFO"):
    if category.upper() == "INFO":
         color = COLOR_INFO
    elif category.upper() == "SUCCESS":
         color = COLOR_SUCCESS
    elif category.upper() == "WARNING":
         color = COLOR_WARNING
    elif category.upper() == "ERROR":
         color = COLOR_ERROR
    elif category.upper() == "DEBUG":
         color = COLOR_DEBUG
    elif category.upper() == "PROCESS":
         color = COLOR_PROCESS
    else:
         color = COLOR_RESET
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}[{timestamp}] {category.upper()}: {message}{COLOR_RESET}")



# This function loads the configuration from config.json, creating it with default values if it doesn't exist.
def load_config():
    """
    Load configuration from config.json (in the script directory). If not present, create it with default values.
    New keys include settings for primary/secondary models, temperatures, RMS threshold, debug audio playback,
    noise reduction, consensus threshold, and now image support.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)
    log_message("Configuration loaded from config.json", "INFO")
    return config

config = load_config()

# ────────────────────────────────────────────────────────────────────────────────
# tool‐schema generation & registry
# ────────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {}

def _extract_usage_from_doc(fn: Callable) -> Optional[str]:
    """Look for a fenced ```…``` block inside the docstring and return it."""
    doc = inspect.getdoc(fn) or ""
    m = re.search(r"```([\s\S]*?)```", doc)
    return m.group(1).strip() if m else None

def _create_tool_schema(fn: Callable) -> Dict[str, Any]:
    """
    Build a JSON-schema dict from the function signature and docstring:
     - parameter names, types (string|integer|number|boolean), defaults
     - overall description from the doc
     - optional 'usage' field from any fenced triple-quotes in the doc
    """
    sig = inspect.signature(fn)
    props: Dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        p_schema: Dict[str, Any] = {}
        ann = param.annotation
        # map simple Python annotations → JSON types
        if ann is inspect._empty:
            p_schema["type"] = "string"
        else:
            p_schema["type"] = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
            }.get(ann, "string")
        if param.default is inspect._empty:
            required.append(name)
        else:
            p_schema["default"] = param.default
        props[name] = p_schema

    schema_obj: Dict[str, Any] = {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required,
        }
    }
    # attach any triple-quoted usage example
    usage = _extract_usage_from_doc(fn)
    if usage:
        schema_obj["usage"] = usage

    return schema_obj


# Here in this class, we define various utility functions for text processing, embedding, and other operations. We also include methods for removing emojis, converting numbers to words, and calculating cosine similarity.
class Utils:
    @staticmethod
    def remove_emojis(text):
        emoji_pattern = re.compile(
            "[" 
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        result = emoji_pattern.sub(r'', text)
        log_message("Emojis removed from text.", "DEBUG")
        return result
    
    @staticmethod
    def convert_numbers_to_words(text):
        def replace_num(match):
            number_str = match.group(0)
            try:
                return num2words(int(number_str))
            except ValueError:
                return number_str
        converted = re.sub(r'\b\d+\b', replace_num, text)
        log_message("Numbers converted to words in text.", "DEBUG")
        return converted
    
    @staticmethod
    def get_current_time():
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message("Current time retrieved.", "DEBUG")
        return current_time
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            log_message("One of the vectors has zero norm in cosine similarity calculation.", "WARNING")
            return 0.0
        similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        log_message("Cosine similarity computed.", "DEBUG")
        return similarity
    
    @staticmethod
    def safe_load_json_file(path, default):
        if not path:
            return default
        if not os.path.exists(path):
            if default == []:
                with open(path, 'w') as f:
                    json.dump([], f)
            return default
        try:
            with open(path, 'r') as f:
                result = json.load(f)
            log_message(f"JSON file loaded from {path}.", "DEBUG")
            return result
        except Exception:
            log_message(f"Failed to load JSON file from {path}, returning default.", "ERROR")
            return default
        
    def _sanitize_tool_call(code: str) -> str:
        """
        Parse the code string as an AST.Call, re-serialize string
        literals with proper escapes (\n, \') so the final Python is valid.
        """
        try:
            tree = ast.parse(code, mode="eval")
            call = tree.body
            if not isinstance(call, ast.Call):
                return code
            func = call.func.id
            parts = []
            # positional args
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    parts.append(repr(arg.value))
                else:
                    parts.append(ast.unparse(arg))
            # keyword args
            for kw in call.keywords:
                val = kw.value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    v = repr(val.value)
                else:
                    v = ast.unparse(val)
                parts.append(f"{kw.arg}={v}")
            return f"{func}({', '.join(parts)})"
        except Exception:
            # if anything goes wrong, just return the original
            return code

    @staticmethod
    def load_format_schema(fmt):
        if not fmt:
            return None
        if fmt.lower() == "json":
            log_message("JSON format schema detected.", "DEBUG")
            return "json"
        if os.path.exists(fmt):
            try:
                with open(fmt, 'r') as f:
                    result = json.load(f)
                log_message("Format schema loaded from file.", "DEBUG")
                return result
            except Exception:
                log_message("Error loading format schema from file.", "ERROR")
                return None
        log_message("No valid format schema found.", "WARNING")
        return None
    
    @staticmethod
    def monitor_script(interval=5):
        script_path = os.path.abspath(__file__)
        last_mtime = os.path.getmtime(script_path)
        log_message("Monitoring script for changes...", "PROCESS")
        while True:
            time.sleep(interval)
            try:
                new_mtime = os.path.getmtime(script_path)
                if new_mtime != last_mtime:
                    log_message("Script change detected. Restarting...", "INFO")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception:
                pass

    @staticmethod
    def embed_text(text):
        """
        Embed into a 1-D numpy array of shape (768,).
        """
        try:
            #log_message("Embedding text for context.", "PROCESS")
            response = embed(model="nomic-embed-text", input=text)
            vec = np.array(response["embeddings"], dtype=float)
            # ensure 1-D
            vec = vec.flatten()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            #log_message("Text embedding computed and normalized.", "SUCCESS")
            return vec
        except Exception as e:
            log_message("Error during text embedding: " + str(e), "ERROR")
            return np.zeros(768, dtype=float)

    @staticmethod
    def speak(text: str):
        """
        Legacy wrapper.  Forwards *text* to the global TTS queue if
        present.  Silently ignores every error.
        """
        try:
            from __main__ import tts_queue
            tts_queue.put(str(text))
        except Exception:
            pass
        
# This class delivers various utility functions for managing tools, such as parsing tool calls, adding subtasks, and listing subtasks for use by an LLM instance or other components.
class Tools:
    _driver = None          # always present, even before first browser launch
    _poll   = 0.05                     
    _short  = 3
    

    # Here we set up the Flask app and CORS for cross-origin requests.
    WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    @staticmethod
    def parse_tool_call(text: str) -> Optional[str]:
        """
        Extract a tool call from:
          1) A fenced ```tool_code```, ```python``` or ```py``` block
          2) An inline-backtick call: `func(arg1, arg2)`
          3) A bare call on its own line: func(arg1, arg2)

        Strips out simple Python‐style type annotations (e.g. filename: str="x")
        and normalizes key: value → key=value for basic literals.
        Returns the raw "func(...)" string, or None if nothing valid is found.
        """

        t = text.strip()

        # 1) Pre-clean: remove type annotations before '='
        #    e.g. 'filename: str = "F.gig"' → 'filename = "F.gig"'
        t = re.sub(
            r'([A-Za-z_]\w*)\s*:\s*[A-Za-z_]\w*(?=\s*=)',
            r'\1',
            t
        )

        # 2) Normalize simple key: value → key=value
        t = re.sub(
            r'(\b[A-Za-z_]\w*\b)\s*:\s*'         # key:
            r'("(?:[^"\\]|\\.)*"|'              #   "quoted"
            r'\'(?:[^\'\\]|\\.)*\'|'            #   'quoted'
            r'\d+|None)',                       #   integer or None
            r'\1=\2',
            t
        )

        candidate: str | None = None

        # 3) Attempt fenced ```tool_code``` / ```python``` / ```py``` block
        m = re.search(
            r"```(?:tool_code|python|py)\s*([\s\S]+?)```",
            t,
            flags=re.DOTALL
        )
        if m:
            candidate = m.group(1).strip()
        else:
            # 4) Inline single-backticks
            if t.startswith("`") and t.endswith("`"):
                inner = t[1:-1].strip()
                if "(" in inner and inner.endswith(")"):
                    candidate = inner

        # 5) Bare call on its own line if still nothing
        if not candidate:
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            if lines:
                first = lines[0]
                if "(" in first and first.endswith(")"):
                    candidate = first

        if not candidate:
            logging.debug("Parsed tool call from text: None")
            return None

        # 6) AST-validate: ensure it’s exactly a Name(...) call
        try:
            expr = ast.parse(candidate, mode="eval")
            call = expr.body
            if (
                isinstance(expr, ast.Expression)
                and isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
            ):
                logging.debug(f"Parsed tool call from text: {candidate!r}")
                return candidate
        except Exception:
            pass

        logging.debug("Parsed tool call from text: None (AST check failed)")
        return None
    
    @staticmethod
    def context_query(
        assembler,  # Instance of Assembler with .repo, .engine, .memman
        time_range: list[str] | None = None,
        tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        domain: list[str] | None = None,
        component: list[str] | None = None,
        semantic_label: list[str] | None = None,
        summary_regex: str | None = None,
        similarity_to: str | None = None,
        query: str | None = None,
        top_k: int = 5,
        wm_only: bool = False,
        wm_type: str | None = None,
        window: str | None = None,
        max_entries: int | None = None,
    ) -> str:
        """
        Query the assembler's context store with flexible filters; never falls back on context.jsonl.
        Prints verbose logs at each major step.
        Returns JSON-formatted string with 'results'.
        """
        print("=== context_query called ===")
        print(f"Args: time_range={time_range}, tags={tags}, exclude_tags={exclude_tags}, "
            f"domain={domain}, component={component}, semantic_label={semantic_label}, "
            f"summary_regex={summary_regex}, similarity_to={similarity_to or query}, "
            f"top_k={top_k}, wm_only={wm_only}, wm_type={wm_type}, window={window}, max_entries={max_entries}")
        if query and not similarity_to:
            similarity_to = query
        ctxs = list(assembler.repo.query(lambda c: True))
        print(f"Loaded {len(ctxs)} context objects from assembler.repo")

        # WM shortcuts
        if wm_only:
            before = len(ctxs)
            ctxs = [c for c in ctxs if (c.domain=="segment" and c.semantic_label in ("user_input","assistant")) or c.semantic_label=="final_inference"]
            print(f"{len(ctxs)} of {before} after wm_only filter")
        if wm_type == "segments":
            before = len(ctxs)
            ctxs = [c for c in ctxs if c.domain=="segment" and c.semantic_label in ("user_input","assistant")]
            print(f"{len(ctxs)} of {before} after wm_type='segments'")
        if wm_type == "infers":
            before = len(ctxs)
            ctxs = [c for c in ctxs if c.semantic_label=="final_inference"]
            print(f"{len(ctxs)} of {before} after wm_type='infers'")

        # parse window
        def _parse(ts: str) -> datetime:
            try: return datetime.fromisoformat(ts)
            except: return datetime.strptime(ts, "%Y%m%dT%H%M%SZ")

        if window:
            m = re.match(r"(\d+)\s*(\w+)", window)
            if m:
                val, unit = int(m.group(1)), m.group(2).lower().rstrip("s")
                delta = {"minute": timedelta(minutes=val), "hour": timedelta(hours=val), "day": timedelta(days=val)}.get(unit, timedelta())
                cutoff = datetime.utcnow() - delta
                before = len(ctxs)
                ctxs = [c for c in ctxs if _parse(c.timestamp) >= cutoff]
                print(f"{len(ctxs)} of {before} after window filter '{window}'")

        # tag/domain/component filters
        if tags:
            before = len(ctxs)
            ctxs = [c for c in ctxs if set(tags) & set(c.tags)]
            print(f"{len(ctxs)} of {before} after tags filter")
        if exclude_tags:
            before = len(ctxs)
            ctxs = [c for c in ctxs if not (set(exclude_tags) & set(c.tags))]
            print(f"{len(ctxs)} of {before} after exclude_tags filter")
        if domain:
            before = len(ctxs)
            ctxs = [c for c in ctxs if c.domain in domain]
            print(f"{len(ctxs)} of {before} after domain filter")
        if component:
            before = len(ctxs)
            ctxs = [c for c in ctxs if c.component in component]
            print(f"{len(ctxs)} of {before} after component filter")
        if semantic_label:
            before = len(ctxs)
            ctxs = [c for c in ctxs if c.semantic_label in semantic_label]
            print(f"{len(ctxs)} of {before} after semantic_label filter")
        if summary_regex:
            pat = re.compile(summary_regex, re.IGNORECASE)
            before = len(ctxs)
            ctxs = [c for c in ctxs if c.summary and pat.search(c.summary)]
            print(f"{len(ctxs)} of {before} after summary_regex filter")

        # similarity or recency ranking
        selected = []
        if similarity_to:
            print(f"Performing similarity ranking to '{similarity_to}'")
            try:
                results = assembler.engine.query(
                    similarity_to=similarity_to,
                    include_tags=tags,
                    exclude_tags=exclude_tags,
                    time_range=time_range,
                    top_k=top_k
                )
                selected = [assembler.repo.get(r.context_id) for r in results]
                print(f"Selected {len(selected)} via vector engine")
            except NameError:
                print("Vector engine unavailable, falling back to fuzzy matching")
                scored = [(re.compile(re.escape(similarity_to), re.IGNORECASE).search((c.summary or "")) and 1.0 or 0.0, c) for c in ctxs]
                scored.sort(key=lambda x: x[0], reverse=True)
                selected = [c for _, c in scored[:top_k]]
                print(f"Selected {len(selected)} via fallback fuzzy match")
        else:
            ctxs.sort(key=lambda c: _parse(c.timestamp), reverse=True)
            selected = ctxs[:top_k]
            print(f"Selected top {len(selected)} by recency")

        # cap by max_entries
        if max_entries is not None:
            before = len(selected)
            selected = selected[-max_entries:]
            print(f"Capped to {len(selected)} entries by max_entries={max_entries}")

        # register recalls
        for c in selected:
            c.record_recall(stage_id="context_query", coactivated_with=[])
            assembler.repo.save(c)
            assembler.memman.register_relationships(c, assembler.engine.embedder)

        # format output
        out = [{"context_id": c.context_id, "timestamp": c.timestamp,
                "domain": c.domain, "component": c.component,
                "semantic_label": c.semantic_label, "summary": c.summary or ""}
            for c in selected]
        json_out = json.dumps({"results": out}, indent=2)
        print("Returning context_query results")
        return json_out


    # Here in this definition, we define a static method to add a subtask under an existing task. If the parent task does not exist or has an ID less than or equal to zero, the subtask will be created as a top-level task.
    @staticmethod
    def add_subtask(parent_id: int, text: str) -> dict:
        """
        Create a new subtask under an existing task.
        If the parent_id does not exist (or is <= 0), the subtask will be top-level (parent=None).
        Returns the created subtask object.
        """
        import os, json

        path = os.path.join(WORKSPACE_DIR, "tasks.json")

        # 1) Safely load existing tasks (empty or invalid → [])
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tasks = json.load(f)
                if not isinstance(tasks, list):
                    tasks = []
            except (json.JSONDecodeError, ValueError):
                tasks = []
        else:
            tasks = []

        # 2) Determine whether the parent exists
        parent = None
        if isinstance(parent_id, int) and parent_id > 0:
            parent = next((t for t in tasks if t.get("id") == parent_id), None)

        # 3) Assign new ID
        new_id = max((t.get("id", 0) for t in tasks), default=0) + 1

        # 4) Build the subtask record
        sub = {
            "id":     new_id,
            "text":   text,
            "status": "pending",
            # if parent was found use its id, otherwise None (top-level)
            "parent": parent.get("id") if parent else None
        }

        # 5) Append and save
        tasks.append(sub)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=2)
        except Exception as e:
            # if saving fails, return an error dict
            return {"error": f"Failed to save subtask: {e}"}

        return sub

    # Here in this definition, we define a static method to list all subtasks for a given parent task ID. It returns a list of subtasks that have the specified parent ID. We also ensure that the tasks are loaded from a JSON file, and if the file does not exist, an empty list is returned.
    @staticmethod
    def list_subtasks(parent_id: int) -> list:
        """
        Return all subtasks for the given parent task.
        """
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        tasks = json.loads(open(path).read()) if os.path.exists(path) else []
        return [t for t in tasks if t.get("parent") == parent_id]

    # Here in this definition, we define a static method to set the status of a task or subtask. It updates the status to 'pending', 'in_progress', or 'done' and saves the updated task list back to the JSON file.
    @staticmethod
    def set_task_status(task_id: int, status: str) -> dict:
        """
        Set status = 'pending'|'in_progress'|'done' on a task or subtask.
        """
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        if not os.path.exists(path):
            return {"error": "No tasks yet."}
        tasks = json.loads(open(path).read())
        # find the task
        t = next((t for t in tasks if t.get("id") == task_id), None)
        if not t:
            return {"error": f"No such task {task_id}"}
        if status not in ("pending", "in_progress", "done"):
            return {"error": f"Invalid status {status}"}
        t["status"] = status
        # save list back
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        return t

    # This following static method loads the tasks dictionary from a JSON file. If the file does not exist, it returns an empty dictionary.
    @staticmethod
    def _load_tasks_dict() -> dict:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        if os.path.isfile(path):
            return json.load(open(path, "r"))
        return {}

    # Here we save the tasks dictionary to a JSON file. This method is used to persist the tasks after they have been modified.
    @staticmethod
    def _save_tasks_dict(tasks: dict) -> None:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        with open(path, "w") as f:
            json.dump(tasks, f, indent=2)

    _process_registry: dict[int, subprocess.Popen] = {}
    
    # This is a special function that takes in arbitrary data types (variables, strings, etc.) and returns a single string.
    @staticmethod
    def assemble_mixed_data(data):
        """
        Assembles a mixed set of variables, strings, and other data types into a single string.

        Args:
            data: An arbitrary collection (list, tuple, set, etc.) containing variables,
                strings, and other data types.  The order of elements in the input
                collection determines the order in the output string.

        Returns:
            A single string formed by converting each element in the input collection
            to a string and concatenating them in the original order.
        """

        result = ""
        for item in data:
            result += str(item)  # Convert each item to a string and append

        return result

    # In this method we define a static method to add a new task. It generates a new task ID, appends the task to the tasks list, and saves it to the JSON file. The method returns a confirmation message with the new task ID.
    @staticmethod
    def add_task(text: str) -> str:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        tasks = json.loads(open(path).read()) if os.path.exists(path) else []
        new_id = max((t["id"] for t in tasks), default=0) + 1
        tasks.append({"id": new_id, "text": text})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        return f"Task {new_id} added."

    # Here we update a task by its ID. It searches for the task in the tasks list, updates its text, and saves the updated list back to the JSON file. If the task is not found, it returns an error message.
    @staticmethod
    def update_task(task_id: int, text: str) -> str:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        if not os.path.exists(path):
            return "No tasks yet."
        tasks = json.loads(open(path).read())
        for t in tasks:
            if t["id"] == task_id:
                t["text"] = text
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(tasks, f, indent=2)
                return f"Task {task_id} updated."
        return f"No task with id={task_id}."

    # In this method we define a static method to remove a task by its ID. It filters out the task with the specified ID from the tasks list and saves the updated list back to the JSON file. If the task is not found, it returns an error message.
    @staticmethod
    def remove_task(task_id: int) -> str:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        if not os.path.exists(path):
            return "No tasks yet."
        tasks = json.loads(open(path).read())
        new = [t for t in tasks if t["id"] != task_id]
        if len(new) == len(tasks):
            return f"No task with id={task_id}."
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new, f, indent=2)
        return f"Task {task_id} removed."

    # Here we can define a static method to list all tasks. It reads the tasks from the JSON file and returns them as a JSON string. If the file does not exist, it returns an empty list.
    @staticmethod
    def list_tasks() -> str:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        tasks = json.loads(open(path).read()) if os.path.exists(path) else []
        return json.dumps(tasks)
    
    # This static method runs an arbitrary Python code snippet in a fresh subprocess, capturing its output and return code. It handles timeouts and errors gracefully, returning a structured dictionary with the results.
    @staticmethod
    def run_python_snippet(
        code: str,
        *,
        stdin: str = "",
        timeout: int = 10,
        dedent: bool = True,
    ) -> dict:
        """
        Execute an arbitrary Python snippet in a fresh subprocess.

        Parameters
        ----------
        code : str
            The snippet to run.  It may already be wrapped in
            ```python``` / ```py``` / ```tool_code``` fences – these are stripped
            automatically so callers don’t have to worry.
        stdin : str
            Text piped to the child process’ STDIN.
        timeout : int
            Hard wall-clock limit (seconds).
        dedent : bool
            If True, run `textwrap.dedent()` on the snippet after stripping fences
            – makes copy-pasted indented code work.

        Returns
        -------
        dict
            {
              "stdout":     <captured STDOUT str>,
              "stderr":     <captured STDERR str>,
              "returncode": <int>,
            }
            On failure an `"error"` key is present instead.
        """
        import re, subprocess, sys, tempfile, textwrap, os

        # 1) strip any ```python``` / ```py``` / ```tool_code``` fences
        fence_rx = re.compile(
            r"```(?:python|py|tool_code)?\s*([\s\S]*?)\s*```",
            re.IGNORECASE
        )
        m = fence_rx.search(code)
        if m:
            code = m.group(1)

        # 2) optional dedent
        if dedent:
            code = textwrap.dedent(code)

        # 3) write to temp .py
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        # 4) run it
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode
            }

        except subprocess.TimeoutExpired:
            return {"error": f"Timed out after {timeout}s"}

        except Exception as e:
            return {"error": str(e)}

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def run_tool_once(tool_call: str) -> dict:
        """
        Parse `tool_call` (e.g. "foo(1, bar='x')" or "Tools.foo(1)"), execute
        the corresponding Tools.foo function, and return a dict with keys:
          - "output": the function’s return value (or its repr if unserializable)
          - "exception": None on success or a clear error/traceback string on failure
        """
        import ast
        import inspect
        import traceback
        import json

        # 1) Parse into AST
        try:
            expr = ast.parse(tool_call.strip(), mode="eval")
        except SyntaxError as e:
            return {"output": None, "exception": f"SyntaxError: {e}"}

        # Must be a function call
        call = expr.body
        if not isinstance(call, ast.Call):
            return {"output": None, "exception": "Invalid call: not a function invocation"}

        # 2) Resolve function name
        #    Accept either foo(...) or Tools.foo(...)
        fn_name = None
        if isinstance(call.func, ast.Name):
            fn_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # e.g. Tools.foo
            if (isinstance(call.func.value, ast.Name)
                    and call.func.value.id == Tools.__name__):
                fn_name = call.func.attr
        if not fn_name:
            return {"output": None, "exception": "Unsupported function reference"}

        # 3) Lookup the function
        fn = getattr(Tools, fn_name, None)
        if not callable(fn):
            return {"output": None, "exception": f"Unknown tool: {fn_name}"}

        # 4) Safely evaluate each arg via literal_eval
        def _safe_eval(node):
            try:
                return ast.literal_eval(node)
            except Exception as e:
                raise ValueError(f"Could not parse argument: {e}")

        args = []
        for i, a in enumerate(call.args):
            try:
                args.append(_safe_eval(a))
            except ValueError as e:
                return {"output": None, "exception": f"Positional arg #{i}: {e}"}

        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                return {"output": None, "exception": "Kwargs unpacking (`**`) not supported"}
            try:
                kwargs[kw.arg] = _safe_eval(kw.value)
            except ValueError as e:
                return {"output": None, "exception": f"Keyword arg '{kw.arg}': {e}"}

        # 5) Bind to signature to catch missing/extra args
        sig = inspect.signature(fn)
        try:
            sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            return {"output": None, "exception": f"Signature error: {e}"}

        # 6) Call the function, catching any exception
        try:
            raw = fn(*args, **kwargs)
        except Exception:
            tb = traceback.format_exc()
            return {"output": None, "exception": tb}

        # 7) Ensure JSON‐serializable
        try:
            json.dumps(raw)
            safe_out = raw
        except Exception:
            safe_out = repr(raw)

        return {"output": safe_out, "exception": None}


    # This static method runs a Python script, either in a new terminal window or synchronously capturing its output. It returns a dictionary with the process ID or captured output.
    @staticmethod
    def run_script(
        script_path: str,
        args: str = "",
        base_dir: str = WORKSPACE_DIR,
        capture_output: bool = False,
        window_title: str | None = None
    ) -> dict:
        """
        Launch or run a Python script.

        • If capture_output=False, opens in a new terminal (per OS) and registers PID.
        • If capture_output=True, runs synchronously, captures stdout/stderr/rc.
        Returns a dict:
          - on new terminal: {"pid": <pid>} or {"error": "..."}
          - on capture: {"stdout": "...", "stderr": "...", "returncode": <int>} or {"error": "..."}
        """
        import os, sys, platform, subprocess, shlex

        full_path = script_path if os.path.isabs(script_path) \
                    else os.path.join(base_dir, script_path)
        if not os.path.isfile(full_path):
            return {"error": f"Script not found at {full_path}"}

        cmd = [sys.executable, full_path] + (shlex.split(args) if args else [])
        system = platform.system()

        try:
            if capture_output:
                proc = subprocess.run(
                    cmd, cwd=base_dir, text=True, capture_output=True
                )
                return {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode
                }

            # non-capture → new terminal window
            if system == "Windows":
                flags = subprocess.CREATE_NEW_CONSOLE
                if window_title:
                    title_cmd = ["cmd", "/c", f"title {window_title} &&"] + cmd
                    p = subprocess.Popen(title_cmd, creationflags=flags)
                else:
                    p = subprocess.Popen(cmd, creationflags=flags)

            elif system == "Darwin":
                osa = f'tell application "Terminal" to do script "{ " ".join(cmd) }"'
                p = subprocess.Popen(["osascript", "-e", osa])

            else:
                term = os.getenv("TERMINAL", "xterm")
                p = subprocess.Popen([term, "-hold", "-e"] + cmd, cwd=base_dir)

            pid = p.pid
            Tools._process_registry[pid] = p
            return {"pid": pid}

        except Exception as e:
            return {"error": str(e)}

    # This static method stops a previously launched script by its process ID (PID). It attempts to terminate the process and waits for it to finish, removing it from the registry if successful.
    @staticmethod
    def stop_script(pid: int) -> dict:
        """
        Terminate a previously launched script by its PID.
        Returns {"stopped": pid} or {"error": "..."}.
        """
        proc = Tools._process_registry.get(pid)
        if not proc:
            return {"error": f"No managed process with PID {pid}"}
        try:
            proc.terminate()
            proc.wait(timeout=5)
            del Tools._process_registry[pid]
            return {"stopped": pid}
        except Exception as e:
            return {"error": str(e)}

    # This static method checks the status of a managed script by its process ID (PID). It returns a dictionary indicating whether the script is running, has exited with a code, or if there was an error.
    @staticmethod
    def script_status(pid: int) -> dict:
        """
        Check status of a managed PID.
        Returns {"running": pid} if alive, or {"exit_code": <int>} if done,
        or {"error": "..."} if unknown.
        """
        proc = Tools._process_registry.get(pid)
        if not proc:
            return {"error": f"No managed process with PID {pid}"}
        rc = proc.poll()
        if rc is None:
            return {"running": pid}
        # finished
        del Tools._process_registry[pid]
        return {"exit_code": rc}

    # This static method provides a simple dispatcher for exploring tools. It allows agents to list available tools or get the source code of a specific tool.
    @staticmethod
    def explore_tools(action: str = "list", tool: str | None = None) -> str:
        """
        Tiny dispatcher so agents can:
            • explore_tools("list")          –> same as list_tools(detail=True)
            • explore_tools("source","foo")  –> same as get_tool_source("foo")
        """
        if action == "list":
            return Tools.list_tools(detail=True)
        if action == "source" and tool:
            return Tools.get_tool_source(tool)
        return "Usage: explore_tools('list')  or  explore_tools('source','tool_name')"

    # This static method lists all callable tools currently available in the Tools class. It can return either a simple list of tool names or detailed metadata about each tool, depending on the `detail` parameter.
    @staticmethod
    def list_tools(detail: bool = False) -> str:
        """
        Return JSON metadata for every callable tool currently on Tools.

        detail=False → ["name", ...]  
        detail=True  → [{"name","signature","doc"}, ...]
        """
        import inspect, json

        tools: list = []
        for name, fn in inspect.getmembers(Tools, predicate=callable):
            if name.startswith("_"):
                continue

            if detail:
                sig = str(inspect.signature(fn))
                # grab the full docstring, or empty string if none
                doc = inspect.getdoc(fn) or ""
                tools.append({
                    "name":      name,
                    "signature": sig,
                    "doc":       doc
                })
            else:
                tools.append(name)

        return json.dumps(tools, indent=2)

    # This static method retrieves the source code of a specified tool function by its name. It uses the `inspect` module to get the source code and handles errors gracefully if the tool is not found or if there are issues retrieving the source.
    @staticmethod
    def get_tool_source(tool_name: str) -> str:
        """
        Return the *source code* for `tool_name`, or an error string.
        """
        import inspect

        fn = getattr(Tools, tool_name, None)
        if not fn or not callable(fn):
            return f"Error: tool '{tool_name}' not found."
        try:
            return inspect.getsource(fn)
        except Exception as e:                     # pragma: no-cover
            return f"Error retrieving source: {e}"

    # This static method creates a new external tool by writing a Python function to a file in the `external_tools` directory. It handles overwriting existing files, auto-reloading the module, and provides error messages for common issues.
    @staticmethod
    def create_tool(
        tool_name: str,
        code: str | None = None,
        *,
        description: str | None = None,           # ← tolerated, but ignored
        tool_call: str | None = None,             # ← dito (compat shim)
        overwrite: bool = False,
        auto_reload: bool = True,
    ) -> str:
        """
        Persist a new *external tool* and optionally hot-reload it.

        Parameters
        ----------
        tool_name    name of the Python file **and** the function inside it
        code         full `def tool_name(...):` **OR** None when you just want
                     to reserve the name (rare – normally provide real code)
        description  ignored → kept for backward compatibility with agents
        tool_call    ignored → ditto
        overwrite    allow clobbering an existing file
        auto_reload  immediately import the new module and attach the function
        """
        import os, re, textwrap

        ext_dir = os.path.join(os.path.dirname(__file__), "external_tools")
        os.makedirs(ext_dir, exist_ok=True)
        file_path = os.path.join(ext_dir, f"{tool_name}.py")

        # ── guard rails ────────────────────────────────────────────────────
        if os.path.exists(file_path) and not overwrite:
            return f"Error: {file_path} already exists (use overwrite=True)."

        if code is None:
            return ("Error: `code` is required.  Pass the full function body "
                    "as a string under the `code=` parameter.")

        if not re.match(rf"^\s*def\s+{re.escape(tool_name)}\s*\(", code):
            return (f"Error: `code` must start with `def {tool_name}(` so that "
                    "the module exposes exactly one top-level function.")

        # ── write the module ───────────────────────────────────────────────
        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                header = textwrap.dedent(f'''\
                    """
                    Auto-generated external tool  –  {tool_name}
                    Created via Tools.create_tool()
                    {('Description: ' + description) if description else ''}
                    """
                    ''')
                fh.write(header.rstrip() + "\n\n" + code.strip() + "\n")
        except Exception as e:
            return f"Error writing file: {e}"

        log_message(f"Created external tool {tool_name} at {file_path}", "SUCCESS")

        if auto_reload:
            Tools.reload_external_tools()

        return f"Tool '{tool_name}' created ✔"

    # This static method lists all external tools currently present in the `external_tools` directory. It returns a sorted list of filenames (without the `.py` extension) that are valid Python files.
    @staticmethod
    def list_external_tools() -> list[str]:
        """
        List *.py files currently present in *external_tools/*.
        """
        import os
        ext_dir = os.path.join(os.path.dirname(__file__), "external_tools")
        if not os.path.isdir(ext_dir):
            return []
        return sorted(f for f in os.listdir(ext_dir) if f.endswith(".py"))

    # This static method removes an external tool by deleting its Python file and detaching it from the Tools class. It handles
    @staticmethod
    def remove_external_tool(tool_name: str) -> str:
        """
        Delete *external_tools/<tool_name>.py* and detach it from Tools.
        """
        import os, sys, importlib

        ext_dir = os.path.join(os.path.dirname(__file__), "external_tools")
        path = os.path.join(ext_dir, f"{tool_name}.py")

        try:
            if os.path.isfile(path):
                os.remove(path)
                # also nuke any stale .pyc
                pyc = path + "c"
                if os.path.isfile(pyc):
                    os.remove(pyc)
            if hasattr(Tools, tool_name):
                delattr(Tools, tool_name)
            sys.modules.pop(f"external_tools.{tool_name}", None)
            log_message(f"External tool {tool_name} removed.", "INFO")
            return f"Tool '{tool_name}' removed."
        except Exception as e:                     # pragma: no-cover
            return f"Error removing tool: {e}"

    @staticmethod
    def interact_with_knowledge_graph(
        graph: nx.DiGraph,
        node_type: str,
        node_id: str,
        relation: Optional[str] = None,
        target_node_id: Optional[str] = None,
        attributes_to_retrieve: Optional[List[str]] = None,
        new_attribute: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interacts with a NetworkX knowledge graph.  This method allows an agent to:
        1. Retrieve information about a node.
        2. Find nodes related to a given node.
        3. Add attributes to a node.

        Args:
            graph (nx.DiGraph): The NetworkX knowledge graph.
            node_type (str): The type of node to interact with (e.g., "agent", "task").
            node_id (str): The ID of the node to interact with.
            relation (Optional[str]): The type of relationship to search for (e.g., "assigned_to").
                                     If provided, `target_node_id` must also be provided.
            target_node_id (Optional[str]): The ID of the target node for a relationship search.
            attributes_to_retrieve (Optional[List[str]]): A list of attributes to retrieve from the node.
                                                          If None, all attributes are retrieved.
            new_attribute (Optional[Dict[str, Any]]): A dictionary containing a new attribute to add to the node.
                                                      e.g., {"priority": "high"}

        Returns:
            Dict[str, Any]: A dictionary containing the results of the interaction.
                            The dictionary will have a "status" key ("success" or "error") and a "data" key
                            containing the retrieved data or an error message.

        Raises:
            TypeError: If the graph is not a NetworkX DiGraph.
            ValueError: If invalid arguments are provided.
        """

        if not isinstance(graph, nx.DiGraph):
            raise TypeError("Graph must be a NetworkX DiGraph.")

        try:
            if not graph.has_node(node_id):
                return {"status": "error", "data": f"Node with ID '{node_id}' not found."}

            node_data = graph.nodes[node_id]

            if relation and target_node_id:
                if not graph.has_edge(node_id, target_node_id):
                    return {"status": "error", "data": f"No edge with relation '{relation}' between '{node_id}' and '{target_node_id}'."}
                edge_data = graph.edges[node_id, target_node_id]
                result = edge_data
            elif new_attribute:
                graph.nodes[node_id].update(new_attribute)  # Update node attributes
                result = graph.nodes[node_id]
            else:
                if attributes_to_retrieve:
                    result = {attr: node_data.get(attr) for attr in attributes_to_retrieve if attr in node_data}
                else:
                    result = dict(node_data)  # Return all attributes

            return {"status": "success", "data": result}

        except Exception as e:
            logging.error(f"Error interacting with graph: {e}")
            return {"status": "error", "data": f"An unexpected error occurred: {e}"}

    @staticmethod
    def create_knowledge_graph(
        node_data: List[Dict[str, Any]],
        edge_data: List[Tuple[str, str, Dict[str, Any]]],
        validate_nodes: bool = True  # Added validation flag
    ) -> nx.DiGraph:
        """
        Creates a NetworkX knowledge graph from node and edge data.

        Args:
            node_data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a node.
            edge_data (List[Tuple[str, str, Dict[str, Any]]]): A list of tuples, where each tuple represents an edge
                                                                (source_node_id, target_node_id, edge_attributes).
            validate_nodes (bool): Whether to validate that source and target nodes exist before adding edges.

        Returns:
            nx.DiGraph: The created NetworkX knowledge graph.

        Raises:
            TypeError: If input data is not in the correct format.
            ValueError: If node or edge data is invalid.
        """

        if not isinstance(node_data, list) or not all(isinstance(node, dict) for node in node_data):
            raise TypeError("node_data must be a list of dictionaries.")
        if not isinstance(edge_data, list) or not all(isinstance(edge, tuple) and len(edge) == 3 for edge in edge_data):
            raise TypeError("edge_data must be a list of tuples (source, target, attributes).")

        graph = nx.DiGraph()

        # Add nodes in bulk
        graph.add_nodes_from(node_data)

        # Add edges in bulk, with validation
        for source, target, attributes in edge_data:
            if validate_nodes and not (graph.has_node(source) and graph.has_node(target)):
                logging.warning(f"Skipping edge ({source}, {target}) because source or target node does not exist.")
                continue  # Skip this edge
            graph.add_edge(source, target, **attributes)

        return graph

    @staticmethod
    def create_sqlite_db(db_name: str = "database.db") -> sqlite3.Connection:
        """
        Creates an SQLite database and returns a connection object.

        Args:
            db_name (str): The name of the database file.

        Returns:
            sqlite3.Connection: A connection object to the database.
        """
        try:
            conn = sqlite3.connect(db_name)
            print(f"Database '{db_name}' created successfully.")
            return conn
        except sqlite3.Error as e:
            print(f"Error creating database: {e}")
            return None

    @staticmethod
    def interact_with_sqlite_db(
        conn: sqlite3.Connection,
        query: str,
        params: Tuple[Any, ...] = None
    ) -> pd.DataFrame:
        """
        Executes a query against an SQLite database and returns the results as a Pandas DataFrame.

        Args:
            conn (sqlite3.Connection): The connection object to the database.
            query (str): The SQL query to execute.
            params (Tuple[Any, ...]): Parameters to pass to the query (optional).

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the query results.  Returns an empty DataFrame on error.
        """
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except pd.Error as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    @staticmethod
    def get_table_info(conn: sqlite3.Connection) -> List[Dict[str, str]]:
        """
        Retrieves information about the tables in an SQLite database, including table names and column names.

        Args:
            conn (sqlite3.Connection): The connection object to the database.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, where each dictionary represents a table and its columns.
                                  Each dictionary has the following keys: "table_name" and "columns".
        """
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in cursor.fetchall()]

        table_info = []
        for table_name in table_names:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [row[1] for row in cursor.fetchall()]  # Column names
            table_info.append({"table_name": table_name, "columns": columns})

        return table_info
    
    # This static method reloads all external tools by detaching any previously loaded tools, purging their modules from `sys.modules`, and then re-importing everything found in the `external_tools` directory. It returns a confirmation message.
    @staticmethod
    def reload_external_tools() -> str:
        """
        Detach any previously-loaded external tools, purge their modules, then
        re-import everything found in *external_tools/*.
        """
        import sys, inspect

        # 1️⃣  Detach current externals from Tools
        for name, fn in list(inspect.getmembers(Tools, predicate=callable)):
            if getattr(fn, "__module__", "").startswith("external_tools."):
                delattr(Tools, name)

        # 2️⃣  Purge from sys.modules so the next import is fresh
        for mod in list(sys.modules):
            if mod.startswith("external_tools."):
                sys.modules.pop(mod, None)

        # 3️⃣  Re-import
        Tools.load_external_tools()
        return "External tools reloaded."

    # This static method provides a quick-and-dirty unit-test harness for testing tools. It accepts a tool name and a list of test cases, executing each case and returning a summary of passed and failed tests.
    @staticmethod
    def test_tool(tool_name: str, test_cases: list[dict]) -> dict:
        """
        Quick-and-dirty unit-test harness.

        Each *test_case* dict may contain:  
          • args   : list – positional args  
          • kwargs : dict – keyword args  
          • expect : any  – expected value (optional)  
          • compare: "eq" | "contains" | "custom" (default "eq")  
          • custom : a **lambda** as string, evaluated only when compare="custom"
        """
        import traceback

        fn = getattr(Tools, tool_name, None)
        if not fn or not callable(fn):
            return {"error": f"Tool '{tool_name}' not found."}

        passed = failed = 0
        results = []

        for idx, case in enumerate(test_cases, 1):
            args   = case.get("args", []) or []
            kwargs = case.get("kwargs", {}) or {}
            expect = case.get("expect")
            mode   = case.get("compare", "eq")
            try:
                out = fn(*args, **kwargs)

                if mode == "eq":
                    ok = (out == expect)
                elif mode == "contains":
                    ok = str(expect) in str(out)
                elif mode == "custom":
                    ok = bool(eval(case.get("custom", "lambda *_: False"))(out))
                else:
                    ok = False

                passed += ok
                failed += (not ok)
                results.append({"case": idx, "passed": ok, "output": out})
            except Exception as e:
                failed += 1
                results.append({
                    "case": idx,
                    "passed": False,
                    "error": f"{e}",
                    "trace": traceback.format_exc(limit=2),
                })

        return {"tool": tool_name, "passed": passed, "failed": failed, "results": results}

    # This static method evaluates a tool using an arbitrary metric defined as a single-line lambda function. It runs the tool with provided sample inputs and returns a dictionary with scores, mean score, and details of each run.
    @staticmethod
    def evaluate_tool(
        tool_name: str,
        metric_code: str,
        sample_inputs: list[dict],
    ) -> dict:
        """
        Evaluate a tool with an arbitrary metric.

        • *metric_code* must be a **single-line λ**:  
          `lambda output, **inputs: <float between 0-1>` (higher = better).

        • *sample_inputs*  → list of **kwargs** dicts supplied to the tool.

        Returns {"scores": [...], "mean_score": <float>, "details": [...]}
        """
        import statistics, traceback

        fn = getattr(Tools, tool_name, None)
        if not fn or not callable(fn):
            return {"error": f"Tool '{tool_name}' not found."}

        try:
            scorer = eval(metric_code)
            assert callable(scorer)          # noqa: S101
        except Exception as e:
            return {"error": f"Invalid metric_code: {e}"}

        scores, details = [], []
        for inp in sample_inputs:
            try:
                out = fn(**inp)
                score = float(scorer(out, **inp))
            except Exception as e:
                score = 0.0
                details.append({"input": inp, "error": str(e),
                                "trace": traceback.format_exc(limit=1)})
            scores.append(score)

        mean = statistics.mean(scores) if scores else 0.0
        return {"scores": scores, "mean_score": mean, "details": details}

    # This static method loads all external tools from the `external_tools` directory, importing each Python file and attaching its public functions as static methods on the Tools class. It also logs the loaded tools.
    @staticmethod
    def load_external_tools() -> None:
        """
        Import every *.py* file in *external_tools/* and attach its **public**
        callables as @staticmethods on Tools.
        """
        import os, inspect, importlib.machinery, importlib.util

        ext_dir = os.path.join(os.path.dirname(__file__), "external_tools")
        os.makedirs(ext_dir, exist_ok=True)

        for fname in os.listdir(ext_dir):
            if fname.startswith("_") or not fname.endswith(".py"):
                continue

            mod_name = f"external_tools.{fname[:-3]}"
            path     = os.path.join(ext_dir, fname)

            loader = importlib.machinery.SourceFileLoader(mod_name, path)
            spec   = importlib.util.spec_from_loader(mod_name, loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)               # actual import

            for name, fn in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("_") or hasattr(Tools, name):
                    continue
                setattr(Tools, name, staticmethod(fn))
                log_message(f"Loaded external tool: {name}()", "INFO")

        # keep the public manifest in sync for other agents
        Tools.discover_agent_stack()


    @staticmethod
    def create_file(filename: str,
                    content: str,
                    base_dir: str = WORKSPACE_DIR) -> str:
        """
       Create (or overwrite) a new text file.

        Args:
            filename (str, required): Name **or relative path** of the file to create under
                `base_dir`. Use forward slashes for sub-dirs (“notes/poem.txt”).
            content  (str, required): Full text to write.  Pass "" for an empty file.
            base_dir (str, optional, default=WORKSPACE_DIR): Root folder for the workspace.
                Should normally be left as default.

        Returns:
            str: "Created file: <absolute_path>" on success, otherwise
                 "Error creating file '<absolute_path>': <reason>".

        Errors:
            • Intermediate directories that cannot be created.
            • I/O permission problems.

        Example:
            Tools.create_file("german_poem.txt",
                              "Goldener Abend, still und rein…")
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Created file: {path}"
        except Exception as e:
            return f"Error creating file {path!r}: {e}"

    @staticmethod
    def append_file(filename: str,
                    content: str,
                    base_dir: str = WORKSPACE_DIR) -> str:
        """
       Append text to the end of a file (create if missing).

        Args:
            filename (str, required): Relative path under `base_dir`.
            content  (str, required): Text to append (no newline automatically added).
            base_dir (str, optional): Defaults to WORKSPACE_DIR.

        Returns:
            str: "Appended to file: <absolute_path>" or
                 "Error appending to file '<absolute_path>': <reason>".

        Example:
            Tools.append_file("log/run.txt", "\\nFinished at 21:03")
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Appended to file: {path}"
        except Exception as e:
            return f"Error appending to file {path!r}: {e}"

    @staticmethod
    def delete_file(filename: str,
                    base_dir: str = WORKSPACE_DIR) -> str:
        """
       Delete a file.

        Args:
            filename (str, required): Relative path under `base_dir`.
            base_dir (str, optional): Workspace root.

        Returns:
            str: "Deleted file: <absolute_path>", "File not found: <absolute_path>",
                 or "Error deleting file '<absolute_path>': <reason>".

        Example:
            Tools.delete_file("old/tmp.txt")
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.remove(path)
            return f"Deleted file: {path}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error deleting file {path!r}: {e}"

    # ────────────────────────────────────────
    #  Directory / listing helpers
    # ────────────────────────────────────────

    @staticmethod
    def list_workspace(base_dir: str = WORKSPACE_DIR) -> str:
        """
       Return the top-level directory listing as **JSON text**.

        Args:
            base_dir (str, optional): Folder to list. Default is WORKSPACE_DIR.

        Returns:
            str (JSON array): e.g. '["file1.txt", "subdir", …]'  — or
            str (JSON object): '{"error":"<reason>"}' on failure.

        Example:
            contents_json = Tools.list_workspace()
        """
        import os, json
        try:
            entries = os.listdir(base_dir)
            return json.dumps(entries)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @staticmethod
    def find_files(pattern: str,
                   path: str = WORKSPACE_DIR) -> str:
        """
       Recursive file search (glob).

        Args:
            pattern (str, required): Unix glob (e.g. "*.md").
            path    (str, optional): Directory root for the walk.

        Returns:
            str (JSON array): Each element = {"file": "<name>", "dir": "<abs dir>"}.

        Example:
            matches_json = Tools.find_files("*.py")
        """
        import os, fnmatch, json
        matches = []
        for root, _, files in os.walk(path):
            for fname in files:
                if fnmatch.fnmatch(fname, pattern):
                    matches.append({"file": fname, "dir": root})
        return json.dumps(matches)

    @staticmethod
    def list_dir(path: str = WORKSPACE_DIR) -> str:
        """
       Non-recursive listing (JSON string).

        Args:
            path (str, optional): Directory to list.

        Returns:
            str (JSON array | JSON object{"error":…})

        Example:
            Tools.list_dir("data")
        """
        import os, json
        try:
            return json.dumps(os.listdir(path))
        except Exception as e:
            return json.dumps({"error": str(e)})

    @staticmethod
    def list_files(path: str = WORKSPACE_DIR,
                   pattern: str = "*") -> list:
        """
       Pythonic wrapper around `find_files()`.

        Args:
            path    (str, optional): Search root.
            pattern (str, optional): Glob (default "*").

        Returns:
            list[dict]: Same objects as `find_files`, but decoded.

        Example:
            py_files = Tools.list_files("src", "*.py")
        """
        import json
        return json.loads(Tools.find_files(pattern, path))

    # ────────────────────────────────────────
    #  File-reading / writing helpers
    # ────────────────────────────────────────

    @staticmethod
    def read_files(path: str, *filenames: str) -> dict:
        """
       Read multiple files in one call.

        Args:
            path       (str, required): Directory containing the files.
            *filenames (str, required): One or more filenames.

        Returns:
            dict[str,str]: {filename: content | error string}.

        Example:
            texts = Tools.read_files("logs", "out.txt", "err.txt")
        """
        out = {}
        for fn in filenames:
            out[fn] = Tools.read_file(fn, path)
        return out

    @staticmethod
    def read_file(filepath: str,
                  base_dir: str = WORKSPACE_DIR) -> str:
        """
       Read a single text file.

        Args:
            filepath (str, required): Absolute path **or** path relative to `base_dir`.
            base_dir (str | None, optional): If provided, `filepath` is resolved
                under it.

        Returns:
            str: File contents on success, or
                 "Error reading '<absolute_path>': <reason>".

        Example:
            body = Tools.read_file("notes/poem.txt", Tools.WORKSPACE_DIR)
        """
        import os
        path = os.path.join(base_dir, filepath) if base_dir else filepath
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path!r}: {e}"

    @staticmethod
    def write_file(filepath: str,
                   content: str,
                   base_dir: str = WORKSPACE_DIR) -> str:
        """
       Write (overwrite) a text file. **This is the canonical write helper.**

        Args:
            filepath (str, required): Destination file path (use forward slashes).
            content  (str, required): Full text to write.  
            base_dir (str | None, optional): Prefix directory; if `None`, `filepath`
                must be absolute.

        Returns:
            str: "Wrote <n> chars to '<absolute_path>'" or
                 "Error writing '<absolute_path>': <reason>".

        Example:
            Tools.write_file("docs/readme.md", "# Intro\\n")
        """
        import os
        path = os.path.join(base_dir, filepath) if base_dir else filepath
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to {path!r}"
        except Exception as e:
            return f"Error writing {path!r}: {e}"

    # ────────────────────────────────────────
    #  Rename / copy helpers
    # ────────────────────────────────────────

    @staticmethod
    def rename_file(old: str,
                    new: str,
                    base_dir: str = WORKSPACE_DIR) -> str:
        """
       Rename (move) a file **within** the workspace.

        Args:
            old (str, required): Existing relative path.
            new (str, required): New relative path.
            base_dir (str, optional): Workspace root.

        Returns:
            str: "Renamed <old> → <new>" or "Error renaming file: <reason>".

        Security:
            • Both paths are `os.path.normpath()`’d and must stay under `base_dir`.

        Example:
            Tools.rename_file("tmp.txt", "archive/tmp.txt")
        """
        import os
        safe_old = os.path.normpath(old)
        safe_new = os.path.normpath(new)
        if safe_old.startswith("..") or safe_new.startswith(".."):
            return "Error: Invalid path"
        src = os.path.join(base_dir, safe_old)
        dst = os.path.join(base_dir, safe_new)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            return f"Renamed {safe_old} → {safe_new}"
        except Exception as e:
            return f"Error renaming file: {e}"

    @staticmethod
    def copy_file(src: str,
                  dst: str,
                  base_dir: str = WORKSPACE_DIR) -> str:
        """
       Copy a file inside the workspace.

        Args:
            src (str, required): Existing file path.
            dst (str, required): Destination path.
            base_dir (str, optional): Workspace root.

        Returns:
            str: "Copied <src> → <dst>" or "Error copying file: <reason>".

        Example:
            Tools.copy_file("data/raw.csv", "backup/raw.csv")
        """
        import os, shutil
        safe_src = os.path.normpath(src)
        safe_dst = os.path.normpath(dst)
        if safe_src.startswith("..") or safe_dst.startswith(".."):
            return "Error: Invalid path"
        src_path = os.path.join(base_dir, safe_src)
        dst_path = os.path.join(base_dir, safe_dst)
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
            return f"Copied {safe_src} → {safe_dst}"
        except Exception as e:
            return f"Error copying file: {e}"

    # ────────────────────────────────────────
    #  Metadata helpers
    # ────────────────────────────────────────

    @staticmethod
    def file_exists(filename: str,
                    base_dir: str = WORKSPACE_DIR) -> bool:
        """
       Check existence of a file.

        Args:
            filename (str, required): Relative path.
            base_dir (str, optional)

        Returns:
            bool: True if present and within workspace, False otherwise.

        Example:
            if Tools.file_exists("output/result.json"): ...
        """
        import os
        safe = os.path.normpath(filename)
        if safe.startswith(".."):
            return False
        return os.path.exists(os.path.join(base_dir, safe))

    @staticmethod
    def file_info(filename: str,
                  base_dir: str = WORKSPACE_DIR) -> dict:
        """
       Stat a file (size & mtime).

        Args:
            filename (str, required)
            base_dir (str, optional)

        Returns:
            dict:
                • "size" (int) in bytes  
                • "modified" (float) UNIX epoch seconds  
              or {"error": "<reason>"}.

        Example:
            meta = Tools.file_info("report.pdf")
        """
        import os
        safe = os.path.normpath(filename)
        if safe.startswith(".."):
            return {"error": "Invalid path"}
        path = os.path.join(base_dir, safe)
        try:
            st = os.stat(path)
            return {"size": st.st_size, "modified": st.st_mtime}
        except Exception as e:
            return {"error": str(e)}

    # ────────────────────────────────────────
    #  Utility helpers
    # ────────────────────────────────────────

    @staticmethod
    def get_workspace_dir() -> str:
        """
       Absolute workspace root.

        Returns:
            str: Value of Tools.WORKSPACE_DIR.

        Example:
            root = Tools.get_workspace_dir()
        """
        return Tools.WORKSPACE_DIR

    @staticmethod
    def get_cwd() -> str:
        """
       Process current working directory.

        Returns:
            str: Result of os.getcwd().

        Example:
            print("Running from", Tools.get_cwd())
        """
        import os
        return os.getcwd()

    
    # We define a static method to introspect the available tools and agents, writing the results to a JSON file named `agent_stack.json`. This method collects tool and agent names, default stages, and the last updated timestamp.
    @staticmethod
    def discover_agent_stack() -> str:
        """
        Introspect Tools + Agent classes, write agent_stack.json.
        """
        import os, sys, json, inspect
        from datetime import datetime

        module_path = os.path.dirname(os.path.abspath(__file__))
        stack_path  = os.path.join(module_path, "agent_stack.json")

        tools = [
            name for name, fn in inspect.getmembers(Tools, predicate=callable)
            if not name.startswith("_")
        ]
        agents = [
            name for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
            if name.endswith("Manager") or name.endswith("ChatManager")
        ]

        default_stages = [
            "summary_request",
            "timeframe_history_query",
            "record_user_message",
            "context_analysis",
            "intent_clarification",
            "external_knowledge_retrieval",
            "memory_summarization",
            "planning_summary",
            "tool_self_improvement", 
            "tool_chaining",
            "assemble_prompt",
            "final_inference",
            "chain_of_thought",
            "notification_audit"
        ]

        config = {
            "tools":   tools,
            "agents":  agents,
            "stages":  default_stages,
            "updated": datetime.now().isoformat()
        }

        with open(stack_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return f"agent_stack.json created with {len(tools)} tools & {len(agents)} agents."
    
    # This static method loads the agent stack from the `agent_stack.json` file. It handles cases where the file is missing or corrupted, regenerating it if necessary, and ensures that all required keys are present in the JSON structure.
    @staticmethod
    def load_agent_stack() -> dict:
        """
        Return the contents of *agent_stack.json*.

        • If the file is missing or unreadable ⇒ call
          `Tools.discover_agent_stack()` to (re)create it, then reload.
        • If the JSON loads but is missing any of the canonical
          top-level keys (“tools”, “agents”, “stages”) ⇒ we *merge-patch*
          those keys from a fresh discovery result.
        """
        import os, json, copy

        module_path = os.path.dirname(os.path.abspath(__file__))
        stack_path  = os.path.join(module_path, "agent_stack.json")

        # helper: (re)generate the stack file on disk
        def _regen() -> dict:
            Tools.discover_agent_stack()          # writes the file
            with open(stack_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 1️⃣  Ensure the file exists – otherwise create from scratch
        if not os.path.isfile(stack_path):
            return _regen()

        # 2️⃣  Try to load it; regenerate on corruption
        try:
            with open(stack_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return _regen()

        # 3️⃣  Patch any missing keys without nuking user additions
        required_keys = {"tools", "agents", "stages"}
        if not required_keys.issubset(data.keys()):
            fresh = _regen()
            merged = copy.deepcopy(fresh)          # start with complete set
            merged.update(data)                    # user keys / overrides win
            # ensure required keys exist after merge
            for k in required_keys:
                merged.setdefault(k, fresh[k])
            # write the patched file back
            with open(stack_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            return merged

        return data

    # This static method updates the `agent_stack.json` file with user-specified changes. It merges the changes into the existing configuration, appends a change history entry if a justification is provided, and writes the updated configuration back to disk.
    @staticmethod
    def update_agent_stack(changes: dict, justification: str | None = None) -> str:
        """
        Merge `changes` into agent_stack.json, and if `justification` is provided,
        append an entry to `change_history` with timestamp, changes, and justification.
        """
        import os, json
        from datetime import datetime

        module_path = os.path.dirname(os.path.abspath(__file__))
        stack_path  = os.path.join(module_path, "agent_stack.json")

        # load existing stack (or discover a fresh one)
        config = Tools.load_agent_stack()

        # apply the user‐requested changes
        config.update(changes)
        now = datetime.now().isoformat()
        config["updated"] = now

        # record a change_history entry
        if justification:
            entry = {
                "timestamp":     now,
                "changes":       changes,
                "justification": justification
            }
            config.setdefault("change_history", []).append(entry)

        # write back to disk
        with open(stack_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return "agent_stack.json updated."

    # This static method attempts to find a working system-wide chromedriver executable. It checks common locations and verifies that the executable can run without errors, returning the path if successful or None if not found.
    @staticmethod
    def _find_system_chromedriver() -> str | None:
        """
        Return a path to a *working* chromedriver executable on this machine
        (correct CPU arch + executable).  We try common locations first;
        each candidate must:
        1. Exist and be executable for the current user.
        2. Run `--version` without raising OSError (catches x86-64 vs arm64).
        If none pass, we return None so the caller can fall back to
        webdriver-manager or raise.
        """
        candidates: list[str | None] = [
            shutil.which("chromedriver"),                         # Anything already in PATH
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver",
            "/snap/bin/chromium.chromedriver",
            "/usr/lib/chromium-browser/chromedriver",
            "/opt/homebrew/bin/chromedriver",                     # macOS arm64
        ]

        for path in filter(None, candidates):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                try:
                    # If this fails with Exec format error, we skip it.
                    subprocess.run([path, "--version"],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
                    return path
                except Exception:
                    continue
        return None
    
    # This static method waits for the document to be fully loaded in a Selenium WebDriver instance. It blocks until the `document.readyState` is "complete", ensuring that the page is fully loaded before proceeding with further actions.
    @staticmethod
    def _wait_for_ready(drv, timeout=6):
        """Block until document.readyState == 'complete'."""
        WebDriverWait(drv, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

    # This static method waits for a specific element to be present and enabled in the DOM, given a list of CSS selectors. It returns the first matching WebElement or None if none are found within the specified timeout.
    @staticmethod
    def _first_present(drv, selectors: list[str], timeout=4):
        """
        Return the first WebElement present+enabled out of a list of CSS selectors.
        Returns None if none arrive within `timeout`.
        """
        for sel in selectors:
            try:
                return WebDriverWait(drv, timeout).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
            except TimeoutException:
                continue
        return None

    # This static method defines a condition for WebDriverWait that checks if an element is both visible and enabled. It returns a callable that can be used with WebDriverWait to wait for the specified condition.
    @staticmethod
    def _visible_and_enabled(locator):
        """condition: element is displayed *and* not disabled."""
        def _cond(drv):
            try:
                el = drv.find_element(*locator)
                return el.is_displayed() and el.is_enabled()
            except Exception:
                return False
        return _cond

    # This static method opens a Chrome/Chromium browser using Selenium WebDriver. It tries multiple methods to find a suitable chromedriver, including Selenium-Manager, system-wide chromedriver, and webdriver-manager. It handles different CPU architectures and returns a message indicating success or failure.
    @staticmethod
    def open_browser(headless: bool = False, force_new: bool = False) -> str:
        """
        Launch Chrome/Chromium on x86-64 and ARM64.
        Order:
          1) Selenium-Manager
          2) Snap’s bundled chromedriver
          3) Debian/Ubuntu ELF chromedriver
          4) /usr/local/bin chromedriver
          5) /usr/bin chromedriver
          6) PATH chromedriver
          7) Fallback: retry with snap chromium binary
          8) Auto-download & install ARM64 chromedriver if on ARM64
          9) webdriver-manager (x86_64 only)
        """
        import os
        import random
        import shutil
        import subprocess
        import platform
    
        from selenium import webdriver
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
    
        # 0️⃣ Tear down any existing driver if requested
        if force_new and Tools._driver:
            try:
                Tools._driver.quit()
            except Exception:
                pass
            Tools._driver = None
    
        if Tools._driver:
            return "Browser already open"
    
        # 1️⃣ Locate your Chrome/Chromium binary
        chrome_bin = (
            os.getenv("CHROME_BIN")
            or shutil.which("google-chrome")
            or shutil.which("chromium")
            or "/usr/bin/chromium"
        )
    
        # 2️⃣ Build ChromeOptions
        opts = Options()
        opts.binary_location = chrome_bin
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--remote-allow-origins=*")
        opts.add_argument(f"--remote-debugging-port={random.randint(45000,65000)}")
        # isolate profile to avoid "in use" errors
        tmp_profile = f"/tmp/chrome_profile_{random.getrandbits(64)}"
        opts.add_argument(f"--user-data-dir={tmp_profile}")
    
        # 3️⃣ Selenium-Manager (bundled driver for Chrome ≥115)
        try:
            log_message("[open_browser] Trying Selenium-Manager…", "DEBUG")
            Tools._driver = webdriver.Chrome(options=opts)
            log_message("[open_browser] Launched via Selenium-Manager.", "SUCCESS")
            return "Browser launched (selenium-manager)"
        except WebDriverException as e:
            log_message(f"[open_browser] Selenium-Manager failed: {e}", "WARNING")
    
        # 4️⃣ Try known chromedriver binaries in order
        candidates = [
            "/snap/chromium/current/usr/lib/chromium-browser/chromedriver",  # snap
            "/usr/lib/chromium-browser/chromedriver",                        # Debian/Ubuntu ELF
            "/usr/local/bin/chromedriver",                                   # local install
            "/usr/bin/chromedriver",                                         # distro install
            shutil.which("chromedriver"),                                    # PATH fallback
        ]
        for drv in candidates:
            if drv and os.path.exists(drv):
                try:
                    log_message(f"[open_browser] Trying chromedriver at {drv}", "DEBUG")
                    Tools._driver = webdriver.Chrome(service=Service(drv), options=opts)
                    log_message(f"[open_browser] Launched via chromedriver at {drv}", "SUCCESS")
                    return f"Browser launched (chromedriver at {drv})"
                except WebDriverException as e:
                    log_message(f"[open_browser] chromedriver at {drv} failed: {e}", "WARNING")
    
        # 7️⃣ Fallback: retry with snap chromium binary if available
        snap_bin = "/snap/bin/chromium"
        if os.path.exists(snap_bin) and os.getenv("CHROME_BIN") != snap_bin:
            os.environ["CHROME_BIN"] = snap_bin
            opts.binary_location = snap_bin
            log_message("[open_browser] Retrying with CHROME_BIN=/snap/bin/chromium", "DEBUG")
            # retry Selenium-Manager
            try:
                Tools._driver = webdriver.Chrome(options=opts)
                log_message("[open_browser] Launched via Selenium-Manager (snap CHROME_BIN)", "SUCCESS")
                return "Browser launched (selenium-manager via snap)"
            except WebDriverException as e:
                log_message(f"[open_browser] Selenium-Manager via snap failed: {e}", "WARNING")
            # retry PATH chromedriver under new CHROME_BIN
            snap_sys = shutil.which("chromedriver")
            if snap_sys:
                try:
                    log_message(f"[open_browser] Trying chromedriver at {snap_sys} (after snap CHROME_BIN)", "DEBUG")
                    Tools._driver = webdriver.Chrome(service=Service(snap_sys), options=opts)
                    log_message("[open_browser] Launched via chromedriver via snap PATH", "SUCCESS")
                    return "Browser launched (chromedriver via snap PATH)"
                except WebDriverException as e:
                    log_message(f"[open_browser] chromedriver via snap PATH failed: {e}", "WARNING")
    
        # 8️⃣ ARM64 auto-download & install if on ARM
        arch = platform.machine().lower()
        if arch in ("aarch64", "arm64", "armv8l", "armv7l"):
            try:
                raw = subprocess.check_output([chrome_bin, "--version"]).decode().strip()
                ver = raw.split()[1]  # e.g. "138.0.7204.92"
                url = (
                    f"https://edgedl.me.gvt1.com/edgedl/"
                    f"chrome/chrome-for-testing/{ver}/linux-arm64/"
                    f"chromedriver-linux-arm64.zip"
                )
                tmp_zip = "/tmp/chromedriver_arm64.zip"
                log_message(f"[open_browser] Downloading ARM64 driver from {url}", "DEBUG")
                subprocess.check_call(["wget", "-qO", tmp_zip, url])
                subprocess.check_call(["unzip", "-o", tmp_zip, "-d", "/tmp"])
                subprocess.check_call(["sudo", "mv", "/tmp/chromedriver", "/usr/local/bin/chromedriver"])
                subprocess.check_call(["sudo", "chmod", "+x", "/usr/local/bin/chromedriver"])
                drv = shutil.which("chromedriver")
                log_message(f"[open_browser] Installed ARM64 driver at {drv}", "DEBUG")
                Tools._driver = webdriver.Chrome(service=Service(drv), options=opts)
                log_message("[open_browser] Launched via downloaded ARM64 chromedriver.", "SUCCESS")
                return "Browser launched (downloaded ARM64 chromedriver)"
            except Exception as e:
                log_message(f"[open_browser] ARM64 download/install failed: {e}", "WARNING")
    
        # 9️⃣ webdriver-manager fallback on x86_64
        if arch in ("x86_64", "amd64"):
            try:
                raw = subprocess.check_output([chrome_bin, "--version"]).decode().strip()
                browser_major = raw.split()[1].split(".")[0]
            except Exception:
                browser_major = "latest"
            try:
                log_message(
                    f"[open_browser] Installing ChromeDriver {browser_major} via webdriver-manager",
                    "DEBUG",
                )
                drv_path = ChromeDriverManager(driver_version=browser_major).install()
                log_message(f"[open_browser] webdriver-manager driver at {drv_path}", "DEBUG")
                Tools._driver = webdriver.Chrome(service=Service(drv_path), options=opts)
                log_message("[open_browser] Launched via webdriver-manager.", "SUCCESS")
                return "Browser launched (webdriver-manager)"
            except Exception as e:
                log_message(f"[open_browser] webdriver-manager failed: {e}", "ERROR")
                raise RuntimeError(
                    "webdriver-manager failed on x86_64; install matching chromedriver or set CHROME_BIN/PATH."
                ) from e
    
        # ❌ If we reach here, no driver worked
        raise RuntimeError(
            "No usable chromedriver found. On ARM64, ensure download/install succeeded; "
            "on x86_64, install a matching chromedriver or set CHROME_BIN/PATH."
        )





    # This static method closes the currently open browser session, if any. It attempts to quit the WebDriver instance and handles exceptions gracefully, returning a message indicating whether the browser was closed or if there was no browser to close.
    @staticmethod
    def close_browser() -> str:
        if Tools._driver:
            try:
                Tools._driver.quit()
                log_message("[close_browser] Browser closed.", "DEBUG")
            except Exception:
                pass
            Tools._driver = None
            return "Browser closed"
        return "No browser to close"

    # This static method navigates the currently open browser to a specified URL. It checks if the browser is open, logs the navigation action, and returns a message indicating success or failure.
    @staticmethod
    def navigate(url: str) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        log_message(f"[navigate] → {url}", "DEBUG")
        Tools._driver.get(url)
        return f"Navigated to {url}"

    # This static method clicks on a specified element in the currently open browser using a CSS selector. It waits for the element to be clickable, scrolls it into view, and logs the action. It returns a message indicating success or failure.
    @staticmethod
    def click(selector: str, timeout: int = 8) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        try:
            drv = Tools._driver
            el = WebDriverWait(drv, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.click()
            focused = drv.execute_script("return document.activeElement === arguments[0];", el)
            log_message(f"[click] {selector} clicked (focused={focused})", "DEBUG")
            return f"Clicked {selector}"
        except Exception as e:
            log_message(f"[click] Error clicking {selector}: {e}", "ERROR")
            return f"Error clicking {selector}: {e}"

    # This static method inputs text into a specified element in the currently open browser using a CSS selector. It waits for the element to be clickable, scrolls it into view, clears any existing text, and sends the specified text followed by a RETURN key. It logs the action and returns a message indicating success or failure.
    @staticmethod
    def input(selector: str, text: str, timeout: int = 8) -> str:
        """
        This static method inputs text into a specified element in the currently open browser using a CSS selector. Do not use for replying
        """
        if not Tools._driver:
            return "Error: browser not open"
        try:
            drv = Tools._driver
            el = WebDriverWait(drv, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.clear()
            el.send_keys(text + Keys.RETURN)
            log_message(f"[input] Sent {text!r} to {selector}", "DEBUG")
            return f"Sent {text!r} to {selector}"
        except Exception as e:
            log_message(f"[input] Error typing into {selector}: {e}", "ERROR")
            return f"Error typing into {selector}: {e}"

    # This static method retrieves the current HTML content of the page in the currently open browser. It checks if the browser is open, retrieves the page source, and returns it as a string. If the browser is not open, it returns an error message.
    @staticmethod
    def get_html() -> str:
        if not Tools._driver:
            return "Error: browser not open"
        return Tools._driver.page_source

    # This static method takes a screenshot of the currently open browser and saves it to a specified filename. It checks if the browser is open, saves the screenshot, and returns the filename. If the browser is not open, it returns an error message.
    @staticmethod
    def screenshot(filename: str = "screenshot.png") -> str:
        if not Tools._driver:
            return "Error: browser not open"
        Tools._driver.save_screenshot(filename)
        return filename

    @staticmethod
    def search_internet(topic: str, num_results: int = 5, wait_sec: int = 1, deep_scrape: bool = True, **kwargs) -> list:
        """
        Ultra-quick DuckDuckGo search (event-driven, JS injection).
        THIS RETURNS A RAW MASSIVE EXTRACTED WEBPAGE, call summarize_search(topic="content") instead IF YOU WANT BRIEF SUMMARIES!

        1. Call search_internet(topic=str, top_n=int)
        - topic (str): the search term, use `topic`
        - top_n (int): how many results, use `top_n`

        • Opens the first *num_results* links in separate tabs and deep-scrapes each.
        • Returns: title, url, snippet, summary, and full page HTML (`content`).
        • Never blocks more than 5 s on any wait—everything is aggressively polled.
        """

        # ——— Argument‐alias handling ——————————————
        # allow users to pass top_n, n, limit, etc.
        if 'top_n' in kwargs:
            num_results = kwargs.pop('top_n')
        if 'n' in kwargs:
            num_results = kwargs.pop('n')
        if 'limit' in kwargs:
            num_results = kwargs.pop('limit')
        # warn about any other unexpected kwargs
        if kwargs:
            log_message(f"[search_internet] Ignoring unexpected args: {list(kwargs.keys())!r}", "WARNING")

        log_message(f"[search_internet] ▶ {topic!r} (num_results={num_results})", "INFO")

        # clamp waits to max 5 s
        wait_sec = min(wait_sec, 5)

        # fresh browser session
        Tools.close_browser()
        Tools.open_browser(headless=False, force_new=True)
        drv = Tools._driver
        wait = WebDriverWait(drv, wait_sec, poll_frequency=0.1)
        results = []

        try:
            # 1️⃣ Home page
            drv.get("https://duckduckgo.com/")
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            log_message("[search_internet] Home page ready.", "DEBUG")

            # 2️⃣ Cookie banner
            try:
                btn = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler")
                ))
                btn.click()
                log_message("[search_internet] Cookie banner dismissed.", "DEBUG")
            except TimeoutException:
                pass

            # 3️⃣ Locate search box
            selectors = (
                "input#search_form_input_homepage",
                "input#searchbox_input",
                "input[name='q']",
            )
            box = next((drv.find_element(By.CSS_SELECTOR, sel)
                        for sel in selectors if drv.find_elements(By.CSS_SELECTOR, sel)),
                       None)
            if not box:
                raise RuntimeError("Search box not found!")

            # submit query
            drv.execute_script(
                "arguments[0].value = arguments[1];"
                "arguments[0].dispatchEvent(new Event('input'));"
                "arguments[0].form.submit();",
                box, topic
            )
            log_message("[search_internet] Query submitted.", "DEBUG")

            # 4️⃣ Wait for results
            try:
                wait.until(lambda d: "?q=" in d.current_url)
                wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#links .result, #links [data-nr]"))
                log_message("[search_internet] Results detected.", "DEBUG")
            except TimeoutException:
                log_message("[search_internet] Results timeout.", "WARNING")

            # 5️⃣ Gather top anchors
            anchors = drv.find_elements(
                By.CSS_SELECTOR,
                "a.result__a, a[data-testid='result-title-a']"
            )[:num_results]

            main_handle = drv.current_window_handle

            for a in anchors:
                try:
                    href  = a.get_attribute("href")
                    title = a.text.strip() or html.unescape(drv.execute_script(
                        "return arguments[0].innerText;", a
                    ))

                    # snippet
                    try:
                        parent = a.find_element(By.XPATH, "./ancestor::*[contains(@class,'result')][1]")
                        sn = parent.find_element(By.CSS_SELECTOR,
                            ".result__snippet, span[data-testid='result-snippet']")
                        snippet = sn.text.strip()
                    except NoSuchElementException:
                        snippet = ""

                    summary = snippet
                    page_content = ""

                    # 6️⃣ Deep scrape in new tab  (hard 10-second cap)
                    if deep_scrape and href:
                        drv.switch_to.new_window("tab")

                        # ── absolute timeout for *everything* in this tab ──
                        TAB_DEADLINE = time.time() + 10        # 10 s wall-clock
                        drv.set_page_load_timeout(10)          # network-level cut-off

                        try:
                            drv.get(href)
                        except TimeoutException:
                            log_message(f"[search_internet] page-load timeout for {href!r}", "WARNING")

                        # track DOM mutation activity
                        drv.execute_script("""
                            window._lastMut = Date.now();
                            new MutationObserver(function() {
                                window._lastMut = Date.now();
                            }).observe(document, {childList:true,subtree:true,attributes:true});
                        """)

                        # wait until DOM is stable *or* we hit the 10-s wall
                        STABLE_MS   = 500                      # ms of silence to call it “done”
                        POLL_DELAY  = 0.1
                        while True:
                            if time.time() >= TAB_DEADLINE:
                                log_message(f"[search_internet] hard 10-s deadline for {href!r}", "WARNING")
                                break
                            last = drv.execute_script("return window._lastMut;")
                            if (time.time()*1000) - last > STABLE_MS:
                                break
                            time.sleep(POLL_DELAY)

                        # grab whatever is available and move on
                        page_content = drv.page_source or ""
                        if not page_content.startswith("<"):
                            # fallback to bs4 helper if primary is empty / error
                            maybe = Tools.bs4_scrape(href)
                            if not maybe.startswith("Error"):
                                page_content = maybe

                        # quick summary extraction
                        summary = snippet
                        try:
                            pg = BeautifulSoup(page_content, "html5lib")
                            meta = pg.find("meta", attrs={"name":"description"})
                            ptag = pg.find("p")
                            if meta and meta.get("content"):
                                summary = meta["content"].strip()
                            elif ptag:
                                summary = ptag.get_text(strip=True)
                        except Exception:
                            pass

                        drv.close()                               # always close the tab
                        drv.switch_to.window(main_handle)

                    clean_content = (
                        BeautifulSoup(page_content, "html.parser")
                        .get_text(separator=" ", strip=True)
                        if page_content else ""
                    )

                    results.append({
                        "title":   title,
                        "url":     href,
                        "snippet": snippet,
                        "summary": summary,
                        "content": clean_content,
                    })

                except Exception as ex:
                    log_message(f"[search_internet] result error: {ex}", "WARNING")
                    continue

        except Exception as e:
            log_message(f"[search_internet] Fatal: {e}\n{traceback.format_exc()}", "ERROR")
        finally:
            Tools.close_browser()

        log_message(f"[search_internet] Collected {len(results)} results.", "SUCCESS")
        return results


    # This static method extracts a summary from a webpage using two stages:
    @staticmethod
    def selenium_extract_summary(url: str, wait_sec: int = 8) -> str:
        """
        Do not use this tools, call summarize_search(topic="content") instead!

        Fast two-stage page summariser:

        1. Try a lightweight `Tools.bs4_scrape()` (no browser).
           If it yields HTML, grab <meta name="description"> or first <p>.
        2. If bs4-scrape fails, fall back to *headless* Selenium.
           Uses same driver logic.
        3. Always cleans up the browser session.
        """

        # 1) quick request-based scrape
        html_doc = Tools.bs4_scrape(url)
        if html_doc and not html_doc.startswith("Error"):
            pg = BeautifulSoup(html_doc, "html5lib")
            m = pg.find("meta", attrs={"name": "description"})
            if m and m.get("content"):
                return m["content"].strip()
            p = pg.find("p")
            if p:
                return p.get_text(strip=True)

        # 2) fall back to headless Selenium
        try:
            Tools.open_browser(headless=True)
            drv = Tools._driver
            wait = WebDriverWait(drv, wait_sec)
            drv.get(url)
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            except TimeoutException:
                log_message(f"[selenium_extract_summary] body never appeared for {url!r}, using page_source anyway", "WARNING")
            pg2 = BeautifulSoup(drv.page_source, "html5lib")
            m2 = pg2.find("meta", attrs={"name": "description"})
            if m2 and m2.get("content"):
                return m2["content"].strip()
            p2 = pg2.find("p")
            if p2:
                return p2.get_text(strip=True)
            return ""
        finally:
            Tools.close_browser()

    # This static method summarizes a local search by calling the search_internet method to get the top_n results for a given topic. It formats the results into a bullet list with titles and summaries, returning the formatted string.
    @staticmethod
    def summarize_local_search(topic: str, top_n: int = 3, deep: bool = False) -> str:
        """
        1) Call search_internet() to get top_n results (optionally deep-scraped)
        2) Return bullet list “1. Title — summary”
        """
        try:
            entries = Tools.search_internet(topic, num_results=top_n, deep_scrape=deep)
        except Exception as e:
            return f"Search error: {e}"
        if not entries:
            return "No results found."
        return "\n".join(
            f"{i}. {e['title']} — {e['summary'] or '(no summary)'}"
            for i, e in enumerate(entries, 1)
        )
        
    @staticmethod
    def get_chat_history(
        assembler,                    # required: Assembler instance (must have `.repo`)
        limit:    int | None = None,
        n:        int | None = None,
        count:    int | None = None,
        direction:str | None = None,
        time:     str | None = None,
        time_range: str | None = None,
        time_ago:  str | None = None,
        domain:   str | list[str] | None = None,
        component:str|list[str] | None = None,
        semantic_label: str|list[str] | None = None,
        keyword:  str | None = None,
        query:    str | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        """
        Retrieve a slice of past chat/context entries from the in-memory repository.

        Parameters
        ----------
        assembler
            An Assembler instance (must have a .repo attribute to query).
        limit, n, count : int, optional
            Alias for the maximum number of entries to return.
        direction : str, optional
            "fwd" or "forward" to sort oldest→newest, anything else (or None)
            for newest→oldest.
        time, time_range, time_ago : str, optional
            Textual time window. Any one may be used; priority is
            time_ago → time_range → time.
            Supported formats:
              - `"today"` / `"yesterday"`
              - `"last N days"`, `"last N hours"`, `"last hour"`
              - `"<N> minutes ago"`, `"<N> hours ago"`, etc.
            If omitted or unrecognized, skips window filtering.
        domain, component, semantic_label : str or list, optional
            Filter entries by ContextObject.domain, .component, or .semantic_label.
        keyword, query : str, optional
            If provided, rank candidates by simple substring match +
            embedding-similarity to this text.
        
        Returns
        -------
        dict
            A plain dict with key `"results"` mapping to a list of entries:
            `[{"timestamp":…, "role":…, "content":…}, …]`.  On error,
            returns `{"results": [], "error": "<message>"}`.

        Notes
        -----
        - `limit`, `n`, and `count` are interchangeable.
        - `time_ago` is sugar for things like `"10 minutes ago"`.
        - Embedding-based relevance only looks at the last 100 candidates for speed.
        """
        import re, traceback, json
        from datetime import datetime, timedelta
        from numpy.linalg import norm
        from numpy import dot

        print("\n=== get_chat_history called ===")
        print(f"Inputs → limit={limit}, n={n}, count={count}, direction={direction!r}, "
              f"time={time!r}, time_range={time_range!r}, time_ago={time_ago!r}, "
              f"domain={domain}, component={component}, semantic_label={semantic_label}, "
              f"keyword={keyword!r}, query={query!r}\n")

        try:
            # 1) alias limit/count/n
            if count is not None:
                print(f"Aliasing count={count} → n")
                n = count
            if limit is not None:
                print(f"Aliasing limit={limit} → n")
                n = limit

            # 2) choose textual window argument
            window = time_ago or time_range or time
            if window is not None:
                print(f"Using window='{window}'")

            # 3) fetch all entries
            entries = []
            for c in assembler.repo.query(lambda c: True):
                entries.append({
                    "timestamp":      c.timestamp,
                    "role":           c.metadata.get("role", ""),
                    "content":        c.metadata.get("content", c.summary or ""),
                    "domain":         c.domain,
                    "component":      c.component,
                    "semantic_label": c.semantic_label,
                })
            print(f"Fetched {len(entries)} entries\n")

            # 4) apply domain/component/semantic_label filters
            def _match(val, flt):
                if flt is None: return True
                if isinstance(flt, (list, tuple)): return val in flt
                return val == flt

            before = len(entries)
            entries = [
                e for e in entries
                if _match(e["domain"], domain)
                and _match(e["component"], component)
                and _match(e["semantic_label"], semantic_label)
            ]
            print(f"{len(entries)} after domain/component/semantic_label (was {before})\n")

            # 5) parse timestamps helper
            def _parse(ts: str) -> datetime:
                try:
                    return datetime.fromisoformat(ts.rstrip("Z"))
                except:
                    return datetime.strptime(ts, "%Y%m%dT%H%M%SZ")

            # 6) timeframe filtering
            results = []
            if window:
                period = window.lower().strip()
                now = datetime.utcnow()
                start = end = None

                # today / yesterday
                if period == "today":
                    d0 = datetime.combine(now.date(), datetime.min.time())
                    start, end = d0, d0 + timedelta(days=1)
                elif period == "yesterday":
                    d0 = datetime.combine(now.date()-timedelta(days=1), datetime.min.time())
                    start, end = d0, d0 + timedelta(days=1)
                else:
                    # last N days/hours
                    m_d = re.match(r"last\s+(\d+)\s+days?", period)
                    m_h = re.match(r"last\s+(\d+)\s+hours?", period)
                    # N minutes ago / N hours ago
                    m_ago = re.match(r"(\d+)\s*(minutes?|hours?|days?)\s+ago", period)
                    if m_d:
                        days = int(m_d.group(1)); start, end = now - timedelta(days=days), now
                    elif m_h:
                        hrs = int(m_h.group(1)); start, end = now - timedelta(hours=hrs), now
                    elif m_ago:
                        val, unit = int(m_ago.group(1)), m_ago.group(2).rstrip("s")
                        delta = {"minute": timedelta(minutes=val),
                                 "hour":   timedelta(hours=val),
                                 "day":    timedelta(days=val)}.get(unit, timedelta())
                        start, end = now - delta, now

                if start and end:
                    for e in entries:
                        try:
                            ts = _parse(e["timestamp"])
                        except:
                            continue
                        if start <= ts < end:
                            results.append(e)
                    print(f"{len(results)} in window '{window}' ({start}→{end})\n")
                else:
                    print(f"Could not parse window '{window}', skipping\n")
                    results = entries.copy()
            else:
                print("No window provided, skipping timeframe filter\n")
                results = entries.copy()

            # 7) relevance vs recency
            top_n = int(n) if n is not None else len(results)
            if keyword or query:
                txt = query or keyword or ""
                print(f"Relevance ranking to '{txt}'")
                qv = assembler.engine.embedder(txt)
                scored = []
                cand = results[-100:]
                for e in cand:
                    txt_score = float(txt.lower() in e["content"].lower())
                    vv = assembler.engine.embedder(e["content"])
                    emb_score = dot(qv, vv)/(norm(qv)*norm(vv)) if norm(qv) and norm(vv) else 0.0
                    scored.append((txt_score + emb_score, e))
                scored.sort(key=lambda x: x[0], reverse=True)
                results = [e for _, e in scored[:top_n]]
                print(f"{len(results)} after relevance (top {top_n})\n")
            else:
                print(f"Recency sort, taking top {top_n}\n")
                results.sort(key=lambda e: _parse(e["timestamp"]), reverse=True)
                results = results[:top_n]

            # 8) direction
            rev = not (direction and direction.lower().startswith("fwd"))
            order = "newest→oldest" if rev else "oldest→newest"
            results.sort(key=lambda e: _parse(e["timestamp"]), reverse=rev)
            print(f"Final direction: {order}\n")

            # 9) return raw list
            print("Returning results:")
            print(json.dumps(results, indent=2, ensure_ascii=False))
            return {"results": results}

        except Exception as exc:
            print("!!! Exception in get_chat_history:", exc)
            traceback.print_exc()
            return {"results": [], "error": str(exc)}

        finally:
            print("=== get_chat_history finished ===\n")



    # This static method retrieves the current local time in a formatted string. It uses the datetime module to get the current time, formats it as "YYYY-MM-DD HH:MM:SS", and logs the action.
    @staticmethod
    def get_current_time():
        """
        Return the current time and date in a human readable format.
        """
        from datetime import datetime
        # Grab current local time
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        # Log the exact timestamp we’re returning
        log_message(f"Current time retrieved: {current_time}", "DEBUG")
        return current_time
    
    # This static method captures the primary monitor's screen using mss, saves it with a timestamp, and returns a JSON string containing the file path and a prompt for the model to describe the screenshot.
    @staticmethod
    def capture_screen_and_annotate():
        """
        Capture the primary monitor’s screen using mss, save it with a timestamp,
        and return a JSON string containing:
          - 'file': the saved file path
          - 'prompt': an instruction for the model to describe the screenshot.

        Usage:
            ```tool_code
            capture_screen_and_annotate()
            ```
        """

        # 1) Build output path

        screen_capture_dir = os.path.join(os.path.dirname(__file__), "captures")
        os.makedirs(screen_capture_dir, exist_ok=True)

        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"screen_{ts}.png"
        path       = os.path.join(screen_capture_dir, filename)

        # 2) Capture with mss
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor
                img     = sct.grab(monitor)
                mss.tools.to_png(img.rgb, img.size, output=path)
            log_message(f"Screen captured and saved to {path}", "SUCCESS")
        except Exception as e:
            log_message(f"Error capturing screen: {e}", "ERROR")
            return json.dumps({"error": str(e)})

        # 3) Return the file path plus a prompt
        return json.dumps({
            "file":   path,
            "prompt": f"Please describe what you see in the screenshot, considering this is a screenshot which is of the computer that you reside on, and activity on the screen may be critical to answering questions, be as verbose as possible and describe any text or images present at '{path}'."
        })


    @staticmethod
    def get_visual_input(
        camera: int | str = 0,
        *,
        query: str | None = None,
        model_tier: str = "primary",
        temperature: float = 0.5
    ) -> str:
        """
        Fetch or capture a frame (HTTP or cv2), then send raw bytes
        to auxiliary_inference so the LLM can see it directly.
        Supports URLs like http://.../camera/default_0 (no .jpg extension).
        """
        import requests, cv2, os, json
        from datetime import datetime

        def _fetch_http(u: str) -> bytes | None:
            try:
                r = requests.get(u, timeout=2); r.raise_for_status()
                return r.content
            except Exception as e:
                log_message(f"[HTTP camera] {u} → {e}", "WARN")
                return None

        def _fetch_cv2(idx: int) -> bytes | None:
            cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name=="nt" else 0)
            if not cam.isOpened(): return None
            ok, frame = cam.read(); cam.release()
            if not ok: return None
            _, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes()

        # 1) Try string camera as URL or shorthand
        image_bytes: bytes | None = None
        if isinstance(camera, str):
            url = camera.strip()
            # handle any rs_* shorthand here...
            if url.lower().startswith(("http://", "https://")):
                image_bytes = _fetch_http(url)

        # 2) If still none and camera is int, try default feeds then cv2
        if image_bytes is None and isinstance(camera, int):
            for u in (
                f"http://127.0.0.1:8080/camera/default_{camera}.jpeg",
                f"http://127.0.0.1:8080/camera/default_{camera}"
            ):
                image_bytes = _fetch_http(u)
                if image_bytes: break
            if image_bytes is None:
                image_bytes = _fetch_cv2(camera)

        # 3) Final fallback for non-HTTP string
        if image_bytes is None and isinstance(camera, str):
            image_bytes = _fetch_http(camera)

        if image_bytes is None:
            log_message("No camera source reachable.", "ERROR")
            return json.dumps({"error":"Camera not reachable"})

        # 4) Build prompt and hand off raw bytes
        prompt = query.strip() if query else "Please describe what you see in this image."
        return Tools.auxiliary_inference(
            prompt,
            temperature=temperature,
            model_tier=model_tier,
            system="You are a highly–skilled visual analyst.",
            images=[image_bytes]
        )


    # This static method captures one frame from the default webcam using OpenCV, saves it with a timestamp, and returns a JSON string containing the file path and a prompt for the model to describe the image.
    @staticmethod
    def capture_webcam_and_annotate(query: str=None):
        """
        Capture the webcam camera using cv2, save it with a timestamp, and then annotate what you see,
        save it with a timestamp, and return a JSON string containing:
          - 'file': the saved file path
          - 'prompt': an instruction for the model to describe the image.

        Usage:
            ```tool_code
            capture_webcam_and_annotate(query="question about image")
            ```
        """

        # 1) Open the default camera
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cam.isOpened():
            log_message("Webcam not accessible via cv2.VideoCapture", "ERROR")
            return json.dumps({"error": "Webcam not accessible."})

        # 2) Grab a frame
        ret, frame = cam.read()
        cam.release()
        if not ret:
            log_message("Failed to read frame from webcam", "ERROR")
            return json.dumps({"error": "Failed to capture frame."})

        # 3) Build output path

        image_capture_dir = os.path.join(os.path.dirname(__file__), "captures")
        os.makedirs(image_capture_dir, exist_ok=True)

        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"webcam_{ts}.png"
        path       = os.path.join(image_capture_dir, filename)

        # 4) Save as PNG
        try:
            cv2.imwrite(path, frame)
            log_message(f"Webcam frame saved to {path}", "SUCCESS")
        except Exception as e:
            log_message(f"Error saving webcam frame: {e}", "ERROR")
            return json.dumps({"error": str(e)})

        # 5) Return the file path plus a prompt
        json.dumps({
            "file":   path,
            "prompt": f"Please describe what you see in the image in great detail, considering the context that this image is coming from a webcam attached to the computer you reside on at '{path}'."
        })

        prompt = (f"{query}', apply those questions to the following image and respond with your analysis of its visual contents '{path}', ")
        final = Tools.auxiliary_inference(prompt, temperature=0.5)

        return final
    

    # This static method converts a user query into a more precise information query related to the content of an image. It uses a secondary agent tool to process the query and image context, returning the refined query.
    @staticmethod
    def convert_query_for_image(query, image_path):
        prompt = (f"Given the user query: '{query}', and the context of the image at '{image_path}', "
                  "convert this query into a more precise information query related to the image content.")
        log_message(f"Converting user query for image using prompt: {prompt}", "PROCESS")
        response = Tools.auxiliary_inference(prompt, temperature=0.5)
        log_message("Image query conversion response: " + response, "SUCCESS")
        return response

    # This static method loads an image from a specified path, checking if the file exists. If the image is found, it returns the absolute path; otherwise, it logs an error and returns an error message.
    @staticmethod
    def load_image(image_path):
        full_path = os.path.abspath(image_path)
        if os.path.isfile(full_path):
            log_message(f"Image found at {full_path}", "SUCCESS")
            return full_path
        else:
            log_message(f"Image file not found: {full_path}", "ERROR")
            return f"Error: Image file not found: {full_path}"
    
    @staticmethod
    def set_sudo_password() -> str:
        """
        Prompt for the sudo password and save it in a .env file as SUDO_PASS.
        """
        from getpass import getpass
        import os

        pwd = getpass("Enter your sudo password: ")
        env_path = os.path.join(os.getcwd(), ".env")

        # Read existing lines, stripping out any old SUDO_PASS entries
        lines = []
        if os.path.isfile(env_path):
            with open(env_path, "r") as f:
                content = f.read()
            # preserve original line endings
            lines = [l for l in content.splitlines(keepends=True) if not l.startswith("SUDO_PASS=")]
            # if the last line doesn’t end with a newline, add one
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"

        # Append the new password entry with its own newline
        lines.append(f"SUDO_PASS={pwd}\n")

        # Write back
        with open(env_path, "w") as f:
            f.writelines(lines)

        return "Sudo password saved to .env"

    @staticmethod
    def _get_sudo_pass() -> str | None:
        """
        Read the SUDO_PASS entry from .env in the current working directory.
        """
        import os
        env_path = os.path.join(os.getcwd(), ".env")
        if not os.path.isfile(env_path):
            return None
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("SUDO_PASS="):
                    return line.split("=", 1)[1].strip()
        return None

    @staticmethod
    def list_networks() -> list[dict[str, object]]:
        """
        List available Wi-Fi networks via sudo nmcli.

        Returns:
            A list of dicts with keys 'ssid', 'signal', 'security'.
        """
        import subprocess
        sudo_pass = Tools._get_sudo_pass()
        cmd = ["sudo", "-S", "nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"]
        proc = subprocess.run(
            cmd,
            input=(sudo_pass + "\n") if sudo_pass else None,
            capture_output=True,
            text=True
        )
        nets = []
        for line in proc.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0]:
                ssid, sig, sec = parts[0], parts[1], parts[2]
                try:
                    signal = int(sig)
                except ValueError:
                    signal = 0
                nets.append({"ssid": ssid, "signal": signal, "security": sec})
        return nets

    @staticmethod
    def connect_network(ssid: str, password: str | None = None, timeout: int = 20) -> bool:
        """
        Connect to a Wi-Fi network via sudo nmcli.

        Args:
            ssid:     The network SSID.
            password: WPA/WPA2 passphrase (None for open networks).
            timeout:  Seconds to wait for connection.

        Returns:
            True if connected successfully, False otherwise.
        """
        import subprocess, time
        sudo_pass = Tools._get_sudo_pass()
        cmd = ["sudo", "-S", "nmcli", "device", "wifi", "connect", ssid]
        if password:
            cmd += ["password", password]
        proc = subprocess.run(
            cmd,
            input=(sudo_pass + "\n") if sudo_pass else None,
            capture_output=True,
            text=True
        )
        if proc.returncode != 0:
            return False

        # Wait until connected
        start = time.time()
        while time.time() - start < timeout:
            status = subprocess.run(
                ["sudo", "-S", "nmcli", "-t", "-f", "DEVICE,STATE", "device"],
                input=(sudo_pass + "\n") if sudo_pass else None,
                capture_output=True,
                text=True
            )
            for line in status.stdout.splitlines():
                dev, state = line.split(":", 1)
                if state == "connected":
                    return True
            time.sleep(1)
        return False

    @staticmethod
    def check_connectivity(host: str = "8.8.8.8", count: int = 1) -> bool:
        """
        Test Internet connectivity by pinging a host.
        """
        import subprocess
        proc = subprocess.run(
            ["ping", "-c", str(count), host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return proc.returncode == 0

    @staticmethod
    def get_signal_strength(ssid: str | None = None) -> int:
        """
        Get the signal strength (%) of the specified or active SSID.
        """
        import subprocess
        target = ssid
        if not target:
            try:
                out = subprocess.check_output(
                    ["nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi", "list"],
                    text=True
                )
                for line in out.splitlines():
                    active, name = line.split(":", 1)
                    if active == "yes":
                        target = name
                        break
            except Exception:
                return -1
        if not target:
            return -1

        for net in Tools.list_networks():
            if net["ssid"] == target:
                return net["signal"]
        return -1

    # This static method retrieves the battery voltage from a file named "voltage.txt" located in the user's home directory. It reads the voltage value, logs the action, and returns the voltage as a float. If an error occurs while reading the file, it logs the error and raises a RuntimeError.
    def get_battery_voltage():
        """
        1. Try to read ~/voltage.txt and return its float value.
        2. If that fails (file missing or parse error), fall back to psutil.sensors_battery():
           - percent: battery.percent
           - power_plugged: battery.power_plugged
           - time_left: formatted hh:mm:ss from battery.secsleft
        Raises RuntimeError on any unexpected failure.
        """
        def convert_time(seconds: float) -> str:
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"

        try:
            home_dir = os.path.expanduser("~")
            file_path = os.path.join(home_dir, "voltage.txt")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    line = f.readline().strip()
                voltage = float(line)
                log_message("Battery voltage retrieved from voltage.txt.", "SUCCESS")
                return voltage

            # Fallback to system battery info
            battery = psutil.sensors_battery()
            if battery is None:
                raise RuntimeError("No system battery information available.")
            percent      = battery.percent
            power_plugged = battery.power_plugged
            secsleft     = battery.secsleft
            time_left    = convert_time(secsleft)

            log_message("Battery info retrieved via system sensors.", "SUCCESS")
            return {
                "percent": percent,
                "power_plugged": power_plugged,
                "time_left": time_left
            }

        except Exception as e:
            msg = f"Error reading battery voltage: {e}"
            log_message(msg, "ERROR")
            raise RuntimeError(msg)

    

    # This static method summarizes a web search by calling the search_internet method to get the top_n results for a given topic. It scrapes each URL for content, asks a secondary agent tool for a summary, and returns a formatted bullet list of the results.
    @staticmethod
    def summarize_search(topic: str, top_n: int = 3) -> str:
        """
        Summarize web pages for a search topic.

        1. Call summarize_search(topic=str, top_n=int)
        - topic (str): the search term
        - top_n (int): how many results

        2. For each page returned (dict with "url", "title", "content"):
        a. Truncate content to 2000 chars, replace newlines.
        b. Calls auxiliary_inference(prompt: str, temperature: float):
            - prompt: "Here is the content of {url}: ... Please give me a 2–3 sentence summary."
            - temperature: 0.3

        Returns a numbered list of summaries, or an error/no-results message.
        """
        try:
            pages = Tools.search_internet(
                topic=topic,
                top_n=top_n,
                deep_scrape=True
            )
        except Exception as e:
            return f"Error retrieving search pages: {e}"

        if not pages:
            return "No results found."

        summaries = []
        for i, page in enumerate(pages, start=1):
            url     = page.get("url", "")
            title   = page.get("title", url)
            content = page.get("content", "")
            snippet = content[:2000].replace("\n", " ")

            prompt = (
                f"Here is the content of {url}:\n\n"
                f"{snippet}\n\n"
                "Please give me a 2–3 sentence summary of the key points."
            )

            try:
                summary = Tools.auxiliary_inference(prompt, temperature=0.3).strip()
            except Exception:
                summary = "Failed to summarise that page."

            summaries.append(f"{i}. {title} — {summary}")

        return "\n".join(summaries)



    # This static method scrapes a webpage using BeautifulSoup and requests. It fetches the content of the URL, parses it with BeautifulSoup, and returns the prettified HTML. If an error occurs during scraping, it logs the error and returns an error message.
    @staticmethod
    def bs4_scrape(url):
        headers = {
            'User-Agent': ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/42.0.2311.135 Safari/537.36 Edge/12.246")
        }
        try:
            import requests
            log_message(f"Scraping URL: {url}", "PROCESS")
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html5lib')
            log_message("Webpage scraped successfully.", "SUCCESS")
            return soup.prettify()
        except Exception as e:
            log_message("Error during scraping: " + str(e), "ERROR")
            return f"Error during scraping: {e}"

    # This static method finds a file by name in a specified search path. It walks through the directory tree, checking each file against the given filename. If the file is found, it logs the success and returns the directory path; otherwise, it logs a warning and returns None.
    @staticmethod
    def find_file(filename, search_path="."):
        log_message(f"Searching for file: {filename} in path: {search_path}", "PROCESS")
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                log_message(f"File found in directory: {root}", "SUCCESS")
                return root
        log_message("File not found.", "WARNING")
        return None

    # This static method retrieves the current location based on the public IP address of the machine. It uses the ip-api.com service to get location data, logs the action, and returns the JSON response. If an error occurs during the request, it logs the error and returns an error message.
    @staticmethod
    def get_current_location():
        """
        Resolve the users current location from a json object that also gets IP address and internet provider.
        """
        try:
            import requests
            log_message("Retrieving current location based on IP.", "PROCESS")
            response = requests.get("http://ip-api.com/json", timeout=5)
            if response.status_code == 200:
                log_message("Current location retrieved.", "SUCCESS")
                return response.json()
            else:
                log_message("Error retrieving location: HTTP " + str(response.status_code), "ERROR")
                return {"error": f"HTTP error {response.status_code}"}
        except Exception as e:
            log_message("Error retrieving location: " + str(e), "ERROR")
            return {"error": str(e)}

    # This static method retrieves the current system utilization metrics such as CPU usage, memory usage, and disk usage. It uses the psutil library to gather these metrics, logs the action, and returns a dictionary containing the utilization percentages.
    @staticmethod
    def get_system_utilization():
        utilization = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        log_message("System utilization retrieved.", "DEBUG")
        return utilization


    @staticmethod
    def auxiliary_inference(
        prompt: str,
        *,
        temperature: float = 0.7,
        system: str | None = None,
        context: object | None = None,
        retrieval_count: int = 0,
        model_tier: str | None = None,
        images: list[bytes] | None = None
    ) -> str:
        """
        Invoke an LLM with a prompt, optional system instructions, and an
        explicit *narrative/context injection* mechanism:

        - `context`: supply a high-level narrative object (e.g. the output of
            `_load_narrative_context()`) so the model can ground its response
            in the overall story arc.
        - `retrieval_count`: indicate how many of the most recent context snippets
            the LLM should consider.  If >0, callers are encouraged to prepend
            a block like:
                “Here are the last {retrieval_count} context snippets: …”
            to the `prompt` (or via `context=`) so that the model can weave them
            into its auxiliary inference.
        - `model_tier`: choose among "primary", "secondary", or "decision" tiers.
        - `images`: raw bytes to embed directly in the final user message;
            falls back to scanning `prompt` for image URLs or file paths.

        This helper is designed both for programmatic use *and* as a tool that
        an LLM can call when it wants to “pull in” narrative context and recent
        retrievals before performing its reasoning.

        Returns the raw string output from the chosen LLM.
        """
        import re, json, requests

        # 1) choose model
        tier_map = {
            "primary":   config.get("primary_model"),
            "secondary": config.get("secondary_model", config.get("primary_model")),
            "decision":  config.get("decision_model",
                        config.get("secondary_model", config.get("primary_model")))
        }
        model_selected = tier_map.get((model_tier or "primary").lower(),
                                    config.get("primary_model"))

        # 2) assemble messages
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})

        # if context is provided, treat it as a narrative block
        if context is not None:
            messages.append({
                "role": "system",
                "content": (
                    f"### Narrative Context (last {retrieval_count} snippets) ###\n"
                    f"{context}"
                )
            })

        messages.append({"role": "user", "content": prompt})

        # 3) load images if none explicitly provided
        images_data: list[bytes] = []
        if images:
            images_data = images
        else:
            # scan for HTTP or absolute local paths ending in image exts
            pattern = r"((?:https?://\S+?\.(?:jpg|jpeg|png|bmp|gif))|(?:/\S+?\.(?:jpg|jpeg|png|bmp|gif)))"
            for loc in re.findall(pattern, prompt, flags=re.IGNORECASE):
                try:
                    if loc.lower().startswith(("http://", "https://")):
                        resp = requests.get(loc, timeout=5)
                        resp.raise_for_status()
                        images_data.append(resp.content)
                    else:
                        with open(loc, "rb") as f:
                            images_data.append(f.read())
                except Exception:
                    continue

        # 4) inject images into the last message if present
        if images_data:
            messages[-1]["images"] = images_data

        # 5) stream via chat()
        try:
            log_message(
                f"auxiliary_inference(model={model_selected}, "
                f"temp={temperature}, tier={model_tier}, "
                f"retrievals={retrieval_count}, images={len(images_data)})",
                "PROCESS"
            )
            content = ""
            print("⟳ Auxiliary-LLM stream:", end="", flush=True)
            for part in chat(
                model=model_selected,
                messages=messages,
                stream=True,
                options={"temperature": temperature},
            ):
                tok = part["message"]["content"]
                content += tok
                print(tok, end="", flush=True)
            print()
            log_message("auxiliary_inference complete.", "SUCCESS")
            return content

        except Exception as e:
            log_message(f"auxiliary_inference error: {e}", "ERROR")
            return json.dumps({"error": str(e)})
        
    @staticmethod
    def generate_tool_schema(tool_name: str) -> Dict[str, Any]:
        """
        Inspect Tools.<tool_name> and generate/store its JSON-schema.
        """
        fn = getattr(Tools, tool_name, None)
        if not callable(fn):
            raise KeyError(f"Unknown tool '{tool_name}'")
        schema = _create_tool_schema(fn)

        # --- prune properties that aren't true keyword args ---
        sig = signature(fn)
        # keep only those parameters that have a default (i.e. real kwargs)
        allowed = {
            name
            for name, param in sig.parameters.items()
            if param.default is not _empty
        }

        props = schema["parameters"]["properties"]
        # filter out anything not in allowed
        schema["parameters"]["properties"] = {
            k: v for k, v in props.items() if k in allowed
        }

        # likewise drop from `required` if it slipped in
        if "required" in schema["parameters"]:
            schema["parameters"]["required"] = [
                p for p in schema["parameters"]["required"]
                if p in allowed
            ]

        TOOL_SCHEMAS[tool_name] = schema
        return schema

    @staticmethod
    def generate_all_tool_schemas() -> None:
        """
        Walk all public callables on Tools and populate TOOL_SCHEMAS.
        """
        for name, fn in inspect.getmembers(Tools, predicate=callable):
            if name.startswith("_"):
                continue
            try:
                Tools.generate_tool_schema(name)
            except KeyError:
                # skip non-tool callables
                continue

    @staticmethod
    def get_tool_schema(tool_name: str) -> Dict[str, Any]:
        """
        Return the JSON-schema for the given tool, generating it on demand.
        """
        if tool_name not in TOOL_SCHEMAS:
            return Tools.generate_tool_schema(tool_name)
        return TOOL_SCHEMAS[tool_name]
    
    @staticmethod
    def get_known_users() -> list[dict]:
        """
        Returns a list of all users we’ve ever seen, each as {'username': str, 'id': int}.
        """
        return _REG.list_all()

    @staticmethod
    def message_user(username: str, message: str) -> str:
        """
        Send a DM to a known user.
        
        Parameters:
        - username: their @username (with or without the leading '@')
        - message: the text to deliver
        
        Returns:
        - "OK" on success
        - "ERROR: <reason>" if something goes wrong
        """
        uname = username.lstrip("@").lower()
        user_id = _REG.id_for(uname)
        if user_id is None:
            return f"ERROR: user @{uname} not found"
        token = os.getenv("BOT_TOKEN")
        if not token:
            return "ERROR: BOT_TOKEN missing"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": user_id, "text": message}
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return "OK"
            else:
                return f"ERROR: HTTP {resp.status_code} – {resp.text}"
        except Exception as e:
            return f"ERROR: {e}"

    @staticmethod
    def get_known_groups() -> list[dict]:
        """
        Returns a list of all registered groups.
        Each entry is:
          { "name": <group_name>, "chat_id": <id> }
        """
        try:
            from group_registry import _GREG
            return _GREG.list_all()
        except Exception as e:
            log_message(f"get_known_groups error: {e}", "ERROR")
            return []

    @staticmethod
    def message_group(group_name: str, message: str) -> dict:
        """
        Send a message into a previously-registered group.

        Parameters
        ----------
        group_name : str
            The exact group name as recorded (via title) in group_registry.
        message : str
            The text to send into that group.

        Returns
        -------
        dict
            { "group_name": <name>, "chat_id": <id>, "status": "sent" }

        Raises
        ------
        ValueError
            If the group_name is not found in the registry.
        RuntimeError
            If BOT_TOKEN is not set.
        """
        import os, requests
        token = os.getenv("BOT_TOKEN")
        if not token:
            raise RuntimeError("BOT_TOKEN not set")
        from group_registry import _GREG

        gid = _GREG.id_for(group_name)
        if gid is None:
            raise ValueError(f"Group '{group_name}' not found")

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": gid, "text": message})
        resp.raise_for_status()

        return {"group_name": group_name, "chat_id": gid, "status": "sent"}

    @staticmethod
    def skip_tools():
        """For simple replies to queries that dont fall under other tools, just skip tools altogether! Be sure to use real tools if the query relates to others, use this as a fallback!"""
        return
    
Tools.generate_all_tool_schemas()
