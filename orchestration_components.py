# orchestration_components.py
# Strategy Graph runtime: ToolRegistry, PolicyManager, Planner, Critic, Executor
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable, Tuple
import time
import json
import re
import traceback
import inspect

try:
    import numpy as np
except Exception:
    np = None

# Expect your project provides these:
try:
    from context import ContextObject
except Exception:
    # Minimal fallback (dev only)
    class ContextObject:
        def __init__(self):
            self.context_id = f"ctx_{int(time.time()*1000)}"
            self.component = ""
            self.semantic_label = ""
            self.stage_id = ""
            self.summary = ""
            self.metadata = {}
            self.tags = []
            self.references = []

        @classmethod
        def make_stage(cls, stage_id, input_refs, output):
            co = cls()
            co.stage_id = stage_id
            co.semantic_label = stage_id
            co.component = stage_id
            co.references = input_refs or []
            co.metadata = output if isinstance(output, dict) else {"output": output}
            co.summary = (json.dumps(output, ensure_ascii=False)[:300] if isinstance(output, dict) else str(output))[:300]
            return co

        @classmethod
        def make_knowledge(cls, stage_id, payload, tags=None):
            co = cls()
            co.stage_id = stage_id
            co.semantic_label = stage_id
            co.component = "knowledge"
            co.metadata = payload if isinstance(payload, dict) else {"payload": payload}
            if tags: co.tags = list(tags)
            co.summary = (json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload))[:300]
            return co

        def touch(self): pass

# Optional: bind Python handlers from your tools module
try:
    import tools as _tools_mod
except Exception:
    _tools_mod = None


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _log_safe(logger: Optional[Callable[[str,str],None]], msg: str, cat: str="INFO"):
    try:
        (logger or (lambda m,c=None: print(f"[{c or 'INFO'}] {m}")))(msg, cat)
    except Exception:
        print(f"[{cat}] {msg}")

def _strip_fences(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[1]
        # strip trailing ```
        if "```" in s:
            s = s.rsplit("```", 1)[0]
    return s.strip()

def _jsonish(x) -> Optional[dict]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        raw = _strip_fences(x)
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None

def _coerce_parameters(blob: dict) -> dict:
    """
    Normalize any of:
      {'schema': {'parameters': {...}}}
      {'parameters': {...}}
      {'json_schema': {...}}
    into {'type':'object','properties':{...}, 'required': [...]}
    """
    if not isinstance(blob, dict):
        return {"type":"object","properties":{}, "required":[]}
    # direct parameters
    if "schema" in blob and isinstance(blob["schema"], dict) and "parameters" in blob["schema"]:
        p = blob["schema"]["parameters"]
        if isinstance(p, dict):
            # assume already parameters shape
            return {"type":"object", **({"properties": p.get("properties", {})} if "properties" in p else {"properties": p}),
                    "required": p.get("required", [])}
    if "parameters" in blob and isinstance(blob["parameters"], dict):
        p = blob["parameters"]
        return {"type":"object", **({"properties": p.get("properties", {})} if "properties" in p else {"properties": p}),
                "required": p.get("required", [])}
    if "json_schema" in blob and isinstance(blob["json_schema"], dict):
        p = blob["json_schema"]
        # if already looks like parameters, keep
        if "properties" in p:
            return {"type": "object", "properties": p.get("properties", {}), "required": p.get("required", [])}
        # else wrap
        return {"type": "object", "properties": p, "required": []}
    # final fallback
    return {"type":"object","properties":{}, "required":[]}


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────
class ToolRegistry:
    """
    Schema-grounded catalogue of tools.

    **SOURCE OF TRUTH = repo JSONL** rows. We support all prior shapes:
      • component=='schema' + tag 'tool_schema' (preferred)
      • component in {'schema','tool','policy'} AND
          ('tool_schema' in tags  OR semantic_label in {'tool_schema','schema:tool'}
           OR stage_id startswith 'tool_schema:')

    Each row's metadata may carry any of:
      - name / tool_name
      - description
      - schema.parameters / parameters / json_schema
    """
    def __init__(self, repo=None, logger: Optional[Callable[[str,str],None]]=None):
        self.repo = repo
        self._log = logger
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._last_refresh_count = -1
        self.refresh_from_repo(bind_handlers=True)

    # Logging helpers
    def log(self, msg, cat="INFO"):
        _log_safe(self._log, msg, cat)

    # Public API --------------------------------------------------------------
    def list(self) -> List[Dict[str, Any]]:
        # Lazy refresh if empty (or repo grew)
        if not self._schemas:
            self.refresh_from_repo(bind_handlers=True)
        return [dict(v) for v in self._schemas.values()]

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._schemas.get(name)

    def register_dict(self, tool: Dict[str, Any]):
        name = tool.get("name")
        if not name: raise ValueError("tool dict requires 'name'")
        if "schema" not in tool or "parameters" not in tool["schema"]:
            raise ValueError("tool dict requires 'schema' with 'parameters'")
        self._schemas[name] = tool

    def register_fn(self, name: str, description: str, parameters: Dict[str, Any],
                    handler: Callable[[Dict[str, Any]], Any]):
        """Directly register a Python callable as a tool with a JSON schema."""
        if not name or not callable(handler):
            raise ValueError("register_fn requires name and callable handler")
        self._schemas[name] = {"name": name, "description": description,
                               "schema": {"parameters": parameters}}
        self._handlers[name] = handler

    def refresh_from_repo(self, bind_handlers: bool = False) -> int:
        """
        Pull tools from repo JSONL with wide compatibility,
        then (optionally) auto-bind Python handlers from `tools` module.
        Returns number of schemas loaded.
        """
        if not self.repo:
            self.log("ToolRegistry.refresh_from_repo: no repo bound", "WARNING")
            return 0

        loaded: Dict[str, Dict[str, Any]] = {}
        try:
            rows = self.repo.query(lambda c:
                (c.component in ("schema","tool","policy"))
                and (
                    (c.tags and "tool_schema" in c.tags) or
                    (c.semantic_label in ("tool_schema","schema:tool")) or
                    (isinstance(c.stage_id, str) and c.stage_id.startswith("tool_schema"))
                )
            )
        except Exception as e:
            self.log(f"ToolRegistry.refresh_from_repo: repo.query failed: {e}", "WARNING")
            rows = []

        for c in rows:
            meta = c.metadata or {}
            # Some people store JSON in summary; try harvesting it too.
            meta_from_summary = _jsonish(c.summary)
            if isinstance(meta_from_summary, dict):
                # Merge summary JSON over metadata if it looks toolish
                if "name" in meta_from_summary or "tool_name" in meta_from_summary:
                    meta = {**meta, **meta_from_summary}

            name = (meta.get("name")
                    or meta.get("tool_name")
                    or (c.stage_id.split(":",1)[1] if isinstance(c.stage_id,str) and ":" in c.stage_id else None)
                    or (c.semantic_label if c.semantic_label and c.semantic_label not in ("tool_schema","schema:tool") else None))
            if not name:
                # try parsing 'summary' for `"name": "..."` if metadata was empty
                if isinstance(c.summary, str):
                    m = re.search(r'"name"\s*:\s*"([^"]+)"', c.summary)
                    if m: name = m.group(1)
            if not name:
                continue

            desc = meta.get("description") or (c.summary if isinstance(c.summary, str) else "")
            params = _coerce_parameters(meta)
            loaded[name] = {
                "name": name,
                "description": desc or "",
                "schema": {"parameters": params}
            }

        self._schemas = loaded
        count = len(self._schemas)
        if count != self._last_refresh_count:
            self.log(f"ToolRegistry: loaded {count} tool schemas from repo", "INFO")
            if count:
                self.log("ToolRegistry: " + ", ".join(sorted(self._schemas.keys()))[:300], "DEBUG")
            self._last_refresh_count = count

        if bind_handlers:
            self._bind_handlers_best_effort()

        return count

    def _bind_handlers_best_effort(self):
        """
        Auto-bind Python handlers for tools whose name matches a callable on:
          • tools.Tools.<name>(**kwargs)
          • tools.<name>(**kwargs)
        Also supports injecting repo if the method is bound to an instance that
        expects it (Assembler already sets tools.repo / tools.Tools.repo).
        """
        if not self._schemas:
            return
        if not _tools_mod:
            self.log("ToolRegistry: tools module not importable; skipping handler bind", "WARNING")
            return

        # Build a single instance of tools.Tools if present
        tools_cls = getattr(_tools_mod, "Tools", None)
        tools_obj = None
        if inspect.isclass(tools_cls):
            try:
                # pass repo if constructor supports it
                sig = inspect.signature(tools_cls)
                if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.name in ("repo","context_repo")
                       for p in sig.parameters.values()):
                    tools_obj = tools_cls(repo=getattr(self, "repo", None))
                else:
                    tools_obj = tools_cls()
            except Exception:
                tools_obj = tools_cls() if callable(tools_cls) else None
            # make sure global module sees the repo too (legacy pattern)
            try:
                _tools_mod.repo = getattr(self, "repo", None)
                tools_cls.repo = getattr(self, "repo", None)
            except Exception:
                pass

        for name in list(self._schemas.keys()):
            if name in self._handlers:
                continue
            handler = None

            # Prefer Tools.<name>
            if tools_obj and hasattr(tools_obj, name) and callable(getattr(tools_obj, name)):
                fn = getattr(tools_obj, name)
                handler = lambda params, _fn=fn: _fn(**(params or {}))

            # Fallback: module-level function
            elif hasattr(_tools_mod, name) and callable(getattr(_tools_mod, name)):
                fn = getattr(_tools_mod, name)
                handler = lambda params, _fn=fn: _fn(**(params or {}))

            if handler:
                # If the schema lacks parameters, create an empty object schema
                sch = self._schemas.get(name) or {}
                params = ((sch.get("schema") or {}).get("parameters") or None)
                if not params:
                    self._schemas[name]["schema"] = {"parameters": {"type":"object","properties":{}}}
                self._handlers[name] = handler

        self.log(f"ToolRegistry: bound {len(self._handlers)} Python handlers", "INFO")

    # Execution ---------------------------------------------------------------
    def _validate_params(self, name: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        sch = self._schemas.get(name)
        if not sch: return False, [f"Unknown tool: {name}"]
        p = (sch.get("schema") or {}).get("parameters") or {}
        required = p.get("required", []) or []
        props = p.get("properties", {}) or {}
        errs = []
        for r in required:
            if r not in params:
                errs.append(f"Missing required param '{r}'")
        # Best-effort type checks
        for k, spec in props.items():
            if k not in params: continue
            typ = (spec or {}).get("type")
            if not typ: continue
            v = params[k]
            if typ == "number" and not isinstance(v, (int, float)): errs.append(f"Param '{k}' should be number")
            if typ == "integer" and not isinstance(v, int): errs.append(f"Param '{k}' should be integer")
            if typ == "string" and not isinstance(v, str): errs.append(f"Param '{k}' should be string")
            if typ == "boolean" and not isinstance(v, bool): errs.append(f"Param '{k}' should be boolean")
            if typ == "array" and not isinstance(v, list): errs.append(f"Param '{k}' should be array")
            if typ == "object" and not isinstance(v, dict): errs.append(f"Param '{k}' should be object")
        # Drop unknown keys silently (planner sometimes adds notes)
        for k in list(params.keys()):
            if props and k not in props:
                params.pop(k, None)
        return (len(errs) == 0, errs)

    def execute(self, name: str, params: Dict[str, Any], assembler=None) -> ContextObject:
        ok, errs = self._validate_params(name, dict(params or {}))
        if not ok:
            out = {"error": "validation_failed", "details": errs, "tool": name, "params": params}
            ctx = ContextObject.make_stage(f"tool:{name}", [], out)
            ctx.component = "tool_output"; ctx.semantic_label = "tool_output"; ctx.tags = ["tool_output", name]
            if assembler:
                try: assembler.repo.save(ctx)
                except Exception: pass
            return ctx

        # Execute Python handler if bound
        if name in self._handlers:
            try:
                result = self._handlers[name](params or {})
                out = {"output": result, "tool": name, "params": params}
            except Exception as e:
                out = {"error": "handler_exception", "message": str(e), "tool": name, "params": params}
        else:
            out = {"error": "no_handler", "message": "No bound Python handler for this tool.", "tool": name, "params": params}

        ctx = ContextObject.make_stage(f"tool:{name}", [], out)
        ctx.component = "tool_output"; ctx.semantic_label = "tool_output"; ctx.tags = ["tool_output", name]
        if assembler:
            try: assembler.repo.save(ctx)
            except Exception: pass
        return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Policy Manager
# ─────────────────────────────────────────────────────────────────────────────
class PolicyManager:
    """Heuristics for plan count & scoring."""
    def __init__(self, logger: Optional[Callable[[str,str],None]]=None):
        self._log = logger

    def num_candidate_plans(self, context: Dict[str, Any]) -> int:
        tools = context.get("tools") or []
        if len(tools) >= 4: return 4
        if len(tools) >= 2: return 3
        return 2

    def score_plan(self, graph: Dict[str, Any], context: Dict[str, Any]) -> float:
        nodes = graph.get("nodes", [])
        n = len(nodes)
        return max(0.0, 1.5 - 0.12*n)

    def decision_priors(self, user_text: str, options: List[str]) -> Dict[str, float]:
        text = (user_text or "").lower()
        priors = {opt: 0.5 for opt in options}
        if any(k in text for k in ("search", "look up", "fetch", "news", "price", "weather", "location")):
            priors = {opt: (0.8 if "TOOL" in opt or opt == "YES" else 0.2) for opt in options}
        if any(k in text for k in ("just talk", "chat", "chit chat")):
            priors = {opt: (0.8 if opt in ("NO_TOOLS","NO") else 0.2) for opt in options}
        return priors


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Planner (multi-plan proposal)
# ─────────────────────────────────────────────────────────────────────────────
class HeuristicPlanner:
    """
    Builds a small set of candidate DAGs from context.
    Graph format:
      {
        "nodes": [
          {"id":"t1","type":"tool","name":"search","params":{"q":"foo"},"depends_on":[]},
          {"id":"s1","type":"llm","name":"LLM_SUMMARIZE","params":{"sources":["t1"]},"depends_on":["t1"]}
        ],
        "meta": {"goal":"answer_user"}
      }
    """
    def __init__(self, embed_fn: Optional[Callable[[str], Any]]=None,
                 tool_registry: Optional[ToolRegistry]=None,
                 logger: Optional[Callable[[str,str],None]]=None):
        self.embed_fn = embed_fn
        self.tr = tool_registry
        self._log = logger

    def _keywords(self, context: Dict[str,Any]) -> List[str]:
        clar = context.get("clar")
        kws: List[str] = []
        if clar and isinstance(getattr(clar, "metadata", None), dict):
            kws = clar.metadata.get("keywords") or []
        if not kws:
            text = (context.get("user_text") or "").strip()
            kws = re.findall(r"[A-Za-z][\w-]{2,}", text)[:8]
        return kws

    def _rank_tools(self, tools: List[Dict[str,Any]], kws: List[str]) -> List[str]:
        if not tools: return []
        scores = []
        lower_kws = [k.lower() for k in kws]
        for t in tools:
            name = (t.get("name") or "").lower()
            desc = (t.get("description") or "").lower()
            text = f"{name} {desc}"
            score = sum(1 for k in lower_kws if k in text)
            scores.append((score, t.get("name")))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [n for s, n in scores if n]

    def _guess_params(self, tool_schema: Dict[str,Any], kws: List[str]) -> Dict[str,Any]:
        """Cheap arg seeding based on common names; relies on schema types."""
        params: Dict[str,Any] = {}
        p = (tool_schema.get("schema") or {}).get("parameters") or {}
        props = p.get("properties", {}) or {}
        # convenience mapping
        textish_keys = [k for k,v in props.items() if (v or {}).get("type") == "string"]
        arrayish_keys = [k for k,v in props.items() if (v or {}).get("type") == "array"]
        joined = " ".join(kws).strip()
        if joined:
            for key in ("query","q","text","keyword","prompt","location","where"):
                if key in props and key in textish_keys and key not in params:
                    params[key] = joined
                    break
            if not params and textish_keys:
                params[textish_keys[0]] = joined
        if arrayish_keys and kws:
            params[arrayish_keys[0]] = kws[:3]
        return params

    def propose(self, context: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
        tools = context.get("tools") or (self.tr.list() if self.tr else [])
        kws = self._keywords(context)
        ranked = self._rank_tools(tools, kws)

        graphs: List[Dict[str,Any]] = []

        # Plan A: Text-only (fast path)
        graphs.append({
            "nodes": [
                {"id":"draft", "type":"llm", "name":"LLM_DRAFT_ONLY", "params":{"style":"concise"}, "depends_on":[]}
            ],
            "meta": {"goal":"answer_user", "strategy":"text_only"}
        })

        # Plan B: Top-1 tool then summarize
        if ranked:
            t1_name = ranked[0]
            t1_schema = next((t for t in tools if t.get("name")==t1_name), None) or {}
            t1_params = self._guess_params(t1_schema, kws)
            graphs.append({
                "nodes": [
                    {"id":"t1", "type":"tool", "name":t1_name, "params":t1_params, "depends_on":[]},
                    {"id":"s1", "type":"llm", "name":"LLM_SUMMARIZE", "params":{"sources":["t1"]}, "depends_on":["t1"]},
                ],
                "meta": {"goal":"answer_user", "strategy":"single_tool_then_summarize"}
            })

        # Plan C: Top-2 in parallel then summarize
        if len(ranked) >= 2:
            t1, t2 = ranked[:2]
            t1s = next((t for t in tools if t.get("name")==t1), None) or {}
            t2s = next((t for t in tools if t.get("name")==t2), None) or {}
            graphs.append({
                "nodes":[
                    {"id":"t1","type":"tool","name":t1,"params":self._guess_params(t1s, kws), "depends_on":[]},
                    {"id":"t2","type":"tool","name":t2,"params":self._guess_params(t2s, kws), "depends_on":[]},
                    {"id":"s1","type":"llm","name":"LLM_SUMMARIZE","params":{"sources":["t1","t2"]},"depends_on":["t1","t2"]},
                ],
                "meta":{"goal":"answer_user","strategy":"parallel_two_then_summarize"}
            })

        return graphs[:max(1, int(k))]


# ─────────────────────────────────────────────────────────────────────────────
# Critic (preflight + simple post checks)
# ─────────────────────────────────────────────────────────────────────────────
class SafetyCritic:
    """Validates plan graphs before execution and returns fixable issues."""
    def __init__(self, tool_registry: Optional[ToolRegistry]=None,
                 logger: Optional[Callable[[str,str],None]]=None):
        self.tr = tool_registry
        self._log = logger

    def _has_cycle(self, nodes: List[Dict[str,Any]]) -> bool:
        graph = {n["id"]: set(n.get("depends_on") or []) for n in nodes}
        seen, stack = set(), set()
        def dfs(u):
            seen.add(u); stack.add(u)
            for v in graph.get(u, ()):
                if v not in seen:
                    if dfs(v): return True
                elif v in stack:
                    return True
            stack.remove(u); return False
        return any(dfs(n["id"]) for n in nodes if n["id"] not in seen)

    def preflight(self, graph: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        issues: List[str] = []
        nodes = graph.get("nodes", [])
        ids = set()
        for n in nodes:
            nid = n.get("id")
            if not nid: issues.append("Node missing 'id'")
            elif nid in ids: issues.append(f"Duplicate node id '{nid}'")
            else: ids.add(nid)
            ntype = n.get("type")
            if ntype not in ("tool","llm"): issues.append(f"Unsupported node type '{ntype}'")
            if ntype == "tool":
                name = n.get("name")
                if not name or not (self.tr and self.tr.get(name)):
                    issues.append(f"Unknown tool '{name}'")
        if self._has_cycle(nodes):
            issues.append("Dependency cycle detected")
        if len(nodes) > 12:
            issues.append("Plan too large (>12 nodes)")
        return issues


# ─────────────────────────────────────────────────────────────────────────────
# DAG Executor (sequential topo; repair-light)
# ─────────────────────────────────────────────────────────────────────────────
class DAGExecutor:
    """
    Executes a plan DAG. Sequential in topological order (parallel-ready later).
    Node types:
      • tool: executed via ToolRegistry
      • llm : calls back into Assembler for composing draft/summaries
    Returns dict: {"reply": str, "tool_ctxs": [ContextObject], "graph": graph, "repairs": int, "issues": [..]}
    """
    def __init__(self, assembler, tool_registry: ToolRegistry, critic: SafetyCritic,
                 logger: Optional[Callable[[str,str],None]]=None):
        self.asm = assembler
        self.tr = tool_registry
        self.critic = critic
        self._log = logger

    def log(self, msg, cat="INFO"):
        _log_safe(self._log, msg, cat)

    def _toposort(self, nodes: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        deps = {n["id"]: set(n.get("depends_on") or []) for n in nodes}
        ready = [n for n in nodes if not deps[n["id"]]]
        order = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for m in nodes:
                if node["id"] in deps.get(m["id"], set()):
                    deps[m["id"]].remove(node["id"])
                    if not deps[m["id"]] and m not in order and m not in ready:
                        ready.append(m)
        if len(order) != len(nodes):
            return nodes
        return order

    def _collect_sources_text(self, source_ids: List[str], results: Dict[str, Any]) -> str:
        chunks = []
        for sid in source_ids or []:
            r = results.get(sid)
            if isinstance(r, ContextObject):
                md = r.metadata or {}
                val = md.get("output") or md.get("exception") or md.get("message") or md
                try:
                    text = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
                except Exception:
                    text = str(val)
                chunks.append(text)
            else:
                chunks.append(str(r))
        return "\n".join(chunks)[:4000]

    def run(self, graph: Dict[str, Any], context: Dict[str, Any],
            critic: Optional[SafetyCritic]=None,
            tool_registry: Optional[ToolRegistry]=None) -> Dict[str, Any]:
        critic = critic or self.critic
        tr = tool_registry or self.tr

        # Ensure we have fresh tools (repo may have been updated)
        try:
            tr.refresh_from_repo(bind_handlers=True)
        except Exception:
            pass

        issues = critic.preflight(graph, context) if critic else []
        if issues:
            return {"reply":"", "tool_ctxs":[], "graph":graph, "issues":issues, "repairs":0}

        nodes = graph.get("nodes", [])
        order = self._toposort(nodes)
        results: Dict[str, Any] = {}
        tool_ctxs: List[ContextObject] = []
        repairs = 0

        budgets = context.get("budgets") or {}
        max_nodes = budgets.get("max_nodes", 16)
        if len(order) > max_nodes:
            return {"reply":"", "tool_ctxs":[], "graph":graph, "issues":["budget:max_nodes"], "repairs":0}

        for node in order:
            ntype = node.get("type")
            nid = node.get("id")
            name = node.get("name")
            params = dict(node.get("params") or {})

            # cheap param repair: fill defaults if schema has them
            if ntype == "tool" and tr:
                sch = tr.get(name) or {}
                pdef = ((sch.get("schema") or {}).get("parameters") or {}).get("properties", {}) or {}
                for k, spec in pdef.items():
                    if k not in params and isinstance(spec, dict) and "default" in spec:
                        params[k] = spec["default"]; repairs += 1

            # LLM nodes
            if ntype == "llm" and name.upper().startswith("LLM_SUMMARIZE"):
                src_ids = node.get("depends_on") or node.get("params", {}).get("sources") or []
                blob = self._collect_sources_text(src_ids, results)
                try:
                    state = {"merged": [], "know_snippets":[blob]}
                    reply = self.asm._stage10_assemble_and_infer(context.get("user_text",""), state)
                    if hasattr(reply, "__await__"):
                        # If _stage10_assemble_and_infer is async in your build, await via loop if exposed
                        reply = self.asm.loop.run_until_complete(reply)  # Best-effort
                    reply = (reply or "").strip()
                except Exception:
                    sys_prompt = "Summarize the following tool results succinctly and answer the user's question."
                    user_prompt = f"User: {context.get('user_text','')}\n\nTool Results:\n{blob}"
                    try:
                        reply = self.asm._stream_and_capture(self.asm.primary_model,
                                                             [{"role":"system","content":sys_prompt},
                                                              {"role":"user","content":user_prompt}],
                                                             tag="[LLM_SUMMARIZE]").strip()
                    except Exception:
                        reply = blob[:800]
                results[nid] = reply
                continue

            if ntype == "llm" and name.upper().startswith("LLM_DRAFT_ONLY"):
                try:
                    reply = self.asm._stage10_assemble_and_infer(context.get("user_text",""), {"merged":[]})
                    if hasattr(reply, "__await__"):
                        reply = self.asm.loop.run_until_complete(reply)
                    reply = (reply or "").strip()
                except Exception:
                    reply = self.asm._stream_and_capture(
                        self.asm.primary_model,
                        [{"role":"system","content":"Answer briefly and helpfully."},
                         {"role":"user","content":context.get("user_text","")}],
                        tag="[LLM_DRAFT]").strip()
                results[nid] = reply
                continue

            # TOOL nodes
            if ntype == "tool" and tr:
                try:
                    ctx = tr.execute(name, params, assembler=self.asm)
                    tool_ctxs.append(ctx)
                    results[nid] = ctx
                except Exception as e:
                    err_ctx = ContextObject.make_stage(f"tool:{name}", [], {"error":"execution_error","message":str(e)})
                    err_ctx.component="tool_output"; err_ctx.semantic_label="tool_output"; err_ctx.tags=["tool_output", name]
                    try: self.asm.repo.save(err_ctx)
                    except Exception: pass
                    tool_ctxs.append(err_ctx)
                    results[nid] = err_ctx
                continue

            # Unknown node type
            results[nid] = {"warning":"unknown_node_type", "node":node}

        # Choose a reply
        reply = ""
        for node in reversed(order):
            if node.get("type")=="llm":
                r = results.get(node["id"])
                if isinstance(r, str) and r.strip():
                    reply = r.strip(); break
        if not reply:
            for r in results.values():
                if isinstance(r, str) and r.strip():
                    reply = r.strip(); break
        if not reply:
            blob = self._collect_sources_text([n["id"] for n in order if n.get("type")=="tool"], results)
            reply = blob[:900]

        return {"reply": reply, "tool_ctxs": tool_ctxs, "graph": graph, "issues": [], "repairs": repairs}
