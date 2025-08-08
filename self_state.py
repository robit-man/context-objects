# self_state.py
from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime

DEFAULT_STATE = {
    "version": 1,
    "traits": {
        "verbosity": 0.55,       # 0..1
        "curiosity": 0.50,
        "risk_appetite": 0.40,
        "clarify_bias": 0.55,
        "safety_bias": 0.70,
    },
    "capabilities": {},          # e.g., "web_search": {"success_rate":0.78,"latency_ms_ema":1200,"arg_error_rate":0.06}
    "policies": {
        "arg_norm_min_sim": 0.86,
        "kg_affinity_threshold": 0.22,
        "planner_retries": 0,
        "default_parallelism": 2,
        "plan_timeout_s": 0.0,
        "retrieval_top_k": 12,
        "mmr_diversity": 0.35,
        "recency_half_life_days": 3.0,
    },
    "signals": {
        "turns": 0,
        "relevance_ema": 0.0,
        "last_update": None,
        "failures": {},          # tool_name -> count
    }
}

@dataclass
class SelfState:
    data: Dict[str, Any] = field(default_factory=lambda: DEFAULT_STATE.copy())

    @property
    def policies(self): return self.data["policies"]
    @property
    def traits(self):   return self.data["traits"]
    @property
    def caps(self):     return self.data["capabilities"]
    @property
    def sig(self):      return self.data["signals"]

    def param(self, key: str, default):
        return self.policies.get(key, default)

    def bump_failure(self, tool: str):
        self.sig["failures"][tool] = self.sig["failures"].get(tool, 0) + 1

    def update_capability(self, tool: str, ok: bool, latency_ms: float | None, arg_error: bool):
        c = self.caps.setdefault(tool, {"success_rate": 0.5, "latency_ms_ema": None, "arg_error_rate": 0.0, "n": 0})
        n0 = c["n"]; n = n0 + 1; c["n"] = n
        # EMA / binomial smoothing
        alpha = 0.15
        c["success_rate"] = (1-alpha)*c["success_rate"] + alpha*(1.0 if ok else 0.0)
        if latency_ms is not None:
            if c["latency_ms_ema"] is None: c["latency_ms_ema"] = latency_ms
            else: c["latency_ms_ema"] = 0.8*c["latency_ms_ema"] + 0.2*latency_ms
        if arg_error:
            c["arg_error_rate"] = (1-alpha)*c["arg_error_rate"] + alpha*1.0
        else:
            c["arg_error_rate"] = (1-alpha)*c["arg_error_rate"]

    def adjust_policies(self):
        """Simple automatic adjustments from observed behaviour."""
        # If arg errors are common, raise mapping strictness; else relax slightly.
        avg_arg_err = 0.0
        if self.caps:
            avg_arg_err = sum(v.get("arg_error_rate", 0.0) for v in self.caps.values()) / max(1, len(self.caps))
        t = self.policies.get("arg_norm_min_sim", 0.86)
        if   avg_arg_err > 0.12: t = min(0.93, t + 0.02)
        elif avg_arg_err < 0.03: t = max(0.80, t - 0.02)
        self.policies["arg_norm_min_sim"] = t

        # If overall success is high and latency stable, allow more parallelism
        succ = [v.get("success_rate", 0.5) for v in self.caps.values()]
        if succ:
            ok = sum(succ)/len(succ)
            if ok > 0.8:  self.policies["default_parallelism"] = min(4, self.policies.get("default_parallelism", 2) + 1)
            if ok < 0.5:  self.policies["default_parallelism"] = max(1, self.policies.get("default_parallelism", 2) - 1)

        self.sig["last_update"] = datetime.utcnow().isoformat() + "Z"

    def update_relevance(self, new_score: float):
        a = 0.10
        self.sig["relevance_ema"] = (1-a)*self.sig["relevance_ema"] + a*max(0.0, min(1.0, new_score))
        self.sig["turns"] += 1
