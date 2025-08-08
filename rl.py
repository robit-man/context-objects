# rl.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from datetime import datetime
from context import ContextObject

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

# -----------------------------
# LinUCB per-tool (contextual bandit)
# -----------------------------
@dataclass
class LinUCBArm:
    d: int
    alpha: float = 0.75    # exploration bonus
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    n: int = 0

    def __post_init__(self):
        self.A = np.eye(self.d)
        self.b = np.zeros((self.d, 1))

    def score(self, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        p = float((theta.T @ x.reshape(-1,1)) + self.alpha * np.sqrt(x.reshape(1,-1) @ A_inv @ x.reshape(-1,1)))
        return p

    def update(self, x: np.ndarray, r: float):
        x = x.reshape(-1,1)
        self.A = self.A + (x @ x.T)
        self.b = self.b + r * x
        self.n += 1

# -----------------------------
# Discrete bandit (UCB1)
# -----------------------------
@dataclass
class UCB1:
    arms: List[Any]
    counts: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    t: int = 0

    def __post_init__(self):
        if not self.counts:
            self.counts = [0]*len(self.arms)
        if not self.values:
            self.values = [0.0]*len(self.arms)

    def choose(self) -> Tuple[int, Any]:
        self.t += 1
        # initialize: pull each arm once
        for i,c in enumerate(self.counts):
            if c == 0:
                self.counts[i] += 1
                return i, self.arms[i]
        import math
        ucb = [ self.values[i] + (2*math.log(self.t)/self.counts[i])**0.5 for i in range(len(self.arms)) ]
        i = int(np.argmax(ucb))
        self.counts[i] += 1
        return i, self.arms[i]

    def update(self, arm_idx: int, reward: float):
        r = _clip01(reward)
        n = self.counts[arm_idx]
        self.values[arm_idx] = ((n-1)/n)*self.values[arm_idx] + (1/n)*r

# -----------------------------
# RL Manager – persistence + APIs
# -----------------------------
class RLManager:
    """
    Stores bandit state in repo (component='rl_store').
    Provides:
      - rank_tools(tools, feats_map) -> ranked names
      - choose_knob(name, options) -> (idx, value)
      - update_from_turn(…): push scalar reward to chosen tools/knobs
    """
    def __init__(self, repo, d_tool: int = 5, alpha: float = 0.75):
        self.repo = repo
        self.d = d_tool
        self.alpha = alpha
        self.tools: Dict[str, LinUCBArm] = {}
        self.knobs: Dict[str, UCB1] = {}
        self._load()

    # ---------- persistence ----------
    def _load(self):
        rows = sorted(self.repo.query(lambda c: c.component=="rl_store"), key=lambda c: c.timestamp, reverse=True)
        if not rows: return
        blob = rows[0].metadata.get("data", {})
        # tools
        for name, d in (blob.get("tools") or {}).items():
            arm = LinUCBArm(d=self.d, alpha=d.get("alpha", self.alpha))
            arm.A = np.array(d["A"]); arm.b = np.array(d["b"]).reshape(-1,1); arm.n = int(d.get("n",0))
            self.tools[name] = arm
        # knobs
        for k, d in (blob.get("knobs") or {}).items():
            band = UCB1(arms=d["arms"], counts=d["counts"], values=d["values"], t=d.get("t",0))
            self.knobs[k] = band

    def _save(self):
        tools_dump = {
            k: {"A": v.A.tolist(), "b": v.b.flatten().tolist(), "n": v.n, "alpha": v.alpha}
            for k,v in self.tools.items()
        }
        knobs_dump = { k: {"arms": v.arms, "counts": v.counts, "values": v.values, "t": v.t} for k,v in self.knobs.items() }
        ctx = ContextObject.make_stage("rl_store", [], {"data": {"tools": tools_dump, "knobs": knobs_dump}})
        ctx.stage_id = "rl_store"; ctx.summary = "rl_store update"
        ctx.touch(); self.repo.save(ctx)

    # ---------- tool ranking ----------
    def _ensure_tool(self, name: str):
        if name not in self.tools:
            self.tools[name] = LinUCBArm(d=self.d, alpha=self.alpha)

    def rank_tools(self, tool_names: List[str], feats_map: Dict[str, List[float]]) -> List[str]:
        # x = [affinity, success_rate, 1-arg_err, 1-norm_latency, 1]  (bias)
        scores = []
        for nm in tool_names:
            self._ensure_tool(nm)
            x = np.array(feats_map.get(nm) or [0,0.5,0.5,0.5,1.0], dtype=float).reshape(-1)
            scores.append((self.tools[nm].score(x), nm))
        scores.sort(key=lambda t: t[0], reverse=True)
        return [nm for _, nm in scores]

    def update_tool(self, name: str, feats: List[float], reward: float):
        self._ensure_tool(name)
        self.tools[name].update(np.array(feats, dtype=float), _clip01(reward))

    # ---------- knobs ----------
    def choose_knob(self, knob_name: str, options: List[Any]) -> Tuple[int, Any]:
        band = self.knobs.get(knob_name)
        if band is None or band.arms != list(options):
            band = UCB1(arms=list(options))
            self.knobs[knob_name] = band
        return band.choose()

    def update_knob(self, knob_name: str, idx: int, reward: float):
        if knob_name in self.knobs:
            self.knobs[knob_name].update(idx, _clip01(reward))

    # ---------- turn-level helper ----------
    def finalize_turn(self):
        self._save()
