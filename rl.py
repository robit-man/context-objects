# rl.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional, Iterable
import numpy as np
import math
from datetime import datetime
from context import ContextObject

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ─────────────────────────────────────────────────────────────
# LinUCB per-tool (contextual bandit) — unchanged API
# ─────────────────────────────────────────────────────────────
@dataclass
class LinUCBArm:
    d: int
    alpha: float = 0.75        # exploration bonus
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    n: int = 0

    def __post_init__(self):
        self.A = np.eye(self.d)
        self.b = np.zeros((self.d, 1))

    def score(self, x: np.ndarray) -> float:
        # p = theta^T x + alpha * sqrt( x^T A^-1 x )
        x = x.reshape(-1, 1)
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        exploit = float((theta.T @ x))
        explore = float(self.alpha * np.sqrt((x.T @ A_inv @ x)))
        return exploit + explore

    def update(self, x: np.ndarray, r: float):
        x = x.reshape(-1, 1)
        self.A = self.A + (x @ x.T)
        self.b = self.b + float(_clip01(r)) * x
        self.n += 1

# ─────────────────────────────────────────────────────────────
# Discrete bandit: UCB1 — unchanged API
# ─────────────────────────────────────────────────────────────
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
        # Pull each arm once first
        for i, c in enumerate(self.counts):
            if c == 0:
                self.counts[i] += 1
                return i, self.arms[i]
        # Then UCB
        ucb = []
        for i in range(len(self.arms)):
            bonus = math.sqrt(2.0 * math.log(self.t) / max(1, self.counts[i]))
            ucb.append(self.values[i] + bonus)
        idx = int(np.argmax(ucb))
        self.counts[idx] += 1
        return idx, self.arms[idx]

    def update(self, arm_idx: int, reward: float):
        r = _clip01(reward)
        n = self.counts[arm_idx]
        if n <= 0:
            # if someone calls update before choose(): guard
            self.counts[arm_idx] = 1
            n = 1
        # running average
        self.values[arm_idx] = ((n-1)/n)*self.values[arm_idx] + (1/n)*r

# ─────────────────────────────────────────────────────────────
# Thompson sampling for prompt variants (per-slot)
# ─────────────────────────────────────────────────────────────
@dataclass
class BetaArm:
    # Posterior of Bernoulli reward with Beta(alpha, beta)
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    share: float = 1.0              # traffic allocation in [0,1]
    parent: Optional[str] = None    # baseline/parent variant id
    created_at: str = field(default_factory=_now_iso)
    disabled: bool = False

    def sample(self) -> float:
        # draw from Beta posterior; larger is better during selection
        return np.random.beta(max(1e-6, self.alpha), max(1e-6, self.beta))

    def mean(self) -> float:
        return self.alpha / max(1e-6, (self.alpha + self.beta))

    def update(self, reward: float):
        r = _clip01(reward)
        # fractional update lets us use continuous reward∈[0,1]
        self.alpha += r
        self.beta  += (1.0 - r)
        self.pulls += 1

@dataclass
class PromptSlot:
    # Per "slot" (e.g. 'final_inference_prompt'), we track multiple variants
    slot: str
    arms: Dict[str, BetaArm] = field(default_factory=dict)  # variant_id -> BetaArm
    baseline_id: Optional[str] = None
    canary_schedule: List[float] = field(default_factory=lambda: [0.05, 0.2, 0.5, 1.0])
    min_pulls_before_bump: int = 30
    lift_threshold: float = 0.03   # minimal posterior-mean lift vs baseline to bump
    decay: float = 0.0             # optional share decay per finalize (0..1)
    last_updated: str = field(default_factory=_now_iso)

    def ensure(self, variant_id: str) -> BetaArm:
        if variant_id not in self.arms:
            self.arms[variant_id] = BetaArm()
        return self.arms[variant_id]

    def choose(self, candidates: Iterable[str]) -> str:
        """
        Thompson-sample among candidate variants, weighted by current share.
        Disabled arms or zero-share arms are ignored.
        """
        best_id, best_score = None, -1.0
        for vid in candidates:
            arm = self.arms.get(vid) or self.ensure(vid)
            if arm.disabled or arm.share <= 1e-6:
                continue
            score = arm.sample() * max(1e-6, arm.share)
            if score > best_score:
                best_score, best_id = score, vid
        # fallback: pick any candidate (even if share==0) to avoid None
        if best_id is None:
            for vid in candidates:
                best_id = vid
                break
        return best_id

    def baseline_mean(self) -> Optional[float]:
        if self.baseline_id and self.baseline_id in self.arms:
            return self.arms[self.baseline_id].mean()
        return None

    def maybe_bump_share(self, variant_id: str):
        """
        Canary ramping: if variant's posterior mean > baseline + lift by threshold
        and pulls exceed min_pulls_before_bump, bump its share to the next rung.
        """
        if variant_id not in self.arms:
            return
        v = self.arms[variant_id]
        if v.disabled:
            return
        base_mean = self.baseline_mean()
        if base_mean is None:
            return
        if v.pulls < self.min_pulls_before_bump:
            return
        if (v.mean() - base_mean) < self.lift_threshold:
            return
        # bump to next schedule step
        for rung in self.canary_schedule:
            if v.share + 1e-9 < rung:
                v.share = rung
                break

    def maybe_decay_shares(self):
        """
        Light share decay to keep exploration in play if configured.
        """
        if self.decay <= 0:
            return
        for arm in self.arms.values():
            if not arm.disabled:
                arm.share = max(0.01, arm.share * (1.0 - self.decay))

# ─────────────────────────────────────────────────────────────
# Reward aggregator (turn-level → scalar)
# ─────────────────────────────────────────────────────────────
@dataclass
class RewardConfig:
    w_tool_success: float = 1.2
    w_reflection_ok: float = 1.0
    w_critic_needed: float = -0.5
    w_exceptions: float = -0.25
    w_latency_over: float = -0.25
    w_confirmation: float = -0.15
    w_user_feedback: float = 0.8     # multiply by feedback ∈ { -1, 0, +1 }
    latency_budget_ms: int = 999999  # set lower in assembler if you have a budget

class RewardAggregator:
    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()

    def compute(self, state: Dict[str, Any]) -> float:
        tool_ctxs = state.get("tool_ctxs", []) or []
        succ = sum(1 for t in tool_ctxs if (t.metadata or {}).get("exception") is None)
        total = len(tool_ctxs)
        tool_score = (succ / max(1, total)) if total else 0.0

        reflect_ok = 1.0 if state.get("reflection_status") == "OK" else -1.0
        critic_fix = -0.5 if state.get("critic_needed", False) else 0.0
        exceptions = sum(1 for t in tool_ctxs if (t.metadata or {}).get("exception"))
        latency_ms = int(state.get("latency_ms", 0))
        latency_over = 1 if latency_ms > self.cfg.latency_budget_ms else 0
        confirm_pen = 1 if state.get("asked_confirmation") else 0
        user_fb = state.get("user_feedback")  # -1, 0, +1 or None

        reward = (
            self.cfg.w_tool_success * tool_score +
            self.cfg.w_reflection_ok * reflect_ok +
            self.cfg.w_critic_needed * (1 if critic_fix < 0 else 0) +
            self.cfg.w_exceptions * exceptions +
            self.cfg.w_latency_over * latency_over +
            self.cfg.w_confirmation * confirm_pen +
            (self.cfg.w_user_feedback * float(user_fb) if user_fb is not None else 0.0)
        )
        # squash a bit into [0,1] with logistic-ish mapping, but keep sign
        # map R∈(-∞,∞) → (0,1) with 0.5 at R=0 for stability
        reward01 = 1.0 / (1.0 + math.exp(-reward))
        return _clip01(reward01)

# ─────────────────────────────────────────────────────────────
# RL Manager – persistence + APIs (extended)
# ─────────────────────────────────────────────────────────────
class RLManager:
    """
    Stores bandit state in repo (component='rl_store').

    Provides:
      - rank_tools(tool_names, feats_map) -> ranked list ( LinUCB )
      - update_tool(name, feats, reward)
      - choose_knob(name, options) -> (idx, value)  ( UCB1 )
      - update_knob(name, idx, reward)
      - choose_prompt_variant(slot, variants) -> variant_id ( Thompson + canary )
      - update_prompt_variant(slot, variant_id, reward) -> bumps shares if warranted
      - register_turn(...): one-shot update from a single run
      - finalize_turn(): persist snapshot to repo
    """

    def __init__(
        self,
        repo,
        d_tool: int = 5,
        alpha: float = 0.75,
        reward_cfg: RewardConfig | None = None
    ):
        self.repo = repo
        self.d = d_tool
        self.alpha = alpha
        self.tools: Dict[str, LinUCBArm] = {}
        self.knobs: Dict[str, UCB1] = {}
        self.prompt_slots: Dict[str, PromptSlot] = {}
        self.reward = RewardAggregator(reward_cfg)
        self._load()

    # ---------- persistence ----------
    def _load(self):
        rows = sorted(
            self.repo.query(lambda c: c.component == "rl_store"),
            key=lambda c: c.timestamp,
            reverse=True
        )
        if not rows:
            return
        blob = rows[0].metadata.get("data", {}) or {}

        # tools
        for name, d in (blob.get("tools") or {}).items():
            arm = LinUCBArm(d=self.d, alpha=d.get("alpha", self.alpha))
            arm.A = np.array(d["A"])
            arm.b = np.array(d["b"]).reshape(-1, 1)
            arm.n = int(d.get("n", 0))
            self.tools[name] = arm

        # knobs
        for k, d in (blob.get("knobs") or {}).items():
            band = UCB1(
                arms=d.get("arms", []),
                counts=d.get("counts", []),
                values=d.get("values", []),
                t=d.get("t", 0)
            )
            self.knobs[k] = band

        # prompt slots
        for slot, d in (blob.get("prompt_slots") or {}).items():
            ps = PromptSlot(
                slot=slot,
                canary_schedule=d.get("canary_schedule", [0.05, 0.2, 0.5, 1.0]),
                min_pulls_before_bump=d.get("min_pulls_before_bump", 30),
                lift_threshold=d.get("lift_threshold", 0.03),
                decay=d.get("decay", 0.0),
            )
            ps.baseline_id = d.get("baseline_id")
            for vid, s in (d.get("arms") or {}).items():
                ps.arms[vid] = BetaArm(
                    alpha=float(s.get("alpha", 1.0)),
                    beta=float(s.get("beta", 1.0)),
                    pulls=int(s.get("pulls", 0)),
                    share=float(s.get("share", 1.0)),
                    parent=s.get("parent"),
                    created_at=s.get("created_at", _now_iso()),
                    disabled=bool(s.get("disabled", False)),
                )
            self.prompt_slots[slot] = ps

    def _save(self):
        tools_dump = {
            k: {"A": v.A.tolist(), "b": v.b.flatten().tolist(), "n": v.n, "alpha": v.alpha}
            for k, v in self.tools.items()
        }
        knobs_dump = {
            k: {"arms": v.arms, "counts": v.counts, "values": v.values, "t": v.t}
            for k, v in self.knobs.items()
        }
        slots_dump = {}
        for slot, ps in self.prompt_slots.items():
            slots_dump[slot] = {
                "baseline_id": ps.baseline_id,
                "canary_schedule": ps.canary_schedule,
                "min_pulls_before_bump": ps.min_pulls_before_bump,
                "lift_threshold": ps.lift_threshold,
                "decay": ps.decay,
                "arms": {
                    vid: {
                        "alpha": a.alpha,
                        "beta": a.beta,
                        "pulls": a.pulls,
                        "share": a.share,
                        "parent": a.parent,
                        "created_at": a.created_at,
                        "disabled": a.disabled,
                    }
                    for vid, a in ps.arms.items()
                },
            }

        ctx = ContextObject.make_stage(
            "rl_store",
            [],
            {
                "data": {
                    "tools": tools_dump,
                    "knobs": knobs_dump,
                    "prompt_slots": slots_dump,
                    "saved_at": _now_iso(),
                }
            },
        )
        ctx.stage_id = "rl_store"
        ctx.summary = "rl_store update"
        ctx.touch()
        self.repo.save(ctx)

    # ---------- tools ----------
    def _ensure_tool(self, name: str):
        if name not in self.tools:
            self.tools[name] = LinUCBArm(d=self.d, alpha=self.alpha)

    def rank_tools(self, tool_names: List[str], feats_map: Dict[str, List[float]]) -> List[str]:
        """
        Rank tools by LinUCB score.
        x feature template suggestion:
          [affinity, success_rate, 1-arg_err, 1-norm_latency, bias(=1)]
        """
        scores = []
        for nm in tool_names:
            self._ensure_tool(nm)
            x = np.array(feats_map.get(nm) or [0, 0.5, 0.5, 0.5, 1.0], dtype=float).reshape(-1)
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

    # ---------- prompt variants ----------
    def _ensure_slot(self, slot: str) -> PromptSlot:
        if slot not in self.prompt_slots:
            self.prompt_slots[slot] = PromptSlot(slot=slot)
        return self.prompt_slots[slot]

    def set_baseline_variant(self, slot: str, variant_id: str):
        ps = self._ensure_slot(slot)
        ps.baseline_id = variant_id
        ps.ensure(variant_id).share = max(ps.ensure(variant_id).share, 1.0)  # baseline has full share by default

    def ensure_variants(self, slot: str, variants: Iterable[str], *, default_share: float = 0.05, parent: Optional[str] = None):
        """
        Ensure all variants exist in the slot. New ones start with canary share.
        """
        ps = self._ensure_slot(slot)
        for vid in variants:
            arm = ps.arms.get(vid)
            if arm is None:
                arm = ps.ensure(vid)
                arm.share = float(default_share)
                arm.parent = parent
                arm.created_at = _now_iso()
        # Make sure baseline (if present) keeps ≥ largest share
        if ps.baseline_id and ps.baseline_id in ps.arms:
            ps.arms[ps.baseline_id].share = max(ps.arms[ps.baseline_id].share, 1.0)

    def choose_prompt_variant(self, slot: str, variants: Iterable[str]) -> str:
        """
        Thompson-sample a variant id among the given variants. Ensure they exist.
        """
        ps = self._ensure_slot(slot)
        self.ensure_variants(slot, variants)
        chosen = ps.choose(variants)
        return chosen

    def update_prompt_variant(self, slot: str, variant_id: str, reward: float):
        ps = self._ensure_slot(slot)
        arm = ps.ensure(variant_id)
        arm.update(_clip01(reward))
        # Canary bump & light decay across arms to keep exploration
        ps.maybe_bump_share(variant_id)
        ps.maybe_decay_shares()

    # ---------- turn-level integration ----------
    def register_turn(
        self,
        *,
        tool_updates: List[Tuple[str, List[float]]],           # [(tool_name, feats)]
        selected_knobs: Dict[str, Tuple[int, Any]] | None,     # {knob_name: (idx, value)}
        prompts_used: Dict[str, str],                          # {slot: variant_id}
        state: Dict[str, Any],                                 # assembler state
    ) -> float:
        """
        Compute reward and update all bandits in one go. Returns reward∈[0,1].
        """
        rew = self.reward.compute(state)

        # Update tools with same reward (or you can make per-tool reward later)
        for name, feats in (tool_updates or []):
            try:
                self.update_tool(name, feats, rew)
            except Exception:
                pass

        # Update knobs
        if selected_knobs:
            for k, (idx, _) in selected_knobs.items():
                try:
                    self.update_knob(k, idx, rew)
                except Exception:
                    pass

        # Update prompt variants
        for slot, vid in (prompts_used or {}).items():
            try:
                self.update_prompt_variant(slot, vid, rew)
            except Exception:
                pass

        # Optional: write a small perf blob for analytics
        perf = ContextObject.make_stage(
            "stage_performance",
            [],
            {
                "reward01": float(rew),
                "tool_count": len(state.get("tool_ctxs", []) or []),
                "reflection_status": state.get("reflection_status"),
                "critic_needed": bool(state.get("critic_needed", False)),
                "latency_ms": int(state.get("latency_ms", 0)),
                "asked_confirmation": bool(state.get("asked_confirmation", False)),
                "prompts_used": prompts_used,
                "timestamp": _now_iso(),
            },
        )
        perf.summary = f"reward={rew:.3f} tools={len(state.get('tool_ctxs', []) or [])}"
        perf.stage_id = "stage_performance"
        perf.touch(); self.repo.save(perf)
        return rew

    def finalize_turn(self):
        self._save()
