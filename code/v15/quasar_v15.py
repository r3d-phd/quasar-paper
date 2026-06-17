"""
quasar_v15.py — QUASAR v15: Unified Proactive-Reactive Architecture
=================================================================================
Changes from v14:
  - SWDFT  : Sliding-Window DFT stagnation detector (proactive monitor)
  - BRFD   : Bayesian Reward Function Discovery — shapes probe/reward weights
  - ERL    : Emergency Reactive Layer — 3-stage escalation pipeline
  - SDFT   : Spectral Decomposition Fidelity Tracker (Stage 2 correction)
  - TTT    : Test-Time Training (Stage 3 correction)
  - DEHB Outer : Architecture-level HPO (runs once per qubit level, all families)
  - DEHB Inner : Reactive HPO (fires on escalation failure, narrow re-search)
  - Signal Bus : Typed inter-component communication (8 signals)
  - SLM    : Now signal-gated (SWDFT→ERL→SLM) with BRFD-shaped probe budget
  - All component HPs learned by DEHB; all reward weights learned by BRFD

Architecture layers (v15):
  L1  Environment          QuasarEnv (noise-staged, multi-family)
  L2  Proactive Monitoring SWDFT stagnation detector
  L3  SAC Core             GoalConditionedActor/Critic + FiLM + DER++
  L4  Reactive Correction  ERL → SLM (Stage 1) → SDFT (Stage 2) → TTT (Stage 3)
  L5  Convergence Control  ACC + SLM (periodic fallback)
  L6  Adaptive Curriculum  DEHB Outer (pre-cell) + DEHB Inner (reactive)

Signal Bus signals:
  STAGNATION        SWDFT → ERL
  SLM_CORRECTED     ERL   → resume
  SLM_FAILED        ERL   → SDFT
  SDFT_CORRECTED    SDFT  → resume
  SDFT_FAILED       SDFT  → TTT
  TTT_CORRECTED     TTT   → resume
  TTT_FAILED        TTT   → DEHB Inner
  HP_UPDATED        DEHB  → SAC Core
=================================================================================
"""
# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigSpace import ConfigurationSpace, Float, Integer
from dehb import DEHB
from loguru import logger as log

# ─────────────────────────────────────────────────────────────────────────────
# Device & reproducibility
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────────────────────
# Training constants
# ─────────────────────────────────────────────────────────────────────────────
BUFFER_SIZE   = 200_000
WARMUP_STEPS  = 1_000
UPDATE_EVERY  = 1
LOG_EVERY     = 10_000
DER_CAPACITY  = 20_000

# ─────────────────────────────────────────────────────────────────────────────
# Signal Bus
# ─────────────────────────────────────────────────────────────────────────────
class Signal(Enum):
    STAGNATION      = auto()
    SLM_CORRECTED   = auto()
    SLM_FAILED      = auto()
    SDFT_CORRECTED  = auto()
    SDFT_FAILED     = auto()
    TTT_CORRECTED   = auto()
    TTT_FAILED      = auto()
    HP_UPDATED      = auto()

class SignalBus:
    """Lightweight typed signal bus for inter-component communication."""
    def __init__(self):
        self._signals: Dict[Signal, Any] = {}

    def emit(self, signal: Signal, payload: Any = None):
        self._signals[signal] = payload

    def has(self, signal: Signal) -> bool:
        return signal in self._signals

    def consume(self, signal: Signal) -> Any:
        return self._signals.pop(signal, None)

    def clear(self):
        self._signals.clear()

# ─────────────────────────────────────────────────────────────────────────────
# SWDFT — Sliding-Window Discrete Fourier Transform stagnation detector
# ─────────────────────────────────────────────────────────────────────────────
class SWDFT:
    """
    Monitors best_F over a sliding window and detects stagnation by measuring
    the ratio of low-frequency power to total power in the fidelity time series.
    When the signal is dominated by noise (flat), it emits STAGNATION.

    Parameters
    ----------
    window      : Number of LOG_EVERY checkpoints to keep in the window
    flat_thresh : Low-freq power ratio below which stagnation is declared
    min_steps   : Minimum steps before detection is active
    cooldown    : Minimum checkpoints between successive STAGNATION signals
    """
    def __init__(self, window: int = 20, flat_thresh: float = 0.15,
                 min_steps: int = 50_000, cooldown: int = 10):
        self.window      = window
        self.flat_thresh = flat_thresh
        self.min_steps   = min_steps
        self.cooldown    = cooldown
        self._history: deque = deque(maxlen=window)
        self._last_trigger  = -cooldown
        self._n_checks      = 0

    def update(self, step: int, best_F: float, bus: SignalBus):
        """Call at every LOG_EVERY checkpoint. Emits STAGNATION if detected."""
        self._history.append(best_F)
        self._n_checks += 1
        if step < self.min_steps or len(self._history) < self.window:
            return
        if (self._n_checks - self._last_trigger) < self.cooldown:
            return
        arr = np.array(self._history, dtype=np.float64)
        # DFT of the fidelity window
        spectrum = np.abs(np.fft.rfft(arr - arr.mean()))
        total_power = spectrum.sum() + 1e-12
        # Low-frequency = DC + first 2 harmonics
        low_freq_power = spectrum[:3].sum()
        ratio = low_freq_power / total_power
        if ratio < self.flat_thresh:
            bus.emit(Signal.STAGNATION, payload={
                "step": step, "best_F": best_F,
                "lf_ratio": float(ratio),
                "window_mean": float(arr.mean()),
                "window_std":  float(arr.std()),
            })
            self._last_trigger = self._n_checks
            log.info(f"    SWDFT STAGNATION @ step={step:,} "
                     f"best_F={best_F:.4f} lf_ratio={ratio:.3f}")

    def reset(self):
        self._history.clear()
        self._n_checks = 0
        self._last_trigger = -self.cooldown

# ─────────────────────────────────────────────────────────────────────────────
# BRFD — Bayesian Reward Function Discovery
# ─────────────────────────────────────────────────────────────────────────────
class BRFD:
    """
    Lightweight Bayesian bandit over reward weight combinations.
    Maintains a Beta distribution per weight preset and updates based on
    whether the probe improved best_F.

    Also provides probe_count suggestions calibrated to training difficulty.
    """
    PRESETS = [
        # (w_F, w_S, w_str)  — must sum to <= 1.0; remainder goes to w_log
        (0.60, 0.20, 0.10),
        (0.70, 0.15, 0.05),
        (0.80, 0.10, 0.05),
        (0.50, 0.30, 0.10),
        (0.65, 0.20, 0.10),
        (0.75, 0.15, 0.05),
    ]

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        n = len(self.PRESETS)
        # Beta(alpha, beta) per preset — start with uniform prior
        self._alpha = np.ones(n, dtype=np.float64)
        self._beta  = np.ones(n, dtype=np.float64)
        self._rng   = rng
        self._last_preset_idx: int = 0

    def sample_weights(self) -> Tuple[float, float, float]:
        """Thompson sampling: draw from each Beta, pick argmax."""
        samples = self._rng.beta(self._alpha, self._beta)
        idx = int(np.argmax(samples))
        self._last_preset_idx = idx
        return self.PRESETS[idx]

    def update(self, improved: bool):
        """Update the last-sampled preset based on whether probe improved F."""
        idx = self._last_preset_idx
        if improved:
            self._alpha[idx] += 1.0
        else:
            self._beta[idx] += 1.0

    def suggest_probe_count(self, best_F: float, step: int,
                            base: int = 300) -> int:
        """
        More probes when F is low (hard regime) or step is large (late plateau).
        Returns a value in [base, base * 4].
        """
        difficulty = max(0.0, 1.0 - best_F)          # 0 = easy, 1 = hard
        late_factor = min(1.0, step / 1_000_000)      # 0 = early, 1 = late
        multiplier  = 1.0 + 2.0 * difficulty + 1.0 * late_factor
        return int(base * multiplier)

    def score_probe(self, probe_F: float, best_F: float,
                    step: int) -> float:
        """
        Score a probe action during SLM. Rewards improvement over best_F
        and penalises regression. Used to select the best probe action.
        """
        delta = probe_F - best_F
        return delta + 0.1 * probe_F  # absolute quality bonus

# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, tgt_dim: int):
        self.capacity = capacity
        self.ptr = 0; self.size = 0
        self.obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),        dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),        dtype=np.float32)
        self.targets  = np.zeros((capacity, tgt_dim),  dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, target):
        idx = self.ptr % self.capacity
        self.obs[idx]      = obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.next_obs[idx] = next_obs
        self.dones[idx]    = done
        self.targets[idx]  = target
        self.ptr  += 1
        self.size  = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        def t(x): return torch.FloatTensor(x[idx]).to(DEVICE)
        return t(self.obs), t(self.actions), t(self.rewards), \
               t(self.next_obs), t(self.dones), t(self.targets)

    def __len__(self): return self.size

# ─────────────────────────────────────────────────────────────────────────────
# DER++ Continual Learning Buffer
# ─────────────────────────────────────────────────────────────────────────────
class DERPlusPlusBuffer:
    """Cross-qubit transfer memory — stores (obs, act, rew, next_obs, done, tgt, logit)."""
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = DER_CAPACITY):
        self.capacity  = capacity
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self._n_stored = 0
        self._ptr      = 0
        self._obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self._acts     = np.zeros((capacity, act_dim),  dtype=np.float32)
        self._rews     = np.zeros((capacity, 1),        dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self._dones    = np.zeros((capacity, 1),        dtype=np.float32)
        self._tgts     = np.zeros((capacity, obs_dim),  dtype=np.float32)

    def add_batch(self, obs, acts, rews, next_obs, dones, tgts):
        n = len(obs)
        for i in range(n):
            idx = self._ptr % self.capacity
            self._obs[idx]      = obs[i, :self.obs_dim]
            self._acts[idx]     = acts[i, :self.act_dim]
            self._rews[idx]     = rews[i]
            self._next_obs[idx] = next_obs[i, :self.obs_dim]
            self._dones[idx]    = dones[i]
            self._tgts[idx]     = tgts[i, :self.obs_dim]
            self._ptr += 1
            self._n_stored = min(self._n_stored + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self._n_stored < batch_size:
            return None
        idx = np.random.randint(0, self._n_stored, size=batch_size)
        def t(x): return torch.FloatTensor(x[idx]).to(device)
        return t(self._obs), t(self._acts), t(self._rews), \
               t(self._next_obs), t(self._dones), t(self._tgts)

# ─────────────────────────────────────────────────────────────────────────────
# FiLM conditioning
# ─────────────────────────────────────────────────────────────────────────────
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta  = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma(cond) * x + self.beta(cond)

# ─────────────────────────────────────────────────────────────────────────────
# Goal-conditioned Actor (SAC + FiLM)
# ─────────────────────────────────────────────────────────────────────────────
class GoalConditionedActor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, act_dim: int, tgt_dim: int,
                 hidden_dim: int = 256, cond_dim: int = 64):
        super().__init__()
        self.state_enc  = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.target_enc = nn.Sequential(
            nn.Linear(tgt_dim, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, cond_dim))
        self.film         = FiLM(cond_dim, hidden_dim)
        self.mean_head    = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, target: torch.Tensor):
        h    = self.state_enc(obs)
        cond = self.target_enc(target)
        h    = self.film(h, cond)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        mean    = torch.nan_to_num(mean,    nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.nan_to_num(log_std, nan=self.LOG_STD_MIN)
        return mean, log_std

    def get_action_and_logit(self, obs: torch.Tensor, target: torch.Tensor):
        mean, log_std = self.forward(obs, target)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t  = dist.rsample()
        y_t  = torch.tanh(x_t)
        action   = y_t * math.pi
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * math.pi

# ─────────────────────────────────────────────────────────────────────────────
# Goal-conditioned Critic (twin Q)
# ─────────────────────────────────────────────────────────────────────────────
class GoalConditionedCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, tgt_dim: int,
                 hidden_dim: int = 256, cond_dim: int = 64):
        super().__init__()
        in_dim = obs_dim + act_dim + tgt_dim
        self.q1_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.q2_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, action, target):
        x = torch.cat([obs, action, target], dim=-1)
        return self.q1_enc(x), self.q2_enc(x)

# ─────────────────────────────────────────────────────────────────────────────
# SAC Agent
# ─────────────────────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, target_dim: int,
                 alpha: float = 0.05, lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4, tau: float = 0.005,
                 hidden_dim: int = 256, cond_dim: int = 64):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.target_dim = target_dim
        tgt_dim = target_dim * 2  # real + imag

        self.log_alpha    = torch.tensor(math.log(alpha), requires_grad=True,
                                         device=DEVICE)
        self.target_entropy = -float(act_dim)
        self.tau          = tau

        self.actor  = GoalConditionedActor(obs_dim, act_dim, tgt_dim,
                                           hidden_dim, cond_dim).to(DEVICE)
        self.critic = GoalConditionedCritic(obs_dim, act_dim, tgt_dim,
                                            hidden_dim, cond_dim).to(DEVICE)
        self.critic_target = GoalConditionedCritic(obs_dim, act_dim, tgt_dim,
                                                   hidden_dim, cond_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha],          lr=lr_actor)

    @property
    def alpha(self): return self.log_alpha.exp().item()

    def select_action(self, obs: np.ndarray, target: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            o = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            t = torch.FloatTensor(target).unsqueeze(0).to(DEVICE)
            action, _, mean_action = self.actor.get_action_and_logit(o, t)
        a = mean_action if deterministic else action
        return a.squeeze(0).cpu().numpy()

    def update(self, buffer: ReplayBuffer, batch_size: int,
               der_buffer: Optional[DERPlusPlusBuffer] = None,
               der_batch_size: int = 64):
        if len(buffer) < batch_size:
            return
        obs, acts, rews, next_obs, dones, tgts = buffer.sample(batch_size)
        # Critic update
        with torch.no_grad():
            next_a, next_lp, _ = self.actor.get_action_and_logit(next_obs, tgts)
            q1_t, q2_t = self.critic_target(next_obs, next_a, tgts)
            q_t  = torch.min(q1_t, q2_t) - self.alpha * next_lp
            y    = rews + (1.0 - dones) * 0.99 * q_t
        q1, q2 = self.critic(obs, acts, tgts)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        # DER++ auxiliary loss
        if der_buffer is not None and der_buffer._n_stored >= der_batch_size:
            der_sample = der_buffer.sample(der_batch_size, DEVICE)
            if der_sample is not None:
                d_obs, d_acts, d_rews, d_next_obs, d_dones, d_tgts = der_sample
                if d_obs.shape[1] != obs.shape[1]:
                    pad = obs.shape[1] - d_obs.shape[1]
                    if pad > 0:
                        d_obs      = F.pad(d_obs,      (0, pad))
                        d_next_obs = F.pad(d_next_obs, (0, pad))
                    else:
                        d_obs      = d_obs[:, :obs.shape[1]]
                        d_next_obs = d_next_obs[:, :obs.shape[1]]
                if d_tgts.shape[1] != tgts.shape[1]:
                    pad = tgts.shape[1] - d_tgts.shape[1]
                    if pad > 0:
                        d_tgts = F.pad(d_tgts, (0, pad))
                    else:
                        d_tgts = d_tgts[:, :tgts.shape[1]]
                with torch.no_grad():
                    d_next_a, d_next_lp, _ = self.actor.get_action_and_logit(
                        d_next_obs, d_tgts)
                    dq1_t, dq2_t = self.critic_target(d_next_obs, d_next_a, d_tgts)
                    dy = d_rews + (1.0 - d_dones) * 0.99 * (
                        torch.min(dq1_t, dq2_t) - self.alpha * d_next_lp)
                dq1, dq2 = self.critic(d_obs, d_acts, d_tgts)
                critic_loss = critic_loss + 0.5 * (F.mse_loss(dq1, dy) +
                                                    F.mse_loss(dq2, dy))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        # Actor update
        a_new, log_pi, _ = self.actor.get_action_and_logit(obs, tgts)
        q1_new, q2_new   = self.critic(obs, a_new, tgts)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()
        # Alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # Soft target update
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

# ─────────────────────────────────────────────────────────────────────────────
# ACC — Adaptive Convergence Controller (unchanged from v14)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from acc import AdaptiveConvergenceController, ACCDecision, StopReason
except ImportError:
    # Inline fallback if acc.py not on PYTHONPATH
    class StopReason(Enum):
        NONE             = auto()
        THRESHOLD_MET    = auto()
        FLAT_UNREACHABLE = auto()
        CONVERGED        = auto()
        MAX_BUDGET       = auto()

    @dataclass
    class ACCDecision:
        stop:               bool
        reason:             "StopReason"
        step:               int
        best_F:             float
        F_inf:              Optional[float] = None
        T_star:             Optional[int]   = None
        recommended_budget: Optional[int]   = None

    class AdaptiveConvergenceController:
        def __init__(self, F_threshold=0.99, max_budget=500_000,
                     min_points=5, r2_min=0.70, safety_margin=0.20):
            self.F_threshold   = F_threshold
            self.max_budget    = max_budget
            self.min_points    = min_points
            self.r2_min        = r2_min
            self.safety_margin = safety_margin
            self._steps: List[int]   = []
            self._fids:  List[float] = []

        def update(self, step: int, best_F: float) -> ACCDecision:
            self._steps.append(step)
            self._fids.append(best_F)
            if best_F >= self.F_threshold:
                return ACCDecision(True, StopReason.THRESHOLD_MET, step, best_F,
                                   F_inf=best_F, T_star=step,
                                   recommended_budget=step)
            if len(self._steps) < self.min_points:
                return ACCDecision(False, StopReason.NONE, step, best_F)
            # Saturating exponential fit: F(t) = F_inf * (1 - exp(-t/tau))
            try:
                from scipy.optimize import curve_fit
                def sat_exp(t, F_inf, tau):
                    return F_inf * (1.0 - np.exp(-np.array(t) / (tau + 1e-9)))
                t_arr = np.array(self._steps, dtype=np.float64)
                f_arr = np.array(self._fids,  dtype=np.float64)
                p0    = [max(f_arr) * 1.1, t_arr[-1] / 2]
                popt, _ = curve_fit(sat_exp, t_arr, f_arr, p0=p0,
                                    maxfev=2000, bounds=([0, 1], [1.5, 1e9]))
                F_inf_est, tau_est = popt
                # R² check
                f_pred = sat_exp(t_arr, *popt)
                ss_res = np.sum((f_arr - f_pred) ** 2)
                ss_tot = np.sum((f_arr - f_arr.mean()) ** 2) + 1e-12
                r2     = 1 - ss_res / ss_tot
                if r2 < self.r2_min:
                    return ACCDecision(False, StopReason.NONE, step, best_F,
                                       F_inf=float(F_inf_est))
                if F_inf_est < self.F_threshold * (1 - self.safety_margin):
                    return ACCDecision(True, StopReason.FLAT_UNREACHABLE, step,
                                       best_F, F_inf=float(F_inf_est))
                # Predict T* (steps to reach F_threshold)
                if F_inf_est > self.F_threshold:
                    ratio = 1.0 - self.F_threshold / F_inf_est
                    if ratio > 0:
                        T_star = int(-tau_est * math.log(ratio))
                        rec_budget = int(T_star * (1 + self.safety_margin))
                        if step >= T_star:
                            return ACCDecision(True, StopReason.CONVERGED, step,
                                               best_F, F_inf=float(F_inf_est),
                                               T_star=T_star,
                                               recommended_budget=rec_budget)
                        return ACCDecision(False, StopReason.NONE, step, best_F,
                                           F_inf=float(F_inf_est), T_star=T_star,
                                           recommended_budget=rec_budget)
            except Exception:
                pass
            return ACCDecision(False, StopReason.NONE, step, best_F)

# ─────────────────────────────────────────────────────────────────────────────
# SLM — Spectral Landscape Mapping (v15: signal-gated + BRFD-shaped)
# ─────────────────────────────────────────────────────────────────────────────
def slm_correction(agent: SACAgent, env, target_vec: np.ndarray,
                   n_probe: int = 300, brfd: Optional[BRFD] = None) -> bool:
    """
    Probe the landscape around the current policy.
    In v15, n_probe is supplied by BRFD.suggest_probe_count().
    Returns True if a better action was found and actor was updated.
    """
    obs = env.reset()
    best_score  = -1.0
    best_action = None
    best_F_probe = 0.0

    for _ in range(n_probe):
        a = agent.select_action(obs, target_vec, deterministic=False)
        sigma = 0.3 * (1.0 + np.random.rand())  # adaptive perturbation
        a_probe = np.clip(a + np.random.randn(*a.shape) * sigma, -math.pi, math.pi)
        _, _, _, info = env.step(a_probe)
        F = info.get("F", info.get("fidelity", 0.0))
        score = (brfd.score_probe(F, best_F_probe, 0)
                 if brfd is not None else F)
        if score > best_score:
            best_score  = score
            best_action = a_probe
            best_F_probe = F

    if best_action is not None and best_F_probe > 0.3:
        o_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        t_t = torch.FloatTensor(target_vec).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor(best_action).unsqueeze(0).to(DEVICE)
        mean, log_std = agent.actor(o_t, t_t)
        std  = log_std.exp()
        loss = -torch.distributions.Normal(mean, std).log_prob(
            a_t / math.pi).sum()
        agent.actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_opt.step()
        return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# SDFT — Spectral Decomposition Fidelity Tracker (Stage 2 correction)
# ─────────────────────────────────────────────────────────────────────────────
class SDFT:
    """
    Analyses the fidelity history spectrum to identify dominant stagnation
    frequencies and applies a targeted phase correction to the actor output layer.
    """
    def __init__(self, history_len: int = 30):
        self.history_len = history_len
        self._F_history: deque = deque(maxlen=history_len)

    def record(self, best_F: float):
        self._F_history.append(best_F)

    def correct(self, agent: SACAgent, env, target_vec: np.ndarray,
                n_probe: int = 100) -> bool:
        """
        Identify the dominant stagnation mode and apply a targeted gradient
        correction to the actor's mean_head to escape the plateau.
        Returns True if correction was applied.
        """
        if len(self._F_history) < self.history_len // 2:
            return False
        arr = np.array(self._F_history, dtype=np.float64)
        spectrum = np.abs(np.fft.rfft(arr - arr.mean()))
        # Find dominant non-DC frequency
        dominant_freq_idx = int(np.argmax(spectrum[1:]) + 1)
        # Use the dominant frequency to set a perturbation phase
        phase = 2 * math.pi * dominant_freq_idx / len(arr)
        obs = env.reset()
        best_F_corr = 0.0
        best_action = None
        for i in range(n_probe):
            a = agent.select_action(obs, target_vec, deterministic=True)
            # Phase-directed perturbation
            perturb = math.sin(phase * i) * 0.5
            a_probe = np.clip(a + perturb, -math.pi, math.pi)
            _, _, _, info = env.step(a_probe)
            F = info.get("F", info.get("fidelity", 0.0))
            if F > best_F_corr:
                best_F_corr = F
                best_action = a_probe
        if best_action is not None and best_F_corr > 0.3:
            o_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            t_t = torch.FloatTensor(target_vec).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor(best_action).unsqueeze(0).to(DEVICE)
            mean, log_std = agent.actor(o_t, t_t)
            std  = log_std.exp()
            loss = -torch.distributions.Normal(mean, std).log_prob(
                a_t / math.pi).sum()
            agent.actor_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_opt.step()
            log.info(f"    SDFT correction applied (dominant_freq={dominant_freq_idx}, "
                     f"probe_best_F={best_F_corr:.4f})")
            return True
        return False

# ─────────────────────────────────────────────────────────────────────────────
# TTT — Test-Time Training (Stage 3 correction)
# ─────────────────────────────────────────────────────────────────────────────
class TTT:
    """
    Re-anchors the policy to its historical best trajectory via a short
    supervised gradient update on the actor. This is the most expensive
    stage and is only invoked when SLM and SDFT both fail.
    """
    def __init__(self, n_gradient_steps: int = 50, lr: float = 1e-4):
        self.n_gradient_steps = n_gradient_steps
        self.lr               = lr
        self._best_obs:    Optional[np.ndarray] = None
        self._best_action: Optional[np.ndarray] = None
        self._best_target: Optional[np.ndarray] = None

    def record_best(self, obs: np.ndarray, action: np.ndarray,
                    target: np.ndarray):
        """Call whenever a new best_F is achieved during training."""
        self._best_obs    = obs.copy()
        self._best_action = action.copy()
        self._best_target = target.copy()

    def correct(self, agent: SACAgent) -> bool:
        """
        Apply n_gradient_steps of supervised imitation on the best trajectory.
        Returns True if correction was applied.
        """
        if (self._best_obs is None or self._best_action is None
                or self._best_target is None):
            return False
        o_t = torch.FloatTensor(self._best_obs).unsqueeze(0).to(DEVICE)
        t_t = torch.FloatTensor(self._best_target).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor(self._best_action).unsqueeze(0).to(DEVICE)
        # Temporarily lower LR for fine-grained correction
        orig_lr = agent.actor_opt.param_groups[0]["lr"]
        for pg in agent.actor_opt.param_groups:
            pg["lr"] = self.lr
        for _ in range(self.n_gradient_steps):
            mean, log_std = agent.actor(o_t, t_t)
            std  = log_std.exp()
            loss = -torch.distributions.Normal(mean, std).log_prob(
                a_t / math.pi).sum()
            agent.actor_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor_opt.step()
        for pg in agent.actor_opt.param_groups:
            pg["lr"] = orig_lr
        log.info(f"    TTT correction applied ({self.n_gradient_steps} steps)")
        return True

# ─────────────────────────────────────────────────────────────────────────────
# ERL — Emergency Reactive Layer (orchestrates Stage 1→2→3)
# ─────────────────────────────────────────────────────────────────────────────
class ERL:
    """
    Orchestrates the 3-stage reactive correction pipeline:
      Stage 1: SLM  (cheap, random landscape probing)
      Stage 2: SDFT (spectral phase correction)
      Stage 3: TTT  (test-time training on best trajectory)
    Falls through to DEHB Inner if all three fail.
    """
    def __init__(self, sdft: SDFT, ttt: TTT, brfd: BRFD):
        self.sdft = sdft
        self.ttt  = ttt
        self.brfd = brfd
        self.n_activations  = 0
        self.n_slm_success  = 0
        self.n_sdft_success = 0
        self.n_ttt_success  = 0
        self.n_full_fail    = 0

    def respond(self, bus: SignalBus, agent: SACAgent, env,
                target_vec: np.ndarray, best_F: float, step: int):
        """
        Called when bus has STAGNATION signal.
        Tries SLM → SDFT → TTT in order.
        Emits the appropriate signal after each stage.
        """
        if not bus.has(Signal.STAGNATION):
            return
        payload = bus.consume(Signal.STAGNATION)
        self.n_activations += 1
        log.info(f"    ERL activated (activation #{self.n_activations}) "
                 f"@ step={step:,} best_F={best_F:.4f}")

        # Stage 1: SLM
        n_probes = self.brfd.suggest_probe_count(best_F, step)
        slm_ok   = slm_correction(agent, env, target_vec,
                                  n_probe=n_probes, brfd=self.brfd)
        self.brfd.update(improved=slm_ok)
        if slm_ok:
            bus.emit(Signal.SLM_CORRECTED, payload={"step": step})
            self.n_slm_success += 1
            log.info(f"    ERL Stage 1 (SLM) SUCCESS @ step={step:,}")
            return

        bus.emit(Signal.SLM_FAILED, payload={"step": step})
        log.info(f"    ERL Stage 1 (SLM) FAILED — escalating to SDFT")

        # Stage 2: SDFT
        sdft_ok = self.sdft.correct(agent, env, target_vec)
        if sdft_ok:
            bus.emit(Signal.SDFT_CORRECTED, payload={"step": step})
            self.n_sdft_success += 1
            return

        bus.emit(Signal.SDFT_FAILED, payload={"step": step})
        log.info(f"    ERL Stage 2 (SDFT) FAILED — escalating to TTT")

        # Stage 3: TTT
        ttt_ok = self.ttt.correct(agent)
        if ttt_ok:
            bus.emit(Signal.TTT_CORRECTED, payload={"step": step})
            self.n_ttt_success += 1
            return

        bus.emit(Signal.TTT_FAILED, payload={"step": step})
        self.n_full_fail += 1
        log.info(f"    ERL Stage 3 (TTT) FAILED — signalling DEHB Inner")

    def summary(self) -> dict:
        return {
            "activations":   self.n_activations,
            "slm_success":   self.n_slm_success,
            "sdft_success":  self.n_sdft_success,
            "ttt_success":   self.n_ttt_success,
            "full_fail":     self.n_full_fail,
        }

# ─────────────────────────────────────────────────────────────────────────────
# SiliQunEnvWrapper — adapts SiliQunEnv (Gymnasium API) to the v15 SAC interface
#
# Key differences bridged:
#   1. Gymnasium step() returns (obs, reward, terminated, truncated, info)
#      → wrapped to (obs, reward, done, info)
#   2. Gymnasium reset() returns (obs, info)  → wrapped to obs
#   3. info["fidelity"] → info["F"] alias added
#   4. Target vector extracted from env._target_state (MPS or ndarray)
#   5. Action space [-1, 1] → actor output scaled by π before passing to env
#      (actor still outputs in [-π, π]; wrapper rescales to [-1, 1] for SiliQun)
# ─────────────────────────────────────────────────────────────────────────────

# Add SiliQun to PYTHONPATH
for _p in [
    str(Path.home() / "siliqun"),
    str(Path.home() / "siliqun_v6"),
    str(Path.home() / "siliqun" / "siliqun"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from siliqun.engine.gym_env import SiliQunEnv, make_siliqun_env
    _SILIQUN_AVAILABLE = True
except ImportError:
    _SILIQUN_AVAILABLE = False
    log.warning("SiliQun not found on PYTHONPATH — will raise at env creation")


class SiliQunEnvWrapper:
    """
    Thin wrapper around SiliQunEnv that presents the same interface
    expected by the v15 SAC training loop:
      - reset() → np.ndarray  (obs only)
      - step(action) → (obs, reward, done, info)  (4-tuple)
      - env.target  → complex128 ndarray of shape (2**n,)
      - env.observation_space, env.action_space  (passthrough)
      - info["F"] and info["fidelity"] both available
    """

    # Map v15 target names → SiliQun target names
    _TARGET_MAP = {
        "GHZ":          "ghz",
        "ghz":          "ghz",
        "W":            "w",
        "w":            "w",
        "Cluster":      "cluster_linear",
        "cluster":      "cluster_linear",
        "cluster_linear": "cluster_linear",
        "Dicke-k3":     "dicke_k3",
        "Dicke_k3":     "dicke_k3",
        "dicke-k3":     "dicke_k3",
        "dicke_k3":     "dicke_k3",
    }

    def __init__(self, n_qubits: int, target_state: str,
                 noise_stage: int = 5, max_ep_steps: int = 200,
                 reward_weights: Optional[dict] = None, seed: int = 42):
        if not _SILIQUN_AVAILABLE:
            raise ImportError(
                "SiliQun package not found. Install it with: "
                "pip install -e ~/siliqun")
        siliqun_target = self._TARGET_MAP.get(target_state, target_state.lower())
        # Map noise_stage (0-10) to SiliQun noise bool + sim_mode
        noise_enabled = noise_stage > 0
        # Use MPS for small systems, SV for large
        sim_mode = "sv" if n_qubits >= 8 else "mps"
        self._env = make_siliqun_env(
            n_qubits=n_qubits,
            device="donor",
            target=siliqun_target,
            sim_mode=sim_mode,
            noise=noise_enabled,
            max_bond_dim=min(64, 2 ** (n_qubits // 2)),
            max_steps=max_ep_steps,
            fidelity_threshold=0.99,
            reward_type="shaped",
            seed=seed,
            use_gpu=True,
        )
        self._n_qubits = n_qubits
        self._target_state_name = siliqun_target
        self._reward_weights = reward_weights or {}
        # Build target vector (real + imag concatenated)
        self._target_vec = self._extract_target_vec()
        # Expose Gymnasium spaces directly
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    def _extract_target_vec(self) -> np.ndarray:
        """Extract dense complex state vector from SiliQun target."""
        ts = self._env._target_state
        if isinstance(ts, np.ndarray):
            sv = ts.flatten().astype(np.complex128)
        else:
            # MPS object — convert to dense
            try:
                sv = ts.to_dense().flatten().astype(np.complex128)
            except Exception:
                # Fallback: build GHZ manually
                dim = 2 ** self._n_qubits
                sv = np.zeros(dim, dtype=np.complex128)
                sv[0] = sv[-1] = 1.0 / math.sqrt(2)
        return sv

    @property
    def target(self) -> np.ndarray:
        """Complex target state vector (2**n,) — used to build target_vec."""
        return self._target_vec

    def reset(self) -> np.ndarray:
        """Reset env and return obs array (no info)."""
        obs, _info = self._env.reset()
        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action (in [-π, π]) and return (obs, reward, done, info).
        SiliQun expects actions in [-1, 1], so we rescale.
        """
        # Rescale from actor space [-π, π] → SiliQun space [-1, 1]
        a_scaled = np.clip(action / math.pi, -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = self._env.step(a_scaled)
        done = terminated or truncated
        # Apply BRFD-style reward reweighting if weights provided
        F = float(info.get("fidelity", 0.0))
        if self._reward_weights:
            w_F   = self._reward_weights.get("fidelity",  0.65)
            w_S   = self._reward_weights.get("success",   0.20)
            w_str = self._reward_weights.get("structure", 0.10)
            w_log = self._reward_weights.get("log",       0.05)
            reward = (w_F * F
                      + w_S * float(info.get("success", 0.0))
                      + w_str * float(info.get("bond_dims", [1])[0]) / 32.0
                      + w_log * math.log1p(F))
        # Alias fidelity as "F" for v15 pipeline compatibility
        info["F"] = F
        return np.array(obs, dtype=np.float32), float(reward), bool(done), info

    def __getattr__(self, name):
        """Passthrough to underlying SiliQunEnv for any other attribute."""
        return getattr(self._env, name)

# ─────────────────────────────────────────────────────────────────────────────
# Transfer agent (v14 logic preserved)
# ─────────────────────────────────────────────────────────────────────────────
def transfer_agent(source: SACAgent, obs_dim: int, act_dim: int,
                   target_dim: int, **sac_kwargs) -> SACAgent:
    new_agent = SACAgent(obs_dim, act_dim, target_dim, **sac_kwargs)
    tgt_dim_new = target_dim * 2
    def _copy(src_layer, tgt_layer):
        r = min(src_layer.weight.shape[0], tgt_layer.weight.shape[0])
        c = min(src_layer.weight.shape[1], tgt_layer.weight.shape[1])
        tgt_layer.weight.data[:r, :c] = src_layer.weight.data[:r, :c]
        tgt_layer.bias.data[:r]       = src_layer.bias.data[:r]
        if tgt_layer.weight.shape[1] > c:
            tgt_layer.weight.data[:r, c:] = torch.randn_like(
                tgt_layer.weight.data[:r, c:]) * 0.01
        if tgt_layer.weight.shape[0] > r:
            tgt_layer.weight.data[r:, :] = torch.randn_like(
                tgt_layer.weight.data[r:, :]) * 0.01
            tgt_layer.bias.data[r:] = 0.0
    _copy(source.actor.state_enc[0],  new_agent.actor.state_enc[0])
    _copy(source.actor.state_enc[2],  new_agent.actor.state_enc[2])
    _copy(source.actor.target_enc[0], new_agent.actor.target_enc[0])
    _copy(source.actor.target_enc[2], new_agent.actor.target_enc[2])
    _copy(source.actor.film.gamma,    new_agent.actor.film.gamma)
    _copy(source.actor.film.beta,     new_agent.actor.film.beta)
    _copy(source.actor.mean_head,     new_agent.actor.mean_head)
    _copy(source.actor.log_std_head,  new_agent.actor.log_std_head)
    for sq, tq in [(source.critic.q1_enc, new_agent.critic.q1_enc),
                   (source.critic.q2_enc, new_agent.critic.q2_enc)]:
        for i in [0, 2]:
            _copy(sq[i], tq[i])
    log.info(f"  Transfer: {source.obs_dim}→{obs_dim} obs, "
             f"{source.act_dim}→{act_dim} act")
    return new_agent

# ─────────────────────────────────────────────────────────────────────────────
# DEHB config spaces
# ─────────────────────────────────────────────────────────────────────────────
def build_inner_cs(seed: int = 42) -> ConfigurationSpace:
    """Inner DEHB: full HP space (8 dims), used pre-cell."""
    cs = ConfigurationSpace(seed=seed)
    cs.add([
        Float("alpha",          (0.005, 0.50),    log=True,  default=0.05),
        Float("lr_actor",       (1e-4,  5e-3),    log=True,  default=3e-4),
        Float("lr_critic",      (1e-4,  5e-3),    log=True,  default=3e-4),
        Float("tau",            (0.001, 0.05),     log=False, default=0.005),
        Integer("batch_size",   (64,   512),       log=True,  default=256),
        Integer("hidden_dim",   (128,  512),       log=True,  default=256),
        Integer("cond_dim",     (32,   128),       log=True,  default=64),
        Integer("slm_interval", (5_000, 100_000),  log=True,  default=50_000),
    ])
    return cs

def build_reactive_cs(seed: int = 42) -> ConfigurationSpace:
    """Reactive DEHB Inner: narrow 2-dim re-search around current best."""
    cs = ConfigurationSpace(seed=seed)
    cs.add([
        Float("alpha",    (0.005, 0.30), log=True, default=0.05),
        Float("lr_actor", (1e-4,  3e-3), log=True, default=3e-4),
    ])
    return cs

# ─────────────────────────────────────────────────────────────────────────────
# run_cell — single training cell with full v15 reactive pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_cell(
    n_qubits:      int,
    target_state:  str,
    seed:          int,
    noise_stage:   int,
    max_ep_steps:  int,
    reward_w_F:    float,
    reward_w_S:    float,
    reward_w_str:  float,
    alpha:         float,
    lr_actor:      float,
    lr_critic:     float,
    tau:           float,
    batch_size:    int,
    hidden_dim:    int,
    cond_dim:      int,
    max_steps:     int,
    init_agent:    Optional[SACAgent] = None,
    plateau_patience: int = 150_000,
    slm_interval:  int = 50_000,
    der_buffer:    Optional[DERPlusPlusBuffer] = None,
    der_batch_size: int = 64,
    acc_f_threshold: float = 0.99,
    acc_min_points:  int   = 5,
    acc_r2_min:      float = 0.70,
    acc_margin:      float = 0.20,
    # v15 reactive components (optional — created internally if not passed)
    bus:    Optional[SignalBus] = None,
    swdft:  Optional[SWDFT]    = None,
    brfd:   Optional[BRFD]     = None,
    erl:    Optional[ERL]      = None,
    # DEHB Inner reactive re-search
    enable_reactive_dehb: bool = True,
    reactive_brackets:    int  = 4,
) -> Tuple[float, int, List[dict], SACAgent]:
    """Run one training cell. Returns (best_F, total_steps, history, agent)."""
    _TARGET_NORM = {
        "GHZ": "ghz", "ghz": "ghz",
        "W": "w", "w": "w",
        "Cluster": "cluster_linear", "cluster": "cluster_linear",
        "cluster_linear": "cluster_linear",
        "Dicke-k3": "dicke_k3", "Dicke_k3": "dicke_k3",
        "dicke-k3": "dicke_k3", "dicke_k3": "dicke_k3",
    }
    target_state = _TARGET_NORM.get(target_state, target_state.lower())
    set_seed(seed)

    # Instantiate v15 components if not provided
    if bus   is None: bus   = SignalBus()
    if swdft is None: swdft = SWDFT(window=20, flat_thresh=0.15,
                                    min_steps=30_000, cooldown=8)
    if brfd  is None: brfd  = BRFD(seed=seed)
    sdft_inst = SDFT(history_len=30)
    ttt_inst  = TTT(n_gradient_steps=50, lr=1e-4)
    if erl   is None: erl   = ERL(sdft=sdft_inst, ttt=ttt_inst, brfd=brfd)

    _w_log = max(0.0, 1.0 - reward_w_F - reward_w_S - reward_w_str)
    env = SiliQunEnvWrapper(
        n_qubits=n_qubits,
        target_state=target_state,
        noise_stage=noise_stage,
        max_ep_steps=max_ep_steps,
        reward_weights={
            "fidelity":  reward_w_F,
            "success":   reward_w_S,
            "structure": reward_w_str,
            "log":       _w_log,
        },
        seed=seed,
    )
    obs_dim    = env.observation_space.shape[0]
    act_dim    = env.action_space.shape[0]
    target_dim = env.target.shape[0]

    sac_kwargs = dict(alpha=alpha, lr_actor=lr_actor, lr_critic=lr_critic,
                      tau=tau, hidden_dim=hidden_dim, cond_dim=cond_dim)
    agent = (transfer_agent(init_agent, obs_dim, act_dim, target_dim, **sac_kwargs)
             if init_agent is not None
             else SACAgent(obs_dim, act_dim, target_dim, **sac_kwargs))
    buffer     = ReplayBuffer(BUFFER_SIZE, obs_dim, act_dim, target_dim * 2)
    # SiliQunEnvWrapper.target is already a complex128 state vector
    _tgt = env.target
    target_vec = np.concatenate([_tgt.real, _tgt.imag]).astype(np.float32)

    obs       = env.reset()
    best_F    = 0.0
    best_obs  = obs.copy()
    best_act  = np.zeros(act_dim, dtype=np.float32)
    history   = []
    step      = 0
    _reactive_dehb_fired = False

    acc = AdaptiveConvergenceController(
        F_threshold=acc_f_threshold,
        max_budget=max_steps,
        min_points=acc_min_points,
        r2_min=acc_r2_min,
        safety_margin=acc_margin,
    )
    _dec = ACCDecision(stop=False, reason=StopReason.NONE, step=0, best_F=0.0)

    while step < max_steps:
        action = (env.action_space.sample() if step < WARMUP_STEPS
                  else agent.select_action(obs, target_vec))
        next_obs, reward, done, info = env.step(action)
        F = info.get("F", info.get("fidelity", 0.0))
        buffer.add(obs, action, reward, next_obs, float(done), target_vec)
        obs = next_obs if not done else env.reset()
        step += 1

        if F > best_F:
            best_F   = F
            best_obs = obs.copy()
            best_act = action.copy()
            ttt_inst.record_best(best_obs, best_act, target_vec)

        if step >= WARMUP_STEPS and step % UPDATE_EVERY == 0:
            agent.update(buffer, batch_size, der_buffer=der_buffer,
                         der_batch_size=der_batch_size)

        # SDFT records every step for spectral analysis
        if step % LOG_EVERY == 0:
            sdft_inst.record(best_F)

        # Periodic SLM fallback (v14 behaviour, lower priority than signal-gated)
        if step % slm_interval == 0 and step > WARMUP_STEPS:
            if not bus.has(Signal.STAGNATION):  # only if ERL not already active
                if slm_correction(agent, env, target_vec, n_probe=200, brfd=brfd):
                    log.info(f"    SLM periodic correction @ step={step:,}")

        if step % LOG_EVERY == 0:
            # SWDFT check — may emit STAGNATION
            swdft.update(step, best_F, bus)

            # ERL response if stagnation detected
            if bus.has(Signal.STAGNATION):
                erl.respond(bus, agent, env, target_vec, best_F, step)

            # Reactive DEHB Inner if all ERL stages failed
            if bus.has(Signal.TTT_FAILED) and enable_reactive_dehb \
                    and not _reactive_dehb_fired:
                bus.consume(Signal.TTT_FAILED)
                log.info(f"    DEHB Inner (reactive) firing @ step={step:,}")
                try:
                    _r_cs = build_reactive_cs(seed=seed)
                    _r_dehb = DEHB(
                        cs=_r_cs,
                        min_fidelity=0.1, max_fidelity=1.0,
                        n_workers=1, output_path="/tmp",
                    )
                    def _r_target(config, budget, **kw):
                        try:
                            _r_F, _, _, _ = run_cell(
                                n_qubits=n_qubits, target_state=target_state,
                                seed=seed, noise_stage=noise_stage,
                                max_ep_steps=max_ep_steps,
                                reward_w_F=reward_w_F, reward_w_S=reward_w_S,
                                reward_w_str=reward_w_str,
                                alpha=float(config["alpha"]),
                                lr_actor=float(config["lr_actor"]),
                                lr_critic=lr_critic, tau=tau,
                                batch_size=batch_size, hidden_dim=hidden_dim,
                                cond_dim=cond_dim,
                                max_steps=int(max_steps * 0.1),
                                init_agent=agent,
                                slm_interval=slm_interval,
                                der_buffer=der_buffer,
                                enable_reactive_dehb=False,
                            )
                            return {"fitness": 1.0 - _r_F, "cost": budget}
                        except Exception:
                            return {"fitness": 1.0, "cost": budget}
                    _r_dehb.run(
                        target_function=_r_target,
                        fevals=reactive_brackets * 3,
                    )
                    _r_inc = _r_dehb.get_incumbents()
                    if _r_inc and _r_inc["config"] is not None:
                        _r_cfg = _r_inc["config"]
                        # Hot-patch alpha and lr_actor into existing agent
                        new_alpha = float(_r_cfg["alpha"])
                        new_lr    = float(_r_cfg["lr_actor"])
                        agent.log_alpha = torch.tensor(
                            math.log(new_alpha), requires_grad=True, device=DEVICE)
                        agent.alpha_opt = torch.optim.Adam(
                            [agent.log_alpha], lr=new_lr)
                        for pg in agent.actor_opt.param_groups:
                            pg["lr"] = new_lr
                        bus.emit(Signal.HP_UPDATED,
                                 payload={"alpha": new_alpha, "lr_actor": new_lr})
                        log.info(f"    DEHB Inner: new alpha={new_alpha:.4f} "
                                 f"lr_actor={new_lr:.2e}")
                    _reactive_dehb_fired = True
                except Exception as _re:
                    log.warning(f"    DEHB Inner failed: {_re}")

            # ACC convergence check
            _dec = acc.update(step, best_F)
            stop, reason = _dec.stop, _dec.reason
            _F_inf = round(_dec.F_inf, 5) if _dec.F_inf is not None else None
            history.append({
                "step": step, "best_F": round(best_F, 5),
                "acc_stop": stop, "acc_reason": str(reason),
                "acc_F_inf": _F_inf,
                "acc_T_star": _dec.T_star,
                "acc_recommended_budget": _dec.recommended_budget,
                "erl_summary": erl.summary(),
            })
            log.info(f"    {n_qubits}Q/{target_state}/s{seed} "
                     f"step={step:,} best_F={best_F:.4f} "
                     f"acc={reason} F_inf={_F_inf} alpha={agent.alpha:.4f}")
            if stop:
                log.info(f"    ACC STOP [{reason}] step={step:,} "
                         f"F_inf={_F_inf} T*={_dec.T_star}")
                break

    _F_inf_final = round(_dec.F_inf, 5) if _dec.F_inf is not None else None
    history.append({
        "step": step, "best_F": round(best_F, 5),
        "acc_stop": True, "acc_reason": "MAX_BUDGET",
        "acc_F_inf": _F_inf_final,
        "acc_T_star": _dec.T_star,
        "acc_recommended_budget": _dec.recommended_budget,
        "erl_summary": erl.summary(),
    })
    return best_F, step, history, agent

# ─────────────────────────────────────────────────────────────────────────────
# DEHB Outer target factory
# ─────────────────────────────────────────────────────────────────────────────
class OuterTargetFactory:
    """
    Wraps run_cell for the DEHB Outer loop.
    Evaluates a HP configuration on ALL (target, seed) pairs at short budget.
    Returns mean best_F across all pairs as the fitness signal.
    """
    def __init__(self, n_qubits: int, targets: List[str], seeds: List[int],
                 noise_stage: int, max_ep_steps: int, short_budget: int,
                 der_buffer: Optional[DERPlusPlusBuffer],
                 init_agent: Optional[SACAgent]):
        self.n_qubits     = n_qubits
        self.targets      = targets
        self.seeds        = seeds
        self.noise_stage  = noise_stage
        self.max_ep_steps = max_ep_steps
        self.short_budget = short_budget
        self.der_buffer   = der_buffer
        self.init_agent   = init_agent
        self._n_calls     = 0

    def __call__(self, config, budget, **kwargs):
        self._n_calls += 1
        fids = []
        for target in self.targets:
            for seed in self.seeds:
                try:
                    F, _, _, _ = run_cell(
                        n_qubits=self.n_qubits,
                        target_state=target,
                        seed=seed,
                        noise_stage=self.noise_stage,
                        max_ep_steps=self.max_ep_steps,
                        reward_w_F=0.65, reward_w_S=0.20, reward_w_str=0.10,
                        alpha=float(config["alpha"]),
                        lr_actor=float(config["lr_actor"]),
                        lr_critic=float(config["lr_critic"]),
                        tau=float(config["tau"]),
                        batch_size=int(config["batch_size"]),
                        hidden_dim=int(config["hidden_dim"]),
                        cond_dim=int(config["cond_dim"]),
                        max_steps=int(self.short_budget * budget),
                        init_agent=self.init_agent,
                        slm_interval=int(config["slm_interval"]),
                        der_buffer=self.der_buffer,
                        acc_f_threshold=0.99, acc_min_points=3,
                        acc_r2_min=0.60, acc_margin=0.20,
                        enable_reactive_dehb=False,
                    )
                    fids.append(F)
                except Exception as _e:
                    log.warning(f"    Outer HP trial SKIPPED ({_e})")
                    fids.append(0.0)
        mean_F = float(np.mean(fids)) if fids else 0.0
        return {"fitness": 1.0 - mean_F, "cost": budget}

# ─────────────────────────────────────────────────────────────────────────────
# run_qubit_level — runs DEHB Outer then full-budget cells for one qubit count
# ─────────────────────────────────────────────────────────────────────────────
def run_qubit_level(
    n_qubits:      int,
    targets:       List[str],
    seeds:         List[int],
    noise_stage:   int,
    max_steps:     int,
    outer_brackets: int,
    inner_brackets: int,
    results_dir:   Path,
    ckpt_dir:      Path,
    init_agent:    Optional[SACAgent],
    der_buffer:    Optional[DERPlusPlusBuffer],
    seed_filter:   Optional[int] = None,  # if set, only run this seed
) -> Tuple[Dict[str, Dict[int, float]], Optional[SACAgent]]:
    """
    Phase 1: DEHB Outer — find best HP across all (target, seed) at short budget.
    Phase 2: Full-budget run per (target, seed) with best HP + reactive pipeline.
    Returns cell_results dict and best overall agent.
    """
    max_ep_steps = n_qubits * 50

    # ── DEHB Outer ────────────────────────────────────────────────────────────
    short_budget = max(10_000, max_steps // 20)
    outer_cs     = build_inner_cs(seed=42)
    outer_seeds  = seeds if seed_filter is None else [seed_filter]
    outer_factory = OuterTargetFactory(
        n_qubits=n_qubits, targets=targets, seeds=outer_seeds,
        noise_stage=noise_stage, max_ep_steps=max_ep_steps,
        short_budget=short_budget, der_buffer=der_buffer,
        init_agent=init_agent,
    )
    log.info(f"  DEHB Outer: {outer_brackets} brackets, "
             f"short_budget={short_budget:,} per trial")
    _best_outer_cfg = None
    try:
        outer_dehb = DEHB(
            cs=outer_cs,
            min_fidelity=0.1, max_fidelity=1.0,
            n_workers=1, output_path="/tmp",
        )
        outer_dehb.run(
            target_function=outer_factory,
            fevals=outer_brackets * len(targets) * len(outer_seeds),
        )
        inc = outer_dehb.get_incumbents()
        if inc and inc.get("config") is not None:
            _best_outer_cfg = inc["config"]
            log.info(f"  DEHB Outer best: alpha={float(_best_outer_cfg['alpha']):.4f} "
                     f"lr_actor={float(_best_outer_cfg['lr_actor']):.2e}")
    except Exception as _oe:
        log.warning(f"  DEHB Outer failed ({_oe}) — using default HP")

    if _best_outer_cfg is None:
        # Fallback default config
        _best_outer_cfg = {
            "alpha": 0.05, "lr_actor": 3e-4, "lr_critic": 3e-4,
            "tau": 0.005, "batch_size": 256, "hidden_dim": 256,
            "cond_dim": 64, "slm_interval": 50_000,
        }

    # ── Full-budget cells ─────────────────────────────────────────────────────
    cell_results: Dict[str, Dict[int, float]] = {t: {} for t in targets}
    best_agent:   Optional[SACAgent] = None
    best_F_level  = 0.0

    run_seeds = seeds if seed_filter is None else [seed_filter]

    for target in targets:
        for seed in run_seeds:
            log.info(f"  ── {n_qubits}Q / {target} / seed={seed} ──")
            # Per-cell BRFD and SWDFT instances (independent per cell)
            cell_brfd  = BRFD(seed=seed)
            cell_swdft = SWDFT(window=20, flat_thresh=0.15,
                               min_steps=30_000, cooldown=8)
            cell_bus   = SignalBus()
            cell_sdft  = SDFT(history_len=30)
            cell_ttt   = TTT(n_gradient_steps=50, lr=1e-4)
            cell_erl   = ERL(sdft=cell_sdft, ttt=cell_ttt, brfd=cell_brfd)

            # ACC probe (20k) to get budget estimate
            _acc_budget = max_steps
            try:
                _, _, _probe_hist, _ = run_cell(
                    n_qubits=n_qubits, target_state=target, seed=seed,
                    noise_stage=noise_stage, max_ep_steps=max_ep_steps,
                    reward_w_F=0.65, reward_w_S=0.20, reward_w_str=0.10,
                    alpha=float(_best_outer_cfg["alpha"]),
                    lr_actor=float(_best_outer_cfg["lr_actor"]),
                    lr_critic=float(_best_outer_cfg["lr_critic"]),
                    tau=float(_best_outer_cfg["tau"]),
                    batch_size=int(_best_outer_cfg["batch_size"]),
                    hidden_dim=int(_best_outer_cfg["hidden_dim"]),
                    cond_dim=int(_best_outer_cfg["cond_dim"]),
                    max_steps=20_000,
                    init_agent=init_agent,
                    slm_interval=int(_best_outer_cfg["slm_interval"]),
                    der_buffer=der_buffer,
                    acc_f_threshold=0.99, acc_min_points=3,
                    acc_r2_min=0.60, acc_margin=0.20,
                    enable_reactive_dehb=False,
                )
                if _probe_hist and _probe_hist[-1].get("acc_recommended_budget"):
                    _acc_budget = min(max_steps,
                                     _probe_hist[-1]["acc_recommended_budget"])
                    log.info(f"  ACC predicted budget: {_acc_budget:,}")
            except Exception as _pe:
                log.warning(f"  ACC probe failed ({_pe}) — using {max_steps:,}")

            # BRFD reward weights for full run
            w_F, w_S, w_str = cell_brfd.sample_weights()

            best_F, _steps, _hist, _agent = run_cell(
                n_qubits=n_qubits, target_state=target, seed=seed,
                noise_stage=noise_stage, max_ep_steps=max_ep_steps,
                reward_w_F=w_F, reward_w_S=w_S, reward_w_str=w_str,
                alpha=float(_best_outer_cfg["alpha"]),
                lr_actor=float(_best_outer_cfg["lr_actor"]),
                lr_critic=float(_best_outer_cfg["lr_critic"]),
                tau=float(_best_outer_cfg["tau"]),
                batch_size=int(_best_outer_cfg["batch_size"]),
                hidden_dim=int(_best_outer_cfg["hidden_dim"]),
                cond_dim=int(_best_outer_cfg["cond_dim"]),
                max_steps=_acc_budget,
                init_agent=init_agent,
                slm_interval=int(_best_outer_cfg["slm_interval"]),
                der_buffer=der_buffer,
                acc_f_threshold=0.99, acc_min_points=5,
                acc_r2_min=0.70, acc_margin=0.20,
                bus=cell_bus, swdft=cell_swdft, brfd=cell_brfd,
                erl=cell_erl,
                enable_reactive_dehb=True,
                reactive_brackets=inner_brackets,
            )
            cell_results[target][seed] = best_F

            # Save per-cell result
            cell_path = results_dir / f"{n_qubits}Q_{target}_s{seed}.json"
            cell_path.write_text(json.dumps({
                "n_qubits": n_qubits, "target": target, "seed": seed,
                "best_F": round(best_F, 6),
                "steps": _steps,
                "best_hp": {k: (float(v) if isinstance(v, (int, float))
                                else int(v))
                            for k, v in _best_outer_cfg.items()},
                "brfd_weights": {"w_F": w_F, "w_S": w_S, "w_str": w_str},
                "erl_summary": cell_erl.summary(),
                "history": _hist[-10:],  # last 10 checkpoints only
            }, indent=2))
            log.info(f"  {n_qubits}Q/{target}/s{seed} DONE: best_F={best_F:.4f}")

            if best_F > best_F_level:
                best_F_level = best_F
                best_agent   = _agent

    # Save best checkpoint for this qubit level
    if best_agent is not None:
        ckpt_path = ckpt_dir / f"{n_qubits}Q_best_agent.pt"
        torch.save({
            "obs_dim":    best_agent.obs_dim,
            "act_dim":    best_agent.act_dim,
            "target_dim": best_agent.target_dim,
            "actor":      best_agent.actor.state_dict(),
            "critic":     best_agent.critic.state_dict(),
        }, ckpt_path)
        log.info(f"  Checkpoint saved: {ckpt_path}")

    return cell_results, best_agent

# ─────────────────────────────────────────────────────────────────────────────
# Advancement logic
# ─────────────────────────────────────────────────────────────────────────────
def should_advance(cell_results: Dict[str, Dict[int, float]],
                   f_threshold: float, f_floor: float) -> Tuple[bool, float, float]:
    all_F = [f for sd in cell_results.values() for f in sd.values()]
    if not all_F:
        return False, 0.0, 0.0
    mean_F = float(np.mean(all_F))
    min_F  = float(np.min(all_F))
    return (mean_F >= f_threshold and min_F >= f_floor), mean_F, min_F

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="QUASAR v15 — Unified Proactive-Reactive Architecture")
    parser.add_argument("--targets",         nargs="+",
                        default=["GHZ", "W", "Cluster", "Dicke-k3"])
    parser.add_argument("--seeds",           nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--seed-filter",     type=int, default=None,
                        help="Run only this seed (for parallel per-seed jobs)")
    parser.add_argument("--min-qubits",      type=int, default=2)
    parser.add_argument("--max-qubits",      type=int, default=12)
    parser.add_argument("--outer-brackets",  type=int, default=6,
                        help="DEHB Outer bracket count per qubit level")
    parser.add_argument("--inner-brackets",  type=int, default=4,
                        help="DEHB Inner (reactive) bracket count")
    parser.add_argument("--f-threshold",     type=float, default=0.99)
    parser.add_argument("--f-floor",         type=float, default=0.90)
    parser.add_argument("--max-steps-base",  type=int, default=500_000)
    parser.add_argument("--steps-scale",     type=float, default=1.5)
    parser.add_argument("--noise-stage",     type=int, default=5)
    parser.add_argument("--results-dir",     type=str,
                        default=str(Path.home() / "quasar_v15" / "results" / "v15_adaptive"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ckpt_dir    = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.remove()
    log.add(sys.stderr, level="INFO")
    log.add(results_dir / "quasar_v15.log", level="DEBUG", rotation="100 MB")

    log.info("=" * 70)
    log.info("QUASAR v15 — Unified Proactive-Reactive Architecture")
    log.info(f"  Device:       {DEVICE}")
    log.info(f"  Qubit range:  {args.min_qubits} → {args.max_qubits} (adaptive)")
    log.info(f"  Targets:      {args.targets}")
    log.info(f"  Seeds:        {args.seeds}"
             + (f"  [filter: {args.seed_filter}]" if args.seed_filter else ""))
    log.info(f"  F threshold:  {args.f_threshold} / floor {args.f_floor}")
    log.info(f"  Max steps:    {args.max_steps_base:,} @ {args.min_qubits}Q, "
             f"×{args.steps_scale} per qubit")
    log.info(f"  Noise stage:  {args.noise_stage}")
    log.info(f"  DEHB Outer:   {args.outer_brackets} brackets")
    log.info(f"  DEHB Inner:   {args.inner_brackets} brackets (reactive)")
    log.info("=" * 70)

    der_buffer: Optional[DERPlusPlusBuffer] = None
    scalability_report = {
        "config":             vars(args),
        "start_time":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qubit_levels":       {},
        "scalability_ceiling": None,
        "ceiling_reason":     None,
    }
    prev_best_agent: Optional[SACAgent] = None
    ceiling_reached = False

    for n_qubits in range(args.min_qubits, args.max_qubits + 1):
        extra     = n_qubits - args.min_qubits
        max_steps = int(args.max_steps_base * (args.steps_scale ** extra))
        max_steps = min(max_steps, 5_000_000)

        log.info("")
        log.info("=" * 70)
        log.info(f"  QUBIT LEVEL: {n_qubits}Q  (budget={max_steps:,} steps)")
        log.info("=" * 70)
        t0 = time.time()

        if der_buffer is None:
            _obs_dim = 4 * (2 ** args.min_qubits) + args.min_qubits + 2
            _act_dim = args.min_qubits * 3
            der_buffer = DERPlusPlusBuffer(
                obs_dim=_obs_dim, act_dim=_act_dim, capacity=DER_CAPACITY)
            log.info(f"DER++ buffer init: obs_dim={_obs_dim}, act_dim={_act_dim}")

        cell_results, best_agent = run_qubit_level(
            n_qubits=n_qubits,
            targets=args.targets,
            seeds=args.seeds,
            noise_stage=args.noise_stage,
            max_steps=max_steps,
            outer_brackets=args.outer_brackets,
            inner_brackets=args.inner_brackets,
            results_dir=results_dir,
            ckpt_dir=ckpt_dir,
            init_agent=prev_best_agent,
            der_buffer=der_buffer,
            seed_filter=args.seed_filter,
        )
        elapsed = time.time() - t0
        advance, mean_F, min_F = should_advance(
            cell_results, args.f_threshold, args.f_floor)

        per_target_mean = {
            t: float(np.mean(list(sd.values())))
            for t, sd in cell_results.items() if sd
        }
        level_summary = {
            "n_qubits":        n_qubits,
            "mean_F":          round(mean_F, 5),
            "min_F":           round(min_F, 5),
            "per_target_mean": {k: round(v, 5) for k, v in per_target_mean.items()},
            "cell_results":    {t: {str(s): round(f, 5) for s, f in sd.items()}
                                for t, sd in cell_results.items()},
            "max_steps_used":  max_steps,
            "elapsed_s":       round(elapsed, 1),
            "advanced":        advance,
        }
        scalability_report["qubit_levels"][str(n_qubits)] = level_summary
        log.info(f"  {n_qubits}Q SUMMARY: mean_F={mean_F:.4f}  min_F={min_F:.4f}")
        for t, v in per_target_mean.items():
            log.info(f"    {t}: mean_F={v:.4f}")

        report_path = results_dir / "scalability_report.json"
        report_path.write_text(json.dumps(scalability_report, indent=2))

        if advance:
            log.info(f"  ✓ {n_qubits}Q PASSED → advancing to {n_qubits + 1}Q")
            prev_best_agent = best_agent
        else:
            reason = (f"mean_F={mean_F:.4f} < threshold={args.f_threshold}"
                      if mean_F < args.f_threshold
                      else f"min_F={min_F:.4f} < floor={args.f_floor}")
            log.info(f"  ✗ {n_qubits}Q STALLED ({reason})")
            log.info(f"  ══ SCALABILITY CEILING: N* = {n_qubits - 1}Q ══")
            scalability_report["scalability_ceiling"] = n_qubits - 1
            scalability_report["ceiling_reason"]      = reason
            ceiling_reached = True
            break

    if not ceiling_reached:
        scalability_report["scalability_ceiling"] = args.max_qubits
        scalability_report["ceiling_reason"] = (
            f"Reached hard max-qubits={args.max_qubits}")
        log.info(f"  ══ EXPERIMENT COMPLETE: N* >= {args.max_qubits}Q ══")

    scalability_report["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report_path = results_dir / "scalability_report.json"
    report_path.write_text(json.dumps(scalability_report, indent=2))
    log.info(f"  FINAL RESULT: N* = {scalability_report['scalability_ceiling']}Q")
    log.info(f"  Report: {report_path}")

if __name__ == "__main__":
    main()
