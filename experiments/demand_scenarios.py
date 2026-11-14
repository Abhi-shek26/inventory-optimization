"""Demand processes for the three experiments.

Exp 1 -- stationary demand:        Poisson(6) every period (the training world).
Exp 2 -- sudden demand change:     Poisson(6) for the first half, Poisson(10) after.
Exp 3 -- high-variance demand:     alternating 6 / 10 / 2 days (same mean, more risk).

Each generator returns (demand_path, window) where `window` is the slice of
periods where the change happens -- used for the "adaptation performance" metric.
"""

from __future__ import annotations

import math

import numpy as np


def truncated_poisson_pmf(lam, values):
    weights = np.array([math.exp(-lam) * lam ** int(v) / math.factorial(int(v))
                        for v in values], dtype=float)
    return weights / weights.sum()


def sample_path(rng, values, pmf, horizon):
    return rng.choice(values, size=horizon, p=pmf).astype(int)


def stationary(rng, cfg, values, pmf):
    path = sample_path(rng, values, pmf, cfg.horizon)
    return path, slice(cfg.horizon // 2, cfg.horizon)


def step_change(rng, cfg, values, pmf, low=6.0, high=10.0, switch_at=0.5):
    """Demand jumps mid-episode; the second regime is never seen in training."""
    switch = int(round(cfg.horizon * switch_at))
    pmf_low = truncated_poisson_pmf(low, values)
    pmf_high = truncated_poisson_pmf(high, values)
    first = sample_path(rng, values, pmf_low, switch)
    rest = sample_path(rng, values, pmf_high, cfg.horizon - switch)
    return np.concatenate([first, rest]), slice(switch, cfg.horizon)


def high_variance(rng, cfg, values, pmf, pattern=(6, 10, 2)):
    """Same average demand, much larger day-to-day swings."""
    pmfs = [truncated_poisson_pmf(float(m), values) for m in pattern]
    path = np.array([rng.choice(values, p=pmfs[t % len(pattern)])
                     for t in range(cfg.horizon)], dtype=int)
    return path, slice(0, cfg.horizon)


SCENARIOS = {
    "Exp1_stationary": stationary,
    "Exp2_step_change": step_change,
    "Exp3_high_variance": high_variance,
}

SCENARIO_LABELS = {
    "Exp1_stationary": "Exp 1 - stationary demand (Poisson 6)",
    "Exp2_step_change": "Exp 2 - sudden demand jump 6 -> 10",
    "Exp3_high_variance": "Exp 3 - high-variance demand (6/10/2 pattern)",
}
