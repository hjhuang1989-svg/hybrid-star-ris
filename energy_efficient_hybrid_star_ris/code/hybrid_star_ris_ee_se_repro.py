#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible simulator for
"Energy-Efficient Hybrid STAR-RIS via Sparse Active Selection and Gain Control"

This implementation evaluates the post-ZF scalar surrogate used in the paper.
It generates the energy-splitting (ES) boundary study, feasibility probabilities,
baseline comparisons, power distributions, and hardware-sensitivity curves
reported in the manuscript. The active baseline uses the same power-minimizing
gain update as the proposed scheme; a gain-saturated active benchmark is
intentionally not mixed into the main comparison.
"""

from __future__ import annotations

import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np


# =========================
# Simulation configuration
# =========================
@dataclass
class SimConfig:
    """System and sweep parameters used by the reproducibility script."""

    N: int = 24
    Kt: int = 2
    Kr: int = 2
    fading: str = "nakagami"
    m_br: float = 1.0
    m_ru: float = 1.0
    pl_br: float = 0.20
    pl_t: Tuple[float, float] = (0.060, 0.050)
    pl_r: Tuple[float, float] = (0.080, 0.060)

    rho_t: float = 0.55
    rho_r: float = 0.45
    beta_max: float = 5.0

    sigma0: float = 0.25
    sigma_amp: float = 0.05

    P_ctrl: float = 0.10
    P_bias: float = 0.015
    f_update: float = 50.0
    E_sw_base: float = 2.0e-4
    E_sw_act: float = 1.2e-4
    P_amp: float = 0.0025
    eta_pa: float = 0.40
    Ptx_max: float = 5.0

    L_uniform: int = 6
    L_step: int = 3
    random_repeats: int = 1
    split_grid: Tuple[float, ...] = tuple(np.linspace(0.2, 0.8, 13))
    R_targets: Tuple[int, ...] = tuple(range(2, 22, 2))

    num_mc: int = 10000
    num_workers: int = os.cpu_count() or 1

    lambda1: Optional[float] = None
    lambda2: Optional[float] = None


def eq23_weights(cfg: SimConfig) -> Tuple[float, float]:
    """Return the two weights used in the practical ranking score.

    If not supplied explicitly, the defaults use the inverse receiver-noise
    power and the inverse BS power budget, respectively.
    """

    lambda1 = cfg.lambda1 if cfg.lambda1 is not None else 1.0 / (cfg.sigma0 ** 2)
    lambda2 = cfg.lambda2 if cfg.lambda2 is not None else 1.0 / cfg.Ptx_max
    return float(lambda1), float(lambda2)


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    """Return mean, spread, standard error, and 95% CI half-width."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "ci95": float("nan")}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(std / np.sqrt(arr.size)) if arr.size > 0 else float("nan")
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "se": se,
        "ci95": float(1.959963984540054 * se),
    }


def runtime_environment(cfg: SimConfig) -> Dict[str, object]:
    """Capture the execution environment used for the reproducibility run."""

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "num_workers": int(cfg.num_workers),
        "mc_realizations": int(cfg.num_mc),
    }


def complex_fading(shape: Sequence[int], fading: str, m: float, avg_power: float, rng: np.random.Generator) -> np.ndarray:
    """Generate complex fading coefficients.

    Envelope power is sampled from a Gamma distribution and combined with a
    uniformly distributed phase.  With ``fading="rayleigh"``, the Nakagami
    parameter is reduced to ``m=1``.
    """

    if fading.lower() == "rayleigh":
        m = 1.0
    power = rng.gamma(shape=m, scale=avg_power / m, size=shape)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    return np.sqrt(power) * np.exp(1j * phase)


def generate_channels(cfg: SimConfig, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one BS-RIS and RIS-user channel realization."""

    rng = np.random.default_rng(seed)
    g = complex_fading((cfg.N,), cfg.fading, cfg.m_br, cfg.pl_br, rng)
    h_t = np.vstack([complex_fading((cfg.N,), cfg.fading, cfg.m_ru, cfg.pl_t[k], rng) for k in range(cfg.Kt)])
    h_r = np.vstack([complex_fading((cfg.N,), cfg.fading, cfg.m_ru, cfg.pl_r[k], rng) for k in range(cfg.Kr)])
    return g, h_t, h_r


# =========================
# Protocols and weighting rules
# =========================
def protocol_rho_tau(protocol: str, cfg: SimConfig) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return effective splitting coefficients and time weights for the ES protocol."""

    mode = protocol.upper()
    if mode != "ES":
        raise ValueError(f"Unsupported protocol in the ES-focused script: {protocol}")
    return (cfg.rho_t, cfg.rho_r), (1.0, 1.0)


def side_rate_weights(alpha: float) -> Tuple[float, float]:
    """Convert the sum-rate splitting variable alpha into side-rate weights."""

    eps = 1e-6
    return float(max(alpha, eps)), float(max(1.0 - alpha, eps))


def eq23_side_score(g: np.ndarray, h_side: np.ndarray, side_weight: float, rho_s_eff: float, cfg: SimConfig) -> np.ndarray:
    """Compute the normalized practical ranking score.

    The numerator measures the effective coherent-gain potential for the
    weakest user on the considered side.  The denominator accounts for the
    normalized amplifier-noise, bias-power, and switching-power costs.  This
    score is used only for ranking; each candidate active set is then checked
    through the surrogate power model and feasibility constraints.
    """

    lambda1, lambda2 = eq23_weights(cfg)
    abs_g = np.abs(g)
    min_abs_h = np.min(np.abs(h_side), axis=0)
    mean_abs_h2 = np.mean(np.abs(h_side) ** 2, axis=0)
    numerator = side_weight * np.sqrt(rho_s_eff) * abs_g * min_abs_h
    denominator = lambda1 * rho_s_eff * (cfg.sigma_amp ** 2) * mean_abs_h2 + lambda2 * (cfg.P_bias + cfg.f_update * cfg.E_sw_act)
    return numerator / (denominator + 1e-12)


def build_active_sets(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, L: int, alpha: float, cfg: SimConfig, protocol: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build active-element sets for the proposed cost-aware sparse design.

    The score is computed separately for the transmission and reflection
    sides.  The L elements with the largest side-wise score are selected and
    assigned to the side on which they score higher.
    """

    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(protocol, cfg)
    weight_t, weight_r = side_rate_weights(alpha)
    xi_t = eq23_side_score(g, h_t, weight_t, rho_t_eff, cfg)
    xi_r = eq23_side_score(g, h_r, weight_r, rho_r_eff, cfg)
    best_side = np.where(xi_t >= xi_r, 0, 1)
    best_score = np.maximum(xi_t, xi_r)
    selected = np.argsort(-best_score)[:L] if L > 0 else np.array([], dtype=int)
    A_t = np.sort(selected[best_side[selected] == 0])
    A_r = np.sort(selected[best_side[selected] == 1])
    return A_t, A_r


def build_uniform_active_sets(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, L: int, alpha: float, cfg: SimConfig, protocol: str) -> Tuple[np.ndarray, np.ndarray]:
    """Select uniformly spaced active elements for the uniform baseline."""

    if L <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    idx = np.linspace(0, cfg.N - 1, L, dtype=int)
    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(protocol, cfg)
    weight_t, weight_r = side_rate_weights(alpha)
    xi_t = eq23_side_score(g, h_t, weight_t, rho_t_eff, cfg)
    xi_r = eq23_side_score(g, h_r, weight_r, rho_r_eff, cfg)
    best_side = np.where(xi_t[idx] >= xi_r[idx], 0, 1)
    return np.sort(idx[best_side == 0]), np.sort(idx[best_side == 1])


def build_strongest_active_sets(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, L: int, alpha: float, cfg: SimConfig, protocol: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build a channel-strength-only greedy baseline.

    This baseline uses the same L/alpha grid as the proposed method but ranks
    elements only by side-rate-weighted coherent gain.  It deliberately
    ignores amplifier-noise, bias, and switching costs.
    """

    if L <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(protocol, cfg)
    weight_t, weight_r = side_rate_weights(alpha)
    abs_g = np.abs(g)
    pi_t = weight_t * np.sqrt(rho_t_eff) * abs_g * np.min(np.abs(h_t), axis=0)
    pi_r = weight_r * np.sqrt(rho_r_eff) * abs_g * np.min(np.abs(h_r), axis=0)
    best_side = np.where(pi_t >= pi_r, 0, 1)
    selected = np.argsort(-np.maximum(pi_t, pi_r))[:L]
    A_t = np.sort(selected[best_side[selected] == 0])
    A_r = np.sort(selected[best_side[selected] == 1])
    return A_t, A_r


def _random_baseline_seed(g: np.ndarray, L: int, alpha: float, repeat: int, protocol: str) -> int:
    """Create a deterministic seed for the single-random sparse baseline."""

    # The hash depends on the channel realization and design point, keeping the
    # random baseline reproducible without passing a separate seed through the
    # boundary-search interface.
    vals = np.round(np.abs(g[: min(8, len(g))]) * 1e9).astype(np.uint64)
    modulus = 2**64
    acc = 1469598103934665603
    for v in vals:
        acc = (acc ^ int(v)) % modulus
        acc = (acc * 1099511628211) % modulus
    acc ^= int(L * 1009 + round(alpha * 1000) * 917 + repeat * 131)
    acc ^= sum(ord(ch) for ch in protocol.upper()) * 65537
    return int(acc % (2**32 - 1))


def build_random_active_sets(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, L: int, alpha: float, cfg: SimConfig, protocol: str, repeat: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Build a reproducible single-random sparse active-set baseline.

    The support is random, while each selected element is assigned to the side
    where it has the larger cost-aware score.  The default configuration uses
    one random repetition, so this is a lightweight sanity baseline rather than
    a heavily optimized random search.
    """

    if L <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    rng = np.random.default_rng(_random_baseline_seed(g, L, alpha, repeat, protocol))
    selected = np.sort(rng.choice(cfg.N, size=L, replace=False))
    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(protocol, cfg)
    weight_t, weight_r = side_rate_weights(alpha)
    xi_t = eq23_side_score(g, h_t, weight_t, rho_t_eff, cfg)
    xi_r = eq23_side_score(g, h_r, weight_r, rho_r_eff, cfg)
    best_side = np.where(xi_t[selected] >= xi_r[selected], 0, 1)
    return np.sort(selected[best_side == 0]), np.sort(selected[best_side == 1])


def side_params(g: np.ndarray, h_side: np.ndarray, passive_idx: np.ndarray, active_idx: np.ndarray, rho_eff: float, sigma0: float, sigma_amp: float) -> Tuple[float, float, float, float]:
    """Extract the four scalar parameters a, b, c, and d.

    ``a`` is the worst-user passive-link coherent gain.  ``b`` is the
    worst-user active-link coherent gain at unit gain.  ``c`` is receiver
    thermal noise.  ``d`` is the worst-user amplifier-noise coefficient after
    the RIS-user channel.

    True zero values are returned instead of epsilon-lifted values so that
    fully passive, fully active, b_s=0, and d_s=0 boundary cases match the
    case classification in the manuscript.  Numerical stability is handled in
    the downstream cost and SINR denominators.
    """

    abs_g = np.abs(g)
    abs_h = np.abs(h_side)
    if len(passive_idx) > 0:
        passive_sum = np.sum(np.sqrt(rho_eff) * abs_g[None, passive_idx] * abs_h[:, passive_idx], axis=1)
    else:
        passive_sum = np.zeros(h_side.shape[0])
    if len(active_idx) > 0:
        active_unit = np.sum(np.sqrt(rho_eff) * abs_g[None, active_idx] * abs_h[:, active_idx], axis=1)
        amp_noise_coeff = np.sum(rho_eff * abs_h[:, active_idx] ** 2 * sigma_amp ** 2, axis=1)
    else:
        active_unit = np.zeros(h_side.shape[0])
        amp_noise_coeff = np.zeros(h_side.shape[0])
    a = float(np.min(passive_sum))
    b = float(np.min(active_unit))
    c = float(sigma0 ** 2)
    d = float(np.max(amp_noise_coeff))
    return a, b, c, d


def beta_se_closed_form(a: float, b: float, c: float, d: float, beta_max: float) -> float:
    """Compute the SE-optimal beta from Proposition 1 and boundary cases.

    This is not the power-minimizing beta used for EE/Ptot minimization.  It
    is kept only as a reference or one-dimensional-search initializer.
    """

    if b <= 1e-12:
        return 1.0
    # Fully active a_s=0 or noiseless-amplifier d_s=0: the SE surrogate is
    # nondecreasing in beta_s, so the SE-optimal gain is the cap.
    if a <= 1e-12 or d <= 1e-12:
        return float(beta_max)
    beta = b * c / (a * d)
    return float(np.clip(beta, 1.0, beta_max))


def side_cost_scalar(beta: float, a: float, b: float, c: float, d: float, rate_side: float, K: int, Ls: int, mu_s: float, eta_pa: float) -> float:
    """Side-wise scalar cost Psi_s(beta_s) used for gain refinement."""

    if K <= 0:
        return 0.0
    snr_req = 2.0 ** (rate_side / K) - 1.0
    denom = (a + b * beta) ** 2 + 1e-15
    tx_part = K * snr_req * (c + d * beta ** 2) / (eta_pa * denom)
    amp_part = mu_s * Ls * max(beta ** 2 - 1.0, 0.0)
    return float(tx_part + amp_part)


def minimize_beta_power(a: float, b: float, c: float, d: float, rate_side: float, K: int, Ls: int, mu_s: float, eta_pa: float, beta_max: float) -> float:
    """Minimize Psi_s(beta_s) over [1, beta_max].

    A stable golden-section search is used, and both endpoints are checked
    explicitly.  This keeps the implementation aligned with the manuscript:
    the frontier approximation minimizes Psi_s instead of directly using the
    SE-optimal closed form.  Fully active and d_s=0 boundary cases are handled
    by the same one-dimensional search.  The gain is set to one only when no
    active element exists or the active coherent gain is zero.
    """

    if Ls <= 0 or b <= 1e-12:
        return 1.0
    lo, hi = 1.0, float(beta_max)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = side_cost_scalar(x1, a, b, c, d, rate_side, K, Ls, mu_s, eta_pa)
    f2 = side_cost_scalar(x2, a, b, c, d, rate_side, K, Ls, mu_s, eta_pa)
    for _ in range(24):
        if f1 > f2:
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = side_cost_scalar(x2, a, b, c, d, rate_side, K, Ls, mu_s, eta_pa)
        else:
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = side_cost_scalar(x1, a, b, c, d, rate_side, K, Ls, mu_s, eta_pa)
    candidates = [1.0, beta_max, 0.5 * (lo + hi), beta_se_closed_form(a, b, c, d, beta_max)]
    return float(min(candidates, key=lambda x: side_cost_scalar(x, a, b, c, d, rate_side, K, Ls, mu_s, eta_pa)))


def exact_first_order_activation_score(delta_u: float, delta_d: float, a: float, b: float, c: float, d: float, beta: float, rate_side: float, K: int, P_bias: float, f_update: float, E_sw_act: float, mu_s: float, eta_pa: float) -> float:
    """Return the exact first-order activation score.

    A score larger than one means that activating the candidate element would
    reduce the side cost under the first-order approximation.
    """

    C_s = K * (2.0 ** (rate_side / K) - 1.0) / eta_pa
    denom_gain = (a + b * beta)
    kappa1 = 2.0 * C_s * (c + d * beta ** 2) * (beta - 1.0) / (denom_gain ** 3 + 1e-15)
    kappa2 = C_s * beta ** 2 / (denom_gain ** 2 + 1e-15)
    denominator = kappa2 * delta_d + P_bias + f_update * E_sw_act + mu_s * max(beta ** 2 - 1.0, 0.0)
    return float(kappa1 * delta_u / (denominator + 1e-15))


def side_required_power(g: np.ndarray, h_side: np.ndarray, passive_idx: np.ndarray, active_idx: np.ndarray, beta: float, rate_target_side: float, rho_eff: float, sigma0: float, sigma_amp: float) -> float:
    """Compute the transmit power required to meet one side-rate target.

    The side target is split equally across users.  The required SNR is then
    converted into per-user transmit-power requirements and summed across the
    side.
    """

    K = h_side.shape[0]
    if K == 0:
        return 0.0
    r_each = rate_target_side / K
    snr_req = 2 ** r_each - 1.0
    abs_g = np.abs(g)
    reqs = []
    for k in range(K):
        eff = np.sum(np.sqrt(rho_eff) * abs_g[passive_idx] * np.abs(h_side[k, passive_idx])) if len(passive_idx) > 0 else 0.0
        if len(active_idx) > 0:
            eff += np.sum(np.sqrt(rho_eff) * beta * abs_g[active_idx] * np.abs(h_side[k, active_idx]))
            noise = sigma0 ** 2 + np.sum(rho_eff * np.abs(h_side[k, active_idx]) ** 2 * sigma_amp ** 2 * beta ** 2)
        else:
            noise = sigma0 ** 2
        reqs.append(snr_req * noise / (eff ** 2 + 1e-15))
    return float(np.sum(reqs))


def protocol_total_power(P_t_req: float, P_r_req: float, Lt: int, Lr: int, beta_t: float, beta_r: float, cfg: SimConfig, protocol: str) -> Tuple[float, float, float, Dict[str, float]]:
    """Aggregate ES total power and return static, dynamic, BS, and amplifier parts."""

    mode = protocol.upper()
    if mode != "ES":
        raise ValueError(f"Unsupported protocol in the ES-focused script: {protocol}")
    static = cfg.P_ctrl + cfg.P_bias * (Lt + Lr)
    dynamic = cfg.f_update * (cfg.E_sw_base + cfg.E_sw_act * (Lt + Lr))
    Ptx_avg = P_t_req + P_r_req
    Ptx_peak = Ptx_avg
    bs = Ptx_avg / cfg.eta_pa
    amp = cfg.P_amp * (Lt * max(beta_t ** 2 - 1.0, 0.0) + Lr * max(beta_r ** 2 - 1.0, 0.0))
    Ptot = static + dynamic + bs + amp
    return Ptot, Ptx_avg, Ptx_peak, {"static": static, "dynamic": dynamic, "bs": bs, "amp": amp}


# =========================
# Candidate evaluation and boundary-point search
# =========================
def evaluate_candidate(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, A_t: np.ndarray, A_r: np.ndarray, Rtarget: float, alpha: float, cfg: SimConfig, protocol: str = "ES") -> Optional[Dict[str, float]]:
    """Evaluate one ES active-set split at a target sum-SE.

    The inputs are fixed active sets A_t/A_r and a side-rate splitting variable alpha.  The
    output includes total power, transmit power, optimized side-wise gains, and
    power breakdown.  If the peak BS power exceeds the budget, ``None`` is
    returned.
    """

    mode = protocol.upper()
    if mode != "ES":
        raise ValueError(f"Unsupported protocol in the ES-focused script: {protocol}")
    if np.intersect1d(A_t, A_r).size > 0:
        raise ValueError("Transmission and reflection active sets must be disjoint.")
    Rt = alpha * Rtarget
    Rr = (1.0 - alpha) * Rtarget
    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(mode, cfg)
    passive_t = np.setdiff1d(np.arange(cfg.N), A_t)
    passive_r = np.setdiff1d(np.arange(cfg.N), A_r)
    a_t, b_t, c_t, d_t = side_params(g, h_t, passive_t, A_t, rho_t_eff, cfg.sigma0, cfg.sigma_amp)
    a_r, b_r, c_r, d_r = side_params(g, h_r, passive_r, A_r, rho_r_eff, cfg.sigma0, cfg.sigma_amp)
    beta_t = minimize_beta_power(a_t, b_t, c_t, d_t, Rt, cfg.Kt, len(A_t), cfg.P_amp, cfg.eta_pa, cfg.beta_max)
    beta_r = minimize_beta_power(a_r, b_r, c_r, d_r, Rr, cfg.Kr, len(A_r), cfg.P_amp, cfg.eta_pa, cfg.beta_max)
    if not (1.0 <= beta_t <= cfg.beta_max and 1.0 <= beta_r <= cfg.beta_max):
        raise ValueError("Optimized gains must remain inside [1, beta_max].")
    P_t = side_required_power(g, h_t, passive_t, A_t, beta_t, Rt, rho_t_eff, cfg.sigma0, cfg.sigma_amp)
    P_r = side_required_power(g, h_r, passive_r, A_r, beta_r, Rr, rho_r_eff, cfg.sigma0, cfg.sigma_amp)
    Ptot, Ptx_avg, Ptx_peak, breakdown = protocol_total_power(P_t, P_r, len(A_t), len(A_r), beta_t, beta_r, cfg, mode)
    if Ptx_peak > cfg.Ptx_max:
        return None
    EE = Rtarget / Ptot
    out = {
        "R": float(Rtarget),
        "EE": float(EE),
        "Ptx": float(Ptx_avg),
        "Ptx_peak": float(Ptx_peak),
        "Ptot": float(Ptot),
        "Lt": int(len(A_t)),
        "Lr": int(len(A_r)),
        "L": int(len(A_t) + len(A_r)),
        "A_t": [int(i) for i in A_t.tolist()],
        "A_r": [int(i) for i in A_r.tolist()],
        "beta_t": float(beta_t),
        "beta_r": float(beta_r),
        "beta_avg": float(0.5 * (beta_t + beta_r)),
        "alpha": float(alpha),
        "Rt": float(Rt),
        "Rr": float(Rr),
        "protocol": mode,
    }
    out.update(breakdown)
    return out


def solve_boundary_point(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, Rtarget: float, cfg: SimConfig, scheme: str = "proposed", protocol: str = "ES") -> Optional[Dict[str, float]]:
    """Search the minimum-power point for a scheme at a fixed target SE.

    The search variables are the total active-cardinality L, the side-rate
    splitting variable alpha, and the active-element assignment rule determined by the
    selected scheme.  The feasible candidate with the smallest total power is
    retained.
    """

    N = len(g)
    selected_candidate: Optional[Dict[str, float]] = None
    if scheme == "passive":
        L_candidates = [0]
    elif scheme == "active":
        L_candidates = [N]
    elif scheme == "uniform":
        L_candidates = [cfg.L_uniform]
    else:
        L_candidates = list(range(0, N + 1, cfg.L_step))

    for alpha in cfg.split_grid:
        for L in L_candidates:
            if scheme == "random":
                repeats = range(max(1, int(cfg.random_repeats)))
            else:
                repeats = range(1)
            for repeat in repeats:
                if scheme == "passive":
                    A_t = np.array([], dtype=int)
                    A_r = np.array([], dtype=int)
                elif scheme in ("uniform", "uniform_opt"):
                    A_t, A_r = build_uniform_active_sets(g, h_t, h_r, L, alpha, cfg, protocol)
                elif scheme == "random":
                    A_t, A_r = build_random_active_sets(g, h_t, h_r, L, alpha, cfg, protocol, repeat=repeat)
                elif scheme == "strongest":
                    A_t, A_r = build_strongest_active_sets(g, h_t, h_r, L, alpha, cfg, protocol)
                elif scheme == "proposed":
                    # The proposed sparse selection rule evaluates two candidate
                    # ranking rules: the cost-aware score and its
                    # channel-strength-only counterpart.  The final choice is
                    # made by the same surrogate feasibility and total-power
                    # evaluation.
                    candidate_sets = (
                        build_active_sets(g, h_t, h_r, L, alpha, cfg, protocol),
                        build_strongest_active_sets(g, h_t, h_r, L, alpha, cfg, protocol),
                    )
                    for A_t_pool, A_r_pool in candidate_sets:
                        res = evaluate_candidate(g, h_t, h_r, A_t_pool, A_r_pool, Rtarget, alpha, cfg, protocol)
                        if res is None:
                            continue
                        if selected_candidate is None or res["Ptot"] < selected_candidate["Ptot"]:
                            selected_candidate = res
                    continue
                else:
                    A_t, A_r = build_active_sets(g, h_t, h_r, L, alpha, cfg, protocol)
                res = evaluate_candidate(g, h_t, h_r, A_t, A_r, Rtarget, alpha, cfg, protocol)
                if res is None:
                    continue
                if selected_candidate is None or res["Ptot"] < selected_candidate["Ptot"]:
                    selected_candidate = res
    return selected_candidate


# =========================
# Monte Carlo averaging and sensitivity analysis
# =========================
def _boundary_worker(args: Tuple[int, SimConfig, str, Tuple[str, ...]]) -> Dict[str, List[Optional[Dict[str, float]]]]:
    """Worker for one Monte Carlo channel realization."""

    seed, cfg, protocol, schemes = args
    g, h_t, h_r = generate_channels(cfg, seed=seed)
    return {
        scheme: [solve_boundary_point(g, h_t, h_r, R, cfg, scheme, protocol) for R in cfg.R_targets]
        for scheme in schemes
    }


def _proposed_fixed_rate_worker(args: Tuple[int, SimConfig, float, str]) -> Optional[Dict[str, float]]:
    """Worker for one proposed-design realization at a fixed rate target."""

    seed, cfg, Rtarget, protocol = args
    g, h_t, h_r = generate_channels(cfg, seed=seed)
    return solve_boundary_point(g, h_t, h_r, Rtarget, cfg, "proposed", protocol)


def _proposed_fixed_rate_indexed_worker(args: Tuple[int, int, SimConfig, float, str]) -> Tuple[int, Optional[Dict[str, float]]]:
    """Worker for batched sensitivity sweeps.

    The first returned value is the sweep-index, preserving deterministic
    grouping even when tasks are evaluated in parallel.
    """

    idx, seed, cfg, Rtarget, protocol = args
    g, h_t, h_r = generate_channels(cfg, seed=seed)
    return idx, solve_boundary_point(g, h_t, h_r, Rtarget, cfg, "proposed", protocol)


def _proposed_power_trace_worker(args: Tuple[int, SimConfig, float, str]) -> List[Optional[float]]:
    """Return the best-so-far proposed-search total-power trace for one channel."""

    seed, cfg, Rtarget, protocol = args
    g, h_t, h_r = generate_channels(cfg, seed=seed)
    n_elem = len(g)
    l_candidates = list(range(0, n_elem + 1, cfg.L_step))

    trace: List[Optional[float]] = []
    best = np.inf
    for alpha in cfg.split_grid:
        for active_count in l_candidates:
            candidate_sets = (
                build_active_sets(g, h_t, h_r, active_count, alpha, cfg, protocol),
                build_strongest_active_sets(g, h_t, h_r, active_count, alpha, cfg, protocol),
            )
            for a_t, a_r in candidate_sets:
                res = evaluate_candidate(g, h_t, h_r, a_t, a_r, Rtarget, alpha, cfg, protocol)
                if res is not None and float(res["Ptot"]) < best:
                    best = float(res["Ptot"])
                trace.append(None if not np.isfinite(best) else float(best))
    return trace


def _ordered_parallel_map(func, tasks: List[Tuple], max_workers: int) -> List:
    """Run a deterministic ordered parallel map with a serial fallback."""

    if max_workers <= 1 or len(tasks) <= 1:
        return [func(t) for t in tasks]
    # Linux/fork keeps the script runnable without requiring importable package
    # installation.  If fork is unavailable, fall back to the platform default.
    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            return list(ex.map(func, tasks))
    except (ValueError, RuntimeError):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(func, tasks))


def average_boundary(cfg: SimConfig, protocol: str = "ES", schemes: Tuple[str, ...] = ("passive", "uniform", "uniform_opt", "random", "strongest", "active", "proposed")) -> Tuple[Dict[str, List[Optional[Dict[str, float]]]], Dict[str, List[List[Optional[Dict[str, float]]]]]]:
    """Average all target-rate points over Monte Carlo channel realizations."""

    raw: Dict[str, List[List[Optional[Dict[str, float]]]]] = {s: [[] for _ in cfg.R_targets] for s in schemes}
    seeds = list(range(1, cfg.num_mc + 1))
    tasks = [(seed, cfg, protocol, schemes) for seed in seeds]
    max_workers = max(1, min(int(cfg.num_workers), cfg.num_mc))
    mc_outputs = _ordered_parallel_map(_boundary_worker, tasks, max_workers)

    for mc_out in mc_outputs:
        for scheme in schemes:
            for i, res in enumerate(mc_out[scheme]):
                raw[scheme][i].append(res)

    avg: Dict[str, List[Optional[Dict[str, float]]]] = {}
    for scheme in schemes:
        pts: List[Optional[Dict[str, float]]] = []
        for i, R in enumerate(cfg.R_targets):
            feas = [r for r in raw[scheme][i] if r is not None]
            if not feas:
                pts.append(None)
                continue
            keys = ["EE", "Ptx", "Ptx_peak", "Ptot", "L", "Lt", "Lr", "beta_t", "beta_r", "beta_avg", "alpha", "static", "dynamic", "bs", "amp"]
            entry = {"R": float(R), "protocol": protocol.upper(), "n_total": int(cfg.num_mc), "n_feasible": int(len(feas)), "feas_prob": len(feas) / cfg.num_mc}
            for key in keys:
                stats = summarize_values([r[key] for r in feas])
                entry[key] = stats["mean"]
                entry[f"{key}_std"] = stats["std"]
                entry[f"{key}_se"] = stats["se"]
                entry[f"{key}_ci95"] = stats["ci95"]
            entry["EE_from_mean_power"] = float(R / entry["Ptot"])
            pts.append(entry)
        avg[scheme] = pts
    return avg, raw


def _nan_percentile_columns(trace_mat: np.ndarray, percentile: float) -> np.ndarray:
    """Column-wise nanpercentile without all-NaN slice warnings."""

    out = np.full(trace_mat.shape[1], np.nan, dtype=float)
    for idx in range(trace_mat.shape[1]):
        col = trace_mat[:, idx]
        col = col[np.isfinite(col)]
        if col.size:
            out[idx] = float(np.percentile(col, percentile))
    return out


def _json_float_series(values: Sequence[float]) -> List[Optional[float]]:
    """Convert finite floats to JSON-safe values and non-finite entries to null."""

    out: List[Optional[float]] = []
    for value in values:
        val = float(value)
        out.append(val if np.isfinite(val) else None)
    return out


def power_trace_by_rate(cfg: SimConfig, r_targets: Sequence[float] = (4.0, 8.0, 12.0, 16.0, 20.0), protocol: str = "ES") -> Dict[str, Dict[str, object]]:
    """Return median and interquartile best-so-far total-power traces by SE target."""

    seeds = list(range(1, cfg.num_mc + 1))
    max_workers = max(1, min(int(cfg.num_workers), cfg.num_mc))
    out: Dict[str, Dict[str, object]] = {}
    for r_target in r_targets:
        tasks = [(seed, cfg, float(r_target), protocol) for seed in seeds]
        traces = _ordered_parallel_map(_proposed_power_trace_worker, tasks, max_workers)
        if not traces:
            continue
        trace_mat = np.asarray(
            [[np.nan if value is None else float(value) for value in trace] for trace in traces],
            dtype=float,
        )
        median = _nan_percentile_columns(trace_mat, 50)
        q25 = _nan_percentile_columns(trace_mat, 25)
        q75 = _nan_percentile_columns(trace_mat, 75)
        candidate_index = np.arange(1, trace_mat.shape[1] + 1, dtype=int)
        key = str(int(r_target)) if float(r_target).is_integer() else str(float(r_target))
        out[key] = {
            "R": float(r_target),
            "protocol": protocol.upper(),
            "num_mc": int(cfg.num_mc),
            "candidate_index": [int(x) for x in candidate_index.tolist()],
            "median": _json_float_series(median),
            "q25": _json_float_series(q25),
            "q75": _json_float_series(q75),
        }
    return out


def proposed_fixed_rate_samples(cfg: SimConfig, Rtarget: float, protocol: str) -> List[Dict[str, float]]:
    """Return feasible proposed-design samples for one target rate."""

    seeds = list(range(1, cfg.num_mc + 1))
    tasks = [(seed, cfg, Rtarget, protocol) for seed in seeds]
    max_workers = max(1, min(int(cfg.num_workers), cfg.num_mc))
    vals = _ordered_parallel_map(_proposed_fixed_rate_worker, tasks, max_workers)
    return [v for v in vals if v is not None]


def sweep_sigma_amp(sigmas: Sequence[float], cfg: SimConfig, Rtarget: float, protocol: str) -> List[Dict[str, float]]:
    """Sweep amplifier-noise standard deviation and average the resulting design."""

    sigmas = [float(x) for x in sigmas]
    cfgs = []
    for sigma_amp in sigmas:
        cfg_s = SimConfig(**asdict(cfg))
        cfg_s.sigma_amp = sigma_amp
        cfgs.append(cfg_s)

    tasks = [(i, seed, cfgs[i], Rtarget, protocol) for i in range(len(sigmas)) for seed in range(1, cfg.num_mc + 1)]
    max_workers = max(1, min(int(cfg.num_workers), len(tasks)))
    grouped: List[List[Dict[str, float]]] = [[] for _ in sigmas]
    for idx, res in _ordered_parallel_map(_proposed_fixed_rate_indexed_worker, tasks, max_workers):
        if res is not None:
            grouped[idx].append(res)

    out = []
    for sigma_amp, cfg_s, vals in zip(sigmas, cfgs, grouped):
        if vals:
            entry = {"sigma_amp": float(sigma_amp), "n_total": int(cfg.num_mc), "n_feasible": int(len(vals)), "feas_prob": float(len(vals) / cfg.num_mc)}
            series = {
                "beta_t": [r["beta_t"] for r in vals],
                "beta_r": [r["beta_r"] for r in vals],
                "beta_avg": [r["beta_avg"] for r in vals],
                "active_ratio": [r["L"] / cfg_s.N for r in vals],
                "EE": [r["EE"] for r in vals],
            }
            for key, values in series.items():
                stats = summarize_values(values)
                entry[key] = stats["mean"]
                entry[f"{key}_std"] = stats["std"]
                entry[f"{key}_se"] = stats["se"]
                entry[f"{key}_ci95"] = stats["ci95"]
            out.append(entry)
        else:
            out.append({"sigma_amp": float(sigma_amp), "n_total": int(cfg.num_mc), "n_feasible": 0, "feas_prob": 0.0, "beta_t": None, "beta_r": None, "beta_avg": None, "active_ratio": None, "EE": None})
    return out


def sweep_update_frequency(f_updates: Sequence[float], cfg: SimConfig, Rtarget: float, protocol: str) -> List[Dict[str, float]]:
    """Sweep update frequency to quantify switching/reconfiguration overhead."""

    f_updates = [float(x) for x in f_updates]
    cfgs = []
    for fu in f_updates:
        cfg_s = SimConfig(**asdict(cfg))
        cfg_s.f_update = fu
        cfgs.append(cfg_s)

    tasks = [(i, seed, cfgs[i], Rtarget, protocol) for i in range(len(f_updates)) for seed in range(1, cfg.num_mc + 1)]
    max_workers = max(1, min(int(cfg.num_workers), len(tasks)))
    grouped: List[List[Dict[str, float]]] = [[] for _ in f_updates]
    for idx, res in _ordered_parallel_map(_proposed_fixed_rate_indexed_worker, tasks, max_workers):
        if res is not None:
            grouped[idx].append(res)

    out = []
    for fu, vals in zip(f_updates, grouped):
        if vals:
            stats = summarize_values([r["EE"] for r in vals])
            out.append({
                "f_update": float(fu),
                "n_total": int(cfg.num_mc),
                "n_feasible": int(len(vals)),
                "feas_prob": float(len(vals) / cfg.num_mc),
                "EE": stats["mean"],
                "EE_std": stats["std"],
                "EE_se": stats["se"],
                "EE_ci95": stats["ci95"],
            })
        else:
            out.append({"f_update": float(fu), "n_total": int(cfg.num_mc), "n_feasible": 0, "feas_prob": 0.0, "EE": None})
    return out


def average_breakdown_for_R(cfg: SimConfig, raw: Dict[str, List[List[Optional[Dict[str, float]]]]], Rtarget: float) -> Dict[str, Dict[str, float]]:
    """Average power components for all schemes at a selected target rate."""

    idx = list(cfg.R_targets).index(int(Rtarget))
    out: Dict[str, Dict[str, float]] = {}
    for scheme in raw:
        feas = [r for r in raw[scheme][idx] if r is not None]
        if not feas:
            continue
        out[scheme] = {"n_total": int(cfg.num_mc), "n_feasible": int(len(feas)), "feas_prob": float(len(feas) / cfg.num_mc)}
        for key in ("static", "dynamic", "bs", "amp", "Ptot", "EE"):
            stats = summarize_values([r[key] for r in feas])
            out[scheme][key] = stats["mean"]
            out[scheme][f"{key}_std"] = stats["std"]
            out[scheme][f"{key}_se"] = stats["se"]
            out[scheme][f"{key}_ci95"] = stats["ci95"]
    return out


SCHEME_LABELS = {
    "passive": "Passive",
    "uniform": "Fixed uniform",
    "uniform_opt": "Opt. uniform",
    "random": "Single-random sparse",
    "strongest": "Strongest greedy",
    "active": "Active",
    "proposed": "Proposed",
}

SCHEME_ORDER = ("passive", "uniform", "uniform_opt", "random", "strongest", "active", "proposed")


def feasibility_table(avg: Dict[str, List[Optional[Dict[str, float]]]], rates: Sequence[int] = (12, 18, 20)) -> Dict[str, Dict[str, float]]:
    """Extract feasibility probabilities at selected target rates."""

    out: Dict[str, Dict[str, float]] = {}
    for scheme in SCHEME_ORDER:
        if scheme not in avg:
            continue
        row = {}
        for rate in rates:
            match = next((p for p in avg[scheme] if p is not None and int(round(p["R"])) == int(rate)), None)
            row[str(rate)] = float(match["feas_prob"]) if match is not None else 0.0
        out[scheme] = row
    return out


def representative_metrics(avg: Dict[str, List[Optional[Dict[str, float]]]], rate: int = 12) -> Dict[str, Dict[str, float]]:
    """Extract compact metrics for the representative rate point."""

    out: Dict[str, Dict[str, float]] = {}
    for scheme in SCHEME_ORDER:
        if scheme not in avg:
            continue
        match = next((p for p in avg[scheme] if p is not None and int(round(p["R"])) == int(rate)), None)
        if match is None:
            continue
        out[scheme] = {
            "EE_mean": float(match["EE"]),
            "EE_se": float(match.get("EE_se", 0.0)),
            "EE_ci95": float(match.get("EE_ci95", 0.0)),
            "Ptot_mean": float(match["Ptot"]),
            "Ptot_se": float(match.get("Ptot_se", 0.0)),
            "Ptot_ci95": float(match.get("Ptot_ci95", 0.0)),
            "EE_from_mean_power": float(match["EE_from_mean_power"]),
            "active_count": float(match["L"]),
            "n_feasible": int(match.get("n_feasible", 0)),
            "n_total": int(match.get("n_total", 0)),
            "feas_prob": float(match["feas_prob"]),
        }
    return out


def point_at_rate(points: Sequence[Optional[Dict[str, float]]], rate: int) -> Optional[Dict[str, float]]:
    """Return the averaged point matching an integer target rate."""

    return next((p for p in points if p is not None and int(round(p["R"])) == int(rate)), None)


def validate_raw_outputs(cfg: SimConfig, raw: Dict[str, List[List[Optional[Dict[str, float]]]]], label: str) -> Dict[str, object]:
    """Validate numerical consistency for raw Monte Carlo samples."""

    checked_total = 0
    checked_feasible = 0
    violations: List[str] = []
    tol = 1e-7
    numeric_keys = ("EE", "Ptx", "Ptx_peak", "Ptot", "L", "Lt", "Lr", "beta_t", "beta_r", "beta_avg", "alpha", "static", "dynamic", "bs", "amp")

    for scheme, by_rate in raw.items():
        for rate_idx, samples in enumerate(by_rate):
            rate = cfg.R_targets[rate_idx]
            for sample_idx, res in enumerate(samples):
                checked_total += 1
                if res is None:
                    continue
                checked_feasible += 1
                prefix = f"{label}:{scheme}:R{rate}:sample{sample_idx + 1}"
                for key in numeric_keys:
                    if key in res and not np.isfinite(float(res[key])):
                        violations.append(f"{prefix}:{key}:nonfinite")
                if not (1.0 - tol <= res["beta_t"] <= cfg.beta_max + tol and 1.0 - tol <= res["beta_r"] <= cfg.beta_max + tol):
                    violations.append(f"{prefix}:gain_bounds")
                if res["Ptx_peak"] > cfg.Ptx_max + tol:
                    violations.append(f"{prefix}:Ptx_peak_budget")
                if abs(res["Ptot"] - (res["static"] + res["dynamic"] + res["bs"] + res["amp"])) > 1e-6:
                    violations.append(f"{prefix}:power_sum")
                if abs(res["EE"] - res["R"] / res["Ptot"]) > 1e-6:
                    violations.append(f"{prefix}:ee_identity")
                if abs(res["L"] - (res["Lt"] + res["Lr"])) > 1e-9:
                    violations.append(f"{prefix}:active_count")
                if len(violations) >= 20:
                    break
            if len(violations) >= 20:
                break
        if len(violations) >= 20:
            break

    if violations:
        raise RuntimeError(f"Numerical validation failed for {label}: {violations[:5]}")
    return {
        "label": label,
        "checked_total": int(checked_total),
        "checked_feasible": int(checked_feasible),
        "violations": 0,
    }


def fairness_audit_proposed_vs_strongest(raw: Dict[str, List[List[Optional[Dict[str, float]]]]]) -> Dict[str, object]:
    """Confirm proposed includes the strongest-greedy candidate pool."""

    if "proposed" not in raw or "strongest" not in raw:
        return {"checked_pairs": 0, "violations": 0}
    checked_pairs = 0
    worst_excess = 0.0
    violations = []
    for rate_idx, (proposed_samples, strongest_samples) in enumerate(zip(raw["proposed"], raw["strongest"])):
        for sample_idx, (prop, strong) in enumerate(zip(proposed_samples, strongest_samples)):
            if prop is None or strong is None:
                continue
            checked_pairs += 1
            excess = float(prop["Ptot"] - strong["Ptot"])
            worst_excess = max(worst_excess, excess)
            if excess > 1e-7:
                violations.append({"rate": int(rate_idx), "sample": int(sample_idx + 1), "excess_power": excess})
                if len(violations) >= 5:
                    break
        if len(violations) >= 5:
            break
    if violations:
        raise RuntimeError(f"Fairness audit failed: {violations}")
    return {"checked_pairs": int(checked_pairs), "violations": 0, "worst_power_excess": float(worst_excess)}

def main() -> None:
    """Run all reproducibility experiments and write figures/JSON results."""

    base_dir = Path(__file__).resolve().parents[1]
    fig_dir = base_dir / "figs"
    results_dir = base_dir / "results"
    fig_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    cfg = SimConfig()

    # Replace only the canonical result file for this configuration.
    result_json = results_dir / "hybrid_star_ris_results.json"
    if result_json.exists():
        result_json.unlink()

    timings: Dict[str, float] = {}
    validations: Dict[str, object] = {}

    t0 = time.perf_counter()
    avg_es, raw_es = average_boundary(cfg, protocol="ES")
    timings["average_boundary_es_seconds"] = time.perf_counter() - t0
    validations["average_boundary_es"] = validate_raw_outputs(cfg, raw_es, "ES main boundary")
    validations["fairness_proposed_vs_strongest_es"] = fairness_audit_proposed_vs_strongest(raw_es)
    breakdown_es_12 = average_breakdown_for_R(cfg, raw_es, Rtarget=12)
    breakdown_es_16 = average_breakdown_for_R(cfg, raw_es, Rtarget=16)


    sigmas = np.linspace(0.02, 0.18, 9)
    f_updates = np.arange(50, 551, 50)
    sensitivity_rates = (6, 9, 12, 15, 18)
    es_noise_by_rate = {}
    es_dyn_by_rate = {}
    for R in sensitivity_rates:
        key = str(R)
        t0 = time.perf_counter()
        es_noise_by_rate[key] = sweep_sigma_amp(sigmas, cfg, Rtarget=float(R), protocol="ES")
        timings[f"noise_sweep_es_R{R}_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        es_dyn_by_rate[key] = sweep_update_frequency(f_updates, cfg, Rtarget=float(R), protocol="ES")
        timings[f"dynamic_sweep_es_R{R}_seconds"] = time.perf_counter() - t0

    reliability_blockers = []

    t0 = time.perf_counter()
    trace_by_rate = power_trace_by_rate(cfg, r_targets=(4.0, 8.0, 12.0, 16.0, 20.0), protocol="ES")
    timings["power_trace_es_seconds"] = time.perf_counter() - t0

    summary = {
        "result_schema": "hybrid_star_ris_es_mc10000_python_eps",
        "config": {**asdict(cfg), "lambda1_effective": eq23_weights(cfg)[0], "lambda2_effective": eq23_weights(cfg)[1]},
        "runtime_environment": runtime_environment(cfg),
        "timings_seconds": timings,
        "scheme_order": list(SCHEME_ORDER),
        "scheme_labels": SCHEME_LABELS,
        "average_boundary_es": avg_es,
        "representative_R12_metrics_es": representative_metrics(avg_es, rate=12),
        "feasibility_probability_es": feasibility_table(avg_es, rates=(12, 18, 20)),
        "es_noise_sweep_by_rate": es_noise_by_rate,
        "es_dynamic_sweep_by_rate": es_dyn_by_rate,
        "power_breakdown_at_R12_es": breakdown_es_12,
        "power_breakdown_at_R16_es": breakdown_es_16,
        "power_trace_by_rate": trace_by_rate,
        "validations": validations,
        "reliability_blockers": reliability_blockers,
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    from plot_hybrid_star_ris_eps import plot_from_json

    plot_from_json(result_json, fig_dir)
    print("Simulation completed.")
    print(f"Figures saved under: {fig_dir}")
    print(f"Summary JSON: {result_json}")
    if reliability_blockers:
        print("Reliability blockers detected:")
        for item in reliability_blockers:
            print(f"- {item}")


if __name__ == "__main__":
    main()
