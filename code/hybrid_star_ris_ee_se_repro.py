#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible simulator for
"EE-SE Pareto-Frontier Approximation and Amplification Structure Optimization for
Hybrid Active-Passive STAR-RIS"

This implementation evaluates the post-ZF scalar surrogate used in the paper.
It generates the ES / TS / SS protocol studies, feasibility probabilities,
baseline ablations, power breakdowns, and normalized EE results reported in
the manuscript.  The side-assigned fully active baseline uses the same
power-minimizing gain update as the proposed scheme; a gain-saturated active
benchmark is intentionally not mixed into the main comparison.
"""

from __future__ import annotations

import json
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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
    pl_br: float = 0.18
    pl_t: Tuple[float, float] = (0.064, 0.052)
    pl_r: Tuple[float, float] = (0.080, 0.064)

    rho_t: float = 0.55
    rho_r: float = 0.45
    tau_t: float = 0.55
    tau_r: float = 0.45
    beta_max: float = 5.0

    sigma0: float = 0.26
    sigma_amp: float = 0.05

    P_ctrl: float = 0.12
    P_bias: float = 0.016
    f_update: float = 50.0
    E_sw_base: float = 2.0e-4
    E_sw_act: float = 1.2e-4
    P_amp: float = 0.0025
    eta_pa: float = 0.38
    Ptx_max: float = 5.0

    L_uniform: int = 6
    L_step: int = 3
    random_repeats: int = 1
    split_grid: Tuple[float, ...] = tuple(np.linspace(0.2, 0.8, 13))
    R_targets: Tuple[int, ...] = tuple(range(2, 22, 2))

    num_mc: int = 100
    num_workers: int = min(8, os.cpu_count() or 1)

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


def complex_fading(shape: Sequence[int], fading: str, m: float, omega: float, rng: np.random.Generator) -> np.ndarray:
    """Generate complex fading coefficients.

    Envelope power is sampled from a Gamma distribution and combined with a
    uniformly distributed phase.  With ``fading="rayleigh"``, the Nakagami
    parameter is reduced to ``m=1``.
    """

    if fading.lower() == "rayleigh":
        m = 1.0
    power = rng.gamma(shape=m, scale=omega / m, size=shape)
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
    """Return effective splitting coefficients and time fractions for ES/TS/SS."""

    mode = protocol.upper()
    if mode == "ES":
        return (cfg.rho_t, cfg.rho_r), (1.0, 1.0)
    if mode == "TS":
        return (1.0, 1.0), (cfg.tau_t, cfg.tau_r)
    if mode == "SS":
        return (1.0, 1.0), (1.0, 1.0)
    raise ValueError(f"Unsupported protocol: {protocol}")


def side_demand_weights(alpha: float) -> Tuple[float, float]:
    """Convert the sum-rate split alpha into side-demand weights."""

    eps = 1e-6
    return float(max(alpha, eps)), float(max(1.0 - alpha, eps))


def eq23_side_score(g: np.ndarray, h_side: np.ndarray, omega_s: float, rho_s_eff: float, cfg: SimConfig) -> np.ndarray:
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
    numerator = omega_s * np.sqrt(rho_s_eff) * abs_g * min_abs_h
    denominator = lambda1 * rho_s_eff * (cfg.sigma_amp ** 2) * mean_abs_h2 + lambda2 * (cfg.P_bias + cfg.f_update * cfg.E_sw_act)
    return numerator / (denominator + 1e-12)


def build_active_sets(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, L: int, alpha: float, cfg: SimConfig, protocol: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build active-element sets for the proposed cost-aware sparse design.

    The score is computed separately for the transmission and reflection
    sides.  The L elements with the largest side-wise score are selected and
    assigned to the side on which they score higher.
    """

    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(protocol, cfg)
    omega_t, omega_r = side_demand_weights(alpha)
    xi_t = eq23_side_score(g, h_t, omega_t, rho_t_eff, cfg)
    xi_r = eq23_side_score(g, h_r, omega_r, rho_r_eff, cfg)
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
    omega_t, omega_r = side_demand_weights(alpha)
    xi_t = eq23_side_score(g, h_t, omega_t, rho_t_eff, cfg)
    xi_r = eq23_side_score(g, h_r, omega_r, rho_r_eff, cfg)
    best_side = np.where(xi_t[idx] >= xi_r[idx], 0, 1)
    return np.sort(idx[best_side == 0]), np.sort(idx[best_side == 1])


def build_strongest_active_sets(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, L: int, alpha: float, cfg: SimConfig, protocol: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build a channel-strength-only greedy baseline.

    This baseline uses the same L/alpha grid as the proposed method but ranks
    elements only by side-demand-weighted coherent gain.  It deliberately
    ignores amplifier-noise, bias, and switching costs.
    """

    if L <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(protocol, cfg)
    omega_t, omega_r = side_demand_weights(alpha)
    abs_g = np.abs(g)
    pi_t = omega_t * np.sqrt(rho_t_eff) * abs_g * np.min(np.abs(h_t), axis=0)
    pi_r = omega_r * np.sqrt(rho_r_eff) * abs_g * np.min(np.abs(h_r), axis=0)
    best_side = np.where(pi_t >= pi_r, 0, 1)
    selected = np.argsort(-np.maximum(pi_t, pi_r))[:L]
    A_t = np.sort(selected[best_side[selected] == 0])
    A_r = np.sort(selected[best_side[selected] == 1])
    return A_t, A_r


def _random_baseline_seed(g: np.ndarray, L: int, alpha: float, repeat: int, protocol: str) -> int:
    """Create a deterministic seed for the single-random sparse baseline."""

    # The hash depends on the channel realization and design point, keeping the
    # random baseline reproducible without passing a separate seed through the
    # solver interface.
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
    one random repetition (Q_rand=1), so this is a lightweight sanity baseline
    rather than a heavily optimized random search.
    """

    if L <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    rng = np.random.default_rng(_random_baseline_seed(g, L, alpha, repeat, protocol))
    selected = np.sort(rng.choice(cfg.N, size=L, replace=False))
    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(protocol, cfg)
    omega_t, omega_r = side_demand_weights(alpha)
    xi_t = eq23_side_score(g, h_t, omega_t, rho_t_eff, cfg)
    xi_r = eq23_side_score(g, h_r, omega_r, rho_r_eff, cfg)
    best_side = np.where(xi_t[selected] >= xi_r[selected], 0, 1)
    return np.sort(selected[best_side == 0]), np.sort(selected[best_side == 1])


def build_ss_subsurfaces(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, A_t: np.ndarray, A_r: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    """Construct transmission and reflection subsurfaces for the SS protocol.

    Under SS, each passive element belongs to only one side.  Non-active
    elements are therefore assigned according to the side-demand-weighted link
    strength, yielding complete subsurfaces S_t and S_r.
    """

    abs_g = np.abs(g)
    omega_t, omega_r = side_demand_weights(alpha)
    pi_t = omega_t * abs_g * np.min(np.abs(h_t), axis=0)
    pi_r = omega_r * abs_g * np.min(np.abs(h_r), axis=0)
    all_idx = np.arange(len(g))
    active_union = np.union1d(A_t, A_r)
    passive = np.setdiff1d(all_idx, active_union)
    P_t = np.sort(passive[pi_t[passive] >= pi_r[passive]])
    P_r = np.sort(passive[pi_t[passive] < pi_r[passive]])
    S_t = np.sort(np.union1d(P_t, A_t))
    S_r = np.sort(np.union1d(P_r, A_r))
    return {"P_t": P_t, "P_r": P_r, "S_t": S_t, "S_r": S_r}


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
    """Aggregate total power and return static, dynamic, BS, and amplifier parts."""

    static = cfg.P_ctrl + cfg.P_bias * (Lt + Lr)
    dynamic = cfg.f_update * (cfg.E_sw_base + cfg.E_sw_act * (Lt + Lr))
    mode = protocol.upper()
    if mode == "TS":
        tau_t, tau_r = cfg.tau_t, cfg.tau_r
        Ptx_avg = tau_t * P_t_req + tau_r * P_r_req
        Ptx_peak = max(P_t_req, P_r_req)
        bs = Ptx_avg / cfg.eta_pa
        amp = cfg.P_amp * (tau_t * Lt * max(beta_t ** 2 - 1.0, 0.0) + tau_r * Lr * max(beta_r ** 2 - 1.0, 0.0))
    else:
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
    """Evaluate one active-set split at a target sum-SE.

    The inputs are fixed active sets A_t/A_r and a side-rate split alpha.  The
    output includes total power, transmit power, optimized common gains, and
    power breakdown.  If the peak BS power exceeds the budget, ``None`` is
    returned.
    """

    Rt = alpha * Rtarget
    Rr = (1.0 - alpha) * Rtarget
    (rho_t_eff, rho_r_eff), (tau_t, tau_r) = protocol_rho_tau(protocol, cfg)
    mode = protocol.upper()
    if mode == "SS":
        ss_sets = build_ss_subsurfaces(g, h_t, h_r, A_t, A_r, alpha)
        passive_t = ss_sets["P_t"]
        passive_r = ss_sets["P_r"]
    else:
        ss_sets = None
        passive_t = np.setdiff1d(np.arange(cfg.N), A_t)
        passive_r = np.setdiff1d(np.arange(cfg.N), A_r)
    a_t, b_t, c_t, d_t = side_params(g, h_t, passive_t, A_t, rho_t_eff, cfg.sigma0, cfg.sigma_amp)
    a_r, b_r, c_r, d_r = side_params(g, h_r, passive_r, A_r, rho_r_eff, cfg.sigma0, cfg.sigma_amp)
    if mode == "TS":
        Rt_eff = Rt / tau_t
        Rr_eff = Rr / tau_r
    else:
        Rt_eff = Rt
        Rr_eff = Rr
    beta_t = minimize_beta_power(a_t, b_t, c_t, d_t, Rt_eff, cfg.Kt, len(A_t), cfg.P_amp, cfg.eta_pa, cfg.beta_max)
    beta_r = minimize_beta_power(a_r, b_r, c_r, d_r, Rr_eff, cfg.Kr, len(A_r), cfg.P_amp, cfg.eta_pa, cfg.beta_max)
    P_t = side_required_power(g, h_t, passive_t, A_t, beta_t, Rt_eff, rho_t_eff, cfg.sigma0, cfg.sigma_amp)
    P_r = side_required_power(g, h_r, passive_r, A_r, beta_r, Rr_eff, rho_r_eff, cfg.sigma0, cfg.sigma_amp)
    Ptot, Ptx_avg, Ptx_peak, breakdown = protocol_total_power(P_t, P_r, len(A_t), len(A_r), beta_t, beta_r, cfg, mode)
    if Ptx_peak > cfg.Ptx_max:
        return None
    EE = Rtarget / Ptot
    out = {"R": float(Rtarget), "EE": float(EE), "Ptx": float(Ptx_avg), "Ptx_peak": float(Ptx_peak), "Ptot": float(Ptot), "Lt": int(len(A_t)), "Lr": int(len(A_r)), "L": int(len(A_t) + len(A_r)), "beta_t": float(beta_t), "beta_r": float(beta_r), "beta_avg": float(0.5 * (beta_t + beta_r)), "alpha": float(alpha), "protocol": mode}
    out.update(breakdown)
    if ss_sets is not None:
        out["surface_t"] = int(len(ss_sets["S_t"]))
        out["surface_r"] = int(len(ss_sets["S_r"]))
    return out


def solve_boundary_point(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, Rtarget: float, cfg: SimConfig, scheme: str = "proposed", protocol: str = "ES") -> Optional[Dict[str, float]]:
    """Search the minimum-power point for a scheme at a fixed target SE.

    The search variables are the total active-cardinality L, the side-rate
    split alpha, and the active-element assignment rule determined by the
    selected scheme.  The feasible candidate with the smallest total power is
    retained.
    """

    N = len(g)
    best: Optional[Dict[str, float]] = None
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
                    # The proposed sparse solver evaluates a small ranker pool:
                    # the cost-aware score and its channel-strength-only
                    # counterpart.  The final choice is made by the same
                    # surrogate feasibility and total-power evaluation.
                    candidate_sets = (
                        build_active_sets(g, h_t, h_r, L, alpha, cfg, protocol),
                        build_strongest_active_sets(g, h_t, h_r, L, alpha, cfg, protocol),
                    )
                    for A_t_pool, A_r_pool in candidate_sets:
                        res = evaluate_candidate(g, h_t, h_r, A_t_pool, A_r_pool, Rtarget, alpha, cfg, protocol)
                        if res is None:
                            continue
                        if best is None or res["Ptot"] < best["Ptot"]:
                            best = res
                    continue
                else:
                    A_t, A_r = build_active_sets(g, h_t, h_r, L, alpha, cfg, protocol)
                res = evaluate_candidate(g, h_t, h_r, A_t, A_r, Rtarget, alpha, cfg, protocol)
                if res is None:
                    continue
                if best is None or res["Ptot"] < best["Ptot"]:
                    best = res
    return best


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
            entry = {"R": float(R), "protocol": protocol.upper(), "feas_prob": len(feas) / cfg.num_mc}
            for key in keys:
                vals = np.array([r[key] for r in feas], dtype=float)
                entry[key] = float(np.mean(vals))
                if key in ("EE", "Ptot"):
                    entry[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    entry[f"{key}_se"] = float(entry[f"{key}_std"] / np.sqrt(len(vals))) if len(vals) > 0 else 0.0
            entry["EE_from_mean_power"] = float(R / entry["Ptot"])
            pts.append(entry)
        avg[scheme] = pts
    return avg, raw


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
            out.append({"sigma_amp": float(sigma_amp), "beta_t": float(np.mean([r["beta_t"] for r in vals])), "beta_r": float(np.mean([r["beta_r"] for r in vals])), "beta_avg": float(np.mean([r["beta_avg"] for r in vals])), "active_ratio": float(np.mean([r["L"] / cfg_s.N for r in vals])), "EE": float(np.mean([r["EE"] for r in vals]))})
        else:
            out.append({"sigma_amp": float(sigma_amp), "beta_t": None, "beta_r": None, "beta_avg": None, "active_ratio": None, "EE": None})
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
            out.append({"f_update": float(fu), "EE": float(np.mean([r["EE"] for r in vals]))})
        else:
            out.append({"f_update": float(fu), "EE": None})
    return out


def average_breakdown_for_R(cfg: SimConfig, raw: Dict[str, List[List[Optional[Dict[str, float]]]]], Rtarget: float) -> Dict[str, Dict[str, float]]:
    """Average power components for all schemes at a selected target rate."""

    idx = list(cfg.R_targets).index(int(Rtarget))
    out: Dict[str, Dict[str, float]] = {}
    for scheme in raw:
        feas = [r for r in raw[scheme][idx] if r is not None]
        if not feas:
            continue
        out[scheme] = {
            "static": float(np.mean([r["static"] for r in feas])),
            "dynamic": float(np.mean([r["dynamic"] for r in feas])),
            "bs": float(np.mean([r["bs"] for r in feas])),
            "amp": float(np.mean([r["amp"] for r in feas])),
            "Ptot": float(np.mean([r["Ptot"] for r in feas])),
            "EE": float(np.mean([r["EE"] for r in feas])),
            "feas_prob": float(len(feas) / cfg.num_mc),
        }
    return out


SCHEME_LABELS = {
    "passive": "Passive",
    "uniform": "Fixed uniform",
    "uniform_opt": "Opt. uniform",
    "random": "Single-random sparse",
    "strongest": "Strongest greedy",
    "active": "Side-assigned active",
    "proposed": "Proposed",
}

SCHEME_ORDER = ("passive", "uniform", "uniform_opt", "random", "strongest", "active", "proposed")


def complexity_summary(cfg: SimConfig) -> Dict[str, Dict[str, str]]:
    """Return the symbolic per-boundary-point complexity reported in the paper."""

    G = "|G_alpha|"
    L = "|L|"
    Qb = "Q_beta"
    Qr = "Q_rand"
    NK = "N(K_t+K_r)"
    rank = "N log N"
    return {
        "passive": {"search_size": G, "ranking": "0", "dominant_order": f"O({G}({NK}+{Qb}))"},
        "uniform": {"search_size": G, "ranking": "0", "dominant_order": f"O({G}({NK}+{Qb}))"},
        "uniform_opt": {"search_size": f"{L}{G}", "ranking": "0", "dominant_order": f"O({L}{G}({NK}+{Qb}))"},
        "random": {"search_size": f"{Qr}{L}{G}", "ranking": "0", "dominant_order": f"O({Qr}{L}{G}({NK}+{Qb}))"},
        "strongest": {"search_size": f"{L}{G}", "ranking": rank, "dominant_order": f"O({L}{G}({rank}+{NK}+{Qb}))"},
        "active": {"search_size": G, "ranking": rank, "dominant_order": f"O({G}({rank}+{NK}+{Qb}))"},
        "proposed": {"search_size": f"2{L}{G}", "ranking": rank, "dominant_order": f"O({L}{G}({rank}+{NK}+{Qb}))"},
    }


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
            "Ptot_mean": float(match["Ptot"]),
            "EE_from_mean_power": float(match["EE_from_mean_power"]),
            "active_count": float(match["L"]),
            "feas_prob": float(match["feas_prob"]),
        }
    return out


# =========================
# Plotting
# =========================
def style() -> None:
    """Set a consistent plotting style for manuscript figures."""

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.dpi": 220, "lines.linewidth": 2.0})


def save_manuscript_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save a figure in PNG and JPG formats."""

    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.jpg", bbox_inches="tight", dpi=220)


def plot_pareto(avg: Dict[str, List[Optional[Dict[str, float]]]], out_dir: Path) -> None:
    """Plot the approximated EE-SE frontier for the ES protocol."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = {
        "passive": "o",
        "uniform": "s",
        "uniform_opt": "v",
        "random": "P",
        "strongest": "X",
        "active": "^",
        "proposed": "D",
    }
    for scheme in SCHEME_ORDER:
        if scheme not in avg:
            continue
        pts = [p for p in avg[scheme] if p is not None]
        if not pts:
            continue
        ax.plot([p["R"] for p in pts], [p["EE"] for p in pts], marker=markers[scheme], label=SCHEME_LABELS[scheme])
        # Hollow markers identify conditional averages with feasibility below one.
        outage_pts = [p for p in pts if p.get("feas_prob", 1.0) < 1.0]
        if outage_pts:
            ax.plot(
                [p["R"] for p in outage_pts],
                [p["EE"] for p in outage_pts],
                linestyle="None",
                marker=markers[scheme],
                markerfacecolor="none",
                markeredgewidth=1.3,
            )
    ax.set_xlabel("Sum spectral efficiency (bit/s/Hz)")
    ax.set_ylabel("Normalized energy efficiency (bit/J/Hz)")
    ax.legend(frameon=True, fontsize=7, ncol=2)
    fig.tight_layout()
    save_manuscript_figure(fig, out_dir, "fig_pareto_boundary")
    plt.close(fig)


def plot_design_rules(protocol_boundary: Dict[str, List[Optional[Dict[str, float]]]], cfg: SimConfig, out_dir: Path) -> None:
    """Plot active-ratio and side-wise-gain rules for ES/TS/SS."""

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for protocol, marker in [("ES", "o"), ("TS", "s"), ("SS", "D")]:
        pts = [p for p in protocol_boundary[protocol] if p is not None]
        axes[0].plot([p["R"] for p in pts], [p["L"] / cfg.N for p in pts], marker=marker, label=protocol)
    axes[0].set_xlabel("Target sum-SE (bit/s/Hz)")
    axes[0].set_ylabel("Active ratio")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=True, fontsize=8, ncol=3)
    style_map = {"ES": ("o", "-"), "TS": ("s", "--"), "SS": ("D", ":")}
    for protocol in ["ES", "TS", "SS"]:
        pts = [p for p in protocol_boundary[protocol] if p is not None]
        marker, linestyle = style_map[protocol]
        axes[1].plot([p["R"] for p in pts], [p["beta_t"] for p in pts], marker=marker, linestyle=linestyle, label=rf"{protocol} $\beta_t^\star$")
        axes[1].plot([p["R"] for p in pts], [p["beta_r"] for p in pts], marker=marker, linestyle=(0, (1, 1)), label=rf"{protocol} $\beta_r^\star$")
    axes[1].set_xlabel("Target sum-SE (bit/s/Hz)")
    axes[1].set_ylabel("Side-wise gain")
    axes[1].set_ylim(1.0, cfg.beta_max + 0.2)
    axes[1].legend(frameon=True, fontsize=7, ncol=2)
    fig.tight_layout()
    save_manuscript_figure(fig, out_dir, "fig_design_rules")
    plt.close(fig)


def plot_noise_and_dynamics(protocol_noise: Dict[str, List[Dict[str, float]]], protocol_dyn: Dict[str, List[Dict[str, float]]], out_dir: Path) -> None:
    """Plot amplifier-noise and update-frequency sensitivities."""

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for protocol, marker in [("ES", "o"), ("TS", "s"), ("SS", "D")]:
        noise = protocol_noise[protocol]
        axes[0].plot([d["sigma_amp"] for d in noise], [np.nan if d["beta_avg"] is None else d["beta_avg"] for d in noise], marker=marker, label=protocol)
        dyn = protocol_dyn[protocol]
        axes[1].plot([d["f_update"] for d in dyn], [np.nan if d["EE"] is None else d["EE"] for d in dyn], marker=marker, label=protocol)
    axes[0].set_xlabel(r"Amplifier-noise std. $\sigma_a$")
    axes[0].set_ylabel("Average side-wise gain")
    axes[0].legend(frameon=True, fontsize=8, ncol=3)
    axes[1].set_xlabel("Update frequency (Hz)")
    axes[1].set_ylabel("Normalized energy efficiency (bit/J/Hz)")
    axes[1].legend(frameon=True, fontsize=8, ncol=3)
    fig.tight_layout()
    save_manuscript_figure(fig, out_dir, "fig_noise_dynamic")
    plt.close(fig)


def plot_power_breakdown(breakdown_map: Dict[int, Dict[str, Dict[str, float]]], out_dir: Path) -> None:
    """Plot two power-breakdown panels for representative target-rate points."""

    rates = [12, 16]
    short_labels = {
        "passive": "Pass.",
        "uniform": "F.-unif.",
        "uniform_opt": "Opt.-unif.",
        "random": "Rand.",
        "strongest": "Strong.",
        "active": "Active",
        "proposed": "Prop.",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), sharey=True)
    legend_handles = None
    legend_labels = None
    ymax = 0.0
    for rate in rates:
        breakdown = breakdown_map[rate]
        schemes = [s for s in SCHEME_ORDER if s in breakdown]
        total = np.array([breakdown[s]["static"] + breakdown[s]["dynamic"] + breakdown[s]["bs"] + breakdown[s]["amp"] for s in schemes])
        ymax = max(ymax, float(np.max(total)))
    for ax, rate in zip(axes, rates):
        breakdown = breakdown_map[rate]
        schemes = [s for s in SCHEME_ORDER if s in breakdown]
        static = np.array([breakdown[s]["static"] for s in schemes])
        dynamic = np.array([breakdown[s]["dynamic"] for s in schemes])
        bs = np.array([breakdown[s]["bs"] for s in schemes])
        amp = np.array([breakdown[s]["amp"] for s in schemes])
        x = np.arange(len(schemes))
        ax.bar(x, static, label="Static")
        ax.bar(x, dynamic, bottom=static, label="Dynamic")
        ax.bar(x, bs, bottom=static + dynamic, label="BS transmit")
        ax.bar(x, amp, bottom=static + dynamic + bs, label="Amplification")
        ax.set_xticks(x)
        ax.set_xticklabels([short_labels[s] for s in schemes], rotation=25, ha="right", fontsize=7)
        ax.set_title(rf"$R_\Sigma={rate}$ bit/s/Hz", fontsize=9)
        ax.set_ylim(0.0, 1.12 * ymax)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Power consumption (W)")
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, frameon=True, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_manuscript_figure(fig, out_dir, "fig_power_breakdown")
    plt.close(fig)



# =========================
# Main workflow
# =========================
def main() -> None:
    """Run all reproducibility experiments and write figures/JSON results."""

    style()
    base_dir = Path(__file__).resolve().parents[1]
    fig_dir = base_dir / "figs"
    results_dir = base_dir / "results"
    fig_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    cfg = SimConfig()

    # Remove any existing result JSON before writing the output for this configuration.
    for old_json in results_dir.glob("*.json"):
        old_json.unlink()

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    avg_es, raw_es = average_boundary(cfg, protocol="ES")
    timings["average_boundary_es_seconds"] = time.perf_counter() - t0
    breakdown_es_12 = average_breakdown_for_R(cfg, raw_es, Rtarget=12)
    breakdown_es_16 = average_breakdown_for_R(cfg, raw_es, Rtarget=16)

    protocol_boundary: Dict[str, List[Optional[Dict[str, float]]]] = {"ES": avg_es["proposed"]}
    for protocol in ("TS", "SS"):
        t0 = time.perf_counter()
        avg_p, _ = average_boundary(cfg, protocol=protocol, schemes=("proposed",))
        timings[f"average_boundary_{protocol.lower()}_proposed_seconds"] = time.perf_counter() - t0
        protocol_boundary[protocol] = avg_p["proposed"]

    sigmas = np.linspace(0.02, 0.18, 9)
    f_updates = np.arange(50, 551, 50)
    protocol_noise = {}
    protocol_dyn = {}
    for p in ("ES", "TS", "SS"):
        t0 = time.perf_counter()
        protocol_noise[p] = sweep_sigma_amp(sigmas, cfg, Rtarget=12.0, protocol=p)
        timings[f"noise_sweep_{p.lower()}_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        protocol_dyn[p] = sweep_update_frequency(f_updates, cfg, Rtarget=12.0, protocol=p)
        timings[f"dynamic_sweep_{p.lower()}_seconds"] = time.perf_counter() - t0

    plot_pareto(avg_es, fig_dir)
    plot_design_rules(protocol_boundary, cfg, fig_dir)
    plot_noise_and_dynamics(protocol_noise, protocol_dyn, fig_dir)
    plot_power_breakdown({12: breakdown_es_12, 16: breakdown_es_16}, fig_dir)
    summary = {
        "result_schema": "hybrid_star_ris_v4_surrogate_baselines",
        "config": {**asdict(cfg), "lambda1_effective": eq23_weights(cfg)[0], "lambda2_effective": eq23_weights(cfg)[1]},
        "scheme_order": list(SCHEME_ORDER),
        "scheme_labels": SCHEME_LABELS,
        "average_boundary_es": avg_es,
        "representative_R12_metrics_es": representative_metrics(avg_es, rate=12),
        "feasibility_probability_es": feasibility_table(avg_es, rates=(12, 18, 20)),
        "protocol_design_rules": protocol_boundary,
        "protocol_noise_sweep": protocol_noise,
        "protocol_dynamic_sweep": protocol_dyn,
        "power_breakdown_at_R12_es": breakdown_es_12,
        "power_breakdown_at_R16_es": breakdown_es_16,
        "complexity_summary": complexity_summary(cfg),
        "runtime_seconds": timings,
    }
    with open(results_dir / "hybrid_star_ris_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Simulation completed.")
    print(f"Figures saved under: {fig_dir}")
    print(f"Summary JSON: {results_dir / 'hybrid_star_ris_results.json'}")


if __name__ == "__main__":
    main()
