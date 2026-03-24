#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible simulator for
"EE-SE Pareto Frontier and Amplification Structure Optimization for
Hybrid Active-Passive STAR-RIS"

This implementation preserves the post-ZF equivalent model used in the paper
and supports the protocol-dependent ES / TS / SS studies required by the
updated figures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =========================
# 仿真配置
# =========================
@dataclass
class SimConfig:
    """集中管理论文复现实验中的系统参数与扫描参数。"""

    N: int = 24
    Kt: int = 2
    Kr: int = 2
    fading: str = "nakagami"
    m_br: float = 1.8
    m_ru: float = 1.3
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
    P_bias: float = 0.004
    P_dyn_base: float = 0.01
    P_dyn_act: float = 0.0015
    P_amp: float = 0.0025
    eta_pa: float = 0.38
    Ptx_max: float = 5.0

    L_uniform: int = 6
    L_step: int = 3
    split_grid: Tuple[float, ...] = tuple(np.linspace(0.2, 0.8, 13))
    R_targets: Tuple[int, ...] = tuple(range(2, 22, 2))

    num_mc: int = 100
    f_ref: float = 50.0

    lambda1: Optional[float] = None
    lambda2: Optional[float] = None


def eq23_weights(cfg: SimConfig) -> Tuple[float, float]:
    """返回式(23)中的两项权重。

    若未显式给出，则按论文复现脚本中的默认规则，
    分别使用接收噪声功率和最大发射功率的倒数来构造。
    """

    lambda1 = cfg.lambda1 if cfg.lambda1 is not None else 1.0 / (cfg.sigma0 ** 2)
    lambda2 = cfg.lambda2 if cfg.lambda2 is not None else 1.0 / cfg.Ptx_max
    return float(lambda1), float(lambda2)


def complex_fading(shape: Sequence[int], fading: str, m: float, omega: float, rng: np.random.Generator) -> np.ndarray:
    """生成复衰落信道系数。

    这里统一用 Gamma 分布生成包络功率，再叠加均匀相位。
    当 `fading="rayleigh"` 时，将 Nakagami 参数退化到 `m=1`。
    """

    if fading.lower() == "rayleigh":
        m = 1.0
    power = rng.gamma(shape=m, scale=omega / m, size=shape)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    return np.sqrt(power) * np.exp(1j * phase)


def generate_channels(cfg: SimConfig, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按给定随机种子生成一组 BS-RIS 与 RIS-用户信道。"""

    rng = np.random.default_rng(seed)
    g = complex_fading((cfg.N,), cfg.fading, cfg.m_br, cfg.pl_br, rng)
    h_t = np.vstack([complex_fading((cfg.N,), cfg.fading, cfg.m_ru, cfg.pl_t[k], rng) for k in range(cfg.Kt)])
    h_r = np.vstack([complex_fading((cfg.N,), cfg.fading, cfg.m_ru, cfg.pl_r[k], rng) for k in range(cfg.Kr)])
    return g, h_t, h_r


# =========================
# 协议与加权规则
# =========================
def protocol_rho_tau(protocol: str, cfg: SimConfig) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """根据 ES / TS / SS 协议返回有效反射/透射系数与时间占比。"""

    mode = protocol.upper()
    if mode == "ES":
        return (cfg.rho_t, cfg.rho_r), (1.0, 1.0)
    if mode == "TS":
        return (1.0, 1.0), (cfg.tau_t, cfg.tau_r)
    if mode == "SS":
        return (1.0, 1.0), (1.0, 1.0)
    raise ValueError(f"Unsupported protocol: {protocol}")


def side_demand_weights(alpha: float) -> Tuple[float, float]:
    """把总速率拆分比例 alpha 转成透射侧/反射侧的需求权重。"""

    eps = 1e-6
    return float(max(alpha, eps)), float(max(1.0 - alpha, eps))


def eq23_side_score(g: np.ndarray, h_side: np.ndarray, omega_s: float, rho_s_eff: float, cfg: SimConfig) -> np.ndarray:
    """计算某一侧每个单元的式(23)打分。

    分子描述“该单元对当前侧最弱用户的有效增益潜力”，
    分母描述“放大噪声与有源偏置/动态功耗带来的代价”。
    该分数越高，说明把单元划为该侧有源单元越划算。
    """

    lambda1, lambda2 = eq23_weights(cfg)
    abs_g = np.abs(g)
    min_abs_h = np.min(np.abs(h_side), axis=0)
    mean_abs_h2 = np.mean(np.abs(h_side) ** 2, axis=0)
    numerator = omega_s * np.sqrt(rho_s_eff) * abs_g * min_abs_h
    denominator = lambda1 * rho_s_eff * (cfg.sigma_amp ** 2) * mean_abs_h2 + lambda2 * (cfg.P_bias + cfg.P_dyn_act)
    return numerator / (denominator + 1e-12)


def build_active_sets(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, L: int, alpha: float, cfg: SimConfig, protocol: str) -> Tuple[np.ndarray, np.ndarray]:
    """按 proposed 策略构造有源单元集合。

    做法是先分别计算单元服务透射侧和反射侧的收益分数，
    再选出综合分数最高的 L 个单元，并按更优服务侧划分为 A_t / A_r。
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
    """按均匀抽样方式选择有源单元，用作 baseline。"""

    if L <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    idx = np.linspace(0, cfg.N - 1, L, dtype=int)
    (rho_t_eff, rho_r_eff), _ = protocol_rho_tau(protocol, cfg)
    omega_t, omega_r = side_demand_weights(alpha)
    xi_t = eq23_side_score(g, h_t, omega_t, rho_t_eff, cfg)
    xi_r = eq23_side_score(g, h_r, omega_r, rho_r_eff, cfg)
    best_side = np.where(xi_t[idx] >= xi_r[idx], 0, 1)
    return np.sort(idx[best_side == 0]), np.sort(idx[best_side == 1])


def build_ss_subsurfaces(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, A_t: np.ndarray, A_r: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    """在 SS 协议下构造透射/反射子表面。

    SS 模式中每个无源单元只能固定归属一侧，因此这里会把
    非有源单元按两侧需求与链路强度比较后分配到 P_t / P_r，
    最终得到完整子表面 S_t / S_r。
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
    """提取闭式增益解所需的四个标量参数 a, b, c, d。

    `a` 对应无源链路的最弱等效增益；
    `b` 对应有源链路在单位增益下的最弱等效增益；
    `c` 是接收端底噪；
    `d` 是放大器噪声经过信道后的最坏放大系数。
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
    a = float(np.min(passive_sum) + 1e-12)
    b = float(np.min(active_unit))
    c = float(sigma0 ** 2)
    d = float(np.max(amp_noise_coeff) + 1e-12)
    return a, b, c, d


def beta_se_closed_form(a: float, b: float, c: float, d: float, beta_max: float) -> float:
    """根据闭式表达式求单侧最优放大增益 beta。"""

    if a <= 1e-12 or b <= 1e-12 or d <= 1e-12:
        return 1.0
    beta = b * c / (a * d)
    return float(np.clip(beta, 1.0, beta_max))


def side_required_power(g: np.ndarray, h_side: np.ndarray, passive_idx: np.ndarray, active_idx: np.ndarray, beta: float, rate_target_side: float, rho_eff: float, sigma0: float, sigma_amp: float) -> float:
    """计算满足某一侧目标速率所需的发射功率。

    这里按该侧用户平均分配速率门限，再逐个用户求满足目标 SNR
    所需的最小功率，最后对该侧用户进行求和。
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
    """汇总总功耗，并拆出静态、动态、基站发射、有源放大四部分。"""

    static = cfg.P_ctrl + cfg.P_bias * (Lt + Lr)
    dynamic = cfg.P_dyn_base + cfg.P_dyn_act * (Lt + Lr)
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
# 单个候选点与单条前沿求解
# =========================
def evaluate_candidate(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, A_t: np.ndarray, A_r: np.ndarray, Rtarget: float, alpha: float, cfg: SimConfig, protocol: str = "ES") -> Optional[Dict[str, float]]:
    """评估一个候选有源划分在给定目标 SE 下的可行性与 EE。

    输入是固定的有源集合 A_t / A_r 与速率拆分比例 alpha。
    输出包含该候选点的总功耗、发射功率、最优增益以及功耗分解。
    若峰值发射功率超过约束，则返回 `None` 表示不可行。
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
    beta_t = 1.0 if len(A_t) == 0 else beta_se_closed_form(a_t, b_t, c_t, d_t, cfg.beta_max)
    beta_r = 1.0 if len(A_r) == 0 else beta_se_closed_form(a_r, b_r, c_r, d_r, cfg.beta_max)
    if mode == "TS":
        Rt_eff = Rt / tau_t
        Rr_eff = Rr / tau_r
    else:
        Rt_eff = Rt
        Rr_eff = Rr
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


def solve_frontier_point(g: np.ndarray, h_t: np.ndarray, h_r: np.ndarray, Rtarget: float, cfg: SimConfig, scheme: str = "proposed", protocol: str = "ES") -> Optional[Dict[str, float]]:
    """在固定目标 SE 下，搜索某种方案的最优 EE 点。

    搜索变量包括：
    1. 总有源单元数 L；
    2. 透射/反射侧速率拆分比例 alpha；
    3. 给定方案下的有源单元分配方式。
    最终保留总功耗最小的候选点。
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
            if scheme == "passive":
                A_t = np.array([], dtype=int)
                A_r = np.array([], dtype=int)
            elif scheme == "uniform":
                A_t, A_r = build_uniform_active_sets(g, h_t, h_r, L, alpha, cfg, protocol)
            else:
                A_t, A_r = build_active_sets(g, h_t, h_r, L, alpha, cfg, protocol)
            res = evaluate_candidate(g, h_t, h_r, A_t, A_r, Rtarget, alpha, cfg, protocol)
            if res is None:
                continue
            if best is None or res["Ptot"] < best["Ptot"]:
                best = res
    return best


# =========================
# Monte Carlo 平均与敏感性分析
# =========================
def average_frontier(cfg: SimConfig, protocol: str = "ES", schemes: Tuple[str, ...] = ("passive", "uniform", "active", "proposed")) -> Tuple[Dict[str, List[Optional[Dict[str, float]]]], Dict[str, List[List[Optional[Dict[str, float]]]]]]:
    """对所有目标速率点做 Monte Carlo 平均，得到平均 Pareto 前沿。"""

    raw: Dict[str, List[List[Optional[Dict[str, float]]]]] = {s: [[] for _ in cfg.R_targets] for s in schemes}
    for mc in range(cfg.num_mc):
        g, h_t, h_r = generate_channels(cfg, seed=mc + 1)
        for i, R in enumerate(cfg.R_targets):
            for scheme in schemes:
                raw[scheme][i].append(solve_frontier_point(g, h_t, h_r, R, cfg, scheme, protocol))
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
                entry[key] = float(np.mean([r[key] for r in feas]))
            pts.append(entry)
        avg[scheme] = pts
    return avg, raw


def sweep_sigma_amp(sigmas: Sequence[float], cfg: SimConfig, Rtarget: float, protocol: str) -> List[Dict[str, float]]:
    """扫描放大器噪声标准差，观察最优增益和 EE 的变化。"""

    out = []
    for sigma_amp in sigmas:
        cfg_s = SimConfig(**asdict(cfg))
        cfg_s.sigma_amp = float(sigma_amp)
        vals = []
        for mc in range(cfg_s.num_mc):
            g, h_t, h_r = generate_channels(cfg_s, seed=mc + 1)
            res = solve_frontier_point(g, h_t, h_r, Rtarget, cfg_s, "proposed", protocol)
            if res is not None:
                vals.append(res)
        if vals:
            out.append({"sigma_amp": float(sigma_amp), "beta_t": float(np.mean([r["beta_t"] for r in vals])), "beta_r": float(np.mean([r["beta_r"] for r in vals])), "beta_avg": float(np.mean([r["beta_avg"] for r in vals])), "active_ratio": float(np.mean([r["L"] / cfg_s.N for r in vals])), "EE": float(np.mean([r["EE"] for r in vals]))})
        else:
            out.append({"sigma_amp": float(sigma_amp), "beta_t": None, "beta_r": None, "beta_avg": None, "active_ratio": None, "EE": None})
    return out


def sweep_update_frequency(f_updates: Sequence[float], cfg: SimConfig, Rtarget: float, protocol: str) -> List[Dict[str, float]]:
    """扫描更新频率，用动态功耗缩放来刻画开关/重配置开销。"""

    out = []
    for fu in f_updates:
        cfg_s = SimConfig(**asdict(cfg))
        scale = fu / cfg.f_ref
        cfg_s.P_dyn_base = cfg.P_dyn_base * scale
        cfg_s.P_dyn_act = cfg.P_dyn_act * scale
        vals = []
        for mc in range(cfg_s.num_mc):
            g, h_t, h_r = generate_channels(cfg_s, seed=mc + 1)
            res = solve_frontier_point(g, h_t, h_r, Rtarget, cfg_s, "proposed", protocol)
            if res is not None:
                vals.append(res)
        if vals:
            out.append({"f_update": float(fu), "EE": float(np.mean([r["EE"] for r in vals]))})
        else:
            out.append({"f_update": float(fu), "EE": None})
    return out


def average_breakdown_for_R(cfg: SimConfig, raw: Dict[str, List[List[Optional[Dict[str, float]]]]], Rtarget: float) -> Dict[str, Dict[str, float]]:
    """在指定目标速率处，对不同方案的功耗分量做平均统计。"""

    idx = list(cfg.R_targets).index(int(Rtarget))
    out: Dict[str, Dict[str, float]] = {}
    for scheme in raw:
        feas = [r for r in raw[scheme][idx] if r is not None]
        if not feas:
            continue
        out[scheme] = {"static": float(np.mean([r["static"] for r in feas])), "dynamic": float(np.mean([r["dynamic"] for r in feas])), "bs": float(np.mean([r["bs"] for r in feas])), "amp": float(np.mean([r["amp"] for r in feas])), "EE": float(np.mean([r["EE"] for r in feas]))}
    return out


# =========================
# 绘图
# =========================
def style() -> None:
    """统一图像风格，便于论文插图保持一致。"""

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.dpi": 220, "lines.linewidth": 2.0})


def plot_pareto(avg: Dict[str, List[Optional[Dict[str, float]]]], out_dir: Path) -> None:
    """绘制 ES 协议下不同方案的 EE-SE Pareto 前沿。"""

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for scheme, label, marker in [("passive", "Passive STAR-RIS", "o"), ("uniform", "Uniform hybrid", "s"), ("active", "Fully active", "^"), ("proposed", "Proposed sparse hybrid", "D")]:
        pts = [p for p in avg[scheme] if p is not None]
        ax.plot([p["R"] for p in pts], [p["EE"] for p in pts], marker=marker, label=label)
    ax.set_xlabel("Sum spectral efficiency (bit/s/Hz)")
    ax.set_ylabel("Energy efficiency (bit/J)")
    ax.set_title("EE-SE Pareto frontier under ES")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)


def plot_design_rules(protocol_frontier: Dict[str, List[Optional[Dict[str, float]]]], cfg: SimConfig, out_dir: Path) -> None:
    """绘制 ES / TS / SS 下的有源比例与侧向增益设计规律。"""

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for protocol, marker in [("ES", "o"), ("TS", "s"), ("SS", "D")]:
        pts = [p for p in protocol_frontier[protocol] if p is not None]
        axes[0].plot([p["R"] for p in pts], [p["L"] / cfg.N for p in pts], marker=marker, label=protocol)
    axes[0].set_xlabel("Target sum-SE (bit/s/Hz)")
    axes[0].set_ylabel("Active ratio")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("(a) Active ratio under ES/TS/SS")
    axes[0].legend(frameon=True, fontsize=8, ncol=3)
    style_map = {"ES": ("o", "-"), "TS": ("s", "--"), "SS": ("D", ":")}
    for protocol in ["ES", "TS", "SS"]:
        pts = [p for p in protocol_frontier[protocol] if p is not None]
        marker, linestyle = style_map[protocol]
        axes[1].plot([p["R"] for p in pts], [p["beta_t"] for p in pts], marker=marker, linestyle=linestyle, label=rf"{protocol} $\beta_t^\star$")
        axes[1].plot([p["R"] for p in pts], [p["beta_r"] for p in pts], marker=marker, linestyle=(0, (1, 1)), label=rf"{protocol} $\beta_r^\star$")
    axes[1].set_xlabel("Target sum-SE (bit/s/Hz)")
    axes[1].set_ylabel("Side-wise gain")
    axes[1].set_ylim(1.0, cfg.beta_max + 0.2)
    axes[1].set_title("(b) Side-wise gains under ES/TS/SS")
    axes[1].legend(frameon=True, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_design_rules.png", bbox_inches="tight")
    plt.close(fig)


def plot_noise_and_dynamics(protocol_noise: Dict[str, List[Dict[str, float]]], protocol_dyn: Dict[str, List[Dict[str, float]]], out_dir: Path) -> None:
    """绘制噪声敏感性与更新频率敏感性两张曲线。"""

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for protocol, marker in [("ES", "o"), ("TS", "s"), ("SS", "D")]:
        noise = protocol_noise[protocol]
        axes[0].plot([d["sigma_amp"] for d in noise], [np.nan if d["beta_avg"] is None else d["beta_avg"] for d in noise], marker=marker, label=protocol)
        dyn = protocol_dyn[protocol]
        axes[1].plot([d["f_update"] for d in dyn], [np.nan if d["EE"] is None else d["EE"] for d in dyn], marker=marker, label=protocol)
    axes[0].set_xlabel(r"Amplifier-noise std. $\sigma_a$")
    axes[0].set_ylabel("Average side-wise gain")
    axes[0].set_title("(a) Gain sensitivity under ES/TS/SS")
    axes[0].legend(frameon=True, fontsize=8, ncol=3)
    axes[1].set_xlabel("Update frequency (Hz)")
    axes[1].set_ylabel("Energy efficiency (bit/J)")
    axes[1].set_title("(b) EE under switching overhead")
    axes[1].legend(frameon=True, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_noise_dynamic.png", bbox_inches="tight")
    plt.close(fig)


def plot_power_breakdown(breakdown: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    """绘制代表性速率点上的功耗分解柱状图。"""

    schemes = list(breakdown.keys())
    static = np.array([breakdown[s]["static"] for s in schemes])
    dynamic = np.array([breakdown[s]["dynamic"] for s in schemes])
    bs = np.array([breakdown[s]["bs"] for s in schemes])
    amp = np.array([breakdown[s]["amp"] for s in schemes])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(schemes))
    ax.bar(x, static, label="Static")
    ax.bar(x, dynamic, bottom=static, label="Dynamic")
    ax.bar(x, bs, bottom=static + dynamic, label="BS transmit")
    ax.bar(x, amp, bottom=static + dynamic + bs, label="Amplification")
    ax.set_xticks(x)
    ax.set_xticklabels(["Passive", "Uniform", "Active", "Proposed"])
    ax.set_ylabel("Power consumption (W)")
    ax.set_title("Power breakdown at representative SE target")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_power_breakdown.png", bbox_inches="tight")
    plt.close(fig)


# =========================
# 主流程
# =========================
def main() -> None:
    """运行完整复现实验，输出图像与 JSON 汇总结果。"""

    style()
    base_dir = Path(__file__).resolve().parents[1]
    fig_dir = base_dir / "figs"
    results_dir = base_dir / "results"
    fig_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    cfg = SimConfig()

    avg_es, raw_es = average_frontier(cfg, protocol="ES")
    breakdown_es = average_breakdown_for_R(cfg, raw_es, Rtarget=12)

    protocol_frontier: Dict[str, List[Optional[Dict[str, float]]]] = {}
    for protocol in ("ES", "TS", "SS"):
        avg_p, _ = average_frontier(cfg, protocol=protocol, schemes=("proposed",))
        protocol_frontier[protocol] = avg_p["proposed"]

    sigmas = np.linspace(0.02, 0.18, 9)
    f_updates = np.arange(50, 551, 50)
    protocol_noise = {p: sweep_sigma_amp(sigmas, cfg, Rtarget=12.0, protocol=p) for p in ("ES", "TS", "SS")}
    protocol_dyn = {p: sweep_update_frequency(f_updates, cfg, Rtarget=12.0, protocol=p) for p in ("ES", "TS", "SS")}

    plot_pareto(avg_es, fig_dir)
    plot_design_rules(protocol_frontier, cfg, fig_dir)
    plot_noise_and_dynamics(protocol_noise, protocol_dyn, fig_dir)
    plot_power_breakdown(breakdown_es, fig_dir)
    summary = {"config": {**asdict(cfg), "lambda1_effective": eq23_weights(cfg)[0], "lambda2_effective": eq23_weights(cfg)[1]}, "average_frontier_es": avg_es, "protocol_design_rules": protocol_frontier, "protocol_noise_sweep": protocol_noise, "protocol_dynamic_sweep": protocol_dyn, "power_breakdown_at_R12_es": breakdown_es}
    with open(results_dir / "hybrid_star_ris_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Simulation completed.")
    print(f"Figures saved under: {fig_dir}")
    print(f"Summary JSON: {results_dir / 'hybrid_star_ris_results.json'}")


if __name__ == "__main__":
    main()
