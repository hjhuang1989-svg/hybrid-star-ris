"""Draw the manuscript EPS figures directly from the generated JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCHEMES = ("passive", "uniform", "uniform_opt", "random", "strongest", "active", "proposed")
LABELS = {
    "passive": "Passive",
    "uniform": "Fixed uniform",
    "uniform_opt": "Opt. uniform",
    "random": "Single-random sparse",
    "strongest": "Strongest greedy",
    "active": "Active",
    "proposed": "Proposed",
}
SHORT_LABELS = {
    "passive": "Pass.",
    "uniform": "F.-unif.",
    "uniform_opt": "Opt.-unif.",
    "random": "Rand.",
    "strongest": "Strong.",
    "active": "Active",
    "proposed": "Prop.",
}
MARKERS = {
    "passive": "o",
    "uniform": "s",
    "uniform_opt": "v",
    "random": "*",
    "strongest": "x",
    "active": "^",
    "proposed": "D",
}
COLORS = {
    "passive": "#66737f",
    "uniform": "#8a40b0",
    "uniform_opt": "#733399",
    "random": "#e6731a",
    "strongest": "#339933",
    "active": "#d92626",
    "proposed": "#1a73b8",
}


def configure_style() -> None:
    """Set IEEE-friendly vector-figure defaults."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 6.6,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "axes.grid": True,
            "grid.alpha": 1.0,
            "grid.color": "0.86",
            "grid.linewidth": 0.45,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.05,
            "lines.markersize": 3.4,
            "patch.linewidth": 0.45,
            "ps.fonttype": 3,
            "pdf.fonttype": 3,
            "savefig.facecolor": "white",
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read the simulation summary JSON."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rate_key(key: str) -> float:
    """Convert a JSON rate key to a numeric rate."""

    return float(str(key).replace("_", "."))


def finite(values: list[Any] | np.ndarray) -> np.ndarray:
    """Convert a sequence to a finite-friendly float array."""

    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def save_eps(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    """Save one figure as EPS and close it."""

    fig.savefig(fig_dir / f"{stem}.eps", format="eps", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def plot_pareto(data: dict[str, Any], fig_dir: Path) -> None:
    """Draw Fig. 2, the approximated EE-SE boundary."""

    fig, ax = plt.subplots(figsize=(3.55, 2.20))
    for scheme in SCHEMES:
        rows = [row for row in data["average_boundary_es"].get(scheme, []) if row is not None]
        if not rows:
            continue
        r_vals = finite([row["R"] for row in rows])
        ee_vals = finite([row["EE"] for row in rows])
        feas = finite([row.get("feas_prob", 1.0) for row in rows])
        ax.plot(
            r_vals,
            ee_vals,
            marker=MARKERS[scheme],
            color=COLORS[scheme],
            label=LABELS[scheme],
            markerfacecolor="white" if scheme in {"passive", "uniform", "uniform_opt"} else COLORS[scheme],
            markeredgewidth=0.85,
        )
        outage = feas < 0.995
        if np.any(outage):
            ax.plot(
                r_vals[outage],
                ee_vals[outage],
                linestyle="None",
                marker=MARKERS[scheme],
                color=COLORS[scheme],
                markerfacecolor="none",
                markeredgewidth=1.0,
            )
            if scheme == "passive":
                for x_val, y_val, p_val in zip(r_vals[outage], ee_vals[outage], feas[outage]):
                    if np.isclose(x_val, 18.0):
                        xytext = (17.05, y_val + 0.36)
                    elif np.isclose(x_val, 20.0):
                        y_text = y_val + 0.34
                        ax.annotate(
                            r"$p_f^{\mathrm{pass}}$",
                            (x_val, y_val),
                            textcoords="data",
                            xytext=(19.78, y_text),
                            fontsize=6.2,
                            ha="right",
                        )
                        ax.annotate(
                            rf"$={p_val:.2f}$",
                            (x_val, y_val),
                            textcoords="data",
                            xytext=(19.78, y_text),
                            fontsize=6.2,
                            ha="left",
                        )
                        continue
                    else:
                        xytext = (x_val + 0.15, y_val + 0.18)
                    ax.annotate(
                        rf"$p_f^{{\mathrm{{pass}}}}={p_val:.2f}$",
                        (x_val, y_val),
                        textcoords="data",
                        xytext=xytext,
                        fontsize=6.2,
                    )
    ax.set_xlabel("Sum spectral efficiency (bit/s/Hz)", fontsize=6.6, labelpad=1.0)
    ax.set_ylabel("Energy efficiency", fontsize=6.6, labelpad=1.0)
    ax.set_xlim(1, 21)
    ax.set_ylim(bottom=2.0)
    ax.set_xticks(np.arange(2, 21, 2))
    ax.tick_params(axis="both", which="major", pad=1.0, length=2.3, labelsize=5.9)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.14, 0.02),
        ncol=2,
        frameon=True,
        framealpha=1.0,
        borderpad=0.16,
        labelspacing=0.10,
        handlelength=0.92,
        handletextpad=0.24,
        columnspacing=0.38,
        fontsize=4.8,
    )
    fig.tight_layout(pad=0.15)
    save_eps(fig, fig_dir, "fig_pareto_boundary")


def plot_noise_sigma(data: dict[str, Any], fig_dir: Path) -> None:
    """Draw Fig. 3(a), amplifier-noise sensitivity."""

    fig, ax = plt.subplots(figsize=(1.62, 1.35))
    markers = ("o", "^", "s", "v", "D")
    rate_keys = sorted(data["es_noise_sweep_by_rate"], key=rate_key)
    for idx, key in enumerate(rate_keys):
        rows = data["es_noise_sweep_by_rate"][key]
        ax.plot(
            finite([row["sigma_amp"] for row in rows]),
            finite([row["beta_avg"] for row in rows]),
            marker=markers[idx % len(markers)],
            label=rf"$R_\Sigma={rate_key(key):g}$",
        )
    ax.set_xlabel(r"Amplifier-noise std. $\sigma_a$", fontsize=6.6, labelpad=0.4)
    ax.set_ylabel(r"Amplification $\beta$", fontsize=6.6, labelpad=0.6)
    ax.tick_params(axis="both", pad=1.0, length=2.0, labelsize=5.9)
    ax.legend(loc="lower left", ncol=2, fontsize=4.3, borderpad=0.18, labelspacing=0.12, handlelength=1.15)
    fig.tight_layout(pad=0.06)
    save_eps(fig, fig_dir, "fig_noise_sigma")


def plot_noise_update(data: dict[str, Any], fig_dir: Path) -> None:
    """Draw Fig. 3(b), switching-overhead sensitivity."""

    fig, ax = plt.subplots(figsize=(1.62, 1.35))
    markers = ("o", "^", "s", "v", "D")
    rate_keys = sorted(data["es_dynamic_sweep_by_rate"], key=rate_key)
    for idx, key in enumerate(rate_keys):
        rows = data["es_dynamic_sweep_by_rate"][key]
        ax.plot(
            finite([row["f_update"] for row in rows]),
            finite([row["EE"] for row in rows]),
            marker=markers[idx % len(markers)],
            label=rf"$R_\Sigma={rate_key(key):g}$",
        )
    ax.set_xlabel("Update frequency (Hz)", fontsize=6.6, labelpad=0.4)
    ax.set_ylabel("Energy efficiency", fontsize=6.6, labelpad=0.6)
    ax.tick_params(axis="both", pad=1.0, length=2.0, labelsize=5.9)
    ax.legend(loc="upper right", ncol=2, fontsize=4.3, borderpad=0.18, labelspacing=0.12, handlelength=1.15)
    fig.tight_layout(pad=0.06)
    save_eps(fig, fig_dir, "fig_noise_update")


def plot_power_breakdown(data: dict[str, Any], fig_dir: Path) -> None:
    """Draw Fig. 4, the two-panel power-component breakdown."""

    blocks = [data["power_breakdown_at_R12_es"], data["power_breakdown_at_R16_es"]]
    rates = [12, 16]
    component_colors = ["#3373d6", "#f28c2b", "#33a34a", "#d62f2f"]
    fig, axes = plt.subplots(1, 2, figsize=(3.65, 1.42))
    for panel, (ax, block, rate) in enumerate(zip(axes, blocks, rates)):
        schemes = [scheme for scheme in SCHEMES if scheme in block]
        schemes = sorted(
            schemes,
            key=lambda s: block[s]["static"] + block[s]["dynamic"] + block[s]["bs"] + block[s]["amp"],
            reverse=True,
        )
        vals = np.asarray(
            [[block[s]["static"], block[s]["dynamic"], block[s]["bs"], block[s]["amp"]] for s in schemes],
            dtype=float,
        )
        totals = np.sum(vals, axis=1)
        x = np.arange(len(schemes))
        bottom = np.zeros(len(schemes))
        labels = ["Static", "Dynamic", "BS transmit", "Amplification"]
        for comp_idx, label in enumerate(labels):
            ax.bar(x, vals[:, comp_idx], bottom=bottom, width=0.72, color=component_colors[comp_idx], label=label)
            bottom += vals[:, comp_idx]
        for x_val, total in zip(x, totals):
            ax.text(x_val, total + 0.028 * np.nanmax(totals), f"{total:.2f}", ha="center", va="bottom", fontsize=5.0)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[s] for s in schemes], rotation=24, ha="right", fontsize=5.9)
        ax.set_ylabel("Power (W)", fontsize=6.6, labelpad=0.6)
        ax.text(
            0.5,
            -0.46,
            rf"({chr(ord('a') + panel)}) $R_\Sigma={rate}$ bit/s/Hz",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )
        ax.set_ylim(0.0, 1.24 * np.nanmax(totals))
        ax.tick_params(axis="both", pad=1.0, length=2.0, labelsize=5.9)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.00, 1.01),
            ncol=2,
            fontsize=4.4,
            frameon=True,
            borderpad=0.16,
            handlelength=1.18,
            handletextpad=0.24,
            columnspacing=0.35,
        )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.34, wspace=0.34)
    save_eps(fig, fig_dir, "fig_power_breakdown")


def plot_power_trace(data: dict[str, Any], fig_dir: Path) -> None:
    """Draw Fig. 5, the candidate-search total-power trace."""

    fields = sorted(data["power_trace_by_rate"], key=rate_key)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(2.34, 1.04))
    for idx, key in enumerate(fields):
        row = data["power_trace_by_rate"][key]
        x_vals = finite(row["candidate_index"])
        med = finite(row["median"])
        q25 = finite(row["q25"])
        q75 = finite(row["q75"])
        color = cmap(idx / max(len(fields) - 1, 1))
        band = np.isfinite(x_vals) & np.isfinite(q25) & np.isfinite(q75)
        if np.any(band):
            band_color = tuple(0.86 + 0.14 * channel for channel in color[:3])
            ax.fill_between(x_vals[band], q25[band], q75[band], color=band_color, linewidth=0.0)
        good = np.isfinite(x_vals) & np.isfinite(med)
        ax.plot(x_vals[good], med[good], color=color, linewidth=0.9, label=rf"$R_\Sigma={row['R']:g}$")
    ax.set_xlabel("Candidate update index", fontsize=6.6, labelpad=0.6)
    ax.set_ylabel("Power (W)", fontsize=6.6, labelpad=0.6)
    ax.tick_params(axis="both", pad=1.0, length=2.0, labelsize=5.9)
    ax.legend(loc="upper right", fontsize=4.6, frameon=True, borderpad=0.18, handlelength=1.15)
    fig.tight_layout(pad=0.06)
    save_eps(fig, fig_dir, "fig_power_trace_preview")


def plot_from_json(json_path: Path, fig_dir: Path) -> None:
    """Draw all paper EPS figures from a simulation JSON file."""

    configure_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = read_json(json_path)
    plot_pareto(data, fig_dir)
    plot_noise_sigma(data, fig_dir)
    plot_noise_update(data, fig_dir)
    plot_power_breakdown(data, fig_dir)
    plot_power_trace(data, fig_dir)


def default_paths() -> tuple[Path, Path]:
    """Return default JSON input and figure output paths."""

    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "results" / "hybrid_star_ris_results.json", base_dir / "figs"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    default_json, default_fig_dir = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=default_json, help="Path to hybrid_star_ris_results.json.")
    parser.add_argument("--fig-dir", type=Path, default=default_fig_dir, help="Directory for manuscript EPS figures.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    plot_from_json(args.json, args.fig_dir)
    print(f"Saved Python EPS figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
