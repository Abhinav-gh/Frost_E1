"""
Frost E1 — plots.py

Generate all required E1 plots from HDF5 log data.

Required outputs:
  1. dirty_fraction_vs_tolerance.png   — phi(epsilon) for each system/T
  2. perturbation_force_error_joint.png — |Delta r| vs induced force error
  3. delta_r_vs_delta_rhat.png         — radial vs angular perturbation
  4. force_error_by_tolerance.png      — force error statistics per epsilon
  5. cache_lifetime.png                — cache lifetime distributions
  6. spatial_autocorrelation.png       — Moran's I vs epsilon
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Design constants ──────────────────────────────────────────────────────────
SYSTEM_COLORS = {
    "Si": "#2196F3",
    "Cu": "#FF9800",
    "Li3PO4": "#9C27B0",
    "water": "#00BCD4",
    "SiO2": "#4CAF50",
}
TEMP_LINESTYLES = {300: "-", 1000: "--"}
SEED_MARKERS = {42: "o", 137: "s", 271: "^"}
FIGURE_DPI = 150


def save_fig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_dirty_fraction_vs_tolerance(
    results: List[dict],  # list of {metadata, dirty_summary_df, ...}
    output_path: Path,
) -> None:
    """
    OUTPUT 1: Dirty fraction phi vs tolerance epsilon.

    One curve per (system, temperature). Log x-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, temp in zip(axes, [300, 1000]):
        ax.set_title(f"T = {temp} K", fontsize=13)
        ax.set_xlabel("Tolerance ε (meV/Å)", fontsize=11)
        ax.set_ylabel("Dirty fraction φ", fontsize=11)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="red", linestyle=":", linewidth=1.5, label="Go/No-go threshold (φ=0.5)")
        ax.grid(True, which="both", alpha=0.3)

        for res in results:
            meta = res["metadata"]
            if int(meta.get("temperature_K", 0)) != temp:
                continue
            system = meta.get("system", "?")
            df = res.get("dirty_summary_df")
            if df is None or df.empty:
                continue

            color = SYSTEM_COLORS.get(system, "gray")
            ax.plot(
                df["epsilon"],
                df["median_dirty_fraction"],
                color=color,
                linestyle=TEMP_LINESTYLES.get(temp, "-"),
                marker="o",
                markersize=5,
                label=system,
            )
            # IQR shading
            if "p25_dirty_fraction" in df.columns:
                ax.fill_between(
                    df["epsilon"],
                    df["p25_dirty_fraction"],
                    df["p75_dirty_fraction"],
                    color=color,
                    alpha=0.15,
                )

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        # deduplicate labels
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="lower left")

    fig.suptitle("Frost E1 — Dirty Fraction vs Tolerance", fontsize=14, y=1.02)
    save_fig(fig, output_path)


def plot_perturbation_force_error_joint(
    force_error_frames: List[dict],  # list of {epsilon, delta_r, mean_force_error, ...}
    output_path: Path,
    model_rmse_eV: Optional[float] = None,
    system: str = "",
    temperature_K: int = 300,
) -> None:
    """
    OUTPUT 2: Joint distribution of |Delta r| vs induced force error.

    This is the decisive E1 plot.
    """
    if not force_error_frames:
        logger.warning("No force-error frames to plot for %s T=%d K", system, temperature_K)
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    epsilons = sorted(set(f["epsilon"] for f in force_error_frames))
    cmap = plt.cm.viridis
    norm = mcolors.LogNorm(vmin=min(epsilons), vmax=max(epsilons))

    for frame in force_error_frames:
        eps = frame["epsilon"]
        dr = frame.get("mean_delta_r", np.nan)
        fe = frame.get("mean_error", np.nan)
        ax.scatter(
            dr * 1000,  # convert A to mA for readability
            fe * 1000,  # eV/A -> meV/A
            c=[eps],
            cmap=cmap,
            norm=norm,
            s=40,
            alpha=0.7,
            edgecolors="none",
        )

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label="Tolerance ε (meV/Å)",
    )

    ax.set_xlabel("|Δr| (mÅ)", fontsize=12)
    ax.set_ylabel("Mean per-atom force error (meV/Å)", fontsize=12)
    ax.set_title(
        f"E1 Perturbation–Force-Error | {system} | T={temperature_K} K",
        fontsize=12,
    )

    if model_rmse_eV is not None:
        ax.axhline(
            model_rmse_eV * 1000,
            color="red",
            linestyle="--",
            label=f"Model RMSE = {model_rmse_eV*1000:.1f} meV/Å",
        )
    else:
        ax.text(
            0.98, 0.95,
            "Model RMSE: unavailable",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="gray",
        )

    ax.axhline(5.0, color="orange", linestyle=":", label="Go/No-go force threshold (5 meV/Å)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_fig(fig, output_path)


def plot_delta_r_vs_delta_rhat(
    perturbation_data: List[dict],  # {delta_r, delta_rhat, epsilon}
    output_path: Path,
    system: str = "",
    temperature_K: int = 300,
) -> None:
    """
    OUTPUT 3: Angular perturbation analysis. |Delta r| vs |Delta rhat|.
    """
    if not perturbation_data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: |Delta r| vs |Delta rhat|
    ax = axes[0]
    dr = np.array([d["delta_r"] for d in perturbation_data]) * 1000  # mA
    drhat = np.array([d["delta_rhat"] for d in perturbation_data])
    ax.hexbin(dr, drhat, gridsize=40, mincnt=1, cmap="Blues")
    ax.set_xlabel("|Δr| (mÅ)", fontsize=11)
    ax.set_ylabel("|Δr̂|", fontsize=11)
    ax.set_title(f"|Δr| vs |Δr̂| — {system} T={temperature_K}K", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: histograms
    ax2 = axes[1]
    ax2.hist(dr, bins=50, alpha=0.6, label="|Δr| (mÅ)", density=True)
    ax2_twin = ax2.twinx()
    ax2_twin.hist(drhat, bins=50, alpha=0.4, color="orange", label="|Δr̂|", density=True)
    ax2.set_xlabel("Perturbation magnitude", fontsize=11)
    ax2.set_ylabel("P(|Δr|)", fontsize=11)
    ax2_twin.set_ylabel("P(|Δr̂|)", fontsize=11)
    ax2.set_title("Perturbation distributions", fontsize=11)

    fig.suptitle(f"E1 Angular Perturbation Analysis — {system}", fontsize=13)
    save_fig(fig, output_path)


def plot_force_error_by_tolerance(
    epsilon_force_stats: Dict[float, dict],
    output_path: Path,
    system: str = "",
    temperature_K: int = 300,
) -> None:
    """
    OUTPUT 4: Force error statistics for each tolerance epsilon.
    """
    if not epsilon_force_stats:
        return

    epsilons = sorted(epsilon_force_stats.keys())
    mean_errors = [epsilon_force_stats[e]["mean_error"] * 1000 for e in epsilons]
    max_errors = [epsilon_force_stats[e]["max_error"] * 1000 for e in epsilons]
    rms_errors = [epsilon_force_stats[e]["rms_error"] * 1000 for e in epsilons]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(epsilons, mean_errors, "o-", label="Mean force error", color="#2196F3")
    ax.loglog(epsilons, rms_errors, "s--", label="RMS force error", color="#4CAF50")
    ax.loglog(epsilons, max_errors, "^:", label="Max force error", color="#F44336")
    ax.axhline(5.0, color="orange", linestyle=":", linewidth=2, label="Go/No-go: 5 meV/Å")
    ax.set_xlabel("Tolerance ε (meV/Å)", fontsize=12)
    ax.set_ylabel("Induced force error (meV/Å)", fontsize=12)
    ax.set_title(f"E1 Force Error by Tolerance — {system} T={temperature_K}K", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, output_path)


def plot_cache_lifetime(
    lifetime_data: Dict[float, dict],  # {epsilon: {lifetime stats}}
    output_path: Path,
    system: str = "",
    temperature_K: int = 300,
) -> None:
    """
    OUTPUT 5: Cache lifetime distribution per tolerance.
    """
    if not lifetime_data:
        return

    epsilons = sorted(lifetime_data.keys())
    mean_lifetimes = [lifetime_data[e].get("estimated_mean_lifetime_steps", 0) for e in epsilons]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(epsilons, mean_lifetimes, "o-", color="#9C27B0", linewidth=2)
    ax.set_xlabel("Tolerance ε (meV/Å)", fontsize=12)
    ax.set_ylabel("Estimated mean cache lifetime (steps)", fontsize=12)
    ax.set_title(f"E1 Cache Lifetime — {system} T={temperature_K}K", fontsize=12)
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)


def plot_spatial_autocorrelation(
    morans_I_data: Dict[float, float],  # {epsilon: Moran's I}
    output_path: Path,
    system: str = "",
    temperature_K: int = 300,
) -> None:
    """
    OUTPUT 6: Moran's I spatial autocorrelation vs epsilon.
    """
    if not morans_I_data:
        return

    epsilons = sorted(morans_I_data.keys())
    morans = [morans_I_data[e] for e in epsilons]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(epsilons, morans, "o-", color="#FF5722", linewidth=2)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="Random (I=0)")
    ax.set_xlabel("Tolerance ε (meV/Å)", fontsize=12)
    ax.set_ylabel("Moran's I", fontsize=12)
    ax.set_title(
        f"E1 Spatial Autocorrelation of Dirty Mask — {system} T={temperature_K}K",
        fontsize=12,
    )
    ax.set_ylim(-0.2, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)
