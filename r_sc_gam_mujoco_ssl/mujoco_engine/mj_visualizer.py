"""
mj_visualizer.py
Generate matplotlib figures from simulation / proxy data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_joint_trajectories(sim_result: Dict[str, Any]) -> plt.Figure:
    """Time-series joint angle plots for left and right sides."""
    t = sim_result["t"]
    joints = ["hip", "knee", "ankle"]
    sides  = ["left", "right"]
    colors = {"left": "#2563eb", "right": "#dc2626"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, j in zip(axes, joints):
        for s in sides:
            if s in sim_result["joints"] and j in sim_result["joints"][s]:
                pos = sim_result["joints"][s][j].get("pos_deg", np.zeros_like(t))
                ax.plot(t, pos, color=colors[s], lw=1.8, label=f"{s} {j}", alpha=0.85)
        ax.set_title(f"{j.capitalize()} angle (deg)", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Degrees")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Simulated Joint Trajectories — {sim_result.get('phenotype','')}\n"
        f"({'MuJoCo' if sim_result.get('used_mujoco') else 'Proxy'})",
        fontsize=11, weight="bold"
    )
    fig.tight_layout()
    return fig


def plot_torque_heatmap(phase_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of joint torque proxy per side/joint/phase."""
    if phase_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    pivot = phase_df.pivot_table(
        index=["side", "joint"], columns="phase", values="torque_proxy", aggfunc="mean"
    )
    phase_order = ["IC", "LR", "MSt", "TSt", "PSw", "ISw", "MSw", "TSw"]
    pivot = pivot.reindex(columns=[p for p in phase_order if p in pivot.columns])

    fig, ax = plt.subplots(figsize=(11, max(4, len(pivot) * 0.7 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    row_labels = [f"{s}·{j}" for s, j in pivot.index]
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > pivot.values.max() * 0.6 else "#111")
    fig.colorbar(im, ax=ax, label="Torque proxy (N·m)", shrink=0.8)
    ax.set_title("Joint Torque Proxy — Phase Heatmap", fontsize=11, weight="bold")
    fig.tight_layout()
    return fig


def plot_loading_index_heatmap(phase_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of joint loading index per side/joint/phase."""
    if phase_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    pivot = phase_df.pivot_table(
        index=["side", "joint"], columns="phase", values="joint_loading_index", aggfunc="mean"
    )
    phase_order = ["IC", "LR", "MSt", "TSt", "PSw", "ISw", "MSw", "TSw"]
    pivot = pivot.reindex(columns=[p for p in phase_order if p in pivot.columns])

    fig, ax = plt.subplots(figsize=(11, max(4, len(pivot) * 0.7 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    row_labels = [f"{s}·{j}" for s, j in pivot.index]
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.6 else "#111")
    fig.colorbar(im, ax=ax, label="Joint Loading Index (0–1)", shrink=0.8)
    ax.set_title("Joint Loading Index — Phase Heatmap", fontsize=11, weight="bold")
    fig.tight_layout()
    return fig


def plot_tendon_lengths(sim_result: Dict[str, Any]) -> plt.Figure:
    """Time-series tendon-proxy lengths."""
    t = sim_result["t"]
    tendon_data = sim_result.get("tendon_length_proxy", {})
    colors = {"left": "#2563eb", "right": "#dc2626"}

    all_tendons = set()
    for s_data in tendon_data.values():
        all_tendons.update(s_data.keys())
    all_tendons = sorted(all_tendons)
    if not all_tendons:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No tendon data", ha="center")
        return fig

    ncols = 2
    nrows = int(np.ceil(len(all_tendons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    for ax, tname in zip(axes_flat, all_tendons):
        for side, s_data in tendon_data.items():
            if tname in s_data:
                l = s_data[tname]
                if len(l) == len(t):
                    ax.plot(t, l, color=colors.get(side, "gray"),
                            lw=1.8, label=side, alpha=0.85)
        ax.set_title(tname.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Est. length (m)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(all_tendons):]:
        ax.axis("off")

    fig.suptitle("Estimated Muscle-Tendon Length Proxies", fontsize=11, weight="bold")
    fig.tight_layout()
    return fig
