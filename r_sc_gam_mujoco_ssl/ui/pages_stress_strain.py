"""
pages_stress_strain.py — Stress-strain and joint loading tab
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def render_stress_strain_tab(ss_df: pd.DataFrame, loading_df: pd.DataFrame):
    st.header("Stress-Strain and Joint Loading Analysis")
    st.warning(
        "**Scientific Boundary**: Values below are **approximate normalised estimates** "
        "for research visualisation and hypothesis generation. They are NOT confirmed "
        "tissue stress/strain, NOT measured forces, and must NOT be used for "
        "clinical decisions."
    )

    if ss_df.empty:
        st.info("Stress-Strain engine is disabled. Enable it in the sidebar.")
        return

    # --- Strain heatmap ---
    st.subheader("Estimated Muscle-Tendon Strain Heatmap")
    _render_heatmap(ss_df, "estimated_strain", "Estimated Strain",
                    cmap="RdYlGn_r", vmin=-0.15, vmax=0.15)

    # --- Stress heatmap ---
    st.subheader("Estimated Actuator Stress Proxy Heatmap (normalised 0–1)")
    _render_heatmap(ss_df, "norm_stress", "Norm. Stress Proxy",
                    cmap="YlOrRd", vmin=0.0, vmax=1.0)

    # --- Burden heatmap ---
    st.subheader("Stress Burden Heatmap (stress × phase functional weight)")
    _render_heatmap(ss_df, "stress_burden", "Stress Burden",
                    cmap="Reds", vmin=0.0, vmax=None)

    # --- Summary bar chart ---
    st.subheader("Mean Stress Burden by Muscle")
    summ = ss_df.groupby(["muscle", "full_name"])[
        ["stress_burden", "strain_burden"]].mean().reset_index()
    summ = summ.sort_values("stress_burden", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(summ))
    ax.bar(x, summ["stress_burden"], label="Stress burden", color="#dc2626", alpha=0.85)
    ax.bar(x, summ["strain_burden"], label="Strain burden", color="#2563eb", alpha=0.65)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summ["muscle"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Burden index")
    ax.set_title("Mean Stress/Strain Burden by Muscle (simulation-estimated)",
                 fontsize=10, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Loading index summary ---
    if not loading_df.empty:
        st.subheader("Joint Loading Index Summary")
        summ_jli = loading_df.groupby(["side", "joint"])[
            "joint_loading_index"].mean().reset_index()
        summ_jli.columns = ["side", "joint", "mean_JLI"]
        summ_jli["mean_JLI"] = summ_jli["mean_JLI"].round(3)
        st.dataframe(
            summ_jli.sort_values("mean_JLI", ascending=False),
            use_container_width=True,
        )

    # --- Full table ---
    st.subheader("Full Stress-Strain Data Table")
    display_cols = ["side", "muscle", "phase", "estimated_strain",
                    "estimated_stress_proxy", "norm_stress",
                    "strain_burden", "stress_burden", "recruitment_likelihood"]
    st.dataframe(
        ss_df[display_cols].style.background_gradient(
            subset=["norm_stress", "stress_burden"], cmap="YlOrRd"),
        use_container_width=True,
    )

    # --- Interpretations ---
    st.subheader("Key Stress-Strain Interpretations")
    high = ss_df[ss_df["norm_stress"] >= 0.35].sort_values(
        "norm_stress", ascending=False).head(6)
    for _, r in high.iterrows():
        st.write(f"**{r['side']} {r['muscle']} ({r['full_name']}) — {r['phase']}** "
                 f"(norm_stress={r['norm_stress']:.2f}): {r['interpretation']}")


def _render_heatmap(df: pd.DataFrame, value_col: str, label: str,
                    cmap: str, vmin, vmax):
    try:
        pivot = df.pivot_table(
            index=["side", "muscle"], columns="phase",
            values=value_col, aggfunc="mean",
        )
        phase_order = ["IC", "LR", "MSt", "TSt", "PSw", "ISw", "MSw", "TSw"]
        pivot = pivot.reindex(columns=[p for p in phase_order if p in pivot.columns])

        if pivot.empty:
            st.write("No data for heatmap.")
            return

        actual_vmax = float(pivot.values[~np.isnan(pivot.values)].max()) if vmax is None else vmax
        cmap_obj = LinearSegmentedColormap.from_list("custom", [
            "#f8fafc", "#fde68a", "#f97316", "#b91c1c"]) if cmap == "custom" else cmap

        fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.55 + 1)))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap_obj,
                       vmin=vmin if vmin is not None else None,
                       vmax=actual_vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        row_labels = [f"{s}·{m}" for s, m in pivot.index]
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(row_labels, fontsize=8)
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(v) > actual_vmax * 0.55 else "#111")
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        ax.set_title(f"{label} — Phase Heatmap (simulation-estimated)",
                     fontsize=10, weight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.write(f"Could not render heatmap: {e}")
