"""
pages_archetype.py — Archetype model, kinematic curves, recruitment, rehab priority
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

from core.r_sc_gam_core import (
    PHENOTYPES, NORMAL_CURVES, GRID, PHASES, PHASE_KEYS, N_PH,
    spasticity_scores, weakness_scores, recruitment_matrix, rehab_priority,
)


PHASE_BAND_COLORS = ["#eef4ff", "#dbe7ff"]


def _add_phase_bands(ax, ymin, ymax, label_top=False):
    for i, (k, full, s, e, c) in enumerate(PHASES):
        ax.axvspan(s, e, color=PHASE_BAND_COLORS[i % 2], zorder=0)
        if label_top:
            ax.text((s + e) / 2, ymax * 0.97, k, ha="center", va="top",
                    fontsize=7, color="#33415c")
    ax.axvline(60, color="#94a3b8", ls="--", lw=1)


def render_archetype_tab(archetype_key: str, severity: str,
                         correction_strength: float):
    pheno = PHENOTYPES[archetype_key]
    dc = pheno.distorted_curves()
    sc_map = {"mild": 0.4, "moderate": 0.7, "severe": 1.0}
    sc = sc_map.get(severity, 0.7)

    st.header(f"R-SC-GAM Archetype Model: {pheno.name}")
    st.write(f"**Classification:** {pheno.classification} | **GMFCS:** {pheno.gmfcs_typical}")

    # --- Joint kinematic curves ---
    st.subheader("Normal vs CP Archetype Joint Curves")
    joints   = ["hip", "knee", "ankle"]
    titles   = ["Hip flexion(+)/extension(-) (deg)",
                "Knee flexion(+) (deg)",
                "Ankle dorsiflexion(+)/plantarflexion(-) (deg)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, j, t in zip(axes, joints, titles):
        n = NORMAL_CURVES[j]
        cp = NORMAL_CURVES[j] + (dc[j] - NORMAL_CURVES[j]) * sc
        corr = n * correction_strength + cp * (1 - correction_strength)
        ymin = min(n.min(), cp.min()) - 6
        ymax = max(n.max(), cp.max()) + 8
        _add_phase_bands(ax, ymin, ymax, label_top=(j == "hip"))
        ax.plot(GRID, n,    color="#1d4ed8", lw=2.2, label="Normal template")
        ax.plot(GRID, cp,   color="#dc2626", lw=2.2, label=f"{pheno.name} ({severity})")
        ax.plot(GRID, corr, color="#16a34a", lw=1.8, ls="--",
                label=f"Hypothetical correction ({int(correction_strength*100)}%)")
        ax.fill_between(GRID, n, cp, color="#dc2626", alpha=0.10)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("% Gait Cycle")
        ax.set_ylabel("degrees")
        ax.set_xlim(0, 100)
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"R-SC-GAM Kinematics — {pheno.name}  |  "
        "Green dashed = hypothetical correction target visualization",
        fontsize=10, weight="bold",
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Green dashed line = hypothetical correction target visualization. "
        "This does NOT claim real therapy outcome."
    )

    # --- Deviation table ---
    st.subheader("Kinematic Deviation from Normal (degrees at phase centres)")
    from core.r_sc_gam_core import phase_values
    rows = []
    for j in ("hip", "knee", "ankle"):
        cp_curve = NORMAL_CURVES[j] + (dc[j] - NORMAL_CURVES[j]) * sc
        dev = phase_values(cp_curve) - phase_values(NORMAL_CURVES[j])
        row = {"joint": j}
        for ph, v in zip(PHASE_KEYS, dev):
            row[ph] = round(float(v), 1)
        rows.append(row)
    dev_df = pd.DataFrame(rows).set_index("joint")
    st.dataframe(dev_df.style.background_gradient(cmap="RdYlGn_r", axis=None), use_container_width=True)

    # --- Spasticity findings ---
    st.subheader("Spasticity Rule Findings")
    spas = spasticity_scores(pheno)
    weak = weakness_scores(pheno)

    spas_rows = []
    for short, v in spas.items():
        spas_rows.append({
            "muscle": short,
            "score": v["score"],
            "flagged": "YES" if v["flagged"] else "no",
            "evidence_phases": ", ".join(v["evidence_phases"]) or "—",
        })
    st.dataframe(pd.DataFrame(spas_rows), use_container_width=True)

    if weak:
        st.subheader("Weakness Findings")
        weak_rows = [{"muscle": k, "severity_score": round(v, 3)} for k, v in weak.items()]
        st.dataframe(pd.DataFrame(weak_rows), use_container_width=True)


def render_recruitment_tab(archetype_key: str):
    pheno = PHENOTYPES[archetype_key]
    df_rec, spas, weak = recruitment_matrix(pheno)

    st.header("Muscle Recruitment Likelihood Matrix")
    st.info(
        "Values are **simulation-estimated recruitment likelihoods (0–1)**, "
        "derived from archetype-specific spasticity and weakness rules. "
        "This is NOT measured EMG."
    )

    cmap = LinearSegmentedColormap.from_list(
        "rec", ["#f8fafc", "#fde68a", "#f97316", "#b91c1c"])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.imshow(df_rec.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(N_PH))
    ax.set_xticklabels(PHASE_KEYS)
    ax.set_yticks(range(len(df_rec)))
    ax.set_yticklabels(df_rec.index, fontsize=8)
    for i in range(len(df_rec)):
        for j in range(N_PH):
            v = df_rec.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if v > 0.55 else "#1f2937")
    ax.set_title(
        f"Phase-wise Muscle Recruitment Likelihood — {pheno.name}",
        fontsize=11, weight="bold")
    ax.set_xlabel("Gait phase")
    fig.colorbar(im, ax=ax, label="Recruitment likelihood (0–1)", shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Recruitment Matrix (DataFrame)")
    st.dataframe(df_rec.style.background_gradient(cmap="YlOrRd", axis=None,
                                                   vmin=0, vmax=1),
                 use_container_width=True)


def render_rehab_priority_tab(archetype_key: str):
    pheno = PHENOTYPES[archetype_key]
    rpi_df = rehab_priority(pheno)

    st.header("Rehabilitation Priority Index")
    st.write(
        "RPI combines spasticity score, weakness score, and kinematic deviation "
        "into a 0–100 ranking. Higher = higher priority for clinical attention."
    )

    cat_color = {
        "Spasticity management": "#dc2626",
        "Strengthening / motor control": "#2563eb",
        "Mixed (tone + weakness)": "#7c3aed",
        "Biomechanical / orthotic": "#059669",
        "Monitor": "#94a3b8",
    }
    colors = [cat_color.get(c, "#94a3b8") for c in rpi_df["intervention"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(rpi_df["muscle"][::-1], rpi_df["RPI"][::-1],
            color=colors[::-1], edgecolor="#1f2937")
    for y, (v, c) in enumerate(zip(rpi_df["RPI"][::-1], rpi_df["intervention"][::-1])):
        ax.text(v + 0.5, y, f"{v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("Rehabilitation Priority Index (0–100)")
    ax.set_title(f"Rehab Priority Ranking — {pheno.name}", fontsize=11, weight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in cat_color.values()]
    ax.legend(handles, cat_color.keys(), fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Priority Table")
    display_cols = ["rank", "muscle", "full_name", "spasticity", "weakness",
                    "deviation", "RPI", "intervention"]
    st.dataframe(rpi_df[display_cols], use_container_width=True)

    st.subheader("Intervention Rationale")
    for _, r in rpi_df.head(4).iterrows():
        with st.expander(f"#{r['rank']} {r['full_name']} — {r['intervention']}"):
            st.write(r["rationale"])
