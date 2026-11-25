#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_batch.py — headless batch runner for R-SC-GAM-MJ-SSL.

Runs the full pipeline (MuJoCo simulation with proxy fallback -> phase mechanics
-> stress-strain proxies -> rehab risk -> reports) for every archetype and
severity level, and saves figures, CSVs and reports under outputs/.

Run:  python run_batch.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core.r_sc_gam_core import PHENOTYPES
from core.stress_strain_engine import compute_stress_strain
from core.rehab_risk_engine import compute_rehab_risk
from core.report_engine import save_all_reports
from mujoco_engine.mj_runner import run_mujoco_archetype_simulation
from mujoco_engine.mj_data_extractor import extract_phase_mechanics
from mujoco_engine.mj_visualizer import (
    plot_joint_trajectories, plot_torque_heatmap,
    plot_loading_index_heatmap, plot_tendon_lengths,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DIRS = {k: os.path.join(OUT, k) for k in ("figures", "csv", "reports", "mujoco")}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

SEVERITIES = ["mild", "moderate", "severe"]
SIDE = "bilateral"


def save_fig(fig, name):
    fig.savefig(os.path.join(DIRS["figures"], name), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = []
    for key, pheno in PHENOTYPES.items():
        for sev in SEVERITIES:
            tag = f"{key}_{sev}_{SIDE}"
            sim = run_mujoco_archetype_simulation(key, sev, SIDE)
            phase_df = extract_phase_mechanics(sim)
            ss_df = compute_stress_strain(key, sev, SIDE, sim_result=sim)
            risk = compute_rehab_risk(key, sev, SIDE, loading_df=phase_df,
                                      stress_strain_df=ss_df)

            phase_df.to_csv(os.path.join(DIRS["mujoco"], f"{tag}_phase_mechanics.csv"), index=False)
            ss_df.to_csv(os.path.join(DIRS["csv"], f"{tag}_stress_strain.csv"), index=False)
            risk["scores_df"].to_csv(os.path.join(DIRS["csv"], f"{tag}_rehab_risk.csv"), index=False)

            save_fig(plot_joint_trajectories(sim), f"{tag}_joint_trajectories.png")
            save_fig(plot_torque_heatmap(phase_df), f"{tag}_torque_heatmap.png")
            save_fig(plot_loading_index_heatmap(phase_df), f"{tag}_loading_index.png")
            save_fig(plot_tendon_lengths(sim), f"{tag}_tendon_lengths.png")

            save_all_reports(key, sev, SIDE, DIRS["reports"], phase_df, ss_df,
                             risk, sim.get("message", ""))

            top = risk["scores_df"].iloc[0]
            summary.append({
                "archetype": pheno.name, "severity": sev,
                "engine": "MuJoCo" if sim.get("used_mujoco") else "proxy",
                "top_muscle": risk.get("top_muscle"),
                "top_joint": risk.get("top_joint"),
                "top_phase": risk.get("top_phase"),
                "top_risk_score": float(top["composite_risk"]),
                "primary_target": risk.get("primary_target"),
            })
            print(f"[{pheno.name:<18}] {sev:<8} engine={summary[-1]['engine']:<6} "
                  f"top muscle={summary[-1]['top_muscle']}")

    pd.DataFrame(summary).to_csv(os.path.join(OUT, "_BATCH_SUMMARY.csv"), index=False)
    print(f"\nDone. Results saved under {OUT}")


if __name__ == "__main__":
    main()
