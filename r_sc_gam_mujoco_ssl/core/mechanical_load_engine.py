"""
mechanical_load_engine.py
Compute per-joint, per-phase mechanical loading burden from simulation data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any

from core.r_sc_gam_core import (
    PHASES, PHASE_KEYS, PHASE_FUNCTIONAL_WEIGHT,
    NORMAL_CURVES, GRID,
)


INERTIA_PROXY   = {"hip": 0.22, "knee": 0.065, "ankle": 0.012}
DAMPING_PROXY   = {"hip": 3.0,  "knee": 1.5,   "ankle": 0.6}
STIFFNESS_PROXY = {"hip": 15.0, "knee": 8.0,   "ankle": 4.0}
MOMENT_ARM      = {"hip": 0.06, "knee": 0.04,  "ankle": 0.05}
BW_PROXY        = 300.0    # N
STANCE_DUTY     = 0.60
CYCLE_DUR       = 1.1      # s


def compute_loading_from_simulation(sim_result: Dict[str, Any]) -> pd.DataFrame:
    """Build per-side / per-joint / per-phase loading DataFrame from sim output."""
    t = sim_result["t"]
    cycle_dur = CYCLE_DUR
    pct = (t / cycle_dur * 100.0) % 100.0
    rows = []

    for side, jdict in sim_result.get("joints", {}).items():
        for jkey, signals in jdict.items():
            pos_deg = signals.get("pos_deg",  np.zeros_like(t))
            vel     = signals.get("vel_rad_s", np.zeros_like(t))
            tau     = signals.get("torque_Nm", np.zeros_like(t))
            cf      = signals.get("contact_N", np.zeros_like(t))

            for pi, (ph_short, ph_full, ph_start, ph_end, ph_c) in enumerate(PHASES):
                mask = (pct >= ph_start) & (pct < ph_end)
                if mask.sum() == 0:
                    mask = np.abs(pct - ph_c) < 5.0
                if mask.sum() == 0:
                    continue

                angle_avg = float(np.mean(pos_deg[mask]))
                vel_avg   = float(np.mean(np.abs(np.degrees(vel[mask]))))
                acc_arr   = np.gradient(vel[mask], np.gradient(t[mask]).mean() if mask.sum() > 1 else 0.01)
                acc_avg   = float(np.mean(np.abs(np.degrees(acc_arr)))) if len(acc_arr) > 0 else 0.0
                tau_avg   = float(np.mean(np.abs(tau[mask])))
                cf_avg    = float(np.mean(cf[mask]))

                fw = PHASE_FUNCTIONAL_WEIGHT[pi]
                jli = float(np.clip(
                    (0.4 * tau_avg / 30.0 + 0.3 * cf_avg / BW_PROXY + 0.3 * vel_avg / 200.0) * fw,
                    0.0, 1.0
                ))

                rows.append({
                    "side": side,
                    "joint": jkey,
                    "phase": ph_short,
                    "angle_deg": round(angle_avg, 2),
                    "velocity_deg_per_gc": round(vel_avg, 2),
                    "acceleration_deg_per_gc2": round(acc_avg, 2),
                    "torque_proxy": round(tau_avg, 3),
                    "contact_force_proxy": round(cf_avg, 1),
                    "joint_loading_index": round(jli, 3),
                    "interpretation": _interpret(jkey, ph_short, jli, side),
                })

    return pd.DataFrame(rows)


def compute_loading_from_curves(
    phenotype_key: str,
    severity: str = "moderate",
    side_involvement: str = "bilateral",
) -> pd.DataFrame:
    """Compute loading directly from R-SC-GAM curves without running MuJoCo."""
    from core.r_sc_gam_core import PHENOTYPES
    from mujoco_engine.mj_runner import SEVERITY_SCALE

    pheno = PHENOTYPES[phenotype_key]
    dc    = pheno.distorted_curves()
    sc    = SEVERITY_SCALE.get(severity, 0.7)
    sides = ["left", "right"] if side_involvement == "bilateral" else [side_involvement]

    t = np.linspace(0, 5.5, 550)
    cycle_dur = CYCLE_DUR
    pct_t = (t / cycle_dur * 100.0) % 100.0
    rows = []

    for side in sides:
        for jkey in ["hip", "knee", "ankle"]:
            base  = NORMAL_CURVES[jkey]
            cp    = dc[jkey]
            curve = base + (cp - base) * sc

            q_deg = np.interp(pct_t, GRID.astype(float), curve)
            q_rad = np.deg2rad(q_deg)
            dq    = np.gradient(q_rad, t)
            ddq   = np.gradient(dq, t)
            dev   = q_rad - np.deg2rad(np.interp(pct_t, GRID.astype(float), base))

            I = INERTIA_PROXY[jkey]
            b = DAMPING_PROXY[jkey]
            k = STIFFNESS_PROXY[jkey]
            tau = I * ddq + b * dq + k * dev
            stance_mask_all = pct_t < (STANCE_DUTY * 100)
            cf_all = np.where(stance_mask_all,
                              BW_PROXY * (1.0 + 0.3 * np.sin(np.pi * pct_t / (STANCE_DUTY * 100))), 0.0)

            for pi, (ph_short, ph_full, ph_start, ph_end, ph_c) in enumerate(PHASES):
                mask = (pct_t >= ph_start) & (pct_t < ph_end)
                if mask.sum() == 0:
                    continue
                angle_avg = float(np.mean(q_deg[mask]))
                vel_avg   = float(np.mean(np.abs(np.degrees(dq[mask]))))
                acc_avg   = float(np.mean(np.abs(np.degrees(ddq[mask]))))
                tau_avg   = float(np.mean(np.abs(tau[mask])))
                cf_avg    = float(np.mean(cf_all[mask]))
                fw        = PHASE_FUNCTIONAL_WEIGHT[pi]
                jli       = float(np.clip(
                    (0.4 * tau_avg / 30.0 + 0.3 * cf_avg / BW_PROXY + 0.3 * vel_avg / 200.0) * fw,
                    0.0, 1.0
                ))
                rows.append({
                    "side": side,
                    "joint": jkey,
                    "phase": ph_short,
                    "angle_deg": round(angle_avg, 2),
                    "velocity_deg_per_gc": round(vel_avg, 2),
                    "acceleration_deg_per_gc2": round(acc_avg, 2),
                    "torque_proxy": round(tau_avg, 3),
                    "contact_force_proxy": round(cf_avg, 1),
                    "joint_loading_index": round(jli, 3),
                    "interpretation": _interpret(jkey, ph_short, jli, side),
                })

    # Fill other side with normal if unilateral
    filled_sides = set([r["side"] for r in rows])
    for s in ["left", "right"]:
        if s not in filled_sides:
            for jkey in ["hip", "knee", "ankle"]:
                base = NORMAL_CURVES[jkey]
                t2 = np.linspace(0, 5.5, 550)
                pct2 = (t2 / cycle_dur * 100.0) % 100.0
                q_deg = np.interp(pct2, GRID.astype(float), base)
                q_rad = np.deg2rad(q_deg)
                dq2 = np.gradient(q_rad, t2)
                ddq2 = np.gradient(dq2, t2)
                tau2 = INERTIA_PROXY[jkey] * ddq2 + DAMPING_PROXY[jkey] * dq2
                cf2 = np.where(pct2 < (STANCE_DUTY * 100), BW_PROXY * 0.9, 0.0)
                for pi, (ph_short, _, ph_start, ph_end, ph_c) in enumerate(PHASES):
                    mask = (pct2 >= ph_start) & (pct2 < ph_end)
                    if mask.sum() == 0:
                        continue
                    tau_avg = float(np.mean(np.abs(tau2[mask])))
                    cf_avg  = float(np.mean(cf2[mask]))
                    vel_avg = float(np.mean(np.abs(np.degrees(dq2[mask]))))
                    jli     = float(np.clip(
                        (0.4 * tau_avg / 30.0 + 0.3 * cf_avg / BW_PROXY + 0.3 * vel_avg / 200.0)
                        * PHASE_FUNCTIONAL_WEIGHT[pi], 0.0, 1.0))
                    rows.append({
                        "side": s, "joint": jkey, "phase": ph_short,
                        "angle_deg": round(float(np.mean(q_deg[mask])), 2),
                        "velocity_deg_per_gc": round(vel_avg, 2),
                        "acceleration_deg_per_gc2": round(float(np.mean(np.abs(np.degrees(ddq2[mask])))), 2),
                        "torque_proxy": round(tau_avg, 3),
                        "contact_force_proxy": round(cf_avg, 1),
                        "joint_loading_index": round(jli, 3),
                        "interpretation": "Normal side reference.",
                    })

    return pd.DataFrame(rows)


def _interpret(joint: str, phase: str, jli: float, side: str) -> str:
    stance = phase in ("IC", "LR", "MSt", "TSt", "PSw")
    swing  = phase in ("ISw", "MSw", "TSw")
    if jli < 0.20:
        return "Low loading — within expected normal range."
    if joint == "ankle" and stance and jli >= 0.55:
        return "High ankle stance loading — may indicate plantarflexor/equinus mechanical burden."
    if joint == "ankle" and stance and jli >= 0.35:
        return "Moderate ankle stance loading — monitor plantarflexor contribution."
    if joint == "knee" and swing and jli >= 0.45:
        return "High knee swing loading — consistent with stiff-knee / rectus femoris hypothesis."
    if joint == "knee" and stance and jli >= 0.50:
        return "High knee stance loading — consistent with crouch or flexed-knee loading pattern."
    if joint == "hip" and jli >= 0.50:
        return "High hip loading — may indicate proximal compensation or hip flexor overactivity."
    if jli >= 0.35:
        return f"Moderate {joint} {phase} loading — {side} side — assess with clinical examination."
    return f"Low-moderate {joint} {phase} loading — routine monitoring recommended."
