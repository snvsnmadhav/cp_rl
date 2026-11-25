"""
mj_data_extractor.py
Extract and summarise per-phase mechanical signals from simulation output.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any

from core.r_sc_gam_core import PHASES, PHASE_KEYS, PHASE_FUNCTIONAL_WEIGHT


def extract_phase_mechanics(sim_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Reduce time-series simulation data to per-phase summary rows.

    Columns: side, joint, phase, angle_deg, velocity_deg_per_gc,
             torque_proxy, contact_force_proxy, joint_loading_index, interpretation
    """
    t = sim_result["t"]
    duration = float(t[-1]) if len(t) > 1 else 1.0
    cycle_dur = 1.1

    rows = []
    for side, jdict in sim_result["joints"].items():
        for jkey, signals in jdict.items():
            pos_deg = signals.get("pos_deg", np.zeros_like(t))
            vel = signals.get("vel_rad_s", np.zeros_like(t))
            torque = signals.get("torque_Nm", np.zeros_like(t))
            contact = signals.get("contact_N", np.zeros_like(t))

            pct = (t / cycle_dur * 100.0) % 100.0

            for ph_idx, (ph_short, ph_full, ph_start, ph_end, ph_centre) in enumerate(PHASES):
                mask = (pct >= ph_start) & (pct < ph_end)
                if mask.sum() == 0:
                    mask = np.abs(pct - ph_centre) < 6.0
                if mask.sum() == 0:
                    continue

                angle   = float(np.mean(pos_deg[mask]))
                vel_avg = float(np.mean(np.degrees(np.abs(vel[mask]))))
                tau_avg = float(np.mean(np.abs(torque[mask])))
                cf_avg  = float(np.mean(contact[mask]))

                # Joint loading index: weighted combination, normalised heuristic
                jli = float(np.clip(
                    (0.4 * tau_avg / 30.0 + 0.3 * cf_avg / 300.0 + 0.3 * vel_avg / 200.0)
                    * PHASE_FUNCTIONAL_WEIGHT[ph_idx],
                    0.0, 1.0
                ))

                interp = _interpret_loading(jkey, ph_short, jli, side)

                rows.append({
                    "side": side,
                    "joint": jkey,
                    "phase": ph_short,
                    "angle_deg": round(angle, 2),
                    "velocity_deg_per_gc": round(vel_avg, 2),
                    "torque_proxy": round(tau_avg, 3),
                    "contact_force_proxy": round(cf_avg, 1),
                    "joint_loading_index": round(jli, 3),
                    "interpretation": interp,
                })

    return pd.DataFrame(rows)


def _interpret_loading(joint: str, phase: str, jli: float, side: str) -> str:
    if jli < 0.25:
        return "Low loading — within expected range."
    stance = phase in ("IC", "LR", "MSt", "TSt", "PSw")
    swing  = phase in ("ISw", "MSw", "TSw")
    if joint == "ankle" and stance and jli >= 0.5:
        return "High ankle stance loading — may indicate plantarflexor/equinus mechanical burden."
    if joint == "ankle" and stance and jli >= 0.35:
        return "Moderate ankle stance loading — monitor plantarflexor contribution."
    if joint == "knee" and swing and jli >= 0.45:
        return "High knee swing loading — may indicate stiff-knee / rectus femoris compensation."
    if joint == "knee" and stance and jli >= 0.5:
        return "High knee stance loading — consistent with crouch or flexed-knee pattern."
    if joint == "hip" and jli >= 0.5:
        return "High hip loading — may indicate proximal compensation or hip flexor contribution."
    if side in ("left", "right"):
        return f"Elevated {joint} {phase} loading — {side} side asymmetry may be present."
    return "Moderate loading detected — contextualise with clinical examination."
