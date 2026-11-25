"""
stress_strain_engine.py
Estimate approximate muscle-tendon strain and actuator stress proxies.

SCIENTIFIC BOUNDARY:
True biological stress/strain requires patient-specific geometry, PCSA, material
properties, and tissue data. This engine produces approximate normalised estimates
for research visualisation and hypothesis generation ONLY. Values must NOT be used
for clinical decisions or interpreted as confirmed tissue mechanics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

from core.r_sc_gam_core import (
    PHASES, PHASE_KEYS, PHASE_FUNCTIONAL_WEIGHT,
    NORMAL_CURVES, GRID, PHENOTYPES, MUSCLES, MUSCLE_BY_SHORT,
)


# ---------------------------------------------------------------------------
# Muscle parameter table (literature adult/paediatric approximations)
# Values are transparent defaults. The UI may expose sliders to edit them.
# Sources: Arnold et al. (2010) lower limb muscle geometry; Schwartz et al.
# ---------------------------------------------------------------------------
@dataclass
class MuscleParams:
    short: str
    name: str
    resting_length_m: float    # pennation-corrected fibre length proxy
    pcsa_cm2: float
    moment_arm_m: float
    primary_joint: str
    action: str
    normal_active_phases: List[str]
    note: str = ""


MUSCLE_PARAMS: Dict[str, MuscleParams] = {
    "GAS": MuscleParams("GAS", "Gastrocnemius",
                        resting_length_m=0.055, pcsa_cm2=30.0, moment_arm_m=0.050,
                        primary_joint="ankle", action="plantarflexor",
                        normal_active_phases=["MSt", "TSt"],
                        note="Biarticular; knee crossing included in length proxy."),
    "SOL": MuscleParams("SOL", "Soleus",
                        resting_length_m=0.040, pcsa_cm2=52.0, moment_arm_m=0.048,
                        primary_joint="ankle", action="plantarflexor",
                        normal_active_phases=["MSt", "TSt", "PSw"]),
    "TA":  MuscleParams("TA",  "Tibialis anterior",
                        resting_length_m=0.065, pcsa_cm2=9.8, moment_arm_m=0.037,
                        primary_joint="ankle", action="dorsiflexor",
                        normal_active_phases=["IC", "LR", "ISw", "MSw", "TSw"]),
    "RF":  MuscleParams("RF",  "Rectus femoris",
                        resting_length_m=0.084, pcsa_cm2=15.0, moment_arm_m=0.045,
                        primary_joint="knee", action="extensor/hip-flexor",
                        normal_active_phases=["PSw", "ISw"],
                        note="Biarticular; hip crossing modelled via offset."),
    "HAM": MuscleParams("HAM", "Hamstrings",
                        resting_length_m=0.095, pcsa_cm2=25.0, moment_arm_m=0.048,
                        primary_joint="knee", action="flexor/hip-extensor",
                        normal_active_phases=["TSw", "IC", "LR"]),
    "ILIO": MuscleParams("ILIO", "Iliopsoas",
                         resting_length_m=0.120, pcsa_cm2=12.0, moment_arm_m=0.060,
                         primary_joint="hip", action="flexor",
                         normal_active_phases=["PSw", "ISw"]),
    "ADD": MuscleParams("ADD", "Hip adductors",
                        resting_length_m=0.130, pcsa_cm2=18.0, moment_arm_m=0.055,
                        primary_joint="hip", action="adductor",
                        normal_active_phases=["PSw", "ISw"],
                        note="Frontal-plane dominant; sagittal model is an approximation only."),
    "GMED": MuscleParams("GMED", "Gluteus medius",
                         resting_length_m=0.070, pcsa_cm2=22.0, moment_arm_m=0.060,
                         primary_joint="hip", action="abductor",
                         normal_active_phases=["IC", "LR", "MSt"],
                         note="Frontal-plane dominant; sagittal model approximation."),
    "GMAX": MuscleParams("GMAX", "Gluteus maximus",
                         resting_length_m=0.145, pcsa_cm2=32.0, moment_arm_m=0.062,
                         primary_joint="hip", action="extensor",
                         normal_active_phases=["TSw", "IC", "LR"]),
    "VAS": MuscleParams("VAS", "Vasti (quadriceps)",
                        resting_length_m=0.070, pcsa_cm2=55.0, moment_arm_m=0.042,
                        primary_joint="knee", action="extensor",
                        normal_active_phases=["TSw", "IC", "LR"]),
}

# Max stress proxy for normalisation (N/cm²)
MAX_STRESS_PROXY = 35.0


def _joint_angle_at_phase(curves: Dict[str, np.ndarray], joint: str, phase: str) -> float:
    """Return mean angle (deg) for the given phase from curves dict."""
    ph_map = {p[0]: (p[2], p[3]) for p in PHASES}
    if phase not in ph_map or joint not in curves:
        return 0.0
    start, end = ph_map[phase]
    mask = (GRID >= start) & (GRID <= end)
    return float(np.mean(curves[joint][mask]))


def _estimate_muscle_length(
    mp: MuscleParams,
    curves: Dict[str, np.ndarray],
    phase: str,
) -> float:
    """
    Approximate current muscle-tendon length via moment-arm rule:
        L ≈ L_rest + MA * Δθ_primary
    For biarticulars, include secondary joint contribution.
    """
    j = mp.primary_joint
    theta_deg = _joint_angle_at_phase(curves, j, phase)
    theta_norm = _joint_angle_at_phase(NORMAL_CURVES, j, phase)
    delta_rad = np.deg2rad(theta_deg - theta_norm)
    L = mp.resting_length_m + mp.moment_arm_m * delta_rad

    # Biarticular corrections (rough)
    if mp.short in ("GAS", "RF"):
        sec_j = "knee" if mp.short == "GAS" else "hip"
        theta2 = _joint_angle_at_phase(curves, sec_j, phase)
        theta2_n = _joint_angle_at_phase(NORMAL_CURVES, sec_j, phase)
        delta2 = np.deg2rad(theta2 - theta2_n)
        L += mp.moment_arm_m * 0.6 * delta2   # secondary coupling ~60%
    if mp.short == "HAM":
        theta2 = _joint_angle_at_phase(curves, "hip", phase)
        theta2_n = _joint_angle_at_phase(NORMAL_CURVES, "hip", phase)
        delta2 = np.deg2rad(theta2 - theta2_n)
        L += mp.moment_arm_m * 0.55 * (-delta2)

    return max(float(L), mp.resting_length_m * 0.5)


def compute_stress_strain(
    phenotype_key: str,
    severity: str = "moderate",
    side_involvement: str = "bilateral",
    sim_result: Dict[str, Any] = None,
    custom_params: Dict[str, Dict] = None,
) -> pd.DataFrame:
    """
    Compute per-muscle / per-phase estimated strain and stress proxies.

    sim_result: if provided, actuator forces are read from it; else proxied from torque.
    custom_params: optional override dict keyed by muscle short with
                   {resting_length_m, pcsa_cm2, moment_arm_m} overrides.
    """
    from mujoco_engine.mj_runner import SEVERITY_SCALE
    pheno = PHENOTYPES[phenotype_key]
    dc    = pheno.distorted_curves()
    sc    = SEVERITY_SCALE.get(severity, 0.7)

    # Apply severity scaling
    curves: Dict[str, np.ndarray] = {}
    for jk in dc:
        base = NORMAL_CURVES[jk]
        curves[jk] = base + (dc[jk] - base) * sc

    sides = ["left", "right"] if side_involvement == "bilateral" else [side_involvement]
    all_sides = ["left", "right"]

    # Recruitment likelihoods for this phenotype
    from core.r_sc_gam_core import recruitment_matrix
    rec_df, spas, weak = recruitment_matrix(pheno)

    params = dict(MUSCLE_PARAMS)
    if custom_params:
        for short, overrides in custom_params.items():
            if short in params:
                for attr, val in overrides.items():
                    setattr(params[short], attr, val)

    rows = []
    for side in all_sides:
        side_curves = curves if side in sides else NORMAL_CURVES

        for short, mp in params.items():
            rec_row_label = next(
                (lbl for lbl in rec_df.index if lbl.startswith(short + " ")), None)

            for pi, (ph_short, ph_full, ph_start, ph_end, ph_c) in enumerate(PHASES):
                # recruitment likelihood for this muscle × phase
                rec_val = 0.0
                if rec_row_label and ph_short in rec_df.columns:
                    rec_val = float(rec_df.loc[rec_row_label, ph_short])

                est_L = _estimate_muscle_length(mp, side_curves, ph_short)
                L0    = mp.resting_length_m
                strain = (est_L - L0) / L0

                # Actuator force proxy: torque / moment_arm, weighted by recruitment
                if sim_result and side in sim_result.get("joints", {}) \
                        and mp.primary_joint in sim_result["joints"][side]:
                    joint_sig = sim_result["joints"][side][mp.primary_joint]
                    t = sim_result["t"]
                    cycle_dur = 1.1
                    pct_t = (t / cycle_dur * 100.0) % 100.0
                    mask = (pct_t >= ph_start) & (pct_t < ph_end)
                    if mask.sum() == 0:
                        mask = np.abs(pct_t - ph_c) < 5.0
                    tau_vals = joint_sig.get("torque_Nm", np.zeros_like(t))
                    tau_avg = float(np.mean(np.abs(tau_vals[mask]))) if mask.sum() > 0 else 0.0
                    act_force = tau_avg / max(mp.moment_arm_m, 0.01) * rec_val
                else:
                    # Proxy: phase-weighted kinematic-torque estimate
                    theta_deg = _joint_angle_at_phase(side_curves, mp.primary_joint, ph_short)
                    theta_norm = _joint_angle_at_phase(NORMAL_CURVES, mp.primary_joint, ph_short)
                    delta_rad = abs(np.deg2rad(theta_deg - theta_norm))
                    k_eff = 8.0  # N·m/rad pseudo-stiffness
                    tau_approx = k_eff * delta_rad * PHASE_FUNCTIONAL_WEIGHT[pi]
                    act_force = tau_approx / max(mp.moment_arm_m, 0.01) * rec_val

                stress_proxy = act_force / max(mp.pcsa_cm2, 0.1)
                norm_stress  = float(np.clip(stress_proxy / MAX_STRESS_PROXY, 0.0, 1.0))

                # Burden = magnitude × functional weight of the phase
                fw         = PHASE_FUNCTIONAL_WEIGHT[pi]
                strain_bur = float(abs(strain) * fw)
                stress_bur = float(norm_stress * fw)

                interp = _interpret_ss(short, ph_short, strain, norm_stress,
                                       strain_bur, stress_bur, side, side_involvement)

                rows.append({
                    "side": side,
                    "muscle": short,
                    "full_name": mp.name,
                    "phase": ph_short,
                    "estimated_length_m": round(est_L, 4),
                    "resting_length_m": round(L0, 4),
                    "estimated_strain": round(strain, 4),
                    "actuator_force_proxy": round(act_force, 2),
                    "pcsa_cm2": mp.pcsa_cm2,
                    "estimated_stress_proxy": round(stress_proxy, 3),
                    "norm_stress": round(norm_stress, 3),
                    "strain_burden": round(strain_bur, 4),
                    "stress_burden": round(stress_bur, 4),
                    "recruitment_likelihood": round(rec_val, 3),
                    "interpretation": interp,
                })

    return pd.DataFrame(rows)


def _interpret_ss(
    short: str, phase: str,
    strain: float, norm_stress: float,
    strain_bur: float, stress_bur: float,
    side: str, side_inv: str,
) -> str:
    high_strain  = abs(strain) > 0.06
    high_stress  = norm_stress > 0.50
    frontal_only = short in ("ADD", "GMED")

    if frontal_only:
        return (
            f"Adductor/scissoring or gluteus medius burden is rule-based here "
            f"because the current model is sagittal-dominant. "
            f"Frontal-plane kinematics required for reliable estimate."
        )

    if not (high_strain or high_stress):
        return "Low estimated burden — no specific mechanical concern flagged."

    parts = []
    if short in ("GAS", "SOL") and high_strain:
        parts.append("high gastrocnemius-soleus strain supports equinus mechanical load hypothesis")
    if short == "RF" and phase in ("ISw", "MSw") and high_strain:
        parts.append("high rectus femoris burden during swing supports stiff-knee recruitment hypothesis")
    if short == "HAM" and phase in ("MSt", "TSt") and high_strain:
        parts.append("high hamstring burden during stance supports crouch/flexed-knee loading pattern")
    if short == "ILIO" and high_stress:
        parts.append("high iliopsoas stress burden noted — hip flexor contribution elevated")
    if short == "VAS" and phase in ("MSt", "TSt") and high_stress:
        parts.append("high quadriceps burden during stance — consistent with crouch pattern")
    if short == "TA" and phase in ("ISw", "MSw") and high_strain:
        parts.append("high tibialis anterior strain in swing — may support drop-foot or clearance deficit")

    if not parts:
        lbl = "high" if (high_strain and high_stress) else ("high strain" if high_strain else "high stress")
        parts.append(f"{lbl} mechanical burden detected — contextualise with archetype and clinical findings")

    if side_inv in ("left", "right"):
        parts.append(f"unilateral {side} side — consistent with hemiplegic asymmetry hypothesis")
    elif side_inv == "bilateral" and (high_strain or high_stress):
        parts.append("bilateral pattern — consistent with diplegic spasticity distribution")

    return "; ".join(parts).capitalize() + "."
