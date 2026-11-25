"""
report_engine.py
Extended report generation — wraps R-SC-GAM reports with MuJoCo/proxy summary.
"""

from __future__ import annotations
import os
import textwrap
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from core.r_sc_gam_core import (
    PHENOTYPES, therapist_report as rscgam_therapist_report,
    patient_report as rscgam_patient_report,
    child_report as rscgam_child_report,
)


_DISCLAIMER = (
    "\n" + "=" * 78 + "\n"
    "COMPUTATIONAL FRAMEWORK DISCLAIMER\n"
    "R-SC-GAM-MJ-SSL produces: muscle recruitment likelihood, estimated actuator\n"
    "stress, estimated muscle-tendon strain, joint torque burden, mechanical loading\n"
    "index, and simulation-based hypothesis generation.\n"
    "It does NOT measure EMG, does NOT confirm tissue stress, does NOT diagnose CP,\n"
    "and does NOT prescribe treatment. Every output must be interpreted by a qualified\n"
    "clinician alongside instrumented 3D gait analysis, dynamic EMG, and physical\n"
    "examination.\n"
    "=" * 78
)


def extended_therapist_report(
    phenotype_key: str,
    severity: str = "moderate",
    side_involvement: str = "bilateral",
    loading_df: Optional[pd.DataFrame] = None,
    stress_strain_df: Optional[pd.DataFrame] = None,
    risk_result: Optional[Dict[str, Any]] = None,
    sim_message: str = "",
) -> str:
    pheno = PHENOTYPES[phenotype_key]
    L = []
    L.append("=" * 78)
    L.append("R-SC-GAM-MJ-SSL  EXTENDED THERAPIST REPORT")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append(f"Archetype: {pheno.name}  |  Severity: {severity}  |  Side: {side_involvement}")
    L.append("=" * 78)

    # Core R-SC-GAM section
    L.append("")
    L.append("[ SECTION 1: R-SC-GAM ARCHETYPE MODEL ]")
    L.append(rscgam_therapist_report(pheno))

    # MuJoCo / proxy summary
    L.append("")
    L.append("[ SECTION 2: MECHANICAL SIMULATION SUMMARY ]")
    L.append(f"Simulation mode: {sim_message if sim_message else 'Proxy-based deterministic approximation'}")
    if loading_df is not None and not loading_df.empty:
        L.append("")
        L.append("Joint loading index summary (mean across phases, by joint):")
        summ = loading_df.groupby(["side", "joint"])["joint_loading_index"].mean().reset_index()
        for _, row in summ.iterrows():
            L.append(f"  {row['side']:>5} {row['joint']:<6}: {row['joint_loading_index']:.3f}")

        top = loading_df.nlargest(3, "joint_loading_index")[
            ["side", "joint", "phase", "joint_loading_index", "interpretation"]]
        L.append("")
        L.append("Top 3 mechanically loaded joint/phase combinations:")
        for _, r in top.iterrows():
            L.append(f"  {r['side']} {r['joint']} {r['phase']}: JLI={r['joint_loading_index']:.3f}")
            L.append(textwrap.fill(f"  -> {r['interpretation']}", 74,
                                   subsequent_indent="     "))

    # Stress-strain summary
    L.append("")
    L.append("[ SECTION 3: ESTIMATED STRESS-STRAIN SUMMARY ]")
    L.append("IMPORTANT: Values are approximate normalised estimates for research")
    L.append("visualisation. They are NOT confirmed tissue mechanics.")
    if stress_strain_df is not None and not stress_strain_df.empty:
        summ_ss = stress_strain_df.groupby(["muscle", "full_name"])[
            ["strain_burden", "stress_burden"]].mean().reset_index()
        summ_ss = summ_ss.sort_values("stress_burden", ascending=False).head(5)
        L.append("")
        L.append("Top 5 muscles by estimated stress burden:")
        for _, r in summ_ss.iterrows():
            L.append(f"  {r['muscle']:<5} {r['full_name']:<25} "
                     f"stress_burden={r['stress_burden']:.3f}  "
                     f"strain_burden={r['strain_burden']:.3f}")

    # Rehab risk
    L.append("")
    L.append("[ SECTION 4: REHABILITATION MECHANICAL RISK ]")
    if risk_result:
        L.append(f"Top overloaded joint : {risk_result.get('top_joint','—').upper()}")
        L.append(f"Top overloaded muscle: {risk_result.get('top_muscle','—')}")
        L.append(f"Top phase of concern : {risk_result.get('top_phase','—')}")
        L.append("")
        L.append("Primary rehab target:")
        L.append(textwrap.fill(risk_result.get("primary_target", ""), 74,
                               initial_indent="  ", subsequent_indent="  "))
        L.append("Secondary rehab target:")
        L.append(textwrap.fill(risk_result.get("secondary_target", ""), 74,
                               initial_indent="  ", subsequent_indent="  "))
        L.append("Compensation monitoring:")
        L.append(textwrap.fill(risk_result.get("compensation_target", ""), 74,
                               initial_indent="  ", subsequent_indent="  "))
        L.append("Ongoing monitoring:")
        L.append(textwrap.fill(risk_result.get("monitoring_target", ""), 74,
                               initial_indent="  ", subsequent_indent="  "))

    # Limitations
    L.append("")
    L.append("[ SECTION 5: LIMITATIONS AND VALIDATION PATHWAY ]")
    L.append(textwrap.fill(
        "This report combines literature-encoded archetype reasoning (R-SC-GAM) "
        "with deterministic mechanical proxy calculations or MuJoCo physics "
        "simulation (R-SC-GAM-MJ-SSL). The mechanical outputs are SIMULATION-BASED "
        "HYPOTHESES, not measured patient data.", 74))
    L.append("")
    L.append("Validation pathway:")
    L.append("  1. Confirm archetype with instrumented 3D gait analysis (force plates + cameras).")
    L.append("  2. Validate muscle recruitment with dynamic surface/fine-wire EMG.")
    L.append("  3. Assess spasticity with MAS/Tardieu scale + passive ROM examination.")
    L.append("  4. Compare simulation torque estimates with inverse dynamics from 3D data.")
    L.append("  5. Obtain imaging (ultrasound/MRI) for muscle architecture if tissue detail needed.")
    L.append("  6. All intervention decisions must be made by qualified clinicians.")

    L.append(_DISCLAIMER)
    return "\n".join(L)


def patient_parent_report(
    phenotype_key: str,
    severity: str = "moderate",
    side_involvement: str = "bilateral",
    risk_result: Optional[Dict[str, Any]] = None,
) -> str:
    pheno = PHENOTYPES[phenotype_key]
    L = []
    L.append("PATIENT / PARENT WALKING PATTERN REPORT")
    L.append("=" * 64)
    L.append(f"Pattern simulated: {pheno.name}")
    L.append(f"Severity modelled: {severity}")
    L.append(f"Side: {side_involvement}")
    L.append("")
    L.append(rscgam_patient_report(pheno))
    L.append("")
    L.append("-" * 64)
    L.append("What the computer simulation added:")
    if risk_result:
        L.append(textwrap.fill(
            f"The simulation estimated that the {risk_result.get('top_joint','ankle')} joint "
            f"and {risk_result.get('top_muscle','—')} muscle group may be carrying more "
            f"mechanical load than normal during the {risk_result.get('top_phase','stance')} "
            f"phase of walking.", 64))
        L.append("")
        L.append("Movement goal this simulation highlights:")
        L.append(textwrap.fill(risk_result.get("primary_target", ""), 64,
                               initial_indent="  ", subsequent_indent="  "))
    L.append("")
    L.append(textwrap.fill(
        "IMPORTANT: This report is produced by a computer model. It does NOT "
        "diagnose your child's condition or prescribe treatment. Your therapist "
        "and doctor will explain what these findings mean for your child's "
        "specific situation and plan therapy accordingly.", 64))
    return "\n".join(L)


def child_friendly_report(
    phenotype_key: str,
    risk_result: Optional[Dict[str, Any]] = None,
) -> str:
    pheno = PHENOTYPES[phenotype_key]
    return rscgam_child_report(pheno)


def save_all_reports(
    phenotype_key: str,
    severity: str,
    side_involvement: str,
    out_dir: str,
    loading_df: Optional[pd.DataFrame] = None,
    stress_strain_df: Optional[pd.DataFrame] = None,
    risk_result: Optional[Dict[str, Any]] = None,
    sim_message: str = "",
):
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{phenotype_key}_{severity}_{side_involvement}")

    with open(prefix + "_THERAPIST.txt", "w", encoding="utf-8") as f:
        f.write(extended_therapist_report(
            phenotype_key, severity, side_involvement,
            loading_df, stress_strain_df, risk_result, sim_message))

    pheno = PHENOTYPES[phenotype_key]
    with open(prefix + "_PATIENT.txt", "w", encoding="utf-8") as f:
        f.write(patient_parent_report(phenotype_key, severity, side_involvement, risk_result))

    with open(prefix + "_CHILD.txt", "w", encoding="utf-8") as f:
        f.write(child_friendly_report(phenotype_key, risk_result))
