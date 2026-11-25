"""
rehab_risk_engine.py
Combine R-SC-GAM + mechanical load + stress-strain into a rehabilitation
mechanical risk score and ranked target list.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from core.r_sc_gam_core import (
    PHENOTYPES, PHASES, PHASE_FUNCTIONAL_WEIGHT,
    spasticity_scores, weakness_scores, rehab_priority,
)


# Composite weight vector (must sum to 1.0)
W = {
    "rehab_priority":  0.25,
    "joint_loading":   0.20,
    "stress_burden":   0.20,
    "strain_burden":   0.15,
    "spasticity":      0.10,
    "asymmetry":       0.10,
}

_ARCHETYPE_RISK_TEMPLATES = {
    "stiff_knee": {
        "primary_concern": "Knee swing-phase burden with rectus femoris dominance pattern.",
        "primary_target":  "Improve swing-phase knee flexion timing and reduce compensatory hip motion.",
        "secondary_target": "Address rectus femoris over-activity during pre/initial swing.",
        "compensation_target": "Monitor hip flexor and ankle plantarflexor compensatory activity.",
        "monitoring_target": "Foot clearance; hip hiking; circumduction strategies.",
    },
    "true_equinus": {
        "primary_concern": "Ankle plantarflexor / Achilles burden during stance and pre-swing.",
        "primary_target":  "Dorsiflexion control, plantarflexor tone monitoring, heel contact training.",
        "secondary_target": "Assess knee recurvatum secondary to plantar-flexed-knee-extended gait.",
        "compensation_target": "Hip and lumbar compensation for distal lever-arm deficit.",
        "monitoring_target": "Fixed vs dynamic equinus distinction; serial casting compliance.",
    },
    "jump": {
        "primary_concern": "Multi-level ankle / knee / hip burden with anterior pelvic tilt.",
        "primary_target":  "Multi-level spasticity management (BoNT-A / SEMLS); stance-phase knee extension.",
        "secondary_target": "Rectus femoris transfer consideration for stiff-knee component.",
        "compensation_target": "Hamstring and iliopsoas contributions at hip during stance.",
        "monitoring_target": "Energy expenditure; pelvic kinematics; trunk stability.",
    },
    "apparent_equinus": {
        "primary_concern": "Proximal (hip / knee) flexion burden producing apparent toe-walking.",
        "primary_target":  "Hamstring and iliopsoas management; avoid heel-cord lengthening.",
        "secondary_target": "Strengthen quadriceps and gluteus maximus for stance extension.",
        "compensation_target": "Lumbar hyperlordosis; anterior pelvic tilt management.",
        "monitoring_target": "Ankle dorsiflexion ROM — confirm no contracture develops.",
    },
    "crouch": {
        "primary_concern": "Knee extensor and hip extensor burden during stance; calcaneal loading.",
        "primary_target":  "Stance-phase extension support; plantarflexor contribution restoration.",
        "secondary_target": "Hamstring lengthening; GRAFO or KAFO consideration.",
        "compensation_target": "Avoid further heel-cord lengthening; monitor lever-arm deformity.",
        "monitoring_target": "Quadriceps fatigue; patellofemoral stress; energy cost.",
    },
    "drop_foot": {
        "primary_concern": "Affected-side ankle / swing clearance deficit; dorsiflexor weakness.",
        "primary_target":  "Dorsiflexor strengthening / FES; leaf-spring AFO for clearance.",
        "secondary_target": "Hip and knee flexion compensation — monitor hip hiking.",
        "compensation_target": "Contralateral side may show step-length asymmetry.",
        "monitoring_target": "Fixed equinus contracture development; long-term AFO fit.",
    },
}

_SIDE_TEMPLATES = {
    "bilateral": {
        "concern_suffix": "Bilateral pattern — consistent with diplegic spasticity distribution.",
        "target_suffix":  "Bilateral tone pattern control, step-width, pelvic stability.",
    },
    "left":  {
        "concern_suffix": "Unilateral left-side burden — consistent with hemiplegic asymmetry.",
        "target_suffix":  "Left-side stance stability and swing clearance.",
    },
    "right": {
        "concern_suffix": "Unilateral right-side burden — consistent with hemiplegic asymmetry.",
        "target_suffix":  "Right-side stance stability and swing clearance.",
    },
}


def compute_rehab_risk(
    phenotype_key: str,
    severity: str = "moderate",
    side_involvement: str = "bilateral",
    loading_df: Optional[pd.DataFrame] = None,
    stress_strain_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute composite rehabilitation mechanical risk scores.

    Returns:
        scores_df: per-muscle composite risk
        top_joint: highest-loaded joint
        top_muscle: highest-risk muscle
        top_phase: phase of greatest concern
        primary_target, secondary_target, etc.
        narrative: therapist-readable explanation
    """
    pheno = PHENOTYPES[phenotype_key]
    rpi_df = rehab_priority(pheno)
    spas   = spasticity_scores(pheno)
    weak   = weakness_scores(pheno)
    sc_map = {"mild": 0.4, "moderate": 0.7, "severe": 1.0}
    sev_sc = sc_map.get(severity, 0.7)

    # --- Per-muscle composite risk score ---
    rows = []
    for _, r in rpi_df.iterrows():
        short = r["muscle"]
        rpi_norm = r["RPI"] / 100.0

        # Joint loading contribution
        jli = 0.0
        if loading_df is not None and not loading_df.empty:
            muscle_obj = next((m for m in __import__(
                "core.r_sc_gam_core", fromlist=["MUSCLES"]).MUSCLES
                if m.short == short), None)
            if muscle_obj:
                primary_j = list(muscle_obj.actions.keys())[0]
                sub = loading_df[loading_df["joint"] == primary_j]
                if not sub.empty:
                    jli = float(sub["joint_loading_index"].mean())

        # Stress / strain contributions
        stress_bur = 0.0
        strain_bur = 0.0
        if stress_strain_df is not None and not stress_strain_df.empty:
            sub_ss = stress_strain_df[stress_strain_df["muscle"] == short]
            if not sub_ss.empty:
                stress_bur = float(sub_ss["stress_burden"].mean())
                strain_bur = float(sub_ss["strain_burden"].mean())

        spas_score = spas.get(short, {}).get("score", 0.0)

        # Asymmetry burden
        asym = 0.0
        if loading_df is not None and not loading_df.empty and side_involvement != "bilateral":
            aff_side = side_involvement
            norm_side = "right" if aff_side == "left" else "left"
            sub_aff  = loading_df[(loading_df["side"] == aff_side)]
            sub_norm = loading_df[(loading_df["side"] == norm_side)]
            if not sub_aff.empty and not sub_norm.empty:
                asym = float(abs(sub_aff["joint_loading_index"].mean()
                                 - sub_norm["joint_loading_index"].mean()))

        composite = (
            W["rehab_priority"] * rpi_norm +
            W["joint_loading"]  * jli +
            W["stress_burden"]  * min(stress_bur, 1.0) +
            W["strain_burden"]  * min(strain_bur, 1.0) +
            W["spasticity"]     * spas_score +
            W["asymmetry"]      * min(asym, 1.0)
        ) * sev_sc

        rows.append({
            "muscle": short,
            "full_name": r["full_name"],
            "rpi_norm": round(rpi_norm, 3),
            "joint_loading": round(jli, 3),
            "stress_burden": round(stress_bur, 3),
            "strain_burden": round(strain_bur, 3),
            "spasticity_score": round(spas_score, 3),
            "asymmetry_burden": round(asym, 3),
            "composite_risk": round(min(composite * 100, 100.0), 1),
            "intervention": r["intervention"],
        })

    scores_df = pd.DataFrame(rows).sort_values(
        "composite_risk", ascending=False).reset_index(drop=True)
    scores_df.insert(0, "rank", scores_df.index + 1)

    # --- Top overloaded joint ---
    top_joint = "ankle"
    if loading_df is not None and not loading_df.empty:
        top_joint = str(loading_df.groupby("joint")["joint_loading_index"]
                        .mean().idxmax())

    # --- Top overloaded muscle ---
    top_muscle = str(scores_df.iloc[0]["full_name"]) if not scores_df.empty else "—"

    # --- Top phase ---
    top_phase = "TSt"
    if loading_df is not None and not loading_df.empty:
        top_phase = str(loading_df.groupby("phase")["joint_loading_index"]
                        .mean().idxmax())

    # --- Build narrative from template ---
    tmpl = _ARCHETYPE_RISK_TEMPLATES.get(phenotype_key, {
        "primary_concern":    f"Elevated mechanical burden in {phenotype_key} archetype pattern.",
        "primary_target":     "Consult archetype-specific guidelines.",
        "secondary_target":   "Contextualise with full 3D gait analysis.",
        "compensation_target": "Monitor compensatory strategies.",
        "monitoring_target":  "Routine clinical review.",
    })
    side_tmpl = _SIDE_TEMPLATES.get(side_involvement,
                                    _SIDE_TEMPLATES["bilateral"])
    sev_str = severity.capitalize()

    primary_concern    = tmpl["primary_concern"] + " " + side_tmpl["concern_suffix"]
    primary_target     = tmpl["primary_target"]  + " " + side_tmpl["target_suffix"]
    secondary_target   = tmpl["secondary_target"]
    compensation_target = tmpl["compensation_target"]
    monitoring_target  = tmpl["monitoring_target"]

    top3 = scores_df.head(3)
    top_muscles_str = ", ".join(top3["full_name"].tolist()) if not top3.empty else "—"

    narrative = (
        f"Severity: {sev_str} | Side involvement: {side_involvement}\n\n"
        f"PRIMARY MECHANICAL CONCERN:\n{primary_concern}\n\n"
        f"Top mechanically loaded muscles (simulation-estimated): {top_muscles_str}\n"
        f"Top loaded joint: {top_joint.upper()} | Top phase: {top_phase}\n\n"
        f"PRIMARY REHAB TARGET:\n{primary_target}\n\n"
        f"SECONDARY REHAB TARGET:\n{secondary_target}\n\n"
        f"COMPENSATION MONITORING:\n{compensation_target}\n\n"
        f"ONGOING MONITORING:\n{monitoring_target}\n\n"
        f"LIMITATIONS:\n"
        f"This risk score combines R-SC-GAM archetype reasoning with simulation-based "
        f"mechanical proxies. It is NOT measured EMG, NOT confirmed tissue stress, and "
        f"NOT a clinical prescription. All outputs require clinician interpretation "
        f"alongside instrumented 3D gait analysis and physical examination."
    )

    return {
        "scores_df": scores_df,
        "top_joint": top_joint,
        "top_muscle": top_muscle,
        "top_phase": top_phase,
        "primary_concern": primary_concern,
        "primary_target": primary_target,
        "secondary_target": secondary_target,
        "compensation_target": compensation_target,
        "monitoring_target": monitoring_target,
        "narrative": narrative,
    }
