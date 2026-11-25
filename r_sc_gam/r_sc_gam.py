#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
R-SC-GAM
Rehabilitation-oriented Spastic Cerebral-palsy Gait Archetype Mapper
================================================================================
A self-contained, literature-encoded computational framework that maps clinically
recognised spastic CP gait archetypes -> phase-wise muscle recruitment likelihood
and rehabilitation-priority indices, using:
    (1) normal sagittal gait templates,
    (2) phenotype-specific joint distortions,
    (3) velocity-dependent spasticity rules,
    (4) anatomical muscle-function mapping.

It produces three audience-specific outputs:
    - THERAPIST : full clinical decision-support report + figures + CSV matrices
    - PATIENT/PARENT : plain-language summary
    - CHILD : friendly, encouraging "muscle helpers" summary

It also includes a classifier: feed observed hip/knee/ankle angle curves
(e.g. from MediaPipe / OpenPose 2D pipelines) and it returns the best-matching
archetype, bridging the model to real patient data.

--------------------------------------------------------------------------------
LITERATURE BASIS (encoded directly in the constants below)
--------------------------------------------------------------------------------
Gait cycle phases (8)                  : Perry J, Burnfield JM. Gait Analysis:
                                         Normal and Pathological Function, 2nd ed.
Normal sagittal kinematics landmarks   : Perry & Burnfield; Kadaba MP et al.
                                         J Orthop Res 1990.
Normal EMG activation timing           : Perry & Burnfield; Kadaba 1990.
Spasticity (velocity dependence)       : Lance JW 1980; Modified Ashworth (Bohannon
                                         & Smith 1987).
Hemiplegic archetypes (Type I-IV)      : Winters TF, Gage JR, Hicks R.
                                         J Bone Joint Surg Am 1987;69:437-441.
Diplegic archetypes (Group I-V)        : Rodda J, Graham HK. Eur J Neurol 2001;
                                         Rodda et al. JBJS Br 2004.
Stiff-knee gait                        : Sutherland DH, Davids JR. Clin Orthop 1993.
Severity classification                : Palisano R et al. (GMFCS) Dev Med Child
                                         Neurol 1997.
Intervention evidence (BoNT-A/SDR/etc) : standard paediatric orthopaedic / neuro-
                                         rehabilitation practice (Graham, Novak et al.).

IMPORTANT CLINICAL DISCLAIMER
-----------------------------
R-SC-GAM is a *knowledge-based decision-support model*. The muscle recruitment
values are LITERATURE-INFORMED LIKELIHOODS, not measured EMG. Joint distortions
are canonical archetype templates, not a specific child's data (unless you feed
observed curves to classify_gait / build_patient_case). Every output must be
interpreted by a qualified clinician alongside instrumented 3D gait analysis,
dynamic EMG and physical examination before any intervention.

Author scaffold: built for the CNS Lab / R2D2, Dept. of Medical Science &
Technology, IIT Madras. Runs end-to-end on plain numpy + pandas + matplotlib.
================================================================================
"""

import os
import json
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # headless: works in scripts, servers, Jupyter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ OUTPUT DIR
# Hardcoded output directory. On Windows/Jupyter change to e.g. r"C:\rscgam_out".
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

GRID = np.arange(0, 101)            # 0..100 % gait cycle, 1% resolution

# ============================================================================ #
# SECTION 1 -- GAIT-CYCLE PHASE MODEL  (Perry & Burnfield)
# ============================================================================ #
# Each phase: (short, full, start%, end%, sampling-centre%)
PHASES: List[Tuple[str, str, float, float, float]] = [
    ("IC",  "Initial Contact",   0,   2,   1),
    ("LR",  "Loading Response",   2,  10,   6),
    ("MSt", "Mid Stance",        10,  30,  20),
    ("TSt", "Terminal Stance",   30,  50,  40),
    ("PSw", "Pre-Swing",         50,  60,  55),
    ("ISw", "Initial Swing",     60,  73,  66),
    ("MSw", "Mid Swing",         73,  87,  80),
    ("TSw", "Terminal Swing",    87, 100,  93),
]
PHASE_KEYS    = [p[0] for p in PHASES]
PHASE_CENTERS = np.array([p[4] for p in PHASES], dtype=float)
N_PH          = len(PHASES)
# Weight of each phase for functional importance (stance / weight-acceptance
# phases matter most for stability & energy cost -> weighted higher in RPI).
PHASE_FUNCTIONAL_WEIGHT = np.array(
    [1.0, 1.3, 1.2, 1.3, 1.1, 0.9, 0.7, 1.0])  # IC..TSw


# ============================================================================ #
# SECTION 2 -- NORMAL SAGITTAL KINEMATIC TEMPLATES
# Landmarks (%, degrees); flexion / dorsiflexion positive.
# Values are textbook means (Perry & Burnfield / Kadaba 1990), interpolated.
# ============================================================================ #
NORMAL_LANDMARKS: Dict[str, List[Tuple[float, float]]] = {
    "hip":   [(0, 32), (15, 25), (30, 10), (45, -7), (50, -8),
              (60, 0), (73, 20), (87, 30), (100, 32)],
    "knee":  [(0, 5), (8, 12), (15, 18), (35, 4), (50, 12),
              (60, 40), (70, 58), (73, 62), (85, 30), (95, 8), (100, 5)],
    "ankle": [(0, -2), (8, -6), (15, -2), (30, 5), (45, 11), (50, 8),
              (58, -12), (62, -15), (73, -6), (85, -1), (100, -2)],
    "pelvis": [(0, 11), (50, 12), (60, 13), (100, 11)],  # anterior tilt
}


def _interp(landmarks: List[Tuple[float, float]]) -> np.ndarray:
    """Linear-interpolate (%, value) landmarks onto the 0..100 GRID."""
    xs = np.array([p[0] for p in landmarks], dtype=float)
    ys = np.array([p[1] for p in landmarks], dtype=float)
    return np.interp(GRID, xs, ys)


NORMAL_CURVES: Dict[str, np.ndarray] = {j: _interp(lm)
                                        for j, lm in NORMAL_LANDMARKS.items()}


def phase_values(curve: np.ndarray) -> np.ndarray:
    """Sample a 0..100 curve at the 8 phase centres -> array length 8."""
    return np.interp(PHASE_CENTERS, GRID, curve)


def phase_velocity(curve: np.ndarray) -> np.ndarray:
    """Angular velocity (deg per %GC) at phase centres -> array length 8.
    Used as a proxy for muscle lengthening/shortening velocity, the driver
    of velocity-dependent spasticity (Lance 1980)."""
    vel = np.gradient(curve, GRID)
    return np.interp(PHASE_CENTERS, GRID, vel)


# ============================================================================ #
# SECTION 3 -- MUSCLE-FUNCTION MAP  (functional anatomy + normal EMG timing)
# ============================================================================ #
@dataclass
class Muscle:
    short: str
    name: str
    actions: Dict[str, str]        # joint -> 'flexor'/'extensor'/'dorsiflexor'/
                                   #          'plantarflexor'/'abductor'/'adductor'
    normal_phases: List[int]       # phase indices normally active (Perry/Kadaba)
    biarticular: bool
    plane: str = "sagittal"
    # Spastic signature: (controlled_joint, [lengthening phase idx],
    #                      sign) where sign=+1 means "spasticity -> MORE angle
    #                      than normal", -1 means "spasticity -> LESS angle".
    spastic_signature: Tuple[str, List[int], int] = (None, [], 0)


MUSCLES: List[Muscle] = [
    Muscle("ILIO", "Iliopsoas", {"hip": "flexor"},
           normal_phases=[4, 5], biarticular=False,
           spastic_signature=("hip", [3], +1)),          # resists hip ext @TSt
    Muscle("GMAX", "Gluteus maximus", {"hip": "extensor"},
           normal_phases=[7, 0, 1], biarticular=False),
    Muscle("GMED", "Gluteus medius", {"hip": "abductor"},
           normal_phases=[0, 1, 2], biarticular=False, plane="frontal"),
    Muscle("ADD", "Hip adductors", {"hip": "adductor"},
           normal_phases=[4, 5], biarticular=False, plane="frontal"),
    # NB: adductor over-activity (scissoring) is a FRONTAL-plane sign; it cannot
    # be inferred from the sagittal kinematics modelled here. Flag it only with
    # frontal-plane input (hip ad/abduction), otherwise leave unscored.
    Muscle("RF", "Rectus femoris", {"hip": "flexor", "knee": "extensor"},
           normal_phases=[4, 5], biarticular=True,
           spastic_signature=("knee", [4, 5], -1)),       # resists knee flex->stiff
    Muscle("VAS", "Vasti (quadriceps)", {"knee": "extensor"},
           normal_phases=[7, 0, 1], biarticular=False),
    Muscle("HAM", "Hamstrings", {"hip": "extensor", "knee": "flexor"},
           normal_phases=[7, 0, 1], biarticular=True,
           spastic_signature=("knee", [7, 0], +1)),       # resists knee ext->flexed
    Muscle("TA", "Tibialis anterior", {"ankle": "dorsiflexor"},
           normal_phases=[0, 1, 5, 6, 7], biarticular=False),
    # NB: insufficient swing dorsiflexion is modelled as TA WEAKNESS (drop foot),
    # not TA spasticity; equinus is driven by gastroc-soleus (below).
    Muscle("GAS", "Gastrocnemius", {"ankle": "plantarflexor", "knee": "flexor"},
           normal_phases=[2, 3], biarticular=True,
           spastic_signature=("ankle", [2, 3], -1)),      # resists DF -> equinus
    Muscle("SOL", "Soleus", {"ankle": "plantarflexor"},
           normal_phases=[2, 3, 4], biarticular=False,
           spastic_signature=("ankle", [2, 3], -1)),
]
MUSCLE_BY_SHORT = {m.short: m for m in MUSCLES}
MUSCLE_SHORTS   = [m.short for m in MUSCLES]


# ============================================================================ #
# SECTION 4 -- PHENOTYPE (ARCHETYPE) LIBRARY
# Each archetype = additive joint-angle offsets (deg) over the normal template
# + the muscles primarily spastic / primarily weak (literature-derived).
# ============================================================================ #
@dataclass
class Phenotype:
    key: str
    name: str
    classification: str
    gmfcs_typical: str
    description: str
    offsets: Dict[str, List[Tuple[float, float]]]   # joint -> (%, deg) offsets
    primary_spastic: List[str]
    primary_weak: List[str]
    clinical_notes: str

    def distorted_curves(self) -> Dict[str, np.ndarray]:
        out = {}
        for j, base in NORMAL_CURVES.items():
            if j in self.offsets:
                out[j] = base + _interp(self.offsets[j])
            else:
                out[j] = base.copy()
        return out


PHENOTYPES: Dict[str, Phenotype] = {
    "true_equinus": Phenotype(
        key="true_equinus", name="True Equinus",
        classification="Winters-Gage-Hicks Type II / Rodda Group I",
        gmfcs_typical="I-II",
        description="Persistent ankle plantarflexion through stance (toe-walking); "
                    "knee may hyperextend (recurvatum). Ankle is the primary lesion.",
        offsets={
            "ankle": [(0, -12), (15, -15), (40, -18), (50, -16),
                      (60, -5), (73, -3), (100, -8)],
            "knee":  [(0, -3), (30, -8), (50, -6), (60, 0), (100, 0)],
        },
        primary_spastic=["GAS", "SOL"],
        primary_weak=["TA"],
        clinical_notes="Dynamic gastroc-soleus spasticity. First-line: stretch / "
                       "serial casting / BoNT-A to triceps surae + AFO. Fixed "
                       "contracture -> consider gastroc lengthening."),

    "jump": Phenotype(
        key="jump", name="Jump Gait",
        classification="Winters-Gage-Hicks Type III / Rodda Group II",
        gmfcs_typical="II",
        description="Equinus ankle + flexed/stiff knee + flexed hip with increased "
                    "anterior pelvic tilt. Knee flexes at IC then partially extends "
                    "in stance; reduced peak knee flexion in swing (stiff knee).",
        offsets={
            "ankle": [(0, -10), (40, -14), (60, -6), (73, -4), (100, -8)],
            "knee":  [(0, 12), (15, 8), (35, 8), (50, 5), (60, -8),
                      (73, -18), (85, -5), (100, 12)],
            "hip":   [(0, 12), (40, 15), (50, 15), (73, 8), (100, 12)],
            "pelvis":[(0, 6), (100, 6)],
        },
        primary_spastic=["GAS", "SOL", "HAM", "RF", "ILIO"],
        primary_weak=["VAS", "TA"],
        clinical_notes="Multi-level involvement. Classic candidate for single-event "
                       "multilevel surgery (SEMLS) and/or BoNT-A; rectus femoris "
                       "transfer for stiff-knee component; address hamstring & psoas."),

    "apparent_equinus": Phenotype(
        key="apparent_equinus", name="Apparent Equinus",
        classification="Rodda Group III",
        gmfcs_typical="II-III",
        description="Toe-walking appearance with NORMAL ankle dorsiflexion ROM; the "
                    "apparent equinus is produced by excessive hip and knee flexion "
                    "in stance. Treating the ankle here is a classic error.",
        offsets={
            "ankle": [(0, -2), (50, 0), (100, -2)],
            "knee":  [(0, 15), (30, 18), (50, 15), (73, -5), (100, 15)],
            "hip":   [(0, 18), (40, 20), (50, 18), (100, 18)],
            "pelvis":[(0, 8), (100, 8)],
        },
        primary_spastic=["HAM", "ILIO"],
        primary_weak=["VAS", "GMAX"],
        clinical_notes="Do NOT lengthen the heel cord. Target hamstrings & psoas "
                       "(spasticity management + lengthening), strengthen quadriceps "
                       "and hip extensors."),

    "crouch": Phenotype(
        key="crouch", name="Crouch Gait",
        classification="Winters-Gage-Hicks Type IV / Rodda Group IV",
        gmfcs_typical="III-IV",
        description="Excessive knee flexion throughout the cycle (never extends), "
                    "ankle excessive dorsiflexion (calcaneus), flexed hip and "
                    "anterior pelvic tilt. High quadriceps demand & energy cost; "
                    "often follows over-lengthened heel cords (lever-arm disease).",
        offsets={
            "ankle": [(0, 5), (40, 10), (50, 8), (60, 12), (73, 5), (100, 5)],
            "knee":  [(0, 25), (30, 28), (50, 25), (60, 10),
                      (73, -2), (85, 10), (100, 25)],
            "hip":   [(0, 22), (40, 25), (50, 22), (100, 22)],
            "pelvis":[(0, 10), (100, 10)],
        },
        primary_spastic=["HAM", "ILIO"],
        primary_weak=["VAS", "SOL", "GMAX"],
        clinical_notes="Quadriceps overload + plantarflexor/knee-extension lever-arm "
                       "failure. Strengthen quads & plantarflexors, hamstring "
                       "lengthening, GRAFO orthoses; surgical correction of lever-arm "
                       "deformities. Avoid further heel-cord lengthening."),

    "stiff_knee": Phenotype(
        key="stiff_knee", name="Stiff-Knee Gait",
        classification="Sutherland & Davids (swing-phase knee dysfunction)",
        gmfcs_typical="I-II",
        description="Reduced and delayed peak knee flexion in swing from rectus "
                    "femoris co-spasticity, impairing foot clearance.",
        offsets={
            "knee":  [(0, 2), (50, 2), (60, -12), (70, -22),
                      (73, -24), (85, -10), (100, 2)],
            "ankle": [(0, -5), (40, -6), (100, -5)],
        },
        primary_spastic=["RF"],
        primary_weak=[],
        clinical_notes="Confirm with dynamic EMG (rectus active in swing). "
                       "Rectus femoris transfer / BoNT-A to rectus; ensure adequate "
                       "knee-flexion velocity at toe-off."),

    "drop_foot": Phenotype(
        key="drop_foot", name="Drop Foot",
        classification="Winters-Gage-Hicks Type I (hemiplegic)",
        gmfcs_typical="I",
        description="Insufficient swing-phase dorsiflexion (foot drop) with intact "
                    "passive ankle ROM; weak / poorly-timed dorsiflexors rather than "
                    "spastic equinus. Compensatory hip & knee flexion to clear foot.",
        offsets={
            "ankle": [(0, -8), (20, 0), (40, 2), (50, 0), (60, -10),
                      (73, -14), (87, -12), (100, -10)],
            "knee":  [(0, 2), (60, 5), (73, 8), (100, 2)],
            "hip":   [(0, 3), (73, 8), (100, 3)],
        },
        primary_spastic=[],
        primary_weak=["TA"],
        clinical_notes="Primarily weakness/timing, not spasticity. Leaf-spring AFO "
                       "for clearance, dorsiflexor strengthening / FES; rule out "
                       "fixed equinus contracture."),
}


# ============================================================================ #
# SECTION 5 -- SPASTICITY ENGINE  (velocity-dependent rule evaluation)
# ============================================================================ #
def spasticity_scores(pheno: Phenotype) -> Dict[str, dict]:
    """
    For each muscle with a spastic signature, test whether the phenotype's joint
    deviation is consistent with velocity-dependent spasticity of that muscle.

    Mechanism (Lance 1980): a spastic muscle resists rapid lengthening, so the
    joint fails to complete the motion that lengthens the muscle. We therefore
    look, in the muscle's lengthening phases, at:
        evidence = sign * (deviation)    (only positive evidence counts)
    weighted by the *normal lengthening velocity* at that phase (faster stretch
    -> stronger velocity-dependent response).

    Returns short -> {score(0..1), flagged(bool), evidence_phases}.
    """
    dc = pheno.distorted_curves()
    out = {}
    for m in MUSCLES:
        joint, leng_phases, sign = m.spastic_signature
        if joint is None or not leng_phases:
            continue
        dev = phase_values(dc[joint]) - phase_values(NORMAL_CURVES[joint])
        vel = np.abs(phase_velocity(NORMAL_CURVES[joint]))
        ev = 0.0
        ev_phases = []
        for pi in leng_phases:
            contrib = sign * dev[pi]                      # positive => consistent
            if contrib > 0:
                w = 1.0 + 0.04 * vel[pi]                  # velocity weighting
                ev += contrib * w
                ev_phases.append(PHASE_KEYS[pi])
        # prior boost if literature lists this muscle as primarily spastic
        prior = 1.35 if m.short in pheno.primary_spastic else 1.0
        raw = ev * prior
        # squash to 0..1 (25 deg-equivalent ~ saturating)
        score = float(1 - np.exp(-raw / 25.0))
        out[m.short] = {"score": round(score, 3),
                        "flagged": score >= 0.30,
                        "evidence_phases": ev_phases}
    return out


def weakness_scores(pheno: Phenotype) -> Dict[str, float]:
    """Literature-listed weak muscles get a severity proportional to how
    functionally loaded their normal-active phases are (stance weakness costs
    more). Returns short -> 0..1."""
    out = {}
    for short in pheno.primary_weak:
        m = MUSCLE_BY_SHORT[short]
        w = np.mean([PHASE_FUNCTIONAL_WEIGHT[pi] for pi in m.normal_phases]) \
            if m.normal_phases else 1.0
        out[short] = float(min(1.0, 0.55 * w))
    return out


# ============================================================================ #
# SECTION 6 -- PHASE-WISE MUSCLE RECRUITMENT LIKELIHOOD MATRIX
# rows = muscles, cols = 8 phases, values 0..1
# ============================================================================ #
BASE_ACTIVE   = 1.00     # normal on-phase baseline
BASE_INACTIVE = 0.12     # normal off-phase baseline
SPASTIC_GAIN  = 0.92     # prolonged / intensified activation level
COCONTRACT    = 0.45     # inappropriate co-contraction floor for spastic muscle
WEAK_FLOOR    = 0.40     # reduced recruitability of a weak muscle
COMP_BOOST    = 0.28     # synergist compensation increment

# Compensation map: weak muscle -> synergists recruited harder (stance focus)
COMPENSATORS: Dict[str, List[str]] = {
    "VAS":  ["SOL", "GMAX", "HAM"],     # crouch: extension couple compensates
    "TA":   ["RF", "ILIO", "HAM"],      # drop foot: hip flexors aid clearance
    "SOL":  ["VAS", "GMAX"],
    "GMAX": ["HAM", "SOL"],
}
STANCE_PHASES = [0, 1, 2, 3, 4]


def recruitment_matrix(pheno: Phenotype) -> Tuple[pd.DataFrame, Dict[str, dict],
                                                  Dict[str, float]]:
    spas = spasticity_scores(pheno)
    weak = weakness_scores(pheno)

    M = np.full((len(MUSCLES), N_PH), BASE_INACTIVE, dtype=float)
    for i, m in enumerate(MUSCLES):
        for pi in m.normal_phases:
            M[i, pi] = BASE_ACTIVE

    for i, m in enumerate(MUSCLES):
        s = spas.get(m.short, {}).get("score", 0.0)
        # --- spasticity: prolong + intensify + premature onset + co-contraction
        if m.short in pheno.primary_spastic or s >= 0.30:
            active = set(m.normal_phases)
            # prolong into neighbouring phases (premature on / delayed off)
            extended = set()
            for pi in active:
                extended |= {pi, (pi - 1) % N_PH, (pi + 1) % N_PH}
            for pi in extended:
                M[i, pi] = max(M[i, pi], SPASTIC_GAIN * (0.7 + 0.3 * s))
            # inappropriate co-contraction floor across the whole cycle
            M[i, :] = np.maximum(M[i, :], COCONTRACT * s)
        # --- weakness: cannot recruit on-phase well
        if m.short in weak:
            for pi in m.normal_phases:
                M[i, pi] = min(M[i, pi], WEAK_FLOOR * (1 - 0.5 * weak[m.short]))

    # --- synergist compensation for weak muscles
    for wshort, wsev in weak.items():
        for cshort in COMPENSATORS.get(wshort, []):
            ci = MUSCLE_SHORTS.index(cshort)
            for pi in STANCE_PHASES:
                M[ci, pi] = min(1.0, M[ci, pi] + COMP_BOOST * wsev)

    M = np.clip(M, 0.0, 1.0)
    df = pd.DataFrame(M, index=[f"{m.short} ({m.name})" for m in MUSCLES],
                      columns=PHASE_KEYS)
    return df, spas, weak


# ============================================================================ #
# SECTION 7 -- REHABILITATION PRIORITY INDEX (RPI) + intervention mapping
# ============================================================================ #
# RPI weights (transparent + tunable)
W_SPASTIC   = 0.45
W_DEVIATION = 0.30
W_WEAKNESS  = 0.25


def deviation_contribution(pheno: Phenotype, muscle: Muscle) -> float:
    """How much the muscle's controlled joint(s) deviate during functionally
    important phases -> 0..1 (saturating at ~30 deg weighted mean)."""
    dc = pheno.distorted_curves()
    total, n = 0.0, 0
    for joint in muscle.actions:
        if joint not in dc:
            continue
        dev = np.abs(phase_values(dc[joint]) - phase_values(NORMAL_CURVES[joint]))
        total += float(np.sum(dev * PHASE_FUNCTIONAL_WEIGHT) /
                       np.sum(PHASE_FUNCTIONAL_WEIGHT))
        n += 1
    if n == 0:
        return 0.0
    mean_dev = total / n
    return float(1 - np.exp(-mean_dev / 30.0))


def intervention_for(spastic: float, weak: float, dev: float) -> Tuple[str, str]:
    """Return (category, clinician-facing rationale) from the dominant driver."""
    if spastic >= 0.5 and spastic >= weak:
        cat = "Spasticity management"
        why = ("Velocity-dependent over-activity dominates. Stretching / serial "
               "casting / focal BoNT-A; if multi-level & severe consider SDR or "
               "tone-reducing surgery. Pair with antagonist strengthening + orthosis.")
    elif weak >= 0.5 and weak > spastic:
        cat = "Strengthening / motor control"
        why = ("Weakness / poor timing dominates. Progressive resistance training, "
               "FES or task-specific motor training; orthosis for compensation while "
               "strength is rebuilt.")
    elif spastic >= 0.3 and weak >= 0.3:
        cat = "Mixed (tone + weakness)"
        why = ("Both over-activity and weakness present. Reduce tone first (BoNT-A / "
               "stretch) THEN strengthen the now-accessible muscle; orthotic support.")
    elif dev >= 0.4:
        cat = "Biomechanical / orthotic"
        why = ("Marked kinematic deviation without clear tone/weakness signal -> "
               "likely contracture or lever-arm issue. Orthotic management; refer "
               "for surgical assessment if fixed.")
    else:
        cat = "Monitor"
        why = "Low priority; routine monitoring and maintenance stretching."
    return cat, why


def rehab_priority(pheno: Phenotype) -> pd.DataFrame:
    df_rec, spas, weak = recruitment_matrix(pheno)
    rows = []
    for m in MUSCLES:
        s = spas.get(m.short, {}).get("score", 0.0)
        w = weak.get(m.short, 0.0)
        d = deviation_contribution(pheno, m)
        rpi = 100.0 * (W_SPASTIC * s + W_WEAKNESS * w + W_DEVIATION * d)
        cat, why = intervention_for(s, w, d)
        rows.append({
            "muscle": m.short, "full_name": m.name,
            "spasticity": round(s, 3), "weakness": round(w, 3),
            "deviation": round(d, 3), "RPI": round(rpi, 1),
            "intervention": cat, "rationale": why,
        })
    out = pd.DataFrame(rows).sort_values("RPI", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


# ============================================================================ #
# SECTION 8 -- ARCHETYPE CLASSIFIER  (bridge to observed 2D/3D pipelines)
# ============================================================================ #
def classify_gait(observed: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    observed: dict joint -> 0..100 angle curve (deg), keys among hip/knee/ankle.
    Returns archetypes ranked by similarity to the observed deviation signature.

    Similarity = 1 / (1 + RMS phase-centre deviation-difference). This lets you
    feed MediaPipe/OpenPose-derived joint angles straight in.
    """
    obs_dev = {}
    for j in ("hip", "knee", "ankle"):
        if j in observed:
            obs_dev[j] = phase_values(observed[j]) - phase_values(NORMAL_CURVES[j])

    rows = []
    for key, ph in PHENOTYPES.items():
        dc = ph.distorted_curves()
        diffs = []
        for j, od in obs_dev.items():
            pd_dev = phase_values(dc[j]) - phase_values(NORMAL_CURVES[j])
            diffs.append(od - pd_dev)
        if not diffs:
            continue
        rms = float(np.sqrt(np.mean(np.concatenate(diffs) ** 2)))
        rows.append({"archetype": ph.name, "key": key,
                     "rms_deg": round(rms, 2),
                     "similarity": round(1.0 / (1.0 + rms / 10.0), 3)})
    out = pd.DataFrame(rows).sort_values("similarity", ascending=False)\
                            .reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


# ============================================================================ #
# SECTION 9 -- VISUALISATION
# ============================================================================ #
PHASE_BAND_COLORS = ["#eef4ff", "#dbe7ff"]   # alternating stripe fill


def _add_phase_bands(ax, ymin, ymax, label_top=False):
    for i, (k, full, s, e, c) in enumerate(PHASES):
        ax.axvspan(s, e, color=PHASE_BAND_COLORS[i % 2], zorder=0)
        if label_top:
            ax.text((s + e) / 2, ymax, k, ha="center", va="bottom",
                    fontsize=7, color="#33415c")
    ax.axvline(60, color="#94a3b8", ls="--", lw=1)   # stance/swing boundary


def plot_kinematics(pheno: Phenotype, path: str):
    dc = pheno.distorted_curves()
    joints = ["hip", "knee", "ankle"]
    titles = ["Hip flexion(+)/extension(-)",
              "Knee flexion(+)",
              "Ankle dorsiflexion(+)/plantarflexion(-)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, j, t in zip(axes, joints, titles):
        n, d = NORMAL_CURVES[j], dc[j]
        ymin = min(n.min(), d.min()) - 6
        ymax = max(n.max(), d.max()) + 8
        _add_phase_bands(ax, ymin, ymax, label_top=(j == "hip"))
        ax.plot(GRID, n, color="#1d4ed8", lw=2.2, label="Normal template")
        ax.plot(GRID, d, color="#dc2626", lw=2.2, label=pheno.name)
        ax.fill_between(GRID, n, d, color="#dc2626", alpha=0.10)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("% Gait Cycle"); ax.set_ylabel("degrees")
        ax.set_xlim(0, 100); ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle(f"R-SC-GAM  |  Kinematics: {pheno.name}  "
                 f"({pheno.classification})", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_recruitment(pheno: Phenotype, path: str):
    df, _, _ = recruitment_matrix(pheno)
    cmap = LinearSegmentedColormap.from_list(
        "rec", ["#f8fafc", "#fde68a", "#f97316", "#b91c1c"])
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    im = ax.imshow(df.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(N_PH)); ax.set_xticklabels(PHASE_KEYS)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index, fontsize=8)
    for i in range(len(df)):
        for j in range(N_PH):
            v = df.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if v > 0.55 else "#1f2937")
    ax.set_title(f"R-SC-GAM  |  Phase-wise muscle recruitment likelihood\n{pheno.name}",
                 fontsize=11, weight="bold")
    ax.set_xlabel("Gait phase")
    fig.colorbar(im, ax=ax, label="recruitment likelihood (0-1)", shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_rpi(pheno: Phenotype, path: str):
    rpi = rehab_priority(pheno)
    cat_color = {
        "Spasticity management": "#dc2626",
        "Strengthening / motor control": "#2563eb",
        "Mixed (tone + weakness)": "#7c3aed",
        "Biomechanical / orthotic": "#059669",
        "Monitor": "#94a3b8",
    }
    colors = [cat_color[c] for c in rpi["intervention"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(rpi["muscle"][::-1], rpi["RPI"][::-1],
            color=colors[::-1], edgecolor="#1f2937")
    for y, (v, c) in enumerate(zip(rpi["RPI"][::-1], rpi["intervention"][::-1])):
        ax.text(v + 0.6, y, f"{v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("Rehabilitation Priority Index (0-100)")
    ax.set_title(f"R-SC-GAM  |  Rehab priority ranking: {pheno.name}",
                 fontsize=11, weight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in cat_color.values()]
    ax.legend(handles, cat_color.keys(), fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_child(pheno: Phenotype, path: str):
    """Friendly 'muscle helpers' star-chart for a child."""
    rpi = rehab_priority(pheno).head(4)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axis("off")
    ax.text(0.5, 0.95, "My Walking Helpers \u2605", ha="center",
            fontsize=18, weight="bold", color="#7c3aed", transform=ax.transAxes)
    ax.text(0.5, 0.86, "These muscle friends need a little extra practice!",
            ha="center", fontsize=11, color="#475569", transform=ax.transAxes)
    nice = {"GAS": "Calf (back of leg)", "SOL": "Deep calf",
            "HAM": "Back-of-thigh", "RF": "Front-of-thigh helper",
            "VAS": "Knee straightener", "ILIO": "Hip lifter",
            "TA": "Toe lifter", "GMAX": "Bottom muscle",
            "GMED": "Side-hip helper", "ADD": "Inner-thigh"}
    y = 0.68
    for _, r in rpi.iterrows():
        stars = int(round(r["RPI"] / 20)) + 1
        stars = max(1, min(5, stars))
        ax.text(0.08, y, "\u2605" * stars, fontsize=15, transform=ax.transAxes)
        ax.text(0.42, y, nice.get(r["muscle"], r["full_name"]),
                fontsize=12, va="center", transform=ax.transAxes, color="#1f2937")
        y -= 0.15
    ax.text(0.5, 0.04, "More stars = let's give this helper more love in practice!",
            ha="center", fontsize=9, style="italic", color="#64748b",
            transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ============================================================================ #
# SECTION 10 -- AUDIENCE-SPECIFIC REPORTS
# ============================================================================ #
def therapist_report(pheno: Phenotype) -> str:
    df_rec, spas, weak = recruitment_matrix(pheno)
    rpi = rehab_priority(pheno)
    dc = pheno.distorted_curves()
    L = []
    L.append("=" * 78)
    L.append(f"R-SC-GAM  CLINICAL DECISION-SUPPORT REPORT  --  {pheno.name}")
    L.append("=" * 78)
    L.append(f"Classification : {pheno.classification}")
    L.append(f"Typical GMFCS  : {pheno.gmfcs_typical}")
    L.append(textwrap.fill("Description    : " + pheno.description, 78))
    L.append("")
    L.append("-- Kinematic deviation from normal (deg, at phase centres) " + "-" * 18)
    L.append(f"{'phase':>6} " + " ".join(f"{k:>6}" for k in PHASE_KEYS))
    for j in ("hip", "knee", "ankle"):
        dev = phase_values(dc[j]) - phase_values(NORMAL_CURVES[j])
        L.append(f"{j:>6} " + " ".join(f"{v:+6.1f}" for v in dev))
    L.append("")
    L.append("-- Spasticity rule findings (velocity-dependent) " + "-" * 28)
    if any(v["flagged"] for v in spas.values()):
        for short, v in spas.items():
            if v["flagged"]:
                ph = ",".join(v["evidence_phases"]) or "-"
                L.append(f"   [FLAG] {MUSCLE_BY_SHORT[short].name:<22} "
                         f"score={v['score']:.2f}  evidence phases: {ph}")
    else:
        L.append("   (no spastic muscle flagged for this archetype)")
    if weak:
        L.append("")
        L.append("-- Weakness findings " + "-" * 57)
        for short, sc in weak.items():
            L.append(f"   [WEAK] {MUSCLE_BY_SHORT[short].name:<22} severity={sc:.2f}")
    L.append("")
    L.append("-- Rehabilitation priority ranking " + "-" * 42)
    L.append(f"{'rank':>4} {'muscle':>6} {'RPI':>6}  {'spas':>5} {'weak':>5} "
             f"{'dev':>5}  intervention")
    for _, r in rpi.iterrows():
        L.append(f"{r['rank']:>4} {r['muscle']:>6} {r['RPI']:>6.1f}  "
                 f"{r['spasticity']:>5.2f} {r['weakness']:>5.2f} "
                 f"{r['deviation']:>5.2f}  {r['intervention']}")
    L.append("")
    L.append("-- Intervention rationale (top 3) " + "-" * 43)
    for _, r in rpi.head(3).iterrows():
        L.append(f"   * {r['full_name']} ({r['intervention']}):")
        L.append(textwrap.fill(r["rationale"], 74,
                               initial_indent="       ", subsequent_indent="       "))
    L.append("")
    L.append("-- Archetype clinical note " + "-" * 51)
    L.append(textwrap.fill(pheno.clinical_notes, 74,
                           initial_indent="   ", subsequent_indent="   "))
    L.append("")
    L.append("DISCLAIMER: literature-informed decision support, NOT a prescription. "
             "Confirm with instrumented 3D gait analysis, dynamic EMG and physical "
             "examination.")
    L.append("=" * 78)
    return "\n".join(L)


def patient_report(pheno: Phenotype) -> str:
    rpi = rehab_priority(pheno)
    top = rpi.head(3)
    L = []
    L.append(f"YOUR WALKING PATTERN SUMMARY  --  closest match: {pheno.name}")
    L.append("-" * 64)
    L.append(textwrap.fill(
        "Your walking pattern most closely resembles a recognised pattern called "
        f"'{pheno.name}'. In plain terms: " + pheno.description.lower(), 64))
    L.append("")
    L.append("What this means for your therapy focus:")
    for _, r in top.iterrows():
        if r["intervention"].startswith("Spasticity"):
            action = "is working too hard / too tight -> focus on stretching & tone relief"
        elif r["intervention"].startswith("Strength"):
            action = "is weak -> focus on strengthening it"
        elif r["intervention"].startswith("Mixed"):
            action = "is both tight and weak -> loosen first, then strengthen"
        elif r["intervention"].startswith("Biomech"):
            action = "may benefit from a brace/orthosis -> ask about support options"
        else:
            action = "just needs monitoring"
        L.append(f"  - {r['full_name']}: {action}")
    L.append("")
    L.append(textwrap.fill(
        "These are general, evidence-based directions. Your therapist or doctor "
        "will tailor the exact exercises, braces or treatments to you after an "
        "in-person assessment.", 64))
    return "\n".join(L)


def child_report(pheno: Phenotype) -> str:
    rpi = rehab_priority(pheno).head(3)
    nice = {"GAS": "your calf", "SOL": "your deep calf", "HAM": "the back of your thigh",
            "RF": "the front of your thigh", "VAS": "your knee-straightener",
            "ILIO": "your hip-lifter", "TA": "your toe-lifter",
            "GMAX": "your bottom muscle", "GMED": "your side-hip", "ADD": "your inner thigh"}
    L = []
    L.append("HELLO, SUPER-WALKER! \u2b50")
    L.append("-" * 40)
    L.append("Your legs have lots of muscle friends that help you walk.")
    L.append("A few of them need some extra practice and love:")
    L.append("")
    for _, r in rpi.iterrows():
        L.append(f"  \u2b50 {nice.get(r['muscle'], r['full_name'])} "
                 f"- let's give it some special exercises!")
    L.append("")
    L.append("Every bit of practice makes you stronger. You're doing great! \U0001F31F")
    return "\n".join(L)


# ============================================================================ #
# SECTION 11 -- MAIN PIPELINE
# ============================================================================ #
def run_phenotype(pheno: Phenotype) -> dict:
    base = os.path.join(OUT_DIR, pheno.key)
    # figures
    plot_kinematics(pheno, base + "_kinematics.png")
    plot_recruitment(pheno, base + "_recruitment.png")
    plot_rpi(pheno, base + "_rpi.png")
    plot_child(pheno, base + "_child.png")
    # tables
    df_rec, _, _ = recruitment_matrix(pheno)
    df_rec.round(3).to_csv(base + "_recruitment_matrix.csv")
    rpi = rehab_priority(pheno)
    rpi.to_csv(base + "_rehab_priority.csv", index=False)
    # reports
    with open(base + "_THERAPIST.txt", "w", encoding="utf-8") as f:
        f.write(therapist_report(pheno))
    with open(base + "_PATIENT.txt", "w", encoding="utf-8") as f:
        f.write(patient_report(pheno))
    with open(base + "_CHILD.txt", "w", encoding="utf-8") as f:
        f.write(child_report(pheno))
    return {"phenotype": pheno.name, "out_prefix": base}


def main():
    print("R-SC-GAM running...  output ->", OUT_DIR, "\n")
    summary = []
    for key, ph in PHENOTYPES.items():
        info = run_phenotype(ph)
        rpi = rehab_priority(ph)
        top = rpi.iloc[0]
        print(f"  [{ph.name:<18}] {ph.classification}")
        print(f"      top priority: {top['full_name']} "
              f"(RPI {top['RPI']:.0f}, {top['intervention']})")
        summary.append({"key": key, "name": ph.name,
                        "top_muscle": top["full_name"],
                        "top_RPI": top["RPI"],
                        "top_intervention": top["intervention"]})
    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "_SUMMARY.csv"), index=False)

    # ---- demonstrate the classifier on a synthetic "observed" gait ----------
    print("\n  Classifier demo (synthetic crouch-like observation + noise):")
    rng = np.random.default_rng(7)
    crouch = PHENOTYPES["crouch"].distorted_curves()
    observed = {j: crouch[j] + rng.normal(0, 2.0, size=GRID.shape)
                for j in ("hip", "knee", "ankle")}
    res = classify_gait(observed)
    res.to_csv(os.path.join(OUT_DIR, "_classifier_demo.csv"), index=False)
    for _, r in res.head(3).iterrows():
        print(f"      #{r['rank']} {r['archetype']:<18} "
              f"similarity={r['similarity']:.3f} (rms {r['rms_deg']} deg)")

    print("\nDone. Per-archetype outputs: kinematics / recruitment / rpi / child "
          "figures + recruitment-matrix & rehab-priority CSVs + THERAPIST/PATIENT/"
          "CHILD reports.")
    return summary


if __name__ == "__main__":
    main()
