# R-SC-GAM-MJ-SSL

**Rehabilitation-Oriented Spasticity-Constrained Gait Archetype Model with MuJoCo Stress-Strain Loading**

---

## Project Purpose

R-SC-GAM-MJ-SSL is a computational rehabilitation reasoning and simulation framework. It estimates muscle recruitment likelihood, joint loading burden, and approximate muscle-tendon stress-strain proxies from archetype-based CP gait models. It does not measure EMG, diagnose CP, or prescribe treatment.

The framework combines:
- **R-SC-GAM**: Literature-encoded CP gait archetype model with spasticity scoring and rehabilitation priority index
- **MuJoCo / Proxy Simulation**: Joint torque, contact force, tendon length estimates from physics or deterministic proxy
- **Stress-Strain Engine**: Approximate normalised muscle-tendon strain and actuator stress proxies
- **Rehab Risk Engine**: Composite mechanical risk ranking and rehabilitation target generation

---

## Scientific Boundary

> This is NOT measured EMG.  
> This is NOT true biological muscle activation.  
> This is NOT clinical prescription.  
> This is a **computational rehabilitation reasoning and simulation framework**.

Correct terminology used throughout:
- muscle recruitment likelihood
- estimated actuator stress
- estimated muscle-tendon strain
- joint torque burden
- mechanical loading index
- rehabilitation priority target
- simulation-based hypothesis generation

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## Folder Structure

```
r_sc_gam_mujoco_ssl/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── README.md
├── core/
│   ├── r_sc_gam_core.py            # Preserved R-SC-GAM scientific model
│   ├── mechanical_load_engine.py   # Joint torque / loading burden
│   ├── stress_strain_engine.py     # Muscle-tendon strain/stress proxies
│   ├── rehab_risk_engine.py        # Composite risk score + targets
│   └── report_engine.py            # Extended report generation
├── mujoco_engine/
│   ├── mj_model_builder.py         # MuJoCo XML biped model
│   ├── mj_runner.py                # Simulation runner + proxy fallback
│   ├── mj_data_extractor.py        # Phase-wise data extraction
│   └── mj_visualizer.py            # Matplotlib figures from sim data
├── ui/
│   ├── pages_home.py               # Overview tab
│   ├── pages_archetype.py          # Archetype, recruitment, RPI tabs
│   ├── pages_mujoco.py             # MuJoCo simulation tab
│   ├── pages_stress_strain.py      # Stress-strain tab
│   └── pages_reports.py            # Risk map + reports tab
└── outputs/
    ├── figures/                    # Exported PNG figures
    ├── csv/                        # Exported CSV data
    ├── reports/                    # Exported text reports
    └── mujoco/                     # MuJoCo outputs
```

---

## How to Interpret Outputs

### Archetype Model Tab
- Blue = normal gait template
- Red = CP archetype distorted pattern (scaled by severity)
- Green dashed = hypothetical correction target visualization (does NOT claim real outcome)

### Recruitment Likelihood Heatmap
- Values 0–1 represent estimated likelihood of muscle activity per phase
- Derived from spasticity rules and weakness scores, NOT measured EMG

### Rehab Priority Index (RPI)
- 0–100 score combining spasticity (45%), weakness (25%), and deviation (30%)
- Higher = higher clinical attention priority

### Mechanical Loading
- Joint torque proxy, contact force proxy, and loading index per phase
- Computed from kinematics via simplified mechanical equations or MuJoCo physics

### Stress-Strain Proxies
- Estimated muscle-tendon strain = (current length - resting length) / resting length
- Estimated stress proxy = actuator force / PCSA
- Both are approximate research estimates, NOT confirmed tissue mechanics

### Rehab Risk Score
- Composite score (0–100) = weighted sum of RPI, loading, stress, strain, spasticity, asymmetry
- Provides ranked rehabilitation target prioritisation

---

## Archetypes Supported

| Key | Name | Classification | Typical GMFCS |
|-----|------|---------------|----------------|
| true_equinus | True Equinus | WGH Type II / Rodda Group I | I–II |
| jump | Jump Gait | WGH Type III / Rodda Group II | II |
| apparent_equinus | Apparent Equinus | Rodda Group III | II–III |
| crouch | Crouch Gait | WGH Type IV / Rodda Group IV | III–IV |
| stiff_knee | Stiff-Knee Gait | Sutherland & Davids | I–II |
| drop_foot | Drop Foot | WGH Type I | I |

---

## Limitations

1. All joint kinematics are **canonical archetype templates**, not individual patient data
2. Mechanical proxies use simplified segment inertia and stiffness values
3. The model is **sagittal-plane dominant** — frontal-plane muscles (adductors, gluteus medius) use rule-based approximations only
4. Stress-strain estimates require patient-specific geometry, PCSA, and material properties for accuracy
5. MuJoCo model is a simplified 6-DOF lower limb — not validated against full clinical gait analysis
6. All outputs are for **research visualisation and hypothesis generation** only

---

## Validation Pathway

1. Confirm archetype with instrumented 3D gait analysis (force plates + cameras)
2. Validate muscle recruitment with dynamic surface/fine-wire EMG
3. Assess spasticity with MAS/Tardieu scale + passive ROM examination
4. Compare simulation torque estimates with inverse dynamics from 3D data
5. Obtain imaging (ultrasound/MRI) for muscle architecture if tissue detail is needed
6. All intervention decisions must be made by qualified clinicians

---

## Literature Basis

- Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function, 2nd ed.
- Kadaba MP et al. J Orthop Res 1990.
- Winters TF, Gage JR, Hicks R. JBJS Am 1987;69:437-441.
- Rodda J, Graham HK. Eur J Neurol 2001; Rodda et al. JBJS Br 2004.
- Sutherland DH, Davids JR. Clin Orthop 1993.
- Lance JW 1980; Bohannon & Smith 1987 (Modified Ashworth Scale).
- Arnold et al. Ann Biomed Eng 2010 (muscle geometry).

---

## Author

Built for computational rehabilitation research.  
CNS Lab / R2D2 — Dept. of Medical Science & Technology, IIT Madras.
