"""
pages_home.py — Overview / landing tab
"""

import streamlit as st


def render_home(archetype_key: str, severity: str, side_involvement: str,
                enable_mujoco: bool, enable_ss: bool):
    from core.r_sc_gam_core import PHENOTYPES

    pheno = PHENOTYPES[archetype_key]

    st.title("R-SC-GAM-MJ-SSL")
    st.markdown(
        "**Rehabilitation-Oriented Spasticity-Constrained Gait Archetype Model "
        "with MuJoCo Stress-Strain Loading**"
    )

    st.info(
        "**Scientific Boundary** — This framework estimates *muscle recruitment "
        "likelihood*, *joint torque burden*, *approximate muscle-tendon strain*, "
        "and *actuator stress proxies* from archetype-based CP gait models.  \n"
        "It does **NOT** measure EMG, diagnose CP, or prescribe treatment."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Archetype", pheno.name)
    col2.metric("Severity", severity.capitalize())
    col3.metric("Side", side_involvement.capitalize())

    st.markdown("---")

    st.subheader("Selected Archetype Description")
    st.write(f"**Classification:** {pheno.classification}")
    st.write(f"**Typical GMFCS:** {pheno.gmfcs_typical}")
    st.write(pheno.description)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Primary Spastic Muscles")
        if pheno.primary_spastic:
            for s in pheno.primary_spastic:
                st.write(f"- {s}")
        else:
            st.write("_None identified for this archetype_")

    with col_b:
        st.subheader("Primary Weak Muscles")
        if pheno.primary_weak:
            for s in pheno.primary_weak:
                st.write(f"- {s}")
        else:
            st.write("_None identified for this archetype_")

    st.markdown("---")
    st.subheader("Clinical Notes")
    st.write(pheno.clinical_notes)

    st.markdown("---")
    st.subheader("Active Modules")
    mc1, mc2, mc3 = st.columns(3)
    mc1.success("R-SC-GAM Core  ✓")
    mc2.success("Mechanical Load Engine  ✓")
    if enable_mujoco:
        mc3.success("MuJoCo Simulation  ✓")
    else:
        mc3.info("MuJoCo: Proxy mode")

    if enable_ss:
        st.success("Stress-Strain Engine  ✓")
    else:
        st.info("Stress-Strain Engine: disabled (toggle in sidebar)")

    st.markdown("---")
    st.subheader("Pipeline Overview")
    st.markdown("""
```
Input: CP Archetype + Severity + Side
          │
          ▼
  R-SC-GAM Core
  ┌─────────────────────────────────────────────────┐
  │  Archetype phenotype distorted joint curves     │
  │  Spasticity scoring (velocity-dependent rules)  │
  │  Muscle recruitment likelihood matrix           │
  │  Rehabilitation Priority Index (RPI)            │
  └─────────────────────────────────────────────────┘
          │
          ▼
  MuJoCo / Mechanical Proxy Simulation
  ┌─────────────────────────────────────────────────┐
  │  Joint torques, velocities, accelerations       │
  │  Contact force proxies                          │
  │  Estimated tendon/muscle lengths                │
  └─────────────────────────────────────────────────┘
          │
          ▼
  Stress-Strain Engine
  ┌─────────────────────────────────────────────────┐
  │  Estimated muscle-tendon strain                 │
  │  Actuator stress proxy                          │
  │  Phase-weighted burden indices                  │
  └─────────────────────────────────────────────────┘
          │
          ▼
  Rehab Risk Engine
  ┌─────────────────────────────────────────────────┐
  │  Composite mechanical risk score                │
  │  Top overloaded joint / muscle / phase          │
  │  Ranked rehab targets                           │
  └─────────────────────────────────────────────────┘
          │
          ▼
  Reports + Exports
```
""")
