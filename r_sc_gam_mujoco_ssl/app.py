#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — R-SC-GAM-MJ-SSL Streamlit Application
Rehabilitation-Oriented Spasticity-Constrained Gait Archetype Model
with MuJoCo Stress-Strain Loading

Run:  streamlit run app.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np

# ── page config must come first ──────────────────────────────────────────────
st.set_page_config(
    page_title="R-SC-GAM-MJ-SSL",
    page_icon="🦿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.r_sc_gam_core import PHENOTYPES, NORMAL_CURVES, GRID

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(os.path.join(OUT_DIR, "figures"),  exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "csv"),      exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "reports"),  exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "mujoco"),   exist_ok=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("R-SC-GAM-MJ-SSL")
    st.caption("Rehab-Oriented CP Gait + MuJoCo Stress-Strain Loading")
    st.markdown("---")

    archetype_key = st.selectbox(
        "CP Gait Archetype",
        options=list(PHENOTYPES.keys()),
        format_func=lambda k: PHENOTYPES[k].name,
        index=0,
    )

    severity = st.selectbox(
        "Severity",
        options=["mild", "moderate", "severe"],
        index=1,
    )

    side_involvement = st.selectbox(
        "Side Involvement",
        options=["bilateral", "left", "right"],
        index=0,
    )

    st.markdown("---")

    enable_mujoco = st.checkbox("Enable MuJoCo Simulation", value=False,
                                help="Attempt real MuJoCo physics; falls back to proxy automatically.")

    enable_ss = st.checkbox("Enable Stress-Strain Approximation", value=True)

    st.markdown("---")
    st.subheader("Hypothetical Correction Strength")
    correction_strength = st.slider(
        "Correction strength (0 = CP pattern, 1 = Normal)",
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        help="Hypothetical correction target visualization — does NOT claim real therapy outcome."
    )
    st.caption("Green dashed lines on kinematic plots show this hypothetical target.")

    st.markdown("---")
    st.markdown(
        "_R-SC-GAM-MJ-SSL v1.0_  \n"
        "Computational Rehabilitation Reasoning Framework  \n"
        "NOT a medical device — NOT a prescription."
    )


# ── Run computation (cached by inputs) ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(archetype_key: str, severity: str, side_involvement: str,
                 enable_mujoco: bool, enable_ss: bool):
    from mujoco_engine.mj_runner import run_mujoco_archetype_simulation
    from core.mechanical_load_engine import compute_loading_from_simulation, compute_loading_from_curves
    from core.stress_strain_engine import compute_stress_strain
    from core.rehab_risk_engine import compute_rehab_risk

    # --- MuJoCo / proxy simulation ---
    if enable_mujoco:
        with st.spinner("Running MuJoCo simulation (or proxy)..."):
            sim_result = run_mujoco_archetype_simulation(
                archetype_key, severity, side_involvement,
                duration_sec=5.5, timestep=0.01,
            )
    else:
        sim_result = run_mujoco_archetype_simulation(
            archetype_key, severity, side_involvement,
            duration_sec=5.5, timestep=0.01,
        )

    sim_message = sim_result.get("message", "")

    # --- Mechanical loading ---
    loading_df = compute_loading_from_simulation(sim_result)
    if loading_df.empty:
        loading_df = compute_loading_from_curves(archetype_key, severity, side_involvement)

    # --- Stress-strain ---
    ss_df = pd.DataFrame()
    if enable_ss:
        ss_df = compute_stress_strain(
            archetype_key, severity, side_involvement, sim_result=sim_result)

    # --- Rehab risk ---
    risk_result = compute_rehab_risk(
        archetype_key, severity, side_involvement,
        loading_df=loading_df if not loading_df.empty else None,
        stress_strain_df=ss_df if not ss_df.empty else None,
    )

    return sim_result, loading_df, ss_df, risk_result, sim_message


# Run the pipeline
with st.spinner("Computing R-SC-GAM-MJ-SSL pipeline..."):
    sim_result, loading_df, ss_df, risk_result, sim_message = run_pipeline(
        archetype_key, severity, side_involvement,
        enable_mujoco, enable_ss,
    )


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Overview",
    "🦿 Archetype Model",
    "💪 Recruitment Likelihood",
    "🎯 Rehab Priority",
    "⚙️ MuJoCo Simulation",
    "📊 Stress-Strain & Loading",
    "🚨 Rehab Risk Map",
    "📄 Reports & Export",
])

tab_overview, tab_arch, tab_rec, tab_rpi, tab_mj, tab_ss, tab_risk, tab_rep = tabs

with tab_overview:
    from ui.pages_home import render_home
    render_home(archetype_key, severity, side_involvement, enable_mujoco, enable_ss)

with tab_arch:
    from ui.pages_archetype import render_archetype_tab
    render_archetype_tab(archetype_key, severity, correction_strength)

with tab_rec:
    from ui.pages_archetype import render_recruitment_tab
    render_recruitment_tab(archetype_key)

with tab_rpi:
    from ui.pages_archetype import render_rehab_priority_tab
    render_rehab_priority_tab(archetype_key)

with tab_mj:
    from ui.pages_mujoco import render_mujoco_tab
    render_mujoco_tab(sim_result, loading_df)

with tab_ss:
    from ui.pages_stress_strain import render_stress_strain_tab
    render_stress_strain_tab(ss_df, loading_df)

with tab_risk:
    from ui.pages_reports import render_risk_map_tab
    render_risk_map_tab(risk_result)

with tab_rep:
    from ui.pages_reports import render_reports_tab
    render_reports_tab(
        archetype_key, severity, side_involvement,
        loading_df, ss_df, risk_result, sim_message,
        OUT_DIR,
    )
