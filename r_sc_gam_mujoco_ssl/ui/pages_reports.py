"""
pages_reports.py — Reports + export tab
"""

import os
import io
import streamlit as st
import pandas as pd
from datetime import datetime


def render_risk_map_tab(risk_result: dict):
    st.header("Rehabilitation Mechanical Risk Map")
    st.info(
        "Composite risk score = weighted combination of R-SC-GAM RPI, "
        "joint loading index, stress/strain burden, spasticity score, and "
        "asymmetry burden. Normalised to 0–100. Higher = higher estimated "
        "mechanical load / priority."
    )

    if not risk_result:
        st.write("Risk engine not yet run.")
        return

    # Metric summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Top Overloaded Joint", risk_result.get("top_joint", "—").upper())
    col2.metric("Top Overloaded Muscle", risk_result.get("top_muscle", "—"))
    col3.metric("Top Phase of Concern", risk_result.get("top_phase", "—"))

    st.markdown("---")

    # Risk ranking chart
    scores_df = risk_result.get("scores_df", pd.DataFrame())
    if not scores_df.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cat_color = {
            "Spasticity management": "#dc2626",
            "Strengthening / motor control": "#2563eb",
            "Mixed (tone + weakness)": "#7c3aed",
            "Biomechanical / orthotic": "#059669",
            "Monitor": "#94a3b8",
        }

        fig, ax = plt.subplots(figsize=(10, 5.5))
        colors = [cat_color.get(c, "#94a3b8") for c in scores_df["intervention"]]
        ax.barh(scores_df["muscle"][::-1], scores_df["composite_risk"][::-1],
                color=colors[::-1], edgecolor="#1f2937")
        for y, v in enumerate(scores_df["composite_risk"][::-1]):
            ax.text(v + 0.5, y, f"{v:.0f}", va="center", fontsize=8)
        ax.set_xlabel("Composite Mechanical Risk Score (0–100)")
        ax.set_title("Rehabilitation Mechanical Risk Ranking", fontsize=11, weight="bold")
        handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in cat_color.values()]
        ax.legend(handles, cat_color.keys(), fontsize=7.5, loc="lower right")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Risk Score Components Table")
        display_cols = ["rank", "muscle", "full_name", "composite_risk",
                        "rpi_norm", "joint_loading", "stress_burden",
                        "strain_burden", "spasticity_score", "intervention"]
        st.dataframe(
            scores_df[display_cols].style.background_gradient(
                subset=["composite_risk"], cmap="YlOrRd"),
            use_container_width=True,
        )

    # Narrative
    st.subheader("Rehab Risk Narrative")
    st.text(risk_result.get("narrative", ""))


def render_reports_tab(
    archetype_key: str,
    severity: str,
    side_involvement: str,
    loading_df: pd.DataFrame,
    ss_df: pd.DataFrame,
    risk_result: dict,
    sim_message: str,
    out_dir_base: str,
):
    from core.report_engine import (
        extended_therapist_report,
        patient_parent_report,
        child_friendly_report,
        save_all_reports,
    )

    st.header("Reports")

    tab_t, tab_p, tab_c = st.tabs(["Therapist Report", "Patient/Parent Report", "Child Report"])

    with tab_t:
        st.subheader("Extended Therapist Report")
        report_t = extended_therapist_report(
            archetype_key, severity, side_involvement,
            loading_df if not loading_df.empty else None,
            ss_df if not ss_df.empty else None,
            risk_result if risk_result else None,
            sim_message,
        )
        st.text_area("Therapist Report", report_t, height=500)
        st.download_button(
            "Download Therapist Report (.txt)",
            data=report_t.encode("utf-8"),
            file_name=f"therapist_{archetype_key}_{severity}.txt",
            mime="text/plain",
        )

    with tab_p:
        st.subheader("Patient / Parent Report")
        report_p = patient_parent_report(
            archetype_key, severity, side_involvement, risk_result)
        st.text_area("Patient/Parent Report", report_p, height=400)
        st.download_button(
            "Download Patient/Parent Report (.txt)",
            data=report_p.encode("utf-8"),
            file_name=f"patient_{archetype_key}_{severity}.txt",
            mime="text/plain",
        )

    with tab_c:
        st.subheader("Child-Friendly Report")
        report_c = child_friendly_report(archetype_key, risk_result)
        st.text_area("Child Report", report_c, height=300)
        st.download_button(
            "Download Child Report (.txt)",
            data=report_c.encode("utf-8"),
            file_name=f"child_{archetype_key}_{severity}.txt",
            mime="text/plain",
        )

    st.markdown("---")
    st.subheader("Export All Outputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save All Reports to outputs/reports/"):
            save_all_reports(
                archetype_key, severity, side_involvement,
                os.path.join(out_dir_base, "reports"),
                loading_df if not loading_df.empty else None,
                ss_df if not ss_df.empty else None,
                risk_result, sim_message,
            )
            st.success("Reports saved.")

    with col2:
        if not loading_df.empty and st.button("Export Mechanics CSV"):
            csv_path = os.path.join(out_dir_base, "csv",
                                    f"{archetype_key}_{severity}_loading.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            loading_df.to_csv(csv_path, index=False)
            st.success(f"Saved: {csv_path}")

    with col3:
        if not ss_df.empty and st.button("Export Stress-Strain CSV"):
            csv_path = os.path.join(out_dir_base, "csv",
                                    f"{archetype_key}_{severity}_stress_strain.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            ss_df.to_csv(csv_path, index=False)
            st.success(f"Saved: {csv_path}")

    # Inline CSV downloads (no disk write required)
    st.markdown("---")
    st.subheader("Inline CSV Downloads")

    if not loading_df.empty:
        csv_buf = io.StringIO()
        loading_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Mechanics CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"mechanics_{archetype_key}_{severity}.csv",
            mime="text/csv",
        )

    if not ss_df.empty:
        csv_buf2 = io.StringIO()
        ss_df.to_csv(csv_buf2, index=False)
        st.download_button(
            "Download Stress-Strain CSV",
            data=csv_buf2.getvalue().encode("utf-8"),
            file_name=f"stress_strain_{archetype_key}_{severity}.csv",
            mime="text/csv",
        )

    if risk_result and "scores_df" in risk_result:
        csv_buf3 = io.StringIO()
        risk_result["scores_df"].to_csv(csv_buf3, index=False)
        st.download_button(
            "Download Risk Scores CSV",
            data=csv_buf3.getvalue().encode("utf-8"),
            file_name=f"risk_scores_{archetype_key}_{severity}.csv",
            mime="text/csv",
        )
