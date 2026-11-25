"""
pages_mujoco.py — MuJoCo simulation tab
"""

import streamlit as st
import pandas as pd
import numpy as np


def render_mujoco_tab(sim_result: dict, phase_df: pd.DataFrame):
    st.header("MuJoCo Mechanical Simulation")

    mode = "MuJoCo Physics" if sim_result.get("used_mujoco") else "Deterministic Mechanical Proxy"
    if sim_result.get("used_mujoco"):
        st.success(f"Simulation mode: {mode}")
    else:
        st.warning(
            f"Simulation mode: **{mode}**  \n"
            f"{sim_result.get('message', '')}"
        )

    st.info(
        "Outputs are **simulation-based estimates**: joint torque proxy, "
        "contact force proxy, and estimated tendon/muscle lengths. "
        "They are NOT measured forces or confirmed biomechanics."
    )

    # Joint trajectory plots
    st.subheader("Simulated Joint Angle Trajectories")
    from mujoco_engine.mj_visualizer import plot_joint_trajectories
    fig_traj = plot_joint_trajectories(sim_result)
    st.pyplot(fig_traj)
    import matplotlib.pyplot as plt
    plt.close(fig_traj)

    # Torque heatmap
    st.subheader("Joint Torque Proxy — Phase Heatmap")
    from mujoco_engine.mj_visualizer import plot_torque_heatmap
    fig_tau = plot_torque_heatmap(phase_df)
    st.pyplot(fig_tau)
    plt.close(fig_tau)

    # Loading index heatmap
    st.subheader("Joint Loading Index — Phase Heatmap")
    from mujoco_engine.mj_visualizer import plot_loading_index_heatmap
    fig_jli = plot_loading_index_heatmap(phase_df)
    st.pyplot(fig_jli)
    plt.close(fig_jli)

    # Tendon length plots
    st.subheader("Estimated Muscle-Tendon Length Proxies")
    from mujoco_engine.mj_visualizer import plot_tendon_lengths
    fig_ten = plot_tendon_lengths(sim_result)
    st.pyplot(fig_ten)
    plt.close(fig_ten)

    # Phase dataframe
    st.subheader("Mechanical Data Table")
    if not phase_df.empty:
        st.dataframe(
            phase_df.style.background_gradient(
                subset=["joint_loading_index"], cmap="YlOrRd"),
            use_container_width=True,
        )
    else:
        st.write("No mechanical data available.")

    # Interpretation bullets
    if not phase_df.empty:
        st.subheader("Key Interpretations")
        high = phase_df[phase_df["joint_loading_index"] >= 0.40].sort_values(
            "joint_loading_index", ascending=False).head(5)
        for _, r in high.iterrows():
            st.write(f"**{r['side']} {r['joint']} — {r['phase']}** "
                     f"(JLI={r['joint_loading_index']:.2f}): {r['interpretation']}")
