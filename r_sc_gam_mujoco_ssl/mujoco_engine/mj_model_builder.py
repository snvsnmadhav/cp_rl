"""
mj_model_builder.py
Build a simplified lower-limb MuJoCo XML model for CP gait archetype simulation.
"""

import textwrap
from typing import Dict


# Severity scaling: how much the archetype's distortion feeds into control limits
SEVERITY_SCALE = {"mild": 0.4, "moderate": 0.7, "severe": 1.0}


def build_simple_biped_model(
    archetype_key: str = "true_equinus",
    side_involvement: str = "bilateral",
    severity: str = "moderate",
) -> str:
    """
    Return a MuJoCo XML string for a simplified lower-limb biped.

    Joints:  left/right hip, knee, ankle
    Bodies:  pelvis, left/right thigh, shank, foot
    Actuators: position motors for all 6 joints
    Sites:   named tendon-proxy sites on each segment
    """
    sc = SEVERITY_SCALE.get(severity, 0.7)

    xml = textwrap.dedent(f"""
    <mujoco model="cp_biped_{archetype_key}">
      <compiler angle="degree" coordinate="local"/>

      <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

      <default>
        <joint limited="true" damping="1.0" stiffness="0" armature="0.02"/>
        <geom condim="3" friction="1.0 0.05 0.01"/>
      </default>

      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.8 0.9 1.0" rgb2="0.5 0.7 1.0" width="256" height="256"/>
        <texture type="2d" name="groundplane" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75" width="300" height="300"/>
        <material name="groundmat" texture="groundplane" texrepeat="5 5" reflectance="0.2"/>
        <material name="body_mat" rgba="0.7 0.5 0.4 1"/>
        <material name="joint_mat" rgba="0.3 0.5 0.8 1"/>
      </asset>

      <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="ground" type="plane" size="5 5 0.1" material="groundmat" contype="1" conaffinity="1"/>

        <!-- Pelvis / trunk -->
        <body name="pelvis" pos="0 0 1.0">
          <freejoint name="root"/>
          <geom name="pelvis_geom" type="box" size="0.12 0.08 0.06" mass="12.0" material="body_mat"/>
          <site name="pelvis_com" pos="0 0 0"/>

          <!-- === LEFT LEG === -->
          <body name="left_thigh" pos="-0.09 0 -0.06">
            <joint name="left_hip" type="hinge" axis="1 0 0" range="-30 90" damping="5.0"/>
            <geom name="left_thigh_geom" type="capsule" fromto="0 0 0  0 0 -0.38" size="0.04" mass="3.0" material="body_mat"/>
            <site name="left_rectus_femoris_tendon" pos="0.02 0 -0.1"/>
            <site name="left_hamstring_tendon"      pos="-0.02 0 -0.15"/>

            <body name="left_shank" pos="0 0 -0.40">
              <joint name="left_knee" type="hinge" axis="1 0 0" range="-5 140" damping="3.0"/>
              <geom name="left_shank_geom" type="capsule" fromto="0 0 0  0 0 -0.36" size="0.032" mass="1.5" material="body_mat"/>
              <site name="left_gastroc_tendon"           pos="-0.015 0 -0.25"/>

              <body name="left_foot" pos="0 0 -0.38">
                <joint name="left_ankle" type="hinge" axis="1 0 0" range="-50 30" damping="2.0"/>
                <geom name="left_foot_geom"  type="box" size="0.06 0.03 0.02" pos="0.04 0 -0.02" mass="0.45" material="body_mat" contype="1" conaffinity="1"/>
                <geom name="left_heel_geom"  type="sphere" size="0.02" pos="-0.02 0 -0.02" mass="0.05" material="body_mat" contype="1" conaffinity="1"/>
                <site name="left_tibialis_anterior_tendon" pos="0.02 0 -0.01"/>
              </body>
            </body>
          </body>

          <!-- === RIGHT LEG === -->
          <body name="right_thigh" pos="0.09 0 -0.06">
            <joint name="right_hip" type="hinge" axis="1 0 0" range="-30 90" damping="5.0"/>
            <geom name="right_thigh_geom" type="capsule" fromto="0 0 0  0 0 -0.38" size="0.04" mass="3.0" material="body_mat"/>
            <site name="right_rectus_femoris_tendon" pos="0.02 0 -0.1"/>
            <site name="right_hamstring_tendon"      pos="-0.02 0 -0.15"/>

            <body name="right_shank" pos="0 0 -0.40">
              <joint name="right_knee" type="hinge" axis="1 0 0" range="-5 140" damping="3.0"/>
              <geom name="right_shank_geom" type="capsule" fromto="0 0 0  0 0 -0.36" size="0.032" mass="1.5" material="body_mat"/>
              <site name="right_gastroc_tendon"           pos="-0.015 0 -0.25"/>

              <body name="right_foot" pos="0 0 -0.38">
                <joint name="right_ankle" type="hinge" axis="1 0 0" range="-50 30" damping="2.0"/>
                <geom name="right_foot_geom"  type="box" size="0.06 0.03 0.02" pos="0.04 0 -0.02" mass="0.45" material="body_mat" contype="1" conaffinity="1"/>
                <geom name="right_heel_geom"  type="sphere" size="0.02" pos="-0.02 0 -0.02" mass="0.05" material="body_mat" contype="1" conaffinity="1"/>
                <site name="right_tibialis_anterior_tendon" pos="0.02 0 -0.01"/>
              </body>
            </body>
          </body>

        </body> <!-- end pelvis -->
      </worldbody>

      <actuator>
        <position name="left_hip_motor"   joint="left_hip"   kp="200" ctrllimited="true" ctrlrange="-1.0 1.5"/>
        <position name="left_knee_motor"  joint="left_knee"  kp="150" ctrllimited="true" ctrlrange="-0.1 2.5"/>
        <position name="left_ankle_motor" joint="left_ankle" kp="100" ctrllimited="true" ctrlrange="-0.9 0.5"/>
        <position name="right_hip_motor"   joint="right_hip"   kp="200" ctrllimited="true" ctrlrange="-1.0 1.5"/>
        <position name="right_knee_motor"  joint="right_knee"  kp="150" ctrllimited="true" ctrlrange="-0.1 2.5"/>
        <position name="right_ankle_motor" joint="right_ankle" kp="100" ctrllimited="true" ctrlrange="-0.9 0.5"/>
      </actuator>

      <sensor>
        <actuatorfrc name="left_hip_force"   actuator="left_hip_motor"/>
        <actuatorfrc name="left_knee_force"  actuator="left_knee_motor"/>
        <actuatorfrc name="left_ankle_force" actuator="left_ankle_motor"/>
        <actuatorfrc name="right_hip_force"   actuator="right_hip_motor"/>
        <actuatorfrc name="right_knee_force"  actuator="right_knee_motor"/>
        <actuatorfrc name="right_ankle_force" actuator="right_ankle_motor"/>
        <jointpos name="left_hip_pos"   joint="left_hip"/>
        <jointpos name="left_knee_pos"  joint="left_knee"/>
        <jointpos name="left_ankle_pos" joint="left_ankle"/>
        <jointpos name="right_hip_pos"   joint="right_hip"/>
        <jointpos name="right_knee_pos"  joint="right_knee"/>
        <jointpos name="right_ankle_pos" joint="right_ankle"/>
        <jointvel name="left_hip_vel"   joint="left_hip"/>
        <jointvel name="left_knee_vel"  joint="left_knee"/>
        <jointvel name="left_ankle_vel" joint="left_ankle"/>
        <jointvel name="right_hip_vel"   joint="right_hip"/>
        <jointvel name="right_knee_vel"  joint="right_knee"/>
        <jointvel name="right_ankle_vel" joint="right_ankle"/>
      </sensor>

    </mujoco>
    """).strip()

    return xml
