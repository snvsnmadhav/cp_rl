"""
mj_runner.py
Run MuJoCo archetype simulation or fall back to deterministic mechanical proxy.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any

from core.r_sc_gam_core import PHENOTYPES, NORMAL_CURVES, GRID, phase_values, PHASE_KEYS


SEVERITY_SCALE = {"mild": 0.4, "moderate": 0.7, "severe": 1.0}

# Approximate body segment inertia proxies (kg·m² — literature adult estimates)
INERTIA_PROXY = {"hip": 0.22, "knee": 0.065, "ankle": 0.012}
DAMPING_PROXY = {"hip": 3.0,  "knee": 1.5,   "ankle": 0.6}
STIFFNESS_PROXY = {"hip": 15.0, "knee": 8.0,  "ankle": 4.0}
MOMENT_ARM_PROXY = {"hip": 0.06, "knee": 0.04, "ankle": 0.05}
BODY_WEIGHT_PROXY = 300.0   # N (~30 kg child)
STANCE_DUTY = 0.60


def _deg2rad(arr: np.ndarray) -> np.ndarray:
    return arr * np.pi / 180.0


def _time_from_gait_pct(pct_arr: np.ndarray, duration_sec: float) -> np.ndarray:
    return pct_arr / 100.0 * duration_sec


def _build_trajectory(curve_deg: np.ndarray, t_sim: np.ndarray,
                      cycle_dur: float) -> np.ndarray:
    """Map a 0–100 % gait curve onto simulation time by repeating cycles."""
    pct = (t_sim / cycle_dur * 100.0) % 100.0
    return np.interp(pct, GRID.astype(float), curve_deg)


def _compute_mechanical_proxy(
    joint: str,
    curve_deg: np.ndarray,
    normal_curve_deg: np.ndarray,
    t: np.ndarray,
    cycle_dur: float,
) -> Dict[str, np.ndarray]:
    q_rad = _deg2rad(_build_trajectory(curve_deg, t, cycle_dur))
    q_norm_rad = _deg2rad(_build_trajectory(normal_curve_deg, t, cycle_dur))

    dq = np.gradient(q_rad, t)
    ddq = np.gradient(dq, t)
    dev = q_rad - q_norm_rad

    I = INERTIA_PROXY[joint]
    b = DAMPING_PROXY[joint]
    k = STIFFNESS_PROXY[joint]
    tau = I * ddq + b * dq + k * dev

    # Contact force proxy — stance phase only, based on body weight
    pct = (t / cycle_dur * 100.0) % 100.0
    stance_mask = pct < (STANCE_DUTY * 100)
    contact = np.where(stance_mask, BODY_WEIGHT_PROXY * (1.0 + 0.3 * np.sin(np.pi * pct / (STANCE_DUTY * 100))), 0.0)

    actuator_force = tau / MOMENT_ARM_PROXY[joint]

    return {
        "pos_rad": q_rad,
        "vel_rad_s": dq,
        "acc_rad_s2": ddq,
        "torque_Nm": tau,
        "contact_N": contact,
        "actuator_force_N": actuator_force,
        "pos_deg": np.degrees(q_rad),
    }


def run_mujoco_archetype_simulation(
    phenotype: str,
    severity: str = "moderate",
    side_involvement: str = "bilateral",
    duration_sec: float = 5.0,
    timestep: float = 0.01,
) -> Dict[str, Any]:
    """
    Attempt MuJoCo simulation; fall back to deterministic proxy if unavailable.

    Returns a dict with keys:
        used_mujoco: bool
        t: time array
        joints: {side: {joint: {pos_rad, vel, acc, torque, contact, actuator_force}}}
        tendon_length_proxy: {side: {tendon_name: array}}
        message: str
    """
    pheno = PHENOTYPES[phenotype]
    dc = pheno.distorted_curves()
    sc = SEVERITY_SCALE.get(severity, 0.7)

    # Apply severity scaling to offset
    def scaled_curve(joint_key: str) -> np.ndarray:
        base = NORMAL_CURVES[joint_key]
        cp = dc[joint_key]
        return base + (cp - base) * sc

    sides = ["left", "right"] if side_involvement == "bilateral" else [side_involvement]

    t = np.arange(0.0, duration_sec, timestep)
    cycle_dur = 1.1  # ~1 gait cycle in seconds

    result: Dict[str, Any] = {
        "used_mujoco": False,
        "t": t,
        "joints": {},
        "tendon_length_proxy": {},
        "message": "",
        "phenotype": pheno.name,
        "severity": severity,
        "side_involvement": side_involvement,
    }

    # --- Try real MuJoCo ---
    mujoco_ok = False
    try:
        import mujoco
        from mujoco_engine.mj_model_builder import build_simple_biped_model

        xml = build_simple_biped_model(phenotype, side_involvement, severity)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)

        # Set initial joint angles from CP curves at GC=0
        joint_names = {
            "left_hip": "hip", "left_knee": "knee", "left_ankle": "ankle",
            "right_hip": "hip", "right_knee": "knee", "right_ankle": "ankle",
        }
        for jname, jkey in joint_names.items():
            try:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                side = "left" if jname.startswith("left") else "right"
                if side in sides:
                    init_deg = float(scaled_curve(jkey)[0])
                else:
                    init_deg = float(NORMAL_CURVES[jkey][0])
                d.qpos[m.jnt_qposadr[jid]] = np.deg2rad(init_deg)
            except Exception:
                pass

        # Actuator name -> joint mapping
        actuator_joint_map = {
            "left_hip_motor": ("left", "hip"),
            "left_knee_motor": ("left", "knee"),
            "left_ankle_motor": ("left", "ankle"),
            "right_hip_motor": ("right", "hip"),
            "right_knee_motor": ("right", "knee"),
            "right_ankle_motor": ("right", "ankle"),
        }

        # Storage
        n_steps = int(duration_sec / m.opt.timestep)
        t_mj = np.arange(n_steps) * m.opt.timestep

        mj_data: Dict = {s: {j: {"pos_deg": [], "vel": [], "torque": [], "contact": []}
                              for j in ["hip", "knee", "ankle"]}
                         for s in ["left", "right"]}

        for step in range(n_steps):
            t_now = step * m.opt.timestep
            pct = (t_now / cycle_dur * 100.0) % 100.0

            # Set actuator targets
            for aname, (side, jkey) in actuator_joint_map.items():
                curve = scaled_curve(jkey) if side in sides else NORMAL_CURVES[jkey]
                target_deg = float(np.interp(pct, GRID.astype(float), curve))
                target_rad = np.deg2rad(target_deg)
                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                if aid >= 0:
                    d.ctrl[aid] = target_rad

            mujoco.mj_step(m, d)
            if (d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                    or not np.all(np.isfinite(d.qpos))
                    or not np.all(np.isfinite(d.qvel))):
                raise RuntimeError(f"physics became unstable at t={t_now:.3f}s")

            for s in ["left", "right"]:
                for jkey in ["hip", "knee", "ankle"]:
                    jname = f"{s}_{jkey}"
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        pos_r = d.qpos[m.jnt_qposadr[jid]]
                        vel_r = d.qvel[m.jnt_dofadr[jid]]
                        mj_data[s][jkey]["pos_deg"].append(np.degrees(pos_r))
                        mj_data[s][jkey]["vel"].append(np.degrees(vel_r))

        # Convert to arrays + compute derived
        for s in ["left", "right"]:
            result["joints"][s] = {}
            for jkey in ["hip", "knee", "ankle"]:
                pos = np.array(mj_data[s][jkey]["pos_deg"])
                vel = np.array(mj_data[s][jkey]["vel"])
                acc = np.gradient(np.deg2rad(vel), m.opt.timestep)
                I = INERTIA_PROXY[jkey]
                b = DAMPING_PROXY[jkey]
                tau = I * acc + b * np.deg2rad(vel)
                pct_arr = (t_mj / cycle_dur * 100.0) % 100.0
                stance_mask = pct_arr < (STANCE_DUTY * 100)
                contact = np.where(stance_mask, BODY_WEIGHT_PROXY * 0.9, 0.0)
                result["joints"][s][jkey] = {
                    "pos_rad": np.deg2rad(pos),
                    "pos_deg": pos,
                    "vel_rad_s": np.deg2rad(vel),
                    "acc_rad_s2": acc,
                    "torque_Nm": tau,
                    "contact_N": contact,
                    "actuator_force_N": tau / MOMENT_ARM_PROXY[jkey],
                }
            result["tendon_length_proxy"][s] = _tendon_length_proxy(
                result["joints"][s], t_mj)

        result["t"] = t_mj
        result["used_mujoco"] = True
        result["message"] = "MuJoCo simulation completed successfully."
        mujoco_ok = True

    except Exception as exc:
        result["message"] = (
            f"MuJoCo simulation unavailable or failed ({exc}); "
            "showing deterministic mechanical proxy outputs."
        )

    # --- Fallback proxy ---
    if not mujoco_ok:
        for s in sides:
            result["joints"][s] = {}
            for jkey in ["hip", "knee", "ankle"]:
                curve = scaled_curve(jkey)
                proxy = _compute_mechanical_proxy(
                    jkey, curve, NORMAL_CURVES[jkey], t, cycle_dur)
                result["joints"][s][jkey] = proxy
            result["tendon_length_proxy"][s] = _tendon_length_proxy(
                result["joints"][s], t)

        # Fill the "other" side with normal if unilateral
        all_sides = ["left", "right"]
        for s in all_sides:
            if s not in result["joints"]:
                result["joints"][s] = {}
                for jkey in ["hip", "knee", "ankle"]:
                    proxy = _compute_mechanical_proxy(
                        jkey, NORMAL_CURVES[jkey], NORMAL_CURVES[jkey], t, cycle_dur)
                    result["joints"][s][jkey] = proxy
                result["tendon_length_proxy"][s] = _tendon_length_proxy(
                    result["joints"][s], t)

    return result


# --- muscle-tendon length proxies ----------------------------------------

_TENDON_PARAMS = {
    "gastroc": {
        "joints": [("knee", -1, 0.042), ("ankle", +1, 0.050)],
        "rest_len": 0.45,
    },
    "rectus_femoris": {
        "joints": [("hip", +1, 0.055), ("knee", -1, 0.045)],
        "rest_len": 0.38,
    },
    "hamstring": {
        "joints": [("hip", -1, 0.060), ("knee", +1, 0.048)],
        "rest_len": 0.40,
    },
    "tibialis_anterior": {
        "joints": [("ankle", -1, 0.040)],
        "rest_len": 0.27,
    },
}


def _tendon_length_proxy(
    joint_data: Dict[str, Dict],
    t: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Estimate muscle-tendon unit length as:
        L(t) = L_rest + Σ  sign_j * moment_arm_j * q_j(t)
    Returns dict tendon_name -> length_array(m).
    """
    out = {}
    n = len(t)
    for tname, params in _TENDON_PARAMS.items():
        L = np.full(n, params["rest_len"])
        for jkey, sign, ma in params["joints"]:
            if jkey in joint_data:
                q = joint_data[jkey].get("pos_rad", np.zeros(n))
                if len(q) == n:
                    L = L + sign * ma * q
        out[tname] = L
    return out
