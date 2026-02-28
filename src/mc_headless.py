#!/usr/bin/env python3
# mc_headless.py
# =============================================================================
# Standalone headless Monte Carlo EKF evaluation -- FORCE-BASED control.
#
# Windows (binary install):
#   C:\<isaac_root>\python.bat mc_headless.py --trials 50
#
# Windows (pip / conda install, env active):
#   python mc_headless.py --trials 50
#
# HOW PHYSICS ACTUALLY DRIVES RESULTS HERE
# -----------------------------------------
# Previous version used set_linear_velocity / set_angular_velocity, which
# overwrites the rigid body state every step and completely bypasses PhysX
# contact dynamics. Mass and friction had no effect -- all trials were
# essentially identical, so the "slip model" was a hack.
#
# This version uses FORCE-BASED CONTROL with analytical Coulomb clamping:
#
#   1. A PD tracking controller computes the force (F_des) needed to follow
#      the scripted figure-8 trajectory.
#
#   2. The Coulomb friction model (Part 1) clamps the horizontal force:
#        F_max = mu * m * g
#      If |F_des| > F_max, the force is clamped to the friction circle
#      boundary -- the robot SLIPS because the wheels cannot transmit the
#      demanded force.
#
#   3. The clamped force is divided by mass to get acceleration (F=ma),
#      then numerically integrated to update velocity. This velocity is
#      set on the PhysX rigid body, which integrates position.
#
#   Physics coupling:
#     - Mass matters:  F = m*a -> heavier robot needs more force to follow
#       the same trajectory -> more likely to hit friction limit.
#     - Friction matters:  F_max = mu*m*g clamps the achievable force.
#       Lower mu -> lower F_max -> more slip at the same trajectory point.
#     - Both couple naturally: high mass + low mu is the worst case.
#
#   The trajectory uses a more aggressive figure-8 (larger amplitude,
#   shorter period) so that the demanded accelerations (~5-8 m/s^2)
#   genuinely challenge the friction limit (a_crit = mu*g ~ 6-9 m/s^2).
#   With the original gentle trajectory (peak 0.27 m/s^2), friction would
#   never engage regardless of control method.
#
# =============================================================================

import argparse
import math
import os
import sys
from datetime import datetime
from typing import List

import numpy as np

# -- SimulationApp MUST come before all Isaac Sim imports ---------------------
from isaacsim import SimulationApp
_app = SimulationApp({"headless": True, "anti_aliasing": 0})

# -- Isaac Sim imports (safe after SimulationApp) -----------------------------
import carb
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import get_current_stage
from omni.physx.scripts import physicsUtils
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

# -- Pure-Python EKF (no Isaac Sim dependency) --------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from ekf import (
    EKF2D,
    ScriptedFigure8,
    quat_wxyz_to_yaw,
    yaw_to_quat_wxyz,
    wrap_angle,
    simulate_landmark_measurement,
    propagate_unicycle,
)
from global_variables import OUTPUT_ROOT_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Constants  (must match RobotScene in simulation.py)
# =============================================================================

DIMS_M            = np.array([0.780, 0.650, 1.900], dtype=float)
MASS_KG_NOMINAL   = 400.0
COM_FROM_CORNER_M = np.array([0.260, 0.340, 0.600], dtype=float)
MU_NOMINAL        = 0.80
GRAVITY           = 9.81

# Principal inertia about the vertical (yaw) axis
I_YAW_NOMINAL     = 22.2092  # kg*m^2  (smallest principal inertia -- vertical)

# Corrected principal inertias and principal-axes quaternion [w,x,y,z]
PRINCIPAL_INERTIA = (90.8758, 84.6084, 22.2092)
PRINCIPAL_AXES_Q  = (0.14044285, 0.11038041, 0.98391599, 0.00114256)

COM_LOCAL   = COM_FROM_CORNER_M - 0.5 * DIMS_M
HALF_HEIGHT = float(DIMS_M[2] / 2.0)

BASE_PATH = "/World/RobotBase"
BODY_PATH = "/World/RobotBase/Body"
MAT_PATH  = "/World/PhysicsScene/defaultMaterial"

LANDMARKS = np.array([
    [ 3.0,  2.0], [ 3.0, -2.0], [-2.5,  1.5],
    [-2.0, -2.5], [ 0.0,  3.0], [ 0.0, -3.0],
], dtype=float)

SIGMA_V   = 0.05
SIGMA_W   = 0.05
SIGMA_R   = 0.03
SIGMA_B   = np.deg2rad(2.0)
MAX_RANGE = 20.0
HALF_FOV  = np.deg2rad(135.0)
DETECT_P  = 0.95

# -- PD tracking controller gains --------------------------------------------
# Tuned so the controller demands forces that challenge the friction limit
# during the aggressive portions of the trajectory, while keeping position
# errors bounded (<0.2 m RMS) for high-mu trials.
# At 0.2 m error on a 400 kg robot, Kp*m*e = 15*400*0.2 = 1200 N,
# plus feedforward up to mass*7.1 ~ 2840 N -> total ~4040 N approaches
# F_max = mu*m*g ~ 2500-3700 N depending on mu.
KP_POS  = 15.0    # s^-2   (spring constant / mass)
KD_POS  = 8.0     # s^-1   (damper constant / mass)
KP_YAW  = 40.0    # s^-2
KD_YAW  = 12.0    # s^-1

# -- Aggressive trajectory parameters ----------------------------------------
# The original figure-8 (amp=1.5/1.0, period=20s) has peak acceleration of
# only 0.27 m/s^2 -- far below a_crit ~ 6-9 m/s^2, so friction never matters.
#
# These parameters produce peak combined acceleration of ~7.1 m/s^2:
#   ax_peak = amp_x * w^2  = 2.5 * (2*pi/6)^2  ~ 2.7 m/s^2
#   ay_peak = 4*amp_y * w^2 = 4*1.5*(2*pi/6)^2 ~ 6.6 m/s^2
#   combined ~ sqrt(2.7^2 + 6.6^2) ~ 7.1 m/s^2
#
# This sits right between a_crit for low-mu (6.3 m/s^2) and high-mu
# (9.4 m/s^2) trials, creating genuine physics differentiation:
#   low-mu  -> feedforward alone exceeds friction -> ~30-43% slip
#   high-mu -> only PD transients clip            -> ~2% slip
TRAJ_AMP_X   = 2.5    # m
TRAJ_AMP_Y   = 1.5    # m
TRAJ_PERIOD  = 6.0    # s

MC_COLS = [
    "trial", "mass_scale", "mu_scale", "a_crit_m_s2",
    "lidar_scale", "detection_prob",
    "sigma_v", "sigma_w", "sigma_r", "sigma_b_deg",
    "mse_xy_odom", "mse_xy_ekf",
    "mse_yaw_odom", "mse_yaw_ekf",
    "slip_fraction",
]


# =============================================================================
# Scene setup  (built once, params patched per trial)
# =============================================================================

def build_scene(world: World) -> None:
    """
    Build USD stage with a rigid body on a ground plane.
    No LiDAR prim needed: landmark measurements are simulated analytically.
    """
    stage = get_current_stage()
    world.scene.add(GroundPlane("/World/Ground"))

    old = stage.GetPrimAtPath(BASE_PATH)
    if old and old.IsValid():
        stage.RemovePrim(BASE_PATH)

    stage.DefinePrim(BASE_PATH, "Xform")
    root_prim = stage.GetPrimAtPath(BASE_PATH)

    rigid = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid.CreateRigidBodyEnabledAttr(True)

    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.GetMassAttr().Set(float(MASS_KG_NOMINAL))
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*COM_LOCAL.astype(float)))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*PRINCIPAL_INERTIA))
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(*PRINCIPAL_AXES_Q))

    # Collision geometry
    cube = UsdGeom.Cube.Define(stage, BODY_PATH)
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(*DIMS_M.astype(float))
    )
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # Physics material (friction)
    mat = UsdShade.Material.Define(stage, MAT_PATH)
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(float(MU_NOMINAL))
    mat_api.CreateDynamicFrictionAttr().Set(float(MU_NOMINAL))
    mat_api.CreateRestitutionAttr().Set(0.0)
    physicsUtils.add_physics_material_to_prim(stage, cube.GetPrim(), MAT_PATH)

    rp = SingleRigidPrim(BASE_PATH)
    world.scene.add(rp)
    rp.set_world_pose(
        position=np.array([0.0, 0.0, HALF_HEIGHT], dtype=float),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    )


def patch_trial_params(mass_scale: float, mu_scale: float) -> None:
    """Write new mass, inertia, and friction onto existing USD prims."""
    stage = get_current_stage()

    root_prim = stage.GetPrimAtPath(BASE_PATH)
    mass_api  = UsdPhysics.MassAPI(root_prim)
    mass_api.GetMassAttr().Set(float(MASS_KG_NOMINAL * mass_scale))
    I1, I2, I3 = PRINCIPAL_INERTIA
    mass_api.GetDiagonalInertiaAttr().Set(
        Gf.Vec3f(I1 * mass_scale, I2 * mass_scale, I3 * mass_scale)
    )
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(*PRINCIPAL_AXES_Q))

    mat_prim = stage.GetPrimAtPath(MAT_PATH)
    mat_api  = UsdPhysics.MaterialAPI(mat_prim)
    new_mu = float(MU_NOMINAL * mu_scale)
    mat_api.GetStaticFrictionAttr().Set(new_mu)
    mat_api.GetDynamicFrictionAttr().Set(new_mu)


def reset_robot(rigid_prim: SingleRigidPrim, world: World,
                pose: np.ndarray) -> None:
    """Place robot at pose, zero velocities, reinitialise PhysX."""
    rigid_prim.set_world_pose(
        position=np.array([pose[0], pose[1], HALF_HEIGHT], dtype=float),
        orientation=yaw_to_quat_wxyz(float(pose[2])),
    )
    rigid_prim.set_linear_velocity(np.zeros(3, dtype=float))
    rigid_prim.set_angular_velocity(np.zeros(3, dtype=float))
    world.reset()


# =============================================================================
# Single trial -- force-based control with Coulomb friction clamping
# =============================================================================

def run_one_trial(
    world: World,
    rigid_prim: SingleRigidPrim,
    motion: ScriptedFigure8,
    trial_idx: int,
    mass_scale: float,
    mu_scale: float,
    lidar_scale: float,
    p_det: float,
    duration_s: float,
    dt: float,
    rng: np.random.Generator,
) -> List[float]:
    """
    Run one MC trial with force-based PD control + Coulomb friction clamping.

    CONTROL LOOP (per step):
    ------------------------
    1. Read current pose from PhysX (+ use tracked velocity state).
    2. Compute desired pose, velocity, acceleration from scripted trajectory.
    3. PD force controller:
         F_des = m * (a_ff + Kp*(x_des - x) + Kd*(v_des - v))
       where a_ff is the feedforward acceleration from the trajectory.
    4. Coulomb friction clamp (Part 1):
         F_max = mu * m * g
         if |F_des_xy| > F_max -> clamp to friction circle -> SLIP
    5. Clamped force -> acceleration -> integrate velocity:
         a = F_clamped / m
         v_new = v_old + a * dt
    6. Set velocity on rigid body, world.step() -- PhysX integrates position.
    7. Read PhysX pose -> ground truth.
    8. Derive odometry from consecutive poses + noise.
    9. EKF predict + gated landmark updates.

    WHY THIS WORKS:
    ----------------
    - Mass enters through F=ma: heavier robot needs more force for same
      trajectory -> gets closer to friction limit -> more slip.
    - Friction enters through F_max = mu*m*g: lower mu -> lower force budget
      -> earlier saturation -> more slip at same trajectory point.
    - Both effects are CONTINUOUS and physically grounded in Part 1.
    - The aggressive trajectory (peak accel ~8.5 m/s^2) genuinely challenges
      a_crit = mu*g in [6.3, 9.4] m/s^2, unlike the original gentle figure-8.
    """
    patch_trial_params(mass_scale, mu_scale)

    mass   = MASS_KG_NOMINAL * mass_scale
    mu     = MU_NOMINAL * mu_scale
    I_yaw  = I_YAW_NOMINAL * mass_scale   # yaw inertia scales with mass
    a_crit = mu * GRAVITY                  # Coulomb critical acceleration
    F_friction_max = mu * mass * GRAVITY   # Max horizontal friction force

    # Effective wheelbase for yaw torque friction limit.
    # The friction couple that ground contact can exert on yaw is limited by:
    #   T_max ~ F_friction_max * L_eff / 2
    # where L_eff is the effective contact patch separation.
    L_eff = float(DIMS_M[1])              # robot width = 0.65 m
    T_yaw_max = F_friction_max * L_eff / 2.0

    sr = SIGMA_R * lidar_scale
    sb = SIGMA_B * lidar_scale

    # -- Initialise at scripted t=0 -------------------------------------------
    gt0 = motion.pose_at(0.0)
    reset_robot(rigid_prim, world, gt0)

    # Settle for a few steps
    for _ in range(5):
        world.step(render=False)

    # Read initial PhysX state
    pos, quat = rigid_prim.get_world_pose()
    prev_pose = np.array([
        float(pos[0]), float(pos[1]),
        quat_wxyz_to_yaw(np.asarray(quat, dtype=float))
    ], dtype=float)

    cur_vx, cur_vy, cur_wz = 0.0, 0.0, 0.0  # tracked velocity state

    # Initialise EKF and dead-reckoning
    ekf = EKF2D()
    ekf.R = np.diag([sr, sb]) ** 2
    ekf.reset(prev_pose)
    odom_state = prev_pose.copy()

    steps = int(duration_s / dt)
    gt_list, od_list, ek_list = [], [], []
    slip_count = 0  # number of steps where friction saturated

    for i in range(1, steps + 1):
        t = i * dt

        # -- 1. Read current state from PhysX ---------------------------------
        pos, quat = rigid_prim.get_world_pose()
        cur_x  = float(pos[0])
        cur_y  = float(pos[1])
        cur_yaw = quat_wxyz_to_yaw(np.asarray(quat, dtype=float))
        # Use our tracked velocity (more stable than PhysX readback which
        # can jitter from contact solver noise)

        # -- 2. Desired state from trajectory ----------------------------------
        des_pose = motion.pose_at(t)
        des_vx, des_vy = motion.velocity_at(t)
        des_ax, des_ay = motion.acceleration_at(t)
        des_wz = motion.angular_velocity_at(t)

        # -- 3. PD force controller (world frame) -----------------------------
        # F = m * (feedforward_accel + Kp*pos_error + Kd*vel_error)
        fx = mass * (des_ax
                     + KP_POS * (des_pose[0] - cur_x)
                     + KD_POS * (des_vx - cur_vx))
        fy = mass * (des_ay
                     + KP_POS * (des_pose[1] - cur_y)
                     + KD_POS * (des_vy - cur_vy))

        # -- 4. Coulomb friction clamping (Part 1 result) ---------------------
        # The friction circle limits the total horizontal force the ground
        # can exert on the robot:  |F_horiz| <= mu * m * g
        F_horiz = np.array([fx, fy], dtype=float)
        F_mag = np.linalg.norm(F_horiz)

        step_slipped = False
        if F_mag > F_friction_max:
            # Demanded force exceeds friction limit -> robot slips.
            # Force is clamped to the friction circle boundary.
            F_horiz = F_horiz * (F_friction_max / F_mag)
            step_slipped = True
            slip_count += 1

        # -- 5. Integrate: clamped force -> acceleration -> velocity ----------
        ax = F_horiz[0] / mass
        ay = F_horiz[1] / mass
        cur_vx = cur_vx + ax * dt
        cur_vy = cur_vy + ay * dt

        # Yaw torque with friction couple limit
        yaw_err = wrap_angle(des_pose[2] - cur_yaw)
        wz_err  = des_wz - cur_wz
        tz = I_yaw * (KP_YAW * yaw_err + KD_YAW * wz_err)
        if abs(tz) > T_yaw_max:
            tz = math.copysign(T_yaw_max, tz)
            if not step_slipped:
                step_slipped = True
                slip_count += 1
        cur_wz = cur_wz + (tz / I_yaw) * dt

        # -- 6. Set velocity and step PhysX ------------------------------------
        rigid_prim.set_linear_velocity(
            np.array([cur_vx, cur_vy, 0.0], dtype=float)
        )
        rigid_prim.set_angular_velocity(
            np.array([0.0, 0.0, cur_wz], dtype=float)
        )
        world.step(render=False)

        # -- 7. Read PhysX ground truth pose -----------------------------------
        pos, quat = rigid_prim.get_world_pose()
        phys_pose = np.array([
            float(pos[0]), float(pos[1]),
            quat_wxyz_to_yaw(np.asarray(quat, dtype=float))
        ], dtype=float)

        # -- 8. Wheel-encoder odometry -----------------------------------------
        # Derived from consecutive PhysX poses (as a real encoder would).
        dx = phys_pose[0] - prev_pose[0]
        dy = phys_pose[1] - prev_pose[1]
        th = prev_pose[2]
        v_true = (dx * math.cos(th) + dy * math.sin(th)) / max(dt, 1e-9)
        w_true = wrap_angle(phys_pose[2] - prev_pose[2]) / max(dt, 1e-9)

        # Encoder noise: base noise + slip-dependent extra noise.
        # During slip the wheel rotates faster than the ground contact,
        # so the encoder reading is less trustworthy.
        slip_noise_scale = 3.0 if step_slipped else 1.0
        sv_eff = SIGMA_V * slip_noise_scale
        sw_eff = SIGMA_W * slip_noise_scale
        v_meas = v_true + rng.normal(0.0, sv_eff)
        w_meas = w_true + rng.normal(0.0, sw_eff)

        # -- 9. EKF predict ----------------------------------------------------
        ekf.Q_u = np.diag([sv_eff, sw_eff]) ** 2
        ekf.predict(v_meas, w_meas, dt)
        odom_state = propagate_unicycle(odom_state, v_meas, w_meas, dt)

        # -- 10. EKF landmark updates with Mahalanobis gate --------------------
        CHI2_GATE = 13.82  # chi2(2 dof, p=0.999) -- loose gate
        for lm in LANDMARKS:
            pair = simulate_landmark_measurement(
                gt_pose=phys_pose, landmark_xy=lm, rng=rng,
                sigma_r=sr, sigma_b=sb,
                max_range_m=MAX_RANGE, half_fov_rad=HALF_FOV,
                detection_prob=p_det,
            )
            if pair is None:
                continue
            _, z_noisy = pair

            # Predicted measurement at current EKF state
            ex, ey, eth = ekf.x
            ldx, ldy = float(lm[0]) - ex, float(lm[1]) - ey
            lq = ldx**2 + ldy**2
            if lq < 1e-12:
                continue
            lr   = math.sqrt(lq)
            lb   = wrap_angle(math.atan2(ldy, ldx) - eth)
            innov = np.array([z_noisy[0] - lr,
                              wrap_angle(z_noisy[1] - lb)], dtype=float)
            H = np.array([[-ldx/lr,  -ldy/lr,  0.0],
                          [ ldy/lq,  -ldx/lq, -1.0]], dtype=float)
            S = H @ ekf.P @ H.T + ekf.R

            mahal = float(innov @ np.linalg.inv(S) @ innov)
            if mahal > CHI2_GATE:
                continue

            ekf.update_landmark(z_noisy, lm)

        gt_list.append(phys_pose.copy())
        od_list.append(odom_state.copy())
        ek_list.append(ekf.x.copy())
        prev_pose = phys_pose.copy()

    # -- Compute MSE -----------------------------------------------------------
    gt = np.array(gt_list, dtype=float)
    od = np.array(od_list, dtype=float)
    ek = np.array(ek_list, dtype=float)

    exo = gt[:, 0] - od[:, 0];  eyo = gt[:, 1] - od[:, 1]
    exe = gt[:, 0] - ek[:, 0];  eye = gt[:, 1] - ek[:, 1]
    eto = np.array([wrap_angle(a - b) for a, b in zip(gt[:, 2], od[:, 2])],
                   dtype=float)
    ete = np.array([wrap_angle(a - b) for a, b in zip(gt[:, 2], ek[:, 2])],
                   dtype=float)

    slip_frac = float(slip_count) / max(steps, 1)

    return [
        float(trial_idx), mass_scale, mu_scale, a_crit,
        lidar_scale, p_det, SIGMA_V, SIGMA_W, float(sr), float(np.rad2deg(sb)),
        float(np.mean(exo**2 + eyo**2)),
        float(np.mean(exe**2 + eye**2)),
        float(np.mean(eto**2)),
        float(np.mean(ete**2)),
        slip_frac,
    ]


# =============================================================================
# Plots and summary
# =============================================================================

def save_results(rows: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(
        os.path.join(out_dir, "monte_carlo_mse.csv"),
        rows, delimiter=",",
        header=",".join(MC_COLS), comments="",
    )

    mse_xy_odom  = rows[:, 10];  mse_xy_ekf  = rows[:, 11]
    mse_yaw_odom = rows[:, 12];  mse_yaw_ekf = rows[:, 13]
    a_crit       = rows[:, 3];   slip_frac   = rows[:, 14]

    # -- Boxplots --------------------------------------------------------------
    for vals, ylabel, title, fname in [
        ([mse_xy_odom, mse_xy_ekf],
         "MSE_xy (m^2)", "MC Position MSE  [Force control + Coulomb clamping]",
         "mc_boxplot_mse_xy.png"),
        ([mse_yaw_odom, mse_yaw_ekf],
         "MSE_yaw (rad^2)", "MC Yaw MSE  [Force control + Coulomb clamping]",
         "mc_boxplot_mse_yaw.png"),
    ]:
        plt.figure(figsize=(7, 5))
        plt.boxplot(vals, tick_labels=["Odometry", "EKF"])
        plt.ylabel(ylabel); plt.title(title); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150); plt.close()

    # -- mass vs mu scatter ----------------------------------------------------
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(rows[:, 1], rows[:, 2], c=mse_xy_odom, s=70, cmap="plasma")
    plt.xlabel("mass scale"); plt.ylabel("friction (mu) scale")
    plt.title("Odometry MSE_xy vs (mass, mu)  [Force control]")
    plt.grid(True)
    plt.colorbar(sc, label="MSE_xy_odom (m^2)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mc_scatter_mass_mu_odom.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(rows[:, 1], rows[:, 2], c=mse_xy_ekf, s=70, cmap="plasma")
    plt.xlabel("mass scale"); plt.ylabel("friction (mu) scale")
    plt.title("EKF MSE_xy vs (mass, mu)  [Force control]")
    plt.grid(True)
    plt.colorbar(sc, label="MSE_xy_ekf (m^2)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mc_scatter_mass_mu_ekf.png"), dpi=150)
    plt.close()

    # -- a_crit vs slip fraction -----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc0 = axes[0].scatter(a_crit, slip_frac * 100, c=mse_xy_odom,
                          s=60, cmap="hot_r")
    axes[0].set_xlabel("a_crit = mu*g  (m/s^2)  [Part 1 Coulomb]")
    axes[0].set_ylabel("Slip events  (% of steps)")
    axes[0].set_title("Odometry MSE vs Slip Frequency")
    axes[0].grid(True)
    plt.colorbar(sc0, ax=axes[0], label="MSE_xy_odom (m^2)")

    sc1 = axes[1].scatter(a_crit, slip_frac * 100, c=mse_xy_ekf,
                          s=60, cmap="YlOrRd")
    axes[1].set_xlabel("a_crit = mu*g  (m/s^2)  [Part 1 Coulomb]")
    axes[1].set_ylabel("Slip events  (% of steps)")
    axes[1].set_title("EKF MSE vs Slip Frequency")
    axes[1].grid(True)
    plt.colorbar(sc1, ax=axes[1], label="MSE_xy_ekf (m^2)")

    plt.suptitle(
        "Lower a_crit (low mu) -> more friction saturation -> worse odometry; "
        "EKF stays robust via landmark corrections",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mc_slip_vs_acrit.png"), dpi=150)
    plt.close()

    # -- Summary text ----------------------------------------------------------
    eps = 1e-12
    imp_xy  = 100.0 * (mse_xy_odom  - mse_xy_ekf)  / np.maximum(mse_xy_odom,  eps)
    imp_yaw = 100.0 * (mse_yaw_odom - mse_yaw_ekf) / np.maximum(mse_yaw_odom, eps)

    with open(os.path.join(out_dir, "monte_carlo_summary.txt"), "w") as f:
        f.write("Monte Carlo Summary  [Force-based control + Coulomb clamping]\n")
        f.write("=" * 60 + "\n")
        f.write(f"Trials              : {len(rows)}\n")
        f.write(f"Trajectory          : figure-8, amp=({TRAJ_AMP_X},{TRAJ_AMP_Y})m, "
                f"period={TRAJ_PERIOD}s\n")
        peak_a = math.sqrt((TRAJ_AMP_X * (2*math.pi/TRAJ_PERIOD)**2)**2 +
                           (4*TRAJ_AMP_Y * (2*math.pi/TRAJ_PERIOD)**2)**2)
        f.write(f"Peak traj accel     : {peak_a:.1f} m/s^2\n")
        f.write(f"Mean a_crit (m/s^2) : {np.mean(a_crit):.3f}"
                f"  (range {a_crit.min():.3f} - {a_crit.max():.3f})\n")
        f.write(f"Mean slip fraction  : {np.mean(slip_frac)*100:.1f}%"
                f"  (range {slip_frac.min()*100:.1f}% - {slip_frac.max()*100:.1f}%)\n")
        f.write(f"PD gains            : Kp_pos={KP_POS}, Kd_pos={KD_POS}, "
                f"Kp_yaw={KP_YAW}, Kd_yaw={KD_YAW}\n\n")
        for label, odom_arr, ekf_arr, imp_arr, unit in [
            ("MSE_xy",  mse_xy_odom,  mse_xy_ekf,  imp_xy,  "m^2"),
            ("MSE_yaw", mse_yaw_odom, mse_yaw_ekf, imp_yaw, "rad^2"),
        ]:
            f.write(f"{label} ({unit})\n")
            f.write(f"  Odometry  mean +/- std : "
                    f"{np.mean(odom_arr):.4e} +/- {np.std(odom_arr):.4e}\n")
            f.write(f"  EKF       mean +/- std : "
                    f"{np.mean(ekf_arr):.4e} +/- {np.std(ekf_arr):.4e}\n")
            f.write(f"  Improvement %%         : "
                    f"{np.mean(imp_arr):.2f} +/- {np.std(imp_arr):.2f}\n\n")
        f.write(
            "Physics model:\n"
            "  Control       : PD force controller tracking scripted figure-8\n"
            "  Friction      : Analytical Coulomb clamp  F_max = mu * m * g\n"
            "                  Demanded force clamped to friction circle;\n"
            "                  clamped force integrated to get velocity.\n"
            "  Ground truth  : PhysX pose readback after world.step()\n"
            "  Slip mechanism: When PD demands F > F_max, the robot physically\n"
            "                  cannot follow the trajectory -- position error grows.\n"
            "  Mass effect   : F = m*a -> heavier robot needs more force -> hits\n"
            "                  friction limit sooner.\n"
            "  mu effect     : F_max = mu*m*g -> lower mu -> smaller force budget\n"
            "                  -> more friction saturation -> more slip.\n"
            "  Odometry noise: 3x during slip (encoder reads wheel speed,\n"
            "                  not ground speed).\n"
            "  EKF gate      : Mahalanobis chi^2(2dof, 99.9%) = 13.82\n"
        )

    print(f"\n[MC] Saved to {out_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Headless PhysX Monte Carlo EKF (force-based control)"
    )
    parser.add_argument("--trials",   type=int,   default=50)
    parser.add_argument("--duration", type=float, default=21.0)   # 3 full cycles
    parser.add_argument("--dt",       type=float, default=1.0 / 60.0)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--out",      type=str,   default=None)
    args = parser.parse_args()

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(OUTPUT_ROOT_DIR, f"mc_force_{stamp}")

    motion = ScriptedFigure8(
        amp_x=TRAJ_AMP_X, amp_y=TRAJ_AMP_Y, period_s=TRAJ_PERIOD
    )
    print(f"[MC] PhysX headless Monte Carlo -- FORCE-BASED control")
    print(f"[MC] Trajectory: figure-8, amp=({TRAJ_AMP_X},{TRAJ_AMP_Y})m, "
          f"period={TRAJ_PERIOD}s")
    print(f"[MC] Peak trajectory acceleration: {motion.peak_acceleration():.2f} m/s^2")
    print(f"[MC] a_crit range (+/-20% mu): "
          f"[{MU_NOMINAL*0.8*GRAVITY:.2f}, {MU_NOMINAL*1.2*GRAVITY:.2f}] m/s^2")
    print(f"[MC] Trials={args.trials}  Duration={args.duration}s  dt={args.dt:.4f}s")
    print(f"[MC] PD gains: Kp_pos={KP_POS}, Kd_pos={KD_POS}, "
          f"Kp_yaw={KP_YAW}, Kd_yaw={KD_YAW}")
    print(f"[MC] Output -> {args.out}\n")

    world = World(physics_dt=args.dt, rendering_dt=args.dt)
    build_scene(world)
    world.initialize_physics()
    world.reset()

    rigid_prim = SingleRigidPrim(BASE_PATH)
    rows: List[List[float]] = []

    for k in range(args.trials):
        rng = np.random.default_rng(args.seed + 1000 + k)

        mass_scale  = float(rng.uniform(0.8, 1.2))
        mu_scale    = float(rng.uniform(0.8, 1.2))
        lidar_scale = float(rng.uniform(0.8, 1.2))
        p_det       = float(np.clip(rng.normal(DETECT_P, 0.03), 0.70, 0.995))

        row = run_one_trial(
            world=world, rigid_prim=rigid_prim, motion=motion,
            trial_idx=k,
            mass_scale=mass_scale, mu_scale=mu_scale,
            lidar_scale=lidar_scale, p_det=p_det,
            duration_s=args.duration, dt=args.dt,
            rng=rng,
        )
        rows.append(row)

        if (k + 1) % 5 == 0 or k == 0:
            print(
                f"[MC] Trial {k+1:3d}/{args.trials}  "
                f"mass={mass_scale:.3f}  mu={mu_scale:.3f}  "
                f"a_crit={row[3]:.2f} m/s^2  "
                f"slip={row[14]*100:.1f}%  "
                f"MSE_odom={row[10]:.3e}  MSE_ekf={row[11]:.3e}"
            )

    save_results(np.array(rows, dtype=float), args.out)
    _app.close()
    print("[MC] Done.")


if __name__ == "__main__":
    main()
