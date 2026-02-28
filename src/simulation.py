# simulation.py
# ─────────────────────────────────────────────────────────────────────────────
# Isaac Sim scene setup (robot + LiDAR) and EKF scenario in one place.
# Depends only on ekf.py (pure Python) and analysis.py (analysis tools).
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Tuple

import carb
import numpy as np
import omni.kit.commands
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from omni.physx.scripts import physicsUtils
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .ekf import (
    EKF2D,
    ScriptedFigure8,
    propagate_unicycle,
    quat_wxyz_to_yaw,
    simulate_landmark_measurement,
    rb_to_xy_robot_frame,
    wrap_angle,
    save_csv,
    save_trajectory_plot,
    save_error_plots,
    save_measurement_plot,
    save_landmark_noise_snapshot,
)
from .analysis import run_mc_trials, save_mc_results, save_variance_sweep
from .global_variables import OUTPUT_ROOT_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 – Robot scene (physical body + LiDAR)
# ═══════════════════════════════════════════════════════════════════════════════

class RobotScene:
    """
    Creates and manages the physical robot and its LiDAR in the USD stage.

    Robot spec: 780 mm × 650 mm × 1900 mm cuboid, 400 kg,
                CoM at [260, 340, 600] mm from one corner.
    """

    # Physical constants
    DIMS_M  = np.array([0.78, 0.65, 1.90], dtype=float)   # length × width × height
    MASS_KG = 400.0
    COM_FROM_CORNER_M = np.array([0.26, 0.34, 0.60], dtype=float)

    # USD prim paths
    BASE_PATH = "/World/RobotBase"
    BODY_PATH = "/World/RobotBase/Body"
    LIDAR_PATH = "/World/RobotBase/Lidar"

    def __init__(self):
        self._half_height = float(self.DIMS_M[2] / 2.0)
        # CoM in cuboid-centred local frame
        self._com_local = self.COM_FROM_CORNER_M - 0.5 * self.DIMS_M

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> Tuple:
        """
        Build ground plane + robot in the current stage.
        Returns objects that should be registered with World.scene.
        """
        ground = GroundPlane("/World/Ground")
        self._build_robot()
        self._build_lidar()
        return ground, SingleXFormPrim(self.BASE_PATH)

    def setup(self) -> None:
        """Call after World/physics initialisation."""
        set_camera_view(
            eye=[4.0, 3.0, 2.0],
            target=[0.0, 0.0, 1.0],
            camera_prim_path="/OmniverseKit_Persp",
        )
        self._snap_to_ground()
        self._apply_physics_material()

    def reset(self) -> None:
        self._snap_to_ground()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _snap_to_ground(self) -> None:
        base = SingleXFormPrim(self.BASE_PATH)
        try:
            _, quat = base.get_world_pose()
            quat = np.asarray(quat, dtype=float)
        except Exception:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        base.set_world_pose(
            position=np.array([0.0, 0.0, self._half_height], dtype=float),
            orientation=quat,
        )

    def _build_robot(self) -> None:
        stage = get_current_stage()

        old = stage.GetPrimAtPath(self.BASE_PATH)
        if old and old.IsValid():
            stage.RemovePrim(self.BASE_PATH)

        # Root rigid body (unscaled — scale must stay on a child, not the rigid root)
        stage.DefinePrim(self.BASE_PATH, "Xform")
        root_prim = stage.GetPrimAtPath(self.BASE_PATH)
        self._snap_to_ground()

        # Physics
        rigid = UsdPhysics.RigidBodyAPI.Apply(root_prim)
        rigid.CreateRigidBodyEnabledAttr(True)

        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.GetMassAttr().Set(float(self.MASS_KG))
        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*self._com_local.astype(float)))

        # Principal inertias and principal axes pre-computed from I_CoM.
        # Derivation:
        #   1. I_centroid from uniform cuboid formula (declared assumption —
        #      true internal distribution unknown, uniform is a lower bound).
        #   2. Steiner correction to shift reference point to actual CoM:
        #      I_CoM = I_centroid - m*(|d|^2*I3 - d⊗d),
        #      d = CoM_corner - geometric_centroid = (-0.130, +0.015, -0.350) m
        #   3. Eigendecomposition of I_CoM gives principal inertias and axes.
        #   4. Eigenvector rotation matrix converted to quaternion [w,x,y,z].
        #      det(V) verified = +1.0 (proper rotation, no reflection).
        #      Reconstruction error: max|V*diag(Ip)*V' - I_CoM| < 1e-13.
        #
        # Previously this block used raw centroidal values (missing Steiner
        # correction) and identity principal-axes quaternion (wrong orientation).
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(90.8758, 84.6084, 22.2092))
        mass_api.GetPrincipalAxesAttr().Set(
            Gf.Quatf(0.14044285, 0.11038041, 0.98391599, 0.00114256)
        )

        # Visual / collision child cube (scaled)
        cube_geom = UsdGeom.Cube.Define(stage, self.BODY_PATH)
        cube_geom.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube_geom.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*self.DIMS_M.astype(float)))
        UsdPhysics.CollisionAPI.Apply(cube_geom.GetPrim())

    def _build_lidar(self) -> None:
        stage = get_current_stage()
        old = stage.GetPrimAtPath(self.LIDAR_PATH)
        if old and old.IsValid():
            stage.RemovePrim(self.LIDAR_PATH)

        _, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="Lidar",
            parent=self.BASE_PATH,
            min_range=0.1,
            max_range=10.0,
            draw_points=True,
            draw_lines=True,
            horizontal_fov=270.0,
            vertical_fov=0.0,
            horizontal_resolution=0.5,
            vertical_resolution=1.0,
            rotation_rate=10.0,
            high_lod=True,
            yaw_offset=0.0,
        )

        # Mount at top-centre of body
        lidar_prim = lidar.GetPrim()
        translate_attr = lidar_prim.GetAttribute("xformOp:translate")
        top_z = float(self.DIMS_M[2] / 2.0)
        if translate_attr and translate_attr.IsValid():
            translate_attr.Set(Gf.Vec3d(0.0, 0.0, top_z))
        else:
            UsdGeom.Xformable(lidar_prim).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, top_z))

    def _apply_physics_material(self) -> None:
        stage = get_current_stage()
        world = World.instance()
        if world is None:
            return
        phys_ctx = world.get_physics_context()
        if phys_ctx is None:
            return
        scene_prim = phys_ctx.get_current_physics_scene_prim()
        if not scene_prim or not scene_prim.IsValid():
            return

        mat_path = str(scene_prim.GetPath()) + "/defaultMaterial"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            mat = UsdShade.Material.Define(stage, mat_path)
            mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            mat_api.CreateStaticFrictionAttr().Set(0.9)
            mat_api.CreateDynamicFrictionAttr().Set(0.8)
            mat_api.CreateRestitutionAttr().Set(0.0)

        cube_prim = stage.GetPrimAtPath(self.BODY_PATH)
        if cube_prim and cube_prim.IsValid():
            physicsUtils.add_physics_material_to_prim(stage, cube_prim, mat_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 – EKF Scenario (EKF + landmark tracking + logging + export)
# ═══════════════════════════════════════════════════════════════════════════════

class EkfScenario:
    """
    Runs the EKF in lock-step with an Isaac Sim physics tick.

    Consumes the robot pose from RobotScene's xform prim, generates
    simulated landmark observations, fuses them with odometry via EKF2D,
    and accumulates logs for later export.
    """

    def __init__(self, robot_base_prim_path: str = RobotScene.BASE_PATH):
        self._robot_base_prim_path = robot_base_prim_path
        self._robot_xform: Optional[SingleXFormPrim] = None

        # Robot geometry (kept in sync with RobotScene)
        self.robot_height_m = float(RobotScene.DIMS_M[2])
        self._robot_half_height = self.robot_height_m / 2.0

        # Known landmark positions [x, y] in world frame
        self.landmarks: List[np.ndarray] = [
            np.array([ 3.0,  2.0], dtype=float),
            np.array([ 3.0, -2.0], dtype=float),
            np.array([-2.5,  1.5], dtype=float),
            np.array([-2.0, -2.5], dtype=float),
            np.array([ 0.0,  3.0], dtype=float),
            np.array([ 0.0, -3.0], dtype=float),
        ]

        # Noise parameters
        self.sigma_v = 0.05   # odometry linear velocity
        self.sigma_w = 0.05   # odometry angular velocity
        self.sigma_r = 0.03   # landmark range
        self.sigma_b = np.deg2rad(2.0)  # landmark bearing

        # LiDAR-like gating
        self.lidar_max_range_m    = 20.0
        self.lidar_half_fov_rad   = np.deg2rad(135.0)
        self.landmark_detect_prob = 0.95

        # Core objects
        self._ekf  = EKF2D()
        self.motion = ScriptedFigure8()
        self.enable_scripted_motion = True

        self._seed = 42
        self._rng: Optional[np.random.Generator] = None

        # Runtime state
        self._t = 0.0
        self._step_count = 0
        self._prev_gt_pose: Optional[np.ndarray] = None
        self._odom_state   = np.zeros(3, dtype=float)

        # Logs
        self.log_t:        List[float]        = []
        self.log_gt:       List[np.ndarray]   = []
        self.log_odom:     List[np.ndarray]   = []
        self.log_ekf:      List[np.ndarray]   = []
        self.log_num_meas: List[int]          = []

        # Last snapshot for noise visualisation
        self._snap_clean: Optional[np.ndarray] = None
        self._snap_noisy: Optional[np.ndarray] = None
        self._snap_time:  float                = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        stage = get_current_stage()
        if stage is None:
            raise RuntimeError("No active USD stage.")

        prim = stage.GetPrimAtPath(self._robot_base_prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Robot prim not found: {self._robot_base_prim_path}")

        self._robot_xform = SingleXFormPrim(self._robot_base_prim_path)
        self._rng = np.random.default_rng(self._seed)

        # Ensure robot is at the right z
        pos, quat = self._robot_xform.get_world_pose()
        pos = np.asarray(pos, dtype=float).reshape(3)
        quat = np.asarray(quat, dtype=float).reshape(4)
        self._robot_xform.set_world_pose(
            position=np.array([pos[0], pos[1], self._robot_half_height], dtype=float),
            orientation=quat,
        )

        self._create_landmark_visuals()

        # Sync EKF noise matrices to the scenario noise parameters.
        self._ekf.R   = np.diag([self.sigma_r, self.sigma_b]) ** 2
        self._ekf.Q_u = np.diag([self.sigma_v, self.sigma_w]) ** 2

        gt = self._gt_pose()
        self._ekf.reset(gt)
        self._odom_state  = gt.copy()
        self._prev_gt_pose = gt.copy()

        self._t = 0.0
        self._step_count = 0
        self.log_t.clear(); self.log_gt.clear()
        self.log_odom.clear(); self.log_ekf.clear(); self.log_num_meas.clear()
        self._snap_clean = None
        self._snap_noisy = None
        self._snap_time  = 0.0

        carb.log_info("[EKF] Setup complete.")

    def reset(self) -> None:
        self.setup()

    def update(self, step: float) -> bool:
        if self._robot_xform is None:
            return False
        dt = float(step)
        if dt <= 0.0:
            return False

        self._t += dt
        self._step_count += 1

        if self.enable_scripted_motion:
            self.motion.apply_to_xform(self._robot_xform, self._t, self._robot_half_height)

        gt = self._gt_pose()
        if self._prev_gt_pose is None:
            self._prev_gt_pose = gt.copy()
            return False

        v_true, w_true = self._compute_odometry(self._prev_gt_pose, gt, dt)
        v_meas = v_true + self._rng.normal(0.0, self.sigma_v)
        w_meas = w_true + self._rng.normal(0.0, self.sigma_w)

        self._ekf.predict(v_meas, w_meas, dt)
        self._odom_state = propagate_unicycle(self._odom_state, v_meas, w_meas, dt)

        # Landmark updates
        clean_xy, noisy_xy, num_meas = [], [], 0
        for lm in self.landmarks:
            pair = simulate_landmark_measurement(
                gt_pose=gt, landmark_xy=lm, rng=self._rng,
                sigma_r=self.sigma_r, sigma_b=self.sigma_b,
                max_range_m=self.lidar_max_range_m,
                half_fov_rad=self.lidar_half_fov_rad,
                detection_prob=self.landmark_detect_prob,
            )
            if pair is None:
                continue
            z_clean, z_noisy = pair
            self._ekf.update_landmark(z_noisy, lm)
            num_meas += 1
            clean_xy.append(rb_to_xy_robot_frame(z_clean))
            noisy_xy.append(rb_to_xy_robot_frame(z_noisy))

        if clean_xy:
            self._snap_clean = np.asarray(clean_xy, dtype=float)
            self._snap_noisy = np.asarray(noisy_xy, dtype=float)
            self._snap_time  = self._t

        self.log_t.append(self._t)
        self.log_gt.append(gt.copy())
        self.log_odom.append(self._odom_state.copy())
        self.log_ekf.append(self._ekf.x.copy())
        self.log_num_meas.append(num_meas)
        self._prev_gt_pose = gt.copy()

        if self._step_count % 120 == 0:
            rxy, ryaw = self.running_rmse()
            carb.log_info(
                f"[EKF] t={self._t:6.2f}s  meas={num_meas}  "
                f"RMSE_xy={rxy:.3f}m  RMSE_yaw={np.rad2deg(ryaw):.2f}°"
            )

        return False

    # ── Outputs ───────────────────────────────────────────────────────────────

    def get_logs(self) -> Optional[dict]:
        if not self.log_t:
            return None
        return {
            "t":        np.asarray(self.log_t,        dtype=float),
            "gt":       np.asarray(self.log_gt,        dtype=float),
            "odom":     np.asarray(self.log_odom,      dtype=float),
            "ekf":      np.asarray(self.log_ekf,       dtype=float),
            "num_meas": np.asarray(self.log_num_meas,  dtype=int),
        }

    def running_rmse(self) -> Tuple[float, float]:
        logs = self.get_logs()
        if logs is None:
            return 0.0, 0.0
        gt, ek = logs["gt"], logs["ekf"]
        e_xy  = gt[:, :2] - ek[:, :2]
        rmse_xy  = float(np.sqrt(np.mean(np.sum(e_xy * e_xy, axis=1))))
        e_yaw = np.array([wrap_angle(a - b) for a, b in zip(gt[:, 2], ek[:, 2])], dtype=float)
        rmse_yaw = float(np.sqrt(np.mean(e_yaw ** 2)))
        return rmse_xy, rmse_yaw

    def save_run_artifacts(self, out_dir: Optional[str] = None) -> Optional[str]:
        logs = self.get_logs()
        if logs is None:
            carb.log_warn("[EKF] No logs to save.")
            return None

        if out_dir is None:
            stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(OUTPUT_ROOT_DIR, f"run_{stamp}")
        os.makedirs(out_dir, exist_ok=True)

        lms_xy = np.asarray(self.landmarks, dtype=float)
        save_csv(logs, out_dir)
        save_trajectory_plot(logs, lms_xy, out_dir)
        save_error_plots(logs, out_dir)
        save_measurement_plot(logs, out_dir)
        save_landmark_noise_snapshot(self._snap_clean, self._snap_noisy, self._snap_time, out_dir)

        try:
            result = save_variance_sweep(
                RobotScene.LIDAR_PATH,
                sigmas_m=(0.0, 0.01, 0.03, 0.05),
                out_dir=out_dir,
                seed=self._seed,
            )
            if result is None:
                carb.log_warn(
                    "[LiDAR] Point cloud empty. Let sim run 1–2 s before stopping."
                )
        except Exception as exc:
            carb.log_warn(f"[LiDAR] Variance sweep failed: {exc}")

        rmse_xy, rmse_yaw = self.running_rmse()
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("EKF Run Summary\n================\n")
            f.write(f"Samples      : {len(logs['t'])}\n")
            f.write(f"RMSE_xy (m)  : {rmse_xy:.6f}\n")
            f.write(f"RMSE_yaw (°) : {np.rad2deg(rmse_yaw):.6f}\n\n")
            f.write("Noise settings\n")
            f.write(f"  sigma_v   : {self.sigma_v}\n")
            f.write(f"  sigma_w   : {self.sigma_w}\n")
            f.write(f"  sigma_r   : {self.sigma_r}\n")
            f.write(f"  sigma_b   : {np.rad2deg(self.sigma_b):.2f} deg\n")

        carb.log_info(f"[EKF] Artifacts saved → {out_dir}")
        return out_dir

    def run_monte_carlo(
        self,
        num_trials: int = 50,
        duration_s: float = 20.0,
        dt: float = 1.0 / 60.0,
        out_dir: Optional[str] = None,
    ) -> Optional[str]:
        rows = run_mc_trials(
            pose_fn=self.motion.pose_at,
            odom_fn=self._compute_odometry,
            landmarks_xy=np.asarray(self.landmarks, dtype=float),
            num_trials=num_trials,
            duration_s=duration_s,
            dt=dt,
            base_seed=self._seed,
            sigma_v=self.sigma_v,
            sigma_w=self.sigma_w,
            sigma_r=self.sigma_r,
            sigma_b=self.sigma_b,
            max_range_m=self.lidar_max_range_m,
            half_fov_rad=self.lidar_half_fov_rad,
            base_detect_prob=self.landmark_detect_prob,
        )
        out = save_mc_results(rows, out_dir=out_dir)
        carb.log_info(f"[EKF-MC] Saved Monte Carlo → {out}")
        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _gt_pose(self) -> np.ndarray:
        pos, quat = self._robot_xform.get_world_pose()
        pos  = np.asarray(pos,  dtype=float).reshape(3)
        quat = np.asarray(quat, dtype=float).reshape(4)
        return np.array([pos[0], pos[1], quat_wxyz_to_yaw(quat)], dtype=float)

    @staticmethod
    def _compute_odometry(
        prev: np.ndarray, curr: np.ndarray, dt: float
    ) -> Tuple[float, float]:
        x0, y0, th0 = prev
        x1, y1, th1 = curr
        v = ((x1 - x0) * np.cos(th0) + (y1 - y0) * np.sin(th0)) / max(dt, 1e-9)
        w = wrap_angle(th1 - th0) / max(dt, 1e-9)
        return float(v), float(w)

    def _create_landmark_visuals(self) -> None:
        stage = get_current_stage()
        if stage is None:
            return

        root = "/World/Landmarks"
        old = stage.GetPrimAtPath(root)
        if old and old.IsValid():
            stage.RemovePrim(root)
        stage.DefinePrim(root, "Xform")

        for i, lm in enumerate(self.landmarks):
            cyl = UsdGeom.Cylinder.Define(stage, f"{root}/L{i}")
            cyl.CreateRadiusAttr(0.08)
            cyl.CreateHeightAttr(2.0)
            UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3f(float(lm[0]), float(lm[1]), 1.0)
            )
