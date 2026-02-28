# ekf.py
# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python EKF module: math, motion model, measurement model, and plots.
# No Isaac Sim / Omniverse dependencies.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 – Angle & Quaternion Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """Return yaw-only quaternion as [w, x, y, z]."""
    h = 0.5 * yaw
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)], dtype=float)


def quat_wxyz_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from quaternion [w, x, y, z]."""
    w, x, y, z = np.asarray(q, dtype=float).reshape(4)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 – EKF2D
# ═══════════════════════════════════════════════════════════════════════════════

class EKF2D:
    """
    Planar EKF with unicycle motion model.

    State   : x = [px, py, yaw]
    Input   : u = [v, w]   (linear vel, angular vel)
    Meas.   : z = [range, bearing]  to a known landmark
    """

    def __init__(self):
        self.x = np.zeros(3, dtype=float)
        self.P = np.diag([0.05, 0.05, np.deg2rad(5.0)]) ** 2

        # Process noise on [v, w]
        self.Q_u = np.diag([0.02, 0.02]) ** 2

        # Measurement noise on [range, bearing]
        self.R = np.diag([0.02, np.deg2rad(1.0)]) ** 2

    def reset(self, x0: np.ndarray) -> None:
        self.x = np.array(x0, dtype=float).reshape(3)
        self.P = np.diag([0.05, 0.05, np.deg2rad(5.0)]) ** 2

    def predict(self, v_meas: float, w_meas: float, dt: float) -> None:
        """EKF predict step (unicycle model)."""
        x, y, th = self.x
        c, s = math.cos(th), math.sin(th)

        self.x = np.array(
            [x + v_meas * c * dt,
             y + v_meas * s * dt,
             wrap_angle(th + w_meas * dt)],
            dtype=float,
        )

        # Jacobian wrt state
        F = np.array(
            [[1.0, 0.0, -v_meas * s * dt],
             [0.0, 1.0,  v_meas * c * dt],
             [0.0, 0.0,  1.0]],
            dtype=float,
        )

        # Jacobian wrt input noise [v, w]
        L = np.array(
            [[c * dt, 0.0],
             [s * dt, 0.0],
             [0.0,    dt]],
            dtype=float,
        )

        self.P = F @ self.P @ F.T + L @ self.Q_u @ L.T

    def update_landmark(self, z_meas: np.ndarray, landmark_xy: np.ndarray) -> None:
        """EKF update step for a single [range, bearing] measurement."""
        lx, ly = float(landmark_xy[0]), float(landmark_xy[1])
        x, y, th = self.x

        dx, dy = lx - x, ly - y
        q = dx * dx + dy * dy
        if q < 1e-12:
            return

        r = math.sqrt(q)
        b = wrap_angle(math.atan2(dy, dx) - th)
        z_hat = np.array([r, b], dtype=float)

        H = np.array(
            [[-dx / r,  -dy / r,  0.0],
             [ dy / q,  -dx / q, -1.0]],
            dtype=float,
        )

        innov = np.array(
            [z_meas[0] - z_hat[0],
             wrap_angle(z_meas[1] - z_hat[1])],
            dtype=float,
        )

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innov
        self.x[2] = wrap_angle(self.x[2])

        # Joseph form for numerical stability
        I_KH = np.eye(3, dtype=float) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 – Scripted Figure-8 Motion
# ═══════════════════════════════════════════════════════════════════════════════

class ScriptedFigure8:
    """
    Deterministic Lissajous figure-8 used for repeatable EKF validation.
    pose_at(t) → [x, y, yaw]
    """

    def __init__(self, amp_x: float = 1.5, amp_y: float = 1.0, period_s: float = 20.0):
        self.center_xy = np.array([0.0, 0.0], dtype=float)
        self.amp_x = amp_x
        self.amp_y = amp_y
        self.period_s = period_s

    def pose_at(self, t: float) -> np.ndarray:
        T = max(self.period_s, 1e-3)
        w = 2.0 * math.pi / T

        x = self.center_xy[0] + self.amp_x * math.sin(w * t)
        y = self.center_xy[1] + self.amp_y * math.sin(2.0 * w * t)

        dx_dt = self.amp_x * w * math.cos(w * t)
        dy_dt = self.amp_y * 2.0 * w * math.cos(2.0 * w * t)
        yaw = math.atan2(dy_dt, dx_dt + 1e-12)

        return np.array([x, y, yaw], dtype=float)

    def velocity_at(self, t: float) -> Tuple[float, float]:
        """Return world-frame (vx, vy) — analytical first derivative."""
        T = max(self.period_s, 1e-3)
        w = 2.0 * math.pi / T
        vx = self.amp_x * w * math.cos(w * t)
        vy = self.amp_y * 2.0 * w * math.cos(2.0 * w * t)
        return float(vx), float(vy)

    def acceleration_at(self, t: float) -> Tuple[float, float]:
        """Return world-frame (ax, ay) — analytical second derivative."""
        T = max(self.period_s, 1e-3)
        w = 2.0 * math.pi / T
        ax = -self.amp_x * w * w * math.sin(w * t)
        ay = -self.amp_y * 4.0 * w * w * math.sin(2.0 * w * t)
        return float(ax), float(ay)

    def angular_velocity_at(self, t: float) -> float:
        """Return yaw rate — numerical central difference of heading."""
        h = 1e-4
        y1 = self.pose_at(t + h)[2]
        y0 = self.pose_at(t - h)[2]
        return float(wrap_angle(y1 - y0) / (2.0 * h))

    def peak_acceleration(self) -> float:
        """Return the theoretical peak combined acceleration of the trajectory."""
        T = max(self.period_s, 1e-3)
        w = 2.0 * math.pi / T
        # Peak ax = amp_x * w^2, Peak ay = 4 * amp_y * w^2
        ax_peak = self.amp_x * w * w
        ay_peak = 4.0 * self.amp_y * w * w
        return float(math.hypot(ax_peak, ay_peak))

    def apply_to_xform(self, xform, t: float, z: float) -> None:
        """Teleport an Isaac Sim xform prim to the pose at time t."""
        pose = self.pose_at(t)
        quat = yaw_to_quat_wxyz(float(pose[2]))
        xform.set_world_pose(
            position=np.array([float(pose[0]), float(pose[1]), float(z)], dtype=float),
            orientation=quat,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 – Landmark Measurement Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_landmark_measurement(
    gt_pose: np.ndarray,
    landmark_xy: np.ndarray,
    rng: np.random.Generator,
    sigma_r: float,
    sigma_b: float,
    max_range_m: float,
    half_fov_rad: float,
    detection_prob: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate a noisy [range, bearing] landmark observation from gt_pose.

    Returns (z_clean, z_noisy) or None if the landmark is out of range/FOV
    or randomly missed according to detection_prob.
    """
    x, y, th = float(gt_pose[0]), float(gt_pose[1]), float(gt_pose[2])
    lx, ly = float(landmark_xy[0]), float(landmark_xy[1])

    dx, dy = lx - x, ly - y
    r = math.hypot(dx, dy)
    if r > max_range_m:
        return None

    b = wrap_angle(math.atan2(dy, dx) - th)
    if abs(b) > half_fov_rad:
        return None

    if rng.random() > detection_prob:
        return None

    z_clean = np.array([r, b], dtype=float)

    r_meas = max(r + rng.normal(0.0, sigma_r), 0.05)
    b_meas = wrap_angle(b + rng.normal(0.0, sigma_b))
    z_noisy = np.array([r_meas, b_meas], dtype=float)

    return z_clean, z_noisy


def rb_to_xy_robot_frame(z_rb: np.ndarray) -> np.ndarray:
    """Convert [range, bearing] to XY in the robot frame."""
    r, b = float(z_rb[0]), float(z_rb[1])
    return np.array([r * math.cos(b), r * math.sin(b)], dtype=float)


def propagate_unicycle(pose: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
    """Dead-reckoning unicycle step."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    return np.array(
        [x + v * math.cos(th) * dt,
         y + v * math.sin(th) * dt,
         wrap_angle(th + w * dt)],
        dtype=float,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 – Plotting & CSV Export
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(logs: dict, out_dir: str) -> None:
    t   = logs["t"]
    gt  = logs["gt"]
    od  = logs["odom"]
    ek  = logs["ekf"]
    nm  = logs["num_meas"]

    e_odom_yaw = np.array([wrap_angle(a - b) for a, b in zip(gt[:, 2], od[:, 2])], dtype=float)
    e_ekf_yaw  = np.array([wrap_angle(a - b) for a, b in zip(gt[:, 2], ek[:, 2])], dtype=float)

    data = np.column_stack([
        t,
        gt[:, 0], gt[:, 1], gt[:, 2],
        od[:, 0], od[:, 1], od[:, 2],
        ek[:, 0], ek[:, 1], ek[:, 2],
        nm,
        gt[:, 0] - od[:, 0], gt[:, 1] - od[:, 1], e_odom_yaw,
        gt[:, 0] - ek[:, 0], gt[:, 1] - ek[:, 1], e_ekf_yaw,
    ])

    header = ",".join([
        "t",
        "gt_x", "gt_y", "gt_yaw",
        "odom_x", "odom_y", "odom_yaw",
        "ekf_x", "ekf_y", "ekf_yaw",
        "num_meas",
        "err_odom_x", "err_odom_y", "err_odom_yaw",
        "err_ekf_x",  "err_ekf_y",  "err_ekf_yaw",
    ])
    np.savetxt(os.path.join(out_dir, "ekf_log.csv"), data, delimiter=",", header=header, comments="")


def save_trajectory_plot(logs: dict, landmarks_xy: np.ndarray, out_dir: str) -> None:
    gt, od, ek = logs["gt"], logs["odom"], logs["ekf"]

    plt.figure(figsize=(8, 7))
    plt.plot(gt[:, 0], gt[:, 1], label="Ground Truth")
    plt.plot(od[:, 0], od[:, 1], label="Odometry Only")
    plt.plot(ek[:, 0], ek[:, 1], label="EKF")
    if landmarks_xy is not None and len(landmarks_xy):
        plt.scatter(landmarks_xy[:, 0], landmarks_xy[:, 1], marker="x", s=80, label="Landmarks")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("Trajectory: GT vs Odometry vs EKF")
    plt.axis("equal"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory.png"), dpi=150)
    plt.close()


def save_error_plots(logs: dict, out_dir: str) -> None:
    t, gt, od, ek = logs["t"], logs["gt"], logs["odom"], logs["ekf"]

    e_od_x  = gt[:, 0] - od[:, 0]
    e_od_y  = gt[:, 1] - od[:, 1]
    e_od_th = np.array([wrap_angle(a - b) for a, b in zip(gt[:, 2], od[:, 2])], dtype=float)
    e_ek_x  = gt[:, 0] - ek[:, 0]
    e_ek_y  = gt[:, 1] - ek[:, 1]
    e_ek_th = np.array([wrap_angle(a - b) for a, b in zip(gt[:, 2], ek[:, 2])], dtype=float)

    specs = [
        (t, e_od_x,  e_ek_x,  "x error (m)",       "X Error vs Time",            "error_x.png"),
        (t, e_od_y,  e_ek_y,  "y error (m)",        "Y Error vs Time",            "error_y.png"),
        (t, np.rad2deg(e_od_th), np.rad2deg(e_ek_th),
                               "yaw error (deg)",    "Yaw Error vs Time",          "error_yaw.png"),
        (t, np.sqrt(e_od_x**2 + e_od_y**2),
             np.sqrt(e_ek_x**2 + e_ek_y**2),
                               "XY error norm (m)",  "Position Error Norm vs Time","error_xy_norm.png"),
    ]

    for (time, odom_err, ekf_err, ylabel, title, fname) in specs:
        plt.figure(figsize=(8, 4))
        plt.plot(time, odom_err, label="Odometry Only")
        plt.plot(time, ekf_err,  label="EKF")
        plt.xlabel("Time (s)"); plt.ylabel(ylabel); plt.title(title)
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()


def save_measurement_plot(logs: dict, out_dir: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(logs["t"], logs["num_meas"])
    plt.xlabel("Time (s)"); plt.ylabel("# landmark measurements")
    plt.title("Measurements per Frame"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "measurements_per_frame.png"), dpi=150)
    plt.close()


def save_landmark_noise_snapshot(
    clean_xy: np.ndarray,
    noisy_xy: np.ndarray,
    t_snap: float,
    out_dir: str,
) -> None:
    if clean_xy is None or noisy_xy is None or len(clean_xy) == 0:
        return

    plt.figure(figsize=(7, 7))
    plt.scatter(clean_xy[:, 0], clean_xy[:, 1], marker="o", s=60, label="Ideal detections")
    plt.scatter(noisy_xy[:, 0], noisy_xy[:, 1], marker="x", s=70, label="Noisy detections")
    for i in range(min(len(clean_xy), len(noisy_xy))):
        plt.plot([clean_xy[i, 0], noisy_xy[i, 0]],
                 [clean_xy[i, 1], noisy_xy[i, 1]], linewidth=1)
    plt.scatter([0.0], [0.0], marker="s", s=80, label="Robot frame origin")
    plt.arrow(0.0, 0.0, 0.5, 0.0, length_includes_head=True, head_width=0.07)
    plt.xlabel("x_robot (m)"); plt.ylabel("y_robot (m)")
    plt.title(f"Landmark measurement noise snapshot (t={t_snap:.2f}s)")
    plt.axis("equal"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "landmark_noise_snapshot.png"), dpi=150)
    plt.close()
