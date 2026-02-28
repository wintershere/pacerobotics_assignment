# analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# Offline analysis tools: Monte Carlo trials and LiDAR point-cloud quality.
# No Isaac Sim dependencies except the PhysX LiDAR interface in capture_pointcloud().
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ekf import EKF2D, wrap_angle, propagate_unicycle, simulate_landmark_measurement
from .global_variables import OUTPUT_ROOT_DIR

# PhysX LiDAR bindings (Isaac Sim 5.x)
try:
    from isaacsim.sensors.physx import _range_sensor
except Exception:
    _range_sensor = None  # Older / non-Isaac environments

if _range_sensor is None:
    try:
        from omni.isaac.range_sensor import _range_sensor
    except Exception:
        _range_sensor = None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 – Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════

# Physical constants (match RobotScene)
_GRAVITY_M_S2  = 9.81
_ROBOT_MASS_KG = 400.0
_MU_NOMINAL    = 0.8    # nominal static friction (matches scene defaultMaterial)


def _physics_odometry_error(
    v_true: float,
    w_true: float,
    prev_v: float,
    dt: float,
    mass: float,
    mu: float,
    rng: np.random.Generator,
    sigma_v_base: float,
    sigma_w_base: float,
) -> tuple[float, float, float, float, float]:
    """
    Compute physically-grounded odometry measurements for one timestep.

    Three continuous effects link mass and friction to odometry quality:

    1. TRACTION UTILISATION (friction-dependent)
       ──────────────────────────────────────────
       From the Part 1 Coulomb friction analysis, the friction circle limits
       total acceleration to  a_max = μ · g.  Both tangential acceleration
       (speed changes) and centripetal acceleration (turning: v²·κ ≈ v·|ω|)
       consume friction budget.  We define:

           η = a_demand / a_max = √(a_t² + a_c²) / (μ · g)

       Real tyres exhibit progressive micro-slip well below the gross-slip
       threshold (cf. the linear region of the Pacejka tyre curve).  Encoder
       accuracy degrades as traction utilisation rises:

           traction_factor = 1 + k_trac · η²        (k_trac = 8.0)

       Lower μ  →  higher η for the same trajectory  →  more noise.

    2. TYRE COMPLIANCE (mass-dependent)
       ──────────────────────────────────
       A heavier robot increases normal force on each wheel, producing greater
       tyre deformation.  Deformed tyres cause the wheel to roll a slightly
       different distance than the encoder expects (effective radius changes).
       This is a direct, non-cancelling mass effect:

           tyre_factor = (mass / mass_nominal)^0.5

       The square-root models the diminishing-returns relationship between
       load and deformation for pneumatic / elastomer tyres.

    3. DYNAMIC LOAD TRANSFER (mass × acceleration interaction)
       ─────────────────────────────────────────────────────────
       During acceleration, the offset CoM (Part 1) shifts normal force
       between front and rear contact patches:

           ΔN / N_static  =  a_demand · h_CoM / (g · L_eff)

       This creates *differential* wheel slip (one side grips more than the
       other), which injects angular odometry error.  The absolute tyre
       deflection that causes the error is proportional to mass (heavier
       robot → more deflection for the same load-transfer ratio):

           load_factor = 1 + k_load · (mass / mass_nominal) · (ΔN / N_static)

       This term couples mass AND acceleration, and through η, couples in
       friction as well.

    Together:  σ_eff = σ_base × tyre_factor × traction_factor × load_factor

    Returns (v_meas, w_meas, traction_eta, sigma_v_eff, sigma_w_eff).
    """
    g = _GRAVITY_M_S2
    mass_nom = _ROBOT_MASS_KG

    # ── Acceleration demands ──────────────────────────────────────────────
    # Tangential (speed change)
    a_tangential = abs(v_true - prev_v) / max(dt, 1e-9)
    # Centripetal (turning) — v · |ω|  is a good approximation for a_c = v²/R
    a_centripetal = abs(v_true * w_true)
    # Combined acceleration demand on the friction circle
    a_demand = np.sqrt(a_tangential**2 + a_centripetal**2)

    # ── 1. Traction utilisation ───────────────────────────────────────────
    a_max = mu * g
    eta = min(a_demand / max(a_max, 1e-9), 1.0)
    k_trac = 8.0  # tuned so that η≈0.15 gives ~20% noise increase
    traction_factor = 1.0 + k_trac * eta * eta

    # ── 2. Tyre compliance ────────────────────────────────────────────────
    tyre_factor = (mass / mass_nom) ** 0.5

    # ── 3. Dynamic load transfer ──────────────────────────────────────────
    h_com   = 0.60   # CoM height from ground (Part 1: 600 mm)
    L_eff   = 0.65   # effective wheelbase ≈ robot width
    dn_over_n = a_demand * h_com / (g * L_eff * 0.5)  # ΔN / N_static
    k_load = 2.0
    load_factor = 1.0 + k_load * (mass / mass_nom) * dn_over_n

    # ── Combined noise ────────────────────────────────────────────────────
    combined = tyre_factor * traction_factor * load_factor
    sv = sigma_v_base * combined
    sw = sigma_w_base * combined

    v_meas = v_true + rng.normal(0.0, sv)
    w_meas = w_true + rng.normal(0.0, sw)

    return v_meas, w_meas, eta, sv, sw


def run_mc_trials(
    pose_fn,
    odom_fn,
    landmarks_xy: np.ndarray,
    *,
    num_trials: int,
    duration_s: float,
    dt: float,
    base_seed: int,
    sigma_v: float,
    sigma_w: float,
    sigma_r: float,
    sigma_b: float,
    max_range_m: float,
    half_fov_rad: float,
    base_detect_prob: float,
) -> np.ndarray:
    # Use realistic sensor constraints for MC trials: 4 m range and 120° FOV
    # (rather than the 20 m / 270° used in the live sim). With 6 landmarks
    # spread over a 6x6 m arena this gives typically 1-2 visible per step,
    # forcing the EKF to rely on odometry between observations and making
    # the mass/friction variation actually show up in EKF MSE.
    _mc_max_range_m  = 4.0
    _mc_half_fov_rad = np.deg2rad(60.0)   # 120° total FOV
    """
    Run `num_trials` offline EKF trials with physically-grounded mass/friction
    variation.

    Odometry error is derived from a continuous tyre-physics model tied to
    the Part 1 Coulomb friction analysis:

    - Friction (μ) controls the traction utilisation ratio η = a_demand/(μg).
      Lower μ → higher η → more tyre micro-slip → worse odometry.
    - Mass affects tyre compliance (heavier → more deformation → encoder drift)
      and dynamic load transfer (heavier + offset CoM → differential wheel slip).

    Both effects operate *continuously* — there is no binary slip/no-slip
    threshold — so ±20 % randomisation of mass and μ always produces
    measurable MSE variation.

    Returns an (N × 16) array; column layout defined by `_MC_COLS`.
    """
    steps = int(duration_s / dt)
    if steps < 2:
        raise ValueError("duration_s / dt is too small — increase duration or decrease dt.")

    rows = []

    for k in range(int(num_trials)):
        rng = np.random.default_rng(base_seed + 1000 + k)

        # Randomise ±20% as required by assignment
        mass_scale  = float(rng.uniform(0.8, 1.2))
        mu_scale    = float(rng.uniform(0.8, 1.2))
        lidar_scale = float(rng.uniform(0.8, 1.2))

        mass = _ROBOT_MASS_KG * mass_scale
        mu   = _MU_NOMINAL    * mu_scale

        # a_crit from Part 1 Coulomb analysis (for reference / plotting)
        a_crit = mu * _GRAVITY_M_S2

        p_det = float(np.clip(base_detect_prob + rng.normal(0.0, 0.03), 0.70, 0.995))
        sr = sigma_r * lidar_scale
        sb = sigma_b * lidar_scale

        ekf = EKF2D()
        # EKF uses NOMINAL (design-time) noise — it does NOT know the
        # trial-specific lidar_scale or the per-step tyre noise. This is
        # realistic: a real filter is tuned once, not adapted per trial.
        ekf.R   = np.diag([sigma_r, sigma_b]) ** 2          # nominal, not sr/sb
        ekf.Q_u = np.diag([sigma_v, sigma_w]) ** 2          # nominal, fixed

        # LiDAR rotates at 10 Hz; only update EKF every 6 physics steps (60Hz/10Hz)
        LIDAR_PERIOD = 6

        gt0 = pose_fn(0.0)
        ekf.reset(gt0)
        odom_state = gt0.copy()
        prev_gt    = gt0.copy()
        prev_v     = 0.0

        gt_list, od_list, ek_list = [], [], []
        eta_accum = 0.0

        for i in range(1, steps):
            t  = i * dt
            gt = pose_fn(t)
            v_true, w_true = odom_fn(prev_gt, gt, dt)

            # Physics-derived odometry — continuous tyre model from Part 1
            v_meas, w_meas, eta, sv_step, sw_step = _physics_odometry_error(
                v_true=v_true, w_true=w_true,
                prev_v=prev_v, dt=dt,
                mass=mass, mu=mu,
                rng=rng,
                sigma_v_base=sigma_v,
                sigma_w_base=sigma_w,
            )
            eta_accum += eta

            # EKF predicts every step with FIXED nominal Q_u (no per-step tuning)
            ekf.predict(v_meas, w_meas, dt)
            odom_state = propagate_unicycle(odom_state, v_meas, w_meas, dt)

            # Landmark updates only at LiDAR rate (every LIDAR_PERIOD steps)
            if i % LIDAR_PERIOD == 0:
                for lm in landmarks_xy:
                    pair = simulate_landmark_measurement(
                        gt_pose=gt, landmark_xy=lm, rng=rng,
                        sigma_r=sr, sigma_b=sb,
                        max_range_m=_mc_max_range_m,
                        half_fov_rad=_mc_half_fov_rad,
                        detection_prob=p_det,
                    )
                    if pair is not None:
                        _, z_noisy = pair
                        ekf.update_landmark(z_noisy, lm)

            gt_list.append(gt.copy())
            od_list.append(odom_state.copy())
            ek_list.append(ekf.x.copy())
            prev_gt = gt
            prev_v  = v_true

        gt_arr = np.asarray(gt_list, dtype=float)
        od_arr = np.asarray(od_list, dtype=float)
        ek_arr = np.asarray(ek_list, dtype=float)

        exo = gt_arr[:, 0] - od_arr[:, 0]
        eyo = gt_arr[:, 1] - od_arr[:, 1]
        eto = np.array([wrap_angle(a - b) for a, b in zip(gt_arr[:, 2], od_arr[:, 2])], dtype=float)
        exe = gt_arr[:, 0] - ek_arr[:, 0]
        eye = gt_arr[:, 1] - ek_arr[:, 1]
        ete = np.array([wrap_angle(a - b) for a, b in zip(gt_arr[:, 2], ek_arr[:, 2])], dtype=float)

        mean_eta = eta_accum / max(steps - 1, 1)

        rows.append([
            float(k), mass_scale, mu_scale, a_crit, lidar_scale, p_det,
            sigma_v, sigma_w, sr, float(np.rad2deg(sb)),
            float(np.mean(exo**2 + eyo**2)),   # MSE_xy_odom
            float(np.mean(exe**2 + eye**2)),    # MSE_xy_ekf
            float(np.mean(eto**2)),             # MSE_yaw_odom
            float(np.mean(ete**2)),             # MSE_yaw_ekf
            mean_eta,                           # mean traction utilisation
        ])

    return np.asarray(rows, dtype=float)


_MC_COLS = [
    "trial", "mass_scale", "mu_scale", "a_crit_m_s2",
    "lidar_scale", "detection_prob",
    "sigma_v", "sigma_w", "sigma_r", "sigma_b_deg",
    "mse_xy_odom", "mse_xy_ekf",
    "mse_yaw_odom", "mse_yaw_ekf",
    "mean_traction_eta",
]


def save_mc_results(rows: np.ndarray, out_dir: Optional[str] = None) -> str:
    if out_dir is None:
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUTPUT_ROOT_DIR, f"monte_carlo_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(
        os.path.join(out_dir, "monte_carlo_mse.csv"),
        rows, delimiter=",",
        header=",".join(_MC_COLS), comments="",
    )
    _mc_plots(rows, out_dir)
    _mc_summary(rows, out_dir)
    return out_dir


# ── Monte Carlo plot helpers ──────────────────────────────────────────────────

def _mc_plots(rows: np.ndarray, out_dir: str) -> None:
    mse_xy_odom,  mse_xy_ekf  = rows[:, 10], rows[:, 11]
    mse_yaw_odom, mse_yaw_ekf = rows[:, 12], rows[:, 13]
    a_crit     = rows[:, 3]
    mean_eta   = rows[:, 14]
    mass_scale = rows[:, 1]
    mu_scale   = rows[:, 2]

    # Boxplots
    for (vals, ylabel, title, fname) in [
        ([mse_xy_odom,  mse_xy_ekf],  "MSE_xy (m2)",   "Monte Carlo: Position MSE (XY)", "mc_boxplot_mse_xy.png"),
        ([mse_yaw_odom, mse_yaw_ekf], "MSE_yaw (rad2)", "Monte Carlo: Yaw MSE",           "mc_boxplot_mse_yaw.png"),
    ]:
        plt.figure(figsize=(7, 5))
        plt.boxplot(vals, labels=["Odometry", "EKF"])
        plt.ylabel(ylabel); plt.title(title); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150); plt.close()

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(mse_xy_odom, bins=25, alpha=0.5, label="Odometry")
    plt.hist(mse_xy_ekf,  bins=25, alpha=0.5, label="EKF")
    plt.xlabel("MSE_xy (m2)"); plt.ylabel("Count")
    plt.title("Monte Carlo: MSE_xy distribution"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mc_hist_mse_xy.png"), dpi=150); plt.close()

    # CDF
    plt.figure(figsize=(8, 5))
    for arr, name in [(mse_xy_odom, "Odometry"), (mse_xy_ekf, "EKF")]:
        s = np.sort(arr)
        plt.plot(s, np.linspace(0.0, 1.0, len(s), endpoint=True), label=name)
    plt.xlabel("MSE_xy (m2)"); plt.ylabel("CDF")
    plt.title("Monte Carlo: CDF of MSE_xy"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mc_cdf_mse_xy.png"), dpi=150); plt.close()

    # Traction utilisation vs MSE — shows how friction drives odometry quality
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc0 = axes[0].scatter(mean_eta, mse_xy_odom, c=mu_scale, s=60, cmap="coolwarm")
    axes[0].set_xlabel("Mean traction utilisation η  (higher = closer to friction limit)")
    axes[0].set_ylabel("MSE_xy odometry (m²)")
    axes[0].set_title("Odometry MSE vs Traction Utilisation")
    axes[0].grid(True)
    plt.colorbar(sc0, ax=axes[0], label="μ scale")
    sc1 = axes[1].scatter(mean_eta, mse_xy_ekf, c=mu_scale, s=60, cmap="coolwarm")
    axes[1].set_xlabel("Mean traction utilisation η  (higher = closer to friction limit)")
    axes[1].set_ylabel("MSE_xy EKF (m²)")
    axes[1].set_title("EKF MSE vs Traction Utilisation")
    axes[1].grid(True)
    plt.colorbar(sc1, ax=axes[1], label="μ scale")
    plt.suptitle("Lower μ → higher η → worse odometry; EKF compensates with landmarks", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mc_eta_vs_mse.png"), dpi=150)
    plt.close()

    # Mass vs friction scatter coloured by odom MSE and EKF MSE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc0 = axes[0].scatter(mass_scale, mu_scale, c=mse_xy_odom, s=60, cmap="hot_r")
    axes[0].set_xlabel("mass scale"); axes[0].set_ylabel("friction (μ) scale")
    axes[0].set_title("Odometry MSE_xy vs (mass, μ)")
    axes[0].grid(True)
    plt.colorbar(sc0, ax=axes[0], label="MSE_xy_odom (m²)")
    sc1 = axes[1].scatter(mass_scale, mu_scale, c=mse_xy_ekf, s=60, cmap="YlOrRd")
    axes[1].set_xlabel("mass scale"); axes[1].set_ylabel("friction (μ) scale")
    axes[1].set_title("EKF MSE_xy vs (mass, μ)")
    axes[1].grid(True)
    plt.colorbar(sc1, ax=axes[1], label="MSE_xy_ekf (m²)")
    plt.suptitle("High mass + low μ → worst odometry; EKF reduces MSE across all conditions", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mc_scatter_mass_mu_mse.png"), dpi=150)
    plt.close()


def _mc_summary(rows: np.ndarray, out_dir: str) -> None:
    mse_xy_odom,  mse_xy_ekf  = rows[:, 10], rows[:, 11]
    mse_yaw_odom, mse_yaw_ekf = rows[:, 12], rows[:, 13]

    eps = 1e-12
    imp_xy  = 100.0 * (mse_xy_odom  - mse_xy_ekf)  / np.maximum(mse_xy_odom,  eps)
    imp_yaw = 100.0 * (mse_yaw_odom - mse_yaw_ekf) / np.maximum(mse_yaw_odom, eps)

    with open(os.path.join(out_dir, "monte_carlo_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Monte Carlo Summary (MSE)\n========================\n")
        f.write(f"Trials: {len(rows)}\n")
        f.write(f"Mean a_crit (m/s2) : {np.mean(rows[:,3]):.3f}  (range {rows[:,3].min():.3f}-{rows[:,3].max():.3f})\n")
        f.write(f"Mean traction η    : {np.mean(rows[:,14]):.4f}  (range {rows[:,14].min():.4f}-{rows[:,14].max():.4f})\n\n")
        for label, odom_arr, ekf_arr, imp_arr, unit in [
            ("MSE_xy",  mse_xy_odom,  mse_xy_ekf,  imp_xy,  "m²"),
            ("MSE_yaw", mse_yaw_odom, mse_yaw_ekf, imp_yaw, "rad²"),
        ]:
            f.write(f"{label} ({unit})\n")
            f.write(f"  Odometry  mean±std : {np.mean(odom_arr):.4e} ± {np.std(odom_arr):.4e}\n")
            f.write(f"  EKF       mean±std : {np.mean(ekf_arr):.4e} ± {np.std(ekf_arr):.4e}\n")
            f.write(f"  Improvement %      : {np.mean(imp_arr):.2f} ± {np.std(imp_arr):.2f}\n\n")
        f.write(
            "Odometry noise model: continuous tyre-physics model where\n"
            "  - Friction (μ) sets the traction utilisation η = a_demand/(μg)\n"
            "  - Mass scales tyre compliance noise (heavier → more deformation)\n"
            "  - Dynamic load transfer couples mass, acceleration, and CoM offset\n"
            "All three factors operate continuously (no binary slip threshold),\n"
            "so ±20% variation in mass and μ always produces measurable MSE spread.\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 – LiDAR Point-Cloud Quality
# ═══════════════════════════════════════════════════════════════════════════════

def capture_pointcloud(lidar_prim_path: str, tries: int = 5) -> np.ndarray:
    """
    Read the current point cloud from a PhysX LiDAR prim and return an (N,3) array.

    Isaac Sim's get_point_cloud_data can return either:
      - A flat  (N, 3)    array  [older builds / high_lod=False]
      - A tiled (H, W, 3) array  [high_lod=True or newer builds]
    Both shapes are normalised to (N, 3) here.

    The LiDAR buffer is populated only after the sensor completes at least one
    full rotation. At 10 Hz that takes 0.1 s; ensure the simulation has been
    running for >= 30 frames before calling this.
    """
    if _range_sensor is None:
        raise RuntimeError(
            "PhysX LiDAR bindings not available. "
            "Enable extension 'isaacsim.sensors.physx' and ensure "
            "'from isaacsim.sensors.physx import _range_sensor' succeeds."
        )

    import carb  # local import — analysis.py has no mandatory Isaac dependency at module level

    iface = _range_sensor.acquire_lidar_sensor_interface()

    for attempt in range(max(1, int(tries))):
        raw = iface.get_point_cloud_data(lidar_prim_path)

        if raw is None:
            carb.log_warn(f"[LiDAR] attempt {attempt+1}: get_point_cloud_data returned None")
            continue

        pc = np.asarray(raw, dtype=float)
        carb.log_info(
            f"[LiDAR] attempt {attempt+1}: raw shape={pc.shape}  dtype={pc.dtype}"
        )

        # Normalise to (N, 3) — the API returns (H, W, 3) when high_lod=True
        if pc.ndim == 3 and pc.shape[2] == 3:
            pc = pc.reshape(-1, 3)          # (H, W, 3) -> (H*W, 3)
        elif pc.ndim == 2 and pc.shape[1] == 3:
            pass                            # already correct
        elif pc.ndim == 1 and len(pc) % 3 == 0:
            pc = pc.reshape(-1, 3)          # flat interleaved xyz
        else:
            carb.log_warn(
                f"[LiDAR] attempt {attempt+1}: unexpected shape {pc.shape}, skipping"
            )
            continue

        keep = np.linalg.norm(pc, axis=1) > 1e-6   # drop no-hit zero returns
        pc = pc[keep]
        carb.log_info(f"[LiDAR] attempt {attempt+1}: {len(pc)} valid points after filtering")

        if len(pc) > 10:    # accept even a sparse cloud — sweep is still meaningful
            return pc

    carb.log_warn(
        f"[LiDAR] No valid points after {tries} attempts at '{lidar_prim_path}'. "
        "Ensure: (1) LOAD was pressed first, (2) sim ticked >= 30 frames, "
        "(3) geometry exists within LiDAR range, (4) prim path is correct."
    )
    return np.zeros((0, 3), dtype=float)

def _add_range_noise(
    points: np.ndarray,
    sigma_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise to the range (distance) of each LiDAR point."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        return np.zeros((0, 3), dtype=float)

    r = np.linalg.norm(p, axis=1)
    valid = r > 1e-6
    if not np.any(valid):
        return np.zeros((0, 3), dtype=float)

    dirs = np.zeros_like(p)
    dirs[valid] = p[valid] / r[valid, None]

    r_noisy = r.copy()
    r_noisy[valid] += rng.normal(0.0, sigma_m, size=int(np.sum(valid)))
    r_noisy = np.clip(r_noisy, 0.0, None)

    return dirs * r_noisy[:, None]


def save_variance_sweep(
    lidar_prim_path: str,
    sigmas_m: Sequence[float] = (0.0, 0.01, 0.03, 0.05),
    out_dir: Optional[str] = None,
    seed: int = 0,
) -> Optional[str]:
    """
    Capture the live LiDAR point cloud then produce two plots:
      - lidar_variance_sweep.png  : XY scatter for each σ value
      - lidar_range_error_hist.png: Δr histogram for each σ value
    Returns out_dir on success, None if the point cloud is empty.
    """
    rng = np.random.default_rng(seed)
    raw = capture_pointcloud(lidar_prim_path, tries=5)
    if len(raw) == 0:
        return None

    if out_dir is None:
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUTPUT_ROOT_DIR, f"lidar_quality_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    sigmas_m = list(sigmas_m)
    n_cols = 2 if len(sigmas_m) > 1 else 1
    n_rows = int(np.ceil(len(sigmas_m) / n_cols))

    # XY scatter panels for each σ
    plt.figure(figsize=(7 * n_cols, 6 * n_rows))
    for i, s in enumerate(sigmas_m):
        noisy = _add_range_noise(raw, float(s), rng)
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.scatter(noisy[:, 0], noisy[:, 1], s=2)
        ax.set_xlabel("x (m) [sensor frame]")
        ax.set_ylabel("y (m) [sensor frame]")
        ax.set_title(f"σ = {s:.3f} m  (var = {s*s:.6f})")
        ax.axis("equal"); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lidar_variance_sweep.png"), dpi=150)
    plt.close()

    # Δr histograms
    raw_r = np.linalg.norm(raw, axis=1)
    plt.figure(figsize=(8, 5))
    for s in sigmas_m:
        noisy = _add_range_noise(raw, float(s), rng)
        dr = np.linalg.norm(noisy, axis=1) - raw_r
        plt.hist(dr, bins=60, alpha=0.5, label=f"σ={s:.3f} m")
    plt.xlabel("Range error Δr (m)"); plt.ylabel("Count")
    plt.title("Range error distributions for different σ")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lidar_range_error_hist.png"), dpi=150)
    plt.close()

    return out_dir


