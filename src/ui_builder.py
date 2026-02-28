# ui_builder.py
# ─────────────────────────────────────────────────────────────────────────────
# Omniverse UI wiring.  All simulation logic lives in simulation.py.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import os
import traceback

import carb
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

from .simulation import RobotScene, EkfScenario
from .analysis import save_variance_sweep
from .global_variables import OUTPUT_ROOT_DIR


class UIBuilder:

    def __init__(self):
        self.frames: list = []
        self.wrapped_ui_elements: list = []
        self._timeline = omni.timeline.get_timeline_interface()

        # Guard: prevents stage-OPENED handler from firing during our own load
        self._loading_stage = False

        # UI handles — initialised in build_ui(), never reset to None afterwards
        self._load_btn           = None
        self._reset_btn          = None
        self._ekf_state_btn      = None
        self._lidar_sweep_btn    = None
        self._mc_btn             = None

        self._init_scenario_objects()

    # ── Standard extension callbacks ─────────────────────────────────────────

    def on_menu_callback(self) -> None:
        pass

    def on_timeline_event(self, event) -> None:
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._set_run_buttons_enabled(False)

    def on_physics_step(self, step: float) -> None:
        pass

    def on_stage_event(self, event) -> None:
        if event.type == int(StageEventType.OPENED):
            # Ignore the OPENED event triggered by our own create_new_stage()
            if self._loading_stage:
                return
            self._reset_ui_state()

    def cleanup(self) -> None:
        for elem in self.wrapped_ui_elements:
            elem.cleanup()

    # ── UI construction ───────────────────────────────────────────────────────

    def build_ui(self) -> None:
        with CollapsableFrame("World Controls", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD",
                    setup_scene_fn=self._setup_scene,
                    setup_post_load_fn=self._setup_scenario,
                )
                self._load_btn.set_world_settings(physics_dt=1/60.0, rendering_dt=1/60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET",
                    pre_reset_fn=None,
                    post_reset_fn=self._on_post_reset,
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        with CollapsableFrame("Run", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._ekf_state_btn = StateButton(
                    "EKF", "RUN EKF", "STOP EKF",
                    on_a_click_fn=self._on_run_ekf,
                    on_b_click_fn=self._on_stop_ekf,
                    physics_callback_fn=self._tick_ekf,
                )
                self._ekf_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._ekf_state_btn)

                self._lidar_sweep_btn = ui.Button(
                    "RUN LiDAR Variance Sweep",
                    clicked_fn=self._on_lidar_sweep_clicked,
                )
                self._lidar_sweep_btn.enabled = False

                self._mc_btn = ui.Button(
                    "RUN Monte Carlo (50 trials)",
                    clicked_fn=self._on_mc_clicked,
                )
                self._mc_btn.enabled = False

    # ── Scene setup / teardown ───────────────────────────────────────────────

    def _init_scenario_objects(self) -> None:
        """Create fresh scenario logic objects (no UI refs touched here)."""
        self._robot_scene  = RobotScene()
        self._ekf_scenario = EkfScenario()

    def _setup_scene(self) -> None:
        """Called by LoadButton — runs before physics init."""
        self._loading_stage = True
        create_new_stage()
        self._add_light()

        stage_objects = self._robot_scene.load()
        world = World.instance()
        for obj in stage_objects:
            world.scene.add(obj)

    def _setup_scenario(self) -> None:
        """Called by LoadButton after physics init."""
        try:
            self._robot_scene.setup()
            self._ekf_scenario.setup()
        except Exception:
            carb.log_error(f"[UIBuilder] Setup failed:\n{traceback.format_exc()}")
        finally:
            self._loading_stage = False

        self._reset_btn.enabled = True
        self._set_run_buttons_enabled(True)

    def _on_post_reset(self) -> None:
        try:
            self._robot_scene.reset()
            self._ekf_scenario.reset()
        except Exception:
            carb.log_error(f"[UIBuilder] Reset failed:\n{traceback.format_exc()}")

        if self._ekf_state_btn:   self._ekf_state_btn.reset()
        self._set_run_buttons_enabled(True)

    # ── Physics tick callbacks ────────────────────────────────────────────────

    def _tick_ekf(self, step: float) -> None:
        self._ekf_scenario.update(step)

    # ── EKF button handlers ───────────────────────────────────────────────────

    def _on_run_ekf(self) -> None:
        try:
            self._ekf_scenario.setup()
        except Exception:
            carb.log_error(f"[UIBuilder] EKF setup failed:\n{traceback.format_exc()}")
            return
        self._timeline.play()

    def _on_stop_ekf(self) -> None:
        """Pause simulation and export artifacts exactly once."""
        self._timeline.pause()
        try:
            out = self._ekf_scenario.save_run_artifacts()
            if out:
                carb.log_info(f"[UIBuilder] EKF artifacts saved → {out}")
        except Exception:
            carb.log_error(f"[UIBuilder] EKF export failed:\n{traceback.format_exc()}")

    # ── LiDAR variance sweep ─────────────────────────────────────────────────

    def _on_lidar_sweep_clicked(self) -> None:
        asyncio.ensure_future(self._lidar_sweep_async())

    async def _lidar_sweep_async(self) -> None:
        """Tick a few frames to fill the LiDAR buffer, then run the sweep."""
        try:
            self._timeline.play()
            app = omni.kit.app.get_app()
            # 60 frames @ 60 fps = 1 s → >= 10 complete LiDAR rotations at 10 Hz
            for _ in range(60):
                await app.next_update_async()
            self._timeline.pause()

            out_dir = os.path.join(
                OUTPUT_ROOT_DIR, "lidar_variance_sweep_manual",
            )
            result = save_variance_sweep(
                RobotScene.LIDAR_PATH,
                sigmas_m=(0.0, 0.01, 0.03, 0.05),
                out_dir=out_dir,
                seed=0,
            )
            if result is None:
                carb.log_warn(
                    "[LiDAR] Point cloud empty after ticking. "
                    "Check the LiDAR prim path and that geometry is within range."
                )
            else:
                carb.log_info(f"[LiDAR] Variance sweep saved → {result}")

        except Exception:
            self._timeline.pause()
            carb.log_error(f"[UIBuilder] LiDAR sweep failed:\n{traceback.format_exc()}")

    # ── Monte Carlo ───────────────────────────────────────────────────────────

    def _on_mc_clicked(self) -> None:
        self._timeline.pause()
        try:
            carb.log_info("[UIBuilder] Running Monte Carlo (50 trials)…")
            out = self._ekf_scenario.run_monte_carlo(num_trials=50, duration_s=20.0, dt=1.0/60.0)
            carb.log_info(f"[UIBuilder] Monte Carlo saved → {out}")
        except Exception:
            carb.log_error(f"[UIBuilder] Monte Carlo failed:\n{traceback.format_exc()}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_run_buttons_enabled(self, state: bool) -> None:
        for btn in (self._ekf_state_btn,
                    self._lidar_sweep_btn, self._mc_btn):
            if btn is not None:
                btn.enabled = state

    def _reset_ui_state(self) -> None:
        """Called when a new stage is opened externally (not by us)."""
        self._init_scenario_objects()
        if self._ekf_state_btn:   self._ekf_state_btn.reset()
        if self._reset_btn:       self._reset_btn.enabled = False
        self._set_run_buttons_enabled(False)

    @staticmethod
    def _add_light() -> None:
        stage = get_current_stage()
        light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/SphereLight"))
        light.CreateRadiusAttr(2)
        light.CreateIntensityAttr(100_000)
        SingleXFormPrim(str(light.GetPath())).set_world_pose([6.5, 0, 12])
