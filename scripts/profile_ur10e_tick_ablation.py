"""
scripts/profile_ur10e_tick_ablation.py — 量測 UR10e RMPflow 控制迴圈裡
每個「每 tick 都做」的操作各佔多少成本，並用 A/B（ablation）確認拿掉之後
真的變快多少。

背景：headless（無算圖）實測只有約 35 tick/秒（每 tick 28.4ms），但把
`_step_rmpflow()` 的每個操作單獨計時只加總到約 5.1ms，加上
`simulation_app.update()` 的 13.4ms 也只有 18.5ms——中間有約 10ms 找不到。
推測是「在 PHYSICS_POST_STEP callback 內、每個 tick 都重寫目標/增益」對
物理管線造成的交互成本，單獨計時量不出來。

做法：monkey-patch controller 實例上的方法（不改生產程式碼），逐項拿掉，
比較 move_to_home() 的實際 tick 速率。

三個候選（都是每 tick 無條件執行）：
1. `_sync_dynamic_obstacles()` —— 讀母球 RigidPrim 世界座標再寫進
   DynamicSphere proxy（USD 寫入）
2. `rmp_flow.update_world()` —— 重新整理 RMPflow 內部的障礙物世界模型
3. `switch_dof_control_mode("position")` —— 每個 tick 重寫全部 7 個 DOF
   的 stiffness/damping，但增益其實幾乎不會變

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/profile_ur10e_tick_ablation.py
"""

import os
import sys
import time
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TICKS_PER_TRIAL = 400


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services.spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/AblationUr10eTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS

    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, UR10eRobot,
    )
    ball_prim_path = table_ball_set.get_ball_prim_paths()[0]

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()
    table_ball_set.place_ball(0, 0.0, 0.5)
    for _ in range(5):
        simulation_app.update()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    _TABLE_OBSTACLE_HEIGHT_M = 0.15
    table_center = table.get_table_center()
    articulation_api.register_static_box_obstacle(
        [table_center[0], table_center[1], table_z - _TABLE_OBSTACLE_HEIGHT_M / 2.0],
        [TABLE_WIDTH, TABLE_LENGTH, _TABLE_OBSTACLE_HEIGHT_M]
    )
    articulation_api.register_dynamic_sphere_obstacle(ball_prim_path, ball_radius)
    for _ in range(5):
        simulation_app.update()

    rmp_ctrl = articulation_api._ur10e_rmpflow_controller
    rmp_flow = rmp_ctrl._rmp_flow
    articulation = articulation_api._articulation

    # 原始方法留底，每個 trial 前還原
    original_sync = rmp_ctrl._sync_dynamic_obstacles
    original_update_world = rmp_flow.update_world
    original_switch_mode = articulation.switch_dof_control_mode

    def _restore_all():
        rmp_ctrl._sync_dynamic_obstacles = original_sync
        rmp_flow.update_world = original_update_world
        articulation.switch_dof_control_mode = original_switch_mode

    def _trial(label: str) -> float:
        """跑一次 move_to_home()，回傳每 tick 平均毫秒數。同時量
        controller.step() 自己佔掉多少（用 instance 層的 wrapper 計時，
        _step_ur10e_motion() 是透過 self._ur10e_active_controller.step()
        呼叫，所以 patch 實例屬性攔得到）。"""
        step_time = {"total": 0.0, "calls": 0}
        current_step = rmp_ctrl.step

        def _timed_step(frame_duration):
            start = time.perf_counter()
            result = current_step(frame_duration)
            step_time["total"] += time.perf_counter() - start
            step_time["calls"] += 1
            return result

        rmp_ctrl.step = _timed_step
        try:
            articulation_api.move_to_home()
            start = time.perf_counter()
            ticks = 0
            while not articulation_api.is_motion_complete() and ticks < _TICKS_PER_TRIAL:
                simulation_app.update()
                ticks += 1
            elapsed = time.perf_counter() - start
        finally:
            rmp_ctrl.step = current_step

        per_tick_ms = elapsed * 1000.0 / ticks
        step_ms = step_time["total"] * 1000.0 / max(step_time["calls"], 1)
        print(f"[ablation] {label:<44} {per_tick_ms:7.2f} ms/tick "
              f"({1000.0 / per_tick_ms:5.1f} tick/秒)  其中 controller.step()={step_ms:6.2f} ms")
        sys.stdout.flush()
        return per_tick_ms

    print("[ablation] === 基準（全部照現況）===")
    _restore_all()
    baseline = _trial("baseline（現況）")

    print("[ablation] === 逐項拿掉 ===")
    _restore_all()
    rmp_ctrl._sync_dynamic_obstacles = lambda: None
    without_sync = _trial("拿掉 _sync_dynamic_obstacles()")

    _restore_all()
    rmp_flow.update_world = lambda: None
    without_update_world = _trial("拿掉 rmp_flow.update_world()")

    _restore_all()
    articulation.switch_dof_control_mode = lambda *a, **k: None
    without_switch = _trial("拿掉 switch_dof_control_mode()")

    print("[ablation] === 三項全部拿掉 ===")
    _restore_all()
    rmp_ctrl._sync_dynamic_obstacles = lambda: None
    rmp_flow.update_world = lambda: None
    articulation.switch_dof_control_mode = lambda *a, **k: None
    without_all = _trial("三項全拿掉")

    _restore_all()

    print("[ablation] === 節省幅度 ===")
    for label, value in (
        ("_sync_dynamic_obstacles()", without_sync),
        ("rmp_flow.update_world()", without_update_world),
        ("switch_dof_control_mode()", without_switch),
        ("三項全部", without_all),
    ):
        saved = baseline - value
        print(f"[ablation] 拿掉 {label:<30} 省下 {saved:6.2f} ms/tick "
              f"（{100.0 * saved / baseline:5.1f}%）→ {1000.0 / value:5.1f} tick/秒")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[ablation] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
