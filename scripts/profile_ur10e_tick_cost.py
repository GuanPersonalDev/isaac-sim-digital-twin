"""
scripts/profile_ur10e_tick_cost.py — 量測 UR10e 每個 physics tick 的成本
組成，找出「headless 也只有 ~22 tick/秒、GUI 只有 15 FPS、手臂看起來
轉很慢」的瓶頸。

背景：手臂動作是用 **tick 數**定義的（RESET 902 tick、AIM 數千 tick），
所以 tick 速率直接決定使用者看到的動作速度。實測 headless（完全沒有算圖）
只有約 22 tick/秒，離 60Hz 差 3 倍——瓶頸不在算圖，在每個 tick 自己做的
事。這支腳本把 `Ur10eRmpflowController._step_rmpflow()` 會呼叫到的每個
操作單獨拉出來各跑 N 次計時，找出真正吃時間的那一個。

刻意不改生產程式碼，只從外面呼叫同樣的 API。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/profile_ur10e_tick_cost.py
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

_SAMPLES = 200


def _time_it(label: str, fn, samples: int = _SAMPLES) -> float:
    """⚠️ 緊迴圈連續呼叫，中間沒有 physics step——tensor API 會回快取值，
    量到的是「快取讀取」而不是真實成本。只保留來對照，實際成本看
    `_time_it_realistic()`。"""
    fn()
    start = time.perf_counter()
    for _ in range(samples):
        fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / samples
    print(f"[profile] {label:<52} {elapsed_ms:8.3f} ms/次（快取，非真實）")
    sys.stdout.flush()
    return elapsed_ms


def _time_it_realistic(label: str, fn, update_fn, samples: int = 60) -> float:
    """每次呼叫前都先跑一個真實 physics step，讓 tensor API 的快取失效——
    這才是每個 tick 真正要付的成本（含 GPU→CPU 同步）。只累加 fn() 本身
    的時間，不含 physics step。"""
    fn()
    total = 0.0
    for _ in range(samples):
        update_fn()  # 讓快取失效，成本不計入
        start = time.perf_counter()
        fn()
        total += time.perf_counter() - start
    elapsed_ms = total * 1000.0 / samples
    print(f"[profile] {label:<52} {elapsed_ms:8.3f} ms/次")
    sys.stdout.flush()
    return elapsed_ms


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

    table_base_path = "/World/ProfileUr10eTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS

    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    robot_manager = TableRobotManager(
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

    print("[profile] === 尚未註冊障礙物 ===")
    _time_it("simulation_app.update()（無動作、無障礙物）", simulation_app.update)

    _TABLE_OBSTACLE_HEIGHT_M = 0.15
    table_center = table.get_table_center()
    articulation_api.register_static_box_obstacle(
        [table_center[0], table_center[1], table_z - _TABLE_OBSTACLE_HEIGHT_M / 2.0],
        [TABLE_WIDTH, TABLE_LENGTH, _TABLE_OBSTACLE_HEIGHT_M],
    )
    articulation_api.register_dynamic_sphere_obstacle(ball_prim_path, ball_radius)
    for _ in range(5):
        simulation_app.update()

    rmp_ctrl = articulation_api._ur10e_rmpflow_controller
    rmp_flow = rmp_ctrl._rmp_flow
    articulation = articulation_api._articulation

    print("[profile] === 已註冊障礙物（球檯方塊 + 母球動態球體）===")
    baseline_ms = _time_it("simulation_app.update()（無動作、有障礙物）", simulation_app.update)

    print("[profile] === _step_rmpflow() 會用到的個別操作（每次呼叫前先跑真實 physics step）===")
    positions = np.asarray(articulation.get_dof_positions())[0]
    velocities = np.asarray(articulation.get_dof_velocities())[0]
    active_positions = positions[rmp_ctrl._active_dof_indices]
    active_velocities = velocities[rmp_ctrl._active_dof_indices]
    full_targets = positions.copy()

    update_world_ms = _time_it_realistic("rmp_flow.update_world()", rmp_flow.update_world,
                                         simulation_app.update)
    sync_obstacles_ms = _time_it_realistic("_sync_dynamic_obstacles()", rmp_ctrl._sync_dynamic_obstacles,
                                           simulation_app.update)
    get_positions_ms = _time_it_realistic("articulation.get_dof_positions()",
                                          lambda: np.asarray(articulation.get_dof_positions())[0],
                                          simulation_app.update)
    get_velocities_ms = _time_it_realistic("articulation.get_dof_velocities()",
                                           lambda: np.asarray(articulation.get_dof_velocities())[0],
                                           simulation_app.update)
    compute_targets_ms = _time_it_realistic(
        "rmp_flow.compute_joint_targets()",
        lambda: rmp_flow.compute_joint_targets(
            active_positions, active_velocities, np.array([]), np.array([]), 1.0 / 60.0
        ),
        simulation_app.update,
    )
    switch_mode_ms = _time_it_realistic("articulation.switch_dof_control_mode('position')",
                                        lambda: articulation.switch_dof_control_mode("position"),
                                        simulation_app.update)
    set_targets_ms = _time_it_realistic("articulation.set_dof_position_targets()",
                                        lambda: articulation.set_dof_position_targets(full_targets[None, :]),
                                        simulation_app.update)
    gravity_ms = _time_it_realistic("articulation.get_dof_gravity_compensation_forces()",
                                    articulation.get_dof_gravity_compensation_forces,
                                    simulation_app.update)
    ee_pose_ms = _time_it_realistic("end_effector_rigid_prim.get_world_poses()",
                                    rmp_ctrl._end_effector_rigid_prim.get_world_poses,
                                    simulation_app.update)

    per_tick_sum = (
        update_world_ms + sync_obstacles_ms + get_positions_ms + get_velocities_ms
        + compute_targets_ms + switch_mode_ms + set_targets_ms + ee_pose_ms
    )
    print(f"[profile] --- _step_rmpflow() 一個 tick 的估計總成本 ≈ {per_tick_sum:.3f} ms"
          f"（不含 simulation_app.update() 本身的 {baseline_ms:.3f} ms）")
    total_ms = per_tick_sum + baseline_ms
    print(f"[profile] --- 估計每 tick 總成本 ≈ {total_ms:.3f} ms → 約 {1000.0 / total_ms:.1f} tick/秒"
          f"（目標 60 tick/秒 需要 <= 16.67 ms）")

    # 對照組：把我們的 PHYSICS_POST_STEP callback 取消註冊，但讓手臂照樣
    # 被位置目標驅動著實際移動——這樣量到的就是「PhysX 模擬一個正在運動的
    # articulation」本身的成本，跟我們的 Python callback 完全分離。
    print("[profile] === 對照組：手臂在動，但我們的 callback 已取消註冊 ===")
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.deregister_callback(articulation_api._ur10e_step_callback_id)
    articulation_api._ur10e_step_callback_id = None

    moving_targets = positions.copy()
    moving_targets[rmp_ctrl._active_dof_indices] += 0.8  # 給一個明顯的位移目標
    articulation.switch_dof_control_mode("position")
    articulation.set_dof_position_targets(moving_targets[None, :])
    for _ in range(5):
        simulation_app.update()

    start = time.perf_counter()
    for _ in range(200):
        simulation_app.update()
    moving_baseline_ms = (time.perf_counter() - start) * 1000.0 / 200
    print(f"[profile] simulation_app.update()（手臂運動中、無我方 callback）      "
          f"{moving_baseline_ms:8.3f} ms/次")
    print(f"[profile] --- 對照：手臂靜止時是 {baseline_ms:.3f} ms"
          f"，運動中是 {moving_baseline_ms:.3f} ms"
          f"（差 {moving_baseline_ms - baseline_ms:+.3f} ms，這段純粹是 PhysX）")

    # 重新註冊回去，讓後面的 move_to_home() 量測正常
    from isaacsim.core.simulation_manager import SimulationEvent

    articulation_api._ur10e_step_callback_id = SimulationManager.register_callback(
        articulation_api._step_ur10e_motion, event=SimulationEvent.PHYSICS_POST_STEP
    )

    print("[profile] === 實際跑一次 move_to_home() 量真實 tick 速率 ===")
    articulation_api.move_to_home()
    start = time.perf_counter()
    steps = 0
    while not articulation_api.is_motion_complete() and steps < 600:
        simulation_app.update()
        steps += 1
    elapsed = time.perf_counter() - start
    print(f"[profile] move_to_home() 跑了 {steps} tick，耗時 {elapsed:.2f}s "
          f"→ {steps / elapsed:.1f} tick/秒（每 tick {elapsed * 1000 / steps:.2f} ms）")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[profile] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
