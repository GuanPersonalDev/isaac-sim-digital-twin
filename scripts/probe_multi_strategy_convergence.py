"""
scripts/probe_multi_strategy_convergence.py — 「多策略嘗試、挑最好的」驗證：
不再找單一根因/單一參數，改成每個 flat（tilt=0）案例都嘗試兩種已驗證過的
到位方式：

  A. 單次大跳：目前 production code（DemoTableOrchestrator._execute_aim()）
     的做法，reposition 後直接一次 move_to_joint_position() 跳到目標。
  B. 分段 6 步線性內插逼近。

用 is_motion_complete()（5mm 容許誤差）判斷是否收斂；兩個都不收斂則挑
final_error 較小的一個。這個「多策略挑最好」的做法跟本次會話早前解決
撞庫邊碰撞問題時用的 roll 候選值搜尋（scan_elevated_bridge_approach.py 的
ROLL_CANDIDATES_DEG）是同一種工程手法：不強求找到單一根因，而是讓系統
對每個案例自動選出實際有效的路徑。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_multi_strategy_convergence.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

_TARGET_BALL = (0.0, 0.635)
_CUE_BALL_X_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)
_CUE_BALL_Y_GRID = (-1.1, -0.6, -0.1, 0.4, 0.9)


def _shot_angle_deg(cue_ball, target):
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.services.base_placement_calculator import CANONICAL_REST_JOINTS, compute_base_pose

    from scripts.scan_elevated_bridge_approach import compute_tilted_wrist_pose

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/MultiStrategyTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_z = 0.0
    ball_radius = 0.028575

    robot_prim_path = BarrettWamRobot.get_prim_path(table_base_path)
    end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, BarrettWamRobot
    )
    robot = robot_manager.get_robot()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    def _wait_converge(target_pos, max_steps):
        for step in range(max_steps):
            simulation_app.update()
            if articulation_api.is_motion_complete():
                return step
        return None

    def _reset_to_extended_pose(dummy_target):
        # move_to_joint_position 需要一個 target_end_effector_position 參數
        # 才能驅動 is_motion_complete() 邏輯，但這裡只是要把關節開回展開姿態
        # 墊底，不需要真的等到那個（不存在對應的）cartesian 目標收斂，固定
        # 跑 400 步（在正常 DOF 最大速度 2 rad/s 下，最大關節位移 ~1.9 rad
        # 綽綽有餘走完）。
        articulation_api.move_to_joint_position([0.0] * 7, dummy_target.tolist())
        for _ in range(400):
            simulation_app.update()

    def _strategy_single_jump(joint_targets, target_pos):
        articulation_api.move_to_joint_position(joint_targets, target_pos.tolist())
        settled_step = _wait_converge(target_pos, max_steps=1000)
        final_position = np.array(articulation_api.get_end_effector_position())
        final_error = float(np.linalg.norm(final_position - target_pos))
        return settled_step, final_error

    def _strategy_staged(joint_targets, target_pos, num_waypoints=6, per_wp_steps=150):
        start_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0].copy()
        target_joints = np.array(joint_targets)
        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            waypoint_joints = (start_joints + (target_joints - start_joints) * t).tolist()
            articulation_api.move_to_joint_position(waypoint_joints, target_pos.tolist())
            for _ in range(per_wp_steps):
                simulation_app.update()
        articulation_api.move_to_joint_position(joint_targets, target_pos.tolist())
        settled_step = _wait_converge(target_pos, max_steps=500)
        final_position = np.array(articulation_api.get_end_effector_position())
        final_error = float(np.linalg.norm(final_position - target_pos))
        return settled_step, final_error

    strategies = [("single_jump", _strategy_single_jump), ("staged_6wp", _strategy_staged)]

    results = []
    for cue_x in _CUE_BALL_X_GRID:
        for cue_y in _CUE_BALL_Y_GRID:
            cue_ball = (cue_x, cue_y)
            angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
            base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], angle_deg, table_z=table_z)
            wrist0, orientation0, tilt0, crossing0 = compute_tilted_wrist_pose(
                cue_ball, angle_deg, table_z, ball_radius, roll_rad=0.0
            )
            if tilt0 is None or tilt0 > 1e-6:
                print(f"cue_ball={cue_ball}: tilt0={tilt0}，不是 flat 案例，跳過")
                continue

            joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]

            best = None
            for name, strategy_fn in strategies:
                _reset_to_extended_pose(wrist0)
                robot.reposition(base_position)
                for _ in range(30):
                    simulation_app.update()
                settled_step, final_error = strategy_fn(joint_targets, wrist0)
                converged = settled_step is not None
                print(f"  cue_ball={cue_ball} strategy={name:12s} converged={converged} "
                      f"settled_step={settled_step} final_error={final_error*1000:.2f} mm")
                if best is None or (converged and not best[2]) or (converged == best[2] and final_error < best[3]):
                    best = (name, settled_step, converged, final_error)
                if converged:
                    break
            results.append((cue_ball, best))
            print(f"cue_ball={cue_ball}  => 挑選策略={best[0]}  converged={best[2]}  final_error={best[3]*1000:.2f} mm")

    n_converged = sum(1 for _, b in results if b[2])
    print(f"\n=== FINAL SUMMARY: flat 案例共 {len(results)} 個，多策略挑選後收斂（<5mm）{n_converged} 個 ===")
    for cue_ball, b in results:
        print(f"  {cue_ball}: strategy={b[0]} converged={b[2]} final_error={b[3]*1000:.2f} mm")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
