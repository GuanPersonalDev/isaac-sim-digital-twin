"""
scripts/probe_staged_joint_motion.py — 驗證「分段小步逼近」能不能取代
「從展開姿態一次大跳到 CANONICAL_REST_JOINTS」這個動作，解決各案例卡在
不同關節、不同穩定但錯誤配置的問題（solver iteration 修正已經 author 進
wam7.usda，這裡驗證的是另一層、動作規劃本身的修正）。

對每個案例：
  1. baseline：目前 production code（core/services/table_orchestrator.py
     DemoTableOrchestrator._execute_aim()）的做法——reposition 後直接一次
     move_to_joint_position() 跳到目標。
  2. staged：改成從目前關節角度往目標線性內插 N 個中繼點，每個中繼點都用
     move_to_joint_position() 個別下達、跑到收斂或跑滿較短步數才進下一段。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_staged_joint_motion.py
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
_CASES = [(-0.25, -0.1), (0.0, 0.4), (0.25, -0.1)]


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

    table_base_path = "/World/StagedMotionTable"
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

    def _move_staged(joint_targets, target_pos, num_waypoints=6, per_wp_steps=150):
        start_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0].copy()
        target_joints = np.array(joint_targets)
        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            waypoint_joints = (start_joints + (target_joints - start_joints) * t).tolist()
            articulation_api.move_to_joint_position(waypoint_joints, target_pos.tolist())
            for _ in range(per_wp_steps):
                simulation_app.update()
        # 最後一段用真正的目標關節角＋較長的收斂預算，確保有機會真的落在
        # is_motion_complete() 的容許帶內，不是卡在中繼點的近似值。
        articulation_api.move_to_joint_position(joint_targets, target_pos.tolist())
        return _wait_converge(target_pos, max_steps=500)

    for cue_ball in _CASES:
        angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], angle_deg, table_z=table_z)
        wrist0, orientation0, tilt0, crossing0 = compute_tilted_wrist_pose(
            cue_ball, angle_deg, table_z, ball_radius, roll_rad=0.0
        )
        joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]

        # baseline：跟 production _execute_aim() 一樣，reposition 後直接一次
        # move_to_joint_position() 跳到目標。
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()
        articulation_api.move_to_joint_position(joint_targets, wrist0.tolist())
        settled_step = _wait_converge(wrist0, max_steps=1000)
        final_position = np.array(articulation_api.get_end_effector_position())
        baseline_error = float(np.linalg.norm(final_position - wrist0))
        print(f"cue_ball={cue_ball}  [baseline 單次大跳] settled_step={settled_step}  "
              f"final_error={baseline_error*1000:.2f} mm")

        # staged：先把關節開回展開姿態（0,0,...,0），重現 baseline 一樣的
        # 「大距離跳躍」起點，但 base 位置維持不變（wrist0 是相對球桌的世界
        # 座標目標，換了 base 位置 wrist0 就不對了，上一輪 probe_solver_
        # iterations.py 就是在這裡犯過同樣的量測基準錯誤）。
        articulation_api.move_to_joint_position([0.0] * 7, wrist0.tolist())
        for _ in range(300):
            simulation_app.update()
        settled_step_staged = _move_staged(joint_targets, wrist0)
        final_position_staged = np.array(articulation_api.get_end_effector_position())
        staged_error = float(np.linalg.norm(final_position_staged - wrist0))
        print(f"cue_ball={cue_ball}  [staged 分段6步] settled_step={settled_step_staged}  "
              f"final_error={staged_error*1000:.2f} mm")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
