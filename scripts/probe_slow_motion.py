"""
scripts/probe_slow_motion.py — 驗證「單次大跳」landing 在不同案例、不同關節的
不同錯誤配置，是不是動態耦合（Coriolis/離心力/動量）造成的：把 DOF 最大速度
壓到非常低（近似準靜態），如果所有案例都能收斂到 <5mm，就證實問題出在快速
過渡時的動力學耦合，不是幾何奇異點或穩態力矩問題。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_slow_motion.py
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

    table_base_path = "/World/SlowMotionTable"
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

    original_max_vel = np.asarray(articulation_api._articulation.get_dof_max_velocities()).copy()
    print(f"original dof max velocities (rad/s)={original_max_vel.tolist()}")
    slow_max_vel = np.full_like(original_max_vel, 0.02)
    articulation_api._articulation.set_dof_max_velocities(slow_max_vel.tolist())
    readback = np.asarray(articulation_api._articulation.get_dof_max_velocities())
    print(f"after slowing down to 0.02 rad/s: {readback.tolist()}")

    for cue_ball in _CASES:
        angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], angle_deg, table_z=table_z)
        wrist0, orientation0, tilt0, crossing0 = compute_tilted_wrist_pose(
            cue_ball, angle_deg, table_z, ball_radius, roll_rad=0.0
        )
        joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]

        # 先把關節開回展開姿態，reposition 到目標 base 位置，再用極低速度
        # 一次性下達完整目標關節角（不分段，純粹測試「同樣的單次跳躍指令，
        # 只是速度極慢」能不能穩定收斂到正確值）。
        articulation_api.move_to_joint_position([0.0] * 7, wrist0.tolist())
        for _ in range(50):
            simulation_app.update()
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        articulation_api.move_to_joint_position(joint_targets, wrist0.tolist())
        settled_step = None
        # 0.02 rad/s 走完最大關節位移（約 1.9 rad）大概需要 95 秒、
        # 60Hz 物理步進約 5700 步，給足夠的步數預算。
        for step in range(6000):
            simulation_app.update()
            if articulation_api.is_motion_complete():
                settled_step = step
                break
        final_position = np.array(articulation_api.get_end_effector_position())
        final_error = float(np.linalg.norm(final_position - wrist0))
        actual_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
        joint_deviation = np.round(actual_joints - np.array(joint_targets), 4).tolist()
        print(f"cue_ball={cue_ball}  [準靜態極慢速] settled_step={settled_step}  "
              f"final_error={final_error*1000:.2f} mm  joint_deviation_rad={joint_deviation}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
