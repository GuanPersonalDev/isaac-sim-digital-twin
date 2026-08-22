"""
scripts/verify_solver_iteration_fix.py — 驗證 wam7.usda 資產本身（不靠任何
runtime 補丁）套用 solverPositionIterationCount/solverVelocityIterationCount=255
之後，(-0.25, -0.1) 這個之前卡住的 flat 案例是否能正常收斂。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_solver_iteration_fix.py
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
_CASES = [(-0.25, -0.1), (0.0, -0.1), (0.0, 0.4), (0.25, -0.1), (-0.5, -1.1)]


def _shot_angle_deg(cue_ball, target):
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import Usd, UsdPhysics, Sdf

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

    table_base_path = "/World/VerifyFixTable"
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

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            pos_attr = prim.GetAttribute("physxArticulation:solverPositionIterationCount")
            vel_attr = prim.GetAttribute("physxArticulation:solverVelocityIterationCount")
            print(f"確認資產已 author solver iteration count：position={pos_attr.Get() if pos_attr else None} "
                  f"velocity={vel_attr.Get() if vel_attr else None}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    for cue_ball in _CASES:
        angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], angle_deg, table_z=table_z)
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        wrist0, orientation0, tilt0, crossing0 = compute_tilted_wrist_pose(
            cue_ball, angle_deg, table_z, ball_radius, roll_rad=0.0
        )
        if tilt0 is None or tilt0 > 1e-6:
            print(f"cue_ball={cue_ball}: tilt0={tilt0}，不是 flat 案例，跳過（這個驗證只測 CANONICAL_REST_JOINTS 路徑）")
            continue

        joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
        articulation_api.move_to_joint_position(joint_targets, wrist0.tolist())
        settled_step = None
        for step in range(1000):
            simulation_app.update()
            if articulation_api.is_motion_complete():
                settled_step = step
                break
        final_position = np.array(articulation_api.get_end_effector_position())
        final_error = float(np.linalg.norm(final_position - wrist0))
        actual_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
        joint_deviation = np.round(actual_joints - np.array(joint_targets), 4).tolist()
        print(f"cue_ball={cue_ball}  settled_step={settled_step}  final_error={final_error*1000:.2f} mm  "
              f"joint_deviation_rad={joint_deviation}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
