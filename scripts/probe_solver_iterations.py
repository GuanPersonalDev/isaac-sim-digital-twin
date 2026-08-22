"""
scripts/probe_solver_iterations.py — 驗證「PhysX solver iteration count 太低」
是不是 (-0.25, -0.1) 案例 wrist_yaw 卡在 +0.096 rad 回不去 0 的根因。

set_solver_iteration_counts() 走的是 USD backend（見 articulation.py 裡的
「Backends: usd」註記），寫入的是 physxArticulation:solverPositionIterationCount
這個 USD attribute，PhysX 只在 articulation 建立/cook 的當下讀取它一次，不是
每個 tick 動態生效——所以這個實驗必須在 timeline.play() 之前、Articulation
tensor view 建立之前就把這個 attribute 設好，不能像前一個 probe 腳本那樣在
模擬中途才調。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_solver_iterations.py
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
_CUE_BALL = (-0.25, -0.1)


def _shot_angle_deg(cue_ball, target):
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import Usd, UsdPhysics, UsdGeom, Sdf

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

    table_base_path = "/World/SolverIterProbeTable"
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

    # 在 timeline.play() 之前，直接找出帶有 ArticulationRootAPI 的 prim，把
    # solverPositionIterationCount / solverVelocityIterationCount 這兩個
    # physxArticulation attribute 從預設值（Isaac Sim 預設通常是 4 / 1）拉高，
    # 讓 PhysX 在 cook articulation 的當下就用新的迭代次數。
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    articulation_root_prim = None
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_root_prim = prim
            break
    if articulation_root_prim is None:
        print("WARNING: 找不到帶 ArticulationRootAPI 的 prim，無法設定 solver iteration count")
    else:
        print(f"ArticulationRootAPI prim = {articulation_root_prim.GetPath()}")
        pos_attr = articulation_root_prim.GetAttribute("physxArticulation:solverPositionIterationCount")
        vel_attr = articulation_root_prim.GetAttribute("physxArticulation:solverVelocityIterationCount")
        print(f"BEFORE: solverPositionIterationCount={pos_attr.Get() if pos_attr else 'N/A(attr不存在)'} "
              f"solverVelocityIterationCount={vel_attr.Get() if vel_attr else 'N/A(attr不存在)'}")
        if not pos_attr:
            articulation_root_prim.ApplyAPI("PhysxArticulationAPI")
            pos_attr = articulation_root_prim.GetAttribute("physxArticulation:solverPositionIterationCount")
            vel_attr = articulation_root_prim.GetAttribute("physxArticulation:solverVelocityIterationCount")
        pos_attr.Set(255)
        vel_attr.Set(255)
        print(f"AFTER: solverPositionIterationCount={pos_attr.Get()} solverVelocityIterationCount={vel_attr.Get()}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    angle_deg = _shot_angle_deg(_CUE_BALL, _TARGET_BALL)
    base_position, base_yaw_rad = compute_base_pose(_CUE_BALL[0], _CUE_BALL[1], angle_deg, table_z=table_z)
    robot.reposition(base_position)
    for _ in range(30):
        simulation_app.update()

    wrist0, orientation0, tilt0, crossing0 = compute_tilted_wrist_pose(
        _CUE_BALL, angle_deg, table_z, ball_radius, roll_rad=0.0
    )
    joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
    print(f"joint_targets={joint_targets}  wrist0={wrist0.tolist()}")

    def _run_once(label, targets, target_pos, max_steps=1000, log_every=20):
        articulation_api.move_to_joint_position(targets, target_pos.tolist())
        settled_step = None
        for step in range(max_steps):
            simulation_app.update()
            current = np.array(articulation_api.get_end_effector_position())
            err = float(np.linalg.norm(current - target_pos))
            complete = articulation_api.is_motion_complete()
            if step % log_every == 0 or complete:
                actual_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
                print(f"  [{label}] step={step:4d} err={err:.5f} m complete={complete} "
                      f"joints={np.round(actual_joints, 4).tolist()}")
            if complete and settled_step is None:
                settled_step = step
                break
        final = np.array(articulation_api.get_end_effector_position())
        final_err = float(np.linalg.norm(final - target_pos))
        print(f"  [{label}] FINAL settled_step={settled_step} final_err={final_err:.5f} m")
        return final_err

    print("\n=== 單次大跳（展開姿態 -> 目標）===")
    _run_once("single_jump", joint_targets, wrist0)

    # 換一個全新的 robot base 位置（避免沿用剛剛單次大跳後的殘留狀態），
    # 改用分段小步：從目前關節角度往目標線性內插 8 個中繼點，每個中繼點
    # 都等 is_motion_complete() 或跑滿較短的步數才前進下一步，驗證「瞬態
    # 動力學／solver 沒收斂完就進入下一段」是不是 shoulder_pitch 殘留誤差
    # 的成因。
    print("\n=== 分段小步逼近（8 個中繼點）===")
    robot.reposition((base_position[0] + 5.0, base_position[1], base_position[2]))
    for _ in range(30):
        simulation_app.update()
    start_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0].copy()
    target_joints = np.array(joint_targets)
    num_waypoints = 8
    for i in range(1, num_waypoints + 1):
        t = i / num_waypoints
        waypoint_joints = (start_joints + (target_joints - start_joints) * t).tolist()
        _run_once(f"staged_wp{i}/{num_waypoints}", waypoint_joints, wrist0, max_steps=300, log_every=100)


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
