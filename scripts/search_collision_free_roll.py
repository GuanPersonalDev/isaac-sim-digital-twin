"""
scripts/search_collision_free_roll.py — 對每個 Y 值段落，依「數值 IK 全程
餘裕」由高到低嘗試候選 roll 值，用真實物理模擬（含正式的碰撞回報，跟
scripts/verify_swing_trajectory.py 同一套 `enable_contact_reporting`／
`ContactEvent` 機制）逐一測試 AIM 是否 (a) 收斂 (b) 無碰撞，找到第一個
兩者都成立的候選就採用，換下一個 Y 值段落。

背景：docs/issue-180-reachability-analysis.md 第十五節記錄了教訓——純數值
IK（scripts/wam7_kinematics.py）沒有建模手臂本體碰撞，之前只用 IK 可達性
排出來的 `_ROLL_LOOKUP_GRID` 在完整 20 案例驗收網格上大多數是 COLLISION。
碰撞沒辦法用純數值方法加速篩選（需要完整的機器人/球檯 USD 幾何），只能
逐點真實模擬驗證；但 IK 排序至少能把「這個 Y 段落照 margin 高低該先試哪些
roll」排好，把窮舉範圍從 24 個候選縮小到「前幾個大機率沒問題的」，比完全
無方向的試誤省時間。

候選清單來自 scripts/search_roll_for_full_swing.py 最近一次的完整輸出（
已手動依 margin 由高到低排序，margin=0.0000 的退化解略過不試）。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_collision_free_roll.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_AIM_MAX_STEPS = 4000

# 每個 Y 值段落的候選 roll（deg），依 scripts/search_roll_for_full_swing.py
# 的全程最小關節餘裕由高到低排序，margin=0.0000 的退化解不列入。
# ⚠️ 這個候選清單本身（IK 可達性排序）已證實跟 X 無關，但「哪一個候選
# 不會撞庫邊/袋口」跟 X 有關（碰撞取決於手臂在世界座標系裡離哪個庫邊/
# 袋口近，不是只看關節構型），所以要對每個 (X,Y) 組合分別測，不能只測一個
# 代表 X 就套用到全部 X。
_CANDIDATES_BY_Y = {
    -0.9382125: [165, -180, -165, -150, -135, 150, -120, -105],
    -0.635: [150, 165, -180, -165, -150, 135, -135],
}
_X_VALUES = [-0.606425, 0.0, 0.606425]


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import Usd, UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.contact_event import ContactEvent
    from core.services.base_placement_calculator import (
        CANONICAL_FLAT_ORIENTATION, CANONICAL_REST_JOINTS, compute_base_pose,
        compute_canonical_wrist_position,
    )
    from core.services import cue_pose_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/SearchCollisionFreeRollTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))

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

    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            physics_api.enable_contact_reporting(prim.GetPath().pathString)
    contacts: list[ContactEvent] = []
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    def _is_self_path(p: str) -> bool:
        return p.startswith(robot_prim_path) or p.startswith(cue_stick_prim_path)

    _RESET_JOINTS = np.array([[0.0, *CANONICAL_REST_JOINTS]])

    def _hard_reset_joints():
        articulation_api._articulation.set_dof_positions(_RESET_JOINTS)
        articulation_api._articulation.set_dof_velocities(np.zeros((1, 7)))
        for _ in range(10):
            simulation_app.update()

    def _try_candidate(cue_ball_xy, roll_deg):
        shot_angle_deg = 0.0
        roll_rad = math.radians(roll_deg)
        _hard_reset_joints()
        base_position, base_yaw_rad = compute_base_pose(cue_ball_xy[0], cue_ball_xy[1], shot_angle_deg, _TABLE_Z)
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball_xy, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
        )
        if tilt_rad is None:
            return "GEOMETRICALLY_INFEASIBLE"

        safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
        safe_target_position = list(compute_canonical_wrist_position(base_position, 0.0))
        bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
            safe_target_position, list(CANONICAL_FLAT_ORIENTATION),
            cue_ball_xy, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, roll_rad=roll_rad,
        )
        if bridge_waypoints is None:
            return "GEOMETRICALLY_INFEASIBLE"

        contacts.clear()
        articulation_api.move_through_poses(
            bridge_waypoints, preceding_joint_targets=(safe_joint_targets, safe_target_position)
        )
        for step in range(_AIM_MAX_STEPS):
            simulation_app.update()
            if articulation_api.is_motion_complete():
                break
        else:
            return "EXHAUSTED"

        if articulation_api.did_last_motion_timeout():
            return "AIM_TIMEOUT"

        blocking_partners = {
            p for c in contacts for p in (c.collider_path_a, c.collider_path_b)
            if not _is_self_path(p) and "Surface" not in p
        }
        if blocking_partners:
            return f"COLLISION({sorted(blocking_partners)[:2]})"

        actual_position = np.array(articulation_api.get_end_effector_position())
        pos_diff = float(np.linalg.norm(actual_position - wrist))
        if pos_diff > 0.05:
            return f"REJECTED(pos_diff={pos_diff:.4f})"

        return "OK"

    results = {}
    for y, candidates in _CANDIDATES_BY_Y.items():
        for x in _X_VALUES:
            cue_ball_xy = (x, y)
            print("=" * 100)
            print(f"[x={x}, y={y}] 候選（依 margin 排序）：{candidates}")
            winner = None
            for roll_deg in candidates:
                status = _try_candidate(cue_ball_xy, roll_deg)
                print(f"  roll={roll_deg:+4d}deg => {status}")
                if status == "OK":
                    winner = roll_deg
                    break
            if winner is None:
                print(f"  => (x={x},y={y}) 所有候選都失敗，需要擴大候選範圍")
            else:
                print(f"  => (x={x},y={y}) 採用 roll={winner}deg")
            results[(x, y)] = winner

    print("=" * 100)
    print("彙總（可直接用來更新 _ROLL_LOOKUP_GRID）：")
    for (x, y), roll_deg in results.items():
        print(f"  x={x} y={y}  roll={roll_deg}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
