"""
scripts/scan_rail_collisions.py — #233 排除範圍的補測：對一組母球位置網格，
用真正的 BilliardTable（含庫邊碰撞幾何）＋ TableRobotManager（含 CueStick／
FixedJoint），走跟 DemoTableOrchestrator._execute_aim() 完全一樣的路徑
（reposition 基座 → move_to_joint_position），量測哪些母球位置會讓球桿撞到
庫邊或其他物件。

背景：isaac-2026-08-20-22-56.png 截圖發現球桿被庫邊擋住，之前所有探測腳本
（probe_canonical_pose.py／validate_fixed_pose_placement.py／
probe_palm_yaw_correction.py）都只測機器人＋球桿本身，沒有放真正的球桌，
這是第一支把球桌放進場景一起測的腳本，用來量化「範圍有多大」。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/scan_rail_collisions.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

# TABLE_WIDTH=1.27（X 半寬 0.635）、TABLE_LENGTH=2.54（Y 半長 1.27），見
# core/services/spread_score_calculator.py。機器人固定站在桌子 X+1.5 處。
_TARGET_BALL = (0.0, 0.635)
_CUE_BALL_X_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)
_CUE_BALL_Y_GRID = (-1.1, -0.6, -0.1, 0.4, 0.9)


def _shot_angle_deg(cue_ball: tuple[float, float], target: tuple[float, float]) -> float:
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _run() -> None:
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.contact_event import ContactEvent
    from core.services.base_placement_calculator import (
        CANONICAL_REST_JOINTS,
        compute_base_pose,
        required_grip_position,
    )
    from core.models.table_ball_set import TableBallSet

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/RailScanTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_z = 0.0

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

    contacts: list[ContactEvent] = []

    def _on_contact(event: ContactEvent) -> None:
        contacts.append(event)

    physics_api.subscribe_contact_events(_on_contact)

    results = []
    for cue_x in _CUE_BALL_X_GRID:
        for cue_y in _CUE_BALL_Y_GRID:
            cue_ball = (cue_x, cue_y)
            angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
            base_position, base_yaw_rad = compute_base_pose(
                cue_ball[0], cue_ball[1], angle_deg, table_z=table_z
            )
            grip_x, grip_y = required_grip_position(cue_ball[0], cue_ball[1], angle_deg)

            base_yaw_limit = 2.6
            if abs(base_yaw_rad) > base_yaw_limit:
                results.append(
                    {"cue_ball": cue_ball, "angle_deg": angle_deg, "status": "OUT_OF_YAW_LIMIT"}
                )
                print(f"cue_ball={cue_ball}  angle_deg={angle_deg:7.2f}  OUT_OF_YAW_LIMIT (base_yaw_rad={base_yaw_rad:.3f})")
                continue

            contacts.clear()
            robot.reposition(base_position)
            joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
            articulation_api.move_to_joint_position(
                joint_targets, [grip_x, grip_y, table_z + TableBallSet.DEFAULT_BALL_RADIUS]
            )
            for _ in range(250):
                simulation_app.update()

            collided = len(contacts) > 0
            partners = sorted({c.collider_path_b if c.collider_path_a == cue_stick_prim_path else c.collider_path_a for c in contacts})
            status = "COLLISION" if collided else "OK"
            results.append(
                {"cue_ball": cue_ball, "angle_deg": angle_deg, "status": status, "partners": partners}
            )
            print(
                f"cue_ball={cue_ball}  angle_deg={angle_deg:7.2f}  base_yaw_rad={base_yaw_rad:6.3f}  "
                f"{status}" + (f"  partners={partners}" if collided else "")
            )

    physics_api.unsubscribe_contact_events()

    n_collision = sum(1 for r in results if r["status"] == "COLLISION")
    n_out_of_limit = sum(1 for r in results if r["status"] == "OUT_OF_YAW_LIMIT")
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n=== SUMMARY: total={len(results)}  OK={n_ok}  COLLISION={n_collision}  OUT_OF_YAW_LIMIT={n_out_of_limit} ===")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
