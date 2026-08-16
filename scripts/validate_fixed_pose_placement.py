"""
scripts/validate_fixed_pose_placement.py — #233：驗證 base_placement_calculator
的固定姿態公式端到端正確，並檢查球桿實際掛上去是否水平。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/validate_fixed_pose_placement.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


_TARGET_BALL = (0.0, 0.635)
_PROBE_POINTS = {
    "far_corner": (0.606425, -1.241425),
}


def _shot_angle_deg(cue_ball, target):
    import math

    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.services.asset_utility import CUE_STICK_PATH
    from core.services.base_placement_calculator import (
        CANONICAL_REST_JOINTS,
        compute_base_pose,
        required_grip_position,
    )

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()

    for label, cue_ball in _PROBE_POINTS.items():
        print(f"\n=== {label} ===")
        angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
        expected_grip = required_grip_position(cue_ball[0], cue_ball[1], angle_deg)
        base_position, base_yaw_rad = compute_base_pose(
            cue_ball[0], cue_ball[1], angle_deg, table_z=0.0
        )
        print(f"  angle_deg={angle_deg:.3f}  expected_grip={expected_grip}")
        print(f"  base_position={base_position}  base_yaw_rad={base_yaw_rad:.4f}")

        base_path = f"/World/Validate_{label}"
        robot = BarrettWamRobot(base_path, stage_api, articulation_api=None, position=base_position)
        robot_prim_path = BarrettWamRobot.get_prim_path(base_path)
        end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(base_path)

        cue_stick_path = base_path + "/CueStick"
        stage_api.create_reference_prim(cue_stick_path, CUE_STICK_PATH)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(5):
            simulation_app.update()

        articulation = Articulation(paths=robot_prim_path)
        articulation.switch_dof_control_mode("position")
        q = [base_yaw_rad] + list(CANONICAL_REST_JOINTS)
        articulation.set_dof_position_targets(np.array([q]))
        for _ in range(150):
            simulation_app.update()

        end_effector = RigidPrim(paths=end_effector_prim_path)
        positions, orientations = end_effector.get_world_poses()
        tip_world = np.array(positions[0].list())
        print(f"  actual_tip_world={tip_world.tolist()}")
        error = float(np.linalg.norm(tip_world[:2] - np.array(expected_grip)))
        print(f"  xy_error={error:.5f} m")
        print(f"  z_error={abs(tip_world[2] - 0.028575):.5f} m (target table_z+ball_radius=0.028575)")

        # 對齊球桿到腕部（跟 TableRobotManager 相同的機制），檢查桿身方向是否水平。
        stage_api.align_prim_to_target(cue_stick_path, end_effector_prim_path)
        for _ in range(20):
            simulation_app.update()

        from pxr import UsdGeom, Usd

        cue_prim = stage.GetPrimAtPath(cue_stick_path)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        cue_world = xform_cache.GetLocalToWorldTransform(cue_prim)
        # 桿身沿本地 Y 延伸（見 ball_stick.usda），本地 Y 軸在世界座標下的方向
        local_y_world = np.array([cue_world[1][0], cue_world[1][1], cue_world[1][2]])
        local_y_world_unit = local_y_world / np.linalg.norm(local_y_world)
        print(f"  cue_stick_local_Y_axis_in_world={local_y_world_unit.tolist()}")
        print(f"  cue_stick_tilt_from_horizontal_deg={np.degrees(np.arcsin(abs(local_y_world_unit[2]))):.2f}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
