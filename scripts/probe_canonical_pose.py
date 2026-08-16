"""
scripts/probe_canonical_pose.py — #233 前置探測：手動試誤找一個合法的
「固定 canonical 姿態」（joint-space 直接下 position target，不依賴差動 IK
收斂），量出 tip 相對 base 原點的向量，供改寫 base_placement_calculator.py 用。

只印數字，不做任何 pass/fail 判斷——用來目視挑一組看起來合理（tip 夠低、
夠水平、沒有明顯貼近關節限位）的候選姿態。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_canonical_pose.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# 7 DOF 順序（見 assets/barrett_wam/wam7.urdf）：
# [base_yaw, shoulder_pitch, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_pitch, palm_yaw]
# 限位（rad）：
#   base_yaw      [-2.6, 2.6]
#   shoulder_pitch[-1.985, 1.985]
#   shoulder_yaw  [-2.8, 2.8]
#   elbow_pitch   [-0.9, 3.14159]
#   wrist_yaw     [-4.55, 1.25]
#   wrist_pitch   [-1.5707, 1.5707]
#   palm_yaw      [-3.0, 3.0]
_CANDIDATES = {
    "Y_reference_yaw0": [0.0, 1.9, 0.0, 1.8, 0.0, 0.0, 0.0],
    "Z_yaw_plus_0.3": [0.3, 1.9, 0.0, 1.8, 0.0, 0.0, 0.0],
}


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from core.models.barrett_wam_robot import BarrettWamRobot

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    base_path = "/World/PoseProbe"
    robot = BarrettWamRobot(base_path, stage_api, articulation_api=None, position=(0.0, 0.0, 0.0))
    robot_prim_path = BarrettWamRobot.get_prim_path(base_path)
    end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(base_path)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=robot_prim_path)
    end_effector = RigidPrim(paths=end_effector_prim_path)
    articulation.switch_dof_control_mode("position")

    dof_lower, dof_upper = None, None
    for attr in ("get_dof_position_lower_limits", "get_dof_position_limits"):
        if hasattr(articulation, attr):
            print(f"  (found limits accessor: {attr})")
            break

    for label, q in _CANDIDATES.items():
        print(f"\n=== {label}: q={q} ===")
        articulation.set_dof_position_targets(np.array([q]))
        for _ in range(150):
            simulation_app.update()

        actual_q = np.asarray(articulation.get_dof_positions())[0]
        positions, orientations = end_effector.get_world_poses()
        tip_world = np.array(positions[0].list())
        tip_orient = np.array(orientations[0].list())

        print(f"  actual_dof_positions={np.round(actual_q, 3).tolist()}")
        print(f"  tip_world_position={tip_world.tolist()}")
        print(f"  tip_world_orientation(wxyz)={tip_orient.tolist()}")
        print(f"  tip_height_above_base_origin={tip_world[2]:.4f} m")
        print(f"  tip_horizontal_dist_from_base_origin={float(np.hypot(tip_world[0], tip_world[1])):.4f} m")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
