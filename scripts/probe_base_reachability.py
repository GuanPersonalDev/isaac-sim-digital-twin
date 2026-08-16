"""
scripts/probe_base_reachability.py — #233 前置探測：單點差動 IK 可達性

只驗證一件事：`base_placement_calculator` 算出的基座位置，WAM7 用專案既有的
差動 IK（`ArticulationAPIImpl`）是否真的能把末端執行器移到握把需求點——不含
球檯碰撞檢查、不含完整 Kitchen 網格掃描（那是 #233 完整範圍，這裡只是先確認
「理論上有解」跟「差動 IK 實際收斂」對不對得起來）。

只測位置，不鎖定最終姿態方向（把目標 orientation 設成起始 orientation，讓
P controller 的角速度項恆為 0）——orientation-constrained 的完整驗證留給
#233 本體，這裡的目的只是判斷「末端執行器的原點」搆不搆得到。

用法（獨立執行，headless，不需要 GUI/RDP）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_base_reachability.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math


# Kitchen 兩個代表角落（docs/issue-180-reachability-analysis.md 第九節同一組數字）
_TARGET_BALL = (0.0, 0.635)
_PROBE_POINTS = {
    "near_corner": (0.606425, -0.635),
    "far_corner": (0.606425, -1.241425),
}


def _shot_angle_deg(cue_ball: tuple[float, float], target: tuple[float, float]) -> float:
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _run_probe() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.services.base_placement_calculator import (
        compute_base_position,
        required_grip_position,
    )

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    table_z = 0.0
    ball_radius = 0.028575

    results = {}

    for label, cue_ball in _PROBE_POINTS.items():
        angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
        grip_x, grip_y = required_grip_position(cue_ball[0], cue_ball[1], angle_deg)
        target_position = [grip_x, grip_y, table_z + ball_radius]

        base_x, base_y, base_z = compute_base_position(
            cue_ball[0], cue_ball[1], angle_deg, table_z=table_z, ball_radius=ball_radius
        )

        base_path = f"/World/Probe_{label}"
        robot_prim_path = BarrettWamRobot.get_prim_path(base_path)
        end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(base_path)

        print(f"\n=== {label} ===")
        print(f"  cue_ball={cue_ball}  angle_deg={angle_deg:.3f}")
        print(f"  target_position(grip)={target_position}")
        print(f"  base_position={(base_x, base_y, base_z)}")

        robot = BarrettWamRobot(
            base_path, stage_api, articulation_api=None, position=(base_x, base_y, base_z)
        )
        results[label] = {
            "robot_prim_path": robot_prim_path,
            "end_effector_prim_path": end_effector_prim_path,
            "target_position": target_position,
        }

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    for label, info in results.items():
        print(f"\n=== solving IK: {label} ===")
        api = ArticulationAPIImpl(info["robot_prim_path"], info["end_effector_prim_path"])
        api.initialize()

        start_orientation = api._get_end_effector_world_orientation().tolist()
        start_position = np.array(api.get_end_effector_position())
        target_position = np.array(info["target_position"])
        print(f"  start_position={start_position.tolist()}")

        # 兩段式：先用一串沿直線的中繼點分段收斂，避開「一步跳完整段 0.9m」
        # 讓 Jacobian 在接近完全伸直的奇異點附近失穩的問題，而不是一次瞄準
        # 最終目標。每個中繼點用同一套差動 IK 收斂（或逾時就前進到下一個）。
        num_waypoints = 12
        max_steps_per_waypoint = 150
        converged = False
        total_steps = 0
        for i in range(1, num_waypoints + 1):
            waypoint = (start_position + (target_position - start_position) * i / num_waypoints).tolist()
            api.move_to_pose(waypoint, start_orientation)
            for step in range(max_steps_per_waypoint):
                simulation_app.update()
                total_steps += 1
                if api.is_motion_complete():
                    break

        # 最後再對準真正的目標點收斂一次（中繼點的 tolerance 是對 waypoint 不是對終點）。
        api.move_to_pose(info["target_position"], start_orientation)
        max_final_steps = 300
        for step in range(max_final_steps):
            simulation_app.update()
            total_steps += 1
            if api.is_motion_complete():
                converged = True
                break

        final_position = api.get_end_effector_position()
        error = float(
            np.linalg.norm(np.array(final_position) - np.array(info["target_position"]))
        )
        print(f"  converged={converged}  total_steps={total_steps}  final_position={final_position}")
        print(f"  final_position_error={error:.5f} m (tolerance={api.POSITION_TOLERANCE})")
        results[label]["converged"] = converged
        results[label]["final_error"] = error

    print("\n=== SUMMARY ===")
    for label, info in results.items():
        status = "REACHABLE" if info["converged"] else "NOT REACHABLE"
        print(f"  {label}: {status} (final_error={info['final_error']:.5f} m)")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run_probe()
    finally:
        simulation_app.close()
