"""
scripts/probe_room_clearance.py — 查詢房間牆壁／地板跟球桌 Head 端（Y=-1.1
附近撞牆案例）的實際世界座標包圍盒，量出淨空距離，判斷要把機器人基座往哪個
方向、移動多少才能避開。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_room_clearance.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run() -> None:
    import numpy as np
    import omni.usd
    import isaacsim.core.utils.bounds as bounds_util
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl

    from core.models.billiard_table import BilliardTable
    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.table_robot_manager import TableRobotManager
    from core.services.base_placement_calculator import compute_base_pose

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/ClearanceProbeTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))

    cache = bounds_util.create_bbox_cache()

    def _aabb(prim_path):
        aabb = bounds_util.compute_aabb(cache, prim_path=prim_path)
        return {
            "min": (aabb[0], aabb[1], aabb[2]),
            "max": (aabb[3], aabb[4], aabb[5]),
        }

    targets = {
        "Cushion_Head": f"{table_base_path}/Table/billiard_env_unit/BilliardEnv/BilliardTable/Cushion_Head",
        "Cushion_Foot": f"{table_base_path}/Table/billiard_env_unit/BilliardEnv/BilliardTable/Cushion_Foot",
        "BilliardTable(whole)": f"{table_base_path}/Table/billiard_env_unit/BilliardEnv/BilliardTable",
        "SimpleRoom(whole)": f"{table_base_path}/Table/billiard_env_unit/BilliardEnv/SimpleRoom",
        "wood_wall_308": f"{table_base_path}/Table/billiard_env_unit/BilliardEnv/SimpleRoom/Towel_Room01_wood_wall_308",
        "wood_wall_318": f"{table_base_path}/Table/billiard_env_unit/BilliardEnv/SimpleRoom/Towel_Room01_wood_wall_318",
        "wood_wall_320": f"{table_base_path}/Table/billiard_env_unit/BilliardEnv/SimpleRoom/Towel_Room01_wood_wall_320",
    }

    print("=== 世界座標包圍盒（桌子中心在世界原點）===")
    for label, path in targets.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"{label}: prim 不存在 ({path})")
            continue
        box = _aabb(path)
        print(f"{label}: min={tuple(round(v,4) for v in box['min'])}  max={tuple(round(v,4) for v in box['max'])}")

    # 對照：base_placement_calculator 對 Y=-1.1 失敗案例算出的機器人基座位置
    print("\n=== compute_base_pose() 對失敗案例算出的基座位置 ===")
    import math

    target_ball = (0.0, 0.635)
    for cue_x in (-0.5, -0.25, 0.0, 0.25, 0.5):
        cue_ball = (cue_x, -1.1)
        angle_deg = math.degrees(math.atan2(cue_ball[0] - target_ball[0], target_ball[1] - cue_ball[1]))
        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], angle_deg, table_z=0.0)
        print(f"cue_ball={cue_ball}  angle_deg={angle_deg:.2f}  base_position={tuple(round(v,4) for v in base_position)}  base_yaw_rad={base_yaw_rad:.3f}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
