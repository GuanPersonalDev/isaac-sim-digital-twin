"""
scripts/probe_floor_geometry.py — 量測球檯場景裡「地板」與「檯面」的真實世界
Z 座標，供 `search_ur3e_placement_constants.py` 的球桿地板淨空限制式使用。

背景：2026-09-02 GUI 重跑確認 UR3e 揮桿卡住的根因是後擺時 CueStick 撞到
`SimpleRoom/GroundPlane/CollisionPlane`。要把「後擺整段軌跡的球桿淨空」寫成
搜尋階段的限制式，就必須知道地板相對於「桿尖接觸點」的垂直距離——2026-09-01
的踩坑紀錄第 (4) 點當時放棄這個方向的理由正是「不知道真實地板的精確世界
座標，沒有可靠的解析代理」，這支腳本就是把那個未知量測出來。

跑法（需要 Isaac Sim headless）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" -u scripts/probe_floor_geometry.py
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
    from pxr import UsdPhysics, Sdf, UsdGeom, Usd

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from core.models.billiard_table import BilliardTable

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    table_base_path = "/World/ProbeTable"
    table = BilliardTable(
        table_base_path, StageAPIImpl(), MaterialAPIImpl(), RigidBodyAPIImpl(), (0.0, 0.0)
    )
    ball_prim_path = table.get_table_ball_set().get_ball_prim_paths()[0]

    def _world_bounds(prim):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        return np.array(rng.GetMin()), np.array(rng.GetMax())

    print("=== 掃描球檯場景所有 prim，找地板/檯面相關者的世界 Z ===")
    interesting = ("GroundPlane", "CollisionPlane", "floor", "Floor", "Surface", "Cushion")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(table_base_path):
            continue
        if not any(k in path for k in interesting):
            continue
        bounds = _world_bounds(prim)
        if bounds is None:
            continue
        lo, hi = bounds
        print(f"{path}\n    z_min={lo[2]:+.5f}  z_max={hi[2]:+.5f}")

    print("")
    print("=== 母球（接觸點高度基準）===")
    ball_prim = stage.GetPrimAtPath(ball_prim_path)
    bounds = _world_bounds(ball_prim)
    if bounds is not None:
        lo, hi = bounds
        print(f"{ball_prim_path}\n    z_min={lo[2]:+.5f}  z_max={hi[2]:+.5f}  z_center={(lo[2]+hi[2])/2:+.5f}")


if __name__ == "__main__":
    import traceback

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
