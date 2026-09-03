"""
scripts/test_load_ur3e.py — 測試能不能載入 Isaac Sim 內建的 UR3e USD 資產
（`Isaac/Robots/UniversalRobots/ur3e/ur3e.usd`，官方文件確認存在，但這台機器
本機 pip 安裝的 isaacsim 套件裡沒有實體檔案，要走 Nucleus/CDN 解析
`isaacsim.storage.native.get_assets_root_path()`）。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_load_ur3e.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_UR3E_PATH = "Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"


def _run() -> None:
    import omni.usd
    from pxr import UsdPhysics, Sdf
    from isaacsim.storage.native import get_assets_root_path

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl

    print("[test] 呼叫 get_assets_root_path() ...")
    assets_root = get_assets_root_path()
    print(f"[test] assets_root_path={assets_root}")
    if assets_root is None:
        print("[test] ⚠️ get_assets_root_path() 回傳 None，代表連不到任何資產來源（本地/Nucleus/CDN 都沒有）")
        return

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    prim_path = "/World/TestUR3e"
    print(f"[test] 嘗試 create_reference_prim({prim_path}, {_UR3E_PATH}) ...")
    prim = stage_api.create_reference_prim(prim_path, _UR3E_PATH)

    for _ in range(10):
        simulation_app.update()

    print(f"[test] prim.IsValid()={prim.IsValid()}")
    children = stage_api.get_child_prim_paths(prim_path)
    print(f"[test] 子層 prim 數量={len(children)}")
    for c in children[:20]:
        print(f"[test]   child: {c}")

    if not children:
        print("[test] ⚠️ 沒有任何子層 prim，代表 reference 沒有真的解析到資產內容（可能是網路連不到 CDN，或路徑錯誤）")
    else:
        print("[test] ✅ UR3e USD 成功載入並展開階層")

        # 順便檢查有沒有 ArticulationRootAPI，確認是可控制的 articulation。
        from pxr import Usd, UsdPhysics as _UsdPhysics
        robot_prim = stage.GetPrimAtPath(prim_path)
        found_articulation_root = False
        for p in Usd.PrimRange(robot_prim):
            if p.HasAPI(_UsdPhysics.ArticulationRootAPI):
                print(f"[test] 找到 ArticulationRootAPI：{p.GetPath()}")
                found_articulation_root = True
                break
        if not found_articulation_root:
            print("[test] ⚠️ 沒找到 ArticulationRootAPI")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
