"""
scripts/measure_swing_speed.py — Issue #176 空揮測速

用法（獨立執行）：
    python.bat scripts/measure_swing_speed.py

也可透過 Tool Menu Registry（extension/ui/tool_menu_registry.py）在 Kit 主選單
「Tools > Billiard/...」點擊執行，此時共用目前 Kit session 已開啟的 stage。

不 import core/ 任何模組，直接用原生 API（見 docs/task-176-swing-speed-spec.md 第 3 節決議）。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

# 必須跟 extension/billiard_digital_twin/billiard_digital_twin.py 用同一種 import
# 路徑（把 extension/ 本身加進 sys.path，import 成 "ui.tool_menu_registry"），
# 否則同一支檔案會被當成兩個不同模組載入，各自有獨立的 _REGISTERED_TOOLS 清單，
# decorator 註冊的內容跟 discover_and_register 讀到的會對不上。
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from pxr import PhysxSchema, Usd, UsdPhysics
import omni.usd

from ui.tool_menu_registry import tool_menu_item

# 注意：不要在檔案最上層 import isaacsim.* 底下「本身也是獨立 Kit extension」的
# 子模組（例如 isaacsim.core.experimental.prims、isaacsim.storage.native）。
# discover_and_register 會在 extension on_startup 當下把整支檔案 import 一次
# 以觸發 decorator 註冊；若這些模組在檔案最上層被 import，會在其底層 DLL
# （例如 isaacsim.core.simulation_manager 的 _simulation_manager）尚未載入
# 完成時就強制觸發 import，導致 "DLL load failed while importing
# _simulation_manager" 之類的錯誤。這些重量級模組請延後到工具函式「真正
# 執行」時才 import（見 _load_ur5/ check_joint_limits 內的 import）。
# omni.usd 與 pxr 屬於 Kit/USD 基礎綁定，開機當下就可用，可放檔案最上層。

UR5_ASSET_PATH = "Isaac/Robots/UniversalRobots/ur5/ur5.usd"
UR5_PRIM_PATH = "/World/ur5"
REAL_ROBOT_LIMIT_DEG_S = 180.0


def _load_ur5():
    from isaacsim.storage.native import get_assets_root_path

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(UR5_PRIM_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(UR5_PRIM_PATH)
        resolved_path = get_assets_root_path() + "/" + UR5_ASSET_PATH
        prim.GetReferences().AddReference(resolved_path)
    return stage, prim


def _list_revolute_joints(stage, root_path: str):
    joints = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        joints.append(prim)
    return joints


@tool_menu_item("Billiard/Measure Swing Speed - Check Joint Limits")
def check_joint_limits():
    """讀取 UR5 各關節 velocity/effort limit，對照實機 ±180 deg/s，輸出逐關節表格。"""
    stage, _ = _load_ur5()

    joints = _list_revolute_joints(stage, UR5_PRIM_PATH)
    if not joints:
        raise RuntimeError(f"在 {UR5_PRIM_PATH} 下找不到 PhysicsRevoluteJoint，檢查 asset 路徑或 prim 結構")

    print(f"{'Joint':30s} {'USD maxVel(deg/s)':>20s} {'maxForce(N*m)':>15s} {'>180deg/s?':>12s}")
    print("-" * 80)

    rows = []
    for joint_prim in joints:
        physx_joint = PhysxSchema.PhysxJointAPI(joint_prim)
        max_vel_attr = physx_joint.GetMaxJointVelocityAttr()
        max_vel_deg_s = max_vel_attr.Get() if max_vel_attr and max_vel_attr.HasValue() else None

        drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        max_force_nm = drive_api.GetMaxForceAttr().Get() if drive_api else None

        exceeds = (
            "YES - 需覆寫"
            if (max_vel_deg_s is not None and max_vel_deg_s > REAL_ROBOT_LIMIT_DEG_S)
            else "OK"
        )
        name = joint_prim.GetName()
        print(f"{name:30s} {str(max_vel_deg_s):>20s} {str(max_force_nm):>15s} {exceeds:>12s}")
        rows.append((name, max_vel_deg_s, max_force_nm, exceeds))

    from isaacsim.core.experimental.prims import Articulation

    print("\n--- Core API 交叉驗證（rad/s -> deg/s）---")
    ur5 = Articulation(paths=UR5_PRIM_PATH)
    dof_names = ur5.dof_names
    max_vel_rad_s = np.asarray(ur5.get_dof_max_velocities())
    max_vel_deg_s_from_core = np.rad2deg(max_vel_rad_s)
    for dof_name, v_deg in zip(dof_names, max_vel_deg_s_from_core.flatten()):
        print(f"{dof_name:30s} core_api={v_deg:.2f} deg/s")

    return rows


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    check_joint_limits()
    simulation_app.close()
