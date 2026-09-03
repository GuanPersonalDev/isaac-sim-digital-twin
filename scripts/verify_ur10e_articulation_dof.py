"""
scripts/verify_ur10e_articulation_dof.py — UR10e+專用出力結構重新設計計畫
步驟 2：技術驗證。

驗證決策 3 的核心假設：`CueStick` 跟 `wrist_3_link` 之間換成 `PrismaticJoint`
（見 core/models/table_robot_manager.py 的 UR10e 分支）之後，PhysX 會不會
自動把這個新關節併入既有 UR10e articulation 樹，讓
`isaacsim.core.experimental.prims.Articulation` 回報 7 個 DOF（6 手臂關節
＋1 滑軌關節），`dof_names` 含新關節名稱（"CueSlideJoint"）。

這是計畫文件裡明確標記「尚未實測」的風險項目，正式接上 RMPflow 控制器
（步驟 3）之前先確認這個假設成立。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_articulation_dof.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EXPECTED_DOF_COUNT = 7
_NEW_JOINT_NAME = "CueSlideJoint"


def _run() -> None:
    import omni.usd
    from pxr import UsdPhysics, Sdf
    from isaacsim.core.experimental.prims import Articulation

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    base_path = "/World/VerifyUr10e"
    print(f"[verify] 建立 TableRobotManager(base_path={base_path}, robot_arm_class=UR10eRobot) ...")
    manager = TableRobotManager(
        table_center=(0.0, 0.0, 0.0),
        base_path=base_path,
        stage_api=stage_api,
        articulation_api=None,  # UR10eRobot.__init__ 只用 stage_api，不用 articulation_api
        robot_arm_class=UR10eRobot,
    )
    robot_prim_path = manager.get_robot_prim_path()
    cue_stick_prim_path = manager.get_cue_stick_prim_path()
    print(f"[verify] robot_prim_path={robot_prim_path}")
    print(f"[verify] cue_stick_prim_path={cue_stick_prim_path}")

    joint_prim_path = cue_stick_prim_path + "/" + _NEW_JOINT_NAME
    joint_prim = stage.GetPrimAtPath(joint_prim_path)
    print(f"[verify] PrismaticJoint prim {joint_prim_path} IsValid()={joint_prim.IsValid()}")
    if joint_prim.IsValid():
        print(f"[verify] PrismaticJoint typeName={joint_prim.GetTypeName()}")

    for _ in range(30):
        simulation_app.update()

    articulation = Articulation(paths=robot_prim_path)
    dof_names = list(articulation.dof_names) if hasattr(articulation, "dof_names") else None
    print(f"[verify] dof_names={dof_names}")
    dof_count = len(dof_names) if dof_names is not None else None
    print(f"[verify] dof_count={dof_count}（預期 {_EXPECTED_DOF_COUNT}）")

    if dof_names is None:
        print("[verify] ❌ Articulation 沒有 dof_names 屬性，驗證失敗")
        return

    if dof_count != _EXPECTED_DOF_COUNT:
        print(
            f"[verify] ❌ dof_count={dof_count} 不等於預期 {_EXPECTED_DOF_COUNT}，"
            "PhysX 沒有把 PrismaticJoint 併入同一個 articulation（或計入了非預期的其他關節）"
        )
        return

    if _NEW_JOINT_NAME not in dof_names:
        print(f"[verify] ❌ dof_names 裡沒有 {_NEW_JOINT_NAME}，PrismaticJoint 沒被 Articulation 辨識為 DOF")
        return

    print("[verify] ✅ 7-DOF articulation 如預期運作，dof_names 含新滑軌關節")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
