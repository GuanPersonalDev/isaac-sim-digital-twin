"""
scripts/verify_ur10e_rmpflow_reach.py — UR10e+專用出力結構重新設計計畫
步驟 3（前半）：RMPflow 手臂控制最小可行流程驗證（不含障礙物、不含球檯）。

驗證 `extension/isaac_sim_impl_6_0/ur10e_rmpflow_controller.py` 的
`Ur10eRmpflowController`：
1. 關節順序對應（RMPflow 6 個活動關節 vs 7-DOF 完整關節陣列的名稱比對）
   有沒有錯——這是 skills/isaac_sim_6_api_cache.md「RmpFlow」條目 Q6 明確
   標記「尚未在專案中實際寫碼驗證」的風險項。
2. 手臂能不能真的收斂到一個指定的末端目標位置（wrist_3_link 世界座標）。

沒有障礙物、沒有球檯：先用最簡單的案例排除「關節順序對應錯誤」這個最基本
的問題，之後步驟才加障礙物（decision 6）跟真實球檯（步驟 7/8）。

⚠️ 實測發現（2026-09-03）：目標位移大小會明顯影響收斂品質。
- 5cm 量級的單軸/小幅位移（本檔案目前的預設案例）：~120 步（2 秒）內穩定
  收斂到 <5mm 誤差，位置+朝向同時約束也一樣收斂良好。
- 但一次給一個 (0.2, 0.2, 0.1) 量級（約 30cm）的對角線大跳躍目標，900 步
  （15 秒）後仍卡在約 0.14m 殘留誤差原地不動——不是還在收斂中，是真的卡住
  了（RMPflow 的 target_rmp/joint_limit_rmp/damping_rmp 等多個 RMP 分量
  互相拉扯出的局部穩定點，不是全域最佳化，這是 reactive controller 的已知
  特性，不是這裡的程式碼有 bug——已用「PhysX 是否確實追蹤 RMPflow 當下算出
  的關節目標」的差距診斷排除過追蹤面的問題，追蹤落差恆為 0）。
- 對後續步驟（尤其 AIM：從 HOME 姿態一次跳到瞄準姿態，位移量通常不小）的
  含意：大位移目標可能需要拆成多個中繼 waypoint（跟 WAM7 舊架構「Phase 0
  安全姿態＋Cartesian waypoint 序列」精神類似，只是用 RMPflow 逐段導航
  取代原本的差動 IK），或者拉長收斂等待時間＋調整 rmp_params 增益，這點
  在正式串進 table_orchestrator.py（步驟 6）之前需要先決定。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_rmpflow_reach.py
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_POSITION_TOLERANCE_M = 0.01
_NUM_STEPS = 180  # 2 秒足夠讓 5cm 量級的目標收斂（見上方實測發現）
_PHYSICS_DT = 1.0 / 60.0


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from isaac_sim_impl_6_0.ur10e_rmpflow_controller import Ur10eRmpflowController

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    base_path = "/World/VerifyUr10eReach"
    manager = TableRobotManager(
        table_center=(0.0, 0.0, 0.0),
        base_path=base_path,
        stage_api=stage_api,
        articulation_api=None,
        robot_arm_class=UR10eRobot,
    )
    robot_prim_path = manager.get_robot_prim_path()
    print(f"[verify] robot_prim_path={robot_prim_path}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=robot_prim_path)
    for _ in range(5):
        simulation_app.update()

    end_effector_prim_path = robot_prim_path + "/wrist_3_link"
    end_effector_rigid_prim = RigidPrim(paths=end_effector_prim_path)

    start_position, start_orientation = end_effector_rigid_prim.get_world_poses()
    start_position = np.asarray(start_position[0])
    start_orientation = np.asarray(start_orientation[0])
    print(f"[verify] 初始 wrist_3_link 世界位置={start_position.tolist()}")
    print(f"[verify] 初始 wrist_3_link 世界朝向={start_orientation.tolist()}")

    target_position = (start_position + np.array([0.05, 0.0, 0.0])).tolist()
    target_orientation = start_orientation.tolist()
    print(f"[verify] 目標位置={target_position}（維持初始朝向）")

    controller = Ur10eRmpflowController(articulation)
    print(f"[verify] RMPflow 活動關節={controller._active_joint_names}")
    print(f"[verify] 對應到 7-DOF 陣列的 index={controller._active_dof_indices}")

    # UR10e 底座透過 TableRobotManager 被搬到
    # table_center + TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER，
    # 不是世界原點——一定要告訴 RMPflow 真正的底座世界位姿，否則它的內部
    # 運動學模型會假設底座在原點，算出來的關節目標會系統性偏移（見
    # Ur10eRmpflowController.set_robot_base_pose() docstring）。UR10eRobot
    # 只用 set_prim_translate（無旋轉），朝向維持 identity 四元數。
    base_offset = TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    base_position = [0.0 + base_offset[0], 0.0 + base_offset[1], 0.0 + base_offset[2]]
    base_orientation = [1.0, 0.0, 0.0, 0.0]
    print(f"[verify] 機器人底座世界位姿：position={base_position} orientation={base_orientation}")
    controller.set_robot_base_pose(base_position, base_orientation)

    controller.set_end_effector_target(target_position, target_orientation)
    print("[verify] set_end_effector_target 呼叫完成，開始逐 tick 呼叫 controller.step() ...")
    sys.stdout.flush()

    final_error = None
    for step in range(_NUM_STEPS):
        try:
            controller.step(_PHYSICS_DT)
        except Exception:
            print(f"[verify] ❌ controller.step() 在 step={step} 拋出例外：")
            traceback.print_exc()
            sys.stdout.flush()
            raise
        simulation_app.update()
        if step % 60 == 0 or step == _NUM_STEPS - 1:
            live_position, live_orientation = end_effector_rigid_prim.get_world_poses()
            live_position = np.asarray(live_position[0])
            error = float(np.linalg.norm(live_position - np.asarray(target_position)))
            final_error = error
            print(f"[verify] step={step} wrist_3_link 位置={live_position.tolist()} 誤差={error:.5f} m")
            sys.stdout.flush()

    print(f"[verify] 最終誤差={final_error:.5f} m（容許 {_POSITION_TOLERANCE_M} m）")
    if final_error is not None and final_error <= _POSITION_TOLERANCE_M:
        print("[verify] ✅ RMPflow 成功把手臂收斂到目標位置，關節順序對應正確")
    else:
        print("[verify] ❌ 沒有在容許誤差內收斂，需要檢查關節順序對應或 RMPflow 參數")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
