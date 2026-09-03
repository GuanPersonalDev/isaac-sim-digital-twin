"""
scripts/verify_ur10e_rmpflow_aim.py - UR10e+專用出力結構重新設計計畫
步驟 3（後半）：用真實 AIM 目標（cue_pose_calculator.py 算出來的實際擊球
姿態，不是任意挑的座標）驗證 Ur10eRmpflowController.move_to_pose() 的
waypoint 拆分機制在真實使用情境下能不能收斂。

跟 scripts/verify_ur10e_rmpflow_reach.py 的差異：後者用一個刻意選的
0.3m 對角線大跳躍當壓力測試（拆成 4 段後仍卡在約 0.138m 殘留誤差，研判
是那個人為座標本身讓手臂中途走進 RMPflow 不友善的姿態區域）；這支腳本
改用真正會被 table_orchestrator.py 使用的計算路徑：

    cue_pose_calculator.compute_tilted_wrist_pose(cue_ball, shot_angle,
        table_z, ball_radius) -> (wrist_position, wrist_orientation, ...)

cue_ball=(-0.036, -0.752) 是實際 GUI Break shot demo 開局母球位置（見
scripts/test_flat_ur3e_table.py 同一個常數的說明），不是隨便挑的。

2026-09-03 修正歷程：
1. 第一版用決策 4 的固定基座偏移
   （TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER=(1.5,0,0)）實測
   發現 wrist 目標距離固定基座遠達 2.6m，遠超過 UR10e 1.3m 可達距離，
   純幾何上到不了（詳見 core/services/ur10e_placement_calculator.py
   模組說明）。改回 per-shot 重新計算基座
   （core/services/ur10e_placement_calculator.py compute_base_position()，
   推翻決策 4 的固定基座假設，但比 WAM7/UR3e 簡單——只需要確保 wrist
   目標落在可達範圍內，不用搜尋特定關節組合）。
2. 換上正確基座後仍卡在約 0.75m 殘留誤差不收斂——這次目標本身在可達
   範圍內，問題出在 Ur10eRmpflowController.move_to_pose() 原本只內插
   位置、方向從第一段就鎖定最終目標，跟高架橋案例真正需要的傾斜方向
   互相拉扯。改成方向也用 slerp 逐段內插（見該方法 docstring）後：
   16 個中繼 waypoint 全部收斂，最終誤差 0.00189m，PASS。

結論：waypoint 拆分機制（位置線性內插＋方向 slerp 內插）＋per-shot 基座
重算，用真實 AIM 目標（cue_ball=(-0.036,-0.752) 的高架橋案例）驗證通過。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_rmpflow_aim.py
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CUE_BALL = (-0.036, -0.752)
_SHOT_ANGLE_DEG = 0.0
_POSITION_TOLERANCE_M = 0.01
_NUM_STEPS = 4000
_PHYSICS_DT = 1.0 / 60.0


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.ur10e_rmpflow_controller import Ur10eRmpflowController

    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services import cue_pose_calculator, ur10e_placement_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/VerifyUr10eAimTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS
    table_center = table.get_table_center()
    print(f"[aim] table_center={table_center} table_z={table_z} ball_radius={ball_radius}")

    wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, table_z, ball_radius,
    )
    if tilt_rad is None:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 幾何無解（tilt_rad=None）")
    print(f"[aim] cue_ball={_CUE_BALL} tilt_rad={tilt_rad:.6f} ({np.degrees(tilt_rad):.2f}度) crossing={crossing}")
    print(f"[aim] 真實 AIM 目標 wrist_position={list(wrist_position)}")
    print(f"[aim] 真實 AIM 目標 wrist_orientation={list(wrist_orientation)}")

    robot_manager = TableRobotManager(
        table_center, table_base_path, stage_api, None, UR10eRobot,
    )
    robot_prim_path = robot_manager.get_robot_prim_path()
    print(f"[aim] robot_prim_path={robot_prim_path}")

    # per-shot 重新計算基座位置（見 core/services/ur10e_placement_calculator.py
    # 模組說明），取代決策 4 原本的固定偏移——固定偏移實測發現對這個 cue_ball
    # 距離目標遠達 2.6m，純幾何上到不了。
    direction_unit = cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad)
    base_position = ur10e_placement_calculator.compute_base_position(
        tuple(wrist_position), tuple(direction_unit), table_z
    )
    base_orientation = [1.0, 0.0, 0.0, 0.0]
    print(f"[aim] per-shot 重新計算的機器人底座世界位姿：position={base_position} orientation={base_orientation}")
    robot_arm = robot_manager.get_robot()
    robot_arm.reposition(base_position)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=robot_prim_path)
    for _ in range(5):
        simulation_app.update()

    end_effector_prim_path = robot_prim_path + "/wrist_3_link"
    end_effector_rigid_prim = RigidPrim(paths=end_effector_prim_path)

    start_position, _ = end_effector_rigid_prim.get_world_poses()
    print(f"[aim] 初始 wrist_3_link 世界位置={np.asarray(start_position[0]).tolist()}")

    controller = Ur10eRmpflowController(articulation, end_effector_prim_path)
    controller.set_robot_base_pose(list(base_position), base_orientation)

    controller.move_to_pose(list(wrist_position), list(wrist_orientation))
    print(f"[aim] move_to_pose 呼叫完成，共拆成 {len(controller._waypoints)} 個中繼 waypoint")
    for i, (wp_pos, _wp_orient) in enumerate(controller._waypoints):
        print(f"[aim]   waypoint[{i}]={wp_pos.tolist()}")
    sys.stdout.flush()

    step = 0
    while not controller.is_motion_complete() and step < _NUM_STEPS:
        try:
            controller.step(_PHYSICS_DT)
        except Exception:
            print(f"[aim] controller.step() 在 step={step} 拋出例外：")
            traceback.print_exc()
            sys.stdout.flush()
            raise
        simulation_app.update()
        if step % 60 == 0:
            live_position, _ = end_effector_rigid_prim.get_world_poses()
            live_position = np.asarray(live_position[0])
            error = float(np.linalg.norm(live_position - np.asarray(wrist_position)))
            print(f"[aim] step={step} waypoint_index={controller._waypoint_index}/{len(controller._waypoints)} wrist_3_link 位置={live_position.tolist()} 誤差={error:.5f} m")
            sys.stdout.flush()
        step += 1

    live_position, live_orientation = end_effector_rigid_prim.get_world_poses()
    live_position = np.asarray(live_position[0])
    final_error = float(np.linalg.norm(live_position - np.asarray(wrist_position)))
    print(f"[aim] 總步數={step} is_motion_complete()={controller.is_motion_complete()} did_last_motion_timeout()={controller.did_last_motion_timeout()}")
    print(f"[aim] 最終誤差={final_error:.5f} m（容許 {_POSITION_TOLERANCE_M} m）")
    if final_error <= _POSITION_TOLERANCE_M:
        print("[aim] PASS：RMPflow 用真實 AIM 目標成功收斂")
    else:
        print("[aim] FAIL：真實 AIM 目標仍未收斂到容許誤差內")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
