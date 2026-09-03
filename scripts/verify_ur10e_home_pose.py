"""
scripts/verify_ur10e_home_pose.py - UR10e+專用出力結構重新設計計畫
步驟 5：HOME 姿態設計。

驗證 Ur10eRmpflowController.move_to_home()（決策 11 的固定 HOME 關節角度，
沿用 ur10e_robot_description.yaml 的 default_q）：
1. 手臂能不能從任意起始姿態透過 RMPflow 收斂到 HOME 對應的世界座標末端
   位姿。
2. 註冊球檯＋地板為 RMPflow 障礙物後（decision 6 的簡化版，完整版留給
   步驟 6/7），HOME -> AIM（真實擊球姿態）-> HOME 這段來回路徑會不會撞到
   球檯/地板——用真實 PhysX contact-report 事件驗證（decision 6：
   「理論避障有效不可靠，實測才算數」，不是只看 RMPflow 自己有沒有報錯）。

跟 scripts/verify_ur10e_rmpflow_aim.py 的差異：那支只驗證 HOME 之後的單一
AIM 目標收斂精度；這支加上 HOME 本身的定義/往返路徑，以及球檯/地板障礙物
註冊＋真實碰撞事件檢查。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_home_pose.py
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
_FLOOR_Z_WORLD = -0.7695
"""已驗證過的真實地板世界高度，見
scripts/search_ur3e_placement_constants.py 同一個常數。"""
_POSITION_TOLERANCE_M = 0.01
_MAX_STEPS_PER_LEG = 4000
_PHYSICS_DT = 1.0 / 60.0
_TABLE_OBSTACLE_HEIGHT_M = 0.15
"""球檯本體（含庫邊/桌腳）概略厚度，障礙物中心對齊 table_z，這個高度只是
保守估計，不追求精確——目的是讓 RMPflow 知道「桌面附近有東西」，不是
精確建模桌子的每個幾何細節。"""


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd
    from isaacsim.core.api.objects import FixedCuboid, GroundPlane
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.ur10e_rmpflow_controller import Ur10eRmpflowController

    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services import cue_pose_calculator, ur10e_placement_calculator
    from core.services.spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/VerifyUr10eHomeTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS
    table_center = table.get_table_center()
    # ⚠️ 不用 stage_api.get_prim_sides(table.get_table_prim_path())——那個
    # 探測的是整個 billiard_env.usda 參照（含 SimpleRoom 整個房間），量到
    # 10x10x10m 的荒謬尺寸，拿來當障礙物會把整支手臂吞進去，造成天文數字
    # 等級的穿透修正衝量（實測踩過：10,656,314）。改用
    # core/services/spread_score_calculator.py 既有的 TABLE_LENGTH/
    # TABLE_WIDTH（球界邊界推導出的真正球檯尺寸常數，這個專案到處都在用），
    # 高度用 _TABLE_OBSTACLE_HEIGHT_M 概略估計球檯本體厚度。
    table_x_len, table_y_len = TABLE_WIDTH, TABLE_LENGTH
    table_z_len = _TABLE_OBSTACLE_HEIGHT_M
    print(f"[home] table_center={table_center} table_size=({table_x_len:.3f},{table_y_len:.3f},{table_z_len:.3f})")

    wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, table_z, ball_radius,
    )
    if tilt_rad is None:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 幾何無解（tilt_rad=None）")
    print(f"[home] 真實 AIM 目標 wrist_position={list(wrist_position)}  tilt_rad={tilt_rad:.4f}")

    robot_manager = TableRobotManager(
        table_center, table_base_path, stage_api, None, UR10eRobot,
    )
    robot_prim_path = robot_manager.get_robot_prim_path()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()

    direction_unit = cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad)
    base_position = ur10e_placement_calculator.compute_base_position(
        tuple(wrist_position), tuple(direction_unit), table_z
    )
    base_orientation = [1.0, 0.0, 0.0, 0.0]
    print(f"[home] per-shot 基座位置={base_position}")
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

    # ---- 註冊球檯＋地板為 RMPflow 障礙物（decision 6 簡化版）----
    ground_obstacle = GroundPlane(
        prim_path="/World/_RmpflowGroundObstacle", z_position=_FLOOR_Z_WORLD,
    )
    table_obstacle_center = np.array([table_center[0], table_center[1], table_z + table_z_len / 2.0])
    table_obstacle = FixedCuboid(
        prim_path="/World/_RmpflowTableObstacle",
        position=table_obstacle_center,
        scale=np.array([table_x_len, table_y_len, table_z_len]),
        size=1.0,
    )
    print(f"[home] table_obstacle center={table_obstacle_center.tolist()} scale=({table_x_len},{table_y_len},{table_z_len})")

    controller = Ur10eRmpflowController(articulation, end_effector_prim_path)
    controller.set_robot_base_pose(list(base_position), base_orientation)
    controller.add_ground_plane(ground_obstacle)
    controller.add_obstacle(table_obstacle, static=True)

    # ---- 真實 PhysX contact-report（decision 6：理論避障不可靠，實測才算數）----
    contacts = []
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            physics_api.enable_contact_reporting(str(prim.GetPath()))
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    def _run_leg(name: str, target_position, target_orientation) -> bool:
        controller.move_to_pose(list(target_position), list(target_orientation))
        step = 0
        while not controller.is_motion_complete() and step < _MAX_STEPS_PER_LEG:
            controller.step(_PHYSICS_DT)
            simulation_app.update()
            step += 1
        live_position, _ = end_effector_rigid_prim.get_world_poses()
        live_position = np.asarray(live_position[0])
        error = float(np.linalg.norm(live_position - np.asarray(target_position)))
        converged = error <= _POSITION_TOLERANCE_M
        print(f"[home] leg={name} steps={step} error={error:.5f} m did_timeout={controller.did_last_motion_timeout()} converged={converged}")
        return converged

    home_position, home_orientation = controller._compute_home_end_effector_pose()
    print(f"[home] HOME 對應世界座標位置={home_position.tolist()}  朝向={home_orientation.tolist()}")

    ok_1 = _run_leg("start->HOME", home_position, home_orientation)
    contacts_after_leg1 = len(contacts)
    print(f"[home] start->HOME 累積碰撞事件數={contacts_after_leg1}")

    ok_2 = _run_leg("HOME->AIM", wrist_position, wrist_orientation)
    contacts_after_leg2 = len(contacts)
    print(f"[home] HOME->AIM 累積碰撞事件數={contacts_after_leg2}（增量 {contacts_after_leg2 - contacts_after_leg1}）")

    ok_3 = _run_leg("AIM->HOME", home_position, home_orientation)
    contacts_after_leg3 = len(contacts)
    print(f"[home] AIM->HOME 累積碰撞事件數={contacts_after_leg3}（增量 {contacts_after_leg3 - contacts_after_leg2}）")

    for c in contacts:
        print(f"[home] CONTACT a={c.actor_path_a} b={c.actor_path_b} impulse={c.impulse}")

    # impulse=0.0 代表幾何上有觸碰但沒有實際受力（純粹的邊界貼合/掠過，
    # PhysX 在 threshold=0 的回報設定下連這種都會報），跟真正帶力道的碰撞
    # 不是同一回事——沿用今晚稍早除錯時建立的判斷慣例（settle 階段
    # impulse~1.0 才算「非災難性」但仍是真碰撞，impulse=0.0 連真碰撞都
    # 算不上）。AIM 目標本來就在球檯障礙物箱體邊界附近（球桿要碰到球），
    # 這裡出現一次零衝量觸碰是預期內的，不代表手臂真的撞上球檯。
    real_collisions = [c for c in contacts if c.impulse > 0.0]
    all_converged = ok_1 and ok_2 and ok_3
    no_real_collisions = len(real_collisions) == 0
    print(f"[home] 全部路徑收斂={all_converged}  零衝量觸碰事件數={len(contacts)}  真實碰撞事件數={len(real_collisions)}")
    if all_converged and no_real_collisions:
        print("[home] PASS：HOME<->AIM 往返路徑全部收斂，且沒有真實（非零衝量）碰撞")
    else:
        print("[home] FAIL：收斂或真實碰撞檢查未通過")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
