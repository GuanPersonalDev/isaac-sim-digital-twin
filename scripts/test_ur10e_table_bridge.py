"""
scripts/test_ur10e_table_bridge.py — UR10e+專用出力結構重新設計計畫
步驟 8：高架橋（傾斜）案例的真實球檯驗證。

跟 scripts/test_ur10e_table_flat.py 是同一套驗收方法論與同一條生產程式碼
路徑（`Ur10eSwingStrategy.execute_aim()`／`execute_strike()` 透過真正的
`ArticulationAPIImpl`），差別只在測試點：這裡選 `cue_pose_calculator.
compute_tilted_wrist_pose()` 會算出 `tilt_rad > 0` 的母球位置——母球太靠近
庫邊，球桿平放會穿過庫邊，必須把握把端抬高（真人用手架「高架橋」的那個
動作）才打得到。

決策 8 的主張是「flat 與高架橋在新架構下是同一條碼路」：RMPflow 解任意
可達姿態，不需要像 UR3e 那樣為每個傾斜角分別搜尋關節組合，
`cue_pose_calculator.py` 本來就統一處理兩種情況。這支腳本要驗的就是這個
主張——**不新增任何架構，只換測試點**，確認：
1. RMPflow 真的能安全把手臂帶到傾斜姿態（決策 6：零非預期碰撞）
2. 傾斜姿態下推桿一樣打得準（決策 7：達成率 >=90%、球桿只碰母球 1 次）

flat 案例已通過的東西這裡不重複詳細診斷（逐 tick 的桿尖-母球距離、
CueSlideJoint 追蹤延遲、原始生成姿態對位等），那些是排查特定 bug 時加的
儀器，問題解決後留在 flat 那支當歷史記錄即可。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_ur10e_table_bridge.py
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CUE_BALL = (0.0, -0.635)
"""高架橋案例代表點，沿用 scripts/test_elevated_bridge_ur3e_table.py 選的
同一個座標：`cue_pose_calculator.py` 的 9 點網格裡 tilt_rad 不是 None、
且不屬於「任何 roll 都到不了目標球速」那排已知不可解案例的代表點。用同
一個點也方便跟 UR3e 時代的結果直接對照。"""
_SHOT_ANGLE_DEG = 0.0
_CUE_BALL_SPEED = 1.995
_MAX_STEPS_PER_ACTION = 4000
_MAX_STEPS_PER_AIM_ACTION = 20000

_ACHIEVEMENT_THRESHOLD = 0.9
"""決策 7 的達成率門檻，跟 flat 案例同一個標準。"""


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.action import Action
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services import cue_pose_calculator
    from core.services.spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH
    from core.services.ur10e_swing_strategy import Ur10eSwingStrategy
    from isaacsim.core.experimental.prims import RigidPrim

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/TestUr10eTableBridge"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS

    wrist_position, _wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, table_z, ball_radius,
    )
    if tilt_rad is None:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 幾何無解（tilt_rad=None），換一個測試點")
    print(f"[bridge] cue_ball={_CUE_BALL}  tilt_rad={tilt_rad:.6f} rad "
          f"({np.degrees(tilt_rad):.2f}°)  crossing={crossing}")
    if tilt_rad <= 1e-6:
        raise RuntimeError(
            f"cue_ball={_CUE_BALL} 其實是 flat 案例（tilt_rad={tilt_rad}），"
            "這支腳本要驗的是傾斜案例，換別的座標"
        )
    print(f"[bridge] 目標 wrist 位置={np.round(wrist_position, 5).tolist()}"
          f"（Z 比桌面高 {wrist_position[2] - table_z:.4f} m，flat 案例只會高一個球半徑 {ball_radius:.4f} m）")

    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, UR10eRobot,
    )
    robot_arm = robot_manager.get_robot()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    ball_prim_path = table_ball_set.get_ball_prim_paths()[0]

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    # place_ball() 必須在 timeline.play() 之後（球的 xformOp:orient 要等
    # physics 初始化過一次才存在），跟 flat 腳本同一個時序理由。
    table_ball_set.place_ball(0, _CUE_BALL[0], _CUE_BALL[1])
    for _ in range(5):
        simulation_app.update()

    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()
    print("[bridge] initialize() 完成")
    sys.stdout.flush()

    # 障礙物註冊沿用 flat 腳本的結論：球檯方塊放在 table_z **之下**（代表
    # 桌面以下的石板/桌腳結構），桌面正上方保持淨空——放在桌面之上會讓
    # 桿尖擊球高度本身落在障礙物範圍內，最終逼近等於在跟自己的目標打架。
    _TABLE_OBSTACLE_HEIGHT_M = 0.15
    table_center = table.get_table_center()
    table_obstacle_center = [table_center[0], table_center[1], table_z - _TABLE_OBSTACLE_HEIGHT_M / 2.0]
    articulation_api.register_static_box_obstacle(
        table_obstacle_center, [TABLE_WIDTH, TABLE_LENGTH, _TABLE_OBSTACLE_HEIGHT_M]
    )
    articulation_api.register_dynamic_sphere_obstacle(ball_prim_path, ball_radius)

    ball_rigid_prim = RigidPrim(paths=ball_prim_path)

    contacts = []
    current_phase = {"name": "SETUP"}
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    physics_api.enable_contact_reporting(ball_prim_path)
    # 手臂本體也開碰撞回報：高架橋案例手臂要伸到桌面上方更高的地方，
    # 「手臂本體（不是球桿）撞到庫邊」的風險比 flat 案例高，決策 6 要求
    # 實測驗證而不是理論推測。
    arm_link_prim_paths = [
        f"{robot_prim_path}/forearm_link",
        f"{robot_prim_path}/wrist_1_link",
        f"{robot_prim_path}/wrist_2_link",
        f"{robot_prim_path}/wrist_3_link",
    ]
    for arm_link_prim_path in arm_link_prim_paths:
        physics_api.enable_contact_reporting(arm_link_prim_path)
    physics_api.subscribe_contact_events(
        lambda e: contacts.append((current_phase["name"], e))
    )

    strategy = Ur10eSwingStrategy(robot_arm, articulation_api)
    action = Action(
        cue_ball_speed=_CUE_BALL_SPEED,
        shot_angle=_SHOT_ANGLE_DEG,
        position_offset=[0.0, 0.0],
        cue_ball_placement=list(_CUE_BALL),
        should_execute_action=True,
    )

    def _run_until_complete(label: str, max_steps: int) -> int:
        step = 0
        while not articulation_api.is_motion_complete() and step < max_steps:
            simulation_app.update()
            step += 1
        print(f"[bridge] {label} 完成，steps={step} "
              f"did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")
        sys.stdout.flush()
        return step

    # production 是 RESET→AIM，不會從 raw USD 預設姿態直接跳 AIM（那需要
    # 接近 180 度的姿態翻轉，RMPflow 會卡住）。move_to_home() 之前要先讓
    # RMPflow 知道底座目前的世界位姿——此時手臂還在 TableRobotManager
    # 建構時的初始固定偏移，還沒被 execute_aim() 的 per-shot reposition()
    # 移動過。
    initial_base_offset = TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    initial_base_position = [
        table.get_table_center()[i] + initial_base_offset[i] for i in range(3)
    ]
    articulation_api.set_robot_base_pose(initial_base_position, [1.0, 0.0, 0.0, 0.0])

    current_phase["name"] = "RESET"
    articulation_api.move_to_home()
    _run_until_complete("RESET(HOME)", _MAX_STEPS_PER_ACTION)

    current_phase["name"] = "AIM"
    strategy.execute_aim(action, tuple(_CUE_BALL), table_z, ball_radius)
    _run_until_complete("AIM", _MAX_STEPS_PER_AIM_ACTION)

    achieved_position = np.asarray(articulation_api.get_end_effector_position())
    achieved_orientation = np.asarray(articulation_api.get_end_effector_orientation())
    # execute_aim() 內部會重新搜尋 roll_rad，最終目標跟腳本開頭那次
    # 預設 roll_rad=0 的計算不同——位置不受 roll_rad 影響可以直接比，
    # 方向則要拿控制器實際收到的目標比，這裡只報位置誤差與實際姿態。
    print(f"[bridge] AIM 收斂後 wrist 位置={achieved_position.tolist()}")
    print(f"[bridge] AIM 收斂後 wrist 方向={achieved_orientation.tolist()}")
    print(f"[bridge] 位置誤差（對預設 roll 的目標，僅供參考）="
          f"{float(np.linalg.norm(achieved_position - np.asarray(wrist_position))):.5f} m")

    ball_velocity_before, _ = ball_rigid_prim.get_velocities()
    ball_speed_before = float(np.linalg.norm(np.asarray(ball_velocity_before[0], dtype=float)))
    print(f"[bridge] STRIKE 前母球速度={ball_speed_before:.6f} m/s（應為 0；非零代表 AIM 過程蹭到球）")

    current_phase["name"] = "STRIKE"
    strategy.execute_strike(action, tuple(_CUE_BALL), table_z, ball_radius)

    # 達成率用整段 STRIKE 的母球速度峰值：STRIKE 迴圈涵蓋揮桿後的縮回＋
    # 上抬，迴圈結束時母球可能已經滾遠或撞到別的球，末速不代表球桿賦予的
    # 速度；球桿脫離接觸後速度只會遞減，峰值就是實際傳遞出去的值
    # （跟 flat 腳本同一個量測方式）。
    peak_ball_speed = 0.0
    strike_steps = 0
    while not articulation_api.is_motion_complete() and strike_steps < _MAX_STEPS_PER_ACTION:
        simulation_app.update()
        strike_steps += 1
        ball_velocity_now, _ = ball_rigid_prim.get_velocities()
        peak_ball_speed = max(
            peak_ball_speed,
            float(np.linalg.norm(np.asarray(ball_velocity_now[0], dtype=float))),
        )
    print(f"[bridge] STRIKE 完成，steps={strike_steps} "
          f"did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")

    achievement = peak_ball_speed / _CUE_BALL_SPEED
    print(f"[bridge] STRIKE 全程母球速度峰值={peak_ball_speed:.4f} m/s  "
          f"目標母球速度={_CUE_BALL_SPEED} m/s  達成率={100 * achievement:.1f}%")

    ball_contacts = [
        (phase_name, c) for phase_name, c in contacts
        if ball_prim_path in (c.actor_path_a, c.actor_path_b) and c.impulse > 0.0
    ]
    print(f"[bridge] 母球碰撞事件數（impulse>0）={len(ball_contacts)}")
    for phase_name, c in ball_contacts:
        print(f"[bridge]   CONTACT phase={phase_name} a={c.actor_path_a} b={c.actor_path_b} impulse={c.impulse}")

    # 決策 7 的「恰好 1 次」指的是球桿碰到母球恰好一次（沒有蹭到、沒有
    # 二次擊球）；母球撞上球堆是這一擊預期中的結果，不列入計數。
    cue_stick_ball_contacts = [
        (phase_name, c) for phase_name, c in ball_contacts
        if cue_stick_prim_path in (c.actor_path_a, c.actor_path_b)
    ]
    print(f"[bridge] 其中球桿-母球碰撞事件數={len(cue_stick_ball_contacts)}（決策 7 要求恰好 1 次）")

    arm_contacts = [
        (phase_name, c) for phase_name, c in contacts
        if any(p in (c.actor_path_a, c.actor_path_b) for p in arm_link_prim_paths)
    ]
    print(f"[bridge] 手臂本體（forearm/wrist_1~3_link）相關事件（不篩選 impulse）共 {len(arm_contacts)} 筆")
    for phase_name, c in arm_contacts:
        print(f"[bridge]   ARM CONTACT phase={phase_name} a={c.actor_path_a} b={c.actor_path_b} impulse={c.impulse}")

    speed_ok = achievement >= _ACHIEVEMENT_THRESHOLD
    contact_ok = len(cue_stick_ball_contacts) == 1
    arm_ok = len(arm_contacts) == 0
    if speed_ok and contact_ok and arm_ok:
        print("[bridge] PASS：高架橋案例達成率 >=90%、球桿只碰到母球一次、手臂本體零碰撞")
    else:
        reasons = []
        if not speed_ok:
            reasons.append(f"達成率 {100 * achievement:.1f}% < {100 * _ACHIEVEMENT_THRESHOLD:.0f}%")
        if not contact_ok:
            reasons.append(f"球桿-母球碰撞 {len(cue_stick_ball_contacts)} 次（應為 1 次）")
        if not arm_ok:
            reasons.append(f"手臂本體碰撞 {len(arm_contacts)} 筆（應為 0）")
        print(f"[bridge] FAIL：{'；'.join(reasons)}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[bridge] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
