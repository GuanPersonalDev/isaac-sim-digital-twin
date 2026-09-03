"""
scripts/test_ur10e_table_flat.py - UR10e+專用出力結構重新設計計畫
步驟 6：整合進 table_orchestrator.py 後的真實驗證（flat 案例）。

直接呼叫**正式生產程式碼**（Ur10eSwingStrategy.execute_aim()/
execute_strike()，透過真正的 ArticulationAPIImpl，不是各自獨立測試用的
Ur10eRmpflowController/Ur10eCueSlideController 直接呼叫）驗證整合後的
完整 AIM->STRIKE 流程：手臂能不能真的把母球打出去，達成合理比例的目標
球速。

跟稍早的 scripts/verify_ur10e_rmpflow_aim.py 差異：那支只驗證手臂能不能
收斂到 AIM 目標姿態；這支往下多驗證 STRIKE（滑軌關節出力）跟真實碰撞
（母球有沒有被打到、達成率多少），且全程走
core/services/ur10e_swing_strategy.py 的正式程式碼路徑（robot_arm.
reposition()/articulation_api.set_robot_base_pose()/move_to_pose()/
move_cue_slide_stroke()），不是繞過 Strategy 直接操作底層 controller。

先只測 flat 案例（cue_ball 選在遠離庫邊、tilt_rad<=1e-6 的位置），高架橋
案例留給步驟 8。

⚠️ 2026-09-03 目前狀態：FAIL，一個尚未解決的發現。cue_ball=(0.0, 0.5)
（flat，tilt_rad=0）走完整 Ur10eSwingStrategy.execute_aim() 流程後，
AIM 卡在約 0.62m 殘留誤差（16 個中繼 waypoint 跑完但多段逾時未收斂）。

已排除的假設（逐一實測驗證過，見對話記錄）：
- ArticulationAPIImpl 包裝層本身的問題——用另一支診斷腳本直接呼叫
  Ur10eRmpflowController（繞開 ArticulationAPIImpl）測同一個 cue_ball，
  結果幾乎一樣（0.69m 殘留誤差），排除是這次步驟 6 整合引入的 bug。
- reposition() 時機（模擬已在跑時才呼叫 vs. 播放前呼叫）——用另一支
  診斷腳本直接比對 Articulation.get_world_poses()，確認 reposition()
  無論何時呼叫都會立即正確反映到 tensor-based Articulation 的 physics
  root transform，這個環節沒有問題。
- PHYSICS_POST_STEP callback 的 step_dt／呼叫頻率——加了計數器＋數值
  log 確認每個 physics tick 正好呼叫一次、step_dt 穩定為 1/60，正常。

還沒解開的部分：這個 flat 案例（tilt_rad=0，目標朝向幾乎是 identity）
明明看起來比稍早驗證通過的高架橋案例（scripts/verify_ur10e_rmpflow_aim.py，
cue_ball=(-0.036,-0.752)，目標朝向也接近但不是 identity）更簡單，卻卡住；
高架橋案例反而 16 個 waypoint 全部順利收斂到 <2mm。兩者的手臂起始姿態、
waypoint 段數、單段位移量級都相近，目前找不到能解釋這個差異的具體原因，
需要更多時間深入研究 RMPflow 在這個特定案例下卡住的 RMP 分量互動，或
考慮調整 waypoint 拆分策略/rmp_params 增益。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_ur10e_table_flat.py
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CUE_BALL = (0.0, 0.5)
_SHOT_ANGLE_DEG = 0.0
_CUE_BALL_SPEED = 1.995
_PHYSICS_DT = 1.0 / 60.0
_MAX_STEPS_PER_ACTION = 4000


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.action import Action
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services import cue_pose_calculator, swing_trajectory_calculator
    from core.services.ur10e_swing_strategy import Ur10eSwingStrategy
    from isaacsim.core.experimental.prims import RigidPrim

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/TestUr10eTableFlat"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS

    wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, table_z, ball_radius,
    )
    if tilt_rad is None:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 幾何無解")
    print(f"[flat] cue_ball={_CUE_BALL}  tilt_rad={tilt_rad:.6f}（應該 <= 1e-6，flat 案例）")
    if tilt_rad > 1e-6:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 不是 flat 案例（tilt_rad={tilt_rad}），改別的座標")

    print("[flat] 建立 ArticulationAPIImpl ...")
    sys.stdout.flush()
    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    print("[flat] 建立 TableRobotManager ...")
    sys.stdout.flush()
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, UR10eRobot,
    )
    robot_arm = robot_manager.get_robot()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    ball_prim_path = table_ball_set.get_ball_prim_paths()[0]

    print("[flat] timeline.play() ...")
    sys.stdout.flush()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    # ⚠️ place_ball() 必須在 timeline.play() 之後才能呼叫——球的
    # xformOp:orient 屬性要等 physics/timeline 先初始化過一次才會存在
    # （實測踩過：AssertionError: Undefined 'xformOp:orient' property）。
    # production 路徑（DemoTableOrchestrator._execute_aim()）本來就是在
    # 模擬進行中呼叫，這裡改成跟正式路徑一致的時序，不是在場景剛建好、
    # timeline 還沒 play 的時候呼叫。
    print("[flat] place_ball ...")
    sys.stdout.flush()
    table_ball_set.place_ball(0, _CUE_BALL[0], _CUE_BALL[1])
    for _ in range(5):
        simulation_app.update()

    print("[flat] articulation_api.initialize() ...")
    sys.stdout.flush()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()
    print("[flat] initialize() 完成")
    sys.stdout.flush()

    ball_rigid_prim = RigidPrim(paths=ball_prim_path)

    contacts = []
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    physics_api.enable_contact_reporting(ball_prim_path)
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    strategy = Ur10eSwingStrategy(robot_arm, articulation_api)
    action = Action(
        cue_ball_speed=_CUE_BALL_SPEED,
        shot_angle=_SHOT_ANGLE_DEG,
        position_offset=[0.0, 0.0],
        cue_ball_placement=list(_CUE_BALL),
        should_execute_action=True,
    )

    def _run_until_complete(label: str) -> int:
        step = 0
        while not articulation_api.is_motion_complete() and step < _MAX_STEPS_PER_ACTION:
            simulation_app.update()
            step += 1
        print(f"[flat] {label} 完成，steps={step} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")
        return step

    print("[flat] 呼叫 Ur10eSwingStrategy.execute_aim() ...")
    strategy.execute_aim(action, tuple(_CUE_BALL), table_z, ball_radius)
    _run_until_complete("AIM")

    live_wrist_position = np.asarray(articulation_api.get_end_effector_position())
    aim_error = float(np.linalg.norm(live_wrist_position - np.asarray(wrist_position)))
    print(f"[flat] AIM 收斂後 wrist 位置={live_wrist_position.tolist()}  跟目標誤差={aim_error:.5f} m")

    ball_velocity_before, _ = ball_rigid_prim.get_velocities()
    print(f"[flat] STRIKE 前母球速度={np.asarray(ball_velocity_before[0]).tolist()}（應接近 0）")

    print("[flat] 呼叫 Ur10eSwingStrategy.execute_strike() ...")
    strategy.execute_strike(action, tuple(_CUE_BALL), table_z, ball_radius)
    strike_steps = _run_until_complete("STRIKE")

    # 揮桿完成後再多跑幾步，讓球的速度穩定下來（衝量剛作用完那一瞬間的
    # get_velocities() 可能還沒反映完整動量轉移）。
    for _ in range(10):
        simulation_app.update()

    ball_velocity_after, _ = ball_rigid_prim.get_velocities()
    ball_velocity_after = np.asarray(ball_velocity_after[0])
    ball_speed_after = float(np.linalg.norm(ball_velocity_after))
    print(f"[flat] STRIKE 後母球速度向量={ball_velocity_after.tolist()}")
    print(f"[flat] STRIKE 後母球速度={ball_speed_after:.4f} m/s  目標母球速度={_CUE_BALL_SPEED} m/s  達成率={100 * ball_speed_after / _CUE_BALL_SPEED:.1f}%")

    ball_contacts = [
        c for c in contacts
        if ball_prim_path in (c.actor_path_a, c.actor_path_b) and c.impulse > 0.0
    ]
    print(f"[flat] 母球碰撞事件數（impulse>0）={len(ball_contacts)}")
    for c in ball_contacts:
        print(f"[flat]   CONTACT a={c.actor_path_a} b={c.actor_path_b} impulse={c.impulse}")

    if ball_speed_after >= 0.5 * _CUE_BALL_SPEED and len(ball_contacts) >= 1:
        print("[flat] PASS：完整 AIM->STRIKE 流程成功把母球打出去")
    else:
        print("[flat] FAIL：母球沒有被打到，或速度遠低於預期")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[flat] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
