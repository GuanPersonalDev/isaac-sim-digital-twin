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

⚠️ 2026-09-03 目前狀態：FAIL，深入 root cause 調查記錄如下。
cue_ball=(0.0, 0.5)（flat，tilt_rad=0）走完整 Ur10eSwingStrategy.
execute_aim() 流程後，AIM 卡在殘留誤差不收斂。

已排除的假設（逐一實測驗證過）：
- ArticulationAPIImpl 包裝層本身的問題——用另一支診斷腳本直接呼叫
  Ur10eRmpflowController（繞開 ArticulationAPIImpl）測同一個 cue_ball，
  結果幾乎一樣（0.69m 殘留誤差），排除是步驟 6 整合引入的 bug。
- reposition() 時機（模擬已在跑時才呼叫 vs. 播放前呼叫）——用另一支
  診斷腳本直接比對 Articulation.get_world_poses()，確認 reposition()
  無論何時呼叫都會立即正確反映到 tensor-based Articulation 的 physics
  root transform，這個環節沒有問題。
- PHYSICS_POST_STEP callback 的 step_dt／呼叫頻率——加了計數器＋數值
  log 確認每個 physics tick 正好呼叫一次、step_dt 穩定為 1/60，正常。
- waypoint 純位置距離決定段數，沒有依旋轉角度加開更多段——加了
  _MAX_WAYPOINT_ROTATION_RAD 補上這個機制（見
  Ur10eRmpflowController.move_to_pose()），對這個案例沒有實質改善
  （本來的段數就已經被位置距離撐夠了），但保留下來當一般性強化。
- HOME 的 wrist_2_joint=0 是不是踩到 UR 家族手臂的手腕奇異點——實測把
  wrist_2 從 0 改到 π/2，結果反而更差（HOME 本身開始逾時、AIM 殘留誤差
  從 0.16m 惡化到 0.20m），已改回官方原始 default_q，這個假設沒有被
  證實。

有實質幫助但沒有完全解決的發現：raw USD 預設姿態（沒走過 RESET）到某些
AIM 目標需要接近 180 度的姿態翻轉，會讓 RMPflow 卡在很差的殘留誤差
（0.62m）；改成先呼叫 move_to_home()（模擬正式流程的 RESET 階段，
production 本來就是 RESET→AIM，不會從 raw 預設姿態直接跳 AIM）之後，
殘留誤差降到 0.16m——明顯更好，但仍未達到容許誤差。

具體診斷證據：逐 waypoint 記錄過六個關節角度，flat 案例卡住時
wrist_1_joint 在短短幾個 waypoint 內從接近 0 衝到超過 -π（-3.5+ rad）
才折返，elbow_joint 也在某個 waypoint 之後由遞增轉為遞減——這個「先衝
過頭、方向反轉、卡住不動」的模式，指向手臂被迫做過大的姿態翻轉。

真正找到的根因：cue_pose_calculator.compute_tilted_wrist_pose() 的
roll_rad（球桿繞自身軸的冗餘自由度，不影響桿頭實際指向）預設用 0（最短
弧慣例），但這個選擇剛好讓 flat 案例的目標姿態跟 HOME 附近的實際朝向
接近正反面，需要接近 180 度翻轉。改用
ur10e_placement_calculator.compute_roll_minimizing_reorientation()
搜尋讓翻轉角度最小的 roll_rad 後（同一個指向改用另一個滾動角表示），
所需翻轉角度從 180 度降到 90 度，AIM 殘留誤差從 0.16m（HOME-first 版）
大幅降到 0.056m——已經是 10 倍以上的改善，但仍未收斂到 1cm 容許值內，
研判是接近舒適姿態邊界的最後一段路徑本身還有殘留的局部穩定點，需要更
進一步的策略（例如收斂到接近後改用差動 IK 之類的精確控制器收尾）才能
完全解決，留給後續處理。

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

    # ⚠️ 2026-09-03 除錯發現：raw USD 預設姿態（沒走過 RESET）到某些 AIM
    # 目標需要接近 180 度的姿態翻轉，會讓 RMPflow 卡住不收斂（flat 案例
    # 實測卡在 0.62m）；真正的 production 流程是 RESET（回 HOME）→ AIM，
    # 不會從 raw 預設姿態直接跳 AIM。這裡先呼叫 move_to_home()，模擬
    # 正式流程的 RESET 階段，再執行 AIM。
    # move_to_home() 需要 RMPflow 先知道目前底座的真實世界位姿——這時候
    # 手臂還在 TableRobotManager 建構時的初始固定偏移位置
    # （TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER），還沒被
    # execute_aim() 的 per-shot reposition() 移動過，先同步這個初始位置。
    initial_base_offset = TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    initial_base_position = [
        table.get_table_center()[0] + initial_base_offset[0],
        table.get_table_center()[1] + initial_base_offset[1],
        table.get_table_center()[2] + initial_base_offset[2],
    ]
    articulation_api.set_robot_base_pose(initial_base_position, [1.0, 0.0, 0.0, 0.0])

    print("[flat] 呼叫 articulation_api.move_to_home()（模擬 RESET 階段）...")
    articulation_api.move_to_home()
    _run_until_complete("RESET(HOME)")

    print("[flat] 呼叫 Ur10eSwingStrategy.execute_aim() ...")
    strategy.execute_aim(action, tuple(_CUE_BALL), table_z, ball_radius)
    _run_until_complete("AIM")

    live_wrist_position = np.asarray(articulation_api.get_end_effector_position())
    aim_error = float(np.linalg.norm(live_wrist_position - np.asarray(wrist_position)))
    print(f"[flat] AIM 收斂後 wrist 位置={live_wrist_position.tolist()}  跟目標誤差={aim_error:.5f} m")

    # 診斷：RMPflow 最後一次算出的關節目標 vs 實際量到的關節位置——
    # 用來分辨殘留誤差是「PhysX joint drive 追不上 RMPflow 給的目標」
    # （tracking gap，可靠 stiffness/damping 補強修，見 UR3e
    # _boost_wrist_gains_for_cue_stick_load() 先例）還是「RMPflow 自己算出
    # 的關節目標，即使完美追上也不對應期望的末端位姿」（RMPflow 本身的
    # reactive 殘留，需要換精確控制器收尾）。
    rmp_ctrl = articulation_api._ur10e_rmpflow_controller
    last_targets = rmp_ctrl._last_active_position_targets
    if last_targets is not None:
        current_full_positions = np.asarray(articulation_api._articulation.get_dof_positions())[0]
        current_active_positions = current_full_positions[rmp_ctrl._active_dof_indices]
        joint_gap = current_active_positions - last_targets
        print(f"[flat] active_joint_names={rmp_ctrl._active_joint_names}")
        print(f"[flat] RMPflow 最後關節目標={last_targets.tolist()}")
        print(f"[flat] 實際量到的關節位置    ={current_active_positions.tolist()}")
        print(f"[flat] 關節 tracking gap (實際-目標)={joint_gap.tolist()}  max_abs_gap={np.max(np.abs(joint_gap)):.6f} rad")

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
