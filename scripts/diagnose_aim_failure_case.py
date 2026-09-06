"""
scripts/diagnose_aim_failure_case.py — 用真實 GUI／完整正式流程（含
ModelController policy）跑出來的實際失敗參數，建立最小可重現案例，深入
診斷 AIM 為什麼收斂失敗（見 scripts/diagnose_production_tick.py 抓到的
數據：cue_ball=(-0.0364, -0.7523) shot_angle=-0.0435 position_offset=
[0.2882, 0.0833]）。

跟 test_ur10e_table_flat.py／test_ur10e_table_bridge.py 的差異：那兩支
只測 position_offset=[0.0, 0.0] 的固定點，這是**第一次**用非零
position_offset 測試——懷疑正是這個維度沒被覆蓋到。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    DEBUG_UR10E_AIM_PHASES=1 \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/diagnose_aim_failure_case.py

結論（2026-09-06）：繞過 orchestrator 直接跑這組參數到底，AIM 在 1453
步後正常收斂（did_last_motion_timeout=False，位置誤差 0.00092m）——證實
「STAGING 階段回報過 timeout=True」不等於「整段動作最終失敗」，確認了
docs/CHANGELOG.md「did_last_motion_timeout() 對 UR10e 提早誤判逾時」的
根因（`ArticulationAPIImpl.did_last_motion_timeout()` 沒有用
`is_motion_complete()` 把關，直接轉發 STAGING 中途翻過一次的累積旗標）。

修好那個 bug 之後用 diagnose_production_tick.py 重跑同一組參數，AIM 不再
卡死，但收斂明顯比 flat/bridge 案例慢很多（STAGING/NEAR_FINAL 大多數
waypoint 都逼近 240 步上限才過）——研判是這組從未測過的大幅
position_offset 把逼近走廊推到接近母球避障力場，RMPflow 收斂變慢（不是
卡死），這個次要問題還沒解決，見 CHANGELOG 該節最後一段。
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CUE_BALL = (-0.036379953073710204, -0.7522665074534715)
_SHOT_ANGLE_DEG = -0.043533146381378174
_POSITION_OFFSET = [0.2882066700108745, 0.08328814658306938]
_CUE_BALL_SPEED = 3.3392
_MAX_STEPS_PER_AIM_ACTION = 20000


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.action import Action
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services import cue_pose_calculator, ur10e_placement_calculator
    from core.services.spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH
    from core.services.ur10e_swing_strategy import Ur10eSwingStrategy

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/TestUr10eAimFailure"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS

    # 先純數學算一次，看這組參數本身的幾何解（不牽扯 RMPflow/避障）長什麼樣。
    wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, table_z, ball_radius, _POSITION_OFFSET,
    )
    print(f"[aimfail] cue_ball={_CUE_BALL} shot_angle={_SHOT_ANGLE_DEG} position_offset={_POSITION_OFFSET}")
    print(f"[aimfail] tilt_rad={tilt_rad} crossing={crossing}")
    if tilt_rad is None:
        print("[aimfail] 幾何無解（tilt_rad=None）——這本身就是 bug：ModelController 不該輸出無解的組合")
        return
    print(f"[aimfail] wrist_position={np.round(wrist_position, 4).tolist()}  "
          f"wrist_orientation={np.round(wrist_orientation, 4).tolist()}")

    direction_unit = cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad)
    base_position = ur10e_placement_calculator.compute_base_position(
        tuple(wrist_position), tuple(direction_unit), table_z
    )
    print(f"[aimfail] direction_unit={np.round(direction_unit, 4).tolist()}  "
          f"base_position={np.round(base_position, 4).tolist()}")
    reach_distance = float(np.linalg.norm(np.asarray(wrist_position) - np.asarray(base_position)))
    print(f"[aimfail] base→wrist 距離={reach_distance:.4f} m（UR10e 可達距離約 1.3m）")

    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, UR10eRobot,
    )
    robot_arm = robot_manager.get_robot()
    ball_prim_path = table_ball_set.get_ball_prim_paths()[0]

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    table_ball_set.place_ball(0, _CUE_BALL[0], _CUE_BALL[1])
    for _ in range(5):
        simulation_app.update()

    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    _TABLE_OBSTACLE_HEIGHT_M = 0.15
    table_center = table.get_table_center()
    table_obstacle_center = [table_center[0], table_center[1], table_z - _TABLE_OBSTACLE_HEIGHT_M / 2.0]
    articulation_api.register_static_box_obstacle(
        table_obstacle_center, [TABLE_WIDTH, TABLE_LENGTH, _TABLE_OBSTACLE_HEIGHT_M]
    )
    articulation_api.register_dynamic_sphere_obstacle(ball_prim_path, ball_radius)

    initial_base_offset = TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    initial_base_position = [table.get_table_center()[i] + initial_base_offset[i] for i in range(3)]
    articulation_api.set_robot_base_pose(initial_base_position, [1.0, 0.0, 0.0, 0.0])

    print("[aimfail] move_to_home()（RESET）...")
    articulation_api.move_to_home()
    reset_steps = 0
    while not articulation_api.is_motion_complete() and reset_steps < 4000:
        simulation_app.update()
        reset_steps += 1
    print(f"[aimfail] RESET steps={reset_steps} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")

    strategy = Ur10eSwingStrategy(robot_arm, articulation_api)
    action = Action(
        cue_ball_speed=_CUE_BALL_SPEED,
        shot_angle=_SHOT_ANGLE_DEG,
        position_offset=_POSITION_OFFSET,
        cue_ball_placement=list(_CUE_BALL),
        should_execute_action=True,
    )

    print("[aimfail] execute_aim() ...")
    strategy.execute_aim(action, tuple(_CUE_BALL), table_z, ball_radius)

    rmp_ctrl = articulation_api._ur10e_rmpflow_controller
    aim_steps = 0
    while not articulation_api.is_motion_complete() and aim_steps < _MAX_STEPS_PER_AIM_ACTION:
        simulation_app.update()
        aim_steps += 1
        if aim_steps % 100 == 0:
            tip_position = articulation_api.get_end_effector_position()
            print(f"[aimfail] AIM step={aim_steps} tip_position={np.round(tip_position, 4).tolist()} "
                  f"awaiting_retract={articulation_api._ur10e_awaiting_arm_move_after_retract} "
                  f"awaiting_staging={articulation_api._ur10e_awaiting_final_approach_after_staging} "
                  f"awaiting_near_final={articulation_api._ur10e_awaiting_final_short_leg_after_near_final} "
                  f"rmp_motion_active={rmp_ctrl._motion_active}")

    print(f"[aimfail] AIM 完成，steps={aim_steps} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")
    achieved_position = np.asarray(articulation_api.get_end_effector_position())
    position_error = float(np.linalg.norm(achieved_position - np.asarray(wrist_position)))
    print(f"[aimfail] 收斂後 wrist 位置={achieved_position.tolist()}  位置誤差（對預設 roll 目標）={position_error:.5f} m")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[aimfail] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
