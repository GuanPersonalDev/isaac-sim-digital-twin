"""
scripts/diagnose_production_tick.py — 一次性診斷：真實 GUI 執行
billiard_digital_twin extension 時，`BILLIARD_DEBUG_LOG_PATH` 逐 tick log
完全沒有任何 `state=...` 行（連 tick=0 都沒有），只有碰撞事件；肉眼看手臂
完全靜止。主控台也沒有任何 Python traceback。

既有的 scripts/test_ur10e_table_flat.py 等驗收腳本全部繞過
`DemoTableOrchestrator`／`ModelController`／`TorchScriptPolicyImpl`／
`TableRuntime.tick()`，直接呼叫 `Ur10eSwingStrategy.execute_aim()`／
`execute_strike()`——換句話說，**這條完整的正式 tick 路徑，接上 UR10e
之後從來沒有被真實執行過**，這支腳本要補的就是這個缺口。

做法：照 `extension/billiard_digital_twin/billiard_digital_twin.py`
`_enable_training()`／`_build_demo_session()` 的呼叫順序，手動組出跟
production 完全一樣的物件圖（不碰 DebugMenu/UI，那些跟這個問題無關且
headless SimulationApp 不一定有對應擴充套件），然後**自己**呼叫
`session.tick()`（不透過 `SimulationManager.register_callback()`），
每次都包 try/except 印出完整 traceback——如果 production 的
`SimulationManager` callback 分派機制真的會吞掉例外，這裡才有辦法
看到真正的錯誤。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/diagnose_production_tick.py

結論（2026-09-06）：`session.tick()` 本身不拋例外——問題出在
`TableOrchestrator.step()` 內部的 try/except 把例外吞進
`ErrorState.mark_error()`，該方法用標準 `logging.exception()` 記錄，這個
Kit 環境的主控台完全看不到。直接讀 `demo_error_state.get_last_exception()`
才抓到真正的例外：`RuntimeError: 手臂動作逾時未收斂`，根因與修法見
docs/CHANGELOG.md「did_last_motion_timeout() 對 UR10e 提早誤判逾時」。
修好之後同一支腳本可以拿來重跑驗證（見程式碼裡的
`_traced_execute_aim`／`DEBUG_UR10E_AIM_PHASES` 輔助輸出）。
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MAX_TICKS = 8000


def _run() -> None:
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaacsim.core.simulation_manager import SimulationManager

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.torch_script_policy_impl import TorchScriptPolicyImpl

    from core.controllers.model_controller import ModelController
    from core.models.billiard_table import BilliardTable
    from core.models.billiard_state import BilliardStatus
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services.error_state import ErrorState
    from core.services.impulse_striking_service import ImpulseStrikingService
    from core.services.observation_builder import DemoTableObservationBuilder, TrainingTableObservationBuilder
    from core.services.pocket_event_handler import PocketEventHandler
    from core.services.robot_swing_strategy import create_swing_strategy_for
    from core.services.rolling_resistance_service import RollingResistanceService
    from core.services.table_orchestrator import DemoTableOrchestrator, TrainingTableOrchestrator
    from core.services.table_runtime import TableRuntime
    from core.services.table_session import DemoTableSession, TableSession
    from core.models.table_ball_set import TableBallSet

    _POLICY_PATH = os.path.join(_PROJECT_ROOT, "models", "rl", "billiard", "policy.pt")
    _EVAL_MAX_OFFSET = 0.6

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    print("[diag] SimulationManager.setup_simulation(dt=1/60) ...")
    SimulationManager.setup_simulation(dt=1 / 60)

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    rolling_resistance_service = RollingResistanceService(rigid_body_api, TableBallSet.DEFAULT_BALL_RADIUS)
    policy = TorchScriptPolicyImpl(_POLICY_PATH)

    def _build_pocket_event_handler(table, table_ball_set):
        from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
        handler = PocketEventHandler(
            physics_api=PhysicsAPIImpl(),
            pocket_prim_paths=table.get_pocket_prim_paths(),
            ball_prim_paths=table_ball_set.get_ball_prim_paths(),
            on_ball_pocketed=table_ball_set.hide_ball,
        )
        handler.start()
        return handler

    # === Training 桌（跟 production _enable_training() 一樣先建）===
    print("[diag] 建立 Training 桌 /World/Table_0 ...")
    training_table = BilliardTable("/World/Table_0", stage_api, material_api, rigid_body_api, (2.6, 2.6))
    training_ball_set = training_table.get_table_ball_set()
    training_pocket_handler = _build_pocket_event_handler(training_table, training_ball_set)
    training_controller = ModelController(policy, training_ball_set.get_table_x_y(), _EVAL_MAX_OFFSET)
    training_error_state = ErrorState()
    impulse_striking_service = ImpulseStrikingService(
        rigid_body_api, training_ball_set.get_ball_prim_paths()[0], training_ball_set.get_ball_radius()
    )
    training_runtime = TableRuntime(
        TrainingTableObservationBuilder(
            training_ball_set, rigid_body_api, training_ball_set.ball_motion_monitor,
            training_error_state, training_table.position_provider,
        ),
        TrainingTableOrchestrator(
            training_controller, training_ball_set, training_table.position_provider,
            impulse_striking_service, training_error_state, rolling_resistance_service,
        ),
    )
    training_session = TableSession("/World/Table_0", training_table, training_runtime, training_pocket_handler, rigid_body_api)

    # === Demo 桌（跟 production _build_demo_session() 一樣）===
    print("[diag] 建立 Demo 桌 /World/Table_Demo ...")
    demo_table_path = "/World/Table_Demo"
    demo_table = BilliardTable(demo_table_path, stage_api, material_api, rigid_body_api, (0, 0))

    robot_prim_path = UR10eRobot.get_prim_path(demo_table_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(demo_table_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    robot_manager = TableRobotManager(
        demo_table.get_table_center(), demo_table_path, stage_api, articulation_api, UR10eRobot,
    )
    demo_ball_set = demo_table.get_table_ball_set()
    robot_arm = robot_manager.get_robot()

    demo_pocket_handler = _build_pocket_event_handler(demo_table, demo_ball_set)
    demo_controller = ModelController(policy, demo_ball_set.get_table_x_y(), _EVAL_MAX_OFFSET)
    demo_error_state = ErrorState()
    swing_strategy = create_swing_strategy_for(robot_arm, articulation_api)
    demo_runtime = TableRuntime(
        DemoTableObservationBuilder(
            demo_ball_set, rigid_body_api, demo_ball_set.ball_motion_monitor,
            demo_error_state, demo_table.position_provider, robot_arm,
        ),
        DemoTableOrchestrator(
            demo_controller, demo_ball_set, demo_table.position_provider, robot_arm,
            articulation_api, swing_strategy, demo_error_state, rolling_resistance_service,
        ),
    )
    demo_session = DemoTableSession(
        demo_table_path, demo_table, demo_runtime, demo_pocket_handler, rigid_body_api,
        robot_manager, articulation_api,
    )

    # 攔截 execute_aim() 印出實際參數——ModelController 的 policy 決定的
    # cue_ball_placement/shot_angle 不是我們自己選的，需要知道真正的值才能
    # 建立最小可重現的 headless 案例。
    _original_execute_aim = swing_strategy.execute_aim

    def _traced_execute_aim(action, cue_ball, table_z, ball_radius):
        print(f"[diag] execute_aim() 實際參數：cue_ball={cue_ball} shot_angle={action.shot_angle} "
              f"cue_ball_speed={action.cue_ball_speed} position_offset={action.position_offset} "
              f"table_z={table_z} ball_radius={ball_radius}")
        sys.stdout.flush()
        return _original_execute_aim(action, cue_ball, table_z, ball_radius)

    swing_strategy.execute_aim = _traced_execute_aim

    print("[diag] timeline.play() ...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    print("[diag] demo_session.initialize_articulation()（跟 _on_play() 一樣）...")
    demo_session.initialize_articulation()
    demo_session.request_full_reset()
    training_session.request_full_reset()
    for _ in range(5):
        simulation_app.update()

    # === 手動驅動 tick，不透過 SimulationManager.register_callback()，
    # 直接暴露例外 ===
    print(f"[diag] 開始手動 tick（最多 {_MAX_TICKS} 次）...")
    last_state = None
    for tick in range(_MAX_TICKS):
        simulation_app.update()
        try:
            training_session.tick()
        except Exception:
            print(f"[diag] tick={tick} training_session.tick() 拋出例外：")
            traceback.print_exc()
            sys.stdout.flush()
            raise
        try:
            demo_session.tick()
        except Exception:
            print(f"[diag] tick={tick} demo_session.tick() 拋出例外：")
            traceback.print_exc()
            sys.stdout.flush()
            raise

        state = demo_session.get_current_state()
        observation = demo_session.get_last_observation()
        if state != last_state:
            print(f"[diag] tick={tick} demo state 變化：{last_state} -> {state}  "
                  f"has_error={observation.has_error if observation else None}")
            last_state = state
        if observation is not None and observation.has_error:
            print(f"[diag] tick={tick} has_error=True，停止")
            last_exception = demo_error_state.get_last_exception()
            if last_exception is not None:
                print("[diag] ErrorState 記錄的例外：")
                traceback.print_exception(type(last_exception), last_exception, last_exception.__traceback__)
                sys.stdout.flush()
            else:
                print("[diag] ErrorState 沒有記錄到例外物件（has_error=True 但 get_last_exception()=None）")
            break
        if tick % 200 == 0:
            tip_position = articulation_api.get_end_effector_position()
            rmp = articulation_api._ur10e_rmpflow_controller
            slide = articulation_api._ur10e_cue_slide_controller
            print(f"[diag] tick={tick} state={state} tip_position={tip_position} "
                  f"awaiting_retract={articulation_api._ur10e_awaiting_arm_move_after_retract} "
                  f"awaiting_staging={articulation_api._ur10e_awaiting_final_approach_after_staging} "
                  f"awaiting_near_final={articulation_api._ur10e_awaiting_final_short_leg_after_near_final} "
                  f"active_ctrl={'rmp' if articulation_api._ur10e_active_controller is rmp else ('slide' if articulation_api._ur10e_active_controller is slide else 'linear_approach')} "
                  f"rmp_motion_active={rmp._motion_active} rmp_finishing={rmp._finishing_active} "
                  f"rmp_joint_finish={rmp._joint_finish_active} rmp_waypoint_index={rmp._waypoint_index}/{len(rmp._waypoints)} "
                  f"did_timeout={articulation_api.did_last_motion_timeout()}")

    print(f"[diag] 手動 tick 結束，最終 state={demo_session.get_current_state()}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[diag] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
