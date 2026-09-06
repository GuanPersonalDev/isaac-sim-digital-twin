import itertools
import sys
import os
import time
import omni.ext
import omni.usd
import omni.timeline
import omni.kit.app
import carb.events
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for p in [_EXT_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.controllers.model_controller import ModelController
from core.models.table_ball_set import TableBallSet
from core.models.robot_arm import RobotArm
from core.models.barrett_wam_robot import BarrettWamRobot
from core.models.ur3e_robot import UR3eRobot
from core.models.ur10e_robot import UR10eRobot
from core.ports import RigidBodyAPI
from core.ports.policy_port import PolicyPort
from core.services.asset_utility import TABLE_PATH
from core.services.observation_builder import DemoTableObservationBuilder, TrainingTableObservationBuilder
from core.services.robot_swing_strategy import create_swing_strategy_for
from core.services.table_orchestrator import DemoTableOrchestrator, TrainingTableOrchestrator
from core.services.table_runtime import TableRuntime
from core.services.table_session import DemoTableSession, TableSession
from core.ports.material_api import MaterialAPI
from core.ports.stage_api import StageAPI
from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
from isaac_sim_impl_6_0.physics_scene_tuning import configure_physics_scene_for_demo_scale
from isaac_sim_impl_6_0.torch_script_policy_impl import TorchScriptPolicyImpl
from ui.debug_menu import DebugMenu
from ui.tool_menu_registry import discover_and_register, unregister
from core.models.billiard_table import BilliardTable
from core.models.table_robot_manager import TableRobotManager
from core.services.error_state import ErrorState
from core.services.impulse_striking_service import ImpulseStrikingService
from core.services.pocket_event_handler import PocketEventHandler
from core.services.rolling_resistance_service import RollingResistanceService

_TABLE_COUNT = 1
_TOOL_MENU_NAME = "Tools"
# Demo 桌實際掛載的手臂類別，換手臂只需要改這一行（見 core/models/robot_arm.py）。
# 2026-09-06（UR10e 重新設計計畫步驟 9）：UR10e＋線性滑軌推桿機構通過 flat
# 與高架橋兩個案例的真實球檯驗收（scripts/test_ur10e_table_flat.py／
# test_ur10e_table_bridge.py：達成率 93.5%／109.2%、球桿-母球碰撞各恰好
# 1 次、手臂本體碰撞 0 筆），從 UR3e 切換過來。
#
# UR3e 這條路徑（core/services/ur3e_placement_calculator.py／
# ArticulationAPIImpl.move_swing_elbow_pivot()）保留但不再被生產路徑呼叫，
# 原因見 core/services/ur3e_swing_strategy.py 開頭的說明（加權多關節驅動
# 在 UR3e 幾何下有 manipulability ellipsoid 的結構性限制）。
_ROBOT_ARM_CLASS: type[RobotArm] = UR10eRobot
_TABLE_SIZE_PROBE_PATH = "/World/_TableSizeProbe"
_POLICY_PATH = os.path.join(_PROJECT_ROOT, "models", "rl", "billiard", "policy.pt")
_EVAL_MAX_OFFSET = 0.6

# 以下兩個環境變數僅供無人值守除錯用（headful GUI 用
# `isaacsim.exe ... --enable billiard_digital_twin` 開啟、跑固定時間後由外部
# timeout 關閉），不需要真人手動點 Play、肉眼盯著畫面，就能拿到逐 tick 的
# 關節角度/桿尖位置/母球位置數據事後分析。兩者預設都是關閉（空字串/0），
# 只有明確設定環境變數才會啟用，正常互動 GUI 行為完全不受影響。
_AUTO_PLAY_DELAY_SEC = float(os.environ.get("BILLIARD_AUTO_PLAY_DELAY_SEC", "0") or "0")
_DEBUG_LOG_PATH = os.environ.get("BILLIARD_DEBUG_LOG_PATH", "")

def _format_vector(values: list[float]) -> str:
    # 固定小數點後 3 位，避免 Debug Menu 每幀因為浮點數位數不一而跳動版面。
    return "(" + ", ".join(f"{v:.3f}" for v in values) + ")"


class BilliardExtension(omni.ext.IExt):
    _TIMELINE_EVENT_NAME = "billiard_digital_twin_timeline_wait"
    _PHYSIC_CALL_BACK = "billiard_table_tick"
    def on_startup(self, ext_id: str):
        self._debug_menu = None
        self._training_sessions: list[TableSession] = []
        self._demo_sessions: list[DemoTableSession] = []
        self._demo_articulation_apis: dict[str, ArticulationAPIImpl] = {}
        # Training 球檯預設關閉（效能，見 docs/CHANGELOG.md「GUI FPS 調校」）：
        # 在 GUI Demo 情境下沒有畫面用途，需要時可從 Debug Menu 的 toggle 開回來。
        self._training_enabled = False
        self._demo_enabled = True
        self._timeline_playing = False
        self._table_unit_side_length = 0.0
        self._tick_callback_id = None
        self._policy: PolicyPort | None = None
        self._debug_log_file = None
        self._debug_tick_counter = 0
        self._auto_play_sub = None
        self._physics_api_debug = None
        scripts_dir = os.path.join(_PROJECT_ROOT, "scripts")
        self._tool_menu_items = discover_and_register(scripts_dir, _TOOL_MENU_NAME)
        if _DEBUG_LOG_PATH:
            self._debug_log_file = open(_DEBUG_LOG_PATH, "w", encoding="utf-8")
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            self._billiard_init()
        else:
            stream = omni.usd.get_context().get_stage_event_stream()
            self._sub = stream.create_subscription_to_pop(
                self._on_stage_event, name="billiard_digital_twin_stage_wait"
            )
        if _AUTO_PLAY_DELAY_SEC > 0:
            self._auto_play_deadline = time.time() + _AUTO_PLAY_DELAY_SEC
            self._auto_play_sub = (
                omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                    self._on_auto_play_update, name="billiard_digital_twin_auto_play"
                )
            )

    def _on_auto_play_update(self, event: carb.events.IEvent) -> None:
        """僅供無人值守除錯用（見 `_AUTO_PLAY_DELAY_SEC` 說明）：Scene 載入
        後不需要真人點 Play，等固定秒數自動觸發，讓外部腳本可以直接開
        headful GUI、等固定時間、讀 debug log、關閉，不需要人在旁邊操作。"""
        if time.time() < self._auto_play_deadline:
            return
        self._auto_play_sub = None
        omni.timeline.get_timeline_interface().play()

    def _on_stage_event(self, event: carb.events.IEvent) -> None:
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._billiard_init()
            self._sub = None

    def _billiard_init(self):
        SimulationManager.setup_simulation(dt=1/60)
        # 覆寫預設的 GPU dynamics（見 configure_physics_scene_for_demo_scale 說明）
        configure_physics_scene_for_demo_scale(omni.usd.get_context().get_stage())

        self._asset_env_init()

        if _DEBUG_LOG_PATH:
            # GUI 逐 tick log 只有關節角度/位置，看不出是不是真的撞到東西
            # （例如球桿後擺過程掃到地板），加碰撞事件回報直接證實。跟
            # PocketEventHandler 各自獨立一個 PhysicsAPIImpl 實例／各自一個
            # subscribe_contact_events()：
            # enable_contact_reporting() 是把 PhysxContactReportAPI 掛到 USD
            # prim 上（stage 層級的效果，不屬於特定 subscriber），
            # subscribe_contact_events() 訂閱的是 PhysX 全域的 contact report
            # 事件流，兩個獨立訂閱互不干擾、也不需要共用同一個實例。
            from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
            self._physics_api_debug = PhysicsAPIImpl()
            self._physics_api_debug.subscribe_contact_events(self._on_debug_contact_event)

        self._debug_menu = DebugMenu(
            self._on_training_toggle,
            self._on_demo_toggle,
            self.get_table_ids,
            self.get_table_debug_info,
            self.get_ball_velocities_text,
        )

        self._event_init()

        self._tick_callback_id = SimulationManager.register_callback(
            self._on_tick, event=SimulationEvent.PHYSICS_POST_STEP
        )

        # 預設兩個開關皆為 True：開機即建立所有桌子，不需要等 Timeline Play。
        self._on_training_toggle(self._training_enabled)
        self._on_demo_toggle(self._demo_enabled)

    def _asset_env_init(self):
        self._stage_api = StageAPIImpl()
        self._material_api = MaterialAPIImpl()
        self._rigid_body_api = RigidBodyAPIImpl()

        self._table_unit_side_length = self._measure_table_unit_side_length(self._stage_api)
        self._rolling_resistance_service = RollingResistanceService(
            self._rigid_body_api, TableBallSet.DEFAULT_BALL_RADIUS
        )

    def _measure_table_unit_side_length(self, stage_api: StageAPI) -> float:
        """
        用一次性量測用 prim 取得單張桌子的邊長，量完立刻移除，不依賴任何一張
        正式的 Training/Demo 桌是否已經建立（Toggle 完全解耦後，兩者都可能
        還沒被啟用）。
        """
        stage_api.create_reference_prim(_TABLE_SIZE_PROBE_PATH, TABLE_PATH)
        x_length, y_length, z_length = stage_api.get_prim_sides(_TABLE_SIZE_PROBE_PATH)
        stage_api.remove_prim(_TABLE_SIZE_PROBE_PATH)
        return max(x_length, y_length, z_length)

    def _build_table(
        self,
        table_name: str,
        stage_api: StageAPI,
        material_api: MaterialAPI,
        rigid_body_api: RigidBodyAPI,
        pos: tuple[float, float],
    ) -> BilliardTable:
        table = BilliardTable(table_name, stage_api, material_api, rigid_body_api, pos)
        return table

    def _build_pocket_event_handler(self, table: BilliardTable, table_ball_set) -> PocketEventHandler:
        physics_api = PhysicsAPIImpl()
        handler = PocketEventHandler(
            physics_api=physics_api,
            pocket_prim_paths=table.get_pocket_prim_paths(),
            ball_prim_paths=table_ball_set.get_ball_prim_paths(),
            on_ball_pocketed=table_ball_set.hide_ball,
        )
        handler.start()
        return handler

    def _enable_training(self) -> None:
        # 計算單邊撞球桌的個數
        side_count = 1
        while _TABLE_COUNT > side_count * side_count:
            side_count += 1

        index = 0
        for i in range(side_count):
            for j in range(side_count):
                x_pos = self._table_unit_side_length * (i + 1)
                y_pos = self._table_unit_side_length * (j + 1)
                table_id = f"/World/Table_{index}"
                table = self._build_table(
                    table_id, self._stage_api, self._material_api, self._rigid_body_api, (x_pos, y_pos)
                )
                self._training_sessions.append(self._build_training_session(table_id, table))
                index += 1

        if self._debug_menu:
            self._debug_menu.set_available_tables(self.get_table_ids())

    def _build_training_session(self, table_id: str, table: BilliardTable) -> TableSession:
        table_ball_set = table.get_table_ball_set()
        if table_ball_set is None:
            raise RuntimeError(f"{table_id} 剛建立卻沒有 TableBallSet，無法建立 TableSession")
        pocket_handler = self._build_pocket_event_handler(table, table_ball_set)
        controller = self._build_model_controller(table_ball_set)
        error_state = ErrorState()
        impulse_striking_service = ImpulseStrikingService(
            self._rigid_body_api, table_ball_set.get_ball_prim_paths()[0], table_ball_set.get_ball_radius()
        )
        runtime = TableRuntime(
            TrainingTableObservationBuilder(
                table_ball_set, self._rigid_body_api, table_ball_set.ball_motion_monitor, error_state, table.position_provider
            ),
            TrainingTableOrchestrator(
                controller, table_ball_set, table.position_provider, impulse_striking_service, error_state, self._rolling_resistance_service
            ),
        )
        return TableSession(table_id, table, runtime, pocket_handler, self._rigid_body_api)

    def _disable_training(self) -> None:
        for session in self._training_sessions:
            session.destroy()
        self._training_sessions = []
        if self._debug_menu:
            self._debug_menu.set_available_tables(self.get_table_ids())

    def _enable_demo(self) -> None:
        demo_table_path = "/World/Table_Demo"
        table = self._build_table(
            demo_table_path, self._stage_api, self._material_api, self._rigid_body_api, (0, 0)
        )

        session = self._build_demo_session(demo_table_path, table)
        if self._timeline_playing:
            session.initialize_articulation()
        self._demo_sessions.append(session)

        if self._debug_menu:
            self._debug_menu.set_available_tables(self.get_table_ids())

    def _build_demo_session(self, table_id: str, table: BilliardTable) -> DemoTableSession:
        robot_prim_path = _ROBOT_ARM_CLASS.get_prim_path(table_id)
        robot_end_effector_prim_path = _ROBOT_ARM_CLASS.get_end_effector_prim_path(table_id)
        articulation_api = ArticulationAPIImpl(robot_prim_path, robot_end_effector_prim_path)
        # 只給 BILLIARD_DEBUG_LOG_PATH 除錯 log 用（見 _debug_log()）——存的是
        # 具體實作類別，不是 ArticulationAPI 抽象介面，因為要呼叫的
        # get_dof_positions_for_debug() 是除錯專用方法，刻意不加進正式 port。
        self._demo_articulation_apis[table_id] = articulation_api

        robot_manager = TableRobotManager(
            table.get_table_center(), table_id, self._stage_api, articulation_api, _ROBOT_ARM_CLASS
        )

        table_ball_set = table.get_table_ball_set()
        if table_ball_set is None:
            raise RuntimeError(f"{table_id} 剛建立卻沒有 TableBallSet，無法建立 DemoTableSession")
        robot_arm = robot_manager.get_robot()
        if robot_arm is None:
            raise RuntimeError(f"{table_id} 剛建立卻沒有 RobotArm，無法建立 DemoTableSession")

        if self._physics_api_debug is not None:
            # 只回報球桿／母球——揮桿卡住時要確認的是「球桿有沒有撞到球檯/
            # 地板」跟「母球到底有沒有被打到」，不需要整支手臂每個連桿都開
            # （那是 scripts/test_elevated_bridge_ur3e_table.py 那種一次性
            # 研究腳本才需要的廣度，GUI 除錯先聚焦在這兩個最關鍵的 prim）。
            self._physics_api_debug.enable_contact_reporting(robot_manager.get_cue_stick_prim_path())
            self._physics_api_debug.enable_contact_reporting(table_ball_set.get_ball_prim_paths()[0])

        pocket_handler = self._build_pocket_event_handler(table, table_ball_set)
        controller = self._build_model_controller(table_ball_set)
        error_state = ErrorState()
        swing_strategy = create_swing_strategy_for(robot_arm, articulation_api)
        runtime = TableRuntime(
            DemoTableObservationBuilder(
                table_ball_set, self._rigid_body_api, table_ball_set.ball_motion_monitor, error_state, table.position_provider, robot_arm
            ),
            DemoTableOrchestrator(
                controller, table_ball_set, table.position_provider, robot_arm, articulation_api, swing_strategy, error_state, self._rolling_resistance_service
            ),
        )
        return DemoTableSession(
            table_id, table, runtime, pocket_handler, self._rigid_body_api, robot_manager, articulation_api
        )

    def _build_model_controller(self, table_ball_set: TableBallSet) -> ModelController:
        policy = self._get_policy()
        return ModelController(policy, table_ball_set.get_table_x_y(), _EVAL_MAX_OFFSET)
    
    def _get_policy(self) -> PolicyPort:
        if self._policy is None:
            self._policy = TorchScriptPolicyImpl(_POLICY_PATH)
        
        return self._policy

    def _disable_demo(self) -> None:
        for session in self._demo_sessions:
            session.destroy()
        self._demo_sessions = []
        self._demo_articulation_apis = {}
        if self._debug_menu:
            self._debug_menu.set_available_tables(self.get_table_ids())

    def _on_training_toggle(self, enable: bool) -> None:
        self._training_enabled = enable
        if enable:
            self._enable_training()
        else:
            self._disable_training()

    def _on_demo_toggle(self, enable: bool) -> None:
        self._demo_enabled = enable
        if enable:
            self._enable_demo()
        else:
            self._disable_demo()

    def _all_sessions(self) -> itertools.chain[TableSession]:
        return itertools.chain(self._training_sessions, self._demo_sessions)

    def get_table_ids(self) -> list[str]:
        return [session.get_table_id() for session in self._all_sessions()]

    def _find_session(self, table_id: str) -> TableSession | None:
        for session in self._all_sessions():
            if session.get_table_id() == table_id:
                return session
        return None

    def get_table_debug_info(self, table_id: str) -> str:
        session = self._find_session(table_id)
        if session is None:
            return ""
        state = session.get_current_state()
        observation = session.get_last_observation()
        if observation is None:
            return f"State: {state.name}\nNo observation yet"
        return (
            f"State: {state.name}\n"
            f"is_ball_moving: {observation.is_ball_moving}\n"
            f"is_motion_complete: {observation.is_motion_complete}\n"
            f"has_error: {observation.has_error}\n"
            f"Cue ball: {_format_vector(observation.cue_ball_position)}"
        )

    def get_ball_velocities_text(self, table_id: str) -> str:
        session = self._find_session(table_id)
        if session is None:
            return ""
        velocities = session.get_ball_velocities()
        lines = [
            f"Ball_{ball_id}: v={_format_vector(linear)} w={_format_vector(angular)}"
            for ball_id, (linear, angular) in sorted(velocities.items())
        ]
        return "\n".join(lines)

    def _event_init(self):
        timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event, name=self._TIMELINE_EVENT_NAME
        )

    def _on_timeline_event(self, event: carb.events.IEvent) -> None:
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._on_play()
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._on_stop()

    def _on_play(self) -> None:
        # Timeline 的 PLAY 事件「Stop 後重新播放」與「Pause 後繼續」都會送出，
        # 只有前者該把狀態機清回 RESET——Pause 續播時場景與手臂都停在原地，
        # 重置狀態機會讓打到一半的擊球憑空中斷。用 _timeline_playing 區分：
        # Stop 會把它設回 False，PAUSE 事件沒有被訂閱、不會動到它。
        resumed_from_pause = self._timeline_playing
        self._timeline_playing = True
        for demo_session in self._demo_sessions:
            if not demo_session.is_articulation_initialized():
                demo_session.initialize_articulation()
        if not resumed_from_pause:
            for session in self._all_sessions():
                session.request_full_reset()

    def _on_stop(self) -> None:
        self._timeline_playing = False

    def _on_tick(self, step_dt, context) -> None:
        for session in self._all_sessions():
            session.tick()
        if self._debug_log_file is not None:
            self._debug_log()

    def _debug_log(self) -> None:
        """僅供 `BILLIARD_DEBUG_LOG_PATH` 除錯用：逐 tick 把每張 Demo 桌的
        狀態機狀態、母球座標、桿尖世界座標/朝向、各關節角度寫成一行，讓
        headful GUI 跑一段固定時間後可以事後讀檔分析，不需要人在旁邊看
        畫面（見模組開頭 `_AUTO_PLAY_DELAY_SEC`/`_DEBUG_LOG_PATH` 說明）。"""
        self._debug_tick_counter += 1
        for session in self._demo_sessions:
            table_id = session.get_table_id()
            state = session.get_current_state()
            observation = session.get_last_observation()
            articulation_api = self._demo_articulation_apis.get(table_id)
            if articulation_api is not None and session.is_articulation_initialized():
                tip_position = articulation_api.get_end_effector_position()
                tip_orientation = articulation_api.get_end_effector_orientation()
                dof_positions = articulation_api.get_dof_positions_for_debug()
            else:
                tip_position = tip_orientation = dof_positions = []
            cue_ball_position = observation.cue_ball_position if observation is not None else []
            is_motion_complete = observation.is_motion_complete if observation is not None else None
            has_error = observation.has_error if observation is not None else None
            self._debug_log_file.write(
                f"tick={self._debug_tick_counter} table={table_id} state={state.name} "
                f"is_motion_complete={is_motion_complete} has_error={has_error} "
                f"cue_ball={cue_ball_position} tip_pos={tip_position} tip_orient={tip_orientation} "
                f"dof_positions={dof_positions}\n"
            )
        self._debug_log_file.flush()

    def _on_debug_contact_event(self, event) -> None:
        """僅供 `BILLIARD_DEBUG_LOG_PATH` 除錯用：球桿/母球的碰撞事件跟逐
        tick 的關節角度 log 用同一個檔案、同一個 tick 計數器，事後可以對照
        「手臂卡住的那個 tick，是不是同時有一筆碰撞事件」，藉此判斷卡住是
        撞到東西還是純粹控制/收斂問題（見 `_boost_wrist_gains_for_cue_stick_
        load()` 那次驗證：wrist_1/wrist_3 修好後肘關節卡在一個乾淨、不再
        收斂的角度，懷疑是撞到地板，但當時沒有碰撞 log 能直接證實）。"""
        if self._debug_log_file is None:
            return
        self._debug_log_file.write(
            f"tick={self._debug_tick_counter} CONTACT a={event.actor_path_a} b={event.actor_path_b} "
            f"collider_a={event.collider_path_a} collider_b={event.collider_path_b} impulse={event.impulse}\n"
        )
        self._debug_log_file.flush()

    def on_shutdown(self):
        self._disable_training()
        self._disable_demo()
        if self._tick_callback_id is not None:
            SimulationManager.deregister_callback(self._tick_callback_id)
            self._tick_callback_id = None
        if self._tool_menu_items:
            unregister(self._tool_menu_items, _TOOL_MENU_NAME)
            self._tool_menu_items = None
        if self._debug_menu:
            self._debug_menu.destroy()
            self._debug_menu = None
        self._sub = None
        self._timeline_sub = None
        self._auto_play_sub = None
        if self._physics_api_debug is not None:
            self._physics_api_debug.unsubscribe_contact_events()
            self._physics_api_debug = None
        if self._debug_log_file is not None:
            self._debug_log_file.close()
            self._debug_log_file = None
