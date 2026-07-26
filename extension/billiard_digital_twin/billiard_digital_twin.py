import itertools
import sys
import os
import omni.ext
import omni.usd
import omni.timeline
import carb.events
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for p in [_EXT_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.controllers.script_controller import ScriptController
from core.models.table_ball_set import TableBallSet
from core.models.robot_arm import RobotArm
from core.models.barrett_wam_robot import BarrettWamRobot
from core.ports import RigidBodyAPI
from core.services.asset_utility import TABLE_PATH
from core.services.observation_builder import DemoTableObservationBuilder, TrainingTableObservationBuilder
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
_ROBOT_ARM_CLASS: type[RobotArm] = BarrettWamRobot
_TABLE_SIZE_PROBE_PATH = "/World/_TableSizeProbe"

class BilliardExtension(omni.ext.IExt):
    _TIMELINE_EVENT_NAME = "billiard_digital_twin_timeline_wait"
    _PHYSIC_CALL_BACK = "billiard_table_tick"
    def on_startup(self, ext_id: str):
        self._debug_menu = None
        self._training_sessions: list[TableSession] = []
        self._demo_sessions: list[DemoTableSession] = []
        self._training_enabled = True
        self._demo_enabled = True
        self._timeline_playing = False
        self._table_unit_side_length = 0.0
        self._tick_callback_id = None
        scripts_dir = os.path.join(_PROJECT_ROOT, "scripts")
        self._tool_menu_items = discover_and_register(scripts_dir, _TOOL_MENU_NAME)
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            self._billiard_init()
        else:
            stream = omni.usd.get_context().get_stage_event_stream()
            self._sub = stream.create_subscription_to_pop(
                self._on_stage_event, name="billiard_digital_twin_stage_wait"
            )

    def _on_stage_event(self, event: carb.events.IEvent) -> None:
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._billiard_init()
            self._sub = None

    def _billiard_init(self):
        SimulationManager.setup_simulation(dt=1/60)

        self._asset_env_init()

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
        controller = ScriptController()
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

        robot_manager = TableRobotManager(
            table.get_table_center(), table_id, self._stage_api, articulation_api, _ROBOT_ARM_CLASS
        )

        table_ball_set = table.get_table_ball_set()
        if table_ball_set is None:
            raise RuntimeError(f"{table_id} 剛建立卻沒有 TableBallSet，無法建立 DemoTableSession")
        robot_arm = robot_manager.get_robot()
        if robot_arm is None:
            raise RuntimeError(f"{table_id} 剛建立卻沒有 RobotArm，無法建立 DemoTableSession")

        pocket_handler = self._build_pocket_event_handler(table, table_ball_set)
        controller = ScriptController()
        error_state = ErrorState()
        runtime = TableRuntime(
            DemoTableObservationBuilder(
                table_ball_set, self._rigid_body_api, table_ball_set.ball_motion_monitor, error_state, table.position_provider, robot_arm
            ),
            DemoTableOrchestrator(
                controller, table_ball_set, table.position_provider, robot_arm, articulation_api, error_state, self._rolling_resistance_service
            ),
        )
        return DemoTableSession(
            table_id, table, runtime, pocket_handler, self._rigid_body_api, robot_manager, articulation_api
        )

    def _disable_demo(self) -> None:
        for session in self._demo_sessions:
            session.destroy()
        self._demo_sessions = []
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
            return f"狀態: {state.name}\n尚未有 Observation"
        return (
            f"狀態: {state.name}\n"
            f"is_ball_moving: {observation.is_ball_moving}\n"
            f"is_motion_complete: {observation.is_motion_complete}\n"
            f"has_error: {observation.has_error}\n"
            f"母球座標: {observation.cue_ball_position}"
        )

    def get_ball_velocities_text(self, table_id: str) -> str:
        session = self._find_session(table_id)
        if session is None:
            return ""
        velocities = session.get_ball_velocities()
        lines = [
            f"Ball_{ball_id}: v={linear} w={angular}"
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
        self._timeline_playing = True
        for demo_session in self._demo_sessions:
            if not demo_session.is_articulation_initialized():
                demo_session.initialize_articulation()

    def _on_stop(self) -> None:
        self._timeline_playing = False

    def _on_tick(self, step_dt, context) -> None:
        for session in self._all_sessions():
            session.tick()

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
