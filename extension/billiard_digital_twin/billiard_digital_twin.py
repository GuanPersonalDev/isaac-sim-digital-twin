from enum import Enum
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
from core.services.observation_builder import DemoTableObservationBuilder, TrainingTableObservationBuilder
from core.services.table_orchestrator import DemoTableOrchestrator, TrainingTableOrchestrator
from core.services.table_runtime import TableRuntime
from core.ports.material_api import MaterialAPI
from core.ports.stage_api import StageAPI
from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
from ui.debug_menu import DebugMenu
from ui.tool_menu_registry import discover_and_register, unregister
from core.models.billiard_table import BilliardTable
from core.models.table_robot_manager import TableRobotManager
from core.services.error_state import ErrorState
from core.services.impulse_striking_service import ImpulseStrikingService
from core.services.rolling_resistance_service import RollingResistanceService

_TABLE_COUNT = 1
_TOOL_MENU_NAME = "Tools"
# Demo 桌實際掛載的手臂類別，換手臂只需要改這一行（見 core/models/robot_arm.py）。
_ROBOT_ARM_CLASS: type[RobotArm] = BarrettWamRobot

class RuntimeState(Enum):
    NOT_READY = 1
    READY = 2
    RUNNING = 3

class BilliardExtension(omni.ext.IExt):
    _TIMELINE_EVENT_NAME = "billiard_digital_twin_timeline_wait"
    _PHYSIC_CALL_BACK = "billiard_table_tick"
    def on_startup(self, ext_id: str):
        self._debug_menu = None
        self._training_tables = []
        self._demo_table = None
        self._robot = None
        self._table_runtimes: list[TableRuntime] = []
        self._runtime_state = RuntimeState.NOT_READY
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
        
        self._training_enabled = False
        self._demo_enabled = False
        self._debug_menu = DebugMenu(self._on_training_toggle, self._on_demo_toggle)
        
        self._event_init()
        
    def _asset_env_init(self):
        self._stage_api = StageAPIImpl()
        material_api = MaterialAPIImpl()
        self._rigid_body_api = RigidBodyAPIImpl()
        self._table_unit_side_length = 0
        self._training_tables: list[BilliardTable] = []

        demo_table_path = "/World/Table_Demo"
        self._demo_table = self._build_table(
            demo_table_path, self._stage_api, material_api, self._rigid_body_api, (0, 0)
        )
        self._rolling_resistance_service = RollingResistanceService(
            self._rigid_body_api, self._demo_table.get_table_ball_set().get_ball_radius()
        )
        robot_prim_path = _ROBOT_ARM_CLASS.get_prim_path(demo_table_path)
        robot_end_effector_prim_path = _ROBOT_ARM_CLASS.get_end_effector_prim_path(demo_table_path)
        self._articulation_api = ArticulationAPIImpl(robot_prim_path, robot_end_effector_prim_path)
       
        self._table_unit_side_length = self._get_table_side_length(
            self._demo_table.get_table_prim_path()
        )

        demo_table_center = self._demo_table.get_table_center()
        self._robot = TableRobotManager(
            demo_table_center, demo_table_path, self._stage_api, self._articulation_api, _ROBOT_ARM_CLASS
        )

        self._build_training_tables(_TABLE_COUNT, self._stage_api, material_api, self._rigid_body_api)
       

    def _build_training_tables(self, total: int, stage_api: StageAPI, material_api: MaterialAPI, rigid_body_api: RigidBodyAPI):
        # 計算單邊撞球桌的個數
        side_count = 1
        while total > side_count * side_count:
            side_count += 1

        index = 0
        for i in range(side_count):
            for j in range(side_count):
                x_pos = self._table_unit_side_length * (i + 1)
                y_pos = self._table_unit_side_length * (j + 1)
                table = self._build_table(
                    f"/World/Table_{index}", stage_api, material_api, rigid_body_api, (x_pos, y_pos)
                )
                self._training_tables.append(table)
                index += 1

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

    def _get_table_side_length(self, prim_path):
        x_length, y_length, z_length = self._stage_api.get_prim_sides(prim_path)
        return max(x_length, y_length, z_length)

    def _on_training_toggle(self, enable: bool):
        self._training_enabled = enable

    def _on_demo_toggle(self, enable: bool):
        self._demo_enabled = enable

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
        if self._runtime_state == RuntimeState.NOT_READY:
            self._articulation_api.initialize()
            if self._demo_table and self._robot:
                robot = self._robot.get_robot()
                if robot:
                    self._register_demo_table_runtime(self._demo_table, robot)
                
            for training_table in self._training_tables:
                self._register_training_table_runtime(training_table)
           
            self._tick_callback_id = SimulationManager.register_callback(
                self._on_tick, event=SimulationEvent.PHYSICS_POST_STEP
               )

            self._runtime_state = RuntimeState.RUNNING
        elif self._runtime_state == RuntimeState.READY:
            self._articulation_api.initialize()
            self._runtime_state = RuntimeState.RUNNING

    def _register_demo_table_runtime(self, table: BilliardTable, robot_arm: RobotArm) -> None:
        table_ball_set = table.get_table_ball_set()
        if table_ball_set:
            controller = ScriptController()
            error_state = ErrorState()
            demo_table_runtime = TableRuntime(
                DemoTableObservationBuilder(table_ball_set, self._rigid_body_api, table_ball_set.ball_motion_monitor, error_state, table.position_provider, robot_arm),
                DemoTableOrchestrator(controller, table_ball_set, table.position_provider, robot_arm, self._articulation_api, error_state, self._rolling_resistance_service))
            self._table_runtimes.append(demo_table_runtime)
            
    def _register_training_table_runtime(self, table: BilliardTable) -> None:
        table_ball_set = table.get_table_ball_set()
        if table_ball_set: 
            controller = ScriptController()
            error_state = ErrorState()
 
            impulse_striking_service = ImpulseStrikingService(self._rigid_body_api, table_ball_set.get_ball_prim_paths()[0], table_ball_set.get_ball_radius())

            training_table_runtime = TableRuntime(
                TrainingTableObservationBuilder(table_ball_set, self._rigid_body_api, table_ball_set.ball_motion_monitor, error_state, table.position_provider), 
                TrainingTableOrchestrator(controller, table_ball_set, table.position_provider, impulse_striking_service, error_state, self._rolling_resistance_service))
            self._table_runtimes.append(training_table_runtime)
    
    def _on_tick(self, step_dt, context) -> None:
        if self._runtime_state != RuntimeState.RUNNING:
            return
        for runtime in self._table_runtimes:
            runtime.tick()
    
    def _on_stop(self) -> None:
        self._runtime_state = RuntimeState.READY
 
    def on_shutdown(self):
        if self._articulation_api is not None:
            self._articulation_api.shutdown()
        if self._runtime_state != RuntimeState.NOT_READY:
            SimulationManager.deregister_callback(self._tick_callback_id)
        if self._tool_menu_items:
            unregister(self._tool_menu_items, _TOOL_MENU_NAME)
            self._tool_menu_items = None
        if self._debug_menu:
            self._debug_menu.destroy()
            self._debug_menu = None
        for t in self._training_tables:
            t.destroy()
        self._training_tables = None
        if self._demo_table:
            self._demo_table.destroy()
            self._demo_table = None
        if self._robot:
            self._robot.destroy()
            self._robot = None
        self._sub = None
        self._timeline_sub = None
