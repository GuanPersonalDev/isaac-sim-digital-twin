import sys
import os
from core.models.ur5_robot import UR5Robot
from core.ports import RigidBodyAPI
import omni.ext
import omni.usd
import carb.events
from isaacsim.core.api.world import World

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for p in [_EXT_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

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

_TABLE_COUNT = 1
_TOOL_MENU_NAME = "Tools"


class BilliardExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._debug_menu = None
        self._tables = []
        self._demo_table = None
        self._robot = None
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
        if World.instance() is None:
            world = World(physics_dt = 1/60, rendering_dt = 1/60, stage_units_in_meter = 1.0)
        else:
            world = World.instance()
        world.reset()

        self._stage_api = StageAPIImpl()
        material_api = MaterialAPIImpl()
        rigid_body_api = RigidBodyAPIImpl()
        self._table_unit_side_length = 0
        self._tables: list[BilliardTable] = []

        demo_table_path = "/World/Table_Demo"
        self._demo_table = self._build_table(
            demo_table_path, self._stage_api, material_api, rigid_body_api, (0, 0)
        )
        robot_prim_path = UR5Robot.get_prim_path(demo_table_path)
        robot_end_effector_prim_path = UR5Robot.get_end_effector_prim_path(demo_table_path)
        self._articulation_api = ArticulationAPIImpl(world, robot_prim_path, robot_end_effector_prim_path)

        self._table_unit_side_length = self._get_table_side_length(
            self._demo_table.get_table_prim_path()
        )

        demo_table_center = self._demo_table.get_table_center()
        self._robot = TableRobotManager(
            demo_table_center, demo_table_path, self._stage_api, self._articulation_api
        )

        self._build_tables(_TABLE_COUNT, self._stage_api, material_api, rigid_body_api)

        self._training_enabled = False
        self._demo_enabled = False
        self._debug_menu = DebugMenu(self._on_training_toggle, self._on_demo_toggle)

    def _build_tables(self, total: int, stage_api: StageAPI, material_api: MaterialAPI, rigid_body_api: RigidBodyAPI):
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
                self._tables.append(table)
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

    def on_shutdown(self):
        if self._articulation_api is not None:
            self._articulation_api.shutdown()
        world = World.instance()
        if world is not None:
            world.clear_instance()
        if self._tool_menu_items:
            unregister(self._tool_menu_items, _TOOL_MENU_NAME)
            self._tool_menu_items = None
        if self._debug_menu:
            self._debug_menu.destroy()
            self._debug_menu = None
        for t in self._tables:
            t.destroy()
        self._tables = None
        if self._demo_table:
            self._demo_table.destroy()
            self._demo_table = None
        if self._robot:
            self._robot.destroy()
            self._robot = None
        self._sub = None
