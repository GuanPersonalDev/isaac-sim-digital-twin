import sys
import os
import omni.ext
import omni.usd
import carb.events

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for p in [_EXT_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.ports.material_api import MaterialAPI
from core.ports.stage_api import StageAPI
from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
from ui.debug_menu import DebugMenu
from core.models.billiard_table import BilliardTable
from core.models.table_robot_manager import TableRobotManager

_TABLE_COUNT = 1


class BilliardExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._debug_menu = None
        self._tables = []
        self._demo_table = None
        self._robot = None
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
        self._stage_api = StageAPIImpl()
        material_api = MaterialAPIImpl()
        self._table_unit_side_length = 0
        self._tables: list[BilliardTable] = []

        demo_table_path = "/World/Table_Demo"
        self._demo_table = self._build_table(
            demo_table_path, self._stage_api, material_api, (0, 0)
        )
        self._table_unit_side_length = self._get_table_side_length(
            self._demo_table.get_table_prim_path()
        )

        demo_table_center = self._demo_table.get_table_center()
        self._robot = TableRobotManager(
            demo_table_center, demo_table_path, self._stage_api
        )

        self._build_tables(_TABLE_COUNT, self._stage_api, material_api)

        self._training_enabled = False
        self._demo_enabled = False
        self._debug_menu = DebugMenu()

    def _build_tables(self, total: int, stage_api: StageAPI, material_api: MaterialAPI):
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
                    f"/World/Table_{index}", stage_api, material_api, (x_pos, y_pos)
                )
                self._tables.append(table)
                index += 1

    def _build_table(
        self,
        table_name: str,
        stage_api: StageAPI,
        material_api: MaterialAPI,
        pos: tuple[float, float],
    ) -> BilliardTable:
        table = BilliardTable(table_name, stage_api, material_api, pos)
        return table

    def _get_table_side_length(self, prim_path):
        x_length, y_length, z_length = self._stage_api.get_prim_sides(prim_path)
        return max(x_length, y_length, z_length)

    def _on_training_toggle(self, enable: bool):
        self._training_enabled = enable

    def _on_demo_toggle(self, enable: bool):
        self._demo_enabled = enable

    def on_shutdown(self):
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
