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
from core.services.break_shot_position_provider import BreakShotPositionProvider

_TABLE_COUNT = 1


class BilliardExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._debug_menu = None
        self._tables = []
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

        self._build_tables(_TABLE_COUNT, self._stage_api, material_api)
        self._debug_menu = DebugMenu()

    def _build_tables(self, total: int, stage_api: StageAPI, material_api: MaterialAPI):
        # 計算單邊撞球桌的個數
        side_count = 1
        while total > side_count * side_count:
            side_count += 1
        print(f"side count : {side_count}")

        index = 0
        for i in range(side_count):
            for j in range(side_count):
                x_pos = self._table_unit_side_length * i
                y_pos = self._table_unit_side_length * j
                table = BilliardTable(
                    f"/World/Table_{index}", stage_api, material_api, (x_pos, y_pos)
                )
                print(f"create table {index}")
                if self._table_unit_side_length == 0:
                    self._table_unit_side_length = self._get_table_side_length(
                        table.get_table_prim_path()
                    )
                    print(f"get length : {self._table_unit_side_length}")
                self._tables.append(table)
                index += 1

    def _get_table_side_length(self, prim_path):
        x_length, y_length, z_length = self._stage_api.get_prim_sides(prim_path)
        return max(x_length, y_length, z_length)

    def on_shutdown(self):
        if self._debug_menu:
            self._debug_menu.destroy()
            self._debug_menu = None
        for t in self._tables:
            t.destroy()
        self._tables = None
        self._sub = None
