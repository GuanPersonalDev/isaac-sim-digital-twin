import sys
import os
import omni.ext

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
        stage_api = StageAPIImpl()
        material_api = MaterialAPIImpl()
        # stage_api.create_reference_prim(
        #     "/World/Environment",
        #     "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Grid/gridroom_black.usd",
        # )
        self._table_unit_side_length = 0
        self._tables: list[BilliardTable] = []

        self._build_tables(_TABLE_COUNT, stage_api, material_api)
        self._debug_menu = DebugMenu()

    def _build_tables(self, total: int, stage_api: StageAPI, material_api: MaterialAPI):
        # 計算單邊撞球桌的個數
        side_count = 1
        while total > side_count * side_count:
            side_count += 1

        index = 0
        for i in range(side_count):
            for j in range(side_count):
                x_pos = self._table_unit_side_length * i
                y_pos = self._table_unit_side_length * j
                table = BilliardTable(
                    f"/World/Table_{index}", stage_api, material_api, (x_pos, y_pos)
                )
                if self._table_unit_side_length == 0:
                    self._table_unit_side_length = self._get_table_side_length(
                        table.get_table_prim_path()
                    )
                self._tables.append(table)

    def _get_table_side_length(self, prim_path):
        # TODO: get side length from usd
        return 10

    def on_shutdown(self):
        if self._debug_menu:
            self._debug_menu.destroy()
            self._debug_menu = None
        for t in self._tables:
            t.destroy()
        self._tables = None
