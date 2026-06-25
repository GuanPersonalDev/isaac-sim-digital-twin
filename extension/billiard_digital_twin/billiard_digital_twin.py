import sys
import os
import omni.ext

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for p in [_EXT_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
from ui.debug_menu import DebugMenu
from core.models.table_ball_set import TableBallSet
from core.services.break_shot_position_provider import BreakShotPositionProvider


class BilliardExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        stage_api = StageAPIImpl()
        material_api = MaterialAPIImpl()
        stage_api.create_reference_prim(
            "/World/Environment",
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Grid/gridroom_black.usd",
        )

        # self._ball_set = TableBallSet(
        #     stage_api=stage_api, material_api=material_api, table_z=0.0
        # )
        # positions = BreakShotPositionProvider().get_positions()
        # self._ball_set.build(positions)
        #
        self._debug_menu = DebugMenu()

    def on_shutdown(self):
        if self._debug_menu:
            self._debug_menu.destroy()
            self._debug_menu = None
        # self._ball_set = None
