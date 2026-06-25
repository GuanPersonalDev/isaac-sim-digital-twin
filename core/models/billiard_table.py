from .table_ball_set import TableBallSet
from ..ports.stage_api import StageAPI
from ..ports.material_api import MaterialAPI
from ..services.ball_position_provider import BallPositionProvider
from ..services.break_shot_position_provider import BreakShotPositionProvider
from ..services.asset_utility import TABLE_PATH


class BilliardTable:
    """
    單一撞球桌的管理介面
    """

    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        material_api: MaterialAPI,
        position: tuple[float, float],
    ):
        self._base_path = base_path

        self._table_prim_path = self._base_path + f"/Table"
        stage_api.create_reference_prim(self._table_prim_path, TABLE_PATH)
        x_pos, y_pos = position
        stage_api.set_prim_translate(self._table_prim_path, x_pos, y_pos, 0)

        self._table_set = TableBallSet(
            stage_api, material_api, table_z=0, base_path=base_path
        )

        positions = BreakShotPositionProvider().get_positions()
        world_positions = {
            ball_id: (x + x_pos, y + y_pos) for ball_id, (x, y) in positions.items()
        }
        self._table_set.build(world_positions)

    def get_table_prim_path(self):
        return self._table_prim_path

    def destroy(self):
        self._table_set = None
