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

        self._table_prim_path = self._base_path + "/Table"
        stage_api.create_reference_prim(self._table_prim_path, TABLE_PATH)
        self._x_pos, self._y_pos = position
        self._z_pos = 0
        stage_api.set_prim_translate(
            self._table_prim_path, self._x_pos, self._y_pos, self._z_pos
        )

        self._table_set = TableBallSet(
            stage_api, material_api, table_z=self._z_pos, base_path=base_path
        )

        positions = BreakShotPositionProvider().get_positions()
        world_positions = {
            ball_id: (x + self._x_pos, y + self._y_pos)
            for ball_id, (x, y) in positions.items()
        }
        self._table_set.build(world_positions)

    def get_table_prim_path(self):
        return self._table_prim_path

    def get_table_center(self) -> tuple[float, float, float]:
        return (self._x_pos, self._y_pos, self._z_pos)

    def destroy(self):
        self._table_set = None
