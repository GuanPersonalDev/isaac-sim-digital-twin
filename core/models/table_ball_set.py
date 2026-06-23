from __future__ import annotations

import logging
import os

from .ball_colors import BALL_COLORS
from ..ports.material_api import MaterialAPI
from ..ports.stage_api import StageAPI

logger = logging.getLogger(__name__)

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
_BALL_TEMPLATE_PATH = os.path.join(_ASSET_DIR, "ball_template.usda")
_STRIPE_MDL_PATH = os.path.join(_ASSET_DIR, "materials", "stripe_ball.mdl")
_STRIPE_IDENTIFIER = "stripe_material"
_BALL_PRIM_BASE = "/World/Balls/ball_{}"


class TableBallSet:
    """
    撞球桌與球的設定類別
    """

    def __init__(
        self,
        stage_api: StageAPI,
        material_api: MaterialAPI,
        table_z: float,
        ball_radius: float = 0.028575,
    ) -> None:
        self._stage_api = stage_api
        self._material_api = material_api
        self._table_z = table_z
        self._ball_radius = ball_radius
        self._built = False

    def build(self, positions: dict[int, tuple[float, float]]) -> None:
        """
        建立撞球桌，positions 的 key 須包含 0–9，value 為相對桌台的 (x, y) 座標（單位 m）。
        """
        if set(positions.keys()) != set(range(10)):
            raise ValueError(
                f"positions 需包含 key 0-9，實際收到：{sorted(positions.keys())}"
            )

        z = self._table_z + self._ball_radius

        for ball_id in range(10):
            prim_path = _BALL_PRIM_BASE.format(ball_id)
            self._stage_api.create_reference_prim(prim_path, _BALL_TEMPLATE_PATH)
            x, y = positions[ball_id]
            self._stage_api.set_prim_translate(prim_path, x, y, z)

            if ball_id == 9:
                self._apply_ball9_material(prim_path)
            else:
                r, g, b = BALL_COLORS[ball_id]
                self._material_api.apply_preview_surface(prim_path, r, g, b)

        self._built = True

    def _apply_ball9_material(self, prim_path: str) -> None:
        if not os.path.exists(_STRIPE_MDL_PATH):
            logger.warning("stripe_ball.mdl 不存在（%s），fallback 為純黃色", _STRIPE_MDL_PATH)
            r, g, b = BALL_COLORS[9]
            self._material_api.apply_preview_surface(prim_path, r, g, b)
            return
        try:
            self._material_api.apply_mdl_shader(
                prim_path=prim_path,
                mdl_path=_STRIPE_MDL_PATH,
                identifier=_STRIPE_IDENTIFIER,
                stripe_color=(BALL_COLORS[9][0], BALL_COLORS[9][1], BALL_COLORS[9][2]),
                base_color=(1.0, 1.0, 1.0),
            )
        except Exception as e:
            logger.warning("MDL Shader 設定失敗（%s），fallback 為純黃色", e)
            r, g, b = BALL_COLORS[9]
            self._material_api.apply_preview_surface(prim_path, r, g, b)

    def hide_ball(self, ball_id: int) -> None:
        """
        關閉球的顯示
        """
        self._check_built()
        self._check_ball_id(ball_id)
        self._stage_api.set_visibility(_BALL_PRIM_BASE.format(ball_id), visible=False)

    def show_ball(self, ball_id: int) -> None:
        """
        顯示球
        """
        self._check_built()
        self._check_ball_id(ball_id)
        self._stage_api.set_visibility(_BALL_PRIM_BASE.format(ball_id), visible=True)

    def reset(self, positions: dict[int, tuple[float, float]]) -> None:
        """
        將全部球設為可見，並移回 positions 指定的座標。
        """
        self._check_built()
        if set(positions.keys()) != set(range(10)):
            raise ValueError(
                f"positions 須包含 key 0-9，實際收到：{sorted(positions.keys())}"
            )

        z = self._table_z + self._ball_radius
        for ball_id in range(10):
            prim_path = _BALL_PRIM_BASE.format(ball_id)
            self._stage_api.set_visibility(prim_path, visible=True)
            x, y = positions[ball_id]
            self._stage_api.set_prim_translate(prim_path, x, y, z)

    def _check_built(self) -> None:
        if not self._built:
            raise RuntimeError("請先呼叫 build() 再操作球的顯示狀態")

    def _check_ball_id(self, ball_id: int) -> None:
        if ball_id not in range(10):
            raise ValueError(f"ball_id 必須在 0–9 之間，收到：{ball_id}")
