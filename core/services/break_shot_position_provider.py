from .ball_position_provider import BallPositionProvider

_D = 0.05715  # ball diameter 57.15 mm
_ROW = _D * (3**0.5) / 2  # y distance = d * sin(60 degree)
_COL = _D / 2  # x offset unit
_FOOT = 0.635  # 1/4 table length = 2.54/4

BREAK_SHOT_POSITIONS: dict = {
    "cue": (0.0, -0.9525),
    1: (0.0, _FOOT),
    2: (-_COL, _FOOT + _ROW),
    3: (_COL, _FOOT + _ROW),
    4: (-_D, _FOOT + 2 * _ROW),
    9: (0.0, _FOOT + 2 * _ROW),
    5: (_D, _FOOT + 2 * _ROW),
    6: (-_COL, _FOOT + 3 * _ROW),
    7: (_COL, _FOOT + 3 * _ROW),
    8: (0.0, _FOOT + 4 * _ROW),
}


class BreakShotPositionProvider(BallPositionProvider):
    def get_positions(self) -> dict:
        return dict(BREAK_SHOT_POSITIONS)
