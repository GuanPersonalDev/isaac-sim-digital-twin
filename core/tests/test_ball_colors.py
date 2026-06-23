def _ball_colors() -> dict[int, list[float]]:
    from core.models.ball_colors import BALL_COLORS

    return BALL_COLORS


def test_ball_colors_keys():
    assert set(_ball_colors().keys()) == set(range(10))


def test_ball_colors_rgb_range():
    for rgb in _ball_colors().values():
        assert len(rgb) == 3
        assert all(0.0 <= value <= 1.0 for value in rgb)
