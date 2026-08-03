import math

import pytest

from core.services.position_offset_limiter import clamp_position_offset


@pytest.fixture(params=[math.nan, math.inf, -math.inf])
def non_finite_value(request) -> float:
    return request.param


@pytest.fixture(params=[True, "invalid", None])
def non_numeric_value(request):
    return request.param


class TestClampWithinLimit:
    def test_returns_vector_inside_circle_unchanged(self):
        assert clamp_position_offset([-0.2, 0.1], 0.5) == [-0.2, 0.1]

    def test_returns_vector_on_boundary_unchanged(self):
        # norm 恰等於 limit 時不得進入縮放分支，否則等比運算的浮點殘差
        # 會讓嚴格相等失敗。
        assert clamp_position_offset([0.7, 0.7], math.hypot(0.7, 0.7)) == [
            0.7,
            0.7,
        ]

    def test_full_capability_does_not_clamp_physical_maximum(self):
        # max_offset = 1.0 代表可用滿物理域 ±0.5R，正規化後角落值為
        # (1.0, 1.0)、norm ≈ 1.414，此時仍應被裁到 1.0。
        result = clamp_position_offset([1.0, 1.0], 1.0)

        assert math.hypot(*result) == pytest.approx(1.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        assert clamp_position_offset([0.0, 0.0], 0.0) == [0.0, 0.0]


class TestClampOutsideLimit:
    def test_scales_norm_down_to_limit(self):
        result = clamp_position_offset([0.8, -0.6], 0.5)

        assert math.hypot(*result) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "position_offset",
        [
            [-0.9, -0.9],
            [0.8, -0.6],
            [-0.9, 0.2],
            [0.9, 0.9],
        ],
    )
    def test_preserves_direction_angle(
        self,
        position_offset: list[float],
    ):
        result = clamp_position_offset(position_offset, 0.5)

        assert math.atan2(result[1], result[0]) == pytest.approx(
            math.atan2(position_offset[1], position_offset[0])
        )

    @pytest.mark.parametrize(
        "position_offset",
        [
            [-0.9, -0.9],
            [0.8, -0.6],
            [-0.9, 0.2],
            [0.9, 0.9],
        ],
    )
    def test_preserves_sign_of_each_axis(
        self,
        position_offset: list[float],
    ):
        result = clamp_position_offset(position_offset, 0.5)

        assert math.copysign(1.0, result[0]) == math.copysign(
            1.0, position_offset[0]
        )
        assert math.copysign(1.0, result[1]) == math.copysign(
            1.0, position_offset[1]
        )

    def test_does_not_clamp_axes_independently(self):
        # 分軸截斷會把 (0.9, 0.2) 變成 (0.5, 0.2)，方向角從 0.219 rad
        # 歪到 0.381 rad。等比縮放必須維持兩軸比例。
        result = clamp_position_offset([0.9, 0.2], 0.5)

        assert result[0] / result[1] == pytest.approx(0.9 / 0.2)

    def test_zero_max_offset_returns_center_strike(self):
        assert clamp_position_offset([0.9, -0.9], 0.0) == [0.0, 0.0]


class TestInvalidInput:
    @pytest.mark.parametrize(
        "position_offset",
        [
            [],
            [0.1],
            [0.1, 0.2, 0.3],
        ],
    )
    def test_rejects_offset_without_exactly_two_values(
        self,
        position_offset: list[float],
    ):
        with pytest.raises(ValueError):
            clamp_position_offset(position_offset, 0.5)

    def test_rejects_non_finite_offset(self, non_finite_value: float):
        with pytest.raises(ValueError):
            clamp_position_offset([non_finite_value, 0.1], 0.5)

        with pytest.raises(ValueError):
            clamp_position_offset([0.1, non_finite_value], 0.5)

    def test_rejects_non_numeric_offset(self, non_numeric_value):
        with pytest.raises(ValueError):
            clamp_position_offset([non_numeric_value, 0.1], 0.5)

        with pytest.raises(ValueError):
            clamp_position_offset([0.1, non_numeric_value], 0.5)

    @pytest.mark.parametrize(
        "max_offset",
        [-0.1, 1.1, -1.0, 2.0],
    )
    def test_rejects_max_offset_out_of_range(self, max_offset: float):
        with pytest.raises(ValueError):
            clamp_position_offset([0.1, 0.1], max_offset)

    def test_rejects_non_finite_max_offset(self, non_finite_value: float):
        with pytest.raises(ValueError):
            clamp_position_offset([0.1, 0.1], non_finite_value)

    def test_rejects_non_numeric_max_offset(self, non_numeric_value):
        with pytest.raises(ValueError):
            clamp_position_offset([0.1, 0.1], non_numeric_value)
