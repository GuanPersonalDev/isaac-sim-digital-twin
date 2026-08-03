import math

import pytest

from core.services.numeric_validation import (
    validate_finite_number,
    validate_max_offset,
)


class TestValidateFiniteNumber:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (1, 1.0),
            (0, 0.0),
            (-2.5, -2.5),
        ],
    )
    def test_converts_real_number_to_float(
        self,
        value,
        expected: float,
    ):
        result = validate_finite_number(value, "field")

        assert isinstance(result, float)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_value(self, value: float):
        with pytest.raises(ValueError):
            validate_finite_number(value, "field")

    @pytest.mark.parametrize(
        "value",
        [True, False, "1.0", None, [1.0]],
    )
    def test_rejects_non_real_value(self, value):
        # bool 是 int 的子類別，必須明確排除，否則 True 會被靜默當成 1.0。
        with pytest.raises(ValueError):
            validate_finite_number(value, "field")

    def test_error_message_contains_field_name(self):
        with pytest.raises(ValueError, match="cue_ball_speed"):
            validate_finite_number("invalid", "cue_ball_speed")


class TestValidateMaxOffset:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 0, 1])
    def test_accepts_value_within_range(self, value):
        assert validate_max_offset(value) == pytest.approx(float(value))

    @pytest.mark.parametrize("value", [-0.1, 1.1, -1.0, 2.0])
    def test_rejects_value_out_of_range(self, value: float):
        with pytest.raises(ValueError):
            validate_max_offset(value)

    @pytest.mark.parametrize(
        "value",
        [math.nan, math.inf, -math.inf, True, "0.5", None],
    )
    def test_rejects_non_finite_or_non_real_value(self, value):
        with pytest.raises(ValueError):
            validate_max_offset(value)

    def test_error_message_names_max_offset(self):
        with pytest.raises(ValueError, match="max_offset"):
            validate_max_offset(1.5)
