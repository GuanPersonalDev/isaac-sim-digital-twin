import dataclasses
import math

import pytest

from core.models.action import Action
from core.models.action_bounds import (
    CUE_BALL_SPEED,
    POSITION_OFFSET_HORIZONTAL,
    POSITION_OFFSET_VERTICAL,
)
from core.models.manual_shot_bounds import (
    CUE_BALL_PLACEMENT_X,
    CUE_BALL_PLACEMENT_Y,
    MANUAL_SHOT_ANGLE,
)
from core.models.manual_shot_parameters import ManualShotParameters
from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS


def _valid_kwargs(**overrides) -> dict:
    fields = {
        "cue_ball_placement": (0.0, CUE_BALL_PLACEMENT_Y[0]),
        "shot_angle": 0.0,
        "cue_ball_speed": CUE_BALL_SPEED[1],
        "position_offset": (0.0, 0.0),
    }
    fields.update(overrides)
    return fields


class TestDefault:
    def test_matches_break_shot_position_zero(self):
        # 「切到手動什麼都不調就擊球」要跟現行 ScriptController 打出同一球，
        # 開球點取自 BREAK_SHOT_POSITIONS[0]，不得硬編碼。
        # Act
        parameters = ManualShotParameters.default()

        # Assert
        assert parameters.cue_ball_placement == BREAK_SHOT_POSITIONS[0]

    def test_matches_max_cue_ball_speed(self):
        # Act
        parameters = ManualShotParameters.default()

        # Assert
        assert parameters.cue_ball_speed == CUE_BALL_SPEED[1]

    def test_angle_is_zero_and_offset_is_zero(self):
        # Act
        parameters = ManualShotParameters.default()

        # Assert
        assert parameters.shot_angle == 0.0
        assert parameters.position_offset == (0.0, 0.0)


class TestToAction:
    def test_returns_a_brand_new_action_instance_every_call(self):
        # Action 是 mutable dataclass，下游會改它（見
        # ModelController._aiming_state_action_result 的說明）；共用同一個
        # instance 會讓兩次分派互相污染。
        # Arrange
        parameters = ManualShotParameters.default()

        # Act
        first = parameters.to_action(should_execute_action=True)
        second = parameters.to_action(should_execute_action=True)

        # Assert
        assert first is not second
        assert first.cue_ball_placement is not second.cue_ball_placement
        assert first.position_offset is not second.position_offset

    def test_maps_every_field_onto_the_action(self):
        # Arrange
        parameters = ManualShotParameters(
            cue_ball_placement=(0.1, -0.9),
            shot_angle=42.0,
            cue_ball_speed=1.5,
            position_offset=(0.2, -0.3),
        )

        # Act
        action = parameters.to_action(should_execute_action=True)

        # Assert
        assert isinstance(action, Action)
        assert action.cue_ball_placement == [0.1, -0.9]
        assert action.shot_angle == 42.0
        assert action.cue_ball_speed == 1.5
        assert action.position_offset == [0.2, -0.3]

    @pytest.mark.parametrize("should_execute_action", [True, False])
    def test_should_execute_action_flows_through_unchanged(
        self, should_execute_action: bool
    ):
        # should_execute_action 是每次狀態轉換才決定的執行期旗標，不是參數
        # 容器本身的欄位——這裡確認呼叫端傳什麼就原封不動出現在 Action 上。
        # Arrange
        parameters = ManualShotParameters.default()

        # Act
        action = parameters.to_action(should_execute_action=should_execute_action)

        # Assert
        assert action.should_execute_action is should_execute_action


class TestFrozen:
    def test_assigning_a_field_raises_frozen_instance_error(self):
        # frozen=True 是刻意的：UI 端「整包做好再一次賦值」、physics 端
        # 「一次讀出整包」，不存在讀到「角度已更新但速度還沒」的中間態。
        # Arrange
        parameters = ManualShotParameters.default()

        # Act / Assert
        with pytest.raises(dataclasses.FrozenInstanceError):
            parameters.shot_angle = 10.0


class TestBoundaryValuesAreAccepted:
    def test_placement_at_the_exact_bounds_does_not_raise(self):
        # Act / Assert（不拋例外即通過）
        ManualShotParameters(
            **_valid_kwargs(
                cue_ball_placement=(CUE_BALL_PLACEMENT_X[0], CUE_BALL_PLACEMENT_Y[0])
            )
        )
        ManualShotParameters(
            **_valid_kwargs(
                cue_ball_placement=(CUE_BALL_PLACEMENT_X[1], CUE_BALL_PLACEMENT_Y[1])
            )
        )

    def test_shot_angle_at_the_exact_bounds_does_not_raise(self):
        # Act / Assert
        ManualShotParameters(**_valid_kwargs(shot_angle=MANUAL_SHOT_ANGLE[0]))
        ManualShotParameters(**_valid_kwargs(shot_angle=MANUAL_SHOT_ANGLE[1]))

    def test_speed_at_the_exact_bounds_does_not_raise(self):
        # Act / Assert
        ManualShotParameters(**_valid_kwargs(cue_ball_speed=CUE_BALL_SPEED[0]))
        ManualShotParameters(**_valid_kwargs(cue_ball_speed=CUE_BALL_SPEED[1]))

    def test_offset_at_the_exact_bounds_does_not_raise(self):
        # Act / Assert
        ManualShotParameters(
            **_valid_kwargs(
                position_offset=(
                    POSITION_OFFSET_VERTICAL[0],
                    POSITION_OFFSET_HORIZONTAL[0],
                )
            )
        )
        ManualShotParameters(
            **_valid_kwargs(
                position_offset=(
                    POSITION_OFFSET_VERTICAL[1],
                    POSITION_OFFSET_HORIZONTAL[1],
                )
            )
        )


class TestOutOfRangeValuesRaise:
    """建構即驗證，越界一律拋 ValueError，不靜默夾住。"""

    def test_placement_x_out_of_range_raises(self):
        # Assert
        with pytest.raises(ValueError, match="cue_ball_placement"):
            ManualShotParameters(
                **_valid_kwargs(
                    cue_ball_placement=(
                        CUE_BALL_PLACEMENT_X[1] + 1.0,
                        CUE_BALL_PLACEMENT_Y[0],
                    )
                )
            )

    def test_placement_y_out_of_range_raises(self):
        # Assert
        with pytest.raises(ValueError, match="cue_ball_placement"):
            ManualShotParameters(
                **_valid_kwargs(
                    cue_ball_placement=(0.0, CUE_BALL_PLACEMENT_Y[0] - 1.0)
                )
            )

    def test_shot_angle_out_of_range_raises(self):
        # Assert
        with pytest.raises(ValueError, match="shot_angle"):
            ManualShotParameters(
                **_valid_kwargs(shot_angle=MANUAL_SHOT_ANGLE[1] + 1.0)
            )

    def test_speed_out_of_range_raises(self):
        # Assert
        with pytest.raises(ValueError, match="cue_ball_speed"):
            ManualShotParameters(
                **_valid_kwargs(cue_ball_speed=CUE_BALL_SPEED[0] - 0.1)
            )

    def test_offset_vertical_out_of_range_raises(self):
        # Assert
        with pytest.raises(ValueError, match="position_offset"):
            ManualShotParameters(
                **_valid_kwargs(
                    position_offset=(POSITION_OFFSET_VERTICAL[1] + 0.1, 0.0)
                )
            )

    def test_offset_horizontal_out_of_range_raises(self):
        # Assert
        with pytest.raises(ValueError, match="position_offset"):
            ManualShotParameters(
                **_valid_kwargs(
                    position_offset=(0.0, POSITION_OFFSET_HORIZONTAL[1] + 0.1)
                )
            )

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_shot_angle_raises(self, value: float):
        # Assert
        with pytest.raises(ValueError):
            ManualShotParameters(**_valid_kwargs(shot_angle=value))
