import math

import pytest

from core.models.manual_shot_parameters import ManualShotParameters
from core.models.manual_shot_bounds import CUE_BALL_PLACEMENT_Y
from core.models.table_ball_set import TableBallSet
from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS
from core.services.manual_shot_feasibility import (
    ManualShotFeasibility,
    evaluate_manual_shot,
    evaluate_manual_shot_parameters,
)

# 產線母球在世界座標系裡的實際 Z（`core/models/billiard_table.py` 的
# `self._z_pos = 0`），不是 0.75——那是部分測試檔用來驗證「table_z 有沒有
# 被正確傳遞」的任意值，不是真實桌面高度。`compute_required_tilt_rad()`
# 拿 `table_z + ball_radius` 直接跟 `_RAIL_TOP_HEIGHT`/`_SAFETY_MARGIN`
# 這兩個相對於桌面的小常數比較，table_z 不是 0 附近時算出來的可行性會
# 整組失真，因此這裡固定用跟生產環境一致的 0.0。
_TABLE_Z = 0.0
_BALL_RADIUS = TableBallSet.DEFAULT_BALL_RADIUS


class TestEvaluateManualShotFeasibleCase:
    def test_break_shot_position_zero_angle_zero_offset_is_feasible(self):
        # 開球點＋0 度＋零偏移是最基本的合法擊球，必須可行——這是
        # 「切到手動什麼都不調就按擊球」的那一球（見 ManualShotParameters.default()）。
        # Act
        feasibility = evaluate_manual_shot(
            BREAK_SHOT_POSITIONS[0], 0.0, _TABLE_Z, _BALL_RADIUS, (0.0, 0.0)
        )

        # Assert
        assert feasibility.is_feasible is True
        assert feasibility.reason == ""

    def test_feasible_reason_is_always_empty_string(self):
        # 面板靠 reason == "" 判斷是否顯示紅線/擋鈕，可行時這裡不能是任何
        # 其他真值（例如 None）。
        # Act
        feasibility = evaluate_manual_shot(
            BREAK_SHOT_POSITIONS[0], 0.0, _TABLE_Z, _BALL_RADIUS, (0.0, 0.0)
        )

        # Assert
        assert feasibility.reason == ""
        assert isinstance(feasibility, ManualShotFeasibility)


class TestEvaluateManualShotInfeasibleCase:
    # ⚠️ 這個座標是實測掃描（不是照抄計畫文件推斷）找出來的：母球在
    # `CUE_BALL_PLACEMENT_Y` 下界、角度 0 度時，
    # `cue_pose_calculator.compute_tilted_wrist_pose()` 在生產環境的
    # `table_z=0.0` 下確實回傳 `tilt_rad is None`——握把→母球連線跟
    # y=-1.295 這面庫邊的交點離母球只有約 0.0536m（`d`），這個距離換算出
    # 的所需仰角 `required_sin ≈ 1.15 > 1.0`（`compute_required_tilt_rad()`
    # 判定「垂直抬桿都不夠，asin 已無定義域」），不是交點與母球重合
    # （d < 1e-6）那一種無解。
    _INFEASIBLE_CUE_BALL_XY = (0.0, CUE_BALL_PLACEMENT_Y[0])
    _INFEASIBLE_SHOT_ANGLE = 0.0

    def test_cue_ball_at_placement_y_lower_bound_is_geometry_unsolvable(self):
        # Act
        feasibility = evaluate_manual_shot(
            self._INFEASIBLE_CUE_BALL_XY,
            self._INFEASIBLE_SHOT_ANGLE,
            _TABLE_Z,
            _BALL_RADIUS,
            (0.0, 0.0),
        )

        # Assert
        assert feasibility.is_feasible is False
        assert feasibility.reason == "geometry_unsolvable"


class TestEvaluateManualShotParameters:
    def test_matches_underlying_function_for_feasible_parameters(self):
        # Arrange
        parameters = ManualShotParameters.default()

        # Act
        via_parameters = evaluate_manual_shot_parameters(parameters, _TABLE_Z, _BALL_RADIUS)
        via_raw_values = evaluate_manual_shot(
            parameters.cue_ball_placement,
            parameters.shot_angle,
            _TABLE_Z,
            _BALL_RADIUS,
            parameters.position_offset,
        )

        # Assert
        assert via_parameters == via_raw_values

    def test_matches_underlying_function_for_infeasible_parameters(self):
        # Arrange
        parameters = ManualShotParameters(
            cue_ball_placement=(0.0, CUE_BALL_PLACEMENT_Y[0]),
            shot_angle=0.0,
            cue_ball_speed=1.0,
            position_offset=(0.0, 0.0),
        )

        # Act
        feasibility = evaluate_manual_shot_parameters(parameters, _TABLE_Z, _BALL_RADIUS)

        # Assert
        assert feasibility.is_feasible is False
        assert feasibility.reason == "geometry_unsolvable"


class TestPositionOffsetAcceptsTupleOrList:
    @pytest.mark.parametrize("offset", [(0.0, 0.0), [0.0, 0.0]])
    def test_tuple_and_list_both_work(self, offset):
        # Act
        feasibility = evaluate_manual_shot(
            BREAK_SHOT_POSITIONS[0], 0.0, _TABLE_Z, _BALL_RADIUS, offset
        )

        # Assert
        assert feasibility.is_feasible is True


class TestNonFiniteInputsRaise:
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_shot_angle_raises(self, value: float):
        # Assert
        with pytest.raises(ValueError):
            evaluate_manual_shot(
                BREAK_SHOT_POSITIONS[0], value, _TABLE_Z, _BALL_RADIUS, (0.0, 0.0)
            )

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_cue_ball_xy_raises(self, value: float):
        # Assert
        with pytest.raises(ValueError):
            evaluate_manual_shot(
                (value, 0.0), 0.0, _TABLE_Z, _BALL_RADIUS, (0.0, 0.0)
            )

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_position_offset_raises(self, value: float):
        # Assert
        with pytest.raises(ValueError):
            evaluate_manual_shot(
                BREAK_SHOT_POSITIONS[0], 0.0, _TABLE_Z, _BALL_RADIUS, (value, 0.0)
            )

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_table_z_raises(self, value: float):
        # Assert
        with pytest.raises(ValueError):
            evaluate_manual_shot(
                BREAK_SHOT_POSITIONS[0], 0.0, value, _BALL_RADIUS, (0.0, 0.0)
            )
