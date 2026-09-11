import pytest

from core.models import action_bounds, manual_shot_bounds
from core.models.action import Action
from core.services.rl_action_decoder import normalize_action


class TestManualShotAngle:
    def test_range_is_plus_minus_160(self):
        # 162.8° 是 Ur10eSwingStrategy 基座不撞進球桌的解析上界（見
        # manual_shot_bounds.py 檔案級 docstring），160 留 2.8° 安全餘裕。
        # Assert
        assert manual_shot_bounds.MANUAL_SHOT_ANGLE == (-160.0, 160.0)

    def test_is_narrower_than_the_collision_free_analytical_bound(self):
        # 安全區必須嚴格小於幾何解析上界，否則就沒有安全餘裕可言。
        # Assert
        _, high = manual_shot_bounds.MANUAL_SHOT_ANGLE
        assert high < 162.8

    def test_is_independent_of_the_rl_training_shot_angle(self):
        # 兩把尺各自獨立（#115）：手動面板是基座不撞桌的安全區 ±160°，
        # RL 契約已復原整圈 (-180, 180)（#232-core）。獨立常數仍有存在
        # 必要——安全區不是整圈，也不是訓練用的收窄尺。
        # Assert
        manual_low, manual_high = manual_shot_bounds.MANUAL_SHOT_ANGLE
        rl_low, rl_high = action_bounds.SHOT_ANGLE
        assert (manual_low, manual_high) != (rl_low, rl_high)
        assert manual_low > rl_low
        assert manual_high < rl_high


class TestReExportedPhysicalBounds:
    """擺位/速度/偏移四項是物理能力邊界，手動與 RL 共用同一份數值，本檔
    只能轉出、不得重新定義——逐一比對確保沒有第二份實作悄悄漂移（#228）。
    """

    @pytest.mark.parametrize(
        "name",
        [
            "CUE_BALL_PLACEMENT_X",
            "CUE_BALL_PLACEMENT_Y",
            "CUE_BALL_SPEED",
            "POSITION_OFFSET_VERTICAL",
            "POSITION_OFFSET_HORIZONTAL",
        ],
    )
    def test_re_exported_value_equals_action_bounds(self, name: str):
        # Assert
        assert getattr(manual_shot_bounds, name) == getattr(action_bounds, name)

    @pytest.mark.parametrize(
        "name",
        [
            "CUE_BALL_PLACEMENT_X",
            "CUE_BALL_PLACEMENT_Y",
            "CUE_BALL_SPEED",
            "POSITION_OFFSET_VERTICAL",
            "POSITION_OFFSET_HORIZONTAL",
        ],
    )
    def test_re_exported_value_is_the_same_object_not_a_copy(self, name: str):
        # 轉出而非重新定義：同一個 tuple 物件，不是數值相同的另一份字面量。
        # Assert
        assert getattr(manual_shot_bounds, name) is getattr(action_bounds, name)


class TestManualActionExpressibleInRlNormalization:
    """給未來想把手動 Action 接進 RL 記錄管線的人看的可執行文件。"""

    def test_manual_safety_zone_is_expressible_in_rl_after_full_circle_restore(
        self,
    ):
        # Milestone A 收窄 ±30 時，手動 ±160 餵 normalize_action 會拋
        # ValueError。#232 復原整圈後 ±160 是合法方向，必須能正規化。
        # Arrange
        high = manual_shot_bounds.MANUAL_SHOT_ANGLE[1]
        rl_low, rl_high = action_bounds.SHOT_ANGLE
        assert rl_low <= high <= rl_high
        manual_action = Action(
            cue_ball_placement=[0.0, action_bounds.CUE_BALL_PLACEMENT_Y[0]],
            shot_angle=high,
            cue_ball_speed=action_bounds.CUE_BALL_SPEED[1],
            position_offset=[0.0, 0.0],
            should_execute_action=True,
        )

        # Act
        recovered = normalize_action(manual_action)

        # Assert
        assert -1.0 <= recovered[2] <= 1.0
