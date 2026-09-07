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

    def test_is_wider_than_the_rl_training_shot_angle(self):
        # 這是兩把尺的核心斷言：手動面板的角度範圍必須明顯寬於 RL 為了
        # 訓練信號密度收窄的 action_bounds.SHOT_ANGLE（#231），否則
        # MANUAL_SHOT_ANGLE 這個獨立常數就沒有存在的必要。
        # Assert
        manual_low, manual_high = manual_shot_bounds.MANUAL_SHOT_ANGLE
        rl_low, rl_high = action_bounds.SHOT_ANGLE
        assert manual_low < rl_low
        assert manual_high > rl_high


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


class TestManualActionRejectedByRlNormalization:
    """給未來想把手動 Action 接進 RL 記錄管線的人看的可執行文件。"""

    def test_widened_angle_is_rejected_by_normalize_action_by_design(self):
        # 這是預期行為，不是 bug：手動面板允許 ±160°，遠寬於 RL 收窄後的
        # action_bounds.SHOT_ANGLE（±30°，Milestone A #231）。手動 Action
        # 直接餵進 normalize_action() 必然因為超出可表達範圍而拋
        # ValueError——夾住會把 90° 謊報成 30°（不同方向），所以刻意不接受
        # 靜默夾住這條路。想把手動 Action 接進 RL 記錄管線的人會在這裡被
        # 擋下並讀到這段說明。
        # Arrange
        low, high = manual_shot_bounds.MANUAL_SHOT_ANGLE
        rl_low, rl_high = action_bounds.SHOT_ANGLE
        assert not (rl_low <= high <= rl_high)  # 前提成立才有意義
        manual_action = Action(
            cue_ball_placement=[0.0, action_bounds.CUE_BALL_PLACEMENT_Y[0]],
            shot_angle=high,
            cue_ball_speed=action_bounds.CUE_BALL_SPEED[1],
            position_offset=[0.0, 0.0],
            should_execute_action=True,
        )

        # Act / Assert
        with pytest.raises(ValueError, match="shot_angle"):
            normalize_action(manual_action)
