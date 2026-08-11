import math

import pytest

from core.controllers.script_controller import ScriptController
from core.models.action_bounds import (
    ACTION_BOUNDS,
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    CUE_BALL_PLACEMENT_X,
    CUE_BALL_PLACEMENT_Y,
    CUE_BALL_SPEED,
    POSITION_OFFSET_HORIZONTAL,
    POSITION_OFFSET_VERTICAL,
    SHOT_ANGLE,
)
from core.models.billiard_state import BilliardStatus
from core.models.observation import Observation
from core.models.table_ball_set import TableBallSet
from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS
from core.services.rolling_resistance_service import GRAVITY, ROLLING_FRICTION_COEFF


# #110 的 6 維索引表。索引、欄位與數值都是對外契約：policy 依索引依序取值，
# 訓練當下的順序會固化進匯出的權重，因此這裡把三者一起釘死。要改動必須先
# 改 Issue，不是改測試。
_ISSUE_110_INDEX_TABLE = (
    ("placement_x", 0, "cue_ball_placement[0]", (-0.606425, 0.606425)),
    ("placement_y", 1, "cue_ball_placement[1]", (-1.241425, -0.635)),
    ("shot_angle", 2, "shot_angle", (-30.0, 30.0)),
    ("cue_ball_speed", 3, "cue_ball_speed", (0.65, 3.3392)),
    ("offset_vertical", 4, "position_offset[0]", (-0.5, 0.5)),
    ("offset_horizontal", 5, "position_offset[1]", (-0.5, 0.5)),
)

_BOUNDS_PARAMS = tuple(
    pytest.param(index, field_name, expected, id=test_id)
    for test_id, index, field_name, expected in _ISSUE_110_INDEX_TABLE
)

_FIELD_PARAMS = tuple(
    pytest.param(index, field_name, id=test_id)
    for test_id, index, field_name, _ in _ISSUE_110_INDEX_TABLE
)

_NAMED_BOUNDS_IN_ISSUE_110_ORDER = (
    CUE_BALL_PLACEMENT_X,
    CUE_BALL_PLACEMENT_Y,
    SHOT_ANGLE,
    CUE_BALL_SPEED,
    POSITION_OFFSET_VERTICAL,
    POSITION_OFFSET_HORIZONTAL,
)


class TestActionBoundsContract:
    def test_action_dim_is_six(self):
        # Assert
        assert ACTION_DIM == 6
        assert len(ACTION_BOUNDS) == ACTION_DIM

    @pytest.mark.parametrize(("index", "field_name", "expected"), _BOUNDS_PARAMS)
    def test_bounds_match_issue_110_index_table(
        self, index: int, field_name: str, expected: tuple[float, float]
    ):
        # Assert
        assert ACTION_BOUNDS[index] == expected, (
            f"index {index}（{field_name}）的物理域範圍與 #110 不一致"
        )

    def test_named_constants_sit_at_their_issue_110_index(self):
        # 兩維偏移的數值完全相同，只比對數值抓不到互換；這裡比對具名常數
        # 本身，確保 ACTION_BOUNDS 的排列順序沒有被調換。
        # Assert
        assert ACTION_BOUNDS == _NAMED_BOUNDS_IN_ISSUE_110_ORDER

    @pytest.mark.parametrize(("index", "field_name"), _FIELD_PARAMS)
    def test_every_dimension_is_a_finite_ordered_pair(
        self, index: int, field_name: str
    ):
        # Act
        low, high = ACTION_BOUNDS[index]

        # Assert
        assert math.isfinite(low)
        assert math.isfinite(high)
        assert low < high, f"index {index}（{field_name}）的上下限顛倒或退化"


class TestAggregatedVectors:
    def test_low_and_high_align_with_action_bounds(self):
        # Assert
        assert len(ACTION_LOW) == ACTION_DIM
        assert len(ACTION_HIGH) == ACTION_DIM
        assert list(ACTION_LOW) == [low for low, _ in ACTION_BOUNDS]
        assert list(ACTION_HIGH) == [high for _, high in ACTION_BOUNDS]

    def test_low_is_strictly_below_high_elementwise(self):
        # gymnasium Box 逐元素比對，任一維反轉都會讓取樣與裁切靜默失效。
        # Assert
        assert all(low < high for low, high in zip(ACTION_LOW, ACTION_HIGH))


class TestDimensionSemantics:
    def test_shot_angle_is_centred_on_the_rack(self):
        # #231 問題 1：Gaussian policy 的初始輸出集中在 normalized 0，那個值
        # 必須對應「正對球堆」（0° 朝桌台 +Y）。原本的 (0, 360) 中點是 180°，
        # 等於未訓練的 policy 預設把母球往 kitchen 底庫打。
        # Assert
        assert (SHOT_ANGLE[0] + SHOT_ANGLE[1]) / 2 == 0.0

    def test_shot_angle_covers_every_legal_aim_at_the_one_ball(self):
        # #231 問題 2：Milestone A 把區間收窄到 ±30° 換取探索解析度
        # （命中質量比 2.9% → 17.2%）。收窄的下限由幾何決定，不是拍腦袋的
        # 數字——母球從**任何**合法 kitchen 擺位都必須瞄得到 1 號球，再加上
        # 接觸本身的容錯窗口。
        #
        # 這條測試現算而不寫死：桌台尺寸、開球擺位或 kitchen 範圍一改，
        # 收窄過頭會直接失敗，而不是變成「某些擺位打不到球堆」的靜默缺陷。
        # Arrange
        one_ball_x, one_ball_y = BREAK_SHOT_POSITIONS[1]
        ball_diameter = 2 * TableBallSet.DEFAULT_BALL_RADIUS

        # Act：kitchen 四個角落是最極端的瞄準需求
        required_aim_deg = 0.0
        contact_window_deg = 0.0
        for cue_x in CUE_BALL_PLACEMENT_X:
            for cue_y in CUE_BALL_PLACEMENT_Y:
                dx = one_ball_x - cue_x
                dy = one_ball_y - cue_y
                # 0° 朝 +Y、正角朝 -X，所以第一引數取 -dx
                required_aim_deg = max(
                    required_aim_deg, abs(math.degrees(math.atan2(-dx, dy)))
                )
                distance = math.hypot(dx, dy)
                contact_window_deg = max(
                    contact_window_deg,
                    math.degrees(math.atan(ball_diameter / distance)),
                )

        # Assert
        assert SHOT_ANGLE[1] >= required_aim_deg + contact_window_deg
        assert SHOT_ANGLE[0] <= -(required_aim_deg + contact_window_deg)

    def test_position_offset_axes_are_symmetric_and_identical(self):
        # 圓形可行域 clamp 的前提：兩軸同尺規且對稱，反正規化才是等比縮放，
        # 圓不會被壓成橢圓（#222）。
        # Assert
        assert POSITION_OFFSET_VERTICAL == POSITION_OFFSET_HORIZONTAL
        assert POSITION_OFFSET_VERTICAL[0] == -POSITION_OFFSET_VERTICAL[1]

    def test_position_offset_stays_within_miscue_limit(self):
        # 超過 0.5R 即滑桿，物理上不可用。
        # Assert
        assert POSITION_OFFSET_VERTICAL[1] <= 0.5
        assert POSITION_OFFSET_HORIZONTAL[1] <= 0.5

    def test_cue_ball_placement_y_stays_inside_kitchen(self):
        # Kitchen 位於 head string（Y = -0.635）以下，開球擺位不得越線。
        # Assert
        assert CUE_BALL_PLACEMENT_Y[1] <= -0.635

    def test_cue_ball_speed_lower_bound_excludes_no_op(self):
        # 執行期 no-op Action 用 0.0，RL 有效下限是 0.65（#110 定 0.5，#123 上調）。
        # Assert
        assert CUE_BALL_SPEED[0] == 0.65
        assert CUE_BALL_SPEED[0] > 0.0

    def test_cue_ball_speed_lower_bound_reaches_the_rack(self):
        # #123：下限必須讓母球從**任何**合法擺位都滾得到 1 號球，否則正規化域
        # 低端是死區。行程 = 最遠 kitchen 擺位到 1 號球的距離扣掉接觸時的 2R；
        # 減速度取純滾動的 μg（rolling_resistance_service 強制純滾動）。
        # Arrange
        rolling_deceleration = ROLLING_FRICTION_COEFF * GRAVITY
        ball_diameter = 2 * TableBallSet.DEFAULT_BALL_RADIUS
        one_ball_y = BREAK_SHOT_POSITIONS[1][1]
        travel = (one_ball_y - CUE_BALL_PLACEMENT_Y[0]) - ball_diameter

        # Act
        minimum_speed = math.sqrt(2 * rolling_deceleration * travel)

        # Assert
        assert CUE_BALL_SPEED[0] > minimum_speed


class TestSingleSourceOfTruth:
    def test_script_controller_no_longer_defines_its_own_speed_limit(self):
        # 3.3392 曾同時存在於 ScriptController.MAX_CUE_BALL_SPEED；#114 要求
        # 單一來源，重新引入類別常數會讓兩處數值有機會漂移。
        # Assert
        assert not hasattr(ScriptController, "MAX_CUE_BALL_SPEED")

    def test_striking_action_uses_the_shared_upper_bound(self):
        # Arrange
        controller = ScriptController()
        controller.get_action(_observation(is_motion_complete=True))
        controller.get_action(_observation(is_init_state=True))

        # Act
        action = controller.get_action(_observation(is_motion_complete=True))

        # Assert
        assert controller.get_current_state() == BilliardStatus.STRIKING
        assert action.cue_ball_speed == CUE_BALL_SPEED[1]


def _observation(
    is_init_state: bool = False,
    is_ball_moving: bool = False,
    is_motion_complete: bool = False,
    has_error: bool = False,
) -> Observation:
    return Observation(
        ball_positions=[[0.0, 0.0, 0.0]],
        cue_ball_position=[-0.3, 0.0, 0.0],
        is_init_state=is_init_state,
        is_ball_moving=is_ball_moving,
        is_motion_complete=is_motion_complete,
        has_error=has_error,
    )
