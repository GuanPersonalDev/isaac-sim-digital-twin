import math

import pytest

from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS
from core.services.spread_score_calculator import (
    SPREAD_RACK,
    SPREAD_REF,
    SPREAD_REWARD_SCALE,
    TABLE_LENGTH,
    TABLE_WIDTH,
    calculate_spread_score,
    spread_score_to_reward,
)

_TABLE_AREA = TABLE_LENGTH * TABLE_WIDTH
_TABLE_DIAGONAL = math.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)


def _grid_positions() -> dict[int, tuple[float, float]]:
    """3x3 網格，間距 0.1m：凸包是邊長 0.2 的正方形（面積 0.04），
    每顆球的最近鄰距離都精確等於 0.1（對角鄰居距離 0.1*sqrt(2) 較遠，
    不會被選為最近鄰），可以手算出精確的期望分數。"""
    ball_id = 1
    positions = {}
    for row in range(3):
        for col in range(3):
            positions[ball_id] = (col * 0.1, row * 0.1)
            ball_id += 1
    return positions


class TestCalculateSpreadScoreExactValues:
    def test_grid_layout_computes_exact_known_score(self):
        positions = _grid_positions()

        score = calculate_spread_score(positions, pocketed_ball_ids=set())

        expected_area = 0.2 * 0.2
        expected_normalized_area = expected_area / _TABLE_AREA
        expected_avg_nn_distance = 0.1
        expected_normalized_distance = expected_avg_nn_distance / _TABLE_DIAGONAL
        expected_score = 0.5 * expected_normalized_area + 0.5 * expected_normalized_distance

        assert score == pytest.approx(expected_score)

    def test_rectangle_hull_with_all_balls_pocketed_isolates_area_component(self):
        # 距離分量在「少於 2 顆未進袋球」時固定是 1.0（滿分），可以用這個
        # trick 單獨驗證凸包面積分量算得精不精確。4 個角落決定凸包，
        # 其餘 5 顆球放在角落內部（不影響凸包)。
        positions = {
            1: (0.0, 0.0),
            2: (1.0, 0.0),
            3: (1.0, 0.5),
            4: (0.0, 0.5),
            5: (0.5, 0.25),
            6: (0.5, 0.25),
            7: (0.5, 0.25),
            8: (0.5, 0.25),
            9: (0.5, 0.25),
        }

        score = calculate_spread_score(positions, pocketed_ball_ids=set(range(1, 10)))

        expected_normalized_area = (1.0 * 0.5) / _TABLE_AREA
        expected_score = 0.5 * expected_normalized_area + 0.5 * 1.0

        assert score == pytest.approx(expected_score)


class TestCalculateSpreadScoreComparative:
    def test_widely_spread_balls_score_higher_than_clustered_balls(self):
        clustered = {i: (0.001 * i, 0.001 * i) for i in range(1, 10)}
        spread = {
            1: (0.0, 0.0),
            2: (TABLE_LENGTH, 0.0),
            3: (TABLE_LENGTH, TABLE_WIDTH),
            4: (0.0, TABLE_WIDTH),
            5: (TABLE_LENGTH / 2, TABLE_WIDTH / 2),
            6: (TABLE_LENGTH / 4, TABLE_WIDTH / 4),
            7: (TABLE_LENGTH * 3 / 4, TABLE_WIDTH / 4),
            8: (TABLE_LENGTH / 4, TABLE_WIDTH * 3 / 4),
            9: (TABLE_LENGTH * 3 / 4, TABLE_WIDTH * 3 / 4),
        }

        clustered_score = calculate_spread_score(clustered, pocketed_ball_ids=set())
        spread_score = calculate_spread_score(spread, pocketed_ball_ids=set())

        assert spread_score > clustered_score

    def test_score_is_clamped_to_one_when_theoretical_max_exceeded(self):
        # 現實中球不會超出桌面範圍，但函式本身要保底夾住，不能回傳 >1。
        huge_square = {
            1: (0.0, 0.0),
            2: (100.0, 0.0),
            3: (100.0, 100.0),
            4: (0.0, 100.0),
            5: (50.0, 50.0),
            6: (25.0, 25.0),
            7: (75.0, 25.0),
            8: (25.0, 75.0),
            9: (75.0, 75.0),
        }

        score = calculate_spread_score(huge_square, pocketed_ball_ids=set())

        assert score <= 1.0


class TestCalculateSpreadScorePocketedBalls:
    def test_pocketed_ball_position_still_counts_toward_hull_area(self):
        base_positions = {
            1: (0.0, 0.0),
            2: (1.0, 0.0),
            3: (1.0, 0.5),
            4: (0.5, 0.25),
            5: (0.5, 0.25),
            6: (0.5, 0.25),
            7: (0.5, 0.25),
            8: (0.5, 0.25),
            9: (0.5, 0.25),
        }
        # ball 4 進袋，用袋口座標 (0.0, 0.5) 代入，把凸包從三角形擴成矩形
        positions_with_pocket_substitution = dict(base_positions)
        positions_with_pocket_substitution[4] = (0.0, 0.5)

        score_without_fourth_corner = calculate_spread_score(
            base_positions, pocketed_ball_ids={4}
        )
        score_with_fourth_corner = calculate_spread_score(
            positions_with_pocket_substitution, pocketed_ball_ids={4}
        )

        assert score_with_fourth_corner > score_without_fourth_corner

    def test_pocketed_balls_excluded_from_nearest_neighbor_distance(self):
        positions = _grid_positions()
        # ball 5 是正中心 (0.1, 0.1)，把它排除後，其餘 8 顆球的最近鄰距離
        # 應該不變（grid 邊長/邊角球的最近鄰本來就不是中心點）。
        score_with_center_included = calculate_spread_score(positions, pocketed_ball_ids=set())
        score_with_center_pocketed = calculate_spread_score(positions, pocketed_ball_ids={5})

        assert score_with_center_included == pytest.approx(score_with_center_pocketed)


class TestCalculateSpreadScoreFewRemainingBalls:
    def test_returns_full_distance_score_when_zero_balls_remain(self):
        positions = _grid_positions()

        score = calculate_spread_score(positions, pocketed_ball_ids=set(range(1, 10)))

        expected_normalized_area = (0.2 * 0.2) / _TABLE_AREA
        expected_score = 0.5 * expected_normalized_area + 0.5 * 1.0
        assert score == pytest.approx(expected_score)

    def test_returns_full_distance_score_when_one_ball_remains(self):
        positions = _grid_positions()

        score = calculate_spread_score(positions, pocketed_ball_ids=set(range(1, 9)))

        expected_normalized_area = (0.2 * 0.2) / _TABLE_AREA
        expected_score = 0.5 * expected_normalized_area + 0.5 * 1.0
        assert score == pytest.approx(expected_score)

    def test_computes_real_distance_when_exactly_two_balls_remain(self):
        positions = _grid_positions()
        # 只留 ball 1 (0,0) 與 ball 2 (0.1,0)，距離精確等於 0.1
        score = calculate_spread_score(positions, pocketed_ball_ids=set(range(3, 10)))

        expected_normalized_area = (0.2 * 0.2) / _TABLE_AREA
        expected_normalized_distance = 0.1 / _TABLE_DIAGONAL
        expected_score = 0.5 * expected_normalized_area + 0.5 * expected_normalized_distance
        assert score == pytest.approx(expected_score)


class TestCalculateSpreadScoreValidation:
    def test_raises_value_error_when_ball_positions_missing_keys(self):
        incomplete_positions = {i: (0.0, 0.0) for i in range(1, 9)}  # 缺 ball 9

        with pytest.raises(ValueError):
            calculate_spread_score(incomplete_positions, pocketed_ball_ids=set())

    def test_raises_value_error_when_ball_positions_has_extra_keys(self):
        positions_with_cue_ball = _grid_positions()
        positions_with_cue_ball[0] = (0.0, 0.0)  # 白球不該出現在這裡

        with pytest.raises(ValueError):
            calculate_spread_score(positions_with_cue_ball, pocketed_ball_ids=set())


class TestSpreadScoreToReward:
    """#123：reward 用的重新正規化。"""

    def test_calibration_matches_two_runpod_controlled_break_runs(self):
        # 2026-08-11，RunPod 真實 PhysX、最大速度控制式開球：seed 123 / 456
        # 各 508 筆 first_contact == 1，pooled mean = 0.0420084。錨點取 0.0420；
        # scale 取 1.0，讓平均控制式開球約 +1 reward，並為極端值保留安全空間。
        assert SPREAD_REF == pytest.approx(0.0420)
        assert SPREAD_REWARD_SCALE == pytest.approx(1.0)

    def test_spread_rack_constant_matches_the_actual_rack_layout(self):
        # SPREAD_RACK 是硬寫的數值（避免 calculator 反向依賴 provider），這條
        # 對拍是它唯一的防線：開球擺位或桌台尺寸一改，常數就會失準，而失準
        # 不會報錯——只會讓「球沒動」的 reward 不再是 0。
        # Arrange
        rack = {ball_id: BREAK_SHOT_POSITIONS[ball_id] for ball_id in range(1, 10)}

        # Act
        score = calculate_spread_score(rack, pocketed_ball_ids=set())

        # Assert
        assert score == pytest.approx(SPREAD_RACK)

    def test_rack_layout_earns_exactly_zero_reward(self):
        # 球完全沒動 = 沒有任何貢獻。這是整個轉換的錨點之一。
        # Assert
        assert spread_score_to_reward(SPREAD_RACK) == pytest.approx(0.0)

    def test_reference_anchor_earns_the_reward_scale(self):
        # 另一個錨點：RunPod 控制式開球平均 = SPREAD_REWARD_SCALE。
        # Assert
        assert spread_score_to_reward(SPREAD_REF) == pytest.approx(SPREAD_REWARD_SCALE)

    def test_reward_is_strictly_increasing_in_spread_score(self):
        # Arrange
        ascending = [0.0, SPREAD_RACK, SPREAD_REF, 0.1, 0.5, 1.0]

        # Act
        rewards = [spread_score_to_reward(score) for score in ascending]

        # Assert
        assert rewards == sorted(rewards)
        assert len(set(rewards)) == len(rewards)

    def test_best_reachable_break_stays_below_the_cue_scratch_penalty(self):
        # 兩輪 RunPod 實測最大值是 0.10395。上修到 0.11 作為護欄後，reward
        # 仍不得蓋過母球落袋的 -3.5，否則 policy 會學成「打得夠散就可以順便
        # scratch」。這條測試把校準後的安全餘裕釘住。
        # Arrange
        best_reachable_spread_score = 0.11
        cue_scratch_penalty_magnitude = 3.5

        # Act
        reward = spread_score_to_reward(best_reachable_spread_score)

        # Assert
        assert reward < cue_scratch_penalty_magnitude
