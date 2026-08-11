import math

import pytest

from core.models.action_bounds import CUE_BALL_PLACEMENT_X, CUE_BALL_PLACEMENT_Y
from core.models.table_ball_set import TableBallSet
from core.services.aim_shaping_calculator import (
    AIM_REFERENCE_GAP,
    AIM_REWARD_SCALE,
    closest_approach_to_reward,
)
from core.services.break_foul_evaluator import (
    FIRST_CONTACT_FOUL_PENALTY,
    NO_CONTACT_FOUL_PENALTY,
)
from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS


class TestReferenceGap:
    def test_reference_gap_is_the_farthest_legal_placement(self):
        """歸零距離必須是「母球擺最遠時的表面間距」，由幾何現算。

        寫死數字的話，`CUE_BALL_PLACEMENT_*` 或 `BREAK_SHOT_POSITIONS` 一改，
        塑形的斜率就會偷偷跑掉而不報錯。
        """
        one_ball = BREAK_SHOT_POSITIONS[1]
        expected = max(
            math.dist((x, y), one_ball)
            for x in CUE_BALL_PLACEMENT_X
            for y in CUE_BALL_PLACEMENT_Y
        ) - 2.0 * TableBallSet.DEFAULT_BALL_RADIUS

        assert AIM_REFERENCE_GAP == pytest.approx(expected)

    def test_every_legal_placement_starts_inside_the_reference_gap(self):
        """任何合法擺位的初始間距都 <= 歸零距離。

        這保證塑形分數天生落在 [0, AIM_REWARD_SCALE]，呼叫端不必再夾一次。
        """
        one_ball = BREAK_SHOT_POSITIONS[1]
        diameter = 2.0 * TableBallSet.DEFAULT_BALL_RADIUS

        for x in CUE_BALL_PLACEMENT_X:
            for y in CUE_BALL_PLACEMENT_Y:
                gap = math.dist((x, y), one_ball) - diameter
                assert gap <= AIM_REFERENCE_GAP + 1e-12


class TestClosestApproachToReward:
    def test_contact_gets_the_full_scale(self):
        assert closest_approach_to_reward(0.0) == pytest.approx(AIM_REWARD_SCALE)

    def test_reference_gap_gets_zero(self):
        assert closest_approach_to_reward(AIM_REFERENCE_GAP) == 0.0

    def test_beyond_the_reference_gap_stays_zero(self):
        assert closest_approach_to_reward(AIM_REFERENCE_GAP * 2.0) == 0.0

    def test_never_measured_gets_zero(self):
        """訓練端的初值是 `math.inf`（這一局沒量到），視同沒靠近。"""
        assert closest_approach_to_reward(math.inf) == 0.0

    def test_reward_decreases_monotonically_with_distance(self):
        gaps = [i * AIM_REFERENCE_GAP / 20.0 for i in range(21)]
        rewards = [closest_approach_to_reward(gap) for gap in gaps]

        assert rewards == sorted(rewards, reverse=True)
        assert len(set(rewards)) == len(rewards)

    def test_reward_is_linear_in_distance(self):
        """線性而非指數：遠端也要有梯度。

        #124 第一輪的 policy 正好收斂在遠端，指數衰減在那裡會壓成 0，
        等於這一項沒做事。
        """
        quarter = closest_approach_to_reward(AIM_REFERENCE_GAP * 0.25)
        half = closest_approach_to_reward(AIM_REFERENCE_GAP * 0.5)
        three_quarters = closest_approach_to_reward(AIM_REFERENCE_GAP * 0.75)

        assert half - three_quarters == pytest.approx(quarter - half)

    def test_stays_inside_the_declared_range(self):
        for gap in (0.0, 0.01, 0.5, 1.0, AIM_REFERENCE_GAP, math.inf):
            assert 0.0 <= closest_approach_to_reward(gap) <= AIM_REWARD_SCALE

    @pytest.mark.parametrize("bad", [-1e-9, -0.5])
    def test_negative_distance_is_rejected(self, bad: float):
        with pytest.raises(ValueError):
            closest_approach_to_reward(bad)

    def test_nan_is_rejected(self):
        """NaN 不能靜默通過：`nan >= x` 是 False，會一路算出 NaN reward。"""
        with pytest.raises(ValueError):
            closest_approach_to_reward(math.nan)


class TestShapingCannotInvertTheFoulLadder:
    def test_scale_is_smaller_than_the_smallest_rung(self):
        """塑形滿分 < 犯規階梯最小級距，否則排序會反轉。

        反轉的後果：「沒碰到球但瞄很準」(-2.0 + 滿分) 超過「碰到錯球」(-1.5)，
        policy 學到靠近而不是命中——正好是這一項要修的那種缺陷。
        """
        smallest_rung = FIRST_CONTACT_FOUL_PENALTY - NO_CONTACT_FOUL_PENALTY

        assert AIM_REWARD_SCALE < smallest_rung

    def test_best_no_contact_still_loses_to_worst_wrong_ball_contact(self):
        best_miss = NO_CONTACT_FOUL_PENALTY + closest_approach_to_reward(0.0)
        worst_wrong_contact = FIRST_CONTACT_FOUL_PENALTY + closest_approach_to_reward(
            AIM_REFERENCE_GAP
        )

        assert best_miss < worst_wrong_contact


class TestGradientIsAboveTheNoiseFloor:
    def test_ten_versus_thirty_degree_aim_error_is_distinguishable(self):
        """10° 與 30° 的瞄準誤差要拉得開，否則推不動 policy。

        #124 第一輪觀測到的 surrogate 雜訊底是 ±0.001 量級；兩者的塑形差
        必須明顯高於它。距離由母球到 1 號球的實際球心距現算。
        """
        cue = BREAK_SHOT_POSITIONS[0]
        one_ball = BREAK_SHOT_POSITIONS[1]
        distance = math.dist(cue, one_ball)
        diameter = 2.0 * TableBallSet.DEFAULT_BALL_RADIUS

        def gap_for(error_deg: float) -> float:
            return max(
                0.0, distance * math.sin(math.radians(error_deg)) - diameter
            )

        difference = closest_approach_to_reward(gap_for(10.0)) - (
            closest_approach_to_reward(gap_for(30.0))
        )

        assert difference > 0.05
