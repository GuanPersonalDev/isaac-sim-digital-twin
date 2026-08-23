import numpy as np
import pytest

from core.services import swing_trajectory_calculator as calc
from core.models.action_bounds import CUE_BALL_SPEED


class TestComputeRequiredTipSpeed:
    def test_round_trips_with_action_bounds_forward_formula(self):
        # action_bounds.py CUE_BALL_SPEED 上界的推導公式：
        # v_ball = v_cue * (1+e) * M/(M+m)。compute_required_tip_speed 是它的
        # 反函式，兩者要能互相還原。
        cue_ball_speed = CUE_BALL_SPEED[1]

        tip_speed = calc.compute_required_tip_speed(cue_ball_speed)
        momentum_ratio = (1.0 + calc.RESTITUTION_COEFFICIENT) * calc.CUE_STICK_MASS_KG / (
            calc.CUE_STICK_MASS_KG + calc.CUE_BALL_MASS_KG
        )
        recovered_cue_ball_speed = tip_speed * momentum_ratio

        assert recovered_cue_ball_speed == pytest.approx(cue_ball_speed)

    def test_matches_documented_peak_tip_speed(self):
        # docs/task-176-swing-speed-spec.md 記載的實測桿尖峰值速度約 2.5302 m/s，
        # 對應 action_bounds.py CUE_BALL_SPEED 上界 3.3392 m/s。
        tip_speed = calc.compute_required_tip_speed(CUE_BALL_SPEED[1])

        assert tip_speed == pytest.approx(2.5302, abs=0.01)

    def test_lower_speed_gives_lower_tip_speed(self):
        low = calc.compute_required_tip_speed(CUE_BALL_SPEED[0])
        high = calc.compute_required_tip_speed(CUE_BALL_SPEED[1])

        assert low < high


class TestComputeFollowThroughDistance:
    def test_monotonically_increasing_with_speed(self):
        low_speed_distance = calc.compute_follow_through_distance(0.5)
        high_speed_distance = calc.compute_follow_through_distance(2.5)

        assert high_speed_distance > low_speed_distance

    def test_clipped_to_lower_bound(self):
        distance = calc.compute_follow_through_distance(1e-6)

        assert distance == pytest.approx(calc._FOLLOW_THROUGH_MIN_M)

    def test_clipped_to_upper_bound(self):
        distance = calc.compute_follow_through_distance(1000.0)

        assert distance == pytest.approx(calc._FOLLOW_THROUGH_MAX_M)


class TestComputeBackswingPosition:
    def test_moves_opposite_to_direction(self):
        contact = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])

        backswing = calc.compute_backswing_position(contact, direction, backswing_distance=0.15)

        assert backswing == pytest.approx([0.0, -0.15, 0.0])


class TestComputeSwingWaypoints:
    def test_returns_two_waypoints(self):
        waypoints = calc.compute_swing_waypoints(
            contact_position=[0.0, 0.0, 0.5],
            contact_orientation=[1.0, 0.0, 0.0, 0.0],
            direction_unit=[0.0, 1.0, 0.0],
            cue_ball_speed=CUE_BALL_SPEED[1],
        )

        assert len(waypoints) == 2

    def test_first_waypoint_has_zero_velocity(self):
        waypoints = calc.compute_swing_waypoints(
            contact_position=[0.0, 0.0, 0.5],
            contact_orientation=[1.0, 0.0, 0.0, 0.0],
            direction_unit=[0.0, 1.0, 0.0],
            cue_ball_speed=CUE_BALL_SPEED[1],
        )

        assert waypoints[0].linear_velocity == pytest.approx([0.0, 0.0, 0.0])

    def test_second_waypoint_velocity_matches_direction_and_required_speed(self):
        direction_unit = [0.0, 1.0, 0.0]
        cue_ball_speed = CUE_BALL_SPEED[1]

        waypoints = calc.compute_swing_waypoints(
            contact_position=[0.0, 0.0, 0.5],
            contact_orientation=[1.0, 0.0, 0.0, 0.0],
            direction_unit=direction_unit,
            cue_ball_speed=cue_ball_speed,
        )

        expected_speed = calc.compute_required_tip_speed(cue_ball_speed)
        actual_velocity = np.array(waypoints[1].linear_velocity)
        assert np.linalg.norm(actual_velocity) == pytest.approx(expected_speed)
        assert actual_velocity / np.linalg.norm(actual_velocity) == pytest.approx(direction_unit)

    def test_second_waypoint_position_is_beyond_contact_point_not_at_it(self):
        # 鎖住隨揮設計：終點必須是 contact_position 前方一小段隨揮距離，
        # 不是 contact_position 本身——否則 P 控制器在球心處還有殘留位置
        # 誤差貢獻，桿尖通過球心當下會提早減速，接觸瞬間速度就不準了。
        contact_position = [0.0, 0.0, 0.5]

        waypoints = calc.compute_swing_waypoints(
            contact_position=contact_position,
            contact_orientation=[1.0, 0.0, 0.0, 0.0],
            direction_unit=[0.0, 1.0, 0.0],
            cue_ball_speed=CUE_BALL_SPEED[1],
        )

        assert waypoints[1].position != pytest.approx(contact_position)
        # 沿 direction_unit (+Y) 前進，Y 應該比 contact_position 大。
        assert waypoints[1].position[1] > contact_position[1]

    def test_both_waypoints_share_orientation(self):
        contact_orientation = [0.7071, 0.7071, 0.0, 0.0]

        waypoints = calc.compute_swing_waypoints(
            contact_position=[0.0, 0.0, 0.5],
            contact_orientation=contact_orientation,
            direction_unit=[0.0, 1.0, 0.0],
            cue_ball_speed=CUE_BALL_SPEED[1],
        )

        assert waypoints[0].orientation == pytest.approx(contact_orientation)
        assert waypoints[1].orientation == pytest.approx(contact_orientation)

    def test_backswing_waypoint_is_behind_contact_point(self):
        contact_position = [0.0, 0.0, 0.5]

        waypoints = calc.compute_swing_waypoints(
            contact_position=contact_position,
            contact_orientation=[1.0, 0.0, 0.0, 0.0],
            direction_unit=[0.0, 1.0, 0.0],
            cue_ball_speed=CUE_BALL_SPEED[1],
            backswing_distance=0.15,
        )

        assert waypoints[0].position[1] == pytest.approx(contact_position[1] - 0.15)
