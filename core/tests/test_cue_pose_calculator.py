import math

import numpy as np
import pytest

from core.services import cue_pose_calculator
from core.services.base_placement_calculator import required_grip_position


class TestComputeRequiredTiltRad:
    def test_no_crossing_returns_zero_tilt(self):
        # 握把→母球連線沿 x=0 這條垂直線，完全落在庫邊圍成的矩形內部，不會
        # 跟任何一面庫邊相交。
        tilt_rad, crossing = cue_pose_calculator.compute_required_tilt_rad(
            (0.0, 0.0), (0.0, 0.3), tip_height=0.028575
        )

        assert tilt_rad == 0.0
        assert crossing is None

    def test_crossing_far_enough_requires_positive_tilt(self):
        # 握把在庫邊之外、母球在庫邊之內，連線會跟 y=-1.295 這面庫邊相交。
        tilt_rad, crossing = cue_pose_calculator.compute_required_tilt_rad(
            (0.0, -2.0), (0.0, 0.0), tip_height=0.028575
        )

        assert tilt_rad is not None
        assert tilt_rad > 0.0
        assert crossing == pytest.approx((0.0, -1.295))

    def test_crossing_too_close_is_infeasible(self):
        # 母球緊貼在庫邊旁邊，即使垂直抬高也無法讓桿身在交點處清過庫邊頂部。
        tilt_rad, crossing = cue_pose_calculator.compute_required_tilt_rad(
            (0.0, -1.3), (0.0, -1.29), tip_height=0.0
        )

        assert tilt_rad is None
        assert crossing == pytest.approx((0.0, -1.295))


class TestComputeTiltedDirection:
    def test_flat_direction_matches_zero_tilt_shot_angle_zero(self):
        direction = cue_pose_calculator.compute_tilted_direction(0.0, 0.0)

        assert direction == pytest.approx([0.0, 1.0, 0.0])

    def test_flat_direction_matches_shot_angle_ninety(self):
        direction = cue_pose_calculator.compute_tilted_direction(90.0, 0.0)

        assert direction == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)

    def test_tilted_direction_has_negative_z_and_unit_norm(self):
        direction = cue_pose_calculator.compute_tilted_direction(0.0, math.radians(10.0))

        assert direction[2] < 0.0
        assert np.linalg.norm(direction) == pytest.approx(1.0)


class TestComputeContactPoint:
    def test_zero_offset_returns_ball_center(self):
        ball_center = np.array([1.0, 2.0, 3.0])
        direction = np.array([0.0, 1.0, 0.0])

        contact = cue_pose_calculator.compute_contact_point(ball_center, direction, [0.0, 0.0], 0.028575)

        assert contact == pytest.approx(ball_center)

    @pytest.mark.parametrize("shot_angle_deg", [0.0, 30.0, -45.0, 90.0])
    def test_side_offset_direction_matches_impulse_striking_service(self, shot_angle_deg):
        # training（impulse_striking_service）跟 demo（cue_pose_calculator）
        # 兩條腿的 position_offset[1]（左右）正負號語意必須一致，否則同一個
        # RL policy 在兩條腿上學到的「往左/右打」語意會相反。
        ball_center = np.array([0.0, 0.0, 0.0])
        direction = cue_pose_calculator.compute_tilted_direction(shot_angle_deg, 0.0)
        ball_radius = 0.028575

        contact = cue_pose_calculator.compute_contact_point(ball_center, direction, [0.0, 1.0], ball_radius)

        theta = math.radians(shot_angle_deg)
        expected_side = np.array([math.cos(theta), math.sin(theta), 0.0])
        assert (contact - ball_center) / ball_radius == pytest.approx(expected_side, abs=1e-9)

    def test_vertical_offset_moves_along_world_up_when_flat(self):
        ball_center = np.array([0.0, 0.0, 0.0])
        direction = cue_pose_calculator.compute_tilted_direction(0.0, 0.0)
        ball_radius = 0.028575

        contact = cue_pose_calculator.compute_contact_point(ball_center, direction, [1.0, 0.0], ball_radius)

        assert (contact - ball_center) / ball_radius == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)

    @pytest.mark.parametrize("position_offset", [[0.5, 0.0], [0.0, 0.5], [-0.5, -0.5]])
    def test_offset_magnitude_scales_with_ball_radius(self, position_offset):
        ball_center = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])
        ball_radius = 0.028575

        contact = cue_pose_calculator.compute_contact_point(ball_center, direction, position_offset, ball_radius)

        expected_magnitude = math.hypot(*position_offset) * ball_radius
        assert np.linalg.norm(contact - ball_center) == pytest.approx(expected_magnitude)


class TestComputeTiltedWristPose:
    def test_infeasible_geometry_returns_all_none(self):
        wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            (0.0, -1.29), 0.0, table_z=-1.3, ball_radius=0.0
        )

        assert wrist is None
        assert orientation is None
        assert tilt_rad is None

    def test_flat_case_wrist_xy_matches_required_grip_position(self):
        cue_ball = (0.0, 0.3)
        shot_angle_deg = 0.0
        table_z = 0.0
        ball_radius = 0.028575

        wrist, _, tilt_rad, _ = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, table_z, ball_radius
        )

        assert tilt_rad == 0.0
        expected_x, expected_y = required_grip_position(cue_ball[0], cue_ball[1], shot_angle_deg)
        assert wrist[0] == pytest.approx(expected_x)
        assert wrist[1] == pytest.approx(expected_y)
        assert wrist[2] == pytest.approx(table_z + ball_radius)

    def test_roll_changes_orientation_but_not_position_or_tilt(self):
        cue_ball = (0.0, 0.3)
        shot_angle_deg = 0.0
        table_z = 0.0
        ball_radius = 0.028575

        wrist_no_roll, orientation_no_roll, tilt_no_roll, _ = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, table_z, ball_radius, roll_rad=0.0
        )
        wrist_roll, orientation_roll, tilt_roll, _ = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, table_z, ball_radius, roll_rad=math.pi / 2
        )

        assert wrist_roll == pytest.approx(wrist_no_roll)
        assert tilt_roll == pytest.approx(tilt_no_roll)
        assert not np.allclose(orientation_roll, orientation_no_roll)


class TestComputeElevatedBridgeWaypoints:
    # (0.0, 0.0) 這個母球位置＋shot_angle=0 時，握把→母球連線會跟 y=-1.295
    # 這面庫邊相交（d=1.295m），需要一個很小但非零的抬高角（~2.7°），是個
    # 乾淨、可行的「需要抬高」測試案例。
    _FEASIBLE_TILT_KWARGS = dict(cue_ball_xy=(0.0, 0.0), shot_angle_deg=0.0, table_z=0.0, ball_radius=0.028575)
    # 單位四元數：代表 Phase 0 結束後（base_yaw=0）CANONICAL_REST_JOINTS
    # 水平指向世界 +Y 的姿態，見 table_orchestrator._execute_aim() 的
    # safe_orientation。
    _CURRENT_ORIENTATION = [1.0, 0.0, 0.0, 0.0]

    def test_returns_waypoints_in_order(self):
        current_position = [0.0, 0.0, 1.0]
        rotate_steps = 8

        waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
            current_position, self._CURRENT_ORIENTATION, rotate_steps=rotate_steps,
            backswing_distance_m=0.05, **self._FEASIBLE_TILT_KWARGS
        )

        assert waypoints is not None
        # B1 + B2 + rotate_steps 個 C1 中繼點 + C2。
        assert len(waypoints) == 2 + rotate_steps + 1
        # B1：xy 沿用 current_position，姿態沿用 current_orientation，只有
        # z 抬到安全高度（不等於原本的 current_position）。
        assert waypoints[0].position[:2] == pytest.approx(current_position[:2])
        assert waypoints[0].position != pytest.approx(current_position)
        assert waypoints[0].orientation == pytest.approx(self._CURRENT_ORIENTATION)
        # B2：跟 B1 同一個安全高度，姿態仍是 current_orientation，只有
        # xy 平移到最終腕部位置正上方。
        assert waypoints[1].position[2] == pytest.approx(waypoints[0].position[2])
        assert waypoints[1].orientation == pytest.approx(self._CURRENT_ORIENTATION)
        # C1 中繼點（index 2..2+rotate_steps-1）：位置全部固定在 B2 那一點，
        # 姿態逐步從 current_orientation 內插到最終傾斜姿態，最後一個中繼點
        # 姿態要等於最終傾斜姿態。
        c1_waypoints = waypoints[2 : 2 + rotate_steps]
        for wp in c1_waypoints:
            assert wp.position == pytest.approx(waypoints[1].position)
        assert c1_waypoints[0].orientation != pytest.approx(self._CURRENT_ORIENTATION)
        assert c1_waypoints[-1].orientation == pytest.approx(waypoints[-1].orientation)
        # C2：姿態不動（跟最後一個 C1 中繼點一致），純垂直下降到最終腕部位置。
        assert waypoints[-1].orientation == pytest.approx(c1_waypoints[-1].orientation)
        assert waypoints[-1].position != pytest.approx(c1_waypoints[-1].position)

    def test_infeasible_geometry_returns_none(self):
        waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
            [0.0, 0.0, 1.0], self._CURRENT_ORIENTATION,
            cue_ball_xy=(0.0, -1.29), shot_angle_deg=0.0, table_z=-1.3, ball_radius=0.0,
            backswing_distance_m=0.05,
        )

        assert waypoints is None

    def test_larger_safe_altitude_margin_raises_approach_height(self):
        current_position = [0.0, 0.0, 1.0]

        low = cue_pose_calculator.compute_elevated_bridge_waypoints(
            current_position, self._CURRENT_ORIENTATION, safe_altitude_margin=0.1,
            backswing_distance_m=0.05, **self._FEASIBLE_TILT_KWARGS
        )
        high = cue_pose_calculator.compute_elevated_bridge_waypoints(
            current_position, self._CURRENT_ORIENTATION, safe_altitude_margin=0.5,
            backswing_distance_m=0.05, **self._FEASIBLE_TILT_KWARGS
        )

        assert high[1].position[2] > low[1].position[2]
