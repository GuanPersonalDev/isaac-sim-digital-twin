from unittest.mock import MagicMock

import pytest

from core.services.ball_motion_monitor import BallMotionMonitor


class TestBallMotionMonitor:
    def test_all_balls_stationary_returns_false(self):
        # Arrange
        rigid_body_api = MagicMock()
        rigid_body_api.get_linear_velocity.return_value = [0.0, 0.0, 0.0]
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1", "/World/Ball2"])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is False

    def test_one_ball_above_threshold_returns_true(self):
        # Arrange
        rigid_body_api = MagicMock()

        def get_linear_velocity(prim_path: str) -> list[float]:
            if prim_path == "/World/Ball2":
                return [0.5, 0.0, 0.0]
            return [0.0, 0.0, 0.0]

        rigid_body_api.get_linear_velocity.side_effect = get_linear_velocity
        monitor = BallMotionMonitor(
            rigid_body_api, ["/World/Ball1", "/World/Ball2", "/World/Ball3"]
        )

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is True

    def test_ball_exactly_at_threshold_counts_as_moving(self):
        # Arrange
        rigid_body_api = MagicMock()
        rigid_body_api.get_linear_velocity.return_value = [0.001, 0.0, 0.0]
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1"])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is True

    def test_ball_just_below_threshold_counts_as_stationary(self):
        # Arrange
        rigid_body_api = MagicMock()
        rigid_body_api.get_linear_velocity.return_value = [0.0009, 0.0, 0.0]
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1"])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is False

    def test_empty_ball_list_returns_false(self):
        # Arrange
        rigid_body_api = MagicMock()
        monitor = BallMotionMonitor(rigid_body_api, [])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is False
        rigid_body_api.get_linear_velocity.assert_not_called()

    def test_speed_magnitude_combines_all_axes(self):
        # Arrange
        rigid_body_api = MagicMock()
        # 3-4-0 vector has magnitude 5.0, well above threshold
        rigid_body_api.get_linear_velocity.return_value = [3.0, 4.0, 0.0]
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1"])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is True

    def test_rigid_body_api_exception_propagates(self):
        # Arrange
        rigid_body_api = MagicMock()
        rigid_body_api.get_linear_velocity.side_effect = RuntimeError("prim not found")
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1"])

        # Act / Assert
        with pytest.raises(RuntimeError, match="prim not found"):
            monitor.is_any_ball_moving()
