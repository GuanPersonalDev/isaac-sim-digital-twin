from unittest.mock import MagicMock

import pytest

from core.services.ball_motion_monitor import BallMotionMonitor


def _batched_rigid_body_mock() -> MagicMock:
    """RigidBodyAPI 的測試替身，理由同 test_rolling_resistance_service.py：
    BallMotionMonitor 改成一次批次讀取，測試資料仍用逐顆描述。"""
    api = MagicMock()
    api.get_velocities.side_effect = lambda paths: (
        [api.get_linear_velocity(path) for path in paths],
        [api.get_angular_velocity(path) for path in paths],
    )
    return api


class TestBallMotionMonitor:
    def test_all_balls_stationary_returns_false(self):
        # Arrange
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.return_value = [0.0, 0.0, 0.0]
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1", "/World/Ball2"])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is False

    def test_one_ball_above_threshold_returns_true(self):
        # Arrange
        rigid_body_api = _batched_rigid_body_mock()

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
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.return_value = [0.001, 0.0, 0.0]
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1"])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is True

    def test_ball_just_below_threshold_counts_as_stationary(self):
        # Arrange
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.return_value = [0.0009, 0.0, 0.0]
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1"])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is False

    def test_empty_ball_list_returns_false(self):
        # Arrange
        rigid_body_api = _batched_rigid_body_mock()
        monitor = BallMotionMonitor(rigid_body_api, [])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is False
        rigid_body_api.get_linear_velocity.assert_not_called()

    def test_speed_magnitude_combines_all_axes(self):
        # Arrange
        rigid_body_api = _batched_rigid_body_mock()
        # 3-4-0 vector has magnitude 5.0, well above threshold
        rigid_body_api.get_linear_velocity.return_value = [3.0, 4.0, 0.0]
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1"])

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is True

    def test_rigid_body_api_exception_propagates(self):
        # Arrange
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.side_effect = RuntimeError("prim not found")
        monitor = BallMotionMonitor(rigid_body_api, ["/World/Ball1"])

        # Act / Assert
        with pytest.raises(RuntimeError, match="prim not found"):
            monitor.is_any_ball_moving()


class TestBallMotionMonitorBatchesReads:
    """效能契約（見 core/ports/rigid_body_api.py）：每次判斷只能做一次批次
    讀取。改回逐顆呼叫 get_linear_velocity() 會讓 GUI 每 frame 多出 10 次
    GPU→CPU 同步，實測是 FPS 從 12 掉下來的主因之一。"""

    def test_reads_all_ball_velocities_in_a_single_batched_call(self):
        # Arrange
        rigid_body_api = MagicMock()
        ball_paths = [f"/World/Ball{i}" for i in range(10)]
        rigid_body_api.get_velocities.return_value = (
            [[0.0, 0.0, 0.0] for _ in ball_paths],
            [[0.0, 0.0, 0.0] for _ in ball_paths],
        )
        monitor = BallMotionMonitor(rigid_body_api, ball_paths)

        # Act
        monitor.is_any_ball_moving()

        # Assert
        rigid_body_api.get_velocities.assert_called_once_with(ball_paths)
        rigid_body_api.get_linear_velocity.assert_not_called()

    def test_still_batches_when_a_ball_is_already_moving(self):
        """會動的球排在第一顆時也不能退化成「讀到就提前跳出」的逐顆版本。"""
        # Arrange
        rigid_body_api = MagicMock()
        ball_paths = ["/World/Ball0", "/World/Ball1"]
        rigid_body_api.get_velocities.return_value = (
            [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        monitor = BallMotionMonitor(rigid_body_api, ball_paths)

        # Act
        result = monitor.is_any_ball_moving()

        # Assert
        assert result is True
        assert rigid_body_api.get_velocities.call_count == 1
