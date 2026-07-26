import math
from unittest.mock import MagicMock

import pytest

from core.models.action import Action
from core.services.impulse_striking_service import (
    ImpulseStrikingService,
    compute_cue_ball_velocities,
)


def _action(
    cue_speed: float = 1.0,
    shot_angle: float = 0.0,
    position_offset: tuple[float, float] = (0.0, 0.0),
    cue_ball_placement: tuple[float, float] = (0.0, 0.0),
) -> Action:
    return Action(
        cue_speed=cue_speed,
        shot_angle=shot_angle,
        position_offset=list(position_offset),
        cue_ball_placement=list(cue_ball_placement),
        should_execute_action=False,
    )


class TestLinearVelocity:
    def test_zero_angle_travels_along_positive_y(self):
        # Arrange
        action = _action(cue_speed=2.0, shot_angle=0.0)

        # Act
        linear, _ = compute_cue_ball_velocities(action, ball_radius=1.0)

        # Assert
        assert linear == pytest.approx([0.0, 2.0, 0.0], abs=1e-9)

    def test_90_degree_angle_travels_along_negative_x(self):
        # Arrange
        action = _action(cue_speed=2.0, shot_angle=90.0)

        # Act
        linear, _ = compute_cue_ball_velocities(action, ball_radius=1.0)

        # Assert
        assert linear == pytest.approx([-2.0, 0.0, 0.0], abs=1e-9)

    def test_linear_velocity_magnitude_equals_cue_speed_regardless_of_angle(self):
        # Arrange
        action = _action(cue_speed=3.5, shot_angle=137.0)

        # Act
        linear, _ = compute_cue_ball_velocities(action, ball_radius=1.0)

        # Assert
        magnitude = math.sqrt(sum(c**2 for c in linear))
        assert magnitude == pytest.approx(3.5)

    def test_position_offset_does_not_affect_linear_velocity(self):
        # Arrange
        no_offset = _action(cue_speed=2.0, shot_angle=20.0, position_offset=(0.0, 0.0))
        with_offset = _action(cue_speed=2.0, shot_angle=20.0, position_offset=(0.4, -0.3))

        # Act
        linear_no_offset, _ = compute_cue_ball_velocities(no_offset, ball_radius=1.0)
        linear_with_offset, _ = compute_cue_ball_velocities(with_offset, ball_radius=1.0)

        # Assert
        assert linear_no_offset == pytest.approx(linear_with_offset)


class TestAngularVelocity:
    def test_zero_offset_produces_zero_angular_velocity(self):
        # Arrange
        action = _action(cue_speed=2.0, shot_angle=45.0, position_offset=(0.0, 0.0))

        # Act
        _, angular = compute_cue_ball_velocities(action, ball_radius=1.0)

        # Assert
        assert angular == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_top_offset_at_zero_angle_spins_about_horizontal_x_axis_only(self):
        # Arrange
        action = _action(cue_speed=2.0, shot_angle=0.0, position_offset=(0.2, 0.0))

        # Act
        _, angular = compute_cue_ball_velocities(action, ball_radius=1.0)

        # Assert
        assert angular[1] == pytest.approx(0.0, abs=1e-9)
        assert angular[2] == pytest.approx(0.0, abs=1e-9)
        assert angular[0] != pytest.approx(0.0)

    def test_side_offset_spins_about_vertical_z_axis_regardless_of_shot_angle(self):
        # Arrange / Act / Assert
        for shot_angle in (0.0, 33.0, 90.0, 210.0):
            action = _action(
                cue_speed=2.0, shot_angle=shot_angle, position_offset=(0.0, 0.3)
            )
            _, angular = compute_cue_ball_velocities(action, ball_radius=1.0)

            assert angular[0] == pytest.approx(0.0, abs=1e-9)
            assert angular[1] == pytest.approx(0.0, abs=1e-9)
            assert angular[2] == pytest.approx(1.2, abs=1e-9)  # k=4.0 (default 0.8 efficiency) * b=0.3

    def test_spin_efficiency_scales_angular_velocity_linearly(self):
        # Arrange
        action = _action(cue_speed=2.0, shot_angle=0.0, position_offset=(0.2, 0.3))

        # Act
        _, angular_full = compute_cue_ball_velocities(
            action, ball_radius=1.0, spin_efficiency=1.0
        )
        _, angular_half = compute_cue_ball_velocities(
            action, ball_radius=1.0, spin_efficiency=0.5
        )

        # Assert
        for full, half in zip(angular_full, angular_half):
            assert half == pytest.approx(full * 0.5)

    def test_larger_ball_radius_reduces_angular_velocity_for_same_offset_ratio(self):
        # Arrange
        action = _action(cue_speed=2.0, shot_angle=0.0, position_offset=(0.2, 0.0))

        # Act
        _, angular_small_radius = compute_cue_ball_velocities(action, ball_radius=0.5)
        _, angular_large_radius = compute_cue_ball_velocities(action, ball_radius=1.0)

        # Assert
        assert abs(angular_large_radius[0]) < abs(angular_small_radius[0])


class TestImpulseStrikingServiceStrike:
    def test_places_cue_ball_at_placement_xy(self):
        # Arrange
        rigid_body_api = MagicMock()
        service = ImpulseStrikingService(
            rigid_body_api, cue_ball_prim="/World/CueBall", ball_radius=1.0
        )
        action = _action(cue_speed=2.0, shot_angle=0.0, cue_ball_placement=(0.1, -0.2))

        # Act
        service.strike(action, table_x=5.0, table_y=3.0, table_z=0.75)

        # Assert
        # 必須用 RigidBodyAPI.set_position()（tensor API），不能用 StageAPI 的
        # raw xform op——見 core/ports/rigid_body_api.py 的說明，混用會讓
        # set_velocities() 靜默失效（實測回報的 bug：母球擊球後完全不動）。
        rigid_body_api.set_position.assert_called_once_with(
            "/World/CueBall", pytest.approx(5.1), pytest.approx(2.8), 1.75
        )

    def test_sets_computed_velocities_on_cue_ball(self):
        # Arrange
        rigid_body_api = MagicMock()
        service = ImpulseStrikingService(
            rigid_body_api, cue_ball_prim="/World/CueBall", ball_radius=1.0
        )
        action = _action(cue_speed=2.0, shot_angle=0.0, position_offset=(0.2, 0.3))

        # Act
        service.strike(action, table_x=0.0, table_y=0.0, table_z=0.75)

        # Assert
        expected_linear, expected_angular = compute_cue_ball_velocities(
            action, ball_radius=1.0
        )
        rigid_body_api.set_velocities.assert_called_once_with(
            "/World/CueBall", expected_linear, expected_angular
        )

    def test_places_ball_before_setting_velocities(self):
        # Arrange
        rigid_body_api = MagicMock()
        manager = MagicMock()
        manager.attach_mock(rigid_body_api.set_position, "set_position")
        manager.attach_mock(rigid_body_api.set_velocities, "set_velocities")
        service = ImpulseStrikingService(
            rigid_body_api, cue_ball_prim="/World/CueBall", ball_radius=1.0
        )
        action = _action()

        # Act
        service.strike(action, table_x=0.0, table_y=0.0, table_z=0.75)

        # Assert
        assert [call[0] for call in manager.mock_calls] == ["set_position", "set_velocities"]

    def test_uses_configured_spin_efficiency(self):
        # Arrange
        rigid_body_api = MagicMock()
        service = ImpulseStrikingService(
            rigid_body_api,
            cue_ball_prim="/World/CueBall",
            ball_radius=1.0,
            spin_efficiency=0.5,
        )
        action = _action(cue_speed=2.0, shot_angle=0.0, position_offset=(0.2, 0.3))

        # Act
        service.strike(action, table_x=0.0, table_y=0.0, table_z=0.75)

        # Assert
        expected_linear, expected_angular = compute_cue_ball_velocities(
            action, ball_radius=1.0, spin_efficiency=0.5
        )
        rigid_body_api.set_velocities.assert_called_once_with(
            "/World/CueBall", expected_linear, expected_angular
        )
