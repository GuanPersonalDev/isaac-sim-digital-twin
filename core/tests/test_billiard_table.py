from unittest.mock import MagicMock, patch

import pytest

from core.models.billiard_table import BilliardTable


class TestBilliardTableRobot:
    def test_init_creates_robot_with_offset_world_position(self):
        stage_api = MagicMock()
        material_api = MagicMock()

        with (
            patch("core.models.billiard_table.TableBallSet") as table_ball_set_class,
            patch("core.models.billiard_table.BreakShotPositionProvider") as position_provider_class,
            patch("core.models.billiard_table.UR5Robot", create=True) as robot_class,
        ):
            position_provider_class.return_value.get_positions.return_value = {
                ball_id: (0.0, 0.0) for ball_id in range(10)
            }

            BilliardTable(
                base_path="/World/BilliardTable",
                stage_api=stage_api,
                material_api=material_api,
                position=(2.0, 3.0),
            )

        table_ball_set_class.return_value.build.assert_called_once()
        robot_class.assert_called_once_with(
            "/World/BilliardTable",
            stage_api,
            (3.5, 3.0, 0.0),
        )


@pytest.fixture
def billiard_table_base_path():
    return "/World/BilliardTable"


@pytest.fixture
def stage_api():
    return MagicMock()


@pytest.fixture
def material_api():
    return MagicMock()


@pytest.fixture
def billiard_table(stage_api, material_api, billiard_table_base_path):
    with (
        patch("core.models.billiard_table.TableBallSet"),
        patch("core.models.billiard_table.BreakShotPositionProvider") as position_provider_class,
        patch("core.models.billiard_table.UR5Robot", create=True),
    ):
        position_provider_class.return_value.get_positions.return_value = {
            ball_id: (0.0, 0.0) for ball_id in range(10)
        }

        return BilliardTable(
            base_path=billiard_table_base_path,
            stage_api=stage_api,
            material_api=material_api,
            position=(2.0, 3.0),
        )


class TestBilliardTableLifecycle:
    def test_get_table_prim_path_returns_table_path(
        self, billiard_table, billiard_table_base_path
    ):
        assert (
            billiard_table.get_table_prim_path()
            == f"{billiard_table_base_path}/Table"
        )

    def test_destroy_clears_internal_state(self, billiard_table):
        billiard_table.destroy()

        assert billiard_table._table_set is None
        assert billiard_table._robot is None
