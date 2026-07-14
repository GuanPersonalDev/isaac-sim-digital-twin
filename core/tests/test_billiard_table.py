from unittest.mock import MagicMock, patch

import pytest

from core.models.billiard_table import BilliardTable


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
def billiard_table_position():
    return (2.0, 3.0)


@pytest.fixture
def billiard_table(
    stage_api, material_api, billiard_table_base_path, billiard_table_position
):
    with (
        patch("core.models.billiard_table.TableBallSet"),
        patch("core.models.billiard_table.BreakShotPositionProvider") as position_provider_class,
    ):
        position_provider_class.return_value.get_positions.return_value = {
            ball_id: (0.0, 0.0) for ball_id in range(10)
        }

        return BilliardTable(
            base_path=billiard_table_base_path,
            stage_api=stage_api,
            material_api=material_api,
            position=billiard_table_position,
        )


class TestBilliardTableCenter:
    def test_billiard_table_get_table_center(
        self, billiard_table, billiard_table_position
    ):
        x_pos, y_pos = billiard_table_position

        assert billiard_table.get_table_center() == (x_pos, y_pos, 0.0)


class TestBilliardTableRobot:
    def test_billiard_table_does_not_create_robot(self, billiard_table):
        assert not hasattr(billiard_table, "_robot")


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
