import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest


def _load_ports_module(module_name: str, filename: str) -> None:
    if module_name in sys.modules:
        return

    if "core.ports" not in sys.modules:
        ports_package = types.ModuleType("core.ports")
        ports_package.__path__ = []
        sys.modules["core.ports"] = ports_package

    path = Path(__file__).resolve().parents[1] / "ports" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _table_ball_set_class():
    _load_ports_module("core.ports.stage_api", "stage_api.py")
    _load_ports_module("core.ports.material_api", "material_api.py")
    from core.models.table_ball_set import TableBallSet
    return TableBallSet


@pytest.fixture
def stage_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def material_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def positions() -> dict[int, tuple[float, float]]:
    return {ball_id: (ball_id * 0.1, ball_id * -0.1) for ball_id in range(10)}


@pytest.fixture
def table_ball_set(stage_api: MagicMock, material_api: MagicMock):
    return _table_ball_set_class()(
        stage_api=stage_api,
        material_api=material_api,
        table_z=0.75,
        ball_radius=0.028575,
    )


def _prim_path(ball_id: int) -> str:
    return f"/World/Balls/ball_{ball_id}"


class TestTableBallSet:
    def test_build_calls_create_reference_prim_10_times(
        self,
        table_ball_set,
        stage_api: MagicMock,
        positions: dict[int, tuple[float, float]],
    ):
        table_ball_set.build(positions)

        assert stage_api.create_reference_prim.call_count == 10

    def test_build_sets_correct_z(
        self,
        table_ball_set,
        stage_api: MagicMock,
        positions: dict[int, tuple[float, float]],
    ):
        expected_z = pytest.approx(0.75 + 0.028575)

        table_ball_set.build(positions)

        z_values = [
            c.args[3]
            for c in stage_api.set_prim_translate.call_args_list
        ]
        assert all(z == expected_z for z in z_values)

    def test_hide_ball_calls_set_visibility_false(
        self,
        table_ball_set,
        stage_api: MagicMock,
        positions: dict[int, tuple[float, float]],
    ):
        table_ball_set.build(positions)

        table_ball_set.hide_ball(3)

        stage_api.set_visibility.assert_called_with(_prim_path(3), visible=False)

    def test_show_ball_calls_set_visibility_true(
        self,
        table_ball_set,
        stage_api: MagicMock,
        positions: dict[int, tuple[float, float]],
    ):
        table_ball_set.build(positions)

        table_ball_set.show_ball(3)

        stage_api.set_visibility.assert_called_with(_prim_path(3), visible=True)

    def test_reset_makes_all_balls_visible(
        self,
        table_ball_set,
        stage_api: MagicMock,
        positions: dict[int, tuple[float, float]],
    ):
        table_ball_set.build(positions)
        stage_api.set_visibility.reset_mock()

        table_ball_set.reset(positions)

        stage_api.set_visibility.assert_has_calls(
            [call(_prim_path(ball_id), visible=True) for ball_id in range(10)],
            any_order=True,
        )

    def test_hide_ball_invalid_id_raises(
        self,
        table_ball_set,
        positions: dict[int, tuple[float, float]],
    ):
        table_ball_set.build(positions)

        with pytest.raises(ValueError):
            table_ball_set.hide_ball(10)

    def test_build_missing_position_raises(self, table_ball_set):
        positions = {ball_id: (0.0, 0.0) for ball_id in range(9)}

        with pytest.raises(ValueError):
            table_ball_set.build(positions)

    def test_build_before_hide_raises(self, table_ball_set):
        with pytest.raises(RuntimeError):
            table_ball_set.hide_ball(0)
