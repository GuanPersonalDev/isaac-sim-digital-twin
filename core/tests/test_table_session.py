from unittest.mock import MagicMock, call

import pytest

from core.models.billiard_state import BilliardStatus
from core.models.observation import Observation
from core.services.table_session import DemoTableSession, TableSession


@pytest.fixture
def table() -> MagicMock:
    return MagicMock()


@pytest.fixture
def runtime() -> MagicMock:
    return MagicMock()


@pytest.fixture
def pocket_handler() -> MagicMock:
    return MagicMock()


@pytest.fixture
def rigid_body_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def robot_manager() -> MagicMock:
    return MagicMock()


@pytest.fixture
def articulation_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def table_session(
    table: MagicMock, runtime: MagicMock, pocket_handler: MagicMock, rigid_body_api: MagicMock
) -> TableSession:
    return TableSession(
        table_id="/World/Table_0",
        table=table,
        runtime=runtime,
        pocket_handler=pocket_handler,
        rigid_body_api=rigid_body_api,
    )


@pytest.fixture
def demo_table_session(
    table: MagicMock,
    runtime: MagicMock,
    pocket_handler: MagicMock,
    rigid_body_api: MagicMock,
    robot_manager: MagicMock,
    articulation_api: MagicMock,
) -> DemoTableSession:
    return DemoTableSession(
        table_id="/World/Table_Demo",
        table=table,
        runtime=runtime,
        pocket_handler=pocket_handler,
        rigid_body_api=rigid_body_api,
        robot_manager=robot_manager,
        articulation_api=articulation_api,
    )


class TestTableSession:
    def test_get_table_id_returns_constructed_id(self, table_session: TableSession):
        assert table_session.get_table_id() == "/World/Table_0"

    def test_tick_delegates_to_runtime(self, table_session: TableSession, runtime: MagicMock):
        table_session.tick()

        runtime.tick.assert_called_once_with()

    def test_get_current_state_delegates_to_runtime(
        self, table_session: TableSession, runtime: MagicMock
    ):
        runtime.get_current_state.return_value = BilliardStatus.STRIKING

        assert table_session.get_current_state() == BilliardStatus.STRIKING

    def test_get_last_observation_delegates_to_runtime(
        self, table_session: TableSession, runtime: MagicMock
    ):
        observation = Observation(
            ball_positions=[],
            cue_ball_position=[0.0, 0.0, 0.0],
            is_init_state=False,
            is_ball_moving=False,
            is_motion_complete=False,
            has_error=False,
        )
        runtime.get_last_observation.return_value = observation

        assert table_session.get_last_observation() is observation

    def test_get_last_observation_returns_none_when_runtime_has_none(
        self, table_session: TableSession, runtime: MagicMock
    ):
        runtime.get_last_observation.return_value = None

        assert table_session.get_last_observation() is None

    def test_get_ball_velocities_queries_each_ball(
        self, table_session: TableSession, table: MagicMock, rigid_body_api: MagicMock
    ):
        table.get_table_ball_set.return_value.get_ball_prim_paths.return_value = [
            "/World/Table_0/Balls/Ball_0",
            "/World/Table_0/Balls/Ball_1",
        ]
        rigid_body_api.get_linear_velocity.side_effect = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        rigid_body_api.get_angular_velocity.side_effect = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]

        velocities = table_session.get_ball_velocities()

        assert velocities == {
            0: ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            1: ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        }
        rigid_body_api.get_linear_velocity.assert_any_call("/World/Table_0/Balls/Ball_0")
        rigid_body_api.get_angular_velocity.assert_any_call("/World/Table_0/Balls/Ball_1")

    def test_destroy_stops_pocket_handler_before_table(
        self, table_session: TableSession, table: MagicMock, pocket_handler: MagicMock
    ):
        manager_mock = MagicMock()
        manager_mock.attach_mock(pocket_handler.stop, "pocket_handler_stop")
        manager_mock.attach_mock(table.destroy, "table_destroy")

        table_session.destroy()

        pocket_handler.stop.assert_called_once_with()
        table.destroy.assert_called_once_with()
        assert manager_mock.mock_calls == [
            call.pocket_handler_stop(),
            call.table_destroy(),
        ]


class TestDemoTableSession:
    def test_is_articulation_initialized_default_false(
        self, demo_table_session: DemoTableSession
    ):
        assert demo_table_session.is_articulation_initialized() is False

    def test_initialize_articulation_calls_api_and_marks_initialized(
        self, demo_table_session: DemoTableSession, articulation_api: MagicMock
    ):
        demo_table_session.initialize_articulation()

        articulation_api.initialize.assert_called_once_with()
        assert demo_table_session.is_articulation_initialized() is True

    def test_destroy_calls_shutdown_when_initialized(
        self, demo_table_session: DemoTableSession, articulation_api: MagicMock
    ):
        demo_table_session.initialize_articulation()

        demo_table_session.destroy()

        articulation_api.shutdown.assert_called_once_with()

    def test_destroy_skips_shutdown_when_not_initialized(
        self, demo_table_session: DemoTableSession, articulation_api: MagicMock
    ):
        demo_table_session.destroy()

        articulation_api.shutdown.assert_not_called()

    def test_destroy_cancels_pending_home_capture_regardless_of_initialized(
        self, demo_table_session: DemoTableSession, articulation_api: MagicMock
    ):
        demo_table_session.destroy()

        articulation_api.cancel_pending_home_capture.assert_called_once_with()

    def test_destroy_calls_robot_manager_destroy_before_super_destroy(
        self,
        demo_table_session: DemoTableSession,
        articulation_api: MagicMock,
        robot_manager: MagicMock,
        pocket_handler: MagicMock,
        table: MagicMock,
    ):
        demo_table_session.initialize_articulation()

        manager_mock = MagicMock()
        manager_mock.attach_mock(articulation_api.shutdown, "shutdown")
        manager_mock.attach_mock(articulation_api.cancel_pending_home_capture, "cancel")
        manager_mock.attach_mock(robot_manager.destroy, "robot_destroy")
        manager_mock.attach_mock(pocket_handler.stop, "pocket_stop")
        manager_mock.attach_mock(table.destroy, "table_destroy")

        demo_table_session.destroy()

        assert manager_mock.mock_calls == [
            call.shutdown(),
            call.cancel(),
            call.robot_destroy(),
            call.pocket_stop(),
            call.table_destroy(),
        ]
