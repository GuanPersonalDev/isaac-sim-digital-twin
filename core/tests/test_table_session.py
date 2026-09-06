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

    def test_request_full_reset_delegates_to_runtime(
        self, table_session: TableSession, runtime: MagicMock
    ):
        table_session.request_full_reset()

        runtime.request_full_reset.assert_called_once_with()

    def test_demo_session_request_full_reset_delegates_to_runtime(
        self, demo_table_session: DemoTableSession, runtime: MagicMock
    ):
        demo_table_session.request_full_reset()

        runtime.request_full_reset.assert_called_once_with()

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

    def test_initialize_articulation_registers_rmpflow_obstacles(
        self, demo_table_session: DemoTableSession, articulation_api: MagicMock, table: MagicMock
    ):
        """決策 6 的第一層防護：球檯與母球要註冊成 RMPflow 避障物。"""
        table_ball_set = table.get_table_ball_set.return_value
        table_ball_set.get_table_z.return_value = 0.8
        table_ball_set.get_ball_prim_paths.return_value = ["/World/T/Balls/Ball_0"]
        table_ball_set.get_ball_radius.return_value = 0.028575
        table.get_table_center.return_value = (1.0, 2.0, 0.0)

        demo_table_session.initialize_articulation()

        articulation_api.register_dynamic_sphere_obstacle.assert_called_once_with(
            "/World/T/Balls/Ball_0", 0.028575
        )
        center, size = articulation_api.register_static_box_obstacle.call_args.args
        assert center[0] == 1.0 and center[1] == 2.0
        # 方塊要整個落在桌面之下：中心低於 table_z、上緣剛好貼齊 table_z，
        # 桌面正上方（球桿實際操作空間）保持淨空。
        assert center[2] < 0.8
        assert center[2] + size[2] / 2.0 == pytest.approx(0.8)

    def test_initialize_articulation_syncs_initial_robot_base_pose(
        self,
        demo_table_session: DemoTableSession,
        articulation_api: MagicMock,
        robot_manager: MagicMock,
    ):
        """第一個動作 RESET 會用 RMPflow 把 HOME 關節角換算成世界座標目標，
        沒先同步底座位姿的話 RMPflow 會當底座在原點。"""
        robot_manager.get_initial_robot_base_position.return_value = (1.5, 0.0, 0.0)

        demo_table_session.initialize_articulation()

        articulation_api.set_robot_base_pose.assert_called_once_with(
            [1.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )

    def test_registers_obstacles_and_base_pose_only_after_initialize(
        self, demo_table_session: DemoTableSession, articulation_api: MagicMock
    ):
        """UR10e 的 RMPflow 控制器是在 initialize() 裡才建立的，先呼叫這兩個
        方法會被當成 no-op 丟掉。"""
        call_order = []
        articulation_api.initialize.side_effect = lambda: call_order.append("initialize")
        articulation_api.set_robot_base_pose.side_effect = (
            lambda *_args: call_order.append("base_pose")
        )
        articulation_api.register_static_box_obstacle.side_effect = (
            lambda *_args: call_order.append("box")
        )

        demo_table_session.initialize_articulation()

        assert call_order == ["initialize", "base_pose", "box"]

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
