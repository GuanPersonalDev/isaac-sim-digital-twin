from unittest.mock import MagicMock

import pytest

from core.models.contact_event import ContactEvent
from core.services.pocket_event_handler import PocketEventHandler

BALL_PRIM_PATHS = [f"/World/BilliardTable_0/Balls/Ball_{i}" for i in range(10)]


def _event(path_a: str, path_b: str) -> ContactEvent:
    return ContactEvent(
        actor_path_a=path_a,
        actor_path_b=path_b,
        collider_path_a=path_a,
        collider_path_b=path_b,
        impulse=0.5,
    )


@pytest.fixture
def physics_api():
    return MagicMock()


@pytest.fixture
def on_ball_pocketed():
    return MagicMock()


@pytest.fixture
def pocket_event_handler(physics_api, on_ball_pocketed):
    return PocketEventHandler(
        physics_api=physics_api,
        pocket_prim_paths=["/World/Table/Pocket_A"],
        ball_prim_paths=BALL_PRIM_PATHS,
        on_ball_pocketed=on_ball_pocketed,
    )


@pytest.fixture
def out_of_range_ball_pocket_contact_event():
    return _event("/World/BilliardTable_0/Balls/Ball_15", "/World/Table/Pocket_A")


@pytest.fixture
def pocket_contact_without_ball_path_event():
    return ContactEvent(
        actor_path_a="/World/CueStick",
        actor_path_b="/World/Table/Pocket_A",
        collider_path_a="/World/Table/Rail",
        collider_path_b="/World/Table/Pocket_A",
        impulse=0.5,
    )


@pytest.fixture
def invalid_ball_path_pocket_contact_event():
    return ContactEvent(
        actor_path_a="/World/BilliardTable_0/Balls/Ball_stick",
        actor_path_b="/World/Table/Pocket_A",
        collider_path_a="/World/BilliardTable_0/Balls/Ball_stick",
        collider_path_b="/World/Table/Pocket_A",
        impulse=0.5,
    )


class TestPocketEventHandler:
    def test_start_subscribes_to_contact_events(self):
        physics_api = MagicMock()
        on_ball_pocketed = MagicMock()
        handler = PocketEventHandler(
            physics_api=physics_api,
            pocket_prim_paths=["/World/Table/Pocket_A"],
            ball_prim_paths=BALL_PRIM_PATHS,
            on_ball_pocketed=on_ball_pocketed,
        )

        handler.start()

        physics_api.subscribe_contact_events.assert_called_once_with(
            handler.handle_contact_event
        )

    def test_start_enables_contact_reporting_for_pockets_and_balls(self):
        physics_api = MagicMock()
        handler = PocketEventHandler(
            physics_api=physics_api,
            pocket_prim_paths=["/World/Table/Pocket_A", "/World/Table/Pocket_B"],
            ball_prim_paths=BALL_PRIM_PATHS,
            on_ball_pocketed=MagicMock(),
        )

        handler.start()

        enabled_paths = [
            call.args[0]
            for call in physics_api.enable_contact_reporting.call_args_list
        ]
        assert "/World/Table/Pocket_A" in enabled_paths
        assert "/World/Table/Pocket_B" in enabled_paths
        assert "/World/BilliardTable_0/Balls/Ball_0" in enabled_paths
        assert "/World/BilliardTable_0/Balls/Ball_9" in enabled_paths

    def test_ball_pocket_contact_calls_ball_handler(self):
        physics_api = MagicMock()
        on_ball_pocketed = MagicMock()
        handler = PocketEventHandler(
            physics_api=physics_api,
            pocket_prim_paths=["/World/Table/Pocket_A"],
            ball_prim_paths=BALL_PRIM_PATHS,
            on_ball_pocketed=on_ball_pocketed,
        )

        handler.handle_contact_event(
            _event("/World/BilliardTable_0/Balls/Ball_3", "/World/Table/Pocket_A")
        )

        on_ball_pocketed.assert_called_once_with(3)

    def test_ball_pocket_contact_is_detected_in_reverse_order(self):
        physics_api = MagicMock()
        on_ball_pocketed = MagicMock()
        handler = PocketEventHandler(
            physics_api=physics_api,
            pocket_prim_paths=["/World/Table/Pocket_A"],
            ball_prim_paths=BALL_PRIM_PATHS,
            on_ball_pocketed=on_ball_pocketed,
        )

        handler.handle_contact_event(
            _event("/World/Table/Pocket_A", "/World/BilliardTable_0/Balls/Ball_4")
        )

        on_ball_pocketed.assert_called_once_with(4)

    def test_non_pocket_contact_is_ignored(self):
        physics_api = MagicMock()
        on_ball_pocketed = MagicMock()
        handler = PocketEventHandler(
            physics_api=physics_api,
            pocket_prim_paths=["/World/Table/Pocket_A"],
            ball_prim_paths=BALL_PRIM_PATHS,
            on_ball_pocketed=on_ball_pocketed,
        )

        handler.handle_contact_event(
            _event("/World/BilliardTable_0/Balls/Ball_3", "/World/BilliardTable_0/Balls/Ball_4")
        )

        on_ball_pocketed.assert_not_called()

    def test_pocket_contact_without_ball_path_is_ignored(
        self,
        pocket_event_handler,
        pocket_contact_without_ball_path_event,
        on_ball_pocketed,
    ):
        pocket_event_handler.handle_contact_event(pocket_contact_without_ball_path_event)

        on_ball_pocketed.assert_not_called()

    def test_invalid_ball_path_pocket_contact_is_ignored(
        self,
        pocket_event_handler,
        invalid_ball_path_pocket_contact_event,
        on_ball_pocketed,
    ):
        pocket_event_handler.handle_contact_event(invalid_ball_path_pocket_contact_event)

        on_ball_pocketed.assert_not_called()

    def test_out_of_range_ball_contact_is_ignored(
        self,
        pocket_event_handler,
        out_of_range_ball_pocket_contact_event,
        on_ball_pocketed,
    ):
        pocket_event_handler.handle_contact_event(out_of_range_ball_pocket_contact_event)

        on_ball_pocketed.assert_not_called()

    def test_stop_unsubscribes_contact_events(self):
        physics_api = MagicMock()
        handler = PocketEventHandler(
            physics_api=physics_api,
            pocket_prim_paths=["/World/Table/Pocket_A"],
            ball_prim_paths=BALL_PRIM_PATHS,
            on_ball_pocketed=MagicMock(),
        )

        handler.stop()

        physics_api.unsubscribe_contact_events.assert_called_once_with()
