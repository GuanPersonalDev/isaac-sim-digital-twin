from unittest.mock import MagicMock

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
