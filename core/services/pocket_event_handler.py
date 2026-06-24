from typing import Callable

from ..ports.physics_api import PhysicsAPI
from ..models.contact_event import ContactEvent
import re

_BALL_PATH_PATTERN = re.compile(r"(?:^|/)ball_(\d+)(?:$|/)")


class PocketEventHandler:
    def __init__(
        self,
        physics_api: PhysicsAPI,
        pocket_prim_paths: list[str],
        ball_prim_paths: list[str],
        on_ball_pocketed: Callable[[int], None],
    ) -> None:
        self._physics_api = physics_api
        self._pocket_prim_paths = pocket_prim_paths
        self._ball_prim_paths = ball_prim_paths
        self._on_ball_pocketed = on_ball_pocketed

    def start(self):
        for prim in self._pocket_prim_paths:
            self._physics_api.enable_contact_reporting(prim)

        for ball_prim_path in self._ball_prim_paths:
            self._physics_api.enable_contact_reporting(ball_prim_path)

        self._physics_api.subscribe_contact_events(self.handle_contact_event)

    def handle_contact_event(self, event: ContactEvent):

        if not self._has_pocket_path(event):
            return

        ball_prim_path = self._get_match_prim_path("ball_", event)
        if ball_prim_path is None:
            return

        ball_id = self._get_ball_id_from_prim_path(ball_prim_path)
        if ball_id is None:
            return
        self._on_ball_pocketed(ball_id)

    def _event_paths(self, event: ContactEvent) -> tuple[str, str, str, str]:
        return (
            event.actor_path_a,
            event.actor_path_b,
            event.collider_path_a,
            event.collider_path_b,
        )

    def _has_pocket_path(self, event: ContactEvent) -> bool:
        return any(self._is_pocket_path(path) for path in self._event_paths(event))

    def _is_pocket_path(self, prim_path: str) -> bool:
        for pocket_path in self._pocket_prim_paths:
            if prim_path == pocket_path:
                return True
            if prim_path.startswith(f"{pocket_path}/"):
                return True

        return False

    def _get_match_prim_path(self, key_word: str, event: ContactEvent):
        for path in self._event_paths(event):
            if key_word in path:
                return path

        return None

    def _get_ball_id_from_prim_path(self, prim_path: str) -> int | None:
        match = _BALL_PATH_PATTERN.search(prim_path)
        if match is None:
            return None

        ball_id = int(match.group(1))
        if ball_id not in range(10):
            return None

        return ball_id

    def stop(self):
        self._physics_api.unsubscribe_contact_events()
