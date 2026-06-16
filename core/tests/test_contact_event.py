from core.models.contact_event import ContactEvent


class TestContactEvent:
    def test_create_valid_contact_event_preserves_all_fields(self):
        event = ContactEvent(
            actor_path_a="/World/BilliardTable",
            actor_path_b="/World/Ball_1",
            collider_path_a="/World/BilliardTable/Rail_North",
            collider_path_b="/World/Ball_1/collision",
            impulse=0.42,
        )

        assert event.actor_path_a == "/World/BilliardTable"
        assert event.actor_path_b == "/World/Ball_1"
        assert event.collider_path_a == "/World/BilliardTable/Rail_North"
        assert event.collider_path_b == "/World/Ball_1/collision"
        assert event.impulse == 0.42

    def test_create_with_zero_impulse_preserves_boundary_value(self):
        event = ContactEvent(
            actor_path_a="/World/Ball_1",
            actor_path_b="/World/Ball_2",
            collider_path_a="/World/Ball_1/collision",
            collider_path_b="/World/Ball_2/collision",
            impulse=0.0,
        )

        assert event.impulse == 0.0

    def test_create_with_negative_impulse_preserves_value(self):
        event = ContactEvent(
            actor_path_a="/World/Ball_1",
            actor_path_b="/World/Ball_2",
            collider_path_a="/World/Ball_1/collision",
            collider_path_b="/World/Ball_2/collision",
            impulse=-1.0,
        )

        assert event.impulse == -1.0
