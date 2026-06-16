from dataclasses import dataclass


@dataclass
class ContactEvent:
    """
    碰撞事件資訊
    """

    actor_path_a: str
    actor_path_b: str
    collider_path_a: str
    collider_path_b: str
    impulse: float
