from ..models.observation import Observation
from .numeric_validation import validate_finite_number, validate_max_offset, validate_2d_value


_EXPECTED_BALL_COUNT = 10
_RL_BALL_ORDER = tuple(range(1, 10)) + (0,)


def encode_rl_observation(
    observation: Observation,
    table_position: tuple[float, float],
    max_offset: float
) -> list[float]:
    """將執行期 Observation 編碼為固定 21 維 RL 球位向量。"""
    table_x, table_y = _validate_table_position(table_position)
    limit = validate_max_offset(max_offset)
    ball_positions = observation.ball_positions

    if len(ball_positions) != _EXPECTED_BALL_COUNT:
        raise ValueError(
            "observation.ball_positions must contain exactly 10 balls"
        )

    encoded: list[float] = []
    for ball_id in _RL_BALL_ORDER:
        world_x, world_y = validate_2d_value(
            ball_positions[ball_id],
            field_name=f"ball_positions[{ball_id}]",
        )
        # RigidBodyAPI 提供的是世界座標；平行環境中的每張桌子可能位於不同
        # 世界位置。扣除桌台世界 XY 後，相同球局會得到相同的桌台相對座標，
        # 避免 RL Policy 把桌子擺放位置誤當成球局特徵。
        encoded.extend(
            (
                world_x - table_x,
                world_y - table_y,
            )
        )
        
    encoded.append(limit)
    return encoded


def _validate_table_position(
    table_position: tuple[float, float],
) -> tuple[float, float]:
    if len(table_position) != 2:
        raise ValueError("table_position must contain exactly two values")

    return (
        validate_finite_number(table_position[0], "table_position[0]"),
        validate_finite_number(table_position[1], "table_position[1]"),
    )
