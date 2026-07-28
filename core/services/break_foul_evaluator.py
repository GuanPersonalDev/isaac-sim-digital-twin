from ..models.break_foul_result import BreakFoulResult

FIRST_CONTACT_FOUL_PENALTY = -1.5
INSUFFICIENT_RAIL_CONTACT_PENALTY = -0.5
MIN_RAIL_CONTACTED_OBJECT_BALLS = 4
_VALID_OBJECT_BALL_IDS = frozenset(range(1, 10))


def evaluate_break_foul(
    first_contacted_ball_id: int | None,
    pocketed_object_ball_ids: set[int],
    rail_contacted_object_ball_ids: set[int],
) -> BreakFoulResult:
    """判斷開球犯規，回傳扣分與是否立即重置。"""
    _validate_inputs(
        first_contacted_ball_id,
        pocketed_object_ball_ids,
        rail_contacted_object_ball_ids,
    )

    if first_contacted_ball_id != 1:
        return BreakFoulResult(
            penalty=FIRST_CONTACT_FOUL_PENALTY,
            should_reset=True,
        )

    if (
        not pocketed_object_ball_ids
        and len(rail_contacted_object_ball_ids)
        < MIN_RAIL_CONTACTED_OBJECT_BALLS
    ):
        return BreakFoulResult(
            penalty=INSUFFICIENT_RAIL_CONTACT_PENALTY,
            should_reset=False,
        )

    return BreakFoulResult(penalty=0.0, should_reset=False)


def _validate_inputs(
    first_contacted_ball_id: int | None,
    pocketed_object_ball_ids: set[int],
    rail_contacted_object_ball_ids: set[int],
) -> None:
    if first_contacted_ball_id is not None:
        _validate_object_ball_ids(
            {first_contacted_ball_id},
            field_name="first_contacted_ball_id",
        )
    _validate_object_ball_ids(
        pocketed_object_ball_ids,
        field_name="pocketed_object_ball_ids",
    )
    _validate_object_ball_ids(
        rail_contacted_object_ball_ids,
        field_name="rail_contacted_object_ball_ids",
    )


def _validate_object_ball_ids(
    ball_ids: set[int],
    field_name: str,
) -> None:
    invalid_ball_ids = ball_ids - _VALID_OBJECT_BALL_IDS
    if invalid_ball_ids:
        raise ValueError(
            f"{field_name} 只能包含 1–9，實際收到無效值："
            f"{sorted(invalid_ball_ids)}"
        )
