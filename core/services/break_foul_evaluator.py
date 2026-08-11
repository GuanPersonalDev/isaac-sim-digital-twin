from ..models.break_foul_result import BreakFoulResult

FIRST_CONTACT_FOUL_PENALTY = -1.5
INSUFFICIENT_RAIL_CONTACT_PENALTY = -0.5
# 整局沒碰到任何球，比碰到錯球更差（#124，2026-08-11）。
#
# 原本兩者同為 -1.5，於是 policy 起點附近整片 reward 是平的——第一輪訓練
# 238 個 iteration 後收斂到「母球一顆球都沒碰到」，spread 與 break_foul 終止
# 率雙雙歸零。分開之後犯規階梯變成嚴格遞增，「碰到東西」本身就是進步：
#
#   -2.0  沒碰到任何球
#   -1.5  碰到錯球
#   -0.5  碰到 1 號球，但未滿 4 顆號碼球碰顆星
#    0.0  合法開球
#
# ⚠️ 級距 0.5 是 `aim_shaping_calculator.AIM_REWARD_SCALE` 的上界依據，那裡有
#    import-time 檢查。調整本值時塑形滿分必須跟著檢查。
NO_CONTACT_FOUL_PENALTY = -2.0
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

    if first_contacted_ball_id is None:
        return BreakFoulResult(
            penalty=NO_CONTACT_FOUL_PENALTY,
            should_reset=True,
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
