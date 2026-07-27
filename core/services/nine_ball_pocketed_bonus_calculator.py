NINE_BALL_POCKETED_BONUS = 3.0


def calculate_nine_ball_pocketed_bonus(
    nine_ball_pocketed: bool,
    cue_ball_pocketed: bool,
) -> float:
    """9 號球進袋且白球未進袋時回傳獎勵，否則回傳 0.0。"""
    if nine_ball_pocketed and not cue_ball_pocketed:
        return NINE_BALL_POCKETED_BONUS

    return 0.0
