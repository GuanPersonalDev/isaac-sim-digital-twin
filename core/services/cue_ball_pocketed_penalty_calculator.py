CUE_BALL_POCKETED_PENALTY = -3.5


def calculate_cue_ball_pocketed_penalty(cue_ball_pocketed: bool) -> float:
    """
    白球進袋懲罰，見 docs/phase3-task-breakdown.md 的 Reward Function 表：
    白球進袋 -3.5（含 9 號球同時進袋情況，9 號球不加分）。

    這裡只單純判斷白球本身進袋與否，回傳無條件的懲罰值；跟 9 號球是否
    同時進袋互斥的規則，屬於 9 號球加分判定那邊的條件（該規則本來就要求
    「9 號球進袋且白球未進袋」才加分），不需要這個函式處理跨球邏輯。
    """
    return CUE_BALL_POCKETED_PENALTY if cue_ball_pocketed else 0.0
