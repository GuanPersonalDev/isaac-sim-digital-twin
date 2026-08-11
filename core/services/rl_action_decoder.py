"""RL 6 維動作向量與執行期 `Action` 的雙向換算。

訓練端（`BilliardEnv`）與 Demo 端（`ModelController`）import 同一份，避免
兩端各自實作導致欄位順序或裁切行為不一致 —— 維度或順序錯位時 policy 不會
報錯，只會安靜輸出垃圾動作。

物理域的上下限一律取自 `core/models/action_bounds.py`（單一來源），本模組
不得出現硬編碼的邊界數值。
"""

from ..models.action import Action
from ..models.action_bounds import ACTION_BOUNDS, ACTION_DIM
from .numeric_validation import (
    validate_2d_value,
    validate_finite_number,
    validate_max_offset,
)
from .position_offset_limiter import clamp_position_offset


_PLACEMENT_X, _PLACEMENT_Y, _SHOT_ANGLE, _SPEED, _OFFSET_V, _OFFSET_H = range(
    ACTION_DIM
)
_ANGLE_PERIOD = 360.0


def _wrap_angle(angle: float) -> float:
    """把任意角度折成「最接近動作空間中心」的等價角度。

    以**區間中心**為錨，而不是下界——這樣涵蓋整圈與收窄兩種情形都正確：

    - `SHOT_ANGLE = (-180, 180)`：中心 0，折回 `[-180, 180)`。兩個端點是同一個
      方向，`+180` 折成 `-180`，半開區間的語意成立。
    - `SHOT_ANGLE = (-30, 30)`（Milestone A）：中心仍是 0，折回 `[-180, 180)`
      之後 `30` 就停在 `30`。兩個端點是**不同方向**，不該互相折回——用下界當錨
      的話會折成 `[-30, 330)`，`+30` 變 `+30` 沒錯，但 `-31` 會變成 `329`，
      離動作空間更遠而不是更近。

    ⚠️ 週期固定取 `360.0` 而**不是**區間寬度：收窄後寬度是 60，拿它當週期會把
    45° 折成 -15°，完全不同的方向。圓的週期永遠是 360。

    折回後**不保證**落在 `[low, high]` 內——區間收窄後有些物理角度根本無法表達。
    需要那個保證的呼叫端請用 `_canonical_shot_angle()`。
    """
    low, high = ACTION_BOUNDS[_SHOT_ANGLE]
    center = (low + high) / 2.0
    half_period = _ANGLE_PERIOD / 2.0
    return (angle - center + half_period) % _ANGLE_PERIOD - half_period + center


def _canonical_shot_angle(angle: float) -> float:
    """`_wrap_angle()` 加上「折完必須落在動作空間內」的檢查。

    `SHOT_ANGLE` 涵蓋整圈時這個檢查永遠通過（任何方向都表達得了）。Milestone A
    把它收窄到 ±30° 之後就不是了——90° 是合法的物理角度，但 policy 輸不出來。

    對這種情形拋 ValueError 而不是夾住：夾住等於把 90° 謊報成 30°，是不同的
    方向。Milestone B 把區間改回整圈時，任何殘留的假設會在這裡大聲失敗。
    """
    low, high = ACTION_BOUNDS[_SHOT_ANGLE]
    wrapped = _wrap_angle(angle)
    if not low <= wrapped <= high:
        raise ValueError(
            f"shot_angle {angle} 折算為 {wrapped}°，超出目前動作空間 "
            f"[{low}, {high}]（Milestone A 期間 SHOT_ANGLE 已收窄，見 #231）"
        )
    return wrapped


def decode_rl_action(
    raw_action: list[float],
    max_offset: float,
    should_execute_action: bool = True,
) -> Action:
    """將 policy 的 6 維正規化輸出還原為執行期 `Action`（物理域）。

    raw_action: 正規化域 `[-1, 1]` 的 6 維向量，欄位順序依 #110 —— 母球
        擺位 XY、擊球方向角、母球目標初速、上下／左右擊球偏移。
    max_offset: 可用偏移能力的比例，`[0.0, 1.0]`，與正規化域同一把尺。
    should_execute_action: 由 Controller 的狀態轉換產生，不是第 7 維模型
        輸出，因此不從 raw_action 取；多送一維會在長度檢查被擋下。

    處理順序固定為 clip → 圓形裁切 → 反正規化 → 組裝，**不可調換**，理由
    見以下各步驟註解。
    """
    if len(raw_action) != ACTION_DIM:
        raise ValueError(f"raw_action must have length {ACTION_DIM}")

    limit = validate_max_offset(max_offset)
    normalized = [
        validate_finite_number(value, f"raw_action[{index}]")
        for index, value in enumerate(raw_action)
    ]

    # 前四維各自獨立，逐軸夾回 [-1, 1] 即可。policy 的高斯取樣本來就可能
    # 溢出 tanh 的值域，這裡不擋就會被反正規化放大成越界的物理量。
    for index in (_PLACEMENT_X, _PLACEMENT_Y, _SHOT_ANGLE, _SPEED):
        normalized[index] = max(-1.0, min(1.0, normalized[index]))

    # 偏移兩維是一個向量，只能走圓形裁切，**不可逐軸 clip** —— 逐軸截斷會
    # 改變方向（[2.0, 0.01] 會被壓成 [1.0, 0.01]，方向整個歪掉），而擊球
    # 偏移的方向就是加旋方向，方向錯了球路就錯了（#222）。
    #
    # 裁切必須在反正規化「之前」：max_offset 與正規化域同尺，hypot(offset)
    # 才能直接與它比較。若改成先反正規化再裁，物理域的最大範數只有
    # hypot(0.5, 0.5) ≈ 0.707，max_offset ∈ (0.707, 1.0] 會整段變成死區，
    # policy 分不出 0.8 與 1.0。
    normalized[_OFFSET_V], normalized[_OFFSET_H] = clamp_position_offset(
        [normalized[_OFFSET_V], normalized[_OFFSET_H]], limit
    )

    physical = [
        _denormalize(value, index) for index, value in enumerate(normalized)
    ]

    return Action(
        cue_ball_placement=[physical[_PLACEMENT_X], physical[_PLACEMENT_Y]],
        # 物理域是半開區間 [-180, 180)，但 ACTION_BOUNDS 的 high 記為 180.0
        # （gymnasium 的 Box 只能表達閉區間）。這裡統一折回，否則 +1 會還原成
        # 180.0 —— 與 -180.0 同方向卻是不同數值，兩端各自處理就會在邊界
        # 靜默不一致。
        shot_angle=_wrap_angle(physical[_SHOT_ANGLE]),
        cue_ball_speed=physical[_SPEED],
        position_offset=[physical[_OFFSET_V], physical[_OFFSET_H]],
        should_execute_action=should_execute_action,
    )


def normalize_action(action: Action) -> list[float]:
    """`decode_rl_action` 的反向：把物理域 `Action` 換算回 6 維正規化向量。

    用於往返驗證、`action_space` 上下限對齊，以及把既有的物理動作（腳本
    開球、人工調參）換算成「policy 該輸出什麼」。

    本函式是公開介面，呼叫者不保證 `Action` 來自 `decode_rl_action`，因此
    驗證強度與 decode 對等；角度先折回最近的等價值，落不進動作空間就拋
    ValueError（見 `_canonical_shot_angle`），不會靜默算出越界的正規化值。
    """
    physical = [
        *validate_2d_value(action.cue_ball_placement, "cue_ball_placement"),
        _canonical_shot_angle(
            validate_finite_number(action.shot_angle, "shot_angle")
        ),
        validate_finite_number(action.cue_ball_speed, "cue_ball_speed"),
        *validate_2d_value(action.position_offset, "position_offset"),
    ]

    return [_normalize(value, index) for index, value in enumerate(physical)]


def _denormalize(value: float, index: int) -> float:
    """正規化域 `[-1, 1]` → 第 index 維的物理域。

    採中點式 `center + x * half_span` 而非端點式
    `low + (x + 1) * (high - low) / 2`：偏移兩維的 center 恰為 0，中點式
    退化成純等比縮放，圓形裁切保住的方向能精確帶到物理域；端點式有加減
    相消（x = 0.1 會算出 0.05000000000000004），兩軸誤差不對稱就會把圓
    壓歪。代價是端點不再位元精確，比較時用近似值。
    """
    low, high = ACTION_BOUNDS[index]
    center = (high + low) / 2.0
    half_span = (high - low) / 2.0
    return center + value * half_span


def _normalize(value: float, index: int) -> float:
    """第 index 維的物理域 → 正規化域 `[-1, 1]`，`_denormalize` 的反函式。"""
    low, high = ACTION_BOUNDS[index]
    center = (high + low) / 2.0
    half_span = (high - low) / 2.0
    return (value - center) / half_span
