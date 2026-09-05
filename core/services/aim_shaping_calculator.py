"""瞄準塑形（dense shaping）：母球在首次接觸之前對 1 號球的最近接近距離。

#124 第一輪訓練的失敗模式不是「訊號太少」而是「reward 地形是平的」：
沒碰到任何球、碰到錯球都是同一個 -1.5，policy 起點附近整片都一樣，唯一的
梯度是命中率約 4.5% 的那根尖刺，PPO 在平原上只會縮變異數、往取樣雜訊剛好
指的方向收斂（訓練數據見 docs/CHANGELOG.md）。這一項把平原變成指向球堆的
斜坡：越接近 1 號球給越多分，碰到就給滿分。

⚠️ 這是 shaping，不是目標。`AIM_REWARD_SCALE` 必須嚴格小於犯規階梯的最小
   級距（見下方 `_assert_scale_cannot_invert_the_ladder` 的說明），否則塑形
   會反轉排序，policy 學到「靠近但不要碰到」。
"""

import math

from ..models.action_bounds import CUE_BALL_PLACEMENT_X, CUE_BALL_PLACEMENT_Y
from ..models.table_ball_set import TableBallSet
from .break_foul_evaluator import (
    FIRST_CONTACT_FOUL_PENALTY,
    NO_CONTACT_FOUL_PENALTY,
)
from .break_shot_position_provider import BREAK_SHOT_POSITIONS


_ONE_BALL_ID = 1
_BALL_DIAMETER = 2.0 * TableBallSet.DEFAULT_BALL_RADIUS


def _max_initial_gap() -> float:
    """母球擺在**最遠的合法位置**時，與 1 號球的表面間距。

    這是「最近接近距離」在物理上的上界：母球一開始就在這個距離，之後只可能
    更近。用它當正規化的分母，塑形分數自然落在 [0, AIM_REWARD_SCALE]，不必
    另外夾。

    四個角落取最大值而不是寫死數字——`CUE_BALL_PLACEMENT_*` 或
    `BREAK_SHOT_POSITIONS` 任何一項改動，這裡跟著變。
    """
    one_ball = BREAK_SHOT_POSITIONS[_ONE_BALL_ID]
    corners = (
        (x, y) for x in CUE_BALL_PLACEMENT_X for y in CUE_BALL_PLACEMENT_Y
    )
    return max(math.dist(corner, one_ball) for corner in corners) - _BALL_DIAMETER


AIM_REFERENCE_GAP = _max_initial_gap()
"""塑形歸零的距離（m）。約 1.9148，來自 kitchen 最遠角落到 1 號球。"""

AIM_REWARD_SCALE = 0.4
"""碰到 1 號球（間距 0）時的塑形滿分。

為什麼是 0.4 而不是更大：犯規階梯的最小級距是
`NO_CONTACT_FOUL_PENALTY (-2.0)` 與 `FIRST_CONTACT_FOUL_PENALTY (-1.5)` 之間
的 0.5。塑形滿分若 >= 0.5，「沒碰到球但瞄很準」就會反超「碰到錯球」，policy
學到的是靠近而不是命中。0.4 保證**任何**塑形值都無法反轉排序。

為什麼不更小：30° 的瞄準誤差在 1.5875 m 上是 0.79 m 的側向偏差、10° 是
0.28 m，兩者的塑形差是 0.4 × (0.79 - 0.28) / 1.9148 ≈ 0.107——必須明顯高於
第一輪觀測到的 surrogate 雜訊底（±0.001 量級）才推得動 policy。
"""


def _assert_scale_cannot_invert_the_ladder() -> None:
    """import 時就檢查塑形不會反轉犯規排序。

    寫成 import-time 檢查而不是只靠單元測試：這兩個常數分屬不同模組，改動
    `break_foul_evaluator` 的罰分時很容易忘記回頭看這裡，而反轉之後訓練照跑、
    不報錯，只是學到錯的東西——正是這一項要修的那種缺陷。
    """
    smallest_rung = FIRST_CONTACT_FOUL_PENALTY - NO_CONTACT_FOUL_PENALTY
    if AIM_REWARD_SCALE >= smallest_rung:
        raise ValueError(
            f"AIM_REWARD_SCALE={AIM_REWARD_SCALE} 不小於犯規階梯的最小級距 "
            f"{smallest_rung}，塑形會反轉排序（沒碰到球但瞄很準 > 碰到錯球）"
        )


_assert_scale_cannot_invert_the_ladder()


def closest_approach_to_reward(closest_approach: float) -> float:
    """最近表面間距（m）→ 塑形分數 `[0, AIM_REWARD_SCALE]`。

    closest_approach: 母球球面與 1 號球球面在**首次接觸之前**的最小距離。
        0.0 = 碰到了；`math.inf` = 這一局沒有量到（訓練端的初值），視同沒靠近。

    線性遞減而不是指數：線性在整個 [0, 1.9148] 區間都有固定斜率，policy 在
    任何瞄準誤差下都拿得到方向資訊。指數衰減在遠端會壓成 0，而第一輪的 policy
    正好收斂在遠端——那裡沒有梯度就等於這一項沒做事。
    """
    if math.isnan(closest_approach):
        raise ValueError("closest_approach must not be NaN")
    if closest_approach < 0.0:
        raise ValueError("closest_approach must not be negative")
    if closest_approach >= AIM_REFERENCE_GAP:
        # `math.inf` 走這一條，不必額外判斷。
        return 0.0
    return AIM_REWARD_SCALE * (1.0 - closest_approach / AIM_REFERENCE_GAP)
