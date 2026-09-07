"""手動擊球參數面板（#115）用的邊界值——跟 `action_bounds.py` 的 RL 動作空間
是**兩把不同的尺**，這是本檔存在的唯一理由。

| 項目 | 本檔常數 | 尺的性質 | 來源 |
|---|---|---|---|
| 母球擺位 XY | `CUE_BALL_PLACEMENT_X/Y` | 物理能力邊界（桌台幾何：桌寬/Kitchen 範圍） | 轉出 `action_bounds`，不重新定義 |
| 母球初速 | `CUE_BALL_SPEED` | 物理能力邊界（桿尖速度上限、純滾動下限） | 轉出 `action_bounds`，不重新定義 |
| 上下／左右偏移 | `POSITION_OFFSET_VERTICAL/HORIZONTAL` | 物理能力邊界（miscue limit 約 0.5R） | 轉出 `action_bounds`，不重新定義 |
| 擊球方向角 | `MANUAL_SHOT_ANGLE` | **本檔自己定義**，安全區而非物理極限 | 見下方推導 |

擺位/速度/偏移四項不管是手動面板還是 RL policy 出手，物理世界能不能接受
都是同一個答案，因此直接 `from .action_bounds import ...` 轉出，**不允許
重新寫一次數值**——那就是第二份實作，兩邊各自改一次就會漂移（#228 的教訓）。

角度不一樣。`action_bounds.SHOT_ANGLE = (-30.0, 30.0)` 是 **Milestone A
為了訓練信號密度收窄的 RL 動作空間**（#231），不是物理限制；手動面板不走
`decode_rl_action()`/`normalize_action()` 的正規化路徑，直接把使用者拖曳
的角度塞進 `Action.shot_angle`，因此完全不受這個收窄約束。

⚠️ 但手動面板也不能開放整圈 (-180, 180)。理由是 `Ur10eSwingStrategy.
execute_aim()` 每一擊都會把基座 `reposition()` 到：

    base = cue_ball − (CUE_STICK_GRIP_TO_TIP + _BASE_STANDOFF_M) × d
         = cue_ball − 2.15 × (−sinθ, cosθ)

母球在開球點 `(0, −0.9525)` 時，解出來要求基座落在桌台外才不會跟球台重疊，
算出來是 `|θ| < 162.8°`；超過這個角度基座會被 teleport 進球桌中央，狀態機
進 `ERROR` 且不會自動復原。`MANUAL_SHOT_ANGLE` 取 ±160° 這個安全值，把
「基座 teleport 進球桌」這整類錯誤從源頭排除，`manual_shot_feasibility.py`
（階段 5）因此只需要處理「幾何無解」一種情況，不用再判斷這一種。

Milestone B 把 `action_bounds.SHOT_ANGLE` 改回整圈（#232）之後，本檔的角度
項應該退化為直接轉出 `SHOT_ANGLE`（走位球需要瞄準任意方向），不再需要獨立
的 `MANUAL_SHOT_ANGLE` 常數——但即使那時候，±160° 的基座碰撞限制依然存在，
屆時要保留的是這條安全區邏輯本身，不是現在的數值。
"""

from .action_bounds import (
    CUE_BALL_PLACEMENT_X,
    CUE_BALL_PLACEMENT_Y,
    CUE_BALL_SPEED,
    POSITION_OFFSET_HORIZONTAL,
    POSITION_OFFSET_VERTICAL,
)

__all__ = [
    "CUE_BALL_PLACEMENT_X",
    "CUE_BALL_PLACEMENT_Y",
    "CUE_BALL_SPEED",
    "POSITION_OFFSET_HORIZONTAL",
    "POSITION_OFFSET_VERTICAL",
    "MANUAL_SHOT_ANGLE",
]

# ±160° 而非 ±180°：162.8° 是 Ur10eSwingStrategy 基座不撞進球桌的解析上界，
# 留 2.8° 安全餘裕。推導見本檔案級 docstring。
MANUAL_SHOT_ANGLE = (-160.0, 160.0)
