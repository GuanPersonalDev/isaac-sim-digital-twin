"""袋口幾何：位置與判定半徑（桌台相對座標）。

Demo 端靠 `PocketEventHandler` 訂閱 physx contact 事件判定進袋，那條路在
向量化的 RL 環境用不了（1024 env × 10 球的逐 prim 回呼）。訓練端改用位置
判定，需要袋口座標——本模組就是那個單一來源，兩端共用。

數值來源：`assets/billiard_env.usda` 的 6 個 `Pocket_*` Cylinder（2026-08-09
從 USD 直接讀出），恰好等於桌台尺寸的半長半寬，因此這裡用 `TABLE_LENGTH` /
`TABLE_WIDTH` 推導而不是硬編一組新數字——桌台尺寸改了袋口會跟著走。

    Pocket_HeadLeft  (-0.635, -1.27)    Pocket_HeadRight  (+0.635, -1.27)
    Pocket_SideLeft  (-0.635,  0.00)    Pocket_SideRight  (+0.635,  0.00)
    Pocket_FootLeft  (-0.635, +1.27)    Pocket_FootRight  (+0.635, +1.27)

⚠️ 袋口在模擬裡是 **trigger 體積，不是洞**——球滾過去不會掉下去，會繼續滾。
   所以訓練端的進袋判定必須是「整局黏著」的：只要曾經進過袋就算，不能只看
   落定時的位置（見 #121 B-3a）。
"""

from .spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH

# USD 裡 Cylinder 的 radius。約等於 2 倍球徑（球徑 0.05715）。
POCKET_RADIUS = 0.057

# key 與 core/services/asset_utility.py 的 POCKET_NAMES 相同，順序也一致。
POCKET_POSITIONS: dict[str, tuple[float, float]] = {
    "Pocket_HeadLeft": (-TABLE_WIDTH / 2, -TABLE_LENGTH / 2),
    "Pocket_HeadRight": (TABLE_WIDTH / 2, -TABLE_LENGTH / 2),
    "Pocket_SideLeft": (-TABLE_WIDTH / 2, 0.0),
    "Pocket_SideRight": (TABLE_WIDTH / 2, 0.0),
    "Pocket_FootLeft": (-TABLE_WIDTH / 2, TABLE_LENGTH / 2),
    "Pocket_FootRight": (TABLE_WIDTH / 2, TABLE_LENGTH / 2),
}


def rail_limits(ball_radius: float) -> tuple[float, float]:
    """回傳 `(x_limit, y_limit)`：球心碰到顆星時的座標絕對值。

    球心離邊界一個球半徑時球面就貼上顆星了。x 的結果恰好等於
    `core/models/action_bounds.py` 的 `CUE_BALL_PLACEMENT_X` 上界——那個常數
    也是同一套推導（桌面半寬扣一顆球半徑），兩者一致是刻意的。
    """
    return (TABLE_WIDTH / 2 - ball_radius, TABLE_LENGTH / 2 - ball_radius)
