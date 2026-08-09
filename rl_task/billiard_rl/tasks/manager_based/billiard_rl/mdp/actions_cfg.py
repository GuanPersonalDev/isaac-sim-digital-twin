# Copyright (c) 2026 GuanPersonalDev
"""B-2 ActionTerm 的設定類別（#121 B-2）。

與實作分兩個檔案是 Isaac Lab 的慣例：cfg 要能在不建構 term 的情況下被 import
（Hydra override、IO descriptors 匯出都會單獨讀 cfg）。
"""

from __future__ import annotations

from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass

from core.models.table_ball_set import TableBallSet

from .actions import BilliardStrikeAction


@configclass
class BilliardStrikeActionCfg(ActionTermCfg):
    """母球衝量式擊球。"""

    class_type: type = BilliardStrikeAction

    asset_name: str = "balls"
    """場景實體名稱，對應 `BilliardRlSceneCfg.balls`（RigidObjectCollection）。"""

    cue_ball_name: str = "ball_0"
    """母球在 collection 裡的 key，對應 `_make_ball_cfgs()` 產生的 `ball_{id}`。"""

    max_offset_range: tuple[float, float] = (0.0, 1.0)
    """`max_offset` 的取樣範圍 `(low, high)`，兩端都必須落在 `[0.0, 1.0]`。

    `max_offset` 是可用偏移能力比例，同時是 `decode_rl_action()` 圓形裁切的
    半徑與 21 維 observation 的最後一格（#222）。**每個 episode 每個 env 重新
    取樣一次**（`BilliardStrikeAction.reset()`），整局固定。

    ⚠️ 它是**條件變數**不是超參數——policy 要學的是「給定這個偏移能力上限，
    該怎麼打」。取樣範圍塌成單點（例如 `(1.0, 1.0)`）的話第 21 維在整個訓練
    期間都是常數，policy 學不到任何條件依賴，#180 量到手臂實際能力後接上去
    就是分布外輸入——不會報錯，只會打不準。

    退化成定值是刻意保留的用法，但只用於評估與 debug：
    `(0.6, 0.6)` ＝ 固定在該能力下重現行為。訓練請維持完整的 `(0.0, 1.0)`。

    B-1 的 ObsTerm 不再有自己的 `max_offset` 參數——它從本 term 的
    `max_offset` property 讀同一份 buffer，兩端不可能不一致（見 `actions.py`）。
    """

    ball_radius: float = TableBallSet.DEFAULT_BALL_RADIUS
    """球半徑（m）。決定母球擺位的 z 與 `compute_cue_ball_velocities()` 的加旋換算。

    球的擺位（A-1）、USD 資產本身、這裡三處必須是同一個值，所以直接取 core 的常數。
    """

    spin_efficiency: float = 0.8
    """加旋效率，`compute_cue_ball_velocities()` 的參數。

    值取自該函式的預設值——Demo 端建構 `ImpulseStrikingService` 時沒有傳這個參數
    （`billiard_digital_twin.py:177`），吃的就是預設。訓練端與 Demo 端的物理必須
    一致，所以這裡也維持 0.8，**不要**為了調參而改（`training/README.md`：
    物理參數不得在訓練期間變動）。
    """
