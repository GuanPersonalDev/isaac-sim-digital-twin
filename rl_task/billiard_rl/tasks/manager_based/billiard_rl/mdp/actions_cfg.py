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

    max_offset: float = 1.0
    """可用偏移能力比例，`decode_rl_action()` 圓形裁切的半徑。

    ⚠️ 必須與 B-1 ObsTerm 的同名參數一致——那是 policy 看到的第 21 維條件值。
    不一致的話 policy 會學到一個「以為自己有 1.0 偏移能力、其實只有 0.6」的
    策略，完全不報錯。`BilliardRlEnvCfg` 兩處都用 `TRAINING_MAX_OFFSET`，
    這裡的預設值只是讓本類別單獨拿出來用時仍有合理行為。
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
