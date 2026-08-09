# Copyright (c) 2026 GuanPersonalDev
"""B-1：21 維 observation 的 torch 向量化實作。

`core/services/rl_observation_encoder.encode_rl_observation()` 是單筆、純 Python
的版本，Demo 端使用，也是本模組的 golden reference。訓練端不能直接呼叫它——
ObsTerm 每個 env step 都會被觸發，1024 環境跑 Python 迴圈是每步數十毫秒，
訓練跑不完。

兩份實作靠兩件事綁死（#121 B-1）：

1. 欄位順序共用 `core` 的 `RL_BALL_ORDER` 常數
2. `rl_task/tests/test_mdp_observations.py` 的對拍測試（隨機球位逐元素比對）

常數共用只防得住順序錯位，語意漂移要靠對拍。

⚠️ 對拍測試**不在 pre-commit 閘門內**——本機沒有 torch，hook 也寫死了
`pytest core/tests/`。改動本模組後必須手動在 pod 上跑一次，指令見該測試檔頂端。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.services.rl_observation_encoder import RL_BALL_ORDER

if TYPE_CHECKING:
    # 只在型別檢查時匯入。執行期不 import isaaclab，讓對拍測試不必啟動 Kit app。
    from isaaclab.envs import ManagerBasedRLEnv


# RL_BALL_ORDER 是 tuple（1,…,9,0），torch 的 advanced indexing 要 list。
# 在模組層級轉一次就好，但**不要**轉成 tensor——tensor 有 device，會跟
# `--device cpu` 打架；Python list 索引沒有這個問題，速度也一樣。
_BALL_INDEX = list(RL_BALL_ORDER)


def encode_ball_positions(
    ball_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    max_offset: float,
) -> torch.Tensor:
    """純張量版的 21 維編碼，對應 `core` 的 `encode_rl_observation()`。

    刻意不吃 `env`，只吃張量——這樣對拍測試可以直接餵造出來的資料，不必建場景，
    也不必啟動 Kit app。

    ball_pos_w: `(num_envs, 10, 3)` 球心世界座標，object 維度順序為 ball_0…ball_9
    env_origins: `(num_envs, 3)` 各子環境原點的世界座標
    max_offset: 可用偏移能力比例 `[0.0, 1.0]`，21 維的最後一格

    回傳 `(num_envs, 21)`：9 顆號碼球的 XY（1→9）+ 母球 XY + `max_offset`。
    """
    # ① 世界座標 → 桌台相對座標。
    #    [:, None, :2] 等同 env_origins.unsqueeze(1)：env_origins 是 (N, 3)，
    #    球位是 (N, 10, 2)，中間要補一維才廣播得到 10 顆球。忘了會直接報錯——
    #    這算幸運的，漏「減」這個動作本身則是靜默算錯（core 收到世界座標後
    #    不做範圍檢查，reward 照算垃圾）。
    #
    #    ⚠️ 這裡減的是 env_origins，core 減的是 table_position（桌台世界 XY）。
    #    兩者相等**只因為 A-1 的 table 沒有給 init_state**，桌台就落在子環境
    #    原點上。桌台之後若加上 offset，這一行要跟著改，而且不會報錯
    #    （見 billiard_rl_env_cfg.py 上方〈座標系與 env_origins 換算〉）。
    #
    #    z 完全不用：observation 是純 2D，core 那邊也只取 world_x / world_y。
    rel_xy = ball_pos_w[:, :, :2] - env_origins[:, None, :2]  # (N, 10, 2)

    # ② 重排成 RL 的球序：1~9 號球在前，母球（index 0）排最後。
    #    Collection 的 object 維度順序 = rigid_objects dict 的插入順序 =
    #    ball_0…ball_9（A-3 的 Body names 已確認），所以 collection index i
    #    就是 ball_i，RL_BALL_ORDER 可以直接當索引用。
    #    順序錯位不會報錯，只會讓 policy 學垃圾。
    ordered = rel_xy[:, _BALL_INDEX, :]  # (N, 10, 2)

    # ③ 攤平成 (N, 20)。用 reshape 不用 view——advanced indexing 的結果不保證
    #    記憶體連續，view 會直接丟例外。
    flat = ordered.reshape(ordered.shape[0], -1)  # (N, 20)

    # ④ 尾端接上 max_offset，補滿第 21 維。
    #    device / dtype 跟著 flat 走，不要寫死 cuda 或 float32。
    limit = torch.full(
        (flat.shape[0], 1), max_offset, device=flat.device, dtype=flat.dtype
    )
    return torch.cat((flat, limit), dim=-1)  # (N, 21)


def ball_positions(
    env: ManagerBasedRLEnv,
    max_offset: float,
    asset_name: str = "balls",
) -> torch.Tensor:
    """ObsTerm 的進入點：從場景取值後交給 `encode_ball_positions()`。

    只做「取值」這件事，換算全在 `encode_ball_positions()` 裡，這樣對拍測試
    涵蓋得到全部邏輯。

    `max_offset` 刻意不給預設值，強迫在 `ObservationsCfg` 明寫——它必須與 B-2
    的 ActionTerm 用同一個值（見 `billiard_rl_env_cfg.TRAINING_MAX_OFFSET`）。

    `asset_name` 用 `str` 而不是 `SceneEntityCfg`：後者要在執行期 import
    isaaclab，本模組就不再是「純 torch + core」，對拍測試會被迫拉進整個 Kit
    相依。這裡只需要一個 key，用不到 SceneEntityCfg 的 body / joint 篩選。
    """
    balls = env.scene[asset_name]

    # .torch 不可省：Isaac Lab 3.0 的 data property 回傳的是 warp array 的包裝
    # 物件，不是 tensor（A-3 實測）。忘了會在減法運算子上炸掉。
    #
    # ⚠️ 用 body_link_pos_w，不要用 object_pos_w——後者在 3.0 標記 deprecated、
    #    4.0 移除。
    return encode_ball_positions(
        balls.data.body_link_pos_w.torch, env.scene.env_origins, max_offset
    )
