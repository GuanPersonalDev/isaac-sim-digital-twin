# Copyright (c) 2026 GuanPersonalDev
"""B-5：episode 重置時套用開球擺位（#121 B-5）。

⚠️ Isaac Lab 的 `_reset_idx()` **不會**自己把 asset 寫回 default state——那件事
是 `mode="reset"` 的 EventTerm 做的。沒有本模組，第二個 episode 起球會維持在
上一局散開的位置，而且完全不報錯：訓練照跑、reward 照算，只是每一局的初始盤面
都不同，policy 學到的東西沒有意義。

（2026-08-09 pod 實測看得到這個現象：B-4 終止後母球被 B-2 擺回 kitchen，但另外
9 顆球留在原地。）

擺位直接取 `BREAK_SHOT_POSITIONS` 而不是讀 asset 的 default state：

- 完成標準寫的就是「套用 BREAK_SHOT_POSITIONS 固定擺位」，直接讀常數最貼近
- 不依賴 `default_object_state` / `default_body_state` 這批在 3.0 改名中的屬性
- A-1 的 `init_state` 也是從同一個 provider 來的，兩者不會漂移

評估場景必須完全固定（見 `training/README.md`〈Demo 素材〉）——這裡不加任何
隨機擾動。要做 domain randomization 的話另開 EventTerm，不要污染這一個。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.models.table_ball_set import TableBallSet
from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 依 ball_id 排序 → 與 collection 的 object 維度順序（ball_0…ball_9）一致。
# A-1 的 `_make_ball_cfgs()` 用的是同一個 `sorted()`，兩邊必須一致，否則
# reset 會把 3 號球擺到 7 號球的位置——不報錯，只是每一局都從錯的盤面開始。
_BREAK_SHOT_XY: list[tuple[float, float]] = [
    BREAK_SHOT_POSITIONS[ball_id] for ball_id in sorted(BREAK_SHOT_POSITIONS)
]


def break_shot_positions(
    env_origins: torch.Tensor,
    ball_radius: float = TableBallSet.DEFAULT_BALL_RADIUS,
) -> torch.Tensor:
    """`(E, 3)` 子環境原點 → `(E, 10, 3)` 開球擺位的世界座標。

    抽成純張量函式讓對拍測試不必建場景。

    桌台相對座標 → 世界座標要「加」env_origins（A-2 換算表第 2 列）。漏加的話
    所有環境的球會疊在世界原點——這個倒是看得出來，但只有在目視時。
    """
    xy = torch.tensor(
        _BREAK_SHOT_XY, device=env_origins.device, dtype=env_origins.dtype
    )  # (10, 2)

    positions = torch.empty(
        (env_origins.shape[0], xy.shape[0], 3),
        device=env_origins.device,
        dtype=env_origins.dtype,
    )
    positions[:, :, :2] = xy.unsqueeze(0) + env_origins[:, None, :2]
    # 桌面在子環境局部座標的 z=0，球心高度就是球半徑（與 A-1 的 init_state 一致）。
    positions[:, :, 2] = env_origins[:, 2:3] + ball_radius
    return positions


def reset_break_shot_layout(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_name: str = "balls",
    ball_radius: float = TableBallSet.DEFAULT_BALL_RADIUS,
) -> None:
    """把指定子環境的 10 顆球擺回開球位置並清空速度。

    `mode="reset"` 的 EventTerm 簽章固定是 `(env, env_ids, **params)`。
    """
    balls = env.scene[asset_name]

    # body_link_state_w 是 (num_envs, 10, 13)：pos(3) + quat(4) + lin(3) + ang(3)，
    # 前 7 格就是位姿。
    pose = balls.data.body_link_state_w.torch[env_ids][:, :, :7].clone()
    pose[:, :, :3] = break_shot_positions(env.scene.env_origins[env_ids], ball_radius)

    # 姿態（pose[:, :, 3:7]）沿用現值，不重設成 identity：Isaac Lab 3.0 的四元數
    # 分量順序是 (x, y, z, w)，與 2.x 相反，而寫入端的約定沒有實測確認過。
    # 球是均質球體，姿態對物理**完全沒有影響**——唯一差別是花色球的條紋朝向，
    # 純視覺。要做成逐局一致（#227 回放）的話，先確認寫入端的分量順序再改。

    velocity = torch.zeros(
        (pose.shape[0], pose.shape[1], 6), device=pose.device, dtype=pose.dtype
    )

    balls.write_body_link_pose_to_sim_index(body_poses=pose, env_ids=env_ids)
    # 速度必須清掉。只搬位置的話上一局的殘留速度會留著，球一放好就自己滑走。
    balls.write_body_com_velocity_to_sim_index(
        body_velocities=velocity, env_ids=env_ids
    )
