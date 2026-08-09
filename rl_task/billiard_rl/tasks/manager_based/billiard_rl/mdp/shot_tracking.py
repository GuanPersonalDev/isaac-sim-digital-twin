# Copyright (c) 2026 GuanPersonalDev
"""B-3a：一局之內的擊球事件偵測（#121 B-3a）。

`core.services.reward_service.calculate_reward()` 需要四樣輸入，其中三樣在
Demo 端來自 physx contact 事件（`PocketEventHandler`），而訂閱式回呼在
1024 env × 10 球的向量化環境用不了。訓練端全部改成位置判定：

    進袋       球心與某個袋口中心的距離 < POCKET_RADIUS
    首次接觸   母球與某顆號碼球的球心距離 < 2×球半徑
    顆星接觸   |x| >= 桌面半寬 − 球半徑，或 |y| >= 桌面半長 − 球半徑

⚠️ 三者都必須**整局黏著**，不能只看落定時的狀態：

- 袋口在模擬裡是 trigger 體積不是洞，球滾過去會繼續滾（見
  `core/services/pocket_geometry.py`）
- 首次接觸依定義就是歷史事件
- 顆星接觸是瞬間事件，落定時球早就離開顆星了

所以判定要每個 physics tick 跑一次、結果累積起來，掛在
`BilliardStrikeAction.apply_actions()`（manager-based 唯一的每 tick hook，
與 B-6 的滾動阻力並排）。全部是 `(N, 10)` 的距離比較，成本可忽略。
"""

from __future__ import annotations

import torch

from core.services.pocket_geometry import (
    POCKET_POSITIONS,
    POCKET_RADIUS,
    rail_limits,
)

# 依 POCKET_POSITIONS 的插入順序，與 asset_utility.POCKET_NAMES 一致。
_POCKET_XY: list[tuple[float, float]] = list(POCKET_POSITIONS.values())

CUE_BALL_INDEX = 0
"""母球在 collection 的 object 維度索引（= ball_0）。"""


def pocket_xy(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """`(6, 2)` 袋口中心的桌台相對座標。"""
    return torch.tensor(_POCKET_XY, device=device, dtype=dtype)


def detect_pocketed(ball_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """回傳 `(is_pocketed, nearest_pocket_index)`。

    ball_xy: `(N, B, 2)` 桌台相對座標

    - `is_pocketed` `(N, B)` bool：球心落在任一袋口半徑內
    - `nearest_pocket_index` `(N, B)` long：最近的袋口索引，給吸附用

    判定用球心而不是球面：袋口半徑 0.057 約為球半徑的 2 倍，球心進到這個範圍
    時球已經有一半以上在袋口上方，真實球檯此時就掉下去了。
    """
    pockets = pocket_xy(ball_xy.device, ball_xy.dtype)  # (6, 2)
    # (N, B, 1, 2) - (1, 1, 6, 2) -> (N, B, 6)
    distance = (ball_xy.unsqueeze(2) - pockets[None, None, :, :]).norm(dim=-1)
    nearest_distance, nearest_index = distance.min(dim=-1)
    return nearest_distance < POCKET_RADIUS, nearest_index


def detect_rail_contact(ball_xy: torch.Tensor, ball_radius: float) -> torch.Tensor:
    """`(N, B)` bool：球此刻是否貼著顆星。

    開球犯規判定要「至少 4 顆號碼球碰到顆星」（`evaluate_break_foul`），
    所以只需要布林事件，不需要知道碰的是哪一條。
    """
    x_limit, y_limit = rail_limits(ball_radius)
    return (ball_xy[..., 0].abs() >= x_limit) | (ball_xy[..., 1].abs() >= y_limit)


def detect_cue_contact(ball_xy: torch.Tensor, ball_radius: float) -> torch.Tensor:
    """`(N, B)` bool：此刻與母球接觸的球（母球自己恆為 False）。

    兩顆球接觸的定義是球心距離等於直徑。用 `<=` 並加一點餘裕——PhysX 的接觸
    解算會讓球稍微分開，剛好等於直徑的瞬間未必被任何一個 tick 取樣到。
    餘裕取 1% 球半徑，遠小於球徑，不會誤判相鄰但沒碰到的球。
    """
    cue_xy = ball_xy[:, CUE_BALL_INDEX : CUE_BALL_INDEX + 1, :]  # (N, 1, 2)
    distance = (ball_xy - cue_xy).norm(dim=-1)  # (N, B)
    touching = distance <= (2.0 * ball_radius) * 1.01
    touching[:, CUE_BALL_INDEX] = False
    return touching


def update_first_contact(
    first_contact: torch.Tensor, ball_xy: torch.Tensor, ball_radius: float
) -> torch.Tensor:
    """更新「母球第一顆碰到的號碼球」，`(N,)` long，-1 代表尚未接觸。

    只在還沒記錄過的 env 上寫入（黏著）。同一個 tick 若有多顆球同時接觸，
    取**最近的**那一顆——多球同時進入接觸範圍時，球心距離最小的就是先碰到的。
    """
    touching = detect_cue_contact(ball_xy, ball_radius)  # (N, B)
    if not bool(touching.any()):
        return first_contact

    cue_xy = ball_xy[:, CUE_BALL_INDEX : CUE_BALL_INDEX + 1, :]
    distance = (ball_xy - cue_xy).norm(dim=-1)
    # 沒接觸的球排到最後面，再取 argmin
    distance = torch.where(touching, distance, torch.full_like(distance, float("inf")))
    candidate = distance.argmin(dim=-1)  # (N,)

    unset = (first_contact < 0) & touching.any(dim=-1)
    return torch.where(unset, candidate, first_contact)
