# Copyright (c) 2026 GuanPersonalDev
"""B-6：滾動阻力與自旋衰減的 torch 向量化實作（#121 B-6）。

Demo 端的 `core.services.rolling_resistance_service.RollingResistanceService`
掛在 `SimulationEvent.PHYSICS_POST_STEP`，每個 physics tick 對桌上每顆球施加
滾動摩擦與自旋衰減，取代 PhysX 的 `torsionalPatchRadius`。

訓練端不能重用它：1024 env × 10 球 = 每個 tick 一萬次 Python 呼叫，量級上不可能。
所以 torch 重寫，物理常數**全部 import 自 core**，一個都不重打。兩份實作由
`rl_task/tests/test_mdp_physics.py` 的四分支對拍測試綁死。

為什麼 B-6 是 B-4 的前置：`BallMotionMonitor.SPEED_THRESHOLD` 是 0.001 m/s，
靠 PhysX 自己的 damping 衰減是指數收斂，從 0.02 掉到 0.001 要等很久，而那個
damping 值我們沒量過。本模組的衰減是線性且會**硬夾停**——
`NEGLIGIBLE_SPEED_THRESHOLD`（0.02）遠大於 0.001，球一跌破就被寫成精確的 0，
下一個 tick 球靜止終止立刻成立。沒有 B-6，B-4 幾乎不會觸發。

⚠️ `PHYSICS_DT` 是 core 寫死的模組常數（1/60），不透過參數傳入。
   `BilliardRlEnvCfg.__post_init__` 的 `sim.dt` 必須維持 1/60，否則衰減量會
   算錯而且不會報錯。
"""

from __future__ import annotations

import torch

from core.services.rolling_resistance_service import (
    GRAVITY,
    NEGLIGIBLE_SPEED_THRESHOLD,
    NEGLIGIBLE_SPIN_THRESHOLD,
    PHYSICS_DT,
    ROLLING_FRICTION_COEFF,
    SETTLING_NOISE_CEILING,
    SPIN_DECAY_RATE,
)

# 除以速度模長時的下限。Python 版靠 `or` 短路，v_h == 0 時根本不會做除法；
# torch.where **沒有短路**，兩個分支都會先算出來。0/0 產生的 NaN 雖然在選取時
# 會被丟掉（forward 結果是對的），但會污染中間張量讓 isnan 診斷失效。
_EPS = 1e-12


def decay_velocities(
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    ball_radius: float,
    rolling_friction_coeff: float = ROLLING_FRICTION_COEFF,
    spin_decay_rate: float = SPIN_DECAY_RATE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """對一個 tick 的球速度施加滾動摩擦與自旋衰減。

    對應 `RollingResistanceService.apply()`，逐行語意相同。

    lin_vel: `(num_envs, num_balls, 3)` 線速度
    ang_vel: `(num_envs, num_balls, 3)` 角速度
    ball_radius: 球半徑（m）

    回傳 `(new_lin_vel, new_ang_vel, is_settling_noise)`：

    - 前兩者形狀同輸入
    - `is_settling_noise` 形狀 `(num_envs, num_balls)`，True 代表這顆球的水平
      線速度與殘留自旋都在 `SETTLING_NOISE_CEILING` 以下——**不是真的在滾動**，
      只是沉降/多球接觸解算的數值雜訊。呼叫端據此決定要不要寫入（見
      `BilliardStrikeAction._apply_rolling_resistance()` 的說明）。
    """
    delta_v = rolling_friction_coeff * GRAVITY * PHYSICS_DT
    delta_w = spin_decay_rate * PHYSICS_DT

    # 水平速度模長。垂直分量不參與滾動摩擦（球是被桌面撐著的）。
    v_h = lin_vel[..., :2].norm(dim=-1)  # (N, B)

    # n̂ × v / R，n̂=(0, 0, 1) → (-vy, vx) / R：由目前線速度反推出的「滾動」
    # 角速度分量。z 分量恆為 0，所以只算前兩軸。
    roll_w = (
        torch.stack((-lin_vel[..., 1], lin_vel[..., 0]), dim=-1) / ball_radius
    )  # (N, B, 2)

    # 其餘分量就是殘留自旋（側旋／english）——跟滾動摩擦是各自獨立的物理現象。
    residual = ang_vel.clone()
    residual[..., :2] -= roll_w
    residual_magnitude = residual.norm(dim=-1)  # (N, B)

    # 低於視覺門檻、或這個 tick 該扣的量已超過目前速度時直接夾到 0——
    # 不會反向，也不會被永遠留在門檻附近。
    at_rest_h = (v_h < NEGLIGIBLE_SPEED_THRESHOLD) | (delta_v >= v_h)
    at_rest_s = (residual_magnitude < NEGLIGIBLE_SPIN_THRESHOLD) | (
        delta_w >= residual_magnitude
    )

    zero = torch.zeros_like(v_h)
    linear_scale = torch.where(at_rest_h, zero, (v_h - delta_v) / v_h.clamp_min(_EPS))
    spin_scale = torch.where(
        at_rest_s,
        zero,
        (residual_magnitude - delta_w) / residual_magnitude.clamp_min(_EPS),
    )

    new_lin_vel = lin_vel.clone()
    new_lin_vel[..., :2] = lin_vel[..., :2] * linear_scale.unsqueeze(-1)
    # vz 原封不動傳遞，與 core 一致（球的垂直運動交給 PhysX 的接觸解算）。

    # 殘留自旋依比例衰減（方向不變），再加回衰減後的滾動分量。
    new_ang_vel = residual * spin_scale.unsqueeze(-1)
    new_ang_vel[..., :2] += roll_w * linear_scale.unsqueeze(-1)

    is_settling_noise = (v_h < SETTLING_NOISE_CEILING) & (
        residual_magnitude < SETTLING_NOISE_CEILING
    )

    return new_lin_vel, new_ang_vel, is_settling_noise
