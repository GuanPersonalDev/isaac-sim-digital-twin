# Copyright (c) 2026 GuanPersonalDev
"""#123 步驟 5：只讓「真正擊球的那一步」產生 policy 梯度。

## 問題

開球是 one-shot：一局只擊一次，之後的 env step 都只是等球落定
（`mdp/actions.py` 的 `_apply_strike()` 對已擊球的 env 直接 return）。但 episode
長度 ≈ 落定秒數（6~12 步），PPO 照樣對那 5~11 個**沒有物理效果**的步計算
policy gradient。

兩種傷害，性質不同：

1. **surrogate**：`A_t = R - V(s_t)`，而動作對 R 無影響，所以 `E[A_t | s_t] = 0`。
   這是**變異數不是偏誤**，critic 收斂後會自我衰減——早期最傷。
2. **entropy bonus**：作用在每一步，而 90% 的步是無效步；Gaussian policy 的
   `std` 又是全域參數，等於設定的 `entropy_coef` 在真正的決策步上實際生效約
   10 倍。**這條不會自我衰減。**

## 做法（#123 的 Plan A）

只覆寫 `compute_returns()`，把無效步的 advantage 歸零。rsl_rl 的 surrogate 是
`-advantages * ratio`，advantage = 0 → 該樣本的 policy 梯度**精確為零**，不必碰
又長又常改版的 `update()`。

「這一步是不是 episode 的第一步」等價於「上一步的 `dones`」，而 `dones` 已經
傳進 `process_env_step()`，所以不需要在 RolloutStorage 上加欄位。

⚠️ **`surrogate_loss` 用的是 `.mean()`，分母沒有跟著縮。** 分子少了約 90%，
   `entropy` 與 `value_loss` 卻沒變，相對權重會整個歪掉。`rsl_rl_ppo_cfg.py`
   已把 `entropy_coef` 與 `value_loss_coef` 同步縮 10 倍補回來；改動這裡的
   遮罩邏輯時那兩個係數要一起重算。三項同縮後總 loss 也是 0.1 倍，但
   `schedule="adaptive"` 以 `desired_kl` 為目標會自動把 LR 拉回來，不用手調。

⚠️ **value_loss 刻意不遮。** 中間狀態的 return 是合法的監督訊號，多的樣本讓
   critic 收斂更快，而 critic 越準、上面第 1 點的殘留雜訊衰減越快。

## 上 pod 前必須確認的三件事

本機沒有 torch / rsl_rl（見 `mdp/observations.py` 開頭），以下三點是照 rsl_rl
main 分支寫的，**5.x 未必相同**：

1. `PPO.construct_algorithm()` 是否用 `cls(...)` 建構——若寫死 `PPO(...)`，
   `class_name` 指到本類別也不會生效（測試法：印 `type(runner.alg)`）。
2. `self.storage.advantages` 這個屬性名是否存在、形狀是否為 `(T, N, 1)`。
3. `compute_returns()` 內部是否已經做過 advantage 正規化——若否，本類別的
   重新正規化就是唯一那次，行為仍正確。

驗證方式：跑 5 個 iteration，印 `(advantages != 0).float().mean()`，應該
≈ `1 / 平均 episode 步數` ≈ 0.1。若是 1.0 代表遮罩根本沒生效。
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class MaskedPPO(PPO):
    """PPO，但只有 episode 的第一步（真正擊球的那一步）貢獻 policy 梯度。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 每個 rollout step 一筆 `(num_envs,)` bool，compute_returns() 時堆疊。
        # 用 list 而不是預先配置張量：不必知道 num_steps_per_env / num_envs，
        # 也就不會因為 storage 的屬性名在不同版本改掉而爆掉。
        self._valid_steps: list[torch.Tensor] = []
        # 下一個 step 是不是新 episode 的第一步。rollout 開始時 runner 已經
        # reset 過環境，所以初值是全 True。跨 iteration 要保留——rollout 邊界
        # 不等於 episode 邊界。
        self._next_is_first: torch.Tensor | None = None

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        dones_flat = dones.reshape(-1).bool()
        if self._next_is_first is None:
            self._next_is_first = torch.ones_like(dones_flat)

        self._valid_steps.append(self._next_is_first)
        # 這一步 done 的 env，下一步就是新 episode 的第一步。
        self._next_is_first = dones_flat.clone()

        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs) -> None:
        super().compute_returns(obs)

        advantages = self.storage.advantages
        valid = torch.stack(self._valid_steps).reshape(advantages.shape)
        self._valid_steps = []

        selected = advantages[valid]
        if selected.numel() < 2:
            # 整個 rollout 沒有任何 episode 起點（num_steps_per_env 遠小於
            # episode 長度時可能發生）。這時遮罩會把所有梯度歸零，寧可原樣
            # 放行也不要讓整個 iteration 空轉。
            return

        # 正規化統計量只用有效樣本算——父類別算的 mean/std 被 90% 的無效步
        # 稀釋，直接沿用會讓有效樣本的 advantage 尺度整個偏掉。
        self.storage.advantages = torch.where(
            valid,
            (advantages - selected.mean()) / (selected.std() + 1e-8),
            torch.zeros_like(advantages),
        )
