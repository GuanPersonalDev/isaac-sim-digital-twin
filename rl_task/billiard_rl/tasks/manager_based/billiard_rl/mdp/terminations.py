# Copyright (c) 2026 GuanPersonalDev
"""B-4：球靜止提前終止（#121 B-4，設計來源 #178）。

撞球開球是 one-shot 問題——policy 一局只擊一次，之後就是等球停下來結算。
所以終止條件不是「時間到」而是「局面已定」：

- `balls_at_rest`：10 顆球全部靜止 **且** 這一局已經擊過球 → 局面已定，
  回報是完整的，**不 bootstrap**
- `time_out`：`episode_length_s` 用完了球還在動 → 我們不知道結果，
  **交給 value function bootstrap**（這正是 `time_out=True` 的語意）

兩者都需要，而且不能把 `balls_at_rest` 標成 `time_out`——那會讓已經確定的
回報被 value function 覆蓋掉。

為什麼不能只留 `time_out`：球沒停就結算 reward 是**系統性偏誤**。
`calculate_spread_score` 會取到飛行途中的隨機構型，而「還在飛」與出桿力道正
相關，policy 會學成「打到時限還沒停」而不是「打出好的散開」；B-3 的
`nine_ball` / `cue_scratch` 更是直接判錯——正朝袋口飛去的球尚未進袋。

⚠️ 依賴 B-6。`BallMotionMonitor.SPEED_THRESHOLD` 是 0.001 m/s，靠 PhysX 自己的
   damping 衰減是指數收斂，要等很久；B-6 的衰減是線性且會硬夾停
   （`NEGLIGIBLE_SPEED_THRESHOLD` = 0.02 遠大於 0.001），球一跌破就被寫成精確
   的 0。2026-08-09 pod 實測：中等力道開球第 6~7 秒最大球速降到 0.00000。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.services.ball_motion_monitor import BallMotionMonitor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def balls_at_rest_mask(lin_vel: torch.Tensor, struck: torch.Tensor) -> torch.Tensor:
    """純張量版的判定，`(num_envs,)` bool。

    lin_vel: `(num_envs, num_balls, 3)` 線速度
    struck: `(num_envs,)` 該 env 這一局是否已經擊過球

    門檻直接取 `BallMotionMonitor.SPEED_THRESHOLD`，不得另寫數字——那是訓練端
    與 Demo 端「球停了沒」的同一把尺，不一致的話 Demo 會在訓練認定已落定的
    狀態上繼續等。

    只檢查線速度，與 `BallMotionMonitor.is_any_ball_moving()` 現有行為一致：
    原地自旋（線速度為 0）視為靜止。這在真實桌面上不完全成立（側旋會把球重新
    推動），但那是 `core` 既有的定義，訓練端先對齊、不單方面「修正」；而 B-6
    上線後殘留自旋會被主動衰減，這個缺口實務上自己消失。

    比較用平方避免開根號。`SPEED_THRESHOLD` 是正數，平方保序。
    """
    speed_sq = (lin_vel**2).sum(dim=-1)  # (N, B)
    at_rest = (speed_sq < BallMotionMonitor.SPEED_THRESHOLD**2).all(dim=1)
    # ⚠️ 必須 AND 上 struck。開球前所有球本來就是靜止的（A-1 的固定擺位），
    #    少了這一項會在第一次評估時就判定終止，episode 長度變成 1 步、
    #    reward 是開球擺位本身的分數——訓練跑得動但完全學不到東西。
    return at_rest & struck


def break_foul_decided_mask(
    first_contact: torch.Tensor, struck: torch.Tensor
) -> torch.Tensor:
    """`(num_envs,)` bool：這一局的開球犯規結果**已經確定**，不必再等球停。

    first_contact: `(num_envs,)` long，母球第一顆碰到的球，-1 = 目前還沒碰到
    struck: `(num_envs,)` 該 env 這一局是否已經擊過球

    `evaluate_break_foul()` 在 `first_contacted_ball_id != 1` 這個分支直接
    short-circuit 回傳 `(-1.5, should_reset=True)`，**完全不看進袋與顆星**，
    而 `calculate_reward()` 在 `should_reset` 時只回傳罰分、其餘三項歸零。所以
    只要首次接觸已經確定且不是 1 號球，整局的 reward 就已經定了——繼續模擬
    5~10 秒只是把算力花在等一個已知的結果停下來，而「首次接觸不是 1 號球」
    在早期訓練是最常見的結果。

    ⚠️ `first_contact == -1`（整局沒碰到任何球）**不能**放進來：那是「還沒碰到」
       與「永遠不會碰到」的同一個值，要等球全停才分得出來，交給
       `all_balls_at_rest()`。
    """
    return struck & (first_contact > 0) & (first_contact != 1)


def break_foul_decided(
    env: ManagerBasedRLEnv,
    action_term_name: str = "strike",
) -> torch.Tensor:
    """DoneTerm 進入點：開球犯規已確定 → 提前終止。

    ⚠️ 這個 term 與 `mdp/rewards.py` 的結算 gate 是**綁在一起的**：
       `_compute_breakdown()` 必須同樣接受 `break_foul_decided` 的 env，否則
       episode 提前結束但 reward 從未結算，那 -1.5 永遠不會被支付——policy 會
       學到「隨便亂打可以免費跳過這一局」，比不做早停還糟。
    """
    strike_term = env.action_manager.get_term(action_term_name)
    return break_foul_decided_mask(strike_term.first_contact, strike_term.struck)


def all_balls_at_rest(
    env: ManagerBasedRLEnv,
    action_term_name: str = "strike",
    asset_name: str = "balls",
) -> torch.Tensor:
    """DoneTerm 進入點：取值後交給 `balls_at_rest_mask()`。

    `struck` 來自 B-2 的 ActionTerm——它才知道母球有沒有被賦速。
    （若 `ActionManager.get_term()` 不存在，退路是 `env.action_manager._terms[name]`，
    但那是私有屬性，優先用公開 API。）
    """
    balls = env.scene[asset_name]
    strike_term = env.action_manager.get_term(action_term_name)
    return balls_at_rest_mask(balls.data.body_com_lin_vel_w.torch, strike_term.struck)
