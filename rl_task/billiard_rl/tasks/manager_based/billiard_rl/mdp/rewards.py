# Copyright (c) 2026 GuanPersonalDev
"""B-3b／B-3c：reward（#121 B-3，邏輯來源 #218/#219）。

拆成四個獨立 RewTerm 是為了讓 TensorBoard 自動分項（Manager-based 的好處，
見 #120 memo）。但 `core.services.reward_service.calculate_reward()` 的邏輯
**不是單純相加**——開球犯規重置時它只回傳罰分、忽略其他所有項；9 號球獎勵
還要 gate 在「沒有犯規」上。

所以這裡的做法是：`decompose_reward()` 把 `calculate_reward()` 拆成四個分量，
並由對拍測試保證**四者之和恆等於 `calculate_reward()`**。分項只是呈現，
數值權威仍在 core。

效能：`calculate_spread_score()` 是純 Python 凸包（Andrew's monotone chain），
不好向量化。但它只需要在**落定的 env** 上算，而落定是每局一次的稀疏事件，
所以走 Python 迴圈直接呼叫 core 的原函式——跟 B-2 一樣是「真正共用同一份」。
四個 term 共用同一份計算結果（每個 env step 只算一次，見 `_breakdown()`）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.models.break_foul_result import BreakFoulResult
from core.models.shot_result import ShotResult
from core.services.cue_ball_pocketed_penalty_calculator import (
    calculate_cue_ball_pocketed_penalty,
)
from core.services.break_foul_evaluator import evaluate_break_foul
from core.services.nine_ball_pocketed_bonus_calculator import (
    calculate_nine_ball_pocketed_bonus,
)
from core.services.pocket_geometry import POCKET_POSITIONS
from core.services.spread_score_calculator import (
    calculate_spread_score,
    spread_score_to_reward,
)

from .shot_tracking import CUE_BALL_INDEX
from .terminations import all_balls_at_rest, break_foul_decided

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_POCKET_XY: list[tuple[float, float]] = list(POCKET_POSITIONS.values())
_OBJECT_BALL_IDS = tuple(range(1, 10))
_NINE_BALL_ID = 9

_CACHE_ATTR = "_billiard_reward_breakdown"


def decompose_reward(
    shot_result: ShotResult, break_foul_result: BreakFoulResult
) -> dict[str, float]:
    """把 `calculate_reward()` 拆成四個分量，四者之和恆等於它。

    回傳 key：`spread` / `cue_scratch` / `foul` / `nine_ball`。

    對照 `reward_service.calculate_reward()` 的三段邏輯：

    1. `should_reset` 時只回傳罰分 → 其餘三項全部歸零，`foul` 帶罰分
    2. 否則四項相加
    3. 9 號球獎勵 gate 在「母球沒進袋且沒有犯規罰分」上

    拆解本身不做任何判斷，只是把同一組條件寫成分項；數值權威在 core。
    """
    if break_foul_result.should_reset:
        # calculate_reward() 在這個分支直接 return penalty，其他項連算都不算。
        return {
            "spread": 0.0,
            "cue_scratch": 0.0,
            "foul": break_foul_result.penalty,
            "nine_ball": 0.0,
        }

    cue_scratch = calculate_cue_ball_pocketed_penalty(shot_result.cue_ball_pocketed)
    has_foul = shot_result.cue_ball_pocketed or break_foul_result.penalty < 0.0
    nine_ball = (
        0.0
        if has_foul
        else calculate_nine_ball_pocketed_bonus(
            shot_result.nine_ball_pocketed, shot_result.cue_ball_pocketed
        )
    )
    return {
        # 與 calculate_reward() 一樣走 spread_score_to_reward()——這裡若直接放
        # 原始分數，四項之和就不再等於 core 的 reward，護欄測試會失效（#123）。
        "spread": spread_score_to_reward(shot_result.spread_score),
        "cue_scratch": cue_scratch,
        "foul": break_foul_result.penalty,
        "nine_ball": nine_ball,
    }


def evaluate_shot(
    ball_xy: list[tuple[float, float]],
    pocket_index: list[int],
    rail_contacted: list[bool],
    first_contact: int,
) -> dict[str, float]:
    """單一 env、單一局面 → 四個 reward 分量。

    ball_xy: 10 顆球的桌台相對 XY，索引 = ball_id
    pocket_index: 10 個袋口索引，-1 = 沒進袋
    rail_contacted: 10 個布林，該球是否碰過顆星
    first_contact: 母球第一顆碰到的球（ball_id），-1 = 整局沒碰到

    **進袋球代入袋口座標**——這是 `calculate_spread_score()` docstring 明文
    要求的呼叫端責任（「進袋球一樣要給一筆資料，value 用該球進的那個袋口座標
    代入」），也是訓練端不需要把球搬離檯面的原因。
    """
    pocketed_object_ball_ids = {
        ball_id for ball_id in _OBJECT_BALL_IDS if pocket_index[ball_id] >= 0
    }
    rail_contacted_object_ball_ids = {
        ball_id for ball_id in _OBJECT_BALL_IDS if rail_contacted[ball_id]
    }

    break_foul_result = evaluate_break_foul(
        # evaluate_break_foul 只接受 1~9 或 None；-1（沒碰到任何球）→ None，
        # 那會落進「首次接觸不是 1 號球」的分支判 -1.5 並重置，語意正確。
        first_contact if first_contact in _OBJECT_BALL_IDS else None,
        pocketed_object_ball_ids,
        rail_contacted_object_ball_ids,
    )
    shot_result = ShotResult(
        final_ball_positions=[list(xy) for xy in ball_xy],
        cue_ball_pocketed=pocket_index[CUE_BALL_INDEX] >= 0,
        nine_ball_pocketed=pocket_index[_NINE_BALL_ID] >= 0,
        spread_score=_spread_score(
            ball_xy, pocket_index, pocketed_object_ball_ids, break_foul_result
        ),
    )
    return decompose_reward(shot_result, break_foul_result)


def _spread_score(
    ball_xy: list[tuple[float, float]],
    pocket_index: list[int],
    pocketed_object_ball_ids: set[int],
    break_foul_result: BreakFoulResult,
) -> float:
    """散開分數，但 `should_reset` 時不算——那個分支的結果會被丟掉。

    `calculate_spread_score()` 是純 Python 的凸包（Andrew's monotone chain）加上
    72 次 `math.dist`，而 `decompose_reward()` 在 `should_reset` 分支直接把
    spread 歸零、根本不看這個值。

    這在 B-4 只有「落定才結算」的時候無關痛癢（犯規局照樣要等球停，一局也只
    算一次）。但 `break_foul_decided` 提前終止上線後，**犯規局變成第 1 步就
    結算**，而首次接觸不是 1 號球在訓練初期是壓倒性多數（瞄準容錯窗口只有
    ±2.06°）——等於每個 env step 都在 `_compute_breakdown()` 的逐 env Python
    迴圈裡算幾百個馬上被丟掉的凸包，把早停省下的物理時間吐回去。

    回傳 0.0 而不是 None：`ShotResult.spread_score` 是 float，而 0.0 也通過
    `reward_service._validate_spread_score()` 的 [0, 1] 檢查，萬一之後有人改成
    走 `calculate_reward()` 也不會炸。
    """
    if break_foul_result.should_reset:
        return 0.0

    positions: dict[int, tuple[float, float]] = {}
    for ball_id in _OBJECT_BALL_IDS:
        pocket = pocket_index[ball_id]
        positions[ball_id] = (
            _POCKET_XY[pocket] if pocket >= 0 else ball_xy[ball_id]
        )
    return calculate_spread_score(positions, pocketed_object_ball_ids)


def _breakdown(env: ManagerBasedRLEnv, action_term_name: str) -> dict[str, torch.Tensor]:
    """四個 term 共用的每步計算，快取在 env 上避免重複四次。

    RewardManager 會分別呼叫每個 term 的 func，但四者需要的是同一份局面分析
    （凸包、進袋集合、犯規判定）。以 `common_step_counter` 當快取鍵——它每個
    env step 遞增一次，而四個 term 都在同一個 step 內被呼叫。
    """
    step = int(env.common_step_counter)
    cached = getattr(env, _CACHE_ATTR, None)
    if cached is not None and cached[0] == step:
        return cached[1]

    value = _compute_breakdown(env, action_term_name)
    setattr(env, _CACHE_ATTR, (step, value))
    return value


def _compute_breakdown(
    env: ManagerBasedRLEnv, action_term_name: str
) -> dict[str, torch.Tensor]:
    """只對「這一步局面已定」的 env 計算 reward，其餘回 0。

    ⚠️ gate 在 `all_balls_at_rest` 上不是優化而是正確性：球還在飛的時候
    `calculate_spread_score()` 取到的是飛行途中的隨機構型，而「還在飛」與
    出桿力道正相關——不 gate 的話 policy 會學成「打到時限還沒停」。
    進袋判定更是直接錯的（正朝袋口飛去的球尚未進袋）。

    ⚠️ 但「局面已定」不只有落定一種：`break_foul_decided` 的 env 首次接觸已經
    確定且不是 1 號球，`calculate_reward()` 在那個分支只回傳 -1.5、其餘三項
    歸零，**完全不看球在哪裡**，所以球還在動也算得出正確答案。這一項必須跟
    `TerminationsCfg.break_foul` 同進同出：那邊提前終止、這邊不結算的話，
    -1.5 永遠不會被支付，policy 會學到「隨便亂打可以免費跳過這一局」。
    """
    device = env.device
    zeros = torch.zeros(env.num_envs, device=device)
    result = {
        "spread": zeros,
        "cue_scratch": zeros.clone(),
        "foul": zeros.clone(),
        "nine_ball": zeros.clone(),
    }

    settled = all_balls_at_rest(env) | break_foul_decided(env, action_term_name)
    settled_ids = settled.nonzero(as_tuple=False).flatten()
    if settled_ids.numel() == 0:
        return result

    strike_term = env.action_manager.get_term(action_term_name)
    balls = env.scene["balls"]

    # 一次搬到 CPU。落定是每局一次的稀疏事件，這個同步不在熱路徑上。
    ball_xy = (
        balls.data.body_link_pos_w.torch[settled_ids][..., :2]
        - env.scene.env_origins[settled_ids][:, None, :2]
    ).cpu().tolist()
    pocket_index = strike_term.pocket_index[settled_ids].cpu().tolist()
    rail_contacted = strike_term.rail_contacted[settled_ids].cpu().tolist()
    first_contact = strike_term.first_contact[settled_ids].cpu().tolist()

    for row, env_id in enumerate(settled_ids.tolist()):
        components = evaluate_shot(
            [tuple(xy) for xy in ball_xy[row]],
            pocket_index[row],
            rail_contacted[row],
            first_contact[row],
        )
        for name, value in components.items():
            result[name][env_id] = value

    return result


def spread(env: ManagerBasedRLEnv, action_term_name: str = "strike") -> torch.Tensor:
    """散開程度，只在落定的那一步給分。

    數值已經過 `spread_score_to_reward()` 重新正規化（rack = 0.0、RunPod
    控制式最大速度開球平均 = +1.0），**不是** `calculate_spread_score()` 的
    0~1 原始分數。RewTerm 的 weight 必須維持 1.0，理由見該函式的 docstring。
    """
    return _breakdown(env, action_term_name)["spread"]


def cue_scratch(env: ManagerBasedRLEnv, action_term_name: str = "strike") -> torch.Tensor:
    """母球落袋懲罰。"""
    return _breakdown(env, action_term_name)["cue_scratch"]


def foul(env: ManagerBasedRLEnv, action_term_name: str = "strike") -> torch.Tensor:
    """開球犯規罰分（首次接觸不是 1 號球 / 未達 4 顆球碰顆星）。"""
    return _breakdown(env, action_term_name)["foul"]


def nine_ball(env: ManagerBasedRLEnv, action_term_name: str = "strike") -> torch.Tensor:
    """9 號球進袋獎勵（母球進袋或犯規時歸零）。"""
    return _breakdown(env, action_term_name)["nine_ball"]
