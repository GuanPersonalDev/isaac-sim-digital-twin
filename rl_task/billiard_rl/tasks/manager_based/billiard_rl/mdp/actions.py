# Copyright (c) 2026 GuanPersonalDev
"""B-2：母球衝量式擊球的 ActionTerm（#121 B-2）。

與 B-1 相反，本模組**直接呼叫 core 的單筆函式**（`decode_rl_action()` 與
`compute_cue_ball_velocities()`），沒有 torch 重寫。這個不對稱是刻意的：

- B-1 的 ObsTerm 每個 env step 都要算全部 num_envs 筆 → 非向量化不可
- B-2 的 decode 只在「該 env 尚未擊球」時才跑 → 每 episode 每 env 一次

撞球一局只擊一次。`process_actions()` 每個 env step 都會被呼叫，但 `_struck`
旗標讓 decode 只在第一次真的執行；之後 `apply_actions()` 直接 return。
所以這裡是「真正共用同一份」，不需要對拍測試。

⚠️ 訓練環境沒有手臂（本 issue 的決定），母球是直接被賦予速度而不是被球桿撞擊。
   手臂的 RL 是之後的延伸（Milestone B / #180）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.action_manager import ActionTerm

from core.models.action_bounds import ACTION_DIM
from core.services.impulse_striking_service import compute_cue_ball_velocities
from core.services.numeric_validation import validate_max_offset
from core.services.rl_action_decoder import decode_rl_action

from .physics import decay_velocities
from .shot_tracking import (
    detect_pocketed,
    detect_rail_contact,
    update_closest_approach,
    update_first_contact,
)

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv

    from .actions_cfg import BilliardStrikeActionCfg


# ⚠️ RigidObjectCollection 的寫入 API 在 Isaac Lab 3.0 有**三層**並存
#    （2026-08-09 於 pod 上實測 dir() + inspect.signature 確認）：
#
#      write_object_*_to_sim         最舊，object_* 系列，4.0 移除
#      write_body_*_to_sim           docstring 明寫 "Deprecated, same as ..._index"
#      write_body_*_to_sim_index     ← 現行版，本模組使用
#
#    現行版是**關鍵字專用**（簽章有 `*`），且參數名是 body_ids 不是 object_ids：
#
#      write_body_link_pose_to_sim_index(*, body_poses, body_ids=None, env_ids=None)
#      write_body_com_velocity_to_sim_index(*, body_velocities, body_ids=None, env_ids=None)
#
#    形狀都是 (len(env_ids), len(body_ids), ...)，位姿 7、速度 6。
#    讀取端同理：用 body_link_pos_w / body_link_state_w，不要用 object_pos_w。


class BilliardStrikeAction(ActionTerm):
    """把 policy 的 6 維正規化輸出還原成母球的擺位與初速度。

    動作語意（欄位順序見 `core/models/action_bounds.py`）：
      0-1 母球擺位 XY（桌台相對座標，限制在 kitchen 內）
      2   擊球方向角
      3   母球目標初速
      4-5 上下／左右擊球偏移（球半徑比例，決定加旋方向）
    """

    cfg: BilliardStrikeActionCfg
    _asset: RigidObjectCollection

    def __init__(self, cfg: BilliardStrikeActionCfg, env: ManagerBasedRLEnv) -> None:
        # 基底類別會把 env.scene[cfg.asset_name] 綁到 self._asset，
        # 並提供 self.num_envs / self.device。
        super().__init__(cfg, env)

        # policy 的原始輸出。值域**不保證**在 [-1, 1]——高斯取樣會溢出 tanh 的
        # 值域，clip 由 decode_rl_action() 內部負責。
        self._raw_actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)

        # 「這個 env 這一局已經擊過球了」。撞球一局只擊一次，這個旗標是
        # 「一次性擊球」的實作，同時也是 B-4 球靜止終止的必要輸入——沒有它，
        # 開球前就會被判定為靜止，episode 長度變成 1 步。
        self._struck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 條件變數 max_offset 的**唯一權威 buffer**（#122）。
        #
        # 每個 episode 每個 env 重新取樣（見 reset()），整局固定。B-1 的
        # ObsTerm 不自己存一份，而是透過 `max_offset` property 讀這個 tensor——
        # 只有一份 buffer，兩端就不可能不一致。原本兩端各自引用模組常數
        # TRAINING_MAX_OFFSET 的做法在改成隨機之後不管用了：兩邊各取樣一次
        # 會得到不同的值，而 policy 看到的條件值與實際生效的裁切半徑不同是
        # **完全不報錯**的錯誤。
        #
        # 範圍在建構時就驗，壞值不會拖到訓練跑完才發現。validate_max_offset()
        # 檢查 [0.0, 1.0]，額外再檢查 low <= high——low > high 會讓下面的
        # 取樣公式安靜產出負的區間長度，torch.rand 不會抱怨。
        low, high = cfg.max_offset_range
        self._offset_low = validate_max_offset(low)
        self._offset_high = validate_max_offset(high)
        if self._offset_low > self._offset_high:
            raise ValueError(
                f"max_offset_range 的下界不得大於上界，收到 {cfg.max_offset_range}"
            )
        self._max_offset = torch.zeros(self.num_envs, device=self.device)
        # 建構時先取樣一次。ManagerBasedEnv.reset() 會再取樣一遍，但在那之前
        # ObsTerm 就可能被讀到（例如 D-2 的 space 檢查），全 0 是合法值卻不是
        # 我們要的語意——「偏移能力為零」而不是「尚未取樣」。
        self._resample_max_offset(slice(None))

        # 用名字查 index，不要寫死 0。_make_ball_cfgs() 的 sorted() 保證了順序，
        # 但那是 A-1 的實作細節，本模組不該依賴它。
        # body_names 是現行名稱；object_names 仍在但屬 deprecated 系列。
        body_names = list(self._asset.body_names)
        if cfg.cue_ball_name not in body_names:
            raise ValueError(
                f"collection 裡沒有母球 '{cfg.cue_ball_name}'，實際有：{body_names}"
            )
        self._cue_index = body_names.index(cfg.cue_ball_name)

        # B-3a 的三個「整局黏著」事件記錄。三者都是**歷史事件**，落定時的狀態
        # 看不出來（袋口是 trigger 不是洞、球滾過去會繼續滾；顆星接觸是瞬間；
        # 首次接觸依定義就是歷史），所以每個 physics tick 更新一次並累積。
        num_balls = len(body_names)
        # 每顆球進了哪個袋（袋口索引），-1 = 沒進袋。存索引而不只是布林，是因為
        # calculate_spread_score() 要求呼叫端替進袋球代入**袋口座標**。
        self._pocket_index = torch.full(
            (self.num_envs, num_balls), -1, dtype=torch.long, device=self.device
        )
        self._rail_contacted = torch.zeros(
            (self.num_envs, num_balls), dtype=torch.bool, device=self.device
        )
        # 母球第一顆碰到的球（collection 索引），-1 = 尚未接觸。
        self._first_contact = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # #124 dense shaping：母球對 1 號球的最近表面間距（m），首次接觸前才更新。
        # 初值 inf = 這一局還沒量到，`closest_approach_to_reward()` 對它回 0 分。
        self._closest_approach = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """6。取自 core 的單一來源，不寫死數字。"""
        return ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """本 term 沒有獨立的「處理後張量」形式。

        decode 的產物是 core 的 `Action` dataclass（物理域），不是張量，而且
        只在擊球那一個 tick 存在。ActionManager 只拿這個 property 做 logging，
        回傳 raw 不影響行為。
        """
        return self._raw_actions

    @property
    def struck(self) -> torch.Tensor:
        """`(num_envs,)` bool：該 env 這一局是否已經擊過球。

        給 B-4 的球靜止終止用——必須 AND 上這個，否則開球前（球還沒被賦速時）
        就會判定所有球靜止而立刻終止。
        """
        return self._struck

    @property
    def max_offset(self) -> torch.Tensor:
        """`(num_envs,)` float：該 env 這一局的可用偏移能力比例 `[0, 1]`（#122）。

        B-1 的 ObsTerm 讀這個當 21 維的最後一格，本 term 自己拿它當
        `decode_rl_action()` 的圓形裁切半徑——**同一份 buffer**，這是兩端一致
        的唯一保證。回傳的是內部 tensor 不是副本，呼叫端不要就地改寫。
        """
        return self._max_offset

    @property
    def pocket_index(self) -> torch.Tensor:
        """`(num_envs, num_balls)` long：該球進了哪個袋，-1 = 沒進袋（B-3a）。"""
        return self._pocket_index

    @property
    def pocketed(self) -> torch.Tensor:
        """`(num_envs, num_balls)` bool：該球這一局是否曾經進袋。"""
        return self._pocket_index >= 0

    @property
    def rail_contacted(self) -> torch.Tensor:
        """`(num_envs, num_balls)` bool：該球這一局是否碰過顆星。"""
        return self._rail_contacted

    @property
    def first_contact(self) -> torch.Tensor:
        """`(num_envs,)` long：母球第一顆碰到的球，-1 = 整局沒碰到任何球。

        `evaluate_break_foul()` 要求首次接觸必須是 1 號球，碰到錯球判 -1.5 並
        重置；-1（整局沒碰到）判 -2.0，比碰到錯球更差（#124）。
        """
        return self._first_contact

    @property
    def closest_approach(self) -> torch.Tensor:
        """`(num_envs,)` float：母球對 1 號球的最近表面間距（m），首次接觸前的最小值。

        0.0 = 碰到了，`inf` = 這一局還沒量到。dense shaping 的唯一輸入，
        換算見 `core.services.aim_shaping_calculator.closest_approach_to_reward()`。
        """
        return self._closest_approach

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        """每個 env step 呼叫一次。只存不解。

        解碼延到 `apply_actions()` 的第一個 tick，因為那裡才知道哪些 env 真的
        需要（`_struck` 為 False 的那些）。這是「直接呼叫 core 單筆函式」在
        效能上站得住腳的原因。
        """
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        """每個 physics tick 呼叫（decimation 次）。

        兩件事，順序不可調換——**先衰減、後擊球**，否則剛寫入的母球初速會在
        同一個 tick 就被扣掉一次滾動摩擦。

        `apply_actions()` 是 manager-based 唯一每個 physics tick 觸發的 hook，
        所以滾動阻力（B-6）只能放這裡。語意上它不是「動作」，掛在 ActionTerm
        裡是框架限制而非設計選擇；`mode="interval"` 的 EventTerm 是每個 env
        step 觸發，粒度差 60 倍。
        """
        self._update_shot_tracking()
        self._apply_rolling_resistance()
        self._apply_strike()

    def _apply_strike(self) -> None:
        """把尚未擊球的 env 的母球賦予擺位與初速度。

        擊完之後整個 episode 剩下的 tick 都是直接 return，讓物理自己跑。
        """
        pending = (~self._struck).nonzero(as_tuple=False).flatten()
        if pending.numel() == 0:
            return

        # 只把需要的那幾筆搬到 CPU。第一個 tick 是全部 num_envs 筆，
        # 之後就是空的，所以這個同步每個 episode 只發生一次。
        raw_rows = self._raw_actions[pending].cpu().tolist()
        # 逐 env 的裁切半徑。不可退回用單一 cfg 值——每個 env 這一局的
        # max_offset 都不同（#122），拿錯的半徑去裁切，policy 看到的條件
        # 與實際生效的偏移就對不上，而且不報錯。
        offsets = self._max_offset[pending].cpu().tolist()

        placements: list[list[float]] = []
        velocities: list[list[float]] = []
        for row, offset in zip(raw_rows, offsets):
            # should_execute_action 寫死 True：它是 Demo 端狀態機的控制旗標
            # （ScriptController 的 IDLE→AIMING→STRIKING），**不是第 7 維模型
            # 輸出**——多送一維會在 decode 的長度檢查被擋下。訓練端只在真的要
            # 擊球時才走到這裡，所以恆為 True。
            action = decode_rl_action(row, offset, should_execute_action=True)
            linear, angular = compute_cue_ball_velocities(
                action, self.cfg.ball_radius, self.cfg.spin_efficiency
            )
            placements.append(action.cue_ball_placement)
            velocities.append(linear + angular)

        self._strike(pending, placements, velocities)
        self._struck[pending] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """episode 重置：清掉擊球旗標並重新取樣條件變數。

        沒有清 `_struck` 的話 env 重置後它仍為 True，母球永遠不會再被賦速，
        訓練從第二局起全部是靜止畫面——而且不會報錯。

        時序（`manager_based_rl_env.py`）：`_reset_idx()` 在
        `action_manager.reset()` 這一步呼叫本方法，而 observation 是在 `step()`
        的尾端才 compute，**在 `_reset_idx()` 之後**。所以這裡取樣的新值會出現
        在該 episode 的第一筆 observation 上，policy 從第一步就看得到自己這局
        的偏移能力。順序反過來的話 policy 每一局看到的都是上一局的條件值。
        """
        # env_ids=None 代表全部。不要就地改寫 env_ids——型別會從 Sequence[int]
        # 變成 slice，靜態檢查會擋。
        index: Sequence[int] | slice = slice(None) if env_ids is None else env_ids
        self._struck[index] = False
        self._raw_actions[index] = 0.0
        # B-3a 的事件記錄同樣是 per-episode 的，不清會把上一局的進袋、顆星接觸
        # 帶到下一局，reward 全部算錯而且不報錯。
        self._pocket_index[index] = -1
        self._rail_contacted[index] = False
        self._first_contact[index] = -1
        # 不清的話上一局命中的 0.0 會留到下一局，等於白送 dense shaping 滿分。
        self._closest_approach[index] = float("inf")
        # 條件變數重新取樣。這是 #122 的核心——第 21 維必須逐局變動，policy
        # 才學得到「偏移能力上限 → 該怎麼打」的條件依賴。
        self._resample_max_offset(index)

    """
    Internal helpers.
    """

    def _resample_max_offset(self, index: Sequence[int] | slice) -> None:
        """對指定的 env 重新取樣 `max_offset`，均勻分布於 `max_offset_range`。

        `torch.rand_like(self._max_offset[index])` 而不是自己算長度：
        `index` 可能是 slice（全部）也可能是索引張量（部分 env），兩種形狀都
        由 `rand_like` 自動處理，device 與 dtype 也跟著 buffer 走，不會寫死
        cuda 或 float32。

        均勻分布是刻意的：條件變數要讓 policy 在**整個範圍**都學得動，不是
        偏重某個常見值。`rand` 的值域是 `[0, 1)`，乘上區間長度再加下界後
        取不到上界——在連續分布上是零測度集，不影響學習；需要精確取到端點
        的場合（評估固定在 1.0）請用 `(1.0, 1.0)` 這種塌成單點的範圍。
        """
        span = self._offset_high - self._offset_low
        sampled = torch.rand_like(self._max_offset[index])
        self._max_offset[index] = sampled * span + self._offset_low

    def _update_shot_tracking(self) -> None:
        """B-3a：累積這一局的進袋／顆星接觸／首次接觸。

        純讀取 + 更新旗標，**不對球做任何物理干預**。進袋的球在模擬裡會繼續
        滾（袋口是 trigger 體積不是洞），但 reward 端不看它的實際位置——
        `calculate_spread_score()` 的 docstring 明文要求呼叫端替進袋球代入
        袋口座標，B-3b 就是那樣做的。

        不搬球的理由：袋口 Cylinder 是有碰撞的（Demo 靠它做 contact reporting），
        把球吸附到袋口中心會造成穿透、被 PhysX 推開，而那個行為我沒辦法在本機
        驗證。已知差異：Demo 端進袋後會把球移出檯面，訓練端不會——影響的是
        進袋球之後還會不會撞到別的球。列為 #121 的已知落差。
        """
        if not bool(self._struck.any()):
            # 還沒擊球，不可能有任何事件。省掉整組距離計算。
            return

        ball_xy = (
            self._asset.data.body_link_pos_w.torch[..., :2]
            - self._env.scene.env_origins[:, None, :2]
        )

        is_pocketed, nearest_pocket = detect_pocketed(ball_xy)
        # 只寫「這個 tick 新進袋」的球：已記錄過的保留原本的袋口索引，
        # 免得球滾出袋口範圍後又被判進另一個袋。
        newly = is_pocketed & (self._pocket_index < 0)
        self._pocket_index = torch.where(newly, nearest_pocket, self._pocket_index)

        self._rail_contacted |= detect_rail_contact(ball_xy, self.cfg.ball_radius)

        # ⚠️ 順序：closest_approach 必須用**這個 tick 之前**的 first_contact。
        #    碰到 1 號球的那個 tick，first_contact 還是 -1，間距才記得到 ~0；
        #    先更新 first_contact 的話命中的那一局反而拿不到塑形滿分。
        self._closest_approach = update_closest_approach(
            self._closest_approach,
            ball_xy,
            self.cfg.ball_radius,
            self._first_contact,
        )
        self._first_contact = update_first_contact(
            self._first_contact, ball_xy, self.cfg.ball_radius
        )

    def _apply_rolling_resistance(self) -> None:
        """B-6：對所有球施加滾動摩擦與自旋衰減。

        **已經整個安靜下來的 env 完全不寫入**，這是本方法唯一的技巧，也是
        `core` 那個 `continue`（`rolling_resistance_service.py:90`）在張量 API
        下能保留多少就保留多少的做法：

        持續每個 tick 呼叫寫入，PhysX 就沒機會把球放進 sleep——接觸解算會在
        雜訊量級持續重新產生殘留（永遠不是精確的 0），球會永遠卡在那個殘留值
        上不消失（GUI 實測：9 顆 rack 球卡在 vz≈0.0687 永久不動）。而 vz 在
        衰減公式裡是原封不動傳遞的，每個 tick 寫入等於每個 tick 把舊的 vz
        重新注入，B-4 的球靜止判定（檢查完整 3D 速度）就永遠不會成立。

        張量 API 的粒度限制：`write_body_com_velocity_to_sim_mask` 的 `body_mask`
        形狀是 `(num_bodies,)`、`env_mask` 是 `(num_instances,)`，兩個是各自
        獨立的一維遮罩，選出來的是矩形區域——表達不出「env 3 的第 7 顆球跳過、
        第 2 顆球寫入」。所以逐球跳過做不到，只能做到**逐 env 跳過**。

        用 `_index` 而不是 `_mask`：兩者行為相同（都只寫選中的 env），但
        `env_ids` 明確接受 `torch.Tensor`，而 `env_mask` 的型別標註只有
        `wp.array`，得多一層 warp 轉換。若之後 profiling 顯示這裡的 gather
        是熱點，再換成 `_mask` 版（那個吃完整資料，省掉 gather）。

        殘留差異：一個 env 裡「9 顆已停、1 顆還在滾」的過渡期，core 會跳過那
        9 顆、這裡會照寫。但過渡期結束時整個 env 就關掉寫入了，PhysX 會在
        約 0.4 秒內收斂並 sleep——`decimation=60` 之下不到一個 env step。
        """
        lin_vel = self._asset.data.body_com_lin_vel_w.torch
        ang_vel = self._asset.data.body_com_ang_vel_w.torch

        new_lin, new_ang, is_noise = decay_velocities(
            lin_vel, ang_vel, self.cfg.ball_radius
        )

        # 這個 env 的 10 顆球全都只剩雜訊 → 完全不寫入，交還給 PhysX 的 sleep。
        active = (~is_noise.all(dim=1)).nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            return

        velocity = torch.cat((new_lin[active], new_ang[active]), dim=-1)  # (A, B, 6)
        # body_ids 省略 = 全部球。寫的是質心速度；球是均質球體，質心與 link
        # 原點重合，所以與 core 走 RigidBodyAPI.set_velocities() 等價。
        self._asset.write_body_com_velocity_to_sim_index(
            body_velocities=velocity, env_ids=active
        )

    def _strike(
        self,
        env_ids: torch.Tensor,
        placements: list[list[float]],
        velocities: list[list[float]],
    ) -> None:
        """把母球搬到擺位並賦予初速度。

        env_ids: `(P,)` 要擊球的子環境索引
        placements: P 筆桌台相對 XY（`decode_rl_action` 已夾在 kitchen 內）
        velocities: P 筆 `[vx, vy, vz, wx, wy, wz]`
        """
        # body_link_state_w 是 (num_envs, 10, 13)：pos(3) + quat(4) + lin(3) + ang(3)，
        # 前 7 格就是位姿。
        state = self._asset.data.body_link_state_w.torch
        pose = state[env_ids, self._cue_index, :7].clone()  # (P, 7)

        origins = self._env.scene.env_origins[env_ids]  # (P, 3)
        placement = torch.tensor(
            placements, device=pose.device, dtype=pose.dtype
        )  # (P, 2)

        # 桌台相對座標 → 世界座標（A-2 換算表第 4 列：寫 action 要「加」）。
        # 漏加的話 env 5 的母球會被擺到 env 0 的桌上，每一局都從桌外開球——
        # 訓練跑得動、loss 有在動、就是學不起來。
        pose[:, 0] = placement[:, 0] + origins[:, 0]
        pose[:, 1] = placement[:, 1] + origins[:, 1]
        # 桌面在子環境局部座標的 z=0（A-1 的球 init z 就是球半徑），
        # 所以世界高度 = 環境原點 z + 球半徑。
        pose[:, 2] = origins[:, 2] + self.cfg.ball_radius

        # 姿態（pose[:, 3:7]）沿用現值，不自己組 identity quat：Isaac Lab 3.0 的
        # 分量順序是 (x, y, z, w)，與 2.x 的 (w, x, y, z) 相反，寫入端的約定
        # 尚未確認。讀現值改 xyz 再寫回可以完全繞開這個問題，而且母球開球前的
        # 姿態本來就無所謂（球是對稱的）。

        velocity = torch.tensor(velocities, device=pose.device, dtype=pose.dtype)  # (P, 6)

        # 形狀是 (len(env_ids), len(body_ids), ...)，只動母球一顆所以中間維是 1。
        # 這兩個方法是關鍵字專用（簽章有 `*`），不能用位置參數；
        # 沒有 _index 後綴的版本已 deprecated（見檔案上方的 API 三層說明）。
        body_ids = [self._cue_index]
        self._asset.write_body_link_pose_to_sim_index(
            body_poses=pose.unsqueeze(1), body_ids=body_ids, env_ids=env_ids
        )
        self._asset.write_body_com_velocity_to_sim_index(
            body_velocities=velocity.unsqueeze(1), body_ids=body_ids, env_ids=env_ids
        )
