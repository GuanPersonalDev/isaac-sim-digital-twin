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
from core.services.rl_action_decoder import decode_rl_action

from .physics import decay_velocities

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

        # 用名字查 index，不要寫死 0。_make_ball_cfgs() 的 sorted() 保證了順序，
        # 但那是 A-1 的實作細節，本模組不該依賴它。
        # body_names 是現行名稱；object_names 仍在但屬 deprecated 系列。
        body_names = list(self._asset.body_names)
        if cfg.cue_ball_name not in body_names:
            raise ValueError(
                f"collection 裡沒有母球 '{cfg.cue_ball_name}'，實際有：{body_names}"
            )
        self._cue_index = body_names.index(cfg.cue_ball_name)

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

        placements: list[list[float]] = []
        velocities: list[list[float]] = []
        for row in raw_rows:
            # should_execute_action 寫死 True：它是 Demo 端狀態機的控制旗標
            # （ScriptController 的 IDLE→AIMING→STRIKING），**不是第 7 維模型
            # 輸出**——多送一維會在 decode 的長度檢查被擋下。訓練端只在真的要
            # 擊球時才走到這裡，所以恆為 True。
            action = decode_rl_action(
                row, self.cfg.max_offset, should_execute_action=True
            )
            linear, angular = compute_cue_ball_velocities(
                action, self.cfg.ball_radius, self.cfg.spin_efficiency
            )
            placements.append(action.cue_ball_placement)
            velocities.append(linear + angular)

        self._strike(pending, placements, velocities)
        self._struck[pending] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """episode 重置：清掉擊球旗標，下一局才擊得了球。

        沒有這個的話 env 重置後 `_struck` 仍為 True，母球永遠不會再被賦速，
        訓練從第二局起全部是靜止畫面——而且不會報錯。
        """
        # env_ids=None 代表全部。不要就地改寫 env_ids——型別會從 Sequence[int]
        # 變成 slice，靜態檢查會擋。
        index: Sequence[int] | slice = slice(None) if env_ids is None else env_ids
        self._struck[index] = False
        self._raw_actions[index] = 0.0

    """
    Internal helpers.
    """

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
