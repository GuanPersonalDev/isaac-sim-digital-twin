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

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv

    from .actions_cfg import BilliardStrikeActionCfg


def _resolve_attr(obj: object, candidates: Sequence[str], purpose: str) -> str:
    """回傳 candidates 中第一個存在於 obj 的屬性名。

    Isaac Lab 3.0 把 `RigidObjectCollection` 的 `object_*` 系列讀取 property
    rename 成 `body_*`（`object_pos_w` → `body_link_pos_w`，4.0 移除舊名）。
    **寫入方法與 `object_names` 有沒有跟著改名，尚未在 pod 上確認**（#121 D 組）。

    在建構期解析而不是直接寫死名字：猜錯的話會在 `gym.make()` 當下就炸並印出
    找過哪些名字，而不是訓練跑了三分鐘才在 `apply_actions()` 裡 AttributeError。

    D-2 確認實際名稱後，這個 helper 與三處呼叫都可以收斂成直接存取。
    """
    for name in candidates:
        if hasattr(obj, name):
            return name
    raise AttributeError(
        f"找不到{purpose}的屬性。已嘗試：{list(candidates)}。"
        f"Isaac Lab 3.0 可能又改了名稱，用下列指令查出實際名字後更新 candidates："
        f"  print([m for m in dir(type(obj)) if 'write' in m or 'name' in m])"
    )


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
        names_attr = _resolve_attr(
            self._asset, ("object_names", "body_names"), "collection 的物件名稱清單"
        )
        object_names = list(getattr(self._asset, names_attr))
        if cfg.cue_ball_name not in object_names:
            raise ValueError(
                f"collection 裡沒有母球 '{cfg.cue_ball_name}'，實際有：{object_names}"
            )
        self._cue_index = object_names.index(cfg.cue_ball_name)

        # 寫入方法在建構期就解析掉，理由見 _resolve_attr 的 docstring。
        self._write_pose = _resolve_attr(
            self._asset,
            ("write_object_link_pose_to_sim", "write_object_pose_to_sim"),
            "位姿寫入方法",
        )
        self._write_velocity = _resolve_attr(
            self._asset,
            ("write_object_com_velocity_to_sim", "write_object_velocity_to_sim"),
            "速度寫入方法",
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

        只有尚未擊球的 env 會做事；擊完之後整個 episode 剩下的 tick 都是
        直接 return，讓物理自己跑。

        ⚠️ B-6（滾動阻力）之後會加在這個方法的**最前面**，排在擊球寫入之前——
           先衰減、後擊球，避免剛寫入的初速在同一個 tick 就被扣掉。
           `apply_actions()` 是 manager-based 唯一每個 physics tick 觸發的 hook。
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

        # 張量 API 的形狀是 (len(env_ids), len(object_ids), ...)，
        # 只動母球一顆所以中間維是 1。
        getattr(self._asset, self._write_pose)(
            pose.unsqueeze(1), env_ids=env_ids, object_ids=[self._cue_index]
        )
        getattr(self._asset, self._write_velocity)(
            velocity.unsqueeze(1), env_ids=env_ids, object_ids=[self._cue_index]
        )
