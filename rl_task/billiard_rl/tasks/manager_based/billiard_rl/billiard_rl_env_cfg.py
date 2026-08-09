# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from . import mdp

##
# Pre-defined configs
##
# /workspace/setup.sh 的 export PYTHONPATH 才能上 core 匯入成立
from core.models.table_ball_set import TableBallSet
from core.services.asset_utility import BALL_TEMPLATE_PATH, TRAINING_TABLE_PATH
from core.services.break_shot_position_provider import BreakShotPositionProvider
from core.services.numeric_validation import validate_max_offset

##
# 共用條件常數
##
# 21 維 observation 的最後一格（B-1），同時也是 B-2 動作偏移量的裁切半徑。
# 訓練環境沒有手臂（本 issue 的決定），偏移能力視為滿載。
#
# 定成模組常數而不是在兩處各寫 1.0：B-1 的 ObsTerm 與 B-2 的 ActionTermCfg
# 必須是同一個值。不一致的話 policy 看到的條件值跟實際生效的裁切半徑不同——
# 會學到一個「以為自己有 1.0 偏移能力、其實只有 0.6」的策略，完全不報錯。
#
# validate_max_offset() 在 import 時就檢查 [0.0, 1.0]，改壞了立刻炸，而不是
# 訓練跑完才發現。Milestone B（#180，量出手臂實際偏移能力）接上時只改這一行。
TRAINING_MAX_OFFSET = validate_max_offset(1.0)

##
# 座標系與 env_origins 換算（A-2）
##
# core/ 一律使用「桌台相對座標」，Isaac Lab 的模擬則是世界座標。
# scene.env_origins 形狀 (num_envs, 3)，是每個子環境原點的世界座標。
# 四個場合的換算方向不同，漏掉任何一格都**不會報錯**：
#
#   場合                          | 屬於 | 換算
#   ------------------------------|------|--------------------------------
#   init_state.pos                | A-1  | 不加。cloner 會把整個子環境搬到
#                                 |      | 各自的 origin，init_state 是相對值
#   reset event 設定 root state   | B-5  | 加。default_root_state 存的是相對值，
#                                 |      | += env.scene.env_origins[env_ids]
#   讀 observation                | B-1  | 減。body_link_pos_w - env_origins.unsqueeze(1)
#   寫 action（母球擺位）         | B-2  | 加。decode_rl_action() 回傳桌台相對座標
#
# B-1 的 unsqueeze(1) 不可省：env_origins 是 (num_envs, 3)，球位是
# (num_envs, 10, 3)，中間要補一維才廣播得到 10 顆球。忘了會直接報錯——
# 這算幸運的，其餘三格漏掉都是靜默算錯：
#   - B-1 漏減 → core/ 收到世界座標，encode 不做範圍檢查，reward 照算垃圾
#   - B-2 漏加 → env 5 的母球被擺到 env 0 的桌上，每一局都從桌外開球
#   - B-5 漏加 → reset 後所有環境的球疊在世界原點
# 三種都是「訓練跑得動、loss 有在動、就是學不起來」。

##
# Scene definition
##

def _make_ball_cfgs() -> dict[str, RigidObjectCfg]:
    positions = BreakShotPositionProvider().get_positions()
    
    z = TableBallSet.DEFAULT_BALL_RADIUS
    
    return {
        f"ball_{ball_id}": RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Ball_{ball_id}",
            spawn=sim_utils.UsdFileCfg(usd_path=BALL_TEMPLATE_PATH),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z)),
            
        ) for ball_id, (x, y) in sorted(positions.items())
    }


@configclass
class BilliardRlSceneCfg(InteractiveSceneCfg):
    """撞球 Scene, 1 table + 10 ball + light"""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 用 TRAINING_TABLE_PATH（去掉 SimpleRoom 的版本）而非 Demo 的 TABLE_PATH：
    # 房間會被複製到每個 env，碰撞體全部進 broadphase，而 policy 根本看不到它。
    # 兩份資產的階層完全相同，POCKET_RELATIVE_PATH 不受影響。
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=TRAINING_TABLE_PATH)
    )
    
    balls = RigidObjectCollectionCfg(rigid_objects=_make_ball_cfgs())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """動作規格：單一 ActionTerm，6 維（母球擺位 XY / 方向角 / 初速 / 上下左右偏移）。

    實作在 mdp/actions.py。與 B-1 相反，那裡**直接呼叫 core 的單筆函式**
    （`decode_rl_action()` + `compute_cue_ball_velocities()`），沒有 torch 重寫——
    撞球一局只擊一次，decode 每個 episode 每個 env 只跑一次，不是每步一次。

    注意
    1. 直接對球做 impulse，訓練環境沒有手臂（手臂 RL 屬 Milestone B / #180）
    2. 母球擺位是桌台相對座標，寫進模擬要加 env_origins（見上方換算表第 4 列）
    3. 正規化／反正規化與圓形裁切都在 core 實作（#225），這邊只負責接線與寫入
    """

    strike: mdp.BilliardStrikeActionCfg = mdp.BilliardStrikeActionCfg(
        # 與 B-1 ObsTerm 的 max_offset 引用同一個常數。兩者必須一致——
        # 不一致會讓 policy 看到的條件值跟實際生效的裁切半徑不同，且不報錯。
        max_offset=TRAINING_MAX_OFFSET,
    )



@configclass
class ObservationsCfg:
    """觀測規格：單一 ObsTerm，21 維（18 球位 + 2 母球 XY + 1 max_offset）。

    實作在 mdp/observations.py。那裡是 torch 向量化版而非直接包住 core 的
    encode_rl_observation()——ObsTerm 每個 env step 都會被觸發，1024 環境跑
    Python 迴圈是每步數十毫秒。兩份實作由 rl_task/tests/ 的對拍測試綁死。

    從 Collection 讀出的球位是 world frame，要減掉 env_origins 換成桌台相對
    座標（換算方向見檔案上方〈座標系與 env_origins 換算〉）。

    ⚠️ 屬性名稱：object_pos_w / object_quat_w / object_lin_vel_w / object_ang_vel_w /
       object_state_w / default_object_state 在 Isaac Lab 3.0 全部標記 deprecated，
       4.0 移除。用新名稱，形狀（torch）如下：

         body_link_pos_w      (num_envs, 10, 3)   球的位置
         body_link_quat_w     (num_envs, 10, 4)   球的姿態，分量順序 (x, y, z, w)
         body_com_lin_vel_w   (num_envs, 10, 3)   球的線速度
         body_link_state_w    (num_envs, 10, 13)  完整狀態

       第 1 維（object 維度）的順序 = RigidObjectCollectionCfg.rigid_objects 的
       插入順序，即 ball_0(母球) → ball_9。順序錯位不會報錯，只會讓 policy 學垃圾。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """送進 policy 網路的觀測群組

        Manager-based 支援多個 observation group, 但本專案 actor 與 critic 用同一份觀測，因此只有 policy 一組
        """

        # func 拿到的第一個參數固定是 env，其餘由 params 以關鍵字傳入。
        ball_positions = ObsTerm(
            func=mdp.ball_positions,
            params={"max_offset": TRAINING_MAX_OFFSET},
        )

        def __post_init__(self) -> None:
            # 觀測噪音關閉：21 維的最後一格是條件變數不是感測值，
            # enable_corruption 會連它一起加噪音。
            self.enable_corruption = False
            # 串接成單一向量。目前只有一個 term，但 concatenate_terms=False
            # 會讓 observation_space 變成 dict，D-2 的維度檢查對不上。
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件設定
    
    完成後會有一個 mode="reset" 的 EventTerm, 每次 episode 重置時把 10 顆球套回 BREAK_SHOT_POSITIONS 的固定擺位
    """


@configclass
class RewardsCfg:
    """獎勵
    
    四個獨立的 RewTerm
    """

@configclass
class TerminationsCfg:
    """
    終止條件
    """

##
# Environment configuration
##


@configclass
class BilliardRlEnvCfg(ManagerBasedRLEnvCfg):
    """
    環境的設定
    
    gym.register 的 env_cfg_entry_point 指向這個類別

    """
    # Scene settings
    #
    # num_envs：除錯用的預設值。三種覆寫方式，優先序由低到高——
    #   1. 這裡的 cfg 值
    #   2. Hydra override：  ... env.scene.num_envs=1024
    #   3. 專用 CLI 參數：   ... --num_envs 1024
    # 第 3 種是 Isaac Lab 內建（common.py:180-181），有給就無條件蓋掉 cfg 值。
    # 1024 環境屬 #223 [10-merge] 的範圍，本 issue 不需要跑到那個規模。
    #
    # env_spacing：grid cloner 擺放各子環境原點的間距，必須大於單一環境在
    # X／Y 上的最大延伸量。9-ball 檯面 2.54 × 1.27 m，USD 含邊框桌腳更大；
    # 球的 y 範圍從母球 -0.9525 到 8 號球約 +1.03。4.0 有充裕餘裕。
    # （若改用含 SimpleRoom 的 TABLE_PATH，房間約 9.5 × 8 m，這個值要提到 10.0
    #  以上才不會互相穿插——2026-08-08 從 USD 的 Floor2/Floor3 位置量得。）
    # ⚠️ 不要為了省空間調到 3.0 以下——filter_collisions=True 只保證物理上
    #    不互撞，視覺上重疊會讓 A-3 目視與 viser 檢視完全無法判讀。
    scene: BilliardRlSceneCfg = BilliardRlSceneCfg(num_envs=64, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        #
        # decimation：一個 env step 涵蓋幾個 physics tick。60 → 1 個 env step
        # 恰好是 1 秒物理。模板預設的 2（1/30 秒）在這個任務是錯的——撞球一局
        # 只擊一次，policy 卻會被要求輸出 150 次動作，其中 149 次沒有任何效果，
        # rollout buffer 99% 是零均值噪音。
        #
        # 也不能反過來把 decimation 開到「一步就是一局」（單步 bandit）：球未
        # 落定就結算 reward 是**系統性偏誤**——calculate_spread_score 取到飛行
        # 途中的隨機構型，而「還在飛」與出桿力道正相關，policy 會學成「打到
        # 時限還沒停」而不是「打出好的散開」；B-3 的進袋判定更是直接判錯。
        #
        # 所以走多步 + B-4 的球靜止提前終止：實際步數 ≈ 落定秒數（預期 6~12），
        # 而不是固定 20。這同時比單步安全版更省——沒有 rendering 時物理解算是
        # 絕對主導項，固定跑滿 20 秒是 3 倍物理成本只為省下 6 筆 buffer。
        self.decimation = 60
        #
        # episode_length_s：純安全網，不是預期長度。20 秒是估算值不是實測值
        # （純摩擦從 2 m/s 減到夾停門檻要 20 秒，靠顆星碰撞洩能才會縮短）。
        # 開跑後看 TensorBoard 的 Episode_Termination/time_out 比例再調：
        # 穩定在 0 附近可以往下調省成本，明顯 > 0 代表有一部分 episode 的
        # reward 是在球還在動的時候結算的。
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        #
        # ⚠️ sim.dt 必須是 1/60，不可改：core/services/rolling_resistance_service.py
        #    把 PHYSICS_DT = 1.0/60.0 寫死成模組常數、不透過參數傳入，B-6 會 import
        #    它。用模板預設的 1/120 衰減量會差一倍，而且不會報錯。
        self.sim.dt = 1 / 60
        # render_interval 不能再寫成 = self.decimation：decimation 現在是 60，
        # 那樣會變成 1 秒才畫一格，viser 目視與影片錄製全毀。
        self.sim.render_interval = 2