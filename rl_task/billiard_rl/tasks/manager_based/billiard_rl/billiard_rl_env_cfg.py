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
# 條件變數 max_offset 的取樣範圍（#122）
##
# `max_offset` 是 21 維 observation 的最後一格（B-1），同時也是 B-2 動作偏移量
# 的圓形裁切半徑。它是**條件變數**：policy 要學的不是「怎麼打」，而是「給定
# 這個偏移能力上限，該怎麼打」——這正是 #121／#222 用單一 policy 取代「訓練
# 兩個 policy」的 fallback 方案所依賴的機制。
#
# 因此它必須**每個 episode 每個 env 重新取樣**，取樣點在
# `BilliardStrikeAction.reset()`，整局固定。
#
# ⚠️ 2026-08-09 修正：本常數原本是定值 `1.0`（「訓練環境沒有手臂，偏移能力視為
# 滿載」）。那樣第 21 維在整個訓練期間都是常數，policy 學不到任何條件依賴，
# Milestone B（#180）量出手臂實際偏移能力後接上去就是分布外輸入——不報錯，
# 只是打不準。要固定值請改成塌成單點的範圍（例如 `(0.6, 0.6)`），且只用於
# 評估與 debug，不要用於訓練。
#
# 兩端一致性不再靠「共用同一個常數」保證——改成隨機之後那不管用了，兩邊各
# 取樣一次會得到不同的值。現在 ObsTerm 直接讀 ActionTerm 的 buffer，**只有
# 一份**，見 `mdp/actions.py` 的 `max_offset` property。
#
# validate_max_offset() 在 import 時就檢查 [0.0, 1.0]，改壞了立刻炸，而不是
# 訓練跑完才發現。
TRAINING_MAX_OFFSET_RANGE = (validate_max_offset(0.0), validate_max_offset(1.0))

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
    4. 條件變數 `max_offset` 的權威 buffer 也在這個 term 上（#122），B-1 的
       ObsTerm 讀它——**欄位名 `strike` 是 ObsTerm 的 `action_term_name` 參數
       指向的目標**，改名兩處要一起改（改錯會 KeyError，不會靜默降級）
    """

    strike: mdp.BilliardStrikeActionCfg = mdp.BilliardStrikeActionCfg(
        max_offset_range=TRAINING_MAX_OFFSET_RANGE,
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
        #
        # 不再傳 max_offset（#122）——它每局取樣一次，權威 buffer 在 B-2 的
        # ActionTerm 上，ObsTerm 執行期直接讀那一份。這裡只指名要讀哪個
        # action term，名字對應 `ActionsCfg.strike` 的欄位名。
        ball_positions = ObsTerm(
            func=mdp.ball_positions,
            params={"action_term_name": "strike"},
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
    """事件設定：每次 episode 重置時把 10 顆球套回 BREAK_SHOT_POSITIONS 固定擺位。

    ⚠️ 這一項不是選配。Isaac Lab 的 `_reset_idx()` **不會**自己把 asset 寫回
    default state，沒有這個 term 的話第二個 episode 起球會維持在上一局散開的
    位置——訓練照跑、reward 照算，只是每一局的初始盤面都不同，policy 學到的
    東西沒有意義，而且完全不報錯。

    評估場景必須完全固定（`training/README.md`〈Demo 素材〉），所以這裡不加任何
    隨機擾動。要做 domain randomization 請另開 EventTerm，不要污染這一個。
    """

    reset_break_shot: EventTerm = EventTerm(
        func=mdp.reset_break_shot_layout,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """獎勵：四個獨立的 RewTerm，分項進 TensorBoard。

    數值權威在 `core.services.reward_service.calculate_reward()`——那裡的邏輯
    不是單純相加（犯規重置時只回傳罰分、9 號球獎勵 gate 在沒犯規上），所以
    `mdp/rewards.py` 的 `decompose_reward()` 負責拆解，並由對拍測試保證
    **四項之和恆等於 `calculate_reward()`**。

    ⚠️ `weight` 一律 1.0。權重調整屬 #123（PPO 超參數）的範圍，而且改權重會讓
    分項總和不再等於 core 的 reward——要調請先確認那是刻意的。

    四項都只在球落定的那一步給分（gate 在 `all_balls_at_rest`）。中間步回 0，
    否則「飛越久分越高」。
    """

    spread: RewTerm = RewTerm(func=mdp.spread, weight=1.0)

    nine_ball: RewTerm = RewTerm(func=mdp.nine_ball, weight=1.0)

    cue_scratch: RewTerm = RewTerm(func=mdp.cue_scratch, weight=1.0)

    foul: RewTerm = RewTerm(func=mdp.foul, weight=1.0)

@configclass
class TerminationsCfg:
    """終止條件：局面已定就結束，不要為了湊滿 episode 長度而空轉。

    兩個 term 的語意不同且**都需要**：

      balls_at_rest  10 顆球全停 + 已擊球 → 局面已定，回報完整，不 bootstrap
      time_out       時間用完球還在動 → 結果未知，交給 value function bootstrap

    不能把 balls_at_rest 也標成 time_out——那會讓已經確定的回報被 value
    function 覆蓋掉。反過來拿掉 time_out 也不行：球沒停的那些 episode 會被當成
    「真的結束了」，reward 就是在飛行途中結算的垃圾。

    TerminationManager 會自動把每個 term 的觸發率寫進
    extras["log"]["Episode_Termination/<name>"]，rsl_rl 直接吐到 TensorBoard。
    所以 time_out 的比例是免費的診斷指標——穩定在 0 附近代表
    episode_length_s 綽綽有餘可以往下調，明顯 > 0 代表有一部分 episode 的
    reward 不可信。
    """

    balls_at_rest: DoneTerm = DoneTerm(func=mdp.all_balls_at_rest)

    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)

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
    #
    # 2026-08-10（#123）：預設值 64 → 1024。這個任務的**有效樣本 = 開球次數**，
    # 不是 transition 數：episode 約 10 步但只有第 1 步的動作有物理效果，所以
    # 64 env × 16 步 ÷ 10 ≈ 每個 iteration 只有 97 次真實開球，minibatch 內更是
    # 只有約 24 次——用 24 個樣本估 6 維連續動作的梯度，雜訊遠大於訊號。
    # 1024 env 把每個 iteration 的開球數拉到約 1560。
    #
    # ⚠️ num_envs 買的是「每個梯度步的樣本數」，不是梯度步數
    #    （= num_mini_batches × num_learning_epochs，與 num_envs 無關）。
    #    所以 max_iterations 必須一起提高，見 agents/rsl_rl_ppo_cfg.py。
    # ⚠️ wall-clock 不是線性的：64 env 下 GPU PhysX 幾乎全是 kernel launch 開銷。
    #    上 pod 後跑 64 / 256 / 1024 / 2048 各 10 個 iteration 記 it/s，找算力
    #    飽和點再決定要不要往 2048 加。先確認 1024 建得起來（桌台碰撞網格是
    #    主要記憶體項）。
    #
    # env_spacing：grid cloner 擺放各子環境原點的間距，必須大於單一環境在
    # X／Y 上的最大延伸量。9-ball 檯面 2.54 × 1.27 m，USD 含邊框桌腳更大；
    # 球的 y 範圍從母球 -0.9525 到 8 號球約 +1.03。4.0 有充裕餘裕。
    # （若改用含 SimpleRoom 的 TABLE_PATH，房間約 9.5 × 8 m，這個值要提到 10.0
    #  以上才不會互相穿插——2026-08-08 從 USD 的 Floor2/Floor3 位置量得。）
    # ⚠️ 不要為了省空間調到 3.0 以下——filter_collisions=True 只保證物理上
    #    不互撞，視覺上重疊會讓 A-3 目視與 viser 檢視完全無法判讀。
    scene: BilliardRlSceneCfg = BilliardRlSceneCfg(num_envs=1024, env_spacing=4.0)
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
        # episode_length_s：純安全網，不是預期長度——實際步數由 B-4 的球靜止
        # 提前終止決定。
        #
        # 2026-08-09 pod 實測（中等力道，正規化動作全 0 → 初速 1.92 m/s ＝
        # 當時的 CUE_BALL_SPEED 中點；#123 把下界 0.5 → 0.65 後中點是 1.9946，
        # 這筆量測的數字仍以量測當下的 1.92 為準，落定秒數的結論不受影響）：
        # 最大球速 1.280 → 0.455 → 0.352 → 0.249 →
        # 0.148 → 0.049 → 0.00000，四個 env 在**第 6~7 秒**全部落定。
        # 滿速 3.34 m/s 約 1.74 倍，推估 11~13 秒，20 仍有餘裕。
        #
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