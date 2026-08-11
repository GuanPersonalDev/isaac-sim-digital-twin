# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    # 每個 iteration 的梯度步數 = num_mini_batches × num_learning_epochs = 20，
    # **與 num_envs 無關**。150 iteration 只有 3000 個梯度步，對放大後的網路
    # 不夠收斂（#123）。num_envs 給樣本、max_iterations 給優化步數，兩者不能
    # 互相取代。
    max_iterations = 1000
    save_interval = 50
    # log 落點是 <CWD>/logs/rsl_rl/<experiment_name>/<timestamp>（見 #121 E-2）。
    # 改名會讓新舊 run 分家，訂下來就不要再動。
    experiment_name = "billiard"
    # actor 與 critic 各自要吃哪些 observation group。值取自 ObservationsCfg 的
    # 群組名（本專案只有一個 'policy' 群組，21 維）。
    #
    # 不設也能跑——rsl_rl 5.0.1 會 fallback 去找名為 'policy' 的群組，而且猜對
    # 了（log 的 `Resolved observation sets` 兩邊都是 ['policy']）。但它每次啟動
    # 都印三行 UserWarning，其中兩行明說 **This behavior will be removed in a
    # future version**。#123 就記著這個缺口，趁 #124 開長訓練之前補掉，免得之後
    # 升 rsl_rl 時變成「跑不起來而且不知道為什麼」。
    #
    # ⚠️ 本機無 isaaclab/rsl_rl，這個欄位名沒辦法在本機驗證。判斷標準很簡單：
    #    上 pod 後那三行 `obs_groups` 的 UserWarning 應該完全消失，而
    #    `Resolved observation sets` 仍然是 actor/critic 各 ['policy']。若警告
    #    還在，代表欄位放錯層級——那也只是回到現況，不會弄壞任何東西。
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        # 32×32 是 cartpole 模板預設值。21 維觀測、6 維連續動作，而 reward 含
        # 凸包面積這種非線性幾何量——兩層 32 撐不住（#123）。
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        # init_std=1.0 在正規化域 [-1, 1] 上等於「幾乎均勻亂打」。#123 的 800 局
        # 實測沒有任何一局合法開球，主因不是 reward 而是根本瞄不準。
        #
        # 0.4 配上 #231 收窄後的 SHOT_ANGLE（Milestone A 為 ±30°）是 ±12° 的
        # 探索半寬，對上 ±2.062° 的接觸窗口 → 命中質量比約 17.2%。收窄前是
        # ±72° 對 ±2.062°，只有 2.9%——2026-08-11 的 pod 短訓練就卡在那裡：
        # critic 五個 iteration 把 value loss 壓到 0.01（答案永遠是 -1.5）、
        # advantage ≈ 0、action std 完全不動。
        #
        # ⚠️ Milestone B 把 SHOT_ANGLE 改回 ±180° 時，這個值要一起重新評估——
        #    同樣的 0.4 在整圈區間上會退回 2.9% 的命中率。
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        # MaskedPPO：只讓 episode 第一步（真正擊球那一步）產生 policy 梯度。
        # 理由與上 pod 前的驗證清單見 billiard_rl/algorithms/masked_ppo.py。
        class_name="billiard_rl.algorithms.MaskedPPO",
        # ⚠️ value_loss_coef 與 entropy_coef 一律縮 10 倍，是 MaskedPPO 的配套，
        #    不是獨立的調參結果：surrogate_loss 用 .mean()，遮罩後分子少約 90%
        #    但分母沒縮，這兩項若不同步縮就會相對放大 10 倍。要拿掉 MaskedPPO
        #    的話，這兩個數字必須同時改回 1.0 / 0.005。
        value_loss_coef=0.1,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 🔴 3.0e-4 是實測值不是猜的（#124 第二輪，2026-08-11）。
        #
        # 那一輪的 reward 地形修好之後 policy 確實學起來了——break_foul 終止率
        # 0.077 → 0.189（上一輪是歸零）、母球對 1 號球的最近接近距離
        # 0.514 m → 0.321 m、spread ×20 在 iter 57 衝到 0.117（起點的 7 倍）。
        # 然後全部退回去：iter 228 的 spread 只剩 0.002、接近距離退到 0.448 m。
        #
        # 退步與 lr 同步：
        #
        #   iter    0    57   114   171   228
        #   lr    1e-3 3.8e-4 8e-5  1e-5  1e-5
        #   spread 0.017 0.117 0.014 0.009 0.002
        #
        # adaptive 排程的規則是「KL 太大就砍 lr」。1.0e-3 配上同一批資料
        # num_learning_epochs × num_mini_batches = 20 個梯度步本來就會產生大
        # KL，排程只好一路砍；砍到 1e-5 時 policy 已經漂過最佳點，之後就凍在
        # 退步後的位置——剩下 770 個 iteration 完全沒有變化。
        #
        # **最佳點 iter 57 對應的 lr 恰好是 3.8e-4**，所以起點直接放在那裡，
        # 排程不必猛砍。
        #
        # 若新一輪 lr 又崩到 1e-5，下一手是 num_learning_epochs 5 → 3（從源頭
        # 減少同批資料的重複更新以壓 KL），**不是**再調這個數字。
        learning_rate=3.0e-4,
        schedule="adaptive",
        # ⚠️ gamma 必須是 1.0，這不是調參是修正方向性錯誤（#123）。
        #    reward 是純 terminal（mdp/rewards.py 只在 all_balls_at_rest 時給分），
        #    episode 步數 = 落定秒數，而**落定秒數與出杆力道正相關**（模擬：
        #    1.92 m/s → 9.8 s，3.34 m/s → 11.8 s）。gamma < 1 於是變成「打越大力
        #    return 打越多折」——折掉的量與 spread 訊號本身同一個數量級，等於在
        #    懲罰我們想學的行為。
        gamma=1.0,
        # 中間步驟沒有任何 reward，GAE 的 lam < 1 只會把 critic 的 bias 混進來。
        # 終點給分的有限 episode 用 lam = 1.0（Monte Carlo return）最乾淨。
        lam=1.0,
        # 0.01 → 0.02（#124 第二輪）。adaptive 排程在 KL > desired_kl × 2 時把
        # lr 除以 1.5，desired_kl=0.01 等於「KL 一超過 0.02 就腰斬」——對這個
        # 任務太敏感，實測是一路砍到 1e-5 且再也沒回來（回升條件是
        # KL < desired_kl / 2，而 action std 在收斂期本來就會讓 KL 偏大）。
        #
        # 放寬到 0.02 是給排程餘裕，不是關掉它：真正發散時它仍然會踩煞車。
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
