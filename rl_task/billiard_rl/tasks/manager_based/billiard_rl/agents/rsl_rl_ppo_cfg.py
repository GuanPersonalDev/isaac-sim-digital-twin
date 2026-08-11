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
    actor = RslRlMLPModelCfg(
        # 32×32 是 cartpole 模板預設值。21 維觀測、6 維連續動作，而 reward 含
        # 凸包面積這種非線性幾何量——兩層 32 撐不住（#123）。
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        # init_std=1.0 在正規化域 [-1, 1] 上等於「幾乎均勻亂打」：對 shot_angle
        # （物理域 [-180, 180)）而言探索標準差是 ±180 度。#123 的 800 局實測沒有
        # 任何一局合法開球，主因不是 reward 而是根本瞄不準。
        #
        # ⚠️ 0.4 仍然遠大於瞄準的容錯窗口：母球到 1 號球 1.5875 m，接觸只容許
        #    側向 2R，換算角度僅 ±2.062°，正規化域上是 ±0.0115——init_std=0.4
        #    的探索半寬是 ±72°，命中質量比只有約 2.9%。#231 已把區間端點搬正
        #    （normalized 0 現在是正對球堆，不再是背對），但**解析度沒有改善**；
        #    是否為 Milestone A 收窄 SHOT_ANGLE 見 #231 的問題 2。
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
        learning_rate=1.0e-3,
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
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
