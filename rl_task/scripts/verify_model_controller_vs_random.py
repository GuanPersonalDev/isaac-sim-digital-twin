# Copyright (c) 2026 GuanPersonalDev
"""#128：ModelController 在 Isaac Sim 實機物理中的效果驗證（只能在本機/pod 上跑）。

`docs/local-training-2026-08-12.md` 已經用 `run_2026-08-12_metrics.csv` 的
rsl_rl 訓練曲線證明「收斂後的參數」比「iteration 0 隨機參數」reward 高，但那組
數字是 rsl_rl rollout 當下算出來的，不是透過 `core.controllers.model_controller.
ModelController` 這個 class 本身跑出來的。#127 結案時明確留白一項：

    Isaac Sim 內實機驗證：訓練桌以模型參數擺位並出桿
    完整路徑（extension 載入 → 建桌 → physics callback → 模型擺位出桿）
    待 #128 一併確認。

本腳本補這一段。做法：每個平行 env 各自建立一個真正的 `ModelController` 實例，
餵它真正的 `Observation`，讓它自己跑 RESET→IDLE→AIMING 走完
`encode_rl_observation()` → `PolicyPort.infer()` 的推論路徑，拿到的原始 6 維
輸出直接送進訓練環境的 `env.step()`——`BilliardStrikeAction.apply_actions()`
內部呼叫的是**同一個** `decode_rl_action()`（#228 已對拍驗證兩端一致），所以
效果等同 `ModelController` 自己執行 `_execute_strike()`，只是省了狀態機的
`AIMING→STRIKING` 過渡（訓練環境沒有手臂，那個過渡不影響物理結果，見
`ModelController` docstring）。

分別對 `models/rl/billiard/policy.pt`（ModelController 目前載入的收斂版）與
`models/rl/billiard/iter0/policy.pt`（隨機初始化參數）各跑一次固定開球擺位，
用訓練環境既有的 `RewardsCfg`（數值權威在 `core.services.reward_service.
calculate_reward()`）算出每個 env 的 episode reward，比較兩者平均值。

用法（本機，在 repo root 執行）::

    $env:PYTHONPATH="C:\\Users\\Kuan\\isaac-project\\isaac-sim-digital-twin"
    $env:ACCEPT_EULA="Y"; $env:PRIVACY_CONSENT="Y"; $env:OMNI_KIT_ACCEPT_EULA="YES"; $env:ISAACSIM_ACCEPT_EULA="YES"
    C:\\Users\\Kuan\\isaac-project\\venv\\Scripts\\Activate.ps1
    C:\\Users\\Kuan\\isaac-project\\IsaacLab\\isaaclab.bat -p rl_task/scripts/verify_model_controller_vs_random.py --headless --num_envs 64
"""

import argparse
import os

from isaaclab.app import AppLauncher

# ⚠️ AppLauncher 必須在任何 isaaclab.* / billiard_rl.* 匯入之前執行完
#    （理由見 view_scene.py 的同位置註解）。
parser = argparse.ArgumentParser(description="ModelController vs 隨機參數 reward 對照（#128）")
parser.add_argument("--num_envs", type=int, default=64, help="平行環境數，越多統計越穩定")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下才能匯入 Isaac Lab 與本專案的模組。"""

import sys  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.billiard_rl_env_cfg import (  # noqa: E402
    BilliardRlEnvCfg,
)
from core.controllers.model_controller import ModelController  # noqa: E402
from core.models.action_bounds import ACTION_DIM  # noqa: E402
from core.models.billiard_state import BilliardStatus  # noqa: E402
from core.models.observation import Observation  # noqa: E402

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
# 跟 extension/billiard_digital_twin/billiard_digital_twin.py 同一種 import 路徑，
# 讓 TorchScriptPolicyImpl 用「extension/ 加進 sys.path」的方式被匯入。
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaac_sim_impl_6_0.torch_script_policy_impl import TorchScriptPolicyImpl  # noqa: E402

# #227 eval 場景參數（models/rl/billiard/README.md）：四段對照回放共用 0.6，
# 本腳本沿用同一個值，與 ModelController 在 extension 裡的 _EVAL_MAX_OFFSET 一致。
_EVAL_MAX_OFFSET = 0.6
_TRAINED_POLICY_PATH = os.path.join(_PROJECT_ROOT, "models", "rl", "billiard", "policy.pt")
_RANDOM_POLICY_PATH = os.path.join(_PROJECT_ROOT, "models", "rl", "billiard", "iter0", "policy.pt")


def _build_raw_actions(env: ManagerBasedRLEnv, policy: TorchScriptPolicyImpl) -> torch.Tensor:
    """每個 env 各自跑一個真正的 ModelController，走到 AIMING 拿推論出的原始 6 維輸出。

    刻意不直接呼叫 `encode_rl_observation()` / `policy.infer()`——要驗的是
    `ModelController` 這個 class 的推論路徑本身能不能正常跑完整個
    RESET→IDLE→AIMING 狀態轉換，不是底層函式本身（那已經被
    `core/tests/test_model_controller.py` 的 35 個案例覆蓋過）。
    """
    balls = env.scene["balls"]
    ball_xy = (
        balls.data.body_link_pos_w.torch[..., :2] - env.scene.env_origins[:, None, :2]
    ).cpu().tolist()

    raw_rows: list[list[float]] = []
    for env_ball_xy in ball_xy:
        ball_positions = [[x, y, 0.0] for x, y in env_ball_xy]
        controller = ModelController(policy, (0.0, 0.0), _EVAL_MAX_OFFSET)

        controller.get_action(
            Observation(
                ball_positions=ball_positions,
                cue_ball_position=ball_positions[0],
                is_init_state=False,
                is_ball_moving=False,
                is_motion_complete=True,
                has_error=False,
            )
        )  # RESET -> IDLE
        controller.get_action(
            Observation(
                ball_positions=ball_positions,
                cue_ball_position=ball_positions[0],
                is_init_state=True,
                is_ball_moving=False,
                is_motion_complete=False,
                has_error=False,
            )
        )  # IDLE -> AIMING，這一步才真正推論並快取原始輸出

        if controller.get_current_state() != BilliardStatus.AIMING or controller._cached_raw_action is None:
            raise AssertionError(
                "ModelController 未能正常推論到 AIMING（可能落進 ERROR），"
                "檢查 policy 輸出是否合法"
            )

        raw_rows.append(controller._cached_raw_action)

    return torch.tensor(raw_rows, dtype=torch.float32, device=env.device)


def _run_episode(
    env: ManagerBasedRLEnv,
    policy: TorchScriptPolicyImpl,
    label: str,
    max_steps: int,
) -> torch.Tensor:
    """整批 env 從固定開球擺位打一次，回傳每個 env 的累積 episode reward。"""
    env.reset()
    raw_actions = _build_raw_actions(env, policy)

    total_reward = torch.zeros(env.num_envs, device=env.device)
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    zeros = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)

    for step in range(max_steps):
        # 第 0 步送出 ModelController 的決策（BilliardStrikeAction 只在
        # `_struck=False` 時才真的解碼擊球，之後幾步送什麼都不影響物理）。
        actions = raw_actions if step == 0 else zeros
        _, rew, terminated, truncated, _ = env.step(actions)

        active = ~done
        total_reward += rew * active
        done |= terminated | truncated

        if bool(done.all()):
            print(f"[{label}] 第 {step + 1} 步全數落定（{env.num_envs}/{env.num_envs}）")
            break
    else:
        remaining = int((~done).sum())
        print(f"[{label}] 警告：{max_steps} 步內仍有 {remaining} 個 env 未落定（time_out），reward 沿用目前累積值")

    return total_reward


def main() -> None:
    cfg = BilliardRlEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    # #227 eval 場景參數：max_offset 鎖成單點而非逐局隨機取樣。
    cfg.actions.strike.max_offset_range = (_EVAL_MAX_OFFSET, _EVAL_MAX_OFFSET)

    max_steps = int(cfg.episode_length_s / (cfg.decimation * cfg.sim.dt)) + 1

    env = ManagerBasedRLEnv(cfg)
    print(f"[verify] 環境建立完成：num_envs={env.num_envs}, device={env.device}, max_steps={max_steps}")

    try:
        random_policy = TorchScriptPolicyImpl(_RANDOM_POLICY_PATH, device="cpu")
        random_reward = _run_episode(env, random_policy, "隨機參數 iter0", max_steps)

        trained_policy = TorchScriptPolicyImpl(_TRAINED_POLICY_PATH, device="cpu")
        trained_reward = _run_episode(env, trained_policy, "ModelController policy.pt", max_steps)

        random_mean = float(random_reward.mean())
        trained_mean = float(trained_reward.mean())
        random_std = float(random_reward.std())
        trained_std = float(trained_reward.std())

        print(f"\n[verify] 隨機參數 iter0                mean_reward = {random_mean:.4f} ± {random_std:.4f}（{env.num_envs} envs）")
        print(f"[verify] ModelController（policy.pt）  mean_reward = {trained_mean:.4f} ± {trained_std:.4f}（{env.num_envs} envs）")

        if trained_mean <= random_mean:
            raise AssertionError(
                f"ModelController 的平均 reward（{trained_mean:.4f}）未優於隨機參數基準"
                f"（{random_mean:.4f}），#128 未通過"
            )
        print(f"\n[verify] ModelController 平均 reward 優於隨機參數基準（+{trained_mean - random_mean:.4f}）✅ #128 通過")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
