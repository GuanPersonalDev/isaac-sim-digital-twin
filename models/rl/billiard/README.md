# billiard RL policy（本機訓練，2026-08-12）

來源 run：`training/outputs/logs/rsl_rl/billiard/2026-08-12_08-55-10`（本機 RTX 4090，
`Isaac-Billiard-v0`，`num_envs=1024`，超參數見
`rl_task/billiard_rl/tasks/manager_based/billiard_rl/agents/rsl_rl_ppo_cfg.py`，
與雲端 #124 第三輪成功版一致）。

完整過程與判讀見 `docs/local-training-2026-08-12.md`。

## 檔案

| 路徑 | 內容 |
|---|---|
| `policy.pt` / `policy.onnx` | **目前版本**——iteration 200（收斂後），複製自 `iter200/` |
| `iter0/` | iteration 0（隨機初始化，訓練前）匯出的 policy，僅供對照回放用 |
| `iter50/` | iteration 50 匯出的 policy，訓練中段初期 |
| `iter100/` | iteration 100 匯出的 policy，訓練中段 |
| `iter200/` | iteration 200（收斂）匯出的 policy，與頂層 `policy.pt` 相同 |
| `env.yaml` | 訓練當下完整環境參數 dump（來源 `training/outputs/.../params/env.yaml`） |
| `run_2026-08-12_metrics.csv` | 整輪 TensorBoard scalar 原始值（未縮放），212 個 iteration |
| `run_2026-08-12_training_curve.png` | 訓練曲線圖（mean_reward／spread／aim），標出 it=100／200 檢查點 |

`policy.pt` 是 TorchScript 格式，`policy.onnx`／`policy.onnx.data` 是 ONNX 格式（batch size
固定為 1）。四個階段都是從 rsl_rl 的 `model_0.pt` / `model_50.pt` / `model_100.pt` /
`model_200.pt`（含 optimizer state，只有 rsl_rl 載得動，未進版控，留在本機
`training/outputs/` 底下）匯出而來。本輪在迭代 ~212 收斂停訓，沒有 iteration 1000；
#227 的四段對照用 **0 / 50 / 100 / 200**。

## #227 eval 場景參數（四次回放必須完全相同）

除了載入的 policy 不同，其餘變因一律鎖定下面這組值。任一項飄了，對照就失去意義。

| 參數 | 值 | 來源 |
|---|---|---|
| eval seed | `42` | `env.yaml` 的 `seed`（與訓練同一顆；本輪沒有另開 eval seed） |
| `max_offset` | `0.6`（四次回放都餵這個定值，不要取樣） | 訓練時是 `(0.0, 1.0)` 逐 episode 取樣；評估鎖成單點，與 `billiard_rl_env_cfg.py` 註解的用法一致 |
| rack 擺位 | `BREAK_SHOT_POSITIONS`（固定值，非隨機） | `core/services/break_shot_position_provider.py` |
| `spin_efficiency` | `0.8` | `env.yaml` → `actions.strike.spin_efficiency` |
| 滾動摩擦係數 | `0.01` | `core/services/rolling_resistance_service.py` 的 `ROLLING_FRICTION_COEFF` |
| 物理 dt | `1/60`（`0.016666666666666666`） | `env.yaml` → `sim.dt` |
| 觀測維度 | 21（18 球位 + 母球 XY + `max_offset`） | 訓練／匯出設定 |

`BREAK_SHOT_POSITIONS` 數值（桌台相對座標，單位 m；球 0 為母球）：

| ball_id | x | y |
|---|---|---|
| 0 | 0.0 | -0.9525 |
| 1 | 0.0 | 0.635 |
| 2 | -0.028575 | 0.6844933518 |
| 3 | 0.028575 | 0.6844933518 |
| 4 | -0.05715 | 0.7339867037 |
| 5 | 0.05715 | 0.7339867037 |
| 6 | -0.028575 | 0.7834800555 |
| 7 | 0.028575 | 0.7834800555 |
| 8 | 0.0 | 0.8329734073 |
| 9 | 0.0 | 0.7339867037 |

完整環境 dump 見同目錄 `env.yaml`。TorchScript 推論時第 21 維必須寫入 `0.6`，例如：

```python
import torch
policy = torch.jit.load("models/rl/billiard/iter200/policy.pt")  # 或 iter0 / iter50 / iter100
policy.eval()
obs = torch.zeros(1, 21)
obs[0, -1] = 0.6  # eval max_offset，四段都用這個值
action = policy(obs)  # 6 維原始輸出，還要過 decode_rl_action()
```

## 給未來 `ModelController`（#127）的重點

- `obs_normalization=False`（見 `rsl_rl_ppo_cfg.py`），TorchScript 裡的 normalizer 是
  `Identity()`——**餵原始 21 維觀測即可**，不需要自己做正規化。
- forward 回傳的是**無界、確定性**的 6 維動作（高斯分布的 mean，沒有 tanh／clip）。
  **不可以自己解讀**，一定要餵給 `core.services.rl_action_decoder.decode_rl_action()`
  做裁切與反正規化，繞過它會把越界值直接當成越界的物理量。
- ONNX 的 `obs`/`actions` 兩個 tensor 的 batch size 都固定為 1，多環境批次推論要重新匯出。

## 重現／重新匯出

```powershell
$env:PYTHONPATH="C:\Users\Kuan\isaac-project\isaac-sim-digital-twin"
$env:ACCEPT_EULA="Y"; $env:PRIVACY_CONSENT="Y"; $env:OMNI_KIT_ACCEPT_EULA="YES"; $env:ISAACSIM_ACCEPT_EULA="YES"
C:\Users\Kuan\isaac-project\venv\Scripts\Activate.ps1
cd C:\Users\Kuan\isaac-project\IsaacLab
.\isaaclab.bat play --rl_library rsl_rl --task Isaac-Billiard-v0 `
    --external_callback billiard_rl.tasks.register_external_tasks `
    --checkpoint "C:\Users\Kuan\isaac-project\isaac-sim-digital-twin\training\outputs\logs\rsl_rl\billiard\2026-08-12_08-55-10\model_200.pt" `
    --num_envs 1
```

會在 checkpoint 同目錄的 `exported/` 下重新產生 `policy.pt` / `policy.onnx`。
