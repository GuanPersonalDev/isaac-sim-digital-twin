# 本機訓練報告（2026-08-12，Milestone A 開球）

第一次在本機（非 RunPod 雲端）跑通 `Isaac-Billiard-v0` 的 RL 訓練，驗證本機環境可以
取代雲端進行後續訓練與除錯，並留下訓練前／後可回放對照的 policy。

## 環境

| 項目 | 值 |
|---|---|
| 硬體 | 本機 RTX 4090 24GB |
| Python | 3.12.10（`C:\Users\Kuan\isaac-project\venv`，與系統既有的 3.14 並存） |
| isaacsim | 6.0.1.0（pip 安裝） |
| Isaac Lab | `release/3.0.0-beta2`（`C:\Users\Kuan\isaac-project\IsaacLab`，與雲端訓練同版本） |
| torch | 2.10.0+cu128（`isaaclab.bat --install` 正確釘回，未被 isaacsim 預設的 2.11.0/cu13 卡住） |
| billiard_rl | `pip install -e rl_task`（editable） |
| PYTHONPATH | 專案根目錄，讓 `core.*` 可以被 `BilliardRlEnvCfg` import |

本機環境建置過程另有記錄；重點是**這是本機第一次成功跑 Isaac Sim**，第一次冷啟動
（含 shader/extension 快取建置）異常地慢，但那是一次性成本，之後的訓練/回放啟動都
在數秒到數十秒內完成。

## 訓練設定

沿用 `rl_task/billiard_rl/tasks/manager_based/billiard_rl/agents/rsl_rl_ppo_cfg.py`
目前的定案值（雲端 #124 第三輪成功後寫回的版本，**沒有另外調參**）：

- `num_envs=1024`、`max_iterations=1000`、`save_interval=50`
- `MaskedPPO`、`learning_rate=3.0e-4`、`schedule=adaptive`、`desired_kl=0.02`
- `gamma=1.0`、`lam=1.0`（reward 是純 terminal）

啟動指令：

```powershell
$env:PYTHONPATH="C:\Users\Kuan\isaac-project\isaac-sim-digital-twin"
$env:ACCEPT_EULA="Y"; $env:PRIVACY_CONSENT="Y"; $env:OMNI_KIT_ACCEPT_EULA="YES"; $env:ISAACSIM_ACCEPT_EULA="YES"
C:\Users\Kuan\isaac-project\venv\Scripts\Activate.ps1
cd C:\Users\Kuan\isaac-project\isaac-sim-digital-twin\training\outputs
C:\Users\Kuan\isaac-project\IsaacLab\isaaclab.bat train --rl_library rsl_rl `
    --task Isaac-Billiard-v0 `
    --external_callback billiard_rl.tasks.register_external_tasks `
    --headless
```

Log／checkpoint 落點：
`training/outputs/logs/rsl_rl/billiard/2026-08-12_08-55-10/`（本機 `.gitignore` 排除，
未進版控；重新訓練即可重現）。

## 迭代 100／200 檢查點

用專案既有的 `training/scripts/check_training.py`（讀 TensorBoard event，判斷「地形問題／
學起來又丟掉／收斂」三種情境）在兩個時間點檢查，不是憑感覺喊停：

### 迭代 100（非正式，僅供提前示警）

| 指標 | it0 | 峰值 | it~104 |
|---|---|---|---|
| mean reward | −1.592 | **+0.423** @it87 | +0.367（跌幅 2.8%） |
| spread ×20 | 0.014 | 0.815 @it81 | 0.770（跌幅 5.7%） |
| aim ×20 | 0.294 | 0.398 @it62 | 0.397（跌幅 0.6%） |

跌幅全部遠低於 40% 的退步門檻，判斷「還沒到決策點（104/200），繼續跑」——不需要中斷。

### 迭代 200（正式決策點）

| 指標 | it0 | 峰值 | it~210 |
|---|---|---|---|
| mean reward | −1.592 | **+0.606** @it147 | +0.545（跌幅 2.7%） |
| spread ×20 | 0.014 | 0.842 @it193 | 0.749（跌幅 11.2%） |
| aim ×20 | 0.294 | 0.398 @it155 | 0.397（跌幅 0.4%） |
| lr | 3.42e-3 | — | 1.00e-5（已被 adaptive 排程踩到底） |

**判定：🟢 指標守在峰值附近、lr 已到底 → 已收斂，繼續跑不會再變，可以收工。**
於是在迭代 ~212 停止訓練，未跑滿設定的 1000。

### 與雲端 #124 第三輪（成功版）對照

| 指標（峰值附近） | 雲端第三輪 | 本機這輪 |
|---|---|---|
| mean reward | +0.437 @it168 | **+0.606 @it147**（更早、更高） |
| spread ×20 | ~0.77 | ~0.84 @it193 |
| foul ×20（收斂值） | −0.510 | −0.492 |

兩輪用同一套超參數、同一顆隨機種子（`env seed=42`），走勢一致（地形正確、無第二輪
那種學起來又丟掉的退步），本機這輪數字略優，但單次跑的差異在 GPU 平行物理的
非決定性範圍內，不代表本機環境「更好」，只代表**同一套超參數在本機一樣可靠**。

## 產出物

| 檔案 | 說明 |
|---|---|
| `models/rl/billiard/policy.pt` / `.onnx` | **iteration 200**（收斂後）匯出的 policy，供未來 `ModelController`（#127）使用 |
| `models/rl/billiard/iter0/` | iteration 0（訓練前，隨機初始化）匯出的 policy，僅供回放對照 |
| `models/rl/billiard/iter200/` | 與頂層 `policy.pt` 相同，明確標註來源 iteration |
| `models/rl/billiard/run_2026-08-12_metrics.csv` | 整輪 TensorBoard 原始 scalar（212 iteration，未縮放） |
| `models/rl/billiard/README.md` | 給 `ModelController` 實作者的重點（normalizer、動作裁切、batch size 限制） |

匯出指令（對 `model_0.pt` 與 `model_200.pt` 各跑一次）：

```powershell
C:\Users\Kuan\isaac-project\IsaacLab\isaaclab.bat play --rl_library rsl_rl `
    --task Isaac-Billiard-v0 `
    --external_callback billiard_rl.tasks.register_external_tasks `
    --checkpoint "<run目錄>\model_0.pt" `   # 或 model_200.pt
    --num_envs 1
```

`play` 會自動把 policy 匯出到 `<checkpoint 所在目錄>\exported\`，兩次執行會互相覆蓋，
所以每次匯出後立刻複製到獨立資料夾（已反映在 `models/rl/billiard/` 的結構裡）。

## 回放方式（看最一開始 vs 訓練完的擊球）

`isaaclab.bat play` 本身就會開 Isaac Sim GUI 執行完整物理模擬（不加 `--headless`），
這是本機環境相對雲端 headless 訓練的優勢——可以直接看，不需要另外接 ModelController。

**訓練前（iteration 0，隨機亂打）：**

```powershell
$env:PYTHONPATH="C:\Users\Kuan\isaac-project\isaac-sim-digital-twin"
$env:ACCEPT_EULA="Y"; $env:PRIVACY_CONSENT="Y"; $env:OMNI_KIT_ACCEPT_EULA="YES"; $env:ISAACSIM_ACCEPT_EULA="YES"
C:\Users\Kuan\isaac-project\venv\Scripts\Activate.ps1
cd C:\Users\Kuan\isaac-project\IsaacLab
.\isaaclab.bat play --rl_library rsl_rl --task Isaac-Billiard-v0 `
    --external_callback billiard_rl.tasks.register_external_tasks `
    --checkpoint "C:\Users\Kuan\isaac-project\isaac-sim-digital-twin\training\outputs\logs\rsl_rl\billiard\2026-08-12_08-55-10\model_0.pt" `
    --num_envs 4
```

**訓練完（iteration 200，收斂後）：** 同上指令，`--checkpoint` 換成同目錄的 `model_200.pt`。

`--num_envs 4` 只是方便一次看多顆球的擺位/角度變化，不影響 policy 本身；改成 1
也可以。視窗跳出來後就是即時物理模擬，Ctrl+C 結束。

## 已知限制／後續

- **`ModelController` 本身尚未實作**（backlog #127）。這次只把「訓練完的模型」放進
  `isaac-sim-digital-twin`，Demo 端要真正把它接上 `ControllerBase` 介面、換掉
  `ScriptController`，是另一項獨立工作，`models/rl/billiard/README.md` 記錄了實作時
  要注意的三個坑（normalizer 是 Identity、動作要過 `decode_rl_action()`、ONNX batch=1）。
- 承襲雲端 #124 的未解疑點：**`foul` 卡在約 −0.49 ⟹ legal break（真正合法開球）比例
  仍然接近 0**，尽管 spread／aim 都已經頂到訓練上限。這在本機這輪一樣沒改善，
  推測 policy 學到的是「切球」而非「正對球堆全力開球」，需要實際看 GUI 回放判讀
  （見上方回放指令），不是本報告能單靠數字下定論的部分。
- 若想再往上推：`num_learning_epochs` 5 → 3（從源頭壓 KL，讓 lr 撐久一點），這是
  雲端文件記錄的下一手，本機尚未嘗試。
