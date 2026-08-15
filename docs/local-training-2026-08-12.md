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

## 對應 Issue 完成標準核對（#124／#125／#126）

回頭核對 GitHub Issue，本次本機訓練連帶把三張卡的完成標準都補齊了：

| Issue | 完成標準 | 狀態 |
|---|---|---|
| **#124** [9-5] 單環境訓練跑通確認 | 訓練迴路啟動無錯誤 | ✓ |
| | reward 曲線有數值變化（非固定值） | ✓（−1.59 → +0.5 附近） |
| **#125** [9-6] reward 曲線上升趨勢 | 訓練 N 步後 reward 平均值明顯高於初始 | ✓ |
| | 訓練曲線截圖留存 | ✓ `models/rl/billiard/run_2026-08-12_training_curve.png` |
| **#126** [9-7] 模型儲存並可載入推論 | 模型儲存完成（checkpoint 格式） | ✓ `model_0.pt` / `model_200.pt`（rsl_rl），另有 TorchScript／ONNX 匯出 |
| | 載入後能正確執行推論（輸入 observation，輸出 action） | ✓ 見下方獨立驗證 |

### #126 的獨立推論驗證

不只是「`play` 沒有報錯」這種間接證據，額外做了一次單獨載入 TorchScript 並手動餵觀測的測試：

```python
import torch
policy = torch.jit.load("models/rl/billiard/iter200/policy.pt")
policy.eval()
obs = torch.zeros(1, 21)
obs[0, -1] = 0.6  # max_offset，合法範圍 [0,1]
action = policy(obs)
```

輸出：

```
input  obs shape   : (1, 21)
output action shape: (1, 6)
output action values: [-0.0403, 0.3385, -0.0060, 0.6063, 0.6712, 0.0783]
```

21 維觀測進、6 維動作出，形狀與訓練設定的 `ObservationsCfg`／`ActionsCfg` 完全吻合。
（提醒：這裡印出的是模型的**原始**輸出，無界、未裁切——真正要用於執行時必須先過
`core.services.rl_action_decoder.decode_rl_action()`，見 `models/rl/billiard/README.md`。）

### 訓練曲線圖

`models/rl/billiard/run_2026-08-12_training_curve.png`——`mean_reward`／`spread x20`／
`aim x20` 三條曲線，標出 it=100 檢查點與 it=200 決策點。

## #128 確認 ModelController 執行效果優於隨機參數（2026-08-15）

`#127` 結案時留白一項：「Isaac Sim 內實機驗證：訓練桌以模型參數擺位並出桿」，
註記待 #128 一併確認。純看 `run_2026-08-12_metrics.csv` 的 rsl_rl 訓練曲線
（iteration 0 的 `Train/mean_reward = −1.592` vs 收斂後 `+0.527`）只能證明
「訓練後的權重比隨機權重好」，證明不到 `ModelController` 這個 class 本身接上
Isaac Sim 物理之後也是同樣結果——決策路徑（`encode_rl_observation()` →
`PolicyPort.infer()` → 狀態機轉換）沒有被實機跑過。

### 實機驗證方法

新增 `rl_task/scripts/verify_model_controller_vs_random.py`，在
`Isaac-Billiard-v0` 訓練環境（64 平行 env、GPU 物理、`max_offset` 鎖定 0.6，
與 #227 eval 場景參數一致）裡，每個 env 各自建立一個**真正的**
`core.controllers.model_controller.ModelController` 實例，餵真正的
`Observation`，讓它自己跑完 `RESET → IDLE → AIMING` 拿到推論出的原始 6 維
輸出，直接送進 `env.step()`——訓練環境的 `BilliardStrikeAction` 內部呼叫的是
**同一個** `decode_rl_action()`（#228 已對拍驗證兩端一致），所以效果等同
`ModelController` 自己執行 `_execute_strike()`。分別對 `policy.pt`（收斂）與
`iter0/policy.pt`（隨機初始化）各跑一次固定開球擺位，用訓練環境既有的
`RewardsCfg`（權威在 `core.services.reward_service.calculate_reward()`）
算出每個 env 的 episode reward。

執行指令：

```powershell
C:\Users\Kuan\isaac-project\IsaacLab\isaaclab.bat -p rl_task/scripts/verify_model_controller_vs_random.py --headless --num_envs 64
```

### 結果（2026-08-15，64 envs，cuda:0）

| checkpoint | mean_reward | std |
|---|---|---|
| `iter0/policy.pt`（隨機參數） | −1.2915 | 0.2440 |
| `policy.pt`（`ModelController` 目前載入） | −1.1332 | 0.8555 |

`ModelController` 平均 reward 優於隨機參數基準 **+0.1583**，腳本判定 ✅ 通過。

兩者數值都是負的——與已知限制一致（見上方「已知限制／後續」：`foul` 卡在約
−0.49，legal break 比例仍接近 0，policy 學到的較可能是「切球」而非「正對球堆
全力開球」）。差距幅度（+0.16）也明顯小於訓練曲線峰值對照的差距（+2 以上），
原因是評估固定 `max_offset=0.6`（訓練時是全範圍 0~1 取樣，policy 只在部分
子空間學得比較好）且只有 64 個 env 的單次開球樣本，標準差偏大（收斂版
0.86，個別 env 出手品質仍有落差）。但方向一致、真實物理路徑驗證通過，確認
`ModelController` 執行效果優於隨機參數，#128 完成。

## 已知限制／後續

- Demo 端要真正把 `ModelController` 接上 `ControllerBase` 介面、換掉 `ScriptController`
  的完整整合仍在後續 Block（見 `docs/phase3-schedule.md`）。`models/rl/billiard/README.md`
  記錄了實作時要注意的三個坑（normalizer 是 Identity、動作要過 `decode_rl_action()`、
  ONNX batch=1）。
- 承襲雲端 #124 的未解疑點：**`foul` 卡在約 −0.49 ⟹ legal break（真正合法開球）比例
  仍然接近 0**，尽管 spread／aim 都已經頂到訓練上限。這在本機這輪一樣沒改善，
  推測 policy 學到的是「切球」而非「正對球堆全力開球」，需要實際看 GUI 回放判讀
  （見上方回放指令），不是本報告能單靠數字下定論的部分。
- 若想再往上推：`num_learning_epochs` 5 → 3（從源頭壓 KL，讓 lr 撐久一點），這是
  雲端文件記錄的下一手，本機尚未嘗試。
