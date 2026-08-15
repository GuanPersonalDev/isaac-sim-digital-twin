# RL 訓練重播指南（Demo 用）

給之後做 Demo 時用的操作手冊：怎麼重播「訓練前 vs 訓練後」的擊球結果、怎麼看訓練
過程的 reward 曲線。對應的訓練紀錄在 `docs/local-training-2026-08-12.md`。

⚠️ **先讀這段：本機 raw checkpoint 是本地檔案，沒有進 git。**
`training/outputs/`（含 `model_0.pt` ~ `model_200.pt`、TensorBoard event 檔）被
`.gitignore` 排除——這是專案既有慣例（體積大，且可由設定檔重現）。下面「方式一」
的 GUI 回放需要這些原始檔案；如果你要在**這台機器以外**或**清過 `training/outputs/`
之後**做 Demo，只能用方式二（已進 git 的 TorchScript／ONNX），或重新跑一次訓練
（見 `docs/local-training-2026-08-12.md`〈訓練設定〉的指令，同一套超參數，
`env seed=42` 固定，走勢會高度相似但不保證逐位元相同）。

目前這台機器上可用的 checkpoint（`training/outputs/logs/rsl_rl/billiard/2026-08-12_08-55-10/`）：

| 檔案 | 對應階段 |
|---|---|
| `model_0.pt` | 訓練前（隨機初始化，亂打） |
| `model_50.pt` | 訓練中段初期 |
| `model_100.pt` | 迭代 100 檢查點 |
| `model_150.pt` | 訓練中段後期 |
| `model_200.pt` | 迭代 200 決策點（收斂，訓練完） |

## 方式一：Isaac Sim GUI 直接回放（畫面最完整，推薦用於錄 Demo）

每個 checkpoint 都可以直接開 Isaac Sim 視窗看真的物理模擬，不需要先匯出：

```powershell
$env:PYTHONPATH="C:\Users\Kuan\isaac-project\isaac-sim-digital-twin"
$env:ACCEPT_EULA="Y"; $env:PRIVACY_CONSENT="Y"; $env:OMNI_KIT_ACCEPT_EULA="YES"; $env:ISAACSIM_ACCEPT_EULA="YES"
C:\Users\Kuan\isaac-project\venv\Scripts\Activate.ps1
cd C:\Users\Kuan\isaac-project\IsaacLab

.\isaaclab.bat play --rl_library rsl_rl --task Isaac-Billiard-v0 `
    --external_callback billiard_rl.tasks.register_external_tasks `
    --checkpoint "C:\Users\Kuan\isaac-project\isaac-sim-digital-twin\training\outputs\logs\rsl_rl\billiard\2026-08-12_08-55-10\model_200.pt" `
    --num_envs 4
```

- `--checkpoint` 換成上表任一個 `.pt` 路徑，就能看該階段的擊球行為。
- `--num_envs 4`（或更多）可以一次看多顆球的擺位/角度變化；改 `1` 只看單一開球。
- **不要加 `--headless`**——加了就沒有畫面，只是空跑。
- 視窗開啟後就是即時物理模擬，錄影 / 截圖都直接對這個視窗操作，`Ctrl+C` 結束。
- 想做「訓練過程」的漸進式對照（不只 0 vs 200），依序播 `model_0` → `model_50` →
  `model_100` → `model_150` → `model_200`，剪接起來就是完整的學習過程演進——
  這正是 `training/README.md`〈Demo 素材〉章節設計 checkpoint 保留策略的目的。
- 每次執行都會在 checkpoint 同目錄的 `exported/` 下重新產生
  `policy.pt`／`policy.onnx`，會覆蓋前一次的匯出，錄 Demo 不需要理會這個副作用。

## 方式二：用已進版控的 TorchScript／ONNX（可攜，不依賴本機 raw checkpoint）

`models/rl/billiard/` 底下的 `iter0/` 與 `iter200/`（=頂層 `policy.pt`）已經 commit
進 git，換機器、清過 `training/outputs/` 都還在。缺點是**不能直接開 Isaac Sim GUI
回放**——這兩個檔案是純推論用的匯出格式，沒有 rsl_rl 的 runner 可以接動 Isaac Lab
的 play 流程。適合的用途：

- **未來 `ModelController`（#127）落地後**，直接讀這兩個資料夾當「訓練前 / 訓練後」
  的示範模型，不需要重新訓練或重新匯出。
- **純推論驗證**（不需要畫面時）：

  ```python
  import torch
  policy = torch.jit.load("models/rl/billiard/iter200/policy.pt")  # 或 iter0
  policy.eval()
  obs = torch.zeros(1, 21)   # 21 維：18 球位 + 母球 XY + max_offset
  action = policy(obs)        # 6 維，原始輸出，未裁切
  ```

  ⚠️ 這裡拿到的動作是**無界、未裁切**的原始輸出，實際要用在模擬/展示上必須先過
  `core.services.rl_action_decoder.decode_rl_action()`，細節見
  `models/rl/billiard/README.md`。

## 看訓練過程本身（reward 曲線）

不需要重新訓練或重新讀 event 檔，兩份現成產出都在 `models/rl/billiard/`：

- `run_2026-08-12_training_curve.png`——現成的圖，`mean_reward` / `spread x20` /
  `aim x20` 三條曲線，標出 it=100／200 兩個檢查點，可以直接放進 Demo 簡報。
- `run_2026-08-12_metrics.csv`——完整 18 欄原始數值（212 個 iteration），要重畫圖、
  換配色、加標註都從這份資料出發，不需要重新開 Isaac Sim。

如果想要互動式的 TensorBoard（拖曳縮放、比對不同 run）：

```powershell
C:\Users\Kuan\isaac-project\venv\Scripts\python.exe -m tensorboard.main `
    --logdir "C:\Users\Kuan\isaac-project\isaac-sim-digital-twin\training\outputs\logs\rsl_rl\billiard" `
    --port 6006
```

開瀏覽器連 `http://localhost:6006`。這一步一樣依賴本機 `training/outputs/`（未進 git）。

## 疑難排解：Windows Remote Desktop 連線時開不出 GUI 視窗（2026-08-15 實測）

這台機器透過 **Windows 遠端桌面（RDP）**連線操作，用「方式一」開 GUI 回放時遇到
三個障礙，記錄下來給下次連這台機器做 Demo 錄影的人參考：

1. **PowerShell 無法啟用 venv**：`Activate.ps1` 報 `UnauthorizedAccess`（執行原則擋
   指令碼）。解法是只在目前這個視窗放行，不動系統全域設定：

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
   C:\Users\Kuan\isaac-project\venv\Scripts\Activate.ps1
   ```

2. **`isaacsim` 單獨開視窗時工具列一開始看不見**：不是 RDP 顯示驅動的問題，是
   Omniverse Kit 第一次啟動要編譯 shader/UI，編譯完工具列就會自己出現。等就好，
   不用中斷重開。

3. **`isaaclab.bat play` 完全不開視窗（但行程有 CPU/GPU 使用率，不是卡死）**：
   這是本專案用的 IsaacLab 版本（`release/3.0.0-beta2`）**行為改版**，不是本機
   環境問題——`--headless` 舊參數已棄用，**新版預設值本身就是 headless**，
   不加 `--viz` 就完全不會建立視窗（見 `IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py:388-390`）。
   之前 `docs/local-training-2026-08-12.md` 記錄的回放指令是舊版語法，這裡更新：

   ```powershell
   .\isaaclab.bat play --rl_library rsl_rl --task Isaac-Billiard-v0 `
       --external_callback billiard_rl.tasks.register_external_tasks `
       --checkpoint "<checkpoint 路徑>\model_0.pt" `   # 或 model_200.pt
       --num_envs 4 `
       --viz kit
   ```

   關鍵是**必須加 `--viz kit`** 才會開 Isaac Sim GUI 視窗；不加就是安靜地在背景
   跑完整個模擬迴圈，沒有任何錯誤訊息，容易誤判成「卡住」。

   驗證結果：加上 `--viz kit` 後，`model_0.pt`（訓練前，隨機亂打）與
   `model_200.pt`（訓練後，收斂）皆可正常開窗回放，對比清楚可用於 Demo 錄影。

### 多桌（高 env 數）Demo 素材的可行性評估（尚未實測，僅記錄評估結論）

曾評估「Demo 用 1024 桌同時開球」的畫面，結論是**不建議直接渲染全部 1024 桌**：

- 訓練時的 1024 env 是 headless、無 camera、純物理負載，GPU 消耗很輕；但 GUI
  即時渲染是完全不同量級的工作（1024 桌 × 11 個物件 ≈ 11,264 個渲染物件），
  從未實測過，且畫面上 1024 張小桌子人眼根本看不出個別球局。
- 用 `--max_visible_envs` 參數（見 `app_launcher.py:603-606`）：物理照樣跑滿
  設定的 env 數，只渲染其中一部分，兼顧「大規模平行」的視覺印象與畫面可讀性。
  建議指令雛型：`--num_envs 1024 --viz kit --max_visible_envs 16`（從小數字開始
  漸進測試 FPS／記憶體，不要一開始就衝大數字）。
- `self.viewer.eye = (8.0, 0.0, 5.0)`（`billiard_rl_env_cfg.py`）是為單桌近拍調的
  相機位置，`env_spacing=4.0` 下多桌網格會攤得很開，要拍多桌全景需要在 Isaac Sim
  視窗裡手動把相機拉遠/拉高，不是改程式碼。
- **記憶體觀察**：此機器（RTX 4090）僅開 4 桌 GUI 回放時，任務管理器「系統記憶體
  ／共用 GPU 記憶體」已到 83%（非「專用 GPU 記憶體／VRAM」，兩者意義不同——
  前者代表主機 RAM 或 VRAM 溢出借用 RAM，後者才是 GPU 顯存本身），暗示往上加
  env 數前應先盯緊這個數字，不要一次跳到大數字。

## 快速對照表：我要做什麼、該用哪個方式

| 目的 | 用哪個 |
|---|---|
| 錄「訓練前 vs 訓練後」的擊球畫面 | 方式一，`model_0.pt` 與 `model_200.pt` 各跑一次 |
| 錄完整學習過程的漸進式對照 | 方式一，依序播 `model_0/50/100/150/200` |
| Demo 簡報要放 reward 曲線圖 | 直接用 `run_2026-08-12_training_curve.png` |
| 換機器 / 清過 `training/outputs/` 後還要示範模型 | 方式二，`models/rl/billiard/` 底下的匯出檔 |
| `ModelController`（#127）實作測試 | 方式二 |
