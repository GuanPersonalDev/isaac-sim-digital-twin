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

## 快速對照表：我要做什麼、該用哪個方式

| 目的 | 用哪個 |
|---|---|
| 錄「訓練前 vs 訓練後」的擊球畫面 | 方式一，`model_0.pt` 與 `model_200.pt` 各跑一次 |
| 錄完整學習過程的漸進式對照 | 方式一，依序播 `model_0/50/100/150/200` |
| Demo 簡報要放 reward 曲線圖 | 直接用 `run_2026-08-12_training_curve.png` |
| 換機器 / 清過 `training/outputs/` 後還要示範模型 | 方式二，`models/rl/billiard/` 底下的匯出檔 |
| `ModelController`（#127）實作測試 | 方式二 |
