# training/

雲端 RL 訓練（#224）。平台是 RunPod **Secure Cloud／EU-RO-1／RTX A4500**
＋ 100GB Network Volume。

這一層屬於部署／維運，不在 `docs/architecture-spec.md` 的分層規則內，因此與
`core/`、`extension/` 平行，不互相依賴。

> **2026-08-02 大幅修訂。** 先前版本記載的「直接使用官方映像
> `nvcr.io/nvidia/isaac-lab:3.0.0-beta2`＋每次開機跑 `bootstrap_runpod.sh`」路線
> **已實測失敗並廢棄**，原因見下方〈為什麼不用官方映像〉。
>
> **2026-08-06 收尾。** 舊路線的兩個產物已處理：`docker/docker-compose.yml`
> 已刪除（它綁在那個 ENTRYPOINT 蓋不掉的官方映像上，沒有可改寫的餘地）；
> `scripts/bootstrap_runpod.sh` 已改寫成 **Volume 從零建置**腳本，不再是開機腳本
> ——開機的工作由 `/workspace/setup.sh` 負責。

## 環境現況（2026-08-02 建置完成並驗證）

| 項目 | 值 |
|---|---|
| 平台 | RunPod **Secure Cloud**（Network Volume 僅 Secure Cloud 支援）|
| Datacenter | **EU-RO-1**（Volume 綁死區域，建立後不可更改）|
| GPU | **不固定**——EU-RO-1 的庫存很不穩，每次開機挑當下有的同級卡。已實測可用：RTX A4500（20GB，$0.25/hr）、**RTX PRO 4500 Blackwell（32GB，sm_120）**。挑選條件只有兩個：在 EU-RO-1、VRAM ≥ 20GB |
| Network Volume | `billiard-isaac-training` **100GB**，掛載 `/workspace`（原 50GB，2026-08-03 擴容；只能加不能減）|
| Container Disk | 60GB，**停機即清空** |
| 基礎映像 | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` |
| Python | **3.12**（Isaac Lab 硬限制 `>=3.12,<3.13`；Ubuntu 24.04 系統 Python 剛好命中）|
| isaacsim | 6.0.1.0（pip 安裝）|
| torch | 2.10.0+cu128 / CUDA 12.8（驅動視配到的卡而定，實測 570.195.03 與 580.173.02 皆可）|
| Isaac Sim 冷啟動 | 約 **58 秒**（A4500 實測兩次相同，與 shader 快取無關）|

### ⚠️ 換卡後必做：驗證 compute capability

停機會把 GPU 釋放回池子，重開時原型號常常已被別人佔走（訊息是
`Your Pod's GPUs are no longer available`）。RunPod **沒有**「更改既有 Pod 的
GPU 型號」功能，只能從 Storage 頁的 volume 直接部署新 Pod（區域自動鎖定），
舊 Pod 再 Terminate。**Network Volume 絕對不要刪。**

換到不同世代的卡（例如 Ampere → Blackwell）之後，**`torch.cuda.is_available()`
回 True 完全不足以證明能用**。torch 若沒為那張卡的 compute capability 編 kernel，
會在實際運算時才炸，或更糟——靜默算出錯誤結果。

```bash
python -c "import torch; print(torch.cuda.get_device_capability()); print(torch.cuda.get_arch_list()); x=torch.randn(4096,4096,device='cuda'); print((x@x).sum().item())"
```

`get_arch_list()` 必須包含該卡的 `sm_XXX`，且 matmul 要跑得出**量級合理**的數字
（4096² 標準常態相乘求和，理論 σ≈262k，落在 ±1σ 內才正常）。2026-08-06 在
RTX PRO 4500 Blackwell 上實測通過：`sm_120` 在 arch list 內，matmul 正確。

### Pod 環境變數（四個都必要）

`ACCEPT_EULA=Y`、`PRIVACY_CONSENT=Y`、`OMNI_KIT_ACCEPT_EULA=YES`、`ISAACSIM_ACCEPT_EULA=YES`

缺 `ISAACSIM_ACCEPT_EULA` 會卡在「Isaac Sim Additional Software and Materials License
must be accepted」而容器直接結束。

### Pod Start command（裝 sshd 並保持容器存活）

```
bash -c 'apt update; DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server; mkdir -p ~/.ssh; chmod 700 ~/.ssh; echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; mkdir -p /var/run/sshd; service ssh start; sleep infinity'
```

## `/workspace` 佈局

```
/workspace/
├── venv/                       Python 環境（約 50GB，含 isaacsim + torch + Isaac Lab）
├── IsaacLab/                   Isaac Lab 原始碼（release/3.0.0-beta2）
├── isaac-sim-digital-twin/     本專案
├── setup.sh                    開機腳本
└── pip-freeze-working.txt      已知可用狀態的回退點
```

**venv 在 Network Volume 上，所以 Python 套件停機後仍在，不需要每次重裝。**
這是與舊路線最大的差別。

`setup.sh` 放在 volume 上而不是 repo 裡，是因為它要在「repo 還沒 clone 下來」
的時候就能用。它的內容由 `scripts/bootstrap_runpod.sh` 產生——**改了 volume 上的
`setup.sh` 記得同步回那支腳本**，否則重建 volume 時改動會失傳。

## Volume 重建

Volume 被回收（餘額不足時真的會發生，checkpoint 會一起消失）、或要換區重來時：

```bash
bash /path/to/bootstrap_runpod.sh
```

它會依序做完系統套件 → venv → isaacsim → Isaac Lab → 專案 repo → 產生
`setup.sh` → 存回退點與驗證，每個階段都會先檢查是否已完成，可以安全重跑。

**安裝順序寫死在腳本裡是有原因的**：isaacsim 必須在 `isaaclab.sh --install`
之前，顛倒的話 isaacsim 會把 torch 換成 CUDA 13 版本（見〈踩過的坑〉第 2 點）。

⚠️ 這支腳本尚未在真實的空 volume 上驗證過，首次使用請逐段確認輸出。

## 為什麼不用官方映像

`nvcr.io/nvidia/isaac-lab:3.0.0-beta2` 的 **ENTRYPOINT 會強制啟動 Isaac Sim Streaming**，
而 RunPod 的 Container Start Command 覆寫的是 CMD，蓋不掉 ENTRYPOINT。結果是容器啟動 →
跑 Streaming → 環境不合而結束 → 容器死掉重啟，SSH 一進去就被踢出來。

診斷方法：把 start command 設成 `bash -c 'echo HELLO_FROM_START_COMMAND; sleep infinity'`，
若 Container log 裡沒有出現該字串，即確認 ENTRYPOINT 吃掉了 CMD。

另外 Network Volume 掛在 `/workspace` 會**遮蔽**官方映像內的 `/workspace/isaaclab`
（小寫），而 Isaac Lab 在該映像內是 editable install，路徑被遮蔽後 `import isaaclab`
直接失效。

改用乾淨的 CUDA 基礎映像 + pip 安裝到 volume 上的 venv，沒有這些問題。

## 使用

### 每個週末：開機

1. RunPod 網頁 → Pods → **Start**
2. Pod 卡片 → **Connect** → 複製 SSH 指令，在本機終端機執行：

```bash
ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

3. **第一件事**：

```bash
source /workspace/setup.sh
```

看到 `torch 2.10.0+cu128 cuda True` 就代表環境正常。

`setup.sh` 做的事：apt 重裝 `git tmux python3-dev build-essential`（container disk 停機
清空）→ 設 `PIP_CACHE_DIR=/tmp/pipcache` → 啟用 venv → `git pull` 專案 → 印 torch/cuda 狀態。

> ⚠️ **只能用 `source`，不能用 `./setup.sh`。** venv 啟用必須作用在當前 shell；用 `./`
> 會開子行程，activate 完就隨子行程消失，但畫面上看起來「成功了」。檔案刻意不給執行位元
> 作為防呆。

### 每個週末：啟動訓練

**一定要放進 tmux。** SSH session 綁在連線上，斷線前景程序就死——而訓練要跑整整一週。

```bash
tmux new -s train
# 在 tmux 內啟動訓練
```

- 卸離（讓它繼續跑）：`Ctrl-B` 放開，再按 `D`
- 回去看：`tmux attach -t train`
- 確認還活著：`tmux ls`

**判斷規則**：超過 1–2 分鐘的作業一律進 tmux；訓練不可省。

### 每個週末：啟動訓練時一併掛上 GPU watchdog

`scripts/gpu_watchdog.sh` 在 GPU 連續閒置達門檻（預設 30 分鐘）後自動停止 Pod。

**它解決的是這個情境**：訓練在週二崩潰或跑完，你到週六才發現 —— 4 天 × 24h × $0.25 ≈ **$24**，
等同整個專案預算。平日 0.5h 巡檢無法可靠攔截。

```bash
tmux new -s train
/workspace/isaac-sim-digital-twin/training/scripts/gpu_watchdog.sh &
# 接著啟動訓練
```

> ⚠️ **不要放進 `setup.sh` 開機自動啟動。** 週末你 SSH 進去寫 code、還沒開始訓練時
> GPU 是閒置的，開機就啟動會讓 Pod 在 30 分鐘後自己關掉。
>
> 腳本本身還有一道保險：**必須先偵測到 GPU 真的忙碌過（armed）才會開始倒數**，
> 所以誤啟動而訓練沒跑起來時它不會關機。但正確用法仍是「跟著訓練啟動」。

**為什麼自動停機在這裡是安全的**：Pod「停止」不是「終止」，`/workspace` 完整保留。
誤判的代價是重啟時間（約 1 分鐘 + Isaac Sim 冷啟動 58 秒），不是資料。

可調參數（環境變數）：`IDLE_LIMIT`（分鐘，預設 30）、`UTIL_THRESHOLD`（%，預設 5）、
`POLL_INTERVAL`（秒，預設 60）、`HOURLY_RATE`（僅用於 log 顯示累計花費）、`LOG`、`DRY_RUN`。

首次使用建議先試跑，確認不會誤觸發：

```bash
DRY_RUN=1 IDLE_LIMIT=2 ./gpu_watchdog.sh
```

**前置需求：只要一個環境變數。** 停機走 REST API（`scripts/runpod_api.py`，只用
Python 標準庫），不需要安裝任何東西。

在 Pod 的 Environment Variables 加一筆 `WATCHDOG_API_KEY`，值是 RunPod
**Settings → API Keys** 建立的金鑰，scope 必須勾選 **Pods 的讀與寫**。

⚠️ **不可使用 `RUNPOD_API_KEY` 這個名稱。** 它是平台保留字，每顆 Pod 啟動時都會
被自動注入一把 **pod-scoped key**，權限不含管理 Pod 生命週期（實測 GraphQL 回
Unauthorized、REST 回 403），而且會覆蓋你在 Pod 設定裡填的同名值——症狀是
「金鑰格式完全正確卻一直 403」，極難追查。

⚠️ 金鑰**不可寫進任何檔案**，本 repo 是 public。直接填在 Pod 設定即可（Pod 設定
不在 git 裡）。RunPod 的 Secret 功能只影響它在 RunPod UI 上是否明文顯示——注入
容器後兩者都是普通環境變數，`env` 一打就看得到，在 Pod 內完全等價。

改設定後**必須重啟 Pod**，環境變數只在啟動時注入。

驗證金鑰（唯讀，不會動到 Pod）：

```bash
python /workspace/isaac-sim-digital-twin/training/scripts/runpod_api.py check
```

watchdog 啟動時也會自動跑這個檢查，**驗不過就直接退出**——不會讓你以為有保險，
卻在閒置滿 30 分鐘後才發現停不了機。

### 為什麼不用 runpodctl

1. 每顆 Pod 都預裝了 runpodctl，但它配的是那把 pod-scoped key，連唯讀的
   `get pod` 都回 Unauthorized。
2. runpodctl 只打 GraphQL endpoint；新版 scoped key 的 GraphQL 與 REST 權限是
   分開勾選的，可以出現「GraphQL 無權、REST 可用」的組合（實測就是如此）。
3. `runpodctl config` 在 `/root/.runpod/` 不存在時不會自建目錄，直接報
   「Config File not found」；就算存檔成功，它還會順手同步 SSH key 到雲端，
   那需要帳號層級權限，失敗會讓整個指令回傳非零 exit code——用 `&&` 串後續
   指令時會被靜默吃掉。
4. runpodctl 與 curl 都不在基礎映像內，container disk 停機即清空，等於每次開機
   都要重裝；而 apt 會跟 Pod 的 start command 搶 dpkg lock。

venv 在 network volume 上持久化，用它的 python 打 REST 沒有任何安裝成本。

log 寫在 `/workspace/watchdog.log`（volume 上，停機後仍在，可事後查為什麼被關）。

### 平日巡檢（0.5h）

```bash
source /workspace/setup.sh
tmux attach -t train                                    # 還在不在跑
ls -lh /workspace/isaac-sim-digital-twin/training/outputs   # checkpoint 有沒有長出來
nvidia-smi                                              # GPU 有沒有掉線
```

reward 曲線用 TensorBoard 看（需 Pod 建立時已開 HTTP Port `6006`）：

```bash
tensorboard --logdir <輸出目錄> --port 6006 --host 0.0.0.0
```

> **`--host 0.0.0.0` 不可省**（`--bind_all` 亦可）。服務只綁 localhost 的話 RunPod 的
> proxy 連不進來，這是開了 port 卻連不上最常見的原因。

費用在 RunPod 網頁的 Billing 頁看。**Pod 停機後 Network Volume 仍然計費**
（100GB × $0.07/GB/月 = $7/月）。GPU $0.25/hr 只在開機時計。

專案期間（8/02–9/19，約 1.6 個月）總估：GPU 約 50h ≈ $12.5 ＋ Volume ≈ $11.2 = **約 $24**。
餘額不足時 Volume 可能被回收並連 checkpoint 一起消失，建議維持 $30–40 餘額。

### 停機

RunPod 網頁 → **Stop**。

| 位置 | 停機後 |
|---|---|
| `/workspace`（Network Volume）— 含 **venv**、專案、IsaacLab、checkpoint | **保留** |
| Container disk（`/`、`/tmp`、apt 裝的套件、sshd、symlink） | **清空** |

所以**任何要留的東西都必須在 `/workspace` 底下**。`/tmp/pipcache` 是刻意放在
container disk 的（見〈踩過的坑〉第 4 點）。

## 只有 log，沒有畫面

雲端是 headless，容器裡沒有顯示伺服器，**看不到 Isaac Sim 的 GUI 視窗**。能看的就是：
terminal 的 stdout／stderr、TensorBoard 曲線、寫進 `/workspace` 的 checkpoint 與檔案。

headless 下 `libGL.so.1` / `libXt.so.6` 缺失的錯誤**是正常的**（RTX 渲染走 Vulkan
不走 GL），可以忽略。

要**看**物理行為，走的是另一條路：把訓練好的 policy 取回本機（#226），在本機
Isaac Sim GUI 播放（#227）。排程把 Milestone B（手臂執行）整段安排在本機跑，
就是因為那部分必須肉眼看物理行為，雲端做不到。

> Isaac Sim 本身有 WebRTC livestream 功能，理論上可以把畫面推到瀏覽器，但它需要 UDP
> port，而 RunPod 的 proxy 只轉 TCP。**不要為了看畫面把時間花在這上面**——訓練是 headless
> 跑 1024 個環境，本來也沒有「一個畫面」可看。

## Demo 素材：訓練開始前就要設好

Block 12 的影片要呈現「訓練過程」，但雲端 headless、沒有訓練當下的畫面可拍。
唯一可行的做法是**用不同訓練階段的 checkpoint 在本機各回放一次**，剪成前後對照
（iteration 0 亂噴 → 最終乾淨散開）。

⚠️ **以下四項全部是訓練開始前的設定，事後補不回來。** 等 #226/#227 那個週末才發現
沒設，代價是重開 pod 重跑一輪。

### 1. 評估場景必須完全固定

不同 checkpoint 的回放要能對照，前提是**除了 policy 以外每個變因都一樣**：

- rack 擺位鎖死用 `BREAK_SHOT_POSITIONS`（已定案的固定值，非隨機生成）
- eval 用的 seed 與訓練 seed 分開，且固定
- 物理參數、`RollingResistanceService` 係數不得在訓練期間變動
- **`max_offset` 條件值必須固定**（21 維 observation 的新欄位，見 #222/#225）——
  這一項會直接改變 policy 行為，回放時若不固定，對照完全失去意義

### 2. checkpoint 要保留多個，且必須在 volume 上

- checkpoint 週期由 `rl_task/billiard_rl/tasks/manager_based/billiard_rl/agents/rsl_rl_ppo_cfg.py`
  的 `save_interval` 決定（目前 50，是 Isaac Lab 模板的預設值；目標值 100 待 #123 定案）
- **需確認所選框架不會自動 rotate 掉舊 checkpoint**（很多框架預設只留最近 N 個）
- 輸出目錄必須在 `/workspace` 底下，否則停機就沒了

預計要保留可回放的階段：**iteration 0 / 200 / 1000 / 最終**（實際數值待首輪跑完後
依收斂速度調整）。

### 3. ⚠️ 中間 checkpoint 也要在雲端 export 成 TorchScript

**這是最容易漏、代價最高的一項。**

rsl_rl 存的 `model_<iter>.pt` 是含 optimizer state 的訓練檔，要靠 rsl_rl 才載得動。
但本機 Isaac Sim **沒有裝 rsl_rl**——#227 的設計是用 Isaac Sim 內建 PyTorch 載
**TorchScript** 格式的 `exported/policy.pt`。

`play` 每跑一次就會自動匯出 `<run>/exported/policy.pt` 與 `<run>/exported/policy.onnx`。
所以每一個要拿來回放的中間 checkpoint，都必須**趁 pod 還活著時就各跑一次 `play`**。
等訓練結束、只下載了最終 policy 才想起來，就得重開 pod、重跑 export。

> 實證：`2026-08-08` 的兩輪訓練目錄底下只有 `model_*.pt`，**沒有 `exported/`**——
> 那幾個 checkpoint 現在要回放就得重跑 export。

#### 匯出檔的實際性質（2026-08-09 於 pod 實測，rsl_rl 5.x）

⚠️ 以下三項是打開檔案量出來的，不是從 isaaclab 舊版原始碼推論的。改動 rsl_rl 版本
或 `agents/rsl_rl_ppo_cfg.py` 的 `obs_normalization` 後**必須重驗**。

**① 目前沒有任何正規化，但「餵原始觀測即可」這個結論成立。**

TorchScript 的 `forward` 裡確實有 `obs_normalizer`，但 `rsl_rl_ppo_cfg.py` 的
`obs_normalization=False`，所以它是 `Identity()`——`named_buffers()` 是空的，
沒有 `running_mean` / `running_var`。ONNX 圖裡也只有 `Gemm ×3 + Elu ×2`，
沒有任何正規化 op。

> 舊版本文寫的是「normalizer 已打包進模型」——結論碰巧對，理由是錯的：
> 現在能餵原始觀測是因為**根本沒有正規化**。#123 調參若把 `obs_normalization`
> 打開，匯出的模型才會真的包一個有作用的 normalizer（屆時仍餵原始觀測，
> 但**不同 checkpoint 的 normalizer 統計量不同**，回放對照要留意）。

**② 匯出的動作是無界的，而且是 deterministic。**

`forward` 是 `obs_normalizer → mlp → deterministic_output`，回傳高斯分布的**平均值**
而不是取樣，也**沒有 tanh 或 clip**。實測全 50 的輸入會得到
`[-4.6, -1.9, -4.8, 5.3, 10.9, -1.7]`——遠超出正規化域 `[-1, 1]`。

**所以回放端不能自己解讀 policy 的輸出**，一定要走
`core.services.rl_action_decoder.decode_rl_action()`：前四維的 clip 與偏移兩維的
圓形裁切都在那裡（#225），繞過它會把越界值直接反正規化成越界的物理量。

**③ ONNX 的 batch size 固定為 1。**

輸入 `obs` 形狀 `[1, 21]`、輸出 `actions` 形狀 `[1, 6]`，都是固定值不是動態軸。
onnxruntime 實測：`batch=1` OK，`batch=4` 與 `batch=64` 都是
`INVALID_ARGUMENT: Got invalid dimensions for input: obs`。

單機器人 Demo 夠用，要批次推論就得重新匯出。TorchScript 版沒有這個限制。

### 4. TensorBoard event 檔要一起帶回來

reward 曲線建議**下載 event 檔回本機自己畫圖**，不要螢幕錄影 TensorBoard UI——
自己畫的圖乾淨、可控、可標註 A-CP 判定點。event 檔同樣要確認寫在 `/workspace`。

### 誠實標註的界線

影片裡若要放「多環境並行」的畫面，那是本機開少量環境（16/64）的**示意**，
不是實際訓練畫面（實際訓練 headless、1024 環境、無渲染）。**必須在影片或說明中
標明**——技術面試會追問這一點。

## 踩過的坑（重建環境時直接避開）

1. **不要用官方映像**——見上方〈為什麼不用官方映像〉。
2. **安裝順序不可顛倒**：必須先
   `pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com`，
   再 `cd /workspace/IsaacLab && ./isaaclab.sh --install`。順序反了 isaacsim 會把 torch
   換成 2.11.0（CUDA 13），而驅動 570.x 不支援 CUDA 13，`torch.cuda.is_available()` 會變 False。
3. **cu12/cu13 雙堆疊**：`nvidia-cublas-cu12` 與 `nvidia-cublas`（CUDA 13 世代拿掉後綴）
   是不同套件名稱，pip 不會互相取代 → venv 曾膨脹到 66GB。目前仍有約 16 個 CUDA 13
   孤兒套件（約 10–15GB）待清。
4. **pip cache 不要放 `/workspace`**：會吃掉 Volume 配額（曾觸發
   `Errno 122 Disk quota exceeded`）。放 `/tmp/pipcache`（container disk）即可——
   venv 本身持久化，本來就不需要重裝。
5. **`python3-dev` 必裝**，否則 `imgui` 編譯失敗（缺 `Python.h`）擋住 `isaaclab_teleop`。
6. **isaacsim 與 isaaclab 有版本 pin 衝突**（`websockets`/`psutil`/`click`/`coverage`）——
   **不要手動對版本**，硬降 `websockets` 會反過來弄壞 `viser`/`rerun`。用 smoke test
   實測是否影響使用路徑。
7. **快取不需重導**：實測 Omniverse 沒把快取寫到 HOME（`/root/.cache/ov` 僅 4K），
   58 秒冷啟動與 shader 快取無關，不要花時間做 symlink 重導。
8. **RunPod proxy SSH 會忽略遠端指令參數**，只給互動 shell；帶指令需 `-t`（要求 PTY）。
9. **Web Terminal 打不開**是因為自訂映像沒有 sshd/RunPod agent——用 SSH（start command
   已處理 sshd 安裝）。
10. **開機後太快 source `setup.sh` 會整個壞掉**：Pod 的 start command 正在
    `apt-get install openssh-server`，與 `setup.sh` 的 apt 搶同一把 dpkg lock
    （`Could not get lock /var/lib/dpkg/lock-frontend`）。連鎖反應是 `git` 與
    `python3-dev` 沒裝回來 → **系統 python3.12 不存在** → venv 的 `bin/python`
    變成斷掉的 symlink，症狀是 `(venv)` 前綴有出來但 `python: command not found`。
    等 `pgrep apt-get` 消失再 source 即可（`setup.sh` 已內建這個等待）。
    **不要 kill 對方，也不要刪 lock 檔**——中斷 dpkg 會留下半裝狀態，之後每個
    apt 指令都得先 `dpkg --configure -a`。
11. **`curl` 不在基礎映像內**（container disk 停機清空後更是如此）。臨時要打 HTTP
    用 `wget -S ... 2>&1 | grep HTTP/`，或用 venv python 的 `urllib`。所以停機腳本
    刻意不依賴 curl——見〈為什麼不用 runpodctl〉。
12. **`RUNPOD_API_KEY` 是平台保留字**，每顆 Pod 都會被自動注入一把權限不足的
    pod-scoped key，且會覆蓋你在 Pod 設定填的同名值。詳見上方 watchdog 的前置需求。

## Smoke test

```bash
cd /workspace/IsaacLab && ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py
```

⚠️ **這個腳本是無限迴圈，不會自己結束**（headless 下沒有視窗可關，
`simulation_app.is_running()` 永遠為 True）。看到 `[INFO]: Setup complete...` 就按
`Ctrl+C`。想確認它真的在跑，另開 tmux window：

```bash
wc -l /workspace/IsaacLab/logs/docker_tutorial/log.txt   # 行數持續增加 = 物理在 step
```

## 環境回退

`/workspace/pip-freeze-working.txt` 是「已知可用」狀態的快照。清理套件或加裝東西弄壞了
可以據此還原。**動任何套件之前先重存一份。**

## 尚未完成

- **`bootstrap_runpod.sh` 尚未在真實的空 volume 上跑過**（2026-08-06 改寫）。
  內容是把〈踩過的坑〉逐條可執行化，但整條建置流程沒有重跑驗證過——真正需要它的
  時候（volume 被回收）風險最高，首次使用請逐段確認輸出。
- **`BilliardEnv` 的場景與 MDP 內容**（#121 A／B 組）：`run_train.sh` 已接上真正的
  訓練呼叫並跑通，但 `billiard_rl_env_cfg.py` 仍是 Isaac Lab 模板帶來的 cartpole
  內容，還不是撞球場景。
- **`gpu_watchdog.sh` 的停機動作尚未在真實 Pod 上驗證**（2026-08-06 更新）。
  已確認：`$RUNPOD_POD_ID` 有被注入；自建金鑰對 REST `GET /v1/pods` 回 **200**；
  腳本邏輯用假 `nvidia-smi`／假 python 測過四條路徑（金鑰驗證失敗即退出、
  找不到 API script 即退出、未武裝不誤觸發、先忙後閒觸發停機且參數正確）。
  待實測：`POST /v1/pods/<id>/stop` 是否真的停得了機（write 權限）。
  這一項沒過之前**不要開整週的訓練**——那正是 watchdog 要防的情境。
- **CUDA 13 孤兒套件清理**（#224 未完項，選配）。

## 注意事項

**憑證不進映像。** 根目錄 `.dockerignore` 已排除 `.env`、`*.key`、`secrets/`。
實測 `nvcr.io` 的 isaac-sim 系列映像**公開可拉，不需要 NGC 金鑰**。

**換行符號。** `.sh` 若帶 CRLF 進容器會噴 `bash: \r: command not found`，且訊息
看不出原因。repo 的 `core.autocrlf=true` 會在 checkout 時把工作區檔案轉成 CRLF，
所以 `training/.gitattributes` 明確標記 `eol=lf`——**這對 RunPod 特別重要**，
因為雲端是直接 clone，若沒有這個設定，pod 上拿到的就會是 CRLF 版本。
