# training/

雲端 RL 訓練（#224）。目標平台是 RunPod Community RTX 4090，本機只做 smoke test。

這一層屬於部署／維運，不在 `docs/architecture-spec.md` 的分層規則內，因此與
`core/`、`extension/` 平行，不互相依賴。

## 核心原則：程式碼不進映像

雲端與本機**都直接使用官方映像** `nvcr.io/nvidia/isaac-lab:3.0.0-beta2`，
不自建映像、不把程式碼烘焙進去。

| | 程式碼來源 | 改一行 code 的代價 |
|---|---|---|
| RunPod | `bootstrap_runpod.sh` 做 git clone/pull | `git pull`，數秒 |
| 本機 | compose bind mount repo 根目錄 | 不需重建，直接生效 |

兩邊的 `/workspace/billiard` 內容因此等價。若改成把程式碼 COPY 進映像，
每次改動都要重建並推送約 25GB 映像，與「週末寫 code → 丟雲端跑」的節奏不相容。

官方映像已預裝 Isaac Lab 於 `/workspace/isaaclab`（**小寫**，#224 body 寫的
`/workspace/IsaacLab` 是錯的），內含 Isaac Sim **6.0.1**（比本機 6.0.0 新一個
patch，查 API 時注意落差）。headless only。

實測規格（`3.0.0-beta2`）：

| 項目 | 值 |
|---|---|
| 映像大小 | 31.8GB（RunPod container disk 建議 80GB） |
| 執行使用者 | uid/gid 1000（`ubuntu`） |
| `HOME` | `/root`，已 chown 給 uid 1000，可寫 |
| Python | **沒有 `python` 也沒有 `python3`**，只有 `/isaac-sim/python.sh`（`/workspace/isaaclab/_isaac_sim` 是指向 `/isaac-sim` 的 symlink） |

腳本裡的 Python 解析順序（與 `isaaclab.sh` 內部邏輯一致）：
`${ISAACLAB_ROOT}/_isaac_sim/python.sh` → `/isaac-sim/python.sh` → `python3`。

## 結構

```
training/
├── docker/
│   └── docker-compose.yml   # 本機 smoke test（官方映像 + bind mount）
├── scripts/
│   ├── bootstrap_runpod.sh  # 雲端：clone/pull + 裝依賴 + 交棒
│   └── run_train.sh         # 共用進入點（必須 LF）
├── configs/
│   └── ppo_billiard.yaml    # 訓練超參數
├── requirements.txt         # 額外依賴（Isaac Lab / torch / RL 框架已內含）
└── .gitattributes           # 強制 .sh 為 LF
```

## 使用

**本機 smoke test**

```powershell
docker compose -f training\docker\docker-compose.yml run --rm smoke
```

**RunPod**

```bash
curl -fsSL https://raw.githubusercontent.com/GuanPersonalDev/isaac-sim-digital-twin/main/training/scripts/bootstrap_runpod.sh | bash
```

repo 為 public，clone 不需要任何憑證。

## 尚未完成

- **`run_train.sh` 的真正訓練呼叫**：目前只跑 smoke test（驗證 `import isaacsim`、
  `import isaaclab`、`core/` 匯入）。`BilliardEnv` 完成後（#121/#122）才有可用的
  `--task`，屆時把註解掉的 `isaaclab.sh -p .../rsl_rl/train.py` 那段換上。
- **`configs/ppo_billiard.yaml`**：數值為佔位值，且格式要改成 Isaac Lab 選定
  框架期望的 agent cfg。`obs_dim` / `action_dim` 兩欄應刪除——Isaac Lab 從 env 的
  space 定義推導，留在 yaml 會變成第二個事實來源。
  另注意 observation 規格是 **21 維**（見 #222/#225），現有
  `core/services/rl_observation_encoder.py` 是 20 維，尚未對齊。
- **compose 的 cache volume 掛載點**：需依映像內實際 `HOME` 調整，待確認。

## 注意事項

**憑證不進映像。** 根目錄 `.dockerignore` 已排除 `.env`、`*.key`、`secrets/`。
NGC 金鑰在雲端用平台的 secret 注入，不要寫進任何檔案。

**換行符號。** `.sh` 若帶 CRLF 進容器會噴 `bash: \r: command not found`，且訊息
看不出原因。repo 的 `core.autocrlf=true` 會在 checkout 時把工作區檔案轉成 CRLF，
所以 `training/.gitattributes` 明確標記 `eol=lf`——**這對 RunPod 特別重要**，
因為雲端是直接 clone，若沒有這個設定，pod 上拿到的就會是 CRLF 版本。
