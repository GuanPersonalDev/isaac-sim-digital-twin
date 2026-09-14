# Isaac Sim Digital Twin — 九球撞球機器人模擬

[English](README_en.md) | 繁體中文

## 專案簡介

這是一個以 **NVIDIA Isaac Sim 6.0** 打造的九球撞球機器人 digital twin 個人學習
專案，用來練習 Isaac Sim／Omniverse Kit 的 API 開發，以及用強化學習（RL）訓練機
械手臂的擊球策略。

專案內容包含：
- 在虛擬撞球桌上用機械手臂執行擊球動作（架構上支援 UR3e / UR5 / UR10e 等手臂，
  目前主要開發方向是 UR10e）
- 透過 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 建立 RL 訓練環境
  （`rl_task/`），在雲端 GPU（RunPod）上訓練擊球策略
- 在本機以 Isaac Sim GUI 回放、驗證球體物理行為與控制邏輯是否正確

因為是練習性質的個人專案，程式碼與文件仍在持續調整中，架構設計文件
（`docs/architecture-spec.md`）與實際目錄結構偶爾會有落差，請以實際程式碼為準。

## 技術棧

- **Isaac Sim**：6.0.0.1（Warp-based Core Experimental API）
- **Python**：3.12
- **Isaac Lab**：`isaaclab` / `isaaclab_assets` / `isaaclab_mimic` /
  `isaaclab_rl` / `isaaclab_tasks`（供 `rl_task/` 的 RL 訓練環境使用）
- **Omniverse Kit**：`omni.usd`、`omni.timeline`、`omni.ui`、
  `omni.kit.menu.utils`、`omni.kit.viewport.utility` 等（供 `extension/` 的
  Kit extension 使用）

## 架構分層

專案採「商業邏輯／平台抽象／平台實作」三層分離設計，目的是讓 Isaac Sim 版本升級
時只需重寫實作層，不影響商業邏輯：

| 目錄 | 職責 |
|---|---|
| `core/ports/` | 平台無關的抽象介面（`stage_api`、`articulation_api`、`physics_api`、`rigid_body_api`、`material_api`、`policy_port`） |
| `core/controllers/` | 控制策略（狀態機控制器、手動控制器、模型控制器等） |
| `core/models/` | 資料模型（球、桌、機械手臂、動作、觀測等） |
| `core/services/` | 業務邏輯（reward 計算、擊球策略、IK、滾動阻力、observation encoder 等） |
| `core/tests/` | core 層的 pytest 單元測試，純 Python、不需啟動 Isaac Sim |
| `extension/isaac_sim_impl_6_0/` | `core/ports/` 抽象介面在 Isaac Sim 6.0 上的具體實作 |
| `extension/ui/` | omni.ui 元件（HUD panel、debug menu 等） |
| `extension/billiard_digital_twin/` | Kit extension 進入點（`extension.toml`、主 extension 類別） |
| `rl_task/` | 獨立的 Isaac Lab RL 訓練套件（`billiard_rl`），有自己的 `pyproject.toml` |
| `training/` | 雲端訓練維運腳本（RunPod bootstrap、watchdog 等），詳見 [`training/README.md`](training/README.md) |
| `docs/` | 架構規範、CHANGELOG、設計與除錯文件 |
| `assets/` | 撞球桌、球、機械手臂等 USD/USDA 場景資產 |
| `scripts/` | 開發過程中的一次性診斷／驗證腳本 |
| `models/`、`outputs/` | 訓練模型與輸出產物 |

## 目錄結構

```
isaac-sim-digital-twin/
├── core/                       # 平台無關的商業邏輯層
│   ├── ports/                  # 抽象介面
│   ├── controllers/            # 控制策略
│   ├── models/                 # 資料模型
│   ├── services/                # 業務邏輯
│   └── tests/                  # 單元測試
├── extension/                  # Isaac Sim 平台橋接層
│   ├── isaac_sim_impl_6_0/     # 6.0 版 API 實作
│   ├── ui/                     # omni.ui 元件
│   └── billiard_digital_twin/  # Kit extension 進入點
├── rl_task/                    # Isaac Lab RL 訓練套件
├── training/                   # 雲端訓練維運腳本（RunPod）
├── docs/                       # 架構規範與設計文件
├── assets/                     # 場景資產（USD/USDA）
├── scripts/                    # 開發診斷腳本
├── models/                     # 訓練產出模型
├── LICENSE
└── README.md
```

## 安裝與執行

### 本機環境（Isaac Sim GUI 回放／驗證）

```bash
pip install isaacsim[all,extscache]==6.0.0.1 --extra-index-url https://pypi.nvidia.com
```

安裝完成後，透過 Isaac Sim 載入 `extension/billiard_digital_twin/` extension
即可在 GUI 中回放撞球場景。

### RL 訓練環境（`rl_task/`）

`rl_task/` 是獨立的 Isaac Lab 套件，依賴 `isaaclab` 生態系，安裝方式請參考
`rl_task/pyproject.toml` 與 `rl_task/setup.py`。

### 雲端訓練（RunPod）

雲端 GPU 訓練環境的建置與執行流程另見 [`training/README.md`](training/README.md)。

## 測試

`core/` 層的商業邏輯以 pytest 撰寫單元測試，外部依賴（omni/pxr/isaacsim）以
`unittest.mock` 隔離，不需啟動 Isaac Sim 即可執行：

```bash
pytest
```

（`pytest.ini` 已設定 `testpaths = core/tests`）

## 授權

本專案採用 [Apache License 2.0](LICENSE)。
