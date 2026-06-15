# Phase 3 任務拆分清單

## 專案目標
建立一個撞球機器人 RL 訓練環境——UR5 持桿擊球，多環境並行訓練，找到最佳 9-ball 開球參數。
作為 Physical AI 部署前的訓練基礎設施展示。

## 核心展示點
- **中途展示點（LinkedIn 篇6）**：單台撞球機器人，UR5 持桿擊球，參數化控制，HUD 顯示擊球結果
- **最終展示點（LinkedIn 篇8）**：RL 訓練迴路跑通，多環境並行，學習曲線收斂，最佳開球參數展示

## 版本資訊
- **Isaac Sim：6.0.0.1**
- **Python：3.12**
- **安裝：** `pip install isaacsim[all,extscache]==6.0.0.1 --extra-index-url https://pypi.nvidia.com`
- **Core API：** Warp-based Core Experimental API
- **架構規範：** 詳見 `architecture-spec.md`

---

## RL 設計定案

### Observation（20個數字）

| 資料 | 數量 | 說明 |
|---|---|---|
| 1–9 號球 XY 位置 | 18 | 座標以桌面中心為原點 |
| 白球 XY 位置 | 2 | 座標以桌面中心為原點 |
| **總計** | **20** | |

### Action（6個數字）

| 資料 | 數量 | 說明 |
|---|---|---|
| 白球擺放位置 XY | 2 | Kitchen 區內（桌面頭端 1/4 範圍） |
| 出桿方向角度 | 1 | 相對桌面的水平角度 |
| 出桿速度 | 1 | 球桿擊球時的速度 |
| 擊球位置偏移 | 2 | 擊打白球的上下左右偏移量 |
| **總計** | **6** | |

### Reward Function

| 條件 | 分數 | 說明 |
|---|---|---|
| 散開程度 | 0.0 ~ 1.0 | 凸包面積×0.5 + 距桌面中心平均距離×0.5，各自正規化 |
| 9號球進袋（白球未進） | +3.0 | 有效勝利 |
| 白球進袋 | -3.5 | 含9號球同時進袋情況，9號球不加分 |
| 白球未先接觸1號球 | -1.5 | 犯規，立刻重置 |
| 沒球進袋且少於4顆球碰邊框 | -0.5 | 開球力道不足 |

### 架構職責分工

| 負責方 | 職責 |
|---|---|
| RL Policy | 看球的位置 → 決定出桿參數（Observation → Action） |
| Isaac Lab IK | 把出桿參數轉換成關節運動 |
| RMPflow | 執行實際的手臂軌跡 |
| BallPositionProvider | 衝球時使用固定值（BreakShotPositionProvider），其他情境使用即時查詢（LivePositionProvider） |

---

## 環境固定值（WPA / BCA 國際標準）

### 撞球桌

| 項目 | 數值 |
|---|---|
| 桌面長度 | 2.54 m |
| 桌面寬度 | 1.27 m |
| 球袋數量 | 6（四角 + 兩側中央） |
| 角袋開口寬度 | 約 11.4–11.7 cm |

### 球

| 項目 | 數值 |
|---|---|
| 球的直徑 | 57.15 mm |
| 球的質量 | 163 g（取中間值） |
| 1號球位置 | foot spot（桌面長軸 1/4 處） |
| 9號球位置 | 菱形中央 |
| 其餘球 | 隨機排列於菱形內 |
| 白球擺放區域 | Kitchen 區（桌面頭端 1/4，全寬） |

### 物理參數

| 項目 | 數值 |
|---|---|
| 球與球摩擦係數 | 0.05 |
| 球與球彈性係數 | 0.95 |
| 球與桌布滾動阻力係數 | 0.01 |
| 球與桌布滑動摩擦係數 | 0.20 |
| 球與邊框彈性係數 | 0.75 |
| 球桿頭與球摩擦係數 | 0.60 |
| 球桿頭與球彈性係數 | 0.73 |

### 訓練參數

| 項目 | 數值 |
|---|---|
| 出桿速度範圍 | 0.5 ~ 7.0 m/s |
| 出桿角度範圍 | 0 ~ 360 度 |
| 擊球位置偏移範圍 | ±0.5 球半徑 |
| 靜止判定閾值 | 所有球速度 < 0.01 m/s |
| 回合超時上限 | 10 秒 |

---

## Block 0：前置準備
預估總時數：2.5h

| # | 任務 | 預估 |
|---|---|---|
| 0-1 | 確認硬體需求、查閱 Isaac Sim 6.0 系統需求文件 | 0.5h |
| 0-2 | 安裝 Isaac Sim 6.0.0.1（pip install） | 1h |
| 0-3 | 啟動 Isaac Sim，確認環境正常 | 0.5h |
| 0-4 | 跑官方 Hello World 範例，確認 Python API 可執行 | 0.5h |

---

## Block 1：Isaac Sim 環境熟悉
預估總時數：5h

| # | 任務 | 預估 |
|---|---|---|
| 1-1 | 熟悉 Isaac Sim 6.0 Stage / Prim 基本操作 | 1h |
| 1-2 | 熟悉 Isaac Sim 6.0 Core Experimental API（Warp-based）基礎操作 | 1h |
| 1-3 | 熟悉 Articulation API 基本操作（讀取關節、設定位置） | 1h |
| 1-4 | 跑官方機械手臂相關範例 | 1h |
| 1-5 | 熟悉 RMPflow 基本概念，跑官方 Follow Target 範例 | 1h |

---

## Block 2：專案架構建立
預估總時數：5h

| # | 任務 | 預估 |
|---|---|---|
| 2-1 | 建立 `core/` + `extension/` 目錄骨架（含 `omniverse_api/`、`isaac_sim_impl_6_0/`、`ui/`） | 0.5h |
| 2-2 | 建立 pytest 執行環境，確認測試可在 WSL2 跑通 | 0.5h |
| 2-3 | 建立核心資料模型：`BilliardState`、`Observation`、`Action`、`ShotResult` | 0.5h |
| 2-4 | 針對資料模型撰寫 Unit Test | 0.5h |
| 2-5 | 建立 `ControllerBase` 抽象介面 + Unit Test | 0.5h |
| 2-6 | 建立 `BallPositionProvider` 抽象介面 + `BreakShotPositionProvider` 實作 + Unit Test | 0.5h |
| 2-7 | 建立 `ArticulationAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-8 | 建立 `PhysicsAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-9 | 建立 `RigidBodyAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-10 | 建立 Debug Menu 骨架（`extension/ui/debug_menu.py`） | 0.5h |

---

## Block 3：撞球場景建立
預估總時數：6h

| # | 任務 | 預估 |
|---|---|---|
| 3-1 | 建立撞球桌幾何體（桌面、邊框、球袋）USD 場景（依 WPA 標準尺寸） | 1h |
| 3-2 | 設定桌面物理材質（摩擦係數 0.20、彈性係數 0.75） | 0.5h |
| 3-3 | 建立 16 顆球（白球 + 1–9 號球）的 USD Prim（直徑 57.15mm、質量 163g） | 0.5h |
| 3-4 | 設定球的物理材質（球與球摩擦 0.05、彈性 0.95） | 0.5h |
| 3-5 | 確認球的碰撞與滾動物理行為正常（PhysX 參數調校） | 1h |
| 3-6 | 實作 9-ball 標準開球擺位邏輯（1號球 foot spot、9號球菱形中央）+ Unit Test | 0.5h |
| 3-7 | 確認球袋進球判定邏輯（接觸感測器或位置判定）+ Unit Test | 0.5h |
| 3-8 | 實作場景重置函式（球回到開球位置）+ Unit Test | 0.5h |
| 3-9 | 場景整體穩定性確認，物理仿真無異常 | 0.5h |

---

## Block 4：UR5 匯入與球桿設計
預估總時數：5h

| # | 任務 | 預估 |
|---|---|---|
| 4-1 | 下載 UR5 URDF，確認檔案結構完整 | 0.5h |
| 4-2 | URDF → USD 轉換（Isaac Sim 6.0 URDF Importer 3.0），確認 Prim 路徑結構正確 | 1h |
| 4-3 | 確認關節結構，能用 API 讀取關節數量與名稱 | 0.5h |
| 4-4 | 設計球桿幾何體（USD Prim），確認尺寸比例合理 | 0.5h |
| 4-5 | 設計球桿與 UR5 末端的固定連結（Fixed Joint） | 0.5h |
| 4-6 | 產出 UR5 的 RMPflow 設定檔 | 1h |
| 4-7 | 確認 UR5 + 球桿整體在場景中的擺放位置合理 | 0.5h |

---

## Block 5：擊球動作實作
預估總時數：7h

| # | 任務 | 預估 |
|---|---|---|
| 5-1 | 設計擊球參數資料格式（桿速 0.5–7m/s、角度 0–360°、位置偏移 ±0.5r）+ Unit Test | 0.5h |
| 5-2 | 設計 `ScriptController` 狀態機（`IDLE` / `AIMING` / `STRIKING` / `WAITING` / `RESET` / `ERROR`） | 0.5h |
| 5-3 | 撰寫狀態機 Unit Test（Mock `ArticulationAPI`） | 1h |
| 5-4 | 實作 `ArticulationAPIImpl`（`isaac_sim_impl_6_0/`） | 0.5h |
| 5-5 | 實作 `AIMING`：RMPflow 將球桿末端移到擊球預備位置 | 1h |
| 5-6 | 實作 `STRIKING`：沿擊球方向加速推進（模擬揮桿衝擊） | 1h |
| 5-7 | 實作 `WAITING`：等待所有球靜止（速度 < 0.01 m/s）+ Unit Test | 0.5h |
| 5-8 | 實作 `RESET`：場景重置 → 回到 `IDLE` | 0.5h |
| 5-9 | 單次擊球循環跑通確認 | 0.5h |
| 5-10 | 物理參數調校：確認球桿衝擊力道傳遞正確，球散開效果合理 | 1h |

---

## Block 6：Reward Function 與結果評估
預估總時數：4h

| # | 任務 | 預估 |
|---|---|---|
| 6-1 | 設計 `ShotResult` 資料格式（各球最終位置、白球狀態、9號球狀態、散開分數） | 0.5h |
| 6-2 | 實作凸包面積計算函式（正規化至 0~1）+ Unit Test | 0.5h |
| 6-3 | 實作各球距桌面中心平均距離計算（正規化至 0~1）+ Unit Test | 0.5h |
| 6-4 | 實作散開程度複合計算（面積×0.5 + 距離×0.5）+ Unit Test | 0.5h |
| 6-5 | 實作白球進袋判定（-3.5）+ Unit Test | 0.5h |
| 6-6 | 實作 9號球進袋判定（白球未進 +3.0 / 白球同進 不加分）+ Unit Test | 0.5h |
| 6-7 | 實作犯規判定（未接觸1號球 -1.5 / 4顆球碰邊框不足 -0.5）+ Unit Test | 0.5h |
| 6-8 | 整合完整 Reward Function + Unit Test | 0.5h |

---

## Block 7：Observation 收集與預留 RL 接口
預估總時數：3.5h

| # | 任務 | 預估 |
|---|---|---|
| 7-1 | 實作 `RigidBodyAPIImpl`（`isaac_sim_impl_6_0/`，查詢球的位置與速度） | 0.5h |
| 7-2 | 實作 `LivePositionProvider`（`isaac_sim_impl_6_0/`，即時查詢球位置） | 0.5h |
| 7-3 | 實作 Observation 收集函式（20個數字）+ Unit Test | 1h |
| 7-4 | 確認 `ControllerBase`：`get_action(observation) → action` 足以支撐未來 `ModelController` | 0.5h |
| 7-5 | 在 `ScriptController` 中加入 Observation 收集與 Action 格式輸出 | 0.5h |
| 7-6 | Debug Menu 新增「印出當前 Observation」按鈕 | 0.5h |

---

## Block 8：參數化控制與中途展示點
預估總時數：3.5h

> **LinkedIn 篇6 發布點**：單台撞球機器人，UR5 持桿擊球，參數化控制

| # | 任務 | 預估 |
|---|---|---|
| 8-1 | 實作擊球參數的可調介面（桿速、角度、位置偏移可即時調整） | 0.5h |
| 8-2 | HUD 新增參數控制面板 | 1h |
| 8-3 | HUD 新增 ShotResult 顯示（散開分數、白球狀態、9號球狀態） | 0.5h |
| 8-4 | 確認手動調整參數 → 擊球 → 結果顯示的完整流程 | 0.5h |
| 8-5 | 錄製中途展示 Demo 影片 | 0.5h |
| 8-6 | LinkedIn 篇6 草稿撰寫與發布 | 0.5h |

---

## Block 9：RL 訓練迴路
預估總時數：8h

> 高風險 Block，RL 訓練收斂時間不可控，需預留緩衝

| # | 任務 | 預估 |
|---|---|---|
| 9-1 | 研究 Isaac Lab 環境設計規範（`gym.Env` 介面，Isaac Lab 3.0） | 1h |
| 9-2 | 實作 `BilliardEnv`：繼承 Isaac Lab 環境介面，整合 `core/` 邏輯 | 1.5h |
| 9-3 | 確認 `BilliardEnv` 的 `reset()`、`step()`、`observation_space`、`action_space` 正確 | 1h |
| 9-4 | 選定 RL 演算法（PPO），設定初始超參數 | 0.5h |
| 9-5 | 單環境訓練跑通確認（確認 reward 有在變化） | 1h |
| 9-6 | 確認訓練過程中 reward 曲線有上升趨勢 | 1h |
| 9-7 | 儲存訓練好的模型，確認可以載入並執行推論 | 0.5h |
| 9-8 | 實作 `ModelController`（載入訓練模型，替換 `ScriptController`） | 0.5h |
| 9-9 | 確認 `ModelController` 執行效果優於隨機參數 | 0.5h |

---

## Block 10：多環境並行
預估總時數：4h

> 硬體上限未知（RTX 4060 8GB），實際可跑環境數以測試結果為準

| # | 任務 | 預估 |
|---|---|---|
| 10-1 | 研究 Isaac Lab 多環境並行設計（Vectorized Environment） | 1h |
| 10-2 | 調整 `BilliardEnv` 支援多環境實例化 | 1h |
| 10-3 | 測試 8 台並行，確認物理仿真穩定 | 0.5h |
| 10-4 | 逐步擴大規模（32 → 64 → 128），記錄每個規模的 FPS | 1h |
| 10-5 | 確認最大可穩定運行的環境數量 | 0.5h |

---

## Block 11：整合測試與補坑
預估總時數：4h

| # | 任務 | 預估 |
|---|---|---|
| 11-1 | 全流程跑通確認（多環境並行 → RL 訓練 → 收斂 → ModelController 執行） | 1h |
| 11-2 | Debug Menu 所有按鈕確認 | 0.5h |
| 11-3 | 穩定性測試（長時間訓練，確認無記憶體洩漏） | 1h |
| 11-4 | API 掃描：執行 `api-migration-agent` 掃描 `isaac_sim_impl_6_0/`，產出 API 使用清單 | 0.5h |
| 11-5 | 補坑收尾 | 1h |

---

## Block 12：Demo + README + LinkedIn
預估總時數：6h

> **LinkedIn 篇8 發布點**：RL 訓練收斂，多環境並行，最佳開球參數展示

| # | 任務 | 預估 |
|---|---|---|
| 12-1 | Demo 影片腳本規劃（單台參數化 → 多台並行訓練 → 學習曲線 → 最佳參數展示） | 0.5h |
| 12-2 | 錄製 Demo 影片（OBS） | 1h |
| 12-3 | 影片剪輯確認 | 1h |
| 12-4 | README 架構圖 + 技術亮點撰寫 | 1.5h |
| 12-5 | README 接口設計說明（RL 訓練架構 + 版本升級策略） | 0.5h |
| 12-6 | README 收尾確認、上傳 GitHub | 0.5h |
| 12-7 | LinkedIn 篇8 草稿撰寫與發布 | 0.5h |
| 12-8 | LinkedIn 篇8 潤稿確認 | 0.5h |

---

## 總時數摘要

| Block | 內容 | 時數 |
|---|---|---|
| Block 0 | 前置準備 | 2.5h |
| Block 1 | Isaac Sim 環境熟悉 | 5h |
| Block 2 | 專案架構建立 | 5h |
| Block 3 | 撞球場景建立 | 6h |
| Block 4 | UR5 匯入與球桿設計 | 5h |
| Block 5 | 擊球動作實作 | 7h |
| Block 6 | Reward Function 與結果評估 | 4h |
| Block 7 | Observation 收集與預留 RL 接口 | 3.5h |
| Block 8 | 參數化控制與中途展示點 | 3.5h |
| Block 9 | RL 訓練迴路 | 8h |
| Block 10 | 多環境並行 | 4h |
| Block 11 | 整合測試與補坑 | 4h |
| Block 12 | Demo + README + LinkedIn | 6h |
| **總計** | | **63.5h** |

可用時數約 87h，緩衝空間約 23.5h。

**高風險項目：**
- Block 5（物理參數調校）：實際可能超時
- Block 9（RL 訓練收斂）：不可控，可能需要多次 Reward Shaping
- Block 10（多環境並行）：RTX 4060 8GB 硬體上限需實測確認

---

## 參考文件

| 文件 | 用途 |
|---|---|
| `architecture-spec.md` | 專案架構規範（三層分離、抽象介面設計） |
| `unit-test-rules.md` | Unit Test 判斷規則與 TDD 流程 |
| `code-review-checklist.md` | Code Review 自我審查清單 |
| `api-migration-agent.md` | Isaac Sim 版本升級輔助 sub-agent 說明 |
