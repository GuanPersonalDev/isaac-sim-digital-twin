# Phase 3 任務拆分清單

## 專案目標
建立一個可安全產生多樣化訓練資料的撞球機器人仿真環境，作為 Physical AI 部署前的訓練基礎設施。

## 核心展示
- **中途展示點（LinkedIn 篇6）**：單台撞球機器人，UR5 持桿擊球，參數化控制
- **最終展示點（LinkedIn 篇8）**：RL 訓練迴路跑通，多環境並行，學習曲線收斂，找到最佳開球參數

## 版本資訊
- Isaac Sim：6.0.0（Workstation 安裝）
- Python：3.11
- 架構規範：詳見 `architecture-spec.md`

---

## Block 0：前置準備
預估總時數：2.5h

| # | 任務 | 預估 |
|---|---|---|
| 0-1 | 確認硬體需求、查閱 Isaac Sim 6.0.0 系統需求文件 | 0.5h |
| 0-2 | 下載並安裝 Isaac Sim 6.0.0 Workstation 版本 | 1h |
| 0-3 | 啟動 Isaac Sim，執行 Compatibility Checker 確認環境正常 | 0.5h |
| 0-4 | 跑官方 Hello World 範例，確認 Python API 可執行 | 0.5h |

---

## Block 1：Isaac Sim 環境熟悉
預估總時數：5h

| # | 任務 | 預估 |
|---|---|---|
| 1-1 | 熟悉 Isaac Sim Stage / Prim 基本操作 | 1h |
| 1-2 | 熟悉 Isaac Sim Python API：建立物件、設定位置、材質 | 1h |
| 1-3 | 熟悉 Articulation API 基本操作（讀取關節、設定位置） | 1h |
| 1-4 | 跑官方機械手臂相關範例（Franka 或 UR10e） | 1h |
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
| 2-6 | 建立 `ArticulationAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-7 | 建立 `PhysicsAPI` 抽象介面（碰撞偵測、接觸事件）（`omniverse_api/`） | 0.5h |
| 2-8 | 建立 `RigidBodyAPI` 抽象介面（球的位置、速度查詢）（`omniverse_api/`） | 0.5h |
| 2-9 | 建立 `StageAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-10 | 建立 Debug Menu 骨架（`extension/ui/debug_menu.py`） | 0.5h |

---

## Block 3：撞球場景建立
預估總時數：6h

| # | 任務 | 預估 |
|---|---|---|
| 3-1 | 研究撞球桌標準尺寸（9-ball 規格），設計場景比例 | 0.5h |
| 3-2 | 建立撞球桌幾何體（桌面、邊框、球袋）USD 場景 | 1h |
| 3-3 | 設定桌面物理材質（摩擦係數、彈性係數） | 0.5h |
| 3-4 | 建立 16 顆球（白球 + 1–9 號球）的 USD Prim | 0.5h |
| 3-5 | 設定球的物理材質（質量、摩擦係數、彈性係數） | 0.5h |
| 3-6 | 確認球的碰撞與滾動物理行為正常（PhysX 參數調校） | 1h |
| 3-7 | 設計 9-ball 標準開球擺位邏輯 + Unit Test | 0.5h |
| 3-8 | 確認球袋進球判定邏輯（接觸感測器或位置判定）+ Unit Test | 0.5h |
| 3-9 | 場景重置函式（球回到開球位置）+ Unit Test | 0.5h |

---

## Block 4：UR5 匯入與球桿設計
預估總時數：5h

| # | 任務 | 預估 |
|---|---|---|
| 4-1 | 下載 UR5 URDF，確認檔案結構完整 | 0.5h |
| 4-2 | URDF → USD 轉換，確認 Prim 路徑結構正確 | 1h |
| 4-3 | 確認關節結構，能用 API 讀取關節數量與名稱 | 0.5h |
| 4-4 | 設計球桿幾何體（USD Prim），確認尺寸比例合理 | 0.5h |
| 4-5 | 設計球桿與 UR5 末端的固定連結（Fixed Joint） | 0.5h |
| 4-6 | 產出 UR5 的 RMPflow 設定檔（`ur5_rmpflow_common.yaml`） | 1h |
| 4-7 | 確認 UR5 + 球桿整體在場景中的擺放位置合理 | 0.5h |

---

## Block 5：擊球動作實作
預估總時數：7h

| # | 任務 | 預估 |
|---|---|---|
| 5-1 | 設計擊球參數資料格式（桿速、擊球角度、擊球位置偏移）+ Unit Test | 0.5h |
| 5-2 | 設計 `ScriptController` 狀態機（`IDLE` / `AIMING` / `STRIKING` / `WAITING` / `RESET` / `ERROR`） | 0.5h |
| 5-3 | 撰寫狀態機 Unit Test（Mock `ArticulationAPI`） | 1h |
| 5-4 | 實作 `ArticulationAPIImpl`（`isaac_sim_impl_6_0/`） | 0.5h |
| 5-5 | 實作 `AIMING`：RMPflow 將球桿末端移到擊球預備位置 | 1h |
| 5-6 | 實作 `STRIKING`：沿擊球方向加速推進（模擬揮桿衝擊） | 1h |
| 5-7 | 實作 `WAITING`：等待所有球靜止（速度閾值判定）+ Unit Test | 0.5h |
| 5-8 | 實作 `RESET`：場景重置 → 回到 `IDLE` | 0.5h |
| 5-9 | 單次擊球循環跑通確認 | 0.5h |
| 5-10 | 物理參數調校：確認球桿衝擊力道傳遞正確，球散開效果合理 | 1h |

---

## Block 6：Reward Function 與結果評估
預估總時數：4h

| # | 任務 | 預估 |
|---|---|---|
| 6-1 | 設計 `ShotResult` 資料格式（各球最終位置、白球是否進袋、9 號球是否進袋、散開分數） | 0.5h |
| 6-2 | 實作散開程度計算函式（球的分布面積或平均距離）+ Unit Test | 1h |
| 6-3 | 實作白球進袋判定 + Unit Test | 0.5h |
| 6-4 | 實作 9 號球進袋加分判定 + Unit Test | 0.5h |
| 6-5 | 實作 Reward Function（整合散開分數、白球進袋懲罰、9 號球進袋加分）+ Unit Test | 1h |
| 6-6 | Debug Menu 新增「顯示當前 ShotResult」按鈕，手動驗證計算正確性 | 0.5h |

---

## Block 7：Observation 收集與預留 RL 接口
預估總時數：3.5h

| # | 任務 | 預估 |
|---|---|---|
| 7-1 | 確認 `Observation` 資料格式（各球位置、白球位置、手臂關節角度、擊球參數） | 0.5h |
| 7-2 | 實作 Observation 收集函式（從 `RigidBodyAPI` / `ArticulationAPI` 取得）+ Unit Test | 1h |
| 7-3 | 確認 `Action` 資料格式（桿速、擊球角度、擊球位置偏移） | 0.5h |
| 7-4 | 確認 `ControllerBase`：`get_action(observation) → action` 足以支撐未來 `ModelController` | 0.5h |
| 7-5 | 在 `ScriptController` 中加入 Observation 收集與 Action 格式輸出 | 0.5h |
| 7-6 | Debug Menu 新增「印出當前 Observation」按鈕 | 0.5h |

---

## Block 8：參數化控制與中途展示點
預估總時數：3.5h

> **LinkedIn 篇6 發布點**：單台撞球機器人，UR5 持桿擊球，參數化控制

| # | 任務 | 預估 |
|---|---|---|
| 8-1 | 實作擊球參數的可調介面（桿速範圍、角度範圍、位置偏移範圍） | 0.5h |
| 8-2 | HUD 新增參數控制面板（可即時調整擊球參數） | 1h |
| 8-3 | HUD 新增 ShotResult 顯示（散開分數、白球狀態、9 號球狀態） | 0.5h |
| 8-4 | 確認手動調整參數 → 擊球 → 結果顯示的完整流程 | 0.5h |
| 8-5 | 錄製中途展示 Demo 影片（單台撞球機器人，參數化擊球） | 0.5h |
| 8-6 | LinkedIn 篇6 草稿撰寫與發布 | 0.5h |

---

## Block 9：RL 訓練迴路
預估總時數：8h

> 高風險 Block，RL 訓練收斂時間不可控，需預留足夠緩衝

| # | 任務 | 預估 |
|---|---|---|
| 9-1 | 研究 Isaac Lab 環境設計規範（`gym.Env` 介面） | 1h |
| 9-2 | 實作 `BilliardEnv`：繼承 Isaac Lab 環境介面，整合 `core/` 邏輯 | 1.5h |
| 9-3 | 確認 `BilliardEnv` 的 `reset()`、`step()`、`observation_space`、`action_space` 正確 | 1h |
| 9-4 | 選定 RL 演算法（PPO，Isaac Lab 內建），設定超參數 | 0.5h |
| 9-5 | 單環境訓練跑通確認（確認 reward 有在變化） | 1h |
| 9-6 | 確認訓練過程中 reward 曲線有上升趨勢 | 1h |
| 9-7 | 儲存訓練好的模型，確認可以載入並執行推論 | 0.5h |
| 9-8 | 實作 `ModelController`（載入訓練模型，替換 `ScriptController`） | 0.5h |
| 9-9 | 確認 `ModelController` 執行效果優於隨機參數 | 0.5h |

---

## Block 10：多環境並行
預估總時數：4h

> 硬體上限未知，實際可跑環境數以測試結果為準

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
- Block 9（RL 訓練收斂）：不可控，可能需要多次調整超參數
- Block 10（多環境並行）：硬體上限需實測確認

---

## 參考文件

| 文件 | 用途 |
|---|---|
| `architecture-spec.md` | 專案架構規範（`core/` / `omniverse_api/` / `isaac_sim_impl_6_0/` 分層原則） |
| `unit-test-rules.md` | Unit Test 判斷規則與 TDD 流程 |
| `code-review-checklist.md` | Code Review 自我審查清單 |
| `api-migration-agent.md` | Isaac Sim 版本升級輔助 sub-agent 說明 |
