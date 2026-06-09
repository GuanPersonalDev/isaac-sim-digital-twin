# Phase 3 任務拆分清單

## 專案目標
產線仿真驗證環境——Isaac Sim + UR5 Pick & Place，預留 Physical AI 訓練接口。

## 版本資訊
- Isaac Sim：5.1.0（Workstation 安裝）
- Python：3.11
- 架構規範：詳見 `architecture-spec.md`

---

## Block 0：前置準備
預估總時數：2.5h

| # | 任務 | 預估 |
|---|---|---|
| 0-1 | 確認硬體需求、查閱 Isaac Sim 5.1.0 系統需求文件 | 0.5h |
| 0-2 | 下載並安裝 Isaac Sim 5.1.0 Workstation 版本 | 1h |
| 0-3 | 啟動 Isaac Sim，執行 Compatibility Checker 確認環境正常 | 0.5h |
| 0-4 | 跑官方 Hello World 範例，確認 Python API 可執行 | 0.5h |

---

## Block 1：Isaac Sim 環境熟悉
預估總時數：5h

| # | 任務 | 預估 |
|---|---|---|
| 1-1 | 熟悉 Isaac Sim Stage / Prim 基本操作 | 1h |
| 1-2 | 熟悉 Isaac Sim Python API：建立物件、設定位置 | 1h |
| 1-3 | 熟悉 Articulation API 基本操作（讀取關節、設定位置） | 1h |
| 1-4 | 跑官方機械手臂相關範例（Franka 或 UR10e） | 1h |
| 1-5 | 熟悉 RMPflow 基本概念，跑官方 Follow Target 範例 | 1h |

---

## Block 2：專案架構建立
預估總時數：5h

| # | 任務 | 預估 |
|---|---|---|
| 2-1 | 建立 `core/` + `extension/` 目錄骨架（含 `omniverse_api/`、`isaac_sim_impl_5_1/`、`ui/`） | 0.5h |
| 2-2 | 建立 pytest 執行環境，確認測試可在 WSL2 跑通 | 0.5h |
| 2-3 | 建立資料模型：`MachineState`、`Observation`、`Action` | 0.5h |
| 2-4 | 針對資料模型撰寫 Unit Test | 0.5h |
| 2-5 | 建立 `ControllerBase` 抽象介面 + Unit Test | 0.5h |
| 2-6 | 建立 `MotionPlannerAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-7 | 建立 `ArticulationAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-8 | 建立 `StageAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-9 | 建立 `RigidBodyAPI` 抽象介面（`omniverse_api/`） | 0.5h |
| 2-10 | 建立 Debug Menu 骨架（`extension/ui/debug_menu.py`） | 0.5h |

---

## Block 3：UR5 匯入與場景建立
預估總時數：5.5h

| # | 任務 | 預估 |
|---|---|---|
| 3-1 | 研究 URDF 格式，確認 UR5 開源 URDF 來源 | 0.5h |
| 3-2 | 下載 UR5 URDF，確認檔案結構完整 | 0.5h |
| 3-3 | URDF → USD 轉換，確認轉換結果 | 1h |
| 3-4 | 匯入場景，確認 Prim 路徑結構正確 | 0.5h |
| 3-5 | 確認關節結構，能用 API 讀取關節數量與名稱 | 0.5h |
| 3-6 | 產出 UR5 的 RMPflow 設定檔（`ur5_rmpflow_common.yaml`） | 1h |
| 3-7 | 建立工廠場景：地板、傳送帶、目標物件（Cube）擺設 | 1h |
| 3-8 | 確認場景穩定運行，物理仿真無異常 | 0.5h |

---

## Block 4：Pick & Place 實作
預估總時數：7.5h

| # | 任務 | 預估 |
|---|---|---|
| 4-1 | 設計 `ScriptController` 狀態機（`IDLE` / `MOVING_TO_PICK` / `PICKING` / `MOVING_TO_PLACE` / `PLACING`） | 0.5h |
| 4-2 | 撰寫狀態機 Unit Test（Mock `MotionPlannerAPI` 與 `ArticulationAPI`） | 1h |
| 4-3 | 實作 `MotionPlannerAPIImpl`（`isaac_sim_impl_5_1/`，包裝 RMPflow） | 1h |
| 4-4 | 實作 `ArticulationAPIImpl`（`isaac_sim_impl_5_1/`，包裝 Articulation） | 0.5h |
| 4-5 | 實作 `ScriptController`：`IDLE` → `MOVING_TO_PICK`（呼叫 `MotionPlannerAPI`） | 1h |
| 4-6 | 實作 `ScriptController`：`PICKING`（呼叫 `ArticulationAPI` 控制夾爪閉合） | 0.5h |
| 4-7 | 實作 `ScriptController`：`MOVING_TO_PLACE` | 0.5h |
| 4-8 | 實作 `ScriptController`：`PLACING`（夾爪張開）→ `IDLE` | 0.5h |
| 4-9 | 單次 Pick & Place 循環跑通確認 | 0.5h |
| 4-10 | 多次循環穩定性測試（連續 10 次以上） | 1h |
| 4-11 | 異常處理：夾取失敗、超時等情境 + Unit Test | 1h |

---

## Block 5：預留接口實作
預估總時數：3.5h

> 目的：以抽象隔離設計預留 Physical AI 訓練接口，未來接上 RL 時只需最小限度的實作與抽換。

| # | 任務 | 預估 |
|---|---|---|
| 5-1 | 確認 `Observation` 資料格式（關節角度、末端位置、夾爪狀態、目標物件位置） | 0.5h |
| 5-2 | 實作 Observation 收集函式（從 `ArticulationAPI` / `RigidBodyAPI` 取得資料）+ Unit Test | 1h |
| 5-3 | 確認 `Action` 資料格式（目標末端位置、夾爪指令） | 0.5h |
| 5-4 | 確認 `ControllerBase` 介面：`get_action(observation) → action` 足以支撐未來 `ModelController` 接入 | 0.5h |
| 5-5 | 在 `ScriptController` 中加入 Observation 收集與 Action 格式輸出，確認介面一致性 | 0.5h |
| 5-6 | Debug Menu 新增「印出當前 Observation」按鈕，手動驗證資料正確性 | 0.5h |

---

## Block 6：MQTT 整合
預估總時數：3h

| # | 任務 | 預估 |
|---|---|---|
| 6-1 | 設計機器人狀態回傳的 MQTT Topic 結構 | 0.5h |
| 6-2 | 實作 MQTT 訊息格式化邏輯（`core/services/mqtt_service.py`）+ Unit Test | 0.5h |
| 6-3 | 實作 `extension/` 層 MQTT Client 連線與 publish | 0.5h |
| 6-4 | MQTT Client 連線整合測試（確認資料送出） | 0.5h |
| 6-5 | Phase 2 HUD 接收機器人狀態，確認 D 區顯示更新 | 1h |

---

## Block 7：整合測試與補坑
預估總時數：4h

| # | 任務 | 預估 |
|---|---|---|
| 7-1 | Phase 3 全流程跑通確認（場景啟動 → Pick & Place → MQTT → HUD） | 1h |
| 7-2 | Debug Menu 所有按鈕確認（視覺功能手動驗證） | 0.5h |
| 7-3 | 穩定性測試（長時間運行，確認無記憶體洩漏或當機） | 1h |
| 7-4 | API 掃描：執行 `api-migration-agent` 掃描 `isaac_sim_impl_5_1/`，產出 API 使用清單 | 0.5h |
| 7-5 | 補坑收尾 | 1h |

---

## Block 8：Demo + README + LinkedIn
預估總時數：6h

| # | 任務 | 預估 |
|---|---|---|
| 8-1 | Demo 影片腳本規劃 | 0.5h |
| 8-2 | 錄製 Demo 影片（OBS） | 1h |
| 8-3 | 影片剪輯確認 | 1h |
| 8-4 | README 架構圖 + 技術亮點撰寫 | 1.5h |
| 8-5 | README 接口設計說明（預留 Physical AI 路徑 + 版本升級策略） | 0.5h |
| 8-6 | README 收尾確認、上傳 GitHub | 0.5h |
| 8-7 | LinkedIn 篇8 草稿撰寫 | 1h |
| 8-8 | LinkedIn 篇8 潤稿與發布 | 0.5h |

---

## 總時數摘要

| Block | 內容 | 時數 |
|---|---|---|
| Block 0 | 前置準備 | 2.5h |
| Block 1 | Isaac Sim 環境熟悉 | 5h |
| Block 2 | 專案架構建立 | 5h |
| Block 3 | UR5 匯入與場景建立 | 5.5h |
| Block 4 | Pick & Place 實作 | 7.5h |
| Block 5 | 預留接口實作 | 3.5h |
| Block 6 | MQTT 整合 | 3h |
| Block 7 | 整合測試與補坑 | 4h |
| Block 8 | Demo + README + LinkedIn | 6h |
| **總計** | | **42h** |

可用時數約 87h，緩衝空間約 45h，用於應對踩坑、Isaac Sim 學習曲線、未預期問題。

---

## 參考文件

| 文件 | 用途 |
|---|---|
| `architecture-spec.md` | 專案架構規範（`core/` / `omniverse_api/` / `isaac_sim_impl_5_1/` 分層原則） |
| `unit-test-rules.md` | Unit Test 判斷規則與 TDD 流程 |
| `code-review-checklist.md` | Code Review 自我審查清單 |
| `api-migration-agent.md` | Isaac Sim 版本升級輔助 sub-agent 說明 |
