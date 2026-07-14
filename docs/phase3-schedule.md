# Phase 3 進度排程表

## 排程規則

- 7 月起：平日（一～五）1h／天，週六 4h（2×2h），週日 4h（2×2h）
- 2026-07-13 依 `phase3-plan-risks-solutions.md` 全面重排：改為雙 Milestone 結構
  - **Milestone A（RL 訓練）**：impulse-based（`set_velocities`）、無手臂、128 平行環境。**硬約束：7 月底前訓練收斂**（Wave 1 求職目標 8 月中倒推）
  - **Milestone B（手臂執行，GitHub M7）**：UR5 + 球桿剛性連結、單一環境；可行性地圖 → 關節空間揮桿軌跡 → 真實接觸物理校正，約 4 週排入 8 月
- 本週順序固定（訓練配置定案前置）：[4-8] 空揮測速（#176）→ [9-0] early termination 確認（#178）→ 完成後 A 訓練配置才算定案
- Fallback 決策點：**8/02** A 收斂判定（#179）、**8/20** 揮桿軌跡判定（#181，不達標降檔位 c）、**8/23** 偏移精度判定（#182，不穩降檔位 b 固定中心）。檔位定義見 #183
- Wave 1（8 月中）交付線：A 收斂成果 + 中途 Demo（篇6）+ README + B 進行中敘事；B 完成線 = fallback 檔位 (b)
- 篇 5（Phase 2 完成文）不依賴 Phase 3，可立即發布，不佔本表開發時數
- 排程結果：Wave 1 所需項目（Demo、篇6、README）於 **2026-08-12** 前完成；全部任務預計 **2026-09-06** 完成
- ⚠️ **2026-07-14 重排（零緩衝）**：因新增 4-3d~4-3g（撞球桌/手臂職責拆分）5.5h 工作量，且 4-3e-impl/4-3e-test/4-3f 時數由 0.5h 上修為 1h，07-14 ~ 08-02（A-CP）共 19 天的額度被排到全滿，部分任務跨日拆半天完成（如 4-8、5-3、9-1、9-2、9-5）。**此區間沒有任何緩衝**，任一天延誤都會讓 A-CP（#179）超過 8/02 硬約束。使用者已確認接受此風險（不犧牲 4-3g/6-6/7-6 等非關鍵任務換取緩衝）。

## 任務排程明細

| 任務ID | Block | 任務名稱 | 預估時數 | Milestone | 完成 | 完成日期 | 預計完成日期 | GitHub Issue |
|---|---|---|---|---|---|---|---|---|
| 0-1 | Block 0 | 確認硬體需求、查閱 Isaac Sim 6.0.0 系統需求文件 | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-09 | 2026-06-12 | #1 |
| 0-2 | Block 0 | 下載並安裝 Isaac Sim 6.0.0 Workstation 版本 | 1h | M1: 環境建立與架構 | TRUE | 2026-06-10 | 2026-06-13 | #2 |
| 0-3 | Block 0 | 啟動 Isaac Sim 確認環境正常 | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-13 | #3 |
| 0-4 | Block 0 | 跑官方 Hello World 範例確認 Python API 可執行 | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-14 | #4 |
| 1-1 | Block 1 | 熟悉 Isaac Sim Stage / Prim 基本操作 | 1h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-14 | #5 |
| 1-2 | Block 1 | 熟悉 Isaac Sim Python API：建立物件設定位置材質 | 1h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-14 | #6 |
| 1-3 | Block 1 | 熟悉 Articulation API 基本操作（讀取關節設定位置） | 1h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-14 | #7 |
| 1-4 | Block 1 | 跑官方機械手臂相關範例（Franka 或 UR10e） | 1h | M1: 環境建立與架構 | TRUE | 2026-06-12 | 2026-06-15 | #8 |
| 1-5 | Block 1 | 熟悉 RMPflow 基本概念跑官方 Follow Target 範例 | 1h | M1: 環境建立與架構 | TRUE | 2026-06-12 | 2026-06-16 | #9 |
| 2-1 | Block 2 | 建立 core/ + extension/ 目錄骨架（含 omniverse_api/ isaac_sim_impl_6_0/ ui/） | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-13 | 2026-06-16 | #10 |
| 2-2 | Block 2 | 建立 pytest 執行環境確認測試可在 WSL2 跑通 | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-13 | 2026-06-17 | #11 |
| 2-3 | Block 2 | 建立核心資料模型：BilliardState Observation Action ShotResult | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-13 | 2026-06-17 | #12 |
| 2-4 | Block 2 | 針對資料模型撰寫 Unit Test | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-14 | 2026-06-18 | #13 |
| 2-5 | Block 2 | 建立 ControllerBase 抽象介面 + Unit Test | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-14 | 2026-06-18 | #14 |
| 2-6 | Block 2 | 建立 ArticulationAPI 抽象介面（omniverse_api/） | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-15 | 2026-06-19 | #71 |
| 2-7 | Block 2 | 建立 PhysicsAPI 抽象介面（碰撞偵測接觸事件）（omniverse_api/） | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-16 | 2026-06-19 | #72 |
| 2-8 | Block 2 | 建立 RigidBodyAPI 抽象介面（球的位置速度查詢）（omniverse_api/） | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-17 | 2026-06-20 | #73 |
| 2-9 | Block 2 | 建立 StageAPI 抽象介面（omniverse_api/） | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-19 | 2026-06-20 | #74 |
| 2-10 | Block 2 | 建立 Debug Menu 骨架（extension/ui/debug_menu.py） | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-19 | 2026-06-21 | #75 |
| 3-1 | Block 3 | 研究撞球桌標準尺寸（9-ball 規格）設計場景比例 | 0.5h | M2: 場景與機器人 | TRUE | 2026-06-20 | 2026-06-21 | #76 |
| 3-2 | Block 3 | 建立撞球桌幾何體（桌面邊框球袋）USD 場景 | 1h | M2: 場景與機器人 | TRUE | 2026-06-20 | 2026-06-21 | #77 |
| 3-3 | Block 3 | 設定桌面物理材質（摩擦係數彈性係數） | 0.5h | M2: 場景與機器人 | TRUE | 2026-06-23 | 2026-06-21 | #78 |
| 3-4 | Block 3 | 建立 16 顆球（白球 + 1–9 號球）的 USD Prim | 0.5h | M2: 場景與機器人 | TRUE | 2026-06-23 | 2026-06-21 | #79 |
| 3-5 | Block 3 | 設定球的物理材質（質量摩擦係數彈性係數） | 0.5h | M2: 場景與機器人 | TRUE | 2026-06-23 | 2026-06-21 | #80 |
| 3-7 | Block 3 | 設計 9-ball 標準開球擺位邏輯 + Unit Test | 0.5h | M2: 場景與機器人 | TRUE | 2026-06-23 | 2026-06-22 | #82 |
| 3-8 | Block 3 | 確認球袋進球判定邏輯（接觸感測器或位置判定）+ Unit Test | 0.5h | M2: 場景與機器人 | TRUE | 2026-06-24 | 2026-06-23 | #83 |
| 3-9 | Block 3 | 場景重置函式（球回到開球位置）+ Unit Test | 0.5h | M2: 場景與機器人 | TRUE | 2026-06-26 | 2026-06-25 | #84 |
| 4-1 | Block 4 | 下載 UR5 URDF 確認檔案結構完整（決議改用 Nucleus 現成 USD） | 0.5h | M2: 場景與機器人 | TRUE | 2026-06-26 | 2026-06-24 | #85 |
| 4-2 | Block 4 | URDF → USD 轉換確認 Prim 路徑結構正確（決議跳過） | 1h | M2: 場景與機器人 | TRUE | 2026-06-26 | 2026-06-25 | #86 |
| 4-3 | Block 4 | 確認關節結構能用 API 讀取關節數量與名稱 | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-13 | 2026-07-13 | #87 |
| 4-3b | Block 4 | 建立 UR5Robot 類別（載入 Nucleus UR5 USD 設定世界座標） | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-13 | 2026-07-13 | #174 |
| 4-3c | Block 4 | UR5Robot Unit Test | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-13 | 2026-07-14 | #175 |
| 4-3d-impl | Block 4 | BilliardTable：移除手臂建立邏輯，新增 get_table_center() | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-14 | 2026-07-14 | #184 |
| 4-3d-test | Block 4 | BilliardTable Unit Test | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-14 | 2026-07-14 | #185 |
| 4-3e-impl | Block 4 | TableRobotManager：新增手臂操作中介層 | 1h | M2: 場景與機器人 | FALSE |  | 2026-07-15 | #186 |
| 4-3e-test | Block 4 | TableRobotManager Unit Test | 1h | M2: 場景與機器人 | FALSE |  | 2026-07-16 | #187 |
| 4-3f | Block 4 | BilliardExtension：分離訓練桌/Demo 桌，接入 TableRobotManager 與開關 callback | 1h | M2: 場景與機器人 | FALSE |  | 2026-07-17 | #188 |
| 4-3g | Block 4 | DebugMenu：新增訓練/Demo 執行中開關 UI | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-07-18 | #189 |
| 4-4 | Block 4 | 設計球桿幾何體（USD Prim）確認尺寸比例合理 | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-07-18 | #88 |
| 4-5 | Block 4 | 設計球桿與 UR5 末端的固定連結（Fixed Joint） | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-07-18 | #89 |
| 4-8 | Block 4 | 空揮測速：單獨場景量測 TCP 峰值速度（含 asset velocity/effort limit 檢查）→ 定 A 動作空間速度上限 | 3h | M2: 場景與機器人 | FALSE |  | 2026-07-19 | #176 |
| 9-0 | Block 9 | Early termination 設計確認（球靜止偵測 → 立即計算 reward 並 reset） | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-19 | #178 |
| 3-6 | Block 3 | 確認球的碰撞與滾動物理行為正常（PhysX 參數調校） | 1h | M2: 場景與機器人 | FALSE |  | 2026-07-19 | #81 |
| 3-9b | Block 3 | 場景整體穩定性確認，物理仿真無異常 | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-07-19 | #153 |
| 5-1 | Block 5 | 設計擊球參數資料格式（放置XY 方向角 速度 偏移2）+ Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-19 | #92 |
| 5-2 | Block 5 | 設計 ScriptController 狀態機（A 版 STRIKING = set_velocities） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-19 | #93 |
| 5-3 | Block 5 | 撰寫狀態機 Unit Test（Mock ArticulationAPI） | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-20 | #94 |
| 5-4 | Block 5 | 實作 ArticulationAPIImpl（isaac_sim_impl_6_0/） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-20 | #95 |
| 5-11 | Block 5 | 實作 impulse-based 擊球（set_velocities + spin_efficiency 轉換）+ Unit Test | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-21 | #177 |
| 5-7 | Block 5 | 實作 WAITING：等待所有球靜止（速度閾值判定）+ Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-22 | #98 |
| 5-8 | Block 5 | 實作 RESET：場景重置 → 回到 IDLE | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-22 | #99 |
| 5-9 | Block 5 | 單次擊球循環跑通確認（set_velocities 版） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-23 | #100 |
| 5-10 | Block 5 | 物理參數調校：確認賦速後球散開效果合理 | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-24 | #101 |
| 6-1 | Block 6 | 設計 ShotResult 資料格式（各球最終位置 進袋狀態 散開分數） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-24 | #102 |
| 6-2 | Block 6 | 實作散開分數（凸包面積×0.5 + 平均最近鄰距離×0.5）+ Unit Test | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-25 | #103 |
| 6-3 | Block 6 | 實作白球進袋判定（−3.5）+ Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-25 | #104 |
| 6-4 | Block 6 | 實作 9 號球進袋加分判定（+3.0 無犯規）+ Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-25 | #105 |
| 6-7 | Block 6 | 實作犯規判定（未先觸 1 號球 −1.5 / 碰庫 < 4 −0.5）+ Unit Test | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-25 | #154 |
| 6-5 | Block 6 | 實作 Reward Function（整合散開分數與進袋獎懲）+ Unit Test | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-25 | #106 |
| 6-8 | Block 6 | 整合完整 Reward Function + Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-26 | #155 |
| 7-1 | Block 7 | 確認 Observation 資料格式（20 維：1–9 號球 XY + 母球 XY，無手臂資訊） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-26 | #108 |
| 7-1b | Block 7 | 實作 RigidBodyAPIImpl（isaac_sim_impl_6_0/，查詢球的位置與速度） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-26 | #156 |
| 7-2b | Block 7 | 實作 LivePositionProvider（isaac_sim_impl_6_0/，即時查詢球位置） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-26 | #157 |
| 7-2 | Block 7 | 實作 Observation 收集函式（從 RigidBodyAPI 取得）+ Unit Test | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-26 | #109 |
| 7-3 | Block 7 | 確認 Action 資料格式（6 維，速度上限依 #176 實測） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-26 | #110 |
| 7-4 | Block 7 | 確認 ControllerBase：get_action(observation) → action 足以支撐未來 ModelController | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-26 | #111 |
| 7-5 | Block 7 | 在 ScriptController 中加入 Observation 收集與 Action 格式輸出 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-27 | #112 |
| 9-1 | Block 9 | 研究 Isaac Lab 環境設計規範（gym.Env 介面） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-28 | #120 |
| 9-2 | Block 9 | 實作 BilliardEnv：繼承 Isaac Lab 環境介面整合 core/ 邏輯（含 early termination） | 1.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-29 | #121 |
| 9-3 | Block 9 | 確認 BilliardEnv 的 reset() step() observation_space action_space 正確 | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-30 | #122 |
| 9-4 | Block 9 | 選定 RL 演算法（PPO）設定超參數 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-31 | #123 |
| 9-5 | Block 9 | 單環境訓練跑通確認（確認 reward 有在變化） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-01 | #124 |
| 10-1 | Block 10 | 研究 Isaac Lab 多環境並行設計（Vectorized Environment） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-01 | #129 |
| 10-2 | Block 10 | 調整 BilliardEnv 支援多環境實例化 | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-02 | #130 |
| 10-3 | Block 10 | 測試 8 台並行確認物理仿真穩定 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-02 | #131 |
| 10-4 | Block 10 | 逐步擴大規模（32 → 64 → 128）記錄每個規模的 FPS | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-02 | #132 |
| 10-5 | Block 10 | 確認最大可穩定運行的環境數量 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-02 | #133 |
| 9-6 | Block 9 | 確認訓練過程中 reward 曲線有上升趨勢（訓練跑背景，此為監控判讀） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-02 | #125 |
| 9-7 | Block 9 | 儲存訓練好的模型確認可以載入並執行推論 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-02 | #126 |
| A-CP | Block 10 | Milestone A 收斂判定點（未收斂 → 偏移三檔/降環境數重跑，Wave 1 改依 fallback） | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-02 | #179 |
| B-1 | Block 13 | 可達性掃描與可行性地圖（orientation-constrained IK + 後擺走廊） | 5h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-08 | #180 |
| 4-7 | Block 4 | 確認 UR5 + 球桿基座位置（併入 B-1 可行性地圖） | 0.5h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-08 | #91 |
| 9-8 | Block 9 | 實作 ModelController（載入訓練模型替換 ScriptController） | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-08 | #127 |
| 9-9 | Block 9 | 確認 ModelController 執行效果優於隨機參數 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-09 | #128 |
| 8-1 | Block 8 | 實作擊球參數的可調介面（桿速範圍角度範圍位置偏移範圍） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-08-09 | #114 |
| 8-2 | Block 8 | HUD 新增參數控制面板（可即時調整擊球參數） | 1h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-08-09 | #115 |
| 8-3 | Block 8 | HUD 新增 ShotResult 顯示（散開分數白球狀態9號球狀態） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-08-09 | #116 |
| 6-6 | Block 6 | Debug Menu 新增「顯示當前 ShotResult」按鈕手動驗證計算正確性 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-08-09 | #107 |
| 7-6 | Block 7 | Debug Menu 新增「印出當前 Observation」按鈕 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-08-09 | #113 |
| 8-4 | Block 8 | 確認手動調整參數 → 擊球 → 結果顯示的完整流程 | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-08-09 | #117 |
| 8-5 | Block 8 | 錄製中途展示 Demo 影片（RL 訓練成果 + 參數化擊球） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-08-09 | #118 |
| 8-6 | Block 8 | LinkedIn 篇6 草稿撰寫與發布（USD Collection + Material Binding） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-08-09 | #119 |
| 12-4 | Block 12 | README 架構圖 + 技術亮點撰寫（Wave 1 前置） | 1.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-11 | #142 |
| 12-5 | Block 12 | README 接口設計說明（RL 訓練架構 + 版本升級策略） | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-12 | #143 |
| 12-6 | Block 12 | README 收尾確認上傳 GitHub | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-12 | #144 |
| B-2 | Block 13 | 關節空間揮桿軌跡生成（後擺 → 加速 → 擊球點，joint target 播放） | 15h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-19 | #181 |
| 4-6 | Block 4 | 評估是否仍需 RMPflow 設定檔（僅非擊球移動用途） | 1h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-20 | #90 |
| 5-5 | Block 5 | 手臂 AIMING（併入 B-2 關節空間軌跡） | 1h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-19 | #96 |
| 5-6 | Block 5 | 手臂 STRIKING（併入 B-2 關節空間軌跡） | 1h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-19 | #97 |
| B-3 | Block 13 | 真實接觸物理校正（physics dt 1e-4 + CCD + spin_efficiency） | 20h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-29 | #182 |
| B-CP | Block 13 | Fallback 檔位決策點（8/20 軌跡判定、8/23 偏移判定） | 0.5h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-23 | #183 |
| 11-1 | Block 11 | 全流程跑通確認（多環境並行 → RL 訓練 → 收斂 → ModelController 執行） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-30 | #134 |
| 11-2 | Block 11 | Debug Menu 所有按鈕確認 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-30 | #135 |
| 11-3 | Block 11 | 穩定性測試（長時間訓練確認無記憶體洩漏） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-30 | #136 |
| 11-4 | Block 11 | API 掃描：執行 api-scanner 掃描 isaac_sim_impl_6_0/ 產出 API 使用清單 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-31 | #137 |
| 11-5 | Block 11 | 補坑收尾 | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-01 | #138 |
| 12-1 | Block 12 | Demo 影片腳本規劃（RL 訓練 → 手臂執行完整 pipeline） | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-02 | #139 |
| 12-2 | Block 12 | 錄製 Demo 影片（OBS） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-03 | #140 |
| 12-3 | Block 12 | 影片剪輯確認 | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-05 | #141 |
| 12-7 | Block 12 | LinkedIn 篇8 草稿撰寫與發布（含精度/速度 trade-off 敘事） | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-05 | #145 |
| 12-8 | Block 12 | LinkedIn 篇8 潤稿確認 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-06 | #146 |
