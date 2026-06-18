# Phase 3 進度排程表

## 排程規則

- 6 月：平日（一～五）1h／天，週六 1h，週日 4h（2×2h）
- 7 月起：平日（一～五）1h／天，週六 4h（2×2h），週日 4h（2×2h）
- 7/3–7/12：台灣旅行，整段不安排工作時間
- 起算日：2026/06/12（之前已完成的任務維持原完成日期，預計完成日期欄位僅供對照排程節奏）
- 排程結果：全部任務預計於 **2026/08/01** 完成，較目標 8/13 提前 12 天

## 任務排程明細

| 任務ID | Block | 任務名稱 | 預估時數 | Milestone | 完成 | 完成日期 | 預計完成日期 | GitHub Issue |
|---|---|---|---|---|---|---|---|---|
| 0-1 | Block 0 | 確認硬體需求、查閱 Isaac Sim 5.1.0 系統需求文件 | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-09 | 2026-06-12 | #1 |
| 0-2 | Block 0 | 下載並安裝 Isaac Sim 5.1.0 Workstation 版本 | 1h | M1: 環境建立與架構 | TRUE | 2026-06-10 | 2026-06-13 | #2 |
| 0-3 | Block 0 | 啟動 Isaac Sim 執行 Compatibility Checker 確認環境正常 | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-13 | #3 |
| 0-4 | Block 0 | 跑官方 Hello World 範例確認 Python API 可執行 | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-14 | #4 |
| 1-1 | Block 1 | 熟悉 Isaac Sim Stage / Prim 基本操作 | 1h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-14 | #5 |
| 1-2 | Block 1 | 熟悉 Isaac Sim Python API：建立物件設定位置材質 | 1h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-14 | #6 |
| 1-3 | Block 1 | 熟悉 Articulation API 基本操作（讀取關節設定位置） | 1h | M1: 環境建立與架構 | TRUE | 2026-06-11 | 2026-06-14 | #7 |
| 1-4 | Block 1 | 跑官方機械手臂相關範例（Franka 或 UR10e） | 1h | M1: 環境建立與架構 | TRUE | 2026-06-12 | 2026-06-15 | #8 |
| 1-5 | Block 1 | 熟悉 RMPflow 基本概念跑官方 Follow Target 範例 | 1h | M1: 環境建立與架構 | TRUE | 2026-06-12 | 2026-06-16 | #9 |
| 2-1 | Block 2 | 建立 core/ + extension/ 目錄骨架（含 omniverse_api/ isaac_sim_impl_5_1/ ui/） | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-13 | 2026-06-16 | #10 |
| 2-2 | Block 2 | 建立 pytest 執行環境確認測試可在 WSL2 跑通 | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-13 | 2026-06-17 | #11 |
| 2-3 | Block 2 | 建立核心資料模型：BilliardState Observation Action ShotResult | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-13 | 2026-06-17 | #12 |
| 2-4 | Block 2 | 針對資料模型撰寫 Unit Test | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-14 | 2026-06-18 | #13 |
| 2-5 | Block 2 | 建立 ControllerBase 抽象介面 + Unit Test | 0.5h | M1: 環境建立與架構 | TRUE | 2026-06-14 | 2026-06-18 | #14 |
| 2-6 | Block 2 | 建立 ArticulationAPI 抽象介面（omniverse_api/） | 0.5h | M1: 環境建立與架構 | TRUE | 6/15 | 2026-06-19 | #71 |
| 2-7 | Block 2 | 建立 PhysicsAPI 抽象介面（碰撞偵測接觸事件）（omniverse_api/） | 0.5h | M1: 環境建立與架構 | TRUE | 6/16 | 2026-06-19 | #72 |
| 2-8 | Block 2 | 建立 RigidBodyAPI 抽象介面（球的位置速度查詢）（omniverse_api/） | 0.5h | M1: 環境建立與架構 | TRUE | 6/17 | 2026-06-20 | #73 |
| 2-9 | Block 2 | 建立 StageAPI 抽象介面（omniverse_api/） | 0.5h | M1: 環境建立與架構 | FALSE |  | 2026-06-20 | #74 |
| 2-10 | Block 2 | 建立 Debug Menu 骨架（extension/ui/debug_menu.py） | 0.5h | M1: 環境建立與架構 | FALSE |  | 2026-06-21 | #75 |
| 3-1 | Block 3 | 研究撞球桌標準尺寸（9-ball 規格）設計場景比例 | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-21 | #76 |
| 3-2 | Block 3 | 建立撞球桌幾何體（桌面邊框球袋）USD 場景 | 1h | M2: 場景與機器人 | FALSE |  | 2026-06-21 | #77 |
| 3-3 | Block 3 | 設定桌面物理材質（摩擦係數彈性係數） | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-21 | #78 |
| 3-4 | Block 3 | 建立 16 顆球（白球 + 1–9 號球）的 USD Prim | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-21 | #79 |
| 3-5 | Block 3 | 設定球的物理材質（質量摩擦係數彈性係數） | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-21 | #80 |
| 3-6 | Block 3 | 確認球的碰撞與滾動物理行為正常（PhysX 參數調校） | 1h | M2: 場景與機器人 | FALSE |  | 2026-06-22 | #81 |
| 3-7 | Block 3 | 設計 9-ball 標準開球擺位邏輯 + Unit Test | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-22 | #82 |
| 3-8 | Block 3 | 確認球袋進球判定邏輯（接觸感測器或位置判定）+ Unit Test | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-23 | #83 |
| 3-9 | Block 3 | 場景重置函式（球回到開球位置）+ Unit Test | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-23 | #84 |
| 4-1 | Block 4 | 下載 UR5 URDF 確認檔案結構完整 | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-24 | #85 |
| 4-2 | Block 4 | URDF → USD 轉換確認 Prim 路徑結構正確 | 1h | M2: 場景與機器人 | FALSE |  | 2026-06-25 | #86 |
| 4-3 | Block 4 | 確認關節結構能用 API 讀取關節數量與名稱 | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-25 | #87 |
| 4-4 | Block 4 | 設計球桿幾何體（USD Prim）確認尺寸比例合理 | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-26 | #88 |
| 4-5 | Block 4 | 設計球桿與 UR5 末端的固定連結（Fixed Joint） | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-26 | #89 |
| 4-6 | Block 4 | 產出 UR5 的 RMPflow 設定檔（ur5_rmpflow_common.yaml） | 1h | M2: 場景與機器人 | FALSE |  | 2026-06-27 | #90 |
| 4-7 | Block 4 | 確認 UR5 + 球桿整體在場景中的擺放位置合理 | 0.5h | M2: 場景與機器人 | FALSE |  | 2026-06-28 | #91 |
| 5-1 | Block 5 | 設計擊球參數資料格式（桿速擊球角度擊球位置偏移）+ Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-06-28 | #92 |
| 5-2 | Block 5 | 設計 ScriptController 狀態機（IDLE/AIMING/STRIKING/WAITING/RESET/ERROR） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-06-28 | #93 |
| 5-3 | Block 5 | 撰寫狀態機 Unit Test（Mock ArticulationAPI） | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-06-28 | #94 |
| 5-4 | Block 5 | 實作 ArticulationAPIImpl（isaac_sim_impl_5_1/） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-06-28 | #95 |
| 5-5 | Block 5 | 實作 AIMING：RMPflow 將球桿末端移到擊球預備位置 | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-06-28 | #96 |
| 5-6 | Block 5 | 實作 STRIKING：沿擊球方向加速推進（模擬揮桿衝擊） | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-06-29 | #97 |
| 5-7 | Block 5 | 實作 WAITING：等待所有球靜止（速度閾值判定）+ Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-06-30 | #98 |
| 5-8 | Block 5 | 實作 RESET：場景重置 → 回到 IDLE | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-06-30 | #99 |
| 5-9 | Block 5 | 單次擊球循環跑通確認 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-01 | #100 |
| 5-10 | Block 5 | 物理參數調校：確認球桿衝擊力道傳遞正確球散開效果合理 | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-02 | #101 |
| 6-1 | Block 6 | 設計 ShotResult 資料格式（各球最終位置白球是否進袋9號球是否進袋散開分數） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-02 | #102 |
| 6-2 | Block 6 | 實作散開程度計算函式（球的分布面積或平均距離）+ Unit Test | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-13 | #103 |
| 6-3 | Block 6 | 實作白球進袋判定 + Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-14 | #104 |
| 6-4 | Block 6 | 實作 9 號球進袋加分判定 + Unit Test | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-14 | #105 |
| 6-5 | Block 6 | 實作 Reward Function（整合散開分數白球進袋懲罰9號球進袋加分）+ Unit Test | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-15 | #106 |
| 6-6 | Block 6 | Debug Menu 新增「顯示當前 ShotResult」按鈕手動驗證計算正確性 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-16 | #107 |
| 7-1 | Block 7 | 確認 Observation 資料格式（各球位置白球位置手臂關節角度擊球參數） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-16 | #108 |
| 7-2 | Block 7 | 實作 Observation 收集函式（從 RigidBodyAPI / ArticulationAPI 取得）+ Unit Test | 1h | M3: 擊球動作與評估 | FALSE |  | 2026-07-17 | #109 |
| 7-3 | Block 7 | 確認 Action 資料格式（桿速擊球角度擊球位置偏移） | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-18 | #110 |
| 7-4 | Block 7 | 確認 ControllerBase：get_action(observation) → action 足以支撐未來 ModelController | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-18 | #111 |
| 7-5 | Block 7 | 在 ScriptController 中加入 Observation 收集與 Action 格式輸出 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-18 | #112 |
| 7-6 | Block 7 | Debug Menu 新增「印出當前 Observation」按鈕 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-07-18 | #113 |
| 8-1 | Block 8 | 實作擊球參數的可調介面（桿速範圍角度範圍位置偏移範圍） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-07-18 | #114 |
| 8-2 | Block 8 | HUD 新增參數控制面板（可即時調整擊球參數） | 1h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-07-18 | #115 |
| 8-3 | Block 8 | HUD 新增 ShotResult 顯示（散開分數白球狀態9號球狀態） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-07-18 | #116 |
| 8-4 | Block 8 | 確認手動調整參數 → 擊球 → 結果顯示的完整流程 | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-07-19 | #117 |
| 8-5 | Block 8 | 錄製中途展示 Demo 影片（單台撞球機器人參數化擊球） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-07-19 | #118 |
| 8-6 | Block 8 | LinkedIn 篇6 草稿撰寫與發布 | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-07-19 | #119 |
| 9-1 | Block 9 | 研究 Isaac Lab 環境設計規範（gym.Env 介面） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-19 | #120 |
| 9-2 | Block 9 | 實作 BilliardEnv：繼承 Isaac Lab 環境介面整合 core/ 邏輯 | 1.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-19 | #121 |
| 9-3 | Block 9 | 確認 BilliardEnv 的 reset() step() observation_space action_space 正確 | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-20 | #122 |
| 9-4 | Block 9 | 選定 RL 演算法（PPO）設定超參數 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-21 | #123 |
| 9-5 | Block 9 | 單環境訓練跑通確認（確認 reward 有在變化） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-22 | #124 |
| 9-6 | Block 9 | 確認訓練過程中 reward 曲線有上升趨勢 | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-23 | #125 |
| 9-7 | Block 9 | 儲存訓練好的模型確認可以載入並執行推論 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-23 | #126 |
| 9-8 | Block 9 | 實作 ModelController（載入訓練模型替換 ScriptController） | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-24 | #127 |
| 9-9 | Block 9 | 確認 ModelController 執行效果優於隨機參數 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-24 | #128 |
| 10-1 | Block 10 | 研究 Isaac Lab 多環境並行設計（Vectorized Environment） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-25 | #129 |
| 10-2 | Block 10 | 調整 BilliardEnv 支援多環境實例化 | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-25 | #130 |
| 10-3 | Block 10 | 測試 8 台並行確認物理仿真穩定 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-25 | #131 |
| 10-4 | Block 10 | 逐步擴大規模（32 → 64 → 128）記錄每個規模的 FPS | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-25 | #132 |
| 10-5 | Block 10 | 確認最大可穩定運行的環境數量 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-07-25 | #133 |
| 11-1 | Block 11 | 全流程跑通確認（多環境並行 → RL 訓練 → 收斂 → ModelController 執行） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-26 | #134 |
| 11-2 | Block 11 | Debug Menu 所有按鈕確認 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-26 | #135 |
| 11-3 | Block 11 | 穩定性測試（長時間訓練確認無記憶體洩漏） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-26 | #136 |
| 11-4 | Block 11 | API 掃描：執行 api-migration-agent 掃描 isaac_sim_impl_5_1/ 產出 API 使用清單 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-26 | #137 |
| 11-5 | Block 11 | 補坑收尾 | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-26 | #138 |
| 12-1 | Block 12 | Demo 影片腳本規劃（單台參數化 → 多台並行訓練 → 學習曲線 → 最佳參數展示） | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-27 | #139 |
| 12-2 | Block 12 | 錄製 Demo 影片（OBS） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-28 | #140 |
| 12-3 | Block 12 | 影片剪輯確認 | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-29 | #141 |
| 12-4 | Block 12 | README 架構圖 + 技術亮點撰寫 | 1.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-30 | #142 |
| 12-5 | Block 12 | README 接口設計說明（RL 訓練架構 + 版本升級策略） | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-31 | #143 |
| 12-6 | Block 12 | README 收尾確認上傳 GitHub | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-07-31 | #144 |
| 12-7 | Block 12 | LinkedIn 篇8 草稿撰寫與發布 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-01 | #145 |
| 12-8 | Block 12 | LinkedIn 篇8 潤稿確認 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-08-01 | #146 |
