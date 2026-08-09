# Phase 3 進度排程表

## 排程規則

- 2026-08-01 全面重排（雲端訓練 + 週末制）。此前的規則（平日 1h/天、週六日各 4h、A 收斂硬約束 7/31、全部完成 9/06）**全數作廢**，以下為現行版本。
- **工時模型**：平日（一～五）0.5h／天、週末（六日）5h／天。
  - ⚠️ **平日 0.5h 只能巡檢，不排開發任務**。有效開發容量 = 週末日 × 5h。
  - 2026-08-02 ~ 2026-09-19 共 14 個週末日 = **70h 有效容量**，工作量約 62h，名目緩衝 8h（11%）。
  - 套 1.3× 歷史低估修正 → 約 81h，短缺約 11h ≈ 2.2 個週末日。**不做調整的達成率約 40–50%**。
  - 2026-08-02 修訂：#226（0.5h→1h）、#227（1.5h→2h）合計 +1h，用於 Block 12「訓練過程」段落的素材。雲端 headless 沒有訓練當下的畫面，只能靠不同階段 checkpoint 的對照回放呈現，且素材需求必須在訓練開始前設好（見 `training/README.md`）。
- **平日巡檢用途**：看雲端訓練 reward 曲線、確認 checkpoint 有寫入、GPU 是否掉線、費用累計；決定要不要調參重啟。不需開 Isaac Sim 的文書/測試工作也可放平日。
- **加速槓桿（兩者疊加約可補齊 9h，對 11h 缺口仍差 2h，達成率升至 ~65–70%）**：
  - 平日 0.5h → 1.0h：省 5–6h，**性價比最高**。再往 1.5h/2.0h 邊際效益快速遞減（平日可做的任務池本身有限）。
  - sub-agent 委派：僅省 ~3.5h（unit-test 寫測試骨架、tech-doc 產 README 初稿、api-scanner、api-lookup）。**B-2 + B-3 共 26h 完全不可委派**——核心是「跑模擬 → 看行為 → 判斷對不對 → 調參」的閉環。
  - **2026-08-03 首次實踐（8/03–8/07）**：把 #222 + #225 + #121 草擬移到平日，共約 3.5h，直接把 8/08 週末時段讓出來。判斷依據是「**本機能不能驗證**」——`core/` 是純 Python 邏輯，pytest 跑得動、不依賴 Isaac Lab 或 GPU，是最適合平日的任務類型。
    - **本週平日需約 1h/天**（0.5h/天只夠 #222+#225 的 2.5h，#121 排不進去）。
    - **#222 與 #225 應合併實作**——#222 是 21 維 Encoder 本身、#225 是把它做成訓練端與 Demo 端共用；Encoder 一開始就寫在 `core/` 共用層裡的話，#225 同時完成。分開做等於寫兩遍。
    - **#121 只能草擬不能驗證**（需 `import isaaclab`，本機刻意不安裝）。平日寫 cfg／`observation_space` 接線／呼叫 `core/`，週末在 RunPod 上花約 0.5h 跑通。接受「寫的時候無法驗證 import 與型別」的代價換取週末時段。
- **訓練平台改為雲端**（**規格 2026-08-03 實測定案**，取代 08-02 的暫定值）：RunPod **Secure Cloud**／**EU-RO-1**／**RTX A4500（$0.25/hr）**＋ **100GB Network Volume**（$0.07/GB/月，**Pod 停機仍計費**）＋ HTTP Port 6006（TensorBoard）。專案期間總成本估 **$24**（GPU 約 50h ≈ $12.5 ＋ Volume 1.6 個月 ≈ $11.2）。**本機不需安裝 Isaac Lab**。訓練在背景跑不佔工時：週末啟動 → 整週背景跑 → 下個週末收成果。需開頻繁 checkpoint，不訓練時停機。
  - 選 Secure Cloud 而非 Community：Network Volume 僅 Secure Cloud 支援。**原定 RTX 4000 Ada（$0.28/hr）在 EU-RO-1 缺貨**（RTX 3090 同樣缺貨），改用 A4500 反而 RAM 與 vCPU 更多且更便宜。不選 4090 的理由：Milestone A 是純剛體物理 + 小 MLP，負載輕，而 4090 的 vCPU 只有 5（全場最少），Isaac Sim 啟動時的 USD 解析是 CPU 密集。
  - ⚠️ **官方 container `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` 已實測失敗並廢棄** —— 其 ENTRYPOINT 強制啟動 Isaac Sim Streaming，而 RunPod 的 start command 只覆寫 CMD 蓋不掉，導致容器不斷重啟、SSH 一進去就被踢。改用 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` + pip 安裝到 volume 上的 venv。
  - ⚠️ **Container Disk 停機即清空**，但 **venv 在 volume 上所以 Python 套件持久化，不需每週重裝**；只有 apt 套件與 symlink 要重建（`source /workspace/setup.sh` 處理）。Network Volume 掛在 `/workspace` 會遮蔽 container 內同路徑內容（官方 container 的 `/workspace/isaaclab` 就是這樣消失的）。**快取不需重導**（實測 Omniverse 未寫入 HOME，58s 冷啟動與 shader 快取無關）。
  - 日常操作流程（連線、tmux、巡檢、停機、Demo 素材要求）見 `training/README.md`。
- **雙 Milestone 結構**
  - **Milestone A（RL 訓練）**：impulse-based（`set_velocities`）、無手臂、雲端 1024+ 平行環境。
  - **Milestone B（手臂執行，GitHub M7）**：UR5 + 球桿剛性連結、單一環境、**在本機跑**（需開 GUI 看物理行為）。承諾範圍 = fallback **(b)**。
- **Fallback 檔位**：(a) 全功能含 spin 偏移 → **(b) 方向+速度正確、中心擊球、真實桿尖接觸 ← 承諾範圍** → (c) 空揮到位 + programmatic impulse。
  - **(c) 已被否決**：手臂揮桿與球的運動在物理上無因果關係，技術面試被問「球是真的被打到的嗎」會答不出來。
  - (b) 與 (c) 差 23.5h ≈ 5 個週末日，關鍵在 B-3——(b) 雖放棄 spin 但仍要求球是真的被桿尖打出去的，B-3 一項都省不掉。
- **決策點**
  - **2026-08-15 A-CP（#179）**：reward 曲線收斂或明確上升趨勢。**未收斂則直接帶不完美 policy 進入 Milestone B，不為 A 犧牲 B 的時間**——B 不依賴 A 的訓練結果，只依賴手臂本身。
  - **2026-08-16 B-CP1（#180）**：IK 有解、後擺空間足夠支撐 (b) 檔位。
  - **2026-08-29 B-CP2（#181）**：B-2 完工確認（動作連續、無關節限位衝突、擊球點速度在 #176 實測上限 1.313 m/s 內）。**這是完工確認，不是降級判斷點**。
  - **2026-09-06 B-CP3（#182，最後防線）**：球的運動方向/速度與揮桿有清楚因果關係，同參數重複 3 次落點一致。**未達標不回頭砍到 (c)，改砍 Block 12 深度換時間**。
- **Observation 設計變更（20 維 → 21 維）**：新增 `max_offset ∈ [0.0, 1.0]` 條件變數，用單一 policy 做 (a)/(b) 行為切換，取代「訓練兩個 policy」。
  - **必須放在 observation，不可放在 action**（放 action 會讓 policy 自己選，它必然選滿檔）。
  - 訓練時每 episode 從 `[0,1]` 均勻取樣；**環境端**負責把 policy 輸出的偏移量 clamp 到該上限。
  - 推論時填入 #180 量到的手臂實際偏移能力（0.0 = 中心擊球 / 1.0 = 完整三檔）。
  - 21 維組裝必須放 `core/` 共用層，訓練端與 Demo 端 import 同一份。**維度或欄位順序不一致時 policy 不報錯，只會安靜輸出垃圾動作**。
- **已合併／作廢的任務（不再列於下表）**
  - #91（UR5+球桿基座位置）→ 併入 #180 作為驗收條件。
  - #96 / #97（RMPflow AIMING / STRIKING）→ superseded，方法論已被 #181 關節空間軌跡取代。
  - #119（LinkedIn 篇6 草稿）→ M4 中途展示合併進最終發布，不獨立發篇6，敘事併入 Block 12 單篇（參數化控制 → RL 訓練 → 手臂執行完整弧線）。
  - #129–#133（多環境漸進放大）→ 合併為 #223，估時 4h → 1h。漸進策略是為 RTX 4060 8GB 本機設計的，雲端 GPU 記憶體充裕後前提消失。
- **Wave 1 求職 outreach 與專案完成度脫鉤**：8 月中即可開始投遞與更新 LinkedIn 個人檔案，不需等專案 100% 完成。
- ⚠️ **最高風險：B-3 真實接觸物理校正（#182，14h）**。定義本身模糊（「看起來正確且大致可重複」無客觀停止條件）、不可委派、且必須在本機跑——而本機有兩個未解的藍屏問題（Avast 過濾驅動、i9-14900HX MCE）。**B-2/B-3 期間每 15–20 分鐘手動存檔一次**，把單次藍屏損失壓在一輪迭代內。
- ⚠️ `project-tools/check_progress.py` 第 97 行仍硬編碼「預計完成日 2026-09-06」與「Milestone A 收斂硬約束 2026-07-31」，兩者皆已作廢（現為 2026-09-19 / 2026-08-15），需另行修正該程式碼。

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
| 4-3e-impl | Block 4 | TableRobotManager：新增手臂操作中介層 | 1h | M2: 場景與機器人 | TRUE | 2026-07-14 | 2026-07-15 | #186 |
| 4-3e-test | Block 4 | TableRobotManager Unit Test | 1h | M2: 場景與機器人 | TRUE | 2026-07-14 | 2026-07-16 | #187 |
| 4-3f | Block 4 | BilliardExtension：分離訓練桌/Demo 桌，接入 TableRobotManager 與開關 callback | 1h | M2: 場景與機器人 | TRUE | 2026-07-15 | 2026-07-17 | #188 |
| 4-3g | Block 4 | DebugMenu：新增訓練/Demo 執行中開關 UI | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-15 | 2026-07-18 | #189 |
| 4-4 | Block 4 | 設計球桿幾何體（USD Prim）確認尺寸比例合理 | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-15 | 2026-07-18 | #88 |
| 4-5 | Block 4 | 設計球桿與 UR5 末端的固定連結（Fixed Joint） | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-16 | 2026-07-18 | #89 |
| 4-8 | Block 4 | 空揮測速：單獨場景量測 TCP 峰值速度（含 asset velocity/effort limit 檢查）→ 定 A 動作空間速度上限 | 3h | M2: 場景與機器人 | TRUE | 2026-07-17 | 2026-07-19 | #176 |
| 9-0 | Block 9 | Early termination 設計確認（球靜止偵測 → 立即計算 reward 並 reset） | 0.5h | M5: RL 訓練與多環境 | TRUE | 2026-07-17 | 2026-07-19 | #178 |
| 3-6 | Block 3 | 確認球的碰撞與滾動物理行為正常（PhysX 參數調校） | 1h | M2: 場景與機器人 | TRUE | 2026-07-26 | 2026-07-19 | #81 |
| 3-9b | Block 3 | 場景整體穩定性確認，物理仿真無異常 | 0.5h | M2: 場景與機器人 | TRUE | 2026-07-26 | 2026-07-19 | #153 |
| 5-1 | Block 5 | 設計擊球參數資料格式（放置XY 方向角 速度 偏移2）+ Unit Test | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-17 | 2026-07-19 | #92 |
| 5-2 | Block 5 | 設計 ScriptController 狀態機（A 版 STRIKING = set_velocities） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-17 | 2026-07-19 | #93 |
| 5-3 | Block 5 | 撰寫狀態機 Unit Test（Mock ArticulationAPI） | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-19 | 2026-07-20 | #94 |
| 5-4 | Block 5 | 實作 ArticulationAPIImpl（isaac_sim_impl_6_0/） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-19 | 2026-07-20 | #95 |
| 5-11 | Block 5 | 實作 impulse-based 擊球（set_velocities + spin_efficiency 轉換）+ Unit Test | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-19 | 2026-07-21 | #177 |
| 5-7 | Block 5 | 實作 WAITING：等待所有球靜止（速度閾值判定）+ Unit Test | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-19 | 2026-07-22 | #98 |
| 5-8 | Block 5 | 實作 RESET：場景重置 → 回到 IDLE | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-22 | 2026-07-22 | #99 |
| 5-9-impl | Block 5 | TableOrchestrator：串接 RESET 全流程（實作） | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-21 | 2026-07-23 | #194 |
| 5-9-test | Block 5 | TableOrchestrator Unit Test | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-21 | 2026-07-24 | #195 |
| 5-9b-impl | Block 5 | ErrorState + TableOrchestrator：錯誤處理與 reset() 支援 | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-21 | 2026-07-25 | #196 |
| 5-9b-test | Block 5 | ErrorState + TableOrchestrator Unit Test | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-21 | 2026-07-25 | #197 |
| 5-9c-impl | Block 5 | ObservationBuilder（Demo/Training）+ 兩個缺口 getter | 1.5h | M3: 擊球動作與評估 | TRUE | 2026-07-22 | 2026-07-25 | #198 |
| 5-9c-test | Block 5 | ObservationBuilder Unit Test | 1.5h | M3: 擊球動作與評估 | TRUE | 2026-07-22 | 2026-07-26 | #199 |
| 5-9d-impl | Block 5 | TableRuntime：組裝 ObservationBuilder + TableOrchestrator | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-22 | 2026-07-26 | #200 |
| 5-9d-test | Block 5 | TableRuntime Unit Test | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-22 | 2026-07-26 | #201 |
| 5-9e | Block 5 | Extension 訓練桌 timeline play/stop 生命週期串接 | 1.5h | M3: 擊球動作與評估 | TRUE | 2026-07-24 | 2026-07-26 | #202 |
| 5-9 | Block 5 | 單次擊球循環跑通確認（set_velocities 版） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-26 | 2026-07-26 | #100 |
| 5-10 | Block 5 | 物理參數調校：確認賦速後球散開效果合理 | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-26 | 2026-07-27 | #101 |
| 6-1 | Block 6 | 設計 ShotResult 資料格式（各球最終位置 進袋狀態 散開分數） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-26 | 2026-07-28 | #102 |
| 6-2 | Block 6 | 實作散開分數（凸包面積×0.5 + 平均最近鄰距離×0.5）+ Unit Test | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-26 | 2026-07-29 | #103 |
| 6-3 | Block 6 | 實作白球進袋判定（−3.5）+ Unit Test | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-26 | 2026-07-29 | #104 |
| 6-4 | Block 6 | 實作 9 號球進袋加分判定（+3.0 無犯規）+ Unit Test | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-27 | 2026-07-30 | #105 |
| 6-7 | Block 6 | 實作犯規判定（未先觸 1 號球 −1.5 / 碰庫 < 4 −0.5）+ Unit Test | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-28 | 2026-07-31 | #154 |
| 6-5 | Block 6 | 實作完整 Reward Function（整合散開分數、進袋獎懲與開球犯規）+ Unit Test | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-28 | 2026-08-01 | #106 |
| 6-8 | Block 6 | 整合完整 Reward Function + Unit Test（由 #106 取代，不另行實作） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-28 | 2026-08-01 | #155 |
| 7-1 | Block 7 | 確認 RL Observation 資料格式（20 維：1–9 號球 XY + 母球 XY，無手臂資訊） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-29 | 2026-08-01 | #108 |
| 7-1b | Block 7 | 實作 RigidBodyAPIImpl（isaac_sim_impl_6_0/，查詢球的位置與速度） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-29 | 2026-08-01 | #156 |
| 7-2b | Block 7 | 取消 LivePositionProvider（由 ObservationBuilder + RigidBodyAPI 即時查詢流程取代） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-29 | 2026-08-01 | #157 |
| 7-2 | Block 7 | 實作 RL Observation Encoder（既有 Observation → 20 維）+ Unit Test | 1h | M3: 擊球動作與評估 | TRUE | 2026-07-29 | 2026-08-01 | #109 |
| 7-3 | Block 7 | 確認 Action 資料格式（6 維，母球初速上限 3.3392 m/s） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-29 | 2026-08-01 | #110 |
| 7-4 | Block 7 | 補正 ControllerBase 完整生命週期契約（get_action/get_current_state/reset） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-29 | 2026-08-02 | #111 |
| 7-5 | Block 7 | 取消：Observation 收集由 ObservationBuilder/TableRuntime 負責，Action 契約已完成 | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-29 | 2026-08-02 | #112 |
| 7-2-rework | Block 7 | RL Observation Encoder 改 21 維（新增 max_offset 條件變數，訓練時均勻取樣、core/ 層 clamp）+ Unit Test【平日/本機可完成，與 #225 合併實作】 | 2h | M3: 擊球動作與評估 | TRUE | 2026-08-03 | 2026-08-05 | #222 |
| 8-1 | Block 8 | 實作 Action 物理域參數設定（母球擺位初速角度擊球偏移）【平日/本機可完成；#225／#122／#127 的前置，已從 9/06 提前】 | 0.5h | M4: 中途展示點 LinkedIn篇6 | TRUE | 2026-08-04 | 2026-08-04 | #114 |
| 9-C2 | Block 9 | core/ 共用 action 還原函式與正規化策略（6 維模型輸出 → Action，反正規化取自 #114）+ Unit Test【平日/本機可完成；observation 組裝已隨 #222 完成，一致性測試拆至 #228】 | 1h | M5: RL 訓練與多環境 | TRUE | 2026-08-06 | 2026-08-05 | #225 |
| 9-C1 | Block 9 | 雲端基礎設施建置（2026-08-06 完成：Pod/volume/venv 跨 Pod 持久化、Blackwell sm_120 驗證、watchdog 走 REST 實測停機成功；bootstrap 改為 volume 重建腳本） | 5h | M5: RL 訓練與多環境 | TRUE | 2026-08-06 | 2026-08-08 | #224 |
| 5-9f-impl | Block 5 | RollingResistanceService 實作（滾動摩擦，影響 reward 品質） | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-26 | 2026-08-08 | #203 |
| 5-9f-test | Block 5 | RollingResistanceService Unit Test | 0.5h | M3: 擊球動作與評估 | TRUE | 2026-07-26 | 2026-08-08 | #204 |
| 9-1 | Block 9 | 研究 Isaac Lab 環境設計規範（改用本機 IsaacLab 3.0.0 原始碼查證，未開 pod；gym.Env / ManagerBasedRLEnv 介面與環境註冊） | 1h | M5: RL 訓練與多環境 | TRUE | 2026-08-07 | 2026-08-08 | #120 |
| 9-2 | Block 9 | 實作 BilliardEnv：繼承 Isaac Lab 環境介面整合 core/ 邏輯（21 維 observation_space、含 early termination）【平日草擬 1h（08-07，本機無法驗證）+ 週末驗證跑通 0.5h】【實際約 6h：原拆分未涵蓋 B-6 滾動阻力 torch 重寫與 B-3a 進袋/接觸偵測，兩者都因 Demo 端走 physx 事件訂閱、向量化環境用不了而必須重寫】 | 1.5h | M5: RL 訓練與多環境 | TRUE | 2026-08-09 | 2026-08-08 | #121 |
| 9-3 | Block 9 | 確認 BilliardEnv 的 reset() step() observation_space(21維) action_space 正確【實際超出原估：#121 交付時 max_offset 未做逐 episode 取樣（第 21 維為定值 1.0，條件變數退化成常數），本項含補實作、權威 buffer 歸屬重新設計（移至 ActionTerm，ObsTerm 讀同一份）、pod 端到端驗證腳本 verify_max_offset.py】 | 1h | M5: RL 訓練與多環境 | TRUE | 2026-08-09 | 2026-08-09 | #122 |
| 9-4 | Block 9 | 選定 RL 演算法（PPO）設定超參數 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-09 | #123 |
| 9-5 | Block 9 | 單環境訓練跑通確認（確認 reward 有在變化）→ 啟動雲端背景訓練 | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-09 | #124 |
| 9-6 | Block 9 | 確認訓練過程中 reward 曲線有上升趨勢（訓練跑背景，此為監控判讀） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-15 | #125 |
| 9-7 | Block 9 | 儲存訓練好的模型確認可以載入並執行推論 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-15 | #126 |
| 9-C3 | Block 9 | 訓練成果取回流程（policy.pt / policy.onnx / env.yaml ＋ 中間 checkpoint 的 TorchScript ＋ TensorBoard event 檔） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-15 | #226 |
| 9-8 | Block 9 | 實作 ModelController（載入 TorchScript exported/policy.pt，normalizer 已內建） | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-15 | #127 |
| 9-C4 | Block 9 | core/ 共用層兩端呼叫路徑一致性測試（訓練端 vs Demo 端輸出一致、無重複實作）【拆分自 #225，需 #121/#122/#127 到位】 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-15 | #228 |
| 9-9 | Block 9 | 確認 ModelController 執行效果優於隨機參數 | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-15 | #128 |
| 10-merge | Block 10 | 雲端多環境並行：直接設定高環境數（1024+）+ 一次性穩定性檢查 | 1h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-15 | #223 |
| A-CP | Block 10 | Milestone A 收斂判定點（未收斂 → 直接帶不完美 policy 進 B，不為 A 犧牲 B 的時間） | 0.5h | M5: RL 訓練與多環境 | FALSE |  | 2026-08-15 | #179 |
| B-1 | Block 13 | 可達性掃描與可行性地圖（orientation-constrained IK + 後擺走廊）→ 量出手臂實際偏移能力填入 max_offset | 4h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-16 | #180 |
| 4-6 | Block 4 | RMPflow 設定檔（僅非擊球移動用途，如回到待命姿態） | 1h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-16 | #90 |
| B-2 | Block 13 | 關節空間揮桿軌跡生成（後擺 → 加速 → 擊球點，joint target 播放） | 12h | M7: Milestone B 手臂執行 | FALSE |  | 2026-08-29 | #181 |
| B-3 | Block 13 | 真實接觸物理校正（physics dt 1e-4 + CCD + spin_efficiency） | 14h | M7: Milestone B 手臂執行 | FALSE |  | 2026-09-06 | #182 |
| B-CP | Block 13 | Fallback 檔位決策點總覽（B-CP1 8/16、B-CP2 8/29、B-CP3 9/06） | 0.5h | M7: Milestone B 手臂執行 | FALSE |  | 2026-09-06 | #183 |
| 8-2 | Block 8 | HUD 新增參數控制面板（可即時調整擊球參數） | 1h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-09-06 | #115 |
| 8-3 | Block 8 | HUD 新增 ShotResult 顯示（散開分數白球狀態9號球狀態） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-09-06 | #116 |
| 6-6 | Block 6 | Debug Menu 新增「顯示當前 ShotResult」按鈕手動驗證計算正確性 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-09-06 | #107 |
| 7-6 | Block 7 | Debug Menu 新增「印出當前 Observation」按鈕 | 0.5h | M3: 擊球動作與評估 | FALSE |  | 2026-09-06 | #113 |
| 8-4 | Block 8 | 確認手動調整參數 → 擊球 → 結果顯示的完整流程 | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-09-06 | #117 |
| 8-5 | Block 8 | 錄製展示片段（參數化擊球，素材併入最終單篇 Demo） | 0.5h | M4: 中途展示點 LinkedIn篇6 | FALSE |  | 2026-09-12 | #118 |
| 11-1 | Block 11 | 全流程跑通確認（雲端訓練 → 匯出 policy → 本機 ModelController 執行） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-12 | #134 |
| 11-2 | Block 11 | Debug Menu 所有按鈕確認 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-12 | #135 |
| 11-3 | Block 11 | 穩定性測試（長時間訓練確認無記憶體洩漏） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-12 | #136 |
| 11-4 | Block 11 | API 掃描：執行 api-scanner 掃描 isaac_sim_impl_6_0/ 產出 API 使用清單 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-12 | #137 |
| 11-5 | Block 11 | 補坑收尾 | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-12 | #138 |
| 12-C1 | Block 12 | 本機 Demo 播放與錄影流程（GUI 運鏡錄影 ＋ 4 階段 checkpoint 對照回放 ＋ 多環境並行示意） | 2h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-13 | #227 |
| 12-1 | Block 12 | Demo 影片腳本規劃（完整故事弧線：參數化控制 → RL 訓練 → 手臂執行） | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-13 | #139 |
| 12-2 | Block 12 | 錄製 Demo 影片（OBS） | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-13 | #140 |
| 12-3 | Block 12 | 影片剪輯確認 | 1h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-13 | #141 |
| 12-4 | Block 12 | README 架構圖 + 技術亮點撰寫 | 1.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-19 | #142 |
| 12-5 | Block 12 | README 接口設計說明（雲端 RL 訓練架構 + 版本升級策略） | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-19 | #143 |
| 12-6 | Block 12 | README 收尾確認上傳 GitHub | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-19 | #144 |
| 12-7 | Block 12 | LinkedIn 草稿撰寫與發布（單篇整合版，含精度/速度 trade-off 敘事） | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-19 | #145 |
| 12-8 | Block 12 | LinkedIn 潤稿確認 | 0.5h | M6: 整合測試與發布 LinkedIn篇8 | FALSE |  | 2026-09-19 | #146 |
