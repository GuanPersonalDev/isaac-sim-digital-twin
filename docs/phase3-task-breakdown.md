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

## 2026-07-13 重排補註（雙 Milestone 結構）

依 `phase3-plan-risks-solutions.md`（專案根目錄）全面重排，該文件為困難點與解法的完整依據：

- **Milestone A（RL 訓練，7 月底前收斂）**：impulse-based（`set_velocities`）、無手臂、128 平行環境。手臂相關擊球任務全部移出 A。
- **Milestone B（手臂執行，8 月，GitHub Milestone M7）**：可行性地圖（#180）→ 關節空間揮桿軌跡（#181，取代 RMPflow 揮桿）→ 真實接觸物理校正（#182）。完成定義採 fallback 階梯（#183），檔位 (b)「方向+速度正確、中心擊球」為 Wave 1 完成線。
- **本週順序固定**：[4-8] 空揮測速（#176）→ [9-0] early termination 確認（#178）→ 完成後 A 訓練配置才定案。
- 排程明細與決策點日期見 `phase3-schedule.md`。

---

## RL 設計定案

### RL Observation（20 個數字）

固定順序：

`[ball_1_x, ball_1_y, ..., ball_9_x, ball_9_y, cue_ball_x, cue_ball_y]`

| 索引 | 資料 | 數量 | 說明 |
|---|---|---:|---|
| 0–17 | 1–9 號球 XY 位置 | 18 | 依 ball_id 由 1 到 9；桌台相對座標，單位 m |
| 18–19 | 母球 Ball_0 XY 位置 | 2 | 桌台相對座標，單位 m |
| | **總計** | **20** | |

資料來源為既有執行期 `Observation.ball_positions`。RL Encoder 將世界座標扣除
桌台世界 XY，只保留 XY、忽略 Z；進袋球沿用該次 `Observation` 中的當前 XY。
RL 向量不包含手臂關節角度、Action 擊球參數，亦不包含 `is_init_state`、
`is_ball_moving`、`is_motion_complete`、`has_error` 等執行期控制旗標。

Core 輸出原始 `list[float]`；`float32`、`observation_space`、正規化與裁切由
後續 `BilliardEnv`（#122）負責。既有 `core/models/observation.py` 保留為
TableRuntime／ScriptController 使用的執行期狀態，兩者不得混為同一資料契約。

### Action（6個數字）

| 索引 | `Action` 欄位 | 資料 | 物理域範圍與語意 |
|---:|---|---|---|
| 0 | `cue_ball_placement[0]` | 母球擺位 X | 桌台相對座標；球心安全範圍 `[-0.606425, 0.606425] m` |
| 1 | `cue_ball_placement[1]` | 母球擺位 Y | Kitchen 球心範圍 `[-1.241425, -0.635] m` |
| 2 | `shot_angle` | 擊球方向角 | **Milestone A：`[-30, 30]` 度**（`0°` 朝桌台 `+Y`，正對球堆；正角朝 `-X` 增加）。<br>2026-08-11 兩次修訂（#231）：① 由 `[0, 360)` 改成以 0° 為中心——舊區間的中點是 180°，等於未訓練的 policy 預設把母球往 kitchen 底庫打，且最佳方向 0° 落在正規化域邊界被不連續點切成兩半；② 再收窄到 ±30° 換取探索解析度（命中質量比 2.9% → 17.2%），下限由幾何決定：任何合法 kitchen 擺位瞄準 1 號球最多需要 ±25.524°，加接觸窗口 ±2.062° 共 ±27.586°。**Milestone B 之前必須改回 `[-180, 180)` 並重訓**——走位球要能瞄任意方向 |
| 3 | `cue_ball_speed` | 母球目標初速 | `[0.65, 3.3392] m/s`（下界 2026-08-10 由 0.5 上調，見下方訓練參數說明） |
| 4 | `position_offset[0]` | 上下擊球偏移 | `[-0.5, 0.5]`，單位為球半徑比例 |
| 5 | `position_offset[1]` | 左右擊球偏移 | `[-0.5, 0.5]`，單位為球半徑比例 |
| | | **總計** | **6** |

此處定義的是 RL Policy 使用的物理域向量。`Action.should_execute_action` 是
TableRuntime／ScriptController 的執行期控制旗標，不計入 6 維向量。執行期
no-op 可使用 `cue_ball_speed = 0.0`；RL action space 的正規化、裁切與
`gymnasium.spaces.Box` 由後續 `BilliardEnv`（#122）負責。

### Reward Function

| 條件 | 分數 | 說明 |
|---|---|---|
| 散開程度 | 0.0 ~ +3.4 | **原始分數（0.0~1.0）不直接當 reward**，要先過 `spread_score_to_reward()` 重新正規化（#123，2026-08-11 RunPod 校準）。原始分數＝凸包面積×0.5（進袋球以袋口座標納入，維持 9 點不退化）+ 檯面上球平均最近鄰距離×0.5（進袋球排除），各自正規化；轉換後 rack 擺位＝0.0、RunPod 控制式開球平均（raw 0.0420）＝+1.0、兩輪實測最大（raw 0.1040）＝+3.05，並在 **+3.4** 夾住以保證不蓋過白球進袋的 -3.5。詳見 `core/services/spread_score_calculator.py` |
| 9號球進袋（白球未進） | +3.0 | 有效勝利 |
| 白球進袋 | -3.5 | 含9號球同時進袋情況，9號球不加分 |
| 白球先接觸的不是1號球 | -1.5 | 犯規，該局其餘各項全部歸零（aim 塑形除外）。訓練環境會在首次接觸確定且非 1 號球時**提前終止**（`mdp.break_foul_decided`），不必等球停 |
| 白球整局沒碰到任何球 | **-2.0** | **2026-08-11（#124）由 -1.5 分出來**。兩者原本同分，導致 policy 起點附近整片 reward 是平的——#124 第一輪訓練 238 個 iteration 後 `Policy/mean_std` 0.400 → 0.196、`Episode_Reward/spread` 與 `Episode_Termination/break_foul` 雙雙歸零，也就是收斂到「母球一顆球都沒碰到」。分開後犯規階梯嚴格遞增（-2.0 → -1.5 → -0.5 → 0.0），「碰到東西」本身就是進步。無法提前判定，仍要等落定 |
| 沒球進袋且少於4顆球碰邊框 | -0.5 | 開球力道不足 |
| **瞄準塑形（dense shaping）** | 0.0 ~ +0.4 | **2026-08-11 新增（#124）**。母球在**首次接觸之前**對 1 號球的最近表面間距，線性映射：碰到＝+0.4、距離 ≥ 1.9148 m（kitchen 最遠角落到 1 號球）＝0.0。**是唯一在犯規重置的局也給分的項目**——訓練初期壓倒性多數的 episode 都是犯規，塑形若被 `should_reset` 分支吃掉就等於沒加。滿分 0.4 嚴格小於犯規階梯最小級距 0.5，保證塑形無法反轉排序（`aim_shaping_calculator` 有 import-time 檢查）。詳見 `core/services/aim_shaping_calculator.py` |

### 架構職責分工

| 負責方 | 職責 |
|---|---|
| RL Policy | 看球的位置 → 決定出桿參數（Observation → Action） |
| Isaac Lab IK | 把出桿參數轉換成關節運動 |
| 關節空間軌跡（2026-07-13 起取代 RMPflow 揮桿） | 預先規劃揮桿關節角度曲線（後擺→加速→擊球點），以 joint position/velocity target 播放；RMPflow 僅保留非擊球移動用途（見 #181、#90） |
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
| 母球目標初速範圍 | 0.65 ~ 3.3392 m/s（**下界 2026-08-10 由 0.5 上調（#123）**：訓練端強制純滾動，母球水平減速度只有 μg = 0.0981 m/s²，可滾行程為 v²/(2×0.0981)。從 kitchen 最遠處（Y = -1.241425）滾到 1 號球需走 1.8193 m，至少要 0.5974 m/s——原本的 0.5 m/s 在碰到球堆之前就停了，正規化域低端整段是死區。0.65 讓任何合法擺位都碰得到球堆（最差抵達速度 0.256 m/s）。上限維持 3.3392 m/s；先前不含袋口的 2D 模型高估 spread，2026-08-11 已用 RunPod 真實 PhysX 兩輪控制式開球重新校準 reward（1,016 筆 pooled mean 0.0420），不再用該模型的 0.216~0.248 作為尺度依據。<br>上界來源：2026-07-26 換裝 Barrett WAM + 差動 IK 取代 RMPflow 後實測：預設姿態桿尖峰值速度 2.5302 m/s，套用真實撞球動量傳遞公式 v_ball=v_cue×(1+e)×M桿/(M桿+m球)（球桿 0.5kg、母球 0.163kg、皮革頭恢復係數 e=0.75）換算為母球初速上限 3.3392 m/s，已寫入 `core/models/action_bounds.py` 的 `CUE_BALL_SPEED`（#114 的物理域單一來源；原 `ScriptController.MAX_CUE_BALL_SPEED` 已於 #114 移除，改為引用此常數）。原 UR5 直線推桿版本 1.313 m/s 已淘汰。姿態仍有優化空間未窮舉，離真實開球水準（業餘 ~7.6 m/s、職業 ~8.9–11.2 m/s）仍有明顯差距，詳見對應調查紀錄） |
| 出桿角度範圍 | **Milestone A：−30 ~ +30 度**（#231，見上方 Action 索引表第 2 列；Milestone B 之前必須改回 −180 ~ +180 並重訓，見 #232） |
| 擊球位置偏移範圍 | ±0.5 球半徑 |
| 靜止判定閾值 | 所有球速度 < 0.01 m/s |
| 回合超時上限 | 20 秒（`episode_length_s`，純安全網不是預期長度；實際落定約 10~12 步，一步 = 1 秒） |
| PPO 超參數 | `lr=3.0e-4`、`desired_kl=0.02`、`gamma=lam=1.0`、`init_std=0.4`、`num_envs=1024`、`max_iterations=1000`。單一來源是 `rl_task/.../agents/rsl_rl_ppo_cfg.py`；lr 與 desired_kl 的實測依據見 [issue-124-training-runs.md](issue-124-training-runs.md) |

三輪訓練的完整紀錄（每輪改了什麼、量到什麼、為什麼失敗）：
**[issue-124-training-runs.md](issue-124-training-runs.md)**

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
| 5-1 | 設計擊球參數資料格式（母球初速 0.65–3.3392m/s、角度 Milestone A `[-30, 30]`、位置偏移 ±0.5r）+ Unit Test | 0.5h |
| 5-2 | 設計 `ScriptController` 狀態機（`IDLE` / `AIMING` / `STRIKING` / `WAITING` / `RESET` / `ERROR`） | 0.5h |
| 5-3 | 撰寫狀態機 Unit Test（Mock `ArticulationAPI`） | 1h |
| 5-4 | 實作 `ArticulationAPIImpl`（`isaac_sim_impl_6_0/`） | 0.5h |
| 5-5 | 實作 `AIMING`：RMPflow 將球桿末端移到擊球預備位置 | 1h |
| 5-6 | 實作 `STRIKING`：沿擊球方向加速推進（模擬揮桿衝擊） | 1h |
| 5-7 | 實作 `WAITING`：等待所有球靜止（速度 < 0.001 m/s）+ Unit Test | 0.5h |
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
預估總時數：4.5h

| # | 任務 | 預估 |
|---|---|---|
| 7-1 | 確認 RL Observation 資料格式（20 維：1–9 號球 XY + 母球 XY） | 0.5h |
| 7-1b | 實作 `RigidBodyAPIImpl`（`isaac_sim_impl_6_0/`，查詢球的位置與速度） | 0.5h |
| 7-2b | 實作 `LivePositionProvider`（`isaac_sim_impl_6_0/`，即時查詢球位置） | 0.5h |
| 7-2 | 實作 RL Observation Encoder（既有 `Observation` → 20 維）+ Unit Test | 1h |
| 7-3 | 確認 Action 資料格式（6 維） | 0.5h |
| 7-4 | 補正 `ControllerBase` 完整生命週期契約（`get_action`／`get_current_state`／`reset`） | 0.5h |
| 7-5 | 取消：Observation 收集已由 `ObservationBuilder`／`TableRuntime`（#198–#202）負責，Action 契約由 #110／#111 完成 | 0.5h |
| 7-6 | Debug Menu 新增「印出當前 Observation」按鈕 | 0.5h |

---

## Block 8：參數化控制與中途展示點
預估總時數：3.5h

> **LinkedIn 篇6 發布點**：單台撞球機器人，UR5 持桿擊球，參數化控制

| # | 任務 | 預估 |
|---|---|---|
| 8-1 | 實作 Action 物理域參數設定（母球擺位、初速、角度、擊球偏移） | 0.5h |
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
| Block 7 | Observation 收集與預留 RL 接口 | 4.5h |
| Block 8 | 參數化控制與中途展示點 | 3.5h |
| Block 9 | RL 訓練迴路 | 8h |
| Block 10 | 多環境並行 | 4h |
| Block 11 | 整合測試與補坑 | 4h |
| Block 12 | Demo + README + LinkedIn | 6h |
| **總計** | | **64.5h** |

可用時數約 87h，緩衝空間約 22.5h。

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
