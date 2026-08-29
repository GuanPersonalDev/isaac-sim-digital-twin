# STRIKE 功能狀態總結（截至 2026-08-29）

本文件是 #181（高架橋擊球）調查的**現況總覽**，給想快速知道「現在卡在
哪裡、之前試過什麼」的人看。完整的逐步調查過程、每個實驗的原始數據、
失敗嘗試的詳細推導都在
[`docs/issue-180-reachability-analysis.md`](issue-180-reachability-analysis.md)
第十三～十六節，本文件只整理結論。

## 總覽

| 問題 | 狀態 | 根因 | 修法 |
|---|---|---|---|
| AIM（瞄準）Kitchen 範圍 0/20 | ✅ 已解決 | roll 查表選錯值，逼關節頂死限位 | 碰撞感知 roll 查表重建 |
| STRIKE 隨揮終點結構性卡死 | ✅ 已解決 | P控制器+feedforward 疊加在靜態目標點的穩態誤差 | 新增 `move_swing()` 速度最優控制器 |
| STRIKE 碰撞衝量為 0 | ✅ 已解決 | AIM／搭橋目標點精確落在母球球心（零間距），慢速 P 控制器收斂時持續把球推走，STRIKE 執行時打的是空氣 | 給 AIM／搭橋最終目標點加 5cm 安全間距（`contact_clearance_m`） |

---

## 一、AIM（瞄準）：已解決 ✅

### 問題

`scripts/verify_swing_trajectory.py` 對 `action_bounds.CUE_BALL_PLACEMENT_X/Y`
完整 Kitchen 網格跑 20 個案例，AIM 階段 **0/20 成功**，全部卡在瞄準逾時。

### 根因

`core/services/cue_pose_calculator.py` 的 `_ROLL_LOOKUP_GRID`（高架橋 C1
轉向階段用來閃避手臂本體撞庫邊/袋口的 9 點查表）選錯值，逼
`shoulder_pitch`／`wrist_pitch`／`palm_yaw` 同時頂死關節硬限位。這個查表
是舊版用物理模擬手動試誤選出來的，只驗證過「無碰撞」，從未驗證過「差動
IK 真的收得斂」。

用新增的 `scripts/wam7_kinematics.py`（純數值 FK/IK，不跑物理模擬，FK
精度已用已知量測常數驗證到 <0.3mm）重新搜尋後發現：正確的 roll 落在
完全不同的範圍（-180°~165°附近），而且**只跟 `cue_ball_y` 有關，跟
`cue_ball_x` 無關**（`wam_base_yaw_joint` 會吸收 X 方向的關節構型差異）。

### 修法

1. `scripts/search_roll_for_full_swing.py`：用「局部延續」數值 IK（不是
   隨機起點，模擬真實差動 IK 不會跳關節分支的行為）掃出每個 Y 值段落
   IK 餘裕最高的候選 roll。
2. `scripts/search_collision_free_roll.py`：依 IK 餘裕排序逐一用真實
   Isaac Sim 物理模擬＋正式的 `enable_contact_reporting` 驗證是否真的
   無碰撞——純數值 IK 沒有建模手臂本體碰撞，純用 IK 餘裕排序的表在
   完整網格上大多數仍是 COLLISION，需要這一步真實驗證。
3. 發現**碰撞跟世界座標系裡離哪個庫邊/袋口近有關，不是只看關節構型**
   ——同一個 Y、不同 X 常常需要不同的 roll 才能避開碰撞，所以最終改成
   對完整 3×3 X×Y 網格逐點驗證，不是只驗證一個代表 X。

### 驗證結果

`_ROLL_LOOKUP_GRID` 已更新（`core/services/cue_pose_calculator.py`），
核心 6 個 (X,Y) 網格點（`y∈{-0.9382125,-0.635}` 各 3 個 X 值）通過真實
Isaac Sim 物理模擬驗證，AIM 6/6 真正收斂（非假陽性，用
`docs/issue-180-reachability-analysis.md` 第十四節記錄的誠實逾時偵測
方法確認）。

### 已知缺口

`shot_angle≠0`／`position_offset≠0` 的案例目前仍沿用最近鄰查到的
`shot_angle=0` 候選，會碰撞——修法路徑明確（同樣方法擴大候選網格搜尋），
只是還沒做。

---

## 二、STRIKE 隨揮終點控制器：已解決 ✅

### 問題

即使 AIM 修好，STRIKE 隨揮終點（`compute_swing_waypoints()` 的第二個
waypoint：靜態位置 + `linear_velocity=required_tip_speed*direction` 的
feedforward）全部逾時，母球速度量到接近 0。

### 根因

`ArticulationAPIImpl._step_motion()` 的控制律是
`twist = P控制器(POSITION_GAIN=5.0 × 位置誤差) + feedforward`。當目標
帶有非零 feedforward 速度時，這個疊加控制律有**結構性穩態誤差**：目標
一旦被通過（feedforward 持續往前推），P 項就會反向煞車，最終在
`|feedforward|/POSITION_GAIN` 這個距離外跟 feedforward 打平，桿尖停在
那裡幾乎不動——這不是位置容許值設太嚴，是這個控制律本身的數學性質。
兩個獨立速度案例的實測穩態誤差都精確吻合這條公式。

### 修法

在 `ArticulationAPIImpl` 新增 `move_swing()`／`_step_swing_motion()`：
先用一般 pose-tracking 收斂到後擺點，再切換成揮桿模式——每個 physics
tick 用線性規劃（`scipy.optimize.linprog`）直接求「姿態修正角速度不超過
`max_angular_speed` 額度的前提下，沿揮桿方向最大化**桿尖**速度」的
關節速度指令，取代會產生結構性穩態誤差的 P控制器+feedforward 路徑。
`core/ports/articulation_api.py` 同步新增抽象方法。

開發過程中修正三個真實 bug：

1. **等式約束又把角速度鎖回 0**：第一版把姿態修正寫成等式約束
   （`Jang@qdot == 目前姿態誤差`），揮桿一開始姿態誤差是 0，這個等式
   因此一直逼近 0，跟原本 STRIKE 卡住的病徵一樣，只是換了個地方重演。
   改成不等式箱型約束才修正。
2. **目標函式用了腕部的線性 Jacobian，不是桿尖的**：桿尖在
   `CUE_STICK_GRIP_TO_TIP`（1.35m）之外，角速度會透過剛體速度合成
   `v_tip = v_wrist + ω × tip_offset` 讓桿尖產生遠比腕部本身位移更大的
   側向偏移。改用 `Jv_tip = Jv - skew(tip_offset) @ Jang`（新增
   `_skew_matrix()`）才是正確的桿尖速度目標函式。
3. **只優化沿揮桿方向速度、沒約束側向漂移**：目標函式只看 1D 投影
   進度，線性規劃可以讓投影進度正常推進、同時桿尖在垂直方向越飄越遠。
   加一個側向位置回正的箱型約束才讓桿尖真正貼著直線走。

### 驗證結果

修完三個 bug 後，對最難的案例（`y=-0.9382125`，24 個 roll 候選裡唯一
在完全鎖死姿態下都無法達標的那組）實測：關節速度全程 4+ rad/s（真的
在高速移動）、桿尖到球最近距離 **12.6mm**（遠小於球半徑 28.575mm，
幾何上確認深入球體範圍）、姿態誤差控制在 <8°（相對「貼近人類擊球姿態」
是合理範圍）。**控制器本身的運動學計算已證實正確**——桿尖真的能又快
又準地衝到球的位置，接下來卡住的是下一節的問題。

---

## 三、STRIKE 碰撞衝量為 0：已解決 ✅

### 問題

`move_swing()` 桿尖確認高速逼近母球到 12.6mm（遠小於球半徑），開啟
正式的 `enable_contact_reporting`／`ContactEvent` 機制後也**確認 PhysX
真的偵測到接觸事件**（`CueStick/Cylinder <-> Ball`）——但這個接觸事件
的 `impulse=0.0`，母球全程沒有獲得任何速度，揮桿的高速通過階段更是
**完全沒有觸發任何新的碰撞事件**（幾何上明明已經重疊）。

### 根因

**不是 PhysX 碰撞求解器的問題，是 AIM／搭橋目標點設計上精確落在母球
球心（零間距），而不是球面**：

- `compute_tilted_wrist_pose()` 算出的 `wrist`（配合
  `CUE_STICK_GRIP_TO_TIP` 反推出的桿尖目標）精確等於 `ball_center`——
  這是 `compute_contact_point()`／`required_grip_position()` 共用的既有
  慣例（`position_offset=[0,0]` 退化為 `ball_center`）。
- **AIM／搭橋是慢速 P 控制器收斂到這個固定目標**，用真實幾何位置
  （不是估算公式，直接讀 CueStick 剛體即時世界姿態＋Cylinder 真實
  半徑/高度重建線段）追蹤發現：AIM 收斂到最後幾步時，桿尖表面間距
  平滑遞減到 0，接著**持續頂在母球表面上把球一路往前推**，母球速度
  被推到 0.3m/s，`is_motion_complete()` 收斂判定完全不知道路徑上有
  一顆球擋著，單純看位置誤差夠不夠小就判定完成。
- 母球被推走之後持續滾動（AIM 收斂 30 步 + 退桿 90 步 + 揮桿本身
  60 步，中間沒有任何重新讀取母球位置的機制），等到 STRIKE 真正執行
  時，母球早已滾到別處（實測偏移 28cm），揮桿桿尖精準命中的是「計算
  出來的目標點」，但那個點球已經不在那裡了——桿尖跟母球即時位置的
  真實表面間距高達 28cm，完全沒有重疊，`CONTACT_FOUND` 不觸發是正確
  行為，不是 PhysX 漏判。
- STRIKE 揮桿本身沿用同一個 `wrist`（=`ball_center`）當方向參考點不
  受影響，因為揮桿是高速通過、真正的物理碰撞遠早於軟體目標點被打到
  就已經發生——只有「慢速收斂、真的會停在目標點上」的 AIM／搭橋會被
  這個零間距目標暴露出問題。

之前排除的 9 個 PhysX 求解器/碰撞參數假說（見下方「已排除的假說」）
全部是在錯誤的層級調查——它們預設「母球位置正確、只是碰撞力沒有正確
傳遞」，但真正的問題是「母球位置已經被 AIM 階段自己推走了」。

### 修法

`core/services/cue_pose_calculator.py` 的
`compute_elevated_bridge_waypoints()` 新增 `contact_clearance_m` 參數
（預設 `0.05`），把 AIM／搭橋收斂的**最終**（C2）目標點沿 `-direction`
方向退開這個距離，讓收斂終點停在母球表面外側、不會頂到球——只改動
這一個函式的最終目標點，`compute_tilted_wrist_pose()` 本身與 STRIKE
揮桿路徑（`_execute_strike()`）完全不受影響。

數值是用 `scripts/diagnose_move_swing.py` 新增的
`AIM_CONTACT_CLEARANCE_M` 覆寫開關實測校準（真實 Isaac Sim 物理模擬，
非解析推算）：

| `contact_clearance_m` | AIM 階段母球殘留速度 | STRIKE 階段結果 |
|---|---|---|
| 0（原本行為） | 0.32 m/s（持續被推） | 桿尖跟母球間距 28cm，完全打空 |
| 0.01m | 0.15 m/s（仍被推，只是變輕） | 未驗證到底 |
| 0.03m | ~0.10 m/s（仍有極小觸碰） | **`impulse=0.201`，母球 `1.06 m/s`** |
| **0.05m（採用值）** | **0.0000 m/s（全程零觸碰事件）** | **`impulse=0.201`，母球 `1.05 m/s`** |

### 驗證結果

用 `scripts/diagnose_move_swing.py`（不帶任何實驗性覆寫，純用新的
`contact_clearance_m=0.05` 預設值）跑最難的 Kitchen 案例
（`(0.0, -0.9382125)`）：

- AIM 全程 `ball_speed=0.0000`，`contact_events_count` 裡完全沒有
  `aim:xxx` 階段的 `CueStick/Cylinder <-> Ball` 事件（之前每次測試都
  一定會有一個 impulse=0 的事件）。
- 揮桿階段 `swing:51`（`is_swing_motion=True`）出現**真實非零衝量
  `impulse=0.20078956438243786`**——整個調查過程第一次在揮桿階段量到
  非零衝量。
- 母球真實速度 `max_ball_speed=1.0545 m/s`（`required_tip_speed=
  1.5116 m/s`——達成率約 70%，速度落差是揮桿控制器可達最大速度的
  另一個獨立問題，見「已知缺口」，不是這次的碰撞衝量問題）。
- `core/tests/` 652 個單元測試全過（`compute_elevated_bridge_waypoints`
  相關測試檢查的是波點結構關係，不是絕對座標，加了間距後仍全數通過）。

### 已知缺口

- **母球實際速度（1.05 m/s）比需求速度（1.51 m/s）低約 30%——已確認
  就是第二節記錄過的運動學可操作性（manipulability）上限，不是這次
  修法引入的新問題**：`DEBUG_MOVE_SWING=1` 印出的線性規劃每步
  `predicted_speed`（桿尖沿揮桿方向的理論最大速度）峰值只有
  0.68 m/s，全程 7 個關節裡有 5 個持續頂在 `_dof_limits` 的
  ±2.0 rad/s 硬限速上——已經是這個關節構型能榨出的極限，換算成球速
  約 0.90 m/s，跟實測 1.05 m/s 同量級。解法方向見
  `docs/issue-180-reachability-analysis.md` 第十六節列出的 4 個選項
  （降 `cue_ball_speed` 上界／進一步放寬姿態漂移／換 margin>0 的
  roll 候選／放寬 `_dof_limits`），都涉及設計取捨，不建議自行拍板。
- `contact_clearance_m=0.05` 只在最難的 Kitchen 案例上實測驗證過，
  尚未跑第一節提到的完整 X×Y 網格回歸測試確認所有案例都不會反而因為
  多退開 5cm 導致新的碰撞或 IK 不可達。
- 只修了「高架橋」（`tilt_rad>0`）分支的 `compute_elevated_bridge_
  waypoints()`；`_execute_aim()` 的 flat 案例（`tilt_rad<=1e-6`，直接用
  `required_grip_position()`+`CANONICAL_REST_JOINTS`）用的是同一個
  「目標點=球心」慣例，理論上有類似風險，但這次沒有動它（flat 案例
  跟 Kitchen 母球擺位範圍幾乎不重疊，見 `docs/issue-flat-case-residual-
  error.md`，優先度較低，且未實測證實有沒有實際發生）。
- 調查過程中發現 `DemoTableOrchestrator._execute_strike()`（正式生產
  路徑）目前呼叫的是舊版 `swing_trajectory_calculator.compute_swing_
  waypoints()` + `move_through_poses()`，**還沒接上這次驗證的
  `move_swing()` 速度最優控制器**——這次的修法對兩條路徑都有效（都
  依賴同一個被推走的母球位置），但 `move_swing()` 本身要正式派上用場
  還需要另外把 `_execute_strike()` 接上它。

---

## 相關檔案

**正式程式碼變更**：
- `core/services/cue_pose_calculator.py`（`_ROLL_LOOKUP_GRID` 重建；
  `compute_elevated_bridge_waypoints()` 新增 `contact_clearance_m`
  參數，預設 `0.05`——**零衝量問題的實際修法**，見上方第三節）
- `extension/isaac_sim_impl_6_0/articulation_api_impl.py`
  （`move_swing()`／`_step_swing_motion()`／`_skew_matrix()`）
- `core/ports/articulation_api.py`（`move_swing()` 抽象方法）

**新增的研究/診斷腳本**（`scripts/`）：
- `wam7_kinematics.py`——WAM7 純數值 FK/IK（不跑物理模擬）
- `search_ik_reachability.py`／`build_roll_lookup_table.py`／
  `search_roll_for_full_swing.py`／`search_roll_swing_capable.py`——
  各階段的 roll 查表搜尋
- `search_collision_free_roll.py`——真實物理模擬碰撞驗證
- `verify_new_roll_table.py`——新查表的端到端驗證
- `search_backswing_ik.py`／`diagnose_strike_followthrough.py`／
  `prototype_moving_target_strike.py`——STRIKE 隨揮終點根因診斷與
  修法原型探索
- `diagnose_ball_impact.py`／`diagnose_move_swing.py`——直接量測母球
  真實物理速度的驗收工具。`diagnose_move_swing.py` 內建：
  - 多組已排除的實驗開關（求解器迭代次數、關節剛性、CCD、
    `excludeFromArticulation`、隨揮距離、
    `enableExternalForcesEveryIteration`、關節 drive `damping`／
    `maxForce`、碰撞 `contactOffset` 等，見上方「已排除的假說」）
  - `AIM_CONTACT_CLEARANCE_M`——**校準出實際修法數值的開關**
  - 用真實幾何（讀 Cylinder/Ball 真實半徑＋即時世界姿態重建線段，
    不是估算公式）算桿身表面到母球的真實間距（`_real_surface_gap()`）
    ——找到根因的關鍵工具
  - AIM 收斂全程的母球位置/速度追蹤——抓到「AIM 階段就已經把球推走」
    這個時序的關鍵證據
- `minimal_repro_cue_impact.py`——自由剛體撞球的最小重現案例（證實
  碰撞物理本身正常，回頭看是排除了錯誤方向的一個假說，不是根因）

**完整調查過程**：
[`docs/issue-180-reachability-analysis.md`](issue-180-reachability-analysis.md)
第十三～十六節
