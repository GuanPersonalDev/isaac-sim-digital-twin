# flat（tilt=0）案例殘留誤差調查報告

## 背景

`scripts/scan_elevated_bridge_approach.py` 的 25 點網格掃描（`_CUE_BALL_X_GRID`
× `_CUE_BALL_Y_GRID`）裡，需要抬高球桿（tilt>0）閃避庫邊的案例走差動 IK 的
「高架橋」管線；不需要抬高（tilt=0，即「flat」案例）的則直接沿用已驗證過的
`CANONICAL_REST_JOINTS + base_yaw` joint-space 目標（`_run_flat_case()`）。
本輪調查的起點：其中兩個 flat 案例收斂後的末端位置誤差達 23-28mm（逼近球
半徑 28.575mm 的量級），遠超過 `POSITION_TOLERANCE=5mm`，且 `is_motion_
complete()` 從未在 1000 步內轉為 True。

## 範圍確認

25 個網格點裡，只有 3 個是真正的 flat 案例（`compute_required_tilt_rad()`
判定不需要抬高）：

| 母球位置 | 結果 |
|---|---|
| (-0.25, -0.1) | ❌ 殘留誤差 ~27mm，`is_motion_complete()` 恆為 False |
| (0.0, 0.4) | ✅ 收斂良好（套用 solver iteration 修正後 4.8-5.0mm） |
| (0.25, -0.1) | ❌ 殘留誤差 ~24-25mm，`is_motion_complete()` 恆為 False |

其餘 22 個網格點都走差動 IK 高架橋管線，不受此問題影響。

## 排除清單（皆有直接實驗驗證，非推測）

用 `scripts/probe_first_case_residual_error.py` 對 (-0.25, -0.1) 做隔離
診斷，依序排除：

1. **`is_motion_complete()` 假陽性提早判定** — 額外多跑 500-1000 步觀察，
   關節角位元級不變，是真穩態，不是還在收斂中途被誤判。
2. **PhysX 力矩上限（URDF `effort=10`）飽和** — 實測讀回的 `max_efforts`
   本來就是 float32 上限（未套用 URDF 限制），刻意拉高到 1000 仍無效。
3. **關節限位** — wrist_yaw 卡在偏離目標 +0.096 rad 處，離該關節的限位
   （-4.55 ~ 1.25 rad）還很遠。
4. **自我碰撞／接觸** — 開了完整碰撞回報（含所有 RigidBodyAPI link），
   卡住當下累積 0 個碰撞事件。
5. **DOF 順序錯位** — `articulation.dof_names` 確認跟程式碼假設的
   `[base_yaw, shoulder_pitch, shoulder_yaw, elbow_pitch, wrist_yaw,
   wrist_pitch, palm_yaw]` 順序一致。
6. **PD stiffness 不足（單一參數）** — 讀回 stiffness=100000（USD
   author 值 1745.33 經度數轉換），拉高 50 倍到 5,000,000 完全無感——
   代表卡住的關節根本不是被彈簧公式的穩態誤差卡住。

## 已驗證有效、但只解決部分問題的修正

### 1. PhysX articulation solver iteration count（已套用進正式資產）

`wam7.usda` 的 `payloads/Physics/physics.usda` 裡，`ArticulationRootAPI`
所在的 `world` prim原本完全沒有 author `physxArticulation:
solverPositionIterationCount` / `solverVelocityIterationCount`，走 PhysX
內建低迭代次數預設值。對高 stiffness（100000）、7 自由度＋球桿 FixedJoint
耦合的鏈，這在大幅度 joint-space 跳躍後容易數值上收斂到「穩定但錯誤」的
解——這正是 wrist_yaw 卡在 +0.096 rad 的根因之一。

**已套用修正**：在該 prim 加上 `PhysxArticulationAPI` schema，
`solverPositionIterationCount = solverVelocityIterationCount = 255`
（PhysX 上限）。

**效果**：(-0.25, -0.1) 案例 wrist_yaw 偏差從 0.0999 rad 降到 0.0061 rad
（約 16 倍改善），但同一案例的 shoulder_pitch 偏差變成新的主要誤差來源
（0.0409 rad），整體 Cartesian 誤差仍未收斂（27.83mm）；(0.25, -0.1) 案例
反而出現新的大偏差（wrist_yaw 0.4368 rad）。**結論：這個修正是合理、無副
作用的 PhysX 調校最佳實踐，值得保留，但不是完整解法。**

### 2. 分段 joint-space 逼近（僅驗證於研究腳本，未套進正式程式碼）

假設「單次大跳」本身的瞬態動力學（Coriolis/耦合效應）造成不同案例掉進不同
局部解，改用 6 段線性內插逼近取代單次 `move_to_joint_position()`。

**效果**：混合結果。(0.0, 0.4) 從 4.98mm 大幅改善到 0.08mm；但 (0.25, -0.1)
從 14.30mm **惡化**到 24.33mm；(-0.25, -0.1) 幾乎沒有變化（27.83→27.46mm）。

### 3. 準靜態極慢速逼近（DOF max velocity 降到 0.02 rad/s）

測試結果本身不可信：關節偏差出現 -3.27、5.89、2.81 rad 等遠超過目標的
離譜量級，追查後判斷是測試腳本本身的問題（案例之間只留 50 步去把手臂重置
回展開姿態，時間不夠導致跨案例殘留誤差累積；且極低速度＋高勁度可能讓 PD
控制表現得像 bang-bang 而非平滑逼近）。**這個實驗沒有給出可信的訊號，需要
更嚴謹的重新設計才能用。**

### 4. 多策略嘗試、挑最好的（`scripts/probe_multi_strategy_convergence.py`）

比照本次會話較早解決撞庫邊碰撞問題時用的 roll 候選值搜尋手法：每個 flat
案例同時嘗試「單次大跳」與「分段 6 步」，挑 `is_motion_complete()` 實際
收斂、或誤差較小的一個。

**結果**：3 個 flat 案例中，(0.0, 0.4) 用單次大跳收斂（4.81mm）；
(-0.25, -0.1) 與 (0.25, -0.1) 兩種策略都不收斂（分別停在 27.46mm、
24.36mm）。**多策略挑選對已知的兩種策略沒有找到讓這兩個案例收斂的組合。**

## 尚未解決

(-0.25, -0.1) 與 (0.25, -0.1) 這兩個母球位置，用目前的 `CANONICAL_REST_
JOINTS + base_yaw` joint-space 目標，无論用單次跳躍、分段逼近、還是拉高
solver iteration，都無法讓末端執行器收斂到 5mm 容許誤差內（穩定停在
24-27mm，接近球半徑 28.575mm 的量級）。這兩個案例對應的 `base_yaw` 分別
約 1.243 rad 與另一個對稱角度，具體的動力學/數值成因尚未查明，只確認：

- 不是力矩飽和、關節限位、自我碰撞、DOF 順序、單純 PD stiffness 不足。
- 對 solver iteration count 與軌跡分段這兩種介入手段的反應「因案例而異」
  （改善一個案例的同時可能惡化另一個），顯示問題本質可能是對初始條件/
  路徑敏感的動力學不穩定性，而非可以用單一參數或單一軌跡策略修正的問題。

## 建議後續方向（尚未執行）

1. 用 Isaac Sim GUI 互動式檢視這兩個案例卡住當下的即時物理診斷（關節
   受力、加速度、接觸力等），而不是繼續 headless 腳本盲測參數。
2. 檢查 `CANONICAL_REST_JOINTS` 在這兩個 `base_yaw` 值下，是否恰好接近
   某種質量矩陣病態（ill-conditioned）的手臂構型。
3. 若確認是這兩個特定位置的已知限制，且對擊球精度的影響可接受，可以考慮
   在 `base_placement_calculator.py` 或 `DemoTableOrchestrator` 層級對
   這兩個特定角度範圍改用其他（例如非 CANONICAL_REST_JOINTS 的）目標
   姿態繞開，而不是繼續嘗試修正這組姿態本身的收斂性。

## GUI 檢查指南（需要人工進 Isaac Sim 操作，無法用 headless 腳本自動完成）

### 重現步驟

```bash
ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/repro_flat_case_gui.py
```

`scripts/repro_flat_case_gui.py` 用 `headless=False` 開啟正常 Isaac Sim
視窗，自動建好球桌＋機器人、定位到 `_CUE_BALL = (-0.25, -0.1)`（腳本開頭
可改成 `(0.25, -0.1)` 測另一個失敗案例，或改成 `(0.0, 0.4)` 當正常收斂的
對照組），下達跟 headless 診斷完全相同的 `move_to_joint_position()`
指令後就會卡在偏離目標的姿態，**不會自動關閉**，console 每秒印一次目前
誤差與關節角，方便跟 GUI 上看到的數值對照；視窗不會自己結束，看完直接關
掉即可。

### 檢查項目（依排除清單優先順序，尚未在 headless 驗證過的项目已標記）

1. **Joint 視覺化**（Physics Debug Visualization 內的 Joints）：卡住之後
   看關節標記是不是真的靜止不動，還是在做人眼看不出來但幅度夠小、
   `is_motion_complete()` 卻檢測不到的高頻微幅震盪。
2. **Drive force / drive error 面板**（Window > Physics > Physics
   Inspector 或同等診斷面板）：直接讀 `wam_wrist_yaw_joint` /
   `wam_shoulder_pitch_joint` 當下實際輸出的 drive force 是多少、
   跟理論上 `stiffness × position_error` 算出來的值是否吻合——如果不
   吻合，代表卡住的原因不是我們以為的那個彈簧公式，值得重新假設。
3. **【新假設，尚未驗證】質量比病態**：`wam_wrist_palm_stump_link` 的
   `physics:mass = 0.000001`（幾乎零質量，見
   `assets/barrett_wam/wam7/payloads/Physics/physics.usda`），跟上游
   `wam_upper_arm_link`（mass=2.2）、`wam_forearm_link`（mass=0.5）
   之間質量比極端懸殊。PhysX 對相鄰剛體質量比過大的鏈條，求解精度容易
   在鏈條末端（正好是 wrist_yaw 附近）劣化。GUI 裡若有印出 mass ratio
   相關警告（console 或 Physics Debug 面板），或者可以直接嘗試臨時把
   `wam_wrist_palm_stump_link` 的 mass 調高（例如 0.01）重新測試同一
   案例是否改善，藉此驗證/排除這個假設。
4. **Contact 視覺化**：即使我們自己的 `physics_api` contact reporting
   沒偵測到事件，也用肉眼確認球桿/手臂有沒有跟環境或自己有微幅穿模
   （interpenetration）但因為太輕微沒觸發 contact report 的偵測門檻。
5. 用 `(0.0, 0.4)`（正常收斂案例）跑一次同樣的檢查當對照組，比較上述
   幾項指標在「正常」與「卡住」兩種情況下的差異，縮小根因範圍。

## 相關檔案

- `scripts/probe_first_case_residual_error.py` — 隔離診斷、排除清單的實驗
- `scripts/probe_solver_iterations.py` — solver iteration count 驗證
- `scripts/verify_solver_iteration_fix.py` — 資產修正後的乾淨驗證
- `scripts/probe_staged_joint_motion.py` — 分段逼近驗證
- `scripts/repro_flat_case_gui.py` — GUI 互動診斷用的重現腳本（`headless=False`）
- `scripts/probe_slow_motion.py` — 準靜態極慢速驗證（結果不可信，僅存檔）
- `scripts/probe_multi_strategy_convergence.py` — 多策略挑選驗證
- `assets/barrett_wam/wam7/payloads/Physics/physics.usda` — 已套用的
  solver iteration count 修正
