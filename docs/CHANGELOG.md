# 除錯與設計歷程 CHANGELOG

**狀態**：持續累積
**用途**：集中收納原本散落在程式碼註解裡的除錯過程、日期、根因追蹤細節。程式碼內只保留「來源＋用意」，完整的推導過程、試過但失敗的方案、量測數據都記在這裡。各 `docs/issue-*.md` 已有完整分析的項目不重複貼，只在此留一行指標。

條目格式：`### <檔案路徑> — <主題>`，內文含日期（若已知）與推導脈絡。

---

## extension/isaac_sim_impl_6_0/articulation_api_impl.py

### `_boost_wrist_gains_for_cue_stick_load()` — wrist_1/wrist_3 增益補強（2026-09-02）

真實 GUI 執行（`billiard_digital_twin.py` 換成 `UR3eRobot` 之後）逐 tick log 顯示：`move_swing_elbow_pivot()` 進入「joint-space 移動到後擺姿態」子動作時，肘關節正常收斂，但 `wrist_1`／`wrist_3` 兩個關節即使目標角度完全沒變，也會在肘關節做大幅度動態擺動的過程中被拖離目標（wrist_1 從 -0.6999 漂移到 -0.433、wrist_3 從 ~0 漂移到 0.072，之後卡住不動），導致這個子動作永遠收斂不了、1000 步後逾時，STRIKE 從此卡死在錯誤姿態、桿尖離母球 1.87m。

根因：UR3e 官方 USD 的關節 PD 增益／扭矩上限是針對「手臂自身負載」調的，沒考慮到球桿透過 FixedJoint 剛性掛在腕部後，1.35m 力臂在肘關節動態擺動時對 wrist_1/wrist_3 產生的額外反作用力矩。修法：提高這兩個關節的 stiffness/damping/max_effort（沿用官方 Gain Tuner「SET STIFF GAINS」慣例值）。

### `_UR10E_AIM_RETRACT_POSITION_M` — 退桿距離加大實驗（2026-09-05，已回退）

查證發現真正蹭到母球的是桿身（不是桿尖），試過把退桿距離從 `DEFAULT_BACKSWING_DISTANCE_M`（0.15m）加大到 `CUE_STICK_GRIP_TO_TIP+安全邊際`（1.45m），指望讓整根 1.35m 長的桿身徹底撤出母球所在的軸向區間。**實測證實這會讓情況更糟**：手臂從 RESET(HOME) 移動到 AIM 目標的過程中，退更遠的桿子拖著更長的尾巴一起掃過更大空間，反而撞到地板（impulse=2.34）、撞到多顆原本沒事的球——比原本輕微蹭到一顆球（impulse=0.024）嚴重得多。「退桿完成後靜止時的安全距離」跟「手臂移動過程中桿子掃過的空間大小」是互相拉扯的兩個需求，不能只靠加大退桿距離單方面解決。已回退為沿用 `DEFAULT_BACKSWING_DISTANCE_M`，接受桿身蹭到球的殘留風險；真正的修法應該是讓 RMPflow 路徑規劃本身知道母球/球檯的存在。

### `_UR10E_AIM_STAGING_OFFSET_M` — 分階段避障的推導過程（2026-09-05，三輪）

**第一輪**：註冊母球/球檯為 RMPflow 障礙物後，直接接觸問題解決（CueStick-母球 impulse 從 0.024 降到 0），但單一長距離、大幅重新定向的 waypoint chain（從 HOME 直接規劃到貼近母球的最終 AIM 姿態）反而更容易在避障的複雜決策空間裡卡進錯誤姿態分支（實測：方向誤差從 0.02rad 惡化到 0.86rad 等級，wxyz 符號幾乎完全相反）。改成兩階段：先移到方向與最終目標相同、沿桿軸往後退開 offset 的「安全中繼姿態」（避障全程啟用），再從中繼姿態沿同一軸向直線平移到最終位置。

**第二輪**：最終逼近（第 2 步）的終點本來就緊貼母球，母球避障全程停用，代表整段（~1.45m）都沒有防護——實測踩過：桿身在抵達終點前的倒數第二個 waypoint（桿尖離球心僅 0.054m）擦到母球，撞出殘留速度，命中率變成 0%。

**第三輪**：嘗試縮短這個距離降低無防護暴露長度，改成 0.3m 跟 0.6m 都讓 AIM 整個收斂變差（0.3m：位置誤差 0.306m，比整段 FINAL_APPROACH 的移動距離本身還長；0.6m：結果幾乎跟 0.3m 一樣，對這個參數不敏感）——研判 STAGING 階段本身的避障規劃在終點離母球太近時會被過度擠壓，不是單純調小數字能解決。退回已驗證能完全收斂的原始值（1.45m），中途擦碰問題改用「分段停用避障」（見下一條）處理。

### 分段停用避障（第四輪，`_UR10E_FINAL_APPROACH_SAFE_MARGIN_M`，2026-09-05）

FINAL_APPROACH 拆成兩段：先在避障開著的情況下逼近到只剩緩衝距離的「逼近緩衝點」，最後才關避障直線逼近剩下一小段，大幅縮小無防護暴露長度（從 ~1.45m 縮到 0.2m）。結果：擦碰沒有完全消失，只是位置往後移到更短的無防護段末端，且撞擊力道變大（impulse 0.042，先前是 0.0）。結論：這是終點姿態本身跟母球物理間距過緊的幾何問題，不是路徑規劃問題，分段避障無法根治，需要調整退桿距離或接觸點計算。

### Default gains 快取污染 bug（差動 IK 收尾方向誤差卡在 0.073rad，2026-09-05）

`Articulation.set_dof_gains()` 的 `update_default_gains` 參數預設 `True`：joint-space 收尾（`_boost_wrist_gains_for_cue_stick_load()` 的增益 boost）呼叫時若沒明確關閉，會永久污染 `switch_dof_control_mode()` 內部快取的「預設增益」，導致差動 IK 收尾切到 velocity 模式時，wrist_1/wrist_2/wrist_3/elbow 的阻尼被錯誤帶入 boost 過的數值，跟同鏈其他關節量級差異懸殊，重演同一類耦合病態。修法：呼叫時加 `update_default_gains=False`，並在建構時提前呼叫一次 `switch_dof_control_mode()` 用原始烘焙值把快取填好，關閉呼叫順序競態。修正後 AIM 位置/方向誤差穩定收斂到 0.0008m / 0.00225rad（此前最好僅達 0.665m / 1.3rad）。

### `is_motion_complete()` 提早判定完成 bug（2026-08-31，Demo 桌真實 GUI 才踩到）

`_on_tick`（驅動狀態機）跟 `_step_motion`（驅動實際換下一個子動作）是兩個各自獨立註冊的 PHYSICS_POST_STEP callback，`_on_tick` 註冊得早，每個 physics step 會搶先執行。當某個「中繼子動作」（`move_through_poses()` 的 Phase 0 joint-space 安全姿態、或 `move_swing()` 的後擺子階段）剛好在這個 physics step 收斂時，`_on_tick` 會搶先讀到「目前子目標已收斂」，讓外部誤以為**整個**動作做完、狀態機直接跳下一個狀態——但 `_step_motion()` 根本還沒機會把動作換到後面真正的目標，手臂因此永遠卡在中繼姿態（實測：球桿跟母球呈現不合理角度，STRIKING 卻已經在執行）。曾經誤把這個守門邏輯加進 `_is_current_target_converged()` 本體，結果連 `_step_motion()` 自己判斷「這個 waypoint 到了、該換下一個」都被擋住，整條 waypoint 序列永遠卡在第一個——改成只在 `is_motion_complete()` 這一層額外把關，不動 `_step_motion()` 依賴的內部判定。

### `_capture_home_position_once()` 擷取時機 bug

`_default_joint_positions` 與 `_home_position` 原本不同步擷取；`get_dof_positions()` 若在 `initialize()` 裡同步呼叫（physics 可能一步都還沒跑）會拿到不可靠的值（跟 `scripts/probe_palm_yaw_correction.py` 除錯時踩到「剛建構的 Articulation 沒等 physics 穩定就讀，拿到垃圾值」同一類問題），導致 `move_to_home()` 把關節開回一個不可靠的 `_default_joint_positions`，永遠碰不到用正確方式量到的 `_home_position`，是 RESET 狀態卡死、`is_motion_complete()` 恆為 False 的根因。修法：兩者搬到同一個 PHYSICS_POST_STEP callback 一起擷取。

### `_is_current_target_converged()` — 曾嘗試放寬 waypoint 容許值（2026-08-28，已還原）

曾經加過「帶 feedforward 速度的 waypoint 放寬容許值」修正（見 docs/issue-180-reachability-analysis.md 第十五節的穩態誤差公式），但用 `scripts/verify_swing_trajectory.py` 的真實桿尖速度量測驗證後發現：那個放寬容許值的門檻剛好落在 P 控制器+feedforward 的穩態平衡點（合力趨近 0，關節速度也趨近 0），系統會在那裡「宣告完成」，但桿尖當下幾乎靜止（實測 speed_error_ratio≈0.98，實際速度只有該有速度的 ~2%），等於沒真正揮桿。已還原，根因修法改成 `move_swing()`（線性規劃直接下令沿揮桿方向速度最優的關節速度指令，不是放寬既有 pose-tracking 的完成判定，見文件第十六節）。

### `register_static_box_obstacle()` — FixedCuboid → VisualCuboid（2026-09-05）

一開始用 `FixedCuboid`——那是「靜態剛體＋真實 PhysX 碰撞」，不是純粹給 RMPflow 內部避障邏輯參考用的幾何標記。實測踩過：這個障礙物箱體跟真正的球檯位置重疊，變成真的會撞的東西，CueStick 反覆跟它產生接觸事件。改用 `VisualCuboid`（純幾何+世界座標，沒有 RigidBodyAPI/CollisionAPI），不會參與真實 PhysX 碰撞反應。

### `register_dynamic_sphere_obstacle()` — 無法直接包既有母球 prim（2026-09-05）

一開始直接 `DynamicSphere(prim_path=母球路徑)` 想包一層現有的母球 prim，實測噴例外「cannot be parsed as a Sphere object」。母球實際的 USD 結構不是頂層就是一個 `UsdGeom.Sphere`（真正的 Sphere geometry 在更深的子節點），`DynamicSphere` 建構子對「包既有 prim」這個用法會嚴格檢查 prim type，包不了。改成建立一個全新、獨立的 `DynamicSphere` 障礙物 proxy，每個 tick 從母球真正的 `RigidPrim` 讀取最新世界座標手動同步過去。

### `move_swing_elbow_pivot()` 設計動機

`move_swing()` 對 WAM7 有效是因為 WAM7 需要多個關節協調（線性規劃跨全部關節求解）才能達到目標桿尖速度。UR3e 已驗證不需要這樣：只讓 `elbow_dof_index` 這一個關節從 0 加速到目標速度，其餘關節角速度指令精確為 0，就足以達到目標桿尖速度（見 `scripts/test_ur3e_human_pose_swing_speed.py`／`scripts/test_elevated_bridge_ur3e_table.py` 的真實 quintic 軌跡執行驗證，分別達成 104.7%／96.1%），而且完全靜止的 base/肩關節更貼近人體揮桿手肘擺動的動作設計。

### `_apply_velocity_targets_with_gravity_compensation()` — UR3e 重力漂移驗證

專案目前用的 WAM7 沒踩到重力漂移問題，但不是因為刻意處理過：URDF→USD 轉換工具幫每個關節寫死了一組偏高的 damping（`drive:angular:physics:damping=174.53`），意外地夠抗重力，不是專案自己調過的值。這是結構性風險：`scripts/test_ur3e_human_pose_swing_speed.py` 在 isolated 測試場景真的踩到過，達成率一度只有理論值的 10~55%——整支手臂在還沒開始揮桿前就先自由落體，量到的低速度是重力漂移的假象，不是姿態設計本身的問題。

### `get_end_effector_position()` — UR10e 誤用 tip_local_offset bug（2026-09-04）

UR10e 模式下這裡曾經誤用 `_compute_tip_local_offset()`（讀 end effector link 自身的 bounding box，找「離原點最遠那一端」當工具尖端）——那是幫 WAM7/UR3e 找「球桿用 align_prim_to_target 掛接的實體參考點」設計的，UR10e 的球桿是透過 CueSlideJoint 掛在 wrist_3_link 之後，跟 wrist_3_link 自己的 flange 幾何體完全無關。套用這個偏移量會加上一個跟 AIM 目標無關的常數位移，讓 AIM 收斂診斷憑空多出約 5cm 的「假誤差」（實測：joint tracking gap 僅約 6e-5 rad，代表關節本身幾乎完美收斂到 RMPflow 目標，但套用偏移量量出來的末端位置卻跟目標差了 5.6cm）。

---

## extension/isaac_sim_impl_6_0/ur10e_rmpflow_controller.py

### 類別設計動機（2026-09-03～09-04）

RMPflow 對「一次給一個很大的末端目標位移」（約 30cm 量級的對角線跳躍）會卡在局部穩定點，殘留誤差可達 0.1m 以上、長時間不再收斂——這是 reactive RMP controller 的已知特性（多個 RMP 分量互相拉扯），不是 bug。`move_to_pose()` 因此把大位移目標拆成一串位移量不超過 `_MAX_WAYPOINT_STEP_M` 的中繼 waypoint。

`scripts/test_ur10e_table_flat.py` 的診斷發現，flat 案例走完整段 waypoint 後仍殘留 5.6cm 誤差，但 RMPflow 算出的關節目標跟 PhysX 實際量到的關節位置幾乎完全吻合（tracking gap 僅約 6e-5 rad）——代表不是 joint drive 追不上目標，而是 RMPflow 這個 reactive controller 本身在這個姿態附近的計算殘留（NVIDIA 官方論壇也有相同回報：forums.developer.nvidia.com/t/imprecise-control-via-rmpflow/253139）。修法：在最後一個 waypoint 收斂（或逾時）後，若仍未進入容許誤差，改用差動 IK 再收尾（見 `_step_finish_ik()`）——此時手臂已經很接近目標姿態，跳過 RMPflow 的避障不是新風險。

### `move_to_pose()` — 方向也需逐段內插（2026-09-03）

只內插位置、方向從第一段就直接設成最終目標，對「位置移動量大＋方向本身需要旋轉」（例如高架橋案例的傾斜姿態）的真實 AIM 目標會卡住不收斂——研判是 RMPflow 被迫在離最終位置還很遠的中繼點就同時追蹤最終方向，跟位置追蹤互相拉扯出局部穩定點。方向也逐段內插後，兩者不再互相打架。

### `_FINAL_ORIENTATION_TOLERANCE_RAD` — 收緊容許值的排查（2026-09-04）

一開始把 `_ORIENTATION_TOLERANCE_RAD` 整體從 0.02 收緊到 0.005，結果 AIM 直接崩潰（3379 步逾時，位置誤差 0.408m、方向誤差 0.86rad）——因為這個常數同時被 waypoint chain 每一段中繼點的收斂判定沿用，中繼點容許值收太緊會讓每一段都逼近 240 步逾時上限，累積誤差整段路徑跑歪。改成只在「最終姿態」精度收緊：實測 AIM 方向誤差 0.00933rad（在原本 0.02 容許值內，判定「收斂成功」）換算桿尖偏移約 1.26cm，跟 STRIKE 實測 miss 向量的橫向分量（約 1.2cm）吻合——這個偏移小於「球桿半徑+母球半徑」，導致 STRIKE 階段球桿的圓柱形桿身（不是桿尖）貼著母球側面蹭過去，衝量沒有正面轉移，達成率只有 42%。

### `_HOME_JOINT_POSITIONS` — wrist_2 奇異點排查（2026-09-03，未證實）

`default_q` 的 `wrist_2_joint=0`，懷疑落在 UR 家族手臂的手腕奇異點（wrist_2=0 時 wrist_1／wrist_3 兩軸平行/耦合）附近，可能是某些 AIM 目標（尤其 flat 案例）從 HOME 出發會卡在局部穩定點的原因之一。實測把 wrist_2 改成 π/2（遠離這個值）之後，HOME 本身跟後續 AIM 反而都變得更難收斂（HOME 自己開始逾時、AIM 殘留誤差從 0.16m 惡化到 0.20m），已改回原始 `default_q`——這個假設沒有被證實，問題根因仍待查。

### `set_solver_iteration_counts()` — 已移除（2026-09-06）

當初拉高 iteration count 是為了壓 wrist_2 那個「對增益免疫」的殘留誤差，但那個症狀只在球桿被推進母球裡的那些回合出現過——研判是 CueSlideJoint body0/body1 寫反造成的，不是求解器收斂問題。方向修好之後回頭實測（flat 案例，同一組程式碼只差這一行）：

| | 128 iterations | 預設值 |
|---|---|---|
| RESET | 2402 步 | **902 步**（−62%） |
| AIM | 2100 步 | 2079 步（持平） |
| AIM 位置誤差 | 0.00083 m | **0.00056 m** |
| 達成率 | 93.5% | 93.3% |
| 球桿-母球碰撞 | 1 次 | 1 次 |
| 手臂本體碰撞 | 0 筆 | 0 筆 |

精度沒有退（反而略好），RESET 快了 2.7 倍，因此移除。下面那段門檻效應的數據是修正方向 bug 之前量的，保留當歷史記錄，不再適用。

### `set_solver_iteration_counts()` 門檻效應實測數據（2026-09-05，已不適用）

拉高 iteration count 有明顯的門檻效應，不是線性漸進：255（WAM7 探針腳本驗證假設用的極端值）能把殘留誤差壓到 0.002rad，但代價是每個 waypoint 收斂明顯變慢（RESET 905→2407 步、STAGING 1999→4164 步仍逾時）；降到 32 幾乎沒有效果（殘留誤差 0.0902，等於沒調）。128 跟 255 效果相同（殘留誤差同樣壓到 0.002~0.005rad），代價也相同——門檻落在 32~128 之間，且修正效果與變慢代價綁在一起，無法只取其一。選 128（不用 255）是「先取一個明確跨過門檻、且非必要不用極端值」的保守選擇。

### `_compute_analytic_finish_joint_target()` — 奇異點解集合退化（2026-09-04）

收緊 `_FINAL_ORIENTATION_TOLERANCE_RAD` 後才踩到的新問題——`move_to_home()` 走的也是同一條收尾路徑，而 HOME 姿態的 `wrist_2_joint=0` 正好卡在 UR 家族手臂的手腕奇異點上。在奇異點附近，closed-form IK 的解集合會退化（實測：只解出 4 組，不是滿額的 8 組），這時候「挑離目前姿態最近的分支」完全不可信——實測踩過：目前關節角幾乎正好在 HOME，選出來的「最近」分支卻離目前姿態達 2.7rad，把手臂拖去完全錯誤的姿態。加一個寬鬆的合理性上限（`_MAX_REASONABLE_FINISH_DELTA_RAD=0.5`），超過就視為不可信、退回差動 IK。

### `_FINISH_GAIN_OVERRIDES` 排查歷程（2026-09-04～09-05）

一開始照抄 `ArticulationAPIImpl._boost_wrist_gains_for_cue_stick_load()`（UR3e 驗證過的同一組常數，stiffness=1e15/damping=1e5），結果 `wrist_1_joint` 在 joint-space 收尾完全卡住不動（240 步、1120N·m 飽和力矩幾乎無效）。逐一排除碰撞（全連桿 contact reporting 確認零接觸）、關節極限（差六圈以上）、drive type（確認是 "force"）、gains 寫入沒生效之後，靠 A/B 對照測試發現：`wrist_1_joint` 原始 baked stiffness 只有約 72,662，1e15 是這個值的一百多億倍，跟同一條運動鏈上其他關節的量級差距過大，讓 PhysX 的 TGS 迭代求解器對這個最僵硬的關節反而欠收斂——數值上病態，不是真的「增益不夠」。改成 1e6/1e4（約為原始值的 14 倍，遠比 UR3e 那組溫和）之後，實測 2 個 physics tick 就收斂（joint_error 從 0.033rad 降到 0.0096rad）。UR3e 跟 UR10e 兩邊的下游負載結構不同，同一組「越硬越好」的增益常數不能直接照搬。

收緊 `_FINAL_ORIENTATION_TOLERANCE_RAD` 到 0.005 之後才浮現：逐關節 log 證實應該針對個別關節依其負載分別調整增益（Isaac Sim 官方 Gain Tuner 文件、PhysX 官方文件一致建議），逾時當下 `elbow_joint` 誤差 0.02205rad（遠高於其他關節），`wrist_1_joint` 0.00504rad，其餘 4 個關節都遠低於容許值——不是「全部關節都不夠力」，是原本沒被列入 boost 名單、扛著整條下游手臂重力力矩的 `elbow_joint` 不夠力。把 "elbow" 加進 boost 名單解決。

加入 RMPflow 障礙物避讓（AIM 兩階段分期）後才浮現：逐關節 log 顯示 `joint_space_finish` 逾時當下 `wrist_2_joint` 誤差 0.089~0.138rad（其餘 5 個關節都在 0.01rad 以內）。一開始誤判成跟 elbow 同一類「增益不夠」問題，把 stiffness/damping 加倍成 2e6/2e4、`max_effort_multiplier` 從 20x 加到 40x（2240N·m），殘留誤差都幾乎沒變（三次都落在 0.0887~0.0890rad）——不是 PD 穩態誤差（加倍 stiffness 應該讓誤差大致減半），也不是 max_effort 飽和（加倍上限應該有反應）。逐 tick log 顯示這個關節一開始其實已經很接近目標（誤差 0.0065rad），是之後 240 步內平滑地被拉往另一個平衡點，典型「多關節耦合系統 solver 迭代次數不足」的假影特徵——真正的修法是 `set_solver_iteration_counts()`（見上一條），不是 gain override，wrist_2 維持跟 wrist_1/wrist_3 一致的數值。

### `_step_joint_space_finish()` — 缺重力補償的排查（2026-09-04）

第一版沒加重力補償，逐 tick log 顯示 `wrist_1_joint` 穩定卡在離目標 0.033rad 的地方完全不動（其餘 5 個關節都準確追到 1e-4rad 等級）——這正是 `ArticulationAPIImpl._boost_wrist_gains_for_cue_stick_load()` 修過的同一類問題。UR10e 的 `_step_rmpflow()` 之所以沒踩到，是因為 RMPflow 每個 tick 都重新給一個貼近目前值的新目標，等於變相用位置追蹤模擬速度追蹤，掩蓋了這個穩態誤差；joint-space 收尾是固定目標長時間 hold，穩態下垂才會顯現。

### `_step_finish_ik()` — 缺速度歸零的排查（2026-09-04）

第一版在收斂/逾時時直接 return，沒有歸零 velocity target——velocity-mode drive 會持續套用「上一次」下達的非零角速度指令，直到有新指令覆寫為止。實測：STRIKE 開始前母球速度就已經非零，代表收斂後手臂漂移的桿子先撞到了球。

### `add_dynamic_sphere_obstacle()` — DynamicSphere 互撞問題（2026-09-05）

`DynamicSphere` 本身是「動態剛體＋真實 PhysX 碰撞」，不是純幾何標記——實測踩過：這個 proxy 跟真正的母球位置完全重疊（故意同步成一樣），兩個都有真實碰撞形狀，直接互撞，母球被撞出 impulse 31 等級的力道，整顆彈飛去撞牆撞地板。改用 `VisualSphere`。

---

## core/models/action_bounds.py

### `SHOT_ANGLE` — 為什麼是 (-180, 180) 而不是 (0, 360)（#231，2026-08-11）

覆蓋範圍完全相同，差別只在端點擺在哪裡：

| | 舊 (0, 360) | 新 (-180, 180) |
|---|---|---|
| normalized -1 | 0°（正對球堆） | -180°（背對球堆） |
| normalized 0 | 180°（背對球堆） | 0°（正對球堆） |
| normalized +1 | 360°（正對球堆） | 180°（背對球堆） |

1. Gaussian policy 的初始輸出集中在 normalized 0。舊區間等於預設瞄準球堆的反方向——#123 短訓練的 valid_ratio ≈ 0.175（平均 episode 5.7 步，遠短於落定的 10~12 步）就是「母球沒撞到球堆、彈幾次庫就停」的長度。
2. 舊區間把開球的最佳方向 0° 放在 -1 邊界上：policy 要瞄它就得把平均值推到邊界、分佈有一半被 clip；而物理上等價的 359° 與 1° 在正規化域是 +0.994 與 -0.994，相距 1.99——幾乎整個動作空間的寬度。Gaussian 表達不了這種週期性。
3. `rl_task/scripts/verify_spread_ref.py` 原本必須從 -1 與 +1 兩端各取一半樣本才能涵蓋 0° 附近，那個 workaround 本身就是端點放錯位置的證據。

### `SHOT_ANGLE` — Milestone A 收窄為 ±30° 的 PPO 實測數據（2026-08-11）

```
Mean value loss   1.1871 → 0.7214 → 0.2653 → 0.0534 → 0.0105
Mean surrogate    0.0037 → -0.0003 → -0.0001 → -0.0006 → -0.0010
Mean action std   0.40 → 0.40 → 0.40 → 0.40 → 0.40
```

critic 五個 iteration 就把 value loss 壓到 0.01——因為答案永遠是 -1.5。value 準了之後 advantage ≈ 0，surrogate 掉到 ±0.001，policy 完全不動。反推信號密度：Episode_Reward/foul = -0.0743 × 20 = -1.486，代表只有約 0.9% 的 episode 拿到 foul = 0；每輪約 180 局裡只有 2 局帶資訊。

根因是解析度：母球到 1 號球 1.5875 m，接觸只容許側向 2R = 0.05715 m，換算角度窗口僅 ±2.062°（擺位偏 3 cm 時只剩 ±0.980°）。init_std = 0.4 在 ±180° 的區間上是 ±72° 的探索半寬——命中質量比只有 2.9%。

| 半寬 | 命中窗口（正規化） | init_std=0.4 命中質量比 |
|---|---|---|
| ±180° | ±0.0115 | 2.9% |
| ±45° | ±0.0458 | 11.5% |
| ±30° | ±0.0687 | 17.2%（採用） |

### `CUE_BALL_SPEED` — 上界飽和假設已被推翻（2026-08-11）

原本寫「spread 要到約 1.8 m/s 才飽和」，那個數字來自一個被 RunPod 實測推翻的 2D 模型，已證實錯誤。真實 PhysX 的速度掃描（各 500+ 筆 first_contact == 1）顯示完全沒有飽和：

```
1.3223 m/s  spread 0.01349   legal break  0.0%
1.9946 m/s  spread 0.01798   legal break  0.0%
2.6669 m/s  spread 0.03451   legal break 28.7%
3.3392 m/s  spread 0.04264   legal break 44.8%
```

扣掉 rack 基準後的彈性 d ln(spread-rack)/d ln(v) 在上界處仍有 1.36（飽和的話會趨近 0），也就是上界 3.3392 本身才是瓶頸。連帶的事實：低於約 2.6 m/s 的整段是策略上的嚴格劣勢——spread 更差，而且 legal break 掛零，保證吃 -0.5，policy 會學成永遠輸出速度上界，這一維不會有有意義的策略。

---

## core/services/cue_pose_calculator.py

### `_ROLL_LOOKUP_GRID` 三輪修正歷程（2026-08-28，見 docs/issue-180-reachability-analysis.md 第十四節）

**第一輪（全面重建）**：舊表（0°/15°/45°/60° 這種小角度）是用物理模擬手動試誤選出來的、只確認「無碰撞」，從未真正驗證「AIM 差動 IK 收得斂」——20 案例 STRIKE 0/20 全滅的根因追到最後，就是這個查表逼 shoulder_pitch／wrist_pitch／palm_yaw 同時頂死關節限位。用 `scripts/wam7_kinematics.py` 的純數值 IK 重新搜尋，發現正確的 roll 落在完全不同的範圍（-180°~165°），且 roll 只跟 cue_ball_y 有關、跟 cue_ball_x 無關（base_yaw 關節會吸收 X 方向的差異）。

**第二輪**：純數值 IK 沒有建模手臂本體碰撞，只用 IK 餘裕排序的表在完整 20 案例網格上大多數是 COLLISION。改用 `scripts/search_collision_free_roll.py`：對每個候選點依 IK 餘裕由高到低嘗試候選，逐一用真實 Isaac Sim 物理模擬驗證，取第一個「IK 收斂 + 無碰撞」都成立的候選。

**第三輪**：「roll 只跟 cue_ball_y 有關」只在數值 IK 可達性這個面向成立——碰撞跟世界座標系裡離哪個庫邊/袋口近有關，不是只看關節構型，同一個 Y、不同 X 的三個案例常常需要不同的 roll。改成對 `action_bounds.CUE_BALL_PLACEMENT_X/Y` 的完整 3×3 網格逐點驗證，不再假設 X 無關。

### `_BACKSWING_DISTANCE_LOOKUP_GRID` — 基座偏移搜尋失敗記錄（2026-08-31～09-01）

`DEFAULT_BACKSWING_DISTANCE_M`（0.15m）跟關節實際能提供的加速能力完全脫鉤，實測揮桿速度只達目標 55%（見 docs/issue-180-reachability-analysis.md 第十八節）。曾經試過把機器人基座水平位移也當自由變數一起搜尋（能讓部分案例的後擺距離大幅提升），但用真實 Isaac Sim headless（`diagnose_move_swing.py`）驗證發現：純運動學可達性分析找到的基座偏移，會讓 `ArticulationAPIImpl` 真正用的差動 IK 控制迴圈（Phase 0→B1→B2→C1→C2）不收斂（逾時 1000 步，揮桿打空）——一次性可達性求解沒有模擬差動 IK 沿路徑逐步收斂的動態行為，偏移越大、路徑幾何改變越多，風險越高。改成基座位置一律用 `compute_base_pose()` 的公式值，完全不搜尋偏移，只保留後擺距離的改善。

### `compute_elevated_bridge_waypoints()` — C1/B1 兩輪修正的實測數據（2026-08-27，見 docs/issue-180-reachability-analysis.md 第十三節）

**C1 二次修正**：原本是單一個大跳躍 waypoint，實測發現即使腕部位置全程沒動，差動 IK 為了在單一 waypoint 內達成姿態變化，會讓 `shoulder_yaw`/`elbow_pitch` 沿路劇烈擺盪，導致手臂本體掃過球檯庫邊/袋口，在 Kitchen 正中心案例撞到 `Cushion_Head`／`Pocket_HeadLeft`。改用 NLERP 拆成多個中繼姿態解決。

**B1 一次修正**：舊版第一階段是「原地轉向朝正上方」，目的是保證轉向過程中桿頭不會掃低撞到桌面。但這個「轉到正上方」是接近 90° 的大幅重新定向，會把 `wrist_yaw`（總行程只有 5.8 rad，起點在 0）／`wrist_pitch`（總行程只有 π rad≈180°，起點在 -32°）逼到硬限位卡死收斂不了，且跟 `shoulder_pitch`/`elbow_pitch` 的固定姿態餘裕無關——不管怎麼調 `CANONICAL_REST_JOINTS` 都救不了（`shoulder_pitch` 從 1.9 降到 1.5 對這個瓶頸完全沒有幫助，殘留誤差幾乎不變）。改用「保持目前姿態原地爬升」解決。

### `backswing_distance_m` 命名沿革與舊校準記錄（2026-08-29～09-01，見 docs/issue-180-reachability-analysis.md 第十七、十八節）

這個參數原本叫 `contact_clearance_m`（預設 0.05m），跟 STRIKE 後擺起點用的 `swing_trajectory_calculator.DEFAULT_BACKSWING_DISTANCE_M`（0.15m）是兩個獨立數字，中間那段差距原本由裸的差動 IK P 控制器走，沒有防撞驗證。2026-09-01 統一成同一個值並改名、移除預設值。

舊 `contact_clearance_m=0.05` 的校準記錄（`scripts/diagnose_move_swing.py` 的 `AIM_CONTACT_CLEARANCE_M` 覆寫開關實測，真實 Isaac Sim 物理模擬，僅存歷史價值）：0.01m 仍會被 P 控制器的收斂爬升「追上」（母球殘留速度從無間距的 0.32m/s 降到 0.15m/s，但沒有歸零）；0.03m AIM 階段仍有一次極小的觸碰（母球殘留 ~0.1m/s），但已經足以讓 STRIKE 揮桿階段量到真實非零衝量（impulse=0.201、母球 1.06m/s）；0.05m 完全消除 AIM 階段的碰撞事件（全程 ball_speed=0.0000）。新的查表值（0.34~0.35m）遠大於這個下限。

---

## core/services/ur3e_placement_calculator.py

### `_VALIDATED_BRIDGE_*` 常數的取值 bug（曾經填錯一次）

`scripts/test_elevated_bridge_ur3e_table.py` 驗證時用的是同一組 `joints_pan0`，但當時記錄的 `direction_local`／`local_tip_position` 是套用 `required_pan` 之後的量測值，跟這裡要存的 pan=0 基準不是同一份數字——曾經直接拿那份驗證 log 的數字填進常數，填錯一次，找到 bug 後才改用 `search_ur3e_placement_constants.py` 重新在 `shoulder_pan=0` 量一次。

### `_FLAT_PLACEMENT` — 與空場景驗證數據不是同一組

跟 `scripts/test_ur3e_human_pose_swing_speed.py` 驗證過 104.7% 達成率的那組 `joints=[0,-1.7,-0.9,-1.6,-1.5708,0]` 不是同一組：那次驗證是空場景（沒有球檯），桿尖高度算出來是 1.8m，真的擺到球檯旁邊會要求基座陷進地板（跟高架橋案例第一版踩過的坑一樣）。改用有「桿尖高度合理」約束的搜尋結果，margin 從空場景版本的 33% 降到 14.6%，換取基座位置落在合理範圍。

---

## core/services/aim_shaping_calculator.py

### #124 第一輪訓練失敗模式（2026-08-11，it 238）

失敗模式不是「訊號太少」而是「reward 地形是平的」：

```
沒碰到任何球                  -1.5
碰到錯球                      -1.5     ← 與上一列完全相同
碰到 1 號球未滿 4 顆碰顆星      -0.5 + spread
合法開球                       0.0 + spread
```

policy 起點附近整片都是 -1.5，唯一的梯度是命中率約 4.5% 的那根尖刺。PPO 在平原上只會縮變異數、往取樣雜訊剛好指的方向收斂——實測 238 個 iteration 後 `Policy/mean_std` 0.400 → 0.196、`Loss/learning_rate` 6.67e-4 → 2.3e-5，而 `Episode_Reward/spread` 與 `Episode_Termination/break_foul` 雙雙歸零：前者代表沒有任何一局合法接觸（1024 env 只要有一局，raw 值就會是 4.9e-5 而不是 0.000000），後者代表也沒有碰到錯球——母球一顆球都沒碰到。

---

## core/services/rolling_resistance_service.py

### 沉降雜訊誤判為「速度未歸零」bug（GUI 實測回報）

即使只是要把沉降/多球接觸解算的數值雜訊「夾到 0」，只要每個 tick 持續呼叫 `set_velocities()`，這個顯式寫入本身就會讓 PhysX 沒機會把球放進 sleep 狀態——接觸解算會持續在雜訊量級重新產生類似大小的殘留（永遠不是精確的 0）。實測回報：9 顆 rack 球卡在 `vz≈0.0687` 永久不動，`is_ball_moving` 永遠是 True，狀態機卡死在 IDLE。修法：低於 `SETTLING_NOISE_CEILING` 的雜訊完全跳過寫入，交還給 PhysX 自己的 sleep 機制處理（純物理環境不受干擾時會在 0.4 秒內自然收斂到 0 並保持 sleep）。

### 兩個常數為何在模組層級（`ROLLING_FRICTION_COEFF`／`SETTLING_NOISE_CEILING`，#121 B-6）

RL 訓練環境（1024 env × 10 球，每 physics tick 上萬次 Python 呼叫）不能重用本類別，改用 torch 向量化重寫，但兩份實作用的物理常數必須是同一個值，否則會出現「Demo 端跟訓練端的滾動摩擦係數各自改了一次」這種靜默漂移。做法是把常數提到模組層級（而不是留在 `__init__` 的預設引數裡），訓練端直接 `import` 使用；因此兩者都是公開（非底線開頭）名稱。

`SETTLING_NOISE_CEILING` 的量級是實測出來的：沉降/多球接觸解算殘留的線速度雜訊約 1e-7~1e-5 m/s、角速度殘留雜訊約 1e-4~1e-3 rad/s，遠低於 `NEGLIGIBLE_SPEED_THRESHOLD`／`NEGLIGIBLE_SPIN_THRESHOLD`（那兩個是「已經停止，直接夾到 0」的**視覺**門檻，跟這裡「分辨真殘留 vs 純數值雜訊」是不同層級的判斷——#203 回報的門檻附近低速蠕動量級明顯大於這裡的雜訊上限，不會被誤判跳過）。

訓練端在這個門檻內的行為與本類別**刻意不同**：本類別完全跳過寫入，把收斂交還給 PhysX sleep；訓練端的張量 API 是整塊寫入，沒辦法逐球跳過，因此改為主動把三軸速度寫成精確的 0（含 vz，本類別是原封不動傳遞），不需要 sleep，只需要 `BallMotionMonitor.SPEED_THRESHOLD` 讀得到 0。

---

## core/services/ur10e_swing_strategy.py

### `ur10e_placement_calculator` 基座策略調整（2026-09-03）

decision 4 原本假設固定基座位置夠用，實測發現對某些母球位置距離目標遠達 2.6m、超過 UR10e 1.3m 可達距離，因此改回 per-shot 重新計算。

### `compute_roll_minimizing_reorientation()` 的排查過程（2026-09-03～09-04）

固定用預設 `roll_rad=0` 算出來的姿態，對某些目前姿態（尤其從 HOME 出發的 flat 案例）剛好是最壞選擇——跟目前姿態接近正反面，RMPflow 被迫做接近 180 度的姿態翻轉，反應式求解容易卡在局部穩定點（實測殘留誤差 0.1-0.6m）。單純改成「翻轉角度最小」還不夠——找到的姿態可能剛好逼近 UR10e 手腕的運動學奇異點（實測：flat 案例卡在方向誤差 0.0294 rad 不再收斂，換算成桿尖偏移超過球半徑，STRIKE 完全打不到球）。

---

## extension/isaac_sim_impl_6_0/ur10e_cue_slide_controller.py

### `_MAX_STRIKE_STEPS`／中點法則取樣的排查過程（2026-09-05）

STRIKE 揮桿全程通常只有 5~6 個 physics tick（T 只有 0.1 秒量級），逐 tick 直接量測發現 velocity-mode PD 追蹤幾乎零誤差（每個 tick 量到的實際速度就是上個 tick 的指令值），排除「阻尼追不上指令」這個假設。真正原因是取樣點數太少造成的離散積分系統性低估：`_step_strike()` 原本用左端點取樣（zero-order hold 取「這個 tick 一開始」的瞬時速度），對一段持續遞增的速度曲線是標準的左黎曼和低估——手動積分實測量到的速度只走了 0.1386m，但 quintic 邊界條件保證的位移是 0.14855m，差了約 1.3cm，導致 CueSlideJoint 停在 q≈-0.03~-0.04（沒有真的到 q=0），桿尖因此沒真正碰到球心。改成中點法則取樣後精度比左端點好一個數量級。

### `switch_dof_control_mode()` 未限定 dof_indices 導致手臂漂移（2026-09-04）

不帶 `dof_indices` 會套用到全部 7 個 DOF——"velocity" 模式把 stiffness 歸零，若不限定只作用在 CueSlideJoint，等於連其餘 6 個手臂關節的 stiffness 也一起歸零，讓 AIM 收斂好的姿態在整段揮桿期間完全沒有位置回復力可以抵抗 CueSlideJoint 加速的反作用力，只剩 damping／重力補償撐著，手臂因此在推桿當下漂移（實測：STRIKE 全程桿尖 Z 座標多爬升 3.24cm，跟純沿桿軸滑動的幾何預期不符，是造成桿尖沒打中球心的真正原因）。

### 揮桿完成判定的雙邊窄窗 bug（2026-09-05）

第一版用 `abs(current_position) <= _POSITION_TOLERANCE_M`（雙邊 2mm 窄窗）判斷收斂，實測直接撞出更嚴重的新問題——目標速度 1.5+ m/s、每個 physics tick 本身就會移動約 2.5cm，遠大於 2mm 容許窗，joint 從「還沒到」一個 tick 就直接「已經衝過頭」，永遠不會有任何一個 tick 剛好落在窄窗內。於是 `timed_out` 之前 `converged` 恆為 False，指令速度持續卡在 `target_velocity` 硬推，實測衝出去 27cm 一路撞上球檯庫邊（impulse=28，比修正前的「差一點點沒到」嚴重多了）。改成「有沒有抵達或超過 0」（單邊判定）解決。

### `post_strike_retract` 階段 — 二次擊球的排查（2026-09-05）

flat 驗收的達成率過了（92.3%），但決策 7 的「母球碰撞事件數恰好 1 次」沒過：一次擊球記錄到 **3 筆 `CueStick↔Ball_0`**（impulse 0.44 / 2.53 / 0.38），中間夾著一筆 `Ball_1↔Ball_0`（0.70）。順序說明了成因——球桿在 q≈0 打出母球後就停在原地不動，而 q≈0 正是母球原本待的位置；母球撞上球堆彈回來時球桿還擋在那裡，於是又撞上兩次。這在撞球規則裡是二次擊球犯規。

根因是決策 5 明文規定的動作沒實作：「滑軌關節推完（到達接觸點）後沿原本同一條軸線自己縮回去，手臂本身保持靜止不動，縮完才由 RMPflow 接手把手臂帶回 home」。當時只做了 `retract_only`（AIM 前退桿）／`backswing`／`strike` 三個階段，揮桿收斂就直接 `_motion_active = False`。補上 `post_strike_retract` 階段後，整段動作要等球桿縮回 backswing 位置才算完成，呼叫端因此不會在球桿還擋在球路上時就進 RESET。

縮回階段刻意沿用 `move_stroke()` 當下存下的那組手臂關節位置目標（`_hold_position_targets`），而不是重新讀「縮回當下量到的實際位置」——後者等於把揮桿反作用力造成的漂移認可成新目標，位置回復力歸零，跟 `ur10e_rmpflow_controller.py` 的 `_passive_dof_hold_targets` 是同一類陷阱。

第一版縮回**完全沒動**：跑滿 180 步只退了 11mm，卡在 q≈-0.011 一動也不動。用 `scripts/test_ur10e_actuator_swing_isolated.py` 加一段 Phase 3 逐 tick 印出關節實際狀態才定位到——`stiffness=1e5`、`damping=1e4`、位置目標 `-0.15` 全都正確，問題出在 `switch_dof_control_mode("position")` **只還原 stiffness，不會動速度目標**：揮桿最後一個 tick 下的 `q̇=target_velocity=1.5116` 原封不動留著。position 模式的 PD 是 `stiffness×(位置誤差) + damping×(速度誤差)`，殘留的前推速度目標貢獻 `1e4×1.5116≈15116`，剛好抵銷位置誤差項最大的 `1e5×0.139≈13900`，兩項僵持在 q≈-0.011 達成力平衡。切模式時一併把速度目標歸零後，縮回在約 22 個 tick（0.37 秒）內回到後擺位置。

副作用：揮桿收斂的那一 tick，控制器在 `step()` 裡就切進縮回（切回 position 模式、目標指向後擺位置），該 tick 的物理是「開始煞車」而不是揮桿。`test_ur10e_actuator_swing_isolated.py` 原本用「全程 |q| 最小的 tick」當接觸瞬間，剛好挑中這一 tick，量到的桿尖速度只有 0.0365 m/s（達成率 -2.3%，還誤報成「推桿方向裝反」）。改成只採計 `step()` 前後都還在 strike 階段的 tick 後回到 102.6%。

連帶影響量測方式：`scripts/test_ur10e_table_flat.py` 的 STRIKE 迴圈現在也涵蓋縮回，迴圈結束時母球早就撞過球堆，`ball_speed_after` 不再代表球桿賦予的速度。達成率改用整段 STRIKE 觀察到的**最大**母球速度（球桿脫離接觸後只會因摩擦/碰撞遞減，峰值就是實際傳遞出去的速度）。同時把 PASS 門檻從原本寬鬆的「達成率 ≥50% 且母球碰撞 ≥1 次」收緊成決策 7 的標準：達成率 ≥90%、且**球桿-母球**碰撞恰好 1 次（母球撞球堆是這一擊預期中的結果，不列入計數）。

### 換掉量測時機後浮出的兩個既有問題（2026-09-05，已解）

**(1) 達成率其實是 131%，不是 92.3%。** 舊的 `ball_speed_after` 在揮桿迴圈結束後才量，而 `_CUE_BALL=(0.0, 0.5)` 離球堆頂點 `Ball_1`（`break_shot_position_provider.py`，y=0.635）只有 13.5cm，中間僅 7.8cm 空隙——母球 2 個 tick 就撞上球堆，所以舊數字量到的是**撞完球堆之後**的殘速，不是球桿賦予的速度。改用峰值後真實數據是：接觸瞬間桿尖 1.6247 m/s、母球 2.6200 m/s，比值 1.6125。`swing_trajectory_calculator.compute_required_tip_speed()` 用的模型是 `v_ball = v_cue × (1+e)·M/(M+m)` = 1.3197（球桿 0.5kg、母球 0.163kg、e=0.75），但滑軌關節的 drive stiffness 是 1e5，撞擊瞬間球桿是被驅動器硬撐住的，**等效質量遠大於 0.5kg**（M→∞ 時比值上限就是 1+e=1.75，實測 1.6125 正好落在 1.32 與 1.75 之間）。模型低估了等效質量，所以母球被打快 31%。

**(2) 母球從球堆彈回來會再撞上球桿。** 決策 5 的沿軸縮回已經實作且生效，但 flat 案例仍記錄到第 2 筆球桿-母球碰撞（step 21，impulse 0.285，桿尖-母球距離 0.0269 ≈ 球半徑，是真接觸不是誤判）。成因是幾何而非揮桿缺陷：母球正面撞進緊密球堆（等效質量很大）後以約 1.4~1.7 m/s 彈回，而縮回是位置驅動的指數收斂（τ=damping/stiffness=0.1s），速度從 1.27 m/s 一路衰減，追不過等速回來的母球。加大縮回距離只能延後、不能避免——球桿的軸線就是母球的回程路徑，要真正閃開得靠 RESET 把手臂帶離，但母球在 0.15 秒內就回來了，RESET 根本來不及啟動。

**兩項的解法（2026-09-06）：**

(1) 新增滑軌專用的 `compute_required_tip_speed_for_cue_slide()`，改用實測端到端比值 `CUE_SLIDE_MEASURED_SPEED_RATIO`（WAM7／UR3e 仍走理論公式）。校準過程踩到一個陷阱值得記：第一次取單次量測 1.7333 得到達成率 93.6%，看起來還差 6.4%，於是用「比值隨指令速度遞增」這個兩點趨勢外推、改成 1.6224——結果達成率跳到 110.7%，量到的比值 1.7967 比第一點還高，直接推翻那個趨勢假設。真正的主導因素是**接觸落在 physics tick 內的哪個位置**（三次接觸分別在 q=-0.00053／+0.00172／-0.00824，而 quintic 在終點附近速度變化陡峭），造成 ±5% 的 run-to-run 散布。少數幾點擬合出來的「趨勢」是在擬合雜訊。最後取三次量測的平均 1.7175，達成率 93.5%。要真的把散布壓下去得改接觸判定本身（例如讓 quintic 在 q=0 附近速度平坦），不是繼續調這個常數。

(2) 改成縮回的同時把手臂上抬 `Ur10eCueSlideController._POST_STRIKE_LIFT_M`＝0.10m，直接離開母球的回程路徑（**這偏離決策 5 的「手臂本身保持靜止不動、縮完才由 RMPflow 接手」，是使用者在看到實測數據後明確改的設計**）。上抬用 `ur10e_analytic_ik` 做 FK → 沿 +Z 平移 → IK，全程在關節空間算完，所以不需要知道底座的世界座標（底座朝向固定是單位四元數，底座 +Z 就是世界 +Z）。上抬寫手臂 6 個 DOF、縮回寫 CueSlideJoint，同一次 `set_dof_position_targets()` 下完，不需要兩個控制器並行搶同一個 articulation。實測桿尖 Z 從 0.027 抬到 0.127，球桿-母球碰撞 2 次 → 1 次。

`scripts/test_ur10e_actuator_swing_isolated.py` 的無球檯場景會走到上抬 IK 無解的 fallback（該場景手臂維持 USD 預設姿態、球桿朝正上方，末端已伸到離底座 1.51m，再抬 0.1m 超出可達範圍），正確退回「只縮回、手臂不動」並記錄 warning——這是預期行為，不是 bug。已離線驗證 FK→IK 往返本身正確（典型瞄準姿態含上抬後都有 8 組解）。

**flat 案例驗收結果（決策 6/7）**：達成率 93.5%、球桿-母球碰撞恰好 1 次、RESET+AIM 全程手臂本體碰撞 0 筆、三段動作皆無逾時。

---

## scripts/test_ur10e_table_bridge.py

### 高架橋（傾斜）案例驗收（2026-09-06，步驟 8）

決策 8 主張「flat 與高架橋在新架構下是同一條碼路，不需要額外的架構工作」。實際驗證：新腳本**只換測試點**（`cue_ball=(0.0, -0.635)`，tilt_rad=5.34°、wrist Z 抬到 0.154m，flat 只有一個球半徑 0.0286m），生產程式碼一行沒改，第一次跑就通過：

| 項目 | 結果 |
|---|---|
| AIM | 1493 步、位置誤差 0.00126 m、無逾時 |
| STRIKE 前母球速度 | 0.000000 m/s（AIM 全程沒蹭到球） |
| 達成率 | 109.2% |
| 球桿-母球碰撞 | 恰好 1 次 |
| 手臂本體碰撞 | 0 筆 |

決策 8 的主張成立。這跟 UR3e 時代形成強烈對比——那時每個傾斜角都要用 `search_ur3e_placement_constants.py` 那套 Stage 1/1.4/1.5 分別搜尋關節組合，而且搜出來的解仍受 manipulability ellipsoid 結構性限制。

達成率 109.2% 比 flat 的 93.3~93.6% 高約 9 個百分點，方向可解釋：球桿傾斜 5.34° 之後重力沿滑軌軸有分量，而 `CUE_SLIDE_MEASURED_SPEED_RATIO` 是在 flat（軸接近水平）量的。兩者都在決策 7 的 ≥90% 內，暫不為傾斜角另外建模——真要收斂得先處理那個 ±5% 的 tick 內接觸位置散布（見該常數說明），傾斜造成的偏差還沒有大到蓋過那個雜訊底。

---

## assets/cue_actuator.usda

### 專用出力機構的外觀件（2026-09-06）

GUI Demo 時球桿是憑空從手腕伸出來、中間沒有任何看得見的機構，看不出來「為什麼它可以平移」。物理上完全正確（`TableRobotManager` 建的 `PrismaticJoint`），缺的是視覺上的解釋。

新增 `assets/cue_actuator.usda`：一個線性致動器造型（安裝座＋缸體＋橘色前端導桿套＋四根拉桿），掛在 `wrist_3_link` **底下**，球桿當活塞桿從缸體前端伸縮。

**刻意不帶任何 physics schema（純外觀）**——機構的物理已經由 PrismaticJoint 正確模擬，再加一個實體剛體只會多出質量、碰撞對、articulation 拓樸的變化，等於去動已經通過 flat／bridge 驗收的行為，風險完全不對等。掛在末端連桿底下靠 USD transform 繼承跟著手臂走，不需要任何額外關節。座標慣例沿用球桿資產：局部 +Y 就是球桿軸向（`align_prim_to_target()` 讓球桿與 `wrist_3_link` 座標系重合）。

`scripts/verify_ur10e_cue_actuator.py` 驗證結果：

| 項目 | 結果 |
|---|---|
| prim 路徑 | `.../Robot/wrist_3_link/CueActuator` |
| 帶 physics schema 的子 prim | 無 |
| `dof_names` | 仍是 7 個且含 `CueSlideJoint` |
| 伸出（q=0）球桿露出導桿套 | 1.0387 m |
| 縮回（q=-0.15）球桿露出導桿套 | 0.8880 m |
| 兩者差 | **0.1507 m**（＝滑軌行程 0.15 m）|

寫這支驗證腳本踩到兩個坑：(1) 第一版沒補光源，算出來的機構整個是黑的；(2) 用 `rep.orchestrator.step()`＋`BasicWriter` 取圖會接管並暫停 timeline，導致之後的 `simulation_app.update()` 不再推進物理——縮回指令下了 180 個 tick，關節位置卻完全沒變，量到的兩個狀態一模一樣。改用 annotator 直接取畫面（不碰 timeline）才正常。

---

## extension/isaac_sim_impl_6_0/ur10e_rmpflow_controller.py（續）

### 中繼 waypoint 容許值太緊 —— 「手臂轉動非常慢」的主因（2026-09-06）

GUI 回報 FPS 只有約 15、手臂關節轉動看起來非常慢。先釐清這是**兩個可以分開的問題**：Kit 預設一幀跑一個 physics step，所以 FPS 低只是「整段變慢動作」；但即使 tick 率正常，動作本身需要的 tick 數也太多——實測 RESET 要 902 tick，換算模擬時間 **15 秒**才把手臂移到 HOME，真實 UR10e 這種動作 2~3 秒就夠了。

**先量再改**（`scripts/profile_ur10e_tick_cost.py`／`profile_ur10e_tick_ablation.py`）：

| 項目 | 成本 |
|---|---|
| `simulation_app.update()`（headless、閒置、單桌） | 11.2 ms |
| 同上但手臂運動中（無我方 callback） | 13.4 ms |
| 我們的 `controller.step()` 全部 | 4.71 ms |
| 實際每 tick | 26.8 ms（37 tick/秒）|

ablation 顯示把 `_sync_dynamic_obstacles()`／`update_world()`／`switch_dof_control_mode()` 三項全部拿掉也只省 4.71ms（37→45 tick/秒）——**瓶頸不在我們的 callback**，82% 是 Isaac Sim 基礎場景（算圖＋物理）的固定開銷，那個改不動。第一版量測還踩到一個陷阱：在緊迴圈裡連續呼叫 tensor API 量到的是**快取讀取**（`get_dof_positions()` 只有 0.009ms），要在每次呼叫前插入真實 physics step 才量得到含 GPU 同步的真實成本。

真正能改的是**動作需要的 tick 數**。根因：`_is_current_waypoint_converged()` 沿用 `_POSITION_TOLERANCE_M=0.005`（5mm），每一個中繼 waypoint 都要收斂到 5mm 才前進。RMPflow 是漸近收斂，要壓到 5mm 等於手臂在每個中繼點都減速到幾乎停住再重新加速。但中繼點只是引導路徑形狀，**最終精度本來就由收尾階段（解析 IK／joint-space finish）負責**。

`scripts/profile_ur10e_waypoint_tolerance.py` 掃描（RESET 到 HOME，只改中繼容許值）：

| 中繼容許值 | RESET tick | 模擬秒數 | 收尾後 HOME 關節誤差 |
|---|---|---|---|
| 0.005m / 0.02rad（舊） | 902 | 15.0s | 0.001876 rad |
| 0.020m / 0.05rad | 494 | 8.2s | 0.003269 rad |
| 0.050m / 0.10rad | 334 | 5.6s | 0.002984 rad |
| 0.100m / 0.20rad | 211 | 3.5s | 0.001899 rad |

四組最終誤差都遠低於驗收門檻 0.005 rad，最寬鬆那組甚至跟最嚴格那組一樣好——中繼點的精度是白花的。新增 `_WAYPOINT_POSITION_TOLERANCE_M=0.05`／`_WAYPOINT_ORIENTATION_TOLERANCE_RAD=0.10` 只給 `_is_current_waypoint_converged()` 用，收尾階段的 `_POSITION_TOLERANCE_M`／`_FINAL_ORIENTATION_TOLERANCE_RAD` 完全不動。

⚠️ 這支掃描腳本第一版忘了呼叫 `set_robot_base_pose()`，四組全部跑滿 3000 tick、關節誤差 1.07 rad——量到的完全是假的（RMPflow 以為底座在原點）。補上之後基準組精確重現 902 tick 才確認量測有效。

改後完整驗收（兩支都 PASS，所有標準都守住）：

| 項目 | flat 舊 → 新 | bridge 舊 → 新 |
|---|---|---|
| RESET | 902 → **340** | 902 → **340** |
| AIM | 2100 → **829** | 1493 → **819** |
| AIM 位置誤差 | 0.00083 → 0.00185 m（容許 0.01） | 0.00126 → 0.00204 m |
| 達成率 | 93.5% → 93.1% | 109.2% → 109.1% |
| 球桿-母球碰撞 | 1 → 1 | 1 → 1 |
| 手臂本體碰撞 | 0 → 0 | 0 → 0 |

RESET+AIM 合計 3002 → 1169 tick，60Hz 下從 50 秒縮到 19.5 秒。要再快可以往 0.1m/0.2rad 調（RESET 211 tick），但要重跑這兩支驗收。

---

## extension/isaac_sim_impl_6_0/articulation_api_impl.py（續）

### `did_last_motion_timeout()` 對 UR10e 提早誤判逾時（2026-09-06，步驟 9 GUI 首測）

步驟 9 切到生產路徑後第一次真的用 headful GUI（含 `ModelController` 真實 policy 選球）跑，手臂完全靜止不動，主控台跟 `BILLIARD_DEBUG_LOG_PATH` 都沒有任何 Python traceback（連 tick=0 的 state log 都沒有，只有球落地的碰撞事件）。

排查過程：`docs/CHANGELOG.md` 之前所有驗收（`test_ur10e_table_flat.py`／`test_ur10e_table_bridge.py`／`verify_ur10e_production_wiring.py`）全部直接呼叫 `Ur10eSwingStrategy.execute_aim()`／`execute_strike()`，**完整正式路徑（`ModelController` 真實 policy → `DemoTableOrchestrator.step()` → `TableRuntime.tick()`）從來沒有被真實跑過**。寫 `scripts/diagnose_production_tick.py` 忠實複製 production 的物件圖（含 Training 桌，跟 `_enable_training()`／`_build_demo_session()` 完全一樣的呼叫順序），自己呼叫 `session.tick()`（不透過 `SimulationManager.register_callback()`，因為那條路徑會把 `TableOrchestrator.step()` 內部 try/except 吞掉的例外整個藏起來，`ErrorState.mark_error()` 用標準 `logging.exception()` 記錄，在這個 Kit 環境的主控台完全看不到），改直接讀 `ErrorState.get_last_exception()`。

抓到的例外：`RuntimeError: 手臂動作逾時未收斂`，在 AIM 開始後僅 276 個 physics tick（不到 5 秒）就出現，狀態機卡在 `ERROR` 永遠不再往前走。

根因：`Ur10eRmpflowController._did_last_motion_timeout` 是「這條 waypoint chain 裡任何一個 waypoint 逾時過」的**累積旗標**，只有換到下一個大階段（STAGING→NEAR_FINAL 等）呼叫新的 `move_to_pose()` 才會重置——中途某個 waypoint 卡頓超過 `_MAX_STEPS_PER_WAYPOINT=240` 步，不代表整段動作最終會失敗（AIM 的多階段設計本來就是 best-effort 繼續，後續階段常常還是收斂得了）。但 `ArticulationAPIImpl.did_last_motion_timeout()` 對 UR10e 完全沒有用 `is_motion_complete()` 把關，直接原樣轉發這個旗標；`DemoTableOrchestrator._check_downstream_failure()` 卻是**每個 tick**都在查它，動作進行到一半、旗標曾經翻過一次 True，就會被誤判成「已經逾時失敗」提早標記 ERROR。

用 `scripts/diagnose_aim_failure_case.py` 拿真實 GUI 跑出來的失敗參數（`cue_ball=(-0.0364, -0.7523)`、`shot_angle=-0.0435`、`position_offset=[0.2882, 0.0833]`——**第一次**測試非零 `position_offset`）繞過 orchestrator 直接跑到底，證實同一組參數其實在 1453 步後正常收斂（`did_last_motion_timeout=False`，位置誤差 0.00092m），中間 STAGING 階段雖然回報過 `timeout=True` 但只是單一 waypoint 的暫時卡頓，NEAR_FINAL/FINAL_APPROACH 接手後照樣收斂——證實了「累積旗標中途讀到 True」跟「整段動作最終失敗」是兩回事。

修法：`did_last_motion_timeout()` 只有在 `is_motion_complete()==True` 時才回報底層旗標，動作還在進行中一律回傳 False。對 WAM7/UR3e 完全不影響行為——那邊 `self._did_last_motion_timeout = True` 本來就跟 `self._stop_motion()` 同一行程式碼、同一時刻發生，兩者從來就是同一個事件，不像 UR10e 的多階段 waypoint chain 會有「中途翻過一次 True、之後又靠新階段重置」這種暫態。

**次要發現，尚未解決**：修好上面的 bug 後用同一組參數重跑，AIM 不再卡死，但收斂明顯比驗收過的 flat／bridge 案例慢很多——STAGING／NEAR_FINAL 階段大多數 waypoint 都逼近 240 步上限才過，推算整個 AIM 可能要跑 8000~10000+ tick（2~3 分鐘）才會走完，不是幾秒內看得出進展的速度。研判是這組從未測過的大幅 `position_offset` 把逼近走廊推到接近母球避障力場的區域，RMPflow 反應式規劃在那附近收斂變慢（不是卡死，是慢），呼應本專案稍早就記錄過的已知限制：「RMPflow：反應式、每 tick 加速度場、收斂不確定」。真人在 GUI 前觀察的時間如果不夠長，這個「動得很慢」很容易被誤認成「完全不動」。這個問題留給下次 GUI 復測後視情況決定是否要處理（例如收窄 `POSITION_OFFSET_VERTICAL/HORIZONTAL` 的訓練/評估上限，或改善 STAGING/NEAR_FINAL 附近的避障参数）。

---

---

## GUI FPS 調校（2026-09-06）

使用者回報 GUI 只有 ~15 FPS、手臂關節轉動看起來很慢。這一節記錄整個量測→定位→修正的過程，含所有走進死路的假設。

相關檔案：`scripts/benchmark_gui_frametime.py`（本次新增的量測工具）、
`core/ports/rigid_body_api.py`、`extension/isaac_sim_impl_6_0/rigid_body_api_impl.py`、
`core/services/observation_builder.py`、`core/services/ball_motion_monitor.py`、
`core/services/rolling_resistance_service.py`、
`extension/billiard_digital_twin/billiard_digital_twin.py`。

### 量測方法

先確認 Isaac Sim 在 headless／無 GUI 的情況下能不能做效能剖析（可以，三種後端本機都已安裝）：

| 方式 | 用途 | 本機位置 |
|---|---|---|
| `carb.profiler-cpu.plugin` | Chrome Trace，單一程序寫 `.gz` 檔，完全不需要視窗 | `kit/kernel/plugins/` |
| `carb.profiler-tracy.plugin` + `capture.exe` | Tracy，client/server，`capture.exe` 是純命令列 | `extscache/omni.kit.profiler.tracy-1.2.0+wx64/bin/` |
| `isaacsim.benchmark.services` | app_update／physics／render／GPU frametime 的 KPI | `exts/` |

⚠️ `SimulationApp` 的 `profiler_backend` 參數**只認 `["tracy", "nvtx"]`**（原始碼
`simulation_app.py` L518-549），傳 `"cpu"` 會被靜默忽略；要用 Chrome Trace 後端必須走
`extra_args`。另外 `carb.profiler-cpu.plugin` 的存檔設定鍵是 `filePath`，不是網路上常見的
`saveFileName`（實際從 plugin binary 裡撈出來確認）。

最後沒有動用 Tracy——`isaacsim.benchmark.services` 那四個 recorder 的取數方式就夠定位了，
本次自製的 `scripts/benchmark_gui_frametime.py` 直接照抄它們的訂閱方式：

- app frametime：`carb.eventdispatcher` 訂閱 `omni.kit.app.GLOBAL_EVENT_PRE_UPDATE`
- physics frametime：`omni.physics.core.get_physics_benchmarks_interface().subscribe_profile_stats_events()`
- GPU frametime：`omni.hydra.engine.stats.HydraEngineStats().get_gpu_profiler_result()`

### 關鍵洞察：我們自己的 tick 是算在 PhysX Update 裡的

`BilliardExtension` 用 `SimulationManager.register_callback(..., PHYSICS_POST_STEP)` 註冊
`_on_tick`，也就是 observation 組裝、RMPflow 計算、每顆球的速度讀取全部**發生在物理步進
內部**，因此整包被算進 `"PhysX Update"` 這個 zone。一開始看到「Physics 佔 App_Update 的
73.4%」時差點誤判成 PhysX 解算太慢；把 `_on_tick` 單獨計時後才發現它佔了那個 zone 的
**68.8%**，真正的 PhysX 解算只有 ~6ms。

量測工具用 `gc.get_objects()` 找出活著的 extension 實例再換掉 callback，刻意不去改
production 程式碼——量測工具不該為了量測污染被量測的對象。

### 根因：逐顆球的 tensor 讀取

`RigidBodyAPIImpl` 為每個 prim path 各建一個單一 prim 的 `RigidPrim` view，
`get_position()`／`get_linear_velocity()` 每次呼叫都是一次獨立的 tensor 讀取，
`.list()` 會強制一次 GPU→CPU 同步。實測單次固定成本 0.27–0.38ms，**跟一次讀 1 顆還是
10 顆幾乎無關**。三個呼叫端每個 tick 合計約 40 次：

| API | 次數/frame | 單次 | 每 frame |
|---|---|---|---|
| `get_linear_velocity` | 19.5 | 0.270ms | 5.25ms |
| `get_position` | 10.0 | 0.384ms | 3.84ms |
| `get_angular_velocity` | 10.0 | 0.266ms | 2.66ms |
| | | **合計** | **11.74ms** |

佔 `_on_tick`（14.08ms）的 **83%**。

修法：port 新增 `get_positions(paths)` 與 `get_velocities(paths)`，實作端用
`RigidPrim(paths=[...])` 包住整批 prim（view 依 path 組合快取重用），三個呼叫端各改成
一次批次讀取。40 次 → 3 次。

`BallMotionMonitor` 順帶失去「發現有球在動就提前 return」的短路，這是刻意的：短路只有
在球真的在滾時省得到，而絕大多數 tick（RESET／AIM 期間）球都靜止，那時逐顆版本必定跑滿
10 次同步；批次版本任何情況都只有 1 次。

### 走進死路的假設（都由實測推翻）

1. **算圖是瓶頸**：`billiard_env.usda`（8.8MB）含整個 SimpleRoom——60 個 mesh、一盞
   DomeLight（4K HDR 天空）＋一盞 RectLight、16 張貼圖全部從 S3 遠端串流。看起來非常
   可疑，實際上 GPU frametime 只佔 App_Update 的 24–38%，從頭到尾都不是瓶頸。
2. **碰撞體太多**：單張桌子 81 個 collider，其中 35 個是 `approximation="none"`（＝原始
   三角網格）。把 SimpleRoom 底下 59 個 collider 全部關掉 → 20.23 FPS，跟沒關的 20.00
   完全在雜訊範圍內。**排除**。
3. **房間的算圖成本**：整個 SimpleRoom 隱藏 → 20.15 FPS。**同樣沒有差別**。
4. **`updateVelocitiesToUsd` 回寫**：關掉 → 20.93 FPS，邊際效益。
5. **物理 substep 疊加**：擔心 app frametime 拉長導致每 frame 跑多個 substep 形成惡性
   循環。實測 `minFrameRate=30`、每 frame 恰好 1.00 個 substep，**沒有這回事**。

### 量測結果

單張 Demo 桌、RTX 4090、i7-12700、RaytracedLighting、1280×720、400 frame：

| # | 設定 | FPS | App_Update | PhysX Update | `_on_tick` |
|---|---|---|---|---|---|
| 01 | 原始（Training 桌開著） | **12.01** | 83.3ms | 61.1ms | — |
| 02 | Training 桌關閉 | 20.00 | 50.0ms | 29.9ms | — |
| 03 | ＋CPU dynamics | 24.03 | — | — | — |
| 04 | SimpleRoom collider 全關 | 20.23 | — | — | — |
| 05 | SimpleRoom 隱藏 | 20.15 | — | — | — |
| 06 | `updateVelocitiesToUsd=False` | 20.93 | — | — | — |
| 08 | CPU dynamics（含 tick 計時） | 26.24 | 38.1ms | 20.0ms | 13.78ms |
| 09 | 同上（含 API 計時） | 25.74 | 38.9ms | 20.1ms | 14.08ms |
| 11 | **批次讀取**（物理設定不動） | **29.97** | 33.4ms | 15.4ms | **3.41ms** |
| 10 | **批次讀取＋CPU dynamics** | **36.31** | 27.6ms | 9.4ms | **3.41ms** |

API 呼叫成本 11.74ms/frame → **1.08ms/frame**。

### 為什麼關掉 Training 球檯

`BilliardTable` 不論 Demo 或 Training 都參照同一份 `assets/billiard_env.usda`，那份資產
含整個 SimpleRoom，所以兩張桌子等於把整個房間連同環境光載入兩次（兩盞 DomeLight 疊在
一起）。`TRAINING_TABLE_PATH`（`billiard_table_only.usda`，26KB，去掉 SimpleRoom 的版本）
只有 RL 訓練環境 `rl_task/billiard_rl/` 在用，GUI 這條路徑從來沒用到。Training 路徑在
GUI Demo 情境下沒有畫面用途，預設關閉；Debug Menu 的 Training toggle 仍可隨時開回來。

### 第二輪：關掉 GPU dynamics 與開啟 async rendering

批次讀取修完後 `_on_tick` 只剩 3.4ms，"PhysX Update" 剩下的部分才真的是 PhysX 在解算。
這時再測 GPU dynamics 就看得很清楚（600 frame，單張 Demo 桌）：

| 設定 | FPS | PhysX Update | 其中 `_on_tick` |
|---|---|---|---|
| async render，GPU dynamics **開** | 30.59 | 20.25ms | 2.85ms |
| async render，GPU dynamics **關** | **40.56** | 9.33ms | 2.86ms |

也就是 GPU 物理管線在這個場景每 frame 白花約 **11ms**。18 個剛體＋1 個 articulation
的規模，GPU 的固定開銷（kernel launch、GPU 記憶體同步、以及每次 tensor 讀取都要
GPU→CPU 搬一次）遠大於它平行化能省下來的。broadphase 一併從 GPU 改成 MBP。

⚠️ 這個結論**只對 Demo 規模成立**。RL 訓練環境（`rl_task/billiard_rl/`）是 1024 個平行
env、上萬個剛體，那個量級 GPU 物理才會贏。設定收斂在
`extension/isaac_sim_impl_6_0/physics_scene_tuning.py`，函式 docstring 裡寫明這個界線。

`SimulationManager.setup_simulation()` 建出來的 PhysicsScene 預設就是開 GPU dynamics，
所以這是覆寫它的預設值，不是「打開某個開關」。

兩支真實球檯驗收腳本也一起套用同一個設定——驗收腳本原本是自己 `UsdPhysics.Scene.
Define()` 一個裸的 scene，跟 GUI 走的不是同一條建立路徑；如果兩邊物理設定不同，
「驗收通過」就不能代表 GUI 的行為。這是把設定抽成共用函式（而不是在三個地方各寫一次）
的理由。

async rendering 則是 app 層級的啟動選項（`--/app/asyncRendering=true`），寫在
`docs/ur10e-step9-gui-verification-checklist.md` 的啟動指令裡，不放進 extension 程式碼
——extension 不該擅自覆寫 app 的算圖設定。

### 剩下沒解決的：週期性卡頓

最佳設定下 App_Update 的 **median 是 17.3ms（≈57.7 FPS）**，但 mean 是 24.7ms
（40.6 FPS）、p95 高達 77ms。也就是穩態其實已經很接近 60 FPS 的完美目標，是少數幾個
特別長的 frame 把平均拉下來（600 frame 裡約 5% 超過 77ms，暖機 240 frame 已排除，
所以不是啟動期的一次性成本）。

還沒查出這些長 frame 的成因。已知不是：物理 substep 疊加（實測固定 1.00 個/frame）、
不是算圖（headless 完全不開視窗也只有 39.09 FPS，跟開視窗的 40.56 差不多）。下一步
應該用 Tracy（`capture.exe`，headless 可用，見本節開頭的表格）抓一段含卡頓的 trace，
看那幾個 frame 的 zone 樹跟正常 frame 差在哪。

### 試過但不採用：關掉 USD 回寫

`updateToUsd`／`updateVelocitiesToUsd` 兩個回寫開關都測過（600 frame，其餘設定為最終
production 設定）：

| 設定 | FPS | App_Update median | App_Update p95 | Physics Update Transforms |
|---|---|---|---|---|
| 最終 production 設定 | 46.62 | 17.4ms | 29.6ms | 2.98ms |
| ＋`updateToUsd=False` | 47.79 | 14.9ms | **75.0ms** | 1.41ms |
| ＋`updateVelocitiesToUsd=False` | 48.43 | 16.4ms | 38.5ms | 2.35ms |

`Physics Update Transforms` 確實從 2.98ms 掉到 1.41ms，但整體只有 +2~4%，而 p95 在各次
執行之間本來就會在 29–77ms 之間大幅擺動（同一組設定重跑就會變），這個量級的差異分辨
不出來。`updateToUsd=False` 另外還有風險：任何直接讀 USD transform 的地方（例如
`StageAPIImpl`）會拿到不再更新的舊值。**收益不明確、風險明確，不採用。**

### 卡頓的追查進度（未結案）

最終設定下 1500 frame × 多輪的穩定數字：mean 48–50 FPS、**median 約 17ms（≈58 FPS）**、
p95 55–77ms。也就是穩態已經很接近 60 FPS 的完美目標，是少數幾個特別長的 frame 把平均
拉下來。`scripts/benchmark_gui_frametime.py` 加了慢 frame 分析（門檻＝median×2）：

| 設定 | 慢 frame | 平均耗時 | 佔總時間 | 出現間隔 |
|---|---|---|---|---|
| 完整算圖 | 59/1500 | 88.4ms | 18.2% | median 18 frame，stdev 22.5 |
| 完全不開視窗（headless） | 91/1500 | 79.5ms | **26.2%** | median 12，stdev 16.6 |
| `renderer=Minimal` | 87/1500 | 80.5ms | 24.8% | median 10，stdev 17.4 |

**已排除的成因**：

1. **算圖／資產串流**——拿掉視窗跟改用 Minimal renderer，卡頓不但沒消失，佔比反而更高。
   （原本很懷疑資產的 16 張貼圖與 4K HDR 天空都是從 S3 遠端串流。）
2. **物理 substep 疊加**——實測固定 1.00 個 substep/frame。
3. **我們自己的 tick**——`_on_tick` 的 p95 只有 3.3–3.6ms，解釋不了 App_Update 的 p95 55–77ms。
4. **Python GC**——用 `gc.callbacks` 掛上計時器實測，1500 frame 的量測窗內**一次 collection
   都沒有發生**。80ms 量級、間隔不規則、偶爾連續兩 frame 的形狀很像 GC，但實測直接否定。

消除這些長 frame 可以把 mean 從約 20ms 降到約 16.5ms，剛好就是 60 FPS，所以這是通往完美
目標的唯一一條路。下一步：用 Chrome Trace 後端（`--/app/profilerBackend=cpu`，輸出是可以
程式解析的 JSON，不像 Tracy 要開 GUI）抓一段含卡頓的 trace，比對長 frame 與正常 frame 的
zone 樹差異。`scripts/benchmark_gui_frametime.py` 的 `BENCH_CHROME_TRACE` 環境變數已經接好
這條路徑。

### 卡頓的根因：USD stage 鎖爭用（已定位，修法待目視確認）

用 Chrome Trace 後端抓了一段 trace 之後定位出來。做法（`scripts/benchmark_gui_frametime.py`
的 `BENCH_CHROME_TRACE` 環境變數）：

```
--/app/profilerBackend=cpu --/app/profileFromStart=1 --/profiler/enabled=true
--/plugins/carb.profiler-cpu.plugin/saveProfile=1
--/plugins/carb.profiler-cpu.plugin/compressProfile=1
--/plugins/carb.profiler-cpu.plugin/filePath=<path>.gz
```

輸出是 NDJSON 包在一個 array 裡，可以 streaming 逐行解析，不需要開 Tracy GUI。分析方式：
取主執行緒（`App Main loop` 所在的 tid）的所有 zone，把落在「慢 frame」時間區間內的
zone 跟落在正常 frame 內的分別加總比對。

⚠️ 這份 trace 本身被 profiler 污染了——`ScopedGzJsonFile.writeFile` 43ms × 313 次是
profiler 自己在寫檔。但比對「慢 frame vs 正常 frame」的相對差異仍然有效。

結果（每 frame 平均）：

| zone | 慢 frame | 正常 frame | 倍數 |
|---|---|---|---|
| `UsdContext hydraRender` | 45.9ms | 1.8ms | 25× |
| `Lock USD` | 44.4ms | 未進前 14 名 | — |

也就是主執行緒**卡在等 USD stage 的鎖**。physics 每步把 transform 回寫進 USD
（`/physics/updateToUsd`）要拿寫鎖，hydra 算圖要拿讀鎖，兩邊互相等。這也回頭解釋了為什麼
關掉算圖反而更多慢 frame（沒有算圖排隊，物理寫入更密集）、以及為什麼
`updateToUsd=False` 那次 median 掉到 14.9ms。

**修法候選：Fabric scene delegate**（GPU-resident，把「物理→算圖」這條資料流整個移出
USD，正是為了消除這種爭用）。實測（1500 frame，其餘同最終 production 設定）：

| 設定 | FPS | median | **p95** | 慢 frame 佔時間 |
|---|---|---|---|---|
| 最終 production 設定 | 48–50 | 17.0–17.3ms | **55–77ms** | 25.1% |
| ＋Fabric | **51.25** | 17.0ms | **29.6ms** | 20.6% |

```
--/app/useFabricSceneDelegate=true
--/physics/fabricUpdateTransformations=true
--/app/usdrt/scene_delegate/enableProxyCubes=false
```

p95 砍半，佐證了鎖爭用的診斷。**但沒有採用**：Fabric 換掉整條算圖資料路徑，畫面是否
完全正確（材質、可見性切換、`TableBallSet.hide_ball()` 的進袋隱藏）必須肉眼確認，而且
`useFabricSceneDelegate` 一般是啟動期設定，在 runtime 才設不保證完全生效。要採用的話應該
走啟動參數並實際看畫面。
