# #180 可達性掃描前置分析：基座位置與球桿握把點

**狀態**：分析完成，基座位置決策已更新（見第九節），待實際 orientation-constrained IK 掃描驗證
**日期**：2026-08-16
**關聯**：#180（可達性掃描與可行性地圖）、#183（B-CP1 決策點）、#91（已併入 #180）

## 目的

在真的跑 orientation-constrained IK 網格掃描（#180 本體工作）之前，先用現有程式碼與資產裡的實際數字，找出「基座位置為什麼可能覆蓋不了 Kitchen 區」的根本原因，把可調的設計變數縮小到幾個具體槓桿，避免一開始就對整個（母球位置 × 方向角）網格做窮舉。

**範圍限定**（已與使用者確認）：只處理 fallback (b) 需要的「瞄向球堆」窄角錐，不處理 Milestone B 走位球需要的整圈方向。

## 問題範圍限定

`action_bounds.py` 的 `SHOT_ANGLE` 目前收窄為 Milestone A 訓練用的 ±30°，但 Milestone B 之前必須改回整圈 (-180, 180)（見該檔案第 43–45 行的註解）。本分析**只處理窄角錐版本**——瞄向球堆所需的實際角度範圍，不含走位球的整圈需求。

## 一、幾何輸入（真實數字，非假設）

| 項目 | 數值 | 來源 |
|---|---|---|
| 母球擺位區（Kitchen） X | ±0.606425 m | `core/models/action_bounds.py` `CUE_BALL_PLACEMENT_X` |
| 母球擺位區 Y | -1.241425 ~ -0.635 m | 同上 `CUE_BALL_PLACEMENT_Y` |
| 1 號球（瞄準目標）位置 | (0, 0.635) m | `core/services/break_shot_position_provider.py` `BREAK_SHOT_POSITIONS[1]` |
| 兩個 head string 角落的最大瞄準角 | ±25.524° | 幾何算出，與 `action_bounds.py` 註解中的數字互相驗證一致 |
| fallback (b) 偏移 | 固定 0（`max_offset=0.0`） | #183 |

角度定義：0° 朝桌台 +Y，正角朝 -X（`action_bounds.py` 第 14 行）。因為偏移固定 0，每個擺位點的桿頭路徑退化成一條沿瞄準方向的線段，不是面或體。

## 二、發現一：手臂已換成 Barrett WAM 7-DOF，臂展需重新計算

`extension/billiard_digital_twin/billiard_digital_twin.py:49` 確認 `_ROBOT_ARM_CLASS = BarrettWamRobot`。原本沿用 issue #180 文字裡的「UR5 850mm」是舊資訊（UR5 已在 2026-07-26 換裝時淘汰，見 `action_bounds.py` `CUE_BALL_SPEED` 註解）。

從 `assets/barrett_wam/wam7.urdf` 的關節鏈量出：

- 肩部到 elbow（上臂）：0.55 m
- elbow 到腕部（前臂）：0.3 m
- 腕部到 `wam_wrist_palm_stump_link`（工具掛載點）：0.06 m

**WAM7 理論最大臂展 ≈ 0.91 m**（完全伸直上限，未計關節限位）。

## 三、發現二：機器人掛載高度與球檯平面差 1.317 m

追查 `table_center` 的來源鏈：

- `core/models/billiard_table.py:29` `self._z_pos = 0`（寫死）→ `get_table_center()` 回傳 `(x, y, 0)`
- 球的實際世界高度：`table_ball_set.py` `z = table_z + ball_radius ≈ 0 + 0.028575`，球檯打球平面幾乎正好在世界 Z ≈ 0
- `TableRobotManager` 把 `world_position.z = table_center[2] + 0 = 0` 傳給機器人
- 但 `wam7.urdf` 內部自帶兩層固定位移：`world→wam_base_link` 的 `z=1.0`，加上 `wam_base_link→wam_shoulder_yaw_link` 的 `z=0.346`

**肩部世界高度 = 0 + 1.0 + 0.346 = 1.346 m，與球檯平面（≈0.0286m）相差 1.317 m。**

這個高度差本身已經接近整條手臂的理論臂展（0.91m），對水平可用臂展的侵蝕非常大。

## 四、發現三（主因）：球桿握把點在最尾端，槓桿臂效應是最大瓶頸

查 `core/models/table_robot_manager.py` 的 `align_prim_to_target` 呼叫（實作在 `extension/isaac_sim_impl_6_0/stage_api_impl.py:74`）：這個函式把 `CueStick` prim 的**整個 world transform（含旋轉）直接設成與 end-effector 一致**，再用 `FixedJointToRobot` 鎖死。

配合 `assets/ball_stick.usd` 的桿身幾何（`Cylinder height=1.5`，本地原點在桿的一端，桿身沿本地 Y 延伸 1.5m）：

**end-effector 的位置就是桿的一端（握把/butt），桿尖（觸球端）在 end-effector 前方固定 1.5m 處，方向由手腕姿態決定。**

也就是說，手臂真正要伸到的點不是擊球點 `p`，而是：

```
end_effector = p − G · d̂(θ)
```

其中 `G` = 握把到桿尖的距離（目前資產是整支 1.5m），`d̂(θ)` = 瞄準方向單位向量。

### 4.1 用四個 Kitchen 角落算 Chebyshev bisector（對稱軸 X=0）

Kitchen 擺位區與目標球都對稱於 X=0，最佳基座位置必落在球檯中心線上，不需要掃描就能得出這個結論。用兩個代表性角落（Y=-1.241425 與 Y=-0.635 各取一個）求等距點，解出：

| 握把到桿尖距離 G | 最佳基座 Y（球檯相對座標） | 所需水平臂展 R_h | 對應 Z 高度差預算上限 `√(0.91²−R_h²)` |
|---|---|---|---|
| 1.5 m（目前資產設計） | -2.01 | 1.25 m | 不存在（已超出理論臂展） |
| 0.6 m | -1.40 | 0.89 m | 0.18 m |
| 0.4 m | -1.25 | 0.82 m | 0.40 m |
| 0.2 m | -1.10 | 0.75 m | 0.52 m |

（R_h 已含一個保守估計的後擺走廊 L≈0.1~0.15m）

**目前 G=1.5m 的設計下，不論基座 X/Y/Z 怎麼擺，所需水平臂展都超出 WAM7 理論最大臂展——這是比「基座位置沒調好」更根本的問題。**

## 五、目前 placeholder 基座位置的驗證

`core/models/table_robot_manager.py:14` `_ROBOT_OFFSET_FROM_TABLE_CENTER = (1.5, 0.0, 0.0)`：基座在球檯相對座標 (1.5, 0)。

即使 ΔZ=0，到 Kitchen 最近角落 (0.606, -0.635) 的水平距離已是 **1.096 m**，同樣超出 0.91m 理論臂展。這個常數目前應視為**未經高度／臂展驗證的佔位值**，不是 #91/#180 要交付的最終設計值。

## 六、可調的三個槓桿（依影響力排序）

1. **縮短球桿握把點 G**：目前握在桿身最尾端（G=1.5m），需要改成握在桿身中段，建議目標 **G ≈ 0.2~0.4 m**。這是影響最大的一步，需要修改 `table_robot_manager.py` 的掛接邏輯（`align_prim_to_target` 的目標改成桿身上偏移一段距離的參考點，而非直接對齊 end-effector）。
2. **重擺基座 X/Y**：從目前 `(1.5, 0)` 改到對稱解附近 **(0, -1.1 ~ -1.25)**（球檯相對座標，依最終選定的 G 而定）。
3. **大幅降低肩部相對球檯的高度**：目前 1.317m 的高度差需要壓到 0.4~0.5m 以內。WAM7 URDF 內建的「地面式底座」假設（`world→base` 內建 1.0m 抬升）本身可能不適合桌面應用，或許需要側掛/懸吊等完全不同的掛載方式，而不只是調 `_ROBOT_OFFSET_FROM_TABLE_CENTER` 的 Z 分量。

## 七、與現有專案決策的銜接

即使做完上述三項調整，理論值餘裕仍然很薄（幾公分等級的水平臂展餘裕），還沒扣掉後擺、關節限位、姿態約束會吃掉的部分。#183 已經明文承認這種情況：

> 覆蓋不全不是失敗：把 policy 的母球放置範圍 clamp 到可行域（設計輸入，非缺陷）

因此 #180 的收斂判準建議調整為：完成第六節三項調整後，跑一次真正的 orientation-constrained IK 掃描，量出實際可行域，再把 `CUE_BALL_PLACEMENT_X/Y` 收窄成該可行域——這是專案既有決策允許的收尾方式，不需要證明能覆蓋整個現有 Kitchen 矩形。

## 八、本分析的已知簡化假設（下一步驗證用）

- 臂展 0.91m 是「完全伸直」的理論值，未計入關節限位、奇異點迴避、orientation-constrained IK 的實際姿態約束——真實可用臂展會更小
- 高度差與水平距離用純球面近似（`R_h² + ΔZ² = R²`），未考慮 WAM7 各關節軸向的實際指向限制
- 後擺走廊長度 L 目前是估計值（0.1~0.15m），未在程式碼或文件中找到實測/設計定案數字
- 表格中「最佳基座 Y」只用了兩個代表性角落做 Chebyshev bisector，未驗證其餘角落與擺位區內部點是否仍在該基座位置的可達範圍內

## 九、決策更新（2026-08-16）：基座位置（含高度）逐球可變，桿長維持標準

### 背景

第六節提出的三個槓桿（縮短握把 G、重擺基座 X/Y、降低掛載高度）都建立在「基座位置是單一固定常數」的假設上。這次決策把這個假設整個推翻，連帶讓前兩個槓桿失效。

### 決策內容

1. **基座位置（含 X/Y/Z）不再是固定常數，改為逐次擊球依 Action 需求動態決定。** `TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER`（`core/models/table_robot_manager.py:14`）目前仍是 class 常數，這個決策代表它未來要改成呼叫端逐球傳入的計算結果，不是初始化時定死的值。
2. **本次不處理「如何實際重新定位」的機構或實作**——真實世界要嘛靠可移動機構（滑軌、機械手臂搬運座），要嘛靠別的解法；這件事會在 Demo 時明確說明是本次不解決的範疇。這裡只確認「假設基座自由，可達性問題是否解得開」這個理論問題。
3. **球桿維持標準桿長（1.5m），握把回復真實比例**（G≈1.35m，桿尾露出 0.15m，見下方資產調整）——取代第六節「縮短握把」的建議，該建議已作廢。

### 為什麼這樣可行

R_h（水平臂展需求）只跟 G 與瞄準角度有關，跟基座位置無關；基座位置只決定「在哪裡量測」。用第四節同一套 Chebyshev bisector 方法重算，G=1.35m 時 Kitchen 兩個代表角落算出的 end-effector 需求點（桌台相對座標）分別在 (1.19, -1.85) 與 (1.02, -2.53)——兩點都已經在球檯外面（頭岸之外），因為 G 越大，握把點沿瞄準反方向被推得越遠，而 Kitchen 瞄向 1 號球的反方向剛好是把 end-effector 推出頭岸。

基座位置一旦可以逐球自由選，直接把基座放在這個需求點附近即可，R_h 可壓到接近 0，不用犧牲 G。

高度同理：肩部世界高度 = `base_z + 1.0`（`world→wam_base_link`）`+ 0.346`（`wam_base_link→wam_shoulder_yaw_link`）。原本 `base_z` 固定為 0，才會有肩部恆定 1.346m、跟球檯平面 0.0286m 差 1.317m（超過 WAM7 理論最大臂展 0.91m 達 0.41m）這個死結。`base_z` 一旦也自由，這個差值同樣可以被抵銷到接近 0。

R_h 與 ΔZ 都壓到接近 0 之後，所需總臂展趨近後擺走廊 L（≈0.1~0.15m），遠低於 0.91m 理論值，餘裕寬裕。

### 這個決策改變了什麼

- 第六節「縮短握把 G」「重擺基座 X/Y」兩個槓桿不再需要，由本節取代。
- 第六節「降低肩部相對球檯高度」這個槓桿，改成「基座高度逐球自由設定」繞過，不是真的把 WAM7 URDF 內建的 1.0m 地面掛載抬升改掉。URDF 本身內建假設帶來的問題還在，只是被「基座自由」這個更高層的決策蓋過去，還沒真正解決根因——未來若真的要做實體機構，第六節第三點的問題還是要面對。
- #180 驗收條件第二條「確認基座位置合理」的意涵，從「確認一個固定座標」變成「確認『逐球最佳化基座位置』這個方法本身合理」。

### 資產調整

`assets/ball_stick.usda` 的 `Cylinder` pivot 已從 G=0.3m（上一輪的「縮短握把」方案，已作廢）調回 G=1.35m：

```
xformOp:translate = (0, 0.6, 0)
```

桿身總長不變（`height = 1.5`），握把（`CueStick` 原點）到桿尖的距離 1.35m，到桿尾的距離 0.15m，符合真實球桿握姿比例。

### 尚未解決／仍需驗證

- ~~這裡的 R_h≈0、ΔZ≈0 只是理論球面距離的近似，不是真實 IK 解~~ → 第十節已用實際差動 IK／固定姿態在 headless Isaac Sim 中驗證過，不再只是近似。
- ~~「由 Action 反推基座位置」的具體演算法還沒定案~~ → 第十節已定案並實作（`core/services/base_placement_calculator.py`）。
- 仍未消除：關節限位、奇異點迴避、球檯庫邊碰撞（見第十節「還沒解決的部分」）。

## 十、實作與驗證（2026-08-16）：差動 IK 收斂失敗 → 改用固定姿態

### 差動 IK 的實測結果：收斂失敗

第九節的決策是理論分析，本節是實際動手驗證。先按原計畫用專案既有的差動 IK（`ArticulationAPIImpl`，Jacobian-based DLS）去解 Kitchen 兩個代表角落的需求點（`scripts/probe_base_reachability.py`，headless Isaac Sim，不需要 GUI/RDP，用 `ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES` 繞過互動式 EULA 提示）。

結果：**兩個點都沒收斂**，最終誤差 0.63~0.88m。深入看數字，問題不是「搆不到」——目標點離肩部只有水平 0.125m、垂直 0m，遠低於 0.91m 理論最大臂展。真正的問題是 WAM7 剛 spawn 時的預設姿態（全關節 0）末端執行器朝正上方完全伸直，離肩部足足 0.91m，貼在工作空間邊界上；差動 IK 要處理的是「從幾乎完全伸直收回」這種大幅度、經過邊界附近的運動，這正是 Jacobian-based 局部線性化方法最容易失穩、卡住的情境。加中繼點分段收斂（12 個 waypoint）改善了垂直方向，但水平方向換了新的瓶頸，兩個探測點誤差幾乎一樣（0.629m vs 0.636m），像是卡在某個固定的關節限位或奇異點附近。

### 改用固定姿態＋`base_yaw` 關節，取代逐次即時 IK

差動 IK 是為「小幅度推桿微調」調校的（`ArticulationAPIImpl` 註解本身也承認這點），不適合「基座逐球重擺」這種大範圍收臂＋轉向的初始定位。改用完全不同的策略：**手臂 6 個關節永遠鎖定同一組角度，只有 `wam_base_yaw_joint`（機器人自己的第一個關節，限位 ±2.6 rad）隨瞄準角變化**——不需要 runtime IK 收斂，是普通的 joint-space 位置控制（跟既有的 `move_to_home()` 同一種機制，穩定可靠）。

這個做法把「每個 Kitchen grid point 都要即時解一次 IK」，降成「離線只需要成功解一次」，而且離線可以用手動試誤慢慢湊，不需要保證 runtime 每次都收斂。跟 #181（關節空間揮桿軌跡生成）的既有設計高度吻合——#181 本來就規劃「預先規劃揮桿的關節角度曲線，直接以 joint position/velocity target 播放」（固定軌跡，不是動態 IK），也已經規劃「偏移量先離散化三檔」，這次的固定姿態是同一個精神的延伸。

用 `scripts/probe_canonical_pose.py` 手動試誤（直接下 joint position target，不靠 IK 收斂）找到一組可行姿態：

```
CANONICAL_REST_JOINTS = (shoulder_pitch=1.9, shoulder_yaw=0, elbow_pitch=1.8,
                          wrist_yaw=0, wrist_pitch=0, palm_yaw=0)
```

`shoulder_pitch` 距限位（1.985）留了 0.085 rad 餘裕。實測確認 `base_yaw` 每轉 δ，桿尖方向角同步偏轉 δ（1:1、同向）——公式的正負號是實測出來的，不是推導假設。

### 反推公式（`core/services/base_placement_calculator.py`）

```
grip = required_grip_position(cue_ball_x, cue_ball_y, shot_angle_deg)   # 沿用第四節公式，未變
base_yaw_rad = radians(shot_angle_deg) + π/2
base_position = (grip.x − _LOCAL_TIP_RADIUS·d̂.x,
                  grip.y − _LOCAL_TIP_RADIUS·d̂.y,
                  grip.z − _LOCAL_TIP_HEIGHT)
```

`_LOCAL_TIP_RADIUS`（0.35342m）與 `_LOCAL_TIP_HEIGHT`（0.79640m）是 `base_yaw=0`、基座在世界原點時量出的桿尖位置，改資產或重新調校關節角度時必須重新跑 `probe_canonical_pose.py` 量測，不能手動猜數字。第九節的 `STANDOFF`／動態 IK 假設已作廢，公式整套改寫。

### 端到端驗證：位置與姿態都對得上，且意外全部水平

`scripts/validate_fixed_pose_placement.py`：把公式算出的基座位置＋`base_yaw`＋固定姿態實際套進場景，真的掛上 `ball_stick.usda`（`align_prim_to_target` 對齊腕部），量 Kitchen 兩個代表角落：

| | XY 誤差 | Z 誤差 | 桿身傾斜角 |
|---|---|---|---|
| near_corner | 0.00004 m | 0.00001 m | 0.04° |
| far_corner | 0.00002 m | 0.00001 m | 0.04° |

位置誤差在 0.05mm 等級，桿身傾斜角 0.04°——幾乎完全水平，不是刻意調出來的，是這組固定姿態的副作用。第九節原本標注「姿態是否水平未驗證」的疑慮，這裡已用實測推翻。

### 還沒解決的部分

- **這組公式只覆蓋 fallback (b) 的窄角錐**：`base_yaw` 目標值必須落在 `wam_base_yaw_joint` 限位 [-2.6, 2.6] rad 內才有效，目前的瞄準角範圍（±30°，legal aim ±27.586°）換算後穩穩落在限位內，但 Milestone B 走位球需要的整圈方向（-180°, 180°）不在覆蓋範圍——那需要另外設計（基座本身的旋轉，不能只用這一個關節）。
- ~~球檯庫邊碰撞完全沒測過~~ **已完成，見第十一節**（2026-08-23）。
- **關節限位、奇異點迴避、後擺走廊仍未驗證**：見第八節既有簡化假設清單，這輪驗證的是「這組固定姿態能不能到位」，不是「掃過整個 Kitchen 網格＋整段揮桿軌跡都合法」。
- **基座沉到地板以下（`base_z` 為負）的問題依然存在，且與姿態固定與否無關**：肩部到桌面高度差 1.317m 本身就超過 WAM7 理論最大臂展 0.91m，這是純幾何限制，不管姿態固定不固定、不管挑哪個關節配置都無法迴避，只要基座 Z 卡在地板（0）就構不到——第九節「基座位置（含高度）逐球可變」這個決策本身沒被取代，本節只是換了達成方式。

## 十一、#233 球檯庫邊碰撞檢查結果（2026-08-22～2026-08-23）

### 方法

用 `scripts/scan_rail_collisions.py` 對 `_CUE_BALL_X_GRID`（-0.5~0.5）×
`_CUE_BALL_Y_GRID`（-1.1~0.9）25 點網格，套用真實 `BilliardTable` +
`TableRobotManager`（跟正式流程同一條建構路徑），複製 `_execute_aim()` 的
`CANONICAL_REST_JOINTS + base_yaw` joint-space 目標，開 PhysX contact
reporting 掃碰撞。初次掃描 76% 撞庫邊（19/25）——固定姿態是平躺的，握把
到球的直線常常會先穿過庫邊上緣才到球。

### 「高架橋」抬高姿態：解法

參考真人打「高架橋」（elevated bridge）技術：握把端抬高、桿頭仍貼著母球，
讓桿身從庫邊上方通過。`scripts/scan_elevated_bridge_approach.py` 把這個
技術公式化：

- `compute_required_tilt_rad()`：握把→母球連線跟四面庫邊的交點，反推最小
  仰角 φ，使桿身在交點處的高度剛好清過庫邊頂部 + 安全餘量。
- `compute_tilted_wrist_pose()`：仰角 φ 決定的目標腕部位置/姿態（水平分量
  乘 cosφ、多一個垂直分量 sinφ），外加一個繞桿身軸的 `roll` 自由度（5 維
  冗餘，不影響擊球結果，純粹用來閃避特定關節配置卡限位）。
- 逼近軌跡：不能從全關節 0 的預設姿態直接跑差動 IK（工作空間邊界附近會
  失穩，見第十節），先用 joint-space 帶到 `CANONICAL_REST_JOINTS`（安全
  起點），再分階段（轉向到「朝上」→ 平移到高處 → 原地轉到最終傾斜姿態 →
  垂直下降）用差動 IK 逼近，避免桿頭在轉換過程中意外下探撞到桌面。

不需要抬高（tilt=0，「flat」案例，全網格僅 3 點）的沿用原本水平的
`CANONICAL_REST_JOINTS`。

### 結果：25 個網格點，房間位移修正後全數無碰撞

初版「高架橋」（固定 roll=90°）只解到 48%；改成逐點嘗試
`ROLL_CANDIDATES_DEG=(90,-90,45)` 挑第一個成功的、且把 3 階段軌跡的每一段
都做碰撞檢查（不是只查終點）後提升到 76%；再補上手臂本體（不只球桿）的
碰撞偵測（原本只對球桿開 contact reporting，漏了手臂連桿本身撞牆/撞地板
的案例）後到 80%。

剩下的 5 個失敗案例（母球 Y=-1.1 那一整排）追查後發現是撞到球桌 Head 端
一道室內隔間牆（`Towel_Room01_wood_wall_308~316`），離算出來的機器人基座
位置只有 ~0.5-0.6m，且該側牆是室內隔間、不是房間外殼，Foot 端（+Y）確認
有充足淨空。把 `assets/billiard_env.usda` 的 `SimpleRoom`
Xform 整體平移 `(0,-2,0)`（只動房間殼，不動 `BilliardTable`，`table_center`
相依的所有計算完全不受影響）後，這 5 個案例全部轉為無碰撞。

**最終：25/25 網格點 0 碰撞（100%）。**

### 已知限制（供 #181 揮桿軌跡設計參考）

- **22/25 網格點需要「高架橋」抬高姿態（tilt>0）才能避開撞庫邊**：#181
  規劃揮桿弧（後擺→加速→擊球點）時，這些位置的整段軌跡（不只是最終
  擊球瞬間）都要維持抬高姿態、沿桿身軸方向揮動，不能直接假設水平揮桿。
  公式與分階段軌跡邏輯見 `scripts/scan_elevated_bridge_approach.py` 的
  `compute_required_tilt_rad()` / `compute_tilted_wrist_pose()` /
  `_run_elevated_bridge_case()`，目前只存在研究腳本，尚未整合進
  `core/services/base_placement_calculator.py` 或
  `core/services/table_orchestrator.py`。
- **3/25 網格點為 flat（tilt=0）案例，其中 2 個仍有未解的殘留定位誤差**：
  `(-0.25, -0.1)` 與 `(0.25, -0.1)` 這兩個母球位置，用
  `CANONICAL_REST_JOINTS + base_yaw` joint-space 目標，已排除力矩飽和、
  關節限位、自我碰撞、DOF 順序、PD stiffness 不足、碰撞（GUI 實測確認
  唯一的幾何重疊是進袋用的 Trigger 區域，沒有反作用力）——仍穩定卡在
  24-27mm 誤差（`is_motion_complete()` 恆為 False），根因未查明。#181
  規劃這兩個位置的揮桿起始姿態時，需要考慮這個已知的 ~25mm 偏移量（例如
  用實測到位姿態而非理論值當起點），或是排入 #181 之前優先解決。完整
  排除清單、已測手法（solver iteration count、分段逼近等）見
  `docs/issue-flat-case-residual-error.md`。

## 十二、#181 實作與驗證發現的新限制（2026-08-23）

第十一節的高架橋公式與分階段軌跡邏輯已經整合進正式程式碼
（`core/services/cue_pose_calculator.py`、`core/services/swing_trajectory_
calculator.py`、`core/ports/articulation_api.py` 的 `move_through_poses()`、
`DemoTableOrchestrator._execute_aim()`/`_execute_strike()`），並修正了一個
從研究腳本直接搬過來的既有缺陷：Phase 0（joint-space 回安全姿態）的
`target_end_effector_position` 原本沿用移動前的舊位置當佔位符，研究腳本
沒踩到是因為它固定跑 300 步、從不真的檢查 `is_motion_complete()`；正式
程式碼用收斂判定驅動自我轉階段，這個佔位符會讓判定永遠等不到，卡死。已用
新增的 `base_placement_calculator.compute_canonical_wrist_position()`（
`compute_base_pose()` 反推公式的正向版本）修正。

### 新發現：高架橋抬高姿態在 Kitchen 母球範圍內撞到 shoulder_pitch 關節限位

用 `scripts/verify_swing_trajectory.py` 對 `core/models/action_bounds.py`
的 `CUE_BALL_PLACEMENT_X/Y`（Kitchen 母球位置範圍，整個範圍都靠近同一側
庫邊）做端到端驗證，**20 個測試案例（座標邊界+中點交叉角度/速度/偏移量）
0 個成功**，全部卡在瞄準階段逾時。用 `scripts/debug_ported_aim_
regression.py` 隔離單一案例（`(0.0, -0.9382125)`，`shot_angle=0`）逐步
追查：關節角穩定停在
`shoulder_pitch=1.985 rad`——這正是這個關節的**硬限位**（`CANONICAL_
REST_JOINTS` 的 `shoulder_pitch=1.9 rad`，特意只留了 0.085 rad 餘裕，見
`base_placement_calculator.py` 的 `CANONICAL_REST_JOINTS` 註解）。高架橋
抬高姿態需要的 shoulder_pitch 角度超出這個餘裕，差動 IK 撞到硬限位動不了，
末端永遠差固定的 2.86cm，是**真正的運動學可達性問題，不是數值收斂 bug**
——逾時保護（`ArticulationAPIImpl.MOTION_TIMEOUT_STEPS`）正確攔截，不會
卡死、也不會帶著錯誤姿態去揮桿，但目前這個座標範圍的擊球會可靠地進
`ERROR` 狀態、打不出去。

作為對照，`scripts/scan_elevated_bridge_approach.py` 原本驗證的 25 點網格
（`_CUE_BALL_X_GRID`/`_CUE_BALL_Y_GRID`，涵蓋桌面較大範圍、Y 最小值只到
-1.1）多數案例成功，代表 Kitchen 這個範圍（整個 Y 範圍都落在 -1.24~-0.64
之間，比原研究網格更靠近庫邊）系統性地需要比原研究網格更大的抬高角，
更容易撞到 shoulder_pitch 餘裕不足的問題。

**已知限制，留給後續 issue**：需要重新設計 `CANONICAL_REST_JOINTS`（例如
把 shoulder_pitch 的預設值降低、換取更多抬高時可用的餘裕）或高架橋的
接近幾何（例如縮小 `safe_altitude_margin`、改變 Phase B 的水平/垂直移動
順序、降低所需的傾斜角），才能讓 Kitchen 範圍內的高架橋案例真正可達。

## 參考檔案

- `core/models/action_bounds.py`
- `core/services/break_shot_position_provider.py`
- `core/models/billiard_table.py`
- `core/services/table_ball_set.py`（`table_ball_set.py` 內 `TableBallSet`）
- `core/models/table_robot_manager.py`
- `core/models/barrett_wam_robot.py`
- `core/services/base_placement_calculator.py`
- `extension/billiard_digital_twin/billiard_digital_twin.py`
- `extension/isaac_sim_impl_6_0/stage_api_impl.py`
- `extension/isaac_sim_impl_6_0/articulation_api_impl.py`
- `scripts/scan_rail_collisions.py`（#233 初次碰撞掃描）
- `scripts/scan_elevated_bridge_approach.py`（#233 高架橋抬高姿態公式與驗證）
- `scripts/probe_room_clearance.py`（#233 房間隔間牆碰撞的淨空量測）
- `docs/issue-flat-case-residual-error.md`（#233 衍生的 flat 案例殘留誤差調查）
- `core/services/cue_pose_calculator.py`（#181 正式程式碼，高架橋幾何＋接觸點偏移）
- `core/services/swing_trajectory_calculator.py`（#181 後擺/隨揮/桿尖速度計算）
- `scripts/verify_swing_trajectory.py`（#181 端到端驗證，發現 Kitchen 範圍撞關節限位）
- `scripts/debug_ported_aim_regression.py`（#181 對照除錯：定位 Phase 0 佔位符 bug 與 shoulder_pitch 限位問題）
- `assets/barrett_wam/wam7.urdf`
- `assets/ball_stick.usda`
- `assets/ball_template.usda`（單位換算交叉驗證用）
- `scripts/probe_base_reachability.py`（差動 IK 探測，記錄收斂失敗過程）
- `scripts/probe_canonical_pose.py`（固定姿態手動試誤）
- `scripts/validate_fixed_pose_placement.py`（端到端驗證）
