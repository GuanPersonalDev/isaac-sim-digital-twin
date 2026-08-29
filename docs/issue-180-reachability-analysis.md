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

### 事後修正說明（2026-08-26）

上面「25/25 網格點 0 碰撞」是針對 `_CUE_BALL_X_GRID`（-0.5~0.5）×
`_CUE_BALL_Y_GRID`（-1.1~0.9）這組研究用網格，**不等於**真實 Kitchen
母球擺位範圍（`action_bounds.py` `CUE_BALL_PLACEMENT_X`=±0.606425、
`CUE_BALL_PLACEMENT_Y`=-1.241425~-0.635）。兩者落差：

- X 方向少測了 ±0.606425 這兩個真正的角落（研究網格只到 ±0.5）
- Y 方向的 -0.1／0.4／0.9 三排根本不在 Kitchen 範圍內；Kitchen 最貼近庫邊
  的極限（Y=-1.241425）也從未實測過

當初 #233 關閉留言宣稱「已涵蓋 Kitchen 邊界代表性網格點（含四角與中心）」，
對照上面的數字並不成立。這個落差直到 #181 的端到端驗證（見第十二節，
`scripts/verify_swing_trajectory.py` 對真實 `CUE_BALL_PLACEMENT_X/Y` 測試）
才被抓到——20 個真實 Kitchen 案例 0 個成功，全部卡在 `shoulder_pitch`
關節限位。**上面「25/25 全數無碰撞」的結論只能代表這組研究網格本身無碰撞，
不能直接當成真實 Kitchen 邊界已驗證過。**

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

## 十三、重新設計 CANONICAL_REST_JOINTS 的嘗試：原假設被推翻，真正根因是姿態/軌跡設計 bug（2026-08-27）

### 起點：第十二節的假設

第十二節推測根因是 `shoulder_pitch` 餘裕不足（0.085 rad），建議重新設計
`CANONICAL_REST_JOINTS` 換取更多餘裕。本節記錄實際動手驗證這個假設的過程
——**結論是這個假設不成立**，`shoulder_pitch` 在絕大多數測試中幾乎沒有
主動移動，真正卡住的是完全不同的三個獨立問題，全部已修正，過程中順便發現
一個新的、還沒解決的獨立限制（後擺可達性，見本節最後）。

### 用 `scripts/search_canonical_pose_candidates.py` 網格搜尋 `(shoulder_pitch, elbow_pitch)`

新增這支腳本對 20 組候選跑 flat 合法性 + 高架橋餘裕實測。**第一個關鍵發現**：
把 `shoulder_pitch` 從 1.9 降到 1.1（餘裕從 0.085 rad 拉到 0.7683 rad），
Kitchen 兩個代表案例（9.91°／29.61° 抬高）**全部逾時**，且 `shoulder_pitch`
從未真正逼近自己的新限位——代表 `shoulder_pitch` 餘裕根本不是瓶頸。

### 真正根因 1：Phase A「先轉指向正上方」逼死 wrist_yaw/wrist_pitch

加逐 waypoint 診斷後發現：卡在原本的 Phase A（原地轉向朝正上方，一個接近
90° 的大幅重定向），`wrist_yaw`（卡在上限 1.25）、`wrist_pitch`（卡在
限位）會被逼死，跟 `shoulder_pitch`/`elbow_pitch` 完全無關。

**修正**：把 `cue_pose_calculator.compute_elevated_bridge_waypoints()` 的
階段順序從「A 轉正上方→B 平移→C1 轉最終姿態→C2 下降」改成「B1 保持目前
姿態原地爬升→B2 保持目前姿態平移→C1 原地轉最終姿態→C2 下降」，把轉向動作
從 90° 的「轉到正上方」換成通常只有 5°~30° 的「直接轉到最終傾斜姿態」。
新增 `current_orientation` 參數（呼叫端提供，見根因 3）。

### 真正根因 2：C1 單一大跳轉向讓手臂本體掃過庫邊/袋口

修正根因 1 後，Kitchen 正中心案例改撞 `Cushion_Head`/`Pocket_HeadLeft`——
不是桿頭撞到，是手臂本體（`shoulder_yaw`/`elbow_pitch` 沿路劇烈擺盪）在
單一大跳的 C1 轉向過程中掃過去的。

**修正**：C1 從單一 waypoint 改成用 NLERP（見 `cue_pose_calculator._nlerp_quat()`）
拆成 `rotate_steps`（預設 8）個中繼姿態，讓差動 IK 不需要走極端關節配置。

### 真正根因 3（最關鍵）：Phase 0 結束後的姿態分析佔位符完全錯誤

即使修正根因 1、2，用跟正式程式碼一模一樣的 `preceding_joint_targets` 打包
呼叫方式測試，Kitchen 案例依然失敗（`shoulder_pitch` 卡在 1.985）；但用
「手動兩步驟，等固定 200 步後直接讀真實姿態」的方式測試同一個案例卻成功。
追查發現：正式程式碼原本假設 Phase 0（`base_yaw=0` 時的 `CANONICAL_REST_
JOINTS`）結束後，腕部朝向是單位四元數 `[1,0,0,0]`（`_shortest_arc_quat
([0,1,0],[0,1,0])` 的計算結果）。**這個假設是錯的**：`_shortest_arc_quat`
構造的是「最短弧」旋轉，對繞目標軸的 roll 分量完全沒有約束；實際量到的
姿態是 `[0.00006, 0.68216, 0.73120, -0.00017]`，跟單位四元數差了將近
**180°**。高架橋 B1/B2 階段（應該只是純位置爬升/平移，姿態不變）因此被迫
在背景多做一個接近 180° 的意外轉向去「修正」一個根本不存在的姿態誤差，
把 `shoulder_pitch` 逼到硬限位。

**修正**：`base_placement_calculator.py` 新增 `CANONICAL_FLAT_ORIENTATION =
(0.0, 0.68216, 0.73120, 0.0)` 常數（跟 `_LOCAL_TIP_RADIUS`/`_LOCAL_TIP_
HEIGHT` 一樣是實測值，不是理論推導——改資產或關節角度時必須重新量測），
取代 `table_orchestrator.py`／`scripts/verify_swing_trajectory.py`／
`scripts/debug_ported_aim_regression.py` 裡原本的 `[1.0,0.0,0.0,0.0]`
佔位符。**這是把 `verify_swing_trajectory.py` 的 `aim_timeout` 從 17/20
壓到 0/20 的關鍵修正。**

### 真正根因 4（獨立、跟高架橋無關）：`verify_swing_trajectory.py` 自己的量測 bug

上面三個根因修正後，STRIKE 階段回報位置誤差 ~1.4 公尺、方向誤差 ~90°——
追查發現這是驗證腳本自己的 bug，不是正式程式碼的問題：`_run_strike()` 把
`get_end_effector_position()`（腕部位置）直接當桿尖位置去跟母球比對距離，
漏了 `CUE_STICK_GRIP_TO_TIP=1.35m` 的偏移量。已修正（加回 `direction_unit
* CUE_STICK_GRIP_TO_TIP` 偏移）。

### 順帶解決：C1 轉向撞庫邊沒有簡單公式，改用離線查表

roll（球桿繞自身軸的閃避自由度）能解決 C1 撞庫邊的問題，但對 9 個 Kitchen
代表點掃描後發現**沒有簡單規則**（同一個 X 在不同 Y 需要的 roll 不一致，
例如 `(0,-0.9382125)` 需要 15°、`(0,-0.7)` 需要 60°、`(0.606425,-1.15)`
需要 45°）。改用 `cue_pose_calculator.lookup_roll_rad()`：離線掃描出的
9 點最近鄰查表，已接進 `_execute_aim()`/`_execute_strike()`（兩邊查同一個
函式，保證瞄準跟擊球用同一個 roll）。

### 目前驗收結果（`scripts/verify_swing_trajectory.py`，20 案例）

⚠️ **這一節的數字已被第十四節推翻，留著只是記錄當時（用了有瑕疵的測試方法）
看到的表面現象，實際情況遠比這裡描述的嚴峻，請直接看第十四節。**

- ~~**AIM**：0 逾時（原本 17/20）、5/20 完全成功、12/20 撞庫邊、3/20 幾何無解~~
- ~~**STRIKE：0/20 成功**~~

## 十四、STRIKE 0/20 的深入調查：測試方法本身有假陽性，真正根因是更深層的多關節可達性問題（2026-08-27）

### 起點

依使用者要求調查 STRIKE 0/20 失敗的原因。先用 `scripts/search_backswing_
distance.py`（新增）對已知 AIM 成功的 Kitchen 代表案例測試不同的
`backswing_distance`，本以為問題就是第十三節記錄的「後擺距離超出可達
範圍」，過程中卻先挖出一個影響範圍更大的問題。

### 關鍵發現：`_AIM_MAX_STEPS`/`_STRIKE_MAX_STEPS` 太短，導致大量假陽性

`verify_swing_trajectory.py`（含第十三節所有驗收數字）用的
`_AIM_MAX_STEPS=1200`。但高架橋序列有 11~27 個 waypoint（B1+B2+`rotate_
steps`×C1+C2），每個 waypoint 最壞情況要跑到自己的
`ArticulationAPIImpl.MOTION_TIMEOUT_STEPS=1000` 才會真正標記逾時
（`did_last_motion_timeout()=True`）。如果外層測試迴圈的步數預算比這個
「最壞情況下才會揭曉真相」的時間點還短，迴圈會在**逾時真正發生之前**就
把 `is_motion_complete()` 仍是 `False`、`did_last_motion_timeout()` 也還是
`False` 的中間狀態當成「沒逾時＝成功」回傳——這正是第十三節「AIM 0 逾時、
5/20 成功」的成因：**不是真的成功，是測試預算不夠長，還沒等到失敗發生**。

用 `search_backswing_distance.py` 把預算拉大到 `_AIM_MAX_STEPS=4000` 重測
第十三節記錄「成功」的兩個代表案例，兩個都在更晚的 waypoint 卡住並真正
逾時。`verify_swing_trajectory.py` 的 `_AIM_MAX_STEPS`/`_STRIKE_MAX_STEPS`
已修正為 4000／2500（保留在正式驗證腳本裡，這是這次調查最重要、最確定的
產出——**任何未來要驗證這條路徑的人，必須用夠大的步數預算，否則會重複
踩到同一個假陽性**）。

### 用足夠的預算重新誠實驗證，發現真正的根因：多關節同時撞死限位，不是單一常數的問題

對兩個代表案例（Kitchen 正中心 `(0,-0.9382125)`、最嚴苛角落
`(0.606425,-0.635)`）分別追查卡住的位置：

1. **中繼 NLERP 轉向 waypoint 卡住**：`orient_err` 穩定停在遠超過
   `ORIENTATION_TOLERANCE(0.02)` 的值（例如 0.32 rad≈18°），對應的關節
   （`shoulder_pitch`、`wrist_yaw`、`wrist_pitch`——依案例與 roll 不同而
   不同）卡在自己的硬限位。把 `rotate_steps` 從 8 加到 24（轉向切更細）
   能讓序列往後推進更遠，但終究還是在某個中繼點或最終 waypoint 撞死，
   不是「切更細」就能根治。
2. **最終接觸姿態（C2）本身撞死三個關節**：對最嚴苛角落案例，序列真的
   走到最後一個 waypoint（C2）時，`pos_err≈0.10m`、`orient_err≈0.59 rad
   （≈34°）`，`shoulder_pitch`（1.985）、`wrist_pitch`（-1.5707）、
   `palm_yaw`（3.0）**三個關節同時卡在硬限位**——代表這個案例真正的
   接觸姿態，用目前的 `CANONICAL_REST_JOINTS` 起點，物理上就是不可達。
3. **換 roll 值只是把問題從一個關節轉移到另一個關節**：對同一個角落案例
   試了 `roll=0°`（卡 shoulder_pitch/wrist_pitch/palm_yaw）、`roll=45°`
   （卡 wrist_yaw，在序列更早的地方就卡住）——沒有一個測過的 roll 值能
   讓這個案例真正收斂。
4. **放寬 `POSITION_TOLERANCE`/`ORIENTATION_TOLERANCE` 只能解決「差一點點」
   的案例，解決不了「差很多」的案例**：Kitchen 正中心案例的某個中繼
   waypoint 只差一點點（`orient_err=0.023` vs 門檻 `0.02`），放寬到 0.035
   後這個點真的過了，但緊接著在下一個 waypoint（原本被這個更早的卡點
   掩蓋住）又卡住，而且差距更大——這證實放寬容許值只是「延後暴露問題」，
   不是解法，兩個實驗值已還原（`POSITION_TOLERANCE=0.005`、
   `ORIENTATION_TOLERANCE=0.02`，不要沿用診斷時暫時調過的值）。

### 結論（初版）

STRIKE（以及很大一部分 AIM）目前打不出去，根因不是後擺距離、不是 roll
查表覆蓋率、也不是任何單一容許值——是 `CANONICAL_REST_JOINTS` 這組固定
起始姿態，對高架橋案例需要的最終接觸姿態而言，**在 `shoulder_pitch`、
`wrist_pitch`、`wrist_yaw`、`palm_yaw` 這幾個關節上的餘裕普遍不足**，這正是
最早（第十三節開頭）被推翻的假設所懷疑的方向，只是當初只看了
`shoulder_pitch`/`elbow_pitch` 兩個關節，範圍不夠全面，而且用來驗證的
測試本身有假陽性，才會一路得出「已經解決」的錯誤印象。

### 後續窮舉搜尋：手動試誤已經證實走不通（2026-08-27 續）

依使用者要求「繼續解決」，在上面的結論之後又做了一輪更systematic的嘗試，
每次都用 `_hard_reset_joints()`／`set_dof_positions()` 瞬間重置關節+速度
（修正掉更早幾輪 roll 掃描沒重置、被前一次測試殘留姿態污染結果的問題），
每組都用誠實的收斂判定（`_AIM_MAX_STEPS` 足夠大＋沒有靠不住的假陽性）：

1. **13 個 roll 值密集掃描**（-90°~90°，每 15°一個，含之前測過的 0°/45°）：
   對最嚴苛角落案例 `(0.606425,-0.635)` **全部真正收斂失敗**，換 roll 只是
   把卡住的關節從一個換到另一個（roll=0 卡 shoulder_pitch/wrist_pitch/
   palm_yaw，roll=45 卡 wrist_yaw，其餘 11 個 roll 值也都各自卡在不同
   關節）。
2. **4 組手動挑選的「反向」安全起點候選**（把 `wrist_pitch`/`palm_yaw`
   起始值反號、或搭配降低 `shoulder_pitch`）：`wrist_pitch` 反號那組
   推進到 waypoint 26/26（27 個 waypoint 中的最後一個，也就是真正的最終
   接觸姿態），但在那裡真正逾時——這是目前為止最接近成功的結果，但終究
   沒有真正收斂。換到抬高需求較小的 Kitchen 正中心案例測同一組候選，反而
   卡在更早的 waypoint 18（`wrist_yaw` 撞上限），沒有比較好。
3. **`wrist_pitch`×`palm_yaw` 3×3=9 組系統化網格**（固定 `shoulder_pitch`/
   `elbow_pitch`=baseline、roll=0，Kitchen 正中心案例）：**9 組全部失敗**。
4. **`shoulder_pitch`/`elbow_pitch`/`wrist_pitch`/`palm_yaw` 4 個關節一起
   調整的 4 組組合**（含「`shoulder_pitch` 最低+`elbow_pitch` 最高」這個
   最早被推翻、這次搭配 wrist 起點再試一次的方向）：**4 組全部失敗**，
   其中把 `shoulder_pitch` 壓到 1.1、`elbow_pitch` 拉到 2.7 的兩組甚至在
   waypoint 1（B2，單純水平平移、都還沒開始轉向）就卡住——這麼極端的關節
   組合本身的可用工作空間反而更差，不是更好。

**累計超過 30 組獨立、誠實驗證過的候選（roll×安全起點的各種組合）沒有一組
能讓即使是抬高需求最小的 Kitchen 案例真正收斂。**

### 結論（更新）

手動/半系統化的試誤搜尋已經證實走不通——這不是「還沒找到對的參數」，而是
目前這套「單一共用的固定安全起點 + differential IK 原地轉向到最終傾斜
姿態」的高架橋架構，本身的可達工作空間可能就不足以涵蓋 Kitchen 範圍需要
的傾角/位置組合。真正可靠的下一步不是繼續手動猜（已經證明沒有效益），
是下列兩條路線之一，**這兩條都需要新的設計決策，不是本次調查能單方面
決定的**：

1. **投入一套真正的自動化數值 IK 求解工具**（不跑物理模擬、直接用 URDF
   的運動學鏈算 Jacobian 做梯度下降/數值 IK），能在幾秒內測試成百上千組
   候選，取代現在每組要跑 1-2 分鐘物理模擬的手動試誤——這樣才有機會在
   合理時間內做到真正涵蓋全部關節維度的系統化搜尋。是一個獨立的、有一定
   份量的工程投入。
2. **重新檢視高架橋技術本身的幾何設計**，而不是繼續在「同一個固定起點+
   原地轉向」這個架構內找參數。例如：用更短的握把到桿尖距離
   （`CUE_STICK_GRIP_TO_TIP`，目前 1.35m）處理需要抬高的案例、改變桌邊
   接近角度、或允許基座在抬高案例採用非固定的 Phase 0 起點（每個案例算
   自己專屬的安全起點，而不是全部案例共用同一個 `CANONICAL_REST_JOINTS`）。

在其中一條路線真正執行之前，Kitchen 範圍的高架橋擊球應該視為**已知不可達
的既有限制**，不建議繼續嘗試手動調參數。

### 參考檔案（本節新增）

- `scripts/search_backswing_distance.py`（第十四節：後擺距離掃描、AIM/
  STRIKE 誠實收斂驗證、roll 掃描、逐 waypoint 位置/姿態誤差與關節數值
  診斷）

## 十五、數值 IK 工具找到 AIM 真正根因並修復（0/20 → 6/6）；STRIKE 隨揮終點的新根因：控制律結構性穩態誤差（2026-08-28）

第十四節結論「累計 30+ 組候選全部失敗，手動試誤已經走不通」之後，依照
提出的兩條路線之一——**寫一套不跑物理模擬的數值 IK 工具**——實作了
`scripts/wam7_kinematics.py`：直接照 `assets/barrett_wam/wam7.urdf` 的
`<joint>` 標籤（全部 7 個 revolute joint 都繞各自局部座標系的 Z 軸）手刻
4×4 齊次變換鏈的正向運動學（FK），加上有限差分 Jacobian 的阻尼最小二乘法
（DLS）數值 IK，用 `_validate_against_known_constants()` 對照真實量測的
`_LOCAL_TIP_RADIUS`/`_LOCAL_TIP_HEIGHT`/`CANONICAL_FLAT_ORIENTATION` 驗證
FK 誤差 <0.3mm 才採信。這支工具讓「測試一組候選」從物理模擬的 1-2 分鐘
降到毫秒級，才有可能做真正窮盡的系統化搜尋。

### AIM 根因：`_ROLL_LOOKUP_GRID` 選錯值，不是關節限位/姿態設計問題

用 `scripts/search_ik_reachability.py` 對 Kitchen 代表案例做 400 組隨機
起點的數值 IK 掃描後發現：**目標姿態本身在關節限位內是可達的**（找得到
收斂解），但用舊的 `_ROLL_LOOKUP_GRID` 查表值（0°/15°/45°/60° 這種小
角度）時 0/400 收斂——問題出在 roll 選錯了。改用
`scripts/build_roll_lookup_table.py`／`scripts/search_roll_for_full_swing.py`
系統化掃描 roll∈[-180°,180°] 後發現：

1. **正確的 roll 落在完全不同的範圍**（-180°~165°附近，收斂率 20-50%，
   解的盆地相當寬，不是知識邊緣的窄解）。
2. **關鍵發現：roll 只跟 `cue_ball_y` 有關，跟 `cue_ball_x` 無關**——
   `wam_base_yaw_joint` 會吸收 X 方向的差異，同一個 Y、三個不同 X 算出來
   的最佳 roll 完全一致（見 `scripts/search_roll_for_full_swing.py` 實測
   輸出）。這讓查表從「9 點各自獨立」簡化成「只跟 Y 座標所在的行有關」。
3. 用 `scripts/search_roll_for_full_swing.py`（**局部延續 IK**：AIM 解
   當後擺起點、後擺解當隨揮終點起點，模仿真實 `ArticulationAPIImpl.
   _step_motion()` 差動 IK 不會跳關節分支的行為，不是隨機起點）確認整條
   AIM→後擺→隨揮終點軌跡在同一分支內都收斂、且沒有任何關節被逼到限位。

新查表（`core/services/cue_pose_calculator.py` `_ROLL_LOOKUP_GRID`，已
更新並附完整推導註解）：`y=-1.15→roll=-180°`、`y=-0.9382125→165°`、
`y=-0.7→150°`、`y=-0.635→150°`（`x` 不影響）。

用 `scripts/verify_new_roll_table.py` 對真實 Isaac Sim 物理模擬重新驗證：
**AIM 從 0/20（第十四節基準）修到 6/6 真正收斂**（`pos_diff<5mm`、
`orient_diff<0.02rad`，非假陽性——用的是第十四節已修正的
`_AIM_MAX_STEPS=4000` 誠實逾時偵測）。這是本次調查最終確認的 AIM 根因與
修復。

### STRIKE 新根因：隨揮終點（follow-through waypoint）有結構性穩態誤差，跟關節限位無關

AIM 修好後 STRIKE 仍然 0/6 逾時。用 `scripts/diagnose_strike_followthrough.py`
逐步印出位置誤差／關節餘裕／`_compute_pose_tracking_twist()` 與
`_feedforward_twist` 的實際數值後找到精確機制：

`compute_swing_waypoints()` 的隨揮終點（follow-through waypoint）是**靜態
目標位置 + 非零目標速度**（`linear_velocity = required_tip_speed *
direction`，全速案例 ≈1.51 m/s）。`ArticulationAPIImpl._step_motion()` 的
控制律是 `twist = P控制器(POSITION_GAIN=5.0 × 位置誤差, clip 2.0) +
feedforward`——**P 項只對自己單獨 clip，跟 feedforward 相加之後不會再
clip**，且更關鍵的是：**這個相加後的和只有在位置誤差 ×
POSITION_GAIN 剛好抵銷 feedforward 時才會趨近 0**，也就是系統會自然收斂
到一個穩態誤差：

```
steady_state_error ≈ |feedforward_velocity| / POSITION_GAIN
```

實測驗證：全速案例（`required_tip_speed≈1.511 m/s`）穩態誤差理論值
1.511/5.0≈0.302m，實測穩定在 pos_err≈0.305m；额外測試最低速案例
（`cue_ball_speed=0.65` → `required_tip_speed≈0.492 m/s`）理論值
0.492/5.0≈0.098m，實測穩定在 pos_err≈0.098m——**兩個獨立案例都精確吻合
公式，不是巧合**。`POSITION_TOLERANCE=0.005m` 在這個結構下**永遠不可能
達成**（除非 `required_tip_speed` 小到 <0.025 m/s，等同不揮桿），逾時是
必然結果，跟關節限位、roll 選擇、`CANONICAL_REST_JOINTS` 都無關（低速
測試中途出現的 `wrist_yaw margin≈0` 是穩態平衡點附近的巧合伴生現象，
不是成因——關節餘裕後來持續回升，但位置誤差仍凍結在穩態值不動）。

**這是一個控制律架構層級的問題**，影響 `ArticulationAPIImpl`（所有呼叫端
共用），修法需要新的設計決策，不是本次調查能單方面決定：

1. **改變隨揮終點的完成判定**：帶有非零 feedforward 速度的 waypoint
   不應該用「靜態位置收斂」判定完成，改成「桿尖以正確方向通過目標點
   附近」或改成固定步數（時間到）就視為完成——影響
   `_is_current_target_converged()`／`is_motion_complete()` 的語意，
   波及所有呼叫端。
2. **改變控制律本身**：相加後的 twist 再做一次 clip、或幫「有 feedforward
   的 waypoint」加大 `POSITION_TOLERANCE`／改用不同的收斂判定公式——同樣
   影響共用類別。
3. **改變 `compute_swing_waypoints()` 的設計**：不要求隨揮終點是「靜態點+
   固定速度」，改成一系列多個遞增位置的中繼 waypoint（真正的軌跡追蹤，
   逐點都是零 feedforward 的靜態收斂目標）——影響範圍侷限在
   `swing_trajectory_calculator.py`，波及面較小，但終點軌跡的擬真度／
   球桿實際觸球瞬間速度需要重新驗證。

在其中一條路線確定之前，STRIKE 的隨揮終點應視為**已知結構性限制**，不是
可以靠調整 roll／`CANONICAL_REST_JOINTS`／後擺距離參數解決的問題。

### 參考檔案（本節新增）

- `scripts/wam7_kinematics.py`（純數值 FK/IK，已用已知常數驗證）
- `scripts/search_ik_reachability.py`（隨機起點可達性掃描＋roll 掃描）
- `scripts/build_roll_lookup_table.py`（3×3 Kitchen 網格 roll 查表重建）
- `scripts/search_roll_for_full_swing.py`（局部延續 IK：AIM→後擺→隨揮
  終點單一分支驗證，發現 roll 只跟 Y 有關）
- `scripts/verify_new_roll_table.py`（新查表的真實物理模擬驗證，AIM 6/6）
- `scripts/diagnose_strike_followthrough.py`（STRIKE 隨揮終點逐步 twist/
  關節診斷，定位穩態誤差公式）
- `scripts/search_backswing_ik.py`（後擺距離數值 IK 可達性測試）

### ⚠️ 重要修正（2026-08-28 同日）：完整 20 案例驗收測試揭露兩個更嚴重的問題

用 `scripts/verify_swing_trajectory.py`（正式 20 案例驗收，跟本節前段用的
6 點手選案例不同——手選案例只涵蓋 Y 座標三個代表值，這支才是完整
`action_bounds` 網格＋角度/速度/偏移量變化）重新驗證後，發現上面的
「AIM 6/6、STRIKE 6/6」是**過度樂觀的結論**，原因是兩支自寫的驗證腳本
（`verify_new_roll_table.py`／`diagnose_strike_followthrough.py`）都沒有
接上真正的碰撞回報，也沒有量測真實桿尖擊球速度：

**問題一：新 roll 查表在完整網格上大多數案例是 COLLISION，不是逾時。**
`scripts/build_roll_lookup_table.py`／`search_roll_for_full_swing.py` 的
搜尋目標函式只看「數值 IK 在關節限位內能不能收斂」，**完全沒有建模手臂
本體碰撞**（C1 轉向時手臂本體可能掃過庫邊/袋口，這正是 roll 這個自由度
原本要解決的問題，見 `_ROLL_LOOKUP_GRID` 上方註解）。20 案例網格顯示
大部分候選點在真實物理模擬中造成碰撞。6 點手選案例之所以看起來全過，
是因為那 6 點剛好落在無碰撞的子集、加上自寫驗證腳本沒開碰撞偵測，兩個
因素疊加造成的假陽性，不是新查表真的解決了問題。

**問題二：「放寬 `_is_current_target_converged()` 位置容許值」的修法是
錯的，已還原。** 用 `verify_swing_trajectory.py` 的真實桿尖速度量測（
`actual_speed` vs `required_tip_speed`）驗證那個修法時發現：放寬後的容許
值剛好等於 P 控制器+feedforward 疊加控制律的**穩態平衡點**，這個平衡點
的物理意義是合力趨近 0、關節速度也趨近 0——系統會在那裡「宣告完成」，
但桿尖當下幾乎靜止（`speed_error_ratio≈0.98`，只有應有速度的 ~2%），
等於沒有真正揮桿。這個修法已在 `extension/isaac_sim_impl_6_0/
articulation_api_impl.py` 還原，`_is_current_target_converged()` 恢復
原本行為。STRIKE 隨揮終點的正確修法**不能只是放寬位置容許值**，需要
上一段列的三個選項中會產生「桿尖真的在移動」結果的那種（例如把隨揮
拆成多段零 feedforward 的位置 waypoint，靠 P 控制器自然的高增益追蹤
產生連續高速移動；或改成不等待收斂、改用足夠長的固定步數讓桿尖自然
通過目標區域時仍保有高速）——這需要新的設計決策，且要用「真實桿尖
速度」而非「位置有沒有收斂」當驗收標準，避免重蹈本節的覆轍。

**目前唯一確定、經得起完整 20 案例驗收考驗的成果**：
`docs/issue-180-reachability-analysis.md` 前段記錄的三個 bug 修復（Phase A
重排、C1 NLERP 細分、`CANONICAL_FLAT_ORIENTATION` 常數）、
`scripts/wam7_kinematics.py` 這套數值 FK/IK 工具本身（FK 已驗證準確、
可以繼續拿來做「排除掉不可能的候選」的快速篩選，但**搜尋出來的候選必須
回真實物理模擬做碰撞+速度雙重驗證，不能只信任數值 IK 的收斂結果**）、
以及本節對 STRIKE 隨揮終點問題根因（結構性穩態誤差公式）的精確定位——
這個診斷本身是對的，只是「放寬容許值」這個特定修法被證明無效。
`_ROLL_LOOKUP_GRID` 的新表**尚未證實在完整網格上無碰撞**，下一步需要
針對碰撞問題重新搜尋（roll 候選需要同時滿足「數值 IK 可達」+「物理模擬
無碰撞」兩個條件，後者目前只能靠真實物理模擬逐點驗證，無法用純數值方法
加速）。

## 十六、碰撞感知的 roll 查表修正（AIM 收斂+無碰撞雙重驗證）；STRIKE 揮桿速度的深層運動學限制（2026-08-28 續）

### AIM：roll 查表改成逐點碰撞驗證，核心 6 點網格確認可行

第十五節結尾指出「碰撞沒辦法用純數值方法加速篩選」。實作
`scripts/search_collision_free_roll.py`：對每個候選點，依
`search_roll_for_full_swing.py` 算出的 IK 全程餘裕由高到低嘗試候選，
每個候選用真實 Isaac Sim 物理模擬＋正式的 `enable_contact_reporting`／
`ContactEvent` 機制驗證是否收斂、是否碰撞，取第一個「兩者都成立」的候選。

先對 4 個 Y 值代表點（X=0）測試，全部一次到位（每個 Y 只需試 1-2 個候選）。
但拿去跑完整 `verify_swing_trajectory.py` 20 案例後發現：**碰撞跟世界
座標系裡離哪個庫邊/袋口近有關，不是只看關節構型**——`wam_base_yaw_joint`
確實會讓不同 X 的關節構型完全一致（IK 可達性因此跟 X 無關，第十五節已
驗證），但同一組關節構型在世界座標系裡對應到的絕對位置隨 X 平移，會
掃到不同的庫邊/袋口，所以「同一個 Y、不同 X」常常需要不同的 roll 才能
避開碰撞。改成對 `action_bounds.CUE_BALL_PLACEMENT_X/Y` 的完整 3×3 網格
（扣掉 `y=-1.241425` 這個純幾何無解的邊界列）逐點驗證，6 個點都在 1-2
個候選內找到「IK 收斂+無碰撞」都成立的解，`_ROLL_LOOKUP_GRID` 已更新。
`shot_angle≠0`／`position_offset≠0` 的案例目前仍沿用最近鄰查到的
`shot_angle=0` 候選，尚未針對這些變化量各自搜尋（20 案例網格裡這些
變化量案例目前仍會 COLLISION，是已知、有明確修法路徑但還沒做的缺口）。

### STRIKE：不是 waypoint 設計問題，是真正的運動學速度上限

依你的指示直接嘗試 STRIKE 隨揮終點的修法：先實作
`scripts/prototype_moving_target_strike.py`（「移動目標點」——每個物理步
把目標沿 `direction_unit` 前進 `required_tip_speed × PHYSICS_DT`
（`core/services/rolling_resistance_service.py` 的 `PHYSICS_DT=1/60`），
讓 P 控制器的角色只剩修正微小追蹤誤差，不會像靜態目標點那樣被
feedforward「越過」後反向煞車）。這個修法在**呼叫端**實作（重複呼叫
`move_to_pose()` 更新目標），不需要碰共用的 `ArticulationAPIImpl` 控制律
本身。

用 `scripts/diagnose_ball_impact.py`（直接 `RigidPrim(paths=ball_prim_
path).get_velocities()` 量測母球真實物理速度，不透過任何軟體的完成
判定）驗收後發現：**移動目標點修法一樣沒用**——追蹤誤差隨時間從 2cm
惡化到 11cm，母球最高速只有 0.22 m/s（該有 1.51 m/s）。逐步印出關節
速度後看到 `wrist_yaw`／`wrist_pitch`／`palm_yaw` 等關節持續頂在
±2.0 rad/s（`_dof_limits`）的速度上限。

用線性規劃驗算後找到精確原因：在 `(0.0,-0.635)` 案例、`roll=150°`（原本
查表選中的值）這組關節構型下，**純追求揮桿方向最大平移速度**（不管
姿態）理論上可以到 1.63 m/s（超過所需的 1.51 m/s），但這樣做會產生
6.93 rad/s 的角速度——等於揮桿全程桿身瘋狂亂轉。若改成正確約束
「角速度必須精確為 0」（模擬 `compute_swing_waypoints()` 要求全程桿身
指向不變的設計），沿揮桿方向真正能達到的最大速度只剩 **0.81 m/s**——
只有所需速度的 54%。**這是這個關節構型本身的速度可操作性
（manipulability）上限，不是控制律、waypoint 設計或收斂判定的問題**，
換哪種完成判定或哪種 waypoint 拆法都無法突破。

線性規劃公式：`max (direction_unit · J_linear) @ qdot`，限制式
`J_angular @ qdot = 0`（角速度鎖零）且 `qdot ∈ [-2, 2]^7`
（`_dof_limits`），`J` 用 `wam7_kinematics._numerical_jacobian()` 在
AIM 收斂到的關節構型上算。

**掃過全部 24 個 roll 候選後的結果**（`scripts/search_roll_swing_
capable.py`，`required_tip_speed=1.5116`，`cue_ball_speed` 用
`action_bounds.CUE_BALL_SPEED` 稀疏網格中點 1.995）：

- `y=-0.9382125`（Kitchen 較遠列，3 個 X 都一樣，跟 IK/碰撞一樣是
  X 無關的性質）：**24 個候選裡沒有任何一個能達到 1.51 m/s**，最好的是
  `roll=-45°` 的 1.33 m/s（且該解 IK margin=0.0000，貼著關節限位，是
  否穩健存疑）。
- `y=-0.635`（Kitchen 較近列）：只有 `roll=-60°` 達標（1.53 m/s），
  但同樣 IK margin=0.0000。次佳 `roll=-75°` 是 1.47 m/s，未達標。

margin=0.0000 的候選是我的 `solve_ik()` 從固定起點（`CANONICAL_REST_
JOINTS`）收斂時剛好卡在關節邊界的數值產物，不代表「這個目標唯一的解
一定要頂到限位」——換一組隨機起點也許能找到 margin>0 又保有高揮桿速度
的解，但這需要對每個候選重新做隨機起點搜尋（第十五節 `search_ik_
reachability.py` 的方法），還沒做。

### 結論：STRIKE 高速揮桿目前是真正的運動學限制，需要新的設計決策

這不是能單靠參數搜尋解決的問題。可能的方向（都有實質的設計/範圍取捨，
不建議自行拍板）：

1. **降低 `cue_ball_speed` 的有效上限**（或針對特定 Kitchen 子範圍收緊
   `action_bounds.CUE_BALL_SPEED`）——`required_tip_speed` 隨
   `cue_ball_speed` 線性下降，`y=-0.9382125` 最佳解 1.33 m/s 反推大約
   對應 `cue_ball_speed≈1.76`（原上界 3.3392 的一半左右），影響 RL
   action 空間，需要跟 #183 的既有「clamp 行動空間」決策放在一起考慮。
2. **放寬「揮桿全程姿態不變」的要求**，允許小角度姿態漂移換取平移速度
   ——`swing_trajectory_calculator.py` 的既有設計說明（見該檔案 docstring）
   明確是刻意鎖定姿態的，放寬會改變擊球的物理真實度，需要重新驗證球的
   實際飛行/旋轉行為有沒有受影響。
3. **對每個候選做隨機起點 IK 搜尋**，找 margin>0 又保有高揮桿速度的解
   （见上段）——範圍最小、最有可能是純粹的搜尋不夠深，但也可能徒勞
   （`y=-0.9382125` 24 個角度都不夠，換起點未必能無中生有出額外速度
   餘裕，因為 manipulability 上限主要取決於目標姿態本身的幾何，不是
   起點選擇）。
4. **提高關節速度上限**（`_dof_limits`，來自 `get_dof_max_velocities()`）
   ——如果這是軟體可調的參數（不是實體致動器硬限制），直接放寬可能是
   最一勞永逸的解法，但需要先確認這組限速的來源與是否有物理意義上的
   限制依據，範圍可能波及所有動作的速度上限，不是只影響 STRIKE。

### `move_swing()`：揮桿專用速度最優控制器的實作與發現（同日續）

依使用者指示（選項 3「寫揮桿專用的速度最優控制器」）在
`ArticulationAPIImpl` 新增 `move_swing()`／`_step_swing_motion()`：先用
一般 pose-tracking 收斂到後擺點，再切換成每個 physics tick 用線性規劃
（`scipy.optimize.linprog`）直接求「姿態修正在有限額度內、沿揮桿方向
最大化桿尖速度」的關節速度指令，取代 P控制器+feedforward 那條會產生
結構性穩態誤差的路徑。`core/ports/articulation_api.py` 同步新增抽象
方法。過程中用真實母球物理速度（`RigidPrim.get_velocities()`，不透過
任何軟體完成判定）反覆驗證，抓到三個真實 bug：

1. **等式約束又把角速度鎖回 0**：第一版把姿態修正寫成
   `Jang@qdot == orientation_gain × 目前姿態誤差`，揮桿剛開始姿態誤差
   是 0，這個等式因此一直逼近 0，`max_angular_speed` 給的額度完全沒被
   用上——跟原本 STRIKE 卡住的病徵一樣，只是換了個地方重演。改成
   `restore_bias ± max_angular_speed` 的**不等式箱型約束**才修正。
2. **目標函式用了「腕部」的線性 Jacobian，不是桿尖的**：桿尖在
   `CUE_STICK_GRIP_TO_TIP`（1.35m）之外，角速度會透過剛體速度合成
   `v_tip = v_wrist + ω × tip_offset` 讓桿尖產生遠比腕部本身位移更大的
   側向偏移。第一版線性規劃找到的「腕部方向最優」角速度反而讓桿尖越
   轉越偏，完全沒碰到球。改用 `Jv_tip = Jv - skew(tip_offset) @ Jang`
   （`_skew_matrix()`）才是正確的桿尖速度目標函式。
3. **只優化沿揮桿方向的速度，沒約束側向漂移**：目標函式只看 1D 投影
   進度（`traveled`），線性規劃可以讓投影進度正常推進、同時桿尖在垂直
   方向越飄越遠，兩者不衝突。加一個側向位置回正項（`lateral_gain=5.0`
   的箱型約束，模式跟角速度的 `restore_bias` 一致）才讓桿尖真正貼著
   後擺→揮桿終點這條直線走。

修完三個 bug 後，對最難的 `y=-0.9382125` 案例（roll=-180，24 個角度中
唯一在完全鎖死姿態下都無法達標的那組）實測：關節速度全程 4+ rad/s
（真的在高速移動）、桿尖到球最近距離 **12.6mm**（遠小於球半徑
28.575mm，桿尖確實深入球體範圍）、姿態誤差控制在 <8°（相對「貼近人類
擊球姿態」的要求是合理範圍）。開啟正式的
`enable_contact_reporting`／`ContactEvent` 機制後**確認 PhysX 真的有
偵測到 `CueStick/Cylinder <-> Ball` 的接觸事件**——證實桿尖幾何計算是
對的，桿子真的碰到球了。

**但這個接觸事件的 `impulse=0.0`**——母球全程沒有獲得任何速度（維持
AIM 階段殘留的緩慢滾動衰減，`max_ball_speed=0.2703m/s` 全程不變，
`required_tip_speed=1.5116m/s`）。這是一個新的、更深層的問題：幾何/
運動學層面（桿尖有沒有碰到球）已經解決，卡住的是物理引擎的碰撞響應
（碰到了為什麼沒有力）。可能相關的線索：整個 session 每次啟動模擬都會
印出一條警告「Detected an articulation ... with more than 4 velocity
iterations being added to a TGS scene」，尚未深入調查是否與此有關；也
可能是 `physxCollision:contactOffset=0.005`／`restOffset=0` 這類 PhysX
碰撞邊界參數，或桿尖掠過球體的相對速度/接觸持續時間太短，solver 來不及
累積衝量。這是 PhysX 求解器/碰撞參數層級的問題，不是控制器邏輯或運動
學計算的問題，需要另外一輪聚焦在物理引擎設定的調查。

### 參考檔案（本節新增）

- `scripts/search_collision_free_roll.py`（碰撞感知 roll 搜尋，已改成
  完整 X×Y 網格逐點驗證）
- `scripts/prototype_moving_target_strike.py`（移動目標點修法原型，
  已證實無效）
- `scripts/diagnose_ball_impact.py`（直接量測母球真實物理速度，不透過
  軟體完成判定）
- `scripts/search_roll_swing_capable.py`（三條件 roll 搜尋：AIM 可達+
  揮桿速度線性規劃+IK margin 排序）
- `scripts/diagnose_move_swing.py`（`move_swing()` 的逐步驗證腳本：
  桿尖到球距離、關節速度、球桿剛體位置對照、碰撞事件回報）
- `extension/isaac_sim_impl_6_0/articulation_api_impl.py`
  `move_swing()`／`_step_swing_motion()`／`_skew_matrix()`（新增的
  揮桿專用速度最優控制器本體）
- `core/ports/articulation_api.py`（`move_swing()` 抽象方法定義）

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
- `scripts/search_canonical_pose_candidates.py`（第十三節：`(shoulder_pitch, elbow_pitch)` 網格搜尋、B1/B2/C1/C2 waypoint 診斷、roll 候選掃描）
- `assets/barrett_wam/wam7.urdf`
- `assets/ball_stick.usda`
- `assets/ball_template.usda`（單位換算交叉驗證用）
- `scripts/probe_base_reachability.py`（差動 IK 探測，記錄收斂失敗過程）
- `scripts/probe_canonical_pose.py`（固定姿態手動試誤）
- `scripts/validate_fixed_pose_placement.py`（端到端驗證）
