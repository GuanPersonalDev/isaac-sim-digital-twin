# #180 可達性掃描前置分析：基座位置與球桿握把點

**狀態**：分析完成，待實際 orientation-constrained IK 掃描驗證
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

## 參考檔案

- `core/models/action_bounds.py`
- `core/services/break_shot_position_provider.py`
- `core/models/billiard_table.py`
- `core/services/table_ball_set.py`（`table_ball_set.py` 內 `TableBallSet`）
- `core/models/table_robot_manager.py`
- `core/models/barrett_wam_robot.py`
- `extension/billiard_digital_twin/billiard_digital_twin.py`
- `extension/isaac_sim_impl_6_0/stage_api_impl.py`
- `assets/barrett_wam/wam7.urdf`
- `assets/ball_stick.usd`
- `assets/ball_template.usda`（單位換算交叉驗證用）
