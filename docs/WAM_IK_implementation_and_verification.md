# WAM/機械手臂 IK 與擊球軌跡：背景知識、實作步驟、驗收方法

> 本文件整理從 WAM 傳動結構、6D 姿態概念，到 RMPflow 軌跡規劃分工、
> 最終 FK+Jacobian 驗收方法的完整討論脈絡，供後續開發與 Claude Code 交接使用。

---

## 1. 背景知識

### 1.1 傳動方式：纜線驅動 / 齒輪驅動 / 「直驅」

| 傳動方式 | 說明 | WAM 對應 |
|---|---|---|
| 纜線驅動（Cable-driven） | 馬達透過鋼纜＋滑輪傳動到關節，近乎零背隙，馬達可放置在遠端減輕手臂慣量 | J1–J6 |
| 齒輪驅動（Geared） | 齒輪組傳動，結構簡單但通常有背隙 | J7（唯一齒輪軸） |
| 「直驅」（WAM 文件用法） | **不是**傳統機器人學的 direct-drive motor（零減速比），而是「1 顆馬達單獨透過纜線控制 1 個關節」，與差動的「2 馬達共控 2 關節」相對比 | M1→J1、M4→J4 |

> 注意：真正的 Direct Drive（馬達轉子直接接關節輸出軸、無任何減速機構）在 WAM 上並不存在，
> 所有軸都是纜線傳動，差別只在「單軸」vs「差動耦合」。

### 1.2 纜線差動（Cable Differential）原理

用兩顆馬達的**轉速和／轉速差**分別驅動兩個自由度：

```
J2（pitch） ∝ θ_M2 + θ_M3   （同向轉動 → J2 動，J3 不動）
J3（roll）  ∝ θ_M2 − θ_M3   （反向轉動 → J3 動，J2 不動）
```

- 優點：兩馬達合力分擔負載、馬達集中在肩部基座減輕遠端慣量、維持零背隙
- 實務影響：若要直接控制馬達層（而非關節空間指令），需要用轉換矩陣把 J2/J3 換算成 θ_M2/θ_M3，精確係數以 `wam.conf` 或 libbarrett 原始碼為準，不可憑記憶手推
- 若透過高階 API（送關節角度指令）操作，這層轉換通常已由驅動層處理，不需自行計算

### 1.3 6D 目標姿態的意義

剛體姿態（pose）＝ **SE(3)**，共 6 個自由度：

- 3 個平移自由度：位置 X, Y, Z
- 3 個旋轉自由度：朝向（roll / pitch / yaw，或四元數）

**應用上的簡化**：若末端工具（球桿）繞自身軸線的自轉不影響任務結果（擊球方向與速度不受桿身自轉影響），則實際只需約束 5 維，IK 會多出 1 個冗餘自由度可自由選擇（例如肘部 swivel 角度）。

### 1.4 硬體限制（影響可行域與安全邊界）

- 7-DOF WAM 最大負載：3 kg（含加速負載）
- 控制頻率：預設 500 Hz，可調至 1 kHz
- 安全系統速度閾值：需留意軌跡規劃的目標速度別觸發安全斷電
- 底層為 torque-controlled，位置/速度控制是上層疊加的 PID

---

## 2. 實作步驟

### 2.1 姿態與速度的正確輸入分工

| | 位置 | 朝向 | 速度 |
|---|---|---|---|
| 後擺姿態 | ✓ | ✓ | 0 |
| 擊球姿態 | ✓ | ✓ | 目標值（方向＋大小，來自 RL policy 的 6D action 轉換） |

**關鍵澄清（常見誤解）**：
- 兩個姿態都需要「完整 6D pose」，不是「一個只給位置、一個只給朝向」
- 速度不是姿態的一部分，而是餵給軌跡規劃器的**邊界條件**
- 關節角度不是姿態的輸入，而是 IK 求解或 RMPflow 的**即時回饋/種子**，作用是維持冗餘度分支一致性，避免軌跡中途跳變

### 2.2 兩條可能的技術路線

**路線 A：libbarrett 原生 API（若直接操作實體 WAM）**

```cpp
jp_type backswing_jp;   // 7 個 rad
jp_type contact_jp;

wam.moveTo(backswing_jp, true, velocity, acceleration);
wam.moveTo(contact_jp, true, velocity_target, acceleration_target);
```

- `moveTo` 支援 `jp_type`（關節空間）、`cp_type`（笛卡爾位置）、`Quaterniond`（朝向）、`pose_type`（完整 6D）多種重載輸入
- 限制：純內插方式無法精確控制「接觸瞬間的速度向量」，若需要精確速度邊界條件，需改用路線 B 的軌跡規劃邏輯

**路線 B：RMPflow / cuMotion（Isaac 端，反應式笛卡爾控制器）—— 目前採用的架構**

```
你負責計算（上游、離線）：
  後擺姿態(位置+朝向+速度=0)
  擊球姿態(位置+朝向+速度=目標)
        ↓
RMPflow 負責（下游、線上即時）：
  持續讀取當下關節角度（回饋）
  對 Cartesian 目標即時解 IK
  產生滿足「位置+朝向+速度」邊界條件的關節軌跡
  維持冗餘度分支連續性（不需手動管理 IK 種子）
```

- 這是持續追蹤移動中的 Cartesian 目標（含速度項）的閉迴路架構，非「解一次 IK 再用軌跡規劃器連起來」的開迴路兩階段做法
- 對應現有規劃：IK 交給 Isaac Lab、RMPflow 執行軌跡

### 2.3 幾何轉換（RL 輸出 → Cartesian 目標）

從 6D action（母球 XY、方向角、速度、tip offset XY）轉換出：

- 接觸姿態：桿頭尖端接觸瞬間的位置＋指向
- 接觸速度：方向＝擊球方向，大小＝訓練出的速度值
- 後擺姿態：沿擊球方向軸線往後拉開的位置＋同軸朝向，速度＝0

---

## 3. 驗收方法（Milestone B 驗收關卡）

### 3.1 為什麼需要這一步

RMPflow 是反應式控制器，理論上會收斂到目標，但實際硬體會有追蹤誤差。Tier (b) 驗收標準本身就是「方向與速度正確、固定中心接觸」，這一步驗證即是驗收本身，不可省略。

### 3.2 實作步驟

**Step 1：判定接觸時刻**（不能用固定 timestep，需用觸發條件）

```python
def find_contact_timestep(cue_tip_positions: np.ndarray, cue_ball_position: np.ndarray) -> int:
    """找出桿頭尖端與母球距離最小的 timestep，視為接觸時刻"""
    distances = np.linalg.norm(cue_tip_positions - cue_ball_position, axis=1)
    return np.argmin(distances)
```

**Step 2：用 FK 取得實際位置與朝向**（Isaac Sim experimental API 即時讀取，不需手動套公式）

```python
from isaacsim.core.experimental.prims import Articulation

robot = Articulation("/World/UR5")
end_effector_position, end_effector_orientation = robot.get_world_poses()
```

**Step 3：用 Jacobian 將關節角速度映射為末端速度**

```python
def compute_end_effector_velocity(robot: Articulation, contact_index: int):
    joint_velocities = robot.get_dof_velocities().numpy()[contact_index]
    jacobian = robot.get_jacobian_matrices().numpy()[contact_index]  # shape (6, num_dof)

    end_effector_velocity = jacobian @ joint_velocities  # (6,)
    return end_effector_velocity[:3], end_effector_velocity[3:]  # linear, angular
```

**Step 4：比對目標值，設定容忍區間**

```python
def verify_contact_pose(
    actual_position, actual_orientation, actual_linear_velocity,
    target_position, target_orientation, target_velocity,
    position_tolerance_m: float = 0.003,
    velocity_direction_tolerance_deg: float = 3.0,
    velocity_magnitude_tolerance_ratio: float = 0.1,
) -> dict:
    position_error = np.linalg.norm(actual_position - target_position)

    actual_speed = np.linalg.norm(actual_linear_velocity)
    target_speed = np.linalg.norm(target_velocity)
    speed_error_ratio = abs(actual_speed - target_speed) / target_speed

    cos_angle = np.dot(actual_linear_velocity, target_velocity) / (actual_speed * target_speed + 1e-8)
    direction_error_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    passed = (
        position_error <= position_tolerance_m
        and direction_error_deg <= velocity_direction_tolerance_deg
        and speed_error_ratio <= velocity_magnitude_tolerance_ratio
    )
    return {
        "passed": passed,
        "position_error_m": position_error,
        "direction_error_deg": direction_error_deg,
        "speed_error_ratio": speed_error_ratio,
    }
```

### 3.3 容忍值設定建議

| 項目 | 建議起始值 | 依據 |
|---|---|---|
| 位置誤差 | 球半徑（28.575mm）的 1/10 左右 | 對應 Tier(b) 固定中心接觸需求 |
| 方向誤差 | 3–5 度 | 角度誤差會直接放大到球的散開結果 |
| 速度誤差 | ±10–15% | 速度大小影響能量，重要性低於方向 |

實測後應搭配散開結果視覺化回頭調整容忍值，非一次定案。

### 3.4 整合位置

1. **Milestone B 硬體驗收 pipeline**：每次試打後自動執行，產出通過/失敗報告
2. **Milestone A 訓練 sanity check（選用）**：包成 evaluation callback，定期抽樣檢查 policy 輸出的 6D action 轉換後是否運動學可達，及早發現不可達目標，避免留到 Milestone B 才發現問題

---

## 4. 待釐清 / 待確認事項

- WAM 是否為你 Milestone B 的實際目標手臂，或僅為前導理解案例（Phase 3 原規劃為 UR5，剛性連接球桿）；若最終仍以 UR5 為準，第 2.2 節「路線 A libbarrett」部分不適用，僅第 2.2 節「路線 B RMPflow」與第 3 節驗收方法可直接沿用
- `wam.conf` 或 URDF 中差動轉換矩陣的精確係數，需在實作前從硬體/官方文件核實，不可用本文件中的通式代入計算
