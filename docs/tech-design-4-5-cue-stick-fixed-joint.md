# 球桿與 UR5 末端的固定連結（Fixed Joint）— 技術設計文件

> 生成時間：2026-07-15
>
> 更新時間：2026-07-16
>
> 所屬專案：isaac-sim-digital-twin
>
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/89

---

## 1. 功能概述

Issue #88 已由 `TableRobotManager` 建立球桿 Prim。Issue #89 進一步使用 PhysX Fixed Joint，將球桿固定連結至 UR5 的 `wrist_3_link`，使球桿能在物理模擬中穩定跟隨手臂末端移動。

原始實作直接建立 Fixed Joint，卻沒有先讓球桿與 `wrist_3_link` 的世界 Transform 重合。停止狀態下，Joint 的零值 local frame 不會自動搬移球桿；開始模擬時，PhysX solver 會嘗試在極短時間內消除兩端的大幅位置誤差，造成球桿高速移動、撞擊撞球，並將約束反力傳回 UR5 Articulation，使手臂持續抖動。

本次修正的核心是在建立 Joint 前，先將球桿完整世界 Transform 對齊 `wrist_3_link`，再排除球桿與連接剛體之間的碰撞，最後建立 Fixed Joint。第一版使用零位置偏移與零旋轉偏移，兩端 Joint local frame 維持零位置與 identity rotation。

---

## 2. 設計範圍

### 納入範圍

- `wrist_3_link` 同時作為 Fixed Joint 的物理 body 與球桿的定位 target。
- `StageAPI` 新增世界 Transform 對齊能力。
- `StageAPI` 新增指定 Prim pair 的碰撞過濾能力。
- `TableRobotManager` 依「建立球桿 → 對齊 → 過濾碰撞 → 建立 Joint」的順序初始化。
- Core 測試驗證呼叫參數及呼叫順序。
- 球桿質量由 USD 資產管理，不由執行階段程式設定。

### 不納入範圍

- 第一版不支援球桿相對 `wrist_3_link` 的位置或旋轉 offset。
- 第一版不新增 Fixed Joint local frame 參數；兩端使用零位置與 identity rotation。
- 不新增檔案、class 或獨立 `JointAPI` Port。
- 不由程式設定 Mass。
- 不停用球桿與撞球、桌面或其他外部物件的碰撞。

---

## 3. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `UR5Robot` | core/models | `get_end_effector_prim_path()` 回傳 `{robot_prim_path}/wrist_3_link` | `core/models/ur5_robot.py` |
| `StageAPI` | core/ports | 定義 `align_prim_to_target()`、`filter_collision_pair()` 與既有 `create_fixed_joint()` 抽象介面 | `core/ports/stage_api.py` |
| `StageAPIImpl` | extension/isaac_sim_impl_6_0 | 計算並寫入 Prim Transform、建立碰撞過濾關係、建立 Fixed Joint | `extension/isaac_sim_impl_6_0/stage_api_impl.py` |
| `TableRobotManager` | core/models | 按正確順序協調球桿對齊、碰撞過濾與 Fixed Joint 建立 | `core/models/table_robot_manager.py` |
| `ball_stick.usd` | assets | 保存球桿 Rigid Body 的 Mass，目標值為 `0.5 kg` | `assets/ball_stick.usd` |

本次沿用既有檔案與 class，不建立任何新檔案或新 class。

---

## 4. 類別與介面設計

### 4.1 UR5Robot

**職責：** 提供已確認的 UR5 末端剛體 Prim 路徑。

```python
class UR5Robot:
    _END_EFFECTOR_LINK_NAME = "wrist_3_link"

    def get_end_effector_prim_path(self) -> str:
        """回傳 {prim_path}/wrist_3_link。"""
        ...
```

`wrist_3_link` 已確認是要連接的 Articulation Link。場景中不存在 `tool0`，因此設計與測試均不得再使用 `tool0`。

### 4.2 StageAPI

**職責：** 將所有 USD／PhysX 細節留在 extension 實作層，讓 core 僅透過平台無關的路徑介面表達初始化流程。

```python
class StageAPI(ABC):
    @abstractmethod
    def align_prim_to_target(self, prim_path: str, target_path: str) -> None:
        """使 prim_path 的完整世界 Transform 與 target_path 重合。"""
        ...

    @abstractmethod
    def filter_collision_pair(self, prim_path_a: str, prim_path_b: str) -> None:
        """排除指定兩個 Prim 之間的碰撞，不影響其他碰撞 pair。"""
        ...

    @abstractmethod
    def create_fixed_joint(
        self,
        joint_path: str,
        body0_path: str,
        body1_path: str,
    ) -> None:
        """在 joint_path 建立 Fixed Joint，固定連結兩端剛體。"""
        ...
```

### 4.3 StageAPIImpl

#### `align_prim_to_target()`

實作必須處理 Prim parent 不同的情況，不能只複製 local translate。Transform 計算原則為：

```text
desired_prim_world = target_world
prim_local = inverse(prim_parent_world) × desired_prim_world
```

寫入內容須包含完整位置與旋轉，而非只處理 translate。第一版不套用額外 offset，因此對齊完成後：

```text
CueStick world transform = wrist_3_link world transform
```

若球桿 root 既有必要的 scale 或 pivot transform，實作時必須保留或納入矩陣計算，不能在未檢查資產結構的情況下遺失既有 transform op。

#### `filter_collision_pair()`

使用 USD Physics filtered-pairs 關係，只過濾以下碰撞 pair：

```text
CueStick ↔ wrist_3_link
```

不得排除以下碰撞：

```text
CueStick ↔ 撞球
CueStick ↔ 桌面
CueStick ↔ 其他外部物件
```

這可避免連接處的幾何重疊使碰撞排斥力與 Fixed Joint 約束互相對抗。

#### `create_fixed_joint()`

```python
def create_fixed_joint(
    self,
    joint_path: str,
    body0_path: str,
    body1_path: str,
) -> None:
    joint = UsdPhysics.FixedJoint.Define(self.get_stage(), joint_path)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])
```

第一版在建立 Joint 前已讓球桿與 `wrist_3_link` 的世界 Transform 重合，因此兩端 local position 使用 `(0, 0, 0)`，local rotation 使用 identity。無需增加 offset 或 local frame 參數。

### 4.4 TableRobotManager

**職責：** 建立場景物件並協調正確的物理初始化順序。

```python
class TableRobotManager:
    def __init__(
        self,
        table_center: tuple[float, float, float],
        base_path: str,
        stage_api: StageAPI,
    ) -> None:
        """建立 UR5 與球桿，先對齊並過濾碰撞，再建立 Fixed Joint。"""
        ...
```

第一版不需要 `_CUE_JOINT_OFFSET`。若既有程式仍保留未使用的 offset 常數，實作階段應移除，避免暗示目前已支援抓持偏移。

---

## 5. 資料流與呼叫順序

```text
BilliardExtension._billiard_init()
  → TableRobotManager.__init__(table_center, base_path, stage_api)
    → 建立 UR5Robot 並設定既有世界位置
    → stage_api.create_reference_prim(cue_stick_path, CUE_STICK_PATH)
    → end_effector_path = robot.get_end_effector_prim_path()
       # {robot_prim_path}/wrist_3_link
    → stage_api.align_prim_to_target(cue_stick_path, end_effector_path)
    → stage_api.filter_collision_pair(cue_stick_path, end_effector_path)
    → stage_api.create_fixed_joint(
          joint_path,
          cue_stick_path,
          end_effector_path,
      )
```

以下順序是功能正確性的必要條件：

```text
create cue → align → filter cue/wrist collision → create fixed joint
```

不能先建立 Fixed Joint 再對齊，也不能依賴 Disabled Joint 或 solver projection 取代初始 Transform 對齊。

---

## 6. 依賴關係

```text
TableRobotManager
  ├── UR5Robot.get_end_effector_prim_path()
  └── StageAPI
      ├── create_reference_prim()
      ├── align_prim_to_target()
      ├── filter_collision_pair()
      └── create_fixed_joint()

StageAPIImpl
  ├── USD Transform API
  └── pxr.UsdPhysics
      ├── FilteredPairsAPI
      └── FixedJoint
```

Core 層不直接依賴 `pxr`、`omni.usd` 或其他 Isaac Sim API。

---

## 7. USD 資產設定

球桿質量不由程式碼設定。使用者需在 Isaac Sim 中開啟 `assets/ball_stick.usd`，選取套用 `PhysicsRigidBodyAPI` 的球桿根 Prim，於 Physics Mass 設定：

```text
Mass = 0.5 kg
```

儲存資產後，重新載入引用球桿的主場景，確認 Mass 仍為 `0.5 kg`。此設定屬於資產本身的物理屬性，`StageAPI` 不新增 `set_rigid_body_mass()`，`TableRobotManager` 也不在執行階段覆寫 Mass。

---

## 8. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| `prim_path`、`target_path` 或 Joint body 路徑不存在 | 沿用專案既有模式，讓底層 USD／PhysX 例外自然拋出，不在 core 重複防呆 |
| 球桿與 `wrist_3_link` 的 parent 不同 | 以世界 Transform 對齊後，換算回球桿 parent 空間的 local Transform，不能複製 local translate |
| 球桿 root 有既有 scale／pivot／transform op | 實作前檢查並保留必要變換，避免清除資產原有設定 |
| 連接處碰撞幾何重疊 | 只排除 CueStick 與 `wrist_3_link` 的碰撞 pair |
| 球桿需要正式抓持位置或方向 | 列為後續功能；屆時需同時設計 offset 與 Joint local attachment frames，不在第一版先猜測數值 |
| Joint 建立後仍有劇烈抖動 | 依序檢查對齊誤差、filtered pair 是否生效、球桿 Mass、碰撞形狀及 UR5 self-collision 設定；不以 `excludeFromArticulation` 作為本案例的預設解法 |

---

## 9. 測試涵蓋

### Core Unit Test

| 測試案例 | 測試檔案 | 驗證內容 |
|---|---|---|
| `test_table_robot_manager_aligns_cue_stick_to_end_effector` | `core/tests/test_table_robot_manager.py` | 驗證 `align_prim_to_target(cue_stick_path, wrist_3_link_path)` 的參數 |
| `test_table_robot_manager_filters_cue_stick_end_effector_collision` | `core/tests/test_table_robot_manager.py` | 驗證 `filter_collision_pair(cue_stick_path, wrist_3_link_path)` 的參數，且不過濾球桿與撞球或桌面 |
| `test_table_robot_manager_creates_fixed_joint` | `core/tests/test_table_robot_manager.py` | 驗證 Joint path、`body0_path = cue_stick_path`、`body1_path = wrist_3_link_path` |
| `test_table_robot_manager_initializes_cue_stick_joint_in_order` | `core/tests/test_table_robot_manager.py` | 使用 Mock 呼叫紀錄驗證 `create_reference_prim → align_prim_to_target → filter_collision_pair → create_fixed_joint` 的相對順序 |
| `test_ur5_robot_get_end_effector_prim_path` | `core/tests/test_ur5_robot.py` | 驗證回傳 `{robot_prim_path}/wrist_3_link`，不得使用 `tool0` |

測試需同時驗證參數與順序；只驗證 `create_fixed_joint()` 曾被呼叫，無法防止本次「未先對齊」的 Bug 再次發生。

### Isaac Sim 手動驗證

- Stop 狀態下，建立 Joint 前球桿已出現在 `wrist_3_link`。
- 球桿與 `wrist_3_link` 的世界位置每軸誤差小於 `0.001 m`。
- 球桿與 `wrist_3_link` 的世界旋轉一致。
- Play 第一幀球桿沒有明顯 snap 或高速掃過場景。
- UR5 不再持續抖動或亂動。
- 撞球不會在 Play 第一幀被球桿意外撞擊。
- UR5 移動時，球桿穩定跟隨且無可見漂移。
- 球桿仍能與撞球及桌面發生預期碰撞。
- 引用後的球桿 Mass 為 `0.5 kg`。

`StageAPIImpl` 的 USD 矩陣運算與 PhysX 行為不寫 core Unit Test，依專案慣例以 Isaac Sim 場景手動驗證。

---

## 10. 已確認決策與後續範圍

- [x] `wrist_3_link` 是 Fixed Joint 連接的物理 body。
- [x] 場景不存在 `tool0`；`wrist_3_link` 同時作為定位 target。
- [x] 第一版使用零位置 offset 與零旋轉 offset。
- [x] 第一版 Joint local frames 使用零位置與 identity rotation。
- [x] Collision filtering 納入 Issue #89，只排除 CueStick 與 `wrist_3_link`。
- [x] Mass 不由程式設定，改存入 `assets/ball_stick.usd`，目標值為 `0.5 kg`。
- [x] 沿用既有檔案與 class，不新增檔案或 class。
- [ ] 後續若需調整正式握持點，再另行量測球桿 Pivot，並同步設計 offset 與 Joint local attachment frames。

Issue #89 完成後，球桿即可穩定跟隨 UR5 末端，供後續 Block 5 擊球動作狀態機使用。
