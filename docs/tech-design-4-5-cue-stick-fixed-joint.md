# 球桿與 UR5 末端的固定連結（Fixed Joint）— 技術設計文件

> 生成時間：2026-07-15
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/89

---

## 1. 功能概述

Issue #88 已在 `TableRobotManager` 建立球桿 Prim（`get_cue_stick_prim_path()`），但球桿目前使用暫定位置（與 UR5 機器人相同世界座標），並未與機器人有任何物理關聯。本次任務以 PhysX Fixed Joint 將球桿 Prim 固定連結至 UR5 末端執行器 Prim，取代暫定定位邏輯。輸入為球桿 Prim 路徑（`TableRobotManager` 既有）與 UR5 末端執行器 Prim 路徑（本次新增查詢方法）；輸出為 Stage 中新增的 Fixed Joint Prim，效果是球桿在物理模擬中會確實跟隨 UR5 末端執行器移動，供後續 Block 5（擊球動作狀態機）使用。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `UR5Robot` | core/models | 新增 `get_end_effector_prim_path()`，回傳末端執行器 Prim 路徑（純字串組合） | `core/models/ur5_robot.py` |
| `StageAPI` | core/ports | 新增抽象方法 `create_fixed_joint()` | `core/ports/stage_api.py` |
| `StageAPIImpl` | extension/isaac_sim_impl_6_0 | 實作 `create_fixed_joint()`，使用 `pxr.UsdPhysics.FixedJoint` | `extension/isaac_sim_impl_6_0/stage_api_impl.py` |
| `TableRobotManager` | core/models | 修改建構子：取得末端執行器路徑並建立 Fixed Joint，移除球桿暫定定位呼叫 | `core/models/table_robot_manager.py` |

---

## 3. 類別設計

### UR5Robot（修改部分）

**職責（新增）：** 提供末端執行器 Prim 路徑查詢，供 Fixed Joint 建立時使用。

**介面：**
```python
class UR5Robot:
    _END_EFFECTOR_LINK_NAME = "tool0"  # 佔位常數，待查證 UR5 USD 實際連桿名稱

    def get_end_effector_prim_path(self) -> str:
        """回傳末端執行器 Prim 的完整路徑，例如 {prim_path}/tool0。
        純字串路徑組合，不呼叫 ArticulationAPI（UR5 是 Nucleus 現成資產，
        結構已知；耦合 ArticulationAPI 會提早綁定控制層生命週期，
        而 TableRobotManager 建構發生在場景初始化階段，
        ArticulationAPI.initialize() 需等場景穩定才能呼叫）。
        """
        ...
```

**依賴：**
- 輸入來源：既有 `self._prim_path`（建構子已組好）
- 輸出去向：`TableRobotManager` 建構子，作為 `create_fixed_joint` 的 `body1_path`

---

### StageAPI / StageAPIImpl（新增部分）

**職責：** 在 Stage 中建立 PhysX Fixed Joint Prim，連結兩個已存在的 Prim。

**介面：**
```python
class StageAPI(ABC):
    @abstractmethod
    def create_fixed_joint(self, joint_path: str, body0_path: str, body1_path: str) -> None:
        """
        在 joint_path 建立 Fixed Joint Prim，將 body0_path 與 body1_path
        兩端固定連結。
        """
        ...
```

實作（`StageAPIImpl`）：
```python
def create_fixed_joint(self, joint_path: str, body0_path: str, body1_path: str) -> None:
    joint = UsdPhysics.FixedJoint.Define(self.get_stage(), joint_path)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])
```

**依賴：**
- 輸入來源：`TableRobotManager`（呼叫端提供三個路徑參數）
- 輸出去向：`pxr.UsdPhysics.FixedJoint`（外部 PhysX API）

**設計理由：** Fixed Joint 本質是「在 Stage 建立一個新 Prim」，與既有的 `create_reference_prim` 同類職責，放進 `StageAPI` 比新開一個獨立 Port（如 `JointAPI`）更符合現有分層慣例；`RigidBodyAPI` 是純查詢職責（`get_position`/`get_linear_velocity`/`get_angular_velocity`），不適合放這裡。

**實作風險提醒（待驗證，見第 8 節）：** 兩端 Prim 通常需要 `PhysicsRigidBodyAPI` 才能正確參與物理模擬；球桿資產（`ball_stick.usd`）是否已有 `RigidBodyAPI`/`CollisionAPI` 尚未查證；UR5 末端執行器屬於 Articulation 的一部分，Fixed Joint 銜接一般 RigidBody 與 Articulation Link 在 PhysX 是支援的，但需注意 `excludeFromArticulation` 等設定，建議建好後在 Isaac Sim Play 模式下實際驗證球桿是否確實跟隨、無漂移。

---

### TableRobotManager（修改部分）

**職責（修改）：** 建構子內以 Fixed Joint 將球桿連結至 UR5 末端執行器，取代原本的球桿暫定定位（`set_prim_translate`）。

**介面：**
```python
class TableRobotManager:
    _ROBOT_OFFSET_FROM_TABLE_CENTER = (1.5, 0.0, 0.0)
    _CUE_JOINT_OFFSET = (0.0, 0.0, 0.0)  # 佔位常數，待球桿 Pivot 點位置查證後填值

    def __init__(
        self,
        table_center: tuple[float, float, float],
        base_path: str,
        stage_api: StageAPI,
    ) -> None:
        """建立 UR5Robot、引用球桿資產（既有，#88），
        並以 Fixed Joint 將球桿固定連結至 UR5 末端執行器
        （新增，#89）。不再對球桿呼叫 set_prim_translate。
        """
        ...

    def get_robot_prim_path(self) -> str:
        """回傳 Robot Prim 的完整路徑。（既有，不變）"""
        ...

    def get_cue_stick_prim_path(self) -> str:
        """回傳球桿 Prim 的完整路徑。（既有，不變）"""
        ...

    def destroy(self) -> None:
        """既有，不變。"""
        ...
```

**依賴：**
- 輸入來源：`UR5Robot.get_end_effector_prim_path()`（新增）、`self._cue_stick_prim_path`（既有）
- 輸出去向：`StageAPI.create_fixed_joint()`（新增用法）

---

## 4. 資料流

```
BilliardExtension._billiard_init()
  → TableRobotManager.__init__(table_center, base_path, stage_api)
    → world_position = table_center + _ROBOT_OFFSET_FROM_TABLE_CENTER   （既有邏輯，不變）
    → UR5Robot(base_path, stage_api, world_position)                    （既有，不變）
    → stage_api.create_reference_prim(base_path + "/CueStick", CUE_STICK_PATH)   （既有，#88）
    → end_effector_path = self._robot.get_end_effector_prim_path()      （新增）
    → joint_path = self._cue_stick_prim_path + "/FixedJointToRobot"     （新增）
    → stage_api.create_fixed_joint(joint_path, self._cue_stick_prim_path, end_effector_path)   （新增）
    ✗ stage_api.set_prim_translate(cue_stick_path, *world_position)     （移除，#89 起由物理引擎接管球桿位置）
  → 回傳 TableRobotManager 實例
  → 之後（Isaac Sim Play 模式）：PhysX 依 Fixed Joint 約束，
    球桿隨 UR5 末端執行器物理模擬移動
```

---

## 5. 依賴關係圖

```
TableRobotManager
  ├── 依賴 UR5Robot（既有 + 新增 get_end_effector_prim_path()）
  └── 依賴 StageAPI（新增用法：create_fixed_joint）

UR5Robot
  └── 無新依賴（固定路徑組合，不呼叫 ArticulationAPI）

StageAPIImpl
  └── 依賴 pxr.UsdPhysics（PhysX Fixed Joint API）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| `create_fixed_joint` 或末端執行器路徑取得失敗 | 不做防呆檢查，沿用專案既有模式（比照 #88 的 `create_reference_prim`）：直接讓底層例外拋出，由呼叫端（Extension 初始化）自然中斷 |
| 球桿與機器人初始位置重疊/偏移問題 | 由 Fixed Joint 建立後接管，不再需要 `TableRobotManager` 手動定位 |
| 球桿或末端執行器缺少 `PhysicsRigidBodyAPI` 導致 Joint 不生效 | 本次不寫防呆邏輯；列入待驗證事項，需在 Isaac Sim Play 模式下手動確認球桿是否確實跟隨、無漂移 |

---

## 7. 測試涵蓋（對應 Unit Test）

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| `test_table_robot_manager_creates_fixed_joint` | `core/tests/test_table_robot_manager.py` | 驗證 `stage_api.create_fixed_joint` 被呼叫，`body0_path` 為球桿路徑、`body1_path` 為 `get_end_effector_prim_path()` 回傳值 |
| `test_table_robot_manager_no_longer_sets_cue_stick_translate` | `core/tests/test_table_robot_manager.py` | 確認暫定定位邏輯（`set_prim_translate` for cue stick）已移除，或既有測試已刪除/調整 |
| `test_ur5_robot_get_end_effector_prim_path` | `core/tests/test_ur5_robot.py`（若不存在則新建） | 驗證回傳值為 `{prim_path}/<連桿名稱>` |

備註：`StageAPIImpl` 與 `UR5Robot` 內實際呼叫 Isaac Sim/PhysX API 的部分不寫 Unit Test（沿用專案慣例，實作層用 Debug Menu 或手動場景驗證，Unit Test 只測 core 層邏輯與 Mock 呼叫參數）。

---

## 8. 待決定事項

- [ ] UR5 末端執行器在 USD 內的實際連桿名稱（需在 Isaac Sim Stage 樹展開 UR5 Prim 確認，例如 tool0 / ee_link / wrist_3_link 等）
- [ ] `ball_stick.usd` 資產是否已套用 `PhysicsRigidBodyAPI` / `CollisionAPI`（需開啟資產確認）
- [ ] `_CUE_JOINT_OFFSET` 實際數值（待球桿 Pivot 點位置查證後填入）
- 後續依賴：本 Issue 完成後，球桿即可隨 UR5 末端執行器物理模擬移動，供後續 Block 5（擊球動作狀態機）使用
