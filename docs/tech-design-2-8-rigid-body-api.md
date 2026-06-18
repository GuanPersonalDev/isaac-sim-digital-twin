# [2-8] RigidBodyAPI 抽象介面 — 技術設計文件

> 生成時間：2026-06-17
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：（待 progress-planner 建立 Issue 後補入）

---

## 1. 功能概述

提供查詢單一剛體（撞球）的世界座標位置、線速度與角速度的抽象介面。呼叫端（`core/services/ShotResultProvider` 或靜止偵測器）透過此介面判斷球是否完全靜止，無需直接依賴任何 Omniverse / Isaac Sim API。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `RigidBodyAPI` | extension/omniverse_api（ABC） | 定義位置、線速度、角速度查詢的抽象契約 | `extension/omniverse_api/rigid_body_api.py` |
| `RigidBodyAPIImpl`（使用者實作） | extension/isaac_sim_impl_6_0 | 以 Isaac Sim 6.0.0 `RigidPrim` 實作上述三個方法 | `extension/isaac_sim_impl_6_0/rigid_body_api_impl.py` |

---

## 3. 類別設計

### RigidBodyAPI

**職責：** 定義剛體物理狀態查詢的抽象介面，使 `core` 層與引擎實作完全解耦。

**介面：**
```python
from abc import ABC, abstractmethod

class RigidBodyAPI(ABC):

    @abstractmethod
    def get_position(self, prim_path: str) -> list[float]:
        """回傳剛體世界座標位置 [x, y, z]（單位：m）"""
        ...

    @abstractmethod
    def get_linear_velocity(self, prim_path: str) -> list[float]:
        """回傳線速度 [vx, vy, vz]（單位：m/s）"""
        ...

    @abstractmethod
    def get_angular_velocity(self, prim_path: str) -> list[float]:
        """回傳角速度 [wx, wy, wz]（單位：rad/s）"""
        ...
```

**依賴：**
- 輸入來源：呼叫端傳入 `prim_path`（USD 路徑字串，例如 `"/World/Ball_1"`）
- 輸出去向：`core/services/ShotResultProvider` 或靜止偵測器，用於計算速度向量範數以判定靜止

---

### RigidBodyAPIImpl（Isaac Sim 6.0.0 實作層，使用者自行實作）

**職責：** 以 `isaacsim.core.experimental.prims.RigidPrim` 實作抽象介面，並負責 warp array 轉換為標準 Python `list[float]`。

**實作參考：**
```python
from isaacsim.core.experimental.prims import RigidPrim
from extension.omniverse_api.rigid_body_api import RigidBodyAPI

class RigidBodyAPIImpl(RigidBodyAPI):

    def get_position(self, prim_path: str) -> list[float]:
        ball = RigidPrim(paths=prim_path)
        positions, _ = ball.get_world_poses()   # 注意：複數形式（6.0.0 新命名）
        return positions.numpy()[0].tolist()    # shape (1, 3) → [x, y, z]

    def get_linear_velocity(self, prim_path: str) -> list[float]:
        ball = RigidPrim(paths=prim_path)
        lin_vel, _ = ball.get_velocities()
        return lin_vel.numpy()[0].tolist()      # [vx, vy, vz]

    def get_angular_velocity(self, prim_path: str) -> list[float]:
        ball = RigidPrim(paths=prim_path)
        _, ang_vel = ball.get_velocities()
        return ang_vel.numpy()[0].tolist()      # [wx, wy, wz]
```

**依賴：**
- 輸入來源：`prim_path` 字串（由呼叫端提供）
- 輸出去向：回傳標準 `list[float]` 給 `core` 層使用

---

## 4. 資料流

```
core/services/ShotResultProvider（或靜止偵測器）
  │
  ├─ rigid_body_api.get_position("/World/Ball_1")
  │    → RigidPrim.get_world_poses() → warp array (1,3) → [x, y, z]
  │
  ├─ rigid_body_api.get_linear_velocity("/World/Ball_1")
  │    → RigidPrim.get_velocities() → lin_vel warp array (1,3) → [vx, vy, vz]
  │
  └─ rigid_body_api.get_angular_velocity("/World/Ball_1")
       → RigidPrim.get_velocities() → ang_vel warp array (1,3) → [wx, wy, wz]

靜止判定邏輯（位於 core 層）：
  norm([vx, vy, vz]) < threshold AND norm([wx, wy, wz]) < threshold
  → True：球已靜止，可進行下一步
```

---

## 5. 依賴關係圖

```
core/services/ShotResultProvider
  └── 依賴 RigidBodyAPI（ABC）（查詢球的物理狀態）

extension/omniverse_api/RigidBodyAPI（ABC）
  └── 無引擎依賴（純 Python ABC）

extension/isaac_sim_impl_6_0/RigidBodyAPIImpl
  └── 依賴 isaacsim.core.experimental.prims.RigidPrim（Isaac Sim 6.0.0 API）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 模擬尚未啟動即呼叫速度查詢 | 回傳零向量 `[0.0, 0.0, 0.0]`（PhysX 尚未執行，為預期行為，呼叫端應在模擬至少執行一步後才判定靜止） |
| `prim_path` 指向不存在的 prim | `RigidPrim` 建立時可能拋出例外，實作層應視需求加 try/except 並回傳預設值或向上傳遞 |
| 球原地自旋（線速度為零、角速度不為零） | 靜止判定必須同時檢查 `get_linear_velocity` 與 `get_angular_velocity`，兩者皆低於閾值才算靜止（此為角速度方法存在的核心理由） |
| warp array shape 非預期 | `[0]` 索引前可加 shape assert，若 `(1, 3)` 不符則拋出 `ValueError` 並附帶 shape 資訊 |

---

## 7. 測試涵蓋

**Unit Test 豁免：** `RigidBodyAPI` 為純 ABC，無實作邏輯，依 `unit-test-rules.md` 豁免 Unit Test。

`RigidBodyAPIImpl` 的整合測試屬 Isaac Sim 環境依賴，不在 `core/tests/` 覆蓋範圍內。

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| （豁免） | — | ABC 無可測試的實作邏輯 |

---

## 8. 關鍵設計決策

1. **角速度必要性**：撞球可能原地自旋（線速度為零但角速度不為零）。依撞球規則需等球完全靜止，因此靜止判定必須同時檢查線速度與角速度。

2. **回傳 `list[float]` 而非 numpy array**：抽象介面回傳標準 Python 型別，與任何數值計算函式庫解耦。實作層負責執行 `.numpy()[0].tolist()` 轉換。

3. **無新增 dataclass**：三個查詢各自語義獨立，回傳 `list[float]` 已足夠，不需要包裝型別，保持介面簡單。

---

## 9. 待決定事項

- [ ] `RigidBodyAPIImpl` 中每次呼叫都重新建立 `RigidPrim(paths=prim_path)` 是否有效能問題（需確認 Isaac Sim 6.0.0 是否有物件快取機制）
- [ ] 靜止判定的 `threshold` 數值由 `ShotResultProvider` 自行管理，或作為可設定參數傳入
