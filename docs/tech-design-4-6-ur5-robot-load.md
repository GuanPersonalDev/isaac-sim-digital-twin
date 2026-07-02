# UR5Robot 載入與場景整合 — 技術設計文件

> 生成時間：2026-06-27
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：待 progress-planner 建立 Issue 後補入

---

## 1. 功能概述

在 `BilliardTable` 初始化時，自動從 Nucleus 載入 UR5 USD 並以世界座標定位於桌台側邊；外部呼叫者無需感知 Robot 存在，Robot 位置由 `BilliardTable` 依自身座標計算偏移後決定。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `UR5Robot` | core/models | 從 Nucleus 載入 UR5 USD、設定世界座標位置、提供 Prim 路徑 | `core/models/ur5_robot.py` |
| `BilliardTable` | core/models | 新增持有 `UR5Robot` 實例；將桌台世界座標轉換為 Robot 世界座標後傳入 | `core/models/billiard_table.py` |
| `asset_utility` | core/services | 新增 `UR5_PATH` 常數，集中管理 Nucleus 路徑 | `core/services/asset_utility.py` |

---

## 3. 類別設計

### UR5Robot

**職責：** 封裝 UR5 USD 載入與 Prim 定位，對外提供 Prim 路徑查詢。

**介面：**
```python
class UR5Robot:
    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        position: tuple[float, float, float],
    ) -> None:
        """從 Nucleus 載入 UR5 USD 並定位至 position（世界座標）。
        Prim 路徑為 {base_path}/Robot。
        """
        ...

    def get_prim_path(self) -> str:
        """回傳 Robot Prim 的完整路徑，例如 /World/BilliardTable/Robot。"""
        ...
```

**依賴：**
- 輸入來源：`StageAPI`（由 `BilliardTable` 注入）、`asset_utility.UR5_PATH`
- 輸出去向：目前僅 `BilliardTable` 持有實例；`get_prim_path()` 供未來關節控制模組使用

---

### BilliardTable（修改部分）

**職責（新增）：** 計算 Robot 世界座標並於 `__init__` 尾端實例化 `UR5Robot`。

**介面（新增邏輯，非新增 public method）：**
```python
# BilliardTable.__init__ 尾端新增
robot_world_pos = (x_pos + 1.5, y_pos + 0.0, 0.0)  # 暫定偏移，Task 4-7 精調
self._robot = UR5Robot(base_path, stage_api, robot_world_pos)
```

**依賴：**
- 輸入來源：`__init__` 既有的 `position`（桌台世界座標）
- 輸出去向：`UR5Robot` 實例（私有持有）

---

### asset_utility（修改部分）

**新增常數：**
```python
UR5_PATH = "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
```

---

## 4. 資料流

```
BilliardTable.__init__(base_path, stage_api, material_api, position)
  → 建立桌台 Prim（現有）
  → 建立 TableBallSet（現有）
  → robot_world_pos = (position.x + 1.5, position.y + 0.0, 0.0)
  → UR5Robot.__init__(base_path, stage_api, robot_world_pos)
      → stage_api.create_reference_prim("{base_path}/Robot", asset_utility.UR5_PATH)
      → stage_api.set_prim_translate("{base_path}/Robot", robot_world_pos)
  → self._robot 持有實例（備未來關節控制使用）
```

---

## 5. 依賴關係圖

```
BilliardTable
  ├── 依賴 UR5Robot（建立並持有 Robot 實例）
  └── 依賴 asset_utility（間接，由 UR5Robot 使用）

UR5Robot
  ├── 依賴 StageAPI（create_reference_prim、set_prim_translate）
  └── 依賴 asset_utility.UR5_PATH（Nucleus 路徑常數）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 最小範圍實作，暫無邊緣案例 | 未來新增關節控制時補充（例如 Nucleus 路徑不存在、Prim 重複建立） |

---

## 7. 測試涵蓋（對應 Unit Test）

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| `test_ur5robot_creates_reference_prim` | `core/tests/test_ur5_robot.py` | 驗證 `create_reference_prim` 以正確路徑與 UR5_PATH 被呼叫 |
| `test_ur5robot_sets_translate` | `core/tests/test_ur5_robot.py` | 驗證 `set_prim_translate` 收到正確世界座標 |
| `test_ur5robot_get_prim_path` | `core/tests/test_ur5_robot.py` | 驗證回傳 `{base_path}/Robot` |
| `test_billiard_table_creates_robot` | `core/tests/test_billiard_table.py` | 驗證 `BilliardTable.__init__` 後 `_robot` 不為 None |
| `test_billiard_table_robot_world_position` | `core/tests/test_billiard_table.py` | 驗證 Robot 世界座標 = 桌台座標 + (1.5, 0.0, 0.0) |

---

## 8. 待決定事項

- [ ] Task 4-7：精確擺位時確認 Robot 偏移值（目前暫定 `x + 1.5`），確認後更新本文件
- [ ] 未來關節控制擴充時，決定 `UR5Robot` 是否需要實作 `ArticulationAPI` 抽象介面
