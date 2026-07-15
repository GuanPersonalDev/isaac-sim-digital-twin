# 撞球桌與手臂職責拆分：BilliardTable / TableRobotManager / BilliardExtension / DebugMenu — 技術設計文件

> 生成時間：2026-07-13
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：[#184](https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/184) [4-3d-impl]、[#185](https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/185) [4-3d-test]、[#186](https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/186) [4-3e-impl]、[#187](https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/187) [4-3e-test]、[#188](https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/188) [4-3f]、[#189](https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/189) [4-3g]（Milestone M2: 場景與機器人）

---

## 1. 功能概述

因 Milestone A（RL 訓練，128 平行環境、無手臂）與 Milestone B（手臂執行，UR5+球桿剛性連結、單一環境）對「桌子是否含手臂」的需求不同，本次設計將 `BilliardTable` 的手臂建立邏輯抽離，新增 `TableRobotManager` 作為手臂操作的唯一對外中介層；`BilliardExtension` 改為分別管理「訓練桌 list（永遠無手臂）」與「唯一一張 Demo 桌（含手臂）」；並在 Debug Menu 新增「訓練環境執行中」、「Demo 手臂執行中」兩個開關（本輪僅做 UI 與狀態存放，不含實際訓練 / 揮桿邏輯）。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `BilliardTable` | core/models | 建立桌台 Prim 與球組，提供桌面幾何中心查詢；不再建立手臂 | `core/models/billiard_table.py` |
| `TableRobotManager` | core/models | 手臂操作的唯一對外窗口，內部持有 `UR5Robot`，計算手臂世界座標並轉呼叫 | `core/models/table_robot_manager.py`（新） |
| `UR5Robot` | core/models | 手臂 prim 建立與座標設定（職責不變，不修改） | `core/models/ur5_robot.py` |
| `BilliardExtension` | extension | 分離管理訓練桌 list 與 Demo 桌，建立 `TableRobotManager` 與 `DebugMenu`，接收開關 callback 並存狀態 | `extension/billiard_digital_twin/billiard_digital_twin.py` |
| `DebugMenu` | extension/ui | 提供訓練/Demo 兩個 CheckBox UI，值變化時呼叫呼叫端傳入的 callback；不持有布林狀態 | `extension/ui/debug_menu.py` |

---

## 3. 類別設計

### BilliardTable（修改）

**職責：** 專心處理桌子本身（桌台 Prim、球組建立），不再建立或持有手臂。

**介面：**
```python
class BilliardTable:
    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        material_api: MaterialAPI,
        position: tuple[float, float],
    ) -> None:
        """建立桌台 Prim 與球組，並將 position 存為 instance 欄位。"""
        ...

    def get_table_prim_path(self) -> str:
        """回傳桌台 Prim 路徑（既有方法，不變）。"""
        ...

    def get_table_center(self) -> tuple[float, float, float]:
        """回傳桌面幾何中心世界座標 (self._x_pos, self._y_pos, 0.0)。"""
        ...

    def destroy(self) -> None:
        """銷毀桌台與球組（不再包含 self._robot = None）。"""
        ...
```

**變更重點：**
- 移除 `UR5Robot` import 與建立邏輯（原第 39-40 行）
- 建構子內把 `x_pos, y_pos` 存成 `self._x_pos`, `self._y_pos`
- `destroy()` 移除 `self._robot = None`

**依賴：**
- 輸入來源：`StageAPI`、`MaterialAPI`、`TableBallSet`、`BreakShotPositionProvider`（皆為既有依賴，不變）
- 輸出去向：`get_table_center()` 供 `BilliardExtension` 傳給 `TableRobotManager` 計算手臂座標

---

### TableRobotManager（新增）

**職責：** 手臂操作的唯一對外窗口／中介層。日後所有手臂相關操作（例如調整手臂位置）一律加在此類別，外部一律透過此類別操作手臂，不直接持有或呼叫 `UR5Robot`。

**介面：**
```python
class TableRobotManager:
    _ROBOT_OFFSET_FROM_TABLE_CENTER = (1.5, 0.0, 0.0)  # 具名常數，避免 magic number

    def __init__(
        self,
        table_center: tuple[float, float, float],
        base_path: str,
        stage_api: StageAPI,
    ) -> None:
        """
        world_position = table_center + _ROBOT_OFFSET_FROM_TABLE_CENTER
        建立並私有持有 self._robot = UR5Robot(base_path, stage_api, world_position)
        """
        ...

    def get_robot_prim_path(self) -> str:
        """轉呼叫 self._robot.get_prim_path()。"""
        ...

    def destroy(self) -> None:
        """銷毀內部持有的 UR5Robot 實例。"""
        ...
```

**依賴：**
- 輸入來源：`BilliardTable.get_table_center()` 的回傳值、`StageAPI`
- 輸出去向：內部建立 `UR5Robot`；`get_robot_prim_path()` 供未來關節控制模組使用

**設計決策：** 偏移量 `(1.5, 0.0, 0.0)` 目前寫死為具名常數，不做成建構子參數（YAGNI）；等 Milestone B 可達性掃描（#180）真的需要動態調整基座位置時再擴充。

---

### BilliardExtension（修改）

**職責（新增部分）：** 分離管理訓練桌 list（`self._tables`，永遠無手臂）與唯一一張 Demo 桌（`self._demo_table` + `self._robot`），建立 `DebugMenu` 並接收開關 callback。

**新增欄位：**
```python
self._demo_table: BilliardTable | None
self._robot: TableRobotManager | None
self._training_enabled: bool = False
self._demo_enabled: bool = False
# self._tables: list[BilliardTable] 維持既有，訓練環境用，永遠無手臂
```

**介面（新增方法）：**
```python
def _on_training_toggle(self, enabled: bool) -> None:
    """接收 DebugMenu 訓練開關 callback，這輪只存狀態。"""
    self._training_enabled = enabled

def _on_demo_toggle(self, enabled: bool) -> None:
    """接收 DebugMenu Demo 開關 callback，這輪只存狀態。"""
    self._demo_enabled = enabled
```

**`_billiard_init()` 變更（Demo 桌建立採 inline 寫法，不獨立成 `_build_demo_table` 方法）：**
- 原設計曾規劃獨立的 `_build_demo_table(stage_api, material_api)` 方法，但實作時發現 Demo 桌建立只有一行（呼叫既有的 `_build_table()` helper），且建立後立刻要用其結果計算 `self._table_unit_side_length`，抽成獨立方法沒有實質好處，因此改為直接寫在 `_billiard_init()` 內：
  1. `demo_table_path = "/World/Table_Demo"`
  2. `self._demo_table = self._build_table(demo_table_path, stage_api, material_api, (0, 0))`（沿用既有的 `_build_table()` helper，訓練桌也用同一個 helper）
  3. `self._table_unit_side_length = self._get_table_side_length(self._demo_table.get_table_prim_path())`
  4. `self._robot = TableRobotManager(self._demo_table.get_table_center(), demo_table_path, stage_api)`（`base_path` 直接傳 `demo_table_path`，不手動加 `/Robot` 後綴，由 `UR5Robot` 內部統一拼接一次）
  5. 接著才呼叫 `self._build_tables(_TABLE_COUNT, stage_api, material_api)` 建立訓練桌

**`_build_tables()` 變更：**
- 移除原本「用第一張訓練桌量測邊長」的邏輯（`if self._table_unit_side_length == 0: ...`），改由 `_billiard_init()` 先用 Demo 桌量出 `self._table_unit_side_length`
- grid 座標計算由 `x_pos = table_unit_side_length * i` 改為 `x_pos = table_unit_side_length * (i + 1)`、`y_pos = table_unit_side_length * j` 改為 `y_pos = table_unit_side_length * (j + 1)`（i、j 皆從概念上的 1 開始）。x、y 兩軸皆刻意位移，使 Demo 桌（原點）的正前後左右格位（x=0 或 y=0 的邊界列/欄）都不會被訓練桌佔用，讓 Demo 桌四周保持淨空，而不只是避免座標完全重疊

**`on_shutdown()` 變更：** 新增 `self._demo_table.destroy()`、`self._robot.destroy()`，與既有 `self._tables` 迴圈、`self._debug_menu.destroy()` 並存。

**依賴：**
- 輸入來源：`StageAPIImpl`、`MaterialAPIImpl`（既有）
- 輸出去向：建立並持有 `BilliardTable`（訓練桌 list + Demo 桌）、`TableRobotManager`、`DebugMenu`

---

### DebugMenu（修改）

**職責：** 提供訓練/Demo 兩個開關的 UI，值變化時呼叫呼叫端傳入的 callback；本身不持有布林狀態（狀態放在呼叫端 `BilliardExtension`）。

**UI 元件變更：** 實作時改用 `omni.ui.ToolButton`（搭配 `SimpleBoolModel` + `extension/ui/ui_style.py` 的 `UiStyle.get_toggle_style()`）取代原規劃的 `omni.ui.CheckBox`，視覺呈現為開/關兩態的開關型按鈕（類似 Extension Manager 的 enable/disable 開關），行為（callback 呼叫時機、不持有狀態）與原設計相同。

**UI 文字語言：** 標籤採**英文**（`"Training"` / `"Break shot demo"`），非原規劃的中文「訓練環境執行中」「Demo 手臂執行中」。原因：此畫面會用於 LinkedIn Demo 影片/截圖，需以英文呈現。

**介面：**
```python
class DebugMenu:
    def __init__(
        self,
        on_training_toggle: Callable[[bool], None],
        on_demo_toggle: Callable[[bool], None],
    ) -> None:
        """建構子新增兩個必要參數，呼叫端傳入自己的方法（於 _build_ui() 之前指派，避免 callback 尚未綁定就被引用）。"""
        ...

    def _build_ui(self) -> None:
        """
        新增兩個 omni.ui.ToolButton（開關型按鈕）：
        - "Training"：model.add_value_changed_fn(
              lambda m: self._on_training_toggle(m.get_value_as_bool())
          )
        - "Break shot demo"：model.add_value_changed_fn(
              lambda m: self._on_demo_toggle(m.get_value_as_bool())
          )
        """
        ...
```

**依賴：**
- 輸入來源：`BilliardExtension` 傳入的 `on_training_toggle` / `on_demo_toggle` callback
- 輸出去向：使用者操作開關按鈕時觸發呼叫端狀態更新

---

## 4. 資料流

```
BilliardExtension._billiard_init()
  → demo_table_path = "/World/Table_Demo"
  → self._demo_table = self._build_table(demo_table_path, stage_api, material_api, (0, 0))
    → 內部：BilliardTable(demo_table_path, stage_api, material_api, (0, 0))
  → self._table_unit_side_length = self._get_table_side_length(self._demo_table.get_table_prim_path())
  → center = self._demo_table.get_table_center()
  → self._robot = TableRobotManager(center, demo_table_path, stage_api)
      → TableRobotManager 內部：world_position = center + _ROBOT_OFFSET_FROM_TABLE_CENTER
      → self._robot = UR5Robot(base_path, stage_api, world_position)
        → stage_api.create_reference_prim(...) + stage_api.set_prim_translate(...)
  → _build_tables(_TABLE_COUNT, stage_api, material_api)
    → for i, j in grid:
        x_pos = table_unit_side_length * (i + 1); y_pos = table_unit_side_length * (j + 1)
        self._tables.append(BilliardTable(f"/World/Table_{index}", stage_api, material_api, (x_pos, y_pos)))
  → self._debug_menu = DebugMenu(
        on_training_toggle=self._on_training_toggle,
        on_demo_toggle=self._on_demo_toggle,
    )
  → 回傳無（初始化完成）

使用者點擊 Debug Menu checkbox
  → DebugMenu 的 CheckBox value_changed callback
  → 呼叫 BilliardExtension._on_training_toggle(enabled) 或 _on_demo_toggle(enabled)
  → 存到 self._training_enabled / self._demo_enabled（尚無其他行為，待未來訓練/Demo 執行功能實作後再接上）

BilliardExtension.on_shutdown()
  → for t in self._tables: t.destroy()
  → self._demo_table.destroy(); self._robot.destroy()
  → self._debug_menu.destroy()
```

---

## 5. 依賴關係圖

```
BilliardExtension
  ├── 依賴 BilliardTable（建立訓練桌 list 與唯一 Demo 桌）
  ├── 依賴 TableRobotManager（建立並持有 Demo 桌的手臂中介層）
  ├── 依賴 DebugMenu（訓練/Demo 開關 UI，傳入自身 callback）
  ├── 依賴 StageAPIImpl、MaterialAPIImpl（既有）
  └── 依賴 TABLE_PATH（既有，經 BilliardTable 間接使用）

TableRobotManager
  ├── 依賴 UR5Robot（唯一對外窗口，私有持有，計算世界座標後建立）
  └── 依賴 StageAPI port（既有）

BilliardTable
  ├── 依賴 StageAPI、MaterialAPI（既有）
  ├── 依賴 TableBallSet（既有）
  ├── 依賴 BreakShotPositionProvider（既有）
  └── （不再依賴 UR5Robot）

UR5Robot
  ├── 依賴 StageAPI（create_reference_prim、set_prim_translate）
  └── 依賴 asset_utility.UR5_PATH（既有，不變）

DebugMenu
  └── 依賴呼叫端傳入的 Callable[[bool], None]（不持有狀態）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 訓練桌 grid 與 Demo 桌（原點）座標重疊，或緊貼 Demo 桌前後左右 | grid 的 i、j 索引皆從概念上的 1 開始（`x_pos = table_unit_side_length * (i + 1)`、`y_pos = table_unit_side_length * (j + 1)`），Demo 桌位置不變，靠訓練桌整體沿 x、y 兩軸平移解決；刻意讓 x=0 與 y=0 兩個邊界列/欄都不被訓練桌佔用，使 Demo 桌四周（前後左右）保持淨空，而非僅避免座標完全重疊 |
| 手臂偏移量是否需依桌子不同而調整 | 目前寫死為 `TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER` 具名常數，不做成建構子參數（YAGNI），等 Milestone B 可達性掃描（#180）有動態需求時再擴充 |
| 外部模組是否可直接持有/呼叫 `UR5Robot` | 不允許；`UR5Robot` 為 `TableRobotManager` 私有欄位，所有呼叫端一律透過 `TableRobotManager` 操作手臂，日後新增手臂操作方法一律加在 `TableRobotManager` |
| Debug Menu 開關觸發後的實際行為（訓練 step / 揮桿軌跡） | 本輪不實作，僅做 UI 與狀態存放（`self._training_enabled` / `self._demo_enabled`），待 Isaac Lab BilliardEnv 訓練迴圈與 Milestone B 揮桿軌跡功能完成後再接上 |
| `_build_tables()` 邊長量測邏輯 | 移除原本「用第一張訓練桌量測」的分支，改由 `_billiard_init()` 先用 Demo 桌統一量測（訓練桌與 Demo 桌共用同一份 `TABLE_PATH` 資產，邊長相同） |

---

## 7. 測試涵蓋（對應 Unit Test）

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| `test_billiard_table_get_table_center` | `core/tests/test_billiard_table.py` | 驗證 `get_table_center()` 回傳值等於建構時傳入的 `position` 延伸為 `(x, y, 0.0)` |
| `test_billiard_table_does_not_create_robot` | `core/tests/test_billiard_table.py` | 驗證 `BilliardTable.__init__` 不再建立 `UR5Robot`（既有的手臂相關測試需一併移除或改寫） |
| `test_table_robot_manager_creates_robot_with_offset_position` | `core/tests/test_table_robot_manager.py`（新） | 驗證以 `table_center + _ROBOT_OFFSET_FROM_TABLE_CENTER` 建立 `UR5Robot` 時，傳入其建構子的世界座標正確 |
| `test_table_robot_manager_get_robot_prim_path` | `core/tests/test_table_robot_manager.py`（新） | 驗證 `get_robot_prim_path()` 正確轉呼叫 `UR5Robot.get_prim_path()` |
| `test_table_robot_manager_destroy` | `core/tests/test_table_robot_manager.py`（新） | 驗證 `destroy()` 正確銷毀內部持有的 `UR5Robot` 實例 |

**豁免/待判斷項目（不在 `core/` pytest 覆蓋範圍，交由 unit-test agent 依專案慣例判斷）：**
- `BilliardExtension` 的 grid 平移邏輯與 Demo 桌建立順序（`extension/` 層，依賴 `omni.ext`、`omni.usd`）
- `DebugMenu` 的 ToolButton callback 呼叫邏輯（UI 層，依賴 `omni.ui`）

---

## 8. 待決定事項

- [ ] `TableRobotManager` 日後新增的手臂操作方法（移動、調整位置等）尚未設計，待 Milestone B 需求明確後再補文件
- [ ] Debug Menu 開關對應的實際訓練 step 迴圈與揮桿觸發邏輯，待 Isaac Lab BilliardEnv 訓練迴圈與 Milestone B 揮桿軌跡功能實作後再設計並更新本文件
