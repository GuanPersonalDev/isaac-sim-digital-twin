# [3-3] 撞球場景球體管理（TableBallSet）— 技術設計文件

> 生成時間：2026-06-21
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：（待 progress-planner 建立 Issue 後補入）

---

## 1. 功能概述

在 USD Stage 上建立並管理 10 顆撞球（白球 0 + 1–9 號球）的完整生命週期：以 `ball_template.usd` 作為 Reference 實例化各球 Prim、依號碼套用對應顏色材質（1–8 號純色 UsdPreviewSurface；9 號條紋 MDL Shader）、支援單顆球的顯示／隱藏（進洞效果），以及全場重置（回到指定起始座標）。呼叫端（`core/services/`）透過 `TableBallSet` 抽象介面操作，不依賴任何 Omniverse / Isaac Sim API。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `ball_template.usd` | assets | 球幾何體（radius=0.028575 m）+ 物理材質（mass=0.163 kg）的基底 USD Asset | `assets/ball_template.usd` |
| `ball_stripe.mdl` | assets | 9 號球條紋 Shader：V ∈ [0.35, 0.65] 渲染白色，其餘渲染 stripe_color（黃） | `assets/ball_stripe.mdl` |
| `BALL_COLORS` | core/models | 號碼→RGB 顏色的純資料常數（平台無關），key 用 int，0=白球 | `core/models/ball_colors.py` |
| `StageAPI`（擴充） | extension/omniverse_api（ABC） | 在原有介面上新增 `create_reference_prim`、`set_visibility` 兩個抽象方法 | `extension/omniverse_api/stage_api.py` |
| `TableBallSet` | extension/omniverse_api（ABC） | 球集合操作的抽象介面：build / hide_ball / show_ball / reset | `extension/omniverse_api/table_ball_set.py` |
| `TableBallSetImpl` | extension/isaac_sim_impl_6_0 | 以 Isaac Sim 6.0 API 實作 `TableBallSet`，整合 `StageAPI`、`BALL_COLORS`、MDL Shader 設定 | `extension/isaac_sim_impl_6_0/table_ball_set_impl.py` |

---

## 3. 類別設計

### BALL_COLORS（core/models/ball_colors.py）

**職責：** 提供撞球號碼到 RGB 顏色的純 Python 對應表，不含任何 Omniverse 型別，可在 Unit Test 中直接使用。

**介面：**
```python
BALL_COLORS: dict[int, list[float]] = {
    0: [1.0, 1.0, 1.0],  # 白（白球）
    1: [1.0, 1.0, 0.0],  # 黃
    2: [0.0, 0.0, 1.0],  # 藍
    3: [1.0, 0.0, 0.0],  # 紅
    4: [0.5, 0.0, 0.5],  # 紫
    5: [1.0, 0.5, 0.0],  # 橙
    6: [0.0, 0.5, 0.0],  # 綠
    7: [0.5, 0.0, 0.0],  # 深紅
    8: [0.0, 0.0, 0.0],  # 黑
    9: [1.0, 1.0, 0.0],  # 黃（條紋球，MDL 端處理顏色）
}
```

**依賴：**
- 輸入來源：無（靜態常數）
- 輸出去向：`TableBallSetImpl`（build 時讀取 diffuseColor）

---

### StageAPI（extension/omniverse_api/stage_api.py，擴充）

**職責：** 在原有 Stage 結構查詢介面上，新增建立 Reference Prim 與控制可見性的抽象方法。

**介面（新增部分）：**
```python
@abstractmethod
def create_reference_prim(self, prim_path: str, asset_path: str) -> None:
    """
    在 prim_path 建立一個 Prim，並以 asset_path 指向的 USD 檔案作為 Reference。
    若 prim_path 已存在則覆蓋 Reference。
    """
    ...

@abstractmethod
def set_visibility(self, prim_path: str, visible: bool) -> None:
    """
    設定 prim_path 的可見性。
    visible=True  → MakeVisible()
    visible=False → MakeInvisible()
    """
    ...
```

**依賴：**
- 輸入來源：`TableBallSetImpl`（呼叫端）
- 輸出去向：USD Stage

---

### StageAPIImpl（extension/isaac_sim_impl_6_0/stage_api_impl.py，擴充）

**職責：** 以 Omniverse Python API 實作 `StageAPI` 的兩個新方法。

**實作邏輯：**
```python
def create_reference_prim(self, prim_path: str, asset_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim(prim_path)
    prim.GetReferences().AddReference(asset_path)

def set_visibility(self, prim_path: str, visible: bool) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
```

**依賴：**
- 輸入來源：`TableBallSetImpl`
- 輸出去向：`omni.usd`、`UsdGeom`

---

### TableBallSet（extension/omniverse_api/table_ball_set.py）

**職責：** 定義球集合操作的抽象契約，使 `core` 層與引擎實作完全解耦。

**介面：**
```python
from abc import ABC, abstractmethod

class TableBallSet(ABC):
    @abstractmethod
    def build(self, positions: dict[int, tuple[float, float]]) -> None:
        """
        在 Stage 上建立 10 顆球的 Prim 並套用材質。
        positions: {ball_id: (x, y)}，XY 為桌台相對座標（單位 m）。
        Z 軸由實作層自行計算（table_z + radius）。
        """
        ...

    @abstractmethod
    def hide_ball(self, ball_id: int) -> None:
        """隱藏指定號碼的球（進洞效果）。"""
        ...

    @abstractmethod
    def show_ball(self, ball_id: int) -> None:
        """顯示指定號碼的球。"""
        ...

    @abstractmethod
    def reset(self, positions: dict[int, tuple[float, float]]) -> None:
        """
        將全部球設為可見，並移回 positions 指定的座標。
        positions 格式同 build()。
        """
        ...
```

**依賴：**
- 輸入來源：`core/services/`（場景初始化、RL reset loop）
- 輸出去向：`TableBallSetImpl`（由 DI 注入具體實作）

---

### TableBallSetImpl（extension/isaac_sim_impl_6_0/table_ball_set_impl.py）

**職責：** 以 Isaac Sim 6.0 API 實作 `TableBallSet`，整合 Reference 建立、UsdPreviewSurface 材質、MDL Shader、座標設定。

**介面：**
```python
class TableBallSetImpl(TableBallSet):
    def __init__(self, stage_api: StageAPI, table_z: float, ball_radius: float = 0.028575) -> None:
        """
        table_z: 桌面 Z 軸高度（單位 m），用於計算球心 Z = table_z + ball_radius。
        ball_radius: 球半徑，預設 0.028575 m（直徑 57.15 mm）。
        """
        ...

    def build(self, positions: dict[int, tuple[float, float]]) -> None: ...
    def hide_ball(self, ball_id: int) -> None: ...
    def show_ball(self, ball_id: int) -> None: ...
    def reset(self, positions: dict[int, tuple[float, float]]) -> None: ...
```

**內部 Prim 路徑慣例：**
- 球 n → `/World/Balls/ball_{n}`

**build 內部邏輯：**
1. 對 ball_id 0–9，各呼叫 `stage_api.create_reference_prim("/World/Balls/ball_{id}", "assets/ball_template.usd")`
2. 設定各球 XYZ 位移（Z = table_z + ball_radius）
3. ball_id 0–8：建立 `UsdPreviewSurface`，設 `diffuseColor = Gf.Vec3f(*BALL_COLORS[ball_id])`
4. ball_id 9：建立 `UsdShade.Shader` 指向 `assets/ball_stripe.mdl`，傳入 `stripe_color=Gf.Vec3f(1,1,0)`（黃）、`base_color=Gf.Vec3f(1,1,1)`（白），連接 surface / displacement / volume 三個 output

**依賴：**
- 輸入來源：`StageAPI`（DI 注入）、`BALL_COLORS`（直接 import）
- 輸出去向：USD Stage（透過 `StageAPI`）

---

## 4. 資料流

### 場景初始化

```
[core/services/ 場景初始化]
    → TableBallSet.build(positions: {0: (x,y), ..., 9: (x,y)})
        → [TableBallSetImpl]
            → StageAPI.create_reference_prim("/World/Balls/ball_{id}", "assets/ball_template.usd") × 10
            → 計算 Z = table_z + ball_radius，設定各球位移
            → ball 0–8：建立 UsdPreviewSurface，設 diffuseColor = BALL_COLORS[ball_id]
            → ball 9：建立 MDL Shader（assets/ball_stripe.mdl），傳入 stripe_color/base_color，連接三個 output
    → Stage 上出現 10 顆已套色的球
```

### 球進洞

```
[RL 環境 / 物理偵測]
    → TableBallSet.hide_ball(ball_id: int)
        → [TableBallSetImpl]
            → StageAPI.set_visibility("/World/Balls/ball_{ball_id}", visible=False)
    → 對應球從視覺上消失（Prim 仍存在於 Stage）
```

### 重置

```
[RL reset loop / 場景重置]
    → TableBallSet.reset(positions: {0: (x,y), ..., 9: (x,y)})
        → [TableBallSetImpl]
            → StageAPI.set_visibility("/World/Balls/ball_{id}", visible=True) × 10
            → 設定各球位移回 positions 指定座標
    → 10 顆球全部可見並回到起始位置
```

---

## 5. 依賴關係圖

```
core/services/（場景初始化 / RL loop）
  └── 依賴 TableBallSet（ABC）（抽象契約，DI 注入）

TableBallSetImpl
  ├── 依賴 StageAPI（ABC）（USD Stage 操作，DI 注入）
  ├── 依賴 BALL_COLORS（純資料常數，直接 import）
  └── 依賴 assets/ball_stripe.mdl（MDL 檔案路徑字串）

StageAPIImpl
  ├── 依賴 omni.usd（取得 Stage context）
  └── 依賴 UsdGeom.Imageable（可見性控制）

assets/ball_template.usd
  └── 被 StageAPIImpl.create_reference_prim 透過路徑字串引用

assets/ball_stripe.mdl
  └── 被 TableBallSetImpl.build 透過路徑字串引用（ball_id=9 時）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| `hide_ball` 傳入不存在的 ball_id（如 10） | `TableBallSetImpl` 應在操作前檢查 ball_id 範圍（0–9），超出範圍拋出 `ValueError` |
| `build` 前呼叫 `hide_ball` / `show_ball` / `reset` | `TableBallSetImpl` 內部記錄 `_built: bool` 旗標，未 build 時拋出 `RuntimeError` |
| `create_reference_prim` 時 asset_path 不存在 | 由 USD 層回報警告（不中斷執行）；建議在 build 前以 `os.path.exists` 確認 ball_template.usd 存在 |
| `positions` 缺少某顆球的座標（如只給 0–8） | `build` / `reset` 應在呼叫前驗證 key 集合為 {0,...,9}，否則拋出 `ValueError` |
| 9 號球 MDL Shader 載入失敗（mdl 檔不存在） | 記錄 warning log，fallback 為 UsdPreviewSurface + BALL_COLORS[9] 的純黃色，不中斷場景初始化 |
| `reset` 時部分球仍在進洞動畫中 | 直接呼叫 `set_visibility(True)` + 移動座標，不等待動畫；由 RL 環境控制呼叫時機 |

---

## 7. 測試涵蓋（對應 Unit Test）

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| `test_ball_colors_keys` | `core/tests/test_ball_colors.py` | 驗證 BALL_COLORS 包含 key 0–9，無多餘 key |
| `test_ball_colors_rgb_range` | `core/tests/test_ball_colors.py` | 驗證所有 RGB 值均在 [0.0, 1.0] 範圍內 |
| `test_build_calls_create_reference_prim_10_times` | `core/tests/test_table_ball_set.py` | Mock StageAPI，確認 build 呼叫 create_reference_prim 恰好 10 次 |
| `test_build_sets_correct_z` | `core/tests/test_table_ball_set.py` | 確認球心 Z = table_z + ball_radius（不依賴 Omniverse） |
| `test_hide_ball_calls_set_visibility_false` | `core/tests/test_table_ball_set.py` | Mock StageAPI，確認 hide_ball 以 visible=False 呼叫正確 prim_path |
| `test_show_ball_calls_set_visibility_true` | `core/tests/test_table_ball_set.py` | Mock StageAPI，確認 show_ball 以 visible=True 呼叫正確 prim_path |
| `test_reset_makes_all_balls_visible` | `core/tests/test_table_ball_set.py` | Mock StageAPI，確認 reset 對 10 顆球均呼叫 set_visibility(True) |
| `test_hide_ball_invalid_id_raises` | `core/tests/test_table_ball_set.py` | 傳入 ball_id=10 應拋出 ValueError |
| `test_build_missing_position_raises` | `core/tests/test_table_ball_set.py` | positions 僅含 0–8（缺 9）應拋出 ValueError |
| `test_build_before_hide_raises` | `core/tests/test_table_ball_set.py` | 未呼叫 build 直接呼叫 hide_ball 應拋出 RuntimeError |

---

## 8. 待決定事項

- [ ] `ball_template.usd` 由 GUI 手動建立後存檔，確認 Asset 路徑為相對路徑或絕對路徑（影響 `AddReference` 的參數格式）
- [ ] `ball_stripe.mdl` 的具體 UV 採樣語法需在 Isaac Sim 6.0 環境下驗證（V 分量讀取方式）
- [ ] `TableBallSetImpl.build` 中 9 號球 MDL Shader 的三個 output 連接語法（surface / displacement / volume）需確認 UsdShade API 版本相容性
- [ ] 桌面 Z 軸高度（`table_z`）的來源：由場景初始化服務從 Stage 查詢後傳入，或以固定常數設定，待場景結構確認後決定
