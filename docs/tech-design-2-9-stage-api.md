# [2-9] StageAPI 抽象介面（USD Stage 結構查詢）— 技術設計文件

> 生成時間：2026-06-18
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：（待 progress-planner 建立 Issue 後補入）

---

## 1. 功能概述

提供查詢 USD Stage 場景結構的抽象介面：確認特定 prim 是否存在、取得某路徑下的直接子 prim 列表。呼叫端（`core/services/`）透過此介面判斷場景是否備妥（例如確認球的 prim 是否已載入、枚舉場景中的所有球），無需直接依賴任何 Omniverse / Isaac Sim API。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `StageAPI` | extension/omniverse_api（ABC） | 定義 USD Stage 結構查詢的抽象契約 | `extension/omniverse_api/stage_api.py` |
| `StageAPIImpl`（使用者實作） | extension/isaac_sim_impl_6_0 | 以 Isaac Sim 6.0.0 Omniverse API 實作上述兩個方法 | `extension/isaac_sim_impl_6_0/stage_api_impl.py` |

---

## 3. 類別設計

### StageAPI

**職責：** 定義 USD Stage 場景結構查詢的抽象介面，使 `core` 層與引擎實作完全解耦。

**介面：**
```python
from abc import ABC, abstractmethod

class StageAPI(ABC):

    @abstractmethod
    def prim_exists(self, prim_path: str) -> bool:
        """確認指定路徑的 prim 是否存在於 Stage"""
        ...

    @abstractmethod
    def get_prim_paths(self, parent_path: str) -> list[str]:
        """回傳指定路徑下所有直接子 prim 的路徑列表"""
        ...
```

**依賴：**
- 輸入來源：呼叫端傳入 `prim_path` / `parent_path`（USD 路徑字串，例如 `"/World/Ball_1"`、`"/World/Balls"`）
- 輸出去向：`core/services/` 中需要確認場景結構或枚舉 prim 的服務

---

### StageAPIImpl（Isaac Sim 6.0.0 實作層，使用者自行實作）

**職責：** 以 Omniverse API 實作抽象介面，提供兩種等效方案（Isaac Sim 工具函式 / 直接使用 `pxr.Usd`）。

**實作參考（主要方案，Isaac Sim 工具函式）：**
```python
from isaacsim.core.utils.prims import (
    is_prim_path_valid,
    get_prim_at_path,
    get_prim_children,
)
from extension.omniverse_api.stage_api import StageAPI

class StageAPIImpl(StageAPI):

    def prim_exists(self, prim_path: str) -> bool:
        return is_prim_path_valid(prim_path)

    def get_prim_paths(self, parent_path: str) -> list[str]:
        parent = get_prim_at_path(parent_path)
        return [str(c.GetPath()) for c in get_prim_children(parent)]
```

**備用方案（直接使用 `pxr.Usd`，更穩定）：**
```python
import omni.usd

class StageAPIImpl(StageAPI):

    def prim_exists(self, prim_path: str) -> bool:
        stage = omni.usd.get_context().get_stage()
        return stage.GetPrimAtPath(prim_path).IsValid()

    def get_prim_paths(self, parent_path: str) -> list[str]:
        stage = omni.usd.get_context().get_stage()
        parent = stage.GetPrimAtPath(parent_path)
        return [str(c.GetPath()) for c in parent.GetChildren()]
```

**依賴：**
- 輸入來源：`prim_path` / `parent_path` 字串（由呼叫端提供）
- 輸出去向：回傳標準 `bool` / `list[str]` 給 `core` 層使用

---

## 4. 資料流

```
core/services/
  │
  ├─ stage_api.prim_exists("/World/Ball_1")
  │    → is_prim_path_valid("/World/Ball_1")（或 stage.GetPrimAtPath().IsValid()）
  │    → True / False
  │    → 呼叫端決定是否繼續執行後續邏輯
  │
  └─ stage_api.get_prim_paths("/World/Balls")
       → get_prim_at_path("/World/Balls") → parent prim
       → get_prim_children(parent) → 直接子 prim 列表（非遞迴）
       → [str(c.GetPath()) for c in children]
       → ["/World/Balls/Ball_1", "/World/Balls/Ball_2", ...]
       → 呼叫端自行決定是否需要多層查詢
```

---

## 5. 依賴關係圖

```
core/services/（場景備妥確認、prim 枚舉）
  └── 依賴 StageAPI（ABC）（查詢 USD Stage 結構）

extension/omniverse_api/StageAPI（ABC）
  └── 無引擎依賴（純 Python ABC）

extension/isaac_sim_impl_6_0/StageAPIImpl
  ├── 依賴 isaacsim.core.utils.prims（主要方案，Isaac Sim 6.0.0 工具函式）
  └── 依賴 omni.usd / pxr.Usd（備用方案，更底層穩定）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| `prim_path` 不存在 | `prim_exists` 直接回傳 `False`，符合語義；不拋出例外 |
| `parent_path` 不存在時呼叫 `get_prim_paths` | `get_prim_at_path` 可能回傳 invalid prim，`get_prim_children` 行為未定義；實作層應先呼叫 `prim_exists` 確認或以 try/except 包覆，回傳空列表 `[]` |
| `parent_path` 存在但無子 prim | 回傳空列表 `[]`，為合法結果，呼叫端自行處理 |
| Stage 尚未載入（`get_stage()` 回傳 None） | 備用方案中 `stage.GetPrimAtPath()` 會拋出例外；實作層應加 None 檢查，必要時回傳預設值或向上傳遞 |
| `get_prim_paths` 結果含隱藏 prim（Instance Proxy 等） | 回傳所有 `GetChildren()` 結果，不過濾；若呼叫端有需要可自行過濾 prim type |

---

## 7. 測試涵蓋

**Unit Test 豁免：** `StageAPI` 為純 ABC，無實作邏輯，依 `unit-test-rules.md` 豁免 Unit Test。

`StageAPIImpl` 的整合測試屬 Isaac Sim 環境依賴，不在 `core/tests/` 覆蓋範圍內。

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| （豁免） | — | ABC 無可測試的實作邏輯 |

---

## 8. 關鍵設計決策

1. **只涵蓋結構查詢**：位置／速度由 `RigidBodyAPI` 涵蓋，碰撞事件由 `PhysicsAPI` 涵蓋，`StageAPI` 職責嚴格限定於 USD 場景結構（存在性、子層列表），避免職責蔓延。

2. **`get_prim_paths` 只取直接子層（非遞迴）**：避免場景結構複雜時因全樹走訪造成效能問題，呼叫端自行決定是否需要多層查詢。

3. **兩套實作方案並列**：主要方案使用 Isaac Sim 工具函式（`isaacsim.core.utils.prims`），備用方案直接使用 `pxr.Usd`；後者更底層穩定，在工具函式升版異動時可快速切換。

---

## 9. 待決定事項

- [ ] 實作層最終採用主要方案（Isaac Sim 工具函式）或備用方案（`pxr.Usd` 直接存取），待使用者驗證 6.0.0 環境中 `is_prim_path_valid` 可用性後確認
- [ ] `get_prim_paths` 是否需要過濾特定 prim type（例如排除 `_class_` 或 `_over_`），視呼叫端需求決定
