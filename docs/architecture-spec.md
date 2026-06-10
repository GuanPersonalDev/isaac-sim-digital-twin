# 專案架構規範

## 適用範圍
Phase 3 Isaac Sim / Omniverse Kit Extension 專案的架構分層原則。
開發新功能或重構時，以此文件作為職責判斷依據。

---

## 核心原則

**商業邏輯、平台抽象、平台實作三層分離。**

- `core/` 描述「要做什麼」
- `extension/omniverse_api/` 描述「需要平台提供什麼能力（抽象）」
- `extension/isaac_sim_impl_6_0/` 描述「Isaac Sim 6.0 怎麼實現這些能力（實作）」

版本升級時，只需要新增 `isaac_sim_impl_{新版本}/` 並重寫實作層，`core/` 與 `omniverse_api/` 完全不動。

---

## 目錄結構

```
project/
  ├── core/                              ← 商業邏輯層（平台無關）
  │   ├── controllers/
  │   │   ├── controller_base.py         ← 控制策略抽象介面
  │   │   └── script_controller.py       ← 腳本控制實作（Phase 3）
  │   ├── models/
  │   │   ├── machine_state.py           ← 資料模型、狀態定義
  │   │   ├── observation.py             ← Observation 資料格式（預留 RL 接口）
  │   │   └── action.py                  ← Action 資料格式（預留 RL 接口）
  │   ├── services/
  │   │   ├── coordinate_service.py      ← 座標轉換、計算邏輯
  │   │   └── mqtt_service.py            ← MQTT 資料處理邏輯
  │   └── tests/
  │       ├── test_controllers.py
  │       ├── test_coordinate_service.py
  │       └── test_mqtt_service.py
  │
  └── extension/                         ← 平台橋接層
      ├── omniverse_api/                 ← 抽象介面層（版本無關）
      │   ├── stage_api.py               ← 對應 Isaac Sim Stage 概念
      │   ├── articulation_api.py        ← 對應 Articulation 概念
      │   ├── physics_api.py             ← 對應 Physics 查詢概念
      │   ├── rigid_body_api.py          ← 對應 RigidBody 概念
      │   └── ui_api.py                  ← 對應 omni.ui 概念
      ├── isaac_sim_impl_6_0/            ← Isaac Sim 6.0 實作層（升版時整個替換）
      │   ├── stage_api_impl.py
      │   ├── articulation_api_impl.py
      │   ├── physics_api_impl.py
      │   ├── rigid_body_api_impl.py
      │   └── ui_api_impl.py
      └── ui/                            ← UI 元件（透過 ui_api 抽象操作）
          ├── hud_panel.py               ← omni.ui HUD 元件
          └── debug_menu.py              ← 視覺功能手動驗證選單
```

---

## core/ 層規範

### 可以放什麼
- 商業邏輯與決策邏輯
- 狀態機定義與狀態切換邏輯
- 純計算函式（座標轉換、數值處理、格式轉換）
- 資料模型與資料驗證
- 抽象介面定義（`ControllerBase` 等）
- MQTT 訊息的解析與格式化邏輯（不含實際連線）

### 不可以放什麼
- 任何 Isaac Sim / Omniverse 的 import
- `omni.*`、`pxr.*`、`isaacsim.*` 等相關模組
- UI 渲染邏輯
- 場景操作（USD Stage 寫入）

### 測試要求
- 所有非豁免的函式必須有對應的 Unit Test
- 使用 `pytest` 在 WSL2 直接執行，不需要啟動 Isaac Sim
- 外部依賴使用 `unittest.mock` 隔離

---

## extension/omniverse_api/ 層規範

### 定位
以 Isaac Sim 原生概念為粒度定義抽象介面，命名直接對應官方 API 的概念分層。
這一層描述「需要平台提供什麼能力」，不包含任何 Isaac Sim 的具體 import。

### 抽象介面對應表

| 檔案 | 對應 Isaac Sim 概念 | 主要職責 |
|---|---|---|
| `stage_api.py` | Stage / USD Scene | 場景載入、Prim 查詢、Stage 存取 |
| `articulation_api.py` | Articulation | 關節控制、手臂移動、夾爪操作 |
| `physics_api.py` | Physics Scene | 物理查詢、碰撞偵測、接觸判定 |
| `rigid_body_api.py` | RigidBody | 物件位置、速度、姿態查詢 |
| `ui_api.py` | omni.ui | UI 元件建立、事件綁定 |

### 範例
```python
# extension/omniverse_api/articulation_api.py
from abc import ABC, abstractmethod

class ArticulationAPI(ABC):
    """關節控制抽象介面，對應 Isaac Sim Articulation 概念"""

    @abstractmethod
    def set_joint_positions(self, positions: list[float]) -> None:
        """設定所有關節角度"""
        pass

    @abstractmethod
    def get_joint_positions(self) -> list[float]:
        """取得當前所有關節角度"""
        pass

    @abstractmethod
    def set_gripper_state(self, open: bool) -> None:
        """控制夾爪開關"""
        pass
```

### 不可以放什麼
- 任何 `omni.*`、`pxr.*`、`isaacsim.*` import
- 具體的 Isaac Sim API 呼叫
- 商業邏輯

---

## extension/isaac_sim_impl_6_0/ 層規範

### 定位
Isaac Sim 6.0 的具體實作，實作 `omniverse_api/` 定義的所有抽象介面。
資料夾名稱明確標示版本號，升版時新增 `isaac_sim_impl_6_0/` 資料夾，不刪除舊版。

### 規範
- 每個實作檔對應一個抽象介面檔
- 只允許在此層出現 `omni.*`、`pxr.*`、`isaacsim.*` import
- 實作邏輯盡量薄，複雜邏輯應移到 `core/`

### 範例
```python
# extension/isaac_sim_impl_6_0/articulation_api_impl.py
from omni.isaac.core.articulations import Articulation
from ..omniverse_api.articulation_api import ArticulationAPI

class ArticulationAPIImpl(ArticulationAPI):
    """Isaac Sim 6.0 Articulation 實作"""

    def __init__(self, prim_path: str):
        self._articulation = Articulation(prim_path=prim_path)

    def set_joint_positions(self, positions: list[float]) -> None:
        self._articulation.set_joint_positions(positions)

    def get_joint_positions(self) -> list[float]:
        return self._articulation.get_joint_positions().tolist()

    def set_gripper_state(self, open: bool) -> None:
        # Isaac Sim 6.0 夾爪控制實作
        ...
```

### API 升版掃描
升版輔助 sub-agent 以此資料夾為掃描目標。
詳見 `api-migration-agent.md`。

---

## 抽象介面設計

### Controller 抽象介面（Phase 3 → Phase 4 延伸）

```python
# core/controllers/controller_base.py
from abc import ABC, abstractmethod

class ControllerBase(ABC):
    """控制策略抽象介面，ScriptController 與 ModelController 均實作此介面"""

    @abstractmethod
    def get_action(self, observation: dict) -> dict:
        """根據當前觀測值決定動作"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置控制器狀態"""
        pass
```

```python
# core/controllers/script_controller.py（Phase 3 實作）
class ScriptController(ControllerBase):
    def get_action(self, observation: dict) -> dict:
        # 腳本控制邏輯
        ...

    def reset(self) -> None:
        ...
```

```python
# core/controllers/model_controller.py（Phase 4 預留）
class ModelController(ControllerBase):
    def get_action(self, observation: dict) -> dict:
        # RL 模型推論
        ...
```

### UI 資料注入介面

```python
# extension/ui/hud_panel.py
class HudPanel:
    def __init__(self, data_provider):
        # data_provider 是抽象資料來源，開發期間注入假資料
        self._data_provider = data_provider

    def refresh(self):
        data = self._data_provider.get_machine_states()
        # 更新 UI 顯示
```

---

## 預留接口定義（Physical AI 延伸路徑）

| 接口 | 位置 | Phase 3 狀態 | Phase 4 延伸 |
|---|---|---|---|
| Observation 收集 | `core/models/observation.py` | 資料格式定義完成 | 接上訓練資料 pipeline |
| Action Space 定義 | `core/models/action.py` | 格式定義完成 | 接上 RL policy 輸出 |
| Controller 抽換點 | `core/controllers/controller_base.py` | `ScriptController` 實作 | 新增 `ModelController` |

---

## 職責判斷快速參考

| 情境 | 放在哪裡 |
|---|---|
| 判斷機台狀態是 WARNING 還是 ERROR | `core/` |
| 呼叫 Isaac Sim API 移動關節 | `isaac_sim_impl_6_0/` |
| XZ 座標轉換為 2D 像素座標 | `core/` |
| 定義「需要能移動關節」的介面 | `omniverse_api/` |
| 在 omni.ui 上渲染狀態顏色 | `ui/`（透過 `ui_api` 抽象） |
| MQTT 訊息 JSON 解析 | `core/` |
| MQTT Client 連線建立 | `extension/`（非 impl 層） |
| Pick & Place 狀態機邏輯 | `core/` |
| 讀取 USD Stage 的 BBox 數值 | `isaac_sim_impl_6_0/`（結果傳入 `core/`） |
| omni.ui Label 建立 | `isaac_sim_impl_6_0/ui_api_impl.py` |
