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

版本升級時，只需要新增對應版本的 impl 資料夾並重寫實作層，`core/` 與 `omniverse_api/` 完全不動。

---

## 版本資訊

- **Isaac Sim 版本：6.0.0.1**
- **Python：3.12**
- **安裝方式：** `pip install isaacsim[all,extscache]==6.0.0.1 --extra-index-url https://pypi.nvidia.com`
- **Core API：** Isaac Sim 6.0 Warp-based Core Experimental API（PyTorch-based Core API 已廢棄）

---

## 目錄結構

```
project/
  ├── core/                              ← 商業邏輯層（平台無關）
  │   ├── controllers/
  │   │   ├── controller_base.py         ← 控制策略抽象介面
  │   │   └── script_controller.py       ← 腳本控制實作（Phase 3）
  │   ├── models/
  │   │   ├── billiard_state.py          ← 撞球場景狀態資料模型
  │   │   ├── observation.py             ← Observation 資料格式
  │   │   ├── action.py                  ← Action 資料格式
  │   │   └── shot_result.py             ← 擊球結果資料模型
  │   ├── services/
  │   │   ├── ball_position_provider.py  ← 取得球位置的抽象介面
  │   │   ├── break_shot_position_provider.py ← 衝球固定位置實作
  │   │   ├── reward_service.py          ← Reward Function 計算邏輯
  │   │   └── mqtt_service.py            ← MQTT 資料處理邏輯
  │   └── tests/
  │       ├── test_controllers.py
  │       ├── test_reward_service.py
  │       ├── test_ball_position_provider.py
  │       └── test_models.py
  │
  └── extension/                         ← 平台橋接層
      ├── omniverse_api/                 ← 抽象介面層（版本無關）
      │   ├── stage_api.py               ← 對應 Isaac Sim Stage 概念
      │   ├── articulation_api.py        ← 對應 Articulation 概念
      │   ├── physics_api.py             ← 對應 Physics 查詢概念
      │   ├── rigid_body_api.py          ← 對應 RigidBody 概念
      │   └── ui_api.py                  ← 對應 omni.ui 概念
      ├── isaac_sim_impl_6_0/            ← Isaac Sim 6.0 實作層（升版時新增資料夾）
      │   ├── stage_api_impl.py
      │   ├── articulation_api_impl.py
      │   ├── physics_api_impl.py
      │   ├── rigid_body_api_impl.py
      │   ├── ui_api_impl.py
      │   └── live_position_provider.py  ← 即時查詢球位置（依賴 RigidBodyAPI）
      └── ui/                            ← UI 元件（透過 ui_api 抽象操作）
          ├── hud_panel.py               ← omni.ui HUD 元件
          └── debug_menu.py              ← 視覺功能手動驗證選單
```

---

## core/ 層規範

### 可以放什麼
- 商業邏輯與決策邏輯
- 狀態機定義與狀態切換邏輯
- 純計算函式（座標轉換、數值處理、Reward 計算）
- 資料模型與資料驗證
- 抽象介面定義（`ControllerBase`、`BallPositionProvider` 等）
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

以 Isaac Sim 原生概念為粒度定義抽象介面，命名直接對應官方 API 的概念分層。

| 檔案 | 對應 Isaac Sim 6.0 概念 | 主要職責 |
|---|---|---|
| `stage_api.py` | Stage / USD Scene | 場景載入、Prim 查詢 |
| `articulation_api.py` | Articulation | 關節控制、手臂移動 |
| `physics_api.py` | Physics Scene | 碰撞偵測、接觸事件 |
| `rigid_body_api.py` | RigidBody | 球的位置、速度查詢 |
| `ui_api.py` | omni.ui | UI 元件建立、事件綁定 |

---

## extension/isaac_sim_impl_6_0/ 層規範

Isaac Sim 6.0 的具體實作，使用 Warp-based Core Experimental API。

- 資料夾名稱明確標示版本號
- 只允許在此層出現 `omni.*`、`pxr.*`、`isaacsim.*` import
- 實作邏輯盡量薄，複雜邏輯應移到 `core/`
- 升版時新增 `isaac_sim_impl_6_1/` 等資料夾，不刪除舊版

---

## 撞球桌台座標系規範

所有球的位置均以**桌台相對座標**表示，與 Isaac Sim 世界座標無關。

### 定義

| 項目 | 定義 |
|---|---|
| 原點 | 桌面幾何中心 |
| +X 方向 | 向右（站在 head end 朝 foot end 的右手側） |
| +Y 方向 | 朝 foot end（rack 側，開球菱形所在方向） |
| +Z 方向 | 向上（右手定則，Z-up） |

此座標系為標準右手座標系 Z-up，從 +Z 俯視與 Isaac Sim 預設慣例完全對齊。

### 固定邊界值

| 項目 | 值 |
|---|---|
| 桌面 Y 範圍 | -1.27 ~ +1.27 m |
| 桌面 X 範圍 | -0.635 ~ +0.635 m |
| Foot spot（1-ball 位置） | (0, +0.635) |
| Head string（Kitchen 邊界） | Y = -0.635 |
| Kitchen 範圍 | Y ∈ (-1.27, -0.635) |

### 注意事項

`LivePositionProvider`（Task 7-2）從 Isaac Sim 取得的是世界座標，必須搭配桌台 prim 的 transform 才能轉換為桌台相對座標。轉換方式（建構子注入 vs 即時查詢）於 Task 7-2 設計時決定。

---

## BallPositionProvider 抽象設計

取得球位置的資料來源可抽換，上層邏輯完全不知道資料來自固定值還是即時查詢。

```python
# core/services/ball_position_provider.py
from abc import ABC, abstractmethod

class BallPositionProvider(ABC):
    """取得各球位置的抽象介面"""

    @abstractmethod
    def get_positions(self) -> dict:
        """
        回傳各球的 XY 位置
        格式：{
            'cue': (x, y),
            1: (x, y), ..., 9: (x, y)
        }
        """
        pass
```

```python
# core/services/break_shot_position_provider.py
class BreakShotPositionProvider(BallPositionProvider):
    """衝球用：直接回傳 9-ball 標準開球固定位置（菱形排列）"""

    def get_positions(self) -> dict:
        return BREAK_SHOT_POSITIONS  # 固定值，不查詢 Isaac Sim
```

```python
# extension/isaac_sim_impl_6_0/live_position_provider.py
class LivePositionProvider(BallPositionProvider):
    """即時查詢：從 Isaac Sim RigidBodyAPI 取得當前位置"""

    def __init__(self, rigid_body_api: RigidBodyAPI):
        self._rigid_body_api = rigid_body_api

    def get_positions(self) -> dict:
        # 從仿真環境即時查詢
        ...
```

---

## Controller 抽象介面（Phase 3 → Phase 4 延伸）

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

Phase 3 實作 `ScriptController`，Phase 4 新增 `ModelController`（載入 RL 訓練模型），高層邏輯完全不動。

---

## 預留接口定義（Physical AI 延伸路徑）

| 接口 | 位置 | Phase 3 狀態 | Phase 4 延伸 |
|---|---|---|---|
| Observation 收集 | `core/models/observation.py` | 格式定義完成 | 接上訓練資料 pipeline |
| Action Space 定義 | `core/models/action.py` | 格式定義完成 | 接上 RL policy 輸出 |
| Controller 抽換點 | `core/controllers/controller_base.py` | `ScriptController` 實作 | 新增 `ModelController` |
| BallPositionProvider | `core/services/ball_position_provider.py` | `BreakShotPositionProvider` 實作 | `LivePositionProvider` 供其他情境使用 |

---

## 職責判斷快速參考

| 情境 | 放在哪裡 |
|---|---|
| Reward Function 計算 | `core/services/reward_service.py` |
| 衝球固定球位資料 | `core/services/break_shot_position_provider.py` |
| 呼叫 Isaac Sim API 移動關節 | `isaac_sim_impl_6_0/articulation_api_impl.py` |
| 即時查詢球的位置 | `isaac_sim_impl_6_0/live_position_provider.py` |
| XY 座標正規化計算 | `core/` |
| 定義「需要能移動關節」的介面 | `omniverse_api/articulation_api.py` |
| Pick & Place 狀態機邏輯 | `core/controllers/script_controller.py` |
| omni.ui Label 建立 | `isaac_sim_impl_6_0/ui_api_impl.py` |
