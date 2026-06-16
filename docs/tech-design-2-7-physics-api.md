# [2-7] PhysicsAPI 抽象介面（碰撞偵測、接觸事件）— 技術設計文件

> 生成時間：2026-06-16
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：待 progress-planner 建立後補入

---

## 1. 功能概述

提供引擎無關的碰撞事件訂閱機制。呼叫端（core/services 層）透過 `PhysicsAPI` 介面對指定剛體啟用碰撞報告、訂閱碰撞回呼，並以純 Python dataclass `ContactEvent` 接收每次物理步驟後的碰撞結果（碰撞雙方的 actor/collider path 及衝量純量）。實作層細節封裝於 `extension/isaac_sim_impl_6_0/`，core 層不接觸任何 PhysX API。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `ContactEvent` | core/models | 描述單次碰撞事件的資料容器（無引擎依賴） | `core/models/contact_event.py` |
| `PhysicsAPI` | extension/omniverse_api | 碰撞報告啟用與事件訂閱的抽象介面（ABC） | `extension/omniverse_api/physics_api.py` |
| `PhysicsAPIImpl`（實作層） | extension/isaac_sim_impl_6_0 | 將 PhysX 原始資料翻譯為 `ContactEvent`，管理 subscription 生命週期 | `extension/isaac_sim_impl_6_0/physics_api_impl.py`（使用者自行實作） |

---

## 3. 類別設計

### ContactEvent

**職責：** 描述一次碰撞事件，攜帶碰撞雙方的 actor prim path、collider prim path，以及所有接觸點衝量向量長度的加總純量。無任何引擎依賴，可直接用於 core 層邏輯。

**介面：**
```python
@dataclass
class ContactEvent:
    actor_path_a: str
    # 碰撞的剛體 prim path A，例如 "/World/BilliardTable"

    actor_path_b: str
    # 碰撞的剛體 prim path B，例如 "/World/Ball_1"

    collider_path_a: str
    # 具體碰撞形狀 path A，例如 "/World/BilliardTable/Rail_North"

    collider_path_b: str
    # 具體碰撞形狀 path B，例如 "/World/Ball_1/collision"

    impulse: float
    # 碰撞衝量（N·s），為該次碰撞所有接觸點 impulse 向量長度的加總
```

**依賴：**
- 輸入來源：由 `PhysicsAPI` 實作層從 PhysX 原始資料建構
- 輸出去向：傳入 core/services 層的 callback（例如 `ShotResultProvider._on_contact`）

---

### PhysicsAPI

**職責：** 定義碰撞報告啟用與事件訂閱的抽象契約，隔離 core 層與 PhysX 引擎 API。

**介面：**
```python
class PhysicsAPI(ABC):
    @abstractmethod
    def enable_contact_reporting(self, prim_path: str) -> None:
        """對指定剛體 prim 啟用碰撞報告（套用 PhysxContactReportAPI，threshold=0）"""
        ...

    @abstractmethod
    def subscribe_contact_events(
        self, callback: Callable[[ContactEvent], None]
    ) -> None:
        """訂閱碰撞事件，每次物理步驟後由引擎觸發"""
        ...

    @abstractmethod
    def unsubscribe_contact_events(self) -> None:
        """取消訂閱（將 subscription 物件設為 None）"""
        ...
```

**依賴：**
- 輸入來源：core/services 層（呼叫 `enable_contact_reporting` 與 `subscribe_contact_events`）
- 輸出去向：實作層（`extension/isaac_sim_impl_6_0/`）負責具體執行並回呼 `ContactEvent`

---

## 4. 資料流

```
core/services/ShotResultProvider（startup 階段）
  → physics_api.enable_contact_reporting("/World/BilliardTable")
      → 實作層對 BilliardTable actor 套用 PhysxContactReportAPI（threshold=0）
      → 涵蓋所有子 collision shape：Rail_North/South/East/West + Surface
  → physics_api.enable_contact_reporting("/World/Ball_1")
      → 實作層對 Ball_1 actor 套用 PhysxContactReportAPI（threshold=0）
  → physics_api.subscribe_contact_events(self._on_contact)
      → 實作層呼叫 get_physx_simulation_interface().subscribe_contact_report_events(...)
      → subscription 物件存為 instance variable（防止 GC 回收）

[每次物理步驟後，PhysX 觸發 callback]
  → 實作層 wrapper：
      將 header.actor0/actor1 token → prim path（via PhysicsSchemaTools.intToSdfPath）
      將 contacts 陣列 → 計算各接觸點 impulse 向量長度並加總
      建構 ContactEvent(actor_path_a, actor_path_b, collider_path_a, collider_path_b, impulse)
  → self._on_contact(event: ContactEvent)
      → event.actor_path_a    == "/World/BilliardTable"
      → event.actor_path_b    == "/World/Ball_1"
      → event.collider_path_a == "/World/BilliardTable/Rail_North"
      → event.collider_path_b == "/World/Ball_1/collision"
      → event.impulse         == float（純量衝量）

core/services/ShotResultProvider（shutdown 階段）
  → physics_api.unsubscribe_contact_events()
      → 實作層將 subscription 物件設為 None
```

---

## 5. 依賴關係圖

```
core/services/（ShotResultProvider 等）
  └── 依賴 PhysicsAPI（extension/omniverse_api/physics_api.py）
        └── 回傳 ContactEvent（core/models/contact_event.py）

PhysicsAPIImpl（extension/isaac_sim_impl_6_0/）
  ├── 依賴 omni.physx.get_physx_simulation_interface()（訂閱事件）
  ├── 依賴 PhysxSchema.PhysxContactReportAPI（啟用碰撞報告）
  └── 依賴 PhysicsSchemaTools.intToSdfPath（token → prim path 轉換）
```

---

## 6. 關鍵設計決策

### PhysxContactReportAPI 套在 Actor 層

`enable_contact_reporting("/World/BilliardTable")` 一次呼叫即涵蓋該 actor 下所有子 collision shape（Rail_North/South/East/West + Surface）。不需對每個子形狀逐一呼叫，也不需在 core 層維護子路徑清單。

### impulse 計算方式

同一碰撞對可能有複數接觸點，各接觸點的 impulse 為 (x, y, z) 向量。實作層取每個接觸點的向量長度（L2 norm）後**加總**，作為 `ContactEvent.impulse` 的純量值。

### Callback 接收引擎無關型別

實作層負責將原始 PhysX 資料（uint64 token、接觸點陣列）完整翻譯為 `ContactEvent`。core 層只接收 `ContactEvent`，不引入任何 `omni.*` 或 `pxr.*` 命名空間。

### Subscription 物件生命週期

`get_physx_simulation_interface().subscribe_contact_report_events()` 回傳 subscription 物件。實作層**必須**將此物件存為 instance variable；若只存本地變數，GC 會立即回收並靜默取消訂閱，導致無法收到任何碰撞事件。

---

## 7. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 對同一 prim 重複呼叫 `enable_contact_reporting` | 實作層應先檢查 `PhysxContactReportAPI` 是否已存在，避免重複 Apply |
| `subscribe_contact_events` 在 subscription 已存在時再次呼叫 | 實作層應先呼叫 unsubscribe，或記錄 warning，防止雙重訂閱 |
| `unsubscribe_contact_events` 在尚未訂閱時呼叫 | 實作層應做 None 檢查，不拋出例外 |
| PhysX callback 的 contact 陣列為空 | impulse 加總結果為 0.0，仍建構並傳遞 ContactEvent，由 core 層決定是否過濾 |
| prim_path 指向不存在或非剛體的 prim | 實作層應捕捉例外並記錄錯誤，不中斷其他已訂閱的回呼 |

---

## 8. 測試涵蓋

**Unit Test 豁免。**

`PhysicsAPI` 為純抽象介面（ABC，所有方法皆為 `@abstractmethod`），無任何邏輯可測試。`ContactEvent` 為純 dataclass，無行為邏輯。兩者均符合 `unit-test-rules.md` 的豁免條件。

實作層（`extension/isaac_sim_impl_6_0/`）的整合行為需在 Isaac Sim 運行環境下驗證，不在 Unit Test 範圍內。

---

## 9. 實作層參考（Omniverse API 6.0.0）

以下僅供實作層參考，不屬於 core 層範圍。

```python
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts.physicsUtils import PhysicsSchemaTools
from pxr import PhysxSchema

# enable_contact_reporting 實作
contact_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
contact_api.CreateThresholdAttr().Set(0)

# subscribe_contact_events 實作（subscription 必須存為 instance variable）
self._sub = get_physx_simulation_interface().subscribe_contact_report_events(self._wrapper)

# actor/collider token → prim path
path = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
```

---

## 10. 待決定事項

- [ ] 確認是否需要支援多個 callback 同時訂閱（目前介面設計為單一 callback）
- [ ] `impulse` 過濾閾值是否由 core 層負責（例如忽略 impulse < 0.01 的微小碰撞），或由實作層在 `threshold` 參數處理
