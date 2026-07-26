# Debug Menu — Training / Break Shot Demo 動態桌子管理與狀態顯示 — 技術設計文件

> 生成時間：2026-07-26
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：尚未建立（下一步由 progress-planner 建立 Issue）
> 關聯文件：`docs/tech-design-2-10-debug-menu.md`（Debug Menu 骨架，第 9 節「候補功能構想」即為本次任務的緣起）、`docs/tech-design-5-9-table-orchestrator.md`（`TableRuntime`／`TableOrchestrator`／`ObservationBuilder` 現況設計）

---

## 1. 功能概述

Debug Menu 的 Training／Break shot demo 兩個開關，目前只把布林值存進 `self._training_enabled`/`self._demo_enabled`（`billiard_digital_twin.py:151-155`），完全沒有接下游行為；且初始值硬編碼為 `False`，USD 場景與各項 runtime 資源（狀態機、`ObservationBuilder`、`PocketEventHandler`）的建立/清理邏輯目前綁在 Extension 啟動與 Timeline PLAY 事件上，`on_shutdown()` 才做一次性整批清理，過程中還漏了 `self._table_runtimes` 完全沒被清理、`BilliardTable.destroy()`/`TableRobotManager.destroy()` 也只清空 Python 參照、沒有真的移除 USD prim。

本次任務把 Toggle 改造成「隨時可切換、立即生效、完全解耦於 Timeline」的桌子生命週期開關：Training 開＝建立所有訓練桌（USD 場景 + 狀態機 + `ObservationBuilder` + `PocketEventHandler`），關＝完整刪除（含 USD prim 與所有已註冊訂閱）；Demo 同理，額外處理機器手臂/球桿與 `ArticulationAPI` 的分階段初始化（因為 tensor-based Articulation API 必須等 Timeline Play 之後才能 `initialize()`）。Extension 啟動時兩個開關預設為 `True`，行為等同開機自動建立所有桌子。同時 Debug Menu 新增「選擇特定桌子」的下拉選單，選中後即時顯示該桌 `BilliardStatus`（狀態機狀態）與每個 tick 的 `Observation`（`is_ball_moving`/`is_motion_complete`/`has_error`/母球座標），並提供一個進階 toggle 顯示每顆球的線速度／角速度，為未來 8→128 張訓練桌的規模化預留「一次只看一張」的可用介面。

核心架構決策：把「一張桌子」的所有相關資源（`BilliardTable` + `TableRuntime` + `PocketEventHandler`）封裝成新的 core 層類別 `TableSession`（以 `table_id`＝prim path 當唯一識別，取代原本三個平行 list 的隱含對應），Demo 桌額外用子類別 `DemoTableSession` 封裝 `TableRobotManager` 與 `ArticulationAPI`，處理 Articulation 分階段建立與銷毀順序。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `TableSession`（新增） | core/services | 封裝單一桌子的 `BilliardTable`＋`TableRuntime`＋`PocketEventHandler`，以 `table_id` 提供狀態查詢、逐球速度查詢、完整銷毀（含 USD prim 移除） | `core/services/table_session.py` |
| `DemoTableSession`（新增） | core/services | `TableSession` 子類別，額外封裝 `TableRobotManager`＋`ArticulationAPI`，處理 Articulation 分階段 `initialize()` 與銷毀時序 | `core/services/table_session.py`（同檔） |
| `TableRuntime`（修改） | core/services | `tick()` 內把 `ObservationBuilder.build()` 的結果快取進 `self._last_observation`；新增 `get_current_state()`/`get_last_observation()` | `core/services/table_runtime.py` |
| `TableOrchestrator`（修改） | core/services | 新增 `get_current_state()`，轉發 `self._script_controller.get_current_state()` | `core/services/table_orchestrator.py` |
| `BilliardTable`（修改） | core/models | 建構子保存 `self._stage_api`；`destroy()` 改為真的呼叫 `stage_api.remove_prim()` 移除 USD prim | `core/models/billiard_table.py` |
| `TableRobotManager`（修改） | core/models | 建構子保存 `self._stage_api`；`destroy()` 改為真的呼叫 `stage_api.remove_prim()` 移除手臂/球桿 prim | `core/models/table_robot_manager.py` |
| `StageAPI`（修改，port） | core/ports | 新增抽象方法 `remove_prim(prim_path: str) -> None` | `core/ports/stage_api.py` |
| `ArticulationAPI`（修改，port） | core/ports | 新增抽象方法 `cancel_pending_home_capture() -> None`，供銷毀 Demo 桌時避免一次性 home-capture callback 對著已刪除的 prim 觸發 | `core/ports/articulation_api.py` |
| `StageAPIImpl`（修改） | extension/isaac_sim_impl_6_0 | `remove_prim()` 呼叫 `isaacsim.core.utils.prims.delete_prim(prim_path)` | `extension/isaac_sim_impl_6_0/stage_api_impl.py` |
| `ArticulationAPIImpl`（修改） | extension/isaac_sim_impl_6_0 | `cancel_pending_home_capture()`：若 `initialize()` 註冊的一次性 home-capture callback 尚未觸發，`deregister_callback()` 取消它 | `extension/isaac_sim_impl_6_0/articulation_api_impl.py` |
| `BilliardExtension`（修改） | extension/billiard_digital_twin | Toggle 完全解耦於 Timeline；管理 `_training_sessions`/`_demo_sessions`；tick callback 於 `_billiard_init()` 就註冊一次（不再綁 Timeline PLAY）；移除 `RuntimeState`，改用 `self._timeline_playing: bool`；提供 `get_table_ids()`/`get_table_debug_info()`/`get_ball_velocities_text()` 供 `DebugMenu` 注入查詢 | `extension/billiard_digital_twin/billiard_digital_twin.py` |
| `DebugMenu`（修改） | extension/ui | 新增選桌下拉選單（預設不選）、狀態/Observation 顯示區塊（每幀更新）、逐球速度進階 toggle | `extension/ui/debug_menu.py` |

`TableSession`/`DemoTableSession` 屬於「組合多個既有 core 資源的協調物件」，與既有 `TableOrchestrator`/`ObservationBuilder` 同層，但職責更上層——`TableRuntime` 是無狀態的 tick 容器，`TableSession` 則是「一張桌子」在其整個生命週期（建立→執行→銷毀）的擁有者。

---

## 3. 類別設計

### StageAPI.remove_prim（port，新增）

**職責：** 抽象出「移除指定路徑的 prim 及其所有子節點」的能力，供 `BilliardTable.destroy()`/`TableRobotManager.destroy()` 呼叫。

```python
class StageAPI(ABC):
    @abstractmethod
    def remove_prim(self, prim_path: str) -> None:
        """移除指定路徑的 prim 及其所有子節點（含所有 local layer 的 spec）"""
        ...
```

**依賴：**
- 輸入來源：`BilliardTable.destroy()`、`TableRobotManager.destroy()` 呼叫
- 輸出去向：`StageAPIImpl.remove_prim()` 實作

---

### StageAPIImpl.remove_prim（extension，新增，屬第三方 API 薄層）

**職責：** 用 Isaac Sim 6.0.0 官方標準做法刪除 prim，明確不用 `pxr.Usd.Stage.RemovePrim()`（社群回報有時候刪不乾淨、殘留 spec）。

```python
from isaacsim.core.utils.prims import delete_prim

class StageAPIImpl(StageAPI):
    def remove_prim(self, prim_path: str) -> None:
        delete_prim(prim_path)
        # 內部呼叫 omni.kit.commands.execute("DeletePrimsCommand",
        #   paths=[prim_path], destructive=True, stage=self.get_stage())
```

**依賴：**
- 輸入來源：`prim_path`（呼叫端提供）
- 輸出去向：`omni.kit.commands` 的 `DeletePrimsCommand`；**呼叫端（`TableSession`/`DemoTableSession`）在整批刪除流程結束後需呼叫一次 `omni.kit.undo.clear_stack()`**，避免 `DeletePrimsCommand` 掛進 Kit undo stack、使用者 Ctrl+Z 復原 USD 場景卻讓已 `destroy()` 的 core 物件跟畫面對不上（見第 6 節）

---

### ArticulationAPI.cancel_pending_home_capture（port，新增）

**職責：** `ArticulationAPIImpl.initialize()` 會註冊一個一次性的 `_capture_home_position_once` physics callback（用來捕捉手臂初始姿態當作 `move_to_home()` 的目標）。若 Demo 桌在這個 callback 觸發前就被 Toggle 關閉、手臂 prim 已被 `remove_prim()` 移除，callback 觸發時會對著不存在的 prim 呼叫 `get_end_effector_position()` 而報錯。這個方法讓 `DemoTableSession.destroy()` 在刪除 prim 之前，先主動取消尚未觸發的 callback。

```python
class ArticulationAPI(ABC):
    @abstractmethod
    def cancel_pending_home_capture(self) -> None:
        """
        取消尚未觸發的一次性 home-capture callback（若存在）。
        initialize() 從未被呼叫、或 callback 已經觸發過，皆為 no-op。
        """
        ...
```

**`ArticulationAPIImpl` 對應修改（`extension/isaac_sim_impl_6_0/articulation_api_impl.py`）：**

```python
def __init__(self, robot_prim_path: str, end_effector_prim_path: str) -> None:
    ...
    self._capture_callback_id = None  # 新增：None 代表「尚未註冊」或「已觸發並清空」

def initialize(self) -> None:
    ...
    self._capture_callback_id = SimulationManager.register_callback(
        self._capture_home_position_once, event=SimulationEvent.PHYSICS_POST_STEP
    )

def _capture_home_position_once(self, step_dt, context) -> None:
    self._home_position = np.array(self.get_end_effector_position())
    SimulationManager.deregister_callback(self._capture_callback_id)
    self._capture_callback_id = None  # 觸發後清空，供 cancel_pending_home_capture() 判斷是否還需取消

def cancel_pending_home_capture(self) -> None:
    if self._capture_callback_id is not None:
        SimulationManager.deregister_callback(self._capture_callback_id)
        self._capture_callback_id = None
```

**依賴：**
- 輸入來源：`DemoTableSession.destroy()` 呼叫
- 輸出去向：`SimulationManager.deregister_callback()`

---

### TableSession（新增）

**職責：** 封裝一張桌子的完整生命週期資源（`BilliardTable`＋`TableRuntime`＋`PocketEventHandler`），對外用 `table_id`（沿用 prim path，例如 `/World/Table_0`）當唯一識別。取代原本 Extension 端三個平行 list（`_training_tables`/`_table_runtimes`/`_pocket_event_handlers`）靠建立順序隱含對應的設計。

```python
class TableSession:
    def __init__(
        self,
        table_id: str,
        table: BilliardTable,
        runtime: TableRuntime,
        pocket_handler: PocketEventHandler,
        rigid_body_api: RigidBodyAPI,
    ) -> None:
        self._table_id = table_id
        self._table = table
        self._runtime = runtime
        self._pocket_handler = pocket_handler
        self._rigid_body_api = rigid_body_api

    def get_table_id(self) -> str:
        return self._table_id

    def tick(self) -> None:
        self._runtime.tick()

    def get_current_state(self) -> BilliardStatus:
        return self._runtime.get_current_state()

    def get_last_observation(self) -> Observation | None:
        return self._runtime.get_last_observation()

    def get_ball_velocities(self) -> dict[int, tuple[list[float], list[float]]]:
        """
        ball_id -> (linear_velocity, angular_velocity)。僅供 Debug Menu
        「顯示各球速度」進階 toggle 勾選時逐幀呼叫，避免預設就對
        tensor-based RigidBodyAPI 做額外查詢、影響效能。
        """
        table_ball_set = self._table.get_table_ball_set()
        velocities: dict[int, tuple[list[float], list[float]]] = {}
        for ball_id, prim_path in enumerate(table_ball_set.get_ball_prim_paths()):
            linear = self._rigid_body_api.get_linear_velocity(prim_path)
            angular = self._rigid_body_api.get_angular_velocity(prim_path)
            velocities[ball_id] = (linear, angular)
        return velocities

    def destroy(self) -> None:
        """依序：pocket_handler.stop() -> table.destroy()（內部呼叫 stage_api.remove_prim）"""
        self._pocket_handler.stop()
        self._table.destroy()
```

**依賴：**
- 輸入來源：建構期由 Extension 端注入的 `BilliardTable`/`TableRuntime`/`PocketEventHandler`/`RigidBodyAPI`（皆各桌一份，不與其他桌共用）
- 輸出去向：`DebugMenu` 透過 Extension 的 `get_table_debug_info()`/`get_ball_velocities_text()` 間接消費；Extension `_on_tick()` 逐一呼叫 `tick()`

---

### DemoTableSession（新增，`TableSession` 子類別）

**職責：** Demo 桌額外持有 `TableRobotManager`（手臂/球桿 prim 與 fixed joint）與 `ArticulationAPI`，處理「Toggle 完全解耦於 Timeline」與「`ArticulationAPI.initialize()` 必須等 Timeline Play 後才能呼叫」兩個限制之間的落差：Toggle ON 時只建 USD 場景（手臂 prim 存在但不能動），等 Timeline PLAY 事件另外補呼叫 `initialize_articulation()`。

```python
class DemoTableSession(TableSession):
    def __init__(
        self,
        table_id: str,
        table: BilliardTable,
        runtime: TableRuntime,
        pocket_handler: PocketEventHandler,
        rigid_body_api: RigidBodyAPI,
        robot_manager: TableRobotManager,
        articulation_api: ArticulationAPI,
    ) -> None:
        super().__init__(table_id, table, runtime, pocket_handler, rigid_body_api)
        self._robot_manager = robot_manager
        self._articulation_api = articulation_api
        self._articulation_initialized = False

    def initialize_articulation(self) -> None:
        """Timeline PLAY 事件觸發時呼叫（僅在尚未 initialize 時）"""
        self._articulation_api.initialize()
        self._articulation_initialized = True

    def is_articulation_initialized(self) -> bool:
        return self._articulation_initialized

    def destroy(self) -> None:
        """
        override：
        1. articulation_api.shutdown()（若已 initialize；現況為空實作 pass，
           但保持生命週期呼叫對稱）
        2. articulation_api.cancel_pending_home_capture()（無論是否已
           initialize 皆呼叫，內部自行判斷是否為 no-op；避免一次性
           home-capture callback 在 prim 已刪除後才觸發）
        3. robot_manager.destroy()（內部呼叫 stage_api.remove_prim 移除
           手臂/球桿 prim）
        4. super().destroy()（pocket_handler.stop() -> table.destroy()）
        """
        if self._articulation_initialized:
            self._articulation_api.shutdown()
        self._articulation_api.cancel_pending_home_capture()
        self._robot_manager.destroy()
        super().destroy()
```

**依賴：**
- 輸入來源：建構期注入的 `TableRobotManager`/`ArticulationAPI`；`initialize_articulation()` 由 Extension `_on_play()`／`_enable_demo()`（若 Toggle ON 時 Timeline 已在播放）呼叫
- 輸出去向：`ArticulationAPIImpl` 實作層（RmpFlow/差動 IK 執行手臂動作）

---

### TableRuntime（修改）

**職責：** 既有「無狀態容器」定位不變，`tick()` 簽名不變，但內部把 `ObservationBuilder.build()` 的結果存下來，供 `TableSession` 事後查詢（原本用完即丟，導致 Debug Menu 完全無法取得目前的 `Observation`）。

```python
class TableRuntime:
    def __init__(self, observation_builder: ObservationBuilder, orchestrator: TableOrchestrator) -> None:
        self._observation_builder = observation_builder
        self._orchestrator = orchestrator
        self._last_observation: Observation | None = None  # 新增

    def tick(self) -> None:
        observation = self._observation_builder.build()
        self._last_observation = observation  # 新增
        self._orchestrator.step(observation)

    def get_current_state(self) -> BilliardStatus:
        """轉發 self._orchestrator.get_current_state()"""
        return self._orchestrator.get_current_state()

    def get_last_observation(self) -> Observation | None:
        return self._last_observation
```

**依賴：**
- 輸入來源：不變（`ObservationBuilder`/`TableOrchestrator`，建構期注入）
- 輸出去向：新增 `TableSession.get_current_state()`/`get_last_observation()` 轉發呼叫

---

### TableOrchestrator（修改）

**職責：** 新增 `get_current_state()`，轉發 `ScriptController.get_current_state()`（`ScriptController` 已存在同名方法，見 `core/controllers/script_controller.py:27-28`），讓 `TableRuntime` 不需要直接接觸 `ScriptController`。

```python
class TableOrchestrator(ABC):
    ...
    def get_current_state(self) -> BilliardStatus:
        """轉發 self._script_controller.get_current_state()"""
        return self._script_controller.get_current_state()
```

**依賴：**
- 輸入來源：`self._script_controller`（既有欄位，建構期已注入）
- 輸出去向：`TableRuntime.get_current_state()`

---

### BilliardTable.destroy（修改）

**職責：** 從「只清空 Python 參照」改為「真的移除 USD prim」。建構子需額外保存 `self._stage_api`（現況只在 `__init__` 內當區域參數用，`destroy()` 之前完全用不到它）。

```python
class BilliardTable:
    def __init__(self, base_path, stage_api, material_api, rigid_body_api, position):
        self._base_path = base_path
        self._stage_api = stage_api  # 新增：destroy() 需要
        ...

    def destroy(self) -> None:
        self._stage_api.remove_prim(self._table_prim_path)
        self._table_set = None
```

**依賴：**
- 輸入來源：建構期注入的 `StageAPI`
- 輸出去向：`StageAPIImpl.remove_prim()` → `TableSession.destroy()` 的呼叫鏈

---

### TableRobotManager.destroy（修改）

**職責：** 同上，改為真的移除手臂/球桿 prim。建構子額外保存 `self._stage_api`。

```python
class TableRobotManager:
    def __init__(self, table_center, base_path, stage_api, articulation_api, robot_arm_class):
        ...
        self._stage_api = stage_api  # 新增：destroy() 需要
        ...

    def destroy(self) -> None:
        self._stage_api.remove_prim(self._robot_base_path)
        self._robot = None
```

**依賴：**
- 輸入來源：建構期注入的 `StageAPI`
- 輸出去向：`StageAPIImpl.remove_prim()` → `DemoTableSession.destroy()` 的呼叫鏈

**注意：** `TableRobotManager` 目前沒有對外暴露球桿 prim 的獨立移除方法——球桿（`CueStick`）是 `base_path` 底下的子 prim（`base_path + "/CueStick"`），`remove_prim(self._robot_base_path)` 會連同球桿一併移除，不需要額外呼叫。

---

### BilliardExtension（修改，extension 層，不寫 Unit Test，改手動驗證）

**職責：** 把 Toggle 接上 `TableSession`/`DemoTableSession` 的建立與銷毀，移除 `RuntimeState`，把 tick callback 提前到 `_billiard_init()` 註冊一次。

**新增/取代欄位：**
```python
self._training_sessions: list[TableSession] = []
self._demo_sessions: list[DemoTableSession] = []  # 實務最多 1 個，維持 list 保持與 training 對稱
self._training_enabled: bool = True   # 原本硬編碼 False
self._demo_enabled: bool = True       # 原本硬編碼 False
self._timeline_playing: bool = False  # 取代 RuntimeState enum（NOT_READY/READY/RUNNING 整個移除）
```

**主要方法（設計定案，見第 4 節資料流的完整呼叫順序）：**

```python
def _billiard_init(self) -> None:
    SimulationManager.setup_simulation(dt=1/60)
    self._asset_env_init()  # 建立共用資源：StageAPI/MaterialAPI/RigidBodyAPI/table_unit_side_length/rolling_resistance_service，不再直接建桌子
    self._training_enabled = True
    self._demo_enabled = True
    self._debug_menu = DebugMenu(
        self._on_training_toggle,
        self._on_demo_toggle,
        self.get_table_ids,
        self.get_table_debug_info,
        self.get_ball_velocities_text,
    )
    self._event_init()  # 訂閱 Timeline PLAY/STOP（不變）
    self._tick_callback_id = SimulationManager.register_callback(
        self._on_tick, event=SimulationEvent.PHYSICS_POST_STEP
    )  # 提前到這裡註冊一次，不再綁 Timeline PLAY
    self._on_training_toggle(self._training_enabled)  # 預設 True，開機即建立訓練桌
    self._on_demo_toggle(self._demo_enabled)           # 預設 True，開機即建立 Demo 桌

def _enable_training(self) -> None:
    """建立 _TABLE_COUNT 張訓練桌 + TableSession，append 進 self._training_sessions；
    完成後呼叫 self._debug_menu.set_available_tables(self.get_table_ids())"""
    ...

def _disable_training(self) -> None:
    """for session in self._training_sessions: session.destroy()
    self._training_sessions = []
    omni.kit.undo.clear_stack()
    self._debug_menu.set_available_tables(self.get_table_ids())"""
    ...

def _enable_demo(self) -> None:
    """建立 Demo 桌 + TableRobotManager + ArticulationAPIImpl + DemoTableSession，
    append 進 self._demo_sessions；若 self._timeline_playing 為 True，
    立即呼叫 session.initialize_articulation()；
    完成後呼叫 self._debug_menu.set_available_tables(self.get_table_ids())"""
    ...

def _disable_demo(self) -> None:
    """同 _disable_training，作用於 self._demo_sessions"""
    ...

def _on_training_toggle(self, enable: bool) -> None:
    self._training_enabled = enable
    if enable:
        self._enable_training()
    else:
        self._disable_training()

def _on_demo_toggle(self, enable: bool) -> None:
    self._demo_enabled = enable
    if enable:
        self._enable_demo()
    else:
        self._disable_demo()

def get_table_ids(self) -> list[str]:
    return [s.get_table_id() for s in self._training_sessions + self._demo_sessions]

def get_table_debug_info(self, table_id: str) -> str:
    """依 table_id 找到對應 session，組成簡單版狀態文字回傳；找不到回傳空字串"""
    ...

def get_ball_velocities_text(self, table_id: str) -> str:
    """依 table_id 找到對應 session，呼叫 session.get_ball_velocities() 組成逐球速度文字回傳"""
    ...

def _on_play(self) -> None:
    self._timeline_playing = True
    for demo_session in self._demo_sessions:
        if not demo_session.is_articulation_initialized():
            demo_session.initialize_articulation()

def _on_stop(self) -> None:
    self._timeline_playing = False

def _on_tick(self, step_dt, context) -> None:
    for session in self._training_sessions + self._demo_sessions:
        session.tick()

def on_shutdown(self) -> None:
    self._disable_training()
    self._disable_demo()
    SimulationManager.deregister_callback(self._tick_callback_id)
    if self._tool_menu_items:
        unregister(self._tool_menu_items, _TOOL_MENU_NAME)
        self._tool_menu_items = None
    if self._debug_menu:
        self._debug_menu.destroy()
        self._debug_menu = None
    self._sub = None
    self._timeline_sub = None
```

**依賴：**
- 輸入來源：`DebugMenu` 的 Toggle callback、Timeline PLAY/STOP 事件、`SimulationManager` 的 `PHYSICS_POST_STEP` 事件
- 輸出去向：`TableSession`/`DemoTableSession` 的建立與銷毀；`DebugMenu.set_available_tables()`

---

### DebugMenu（修改，extension/ui 層，不寫 Unit Test，改手動驗證）

**職責：** 新增選桌下拉選單（預設不選任何桌）、狀態/Observation 顯示區塊（簡單版每幀更新）、逐球速度進階 toggle（勾選才查詢）。

```python
class DebugMenu:
    def __init__(
        self,
        on_training_toggle: Callable[[bool], None],
        on_demo_toggle: Callable[[bool], None],
        get_table_ids: Callable[[], list[str]],
        get_table_debug_info: Callable[[str], str],
        get_ball_velocities_text: Callable[[str], str],
    ) -> None:
        ...
        self._get_table_ids = get_table_ids
        self._get_table_debug_info = get_table_debug_info
        self._get_ball_velocities_text = get_ball_velocities_text
        self._selected_table_id: str | None = None  # 預設不選任何桌
        self._show_ball_velocities = False
        self._build_ui()  # 新增 ComboBox、狀態 Label、速度 Label、速度 CheckBox
        asyncio.ensure_future(self._dock_to_viewport())
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="billiard_debug_menu_refresh")
        )

    def set_available_tables(self, table_ids: list[str]) -> None:
        """
        Extension 端 _enable_*/_disable_* 完成後呼叫，更新 ComboBox 選項。
        若目前選中的 table_id 已不在 table_ids 內（桌子被 Toggle 關閉刪除），
        自動清空選擇（面板回到空白，不自動切換到其他桌）。
        """
        if self._selected_table_id is not None and self._selected_table_id not in table_ids:
            self._selected_table_id = None
            self._status_label.text = ""
            self._velocity_label.text = ""
        # 更新 ComboBox 底層 model 的選項清單為 table_ids

    def _on_update(self, event) -> None:
        if self._selected_table_id is None:
            return
        self._status_label.text = self._get_table_debug_info(self._selected_table_id)
        if self._show_ball_velocities:
            self._velocity_label.text = self._get_ball_velocities_text(self._selected_table_id)

    def destroy(self) -> None:
        if self._window:
            self._window.destroy()
            self._window = None
        self._update_sub = None  # 新增：釋放 update_event_stream 訂閱
```

**依賴：**
- 輸入來源：Extension 注入的三個查詢 callback；`omni.kit.app.get_app().get_update_event_stream()`（每幀 refresh）
- 輸出去向：無回傳值，純 UI 呈現

---

## 4. 資料流

**Training Toggle ON（`_enable_training()`）：**
```
DebugMenu ToolButton（Training）→ True
  → on_training_toggle(True) = BilliardExtension._on_training_toggle(True)
    → self._training_enabled = True
    → self._enable_training()
      → 依 _TABLE_COUNT 計算網格排列座標
      → for each table_id in ["/World/Table_0", ...]：
          → self._build_table(table_id, ...) 建立 BilliardTable（USD 場景，立即生效，不受 Timeline 影響）
          → 建立 PocketEventHandler(physics_api, pocket_prim_paths, ball_prim_paths, on_ball_pocketed) → .start()
          → 建立 ScriptController / ErrorState / ImpulseStrikingService
          → 建立 TrainingTableObservationBuilder / TrainingTableOrchestrator
          → 組成 TableRuntime(observation_builder, orchestrator)
          → 組成 TableSession(table_id, table, runtime, pocket_handler, rigid_body_api)
          → self._training_sessions.append(session)
      → self._debug_menu.set_available_tables(self.get_table_ids())
        → ComboBox 選項更新為所有現存 table_id
```

**Training Toggle OFF（`_disable_training()`）：**
```
DebugMenu ToolButton（Training）→ False
  → on_training_toggle(False) = BilliardExtension._on_training_toggle(False)
    → self._training_enabled = False
    → self._disable_training()
      → for session in self._training_sessions:
          → session.destroy()
            → pocket_handler.stop()（解除 physics contact 訂閱，只影響這張桌）
            → table.destroy()
              → stage_api.remove_prim(table_prim_path)
                → isaacsim.core.utils.prims.delete_prim(...)
                  → omni.kit.commands.execute("DeletePrimsCommand", ..., destructive=True)
              → self._table_set = None
      → self._training_sessions = []
      → omni.kit.undo.clear_stack()（避免使用者 Ctrl+Z 復原已刪除的 USD prim，跟已銷毀的 core 物件對不上）
      → self._debug_menu.set_available_tables([])
        → 若目前選中的桌子在被刪除的清單中 → DebugMenu 自動清空選擇、清空顯示文字
```

**Demo Toggle ON（`_enable_demo()`，含分階段 Articulation 建立）：**
```
DebugMenu ToolButton（Break shot demo）→ True
  → on_demo_toggle(True) = BilliardExtension._on_demo_toggle(True)
    → self._demo_enabled = True
    → self._enable_demo()
      → self._build_table("/World/Table_Demo", ...) 建立 BilliardTable（USD 場景，立即生效）
      → 建立 ArticulationAPIImpl(robot_prim_path, robot_end_effector_prim_path)
        （只建構 instance，不呼叫 initialize()）
      → 建立 TableRobotManager(demo_table_center, ..., articulation_api, _ROBOT_ARM_CLASS)
        → 內部立即建立手臂 prim / 球桿 prim / FixedJoint（USD prim 建立不受 Timeline 限制，
          即使 Timeline 尚未 Play，手臂 prim 也已存在，只是還不能動）
      → 建立 PocketEventHandler → .start()
      → 建立 ScriptController / ErrorState
      → 建立 DemoTableObservationBuilder / DemoTableOrchestrator
      → 組成 TableRuntime、組成 DemoTableSession(table_id, table, runtime, pocket_handler,
        rigid_body_api, robot_manager, articulation_api)
      → self._demo_sessions.append(session)
      → 若 self._timeline_playing == True（使用者在 Timeline 已經在播放時才勾選 Demo）：
          → session.initialize_articulation()
            → articulation_api.initialize()（tensor-based Articulation 綁定，這時才合法）
            → self._articulation_initialized = True
      → 否則（Timeline 尚未 Play）：is_articulation_initialized() 維持 False，
        手臂 prim 存在但不能動，等下面 Timeline PLAY 流程補呼叫
      → self._debug_menu.set_available_tables(self.get_table_ids())
```

**Timeline PLAY 事件（`_on_play()`，只負責「補呼叫」）：**
```
omni.timeline PLAY 事件
  → BilliardExtension._on_play()
    → self._timeline_playing = True
    → for demo_session in self._demo_sessions:
        → if not demo_session.is_articulation_initialized():
            → demo_session.initialize_articulation()
              → articulation_api.initialize()
```

**Demo Toggle OFF（`_disable_demo()`，隨時可執行，不受 Timeline 狀態影響）：**
```
DebugMenu ToolButton（Break shot demo）→ False
  → on_demo_toggle(False) = BilliardExtension._on_demo_toggle(False)
    → self._demo_enabled = False
    → self._disable_demo()
      → for session in self._demo_sessions:
          → session.destroy()  # DemoTableSession.destroy() override
            → if self._articulation_initialized: articulation_api.shutdown()（現況空實作，保持對稱）
            → articulation_api.cancel_pending_home_capture()
              → 若一次性 home-capture callback 尚未觸發，deregister 它
                （避免之後對著已刪除的手臂 prim 讀位置報錯）
            → robot_manager.destroy()
              → stage_api.remove_prim(robot_base_path)（連同球桿一併移除）
            → super().destroy()
              → pocket_handler.stop()
              → table.destroy() → stage_api.remove_prim(table_prim_path)
      → self._demo_sessions = []
      → omni.kit.undo.clear_stack()
      → self._debug_menu.set_available_tables(self.get_table_ids())
```

**每 tick 執行迴圈（`_on_tick`，Extension 啟動時就註冊一次，完全不受 Timeline 影響，也不看任何 enabled 旗標）：**
```
SimulationManager PHYSICS_POST_STEP 事件（只有 Timeline 真的在跑 physics 時才會 fire）
  → BilliardExtension._on_tick(step_dt, context)
    → for session in self._training_sessions + self._demo_sessions:
        → session.tick()
          → runtime.tick()
            → observation = observation_builder.build()
            → self._last_observation = observation  # 新增快取
            → orchestrator.step(observation)（既有邏輯不變：rolling resistance → 決策 → 分派下游動作）
```

**Debug Menu 選桌與狀態顯示（每幀）：**
```
使用者在 ComboBox 選擇某個 table_id
  → DebugMenu._selected_table_id = table_id

omni.kit.app update_event_stream（每幀觸發）
  → DebugMenu._on_update(event)
    → if self._selected_table_id is None: return
    → status_text = get_table_debug_info(table_id) = BilliardExtension.get_table_debug_info(table_id)
      → session = 依 table_id 找到對應 TableSession/DemoTableSession
      → state = session.get_current_state()          # BilliardStatus
      → observation = session.get_last_observation()  # Observation | None
      → 組成文字：狀態 / is_ball_moving / is_motion_complete / has_error / cue_ball_position
    → self._status_label.text = status_text
    → if self._show_ball_velocities:
        → velocity_text = get_ball_velocities_text(table_id) = BilliardExtension.get_ball_velocities_text(table_id)
          → session.get_ball_velocities()
            → for ball_id, prim_path in enumerate(get_ball_prim_paths()):
                → rigid_body_api.get_linear_velocity(prim_path)
                → rigid_body_api.get_angular_velocity(prim_path)
        → self._velocity_label.text = velocity_text
```

---

## 5. 依賴關係圖

```
TableSession（每張訓練桌一個）
  ├── 依賴 BilliardTable（USD 場景 + destroy() 移除 prim）
  ├── 依賴 TableRuntime（tick 執行 + 狀態/Observation 查詢）
  ├── 依賴 PocketEventHandler（各自獨立的 physx 訂閱，stop() 只影響自己）
  └── 依賴 RigidBodyAPI（get_ball_velocities() 逐球查詢，僅進階 toggle 使用）

DemoTableSession（最多一個，繼承 TableSession）
  ├── 額外依賴 TableRobotManager（手臂/球桿 prim + destroy() 移除 prim）
  └── 額外依賴 ArticulationAPI（initialize()/shutdown()/cancel_pending_home_capture()）

TableRuntime
  ├── 依賴 ObservationBuilder（DemoTableObservationBuilder / TrainingTableObservationBuilder，既有）
  └── 依賴 TableOrchestrator（DemoTableOrchestrator / TrainingTableOrchestrator，既有；
      新增 get_current_state() 轉發 ScriptController）

BilliardTable
  └── 新增依賴 StageAPI.remove_prim()（destroy() 使用）

TableRobotManager
  └── 新增依賴 StageAPI.remove_prim()（destroy() 使用）

StageAPIImpl.remove_prim
  └── 依賴 isaacsim.core.utils.prims.delete_prim（第三方 API）

ArticulationAPIImpl.cancel_pending_home_capture
  └── 依賴 SimulationManager.deregister_callback（既有機制，沿用 _capture_callback_id）

BilliardExtension（extension 組裝層）
  ├── 持有 list[TableSession]（training）+ list[DemoTableSession]（demo，實務最多 1 個）
  ├── 依賴 omni.timeline（PLAY/STOP，決定 self._timeline_playing 與是否補呼叫 initialize_articulation()）
  ├── 依賴 SimulationManager.register_callback(PHYSICS_POST_STEP)（tick callback，
  │     Extension 啟動時註冊一次，不再綁 Timeline PLAY；取代原本的 RuntimeState guard）
  └── 依賴 omni.kit.undo.clear_stack()（每次整批刪除完成後呼叫）

DebugMenu
  ├── 依賴 Extension 注入的 get_table_ids / get_table_debug_info / get_ball_velocities_text
  ├── 依賴 omni.kit.app.get_app().get_update_event_stream()（每幀 refresh）
  └── 依賴 omni.ui（ComboBox / Label / CheckBox，既有 Window/VStack 骨架不變）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| Toggle 觸發的建立/刪除發生時 Timeline 狀態為何 | 完全不受影響，`_enable_training`/`_disable_training`/`_enable_demo`/`_disable_demo` 隨時可呼叫、立即生效；USD prim 建立本身不受 Timeline 限制 |
| Demo Toggle ON 但 Timeline 尚未 Play | `DemoTableSession` 正常建立（手臂 prim 存在），但 `is_articulation_initialized()` 為 `False`，手臂不能動；等 Timeline PLAY 事件觸發時由 `_on_play()` 補呼叫 `initialize_articulation()` |
| Demo Toggle OFF 發生在 Timeline 尚未 Play（`is_articulation_initialized() == False`） | `destroy()` 內 `if self._articulation_initialized:` 為 `False`，跳過 `shutdown()`；`cancel_pending_home_capture()` 因 `_capture_callback_id` 從未被設定（仍是 `None`）而為 no-op；後續 `robot_manager.destroy()`/`super().destroy()` 正常執行 |
| Demo Toggle OFF 發生在 `initialize()` 已呼叫、但一次性 home-capture callback 尚未觸發 | `cancel_pending_home_capture()` 主動 `deregister_callback()`，避免這個 callback 之後對著已被 `remove_prim()` 移除的手臂 prim 呼叫 `get_end_effector_position()` 而報錯 |
| `DeletePrimsCommand` 掛進 Kit undo stack | 每次 `_disable_training()`/`_disable_demo()` 整批刪除完成後呼叫一次 `omni.kit.undo.clear_stack()`，避免使用者 Ctrl+Z 復原 USD 場景卻讓已 `destroy()` 的 core 物件跟畫面對不上 |
| 訓練桌之間互相影響 | 每張桌各自獨立的 `ScriptController`/`ErrorState`/`ImpulseStrikingService`/`PocketEventHandler`（各自持有獨立的 physx 訂閱控制代碼），`RigidBodyAPI`/`StageAPI` 沿用共用單例；單一桌子的 `destroy()`/例外都不影響其他桌 |
| Debug Menu 目前選中的桌子被 Toggle 關閉刪除 | `set_available_tables()` 檢查目前選擇是否還在最新的 `table_ids` 內，不在則自動清空選擇與顯示文字（面板回到空白，**不**自動切換到其他桌） |
| Debug Menu 尚未選擇任何桌（初始狀態） | `_selected_table_id = None`，`_on_update()` 直接 `return`，面板保持空白，不查詢任何 session |
| 「顯示各球速度」未勾選 | `_show_ball_velocities = False` 時完全不呼叫 `get_ball_velocities_text()`，避免預設就對 tensor-based `RigidBodyAPI` 做額外查詢、影響效能 |
| `get_table_debug_info(table_id)`/`get_ball_velocities_text(table_id)` 找不到對應 session（例如剛好在同一幀被刪除） | 回傳空字串 `""`，UI 端顯示空白，不拋例外中斷 refresh loop |
| `TableRuntime.get_last_observation()` 在第一次 `tick()` 之前被呼叫 | 回傳 `None`（`self._last_observation` 初始值），`get_table_debug_info()` 需處理 `observation is None` 的情況（顯示「尚未有 Observation」而非直接存取欄位拋 `AttributeError`） |
| Extension 啟動時預設 `_training_enabled`/`_demo_enabled` 皆為 `True` | `_billiard_init()` 設好旗標後主動呼叫 `_on_training_toggle(True)`/`_on_demo_toggle(True)`，行為等同使用者一開機就手動勾選兩個開關，不需要額外的「啟動即建立」特殊路徑 |
| `RuntimeState`（`NOT_READY`/`READY`/`RUNNING`）enum 整個移除 | 原本用來防止 tick callback 在 Timeline 未播放時執行、以及防止 PLAY 事件重複初始化；新設計下 tick callback 於 Extension 啟動時就註冊一次（`PHYSICS_POST_STEP` 事件本身只有 Timeline 真的在跑 physics 時才會 fire，提早訂閱沒有副作用），`_on_play()` 改用 `demo_session.is_articulation_initialized()` 逐一判斷是否需要補初始化，天然具備冪等性，不需要額外的整體 guard flag |
| `PocketEventHandler` 是否會互相干擾 | 不會：每個實例各自持有獨立的 physx 訂閱控制代碼，`stop()` 只影響自己，可放心單桌呼叫（見 `core/services/pocket_event_handler.py`） |

---

## 7. 測試涵蓋（對應 Unit Test）

`TableSession`/`DemoTableSession`/`TableRuntime`/`TableOrchestrator` 屬於 core 層純邏輯（狀態機轉發、資源生命週期協調、逐球速度組裝），依 `docs/unit-test-rules.md` 需要 Unit Test，全部以 Mock 隔離 `BilliardTable`/`TableRuntime`/`PocketEventHandler`/`RigidBodyAPI`/`ArticulationAPI` 等依賴。

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| test_table_session_get_table_id_returns_constructed_id | core/tests/test_table_session.py | `get_table_id()` 回傳建構時傳入的 `table_id` |
| test_table_session_tick_delegates_to_runtime | core/tests/test_table_session.py | `tick()` 呼叫 `runtime.tick()` 一次 |
| test_table_session_get_current_state_delegates_to_runtime | core/tests/test_table_session.py | `get_current_state()` 回傳 `runtime.get_current_state()` 的結果 |
| test_table_session_get_last_observation_delegates_to_runtime | core/tests/test_table_session.py | `get_last_observation()` 回傳 `runtime.get_last_observation()` 的結果（含 `None` 情況） |
| test_table_session_get_ball_velocities_queries_each_ball | core/tests/test_table_session.py | 依 `table.get_table_ball_set().get_ball_prim_paths()` 逐一呼叫 `rigid_body_api.get_linear_velocity()`/`get_angular_velocity()`，回傳的 dict key 為 ball_id（依 prim path 順序，從 0 起算） |
| test_table_session_destroy_stops_pocket_handler_before_table | core/tests/test_table_session.py | `destroy()` 依序呼叫 `pocket_handler.stop()` → `table.destroy()`，順序正確（`assert_has_calls` 驗證） |
| test_demo_table_session_initialize_articulation_calls_api_and_marks_initialized | core/tests/test_table_session.py | `initialize_articulation()` 呼叫 `articulation_api.initialize()`，之後 `is_articulation_initialized()` 回傳 `True` |
| test_demo_table_session_is_articulation_initialized_default_false | core/tests/test_table_session.py | 建構後尚未呼叫 `initialize_articulation()` 時 `is_articulation_initialized()` 為 `False` |
| test_demo_table_session_destroy_calls_shutdown_when_initialized | core/tests/test_table_session.py | 已 `initialize_articulation()` 後呼叫 `destroy()`，驗證 `articulation_api.shutdown()` 被呼叫 |
| test_demo_table_session_destroy_skips_shutdown_when_not_initialized | core/tests/test_table_session.py | 未曾 `initialize_articulation()` 時呼叫 `destroy()`，驗證 `articulation_api.shutdown()` 不被呼叫 |
| test_demo_table_session_destroy_cancels_pending_home_capture | core/tests/test_table_session.py | `destroy()` 呼叫 `articulation_api.cancel_pending_home_capture()`（無論是否已 initialize） |
| test_demo_table_session_destroy_calls_robot_manager_destroy_before_super | core/tests/test_table_session.py | `destroy()` 依序呼叫 `shutdown()`（若已 init）→ `cancel_pending_home_capture()` → `robot_manager.destroy()` → `pocket_handler.stop()` → `table.destroy()`，用 `MagicMock` + 共用 `call` 序列驗證完整順序 |
| test_table_runtime_tick_caches_last_observation | core/tests/test_table_runtime.py | `tick()` 呼叫後 `get_last_observation()` 回傳與 `observation_builder.build()` 相同的 instance |
| test_table_runtime_get_last_observation_none_before_first_tick | core/tests/test_table_runtime.py | 建構後、`tick()` 呼叫前，`get_last_observation()` 回傳 `None` |
| test_table_runtime_get_current_state_delegates_to_orchestrator | core/tests/test_table_runtime.py | `get_current_state()` 回傳 `orchestrator.get_current_state()` 的結果 |
| test_table_orchestrator_get_current_state_delegates_to_script_controller | core/tests/test_table_orchestrator.py | `get_current_state()` 回傳 `script_controller.get_current_state()` 的結果 |
| test_billiard_table_destroy_calls_stage_api_remove_prim | core/tests/test_billiard_table.py | `destroy()` 呼叫 `stage_api.remove_prim(table_prim_path)`，且 `self._table_set` 變為 `None` |
| test_table_robot_manager_destroy_calls_stage_api_remove_prim | core/tests/test_table_robot_manager.py | `destroy()` 呼叫 `stage_api.remove_prim(robot_base_path)`，且 `self._robot` 變為 `None` |

`StageAPIImpl.remove_prim`（薄層包裝 `isaacsim.core.utils.prims.delete_prim`）與 `ArticulationAPIImpl.cancel_pending_home_capture`（薄層包裝 `SimulationManager.deregister_callback`）依 `docs/unit-test-rules.md` 條件 3（直接包裝第三方 API 的薄層函式）**Unit Test 豁免**。`BilliardExtension`/`DebugMenu` 屬於 extension 層（條件 4/5：UI 元件與視覺呈現邏輯），比照既有 `docs/tech-design-2-10-debug-menu.md` 第 7 節先例，**Unit Test 豁免**，改用 Debug Menu 手動驗證：

1. Extension 啟動後，Training／Break shot demo 兩個 ToolButton 預設皆為 ON，對應桌子與手臂已出現在 Viewport。
2. 手動關閉 Training → 對應桌子（USD prim）從 Viewport 消失；重新開啟 → 桌子重新出現在原位。
3. 手動關閉 Break shot demo → 桌子與手臂皆消失；按 Ctrl+Z 確認場景不會意外復原（`clear_stack()` 生效）。
4. Timeline 尚未 Play 時關閉再開啟 Demo，觀察 Console 無例外拋出；按下 Play 後手臂可正常回到 Home 姿態。
5. 在 Debug Menu 選桌下拉選單選擇某張桌子，狀態文字隨 tick 即時更新（`IDLE`/`AIMING`/`STRIKING`/... 與母球座標）；勾選「顯示各球速度」後出現逐球速度文字。
6. 關閉目前正在檢視的那張桌子的 Toggle，確認選桌下拉選單自動清空、狀態文字回到空白。

---

## 8. 待決定事項

- [ ] **`table_unit_side_length`（訓練桌網格排列間距）的量測時機**：現況 `_get_table_side_length()` 靠量測「已建立的 Demo 桌」的 AABB 取得，且訓練桌網格座標計算需要這個常數。Toggle 解耦後，使用者可能只開 Training 不開 Demo（或反過來、或兩者不同時開），無法再保證「Demo 桌一定比 Training 桌先建立」。建議做法（待實作時定案）：`_billiard_init()` 內用新增的 `stage_api.remove_prim()` 能力，建立一個一次性的量測用 prim（例如 `/World/_TableSizeProbe`），量完 AABB 後立刻移除，取得的常數快取供 `_enable_training()`/`_enable_demo()` 共用，不再依賴任一張正式桌子是否存在。
- [ ] **`self._rolling_resistance_service` 的 `ball_radius` 來源**：現況建構時讀 `self._demo_table.get_table_ball_set().get_ball_radius()`，同樣依賴 Demo 桌已建立。`TableBallSet.get_ball_radius()` 預設值固定為 `0.028575`（見 `core/models/table_ball_set.py:34`），建議直接改用這個常數或另外定義共用常數，不再依賴任何一張桌子的 live instance。
- [ ] `DebugMenu` 選桌 `ComboBox` 的底層 model 更新寫法（`omni.ui.ComboBox` 動態改變選項清單的標準做法）— 屬於第三方 UI API 用法細節，交由 `api-lookup` agent 在實作階段查詢 `skills/isaac_sim_6_api_cache.md`／官方文件確認。
- [ ] `PocketEventHandler` 是否需要新增 `is_started()`/`is_stopped()` 之類的狀態查詢，供 `TableSession.destroy()` 防止重複呼叫 `stop()`（目前設計假設 `destroy()` 只會被呼叫一次，`_disable_training()`/`_disable_demo()` 呼叫完就立刻清空對應的 session list，理論上不會有重複呼叫風險，暫不需要）。
