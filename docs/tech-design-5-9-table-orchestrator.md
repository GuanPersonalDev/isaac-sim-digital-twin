# TableOrchestrator — 技術設計文件

> 生成時間：2026-07-20（第 1-7 節於 2026-07-22 依第 8 節決策全面整理為最終版）
>
> 所屬專案：isaac-sim-digital-twin
>
> 關聯 GitHub：延續 Issue #99「[5-8] 實作 RESET：場景重置 → 回到 IDLE」；對應 Block 5 任務 5-9「單次擊球循環跑通確認」
>
> 關聯文件：`docs/tech-design-5-2-script-controller-state-machine.md`（第 5 節「Controller／執行層職責分離」）

---

## 1. 功能概述

單一撞球桌（Demo 桌／訓練桌）的每 tick 執行迴圈由三個角色分工組成：

1. **`ObservationBuilder`**：每個 tick 開始時，即時查詢所有下游狀態（球位置、是否還在移動、下游動作是否完成、是否為初始擺球狀態、是否有錯誤），完整生成一份 `Observation`（不接受任何既有值，永遠從頭組裝）。
2. **`TableOrchestrator`**：拿到 `Observation` 後，透過 `ControllerBase.get_action()` 取得純決策結果 `Action`，再依 `ControllerBase.get_current_state()` 回傳的 `BilliardStatus` 把 `Action` 轉譯成真正的下游副作用（球位置重置、手臂歸位、瞄準、擊球）。`step()` 只負責「決策 + 執行動作」，不組裝／回傳 `Observation`。
3. **`TableRuntime`**：不持有任何狀態，單純把上述兩者串起來——`tick()` 呼叫 `ObservationBuilder.build()` 取得本次 `Observation`，再傳給 `TableOrchestrator.step()` 執行。

這個設計填補了 Issue #99 遺留的缺口——`TableBallSet.reset()` 與 `UR5Robot.reset()`/`is_reset_complete()` 目前都只是「有能力被呼叫」，先前沒有生產程式碼真的呼叫它們——並補上執行期間的錯誤處理（`ErrorState`：下游動作拋例外時記錄但不中斷其他桌子）與外部重新初始化入口（`TableOrchestrator.reset()`）。

本次（Block 5 任務 5-9）的**實作範圍**：
- 已完成：RESET 狀態的完整資料流（`ControllerBase.get_current_state()` 契約與 `ScriptController` 實作、`TableOrchestrator`/`DemoTableOrchestrator`/`TrainingTableOrchestrator` 骨架、`Observation` 死欄位清理、`TableBallSet` 世界偏移量修正）。
- 本次待實作：`ErrorState`、`ObservationBuilder`（含 Demo/Training 兩種子類別）、`TableOrchestrator`/`TrainingTableOrchestrator` 的錯誤處理與 `reset()` 支援、`TableRuntime`、兩個缺口 getter、Extension 端訓練桌的 timeline play/stop 生命週期串接。
- 明確排除在本次範圍外：Demo 桌的完整 `TableRuntime` 組裝與 `ArticulationAPI` 注入方式（留待 #96/#97 手臂操作實作時處理）、AIMING/STRIKING 的實際手臂動作內容。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 | 狀態 |
|---|---|---|---|---|
| `ControllerBase.get_current_state()` | core/controllers | 定義回傳 `BilliardStatus` 的共同契約；`ScriptController` 回傳 `self._current_state`，未來 `ModelController` 必須提供相同行為 | `core/controllers/controller_base.py`、`script_controller.py` | 已完成 |
| `TableOrchestrator`（抽象基底） | core/services | 只依賴 `ControllerBase` 的三個生命週期方法，依狀態分派下游動作、`try/except` 包住下游動作並回報 `ErrorState`、提供統一 `reset()` 入口 | `core/services/table_orchestrator.py` | 已完成 |
| `DemoTableOrchestrator` | core/services | Demo 桌差異實作：RESET 呼叫 `UR5Robot.reset()`；AIMING/STRIKING 僅定義介面（留給 #96/#97） | `core/services/table_orchestrator.py` | 骨架已完成／`ErrorState` 支援待實作 |
| `TrainingTableOrchestrator` | core/services | 訓練桌差異實作：RESET/AIMING no-op 或恆真，STRIKING 呼叫 `ImpulseStrikingService.strike()` | `core/services/table_orchestrator.py` | 骨架已完成／`ErrorState` 支援待實作 |
| `ErrorState`（新元件） | core/services | 集中記錄下游執行例外：`mark_error()`（log + 記錄，不重拋）、`has_error()`、`get_last_exception()`、`clear()` | `core/services/error_state.py` | 待實作 |
| `ObservationBuilder`（抽象基底，新元件） | core/services | 每個 tick 完整組裝一份 `Observation`（不吃 `previous_observation`）：球位置、母球位置、是否移動中、是否為初始狀態、是否有錯誤；下游動作完成度交由子類別 | `core/services/observation_builder.py` | 待實作 |
| `DemoObservationBuilder` | core/services | 額外注入 `UR5Robot`，`_is_downstream_motion_complete()` 回傳 `ur5_robot.is_reset_complete()` | `core/services/observation_builder.py` | 待實作 |
| `TrainingObservationBuilder` | core/services | `_is_downstream_motion_complete()` 恆回傳 `True`（沒有手臂） | `core/services/observation_builder.py` | 待實作 |
| `TableRuntime`（新元件） | core/services | 無狀態容器：`tick()` = `observation_builder.build()` → `orchestrator.step(observation)` | `core/services/table_runtime.py` | 待實作 |
| `TableBallSet.get_ball_prim_paths()` | core/models | 回傳 10 顆球的 prim path 清單（依 ball_id 升冪排序），供 `BallMotionMonitor`/`ObservationBuilder` 使用 | `core/models/table_ball_set.py` | 已完成 |
| `TableBallSet.get_ball_radius()`（缺口） | core/models | 回傳 `self._ball_radius`，供 `ImpulseStrikingService` 建構子取得球半徑 | `core/models/table_ball_set.py` | 待實作 |
| `BilliardTable.get_table_ball_set()`（缺口） | core/models | 對外暴露內部的 `TableBallSet`，供 Extension 端組裝 `TableRuntime` 時取得 | `core/models/billiard_table.py` | 待實作 |
| `TableBallSet` 世界偏移量 | core/models | 建構子新增 `table_position: tuple[float, float] = (0.0, 0.0)`，`build()`/`reset()` 內部統一套用偏移量，語意一致（皆吃「相對桌台座標」） | `core/models/table_ball_set.py` | 已完成 |
| `Observation` 死欄位移除 | core/models | 移除 `joint_angles`、`shot_params`（設計演進留下的死欄位，從未被消費，真正承載擊球參數的是 `Action` 的 RL 規格欄位） | `core/models/observation.py` | 已完成 |
| Extension timeline 生命週期（訓練桌） | extension | `on_startup` 訂閱 `omni.timeline` PLAY/STOP 事件；PLAY 時呼叫 `articulation_api.initialize()` + 建立訓練桌 `TableRuntime` 清單 + 註冊 `world.add_physics_callback` tick；STOP 時重置 guard flag | `extension/billiard_digital_twin/billiard_digital_twin.py` | 待實作（僅訓練桌，Demo 桌延後） |

三個 Orchestrator 類別與 `ImpulseStrikingService`、`BallMotionMonitor`、`ObservationBuilder` 同層——屬於組合多個 model/port 的協調邏輯，不是純決策（`ScriptController`）也不是單一資源擁有者（`TableBallSet`、`UR5Robot`）。`ErrorState` 是跨 `TableOrchestrator`/`ObservationBuilder` 共享的輕量狀態物件，兩者建構時注入同一個 instance。`TableRuntime` 則是更上一層的組裝容器，不持有狀態，只負責把每個 tick 的兩個步驟串起來。

---

## 3. 類別設計

### ControllerBase.get_current_state()（已完成）

**職責：** 要求所有 Controller 回傳目前所在的 `BilliardStatus`。狀態單一
事實來源維持在具體 Controller 內部，不複製到 `Observation` 或 `Action`。

**介面（現況程式碼）：**
```python
class ControllerBase(ABC):
    @abstractmethod
    def get_current_state(self) -> BilliardStatus: ...


class ScriptController(ControllerBase):
    def __init__(self) -> None:
        self._current_state = BilliardStatus.RESET

    def _change_state(self, status: BilliardStatus):
        self._current_state = status

    def get_current_state(self) -> BilliardStatus:
        return self._current_state

    def reset(self):
        self._change_state(BilliardStatus.RESET)
```

`get_action()` 可以同步轉換狀態；`TableOrchestrator.step()` 在其後立即呼叫
`get_current_state()`，因此回傳值必須是該 `Action` 對應的分派狀態。

**依賴：**
- 輸入來源：無（純讀內部欄位）
- 輸出去向：`TableOrchestrator.step()` 用於分派下游動作；`TableOrchestrator.reset()` 呼叫 Controller 的 `reset()`

---

### TableOrchestrator（抽象基底）

**職責：** 定義共用執行骨架：取得 `Action` → 依 `get_current_state()` 分派下游動作 → 若下游動作拋出例外則交給 `ErrorState` 記錄，不重新拋出。另提供 `reset()` 統一入口，同時清除 `ErrorState` 並讓狀態機回到 `RESET`。

**現況程式碼（骨架已完成，尚未含 `ErrorState`/`reset()`）：**
```python
class TableOrchestrator(ABC):
    def __init__(
        self,
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
    ) -> None:
        self._script_controller = script_controller
        self._table_ball_set = table_ball_set
        self._ball_position_provider = ball_position_provider

    def step(self, observation: Observation) -> None:
        action = self._script_controller.get_action(observation)
        current_state = self._script_controller.get_current_state()
        if action.should_execute_action:
            match current_state:
                case BilliardStatus.RESET:
                    self._reset_balls()
                    self._reset_downstream()
                case BilliardStatus.AIMING:
                    self._execute_aim(action)
                case BilliardStatus.STRIKING:
                    self._execute_strike(action)

    def _reset_balls(self) -> None:
        positions = self._ball_position_provider.get_positions()
        self._table_ball_set.reset(positions)

    @abstractmethod
    def _reset_downstream(self) -> None: ...

    @abstractmethod
    def _execute_aim(self, action: Action) -> None: ...

    @abstractmethod
    def _execute_strike(self, action: Action) -> None: ...
```

**待實作修改（本次範圍）：**
1. 建構子新增 `error_state: ErrorState` 參數。
2. `step()` 內把「分派下游動作」這段包進 `try/except Exception as e: self._error_state.mark_error(e)`（不重新拋出），確保單一桌子的下游錯誤不會中斷共用的 tick loop：
```python
def step(self, observation: Observation) -> None:
    action = self._script_controller.get_action(observation)
    current_state = self._script_controller.get_current_state()
    if action.should_execute_action:
        try:
            match current_state:
                case BilliardStatus.RESET:
                    self._reset_balls()
                    self._reset_downstream()
                case BilliardStatus.AIMING:
                    self._execute_aim(action)
                case BilliardStatus.STRIKING:
                    self._execute_strike(action)
        except Exception as e:
            self._error_state.mark_error(e)
```
3. 新增 `reset()` 方法，`clear()` 與 `script_controller.reset()` 必須同時發生（`ScriptController.get_action()` 判斷順序是 `observation.has_error` 優先於 `current_state`，若只清一邊會導致狀態機瞬間又跳回 `ERROR`）：
```python
def reset(self) -> None:
    """外部重新初始化用：清除錯誤旗標並讓狀態機回到 RESET，兩者必須同時發生。"""
    self._error_state.clear()
    self._script_controller.reset()
```

**依賴：**
- 輸入來源：`TableRuntime.tick()` 每次呼叫時傳入的 `Observation`（由 `ObservationBuilder` 組裝）
- 輸出去向：內部透過 `ScriptController.get_action()` 取得決策，透過 `TableBallSet`/`UR5Robot`/`ImpulseStrikingService` 產生下游副作用；例外經 `ErrorState.mark_error()` 記錄

---

### DemoTableOrchestrator

**職責：** Demo 桌差異實作，RESET 呼叫 `UR5Robot.reset()`；AIMING/STRIKING 目前僅為 TODO 佔位，內容留給 #96/#97。

**現況程式碼：**
```python
class DemoTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        ur5_robot: UR5Robot,
        articulation_api: ArticulationAPI,
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider)
        self._ur5_robot = ur5_robot
        self._articulation_api = articulation_api

    def _reset_downstream(self) -> None:
        self._ur5_robot.reset()

    def _execute_aim(self, action: Action) -> None:
        # TODO: 把 action 轉譯成 ur5_robot 需要的操作（#96）
        ...

    def _execute_strike(self, action: Action) -> None:
        # TODO: 把 action 轉譯成 ur5_robot 需要的操作（#97）
        ...
```

**待實作修改：** 建構子新增 `error_state: ErrorState`，傳給 `super().__init__(...)`。

**依賴：**
- 輸入來源：`TableOrchestrator` 共用骨架、`UR5Robot`（`reset()`/`is_reset_complete()` 皆已存在）
- 輸出去向：`ArticulationAPI` 實作層（`isaac_sim_impl_6_0/`，透過 RmpFlow 執行實際路徑規劃）；本次 Demo 桌的 `TableRuntime` 組裝與 `ArticulationAPI.initialize()` 串接不在範圍內

---

### TrainingTableOrchestrator

**職責：** 訓練桌差異實作，沒有手臂，RESET/AIMING 皆為 no-op 或恆真，STRIKING 呼叫既有 `ImpulseStrikingService.strike()` 直接對母球賦予衝量速度。

**現況程式碼：**
```python
class TrainingTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        impulse_striking_service: ImpulseStrikingService,
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider)
        self._impulse_striking_service = impulse_striking_service

    def _reset_downstream(self) -> None:
        pass  # 沒有手臂，沒有其他需要處理的 reset 元件

    def _execute_aim(self, action: Action) -> None:
        pass  # 訓練桌沒有瞄準動作，隨時可以準備好擊球

    def _execute_strike(self, action: Action) -> None:
        self._impulse_striking_service.strike(action, table_z=self._table_ball_set.get_table_z())
```

**待實作修改：** 建構子新增 `error_state: ErrorState`，傳給 `super().__init__(...)`。

**依賴：**
- 輸入來源：`TableOrchestrator` 共用骨架、`ImpulseStrikingService`（已存在，對應 Issue #177）
- 輸出去向：`RigidBodyAPI.set_velocities()`（由 `ImpulseStrikingService` 內部呼叫）

---

### ErrorState（新元件，待實作）

**職責：** 集中記錄下游動作執行時發生的例外，讓 `TableOrchestrator` 可以「不重新拋出」（避免一張桌子的錯誤讓共用 tick loop 的其他桌子跟著中斷），同時保留可見性（完整 log + `get_last_exception()` 事後查詢）。

**介面（設計定案）：**
```python
import logging

logger = logging.getLogger(__name__)


class ErrorState:
    def __init__(self) -> None:
        self._has_error = False
        self._last_exception: Exception | None = None

    def mark_error(self, exception: Exception) -> None:
        logger.exception("下游執行發生例外", exc_info=exception)
        self._has_error = True
        self._last_exception = exception

    def has_error(self) -> bool:
        return self._has_error

    def get_last_exception(self) -> Exception | None:
        return self._last_exception

    def clear(self) -> None:
        self._has_error = False
        self._last_exception = None
```

**依賴：**
- 輸入來源：`TableOrchestrator.step()` 的 `except Exception as e:` 分支呼叫 `mark_error(e)`
- 輸出去向：`ObservationBuilder.build()` 讀 `has_error()` 寫入 `Observation.has_error`；`TableOrchestrator.reset()` 呼叫 `clear()`（必須與 `ScriptController.reset()` 同時發生，見第 8 節理由）

**注意：** 每張桌子（Demo/Training）各自持有一個獨立的 `ErrorState` instance，`TableOrchestrator` 與對應的 `ObservationBuilder` 建構時注入同一個 instance，不與其他桌子共用。

---

### ObservationBuilder（抽象基底，待實作）

**職責：** 每個 tick 完整組裝一份 `Observation`。**不接受 `previous_observation` 參數**——Builder 的職責就是完整生成，接受既有值再局部修改不符合這個職責定位，因此 `Observation` 每個欄位都要有獨立的即時查詢來源。

**介面（設計定案）：**
```python
from abc import ABC, abstractmethod

from ..models.observation import Observation
from ..models.table_ball_set import TableBallSet
from ..ports.rigid_body_api import RigidBodyAPI
from .ball_motion_monitor import BallMotionMonitor
from .ball_position_provider import BallPositionProvider
from .error_state import ErrorState

_INIT_STATE_TOLERANCE_M = 0.005  # 5mm


class ObservationBuilder(ABC):
    def __init__(
        self,
        table_ball_set: TableBallSet,
        rigid_body_api: RigidBodyAPI,
        ball_motion_monitor: BallMotionMonitor,
        ball_position_provider: BallPositionProvider,
        table_position: tuple[float, float],
        error_state: ErrorState,
    ) -> None:
        """組合建構：所有依賴皆由外部注入"""
        ...

    def build(self) -> Observation:
        """
        每個 tick 組裝一份完整的 Observation：
        1. ball_positions：迴圈 table_ball_set.get_ball_prim_paths()，逐一呼叫 rigid_body_api.get_position()
        2. cue_ball_position = ball_positions[0]（白球固定 ball_id=0，見 ball_colors.py 白色定義）
        3. is_init_state：ball_position_provider.get_positions()（相對座標）逐球加上 table_position
           換算世界座標，跟 ball_positions 逐球比對歐氏距離，容許誤差 5mm（0.005m），
           任一顆球超出誤差即為 False
        4. is_ball_moving = ball_motion_monitor.is_any_ball_moving()
        5. is_motion_complete = self._is_downstream_motion_complete()（抽象方法，交由子類別）
        6. has_error = error_state.has_error()
        """
        ...

    @abstractmethod
    def _is_downstream_motion_complete(self) -> bool:
        """Demo：ur5_robot.is_reset_complete()；Training：恆 True"""
        ...
```

**依賴：**
- 輸入來源：`TableBallSet`（取得球 prim path 清單）、`RigidBodyAPI`（即時位置查詢）、`BallMotionMonitor`（是否還在移動）、`BallPositionProvider`（初始擺球座標，判斷 `is_init_state`）、`ErrorState`（是否有下游錯誤）
- 輸出去向：`TableRuntime.tick()` 拿到組裝好的 `Observation`，傳給 `TableOrchestrator.step()`

---

### DemoObservationBuilder（待實作）

**職責：** Demo 桌差異實作，額外注入 `UR5Robot`，動作完成度需同時考慮手臂是否歸位。

**介面（設計定案）：**
```python
class DemoObservationBuilder(ObservationBuilder):
    def __init__(
        self,
        table_ball_set: TableBallSet,
        rigid_body_api: RigidBodyAPI,
        ball_motion_monitor: BallMotionMonitor,
        ball_position_provider: BallPositionProvider,
        table_position: tuple[float, float],
        error_state: ErrorState,
        ur5_robot: UR5Robot,
    ) -> None:
        super().__init__(table_ball_set, rigid_body_api, ball_motion_monitor, ball_position_provider, table_position, error_state)
        self._ur5_robot = ur5_robot

    def _is_downstream_motion_complete(self) -> bool:
        return self._ur5_robot.is_reset_complete()
```

**依賴：**
- 輸入來源：`ObservationBuilder` 共用骨架、`UR5Robot.is_reset_complete()`
- 輸出去向：同 `ObservationBuilder`

---

### TrainingObservationBuilder（待實作）

**職責：** 訓練桌差異實作，沒有手臂，動作完成度只需球 reset 完成即可。

**介面（設計定案）：**
```python
class TrainingObservationBuilder(ObservationBuilder):
    def _is_downstream_motion_complete(self) -> bool:
        return True
```

**依賴：**
- 輸入來源：`ObservationBuilder` 共用骨架
- 輸出去向：同 `ObservationBuilder`

---

### TableRuntime（新元件，待實作）

**職責：** 無狀態容器，把一組 `(TableOrchestrator, ObservationBuilder)` 包在一起，`tick()` 依序呼叫兩者。不持有任何狀態（因為 `ObservationBuilder.build()` 不吃 `previous_observation`）。

**介面（設計定案）：**
```python
class TableRuntime:
    def __init__(self, orchestrator: TableOrchestrator, observation_builder: ObservationBuilder) -> None:
        self._orchestrator = orchestrator
        self._observation_builder = observation_builder

    def tick(self) -> None:
        observation = self._observation_builder.build()
        self._orchestrator.step(observation)
```

**依賴：**
- 輸入來源：建構期注入的 `TableOrchestrator`、`ObservationBuilder`（每張桌子各一組，`ScriptController`、`BreakShotPositionProvider`、`BallMotionMonitor`、`ImpulseStrikingService`、`ErrorState` 皆各桌一份不共用；`RigidBodyAPI`/`StageAPI` 沿用既有共用單例）
- 輸出去向：Extension 端的 `_on_tick` physics callback，逐一呼叫每張桌子的 `runtime.tick()`

---

### TableBallSet.get_ball_prim_paths()（已完成）

**職責：** 回傳 10 顆球（Ball_0 ~ Ball_9）的完整 prim path 清單，依 ball_id 升冪排序。

**現況程式碼：**
```python
def get_ball_prim_paths(self) -> list[str]:
    """回傳 10 顆球的 prim path 清單"""
    return self._ball_prim_list
```

**依賴：**
- 輸入來源：`build()`/`reset()` 過程中累積的 `self._ball_prim_list`
- 輸出去向：`BallMotionMonitor` 建構子、`ObservationBuilder.build()` 組 `ball_positions`

---

### TableBallSet.get_ball_radius()（缺口，待實作）

**職責：** 回傳建構時注入的球半徑，供 `ImpulseStrikingService` 建構子使用（目前該建構子的 `ball_radius` 參數只能由呼叫端另外硬編碼取得，缺乏單一事實來源）。

**介面（設計定案）：**
```python
def get_ball_radius(self) -> float:
    return self._ball_radius
```

**依賴：**
- 輸入來源：`self._ball_radius`（建構時已注入，預設 `0.028575`）
- 輸出去向：Extension 端組裝 `ImpulseStrikingService` 時呼叫

---

### BilliardTable.get_table_ball_set()（缺口，待實作）

**職責：** 對外暴露內部持有的 `TableBallSet` instance，供 Extension 端組裝 `TableRuntime`（`TableOrchestrator`/`ObservationBuilder` 皆需要 `TableBallSet`）時取得，目前 `BilliardTable` 完全沒有對外暴露這個內部物件。

**介面（設計定案）：**
```python
def get_table_ball_set(self) -> TableBallSet:
    return self._table_set
```

**依賴：**
- 輸入來源：建構子已建立的 `self._table_set`
- 輸出去向：Extension 端組裝 `TrainingTableOrchestrator`/`TrainingObservationBuilder`/`ImpulseStrikingService` 時取得

---

## 4. 資料流

**RESET 狀態（已完成，本次落地的核心流程）：**
```
TableRuntime.tick()
  → observation = observation_builder.build()
      → ball_positions：迴圈 get_ball_prim_paths() × rigid_body_api.get_position()
      → is_init_state：比對 ball_position_provider.get_positions() + table_position 換算世界座標（容許誤差 5mm）
      → is_ball_moving = ball_motion_monitor.is_any_ball_moving()
      → is_motion_complete = _is_downstream_motion_complete()（Demo：ur5_robot.is_reset_complete()；Training：True）
      → has_error = error_state.has_error()
  → orchestrator.step(observation)
    → action = script_controller.get_action(observation)   # 純決策，不接觸執行層
    → current_state = script_controller.get_current_state()
    → 若 action.should_execute_action == True 且 current_state == RESET：
        → try:
            → self._reset_balls()
              → ball_position_provider.get_positions() → table_ball_set.reset(positions)
                → StageAPI.set_prim_translate(...) ×10（Teleport，同 tick 完成，含 table_position 世界偏移量）
                → RigidBodyAPI.set_velocities(..., [0,0,0], [0,0,0]) ×10
            → self._reset_downstream()
              → Demo：ur5_robot.reset() → articulation_api.move_to_home()（RmpFlow，動畫式，多 tick）
              → Training：no-op（沒有手臂）
          except Exception as e:
            → error_state.mark_error(e)（log + 記錄，不重新拋出）
  → 下一個 tick：observation_builder 重新查詢一次，is_motion_complete 更新後
    script_controller.get_action() 才會判斷是否轉換至 IDLE
```

**STRIKING 狀態（Training 桌，本次僅定義介面）：**
```
TableRuntime.tick()
  → observation = training_observation_builder.build()
  → orchestrator.step(observation)
    → action = script_controller.get_action(observation)
    → 若 action.should_execute_action == True 且 current_state == STRIKING：
        → try: self._execute_strike(action)
            → impulse_striking_service.strike(action, table_z)
              → StageAPI.set_prim_translate(cue_ball_prim, x, y, table_z)
              → compute_cue_ball_velocities(action, ball_radius, spin_efficiency)
              → RigidBodyAPI.set_velocities(cue_ball_prim, linear_velocity, angular_velocity)
          except Exception as e: error_state.mark_error(e)
  → 下一個 tick 的 observation_builder.build()：is_ball_moving 應變為 True
```

**外部重新初始化流程（`TableOrchestrator.reset()`，待實作）：**
```
呼叫端（例如 DebugMenu 的重新初始化操作）
  → table_runtime.orchestrator.reset()
    → error_state.clear()          # 兩者必須同時發生
    → script_controller.reset()    # 否則下一個 tick 仍讀到 has_error=True，狀態機瞬間又跳回 ERROR
```

**Extension 端訓練桌 timeline 生命週期（待實作）：**
```
on_startup
  → 訂閱 omni.timeline 的 PLAY/STOP 事件（stage-open 時機仍先建場景/prim/ArticulationAPIImpl 實例，不呼叫 initialize()）

PLAY 事件觸發（_on_timeline_event）
  → 若 self._runtime_initialized: return（防止 Stop→Play 重複觸發）
  → articulation_api.initialize()（Demo 桌手臂，本次先確保不再是永遠 None）
  → 為每張訓練桌組出 TableRuntime（各自獨立的 ScriptController/BreakShotPositionProvider/
    BallMotionMonitor/ImpulseStrikingService/ErrorState，RigidBodyAPI/StageAPI 沿用共用單例）
  → world.add_physics_callback("billiard_table_tick", self._on_tick)
  → self._runtime_initialized = True

_on_tick（physics callback，每步呼叫）
  → 若 not self._training_enabled: return（閘門，沒開 training 完全不 tick，狀態機暫停在原地）
  → 逐一呼叫 runtime.tick()（每張訓練桌各一次）

STOP 事件觸發
  → self._runtime_initialized = False（下次 PLAY 才會重新初始化）
```

---

## 5. 依賴關係圖

```
TableRuntime（每張桌子一組，無狀態）
  ├── 依賴 TableOrchestrator（DemoTableOrchestrator / TrainingTableOrchestrator）
  └── 依賴 ObservationBuilder（DemoObservationBuilder / TrainingObservationBuilder）

DemoTableOrchestrator
  ├── 依賴 ScriptController（取得純決策 Action，共用）
  ├── 依賴 TableBallSet（RESET 球重置，共用；get_ball_prim_paths() 供建構期使用）
  ├── 依賴 BallPositionProvider（RESET 重置座標的來源，共用）
  ├── 依賴 UR5Robot（RESET 手臂歸位，Demo 專屬）
  ├── 依賴 ArticulationAPI（AIMING/STRIKING 手臂路徑規劃，Demo 專屬，經 UR5Robot 或直接注入）
  └── 依賴 ErrorState（下游動作例外記錄，與同桌的 DemoObservationBuilder 共用同一個 instance）

TrainingTableOrchestrator
  ├── 依賴 ScriptController（共用，與 Demo 同一個類別）
  ├── 依賴 TableBallSet（共用）
  ├── 依賴 BallPositionProvider（共用）
  ├── 依賴 ImpulseStrikingService（STRIKING 衝量式擊球，Training 專屬）
  │     └── 依賴 RigidBodyAPI.set_velocities（外部物理引擎 API）
  └── 依賴 ErrorState（與同桌的 TrainingObservationBuilder 共用同一個 instance）

DemoObservationBuilder / TrainingObservationBuilder
  ├── 依賴 TableBallSet（get_ball_prim_paths()）
  ├── 依賴 RigidBodyAPI（get_position() 逐球查詢）
  ├── 依賴 BallMotionMonitor（is_any_ball_moving()）
  ├── 依賴 BallPositionProvider（is_init_state 比對基準）
  ├── 依賴 ErrorState（has_error()）
  └── DemoObservationBuilder 額外依賴 UR5Robot（is_reset_complete()）

ErrorState
  └── 無外部依賴，純記憶體狀態，跨 TableOrchestrator/ObservationBuilder 共享

TableBallSet
  └── 依賴 StageAPI / RigidBodyAPI / MaterialAPI（既有，Port 抽象層）

ScriptController
  └── 不依賴任何執行層 API（純讀 Observation、純回傳 Action，見 5-2 文件第 5.2 節）

BilliardTable
  └── 依賴 TableBallSet（持有並透過 get_table_ball_set() 對外暴露，供 Extension 端組裝 TableRuntime 使用）

Extension（billiard_digital_twin.py）
  ├── 依賴 omni.timeline（PLAY/STOP 事件，決定何時呼叫 ArticulationAPIImpl.initialize() 與建立 TableRuntime）
  └── 依賴 world.add_physics_callback（訓練桌 tick loop 的掛載點，經 training_enabled 旗標閘門）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 | 狀態 |
|---|---|---|
| `action.should_execute_action == False`（同一狀態持續中的非觸發 tick） | 跳過 `_reset_downstream()`/`_execute_aim()`/`_execute_strike()`，`ObservationBuilder` 仍照常每 tick 完整組裝下一份 `Observation` | 已完成 |
| Demo 桌 RESET：球已 reset 完成但手臂尚未到位 | `DemoObservationBuilder._is_downstream_motion_complete()` 回傳 `False`，`Observation.is_motion_complete` 為 `False`，`ScriptController` 停留在 `RESET`，下個 tick 繼續檢查直到手臂到位 | 待實作（`ObservationBuilder`） |
| Training 桌 RESET | 沒有手臂，`_is_downstream_motion_complete()` 恆 `True`，`is_motion_complete` 只反映球的狀態，理論上同一 tick 即可轉換至 `IDLE` | 待實作（`ObservationBuilder`） |
| `observation.has_error == True` | 由 `ScriptController.get_action()` 內部優先判斷轉入 `ERROR`（優先序高於 `current_state`）；`TableOrchestrator.step()` 本身不做額外錯誤攔截，僅依回傳的 `Action`/狀態走既定分派邏輯，`ERROR` 狀態下 `should_execute_action` 恆為 `False` | 已完成（`ScriptController` 判斷邏輯） |
| `TableOrchestrator.step()` 內下游動作（`_reset_downstream`/`_execute_aim`/`_execute_strike`）拋出任何例外 | 包在 `try/except Exception as e: self._error_state.mark_error(e)`，**不重新拋出**：避免單一桌子的下游錯誤讓共用 tick loop 的其他桌子跟著中斷；`mark_error()` 內部用 `logger.exception()` 完整記錄含 traceback，並保留 `get_last_exception()` 供事後查詢 | 待實作（`ErrorState` + `TableOrchestrator` 修改） |
| `BallMotionMonitor.is_any_ball_moving()` 內部呼叫 `get_linear_velocity()` 拋出例外 | `BallMotionMonitor` 內部已 `try/except` 並 log 後 re-raise；此呼叫位於 `ObservationBuilder.build()` 內，屬於「查詢」而非「下游動作」，不經 `ErrorState` 攔截，例外會直接往上拋給 `TableRuntime.tick()` 的呼叫端（未來 Extension physics callback） | 已完成（`BallMotionMonitor`）／`ObservationBuilder` 呼叫點待實作 |
| `is_init_state` 判定容許誤差 | 逐球比對 `ball_position_provider.get_positions()`（換算世界座標後）與 `ball_positions` 的歐氏距離，容許誤差 `5mm`（`0.005`），任一顆球超出誤差即視為 `False` | 待實作 |
| `ArticulationAPIImpl.initialize()` 從未被呼叫（Demo 桌 RESET 每次都失敗） | Extension `_billiard_init()`（stage-open）只建構 `ArticulationAPIImpl` 實例，不呼叫 `initialize()`；改為訂閱 `omni.timeline` PLAY 事件，在 timeline 真正播放後才呼叫 `initialize()`（docstring 本來就要求「在 timeline play 之後呼叫」） | 待實作 |
| Stop→Play 會重複觸發 PLAY 事件，且 Stop 會銷毀 physics view 讓 `initialize()` 內部 handle 失效 | 用 `self._runtime_initialized` guard flag：PLAY 時若已初始化則直接 `return`；STOP 時重置為 `False`，下次 PLAY 才會重新初始化 | 待實作 |
| 沒開啟 `training_enabled` 時的 tick loop | 閘門邏輯放在 Extension 層的 `_on_tick`：`if not self._training_enabled: return`，完全不呼叫任何 `runtime.tick()`，狀態機整個暫停在原地，而不是持續 tick 但下游動作被其他機制擋住 | 待實作 |
| 一次性 physics callback（Demo 桌 `_capture_home_position_once`）在 Stop 發生但尚未觸發 | 待實測項目（記錄於 `skills/isaac_sim_6_api_cache.md`），非本次阻塞；若實測發現有殘留 callback，可能需額外呼叫 `world.remove_physics_callback(...)` 清理 | 待確認（不阻塞本次範圍） |

---

## 7. 測試涵蓋（對應 Unit Test）

| 測試案例 | 測試檔案 | 說明 | 狀態 |
|---|---|---|---|
| test_step_dispatches_reset_when_should_execute_action_true | core/tests/test_table_orchestrator.py | `should_execute_action=True` 且狀態為 RESET 時，正確呼叫 `table_ball_set.reset()` 與 `_reset_downstream()` | 已完成 |
| test_step_skips_downstream_when_should_execute_action_false | core/tests/test_table_orchestrator.py | `should_execute_action=False` 時不觸發任何下游動作方法 | 已完成 |
| test_demo_reset_downstream_calls_ur5_reset | core/tests/test_table_orchestrator.py | `DemoTableOrchestrator._reset_downstream()` 呼叫 `ur5_robot.reset()` | 已完成 |
| test_training_reset_downstream_noop | core/tests/test_table_orchestrator.py | `TrainingTableOrchestrator._reset_downstream()` 為 no-op，不拋例外 | 已完成 |
| test_training_execute_strike_calls_impulse_service | core/tests/test_table_orchestrator.py | `TrainingTableOrchestrator._execute_strike()` 正確呼叫 `impulse_striking_service.strike(action, table_z)` | 已完成 |
| test_get_ball_prim_paths_returns_ten_paths_in_order | core/tests/test_models.py | `TableBallSet.get_ball_prim_paths()` 回傳 10 筆、依 ball_id 0-9 升冪排序 | 已完成 |
| test_table_ball_set_applies_table_position_offset | core/tests/test_models.py | `TableBallSet.build()`/`reset()` 皆正確套用 `table_position` 世界偏移量 | 已完成 |
| test_error_state_mark_error_records_and_does_not_raise | core/tests/test_error_state.py | `mark_error()` 記錄例外、`has_error()` 變為 `True`，且不重新拋出 | 待實作 |
| test_error_state_get_last_exception_returns_recorded_exception | core/tests/test_error_state.py | `get_last_exception()` 回傳 `mark_error()` 記錄的同一個例外物件 | 待實作 |
| test_error_state_clear_resets_flag_and_exception | core/tests/test_error_state.py | `clear()` 後 `has_error()` 為 `False`、`get_last_exception()` 為 `None` | 待實作 |
| test_step_catches_downstream_exception_and_marks_error | core/tests/test_table_orchestrator.py | 下游動作（`_reset_downstream`/`_execute_aim`/`_execute_strike`）拋出例外時，`step()` 不往外拋，且呼叫 `error_state.mark_error(e)` | 待實作 |
| test_orchestrator_reset_clears_error_state_and_script_controller | core/tests/test_table_orchestrator.py | `TableOrchestrator.reset()` 同時呼叫 `error_state.clear()` 與 `script_controller.reset()` | 待實作 |
| test_observation_builder_builds_ball_positions_from_rigid_body_api | core/tests/test_observation_builder.py | `build()` 迴圈 `get_ball_prim_paths()` 逐一呼叫 `rigid_body_api.get_position()` 組出 `ball_positions` | 待實作 |
| test_observation_builder_cue_ball_position_is_ball_zero | core/tests/test_observation_builder.py | `cue_ball_position` 等於 `ball_positions[0]` | 待實作 |
| test_observation_builder_is_init_state_true_within_tolerance | core/tests/test_observation_builder.py | 球位置與 `ball_position_provider` 換算世界座標誤差在 5mm 內時 `is_init_state=True` | 待實作 |
| test_observation_builder_is_init_state_false_outside_tolerance | core/tests/test_observation_builder.py | 任一顆球誤差超出 5mm 時 `is_init_state=False` | 待實作 |
| test_observation_builder_has_error_reflects_error_state | core/tests/test_observation_builder.py | `error_state.has_error()==True` 時 `Observation.has_error` 為 `True` | 待實作 |
| test_observation_builder_does_not_accept_previous_observation | core/tests/test_observation_builder.py | `build()` 簽名不接受任何既有 `Observation` 參數（介面層級的設計驗證） | 待實作 |
| test_demo_observation_builder_motion_complete_uses_ur5_reset_complete | core/tests/test_observation_builder.py | `DemoObservationBuilder._is_downstream_motion_complete()` 回傳 `ur5_robot.is_reset_complete()` | 待實作 |
| test_training_observation_builder_motion_complete_always_true | core/tests/test_observation_builder.py | `TrainingObservationBuilder._is_downstream_motion_complete()` 恆回傳 `True` | 待實作 |
| test_table_runtime_tick_builds_observation_then_steps_orchestrator | core/tests/test_table_runtime.py | `tick()` 依序呼叫 `observation_builder.build()` 再把結果傳給 `orchestrator.step()`，且順序正確 | 待實作 |
| test_table_runtime_holds_no_internal_state | core/tests/test_table_runtime.py | 連續呼叫多次 `tick()`，`TableRuntime` 本身不緩存任何跨 tick 資料（僅委派） | 待實作 |
| test_get_ball_radius_returns_constructed_value | core/tests/test_models.py | `TableBallSet.get_ball_radius()` 回傳建構時注入的 `ball_radius` | 待實作 |
| test_billiard_table_get_table_ball_set_returns_internal_instance | core/tests/test_models.py | `BilliardTable.get_table_ball_set()` 回傳與內部 `self._table_set` 相同的 instance | 待實作 |
| test_timeline_play_calls_articulation_initialize_once | extension 手動驗證 或 core/tests（依實際可測試邊界拆分） | PLAY 事件觸發時呼叫 `articulation_api.initialize()`；重複 PLAY（`_runtime_initialized=True`）不重複呼叫 | 待實作 |
| test_timeline_stop_resets_runtime_initialized_guard | 同上 | STOP 事件觸發後 `_runtime_initialized` 變為 `False` | 待實作 |
| test_on_tick_skips_when_training_disabled | 同上 | `training_enabled=False` 時 `_on_tick` 不呼叫任何 `runtime.tick()` | 待實作 |

---

## 8. 待決定事項

- [x] **狀態分派的資料來源（已定案 2026-07-21，#111 於 2026-07-29 補正介面）**：採方案 (b)——由 Controller 提供 `get_current_state() -> BilliardStatus`，供 `TableOrchestrator.step()` 查詢後分派下游動作。理由：狀態單一事實來源維持在具體 Controller 內部，不需要在 `Observation`/`Action` 額外複製一份狀態、不用擔心同步問題。最初只將方法加到 `ScriptController`；#111 檢查發現 `TableOrchestrator` 以 `ControllerBase` 型別呼叫此方法，因此已將它提升為 `ControllerBase` 的抽象契約。
- [x] **`TableBallSet.reset(positions)` 座標來源與世界偏移量（已定案 2026-07-22）**：`_reset_balls()` 呼叫 `ball_position_provider.get_positions()`（`BreakShotPositionProvider`）取得桌台相對座標，直接傳給 `table_ball_set.reset(positions)`。過程中發現既有 bug：`TableBallSet.build()`/`reset()` 原本完全不處理世界偏移量，全靠呼叫端（`BilliardTable.__init__`）自己在呼叫 `build()` 前手動加總 `x_pos`/`y_pos`，但 `TableOrchestrator._reset_balls()` 繞過 `BilliardTable`、直接呼叫 `TableBallSet.reset()`，導致非原點桌子（Demo 桌以外的所有訓練桌）reset 後球會出現在錯誤的世界座標。修法：`TableBallSet` 建構子新增 `table_position: tuple[float, float] = (0.0, 0.0)`，`build()`/`reset()` 內部統一套用偏移量，兩者語意從此一致（皆吃「相對桌台座標」）；`BilliardTable.__init__` 拿掉手動偏移量加總，改傳 `table_position` 給 `TableBallSet`。
- [ ] `DemoTableOrchestrator` 建構子是否需要同時注入 `UR5Robot` 與 `ArticulationAPI`，或只需注入 `UR5Robot`（`ArticulationAPI` 透過 `UR5Robot` 內部間接使用）——AIMING/STRIKING 的 `move_to_pose`/`execute_strike` 目前只存在於 `ArticulationAPI`，`UR5Robot` 本身未包裝這两个方法，待 #96/#97 实作時一併確認是否要在 `UR5Robot` 新增對應包裝方法。
- [ ] Extension tick／physics callback 尚未存在，`TableOrchestrator.step()` 目前沒有任何生產程式碼呼叫入口，串接時機留待後續任務。
- [x] **`step()` 職責拆分（已定案 2026-07-21）**：`TableOrchestrator.step()` 不再負責組裝/回傳 `Observation`，改回 `step(observation: Observation) -> None`，只保留「決策 + 執行下游動作」；`_is_downstream_motion_complete()` 抽象方法移除。理由：查詢 `is_ball_moving`/`is_motion_complete` 的時機（tick 開始前查 vs. tick 結束後查）在物理上等價（physics callback 與 orchestrator tick 是兩條獨立時間軸），純粹是程式碼歸屬問題；把「執行動作」（命令式）與「組裝 Observation」（查詢式）拆成兩個獨立職責更符合單一職責原則。
  - 「組裝 `Observation`」這段查詢邏輯（`BallMotionMonitor.is_any_ball_moving()`、Demo 桌另需 `UR5Robot.is_reset_complete()`）改移到**新增的對稱元件**：`DemoObservationBuilder`/`TrainingObservationBuilder`，比照 `DemoTableOrchestrator`/`TrainingTableOrchestrator` 依桌子類型分兩種（Demo 桌需要多查手臂完成度、Training 桌不用）。這兩個 Builder 類別本次僅記錄決策，介面與程式碼待後續回合設計。
  - 呼叫端（未來 Extension tick loop）的順序會變成：`ObservationBuilder` 組出本次 tick 的 `Observation` → 呼叫 `TableOrchestrator.step(observation)` 執行動作 → 下一個 tick 再由 `ObservationBuilder` 重新查詢一次。
- [x] **`ObservationBuilder` 不吃 `previous_observation`（已定案 2026-07-22）**：`ObservationBuilder.build()` 不接受既有 `Observation` 當輸入參數——「Builder」的職責就是完整生成一份 `Observation`，接受既有值再局部修改不符合這個職責定位。因此 `Observation` 的每一個欄位都要有獨立的即時查詢來源，而不是靠 `dataclasses.replace()` 沿用舊資料。盤點後處理如下：
  - `is_ball_moving`／下游動作完成度：沿用 `BallMotionMonitor`／`UR5Robot.is_reset_complete()`，跟先前設計一致。
  - `ball_positions`／`cue_ball_position`：迴圈 `table_ball_set.get_ball_prim_paths()` 逐一呼叫 `rigid_body_api.get_position()`；白球固定是 `ball_id=0`（依 `ball_colors.py` 白色 `[1.0, 1.0, 1.0]` 確認），對應 `get_ball_prim_paths()[0]`。
  - `joint_angles`、`shot_params`：追查後確認是設計演進留下的死欄位（`shot_params` 前身是最早期泛用機械手臂模板的 `target_position`，改名後從未被賦予撞球語意也從未被消費；`Action` 的 6 維 RL 規格才是真正承載擊球參數的地方），**兩者已從 `Observation` 移除**（`core/models/observation.py`），不需要 `ObservationBuilder` 生成。
  - `is_init_state`：判斷邏輯（比對目前 `ball_positions` 與初始擺球位置，需考慮誤差範圍）本次尚未設計，留待 `ObservationBuilder` 實作回合處理。
- [x] **`is_init_state` 判定方式與 `has_error` 來源（已定案 2026-07-22）**：
  - `is_init_state`：`ObservationBuilder` 注入 `ball_position_provider` + 該桌 `table_position: tuple[float, float]`，將 `ball_position_provider.get_positions()`（相對座標）逐一加上 `table_position` 換算成世界座標，跟實際 `ball_positions` 逐球比對歐氏距離，容許誤差 `5mm`（`0.005`），任一顆球超出誤差即視為 `False`。
  - `has_error`：新增共享物件 `ErrorState`（`mark_error()` / `has_error() -> bool`），`TableOrchestrator` 與 `ObservationBuilder` 建構時注入同一個 instance。`TableOrchestrator.step()` 把下游動作分派包進 try/except，捕捉到例外時呼叫 `error_state.mark_error()`，**不重新拋出**（靜默吞掉，讓下一個 tick `ObservationBuilder` 讀到 `has_error=True`，`ScriptController` 自然轉進 `ERROR` 狀態），理由是避免單一桌子的下游執行錯誤讓共用的 Extension tick loop（若多桌共用同一個 physics callback）整個中斷。`ObservationBuilder.build()` 直接讀 `error_state.has_error()` 寫入 `Observation.has_error`。
- [x] **`ErrorState.clear()` 機制（已定案 2026-07-22）**：`ErrorState` 新增 `clear() -> None`。`clear()` 必須跟 `ScriptController.reset()` 同時發生，不能分開呼叫——因為 `ScriptController.get_action()` 判斷順序是 `observation.has_error` 優先於 `current_state`，若只呼叫 `script_controller.reset()` 而沒清除 `ErrorState`，下一個 tick 仍會讀到 `has_error=True`，狀態機會瞬間又跳回 `ERROR`，讓 `reset()` 白做。因此不讓外部個別呼叫 `error_state.clear()`/`script_controller.reset()`，改由 `TableOrchestrator` 開一個統一入口：

  ```python
  def reset(self) -> None:
      """外部重新初始化用：清除錯誤旗標並讓狀態機回到 RESET，兩者必須同時發生。"""
      self._error_state.clear()
      self._script_controller.reset()
  ```

  未來 Extension 端（例如 `DebugMenu` 的重新初始化操作）只需呼叫 `table_runtime.orchestrator.reset()`，不需要知道背後牽動 `ErrorState` 與 `ScriptController` 兩個物件。

- [x] **Extension tick loop（訓練桌，已定案 2026-07-22）**：`ArticulationAPI` 的注入方式（第 322 行待決定事項）延後到實作手臂操作時再處理；本次先讓訓練桌（衝量式擊球）的 tick loop 跑通。
  - **`TableRuntime` 不需要持有任何狀態**（因為 `ObservationBuilder.build()` 不吃 `previous_observation`），只是把 `ObservationBuilder`/`TableOrchestrator` 包在一起：
    ```python
    class TableRuntime:
        def __init__(self, orchestrator: TableOrchestrator, observation_builder: ObservationBuilder) -> None:
            self._orchestrator = orchestrator
            self._observation_builder = observation_builder

        def tick(self) -> None:
            observation = self._observation_builder.build()
            self._orchestrator.step(observation)
    ```
  - **新增缺口 getter**：`BilliardTable.get_table_ball_set() -> TableBallSet`（組裝 `TableRuntime` 需要拿到內部的 `TableBallSet`，目前沒有對外暴露）、`TableBallSet.get_ball_radius() -> float`（`ImpulseStrikingService` 建構子需要）。
  - `TrainingTableOrchestrator`/`TrainingObservationBuilder` 建構子皆新增 `error_state: ErrorState` 參數。
  - Extension 端每張訓練桌組出一組 `TableRuntime`（`ScriptController`、`BreakShotPositionProvider`、`BallMotionMonitor`、`ImpulseStrikingService`、`ErrorState` 皆各桌一份，不共用；`RigidBodyAPI`/`StageAPI` 沿用既有的共用單例）。
  - **`training_enabled` 作為 tick loop 的閘門**：`self._training_enabled`（現有旗標，目前無人消費）改為決定要不要呼叫 `runtime.tick()`——沒開 training 時完全不 tick，狀態機整個暫停在原地，而不是持續 tick 但下游動作被其他機制擋住。閘門邏輯放在 Extension 層，`TableRuntime`/`TableOrchestrator` 不需要知道這個開關存在：
    ```python
    def _on_tick(self, step_size: float) -> None:
        if not self._training_enabled:
            return
        for runtime in self._table_runtimes:
            runtime.tick()
    ```
    掛在 `world.add_physics_callback(...)`。
  - Demo 桌的 `TableRuntime` 組裝本次不處理，待 `ArticulationAPI` 注入方式定案後再補。

- [x] **`ErrorState` 錯誤可見性（已定案 2026-07-22）**：討論 Demo 桌串接時發現 `ArticulationAPIImpl.initialize()` 目前完全沒有被呼叫（見下一項），若照原設計靜默吞例外，這類錯誤會完全無法被發現。修正：保留「不重新拋出」（避免一張桌子的錯誤讓共用 tick loop 的其他桌子跟著中斷），但 `mark_error()` 強制留下痕跡：

  ```python
  import logging

  logger = logging.getLogger(__name__)

  class ErrorState:
      def __init__(self) -> None:
          self._has_error = False
          self._last_exception: Exception | None = None

      def mark_error(self, exception: Exception) -> None:
          logger.exception("下游執行發生例外", exc_info=exception)
          self._has_error = True
          self._last_exception = exception

      def has_error(self) -> bool:
          return self._has_error

      def get_last_exception(self) -> Exception | None:
          return self._last_exception

      def clear(self) -> None:
          self._has_error = False
          self._last_exception = None
  ```

  `TableOrchestrator.step()` 對應改成 `except Exception as e: self._error_state.mark_error(e)`。錯誤會完整記錄在 log（含 traceback），也能透過 `get_last_exception()` 事後查詢，但不會讓其他桌子的 tick 被中斷。

- [x] **`ArticulationAPIImpl.initialize()` 生命週期缺口與修法（已定案 2026-07-22）**：追查 Demo 桌串接可行性時發現，`extension/billiard_digital_twin/billiard_digital_twin.py` 建構了 `ArticulationAPIImpl` 但從未呼叫其 `initialize()`，導致 `self._articulation` 永遠是 `None`，任何 `move_to_home()`/`move_to_pose()` 呼叫都會拋 `AttributeError`（會被上面的 `ErrorState` 吞掉，但 Demo 桌會每次 RESET 都卡進 `ERROR`）。修法：把 Extension 的初始化拆成兩個時機：
  - **stage-open 時機**（現有 `_billiard_init()`）：建場景、建 prim、建 `ArticulationAPIImpl` 實例，但不呼叫 `initialize()`。
  - **timeline play 時機**（新增）：呼叫 `articulation_api.initialize()`，再建立 `TableRuntime` 清單並註冊 tick callback。理由：`initialize()` 內部會註冊一次性 physics callback 捕捉手臂 home 姿態，必須等 timeline 真的在播放才會被觸發（docstring 本來就寫「在 timeline play 之後呼叫」，但先前沒有對應的呼叫時機）。

  API 確認（Isaac Sim 6.0.0，見 `skills/isaac_sim_6_api_cache.md`）：訂閱寫法跟現有 stage-event 訂閱同一套模式（`omni.timeline.get_timeline_interface().get_timeline_event_stream()` + `int(omni.timeline.TimelineEventType.PLAY)`），可以在 `on_startup` 就建立訂閱，不需要等 stage 開啟。

  **邊界情況：Stop→Play 會重複觸發 PLAY 事件**，且 Stop 會銷毀 physics view，讓先前 `initialize()` 的內部 handle 失效。呼叫端需要自己防護，用一個旗標避免重複初始化、並在 Stop 時重置：

  ```python
  def _on_timeline_event(self, event: carb.events.IEvent) -> None:
      if event.type == int(omni.timeline.TimelineEventType.PLAY):
          if self._runtime_initialized:
              return
          self._articulation_api.initialize()
          self._table_runtimes = [
              self._build_training_runtime(table, self._stage_api, rigid_body_api)
              for table in self._tables
          ]
          self._world.add_physics_callback("billiard_table_tick", self._on_tick)
          self._runtime_initialized = True
      elif event.type == int(omni.timeline.TimelineEventType.STOP):
          self._runtime_initialized = False
  ```

  待實測項目（記錄在快取，非本次阻塞）：一次性 physics callback（`_capture_home_position_once`）在 Stop 發生但尚未觸發時，Kit/PhysX 是否會自動清除，若實測發現有殘留 callback，可能需要額外呼叫 `world.remove_physics_callback(...)` 清理。
