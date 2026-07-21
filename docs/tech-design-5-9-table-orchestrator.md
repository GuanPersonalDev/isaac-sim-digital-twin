# TableOrchestrator — 技術設計文件

> 生成時間：2026-07-20
>
> 所屬專案：isaac-sim-digital-twin
>
> 關聯 GitHub：延續 Issue #99「[5-8] 實作 RESET：場景重置 → 回到 IDLE」；對應 Block 5 任務 5-9「單次擊球循環跑通確認」
>
> 關聯文件：`docs/tech-design-5-2-script-controller-state-machine.md`（第 5 節「Controller／執行層職責分離」）

---

## 1. 功能概述

`TableOrchestrator` 是 Demo 桌／訓練桌各自獨立擁有的執行迴圈，負責在每個 tick 呼叫 `ScriptController.get_action()` 取得純決策結果 `Action`，再依 `BilliardStatus` 對應的狀態把 `Action` 轉譯成真正的下游副作用（球位置重置、手臂歸位、瞄準、擊球），最後把下游偵測到的結果回寫進下一個 tick 要用的 `Observation`。它填補了 Issue #99 遺留的缺口——`TableBallSet.reset()` 與 `UR5Robot.reset()`/`is_reset_complete()` 目前都只是「有能力被呼叫」，完全沒有生產程式碼真的呼叫它們。本次設計把「全狀態」（IDLE/AIMING/STRIKING/WAITING/RESET/ERROR）的呼叫流程都定義出來，但只有 RESET 是本次要接通的實作範圍，其餘狀態（AIMING/STRIKING）只定義介面留給 #96/#97（Demo 桌手臂路徑規劃）與 #177（訓練桌衝量式擊球）。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `TableOrchestrator` | core/services（抽象基底） | 共用執行迴圈骨架：呼叫 `ScriptController.get_action()`、依狀態分派下游動作、組裝下一個 `Observation` | `core/services/table_orchestrator.py`（新檔案，待建立） |
| `DemoTableOrchestrator` | core/services | Demo 桌差異實作：下游動作透過 `UR5Robot`/`ArticulationAPI` 執行（動畫式、多 tick） | `core/services/table_orchestrator.py`（新檔案，同檔） |
| `TrainingTableOrchestrator` | core/services | 訓練桌差異實作：手臂相關動作皆為 no-op 或恆真，STRIKING 呼叫 `ImpulseStrikingService`（瞬時、單 tick） | `core/services/table_orchestrator.py`（新檔案，同檔） |
| `TableBallSet.get_ball_prim_paths()` | core/models | 新增公開方法，回傳 10 顆球的 prim path 清單，供 `TableOrchestrator` 建構 `BallMotionMonitor` 使用 | `core/models/table_ball_set.py`（既有檔案，新增方法） |
| `ScriptController.get_current_state()` | core/controllers | 新增公開方法，回傳 `self._current_state`（`BilliardStatus`），供 `TableOrchestrator.step()` 查詢目前狀態以分派下游動作（見第 8 節決策） | `core/controllers/script_controller.py`（既有檔案，新增方法） |

以上三個 Orchestrator 類別與 `ImpulseStrikingService`、`BallMotionMonitor` 同層——屬於組合多個 model/port 的協調邏輯，不是純決策（`ScriptController`）也不是單一資源擁有者（`TableBallSet`、`UR5Robot`）。

---

## 3. 類別設計

### TableOrchestrator（抽象基底）

**職責：** 定義共用執行骨架：取得 `Action` → 依 `should_execute_action` 決定是否分派下游動作 → 查詢球是否還在移動 → 查詢下游動作是否完成 → 組裝並回傳下一個 tick 的 `Observation`。差異部分（RESET 的手臂處理、AIMING/STRIKING 的實際動作、動作完成判定）交由子類別實作。

**介面：**
```python
from abc import ABC, abstractmethod

from ..controllers.script_controller import ScriptController
from ..models.action import Action
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.table_ball_set import TableBallSet
from .ball_motion_monitor import BallMotionMonitor
from .ball_position_provider import BallPositionProvider


class TableOrchestrator(ABC):
    def __init__(
        self,
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_motion_monitor: BallMotionMonitor,
        ball_position_provider: BallPositionProvider,
    ) -> None:
        """組合建構：所有依賴皆由外部注入，不在內部 new 任何 model/port"""
        ...

    def step(self, observation: Observation) -> Observation:
        """
        每個 tick 呼叫一次的共用骨架：
        1. action = self._script_controller.get_action(observation)
        2. current_state = self._script_controller.get_current_state()（新增公開方法，回傳 BilliardStatus，狀態單一事實來源仍在 ScriptController）
        3. 依 current_state 分派：
           - RESET    → 若 action.should_execute_action：self._reset_balls() + self._reset_downstream()
           - AIMING   → 若 action.should_execute_action：self._execute_aim(action)
           - STRIKING → 若 action.should_execute_action：self._execute_strike(action)
           - WAITING / IDLE → 無下游動作
        4. is_ball_moving = self._ball_motion_monitor.is_any_ball_moving()
        5. is_motion_complete = self._is_downstream_motion_complete()
        6. 回傳組裝好的下一個 tick Observation
        """
        ...

    def _reset_balls(self) -> None:
        """共用：呼叫 table_ball_set.reset(positions)，Teleport 語意，呼叫完當下即完成"""
        ...

    @abstractmethod
    def _reset_downstream(self) -> bool:
        """回傳下游（手臂等）reset 是否完成。Demo：等待手臂到位；Training：永遠 True"""
        ...

    @abstractmethod
    def _execute_aim(self, action: Action) -> None:
        """AIMING 狀態下游動作，本次僅定義介面，內容留給 #96"""
        ...

    @abstractmethod
    def _execute_strike(self, action: Action) -> None:
        """STRIKING 狀態下游動作，本次僅定義介面，內容留給 #97（Demo）／#177（Training，本次尚未落地）"""
        ...

    @abstractmethod
    def _is_downstream_motion_complete(self) -> bool:
        """Demo：球 reset 完成 且 手臂 is_reset_complete()；Training：僅需球 reset 完成"""
        ...
```

**依賴：**
- 輸入來源：呼叫端每 tick 傳入的 `Observation`（來源為未來 Extension tick / physics callback，本次不實作）
- 輸出去向：回傳的 `Observation` 供下一個 tick 使用；內部透過 `ScriptController.get_action()` 取得決策，透過 `TableBallSet`/`BallMotionMonitor`/`BallPositionProvider` 讀寫模型層狀態

---

### DemoTableOrchestrator

**職責：** Demo 桌差異實作，下游動作透過真實手臂（`UR5Robot` + `ArticulationAPI`）執行，屬於動畫式、多 tick 才完成的動作。

**介面：**
```python
class DemoTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_motion_monitor: BallMotionMonitor,
        ball_position_provider: BallPositionProvider,
        ur5_robot: UR5Robot,
        articulation_api: ArticulationAPI,
    ) -> None:
        """新增 ur5_robot、articulation_api 兩個 Demo 桌專屬依賴"""
        ...

    def _reset_downstream(self) -> bool:
        """呼叫 ur5_robot.reset()（僅在狀態剛轉換的 tick 呼叫一次，由 should_execute_action 控制）
        回傳 ur5_robot.is_reset_complete()（每 tick 檢查，內部誤差 < 0.001m 判定到位）"""
        ...

    def _execute_aim(self, action: Action) -> None:
        """預期呼叫 articulation_api.move_to_pose(position, orientation)，本次僅定義介面
        對應未來 Issue #96，本次不實作內容"""
        ...

    def _execute_strike(self, action: Action) -> None:
        """預期呼叫 articulation_api.execute_strike(direction, distance, speed)，本次僅定義介面
        對應未來 Issue #97，本次不實作內容"""
        ...

    def _is_downstream_motion_complete(self) -> bool:
        """球（同 tick 完成）且手臂 ur5_robot.is_reset_complete() 皆為 True"""
        ...
```

**依賴：**
- 輸入來源：`TableOrchestrator` 共用骨架、`UR5Robot`/`ArticulationAPI`（皆已存在，Issue #99 完成的模型層能力）
- 輸出去向：`ArticulationAPI` 實作層（`isaac_sim_impl_6_0/`，透過 RmpFlow 執行實際路徑規劃）

---

### TrainingTableOrchestrator

**職責：** 訓練桌差異實作，沒有手臂，RESET/AIMING 皆為 no-op 或恆真，STRIKING 呼叫既有 `ImpulseStrikingService.strike()` 直接對母球賦予衝量速度，屬於單一 tick 內完成的瞬時物理事件。

**介面：**
```python
class TrainingTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_motion_monitor: BallMotionMonitor,
        ball_position_provider: BallPositionProvider,
        impulse_striking_service: ImpulseStrikingService,
        table_z: float,
    ) -> None:
        """新增 impulse_striking_service、table_z 兩個 Training 桌專屬依賴"""
        ...

    def _reset_downstream(self) -> bool:
        """沒有手臂，永遠回傳 True"""
        ...

    def _execute_aim(self, action: Action) -> None:
        """no-op：訓練桌沒有瞄準動作"""
        ...

    def _execute_strike(self, action: Action) -> None:
        """呼叫 impulse_striking_service.strike(action, table_z)，單一 tick 內完成"""
        ...

    def _is_downstream_motion_complete(self) -> bool:
        """只要球 reset 完成（同 tick）即為 True，不需等待手臂"""
        ...
```

**依賴：**
- 輸入來源：`TableOrchestrator` 共用骨架、`ImpulseStrikingService`（已存在，對應 Issue #177）
- 輸出去向：`RigidBodyAPI.set_velocities()`（由 `ImpulseStrikingService` 內部呼叫）

---

### TableBallSet.get_ball_prim_paths()（既有類別新增方法）

**職責：** 回傳 10 顆球（Ball_0 ~ Ball_9）的完整 prim path 清單，供 `TableOrchestrator` 建構子取得後傳給 `BallMotionMonitor`。

**介面：**
```python
class TableBallSet:
    def get_ball_prim_paths(self) -> list[str]:
        """
        回傳 10 顆球（ball_id 0-9）的 prim path 清單，依 ball_id 升冪排序。
        沿用既有 self._get_ball_prim_path(ball_id) 的路徑組成規則（self._base_path + f"/Balls/Ball_{ball_id}"）。
        """
        ...
```

**依賴：**
- 輸入來源：`TableBallSet` 內部 `self._base_path`（建構時已注入）
- 輸出去向：`TableOrchestrator` 建構子 → `BallMotionMonitor(rigid_body_api, ball_prim_paths)`

---

## 4. 資料流

**RESET 狀態（本次唯一要接通的實際流程）：**
```
呼叫端（未來 Extension tick，本次不實作）
  → DemoTableOrchestrator.step(observation)
    → action = script_controller.get_action(observation)   # ScriptController 純決策，不接觸執行層
    → 若 action.should_execute_action == True 且目前狀態為 RESET：
        → self._reset_balls()
          → table_ball_set.reset(positions)
            → StageAPI.set_prim_translate(...) ×10（Teleport，同 tick 完成）
            → RigidBodyAPI.set_velocities(..., [0,0,0], [0,0,0]) ×10
        → self._reset_downstream()
          → ur5_robot.reset() → articulation_api.move_to_home()（RmpFlow 路徑規劃，動畫式，多 tick）
          → 回傳 ur5_robot.is_reset_complete() → articulation_api.is_motion_complete()（誤差 < 0.001m）
    → is_ball_moving = ball_motion_monitor.is_any_ball_moving()
    → is_motion_complete = self._is_downstream_motion_complete()
        = True（球，同 tick）且 ur5_robot.is_reset_complete()（手臂，可能仍為 False，需等待後續 tick）
  → 組裝下一個 tick 的 Observation（is_ball_moving、is_motion_complete 皆已更新）
  → 回傳給呼叫端，供下一個 tick 的 script_controller.get_action() 判斷是否轉換至 IDLE
```

**STRIKING 狀態（Training 桌，本次僅定義介面，內容待 #177 落地）：**
```
呼叫端
  → TrainingTableOrchestrator.step(observation)
    → action = script_controller.get_action(observation)
    → 若 action.should_execute_action == True 且目前狀態為 STRIKING：
        → self._execute_strike(action)
          → impulse_striking_service.strike(action, table_z)
            → StageAPI.set_prim_translate(cue_ball_prim, x, y, table_z)
            → compute_cue_ball_velocities(action, ball_radius, spin_efficiency)
            → RigidBodyAPI.set_velocities(cue_ball_prim, linear_velocity, angular_velocity)
    → is_ball_moving = ball_motion_monitor.is_any_ball_moving()   # 衝量賦速後下個 tick 應為 True
    → is_motion_complete = self._is_downstream_motion_complete()  # Training：球 reset 完成即 True，與擊球動作本身無關
  → 組裝下一個 tick 的 Observation
  → 回傳給呼叫端
```

---

## 5. 依賴關係圖

```
DemoTableOrchestrator
  ├── 依賴 ScriptController（取得純決策 Action，共用）
  ├── 依賴 TableBallSet（RESET 球重置，共用；get_ball_prim_paths() 供建構期使用）
  ├── 依賴 BallMotionMonitor（查詢是否還有球在動，共用）
  ├── 依賴 BallPositionProvider（RESET 重置座標的來源，共用）
  ├── 依賴 UR5Robot（RESET 手臂歸位，Demo 專屬）
  └── 依賴 ArticulationAPI（AIMING/STRIKING 手臂路徑規劃，Demo 專屬，經 UR5Robot 或直接注入）

TrainingTableOrchestrator
  ├── 依賴 ScriptController（共用，與 Demo 同一個類別）
  ├── 依賴 TableBallSet（共用）
  ├── 依賴 BallMotionMonitor（共用）
  ├── 依賴 BallPositionProvider（共用）
  └── 依賴 ImpulseStrikingService（STRIKING 衝量式擊球，Training 專屬）
        └── 依賴 RigidBodyAPI.set_velocities（外部物理引擎 API）

TableBallSet
  └── 依賴 StageAPI / RigidBodyAPI / MaterialAPI（既有，Port 抽象層）

ScriptController
  └── 不依賴任何執行層 API（純讀 Observation、純回傳 Action，見 5-2 文件第 5.2 節）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| `action.should_execute_action == False`（同一狀態持續中的非觸發 tick） | 跳過 `_reset_downstream()`/`_execute_aim()`/`_execute_strike()`，但仍繼續執行 `is_ball_moving` 查詢與組裝下一個 `Observation`，避免中斷共用流程 |
| Demo 桌 RESET：球已 reset 完成但手臂尚未到位（`is_reset_complete() == False`） | `_is_downstream_motion_complete()` 回傳 `False`，`Observation.is_motion_complete` 維持 `False`，`ScriptController` 停留在 `RESET` 狀態，下個 tick 繼續檢查，直到手臂到位誤差 < 0.001m |
| Training 桌 RESET | 沒有手臂，`_reset_downstream()` 永遠 `True`，`is_motion_complete` 只反映球的狀態，理論上同一 tick 即可轉換至 `IDLE` |
| `observation.has_error == True` | 由 `ScriptController` 內部優先判斷轉入 `ERROR`（見 5-2 文件第 3 節），`TableOrchestrator.step()` 本身不做額外錯誤攔截，僅依 `ScriptController` 回傳的 `Action`/狀態走既定分派邏輯；`Action` 內容在 `ERROR` 狀態下 `should_execute_action` 恆為 `False`（見 `ScriptController._error_state_action_result` 呼叫 `_generate_action_result()` 未覆寫此欄位） |
| `BallMotionMonitor.is_any_ball_moving()` 內部呼叫 `get_linear_velocity()` 拋出例外 | 沿用既有實作：`BallMotionMonitor` 內部已 `try/except` 並 log 後 re-raise，`TableOrchestrator.step()` 不吞例外，交由上層呼叫端（未來 Extension tick）決定是否轉為 `has_error` |
| 呼叫 `_reset_downstream()`/`_execute_aim()`/`_execute_strike()` 時下游 API（`ArticulationAPI`）拋出例外 | 本次設計不在 `TableOrchestrator` 內攔截，例外直接往上拋；是否要在此層新增 try/except 並回寫 `has_error`，留待實作 `_execute_aim`/`_execute_strike` 真正內容時（對應 #96/#97/#177）一併決定 |

---

## 7. 測試涵蓋（對應 Unit Test）

> 本次僅記錄設計對應的測試涵蓋範圍，實作與撰寫留待後續任務。

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| test_step_dispatches_reset_when_should_execute_action_true | core/tests/test_table_orchestrator.py | `should_execute_action=True` 且狀態為 RESET 時，正確呼叫 `table_ball_set.reset()` 與 `_reset_downstream()` |
| test_step_skips_downstream_when_should_execute_action_false | core/tests/test_table_orchestrator.py | `should_execute_action=False` 時不觸發任何下游動作方法，但仍完成 `is_ball_moving` 查詢與 `Observation` 組裝 |
| test_step_queries_ball_motion_every_tick | core/tests/test_table_orchestrator.py | 驗證 `ball_motion_monitor.is_any_ball_moving()` 每個 `step()` 呼叫皆會執行一次，且時機在下游動作分派之後 |
| test_demo_reset_downstream_calls_ur5_reset | core/tests/test_table_orchestrator.py | `DemoTableOrchestrator._reset_downstream()` 呼叫 `ur5_robot.reset()` 並回傳 `ur5_robot.is_reset_complete()` |
| test_demo_motion_complete_requires_ball_and_arm | core/tests/test_table_orchestrator.py | `DemoTableOrchestrator._is_downstream_motion_complete()` 僅在球與手臂皆完成時回傳 `True`（分別測手臂未到位、球未完成兩種 False 情境） |
| test_training_reset_downstream_always_true | core/tests/test_table_orchestrator.py | `TrainingTableOrchestrator._reset_downstream()` 恆回傳 `True` |
| test_training_execute_strike_calls_impulse_service | core/tests/test_table_orchestrator.py | `TrainingTableOrchestrator._execute_strike()` 正確呼叫 `impulse_striking_service.strike(action, table_z)` |
| test_training_motion_complete_ignores_arm | core/tests/test_table_orchestrator.py | `TrainingTableOrchestrator._is_downstream_motion_complete()` 僅需球 reset 完成即為 `True` |
| test_get_ball_prim_paths_returns_ten_paths_in_order | core/tests/test_models.py | `TableBallSet.get_ball_prim_paths()` 回傳 10 筆、依 ball_id 0-9 升冪排序、路徑格式與 `_get_ball_prim_path()` 一致 |

---

## 8. 待決定事項

- [x] **狀態分派的資料來源（已定案 2026-07-21）**：採方案 (b)——`ScriptController` 新增公開方法 `get_current_state() -> BilliardStatus`，回傳 `self._current_state`，供 `TableOrchestrator.step()` 查詢後分派下游動作。理由：狀態單一事實來源維持在 `ScriptController` 內部，不需要在 `Observation`/`Action` 額外複製一份狀態、不用擔心同步問題，改動範圍最小。此方法尚未實作，屬於 `TableOrchestrator` 實作前置工作的一部分。
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
