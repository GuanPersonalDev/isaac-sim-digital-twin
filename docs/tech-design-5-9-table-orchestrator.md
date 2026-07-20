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
        2. 依當前狀態（見第 8 節「待決定事項」：狀態來源尚待定案）分派：
           - RESET    → 若 action.should_execute_action：self._reset_balls() + self._reset_downstream()
           - AIMING   → 若 action.should_execute_action：self._execute_aim(action)
           - STRIKING → 若 action.should_execute_action：self._execute_strike(action)
           - WAITING / IDLE → 無下游動作
        3. is_ball_moving = self._ball_motion_monitor.is_any_ball_moving()
        4. is_motion_complete = self._is_downstream_motion_complete()
        5. 回傳組裝好的下一個 tick Observation
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

- [ ] **狀態分派的資料來源尚未定案**：本次設計敘述「依 observation 目前狀態（對照 BilliardStatus）分派」，但目前 `Observation`（`core/models/observation.py`）並未包含 `status: BilliardStatus` 欄位；`BilliardStatus` 目前是 `ScriptController` 內部私有的 `_current_state`，沒有對外公開的 getter。`TableOrchestrator.step()` 若要依狀態分派，必須先解決以下其中一種方案，本次設計不代為決定：
  - (a) `Observation` 新增 `status: BilliardStatus` 欄位，由 `ScriptController` 或上游回寫；
  - (b) `ScriptController` 新增公開方法（如 `get_current_state() -> BilliardStatus`）供 `TableOrchestrator` 查詢；
  - (c) 改由 `Action` 本身攜帶足夠資訊讓 `TableOrchestrator` 判斷該做什麼（例如新增一個「動作種類」欄位），不必反查狀態。
- [ ] `TableBallSet.reset(positions)` 需要傳入 `positions: dict[int, tuple[float, float]]`；`TableOrchestrator` 呼叫 `_reset_balls()` 時這組座標從何取得（`BallPositionProvider` 是否已有對應方法回傳「初始擺球座標」）尚待確認介面是否吻合，本次僅假設 `BallPositionProvider` 能提供，未逐一核對既有方法簽章。
- [ ] `DemoTableOrchestrator` 建構子是否需要同時注入 `UR5Robot` 與 `ArticulationAPI`，或只需注入 `UR5Robot`（`ArticulationAPI` 透過 `UR5Robot` 內部間接使用）——AIMING/STRIKING 的 `move_to_pose`/`execute_strike` 目前只存在於 `ArticulationAPI`，`UR5Robot` 本身未包裝這两个方法，待 #96/#97 实作時一併確認是否要在 `UR5Robot` 新增對應包裝方法。
- [ ] Extension tick／physics callback 尚未存在，`TableOrchestrator.step()` 目前沒有任何生產程式碼呼叫入口，串接時機留待後續任務。
