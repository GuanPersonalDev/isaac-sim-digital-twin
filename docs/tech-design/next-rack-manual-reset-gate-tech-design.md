# Next Rack 手動重置閘門 — 技術設計文件

> 生成時間：2026-09-10
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：尚未建立（下一步驟由 progress-planner 建立 Issue）
> 交叉引用：`HUD ShotResult 顯示`（#116）依賴本功能——它假設球停止移動後會停在
> READY_TO_RESET 狀態直到使用者手動確認，才有時間顯示擊球結果。#116 目前尚未進入
> feature-design-flow，對應的技術文件路徑尚不存在，待該功能建立技術文件後應回頭
> 在該文件中反向引用本文件。

---

## 1. 功能概述

目前球停止移動後，狀態機會在同一個 tick 自動把全部球瞬移回固定的開球擺位
（reset），使用者完全沒有機會看到球停下來那一刻的畫面。本功能新增一個中繼狀態
`READY_TO_RESET`：球停止移動後，狀態機停在這個狀態、不自動重擺；HUD 面板新增一顆
「Next Rack」按鈕，只有狀態是 `READY_TO_RESET` 時才能按，按下後才真正觸發重擺
（轉進 `RESET`），下一次球又停止移動、重新進入 `READY_TO_RESET` 之前，按鈕保持
disabled。適用範圍涵蓋 Manual 模式與 AI 模式，因為轉換邏輯定義在兩者共用的基底
類別 `BilliardStateMachineController`。輸入是「球是否還在移動」（`Observation.is_ball_moving`）
與「使用者是否點擊 Next Rack」（UI 執行緒呼叫），輸出是狀態機的狀態轉換與 HUD
按鈕的 enabled/disabled。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `BilliardStatus`（新增 enum 成員） | core.models | 新增 `READY_TO_RESET` 成員，放在 `WAITING` 與 `RESET` 之間 | `core/models/billiard_state.py` |
| `ControllerBase`（新增抽象方法） | core.controllers | 新增 `request_reset_confirm()` 抽象方法，跟 `get_action`/`get_current_state`/`reset` 並列 | `core/controllers/controller_base.py` |
| `BilliardStateMachineController`（擴充） | core.controllers | 新增雙序號計數器（`_reset_requested_seq`/`_reset_handled_seq`）與 `request_reset_confirm()`/`_is_reset_confirm_pending()`；`_waiting_state_action_result()` 改為轉出 `READY_TO_RESET`；新增 `_ready_to_reset_state_action_result()`；`get_action()` 的 match 新增一個 case；`reset()` 同步清掉排隊中的確認請求 | `core/controllers/billiard_state_machine_controller.py` |
| `TableOrchestrator`（擴充） | core.services | 新增 `confirm_reset()` 純轉呼叫至 `self._script_controller.request_reset_confirm()` | `core/services/table_orchestrator.py` |
| `TableRuntime`（擴充） | core.services | 新增 `confirm_reset()` 純轉呼叫至 `self._orchestrator.confirm_reset()` | `core/services/table_runtime.py` |
| `TableSession`（擴充） | core.services | 新增 `confirm_reset()` 純轉呼叫至 `self._runtime.confirm_reset()` | `core/services/table_session.py` |
| `BilliardDigitalTwin`（擴充） | extension.billiard_digital_twin | 新增 `confirm_reset(table_id)` 與 `is_ready_to_reset(table_id)`，各自轉呼叫對應 session；`HudPanel(...)` 建構呼叫多傳這兩個參數 | `extension/billiard_digital_twin/billiard_digital_twin.py` |
| `HudPanel`（擴充） | extension.ui | 新增「Next Rack」按鈕（獨立於既有「Reset Table」按鈕）；`_on_update()` 輪詢按鈕 enabled 狀態 | `extension/ui/hud_panel.py` |

---

## 3. 類別設計

### BilliardStatus（enum 擴充）

**職責：** 新增一個狀態值，代表「球已停止移動、等待使用者確認重擺」。

**介面：**
```python
class BilliardStatus(Enum):
    ...
    WAITING = "waiting"
    READY_TO_RESET = "ready_to_reset"  # 新增，排版放在 WAITING 與 RESET 之間，enum 本身無順序語意
    RESET = "reset"
    ...
```

**依賴：**
- 輸入來源：無（enum 定義本身）
- 輸出去向：`BilliardStateMachineController`（狀態轉換）、`TableOrchestrator.step()`（match 語句，本次不新增 case，`READY_TO_RESET` 是純 no-op）、`BilliardDigitalTwin.is_ready_to_reset()`（狀態比對）

---

### ControllerBase（抽象介面擴充）

**職責：** 強制所有 controller 子類別提供「確認重擺」的入口，跟既有三個抽象方法並列。

**介面：**
```python
class ControllerBase(ABC):
    ...
    @abstractmethod
    def request_reset_confirm(self) -> None:
        """UI 執行緒呼叫：確認可以觸發真正的重擺（瞬移回開球擺位）。"""
        ...
```

**依賴：**
- 輸入來源：無（抽象介面定義）
- 輸出去向：`BilliardStateMachineController` 實作此方法，`ManualController`/`ModelController` 透過繼承自動滿足

---

### BilliardStateMachineController（既有類別擴充，三個共用狀態轉換不開放子類別覆寫）

**職責：** 在 `WAITING` → `READY_TO_RESET` → `RESET` 的轉換中插入一個「等待使用者手動確認」的關卡；沿用 `ManualController.request_shot()`/`is_shot_pending()` 已驗證過的無鎖雙序號計數器手法，UI 執行緒與 physics 執行緒（`get_action()`，每 tick 呼叫）各自只寫自己的欄位，靠 int 賦值的原子性避免 lost update，不需要 `threading.Lock`。

**介面：**
```python
class BilliardStateMachineController(ControllerBase):
    def __init__(self, ...):
        ...
        self._reset_requested_seq = 0  # 只有 UI 執行緒寫
        self._reset_handled_seq = 0    # 只有 physics 執行緒寫

    def request_reset_confirm(self) -> None:
        """UI 執行緒呼叫：遞增請求序號。"""
        self._reset_requested_seq += 1

    def _is_reset_confirm_pending(self) -> bool:
        """比較兩個序號是否相等，判斷是否有尚未消費的確認請求。"""
        return self._reset_requested_seq != self._reset_handled_seq

    def _waiting_state_action_result(self, observation: Observation) -> Action:
        """球停止移動後只轉進 READY_TO_RESET，不再自動觸發重擺。"""
        result = self._generate_action_result()
        if not observation.is_ball_moving:
            self._change_state(BilliardStatus.READY_TO_RESET)
        return result

    def _ready_to_reset_state_action_result(self, observation: Observation) -> Action:
        """有待消費的確認請求時才真正轉進 RESET 並觸發重擺。"""
        result = self._generate_action_result()
        if self._is_reset_confirm_pending():
            self._reset_handled_seq = self._reset_requested_seq
            result.should_execute_action = True
            self._change_state(BilliardStatus.RESET)
        return result

    def get_action(self, observation: Observation) -> Action:
        ...
        # match 語句新增：
        # case BilliardStatus.READY_TO_RESET:
        #     return self._ready_to_reset_state_action_result(observation)

    def reset(self) -> None:
        """既有方法，新增一行同步丟棄排隊中的確認請求，避免下一次進
        READY_TO_RESET 時被誤判成已確認而立刻自動重擺。"""
        self._change_state(BilliardStatus.RESET)
        self._reset_handled_seq = self._reset_requested_seq
        self._on_reset()
```

**依賴：**
- 輸入來源：`Observation.is_ball_moving`（physics 執行緒）、`TableOrchestrator.confirm_reset()` → `request_reset_confirm()`（UI 執行緒）
- 輸出去向：`TableOrchestrator.step()` 讀取 `get_current_state()` 判斷是否進入 `RESET` 分支（`_reset_balls()` + `_reset_downstream()`）

---

### TableOrchestrator / TableRuntime / TableSession（純轉呼叫擴充）

**職責：** 沿用 `get_current_state()`/`request_full_reset()` 既有的逐層轉呼叫慣例，把 UI 層的確認動作往下傳遞到狀態機。

**介面：**
```python
# TableOrchestrator
def confirm_reset(self) -> None:
    self._script_controller.request_reset_confirm()

# TableRuntime
def confirm_reset(self) -> None:
    self._orchestrator.confirm_reset()

# TableSession
def confirm_reset(self) -> None:
    self._runtime.confirm_reset()
```

**依賴：**
- 輸入來源：`BilliardDigitalTwin.confirm_reset(table_id)`
- 輸出去向：`BilliardStateMachineController.request_reset_confirm()`

---

### BilliardDigitalTwin（既有類別擴充）

**職責：** 依 `table_id` 找到對應 session，轉呼叫確認重擺、查詢是否處於 `READY_TO_RESET`；供 `HudPanel` 以 DI 方式呼叫。

**介面：**
```python
def confirm_reset(self, table_id: str) -> None:
    session = self._find_session(table_id)
    if session is None:
        return
    session.confirm_reset()

def is_ready_to_reset(self, table_id: str) -> bool:
    session = self._find_session(table_id)
    if session is None:
        return False
    return session.get_current_state() == BilliardStatus.READY_TO_RESET
```

**依賴：**
- 輸入來源：`HudPanel._on_next_rack_button_clicked()`（呼叫 `confirm_reset`）、`HudPanel._on_update()`（呼叫 `is_ready_to_reset` 輪詢）
- 輸出去向：`TableSession.confirm_reset()`/`get_current_state()`；需新增 import `from core.models.billiard_state import BilliardStatus`

---

### HudPanel（既有類別擴充）

**職責：** 提供「Next Rack」按鈕，語意是「一般擊球後、確認要重擺才重擺」，與既有「Reset Table」按鈕（語意是「立即強制重開整局」，ERROR 復原用，任何時候都能按、不經過這次的閘門）是兩個獨立的按鈕，不合併、不共用。

**介面：**
```python
def __init__(self, ..., confirm_reset: Callable[[str], None], is_ready_to_reset: Callable[[str], bool]):
    ...
    self._confirm_reset = confirm_reset
    self._is_ready_to_reset = is_ready_to_reset

def _build_body_ui(self) -> None:
    ...
    self._next_rack_button = ui.Button("Next Rack", clicked_fn=self._on_next_rack_button_clicked)

def _on_next_rack_button_clicked(self) -> None:
    if self._table_combo_model is None:
        return
    table_id = self._table_combo_model.get_selected_table_id()
    if table_id is None:
        return
    self._confirm_reset(table_id)

def _on_update(self, ...) -> None:
    ...
    self._next_rack_button.enabled = (
        self._is_ready_to_reset(table_id) if table_id is not None else False
    )
```

**依賴：**
- 輸入來源：使用者點擊（UI 執行緒）、`_on_update()` 每輪輪詢（沿用既有 `self._controller_mode_button.enabled = table_id is not None` 寫法）
- 輸出去向：`BilliardDigitalTwin.confirm_reset()`/`is_ready_to_reset()`（透過建構時注入的 callable）

---

## 4. 資料流

```
physics tick（get_action 被呼叫）
  → BilliardStateMachineController._waiting_state_action_result(observation)
    → 若 observation.is_ball_moving == False：_change_state(READY_TO_RESET)
  → HudPanel._on_update() 輪詢 is_ready_to_reset(table_id)
    → self._next_rack_button.enabled = True

使用者點擊 Next Rack 按鈕（UI 執行緒）
  → HudPanel._on_next_rack_button_clicked()
    → self._confirm_reset(table_id)（DI 注入的 callable）
      → BilliardDigitalTwin.confirm_reset(table_id)
        → TableSession.confirm_reset()
          → TableRuntime.confirm_reset()
            → TableOrchestrator.confirm_reset()
              → BilliardStateMachineController.request_reset_confirm()
                → self._reset_requested_seq += 1（只有 UI 執行緒寫）

下一個 physics tick（get_action 被呼叫，狀態已是 READY_TO_RESET）
  → BilliardStateMachineController._ready_to_reset_state_action_result(observation)
    → _is_reset_confirm_pending() 比對 _reset_requested_seq != _reset_handled_seq → True
    → self._reset_handled_seq = self._reset_requested_seq（只有 physics 執行緒寫）
    → result.should_execute_action = True
    → _change_state(RESET)
  → TableOrchestrator.step() 的 match 進入 case BilliardStatus.RESET
    → self._reset_balls()（table_ball_set.reset(positions)，真正瞬移回開球擺位）
    → self._reset_downstream()（手臂歸位）
  → HudPanel._on_update() 下一輪輪詢 is_ready_to_reset() → False
    → self._next_rack_button.enabled = False
```

---

## 5. 依賴關係圖

```
HudPanel（extension.ui）
  ├── 依賴 confirm_reset: Callable（UI 執行緒呼叫，DI 注入，實際指向 BilliardDigitalTwin.confirm_reset）
  └── 依賴 is_ready_to_reset: Callable（輪詢按鈕是否可按，DI 注入，實際指向 BilliardDigitalTwin.is_ready_to_reset）

BilliardDigitalTwin（extension.billiard_digital_twin）
  ├── 依賴 TableSession.confirm_reset()（純轉呼叫）
  ├── 依賴 TableSession.get_current_state()（既有方法，判斷 READY_TO_RESET）
  └── 依賴 core.models.billiard_state.BilliardStatus（新增 import，用於狀態比對）

TableSession（core.services）
  └── 依賴 TableRuntime.confirm_reset()（純轉呼叫）

TableRuntime（core.services）
  └── 依賴 TableOrchestrator.confirm_reset()（純轉呼叫）

TableOrchestrator（core.services）
  └── 依賴 BilliardStateMachineController.request_reset_confirm()（透過 self._script_controller，實際持有的是 ManualController 或 ModelController 的實例）

BilliardStateMachineController（core.controllers）
  ├── 定義 request_reset_confirm() / _is_reset_confirm_pending()（雙序號計數器，沿用 ManualController.request_shot() 手法）
  ├── 定義 _waiting_state_action_result()（改為轉出 READY_TO_RESET，不再自動轉 RESET）
  ├── 定義 _ready_to_reset_state_action_result()（新增，消費確認請求後才轉 RESET）
  └── 依賴 core.models.billiard_state.BilliardStatus.READY_TO_RESET（新增 enum 成員）

ManualController / ModelController（core.controllers）
  └── 繼承 BilliardStateMachineController，不覆寫本次新增的共用行為

ControllerBase（core.controllers，抽象介面）
  └── 新增抽象方法 request_reset_confirm()，ManualController／ModelController 透過繼承 BilliardStateMachineController 自動滿足
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 使用者在 ERROR 狀態按「Reset Table」（既有 `request_manual_reset()` 路徑，非本功能的 Next Rack 按鈕，走 `full_reset()`） | `reset()` 方法已同步清掉排隊中的確認請求（`_reset_handled_seq = _reset_requested_seq`），下一次進入 READY_TO_RESET 時不會被誤判成已確認而立刻自動重擺 |
| 使用者連續點擊 Next Rack 按鈕多次 | 雙序號計數器只比較是否相等，多次遞增 `_reset_requested_seq` 等效合併成一次確認，不會佇列化、不會觸發多次重擺（沿用 `request_shot()` 同一設計決定） |
| Next Rack 按下當下 `table_id` 尚未選定（`_table_combo_model` 為 None 或 `get_selected_table_id()` 回傳 None） | `_on_next_rack_button_clicked()` 提前 return，不呼叫 `confirm_reset`，沿用既有 UI 防呆慣例 |
| `confirm_reset(table_id)`／`is_ready_to_reset(table_id)` 傳入的 `table_id` 找不到對應 session | `BilliardDigitalTwin._find_session()` 回傳 None；`confirm_reset` 直接 return（no-op）；`is_ready_to_reset` 回傳 False（按鈕維持 disabled），不拋例外 |
| 狀態不在 READY_TO_RESET 時仍呼叫 `request_reset_confirm()`（例如繞過 UI enabled 檢查的程式化呼叫） | `_reset_requested_seq` 照樣遞增，但只有狀態機真正處於 READY_TO_RESET 時 `_ready_to_reset_state_action_result()` 才會被 `get_action()` 呼叫到並消費它；其餘狀態下這次遞增不會遺失，但也不會提前生效，要等到真正進入 READY_TO_RESET 才視為已確認 |
| ScriptController 移除後，殘留程式碼引用它 | 見第 7 節「需要移除的死碼」，`core/controllers/__init__.py`、`core/tests/test_action_bounds.py` 的 import 一併清除，避免 `ImportError` |

---

## 7. 需要移除的死碼（本功能引發的必要延伸，使用者已確認選項 (a)：直接移除）

`ScriptController`（`core/controllers/script_controller.py`）繼承同一個共用基底類別，但目前沒有被
`extension/billiard_digital_twin/billiard_digital_twin.py` 使用（GUI 正式流程只建構 `ManualController`/
`ModelController`）。本次改動後若不移除它，它會卡死在 `READY_TO_RESET` 永遠出不來（沒有任何地方會呼叫
`request_reset_confirm()` 餵它），它原本「自動連續循環」的定位會被破壞。使用者已確認直接移除：

- 刪除 `core/controllers/script_controller.py`
- 刪除 `core/tests/test_script_controller.py`
- `core/controllers/__init__.py` 移除 `from .script_controller import ScriptController` 這行 re-export
- `core/tests/test_action_bounds.py`：移除 `TestSingleSourceOfTruth` class（含其下兩個測試方法
  `test_script_controller_no_longer_defines_its_own_speed_limit`、
  `test_striking_action_uses_the_shared_upper_bound`，對應 #114 的單一來源驗證），同時移除檔案開頭的
  `from core.controllers.script_controller import ScriptController` import（不移除檔案其餘測試）
- `scripts/verify_swing_trajectory.py` 的純文字 docstring 提及、其餘檔案 docstring 裡的比較性文字說明，
  皆非程式碼依賴，沿用「不主動改舊註解」慣例，不需要修改

---

## 8. 測試涵蓋（對應 Unit Test）

> `extension/` 層（`BilliardDigitalTwin.confirm_reset`/`is_ready_to_reset`、`HudPanel`）依
> `skills/unit-test.md` 職責一 Step 1 不寫 Unit Test。

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| test_waiting_state_transitions_to_ready_to_reset_when_ball_stops | core/tests/test_manual_controller.py | 球停止移動後狀態轉進 READY_TO_RESET，而非直接轉 RESET |
| test_ready_to_reset_state_stays_until_confirm_requested | core/tests/test_manual_controller.py | 未呼叫 request_reset_confirm() 前，狀態停留在 READY_TO_RESET，should_execute_action 為 False |
| test_ready_to_reset_state_transitions_to_reset_after_confirm | core/tests/test_manual_controller.py | 呼叫 request_reset_confirm() 後下一次 get_action() 轉進 RESET，should_execute_action 為 True |
| test_multiple_confirm_requests_are_coalesced_into_one_reset | core/tests/test_manual_controller.py | 連續呼叫 request_reset_confirm() 多次，仍只觸發一次重擺（雙序號計數器合併） |
| test_reset_drops_pending_confirm_request | core/tests/test_manual_controller.py | reset()（ERROR 復原路徑）呼叫後，殘留的確認請求被清空，下一輪進入 READY_TO_RESET 不會自動重擺 |
| test_confirm_reset_delegates_to_orchestrator | core/tests/test_table_session.py | TableSession.confirm_reset() 正確轉呼叫 TableRuntime.confirm_reset() |
| test_confirm_reset_delegates_to_orchestrator | core/tests/test_table_runtime.py | TableRuntime.confirm_reset() 正確轉呼叫 TableOrchestrator.confirm_reset() |
| test_confirm_reset_delegates_to_controller | core/tests/test_table_orchestrator.py | TableOrchestrator.confirm_reset() 正確轉呼叫 controller.request_reset_confirm() |

死碼移除同步異動：`core/tests/test_action_bounds.py` 移除 `TestSingleSourceOfTruth` class 後，
需重新執行 `python -m pytest core/tests/` 確認沒有殘留的 `ImportError`。

---

## 9. 待決定事項

- [ ] #116（HUD ShotResult 顯示）尚未走 feature-design-flow，其技術文件建立後應回頭在該文件中
      加入對本文件的交叉引用（目前只能由本文件單向引用）
