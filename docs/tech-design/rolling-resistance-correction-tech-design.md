# 滾動摩擦修正（Rolling Resistance Correction）— 技術設計文件

> 生成時間：2026-07-25
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：尚未建立（下一步由 progress-planner 建立 Issue）

---

## 1. 功能概述

用明確的物理修正取代 PhysX 的 `torsionalPatchRadius`／`minTorsionalPatchRadius` 機制，在數位孿生的模擬層自行實作撞球「滾動摩擦」行為。輸入為桌上目前所有球的 prim path 清單（由 `TableBallSet.get_ball_prim_paths()` 取得），每個物理 tick 開始時對每顆球讀取目前的線速度與角速度，依真實滾動摩擦係數（μ_r ≈ 0.01）計算應扣減的水平速度量，並同步衰減「由線速度反推出的滾動角速度分量」，寫回 `RigidBodyAPI`。使用場景：`TableOrchestrator.step()` 每個 tick 無條件呼叫，讓 Demo 桌與訓練桌上的球在滾動過程中會自然減速直到停止，行為貼近真實撞球物理，不再依賴 PhysX 對穿透深度敏感、實測完全無效的扭轉阻力機制。

**背景與動機：** 實測發現即使把 `torsionalPatchRadius` 調到球半徑（0.028575m）的 35 倍（1.0），依然完全沒有可觀察的滾動阻力行為。已排除 solver（TGS）設定錯誤、跨層 RigidBody/Collision 結構問題、Newton 引擎誤判等可能性，確認根因是 TGS solver 為求接觸穩定會把穿透深度壓到趨近零，而 PhysX 扭轉力矩公式「正比於穿透深度」，導致力矩恆趨近零——這是通用物理引擎在此場景下的先天限制。依數位孿生工程原則：當底層引擎的隱式模型無法可靠重現已知真實物理行為時，應在模擬層明確實作已知物理定律取代之。真實撞球滾動摩擦係數公認值 μ_r ≈ 0.005–0.015（來源：Dr. Dave / Witters & Duymelinck, *American Journal of Physics* 1986），減速度 a = μ_r × g。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `RollingResistanceService`（新增） | core/services | 每個 tick 對桌上所有球套用滾動摩擦速度衰減 | `core/services/rolling_resistance_service.py` |
| `TableOrchestrator`（修改） | core/services | `step()` 最前面無條件呼叫 `rolling_resistance_service.apply(...)`；建構子新增依賴注入 | `core/services/table_orchestrator.py` |
| `DemoTableOrchestrator`（修改） | core/services | 建構子新增 `rolling_resistance_service` 參數，透過 `super().__init__()` 往上傳 | `core/services/table_orchestrator.py`（同檔案） |
| `TrainingTableOrchestrator`（修改） | core/services | 建構子新增 `rolling_resistance_service` 參數，透過 `super().__init__()` 往上傳 | `core/services/table_orchestrator.py`（同檔案） |
| `billiard_digital_twin.py`（修改） | extension | 在 `_asset_env_init()` 建立一個共用的 `RollingResistanceService` 實例（Demo/Training 兩桌共用），組裝 orchestrator 時傳入 | `extension/billiard_digital_twin/billiard_digital_twin.py` |

不新增任何 port（完全用既有的 `RigidBodyAPI.get_linear_velocity` / `get_angular_velocity` / `set_velocities`），不需要 extension 層的新實作。

---

## 3. 類別設計

### RollingResistanceService

**職責：** 對指定的球 prim path 清單，依真實滾動摩擦物理逐顆計算並寫回衰減後的線速度／角速度。

**介面：**
```python
GRAVITY = 9.81
NEGLIGIBLE_SPEED_THRESHOLD = 0.02  # m/s，跟 scripts/measure_rolling_friction.py 的 STOP_SPEED_THRESHOLD 同數量級
PHYSICS_DT = 1.0 / 60.0            # 跟 SimulationManager.setup_simulation(dt=1/60) 一致的固定常數，不作為 apply() 參數傳入


class RollingResistanceService:
    def __init__(
        self,
        rigid_body_api: RigidBodyAPI,
        ball_radius: float,
        rolling_friction_coeff: float = 0.01,
    ) -> None:
        """依賴注入 RigidBodyAPI；ball_radius／rolling_friction_coeff 為固定係數。"""
        ...

    def apply(self, ball_prim_paths: list[str]) -> None:
        """
        對每個 prim path 獨立執行一次滾動摩擦速度衰減（見第 4 節資料流的演算法步驟）。
        已停止（水平速度 < NEGLIGIBLE_SPEED_THRESHOLD）的球會被跳過，不寫入速度。
        """
        ...
```

**依賴：**
- 輸入來源：`TableOrchestrator.step()` 呼叫時傳入的 `TableBallSet.get_ball_prim_paths()`
- 輸出去向：透過 `RigidBodyAPI.set_velocities()` 寫回引擎層，供下一個 tick 的物理模擬與 `BallMotionMonitor` 讀取

---

### TableOrchestrator（含 DemoTableOrchestrator / TrainingTableOrchestrator）

**職責：** `step()` 骨架新增無條件執行的滾動摩擦修正步驟，放在既有的 action/state 分派邏輯之前。

**介面（僅列出變更部分）：**
```python
class TableOrchestrator(ABC):
    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService,  # 新增
    ) -> None:
        """新增 rolling_resistance_service 依賴注入。"""
        ...

    def step(self, observation: Observation) -> None:
        """
        每個 tick 呼叫一次：
        1. 無條件呼叫 self._rolling_resistance_service.apply(...)（新增，在最前面）
        2. 既有的 action/current_state 分派邏輯（不變）
        """
        ...


class DemoTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        ur5_robot: UR5Robot,
        articulation_api: ArticulationAPI,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService,  # 新增
    ) -> None:
        """透過 super().__init__() 往上傳 rolling_resistance_service。"""
        ...


class TrainingTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        impulse_striking_service: ImpulseStrikingService,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService,  # 新增
    ) -> None:
        """透過 super().__init__() 往上傳 rolling_resistance_service。"""
        ...
```

**依賴：**
- 輸入來源：`RollingResistanceService`（建構子注入）、`TableBallSet.get_ball_prim_paths()`
- 輸出去向：無直接輸出，效果反映在下一個 tick 的球體物理狀態

---

### billiard_digital_twin.py（extension 組裝層，修改）

**職責：** 在 `_asset_env_init()` 建立一個共用的 `RollingResistanceService` 實例，Demo 桌與訓練桌組裝各自的 orchestrator 時共同傳入同一個實例。

**依賴：**
- 輸入來源：既有的 `self._rigid_body_api`（全桌共用）、球半徑常數
- 輸出去向：`DemoTableOrchestrator` 與 `TrainingTableOrchestrator` 建構子

---

## 4. 資料流

```
TableRuntime.tick()                                              [不變]
  → TableOrchestrator.step(observation)
      → self._rolling_resistance_service.apply(                  ← 新增，放在最前面，無條件執行
            self._table_ball_set.get_ball_prim_paths()
        )
          對每顆球獨立處理：
          1. v = rigid_body_api.get_linear_velocity(prim_path)
             v_h = sqrt(vx² + vy²)                                # 只看水平分量，vz 不動
             若 v_h < NEGLIGIBLE_SPEED_THRESHOLD → 跳過這顆球（不寫入）

          2. Δv = rolling_friction_coeff * GRAVITY * PHYSICS_DT   # 固定常數，不外部傳入

          3. 若 Δv >= v_h：
                 v_after_h = (0, 0)                                # 本 tick 內完全停止，不反向
                 scale = 0
             否則：
                 scale = (v_h - Δv) / v_h
                 v_after_h = (vx*scale, vy*scale)

          4. 角速度採「精確分解版」：
             ω_actual = rigid_body_api.get_angular_velocity(prim_path)
             n̂ = (0, 0, 1)
             ω_roll_before = (n̂ × v_before) / ball_radius          # 由線速度反推出的滾動分量
             ω_residual = ω_actual - ω_roll_before                 # 加塞／未來 side-spin，不衰減
             ω_roll_after = ω_roll_before * scale                  # 隨線速度等比例衰減
             ω_after = ω_roll_after + ω_residual

          5. rigid_body_api.set_velocities(
                 prim_path,
                 (v_after_h[0], v_after_h[1], vz),
                 ω_after
             )
      → action = self._script_controller.get_action(observation)          [不變]
      → current_state = self._script_controller.get_current_state()       [不變]
      → if action.should_execute_action:
            match current_state:
                case RESET: ...
                case AIMING: ...
                case STRIKING: ...                                          [不變]
```

---

## 5. 依賴關係圖

```
RollingResistanceService
  └── 依賴 RigidBodyAPI（既有 port，get_linear_velocity / get_angular_velocity / set_velocities）

TableOrchestrator（base class，DemoTableOrchestrator / TrainingTableOrchestrator 皆繼承）
  └── 新增依賴 RollingResistanceService（建構子注入）

billiard_digital_twin.py（extension 組裝層）
  └── 在 _asset_env_init() 建立一個 RollingResistanceService 共用實例
      （Demo 桌與訓練桌共用同一個實例，因為此 service 不持有跨桌狀態，
      只包一個 rigid_body_api 參考跟兩個係數常數，跟現有 self._rigid_body_api
      也是全桌共用的模式一致）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 球速已經低於 `NEGLIGIBLE_SPEED_THRESHOLD` | 跳過這顆球，不做任何速度寫入，避免除以趨近 0 或抖動 |
| 這個 tick 該扣減的量 `Δv` 超過目前球速 `v_h` | 直接把線速度水平分量歸零、`scale=0`，不會反向 |
| 球有加塞（side-spin，繞垂直軸自旋） | 這次不影響：`ω_residual` 完全不衰減，只有由線速度反推出的滾動分量會衰減 |
| Demo 桌與訓練桌 | 共用同一個 `RollingResistanceService` 實例，`step()` 的呼叫邏輯在 base class `TableOrchestrator` 裡，兩個子類別都會自動套用 |

**關鍵設計決策（需保留紀錄）：** 角速度衰減採用「精確分解版」而非「整體乘 scale 的簡單版」——把角速度分解成「由線速度決定的滾動分量」與「殘留分量（含未來加塞／english 自旋）」，只衰減滾動分量，殘留分量完全不動。此為使用者在設計討論階段主動選擇的方向，明確理由是為了讓未來實作加塞球路時不需要回頭修改這裡的邏輯。

---

## 7. 測試涵蓋（對應 Unit Test，不在本次實作範圍）

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| test_apply_decays_linear_velocity_by_rolling_friction | core/tests/test_rolling_resistance_service.py | 給定已知水平線速度，驗證扣減量等於 `rolling_friction_coeff * GRAVITY * PHYSICS_DT` |
| test_apply_clamps_to_zero_when_delta_exceeds_speed | core/tests/test_rolling_resistance_service.py | 速度很小、`Δv >= v_h` 時，驗證線速度水平分量歸零，不會變成負值/反向 |
| test_apply_skips_ball_below_negligible_threshold | core/tests/test_rolling_resistance_service.py | 球速已經低於門檻時，`set_velocities` 不應被呼叫 |
| test_apply_preserves_residual_angular_velocity | core/tests/test_rolling_resistance_service.py | 給定一個有「額外自旋分量」（模擬加塞）的球，驗證這個殘留分量衰減前後完全不變，只有滾動分量按比例衰減 |
| test_apply_scales_rolling_angular_component_with_linear_velocity | core/tests/test_rolling_resistance_service.py | 驗證滾動分量的衰減比例跟線速度的衰減比例一致（同一個 `scale`） |
| test_step_calls_rolling_resistance_before_state_dispatch | core/tests/test_table_orchestrator.py | 驗證 `TableOrchestrator.step()` 不論 `should_execute_action` / `current_state` 為何，都會呼叫 `rolling_resistance_service.apply()`，且在 state dispatch 之前 |

---

## 8. 待決定事項

- [ ] 無（設計討論已在四階段流程中確認完畢，本文件為定案內容）

---

## 參考資料

- 相關驗證工具（已存在）：`scripts/measure_rolling_friction.py`
- 現有相關檔案：
  - `core/services/table_orchestrator.py`
  - `core/models/table_ball_set.py`
  - `core/ports/rigid_body_api.py`
  - `core/services/impulse_striking_service.py`（參考既有建構參數風格）
  - `extension/billiard_digital_twin/billiard_digital_twin.py`
- 滾動摩擦係數來源：Witters, J. & Duymelinck, D., "Rolling and sliding resistive forces on balls moving on a flat surface," *American Journal of Physics*, 1986.
