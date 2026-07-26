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

**介面（2026-07-25 事後修正版，見第 6 節）：**
```python
GRAVITY = 9.81
NEGLIGIBLE_SPEED_THRESHOLD = 0.02  # m/s，低於此值視覺上等同停止，直接夾到 0（非跳過不處理）
NEGLIGIBLE_SPIN_THRESHOLD = 0.1    # rad/s，殘留自旋低於此值視覺上等同停止，直接夾到 0
PHYSICS_DT = 1.0 / 60.0            # 跟 SimulationManager.setup_simulation(dt=1/60) 一致的固定常數，不作為 apply() 參數傳入
SPIN_DECAY_RATE = 10.0             # rad/s²，Dr. Dave Pool Info 記載的球-呢絨自旋衰減率 5–15 rad/s² 中間值


class RollingResistanceService:
    def __init__(
        self,
        rigid_body_api: RigidBodyAPI,
        ball_radius: float,
        rolling_friction_coeff: float = 0.01,
        spin_decay_rate: float = SPIN_DECAY_RATE,
    ) -> None:
        """依賴注入 RigidBodyAPI；ball_radius／rolling_friction_coeff／spin_decay_rate 為固定係數。"""
        ...

    def apply(self, ball_prim_paths: list[str]) -> None:
        """
        對每個 prim path 獨立執行一次速度衰減（見第 4 節資料流的演算法步驟）：
        線速度＋對應滾動角速度分量依滾動摩擦衰減；殘留角速度分量（含加塞／
        english）依自旋衰減率獨立衰減。兩者都會被夾到精確 0，只有線速度與
        殘留自旋皆已精確為 0 的球才會被跳過（純效能考量）。
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
          對每顆球獨立處理（2026-07-25 事後修正版，見第 6 節）：
          1. v = rigid_body_api.get_linear_velocity(prim_path)
             ω_actual = rigid_body_api.get_angular_velocity(prim_path)
             v_h = sqrt(vx² + vy²)                                # 只看水平分量，vz 不動
             n̂ = (0, 0, 1)
             ω_roll_before = (n̂ × v_before) / ball_radius          # 由線速度反推出的滾動分量
             ω_residual = ω_actual - ω_roll_before                 # 加塞／english 殘留分量
             residual_mag = |ω_residual|

             若 v_h 與 residual_mag 都精確為 0 → 跳過這顆球（純效能考量，不寫入）

          2. 線速度＋滾動角速度分量（滾動摩擦）：
             Δv = rolling_friction_coeff * GRAVITY * PHYSICS_DT
             若 v_h < NEGLIGIBLE_SPEED_THRESHOLD 或 Δv >= v_h：
                 linear_scale = 0                                   # 明確夾到 0，不是跳過不處理
             否則：
                 linear_scale = (v_h - Δv) / v_h
             v_after_h = (vx*linear_scale, vy*linear_scale)
             ω_roll_after = ω_roll_before * linear_scale

          3. 殘留角速度分量（側旋／english 自旋衰減，跟滾動摩擦是獨立的物理現象）：
             Δw = spin_decay_rate * PHYSICS_DT
             若 residual_mag < NEGLIGIBLE_SPIN_THRESHOLD 或 Δw >= residual_mag：
                 spin_scale = 0                                     # 明確夾到 0
             否則：
                 spin_scale = (residual_mag - Δw) / residual_mag
             ω_residual_after = ω_residual * spin_scale

          4. ω_after = ω_roll_after + ω_residual_after

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
| 球速已經低於 `NEGLIGIBLE_SPEED_THRESHOLD` | 直接把線速度水平分量夾到精確 0（**不是跳過不處理**，見下方「事後修正」） |
| 這個 tick 該扣減的量 `Δv` 超過目前球速 `v_h` | 直接把線速度水平分量歸零、`linear_scale=0`，不會反向 |
| 殘留自旋（含加塞／english）幅度已經低於 `NEGLIGIBLE_SPIN_THRESHOLD` | 直接把殘留分量夾到精確 0 |
| 這個 tick 該扣減的自旋量 `Δw` 超過目前殘留自旋幅度 | 直接把殘留分量歸零、`spin_scale=0` |
| 線速度與殘留自旋都已經精確為 0 | 完全跳過這顆球，不重複呼叫 `set_velocities`（純效能考量，用極小 epsilon 判斷，跟上面的視覺門檻是不同用途） |
| Demo 桌與訓練桌 | 共用同一個 `RollingResistanceService` 實例，`step()` 的呼叫邏輯在 base class `TableOrchestrator` 裡，兩個子類別都會自動套用 |

**關鍵設計決策（需保留紀錄）：** 角速度衰減採用「精確分解版」——把角速度分解成「由線速度決定的滾動分量」與「殘留分量（含加塞／english 自旋）」。此為使用者在設計討論階段主動選擇的方向，明確理由是為了讓未來實作加塞球路時不需要回頭修改這裡的邏輯。

### 事後修正（2026-07-25，實測回報後追加）

第一版把「殘留分量完全不衰減」跟「低於門檻就跳過不處理」兩件事放在一起，實測發現兩個問題：

1. **「低於門檻跳過」等於放棄處理**：球速一旦降到門檻附近，就不會再被寫入任何新速度，導致球用門檻附近的殘留速度/自旋永遠移動下去，達不到真正的靜止——這是實作疏漏，不是原本的設計意圖。修正為：低於門檻時**明確夾到精確 0**，只有在「線速度與殘留自旋都已經精確為 0」時才真正跳過（純效能優化，避免對已靜止的球重複寫入）。
2. **殘留自旋（側旋／english）完全不衰減，不符合真實世界**：真實撞球的側旋也會因為跟呢絨的摩擦逐漸停止，這是跟滾動摩擦不同、但同樣重要的物理現象。追加 `SPIN_DECAY_RATE`（取 Dr. Dave Pool Info 記載的球-呢絨自旋衰減率 5–15 rad/s² 中間值，10 rad/s²），對殘留分量獨立衰減，衰減邏輯跟滾動摩擦的線速度衰減對稱（同樣有「夾到 0」的 clamp 機制）。

---

## 7. 測試涵蓋（對應 Unit Test）

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| test_apply_decays_linear_velocity_by_rolling_friction | core/tests/test_rolling_resistance_service.py | 給定已知水平線速度，驗證扣減量等於 `rolling_friction_coeff * GRAVITY * PHYSICS_DT` |
| test_apply_clamps_to_zero_when_delta_exceeds_speed | core/tests/test_rolling_resistance_service.py | 速度很小、`Δv >= v_h` 時，驗證線速度水平分量歸零，不會變成負值/反向 |
| test_apply_clamps_linear_velocity_to_exact_zero_below_negligible_threshold | core/tests/test_rolling_resistance_service.py | 速度低於視覺門檻時，驗證確實被夾到精確 0（而非放著不管） |
| test_apply_skips_ball_already_fully_at_rest | core/tests/test_rolling_resistance_service.py | 線速度與殘留自旋都已經精確為 0 時，`set_velocities` 不應被呼叫 |
| test_apply_decays_residual_angular_velocity_by_spin_decay_rate | core/tests/test_rolling_resistance_service.py | 給定一個有「額外自旋分量」（模擬加塞）的球，驗證這個殘留分量依 `SPIN_DECAY_RATE` 衰減，衰減量等於 `spin_decay_rate * PHYSICS_DT` |
| test_apply_clamps_residual_angular_velocity_to_zero_below_negligible_threshold | core/tests/test_rolling_resistance_service.py | 殘留自旋幅度低於視覺門檻時，驗證確實被夾到精確 0 |
| test_apply_scales_rolling_angular_component_with_linear_velocity | core/tests/test_rolling_resistance_service.py | 驗證滾動分量的衰減比例跟線速度的衰減比例一致（同一個 `linear_scale`） |
| test_step_calls_rolling_resistance_before_state_dispatch | core/tests/test_table_orchestrator.py | 驗證 `TableOrchestrator.step()` 不論 `should_execute_action` / `current_state` 為何，都會呼叫 `rolling_resistance_service.apply()`，且在 state dispatch 之前 |

---

## 8. 待決定事項

- [ ] 無（設計討論已在四階段流程中確認完畢；2026-07-25 依實測回報追加「精確歸零」與「自旋衰減」兩項修正，已同步更新本文件）

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
- 自旋衰減率／球-呢絨摩擦係數來源：Dr. Dave Pool Info, "Pool Physics Property Constants," https://drdavepoolinfo.com/faq/physics/physical-properties/ （ball-cloth spin deceleration rate: 5–15 rad/s²；ball-cloth coefficient of sliding friction: 0.15–0.4，typical 0.2）
