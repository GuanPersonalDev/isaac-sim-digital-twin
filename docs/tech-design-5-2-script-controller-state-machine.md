# ScriptController 狀態機 — 技術設計文件

> 生成時間：2026-07-17
>
> 所屬專案：isaac-sim-digital-twin
>
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/93

---

## 1. 功能概述

設計 `ScriptController`（Milestone B 手臂執行用的程式控制器，實作 `ControllerBase`）所依循的狀態機，涵蓋一次完整擊球循環：`IDLE → AIMING → STRIKING → WAITING → RESET → IDLE`，並定義任何狀態下發生錯誤時轉入 `ERROR`。

---

## 2. 設計決策

直接擴充既有 `BilliardStatus`（`core/models/billiard_state.py`），不另外新增獨立的狀態 enum。原因：

- `BilliardStatus` 目前只被 `BilliardState`、`core/tests/test_models.py`、`core/models/__init__.py` 引用，改動範圍小、無其他消費者。
- 避免同一個「撞球狀態」概念在專案中出現兩套平行的狀態定義。

變更內容：

| 原值 | 新值 | 說明 |
|---|---|---|
| `SHOOTING = "shooting"` | `STRIKING = "striking"` | 語意改為明確對應「揮桿擊球」動作 |
| `RESETTING = "resetting"` | `RESET = "reset"` | 與 task-breakdown 命名對齊 |
| （無） | `WAITING = "waiting"` | 新增，等待所有球靜止（供 #98 `ScriptController.WAITING` 沿用） |
| （無） | `ERROR = "error"` | 新增，任何狀態下發生非預期錯誤時轉入 |

---

## 3. 狀態轉換條件（2026-07-18 修訂，見第 5 節）

```text
任何狀態  → ERROR    : observation.has_error == True（優先權最高，蓋過其他轉換判斷）
RESET     → IDLE     : observation.is_motion_complete == True
IDLE      → AIMING   : observation.is_init_state == True and observation.is_ball_moving == False
AIMING    → STRIKING : observation.is_motion_complete == True
STRIKING  → WAITING  : observation.is_motion_complete == True
WAITING   → RESET    : observation.is_ball_moving == False
```

`ERROR` 目前不定義自動恢復路徑；發生後交由呼叫端（`ScriptController` 的使用者或上層流程）決定是否重新初始化，不在本次設計範圍內臆測復原邏輯。

初始狀態為 `RESET`：不論手臂／場景實際處於什麼姿態開機，一律先走 `RESET → IDLE` 的到位確認，確保 `IDLE` 是可信賴的已知起點。

---

## 4. 已確認決策與後續範圍

- [x] 直接擴充 `BilliardStatus`，不建立獨立狀態 enum。
- [x] `STRIKING` / `RESET` 命名對齊 task-breakdown 5-2 條目。
- [x] `WAITING` 判定閾值沿用 #178 定案的 0.001 m/s（由下游偵測後換算為 `Observation.is_ball_moving`，狀態機本身不比較速度數值）。
- [x] `ScriptController` 類別本體與其 Unit Test 已完成第 5 節的架構修訂，不再依賴 `ArticulationAPI`。

---

## 5. 決策修訂：Controller／執行層職責分離（2026-07-18）

### 5.1 背景

原始設計（第 1～4 節）假設 `ScriptController` 是「Milestone B 手臂執行用」的控制器，狀態轉換直接呼叫 `ArticulationAPI`（`move_to_pose` / `move_to_home` / `is_motion_complete`）判斷是否到位。後續盤點 [phase3-schedule.md](phase3-schedule.md) 發現 Milestone A（訓練桌）已定案採用 `set_velocities` 衝量式擊球（#177），與手臂路徑規劃（#96、#97，已排入 M7 Milestone B）走的是兩條不同機制，若 `ScriptController` 綁死 `ArticulationAPI`，Demo 桌與訓練桌就無法共用同一顆 Controller。

### 5.2 修訂後的職責劃分

- **`ControllerBase` 的實作只分兩種決策風格**：狀態機操作（`ScriptController`，本文件描述的對象）與衝量式操作，這個分類跟「有沒有手臂」無關。
- **`ScriptController` 是純決策類別**：建構子不收任何參數（不再依賴 `ArticulationAPI`），`get_action(observation) -> Action` 只讀 `Observation`、只回傳 `Action`，內部不呼叫任何執行層 API，因此不會拋出執行層例外。
- **執行（如何把 Action 變成物理動作）是下游職責，依桌子類型分流**：
  - Demo 桌：將 `Action` 解算為手臂路徑規劃，再呼叫 `ArticulationAPI` 操作手臂（對應 #95 `ArticulationAPIImpl`、#96/#97 手臂 AIMING/STRIKING，Milestone B）。
  - 訓練桌：將 `Action` 直接轉換為母球衝量，餵入 RL 訓練迴圈更新 Model（對應 #177）。
- **Demo 桌與訓練桌共用同一個 `ScriptController` 類別**，差異只在下游怎麼解讀 `Action`；要使用「狀態機」或「衝量式」哪一種 `ControllerBase` 實作，由 DebugMenu 切換，與桌子類型無關。

### 5.3 資料模型異動

`Observation` 新增（下游執行層偵測後回寫，`ScriptController` 只讀）：

| 欄位 | 型別 | 用途 |
|---|---|---|
| `is_init_state` | `bool` | 撞球桌是否處於初始擺球位置 |
| `is_ball_moving` | `bool` | 是否有球仍在移動（沿用 #178 的 0.001 m/s 閾值換算） |
| `is_motion_complete` | `bool` | 手臂／場景是否已到達下游執行的目標（取代原本 `ArticulationAPI.is_motion_complete()` 直接呼叫） |
| `has_error` | `bool` | 下游執行層偵測到異常（API 失敗、非預期物理狀態）時回寫，觸發 `ScriptController` 進入 `ERROR` |

`Action` 新增一個 `bool` 欄位 `should_execute_action`，表示「此 tick 下游是否需要真的觸發一次動作」——語意不限於手臂，Demo 桌（手臂路徑規劃）與訓練桌（衝量式擊球，#177）皆共用同一個欄位判斷「是否為狀態剛轉換的觸發 tick」，避免同一動作在同一狀態持續的多個 tick 內被重複觸發。

`STRIKING` 狀態固定輸出 `cue_ball_speed = ScriptController.MAX_CUE_BALL_SPEED`（`3.3392`，2026-07-26 換裝 Barrett WAM + 差動 IK 後實測母球初速上限，取代原 Issue #176 UR5 實測的 `1.313`）、`shot_angle = 0`、`position_offset = [0.0, 0.0]`；非零的角度/位移偏移量欄位保留給未來 `ModelController`（RL 模型）輸出使用，`ScriptController` 本身不做動態計算。

`ArticulationAPI` **不需要**新增 `stop()`（曾於討論中提出，後定案改由下游在偵測到異常時直接回寫 `Observation.has_error`，不需要 `ScriptController` 呼叫任何停止方法）。

### 5.4 Action 欄位改為對齊 RL 6 維規格（2026-07-19 修訂，Issue #177 前置）

`Action` 原先的 `position_offset` 是 3 維，語意未定案。依 `phase3-task-breakdown.md`「RL 設計定案 → Action（6 個數字）」表格重新對齊：

| 欄位 | 型別 | 用途 |
|---|---|---|
| `cue_ball_placement` | `list[float]`（2 維） | RL 索引 0–1；母球桌台相對 XY（m），球心須位於 Kitchen 安全範圍 |
| `shot_angle` | `float` | RL 索引 2；`[0, 360)` 度，`0°` 朝桌台 `+Y`，正角朝 `-X` 增加 |
| `cue_ball_speed` | `float` | RL 索引 3；母球目標初速，範圍 `0.5–3.3392 m/s` |
| `position_offset` | `list[float]`（2 維） | RL 索引 4–5；依序為上下／左右偏移，範圍各為 `[-0.5, 0.5]` 球半徑 |
| `should_execute_action` | `bool` | 見 5.3 節 |

`cue_ball_placement`（2）+ `shot_angle` + `cue_ball_speed` + `position_offset`（2）＝ 6 維，對應 RL Action 空間；`should_execute_action` 是 `ScriptController` 執行層額外需要的欄位（Demo 桌、訓練桌皆共用），不計入 RL 的 6 維。執行期 no-op 可使用 `cue_ball_speed = 0.0`；RL action space 的正規化、裁切及 `gymnasium.spaces.Box` 由 `BilliardEnv`（#122）負責。

`RigidBodyAPI`（`core/ports/rigid_body_api.py`）新增 `set_velocities(prim_path, linear_velocity, angular_velocity)`，供 Issue #177（impulse-based 擊球）直接對母球賦速使用。
