# HUD ShotResult 顯示 — 技術設計文件

> 生成時間：2026-09-10
> 所屬專案：isaac-sim-digital-twin
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/116
> 前置依賴：本設計建立在「Next Rack 手動重置閘門」功能之上（`core/models/billiard_state.py` 的
> `BilliardStatus` 新增 `READY_TO_RESET` 狀態，球停止移動後停在此狀態，等使用者按 HUD
> 「Next Rack」按鈕才真正瞬移回開球擺位）。該功能技術文件見
> `docs/tech-design/next-rack-manual-reset-gate-tech-design.md`；其 GitHub Issue 尚未建立
> （由 progress-planner 於下一步驟建立）。

---

## 1. 功能概述

HUD 面板新增一個 ShotResult 顯示區塊，每 frame 輪詢顯示「這一局」的散開分數（呼叫既有
`core/services/spread_score_calculator.py` 的 `calculate_spread_score()`）、母球是否進袋、
9 號球是否進袋。輸入是當前桌台的 `TableSession` 狀態與 `TableBallSet` 落袋紀錄，輸出是一段
多行文字，顯示在 HUD 面板既有 `_status_label` 下方的新 Label。狀態機在 `READY_TO_RESET` 時
顯示真值結果，其餘狀態（`RESET` / `IDLE` / `AIMING` / `STRIKING` / `WAITING` / `ERROR`）顯示
佔位文字 `"Shot Result: -"`。使用場景：使用者打完一桿、球停止移動後，可以立即在 HUD 上看到
這一桿的散開品質與關鍵球進袋狀況，再決定是否按 Next Rack 重擺。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `TableBallSet`（擴充） | core.models | 記錄球落袋當下的世界座標快照，提供落袋 id 集合與位置查詢 | `core/models/table_ball_set.py` |
| `BilliardDigitalTwin`（擴充） | extension（Application 組裝層） | 組出 ShotResult 文字：讀狀態機、讀落袋紀錄、讀即時觀測、呼叫 `calculate_spread_score()` | `extension/billiard_digital_twin/billiard_digital_twin.py` |
| `HudPanel`（擴充） | extension.ui | 顯示 ShotResult 文字的新 Label，每 frame 輪詢更新 | `extension/ui/hud_panel.py` |

（`calculate_spread_score()` 為既有 `core/services/spread_score_calculator.py` 函式，本功能純呼叫，不修改）

---

## 3. 類別設計

### TableBallSet（擴充）

**職責：** 追蹤球的落袋狀態，並在落袋當下存一次位置快照（因為落袋後 PhysX 模擬可能持續讓球
掉落，trigger 體積不保證底下有實體地板接住，事後讀即時位置不可靠）。

**新增介面：**
```python
class TableBallSet:
    def mark_ball_pocketed(self, ball_id: int) -> None:
        """球進袋時呼叫（取代原本直接接 hide_ball）。沿用既有 hide_ball 做視覺隱藏，
        並額外讀取該球落袋當下的世界座標、換算成桌台相對座標存進快照，
        供之後 ShotResult 查詢（此時球可能已離開合理位置）。"""
        ...

    def get_pocketed_ball_ids(self) -> set[int]:
        """回傳目前落袋球 id 集合的副本，避免外部修改內部狀態。"""
        ...

    def get_pocketed_ball_position(self, ball_id: int) -> tuple[float, float]:
        """回傳該球落袋當下的桌台相對座標快照。只會在呼叫端已透過
        get_pocketed_ball_ids() 確認 ball_id 存在時呼叫，不做額外防呆。"""
        ...
```

**既有方法變更：**
- `reset(self, positions)` 尾端新增清空落袋狀態的兩行（`self._pocketed_ball_ids = set()`、
  `self._pocketed_ball_positions = {}`），語意對齊「新的一局開始」。由於「Next Rack 手動重置
  閘門」功能上線後 `reset()` 只在使用者按下 Next Rack 觸發真正重擺時才呼叫，落袋快照會
  持續保留到那個時間點，正好支撐 ShotResult 在 `READY_TO_RESET` 期間穩定顯示的需求。

**依賴：**
- 輸入來源：`core/services/pocket_event_handler.py`（`PocketEventHandler` 的
  `on_ball_pocketed` callback 觸發）、`core/ports/rigid_body_api.py`（讀球即時世界座標）
- 輸出去向：`BilliardDigitalTwin.get_shot_result_text()` 查詢落袋 id 集合與快照位置

---

### BilliardDigitalTwin（擴充）

**職責：** 組裝層，串接 `TableSession`（狀態機、即時觀測）、`TableBallSet`（落袋紀錄）與
`calculate_spread_score()`，產出 HUD 要顯示的最終文字。

**介面：**
```python
class BilliardDigitalTwin:
    def get_shot_result_text(self, table_id: str) -> str:
        """HUD 每 frame 輪詢的 ShotResult 文字。狀態機不在 READY_TO_RESET 時
        回傳佔位文字；查無 table_id 回傳空字串（沿用 get_manual_shot_status_text()
        的既定慣例）。"""
        ...
```

**既有方法變更：**
- `_build_pocket_event_handler()`：`on_ball_pocketed=table_ball_set.hide_ball` 改為
  `on_ball_pocketed=table_ball_set.mark_ball_pocketed`（單行修改）。此方法由 Training 桌與
  Demo 桌共用，兩者都會受影響，但 Training 桌只是多存一份沒人查詢的落袋狀態，無風險。
- `HudPanel(...)` 建構呼叫多傳一個參數 `self.get_shot_result_text`。

**依賴：**
- 輸入來源：`TableSession.get_current_state()` / `get_last_observation()`（既有）、
  `TableBallSet.get_pocketed_ball_ids()` / `get_pocketed_ball_position()`（新增）、
  `TableBallSet.get_table_x_y()`（既有）、`calculate_spread_score()`（既有）
- 輸出去向：`HudPanel` 建構時注入的 `get_shot_result_text` callable

---

### HudPanel（擴充）

**職責：** 顯示層，每 frame 輪詢 `get_shot_result_text(table_id)` 並更新畫面文字。

**介面：**
```python
class HudPanel:
    def __init__(self, ..., get_shot_result_text: Callable[[str], str]):
        """新增 DI 參數，存成 self._get_shot_result_text，沿用既有
        get_manual_shot_status_text 等 callable 注入的既定模式。"""
        ...
```

**既有方法變更：**
- `_build_body_ui()`：在既有 `self._status_label = ui.Label("", word_wrap=True)` 後面新增
  `self._shot_result_label = ui.Label("", word_wrap=True)`，沿用既有多行文字 Label 寫法。
- `_on_update()`：仿照 `self._status_label.text = ...` 新增一行
  `self._shot_result_label.text = self._get_shot_result_text(table_id) if table_id is not None else ""`。

**依賴：**
- 輸入來源：`BilliardDigitalTwin.get_shot_result_text`（建構時 DI 注入）
- 輸出去向：無（終端顯示層）

---

## 4. 資料流

```
PocketEventHandler（球進袋觸發）
  → TableBallSet.mark_ball_pocketed(ball_id)
    → hide_ball(ball_id)（既有視覺隱藏）
    → 讀取即時世界座標，換算桌台相對座標，存進 _pocketed_ball_positions
    → ball_id 加進 _pocketed_ball_ids
  （落袋快照建立完成，等待狀態機推進到 READY_TO_RESET）

HudPanel._on_update()（每 frame）
  → BilliardDigitalTwin.get_shot_result_text(table_id)
    → TableSession.get_current_state() 讀狀態
    → 若非 READY_TO_RESET → 回傳 "Shot Result: -"
    → 若為 READY_TO_RESET：
        → TableSession.get_last_observation() 讀即時觀測
        → TableBallSet.get_pocketed_ball_ids() 讀落袋集合
        → 組出 1~9 號球的桌台相對座標（落袋球用快照位置，未落袋球用即時觀測位置）
        → calculate_spread_score(ball_positions, pocketed_ids) 算散開分數
        → 組成三行文字（Spread Score / Cue Ball Pocketed / 9-Ball Pocketed）
  → 回傳文字
  → HudPanel._shot_result_label.text 更新畫面
```

---

## 5. 依賴關係圖

```
TableBallSet
  ├── 依賴 core/ports/rigid_body_api（讀球即時世界座標，用於落袋快照）
  └── 依賴既有 hide_ball()（沿用視覺隱藏邏輯，不重複實作）

BilliardDigitalTwin.get_shot_result_text()
  ├── 依賴 TableSession（core/services/table_session.py，狀態機與即時觀測）
  ├── 依賴 TableBallSet（落袋 id 集合與落袋快照座標）
  ├── 依賴 core/models/billiard_state.py 的 BilliardStatus.READY_TO_RESET
  │     （⚠️ 由「Next Rack 手動重置閘門」功能新增，見
  │     docs/tech-design/next-rack-manual-reset-gate-tech-design.md，本功能不負責實作該狀態本身）
  └── 依賴 core/services/spread_score_calculator.py 的 calculate_spread_score()（既有純函式）

HudPanel
  └── 依賴 BilliardDigitalTwin.get_shot_result_text（建構時 DI 注入的 callable）
```

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 使用者從沒按過 Next Rack（剛開機、Timeline 剛 PLAY） | 狀態機尚未進入過 `READY_TO_RESET`，`get_shot_result_text()` 顯示 `"Shot Result: -"`，正常行為 |
| Training 桌呼叫 `get_shot_result_text()` | `_demo_table_ball_sets` 只存 Demo 桌，`table_ball_set` 為 `None` 時直接回傳空字串；但 Training 桌本來就不會出現在 HUD 桌台下拉選單（`get_demo_table_ids()` 已排除），不會有這條呼叫路徑 |
| `get_pocketed_ball_position()` 查詢不存在的 ball_id | 不做防呆，沿用專案「不為不可能發生的情境寫錯誤處理」慣例——呼叫端只會在 `ball_id` 已確認存在於 `get_pocketed_ball_ids()` 回傳集合內時才呼叫 |
| 查無 table_id | `get_shot_result_text()` 回傳空字串，沿用 `get_manual_shot_status_text()` 的既定慣例 |

---

## 7. 測試涵蓋（對應 Unit Test）

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| test_mark_ball_pocketed_hides_ball_and_records_snapshot | core/tests/test_table_ball_set.py | 驗證 `mark_ball_pocketed()` 呼叫 `hide_ball()` 並正確存下桌台相對座標快照 |
| test_get_pocketed_ball_ids_returns_copy | core/tests/test_table_ball_set.py | 驗證回傳的是副本，外部修改不影響內部狀態 |
| test_reset_clears_pocketed_state | core/tests/test_table_ball_set.py | 驗證 `reset()` 會清空落袋 id 集合與快照位置 |
| test_get_shot_result_text_placeholder_when_not_ready_to_reset | isaac-sim-digital-twin 對應的 billiard_digital_twin 測試檔（若既有測試檔案已涵蓋 `BilliardDigitalTwin` 的其餘方法，沿用同一檔） | 驗證非 `READY_TO_RESET` 狀態下回傳固定佔位文字 |
| test_get_shot_result_text_computes_spread_score_when_ready | 同上 | 驗證 `READY_TO_RESET` 狀態下正確組出 spread score 與母球／9 號球進袋狀態文字 |
| test_get_shot_result_text_empty_when_table_not_found | 同上 | 驗證查無 table_id 回傳空字串 |

（`extension/ui/hud_panel.py` 屬於 UI 顯示層，依 `skills/unit-test.md` 豁免條件，不強制寫 Unit Test）

---

## 8. 待決定事項

- [ ] 「Next Rack 手動重置閘門」功能的 GitHub Issue 尚未建立（技術文件已存在於
  `docs/tech-design/next-rack-manual-reset-gate-tech-design.md`），待建立後回填本文件開頭的
  Issue 連結
- [ ] `BilliardStatus.READY_TO_RESET` 狀態本身由前置依賴功能負責新增，本功能實作前需確認該狀態已合併，否則 `get_shot_result_text()` 無法正確判斷顯示時機
