# 開球犯規判定 — 技術設計文件

> 生成時間：2026-07-28  
> 所屬專案：isaac-sim-digital-twin  
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/154

---

## 1. 功能概述

本功能接收白球第一個接觸的號碼球、進袋號碼球集合與碰庫號碼球集合，依序判斷「未先接觸 1 號球」及「無號碼球進袋且少於 4 顆不同號碼球碰庫」兩種開球犯規。輸出包含扣分與是否需要立即重置，供後續完整 Reward Function 與控制流程使用。本功能只處理平台無關的純判定邏輯，不負責從 Isaac Sim 接觸事件收集輸入資料。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `BreakFoulResult` | core / models | 保存犯規扣分與立即重置旗標 | `core/models/break_foul_result.py` |
| `evaluate_break_foul` | core / services | 驗證輸入並依優先順序判斷兩種開球犯規 | `core/services/break_foul_evaluator.py` |
| `TestEvaluateBreakFoul` | core / tests | 驗證犯規優先順序、碰庫邊界及輸入驗證 | `core/tests/test_break_foul_evaluator.py` |

事件收集模組不屬於 #154，後續應由獨立任務提供第一接觸球、進袋球與碰庫球資料。

---

## 3. 類別與函式設計

### `BreakFoulResult`

**職責：** 以不可變資料物件表示單次開球犯規判定結果。

**介面：**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BreakFoulResult:
    penalty: float
    should_reset: bool
```

**依賴：**

- 輸入來源：`evaluate_break_foul` 建立。
- 輸出去向：後續完整 Reward Function 使用 `penalty`，控制流程使用 `should_reset`。
- 外部依賴：僅 Python 標準函式庫 `dataclasses`。

### `evaluate_break_foul`

**職責：** 驗證號碼球 ID，依優先順序判斷第一接觸犯規與開球力道不足犯規。

**介面：**

```python
FIRST_CONTACT_FOUL_PENALTY = -1.5
INSUFFICIENT_RAIL_CONTACT_PENALTY = -0.5
MIN_RAIL_CONTACTED_OBJECT_BALLS = 4


def evaluate_break_foul(
    first_contacted_ball_id: int | None,
    pocketed_object_ball_ids: set[int],
    rail_contacted_object_ball_ids: set[int],
) -> BreakFoulResult:
    """判斷開球犯規，回傳扣分與是否立即重置。"""
    ...
```

**依賴：**

- 輸入來源：未來的事件收集模組。
- 輸出去向：後續完整 Reward Function 與控制流程。
- 內部依賴：`BreakFoulResult`。
- 外部依賴：無。

---

## 4. 資料流

```text
未來的事件收集模組
  ├─ first_contacted_ball_id
  ├─ pocketed_object_ball_ids
  └─ rail_contacted_object_ball_ids
           │
           ▼
evaluate_break_foul(...)
  ├─ 驗證所有 ball ID 皆為 1–9（first_contacted_ball_id 可為 None）
  ├─ 第一接觸不是 1 號球
  │    → BreakFoulResult(-1.5, True)
  ├─ 第一接觸合法，但無球進袋且不同碰庫球少於 4 顆
  │    → BreakFoulResult(-0.5, False)
  └─ 其餘合法情況
       → BreakFoulResult(0.0, False)
           │
           ├─ penalty → 後續完整 Reward Function
           └─ should_reset → 控制流程
```

第一接觸犯規具有最高優先權；命中後立即回傳，不與碰庫不足懲罰疊加。

---

## 5. 依賴關係圖

```text
未來事件收集模組（不屬於 #154）
  └── 提供開球事件摘要

evaluate_break_foul
  └── 依賴 BreakFoulResult（建立判定結果）

後續完整 Reward Function（#155）
  └── 使用 BreakFoulResult.penalty

後續控制流程
  └── 使用 BreakFoulResult.should_reset
```

`core/` 不依賴事件資料來自哪個 Isaac Sim API；事件收集與平台橋接必須維持在獨立責任邊界。

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| `first_contacted_ball_id is None` | 視為未接觸 1 號球，回傳 `-1.5`、立即重置 |
| 第一接觸為 2–9 號球 | 回傳 `-1.5`、立即重置 |
| 第一接觸合法但兩種犯規同時成立 | 第一接觸犯規優先，不疊加為 `-2.0` |
| 有任一號碼球進袋、碰庫球少於 4 顆 | 不構成碰庫不足犯規 |
| 無號碼球進袋、3 顆不同號碼球碰庫 | 回傳 `-0.5`、不立即重置 |
| 無號碼球進袋、剛好 4 顆不同號碼球碰庫 | 合法，回傳 `0.0` |
| 同一顆球同時出現在進袋與碰庫集合 | 允許，代表球先碰庫再進袋 |
| 集合重複表示同一顆球 | 使用 `set[int]`，只計算一顆不同號碼球 |
| `first_contacted_ball_id` 為 0 或超出 1–9 | 拋出 `ValueError` |
| 進袋或碰庫集合包含 0 或超出 1–9 | 拋出 `ValueError` |

白球不屬於 `pocketed_object_ball_ids` 或 `rail_contacted_object_ball_ids`；呼叫端若傳入白球 ID `0`，視為違反介面契約。

---

## 7. 測試涵蓋（對應 Unit Test）

此功能是確定性的純計算，不符合 Unit Test 豁免條件，不需要 Mock、Isaac Sim 環境或物理模擬。

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| `test_returns_first_contact_penalty_when_first_ball_is_not_one` | `core/tests/test_break_foul_evaluator.py` | 第一接觸為其他號碼球時回傳 `-1.5` 並立即重置 |
| `test_returns_first_contact_penalty_when_no_ball_contacted` | `core/tests/test_break_foul_evaluator.py` | 第一接觸為 `None` 時視為犯規 |
| `test_first_contact_foul_takes_precedence_over_rail_foul` | `core/tests/test_break_foul_evaluator.py` | 兩種犯規同時成立時不疊加 |
| `test_returns_rail_penalty_when_no_ball_pocketed_and_three_balls_hit_rail` | `core/tests/test_break_foul_evaluator.py` | 驗證少於 4 顆碰庫的犯規路徑 |
| `test_returns_no_penalty_when_four_balls_hit_rail` | `core/tests/test_break_foul_evaluator.py` | 驗證剛好 4 顆的合法邊界 |
| `test_returns_no_penalty_when_object_ball_pocketed` | `core/tests/test_break_foul_evaluator.py` | 有號碼球進袋時略過碰庫不足判定 |
| `test_counts_each_rail_contacted_ball_once` | `core/tests/test_break_foul_evaluator.py` | `set[int]` 以不同球數量判定 |
| `test_allows_ball_in_both_pocketed_and_rail_sets` | `core/tests/test_break_foul_evaluator.py` | 允許同一顆球先碰庫再進袋 |
| `test_raises_value_error_for_invalid_first_contact_ball_id` | `core/tests/test_break_foul_evaluator.py` | 驗證第一接觸球 ID 範圍 |
| `test_raises_value_error_for_invalid_pocketed_ball_id` | `core/tests/test_break_foul_evaluator.py` | 驗證進袋球 ID 範圍 |
| `test_raises_value_error_for_invalid_rail_contacted_ball_id` | `core/tests/test_break_foul_evaluator.py` | 驗證碰庫球 ID 範圍 |

測試資料使用 `pytest.fixture` 建立。執行順序遵循 TDD：先建立資料模型與函式骨架，再撰寫並執行測試以確認失敗，最後補上實作使完整測試通過。

---

## 8. 待決定事項

無。函式介面、輸入驗證、犯規優先順序、邊界行為與測試情境均已確認。
