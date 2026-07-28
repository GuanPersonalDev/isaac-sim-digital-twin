# 完整 Reward Function — 技術設計文件

> 生成時間：2026-07-28  
> 所屬專案：isaac-sim-digital-twin  
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/106

---

## 1. 功能概述

本功能接收 `ShotResult` 與 `BreakFoulResult`，將散開分數、白球進袋懲罰、9 號球進袋獎勵及開球犯規懲罰整合為單一 `float reward`，供後續 `BilliardEnv` 與 PPO 訓練使用。犯規透過負 reward 影響 return、advantage、policy loss 與 value loss，不另外建立自訂犯規 loss。第一接觸犯規會立即終止本次評估並回傳固定 `-1.5`；其他情況則依規則累加各 reward 分量。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `calculate_reward` | core / services | 驗證輸入、處理立即終止規則並整合所有 reward 分量 | `core/services/reward_service.py` |
| `TestCalculateReward` | core / tests | 驗證 reward 組合、犯規優先順序、9 號球獎勵取消及輸入驗證 | `core/tests/test_reward_service.py` |

本功能沿用既有 `ShotResult`、`BreakFoulResult` 與三個 reward 元件，不新增資料模型，也不修改 PPO loss。

---

## 3. 函式設計

### `calculate_reward`

**職責：** 將已計算完成的擊球結果與犯規結果整合為訓練使用的 reward。

**介面：**

```python
def calculate_reward(
    shot_result: ShotResult,
    break_foul_result: BreakFoulResult,
) -> float:
    """整合散開分數、進袋獎懲與開球犯規，回傳訓練 reward。"""
    ...
```

**依賴：**

- 輸入來源：
  - `ShotResult.spread_score`
  - `ShotResult.cue_ball_pocketed`
  - `ShotResult.nine_ball_pocketed`
  - `BreakFoulResult.penalty`
  - `BreakFoulResult.should_reset`
- 內部依賴：
  - `calculate_cue_ball_pocketed_penalty`
  - `calculate_nine_ball_pocketed_bonus`
- 輸出去向：後續 `BilliardEnv.step()` 的 reward。
- 外部依賴：無。

`ShotResult.final_ball_positions` 不在此函式使用；散開程度已由 #103 計算並存入 `spread_score`。

---

## 4. 資料流

```text
ShotResult + BreakFoulResult
           │
           ▼
calculate_reward(...)
  ├─ 驗證 spread_score 為有限值且介於 0.0–1.0
  ├─ 驗證 BreakFoulResult 為合法 domain state
  │
  ├─ should_reset=True
  │    → 直接回傳 -1.5
  │
  └─ should_reset=False
       ├─ reward = spread_score
       ├─ + cue_ball_pocketed_penalty
       ├─ + break_foul_result.penalty
       └─ 完全無犯規時
            + nine_ball_pocketed_bonus
           │
           ▼
      BilliardEnv.step()
       ├─ reward → PPO rollout buffer
       └─ BreakFoulResult.should_reset → episode termination
```

PPO 使用 reward 計算 return 與 advantage，再形成 policy loss 與 value loss；`reward_service` 不直接知道或修改 loss function。

---

## 5. 依賴關係圖

```text
calculate_reward
  ├── 依賴 ShotResult（散開與進袋狀態）
  ├── 依賴 BreakFoulResult（犯規扣分與立即重置）
  ├── 呼叫 calculate_cue_ball_pocketed_penalty
  └── 呼叫 calculate_nine_ball_pocketed_bonus

BilliardEnv（#121／#122）
  ├── 呼叫 calculate_reward
  ├── 將 reward 交給 PPO
  └── 使用 BreakFoulResult.should_reset 控制 episode termination
```

`reward_service` 位於平台無關的 `core/services`，不依賴 Isaac Sim、Isaac Lab、PPO、UI 或 extension 層。

---

## 6. Reward 規則與邊緣案例

### 基本公式

非立即終止時：

```text
reward
= spread_score
+ cue_ball_pocketed_penalty
+ break_foul_result.penalty
+ nine_ball_pocketed_bonus（僅完全無犯規時）
```

### 規則表

| 情境 | Reward 處理 |
|---|---|
| 合法一般擊球 | `spread_score` |
| 合法 9 號球進袋 | `spread_score + 3.0` |
| 白球進袋 | `spread_score - 3.5` |
| 白球與 9 號球同時進袋 | `spread_score - 3.5`，不加 9 號球獎勵 |
| 碰庫不足 | `spread_score - 0.5` |
| 白球進袋且碰庫不足 | `spread_score - 3.5 - 0.5` |
| 碰庫不足且 9 號球進袋 | `spread_score - 0.5`，不加 9 號球獎勵 |
| 第一接觸犯規 | 固定 `-1.5`，不計其他 reward 分量 |

### 輸入驗證

| 情境 | 處理方式 |
|---|---|
| `spread_score` 位於 `0.0–1.0` 且為有限值 | 接受 |
| `spread_score` 為 `NaN`、正負無限大或範圍外 | 拋出 `ValueError` |
| `BreakFoulResult(0.0, False)` | 合法：無開球犯規 |
| `BreakFoulResult(-0.5, False)` | 合法：碰庫不足 |
| `BreakFoulResult(-1.5, True)` | 合法：第一接觸犯規 |
| 其他 penalty／should_reset 組合 | 拋出 `ValueError` |

### Reward 數值範圍

- Reward 不做 clamp。
- 合法最高值：`4.0`（散開 `1.0` + 9 號球 `3.0`）。
- 非立即終止最低值：`-4.0`（散開 `0.0`、白球 `-3.5`、碰庫不足 `-0.5`）。
- 第一接觸犯規固定為 `-1.5`。
- Reward normalization／scaling 留給 #123 的 PPO 超參數設計。

---

## 7. 測試涵蓋（對應 Unit Test）

此功能為確定性的純計算，不符合 Unit Test 豁免條件，不需要 Mock、Isaac Sim、Isaac Lab 或 PPO 環境。

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| `test_returns_spread_score_for_legal_shot` | `core/tests/test_reward_service.py` | 合法一般擊球只回傳散開分數 |
| `test_adds_nine_ball_bonus_for_foul_free_shot` | `core/tests/test_reward_service.py` | 合法 9 號球進袋加入 `3.0` |
| `test_applies_cue_ball_penalty` | `core/tests/test_reward_service.py` | 白球進袋扣除 `3.5` |
| `test_cue_ball_pocketed_cancels_nine_ball_bonus` | `core/tests/test_reward_service.py` | 白球與 9 號球同時進袋不加 bonus |
| `test_applies_insufficient_rail_penalty` | `core/tests/test_reward_service.py` | 碰庫不足扣除 `0.5` |
| `test_accumulates_cue_ball_and_rail_penalties` | `core/tests/test_reward_service.py` | 白球與碰庫不足懲罰可累加 |
| `test_break_foul_cancels_nine_ball_bonus` | `core/tests/test_reward_service.py` | 任一開球犯規取消 9 號球獎勵 |
| `test_first_contact_foul_returns_only_terminal_penalty` | `core/tests/test_reward_service.py` | 立即終止時固定回傳 `-1.5` |
| `test_accepts_spread_score_boundaries` | `core/tests/test_reward_service.py` | 接受 `0.0` 與 `1.0` |
| `test_rejects_invalid_spread_score` | `core/tests/test_reward_service.py` | 拒絕範圍外、`NaN` 與無限大 |
| `test_rejects_invalid_break_foul_result` | `core/tests/test_reward_service.py` | 拒絕不可能的犯規狀態組合 |

測試資料使用 `pytest.fixture` 建立。執行順序遵循 TDD：先建立函式骨架，再撰寫並執行測試確認失敗，最後補上實作使完整測試通過。

---

## 8. Training 與 Issue 邊界

- #106：完整 Reward Function，包含所有進袋獎懲與犯規。
- #121／#122：`BilliardEnv` 串接 reward 與 episode termination。
- #123：PPO 選擇、超參數、reward normalization／scaling。
- #155：原本的「整合完整 Reward Function」已由修正版 #106 完整涵蓋，應標記為 superseded 並關閉。

---

## 9. 待決定事項

無。函式介面、reward 規則、犯規優先順序、輸入驗證、Training 資料流與 Issue 邊界均已確認。
