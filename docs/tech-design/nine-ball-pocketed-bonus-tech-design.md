# 9 號球進袋加分判定 — 技術設計文件

> 生成時間：2026-07-27  
> 所屬專案：isaac-sim-digital-twin  
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/105

---

## 1. 功能概述

本功能接收 9 號球與白球是否進袋的兩個布林狀態，判斷該次擊球是否符合 9 號球有效進袋條件；只有 9 號球進袋且白球未進袋時回傳 `3.0`，其餘情況回傳 `0.0`。此計算器是平台無關的 `core/` 純函式，供後續 Reward Function 整合，不直接依賴 `ShotResult`、Isaac Sim 或 Omniverse API。

---

## 2. 模組清單與職責

| 模組 | 所在層級 | 職責 | 檔案路徑 |
|---|---|---|---|
| `calculate_nine_ball_pocketed_bonus` | core / services | 根據 9 號球與白球進袋狀態計算 9 號球獎勵 | `core/services/nine_ball_pocketed_bonus_calculator.py` |
| `TestCalculateNineBallPocketedBonus` | core / tests | 驗證四種進袋狀態組合與獎勵常數 | `core/tests/test_nine_ball_pocketed_bonus_calculator.py` |

---

## 3. 函式設計

### `calculate_nine_ball_pocketed_bonus`

**職責：** 判斷 9 號球進袋是否構成有效獎勵，並回傳固定分數。

**介面：**

```python
NINE_BALL_POCKETED_BONUS = 3.0


def calculate_nine_ball_pocketed_bonus(
    nine_ball_pocketed: bool,
    cue_ball_pocketed: bool,
) -> float:
    """9 號球進袋且白球未進袋時回傳獎勵，否則回傳 0.0。"""
    ...
```

**依賴：**

- 輸入來源：後續 Reward Function 從 `ShotResult.nine_ball_pocketed` 與 `ShotResult.cue_ball_pocketed` 取得狀態後傳入。
- 輸出去向：後續 Reward Function 將回傳值與散開分數、白球進袋懲罰及犯規分數整合。
- 外部依賴：無。

---

## 4. 資料流

```text
ShotResult
  → Reward Function 讀取 nine_ball_pocketed、cue_ball_pocketed
    → calculate_nine_ball_pocketed_bonus(
          nine_ball_pocketed,
          cue_ball_pocketed,
      )
      → 9 號球進袋且白球未進袋：3.0
      → 其他情況：0.0
    → Reward Function 整合其他 reward 分量
```

---

## 5. 依賴關係圖

```text
後續 Reward Function（#106）
  ├── 讀取 ShotResult 的進袋狀態
  └── 依賴 calculate_nine_ball_pocketed_bonus（取得 9 號球獎勵）

calculate_nine_ball_pocketed_bonus
  └── 無外部依賴，僅使用兩個 bool 輸入
```

本功能不反向依賴 `ShotResult`，以維持計算器可獨立測試並與既有白球進袋懲罰計算器的設計一致。

---

## 6. 邊緣案例與錯誤處理

| 情境 | 處理方式 |
|---|---|
| 9 號球進袋、白球未進袋 | 回傳 `3.0` |
| 9 號球與白球同時進袋 | 回傳 `0.0`，白球進袋時取消 9 號球獎勵 |
| 9 號球未進袋、白球進袋 | 回傳 `0.0`；白球懲罰由獨立計算器處理 |
| 兩球皆未進袋 | 回傳 `0.0` |
| 未先接觸 1 號球、碰庫不足等其他犯規 | 不在本函式處理，交由 #154 的犯規判定功能 |
| 傳入非 `bool` 值 | 不做執行期型別轉換或額外驗證，依賴 Python 型別標註與呼叫端契約 |

---

## 7. 測試涵蓋（對應 Unit Test）

此功能是確定性的純計算，不符合 Unit Test 豁免條件，不需要 Mock、Isaac Sim 環境或物理模擬。

| 測試案例 | 測試檔案 | 說明 |
|---|---|---|
| `test_returns_bonus_when_nine_ball_pocketed_without_cue_ball` | `core/tests/test_nine_ball_pocketed_bonus_calculator.py` | 9 號球進袋且白球未進袋時回傳 `3.0` |
| `test_returns_zero_when_nine_ball_and_cue_ball_both_pocketed` | `core/tests/test_nine_ball_pocketed_bonus_calculator.py` | 兩球同時進袋時不加分 |
| `test_returns_zero_when_only_cue_ball_pocketed` | `core/tests/test_nine_ball_pocketed_bonus_calculator.py` | 只有白球進袋時不加分 |
| `test_returns_zero_when_neither_ball_pocketed` | `core/tests/test_nine_ball_pocketed_bonus_calculator.py` | 兩球皆未進袋時不加分 |
| `test_bonus_constant_is_three` | `core/tests/test_nine_ball_pocketed_bonus_calculator.py` | 驗證獎勵常數固定為 `3.0` |

測試執行順序遵循 TDD：先建立測試並確認因目標模組尚未存在而失敗，再由實作任務補上純函式，最後確認完整測試通過。

---

## 8. 待決定事項

無。函式介面、責任邊界、輸入輸出與測試情境均已確認。
