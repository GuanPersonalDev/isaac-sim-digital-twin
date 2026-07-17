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

## 3. 狀態轉換條件

```text
IDLE      → AIMING   : 收到擊球 Action（cue_speed / shot_angle / position_offset）
AIMING    → STRIKING : 球桿末端已移動到擊球預備位置（RMPflow 到位確認）
STRIKING  → WAITING  : 揮桿衝擊動作執行完畢（沿擊球方向加速推進完成）
WAITING   → RESET    : 所有球速度 < 0.001 m/s（與 #178 定案閾值一致，見 phase3-plan-risks-solutions.md #6）
RESET     → IDLE     : 場景重置完成（球回到開球位置）
任何狀態   → ERROR    : API 呼叫失敗或偵測到非預期物理狀態（例如球飛出桌面、Prim 遺失）
```

`ERROR` 目前不定義自動恢復路徑；發生後交由呼叫端（`ScriptController` 的使用者或上層流程）決定是否重新初始化，不在本次設計範圍內臆測復原邏輯。

---

## 4. 已確認決策與後續範圍

- [x] 直接擴充 `BilliardStatus`，不建立獨立狀態 enum。
- [x] `STRIKING` / `RESET` 命名對齊 task-breakdown 5-2 條目。
- [x] `WAITING` 判定閾值沿用 #178 定案的 0.001 m/s。
- [ ] `ScriptController` 類別本體（狀態轉換的實際程式邏輯）留待 #94（Unit Test）與後續 impl 任務實作，本項僅完成狀態機設計與資料模型擴充。
