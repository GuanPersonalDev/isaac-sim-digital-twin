# SHOT_ANGLE 復原為 (-180, 180) — 技術設計文件（#232-core）

> 生成時間：2026-09-11
>
> 所屬專案：isaac-sim-digital-twin
>
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/232
>
> 本輪範圍：**#232-core**（契約常數 + decoder 訊息 + core pytest + 文件同步）。**#232-train**（`init_std` 重評、重訓／熱啟動）延後，不在本文件實作範圍。

---

## 1. 功能概述

把 RL 動作空間的 `SHOT_ANGLE` 由 Milestone A 權宜值 `(-30.0, 30.0)` 復原為設計值 `(-180.0, 180.0)`，使走位球能表達任意水平瞄準方向。此為 Milestone B 硬前置。

一句話定義：

> **在不改換算演算法的前提下，把 `SHOT_ANGLE` 契約改回整圈半開語意，並讓 core 測試與文件與之一致；訓練超參與重訓另案處理。**

---

## 2. 設計決策（階段 1–4 已確認）

| 項目 | 決策 |
|---|---|
| 目標區間 | `SHOT_ANGLE = (-180.0, 180.0)`（物理域度；Box 記閉區間，反正規化尾端折回半開 `[-180, 180)`） |
| 端點語意 | 維持 #231 問題 1：用 `(-180, 180)` 而非 `(0, 360)`，讓 normalized 0 對應正對球堆（0°） |
| 新檔／新 port | **不新增**；只改既有常數、註解／錯誤訊息、測試、文件 |
| Decoder 演算法 | `_wrap_angle` / `denormalize_axis` **零邏輯變更**（已依區間中心設計） |
| `_canonical_shot_angle` | **保留**防禦性範圍檢查；拿掉錯誤訊息中「Milestone A 已收窄」字樣 |
| 手動 HUD 尺 | **不改** `manual_shot_bounds.py`（兩把尺各自獨立） |
| PPO `init_std` | 本輪**完全不動** `rsl_rl_ppo_cfg.py`；留給 #232-train |
| 舊 checkpoint | 正規化語意改變後**全部失效**（與 #227 相關）；文件與 #232 註明即可 |
| 追蹤 | 不建子 Issue；在 #232 留言標 core／train 拆分與驗收歸屬 |

---

## 3. 資料流與依賴

```text
policy 正規化輸出 [-1, 1]
        │
        ▼
decode_rl_action()          ← core/services/rl_action_decoder.py
  denormalize_axis(SHOT_ANGLE)
  _wrap_angle（以區間中心為錨，週期 360）
        │
        ▼
Action.shot_angle（物理域）
        │
        ├─→ 訓練桌：impulse strike / BilliardEnv
        └─→ Demo 桌：下游手臂解算（本輪不碰）

反向：
Action → normalize_action()
  _canonical_shot_angle（折回後若仍在區間外 → ValueError，整圈下有限角理論上不觸發）
```

**依賴（只讀／沿用）：**

- `core/models/action_bounds.py`：`SHOT_ANGLE`、`ACTION_BOUNDS`、`ACTION_CENTER`、`ACTION_HALF_SPAN`
- `core/services/rl_action_decoder.py`：`decode_rl_action`、`normalize_action`
- `core/models/action.py`：`Action` 資料類別（欄位不變）

**明確不依賴／不修改：**

- `core/models/manual_shot_bounds.py`
- `rl_task/.../rsl_rl_ppo_cfg.py`（本輪）
- `extension/`、Isaac Sim GUI 路徑

---

## 4. 模組改動清單（#232-core）

### 4.1 `core/models/action_bounds.py`

- 將 `SHOT_ANGLE` 改為 `(-180.0, 180.0)`。
- 刪除 Milestone A ±30° 權宜說明（含「30 不是拍腦袋」幾何推導段落作為**收窄理由**的那一段）。
- **保留**：半開區間／gym Box 閉區間說明；選 `(-180, 180)` 而非 `(0, 360)` 的理由（#231 問題 1）。
- 保留「手動面板不受此尺約束」的導向註解。
- `ACTION_CENTER`／`ACTION_HALF_SPAN` 由 bounds 現算，無需手改。

### 4.2 `core/services/rl_action_decoder.py`

- `_wrap_angle`／`denormalize_axis`：不改。
- `_canonical_shot_angle`：保留 `ValueError`；更新訊息，移除 Milestone A 收窄措辭。

### 4.3 測試

| 檔案／測試 | 動作 |
|---|---|
| `test_action_bounds._ISSUE_110_INDEX_TABLE` | `shot_angle` 字面量 `(-30, 30)` → `(-180, 180)` |
| `test_shot_angle_covers_every_legal_aim_at_the_one_ball` | **保留**（下界幾何；放寬後仍通過） |
| `test_shot_angle_is_centred_on_the_rack` | 不改 |
| `test_angles_outside_the_narrowed_action_space_are_rejected` | **刪除** |
| 新測試（建議名） | 有限角（含 90°／270°／±180）`normalize_action` 不拋；`+180` 與 `-180` 方向等價 |
| `test_angle_round_trips_exactly_inside_the_action_space` | `±1` 改為方向等價；中間值仍精確往返 |
| 其餘從 `SHOT_ANGLE`／`ACTION_BOUNDS` 現算的測試 | 預期自動對齊 |

驗收指令：

```bash
python -m pytest core/tests/test_action_bounds.py core/tests/test_rl_action_decoder.py -q
```

### 4.4 文件同步（實作 PR 時一併改，不單開本設計檔以外的設計循環）

- `docs/phase3-task-breakdown.md`：Action 索引表第 2 列角度範圍
- `docs/tech-design-5-2-script-controller-state-machine.md`：§5.4 `shot_angle` 列改為已復原整圈，並指向本文件／#232
- Issue #110：6 維契約索引表（留言或改 body）
- 本文件：`docs/tech-design-232-shot-angle-restore.md`

### 4.5 明確不做（#232-train）

- 調整 `init_std`、重訓、熱啟動權重縮放
- 雙尺度相容層
- 改 `manual_shot_bounds`
- 以訓練曲線為本輪 gate

---

## 5. 驗收標準對照

| #232 Issue 原驗收 | 本輪（core） | 延後（train） |
|---|---|---|
| 1. `SHOT_ANGLE == (-180, 180)` 且 core tests 綠 | ✅ | |
| 2. 任意有限角 `normalize_action` 不再拋 | ✅ | |
| 3. 重訓後指標不低於 Milestone A | | ✅ |
| 4. 文件與 #110 索引同步 | ✅ | |

---

## 6. 風險與副作用

- **舊 RL checkpoint 失效**：同一 normalized 輸出還原成不同物理角（約 6 倍尺度差）。Demo／#227 回放需新權重或接受瞄錯。
- **探索解析度回落**：`init_std=0.4` 在整圈上命中質量比約 2.9%（見 #232／#231）；不處理會「學不動」——由 #232-train 處理，本輪文件標明即可。
- **`SPREAD_REF` 不受影響**：物理角定義的開球取樣不變。

---

## 7. 實作順序建議

1. 改 `SHOT_ANGLE` 與註解（`action_bounds.py`）
2. 更新 decoder 錯誤訊息
3. 改／刪／補 pytest（含 `_ISSUE_110_INDEX_TABLE`）
4. 跑 §4.3 驗收指令
5. 同步 §4.4 文件與 #110／#232 留言
6. 開 PR（建議標題含 `#232-core`）；合入後 **不關閉** #232，直到 #232-train 完成或另決議關閉條件

---

## 8. 已確認決策 checklist

- [x] 階段 1：本輪只做 core 契約；train 延後
- [x] 階段 2：無新檔／無新 port；改 bounds + decoder 訊息 + tests + docs
- [x] 階段 3：演算法零改；測試改寫策略；文件清單
- [x] 階段 4：D1 方向等價；D2 刪 narrowed 測試並補全圈案例；D3 保留防禦 ValueError；D4 #232 留言拆分；D5 本技術文件；D6 PPO cfg 本輪不動
