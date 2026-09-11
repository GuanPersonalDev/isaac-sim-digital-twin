# 整圈 SHOT_ANGLE 下重評 init_std — 技術設計文件（#245 / #232-train）

> 生成時間：2026-09-11
>
> 所屬專案：isaac-sim-digital-twin
>
> 關聯 GitHub：https://github.com/GuanPersonalDev/isaac-sim-digital-twin/issues/245
>
> 前身：#232（core 於 PR #244 完成並關閉）；本文件承接原 #232 驗收 3。
>
> 設計依據銜接：[tech-design-232-shot-angle-restore.md](tech-design-232-shot-angle-restore.md) §4.5／§6。

---

## 1. 功能概述

在 `SHOT_ANGLE = (-180, 180)` 已合入（#244）後，把 PPO actor 的 `init_std` 從為 ±30° 調校的 `0.4` 改為整圈下可維持足夠瞄準解析度的值，並由使用者在 RunPod／本機完成重訓與指標驗收。

一句話定義：

> **以冷啟動調小 `init_std`（0.067）為主路徑，使探索半寬約 ±12°、命中質量比對齊 Milestone A ~17.2%；cfg／註解／訓練紀錄文件進 repo，實際訓練曲線由使用者跑完後對照 A 第三輪基準關 #245。**

---

## 2. 設計決策（階段 1–4 已確認）

| 項目 | 決策 |
|---|---|
| 本輪模式 | **Round B**：設計＋cfg PR＋使用者訓練驗收才關 Issue |
| 主路徑 | 調小 `init_std`（方案 1）；**不做**熱啟動腳本（方案 2 僅備援敘述） |
| `init_std` | **`0.067`**（180°×0.067≈12°；2.062/12≈17.2%） |
| 排除 | 收窄 `CUE_BALL_PLACEMENT_X`；改 `SHOT_ANGLE`／decoder／`manual_shot_bounds` |
| 其他超參 | **不動** `lr`／`desired_kl`／網路／MaskedPPO 配套（沿用 A 第三輪成功組） |
| 新 port／新 runtime 類別 | **不新增** |
| 訓練紀錄文件 | `docs/issue-245-training-runs.md`（結構對齊 #124；先寫計畫與基準表） |

---

## 3. 資料流與依賴

```text
PPORunnerCfg.actor.distribution_cfg.init_std = 0.067
        │
        ▼
Gaussian policy（正規化域）探索半寬 ≈ ±0.067
        │  × ACTION_HALF_SPAN[shot_angle] = 180°
        ▼
物理角探索半寬 ≈ ±12°
        │
        ▼
decode_rl_action（#244 已就緒）→ SHOT_ANGLE ∈ [-180, 180)
        │
        ▼
MaskedPPO / BilliardEnv 訓練
        │
        ▼
logs：break_foul、spread×20、mean reward、action std、lr
        │
        ▼
對照 docs/issue-124-training-runs.md 第三輪基準 → 關 #245
```

**依賴：**

- `rl_task/.../agents/rsl_rl_ppo_cfg.py`（唯一程式改動點）
- `core/models/action_bounds.py`（只讀；`SHOT_ANGLE` 已是整圈）
- `docs/issue-124-training-runs.md`（基準參考，不改內容除非需交叉連結）

---

## 4. 模組改動清單

### 4.1 `rsl_rl_ppo_cfg.py`

- `init_std`：`0.4` → **`0.067`**
- 註解：刪「配 Milestone A ±30 的 0.4」現況語氣；寫明
  - 整圈半跨 180°、`0.067` → ±12°、命中質量比 ~17.2%（對齊 A）
  - 指向 #245
  - 一句：若短訓仍學不動，再考慮熱啟動（角度維 ÷6）或另開備援，**本輪不實作**

### 4.2 `docs/issue-245-training-runs.md`（新建）

對齊 #124 風格，初版至少含：

1. 環境假設（RunPod／GPU／`num_envs`／`max_iterations` 若沿用則註明）
2. 本輪 cfg 差異表（相對 A 第三輪：僅 `SHOT_ANGLE` 整圈 + `init_std=0.067`）
3. Milestone A 第三輪基準摘要（從 #124 引用關鍵列）
4. 待填結果表（iter／break_foul／spread×20／mean reward／lr／action std）
5. 判定慣例（沿用 #124：`break_foul` 不可當退步訊號等）

### 4.3 可選短記

- `docs/CHANGELOG.md` 一行指向 #245／本設計文件（實作 PR 時可加）

### 4.4 明確不做

- 熱啟動權重腳本、`CUE_BALL_PLACEMENT_X`、core 契約、extension／GUI
- 在 bot 電腦上跑 Isaac Sim／RL

---

## 5. 驗收標準（對齊 #245）

| # | 標準 | 誰 |
|---|---|---|
| 1 | 選定並記錄策略：`init_std=0.067` 冷啟動（本文件 + cfg 註解） | 設計／PR |
| 2 | 重訓後 `break_foul`／`spread` **不低於** Milestone A 收斂水準 | 使用者／pod |
| 3 | 訓練紀錄寫入 `docs/issue-245-training-runs.md`（結果表填完） | 使用者為主；可授權補 PR |
| 4 | #245 關閉前留言標策略與關鍵指標對照 | 使用者 |

**A 第三輪參考錨（詳見 #124，非逐字複製為硬門檻）：**  
`spread×20` 高峰約 0.7+ 且守住、`mean reward` 轉正、`break_foul` 降至接近 0（表示瞄對球而非退步）。

---

## 6. 風險

- `0.067` 過窄導致探索不足 → 短訓探路；必要時改試 `0.1` 或啟用熱啟動備援（新工作）
- 舊 checkpoint 在 #244 後已失效；本輪冷啟動不依賴它們
- adaptive lr 行為見 #124（起始 lr 可能被排程放大）——本輪不改排程，只觀察

---

## 7. 實作順序建議

1. 合入本技術文件（docs-only 或與 cfg 同 PR，依授權）
2. 改 `init_std` + 註解；新建 `issue-245-training-runs.md` 骨架
3. PR（建議標題含 `#245`）
4. 使用者上 pod 訓練 → 填結果表 → 對照基準 → 關 #245

---

## 8. 已確認決策 checklist

- [x] 階段 1：Round B（含訓練驗收）
- [x] 階段 2：改 `rsl_rl_ppo_cfg`；無新 core port
- [x] 階段 3：主路徑調 `init_std`；熱啟動備援；排除擺位收窄
- [x] 階段 4：`0.067`；不建熱啟動腳本；`issue-245-training-runs.md`；完整註解；其他超參不動
