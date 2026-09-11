# #245 RL 訓練紀錄（#232-train：整圈 SHOT_ANGLE × `init_std=0.067`）

**狀態：計畫／基準摘要 only。** 結果表待使用者在 pod 跑完後填入；本文件此時不構成 #245 驗收。

#244 已把 `SHOT_ANGLE` 復原為 `(-180, 180)`。本輪只改 PPO actor 的 `init_std`（`0.4` → `0.067`），冷啟動重訓，對照 Milestone A 第三輪基準。設計與選定理由見 [tech-design-245-init-std.md](tech-design-245-init-std.md)；A 第三輪完整曲線見 [issue-124-training-runs.md](issue-124-training-runs.md)。

---

## 環境假設

尚未實際開跑。若沿用 #124 的 pod 設定，假設如下（**標成假設，跑前請核對 log**）：

| 項目 | 假設值 | 備註 |
|---|---|---|
| 平台 | RunPod、1× RTX 4090 | 同 #124 |
| `num_envs` | 1024 | 同 #124 |
| `max_iterations` | 1000 | cfg 現值；#124 第三輪在 it 224 手動停 |
| 耗時 | 約 9.2 s/iteration（整輪約 2.6 小時） | 沿用 #124 量測，未重測 |

判讀慣例（沿用 #124）：TensorBoard 的 `Episode_Reward/*` 是「每局累加 ÷ `max_episode_length_s`(20)」，**本文一律標成 `×20`（＝每局實際值）**。

---

## 相對 Milestone A 第三輪的 cfg 差異

A 第三輪：commit `92ade53`｜`lr=3.0e-4`、`desired_kl=0.02`｜`SHOT_ANGLE = (-30, 30)`｜`init_std=0.4`。

| 項目 | A 第三輪 | 本輪（#245） |
|---|---|---|
| `SHOT_ANGLE` | `(-30, 30)` | `(-180, 180)`（#244） |
| `init_std` | 0.4 | **0.067**（180°×0.067≈±12° 探索半寬） |
| `lr` | 3.0e-4 | 不變 |
| `desired_kl` | 0.02 | 不變 |
| 網路 | `[256, 128, 64]` | 不變 |
| MaskedPPO 配套（`value_loss_coef`／`entropy_coef`） | 0.1／0.0005 | 不變 |
| `gamma`／`lam` | 1.0／1.0 | 不變 |
| `CUE_BALL_PLACEMENT_X` | 未收窄 | **不收窄** |
| 熱啟動 | — | **不實作**（角度維 ÷6 僅備援敘述） |

一句話：契約變整圈之後，用更小的 `init_std` 把瞄準解析度拉回 A 期間的命中質量比（~17.2%）；其餘 PPO 超參沿用 A 第三輪成功組。

---

## Milestone A 第三輪基準摘要（引自 #124）

log：`2026-08-11_07-22-55`，跑到 it 224 後手動停止。**不是逐字硬門檻**，是對照錨。

| iter | `break_foul` | `aim` ×20 | `spread` ×20 | `mean reward` | lr |
|---|---|---|---|---|---|
| 0 | 0.077 | 0.293 | 0.017 | −1.527 | 3.42e-3 |
| 56 | 0.065 | 0.398 | **0.820** | +0.029 | 2.6e-4 |
| 112 | 0.036 | 0.398 | 0.762 | +0.365 | 1.7e-4 |
| 168 | 0.039 | 0.398 | 0.771 | **+0.437** | 5e-5 |
| 223 | **0.005** | 0.398 | 0.722 | +0.115 | 3e-5 |

關鍵錨（#245 驗收「不低於 A 收斂水準」時對這三項）：

- `spread×20` 高峰約 **0.7+** 且守住（it 223 仍 0.722）
- `mean reward` **由負轉正**（峰值 +0.437）
- `break_foul` 降至接近 **0**（it 223＝0.005）——這是瞄對 1 號球的證據，不是退步

---

## 本輪結果（待填）

跑完後把 TensorBoard／`check_training.py` 對應列填進來。空欄表示尚未開跑。

| iter | `break_foul` | `aim` ×20 | `spread` ×20 | `mean reward` | `Policy/mean_std` | lr |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

補充（可選）：log 目錄、commit、實際 `num_envs`／GPU、停在第幾個 iteration、是否手動停止。

---

## 判定慣例（沿用 #124）

- **`break_foul` 不能拿來判斷退步。** 該項數的是「碰到錯球」；學會瞄 1 號球之後本來就該歸零。A 第三輪 0.077 → 0.005 是成功訊號。
- **一律看 `×20`。** TensorBoard 的 `Episode_Reward/*` 已除以 `max_episode_length_s`(20)；跟 #124 第三輪數字比時兩邊都要用每局實際值。
- **lr 崩到 1e-5 有兩種完全相反的意思**：在壞位置凍結（#124 第二輪）或單純收斂（第三輪）。差別只看 `aim`／`spread`／`mean reward` 守不守得住峰值，不能單看 lr。
- 地形問題：`aim` 從頭到尾沒動（第一輪）。學起來又丟掉：`aim`／`spread` 從峰值跌超過 40%（第二輪）。
- 舊 Milestone A checkpoint 在 #244 後已失效；本輪冷啟動，不拿來對曲線。

判定腳本（沿用）：

```bash
/workspace/venv/bin/python /workspace/isaac-sim-digital-twin/training/scripts/check_training.py
```

---

## 待辦

| 項目 | 說明 |
|---|---|
| pod 重訓 | 冷啟動、`init_std=0.067`；填上方結果表 |
| 對照基準 | `spread×20`、`mean reward`、`break_foul` 對 A 第三輪錨 |
| 關 #245 | 使用者留言標策略與關鍵指標後再關；**本 cfg／docs PR 不關閉 Issue** |
| 備援（若短訓學不動） | 試 `init_std=0.1`，或熱啟動（角度維權重／bias ÷6）——另開工作，本輪不實作 |
