#!/usr/bin/env python3
"""讀一輪訓練的 TensorBoard event file：判斷該不該中斷，或匯出成 CSV（#124）。

    # 看最新一輪的現況與判定
    /workspace/venv/bin/python training/scripts/check_training.py

    # 把整份 scalar 序列匯出成 CSV（進 git 用）
    /workspace/venv/bin/python training/scripts/check_training.py \\
        --csv /workspace/run3.csv

    # 指定某一輪（預設是 mtime 最新的）
    /workspace/venv/bin/python training/scripts/check_training.py \\
        --run /workspace/training-runs/logs/rsl_rl/billiard/2026-08-11_06-36-37 \\
        --csv /workspace/run2.csv

不需要進 venv 以外的環境，也不會碰 GPU，訓練跑著的時候可以隨時跑。

為什麼要有 CSV
--------------
event file 是二進位、體積大，`training/outputs/` 因此被 gitignore。但曲線本身
是這個專案最有 diff 價值的產出之一——匯成 CSV（幾十 KB 純文字）進 git 之後，
畫圖、跨輪對照、驗證文件裡的數字都不必再開 pod。

匯出的是**原始值**，不做 ×20 之類的縮放：縮放屬於分析，存檔要忠於來源。

判定邏輯的由來
--------------
第一版只比對「後 N 個 iteration 對前 N 個」的趨勢，結果在 #124 第二輪判錯：
那一輪其實學起來了（break_foul 終止率 0.077 → 0.189、spread ×20 在 iter 57
衝到起點的 7 倍），然後又整個退回去，首尾相減看起來就像「完全沒動」。

所以改成追**歷史最佳值與最新值的落差**（drawdown）。第二版又把 learning rate
崩塌設成無條件最高優先，於是把成功的第三輪判成「中斷」——lr 低有兩種完全相反
的意思（在壞位置凍結／單純收斂），差別只看指標守不守得住，所以它不能當獨立的
判準。現行順序是：先看學到了沒 → 再看守不守得住 → lr 放最後。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


LOG_ROOT = os.environ.get(
    "BILLIARD_LOG_ROOT", "/workspace/training-runs/logs/rsl_rl/billiard"
)

# (顯示名, tag 關鍵字, 倍率)
# 倍率 20 = max_episode_length_s：Isaac Lab 的 Episode_Reward/* 是「每局累加
# ÷ max_episode_length_s」，×20 才是每局的實際值。
METRICS: list[tuple[str, tuple[str, ...], float]] = [
    ("break_foul", ("Episode_Termination", "break_foul"), 1.0),
    ("aim x20", ("Episode_Reward", "aim"), 20.0),
    ("spread x20", ("Episode_Reward", "spread"), 20.0),
    ("foul x20", ("Episode_Reward", "foul"), 20.0),
    # rsl_rl 5.0.1 記的是 `Policy/mean_std`；舊版與部分分支叫 `mean_noise_std`。
    # 兩個都試，不然這一欄會靜靜地消失（#124 第三輪就漏了它）。
    ("action std", ("mean_std",), 1.0),
    ("lr", ("learning_rate",), 1.0),
    ("mean reward", ("mean_reward",), 1.0),
    ("time_out", ("Episode_Termination", "time_out"), 1.0),
]

# 同一個顯示名可以有多組候選關鍵字，依序試。
TAG_FALLBACKS: dict[str, tuple[tuple[str, ...], ...]] = {
    "action std": (("mean_std",), ("noise_std",)),
}

# 「越高越好」的指標才適合拿來判斷退步。
#
# ⚠️ `break_foul`（碰到錯球的終止率）**故意不在這裡**：它的好壞方向會翻轉。
#    逃離「母球什麼都不碰」的階段，它上升代表開始碰到球，是好事；policy 學會
#    瞄準 1 號球之後，它下降到 0 才是好事。#124 第三輪就是這樣被誤判成
#    「跌幅 137%」的災難，實際上那是成功的證據。
PROGRESS_METRICS = ("aim x20", "spread x20", "mean reward")

# aim 塑形的滿分，用來把 reward 反推回物理距離。與
# core.services.aim_shaping_calculator 的常數對齊；這支腳本在 pod 上獨立跑，
# 不 import core 以免多拉一條相依。數值若在 core 端改了，這裡跟著改。
AIM_REWARD_SCALE = 0.4
AIM_REFERENCE_GAP = 1.9148

DECISION_ITERATION = 200
SMOOTH_WINDOW = 10
# 從歷史最佳跌掉這個比例就算「退步」。0.4 是為了跳過正常的迭代雜訊——
# #124 第二輪的 spread 從 0.117 跌到 0.002，跌幅 98%，離這條線很遠。
DRAWDOWN_THRESHOLD = 0.4
# lr 掉到起始值的這個比例以下 = adaptive 排程已經把學習踩到底。
LR_COLLAPSE_RATIO = 0.05


def _latest_run() -> str:
    runs = glob.glob(os.path.join(LOG_ROOT, "*"))
    if not runs:
        sys.exit(f"[check] {LOG_ROOT} 底下沒有任何 run")
    return max(runs, key=os.path.getmtime)


def _resolve(tags: list[str], needles: tuple[str, ...]) -> str | None:
    for tag in tags:
        low = tag.lower()
        if all(needle.lower() in low for needle in needles):
            return tag
    return None


def dump_csv(accumulator: EventAccumulator, tags: list[str], path: str) -> None:
    """把**全部** scalar tag 逐 iteration 寫成 CSV。

    刻意不篩選 tag：`METRICS` 只挑了判定要用的八項，但存檔的目的是「以後不必
    再開 pod」，少存一欄就等於少一個之後回答得了的問題。

    ⚠️ 排除結尾是 `/time` 的 tag。rsl_rl 會替部分指標多記一份以**牆鐘秒數**當
    step 的版本（例如 `Train/mean_reward/time`），混進來會把 step 聯集撐成幾千
    列全是空格。

    值一律原始值，不做 ×20 之類的縮放——縮放屬於分析，存檔要忠於來源。
    """
    keep = sorted(tag for tag in tags if not tag.endswith("/time"))
    series = {
        tag: {event.step: event.value for event in accumulator.Scalars(tag)}
        for tag in keep
    }
    steps = sorted({step for values in series.values() for step in values})

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", *keep])
        for step in steps:
            writer.writerow(
                [step, *(series[tag].get(step, "") for tag in keep)]
            )

    print(f"[check] CSV → {path}（{len(steps)} 列 × {len(keep)} 欄）")


def _smooth(values: list[float], window: int) -> list[float]:
    """滑動平均。單點雜訊會讓 peak 偏高、drawdown 誤判成退步。"""
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _aim_to_gap(aim_reward: float) -> float:
    """塑形分數 → 母球對 1 號球的最近表面間距（m）。線性映射的反函式。"""
    return AIM_REFERENCE_GAP * (1.0 - aim_reward / AIM_REWARD_SCALE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        help=f"要讀的 run 目錄。預設是 {LOG_ROOT} 底下 mtime 最新的那一個。",
    )
    parser.add_argument(
        "--csv",
        help="把全部 scalar 序列寫成 CSV（原始值，不縮放）。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run = args.run or _latest_run()
    accumulator = EventAccumulator(run)
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]

    if args.csv:
        dump_csv(accumulator, tags, args.csv)

    series: dict[str, list[float]] = {}
    steps: list[int] = []
    missing: list[str] = []
    for name, needles, scale in METRICS:
        candidates = TAG_FALLBACKS.get(name, (needles,))
        tag = next(
            (found for c in candidates if (found := _resolve(tags, c))), None
        )
        if tag is None:
            missing.append(name)
            continue
        events = accumulator.Scalars(tag)
        series[name] = [event.value * scale for event in events]
        if len(events) > len(steps):
            steps = [event.step for event in events]

    if not series:
        print("[check] 一個 tag 都對不到，實際有的是：")
        for tag in tags:
            print("   ", tag)
        sys.exit(1)
    if missing:
        print(f"[check] ⚠️ 對不到這些 tag（判定會略過）：{missing}\n")

    total = len(steps)
    print(f"run       : {run}")
    print(f"iteration : {total}\n")

    # ---- 軌跡表 ----
    marks = sorted({0, total // 4, total // 2, 3 * total // 4, total - 1})
    header = "iter " + "".join(f"{name:>13s}" for name in series)
    if "aim x20" in series:
        header += f"{'接近距離(m)':>14s}"
    print(header)
    print("-" * len(header))
    for i in marks:
        row = f"{steps[i]:4d} "
        for values in series.values():
            row += f"{values[i]:>13.5f}" if i < len(values) else f"{'-':>13s}"
        if "aim x20" in series:
            row += f"{_aim_to_gap(series['aim x20'][i]):>14.3f}"
        print(row)

    # ---- 歷史最佳 vs 最新 ----
    print(f"\n=== 歷史最佳 vs 最新（滑動平均 {SMOOTH_WINDOW}）===")
    drawdowns: dict[str, float] = {}
    for name in PROGRESS_METRICS:
        if name not in series:
            continue
        smooth = _smooth(series[name], SMOOTH_WINDOW)
        peak = max(smooth)
        peak_at = steps[smooth.index(peak)]
        last = smooth[-1]
        # 以「距離起點走了多遠」為分母，才不會被負值 reward 的符號搞亂。
        gained = peak - smooth[0]
        drop = (peak - last) / gained if gained > 1e-9 else 0.0
        drawdowns[name] = drop
        print(
            f"  {name:<12s} 起點 {smooth[0]:>9.5f}"
            f"  最佳 {peak:>9.5f} @it{peak_at:<5d}"
            f"  最新 {last:>9.5f}   跌幅 {drop * 100:>6.1f}%"
        )

    if "break_foul" in series:
        smooth = _smooth(series["break_foul"], SMOOTH_WINDOW)
        print(
            f"\n  break_foul（碰到錯球）{smooth[0]:.5f} → {smooth[-1]:.5f}"
            "　※ 不進退步判定，好壞方向會翻轉：\n"
            "     逃離「什麼都不碰」時它該上升；學會瞄 1 號球後它該降到 0。"
        )

    lr_collapsed = False
    if "lr" in series:
        lr_first, lr_last = series["lr"][0], series["lr"][-1]
        lr_collapsed = lr_last < lr_first * LR_COLLAPSE_RATIO
        print(
            f"\n  lr {lr_first:.2e} → {lr_last:.2e}"
            f"（{lr_last / lr_first * 100:.1f}%）"
            + ("   🔴 已被 adaptive 排程踩到底" if lr_collapsed else "")
        )

    # ---- 判定 ----
    print("\n=== 判定 ===")
    regressed = [n for n, d in drawdowns.items() if d > DRAWDOWN_THRESHOLD]
    aim_smooth = _smooth(series.get("aim x20", [0.0]), SMOOTH_WINDOW)
    learned_something = max(aim_smooth) - aim_smooth[0] > 0.01

    # ⚠️ 判斷順序是有意義的：先看「學到了沒」，再看「守不守得住」，
    #    **lr 放最後**。
    #
    #    lr 低有兩種完全相反的意思——policy 在壞位置凍結（#124 第二輪），
    #    或單純收斂了（第三輪）。差別只看指標守不守得住，所以 lr 不能當
    #    獨立的判準。第一版把它設成無條件最高優先，於是把成功的第三輪
    #    判成「中斷」。
    if total < DECISION_ITERATION:
        print(f"還沒到決策點（{total}/{DECISION_ITERATION}），繼續跑。")
        return

    if not learned_something:
        print("🔴 aim 從頭到尾沒動 → 不是優化器問題，是 reward 地形。")
        print("   下一手：收窄 CUE_BALL_PLACEMENT_X（擺位變異 σ ≈ 8.8°）。")
    elif regressed:
        print(f"🔴 學起來過但已退步（{', '.join(regressed)}）→ 中斷。")
        print("   最佳點附近的 checkpoint 是目前最好的 policy，別刪。")
        if lr_collapsed:
            print("   lr 也崩了 → 下一手是 num_learning_epochs 5 → 3（從源頭壓 KL）。")
    elif lr_collapsed:
        print("🟢 指標守在峰值附近，但 lr 已到底 → **已收斂**，繼續跑不會再變。")
        print("   可以收工。想再往上推的話改 num_learning_epochs 5 → 3 重跑。")
    else:
        print("✅ 仍在進步且沒有回落 → 跑完 1000。")


if __name__ == "__main__":
    main()
