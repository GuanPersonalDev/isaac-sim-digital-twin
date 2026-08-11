#!/usr/bin/env python3
"""讀最新一輪訓練的 TensorBoard event file，判斷該不該中斷（#124）。

    /workspace/venv/bin/python training/scripts/check_training.py

不需要進 venv 以外的環境，也不會碰 GPU，訓練跑著的時候可以隨時跑。

判定邏輯的由來
--------------
第一版只比對「後 N 個 iteration 對前 N 個」的趨勢，結果在 #124 第二輪判錯：
那一輪其實學起來了（break_foul 終止率 0.077 → 0.189、spread ×20 在 iter 57
衝到起點的 7 倍），然後又整個退回去，首尾相減看起來就像「完全沒動」。

所以這一版改成追**歷史最佳值與最新值的落差**（drawdown），並且把
learning rate 崩塌獨立成一條——那是「policy 已經凍結，繼續跑不會有任何
變化」的直接證據，優先於其他所有判斷。
"""

from __future__ import annotations

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
    ("action std", ("noise_std",), 1.0),
    ("lr", ("learning_rate",), 1.0),
    ("mean reward", ("mean_reward",), 1.0),
    ("time_out", ("Episode_Termination", "time_out"), 1.0),
]

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


def main() -> None:
    run = _latest_run()
    accumulator = EventAccumulator(run)
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]

    series: dict[str, list[float]] = {}
    steps: list[int] = []
    missing: list[str] = []
    for name, needles, scale in METRICS:
        tag = _resolve(tags, needles)
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
    for name in ("aim x20", "spread x20", "mean reward", "break_foul"):
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
    if total < DECISION_ITERATION:
        print(f"還沒到決策點（{total}/{DECISION_ITERATION}），繼續跑。")
        return

    regressed = [n for n, d in drawdowns.items() if d > DRAWDOWN_THRESHOLD]
    progressed = [
        n
        for n in ("aim x20", "spread x20")
        if n in drawdowns and drawdowns[n] <= DRAWDOWN_THRESHOLD
    ]
    peak_aim = max(_smooth(series["aim x20"], SMOOTH_WINDOW)) if "aim x20" in series else 0.0
    start_aim = _smooth(series["aim x20"], SMOOTH_WINDOW)[0] if "aim x20" in series else 0.0
    learned_something = peak_aim - start_aim > 0.01

    # lr 崩塌優先於一切：policy 已經凍結，繼續跑不會有任何變化。
    if lr_collapsed:
        print("🔴 lr 已崩到起始值的 5% 以下 → policy 凍結，繼續跑不會再變。中斷。")
        print("   下一手：num_learning_epochs 5 → 3（從源頭壓 KL），不是再調 lr。")
    elif regressed and learned_something:
        print(f"🔴 學起來過但已退步（{', '.join(regressed)}）→ 中斷。")
        print("   最佳點附近的 checkpoint 是目前最好的 policy，別刪。")
    elif not learned_something:
        print("🔴 aim 從頭到尾沒動 → 不是優化器問題，是 reward 地形。")
        print("   下一手：收窄 CUE_BALL_PLACEMENT_X（擺位變異 σ ≈ 8.8°）。")
    elif progressed:
        print("✅ 仍在最佳點附近且沒有明顯回落 → 跑完 1000。")
    else:
        print("🟡 訊號混雜，再看 100 個 iteration。")


if __name__ == "__main__":
    main()
