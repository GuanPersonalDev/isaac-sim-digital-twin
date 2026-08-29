"""
scripts/build_roll_lookup_table.py — 用 scripts/search_ik_reachability.py 的
roll 掃描機制，對 verify_swing_trajectory.py 實際測試的 3x3 Kitchen 位置網格
（`action_bounds.CUE_BALL_PLACEMENT_X/Y` 的兩端＋中點）重新搜尋 roll 值，
取代 `cue_pose_calculator._ROLL_LOOKUP_GRID`。

背景：`scripts/search_ik_reachability.py` 對 3 個代表點（兩個角落＋中線遠點）
的實測發現，舊查表的 roll 值（0°/15°/45° 這種小角度）會逼 shoulder_pitch／
wrist_pitch／palm_yaw 同時頂死限位，這正是先前 20 案例 STRIKE 全滅的根因；
改用掃描找出的 roll（落在 ±120°~180° 附近）之後，所有七軸關節都有 >0.15rad
的健康餘裕（不只是最終誤差小，是真的有裕度）。

⚠️ 這裡只驗證「數值 IK 在忽略碰撞下可達」，不驗證「C1 旋轉過程中手臂本體
會不會撞庫邊/袋口」——舊查表的 roll 選擇邏輯本來就是為了閃避這個碰撞，新
roll 值必須回真實物理模擬（scripts/search_canonical_pose_candidates.py 的
bridge 模式，或直接跑 move_through_poses）重新確認無碰撞，才能正式取代
_ROLL_LOOKUP_GRID。這支腳本的輸出是「下一步該去物理模擬驗證哪些候選」，
不是可以直接上線的最終答案。

跑法（不需要 Isaac Sim，純 numpy）：
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/build_roll_lookup_table.py
"""

import math
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from search_ik_reachability import sweep_roll  # noqa: E402
from core.models.action_bounds import CUE_BALL_PLACEMENT_X, CUE_BALL_PLACEMENT_Y  # noqa: E402

_RNG_SEED = 1

_X_CANDIDATES = (CUE_BALL_PLACEMENT_X[0], sum(CUE_BALL_PLACEMENT_X) / 2, CUE_BALL_PLACEMENT_X[1])
_Y_CANDIDATES = (CUE_BALL_PLACEMENT_Y[0], sum(CUE_BALL_PLACEMENT_Y) / 2, CUE_BALL_PLACEMENT_Y[1])


def main():
    rng = np.random.default_rng(_RNG_SEED)
    new_grid = []

    for x in _X_CANDIDATES:
        for y in _Y_CANDIDATES:
            case_name = f"({x:.6f},{y:.6f})"
            print("=" * 100)
            results = sweep_roll(case_name, (x, y), 0.0, rng, roll_steps_deg=15, starts_per_roll=50, max_iters=150)
            if not results:
                print(f"  -> ({x:.6f},{y:.6f}) 純幾何無解（None），跳過，不列入新查表。")
                continue
            top_roll_deg, top_converged, _top_best = results[0]
            print(f"  => 建議 roll={top_roll_deg}deg（收斂 {top_converged}/50）")
            new_grid.append((round(x, 6), round(y, 6), top_roll_deg))

    print("=" * 100)
    print("建議的新 _ROLL_LOOKUP_GRID（尚未驗證碰撞，需回物理模擬確認才能上線）：")
    print("_ROLL_LOOKUP_GRID = [")
    for x, y, roll_deg in new_grid:
        print(f"    ({x}, {y}, {roll_deg}),")
    print("]")


if __name__ == "__main__":
    main()
