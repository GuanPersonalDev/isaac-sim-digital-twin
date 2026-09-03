"""
scripts/search_backswing_distance_ik.py — 用數值 IK（不跑物理模擬）反推每個
高架橋案例的後擺距離上限：沿擊球反方向從 contact pose 逐步退開，找到「還能
收斂、且離關節硬限位有安全餘裕」的最大距離。

背景：docs/issue-180-reachability-analysis.md 第十六節量到，`DEFAULT_
BACKSWING_DISTANCE_M=0.15` 這個寫死常數跟關節實際能提供的加速能力完全脫鉤，
懷疑是揮桿速度嚴重不足（實測只達目標 55%）的部分原因。這支腳本用 IK 可達性
邊界法：假設 PhysX 速度模式的關節驅動器夠快，後擺距離只需要給幾何/IK 留
餘裕，不用模擬從靜止加速到最大轉速需要的時間/距離。

⚠️ 2026-09-01：**基座位置維持 `compute_base_pose()` 的公式值，不搜尋偏移**。
中間試過「把基座水平位移也當自由變數搜尋」（v2/v3），純 IK 可達性分析找到
的偏移（例如讓後擺距離打到搜尋上限）用真實 Isaac Sim headless
（`diagnose_move_swing.py`）驗證時，AIM 的差動 IK 控制迴圈（Phase 0→B1→
B2→C1→C2）反而不收斂（逾時 1000 步，揮桿打空）——純運動學可達性一次性
求解沒有模擬差動 IK 沿路徑逐步收斂的動態行為，偏移越大、路徑幾何改變越多，
踩到這個問題的風險越高。也試過「先算不偏移時的理論速度上限，只在打不到
才搜最小偏移」，但對確實有 manipulability 上限的案例（第十六節：
`y=-0.9382125` 這排任何 roll 都到不了目標）會在小偏移範圍內窮舉不到解，
每個候選都要跑多起點 IK＋LP，實測單一候選就要 2-3 分鐘，最壞情況會拖到
數小時。

改成最保守的做法：**完全不碰基座位置**（消除差動 IK 不收斂的風險），只用
這支腳本原本的方法算後擺距離。這樣才能真正確定不會引入新的收斂問題——
唯一的代價是 manipulability 上限本來就低的案例（`y=-0.9382125`）速度依然
達不到目標，這是已知、可接受的既有限制（見第十六節），不是這次任務要
解決的範圍。

跑法（不需要 Isaac Sim，純 numpy）：
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_backswing_distance_ik.py

輸出人工核對後才寫進 core/services/cue_pose_calculator.py 的
_BACKSWING_DISTANCE_LOOKUP_GRID，不自動套用；每一組數值都還要再用
scripts/diagnose_move_swing.py 這類真實 Isaac Sim headless 腳本驗證過 AIM
真的收斂、無新增碰撞、揮桿速度確實提升，才能視為定案。
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

import wam7_kinematics as wk  # noqa: E402
from core.services.base_placement_calculator import compute_base_pose  # noqa: E402
from core.services.cue_pose_calculator import (  # noqa: E402
    _ROLL_LOOKUP_GRID,
    compute_tilted_direction,
    compute_tilted_wrist_pose,
)

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_SHOT_ANGLE_DEG = 0.0
_RNG_SEED = 3

_STEP_M = 0.01
_MAX_DISTANCE_M = 0.35
# 關節餘裕安全門檻（rad）——離硬限位小於這個值就視為不可行，即使 solve_ik()
# 回報收斂。跟 docs/issue-180-reachability-analysis.md 第十三節的量級一致。
_MARGIN_THRESHOLD_RAD = 0.1

# 多分支延伸：同一個接觸點常有多組不同關節構型都收斂，不同分支能延伸的
# 後擺空間可能天差地遠，只挑「位置/姿態誤差最小」的單一種子分支容易漏掉
# 空間更大的分支（實測：篩選跟確認用不同起點數會量出差好幾倍的距離）。
_SEED_SEARCH_STARTS = 100
_MAX_DISTINCT_SEEDS = 10

_JOINT_NAMES = [
    "base_yaw", "shoulder_pitch", "shoulder_yaw", "elbow_pitch",
    "wrist_yaw", "wrist_pitch", "palm_yaw",
]


def _random_joints(rng: np.random.Generator) -> np.ndarray:
    lowers = np.array([lo for lo, hi in wk.JOINT_LIMITS])
    uppers = np.array([hi for lo, hi in wk.JOINT_LIMITS])
    return lowers + rng.random(wk.NUM_JOINTS) * (uppers - lowers)


def _distinct_converged_seeds(target_position, target_orientation, base_position, rng, num_starts, max_distinct):
    seeds = []
    for _ in range(num_starts):
        start = _random_joints(rng)
        joints, converged, _pe, _oe = wk.solve_ik(
            target_position, target_orientation, start, base_position=base_position
        )
        if not converged:
            continue
        if any(np.linalg.norm(joints - s) < 0.3 for s in seeds):
            continue
        seeds.append(joints)
        if len(seeds) >= max_distinct:
            break
    return seeds


def _min_margin(joints: np.ndarray) -> float:
    return min(min(a - lo, hi - a) for a, (lo, hi) in zip(joints, wk.JOINT_LIMITS))


def _margins_str(joints: np.ndarray) -> str:
    return "  ".join(
        f"{n}=(margin={min(a - lo, hi - a):.3f})"
        for n, a, (lo, hi) in zip(_JOINT_NAMES, joints, wk.JOINT_LIMITS)
    )


def _extend_from_seed(wrist, orientation, direction_unit, base_position, seed_joints):
    current_joints = seed_joints
    max_valid_distance = 0.0
    final_joints = None
    num_steps = int(_MAX_DISTANCE_M / _STEP_M)
    for i in range(1, num_steps + 1):
        distance = i * _STEP_M
        candidate_position = wrist - distance * direction_unit
        joints, ik_converged, _pe, _oe = wk.solve_ik(
            candidate_position, orientation, current_joints, base_position=base_position
        )
        if not ik_converged:
            break
        margin = _min_margin(joints)
        if margin < _MARGIN_THRESHOLD_RAD:
            break
        current_joints = joints
        max_valid_distance = distance
        final_joints = joints
    return max_valid_distance, final_joints


def main():
    rng = np.random.default_rng(_RNG_SEED)
    results = []

    for cue_ball_x, cue_ball_y, roll_deg in _ROLL_LOOKUP_GRID:
        cue_ball_xy = (cue_ball_x, cue_ball_y)
        roll_rad = math.radians(roll_deg)
        base_position, _base_yaw_rad = compute_base_pose(
            cue_ball_x, cue_ball_y, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS
        )
        wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
            cue_ball_xy, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
        )
        print("=" * 100, flush=True)
        if tilt_rad is None:
            print(f"[{cue_ball_xy}] roll={roll_deg}deg  幾何無解（tilt_rad=None），跳過", flush=True)
            results.append((cue_ball_x, cue_ball_y, None))
            continue
        if tilt_rad <= 1e-6:
            print(f"[{cue_ball_xy}] roll={roll_deg}deg  flat 案例（tilt_rad={tilt_rad:.4f}），不套用這套統一，跳過", flush=True)
            results.append((cue_ball_x, cue_ball_y, None))
            continue

        direction_unit = compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad)
        print(
            f"[{cue_ball_xy}] roll={roll_deg}deg  tilt_rad={tilt_rad:.4f}  "
            f"contact={wrist.tolist()}  direction={direction_unit.tolist()}",
            flush=True,
        )

        seeds = _distinct_converged_seeds(
            wrist, orientation, base_position, rng, _SEED_SEARCH_STARTS, _MAX_DISTINCT_SEEDS
        )
        print(f"  收集到 {len(seeds)} 組不同分支的收斂種子", flush=True)
        if not seeds:
            print("  [FAIL] 接觸點本身都無法收斂，跳過這個案例", flush=True)
            results.append((cue_ball_x, cue_ball_y, None))
            continue

        best_distance = -1.0
        best_final = None
        for seed_joints in seeds:
            distance, final_joints = _extend_from_seed(wrist, orientation, direction_unit, base_position, seed_joints)
            print(f"    分支延伸距離={distance:.2f}m", flush=True)
            if distance > best_distance:
                best_distance = distance
                best_final = final_joints

        print(f"  => backswing_distance_m = {best_distance:.2f}（{len(seeds)} 組分支中取最遠）", flush=True)
        if best_final is not None:
            print(f"    最終關節餘裕：{_margins_str(best_final)}", flush=True)
        results.append((cue_ball_x, cue_ball_y, best_distance))

    print("\n" + "=" * 100)
    print("彙總（貼進 cue_pose_calculator._BACKSWING_DISTANCE_LOOKUP_GRID 前人工核對；")
    print("基座位置一律用 compute_base_pose() 的公式值，不套用任何偏移）：")
    print("_BACKSWING_DISTANCE_LOOKUP_GRID = [")
    for cue_ball_x, cue_ball_y, distance in results:
        if distance is None:
            print(f"    ({cue_ball_x}, {cue_ball_y}, None, 0.0, 0.0),  # flat 或幾何無解，不use")
        else:
            print(f"    ({cue_ball_x}, {cue_ball_y}, {distance:.2f}, 0.0, 0.0),")
    print("]")


if __name__ == "__main__":
    main()
