"""
scripts/search_backswing_ik.py — 用數值 IK（不跑物理模擬）檢查 STRIKE 階段
「後擺」目標姿態（`swing_trajectory_calculator.compute_backswing_position()`）
在新 roll 查表下是否可達，以及可達的 `backswing_distance` 上限。

背景：scripts/verify_new_roll_table.py 用真實物理模擬證實新 roll 查表把
AIM 從 0/20 修到 5/6 真正收斂，但接著 STRIKE 全部逾時（`DEFAULT_BACKSWING_
DISTANCE_M=0.15`）。這支腳本沿用 scripts/wam7_kinematics.py 的快速 IK，
對每個案例的「後擺」目標姿態（跟 AIM 終點方向相反、沿桿身軸退開
backswing_distance）做隨機起點掃描＋逐步縮小 backswing_distance，找出
實際可行的距離上限，取代物理模擬慢速試誤。

跑法（不需要 Isaac Sim，純 numpy）：
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_backswing_ik.py
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
from core.services.cue_pose_calculator import compute_tilted_wrist_pose, compute_tilted_direction  # noqa: E402
from core.services.swing_trajectory_calculator import compute_backswing_position  # noqa: E402

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_NUM_RANDOM_STARTS = 150
_RNG_SEED = 2

# scripts/verify_new_roll_table.py 實測 AIM_OK 的 5 個案例（roll 已修正）。
_AIM_OK_CASES = [
    ((-0.606425, -0.9382125), 165),
    ((-0.606425, -0.635), -165),
    ((0.0, -0.635), 150),
    ((0.606425, -0.9382125), -135),
    ((0.606425, -0.635), -165),
]

_BACKSWING_CANDIDATES_M = (0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.0)


def _random_joints(rng: np.random.Generator) -> np.ndarray:
    lowers = np.array([lo for lo, hi in wk.JOINT_LIMITS])
    uppers = np.array([hi for lo, hi in wk.JOINT_LIMITS])
    return lowers + rng.random(wk.NUM_JOINTS) * (uppers - lowers)


def _best_ik(target_position, target_orientation, base_position, rng, num_starts=_NUM_RANDOM_STARTS):
    best = None
    converged_count = 0
    for _ in range(num_starts):
        start = _random_joints(rng)
        joints, converged, pos_err, orient_err = wk.solve_ik(
            target_position, target_orientation, start, base_position=base_position
        )
        if converged:
            converged_count += 1
        score = pos_err + orient_err
        if best is None or score < best[0] + best[1]:
            best = (pos_err, orient_err, joints, converged)
    return best, converged_count


def main():
    rng = np.random.default_rng(_RNG_SEED)

    for cue_ball_xy, roll_deg in _AIM_OK_CASES:
        shot_angle_deg = 0.0
        roll_rad = math.radians(roll_deg)
        base_position, _base_yaw_rad = compute_base_pose(cue_ball_xy[0], cue_ball_xy[1], shot_angle_deg, _TABLE_Z, _BALL_RADIUS)
        wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
            cue_ball_xy, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
        )
        direction_unit = compute_tilted_direction(shot_angle_deg, tilt_rad)

        print("=" * 100)
        print(f"[{cue_ball_xy}] roll={roll_deg}deg  contact_position={wrist.tolist()}  direction={direction_unit.tolist()}")

        # 對照組：接觸點本身（backswing_distance=0）以外，contact_position
        # 先確認可達，當基準線。
        best, converged_count = _best_ik(wrist, orientation, base_position, rng)
        pos_err, orient_err, joints, converged = best
        print(f"  [基準:接觸點] 收斂 {converged_count}/{_NUM_RANDOM_STARTS}  最佳 pos_err={pos_err:.5f} orient_err={orient_err:.5f}")

        for backswing_distance in _BACKSWING_CANDIDATES_M:
            backswing_position = compute_backswing_position(wrist, direction_unit, backswing_distance)
            best, converged_count = _best_ik(backswing_position, orientation, base_position, rng)
            pos_err, orient_err, joints, converged = best
            status = "OK" if converged_count > 0 else "FAIL"
            print(f"  backswing_distance={backswing_distance:.2f}m  收斂 {converged_count}/{_NUM_RANDOM_STARTS}  最佳 pos_err={pos_err:.5f} orient_err={orient_err:.5f}  [{status}]")
            if status == "OK":
                names = ["base_yaw", "shoulder_pitch", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "palm_yaw"]
                margins = "  ".join(
                    f"{n}=(margin={min(a-lo,hi-a):.3f})" for n, a, (lo, hi) in zip(names, joints, wk.JOINT_LIMITS)
                )
                print(f"    關節餘裕：{margins}")
                break


if __name__ == "__main__":
    main()
