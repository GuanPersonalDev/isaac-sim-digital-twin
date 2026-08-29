"""
scripts/search_ik_reachability.py — 用 scripts/wam7_kinematics.py 的純數值
FK/IK（不跑 Isaac Sim 物理模擬）大量測試 Kitchen 邊界目標姿態是否「在關節
限位內、忽略碰撞」可達。

目的：docs/issue-180-reachability-analysis.md 第十四節記錄的手動/半系統化
試誤（30+ 組候選，每組要跑 1-2 分鐘真實物理模擬）已經證實窮舉不完，也一直
沒有定論「這些目標姿態本質上是否可達」。這支工具用數值 IK 從幾百組隨機起始
關節組合去解同一個世界座標目標，秒級跑完，用來回答兩個問題：

  1. 忽略碰撞、只看關節限位，這個目標姿態有沒有 ANY 解？
     - 如果連數值 IK（沒有碰撞、沒有物理時間步限制、可以从任意起點收斂）
       都找不到解，代表目標本身在目前的 CUE_STICK_GRIP_TO_TIP／wrist 幾何
       下是真正不可達，就要往「重新設計高架橋幾何」的方向走，不用再試
       CANONICAL_REST_JOINTS 候選了。
     - 如果數值 IK 找得到解，把收斂到的關節組合印出來，可以直接拿去當
       scripts/search_canonical_pose_candidates.py 或
       scripts/search_backswing_distance.py 的新候選，帶著碰撞/軌跡再驗證。
  2. 找得到解時，哪些關節傾向頂到限位？—— 用來判斷要放寬搜尋範圍還是
     根本要換路線設計。

跑法（不需要 Isaac Sim，純 numpy）：
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_ik_reachability.py
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
from core.services.cue_pose_calculator import compute_tilted_wrist_pose, lookup_roll_rad  # noqa: E402

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575

# 涵蓋 action_bounds.CUE_BALL_PLACEMENT_X/Y 的代表點：兩個角落（最嚴苛，
# tilt≈29.61°）＋中線兩點（次嚴苛／最寬鬆）。跟 verify_swing_trajectory.py
# 的候選點取法一致。
_CASES = [
    ("far_left_corner", (-0.606425, -0.635), 0.0),
    ("far_right_corner", (0.606425, -0.635), 0.0),
    ("mid_far", (0.0, -0.9382125), 0.0),
    ("mid_nearest", (0.0, -1.241425), 0.0),
]

_NUM_RANDOM_STARTS = 400
_RNG_SEED = 0


def _build_target(cue_ball_xy, shot_angle_deg):
    base_position, base_yaw_rad = compute_base_pose(
        cue_ball_xy[0], cue_ball_xy[1], shot_angle_deg, _TABLE_Z, _BALL_RADIUS
    )
    roll_rad = lookup_roll_rad(cue_ball_xy)
    wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
        cue_ball_xy, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
    )
    return base_position, base_yaw_rad, roll_rad, wrist, orientation, tilt_rad


def _random_joints(rng: np.random.Generator) -> np.ndarray:
    lowers = np.array([lo for lo, hi in wk.JOINT_LIMITS])
    uppers = np.array([hi for lo, hi in wk.JOINT_LIMITS])
    return lowers + rng.random(wk.NUM_JOINTS) * (uppers - lowers)


def _margin_report(joints: np.ndarray) -> str:
    parts = []
    names = ["base_yaw", "shoulder_pitch", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "palm_yaw"]
    for name, angle, (lo, hi) in zip(names, joints, wk.JOINT_LIMITS):
        margin = min(angle - lo, hi - angle)
        flag = " <-- 貼限位" if margin < 0.05 else ""
        parts.append(f"{name}={angle:+.3f}(margin={margin:.3f}{flag})")
    return "  ".join(parts)


def _build_target_with_roll(cue_ball_xy, shot_angle_deg, roll_rad):
    base_position, base_yaw_rad = compute_base_pose(
        cue_ball_xy[0], cue_ball_xy[1], shot_angle_deg, _TABLE_Z, _BALL_RADIUS
    )
    wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
        cue_ball_xy, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
    )
    return base_position, base_yaw_rad, wrist, orientation, tilt_rad


def sweep_roll(case_name, cue_ball_xy, shot_angle_deg, rng, roll_steps_deg=15, starts_per_roll=60, max_iters=150):
    """在 [-180,165] 每 `roll_steps_deg` 度掃一次，對每個 roll 值跑
    `starts_per_roll` 組隨機起點，回報每個 roll 的收斂率——用來找出比 9 點
    查表更準確的 roll 值，或確認整個角落案例不論 roll 怎麼選都不可達。"""
    roll_candidates_deg = list(range(-180, 180, roll_steps_deg))
    results = []
    for roll_deg in roll_candidates_deg:
        roll_rad = math.radians(roll_deg)
        base_position, _base_yaw_rad, wrist, orientation, tilt_rad = _build_target_with_roll(
            cue_ball_xy, shot_angle_deg, roll_rad
        )
        if tilt_rad is None:
            continue
        converged_count = 0
        best_score = float("inf")
        best = None
        for _ in range(starts_per_roll):
            start = _random_joints(rng)
            final_joints, converged, pos_err, orient_err = wk.solve_ik(
                wrist, orientation, start, base_position=base_position, max_iters=max_iters
            )
            if converged:
                converged_count += 1
            score = pos_err + orient_err
            if score < best_score:
                best_score = score
                best = (pos_err, orient_err, final_joints)
        results.append((roll_deg, converged_count, best))

    results.sort(key=lambda r: (-r[1], r[2][0] + r[2][1]))
    print(f"  [roll 掃描] {case_name}：由收斂率高到低排序（前 6 名）")
    for roll_deg, converged_count, best in results[:6]:
        pos_err, orient_err, joints = best
        print(f"    roll={roll_deg:+4d}deg  收斂 {converged_count}/{starts_per_roll}  最佳 pos_err={pos_err:.5f} orient_err={orient_err:.5f}")
    if results:
        top_roll_deg, top_converged, top_best = results[0]
        _pos_err, _orient_err, top_joints = top_best
        print(f"  [roll 掃描] 最佳 roll={top_roll_deg}deg 的完整關節餘裕：{_margin_report(top_joints)}")
    return results


def main():
    rng = np.random.default_rng(_RNG_SEED)

    for case_name, cue_ball_xy, shot_angle_deg in _CASES:
        base_position, base_yaw_rad, roll_rad, wrist, orientation, tilt_rad = _build_target(
            cue_ball_xy, shot_angle_deg
        )
        print("=" * 100)
        print(f"[{case_name}] cue_ball={cue_ball_xy}")

        if tilt_rad is None:
            print("  -> compute_required_tilt_rad() 回傳 None：純幾何無解（跟關節限位無關），跳過。")
            continue

        print(f"  tilt_rad={tilt_rad:.4f} ({math.degrees(tilt_rad):.2f} deg)  roll_rad(查表值)={roll_rad:.4f}")
        print(f"  target world position={wrist.tolist()}  orientation(wxyz)={orientation.tolist()}")
        print(f"  base_position={base_position}  base_yaw_rad(Phase0 joint value)={base_yaw_rad:.4f}")

        best = None  # (pos_err, orient_err, joints, converged)
        converged_count = 0
        for _ in range(_NUM_RANDOM_STARTS):
            start = _random_joints(rng)
            final_joints, converged, pos_err, orient_err = wk.solve_ik(
                wrist, orientation, start, base_position=base_position
            )
            if converged:
                converged_count += 1
            score = pos_err + orient_err
            if best is None or score < best[0] + best[1]:
                best = (pos_err, orient_err, final_joints, converged)

        pos_err, orient_err, joints, converged = best
        print(f"  隨機起點測試：{_NUM_RANDOM_STARTS} 組中 {converged_count} 組收斂（pos<3mm 且 orient<0.015rad）")
        print(f"  最佳解：pos_err={pos_err:.5f}m  orient_err={orient_err:.5f}rad  收斂={converged}")
        print(f"  最佳解關節角：{_margin_report(joints)}")

        if not converged_count:
            # 額外用「以 CANONICAL_REST_JOINTS 為起點」再試一次，確認不是
            # 純隨機取樣運氣不好——這組起點在真實物理模擬裡至少走得到
            # Phase 0，是最有參考價值的對照點。
            canonical_start = np.array([base_yaw_rad, 1.9, 0.0, 1.8, 0.0, -0.5585, 1.5010])
            final_joints, converged2, pos_err2, orient_err2 = wk.solve_ik(
                wrist, orientation, canonical_start, base_position=base_position, max_iters=1000
            )
            print(f"  [對照] 以 CANONICAL_REST_JOINTS 為起點：收斂={converged2}  pos_err={pos_err2:.5f}  orient_err={orient_err2:.5f}")
            print(f"  [對照] 最終關節角：{_margin_report(final_joints)}")

        # 不論查表 roll 有沒有收斂，都掃一次 roll，確認查表值是不是最佳選擇
        # （far_left_corner 雖然查表 roll=45° 收斂，但 shoulder_pitch margin
        # 只有 0.003 rad，貼著限位走，掃描看看有沒有餘裕更大的替代 roll）。
        sweep_roll(case_name, cue_ball_xy, shot_angle_deg, rng)

    print("=" * 100)
    print("完成。若某案例 400 組隨機起點全部不收斂，代表目標姿態在目前 wrist/cue 幾何下")
    print("『不論用哪組 CANONICAL_REST_JOINTS 候選、不論怎麼設計中繼路徑』都無法用關節角度")
    print("精確抵達（這是必要條件的否定，比任何有限候選搜尋都更有結論性）。")


if __name__ == "__main__":
    main()
