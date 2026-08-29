"""
scripts/search_roll_for_full_swing.py — 用「局部延續」的數值 IK（不是隨機
起點）搜尋 roll 值，讓 AIM 目標→後擺→隨揮終點整條軌跡都收斂在同一個關節
分支內，不撞關節限位。

背景：scripts/diagnose_strike_followthrough.py 用真實物理模擬證實
STRIKE 隨揮終點卡住的確切機制是——`ArticulationAPIImpl._step_motion()` 的
差動 IK（DLS）是局部方法，只會沿著 AIM 階段收斂到的那個關節分支繼續走；
即使隨揮終點本身在「別的關節分支」是可達的（scripts/search_backswing_ik.py
用隨機起點證實過），差動 IK 也無法跳過去，會被沿途頂到的 shoulder_pitch
限位卡死。

跟 scripts/build_roll_lookup_table.py 的差異：那支腳本只用隨機起點測試
「AIM 目標本身」的可達性，這支腳本額外模擬「同一個關節分支能不能連續走完
AIM→後擺→隨揮終點三段」——用 `wam7_kinematics.solve_ik()` 串接：AIM 解
當後擺的起點、後擺解當隨揮終點的起點（完全模仿真實差動 IK 不會跳分支的
行為），確認整條軌跡三段都收斂且沒有任何關節被逼到限位（margin 太小）。

跑法（不需要 Isaac Sim，純 numpy）：
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_roll_for_full_swing.py
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
from core.services.base_placement_calculator import compute_base_pose, CANONICAL_REST_JOINTS  # noqa: E402
from core.services.cue_pose_calculator import compute_tilted_wrist_pose, compute_tilted_direction  # noqa: E402
from core.services.swing_trajectory_calculator import (  # noqa: E402
    compute_backswing_position, compute_follow_through_distance, compute_required_tip_speed,
)

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_BACKSWING_DISTANCE_M = 0.15
_CUE_BALL_SPEED = 1.995

_CASES = [
    (0.0, -1.15),
    (0.0, -0.9382125),
    (0.0, -0.7),
    (0.0, -0.635),
]

_ROLL_CANDIDATES_DEG = list(range(-180, 180, 15))
_MIN_HEALTHY_MARGIN = 0.05  # rad，低於這個門檻視為「頂到限位」


def _margin_vector(joints: np.ndarray) -> np.ndarray:
    lowers = np.array([lo for lo, hi in wk.JOINT_LIMITS])
    uppers = np.array([hi for lo, hi in wk.JOINT_LIMITS])
    return np.minimum(joints - lowers, uppers - joints)


def try_roll(cue_ball_xy, roll_deg):
    shot_angle_deg = 0.0
    roll_rad = math.radians(roll_deg)
    base_position, base_yaw_rad = compute_base_pose(cue_ball_xy[0], cue_ball_xy[1], shot_angle_deg, _TABLE_Z, _BALL_RADIUS)
    wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
        cue_ball_xy, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
    )
    if tilt_rad is None:
        return None
    direction_unit = compute_tilted_direction(shot_angle_deg, tilt_rad)

    required_tip_speed = compute_required_tip_speed(_CUE_BALL_SPEED)
    follow_through_distance = compute_follow_through_distance(required_tip_speed)
    backswing_position = compute_backswing_position(wrist, direction_unit, _BACKSWING_DISTANCE_M)
    follow_through_position = wrist + follow_through_distance * direction_unit

    # Phase 0 起點：CANONICAL_REST_JOINTS（base_yaw=0），跟真實 Phase 0 一致。
    start = np.array([0.0, *CANONICAL_REST_JOINTS])

    aim_joints, aim_ok, aim_pos_err, aim_orient_err = wk.solve_ik(
        wrist, orientation, start, base_position=base_position, max_iters=250
    )
    if not aim_ok:
        return {"stage": "AIM", "ok": False, "pos_err": aim_pos_err, "orient_err": aim_orient_err}

    backswing_joints, bs_ok, bs_pos_err, bs_orient_err = wk.solve_ik(
        backswing_position, orientation, aim_joints, base_position=base_position, max_iters=250
    )
    if not bs_ok:
        return {"stage": "BACKSWING", "ok": False, "pos_err": bs_pos_err, "orient_err": bs_orient_err,
                "aim_margin": _margin_vector(aim_joints)}

    ft_joints, ft_ok, ft_pos_err, ft_orient_err = wk.solve_ik(
        follow_through_position, orientation, backswing_joints, base_position=base_position, max_iters=250
    )
    margins = {
        "aim": _margin_vector(aim_joints),
        "backswing": _margin_vector(backswing_joints),
        "follow_through": _margin_vector(ft_joints),
    }
    min_margin = min(float(np.min(m)) for m in margins.values())
    return {
        "stage": "FOLLOW_THROUGH", "ok": ft_ok, "pos_err": ft_pos_err, "orient_err": ft_orient_err,
        "margins": margins, "min_margin": min_margin,
    }


def main():
    names = ["base_yaw", "shoulder_pitch", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "palm_yaw"]

    for cue_ball_xy in _CASES:
        print("=" * 100)
        print(f"[{cue_ball_xy}]")
        best = None
        for roll_deg in _ROLL_CANDIDATES_DEG:
            print(f"  ...trying roll={roll_deg}", flush=True)
            result = try_roll(cue_ball_xy, roll_deg)
            if result is None:
                print("  純幾何無解，跳過整個案例")
                break
            if result["stage"] != "FOLLOW_THROUGH" or not result["ok"]:
                continue
            min_margin = result["min_margin"]
            print(f"  roll={roll_deg:+4d}deg  全程收斂  min_margin={min_margin:.4f}")
            if best is None or min_margin > best[1]:
                best = (roll_deg, min_margin, result)
        if best is None:
            print("  -> 沒有任何 roll 讓 AIM->後擺->隨揮終點全程收斂在同一分支（局部 DLS 無解）")
        else:
            roll_deg, min_margin, result = best
            print(f"  => 最佳 roll={roll_deg}deg  全程最小關節餘裕={min_margin:.4f}")
            for label, m in result["margins"].items():
                worst_idx = int(np.argmin(m))
                print(f"     {label} 最小餘裕: {names[worst_idx]}={m[worst_idx]:.4f}")


if __name__ == "__main__":
    main()
