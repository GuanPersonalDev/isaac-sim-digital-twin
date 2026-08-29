"""
scripts/search_roll_swing_capable.py — 三條件 roll 搜尋：(1) AIM 差動 IK
真的收得斂、(2) 揮桿方向在「姿態鎖定不變」約束下的真正最大可達速度要
≥ required_tip_speed、(3) 真實物理模擬無碰撞。前兩個條件純數值、秒級可測，
第三個條件才需要物理模擬逐點驗證。

背景：docs/issue-180-reachability-analysis.md 第十五節先前的 roll 搜尋
（scripts/search_roll_for_full_swing.py／search_collision_free_roll.py）
只驗證了 AIM 目標可達 + 無碰撞，沒有驗證「保持姿態不變的前提下，沿揮桿
方向到底能不能真的加速到 required_tip_speed」。用
scripts/prototype_moving_target_strike.py 實測＋線性規劃驗算後發現：
`(0.0,-0.635)` 案例先前選中的 roll=150°，在「角速度=0」（姿態鎖定）
約束下沿揮桿方向的最大可達速度只有 0.81 m/s，遠低於所需 1.51 m/s——這是
真正的運動學速度上限，不是控制律或 waypoint 設計的問題。掃過全部 24 個
roll 候選後找到 roll=-60° 能達到 1.53 m/s（超過所需），證實**不同 roll
選擇的關節構型有截然不同的揮桿速度可操作性**，先前的排序（只看 AIM
margin）完全沒有考慮到這件事。

線性規劃：`max (direction_unit · Jv) @ qdot` s.t. `Jang @ qdot = 0`
（角速度鎖定為 0，模擬 contact_orientation 全程不變的需求）、
`qdot ∈ [-2, 2]^7`（`_dof_limits`，見 articulation_api_impl.py）。

用法：
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_roll_swing_capable.py
"""

import math
import os
import sys

import numpy as np
from scipy.optimize import linprog

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wam7_kinematics as wk  # noqa: E402
from core.services.base_placement_calculator import compute_base_pose, CANONICAL_REST_JOINTS  # noqa: E402
from core.services.cue_pose_calculator import compute_tilted_wrist_pose, compute_tilted_direction  # noqa: E402
from core.services.swing_trajectory_calculator import compute_required_tip_speed  # noqa: E402

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_CUE_BALL_SPEED = 1.995  # verify_swing_trajectory.py 稀疏網格中點案例一致

_XY_CASES = [
    (-0.606425, -0.9382125),
    (0.0, -0.9382125),
    (0.606425, -0.9382125),
    (-0.606425, -0.635),
    (0.0, -0.635),
    (0.606425, -0.635),
]
_ROLL_CANDIDATES_DEG = list(range(-180, 180, 15))
_QDOT_MAX = 2.0


def _max_swing_speed(joints, base_position, direction_unit):
    J = wk._numerical_jacobian(joints, base_position)
    Jv = J[:3, :]
    Jang = J[3:, :]
    c = direction_unit @ Jv
    bounds = [(-_QDOT_MAX, _QDOT_MAX)] * wk.NUM_JOINTS
    res = linprog(c=-c, A_eq=Jang, b_eq=np.zeros(3), bounds=bounds, method="highs")
    if not res.success:
        return None
    return float(c @ res.x)


def evaluate(cue_ball_xy, shot_angle_deg, roll_deg, required_tip_speed):
    roll_rad = math.radians(roll_deg)
    base_position, _base_yaw_rad = compute_base_pose(cue_ball_xy[0], cue_ball_xy[1], shot_angle_deg, _TABLE_Z, _BALL_RADIUS)
    wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
        cue_ball_xy, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
    )
    if tilt_rad is None:
        return None
    start = np.array([0.0, *CANONICAL_REST_JOINTS])
    joints, ok, pos_err, orient_err = wk.solve_ik(wrist, orientation, start, base_position=base_position, max_iters=500)
    if not ok:
        return None
    ik_margin = float(np.min(np.minimum(
        joints - np.array([lo for lo, hi in wk.JOINT_LIMITS]),
        np.array([hi for lo, hi in wk.JOINT_LIMITS]) - joints,
    )))
    direction_unit = np.array(compute_tilted_direction(shot_angle_deg, tilt_rad))
    swing_speed = _max_swing_speed(joints, base_position, direction_unit)
    if swing_speed is None:
        return None
    return {
        "roll_deg": roll_deg, "ik_margin": ik_margin, "swing_speed": swing_speed,
        "swing_speed_ok": swing_speed >= required_tip_speed,
    }


def main():
    required_tip_speed = compute_required_tip_speed(_CUE_BALL_SPEED)
    print(f"required_tip_speed={required_tip_speed:.4f}")

    for cue_ball_xy in _XY_CASES:
        print("=" * 100)
        print(f"[{cue_ball_xy}]")
        results = []
        for roll_deg in _ROLL_CANDIDATES_DEG:
            r = evaluate(cue_ball_xy, 0.0, roll_deg, required_tip_speed)
            if r is not None:
                results.append(r)

        # 排序：先篩「揮桿速度足夠」的，再依 IK margin 高低排（IK margin 也
        # 重要，太貼近限位的候選在真實 PD 控制/物理雜訊下可能不穩）；揮桿
        # 速度不足的候選排在後面，依 swing_speed 高低排（萬一全部候選都不
        # 夠，至少知道哪個最接近）。
        results.sort(key=lambda r: (not r["swing_speed_ok"], -r["ik_margin"] if r["swing_speed_ok"] else -r["swing_speed"]))

        print(f"  由「揮桿速度是否足夠」+「IK margin」排序（前 8 名）：")
        for r in results[:8]:
            flag = "OK" if r["swing_speed_ok"] else "不足"
            print(f"    roll={r['roll_deg']:+4d}deg  swing_speed={r['swing_speed']:.4f}({flag})  ik_margin={r['ik_margin']:.4f}")


if __name__ == "__main__":
    main()
