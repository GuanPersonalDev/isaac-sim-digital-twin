"""球桿姿態幾何計算：給定母球位置＋擊球角度＋偏移量，算出桿身應該指向哪、
腕部（=握把）目標在哪、需不需要抬高閃避庫邊。

從 `scripts/scan_elevated_bridge_approach.py`（Issue #233 的「高架橋」研究，
25 點網格掃描後驗證 100% 無碰撞，見 `docs/issue-180-reachability-analysis.md`
第十一節）搬進正式程式碼，是 `DemoTableOrchestrator._execute_aim()`／
`_execute_strike()` 共用的幾何單一事實來源。

⚠️ `_RAILS` 是庫邊「頂部外緣」的碰撞淨空邊界（含安全餘量），跟
`pocket_geometry.TABLE_WIDTH/TABLE_LENGTH` 推導出的球心邊界是不同的物理
特徵，不要合併成同一個常數來源。
"""

import math
import re

import numpy as np

from .base_placement_calculator import CUE_STICK_GRIP_TO_TIP, required_grip_position
from ..models.pose_waypoint import PoseWaypoint

_RAIL_TOP_HEIGHT = 0.04
_SAFETY_MARGIN = 0.05
_RAILS = [
    ("x", -0.66, (-1.27, 1.27)),
    ("x", 0.66, (-1.27, 1.27)),
    ("y", -1.295, (-0.635, 0.635)),
    ("y", 1.295, (-0.635, 0.635)),
]


def _segment_rail_crossings(p0, p1, rails):
    # 計算線段 p0→p1 跟 rails 列表（每個元素是 (axis, coord, other_range)）
    # 的所有交點：沿線段參數化 (x,y) = p0 + t*(p1-p0)，t∈[0,1]，對每面
    # rail 解出 t、代回另一軸座標並檢查落在 other_range 內，回傳
    # [((x, y), 交點到 p1 的距離), ...]。
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    result = []
    for axis, coord, other_range in rails:
        if axis == "x":
            if dx == 0:
                continue
            t = (coord - x0) / dx
            if not (0.0 <= t <= 1.0):
                continue
            y = y0 + t * dy
            if other_range[0] <= y <= other_range[1]:
                d = math.hypot(coord - x1, y - y1)
                result.append(((coord, y), d))
        else: # axis == "y"
            if dy == 0:
                continue
            t = (coord - y0) / dy
            if not (0.0 <= t <= 1.0):
                continue
            x = x0 + t * dx
            if other_range[0] <= x <= other_range[1]:
                d = math.hypot(x - x1, coord - y1)
                result.append(((x, coord), d))
    return result


def compute_required_tilt_rad(grip_xy, ball_xy, tip_height):
    """回傳 (tilt_rad, crossing_point_or_None)。tilt_rad=0 代表不需要抬；
    tilt_rad=None 代表無解（即使垂直也不夠高，這個交點物理上過不去）。"""
    crossings = _segment_rail_crossings(grip_xy, ball_xy, _RAILS)
    if not crossings:
        return 0, None
    
    crossing, d = min(crossings, key=lambda c: c[1])
    if d < 1e-6:
        return None, crossing

    required_sin = (_RAIL_TOP_HEIGHT + _SAFETY_MARGIN - tip_height) / d
    if required_sin <= 0:
        return 0, crossing
    if required_sin >= 1.0:
        return None, crossing

    return math.asin(required_sin), crossing


def compute_tilted_direction(shot_angle_deg: float, tilt_rad: float) -> np.ndarray:
    """握把→桿尖方向單位向量。tilt_rad=0 時純水平（等同 base_placement_calculator
    的 `_aim_direction`），tilt_rad>0 時水平分量按 cos(tilt) 縮小、多一個向下
    （朝母球）的 Z 分量。"""
    theta = math.radians(shot_angle_deg)
    dx, dy = -math.sin(theta), math.cos(theta)
    return np.array([dx * math.cos(tilt_rad), dy * math.cos(tilt_rad), -math.sin(tilt_rad)])


def _shortest_arc_quat(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    # 回傳把 v_from 最短路徑旋轉到 v_to 的四元數（wxyz）。v_from≈v_to 回傳
    # 單位四元數；v_from≈-v_to（180°）時任取一個跟 v_from 正交的軸當旋轉軸。
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    dot = float(np.dot(v_from, v_to))

    if dot > 0.999999:
        return np.array([1, 0, 0, 0])
    
    if dot < -0.999999:
        axis = np.cross(v_from, np.array([1, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, np.array([0, 1, 0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0, *axis])

    half = v_from + v_to
    half = half / np.linalg.norm(half)
    w = float(np.dot(v_from, half))
    xyz = np.cross(v_from, half)

    return np.array([w, *xyz])



def _axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    # 軸角表示法轉四元數（wxyz）：w=cos(angle/2)，xyz=axis(normalized)*sin(angle/2)。
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([math.cos(half), *(axis * math.sin(half))])


def _quat_multiply(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """q1 ⊗ q0：先套用 q0、再套用 q1（wxyz）。"""
    w1, x1, y1, z1 = q1
    w0, x0, y0, z0 = q0
    return np.array([
        w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
        w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
        w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
        w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
    ])


def compute_contact_point(
    ball_center: np.ndarray,
    direction_unit: np.ndarray,
    position_offset: list[float],
    ball_radius: float,
) -> np.ndarray:
    """把 `position_offset`（[上下, 左右]，球半徑比例）換算成球面上的實際
    接觸點。以 `direction_unit`（握把→桿尖方向）為主軸，取兩個正交基向量：
    - `e_up`：跟 `direction_unit` 正交、盡量指向世界 +Z 的分量
    - `e_side`：`direction_unit × e_up`（水平面內、垂直於桿身的側向分量）

    在 `tilt_rad=0`（水平）的情況下，`e_up=(0,0,1)`、`e_side` 精確等於
    `impulse_striking_service.compute_cue_ball_velocities()` 的 `side` 向量
    `(cosθ, sinθ, 0)`——這是刻意對齊的，確保 training／demo 兩條腿的
    `position_offset` 正負號語意一致（否則同一個 RL policy 在兩條腿上學到
    的「往上打會怎樣」語意會相反）。`position_offset=[0,0]` 時回傳值精確
    等於 `ball_center`（零偏移退化，不影響任何既有零偏移呼叫端的行為）。

    `direction_unit` 接近垂直（世界 +Z 分量幾乎全部）時 `e_up` 的 Gram-Schmidt
    取法會退化（norm≈0），需要另外取一個跟 direction_unit 正交的水平向量
    當備援基向量。
    """
    world_z = np.array([0, 0, 1])
    e_up_raw = world_z - np.dot(world_z, direction_unit) * direction_unit
    norm_up = np.linalg.norm(e_up_raw)

    if norm_up < 1e-6:
        fallback = np.array([1, 0, 0])
        e_up_raw = fallback - np.dot(fallback, direction_unit) * direction_unit
        norm_up = np.linalg.norm(e_up_raw)

    e_up = e_up_raw / norm_up
    e_side = np.cross(direction_unit, e_up)

    return ball_center + ball_radius * (position_offset[0] * e_up + position_offset[1] * e_side)



def compute_tilted_wrist_pose(
    cue_ball: tuple[float, float],
    shot_angle_deg: float,
    table_z: float,
    ball_radius: float,
    position_offset: list[float] = [0.0, 0.0],
    roll_rad: float = 0.0,
):
    """回傳 (wrist_position, wrist_orientation_wxyz, tilt_rad, crossing)。
    `tilt_rad=None` 代表這個母球位置無解（純幾何上過不去，不是差動 IK 的
    問題），此時 wrist/orientation 也回傳 None。

    做法：`required_grip_position()` 算水平握把點 → `compute_required_tilt_rad()`
    判斷需不需要抬高 → `compute_tilted_direction()` 算方向 →
    `compute_contact_point()` 算球面上實際接觸點 → 沿方向反方向退開
    `CUE_STICK_GRIP_TO_TIP` 得腕部位置 → `_shortest_arc_quat()` 從 +Y 轉到
    `direction` 得基礎朝向，`roll_rad` 非 0 時再疊一個繞 `direction` 軸的
    `_axis_angle_quat()` 旋轉。

    `roll_rad`：球桿繞自身軸（=繞 direction 這個世界向量）額外旋轉的角度，
    是 5 維冗餘的那個自由度，不影響桿頭實際指向或位置，純粹用來閃避特定
    關節配置下的關節限位。
    """
    tip_height = table_z + ball_radius
    grip_x, grip_y = required_grip_position(cue_ball[0], cue_ball[1], shot_angle_deg)

    tilt_rad, crossing = compute_required_tilt_rad((grip_x, grip_y), cue_ball, tip_height)
    if tilt_rad is None:
        return None, None, None, crossing

    direction = compute_tilted_direction(shot_angle_deg, tilt_rad)
    ball_center = np.array([cue_ball[0], cue_ball[1], tip_height])

    contact = compute_contact_point(ball_center, direction, position_offset, ball_radius)

    wrist = contact - CUE_STICK_GRIP_TO_TIP * direction

    base_orientation = _shortest_arc_quat(np.array([0, 1, 0]), direction)
    if roll_rad != 0.0:
        orientation = _quat_multiply(_axis_angle_quat(direction, roll_rad), base_orientation)
    else:
        orientation = base_orientation
    
    return wrist, orientation, tilt_rad, crossing


def compute_elevated_bridge_waypoints(
    current_position: list[float],
    cue_ball_xy: tuple[float, float],
    shot_angle_deg: float,
    table_z: float,
    ball_radius: float,
    position_offset: list[float] = [0.0, 0.0],
    roll_rad: float = 0.0,
    safe_altitude_margin: float = 0.3,
) -> list[PoseWaypoint] | None:
    """把 `scan_elevated_bridge_approach.py` 的 Phase A/B/C1/C2 幾何轉成
    4 個 `PoseWaypoint`（不含 Phase 0——Phase 0 是先用 joint-space 回安全姿態
    避開差動 IK 奇異點，由呼叫端透過 `ArticulationAPI.move_through_poses()`
    的 `preceding_joint_targets` 參數處理，不在這支函式的職責內）：

      A  (current_position, up_orientation)   —— 原地轉向朝上
      B  (approach_point,   up_orientation)   —— 平移到接觸點正上方安全高度
      C1 (approach_point,   final_orientation) —— 安全高度先轉到最終傾斜姿態
      C2 (final_wrist_position, final_orientation) —— 純垂直下降

    C1/C2 拆兩段是為了避免「位置與姿態同時收斂」導致桿頭中途掃過安全高度
    以下（實測撞過桌面 Surface 一次）：C1 腕部不動只轉向，C2 姿態已固定
    只下降，桿頭高度隨腕部線性下降、不會中途下探。

    `up_orientation` 是從 +Y 轉到世界 +Z 的 `_shortest_arc_quat()`；
    `safe_high_z` 取 `current_position` 跟目標腕部高度兩者較大值再加
    `safe_altitude_margin`；`approach_point` 是目標腕部 xy、高度用
    `safe_high_z`。

    先呼叫 `compute_tilted_wrist_pose()` 算出最終 wrist/orientation/tilt_rad，
    回傳 `None` 代表它判定幾何無解。
    """
    wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
        cue_ball_xy, shot_angle_deg, table_z, ball_radius, position_offset, roll_rad
    )
    if tilt_rad is None or wrist is None or orientation is None:
        return None

    up_orientation = _shortest_arc_quat(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))

    safe_high_z = max(float(current_position[2]), float(wrist[2])) + safe_altitude_margin
    approach_point = [float(wrist[0]), float(wrist[1]), safe_high_z]

    wrist_list = wrist.tolist()
    orientation_list = orientation.tolist()
    up_orientation_list = up_orientation.tolist()

    return [
        PoseWaypoint(position=list(current_position), orientation=up_orientation_list),
        PoseWaypoint(position=approach_point, orientation=up_orientation_list),
        PoseWaypoint(position=approach_point, orientation=orientation_list),
        PoseWaypoint(position=wrist_list, orientation=orientation_list),
    ]
