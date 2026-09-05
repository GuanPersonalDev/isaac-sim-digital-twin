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

# 高架橋轉向（C1）時手臂本體（不是桿頭）可能掃過球檯庫邊/袋口，用
# `roll_rad` 這個閃避自由度可以避開。
#
# 這份表用 `scripts/search_collision_free_roll.py` 對
# `action_bounds.CUE_BALL_PLACEMENT_X/Y` 的完整 3×3 網格逐點搜尋：對每個
# 候選 roll 依差動 IK 餘裕排序，逐一用真實 Isaac Sim 物理模擬＋
# `enable_contact_reporting` 碰撞回報驗證，取第一個「AIM→後擺→隨揮終點
# 整條軌跡在同一分支內收斂、且無碰撞」的候選——比只驗證「目標可達」更
# 貼近真實 `ArticulationAPIImpl._step_motion()` 的行為。roll 不能只看
# cue_ball_y（碰撞跟世界座標系裡離哪個庫邊/袋口近有關，不是只看關節
# 構型），因此對 X/Y 都逐點驗證。推導歷程見 docs/CHANGELOG.md。
#
# `y=-1.241425`（`CUE_BALL_PLACEMENT_Y` 下界）純幾何無解
# （`compute_required_tilt_rad()` 回傳 `None`），roll 用不到，沿用鄰近列的
# 值只是讓 nearest-neighbor 查表有合理落點。
#
# (cue_ball_x, cue_ball_y, roll_deg)
_ROLL_LOOKUP_GRID = [
    (-0.606425, -1.241425, 165),
    (-0.606425, -0.9382125, 165),
    (-0.606425, -0.635, 165),
    (0.0, -1.241425, -180),
    (0.0, -0.9382125, -180),
    (0.0, -0.635, 150),
    (0.606425, -1.241425, 165),
    (0.606425, -0.9382125, 165),
    (0.606425, -0.635, 165),
]


def lookup_roll_rad(cue_ball_xy: tuple[float, float]) -> float:
    """回傳離查表座標歐氏距離最近的網格點的 `roll_rad`。這是 9 點粗網格的
    最近鄰查表，不是連續公式——網格點之間的座標實際覆蓋率（是否也剛好落在
    正確的 roll 範圍內）要靠 `scripts/verify_swing_trajectory.py` 對更密的
    真實 Action 網格驗證，不能假設一定成立。"""
    cue_x, cue_y = cue_ball_xy
    _, _, roll_deg = min(
        _ROLL_LOOKUP_GRID,
        key=lambda p: (p[0] - cue_x) ** 2 + (p[1] - cue_y) ** 2,
    )
    return math.radians(roll_deg)


# `DEFAULT_BACKSWING_DISTANCE_M`（0.15m）跟關節實際能提供的加速能力脫鉤，
# 這份表用 `scripts/search_backswing_distance_ik.py`（純數值 IK 可達邊界
# 法：沿擊球反方向從 contact pose 逐步後退，找到「還能收斂、且離關節硬
# 限位有安全餘裕」的最大距離，多組不同關節構型分支各自延伸取最遠）反推
# 每個高架橋案例的後擺距離，取代寫死常數。基座位置一律用
# `compute_base_pose()` 的公式值，不額外搜尋偏移——曾經嘗試把基座偏移也
# 當自由變數搜尋，但一次性可達性求解沒有模擬差動 IK 沿路徑逐步收斂的
# 動態行為，會導致真實控制迴圈不收斂，已放棄，推導歷程見
# docs/CHANGELOG.md。manipulability 上限本來就低的案例（`y=-0.9382125`
# 這排）速度依然達不到目標，是已知、可接受的既有限制。
#
# (cue_ball_x, cue_ball_y, backswing_distance_m)
# y=-1.241425（CUE_BALL_PLACEMENT_Y 下界）純幾何無解，沿用鄰近列（y=
# -0.9382125）的值只是讓 nearest-neighbor 查表有合理落點，實際用不到（跟
# _ROLL_LOOKUP_GRID 同一個慣例）。
_BACKSWING_DISTANCE_LOOKUP_GRID = [
    (-0.606425, -1.241425, 0.21),
    (-0.606425, -0.9382125, 0.21),
    (-0.606425, -0.635, 0.15),
    (0.0, -1.241425, 0.20),
    (0.0, -0.9382125, 0.20),
    (0.0, -0.635, 0.14),
    (0.606425, -1.241425, 0.20),
    (0.606425, -0.9382125, 0.20),
    (0.606425, -0.635, 0.15),
]


def lookup_backswing_distance_m(cue_ball_xy: tuple[float, float]) -> float:
    """回傳離查表座標歐氏距離最近的網格點的後擺距離（m）。用法跟
    `lookup_roll_rad()` 完全一樣（同一個 9 點粗網格，最近鄰查表）。"""
    cue_x, cue_y = cue_ball_xy
    _, _, distance_m = min(
        _BACKSWING_DISTANCE_LOOKUP_GRID,
        key=lambda p: (p[0] - cue_x) ** 2 + (p[1] - cue_y) ** 2,
    )
    return distance_m


def lookup_base_offset(cue_ball_xy: tuple[float, float]) -> tuple[float, float]:
    """回傳離查表座標歐氏距離最近的網格點的基座水平位移 (dx, dy)（m），要
    疊加在 `base_placement_calculator.compute_base_pose()` 算出的基座位置
    上（Z 不動）。用法跟 `lookup_roll_rad()`／`lookup_backswing_distance_m()`
    一樣（同一個 9 點粗網格，最近鄰查表，三者共用同一份 cue_ball_xy 座標
    才能保證算出彼此一致的目標姿態）。"""
    cue_x, cue_y = cue_ball_xy
    _, _, _distance_m, dx, dy = min(
        _BACKSWING_DISTANCE_LOOKUP_GRID,
        key=lambda p: (p[0] - cue_x) ** 2 + (p[1] - cue_y) ** 2,
    )
    return dx, dy


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


def _nlerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    # 四元數線性內插＋正規化（NLERP，不是精確的球面內插 SLERP，但角度差
    # 不大、切成夠多段時誤差可忽略），用來把 compute_elevated_bridge_waypoints()
    # 的 C1 轉向階段拆成多個中繼姿態。q0/q1 可能差了正負號（同一個旋轉的
    # 兩種表示），內積為負時先取反 q1 走最短路徑，否則內插會繞遠路甚至
    # 反向轉。
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - t) * q0 + t * q1
    return q / np.linalg.norm(q)


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
    current_orientation: list[float],
    cue_ball_xy: tuple[float, float],
    shot_angle_deg: float,
    table_z: float,
    ball_radius: float,
    position_offset: list[float] = [0.0, 0.0],
    roll_rad: float = 0.0,
    safe_altitude_margin: float = 0.3,
    rotate_steps: int = 8,
    *,
    backswing_distance_m: float,
) -> list[PoseWaypoint] | None:
    """把「先垂直爬升、再水平平移、最後才轉向」的高架橋逼近幾何轉成一串
    `PoseWaypoint`（不含 Phase 0——Phase 0 是先用 joint-space 回安全姿態避開
    差動 IK 奇異點，由呼叫端透過 `ArticulationAPI.move_through_poses()` 的
    `preceding_joint_targets` 參數處理，不在這支函式的職責內）：

      B1 (climb_point,    current_orientation) —— 保持目前姿態原地垂直爬升到安全高度
      B2 (approach_point, current_orientation) —— 保持目前姿態水平平移到最終腕部 xy 正上方
      C1×rotate_steps (approach_point, NLERP(current→final, i/rotate_steps))
          —— 安全高度原地轉到最終傾斜姿態，拆成 `rotate_steps` 個中繼姿態
      C2 (final_wrist_position, final_orientation) —— 純垂直下降

    B1/B2 用「保持目前姿態原地爬升＋平移」而非「先轉向正上方再爬升」：
    轉向動作延後到 C1，此時只需要從水平姿態直接轉到最終傾斜姿態（通常
    只有 5°~30°），不會像先轉 90° 到正上方那樣逼死 wrist_yaw/wrist_pitch
    的硬限位。C1 用 NLERP 把轉向拆成多個中繼姿態（不是單一大跳躍
    waypoint）：單一大跳躍會讓差動 IK 為了在一個 waypoint 內達成整個姿態
    變化，逼 `shoulder_yaw`/`elbow_pitch` 沿路劇烈擺盪、掃過球檯庫邊/
    袋口；拆成多個小角度中繼姿態後手臂本體的運動軌跡更貼近「原地小角度
    轉」。兩輪修正的實測數據見 docs/CHANGELOG.md。

    `current_orientation`：呼叫端在真正下達 Phase 0（joint-space 回安全
    姿態）之前，這個姿態還沒真的在場景裡發生，不能用
    `ArticulationAPI.get_end_effector_orientation()` 讀（讀到的是移動前的
    舊姿態）。呼叫端應該用分析算出的目標姿態（跟 Phase 0 的
    `target_end_effector_position` 用 `compute_canonical_wrist_position()`
    算出來、不能沿用移動前舊值同一個道理），不是靠 runtime 讀值。

    `safe_high_z` 取 `current_position` 跟目標腕部高度兩者較大值再加
    `safe_altitude_margin`；`climb_point` 是目前腕部 xy、高度用
    `safe_high_z`；`approach_point` 是目標腕部 xy、高度用 `safe_high_z`。

    先呼叫 `compute_tilted_wrist_pose()` 算出最終 wrist/orientation/tilt_rad，
    回傳 `None` 代表它判定幾何無解。

    `backswing_distance_m`：`compute_tilted_wrist_pose()` 回傳的 `wrist`
    精確落在母球球心，不是球面（`compute_contact_point()`／
    `required_grip_position()` 共用的既有慣例，STRIKE 高速通過用同一個
    `wrist` 當揮桿參考點無妨）。但 AIM／搭橋是慢速 P 控制器收斂到這個
    固定目標，一路收斂會把桿尖持續往球心推、真的把母球推走。這裡把
    AIM／搭橋收斂的最終（C2）目標點沿 `-direction` 方向退開
    `backswing_distance_m`，讓收斂終點停在母球表面外側，不是球心；此為
    唯一改動 wrist 目標的地方，`wrist`／`compute_tilted_wrist_pose()`
    本身不受影響。呼叫端一律傳
    `cue_pose_calculator.lookup_backswing_distance_m(cue_ball_xy)`（AIM
    收斂終點＝STRIKE 後擺起點，統一成同一個值，不再有獨立預設值，強制
    呼叫端明示），數值來源與校準記錄見 docs/CHANGELOG.md。
    """
    wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
        cue_ball_xy, shot_angle_deg, table_z, ball_radius, position_offset, roll_rad
    )
    if tilt_rad is None or wrist is None or orientation is None:
        return None

    direction = compute_tilted_direction(shot_angle_deg, tilt_rad)
    safe_wrist = wrist - backswing_distance_m * direction

    safe_high_z = max(float(current_position[2]), float(wrist[2])) + safe_altitude_margin
    climb_point = [float(current_position[0]), float(current_position[1]), safe_high_z]
    approach_point = [float(safe_wrist[0]), float(safe_wrist[1]), safe_high_z]

    wrist_list = safe_wrist.tolist()
    orientation_list = orientation.tolist()
    current_orientation_array = np.array(current_orientation, dtype=float)

    waypoints = [
        PoseWaypoint(position=climb_point, orientation=list(current_orientation)),
        PoseWaypoint(position=approach_point, orientation=list(current_orientation)),
    ]
    for step in range(1, rotate_steps + 1):
        t = step / rotate_steps
        interpolated = _nlerp_quat(current_orientation_array, orientation, t)
        waypoints.append(PoseWaypoint(position=approach_point, orientation=interpolated.tolist()))
    waypoints.append(PoseWaypoint(position=wrist_list, orientation=orientation_list))
    return waypoints
