"""UR3e 姿態/基座計算：`base_placement_calculator.py`（WAM7 專屬）的 UR3e
對應模組。給定球檯幾何算出的目標腕部位置＋揮桿方向，回傳機器人基座座標
（純平移，不旋轉，跟 `RobotArm.reposition()` 介面慣例一致）與 6 個關節
的目標角度。

## 跟 WAM7 版本的架構差異

`base_placement_calculator.py` 用「單一固定姿態（`CANONICAL_REST_JOINTS`）
＋一個瞄準關節（`base_yaw`）」——任何瞄準角都靠 `base_yaw` 吸收，跟母球
Y 座標無關；只有高架橋（tilt>0）才需要額外查表（`_ROLL_LOOKUP_GRID`／
`_BACKSWING_DISTANCE_LOOKUP_GRID`）。

UR3e 這邊已驗證的方法（見 `scripts/design_human_like_ur3e_pose.py`／
`scripts/test_elevated_bridge_ur3e_table.py`／
`scripts/search_ur3e_placement_constants.py`）不是「單一固定姿態」——
純肘關節轉動要同時滿足「桿尖高度合理」＋「傾斜角跟母球 Y 座標對應的
`tilt_rad` 一致」，只有 `shoulder_pan`（角色等同 WAM7 的 `base_yaw`）
真的跟瞄準角無關、可以解析算，其餘 5 個關節（`shoulder_lift, elbow,
wrist1, wrist2, wrist3`）**依 `tilt_rad`（=依母球 Y 座標）分組查表**，
不是單一固定值。這個模組因此提供的是「依 Y 分組的姿態查表」＋「解析算
`shoulder_pan`＋基座平移」，跟 WAM7 版「單一固定姿態＋依 Y 查 roll/
backswing」剛好是對稱但不同的分工。

`table_orchestrator.DemoTableOrchestrator._execute_aim()`／`_execute_strike()`
用 `isinstance(self._robot_arm, UR3eRobot)` 分流到這個模組（見該檔案
`_execute_aim_ur3e()`／`_execute_strike_ur3e()`），WAM7 那條既有路徑完全
不變。

⚠️ 這條路徑目前只有**運動學可行性**（`elbow_margin_ratio_est<=1`）跟高架橋
5.34° 案例的真實揮桿速度被驗證過（見 `_VALIDATED_BRIDGE_*` 常數說明）；
flat 案例、9.91° 案例、以及「從手臂目前姿態安全接近到後擺姿態」這一整段
都還沒有實測驗證，見各常數與 `ArticulationAPIImpl.move_swing_elbow_pivot()`
docstring 的個別說明。把 `extension/billiard_digital_twin/billiard_digital_
twin.py` 的 `_ROBOT_ARM_CLASS` 換成 `UR3eRobot` 之前，應該先跑一次真實
Isaac Sim GUI 驗證，不能只憑這裡的常數就當作已經可以正式上線。
"""

import math

import numpy as np

UR3E_ELBOW_DOF_INDEX = 2
"""UR3e 標準關節順序 [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2,
wrist3] 裡 elbow 的索引，跟 `scripts/search_ur3e_placement_constants.py`／
`extension/isaac_sim_impl_6_0/articulation_api_impl.py` 用同一個慣例
（實際用哪個索引應該用 `dof_names` 現場確認，這個常數只是找不到
`dof_names` 時的 fallback，見 `ArticulationAPIImpl._resolve_end_effector_
jacobian_index()` 同一類防呆模式）。"""


# 每筆紀錄：(cue_ball_y, joints_pan0, direction_local, local_tip_position)
# - `joints_pan0`：(shoulder_lift, elbow, wrist1, wrist2, wrist3) 5 個角度
#   （rad），`shoulder_pan` 固定在 0 時量出來的姿態——跟 WAM7 的
#   `CANONICAL_REST_JOINTS` 是同一個角色，但依 `tilt_rad`（=母球 Y 座標）
#   分成多組，不是單一固定值。
# - `direction_local`：`shoulder_pan=0` 時，純肘關節轉動能給桿尖的線速度
#   方向（單位向量）。跟 `local_tip_position` 一樣是這組關節角本身的固有
#   幾何量，不隨基座怎麼擺而改變。
# - `local_tip_position`：`shoulder_pan=0`、基座在世界座標原點時，桿尖
#   （已經加上 `CUE_STICK_GRIP_TO_TIP` 偏移）的世界座標。
#
# 資料來源：`scripts/search_ur3e_placement_constants.py`（`SEARCH_TILT_DEG`
# 環境變數指定目標傾斜角），對應 `cue_pose_calculator.compute_required_
# tilt_rad()` 在 `shot_angle_deg=0` 時，`CUE_BALL_PLACEMENT_Y` 三個代表 Y
# 值算出的 `tilt_rad`（跟 X 座標無關，只跟 Y 有關——`_ROLL_LOOKUP_GRID`／
# `_BACKSWING_DISTANCE_LOOKUP_GRID` 也是同一個既有發現，見
# `cue_pose_calculator.py`）。
#
# ⚠️ `cue_ball_y=-0.9382125`（tilt≈9.91°）目前**沒有找到可行解**——
# `scripts/search_ur3e_placement_constants.py`（6825 組網格）唯一通過
# 「桿尖高度合理＋傾斜角對齊」篩選的候選，肘關節需要 5.24 rad/s，遠超過
# 3.14 rad/s 的馬達限制（`elbow_margin_ratio_est=1.6683`，>1 代表不可行）。
# 這是目前搜尋方法論的已知限制，不是遺漏——WAM7 版的
# `_BACKSWING_DISTANCE_LOOKUP_GRID` 對同一排（`y=-0.9382125`）也有一模
# 一樣的既有限制（見該檔案「manipulability 上限本來就低的案例...任何 roll
# 都到不了目標球速」），這裡沿用同一個「已知、可接受」的處理方式：這排
# 暫時沿用 `y=-0.635`（tilt≈5.34°，已驗證 96.1% 達成率）的姿態當
# nearest-neighbor 查表的合理落點，不是真的可行解，之後若要解決需要更大
# 的搜尋範圍或放寬「純肘關節驅動」的設計限制（見對話紀錄的討論）。
_VALIDATED_BRIDGE_JOINTS_PAN0 = (-0.4, -0.8, -0.7, 0.25, 0.0)
_VALIDATED_BRIDGE_DIRECTION_LOCAL = (0.996146, -0.001016, -0.08771)
_VALIDATED_BRIDGE_LOCAL_TIP_POSITION = (0.274152, 1.528253, 0.807831)
_VALIDATED_BRIDGE_SPEED_PER_UNIT_OMEGA = 0.5602438975576889
"""tilt≈5.34°（`cue_ball_y=-0.635`）的驗證結果，來源：
`SEARCH_TILT_DEG=5.3401421352383425 scripts/search_ur3e_placement_
constants.py`（`shoulder_pan=0` 時量測，這裡存的是 Stage 1 的原始量測值，
不是套用 `required_pan` 之後的值——套用旋轉是 `_solve_base_position_and_
joint_targets()` 這個函式自己的職責，常數只存 pan=0 基準）。真實揮桿執行
達成率驗證見 `scripts/test_elevated_bridge_ur3e_table.py`（96.1%，settle
階段僅輕微擦地 impulse 1.07，非碰撞等級，見 docs/CHANGELOG.md 的取值
說明）。

`speed_per_unit_omega`：肘關節每 1 rad/s 角速度能給桿尖多少沿 `target_
direction` 的線速度（m/s），是這組姿態本身的固有幾何量（跟 cue_ball_speed
無關）——`compute_target_elbow_velocity()` 用
`required_tip_speed / speed_per_unit_omega` 換算任意 `cue_ball_speed`
需要的肘關節角速度，不用每次重新讀 Jacobian。來源：
`required_tip_speed(1.995)=1.51164 / omega_elbow_needed(2.698182)`（見
`search_ur3e_placement_constants.py` 該次執行的印出值）。"""

_PLACEMENT_LOOKUP_GRID: list[tuple[float, tuple[float, ...], tuple[float, float, float], tuple[float, float, float], float]] = [
    (
        -1.241425,  # 跟母球位置幾何無解（compute_tilted_wrist_pose 回傳 tilt_rad=None）那排，
        # 沿用 y=-0.635 的值只是讓 nearest-neighbor 查表有合理落點，實際用不到，
        # 跟 cue_pose_calculator.py 既有慣例一致。
        _VALIDATED_BRIDGE_JOINTS_PAN0,
        _VALIDATED_BRIDGE_DIRECTION_LOCAL,
        _VALIDATED_BRIDGE_LOCAL_TIP_POSITION,
        _VALIDATED_BRIDGE_SPEED_PER_UNIT_OMEGA,
    ),
    (
        -0.9382125,  # tilt≈9.91°——目前無可行解（見上方模組說明），沿用
        # y=-0.635 的值當佔位符，不是真的可行解。
        _VALIDATED_BRIDGE_JOINTS_PAN0,
        _VALIDATED_BRIDGE_DIRECTION_LOCAL,
        _VALIDATED_BRIDGE_LOCAL_TIP_POSITION,
        _VALIDATED_BRIDGE_SPEED_PER_UNIT_OMEGA,
    ),
    (
        -0.635,  # tilt≈5.34°，已驗證可行（見上方常數說明）。
        _VALIDATED_BRIDGE_JOINTS_PAN0,
        _VALIDATED_BRIDGE_DIRECTION_LOCAL,
        _VALIDATED_BRIDGE_LOCAL_TIP_POSITION,
        _VALIDATED_BRIDGE_SPEED_PER_UNIT_OMEGA,
    ),
]

_FLAT_SPEED_PER_UNIT_OMEGA = 0.563768992108711
"""同 `_VALIDATED_BRIDGE_SPEED_PER_UNIT_OMEGA` 的意義，flat 案例版本。
來源：`required_tip_speed(1.995)=1.51164 / omega_elbow_needed(2.681311)`
（`SEARCH_TILT_DEG=0.0 scripts/search_ur3e_placement_constants.py`）。"""

_FLAT_PLACEMENT: tuple[tuple[float, ...], tuple[float, float, float], tuple[float, float, float], float] = (
    # tilt_rad=0（flat 案例）——跟母球位置無關，任何瞄準角都靠 shoulder_pan
    # 吸收，是跟 WAM7 CANONICAL_REST_JOINTS 同一個角色的「單一固定值」。
    # 來源：`SEARCH_TILT_DEG=0.0 scripts/search_ur3e_placement_constants.py`
    # （elbow_margin_ratio_est=0.8535，餘裕 14.6%）。
    #
    # ⚠️ 跟 `scripts/test_ur3e_human_pose_swing_speed.py` 驗證過 104.7%
    # 達成率的那組 joints **不是同一組**——那次驗證是空場景（沒有球檯），
    # 桿尖高度算出來是 1.8m，真的擺到球檯旁邊會要求基座陷進地板，不能
    # 直接沿用（見 docs/CHANGELOG.md）。這裡的常數目前只確認了 Stage 1/2
    # 的運動學可行性，還沒有像高架橋案例那樣接上真實球檯＋quintic 揮桿
    # 執行驗證，是這個模組目前最大的未驗證缺口。
    (-2.8, -0.8, -0.7, -0.5, 0.0),
    (0.99999, 0.000123, 0.004412),
    (-0.229146, 1.397064, 0.803869),
    _FLAT_SPEED_PER_UNIT_OMEGA,
)


def lookup_placement_constants(
    cue_ball_y: float,
) -> tuple[tuple[float, ...], tuple[float, float, float], tuple[float, float, float], float]:
    """回傳離查表座標最近的 `(joints_pan0, direction_local,
    local_tip_position, speed_per_unit_omega)`——用法跟
    `cue_pose_calculator.lookup_roll_rad()` 完全一樣（最近鄰查表，不是
    連續公式）。"""
    _, joints_pan0, direction_local, local_tip_position, speed_per_unit_omega = min(
        _PLACEMENT_LOOKUP_GRID, key=lambda row: abs(row[0] - cue_ball_y)
    )
    return joints_pan0, direction_local, local_tip_position, speed_per_unit_omega


def compute_flat_base_position_and_joint_targets(
    target_wrist_position: tuple[float, float, float],
    shot_angle_deg: float,
) -> tuple[tuple[float, float, float], list[float]]:
    """flat 案例（tilt_rad≈0）：直接用 `_FLAT_PLACEMENT`，`shoulder_pan`
    由 `shot_angle_deg` 解析算出（`compute_tilted_direction(shot_angle_deg,
    0)` 的方向），不用查表。"""
    theta = math.radians(shot_angle_deg)
    target_direction = (-math.sin(theta), math.cos(theta), 0.0)
    joints_pan0, direction_local, local_tip_position, _speed_per_unit_omega = _FLAT_PLACEMENT
    return _solve_base_position_and_joint_targets(
        target_wrist_position, target_direction, joints_pan0, direction_local, local_tip_position
    )


def compute_flat_target_elbow_velocity(cue_ball_speed: float) -> float:
    """flat 案例：`required_tip_speed / speed_per_unit_omega`，見
    `_FLAT_SPEED_PER_UNIT_OMEGA` 說明。跟 `compute_bridge_target_elbow_
    velocity()` 分開兩個函式（不是同一個查表 key），因為 `_FLAT_PLACEMENT`
    本身就不查表（跟母球位置無關）。"""
    from . import swing_trajectory_calculator

    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(cue_ball_speed)
    return required_tip_speed / _FLAT_SPEED_PER_UNIT_OMEGA


def compute_bridge_target_elbow_velocity(cue_ball_speed: float, cue_ball_y: float) -> float:
    """高架橋案例：跟 `compute_bridge_base_position_and_joint_targets()`
    一樣依 `cue_ball_y` 查表決定 `speed_per_unit_omega`。"""
    from . import swing_trajectory_calculator

    _, _, _, speed_per_unit_omega = lookup_placement_constants(cue_ball_y)
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(cue_ball_speed)
    return required_tip_speed / speed_per_unit_omega


def compute_bridge_base_position_and_joint_targets(
    target_wrist_position: tuple[float, float, float],
    target_direction: tuple[float, float, float],
    cue_ball_y: float,
) -> tuple[tuple[float, float, float], list[float]]:
    """高架橋案例（tilt_rad>0）：`target_direction` 直接沿用呼叫端已經算好
    的 `cue_pose_calculator.compute_tilted_direction(shot_angle_deg,
    tilt_rad)`（跟這個模組的幾何計算共用同一個事實來源，不重複實作），
    姿態依 `cue_ball_y` 查表決定。"""
    joints_pan0, direction_local, local_tip_position, _speed_per_unit_omega = lookup_placement_constants(cue_ball_y)
    return _solve_base_position_and_joint_targets(
        target_wrist_position, target_direction, joints_pan0, direction_local, local_tip_position
    )


def _solve_base_position_and_joint_targets(
    target_wrist_position: tuple[float, float, float],
    target_direction: tuple[float, float, float],
    joints_pan0: tuple[float, ...],
    direction_local: tuple[float, float, float],
    local_tip_position: tuple[float, float, float],
) -> tuple[tuple[float, float, float], list[float]]:
    """核心公式，`compute_flat_base_position_and_joint_targets()`／
    `compute_bridge_base_position_and_joint_targets()` 共用。

    `shoulder_pan` 解析解：繞基座 Z 軸旋轉只改變 `direction_local`／
    `local_tip_position` 的 XY 分量方向角，不改變 Z 分量或量值（見
    `scripts/test_elevated_bridge_ur3e_table.py` Stage 2 的推導與實測
    `dot=0.99999` 驗證），所以只要算「目前 XY 方向角」跟「目標 XY 方向角」
    的差角即可，不需要數值迭代或現場讀 Jacobian。

    `base_position`：純平移（不旋轉，見 `RobotArm.reposition()` 介面）。
    `local_tip_position` 是 **`shoulder_pan=0`** 時基座在原點的桿尖世界
    座標——`shoulder_pan` 一旦轉成 `required_pan`，桿尖相對基座的位置
    也會跟著繞基座 Z 軸轉動同一個角度（跟 `direction_local` 的 XY 分量
    會被同一個旋轉重新定向是同一件事，Z 分量不變、XY 量值不變），所以
    不能直接拿 `shoulder_pan=0` 量到的 `local_tip_position` 去減，必須
    先繞 Z 軸把它轉到 `required_pan` 才是「实際下達 `joint_targets` 後」
    桿尖相對基座的真正位置，兩者相減才是「要把基座移到哪裡，桿尖才會
    剛好落在目標點」——跟 `base_placement_calculator.compute_base_pose()`
    的 `base_x = grip_x - _LOCAL_TIP_RADIUS*dx` 同一個代數結構，只是這裡
    用向量/旋轉矩陣表示、沒有把 XY/Z 拆開寫死三角函數公式（UR3e 沒有
    WAM7 那樣單一固定姿態的解析半徑/高度常數可以拆）。
    """
    direction_local_arr = np.asarray(direction_local, dtype=float)
    target_direction_arr = np.asarray(target_direction, dtype=float)

    angle_current_xy = math.atan2(direction_local_arr[1], direction_local_arr[0])
    angle_target_xy = math.atan2(target_direction_arr[1], target_direction_arr[0])
    required_pan = angle_target_xy - angle_current_xy
    required_pan = (required_pan + math.pi) % (2 * math.pi) - math.pi

    joint_targets = [required_pan, *joints_pan0]

    local_tip_position_arr = np.asarray(local_tip_position, dtype=float)
    cos_p, sin_p = math.cos(required_pan), math.sin(required_pan)
    rotated_tip_position = np.array([
        local_tip_position_arr[0] * cos_p - local_tip_position_arr[1] * sin_p,
        local_tip_position_arr[0] * sin_p + local_tip_position_arr[1] * cos_p,
        local_tip_position_arr[2],
    ])

    base_position_arr = np.asarray(target_wrist_position, dtype=float) - rotated_tip_position
    base_position = (float(base_position_arr[0]), float(base_position_arr[1]), float(base_position_arr[2]))

    return base_position, joint_targets
