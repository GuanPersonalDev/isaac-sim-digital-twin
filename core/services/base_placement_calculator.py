"""逐球基座位置反推：由 Action 前三維（母球擺位＋瞄準角）算出 WAM7 基座座標
與固定姿態的關節目標。

見 docs/issue-180-reachability-analysis.md 第九節。fallback (b) 假設偏移固定
0，桿尖擊球點簡化為母球擺位點本身（忽略球半徑量級的誤差，跟該文件第四節同一
套簡化一致）。

## 設計：固定姿態＋基座位移/旋轉，取代逐次即時 IK

差動 IK（`ArticulationAPIImpl`）從機器人預設姿態（全關節 0，末端執行器朝正
上方完全伸直）收斂到 Kitchen 需求點，實測會在接近工作空間邊界時失穩、卡在
遠離目標的位置（見 #233 探測紀錄，`scripts/probe_base_reachability.py`）。
改用「固定姿態」徹底避開這個問題：手臂 6 個關節（除了 base_yaw）永遠鎖定在
同一組經驗證可行的角度，只有 base_yaw 這個關節隨瞄準角變化——不需要 runtime
IK 收斂，是普通的 joint-space 位置控制（跟 `move_to_home()` 同一種機制）。

`CANONICAL_REST_JOINTS`／`_LOCAL_TIP_RADIUS`／`_LOCAL_TIP_HEIGHT` 由
`scripts/probe_canonical_pose.py` 手動試誤量出：base 放在世界原點、
base_yaw=0 時，量出桿尖世界座標 (0.353, 0.000, 0.796)。改資產或重新調校
這組關節角度時，必須重新跑那支腳本量測，不能手動猜數字。

已驗證（`scripts/validate_fixed_pose_placement.py`，Kitchen 兩個代表角落）：
- 桿尖水平位置隨 base_yaw 同步旋轉（base_yaw=+0.3 rad 時桿尖方向角同步偏轉
  +0.3 rad）
- 實際掛上 `ball_stick.usda` 後（`align_prim_to_target` 對齊腕部），公式算出
  的基座位置與 `base_yaw` 讓桿尖世界座標對上 `required_grip_position()` 的
  需求點，XY 誤差 <0.05mm、Z 誤差 <0.02mm
- 桿身傾斜角 <0.05°，實質上完全水平，不需要額外修正

只處理 fallback (b)：偏移固定 0，桿身水平躺平沿瞄準方向。`base_yaw` 目標值
落在機器人 `wam_base_yaw_joint` 的限位 [-2.6, 2.6] rad 內才有效——這支公式
只覆蓋「瞄向球堆」的窄角錐（見文件開頭〈範圍限定〉），Milestone B 走位球需要
的整圈方向不在這支公式的覆蓋範圍內，需要另外設計（基座本身的旋轉，而不是
只用這一個關節）。
"""

import math

from ..models.table_ball_set import TableBallSet

CUE_STICK_GRIP_TO_TIP = 1.35
"""握把（CueStick 原點／FixedJoint 端）到桿尖的距離。

對應資產 `assets/ball_stick.usda` 的 `Cylinder xformOp:translate=(0,0.6,0)`、
`height=1.5`：桿尾露出 0.15m，符合真實球桿握姿比例（見
issue-180-reachability-analysis.md 第九節）。改資產時這個常數要同步更新。
"""

CANONICAL_REST_JOINTS = (1.9, 0.0, 1.8, 0.0, 0.0, 0.0)
"""固定姿態的 6 個關節目標（rad），依 `assets/barrett_wam/wam7.urdf` 的順序：
`(shoulder_pitch, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_pitch, palm_yaw)`
——不含 `base_yaw`（那個關節每次擊球都要重算，見 `compute_base_pose()`）。

由 `scripts/probe_canonical_pose.py` 手動試誤選出：shoulder_pitch 留了
0.085 rad 的限位餘裕（限位 1.985），在這個前提下桿尖盡量往下、往內收，
避免手臂在預設「完全伸直朝上」姿態附近卡住。不是理論最優解，是可行解。
"""

_LOCAL_TIP_RADIUS = 0.35342
"""base_yaw=0 時，桿尖到 base_yaw 轉軸的水平距離（m）。"""

_LOCAL_TIP_HEIGHT = 0.79640
"""base_yaw=0、基座 z=0 時，桿尖的世界 Z 高度（m）。"""


def _aim_direction(shot_angle_deg: float) -> tuple[float, float]:
    """瞄準方向單位向量。0°朝桌台 +Y，正角朝 -X（見 action_bounds.py SHOT_ANGLE）。"""
    theta = math.radians(shot_angle_deg)
    return (-math.sin(theta), math.cos(theta))


def required_grip_position(
    cue_ball_x: float, cue_ball_y: float, shot_angle_deg: float
) -> tuple[float, float]:
    """握把（= end-effector）需求位置：擊球點沿瞄準反方向退開 `CUE_STICK_GRIP_TO_TIP`。"""
    dx, dy = _aim_direction(shot_angle_deg)
    return (
        cue_ball_x - CUE_STICK_GRIP_TO_TIP * dx,
        cue_ball_y - CUE_STICK_GRIP_TO_TIP * dy,
    )


def compute_base_pose(
    cue_ball_x: float,
    cue_ball_y: float,
    shot_angle_deg: float,
    table_z: float,
    ball_radius: float = TableBallSet.DEFAULT_BALL_RADIUS,
) -> tuple[tuple[float, float, float], float]:
    """由 Action 前三維反推 WAM7 基座座標與 `base_yaw` 關節目標（桌台相對座標）。

    回傳 `(base_position, base_yaw_rad)`：
    - `base_position`：`(x, y, z)`，餵給 `BarrettWamRobot` 建構子的 `position`。
    - `base_yaw_rad`：`wam_base_yaw_joint` 的 joint-space 位置目標，跟
      `CANONICAL_REST_JOINTS` 一起下給 articulation（`switch_dof_control_mode
      ("position")` + `set_dof_position_targets`），不需要跑差動 IK。

    只處理 fallback (b)：偏移固定 0，桿身水平躺平。
    """
    dx, dy = _aim_direction(shot_angle_deg)
    grip_x, grip_y = required_grip_position(cue_ball_x, cue_ball_y, shot_angle_deg)
    grip_z = table_z + ball_radius

    base_yaw_rad = math.radians(shot_angle_deg) + math.pi / 2.0

    base_x = grip_x - _LOCAL_TIP_RADIUS * dx
    base_y = grip_y - _LOCAL_TIP_RADIUS * dy
    base_z = grip_z - _LOCAL_TIP_HEIGHT

    return ((base_x, base_y, base_z), base_yaw_rad)
