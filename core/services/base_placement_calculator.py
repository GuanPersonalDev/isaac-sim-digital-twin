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

`CANONICAL_REST_JOINTS`／`_LOCAL_TIP_RADIUS`／`_LOCAL_TIP_HEIGHT` 的
shoulder/elbow 部分由 `scripts/probe_canonical_pose.py` 手動試誤量出；
wrist_pitch／palm_yaw 部分由 `scripts/probe_palm_yaw_correction.py` 網格
搜尋量出（見下方「2026-08-20 wrist_pitch/palm_yaw 修正」）。改資產或重新
調校這組關節角度時，必須重新跑這兩支腳本量測，不能手動猜數字。

已驗證（`scripts/validate_fixed_pose_placement.py`，Kitchen 兩個代表角落，
位置精度部分；`scripts/probe_palm_yaw_correction.py`，指向與傾斜角部分）：
- 桿尖水平位置隨 base_yaw 同步旋轉（base_yaw=+0.3 rad 時桿尖方向角同步偏轉
  +0.3 rad）
- 實際掛上 `ball_stick.usda` 後（`align_prim_to_target` 對齊腕部），公式算出
  的基座位置與 `base_yaw` 讓桿尖世界座標對上 `required_grip_position()` 的
  需求點，XY 誤差 <0.05mm、Z 誤差 <0.02mm
- 球桿水平指向角偏差 <2°（在 `docs/WAM_IK_implementation_and_verification.md`
  記載的 3–5° 方向誤差容許值內），離水平面傾斜角 <0.3°

⚠️ `validate_fixed_pose_placement.py` 本身的傾斜角檢查（`arcsin(abs(z 分量))`）
對繞垂直軸轉幾度是不敏感的，轉 90° 一樣會通過，**檢查不到水平面內的指向偏
差**——下面這段修正就是這樣被 `validate_fixed_pose_placement.py` 的「已驗
證」漏掉的，之後改這組常數要額外用 `probe_palm_yaw_correction.py` 或等效
方法核對指向角，不能只看 `validate_fixed_pose_placement.py` 過了就放心。

**2026-08-20 wrist_pitch/palm_yaw 修正**：2026-08-19 修 `ball_stick.usda` 的
FixedJoint 之後用實機 log 量到：原本 `wrist_pitch=palm_yaw=0` 時，`base_yaw=0`
的球桿實際指向 +Y，但手腕徑向參考方向是 +X，兩者差 90 度，是 `CANONICAL_
REST_JOINTS` 這組姿態裡「手腕位置」與「球桿指向」之間內建的固定夾角，不是
`base_yaw_rad = shot_angle_rad + math.pi/2` 這條算式的錯（曾經誤判拿掉這個
`+π/2` 修正項，實測證明會讓手腕位置整個算錯，xy_error 從 <0.05mm 惡化到
0.5m，已改回來）。用 `scripts/probe_palm_yaw_correction.py` 對 `(wrist_pitch,
palm_yaw)` 網格搜尋後粗定位、再細搜尋，找到 `wrist_pitch=-32°、palm_yaw=+86°`
這組合能把指向偏差壓到 <2°、傾斜角壓到 <0.3°，且已同步重新量測
`_LOCAL_TIP_RADIUS`／`_LOCAL_TIP_HEIGHT`（wrist_pitch 改變會移動手腕位置，
不像 palm_yaw 對位置沒有影響，這兩個常數不能沿用舊值）。

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

CANONICAL_REST_JOINTS = (1.9, 0.0, 1.8, 0.0, -0.5585, 1.5010)
"""固定姿態的 6 個關節目標（rad），依 `assets/barrett_wam/wam7.urdf` 的順序：
`(shoulder_pitch, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_pitch, palm_yaw)`
——不含 `base_yaw`（那個關節每次擊球都要重算，見 `compute_base_pose()`）。

shoulder_pitch/shoulder_yaw/elbow_pitch/wrist_yaw 由 `scripts/probe_canonical_pose.py`
手動試誤選出：shoulder_pitch 留了 0.085 rad 的限位餘裕（限位 1.985），在這個
前提下桿尖盡量往下、往內收，避免手臂在預設「完全伸直朝上」姿態附近卡住。
不是理論最優解，是可行解。

2026-08-20 追加：原本 wrist_pitch=palm_yaw=0 時，手腕位置的參考方向（+X）
跟球桿實際指向（+Y）差了 90 度（見本檔案開頭「曾經誤判」段落）。用
`scripts/probe_palm_yaw_correction.py` 網格搜尋 `(wrist_pitch, palm_yaw)`
找到 `wrist_pitch=-32°(-0.5585 rad)、palm_yaw=+86°(1.5010 rad)`：這組合在
base_yaw=0 時球桿指向偏差縮到 1.25°（在 WAM_IK_implementation_and_verification.md
記載的 3–5° 方向誤差容許值內）、離水平面傾斜 0.22°（跟原本 <0.05° 同量級）。
改這組關節角時，`_LOCAL_TIP_RADIUS`／`_LOCAL_TIP_HEIGHT` 兩個常數必須同步
重新量測（wrist_pitch 改變會移動手腕位置，不像 palm_yaw 對位置沒有影響），
已經一起更新，不能只改其中一半。
"""

_LOCAL_TIP_RADIUS = 0.38511
"""base_yaw=0 時，桿尖到 base_yaw 轉軸的水平距離（m）。"""

_LOCAL_TIP_HEIGHT = 0.78731
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


def compute_joint_targets(shot_angle_deg: float) -> list[float]:
    """完整 7-DOF joint-space 位置目標，依 `assets/barrett_wam/wam7.urdf` 的
    關節順序：`[base_yaw, shoulder_pitch, shoulder_yaw, elbow_pitch, wrist_yaw,
    wrist_pitch, palm_yaw]`。

    直接對應 `scripts/validate_fixed_pose_placement.py` 驗證過的下達方式：
    `articulation.switch_dof_control_mode("position")` 後
    `set_dof_position_targets(...)`，跟既有 `move_to_home()` 走同一種機制，
    不需要跑差動 IK。

    只是資料層的純函式——尚未接進任何呼叫端（`TableRobotManager` 仍用固定的
    `_ROBOT_OFFSET_FROM_TABLE_CENTER`），要接的話還需要決定「逐球重新定位」
    要放在初始化流程的哪個時間點，這個決定本身列為 #180 第九節明文排除的範圍
    （本次不處理實際重新定位的機構或實作），留給後續 issue。
    """
    base_yaw_rad = math.radians(shot_angle_deg) + math.pi / 2.0
    return [base_yaw_rad, *CANONICAL_REST_JOINTS]
