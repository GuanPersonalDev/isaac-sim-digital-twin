import numpy as np

from ..models.action import Action
from ..models.robot_arm import RobotArm
from ..ports.articulation_api import ArticulationAPI
from .base_placement_calculator import (
    CANONICAL_FLAT_ORIENTATION, CANONICAL_REST_JOINTS, compute_base_pose,
    compute_canonical_wrist_position, required_grip_position,
)
from . import cue_pose_calculator, swing_trajectory_calculator
from .robot_swing_strategy import RobotSwingStrategy


class Wam7SwingStrategy(RobotSwingStrategy):
    """
    WAM7 專屬的瞄準/揮桿策略。

    搬移自 core/services/table_orchestrator.py 現有 _execute_aim()/
    _execute_strike() 裡的 WAM7 分支，邏輯本身不變（見階段 3 的「零行為
    變化」承諾）：
    - flat 案例（tilt_rad<=1e-6）：CANONICAL_REST_JOINTS+base_yaw
      joint-space 直接到位
    - 高架橋案例（tilt_rad>0）：Phase 0 joint-space 回安全姿態 +
      compute_elevated_bridge_waypoints() 多階段 Cartesian waypoint 序列
    - 揮桿：move_swing()（全關節線性規劃，取代原本 move_through_poses()
      的兩段式呼叫）
    """

    def __init__(self, robot_arm: RobotArm, articulation_api: ArticulationAPI) -> None:
        self._robot_arm = robot_arm
        self._articulation_api = articulation_api

    def execute_aim(
        self,
        action: Action,
        cue_ball: tuple[float, float],
        table_z: float,
        ball_radius: float,
    ) -> None:
        # 1. compute_base_pose(cue_ball_x, cue_ball_y, shot_angle_deg, table_z)
        #    -> (base_position, base_yaw_rad)
        # 2. cue_pose_calculator.compute_tilted_wrist_pose(cue_ball, shot_angle,
        #    table_z, ball_radius, position_offset) -> (_, _, tilt_rad, crossing)
        #    tilt_rad is None：幾何無解（即使垂直抬高也無法閃避庫邊），raise ValueError
        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], action.shot_angle, table_z, ball_radius)
        wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset)
        if tilt_rad is None:
            raise ValueError("幾何無解（即使垂直抬高也無法閃避庫邊）")
        #
        # tilt_rad <= 1e-6：flat 案例（tilt=0）。沿用已驗證過的
        # CANONICAL_REST_JOINTS+base_yaw joint-space 路徑，不進差動 IK 管線
        # （見 docs/issue-flat-case-residual-error.md：差動 IK 建構出的姿態跟
        # CANONICAL_REST_JOINTS 實際 FK 姿態即使指向相同，roll 分量不保證
        # 一樣，會逼手臂多繞路）。這個分支目前不處理 position_offset（維持
        # 既有行為，跟 22/25 高架橋案例的偏移支援分開處理）。
        if tilt_rad <= 1e-6:
            grip_position = required_grip_position(cue_ball[0], cue_ball[1], action.shot_angle)
            joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
            self._robot_arm.reposition(base_position)
            self._articulation_api.move_to_joint_position(
                joint_targets, [grip_position[0], grip_position[1], table_z + ball_radius]
            )
            return
        #
        # tilt_rad > 0：高架橋案例。Phase 0（joint-space 回安全姿態避開差動
        # IK 奇異點，base_yaw 固定用 0.0，不是這次瞄準角的目標值——單純只是
        # 一個通用、跟瞄準角無關的安全起點）+ Cartesian waypoint 序列
        # （B1 爬升→B2 平移→C1 轉向→C2 下降），一次呼叫 move_through_poses()
        # 涵蓋整條鏈。
        self._robot_arm.reposition(base_position)
        safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
        safe_target_position = compute_canonical_wrist_position(base_position, 0.0)
        # Phase 0 的 base_yaw 固定用 0.0——CANONICAL_FLAT_ORIENTATION 是這組
        # 姿態下腕部的真實世界朝向（實測值，不是單位四元數！見該常數註解：
        # _shortest_arc_quat 構造出的單位四元數對 roll 沒有約束，跟真實姿態
        # 差了將近 180°，這正是高架橋 B1/B2 階段被迫多轉一圈、shoulder_pitch
        # 卡死的根因）。不能用 get_end_effector_orientation() 讀（跟
        # safe_target_position 同一個道理：Phase 0 這時候還沒真的執行，讀到
        # 的會是移動前的舊姿態）。
        safe_orientation = list(CANONICAL_FLAT_ORIENTATION)
        # C1 轉向時手臂本體可能掃過球檯庫邊/袋口，roll_rad 是用來閃避這個
        # 問題的自由度（不影響擊球結果）——lookup_roll_rad() 是離線掃描真實
        # Kitchen 網格找出的最近鄰查表，見該函式與
        # docs/issue-180-reachability-analysis.md 第十三節。execute_strike()
        # 必須用同一顆 cue_ball 座標查同一個值，才能保證瞄準跟擊球算出同一個
        # 目標姿態。
        roll_rad = cue_pose_calculator.lookup_roll_rad(cue_ball)
        backswing_distance_m = cue_pose_calculator.lookup_backswing_distance_m(cue_ball)
        bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
            [safe_target_position[0], safe_target_position[1], safe_target_position[2]], safe_orientation,
            cue_ball, action.shot_angle, table_z, ball_radius, position_offset=action.position_offset,
            roll_rad=roll_rad, backswing_distance_m=backswing_distance_m,
        )
        if bridge_waypoints is None:
            raise ValueError("高架橋姿態無解")
        self._articulation_api.move_through_poses(
            bridge_waypoints,
            preceding_joint_targets=(safe_joint_targets, [safe_target_position[0], safe_target_position[1], safe_target_position[2]])
        )

    def execute_strike(
        self,
        action: Action,
        cue_ball: tuple[float, float],
        table_z: float,
        ball_radius: float,
    ) -> None:
        # 先用 roll_rad=0 算一次只為了拿 tilt_rad 判斷 flat/bridge——roll 只
        #在高架橋 C1 轉向階段用來閃避手臂本體撞庫邊/袋口有意義，flat 案例
        # （tilt_rad<=1e-6）套用非零 roll 反而會讓 orientation 繞著水平桿身
        # 軸多轉一個跟原本行為無關的角度，所以 flat 案例維持 roll_rad=0，
        # 不查表。
        wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset
        )
        if tilt_rad is None or wrist_position is None or wrist_orientation is None:
            raise ValueError("幾何無解（即使垂直抬高也無法閃避庫邊）")

        if tilt_rad > 1e-6:
            # 高架橋案例：跟 execute_aim() 的高架橋分支用同一顆 cue_ball
            # 座標查同一個 roll_rad（見該處註解），保證瞄準跟擊球算出同一個
            # 目標姿態，重算一次拿套用 roll 後的正確 wrist/orientation。
            roll_rad = cue_pose_calculator.lookup_roll_rad(cue_ball)
            wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
                cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset, roll_rad=roll_rad
            )

        direction_unit = cue_pose_calculator.compute_tilted_direction(action.shot_angle, tilt_rad)
        # move_swing()（線性規劃求「姿態修正在有限額度內、沿揮桿方向最大化
        # 速度」）取代靜態目標點的 P 控制器+feedforward pose tracking——後者
        # 有結構性穩態誤差，隨揮終點永遠差一截到不了（見
        # docs/issue-180-reachability-analysis.md 第十五節）。
        # orientation_gain=1.0／max_angular_speed=1.0 沿用實測驗證過的數值
        # （比 move_swing() 的預設 max_angular_speed=0.5 更寬）。
        #
        # 高架橋案例的退桿距離跟 execute_aim() 的 AIM 收斂終點統一成同一個
        # 值（同一顆 cue_ball 座標查 lookup_backswing_distance_m()）；flat
        # 案例維持現狀，繼續用 DEFAULT_BACKSWING_DISTANCE_M（見
        # docs/issue-180-reachability-analysis.md 第十八節）。
        if tilt_rad > 1e-6:
            backswing_distance = cue_pose_calculator.lookup_backswing_distance_m(cue_ball)
        else:
            backswing_distance = swing_trajectory_calculator.DEFAULT_BACKSWING_DISTANCE_M

        required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(action.cue_ball_speed)
        follow_through_distance = swing_trajectory_calculator.compute_follow_through_distance(required_tip_speed)
        contact_position = np.array(wrist_position)
        backswing_position = swing_trajectory_calculator.compute_backswing_position(
            contact_position, direction_unit, backswing_distance
        )
        follow_through_position = contact_position + follow_through_distance * direction_unit
        self._articulation_api.move_swing(
            backswing_position.tolist(), list(wrist_orientation), follow_through_position.tolist(),
            orientation_gain=1.0, max_angular_speed=1.0,
        )
