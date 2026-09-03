from ..models.action import Action
from ..models.robot_arm import RobotArm
from ..ports.articulation_api import ArticulationAPI
from . import cue_pose_calculator, ur3e_placement_calculator
from .robot_swing_strategy import RobotSwingStrategy

_UR3E_BACKSWING_SWEEP_RAD = 0.5236  # 30°，見 ur3e_placement_calculator.py 常數來源腳本的 _SWEEP_DEG


class Ur3eSwingStrategy(RobotSwingStrategy):
    """
    UR3e 專屬的瞄準/揮桿策略。

    搬移自 core/services/table_orchestrator.py 現有 _execute_aim_ur3e()/
    _execute_strike_ur3e() 的程式碼，邏輯本身不變（見階段 3 的「零行為
    變化」承諾）：
    - 瞄準：ur3e_placement_calculator.py 依 tilt_rad 分流 flat/bridge 兩套
      查表/計算邏輯，算出 base_position/joint_targets 後
      move_to_joint_position()
    - 揮桿：move_swing_elbow_pivot()（純/加權多關節 elbow-pivot，跟
      WAM7 的 move_swing() 是不同的控制策略）

    ⚠️ 這套設計已經證實在 UR3e 幾何下有結構性限制（manipulability
    ellipsoid 對齊問題，margin/align 無法同時達標，見對話紀錄），暫緩
    維護、不再被生產路徑呼叫（billiard_digital_twin.py 的
    _ROBOT_ARM_CLASS 已改指向新手臂）。保留這份程式碼是刻意決定（見
    決策 9），未來若要整理，方向是比照新架構接上專用出力機構，而不是
    繼續調整這套多關節驅動的參數。
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
        # 跟 WAM7 版的差異：沒有「Phase 0 安全姿態＋Cartesian waypoint
        # 序列」這個中繼步驟——UR3e 目前直接用單一 joint-space 動作瞄準
        # （跟 WAM7 flat 案例的做法一樣簡單），高架橋案例也是（WAM7 需要
        # 那套多階段序列是為了閃避差動 IK 在奇異點附近的失穩，UR3e 這裡
        # 完全不跑差動 IK，用不到）。
        #
        # ⚠️ 這代表「從手臂目前姿態安全接近到瞄準姿態」這一段沒有像 WAM7
        # 那樣被驗證過不會讓球桿掃過球檯/球——見
        # ArticulationAPIImpl.move_swing_elbow_pivot() docstring 同一個
        # 已知限制。
        wrist_position, _wrist_orientation, tilt_rad, _crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset
        )
        if tilt_rad is None:
            raise ValueError("幾何無解（即使垂直抬高也無法閃避庫邊）")

        if tilt_rad <= 1e-6:
            base_position, joint_targets = ur3e_placement_calculator.compute_flat_base_position_and_joint_targets(
                tuple(wrist_position), action.shot_angle
            )
        else:
            target_direction = cue_pose_calculator.compute_tilted_direction(action.shot_angle, tilt_rad)
            base_position, joint_targets = ur3e_placement_calculator.compute_bridge_base_position_and_joint_targets(
                tuple(wrist_position), tuple(target_direction), cue_ball[1]
            )

        self._robot_arm.reposition(base_position)
        self._articulation_api.move_to_joint_position(joint_targets, list(wrist_position))

    def execute_strike(
        self,
        action: Action,
        cue_ball: tuple[float, float],
        table_z: float,
        ball_radius: float,
    ) -> None:
        # 用 ArticulationAPIImpl.move_swing_elbow_pivot()（純肘關節轉動，
        # 見該方法 docstring）取代 WAM7 版的全關節線性規劃 move_swing()——
        # 兩者是不同的控制策略，UR3e 沒有驗證過 move_swing() 能不能達到
        # 同樣的目標速度，不能直接沿用 WAM7 那條路徑。
        #
        # 接觸姿態（contact_joint_targets）用跟 execute_aim() 相同的
        # cue_ball/shot_angle/tilt_rad 算出同一組 base_position／
        # joint_targets（AIM 已經把基座搬到這個位置，這裡不重複呼叫
        # self._robot_arm.reposition()）。後擺姿態只把 elbow_dof_index
        # 那個分量往回轉 _UR3E_BACKSWING_SWEEP_RAD（30°，來源見該常數），
        # 其餘分量跟接觸姿態相同——這是 move_swing_elbow_pivot() 的前提
        # 假設（見該方法 docstring）。
        wrist_position, _wrist_orientation, tilt_rad, _crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset
        )
        if tilt_rad is None or wrist_position is None:
            raise ValueError("幾何無解（即使垂直抬高也無法閃避庫邊）")

        if tilt_rad <= 1e-6:
            _base_position, contact_joint_targets = ur3e_placement_calculator.compute_flat_base_position_and_joint_targets(
                tuple(wrist_position), action.shot_angle
            )
            target_elbow_velocity = ur3e_placement_calculator.compute_flat_target_elbow_velocity(action.cue_ball_speed)
        else:
            target_direction = cue_pose_calculator.compute_tilted_direction(action.shot_angle, tilt_rad)
            _base_position, contact_joint_targets = ur3e_placement_calculator.compute_bridge_base_position_and_joint_targets(
                tuple(wrist_position), tuple(target_direction), cue_ball[1]
            )
            target_elbow_velocity = ur3e_placement_calculator.compute_bridge_target_elbow_velocity(
                action.cue_ball_speed, cue_ball[1]
            )

        elbow_dof_index = ur3e_placement_calculator.UR3E_ELBOW_DOF_INDEX
        backswing_joint_targets = list(contact_joint_targets)
        backswing_joint_targets[elbow_dof_index] -= _UR3E_BACKSWING_SWEEP_RAD

        # ⚠️ joint-space 動作的收斂判定只看關節角度、不看這個末端位置參數
        # （見 ArticulationAPIImpl._is_current_target_converged() 的
        # joint-space 分支說明），這裡沒有 UR3e 的解析 FK 可以精確算出
        # 後擺姿態對應的末端位置，用接觸位置近似——不影響收斂判定，只是
        # 少了一個精確的除錯用參考值。
        self._articulation_api.move_swing_elbow_pivot(
            backswing_joint_targets, list(wrist_position),
            list(contact_joint_targets), elbow_dof_index, target_elbow_velocity,
        )
