import logging
from abc import ABC, abstractmethod
from tkinter import W

from ..services.base_placement_calculator import (
    CANONICAL_REST_JOINTS, compute_base_pose, compute_canonical_wrist_position, required_grip_position,
)
from ..services import cue_pose_calculator, swing_trajectory_calculator
from ..controllers.controller_base import ControllerBase
from ..models.action import Action
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.table_ball_set import TableBallSet
from ..ports.articulation_api import ArticulationAPI
from ..models.robot_arm import RobotArm
from .ball_motion_monitor import BallMotionMonitor
from .ball_position_provider import BallPositionProvider
from .impulse_striking_service import ImpulseStrikingService
from .error_state import ErrorState
from .rolling_resistance_service import RollingResistanceService

logger = logging.getLogger(__name__)


class TableOrchestrator(ABC):
    """
    共用執行骨架：取得 Action → 依 ControllerBase.get_current_state() 分派下游動作
    → 查詢球是否還在移動 → 查詢下游動作是否完成 → 組裝下一個 tick 的 Observation。
    差異部分（RESET 的手臂處理、AIMING/STRIKING 的實際動作、動作完成判定）交由子類別實作。
    """

    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService
    ) -> None:
        self._script_controller = script_controller
        self._table_ball_set = table_ball_set
        self._ball_position_provider = ball_position_provider
        self._error_state = error_state
        self._rolling_resistance_service = rolling_resistance_service

    def step(self, observation: Observation) -> None:
        """
        每個 tick 呼叫一次的共用骨架，見 docs/tech-design-5-9-table-orchestrator.md 第 3 節、
        docs/tech-design/rolling-resistance-correction-tech-design.md 第 4 節。
        """
        self._rolling_resistance_service.apply(self._table_ball_set.get_ball_prim_paths())

        action = self._script_controller.get_action(observation)
        current_state = self._script_controller.get_current_state()
        
        if action.should_execute_action:
            try:
                match current_state:
                    case BilliardStatus.RESET:
                        self._reset_balls()
                        self._reset_downstream()
                    case BilliardStatus.AIMING:
                        self._execute_aim(action)
                    case BilliardStatus.STRIKING:
                        self._execute_strike(action)
            except Exception as e:
                self._error_state.mark_error(e)


    def _reset_balls(self) -> None:
        """
        共用：呼叫 table_ball_set.reset(positions)，positions 來自 ball_position_provider。
        Teleport 語意，呼叫完當下即完成。
        """
        positions = self._ball_position_provider.get_positions()
        self._table_ball_set.reset(positions)
        
    def reset(self) -> None:
        """
        外部重新初始化入口，必須同時清除 error_state 與重置狀態機：
        ControllerBase.get_action() 的具體實作可能優先處理 has_error，
        只清一邊會讓狀態機瞬間又跳回 ERROR。
        """
        self._error_state.clear()
        self._script_controller.reset()

    def get_current_state(self) -> BilliardStatus:
        return self._script_controller.get_current_state()

    @abstractmethod
    def _reset_downstream(self) -> None:
        """下游（手臂等）reset """
        ...

    @abstractmethod
    def _execute_aim(self, action: Action) -> None:
        """AIMING 狀態下游動作，內容留給 #96"""
        ...

    @abstractmethod
    def _execute_strike(self, action: Action) -> None:
        """STRIKING 狀態下游動作，內容留給 #97（Demo）／#177（Training）"""
        ...

class DemoTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        robot_arm: RobotArm,
        articulation_api: ArticulationAPI,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider, error_state, rolling_resistance_service)
        self._robot_arm = robot_arm
        self._articulation_api = articulation_api

    def _reset_downstream(self) -> None:
        self._robot_arm.reset()

    def _execute_aim(self, action: Action) -> None:
        # 1. table_z / ball_radius 從 self._table_ball_set 取得；
        #    cue_ball = (action.cue_ball_placement[0], action.cue_ball_placement[1])
        # 2. compute_base_pose(cue_ball_x, cue_ball_y, shot_angle_deg, table_z)
        #    -> (base_position, base_yaw_rad)
        # 3. cue_pose_calculator.compute_tilted_wrist_pose(cue_ball, shot_angle,
        #    table_z, ball_radius, position_offset) -> (_, _, tilt_rad, crossing)
        #    tilt_rad is None：幾何無解（即使垂直抬高也無法閃避庫邊），raise ValueError
        table_z = self._table_ball_set.get_table_z()
        ball_radius = self._table_ball_set.get_ball_radius()
        cue_ball = (action.cue_ball_placement[0], action.cue_ball_placement[1])
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
        #   grip_position = required_grip_position(cue_ball_x, cue_ball_y, shot_angle_deg)
        #   joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
        #   self._robot_arm.reposition(base_position)
        #   self._articulation_api.move_to_joint_position(
        #       joint_targets, [grip_position[0], grip_position[1], table_z + ball_radius]
        #   )
        #   return
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
        # （A→B→C1→C2），一次呼叫 move_through_poses() 涵蓋整條鏈。
        #   self._robot_arm.reposition(base_position)
        #   safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
        #   safe_target_position = compute_canonical_wrist_position(base_position, 0.0)
        #   （不能沿用移動前的舊位置當佔位符——那對應的是移動前的姿態，不是
        #    safe_joint_targets 真正會到達的位置，會讓 is_motion_complete()
        #    永遠等不到收斂）
        #   bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
        #       safe_target_position, cue_ball, action.shot_angle, table_z, ball_radius,
        #       position_offset=action.position_offset,
        #   )
        #   bridge_waypoints is None：高架橋姿態無解，raise ValueError
        #   self._articulation_api.move_through_poses(
        #       bridge_waypoints,
        #       preceding_joint_targets=(safe_joint_targets, safe_target_position),
        #   )
        self._robot_arm.reposition(base_position)
        safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
        safe_target_position = compute_canonical_wrist_position(base_position, 0.0)
        bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
            [safe_target_position[0], safe_target_position[1], safe_target_position[2]], cue_ball, action.shot_angle, table_z, ball_radius, position_offset=action.position_offset
        )
        if bridge_waypoints is None:
            raise ValueError("高架橋姿態無解")
        self._articulation_api.move_through_poses(
            bridge_waypoints,
            preceding_joint_targets=(safe_joint_targets, [safe_target_position[0], safe_target_position[1], safe_target_position[2]])
        )

    def _execute_strike(self, action: Action) -> None:
        # 上一個動作（瞄準）若逾時未收斂
        if self._articulation_api.did_last_motion_timeout():
            raise RuntimeError("瞄準動作逾時未收斂")

        cue_ball = (action.cue_ball_placement[0], action.cue_ball_placement[1])
        table_z = self._table_ball_set.get_table_z()
        ball_radius = self._table_ball_set.get_ball_radius()

        wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset)
        if tilt_rad is None or wrist_position is None or wrist_orientation is None:
            raise ValueError("幾何無解（即使垂直抬高也無法閃避庫邊）")

        direction_unit = cue_pose_calculator.compute_tilted_direction(action.shot_angle, tilt_rad)
        waypoints = swing_trajectory_calculator.compute_swing_waypoints(
            contact_position=list(wrist_position),
            contact_orientation=list(wrist_orientation),
            direction_unit=direction_unit, cue_ball_speed=action.cue_ball_speed
        )
        self._articulation_api.move_through_poses(waypoints)

class TrainingTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        impulse_striking_service: ImpulseStrikingService,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider, error_state, rolling_resistance_service)
        self._impulse_striking_service = impulse_striking_service

    def _reset_downstream(self) -> None:
        """
        目前沒有其他需要處理的 reset 元件
        """
        pass

    def _execute_aim(self, action: Action) -> None:
        """
        Training Table 沒有手臂，隨時可以準備好擊球
        """
        pass


    def _execute_strike(self, action: Action) -> None:
        table_x, table_y = self._table_ball_set.get_table_x_y()
        table_z = self._table_ball_set.get_table_z()
        self._impulse_striking_service.strike(action, table_x, table_y, table_z=table_z)
