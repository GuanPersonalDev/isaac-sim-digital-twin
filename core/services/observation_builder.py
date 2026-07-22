from abc import ABC, abstractmethod


from ..models.table_ball_set import TableBallSet
from ..models.observation import Observation
from ..models.ur5_robot import UR5Robot
from ..ports.rigid_body_api import RigidBodyAPI
from ..services.error_state import ErrorState
from .ball_motion_monitor import BallMotionMonitor
from .ball_position_provider import BallPositionProvider


class ObservationBuilder(ABC):
    _BALL_POS_TOLERANCE = 0.005
    def __init__(self, table_ball_set: TableBallSet, rigid_body_api: RigidBodyAPI, ball_motion_monitor: BallMotionMonitor, error_state: ErrorState, ball_position_provider: BallPositionProvider):
        self._table_ball_set = table_ball_set
        self._rigid_body_api = rigid_body_api
        self._ball_motion_monitor = ball_motion_monitor
        self._error_state = error_state
        self._ball_position_provider = ball_position_provider
        
    def build(self) -> Observation:

        ball_positions, is_init_state = self._ball_pos_check()
        cue_ball_position = ball_positions[0]

        is_ball_moving = self._ball_motion_monitor.is_any_ball_moving()
        is_motion_complete = self._is_motion_complete()
        has_error = self._error_state.has_error()
        return Observation(ball_positions=ball_positions,
                             cue_ball_position=cue_ball_position,
                             is_init_state=is_init_state,
                             is_ball_moving=is_ball_moving,
                             is_motion_complete=is_motion_complete,
                             has_error=has_error)

    def _ball_pos_check(self):
        ball_positions = []

        is_init_state = True
        default_pos = self._ball_position_provider.get_positions()
        ball_index = 0
        table_x, table_y = self._table_ball_set.get_table_x_y()

        for prim_path in self._table_ball_set.get_ball_prim_paths():
            world_pos = self._rigid_body_api.get_position(prim_path)
            ball_positions.append(world_pos)
            
            relative_table_pos_x = world_pos[0] - table_x
            relative_table_pos_y = world_pos[1] - table_y
            default_pos_x, default_pos_y = default_pos[ball_index]
            dx = relative_table_pos_x - default_pos_x
            dy = relative_table_pos_y - default_pos_y
            if is_init_state and (dx**2 + dy**2) >= self._BALL_POS_TOLERANCE**2:
                is_init_state = False
            ball_index += 1

        return ball_positions, is_init_state
       
    @abstractmethod
    def _is_motion_complete(self) -> bool:
        """
        動作是否完成
        """
        ...
       
class DemoTableObservationBuilder(ObservationBuilder):
    def __init__(self, table_ball_set, rigid_body_api, ball_motion_monitor, error_state, ball_position_provider, ur5_robot: UR5Robot):
        super().__init__(table_ball_set, rigid_body_api, ball_motion_monitor, error_state, ball_position_provider)
        self._robot = ur5_robot

    def _is_motion_complete(self):
        return self._robot.is_reset_complete()

class TrainingTableObservationBuilder(ObservationBuilder):
    def __init__(self, table_ball_set, rigid_body_api, ball_motion_monitor, error_state, ball_position_provider):
        super().__init__(table_ball_set, rigid_body_api, ball_motion_monitor, error_state, ball_position_provider)
        
    def _is_motion_complete(self):
        return True