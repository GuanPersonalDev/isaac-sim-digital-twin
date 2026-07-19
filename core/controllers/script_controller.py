from .controller_base import ControllerBase
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.action import Action


class ScriptController(ControllerBase):
    MAX_ARM_SPEED = 1.313

    def __init__(self) -> None:
        self._current_state = BilliardStatus.RESET

    def _change_state(self, status: BilliardStatus):
        self._current_state = status


    def get_action(self, observation: Observation) -> Action:
        if observation.has_error:
            return self._error_state_action_result()

        match self._current_state:
            case BilliardStatus.IDLE:
                return self._idle_state_action_result(observation)
            case BilliardStatus.AIMING:
                return self._aiming_state_action_result(observation)
            case BilliardStatus.STRIKING:
                return self._striking_state_action_result(observation)
            case BilliardStatus.WAITING:
                return self._waiting_state_action_result(observation)
            case BilliardStatus.RESET:
                return self._reset_state_action_result(observation)
            case BilliardStatus.ERROR:
                return self._error_state_action_result()
                
    def _error_state_action_result(self) -> Action:
        result = self._generate_action_result()
        self._change_state(BilliardStatus.ERROR)
        return result
    
    def _idle_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if observation.is_init_state and not observation.is_ball_moving:
            result.should_control_articulation = True
            self._change_state(BilliardStatus.AIMING)
        
        return result
    
    def _aiming_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if observation.is_motion_complete:
            result.cue_speed = self.MAX_ARM_SPEED
            result.should_control_articulation = True
            self._change_state(BilliardStatus.STRIKING)
        
        return result
    
    def _striking_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if observation.is_motion_complete:
            result.should_control_articulation = True
            self._change_state(BilliardStatus.WAITING)
        
        return result
    
    def _waiting_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if not observation.is_ball_moving:
            result.should_control_articulation = True
            self._change_state(BilliardStatus.RESET)
        return result
    
    def _reset_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if observation.is_motion_complete:
            self._change_state(BilliardStatus.IDLE)
        return result
        
    def _generate_action_result(self) -> Action:
        return Action(cue_speed=0, position_offset=[0, 0, 0], shot_angle=0, should_control_articulation=False)

    def reset(self):
        self._change_state(BilliardStatus.RESET)