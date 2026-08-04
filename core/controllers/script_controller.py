from .controller_base import ControllerBase
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.action import Action
from ..models.action_bounds import CUE_BALL_SPEED


class ScriptController(ControllerBase):
    def __init__(self) -> None:
        self._current_state = BilliardStatus.RESET

    def _change_state(self, status: BilliardStatus):
        self._current_state = status

    def get_current_state(self) -> BilliardStatus:
        return self._current_state


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
            result.should_execute_action = True
            self._change_state(BilliardStatus.AIMING)
        
        return result
    
    def _aiming_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if observation.is_motion_complete:
            # 固定以物理域上限出桿；上下限的單一來源在 action_bounds（#114），
            # 此處不得硬編碼數值。
            # Demo 桌（真實揮桿）路徑若之後把 _execute_strike 接上
            # ArticulationAPI.execute_strike()，該處的 speed 參數語意是
            # 「桿尖速度」不是「母球速度」，屆時這個共用常數的語意需要重新檢視。
            result.cue_ball_speed = CUE_BALL_SPEED[1]
            result.should_execute_action = True
            self._change_state(BilliardStatus.STRIKING)
        
        return result
    
    def _striking_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if observation.is_motion_complete:
            result.should_execute_action = True
            self._change_state(BilliardStatus.WAITING)
        
        return result
    
    def _waiting_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if not observation.is_ball_moving:
            result.should_execute_action = True
            self._change_state(BilliardStatus.RESET)
        return result
    
    def _reset_state_action_result(self, observation: Observation) -> Action:
        result = self._generate_action_result()
        if observation.is_motion_complete:
            self._change_state(BilliardStatus.IDLE)
        return result
        
    def _generate_action_result(self) -> Action:
        return Action(
            cue_ball_placement=[0, 0],
            shot_angle=0,
            cue_ball_speed=0,
            position_offset=[0, 0],
            should_execute_action=False,
        )

    def reset(self):
        self._change_state(BilliardStatus.RESET)
