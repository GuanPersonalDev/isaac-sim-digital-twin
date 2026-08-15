from .billiard_state_machine_controller import BilliardStateMachineController
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.action import Action
from ..models.action_bounds import CUE_BALL_SPEED


class ScriptController(BilliardStateMachineController):
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
