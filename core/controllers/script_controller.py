from .billiard_state_machine_controller import BilliardStateMachineController
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.action import Action
from ..models.action_bounds import CUE_BALL_SPEED
from ..services.break_shot_position_provider import BREAK_SHOT_POSITIONS


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
            # 開球母球位置固定為 BREAK_SHOT_POSITIONS[0]（BilliardTable 用同一份
            # 常數透過 BreakShotPositionProvider 實際把球放到這個位置，見
            # core/models/billiard_table.py）——這裡沒有一起覆寫的話，
            # cue_ball_placement 會停留在 _generate_action_result() 的預設值
            # [0, 0]（桌台中心，甚至不在合法 Kitchen 範圍內），導致
            # _execute_aim()/_execute_strike() 拿錯誤的錨點算出整組跟真實母球
            # 位置對不上的機器人姿態。
            result.cue_ball_placement = list(BREAK_SHOT_POSITIONS[0])
            # 固定以物理域上限出桿；上下限的單一來源在 action_bounds（#114），
            # 此處不得硬編碼數值。cue_ball_speed 語意一直是「母球目標初速」；
            # Demo 桌把這個值換算成桿尖接觸速度的邏輯在
            # swing_trajectory_calculator.compute_required_tip_speed()，
            # ScriptController 不需要知道換算細節。
            result.cue_ball_speed = CUE_BALL_SPEED[1]
            result.should_execute_action = True
            self._change_state(BilliardStatus.STRIKING)

        return result
