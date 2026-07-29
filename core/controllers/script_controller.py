from .controller_base import ControllerBase
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.action import Action


class ScriptController(ControllerBase):
    # 訓練桌（impulse strike）路徑把這個值直接當成母球初速（見
    # core/services/impulse_striking_service.py 的 compute_cue_ball_velocities，
    # 目前是 1:1 直接賦值，沒有套用動量轉換）。數值來源：2026-07-26 換裝
    # Barrett WAM + 差動 IK 後實測桿尖峰值速度 2.5302 m/s（預設姿態，見
    # docs/phase3-task-breakdown.md 出桿速度範圍列的更新說明），套用真實
    # 撞球動量傳遞公式 v_ball = v_cue×(1+e)×M桿/(M桿+m球)（球桿 0.5kg、
    # 母球 0.163kg、皮革頭恢復係數 e=0.75，Dr. Dave Pool Info 引用範圍
    # 0.71–0.75）換算為母球初速上限：2.5302×1.75×0.5/0.663 ≈ 3.3392 m/s。
    # Demo 桌（真實揮桿）路徑若之後把 _execute_strike 接上
    # ArticulationAPI.execute_strike()，該處的 speed 參數語意是「桿尖速度」
    # 不是「母球速度」，屆時這個共用常數的語意需要重新檢視。
    MAX_CUE_BALL_SPEED = 3.3392

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
            result.cue_ball_speed = self.MAX_CUE_BALL_SPEED
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
