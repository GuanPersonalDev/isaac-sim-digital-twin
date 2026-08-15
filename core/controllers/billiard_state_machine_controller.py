from abc import abstractmethod

from .controller_base import ControllerBase
from ..models.action import Action
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation

class BilliardStateMachineController(ControllerBase):
    """
    撞球流程 RESET -> IDLE -> AIMING -> STRIKING -> WAITING -> RESET

    只有 IDLE 與 AIMING 需要決定「要打什麼」，交由子類別實作；其餘狀態的轉換
    條件與 no-op Action 格式是 ScriptController 與 ModelController 共用的契約，
    不開放覆寫（兩端呼叫路徑的一致性測試依賴這份契約，見 #228）。
    """

    def __init__(self):
        self._current_state = BilliardStatus.RESET

    def _change_state(self, new_state: BilliardStatus):
        self._current_state = new_state

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

    @abstractmethod
    def _idle_state_action_result(self, observation: Observation) -> Action:
        """
        球已擺好且靜止時進入 AIMING，回傳的 Action 會被送進 _execute_aim()。
        """
        ...

    @abstractmethod
    def _aiming_state_action_result(self, observation: Observation) -> Action:
        """
        下游動作完成時進入 STRIKING，回傳的 Action 會被送進 _execute_strike()，
        六維擊球參數在這一次分派才真正變成物理量。
        """
        ...

    def _error_state_action_result(self) -> Action:
        result = self._generate_action_result()
        self._change_state(BilliardStatus.ERROR)
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
        self._on_reset()

    def _on_reset(self) -> None:
        """
        子類別清除自己的暫存，預設無事可做。
        """
        ...
