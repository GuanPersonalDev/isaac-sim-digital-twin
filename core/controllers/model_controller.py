import logging

from .billiard_state_machine_controller import BilliardStateMachineController
from ..models.action import Action
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..ports.policy_port import PolicyPort
from ..services.numeric_validation import validate_2d_value, validate_max_offset
from ..services.rl_action_decoder import decode_rl_action
from ..services.rl_observation_encoder import encode_rl_observation

logger = logging.getLogger(__name__)


class ModelController(BilliardStateMachineController):
    """
    以訓練好的 RL policy 決定六維擊球參數，狀態機時序與 ScriptController 相同。

    推論固定在 IDLE -> AIMING 發生一次，AIMING -> STRIKING 重用同一份輸出：
    IDLE -> AIMING 當下球已排好且靜止，正是訓練時 policy 看到的觀測分布；重用
    則確保手臂瞄準（#96 之後的 _execute_aim）與實際擊球是同一個決策。

    table_position 是桌台的世界 XY，用來把 Observation 的世界座標換算成桌台
    相對座標。傳錯不會拋錯，policy 只會以為球在別的地方然後安靜輸出垃圾動作。

    max_offset 是可用偏移能力的比例，同時決定觀測第 21 維與偏移量的裁切半徑，
    只存一份餵兩處。評估固定 0.6（見 models/rl/billiard/README.md）。
    """

    def __init__(
        self,
        policy: PolicyPort,
        table_position: tuple[float, float],
        max_offset: float,
    ) -> None:
        super().__init__()
        self._policy = policy
        # 建構時就驗，壞值不會拖到 physics callback 執行中才炸
        self._table_position = validate_2d_value(list(table_position), "table_position")
        self._max_offset = validate_max_offset(max_offset)
        self._cached_raw_action: list[float] | None = None

    def _idle_state_action_result(self, observation: Observation) -> Action:
        """
        球已擺好且靜止時推論一次，快取原始六維輸出後進入 AIMING。

        流程：
        1. 條件未成立（未排好或還在動）就回 no-op，不推論。
        2. encode_rl_observation(observation, self._table_position, self._max_offset)
           取得 21 維觀測，餵給 self._policy.infer()。
        3. decode_rl_action(raw_action, self._max_offset, should_execute_action=True)
           還原成物理域 Action。
        4. 2 與 3 都要包在 try/except 內，失敗一律走 _enter_error_state()。
        5. 成功才寫入 self._cached_raw_action 並 _change_state(AIMING)。

        快取無條件覆寫即可：AIMING 只能從這個分支進入，STRIKING 讀到的絕不可能
        是上一局殘留的值。
        """
        if not observation.is_init_state or observation.is_ball_moving:
            return self._generate_action_result()

        try:
            raw_action = self._policy.infer(encode_rl_observation(observation, self._table_position, self._max_offset))
            action = decode_rl_action(raw_action, self._max_offset, should_execute_action=True)
        except Exception as e:
            return self._enter_error_state(e)
        
        self._cached_raw_action = raw_action
        self._change_state(BilliardStatus.AIMING)
        return action

    def _aiming_state_action_result(self, observation: Observation) -> Action:
        """
        下游動作完成時，用 IDLE 快取的那份輸出重新 decode 並進入 STRIKING。

        流程：
        1. is_motion_complete 未成立就回 no-op。
        2. self._cached_raw_action 為 None 代表時序被破壞，走 _enter_error_state()。
        3. 重新 decode_rl_action(should_execute_action=True) 後 _change_state(STRIKING)。

        重新 decode 而不是快取 Action 物件：Action 是 mutable dataclass，兩次分派
        共用同一個 instance 會被下游改到；should_execute_action 也必須由這次狀態
        轉換產生，不是模型的第 7 維輸出。
        """

        if not observation.is_motion_complete:
            return self._generate_action_result()
        
        if self._cached_raw_action is None:
            return self._enter_error_state(ValueError("ModelController 時序被破壞"))
        
        try:
            action = decode_rl_action(self._cached_raw_action, self._max_offset, should_execute_action=True)
        except Exception as e:
            return self._enter_error_state(e)
        
        self._change_state(BilliardStatus.STRIKING)
        return action

    def _enter_error_state(self, exception: Exception) -> Action:
        """
        吸收推論／還原的例外，不重新拋出。

        TableOrchestrator.step() 的 try/except 只包住下游分派，TableRuntime.tick()
        也沒有包，例外從 get_action() 拋出去會直接穿透 physics callback，一次中斷
        所有桌子的 tick loop。復原走 TableOrchestrator.reset()。
        """
        logger.exception("ModelController 推論失敗", exc_info=exception)
        self._change_state(BilliardStatus.ERROR)
        return self._generate_action_result()

    def _on_reset(self) -> None:
        self._cached_raw_action = None
