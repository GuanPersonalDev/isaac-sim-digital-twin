import pytest

from core.controllers.manual_controller import ManualController
from core.models.action_bounds import CUE_BALL_SPEED
from core.models.billiard_state import BilliardStatus
from core.models.manual_shot_parameters import ManualShotParameters
from core.models.observation import Observation
from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS


def _observation(
    is_init_state: bool = False,
    is_ball_moving: bool = False,
    is_motion_complete: bool = False,
    has_error: bool = False,
) -> Observation:
    return Observation(
        ball_positions=[[0.0, 0.0, 0.0]],
        cue_ball_position=[-0.3, 0.0, 0.0],
        is_init_state=is_init_state,
        is_ball_moving=is_ball_moving,
        is_motion_complete=is_motion_complete,
        has_error=has_error,
    )


# 用來驗證快照語意的第二組參數，刻意跟 default() 全部不同，這樣任何一個
# 欄位被漏改都會讓比對失敗，而不是巧合地跟 default 撞在一起。
_ALT_PARAMETERS = ManualShotParameters(
    cue_ball_placement=(0.3, -0.8),
    shot_angle=12.0,
    cue_ball_speed=2.0,
    position_offset=(0.2, -0.1),
)


@pytest.fixture
def controller() -> ManualController:
    return ManualController()


def _advance_to_idle(controller: ManualController) -> None:
    controller.get_action(_observation(is_motion_complete=True))


def _advance_to_aiming(controller: ManualController) -> None:
    _advance_to_idle(controller)
    controller.request_shot()
    controller.get_action(_observation(is_init_state=True, is_ball_moving=False))


def _advance_to_striking(controller: ManualController) -> None:
    _advance_to_aiming(controller)
    controller.get_action(_observation(is_motion_complete=True))


def _advance_to_waiting(controller: ManualController) -> None:
    _advance_to_striking(controller)
    controller.get_action(_observation(is_motion_complete=True))


class TestInitialState:
    def test_starts_in_reset_state(self, controller: ManualController):
        # Assert
        assert controller.get_current_state() == BilliardStatus.RESET

    def test_default_parameters_match_break_shot(self, controller: ManualController):
        # Assert：切到手動、什麼都不調就按擊球，要打出跟 ScriptController 同一球
        parameters = controller.get_parameters()
        assert parameters.cue_ball_placement == pytest.approx(list(BREAK_SHOT_POSITIONS[0]))
        assert parameters.shot_angle == 0.0
        assert parameters.cue_ball_speed == CUE_BALL_SPEED[1]
        assert parameters.position_offset == pytest.approx([0.0, 0.0])


class TestStaysIdleWithoutShotRequest:
    def test_stays_in_idle_forever_without_shot_request(self, controller: ManualController):
        """驗收錨點：沒按擊球鈕，控制器必須無限期停在 IDLE，不能像
        ScriptController 一樣自動循環。跑 100 tick 逐一檢查狀態與 Action，
        任何一 tick 洩漏成 AIMING 或吐出 should_execute_action=True 都算失敗。
        """
        # Arrange
        _advance_to_idle(controller)

        # Act & Assert
        for _ in range(100):
            action = controller.get_action(
                _observation(is_init_state=True, is_ball_moving=False)
            )
            assert controller.get_current_state() == BilliardStatus.IDLE
            assert action.should_execute_action is False


class TestIdleToAiming:
    def test_stays_idle_when_no_shot_requested(self, controller: ManualController):
        # Arrange
        _advance_to_idle(controller)

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))

        # Assert
        assert controller.get_current_state() == BilliardStatus.IDLE

    def test_holds_request_when_table_not_at_initial_position(
        self, controller: ManualController
    ):
        # Arrange
        _advance_to_idle(controller)
        controller.request_shot()

        # Act
        controller.get_action(_observation(is_init_state=False, is_ball_moving=False))

        # Assert：請求還在排隊，不會被這一 tick 默默丟掉
        assert controller.get_current_state() == BilliardStatus.IDLE
        assert controller.is_shot_pending() is True

    def test_holds_request_when_ball_still_moving(self, controller: ManualController):
        # Arrange
        _advance_to_idle(controller)
        controller.request_shot()

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=True))

        # Assert
        assert controller.get_current_state() == BilliardStatus.IDLE
        assert controller.is_shot_pending() is True

    def test_transitions_to_aiming_once_table_ready(self, controller: ManualController):
        # Arrange：請求先發生，球晚一點才靜止——請求必須撐到那一刻
        _advance_to_idle(controller)
        controller.request_shot()
        controller.get_action(_observation(is_init_state=True, is_ball_moving=True))

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))

        # Assert
        assert controller.get_current_state() == BilliardStatus.AIMING

    def test_idle_action_carries_current_parameters(self, controller: ManualController):
        """呼應時序坑：AIMING 消費的是 IDLE handler 回傳的 action，四項參數
        必須在這裡就填好，不能沿用 _generate_action_result() 的全零預設
        （ScriptController 忘記填 cue_ball_placement 的既有 bug 不可以重演）。
        """
        # Arrange
        _advance_to_idle(controller)
        controller.set_parameters(_ALT_PARAMETERS)
        controller.request_shot()

        # Act
        action = controller.get_action(_observation(is_init_state=True, is_ball_moving=False))

        # Assert：逐一比對四項參數
        assert action.cue_ball_placement == list(_ALT_PARAMETERS.cue_ball_placement)
        assert action.shot_angle == _ALT_PARAMETERS.shot_angle
        assert action.cue_ball_speed == _ALT_PARAMETERS.cue_ball_speed
        assert action.position_offset == list(_ALT_PARAMETERS.position_offset)
        assert action.should_execute_action is True


class TestAimingSnapshot:
    def test_striking_action_uses_snapshot_taken_at_idle(self, controller: ManualController):
        """AIMING 期間換參數，STRIKING 仍必須用進 AIMING 那一刻的舊值——
        否則會變成「照舊角度瞄、照新速度打」，或更糟：AIM 已經把母球
        teleport 到舊擺位，STRIKE 卻對著空氣揮。
        """
        # Arrange
        _advance_to_idle(controller)
        controller.set_parameters(_ALT_PARAMETERS)
        controller.request_shot()
        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))
        assert controller.get_current_state() == BilliardStatus.AIMING

        # Act：瞄準途中改參數
        controller.set_parameters(ManualShotParameters.default())
        action = controller.get_action(_observation(is_motion_complete=True))

        # Assert：STRIKE 拿到的仍是舊快照
        assert controller.get_current_state() == BilliardStatus.STRIKING
        assert action.cue_ball_placement == list(_ALT_PARAMETERS.cue_ball_placement)
        assert action.shot_angle == _ALT_PARAMETERS.shot_angle
        assert action.cue_ball_speed == _ALT_PARAMETERS.cue_ball_speed
        assert action.position_offset == list(_ALT_PARAMETERS.position_offset)

    def test_waits_until_motion_complete(self, controller: ManualController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=False))

        # Assert
        assert controller.get_current_state() == BilliardStatus.AIMING


class TestFullShotCycle:
    def test_full_cycle_returns_to_idle_and_stays(self, controller: ManualController):
        # Act & Assert：整個循環走一輪
        _advance_to_idle(controller)
        controller.request_shot()

        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))
        assert controller.get_current_state() == BilliardStatus.AIMING

        controller.get_action(_observation(is_motion_complete=True))
        assert controller.get_current_state() == BilliardStatus.STRIKING

        controller.get_action(_observation(is_motion_complete=True))
        assert controller.get_current_state() == BilliardStatus.WAITING

        controller.get_action(_observation(is_ball_moving=False))
        assert controller.get_current_state() == BilliardStatus.RESET

        controller.get_action(_observation(is_motion_complete=True))
        assert controller.get_current_state() == BilliardStatus.IDLE

        # 回到 IDLE 之後請求已消費，不會自己再打第二桿
        action = controller.get_action(_observation(is_init_state=True, is_ball_moving=False))
        assert controller.get_current_state() == BilliardStatus.IDLE
        assert action.should_execute_action is False

    def test_multiple_requests_before_shot_taken_merge_into_one(
        self, controller: ManualController
    ):
        # Arrange：連按 3 次
        _advance_to_idle(controller)
        controller.request_shot()
        controller.request_shot()
        controller.request_shot()

        # Act：只應該觸發一次 IDLE -> AIMING
        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))

        # Assert
        assert controller.get_current_state() == BilliardStatus.AIMING
        assert controller.is_shot_pending() is False

        # 走完整個循環回到 IDLE 之後也不會補打欠下的請求
        controller.get_action(_observation(is_motion_complete=True))  # -> STRIKING
        controller.get_action(_observation(is_motion_complete=True))  # -> WAITING
        controller.get_action(_observation(is_ball_moving=False))  # -> RESET
        controller.get_action(_observation(is_motion_complete=True))  # -> IDLE
        assert controller.get_current_state() == BilliardStatus.IDLE

        action = controller.get_action(_observation(is_init_state=True, is_ball_moving=False))
        assert controller.get_current_state() == BilliardStatus.IDLE
        assert action.should_execute_action is False


class TestStrikingToWaiting:
    def test_waits_until_motion_complete(self, controller: ManualController):
        # Arrange
        _advance_to_striking(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=False))

        # Assert
        assert controller.get_current_state() == BilliardStatus.STRIKING

    def test_transitions_to_waiting_when_motion_complete(self, controller: ManualController):
        # Arrange
        _advance_to_striking(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=True))

        # Assert
        assert controller.get_current_state() == BilliardStatus.WAITING


class TestWaitingToReset:
    def test_stays_waiting_while_balls_moving(self, controller: ManualController):
        # Arrange
        _advance_to_waiting(controller)

        # Act
        controller.get_action(_observation(is_ball_moving=True))

        # Assert
        assert controller.get_current_state() == BilliardStatus.WAITING

    def test_transitions_to_reset_when_balls_stop(self, controller: ManualController):
        # Arrange
        _advance_to_waiting(controller)

        # Act
        controller.get_action(_observation(is_ball_moving=False))

        # Assert
        assert controller.get_current_state() == BilliardStatus.RESET


class TestPendingNoneEntersErrorWithoutRaising:
    def test_pending_none_in_aiming_enters_error(self, controller: ManualController):
        """`_pending` 只在 IDLE handler 消費請求時才會被填值。若時序被破壞
        （例如外部直接把狀態撥到 AIMING），AIMING handler 必須安全地轉進
        ERROR，而不是對 None 呼叫 to_action() 炸出 AttributeError 中斷
        physics callback。
        """
        # Arrange：模擬時序被破壞——沒有經過 IDLE handler 就進了 AIMING
        controller._change_state(BilliardStatus.AIMING)

        # Act
        action = controller.get_action(_observation(is_motion_complete=True))

        # Assert：例外被吸收，不往外拋
        assert controller.get_current_state() == BilliardStatus.ERROR
        assert action.should_execute_action is False


class TestErrorHandling:
    def test_any_state_transitions_to_error_when_has_error(self, controller: ManualController):
        # Act
        controller.get_action(_observation(has_error=True))

        # Assert
        assert controller.get_current_state() == BilliardStatus.ERROR

    def test_error_takes_priority_over_normal_transition(self, controller: ManualController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=True, has_error=True))

        # Assert
        assert controller.get_current_state() == BilliardStatus.ERROR

    def test_error_state_does_not_auto_recover(self, controller: ManualController):
        # Arrange
        controller.get_action(_observation(has_error=True))

        # Act
        controller.get_action(
            _observation(is_init_state=True, is_ball_moving=False, is_motion_complete=True)
        )

        # Assert
        assert controller.get_current_state() == BilliardStatus.ERROR


class TestReset:
    def test_reset_returns_controller_to_reset_state(self, controller: ManualController):
        # Arrange
        _advance_to_idle(controller)

        # Act
        controller.reset()

        # Assert
        assert controller.get_current_state() == BilliardStatus.RESET

    def test_reset_drops_pending_request(self, controller: ManualController):
        # Arrange
        _advance_to_idle(controller)
        controller.request_shot()
        assert controller.is_shot_pending() is True

        # Act
        controller.reset()

        # Assert：排隊的請求被丟棄，不是保留到下次 IDLE 補打
        assert controller.is_shot_pending() is False

    def test_reset_keeps_parameters(self, controller: ManualController):
        # Arrange
        controller.set_parameters(_ALT_PARAMETERS)

        # Act
        controller.reset()

        # Assert：參數本身不因 reset 而消失，只有「排隊的請求」被丟棄
        assert controller.get_parameters() == _ALT_PARAMETERS

    def test_reset_during_aiming_clears_pending_snapshot(self, controller: ManualController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.reset()
        # 模擬時序異常：reset 之後又被外部撥回 AIMING
        controller._change_state(BilliardStatus.AIMING)
        action = controller.get_action(_observation(is_motion_complete=True))

        # Assert：_pending 已被 _on_reset() 清空，不會用舊快照打出一桿
        assert controller.get_current_state() == BilliardStatus.ERROR
        assert action.should_execute_action is False


class TestIsShotPending:
    def test_false_initially(self, controller: ManualController):
        # Assert
        assert controller.is_shot_pending() is False

    def test_true_after_request(self, controller: ManualController):
        # Act
        controller.request_shot()

        # Assert
        assert controller.is_shot_pending() is True

    def test_false_after_consumed_by_idle_handler(self, controller: ManualController):
        # Arrange
        _advance_to_idle(controller)
        controller.request_shot()

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))

        # Assert
        assert controller.is_shot_pending() is False
