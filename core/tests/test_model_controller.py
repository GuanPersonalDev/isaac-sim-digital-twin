import math

import pytest

from core.controllers.model_controller import ModelController
from core.models.action_bounds import (
    ACTION_BOUNDS,
    ACTION_DIM,
    POSITION_OFFSET_VERTICAL,
)
from core.models.billiard_state import BilliardStatus
from core.models.observation import Observation
from core.ports.policy_port import PolicyPort
from core.services.rl_observation_encoder import RL_BALL_ORDER


_TABLE_POSITION = (1.5, -2.0)
_MAX_OFFSET = 0.6
_BALL_COUNT = 10
# 動作空間各維的中心；正規化域的 0 反正規化後就是這個值。
# 一律現算，不得寫死 3.3392 或 ±30——Milestone B 把 SHOT_ANGLE 改回整圈時，
# 任何殘留的假設要在這裡大聲失敗（見 core/models/action_bounds.py）。
_ACTION_CENTER = [(low + high) / 2.0 for low, high in ACTION_BOUNDS]


class _FakePolicy(PolicyPort):
    """Fake model：記錄每次收到的觀測，回傳可在測試中途改寫的固定輸出。"""

    def __init__(self, output: list[float] | None = None) -> None:
        self.output = [0.0] * ACTION_DIM if output is None else output
        self.exception: Exception | None = None
        self.calls: list[list[float]] = []

    def infer(self, observation: list[float]) -> list[float]:
        self.calls.append(list(observation))
        if self.exception is not None:
            raise self.exception
        return list(self.output)


def _ball_positions() -> list[list[float]]:
    """球 i 的桌台相對座標固定為 (0.01i, -0.02i)，回傳時加上桌台世界位置。

    桌台位置刻意取非零值：Controller 若忘了扣掉桌台 XY，編碼結果會整組偏移，
    但不會拋任何錯——只有斷言抓得到。
    """
    table_x, table_y = _TABLE_POSITION
    return [
        [table_x + ball_id * 0.01, table_y - ball_id * 0.02, 0.028575]
        for ball_id in range(_BALL_COUNT)
    ]


def _observation(
    is_init_state: bool = False,
    is_ball_moving: bool = False,
    is_motion_complete: bool = False,
    has_error: bool = False,
) -> Observation:
    ball_positions = _ball_positions()
    return Observation(
        ball_positions=ball_positions,
        cue_ball_position=ball_positions[0],
        is_init_state=is_init_state,
        is_ball_moving=is_ball_moving,
        is_motion_complete=is_motion_complete,
        has_error=has_error,
    )


@pytest.fixture
def policy() -> _FakePolicy:
    return _FakePolicy()


@pytest.fixture
def controller(policy: _FakePolicy) -> ModelController:
    return ModelController(policy, _TABLE_POSITION, _MAX_OFFSET)


def _advance_to_idle(controller: ModelController) -> None:
    controller.get_action(_observation(is_motion_complete=True))


def _advance_to_aiming(controller: ModelController) -> None:
    _advance_to_idle(controller)
    controller.get_action(_observation(is_init_state=True, is_ball_moving=False))


def _advance_to_striking(controller: ModelController) -> None:
    _advance_to_aiming(controller)
    controller.get_action(_observation(is_motion_complete=True))


def _advance_to_waiting(controller: ModelController) -> None:
    _advance_to_striking(controller)
    controller.get_action(_observation(is_motion_complete=True))


def _strike_action(controller: ModelController):
    """跑到 AIMING→STRIKING 那一次轉換，回傳會被 _execute_strike() 消費的 Action。"""
    _advance_to_aiming(controller)
    return controller.get_action(_observation(is_motion_complete=True))


class TestInitialState:
    def test_starts_in_reset_state(self, controller: ModelController):
        # Assert
        assert controller._current_state == BilliardStatus.RESET

    def test_does_not_infer_before_any_tick(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Assert
        assert policy.calls == []


class TestLifecycle:
    def test_full_lap_follows_script_controller_timing(self, controller: ModelController):
        # Act & Assert
        controller.get_action(_observation(is_motion_complete=True))
        assert controller.get_current_state() == BilliardStatus.IDLE

        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))
        assert controller.get_current_state() == BilliardStatus.AIMING

        controller.get_action(_observation(is_motion_complete=True))
        assert controller.get_current_state() == BilliardStatus.STRIKING

        controller.get_action(_observation(is_motion_complete=True))
        assert controller.get_current_state() == BilliardStatus.WAITING

        controller.get_action(_observation(is_ball_moving=False))
        assert controller.get_current_state() == BilliardStatus.RESET

    def test_stays_idle_when_balls_not_ready(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange
        _advance_to_idle(controller)

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=True))

        # Assert
        assert controller._current_state == BilliardStatus.IDLE
        assert policy.calls == []

    def test_waits_in_aiming_until_motion_complete(self, controller: ModelController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=False))

        # Assert
        assert controller._current_state == BilliardStatus.AIMING


class TestInference:
    def test_infers_exactly_once_when_entering_aiming(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Act
        _advance_to_aiming(controller)

        # Assert
        assert len(policy.calls) == 1

    def test_encoded_observation_is_21_dim_table_relative_with_max_offset(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange
        expected: list[float] = []
        for ball_id in RL_BALL_ORDER:
            expected.extend([ball_id * 0.01, ball_id * -0.02])
        expected.append(_MAX_OFFSET)

        # Act
        _advance_to_aiming(controller)

        # Assert
        assert len(policy.calls[0]) == 21
        assert policy.calls[0] == pytest.approx(expected)

    def test_does_not_infer_again_when_entering_striking(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Act
        _advance_to_striking(controller)

        # Assert
        assert len(policy.calls) == 1

    def test_infers_again_on_next_episode(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange
        _advance_to_waiting(controller)
        controller.get_action(_observation(is_ball_moving=False))
        policy.output = [1.0] * ACTION_DIM

        # Act
        action = _strike_action(controller)

        # Assert
        assert len(policy.calls) == 2
        assert action.cue_ball_speed == pytest.approx(ACTION_BOUNDS[3][1])


class TestActionDecoding:
    def test_zero_output_maps_to_action_space_center(self, controller: ModelController):
        # Act
        action = _strike_action(controller)

        # Assert
        assert action.cue_ball_placement == pytest.approx(_ACTION_CENTER[0:2])
        assert action.shot_angle == pytest.approx(_ACTION_CENTER[2])
        assert action.cue_ball_speed == pytest.approx(_ACTION_CENTER[3])
        assert action.position_offset == pytest.approx(_ACTION_CENTER[4:6])

    def test_out_of_range_output_is_clipped_to_physical_bounds(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange：policy 輸出是無界的高斯 mean，越界是正常情形而不是錯誤
        policy.output = [10.0, -10.0, 10.0, -10.0, 0.0, 0.0]

        # Act
        action = _strike_action(controller)

        # Assert
        assert action.cue_ball_placement[0] == pytest.approx(ACTION_BOUNDS[0][1])
        assert action.cue_ball_placement[1] == pytest.approx(ACTION_BOUNDS[1][0])
        assert action.shot_angle == pytest.approx(ACTION_BOUNDS[2][1])
        assert action.cue_ball_speed == pytest.approx(ACTION_BOUNDS[3][0])

    def test_offset_is_circle_clipped_to_max_offset_preserving_direction(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange：hypot(1, 1) = 1.414 > max_offset，整條向量必須等比縮短
        policy.output = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

        # Act
        action = _strike_action(controller)

        # Assert：長度縮到 max_offset（換算到物理域的同一把尺），方向不變
        assert math.hypot(*action.position_offset) == pytest.approx(
            _MAX_OFFSET * POSITION_OFFSET_VERTICAL[1]
        )
        assert action.position_offset[0] == pytest.approx(action.position_offset[1])

    def test_aim_and_strike_share_one_decision_but_not_one_instance(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange
        policy.output = [0.5, -0.5, 0.25, 0.75, 0.1, -0.2]
        _advance_to_idle(controller)

        # Act
        aim_action = controller.get_action(
            _observation(is_init_state=True, is_ball_moving=False)
        )
        strike_action = controller.get_action(_observation(is_motion_complete=True))

        # Assert：手臂瞄的與實際打的必須是同一個決策（#96 接上手臂後才看得出差別）
        assert aim_action.cue_ball_placement == pytest.approx(
            strike_action.cue_ball_placement
        )
        assert aim_action.shot_angle == pytest.approx(strike_action.shot_angle)
        assert aim_action.cue_ball_speed == pytest.approx(strike_action.cue_ball_speed)
        assert aim_action.position_offset == pytest.approx(strike_action.position_offset)
        # Action 是 mutable dataclass，兩次分派共用同一個 instance 會被下游改到
        assert aim_action is not strike_action


class TestShouldExecuteAction:
    def test_no_op_ticks_do_not_execute(self, controller: ModelController):
        # Act
        reset_tick = controller.get_action(_observation(is_motion_complete=False))
        _advance_to_idle(controller)
        idle_tick = controller.get_action(_observation(is_init_state=False))

        # Assert
        assert reset_tick.should_execute_action is False
        assert idle_tick.should_execute_action is False

    def test_aiming_and_striking_transitions_execute(self, controller: ModelController):
        # Arrange
        _advance_to_idle(controller)

        # Act
        aim_action = controller.get_action(
            _observation(is_init_state=True, is_ball_moving=False)
        )
        strike_action = controller.get_action(_observation(is_motion_complete=True))

        # Assert
        assert aim_action.should_execute_action is True
        assert strike_action.should_execute_action is True

    def test_seventh_dimension_is_rejected_not_used_as_flag(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange：should_execute_action 由狀態轉換產生，多送一維必須被擋下
        policy.output = [0.0] * (ACTION_DIM + 1)
        _advance_to_idle(controller)

        # Act
        action = controller.get_action(
            _observation(is_init_state=True, is_ball_moving=False)
        )

        # Assert
        assert controller._current_state == BilliardStatus.ERROR
        assert action.should_execute_action is False


class TestErrorHandling:
    def test_policy_exception_enters_error_state_without_raising(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange：例外若穿透 get_action()，physics callback 會一次帶走所有桌子
        policy.exception = RuntimeError("inference failed")
        _advance_to_idle(controller)

        # Act
        action = controller.get_action(
            _observation(is_init_state=True, is_ball_moving=False)
        )

        # Assert
        assert controller._current_state == BilliardStatus.ERROR
        assert action.should_execute_action is False

    def test_non_finite_output_enters_error_state(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange
        policy.output = [float("nan")] + [0.0] * (ACTION_DIM - 1)
        _advance_to_idle(controller)

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))

        # Assert
        assert controller._current_state == BilliardStatus.ERROR

    def test_wrong_length_output_enters_error_state(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange
        policy.output = [0.0] * (ACTION_DIM - 1)
        _advance_to_idle(controller)

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))

        # Assert
        assert controller._current_state == BilliardStatus.ERROR

    def test_observation_error_takes_priority_over_transition(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange
        _advance_to_idle(controller)

        # Act
        controller.get_action(
            _observation(is_init_state=True, is_ball_moving=False, has_error=True)
        )

        # Assert
        assert controller._current_state == BilliardStatus.ERROR
        assert policy.calls == []

    def test_error_state_does_not_auto_recover(self, controller: ModelController):
        # Arrange
        controller.get_action(_observation(has_error=True))

        # Act
        controller.get_action(
            _observation(is_init_state=True, is_ball_moving=False, is_motion_complete=True)
        )

        # Assert
        assert controller._current_state == BilliardStatus.ERROR


class TestReset:
    def test_reset_returns_controller_to_reset_state(self, controller: ModelController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.reset()

        # Assert
        assert controller._current_state == BilliardStatus.RESET

    def test_reset_clears_cached_inference(self, controller: ModelController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.reset()

        # Assert
        assert controller._cached_raw_action is None

    def test_next_episode_infers_again_after_reset(
        self, controller: ModelController, policy: _FakePolicy
    ):
        # Arrange
        _advance_to_aiming(controller)
        controller.reset()

        # Act
        _advance_to_aiming(controller)

        # Assert
        assert len(policy.calls) == 2


class TestConstructorValidation:
    def test_rejects_max_offset_out_of_range(self, policy: _FakePolicy):
        # Act & Assert
        with pytest.raises(ValueError):
            ModelController(policy, _TABLE_POSITION, 1.5)

    def test_rejects_incomplete_table_position(self, policy: _FakePolicy):
        # Act & Assert
        with pytest.raises(ValueError):
            ModelController(policy, (0.0,), _MAX_OFFSET)
