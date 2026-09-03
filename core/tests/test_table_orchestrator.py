from unittest.mock import MagicMock, patch

import pytest

from core.models.action import Action
from core.models.billiard_state import BilliardStatus
from core.models.observation import Observation
from core.controllers.controller_base import ControllerBase
from core.services.error_state import ErrorState
from core.services.robot_swing_strategy import RobotSwingStrategy
from core.services.table_orchestrator import (
    DemoTableOrchestrator,
    TrainingTableOrchestrator,
)


def _observation() -> Observation:
    return Observation(
        ball_positions=[],
        cue_ball_position=[0.0, 0.0, 0.0],
        is_init_state=False,
        is_ball_moving=False,
        is_motion_complete=False,
        has_error=False,
    )


def _action(should_execute_action: bool) -> Action:
    return Action(
        cue_ball_speed=0.0,
        shot_angle=0.0,
        position_offset=[0.0, 0.0],
        cue_ball_placement=[0.0, 0.0],
        should_execute_action=should_execute_action,
    )


@pytest.fixture
def script_controller() -> MagicMock:
    return MagicMock(spec=ControllerBase)


@pytest.fixture
def table_ball_set() -> MagicMock:
    return MagicMock()


@pytest.fixture
def ball_position_provider() -> MagicMock:
    return MagicMock()


@pytest.fixture
def robot_arm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def articulation_api() -> MagicMock:
    api = MagicMock()
    # 預設「上一個動作沒有逾時」——不設的話 MagicMock 回傳的是 truthy Mock，
    # 每個 step() 都會被 _check_downstream_failure() 標記成錯誤
    api.did_last_motion_timeout.return_value = False
    return api


@pytest.fixture
def swing_strategy() -> MagicMock:
    return MagicMock(spec=RobotSwingStrategy)


@pytest.fixture
def impulse_striking_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def error_state() -> MagicMock:
    # wraps 真正的 ErrorState：既能斷言呼叫參數，has_error()/get_last_exception() 也反映真實狀態
    return MagicMock(wraps=ErrorState())


@pytest.fixture
def rolling_resistance_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def demo_orchestrator(
    script_controller: MagicMock,
    table_ball_set: MagicMock,
    ball_position_provider: MagicMock,
    robot_arm: MagicMock,
    articulation_api: MagicMock,
    swing_strategy: MagicMock,
    error_state: MagicMock,
    rolling_resistance_service: MagicMock,
) -> DemoTableOrchestrator:
    return DemoTableOrchestrator(
        script_controller=script_controller,
        table_ball_set=table_ball_set,
        ball_position_provider=ball_position_provider,
        robot_arm=robot_arm,
        articulation_api=articulation_api,
        swing_strategy=swing_strategy,
        error_state=error_state,
        rolling_resistance_service=rolling_resistance_service,
    )


@pytest.fixture
def training_orchestrator(
    script_controller: MagicMock,
    table_ball_set: MagicMock,
    ball_position_provider: MagicMock,
    impulse_striking_service: MagicMock,
    error_state: MagicMock,
    rolling_resistance_service: MagicMock,
) -> TrainingTableOrchestrator:
    return TrainingTableOrchestrator(
        script_controller=script_controller,
        table_ball_set=table_ball_set,
        ball_position_provider=ball_position_provider,
        impulse_striking_service=impulse_striking_service,
        error_state=error_state,
        rolling_resistance_service=rolling_resistance_service,
    )


class TestGetCurrentState:
    def test_demo_orchestrator_get_current_state_delegates_to_script_controller(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
    ):
        script_controller.get_current_state.return_value = BilliardStatus.STRIKING

        assert demo_orchestrator.get_current_state() == BilliardStatus.STRIKING

    def test_training_orchestrator_get_current_state_delegates_to_script_controller(
        self,
        training_orchestrator: TrainingTableOrchestrator,
        script_controller: MagicMock,
    ):
        script_controller.get_current_state.return_value = BilliardStatus.IDLE

        assert training_orchestrator.get_current_state() == BilliardStatus.IDLE


class TestStepDispatch:
    def test_step_dispatches_reset_when_should_execute_action_true(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
        robot_arm: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.RESET
        ball_position_provider.get_positions.return_value = {0: (0.0, 0.0)}

        demo_orchestrator.step(_observation())

        ball_position_provider.get_positions.assert_called_once_with()
        table_ball_set.reset.assert_called_once_with({0: (0.0, 0.0)})
        robot_arm.reset.assert_called_once_with()

    def test_step_dispatches_aiming(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
    ):
        action = _action(should_execute_action=True)
        script_controller.get_action.return_value = action
        script_controller.get_current_state.return_value = BilliardStatus.AIMING
        demo_orchestrator._execute_aim = MagicMock()

        demo_orchestrator.step(_observation())

        demo_orchestrator._execute_aim.assert_called_once_with(action)

    def test_step_dispatches_striking(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
    ):
        action = _action(should_execute_action=True)
        script_controller.get_action.return_value = action
        script_controller.get_current_state.return_value = BilliardStatus.STRIKING
        demo_orchestrator._execute_strike = MagicMock()

        demo_orchestrator.step(_observation())

        demo_orchestrator._execute_strike.assert_called_once_with(action)

    def test_step_skips_downstream_when_should_execute_action_false(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        robot_arm: MagicMock,
        ball_position_provider: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=False)
        script_controller.get_current_state.return_value = BilliardStatus.RESET

        demo_orchestrator.step(_observation())

        ball_position_provider.get_positions.assert_not_called()
        table_ball_set.reset.assert_not_called()
        robot_arm.reset.assert_not_called()

    @pytest.mark.parametrize("state", [BilliardStatus.WAITING, BilliardStatus.IDLE])
    def test_step_has_no_downstream_action_for_waiting_or_idle(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        robot_arm: MagicMock,
        state: BilliardStatus,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = state

        demo_orchestrator.step(_observation())

        table_ball_set.reset.assert_not_called()
        robot_arm.reset.assert_not_called()

    def test_step_calls_rolling_resistance_regardless_of_should_execute_action(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        rolling_resistance_service: MagicMock,
    ):
        table_ball_set.get_ball_prim_paths.return_value = ["/World/Table_Demo/Balls/Ball_0"]
        script_controller.get_action.return_value = _action(should_execute_action=False)
        script_controller.get_current_state.return_value = BilliardStatus.IDLE

        demo_orchestrator.step(_observation())

        rolling_resistance_service.apply.assert_called_once_with(["/World/Table_Demo/Balls/Ball_0"])

    def test_step_calls_rolling_resistance_before_state_dispatch(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
        rolling_resistance_service: MagicMock,
    ):
        manager = MagicMock()
        manager.attach_mock(rolling_resistance_service.apply, "rolling_resistance_apply")
        manager.attach_mock(table_ball_set.reset, "table_ball_set_reset")
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.RESET
        ball_position_provider.get_positions.return_value = {0: (0.0, 0.0)}

        demo_orchestrator.step(_observation())

        assert [call[0] for call in manager.mock_calls] == [
            "rolling_resistance_apply",
            "table_ball_set_reset",
        ]


class TestDownstreamFailure:
    def test_step_marks_an_error_when_the_previous_motion_timed_out(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        articulation_api: MagicMock,
        error_state: MagicMock,
    ):
        articulation_api.did_last_motion_timeout.return_value = True
        script_controller.get_action.return_value = _action(should_execute_action=False)
        script_controller.get_current_state.return_value = BilliardStatus.RESET

        demo_orchestrator.step(_observation())

        error_state.mark_error.assert_called_once()
        assert isinstance(error_state.mark_error.call_args.args[0], RuntimeError)

    def test_step_marks_no_error_while_motions_keep_converging(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        error_state: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=False)
        script_controller.get_current_state.return_value = BilliardStatus.IDLE

        demo_orchestrator.step(_observation())

        error_state.mark_error.assert_not_called()

    def test_timeout_is_checked_before_the_controller_decides(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        articulation_api: MagicMock,
        error_state: MagicMock,
    ):
        """
        逾時必須在 get_action() 之前標記，否則狀態機這一 tick 還是會用
        「動作已完成」的舊語意往下推進一格。
        """
        calls = []
        articulation_api.did_last_motion_timeout.side_effect = lambda: calls.append("check") or True
        error_state.mark_error.side_effect = lambda _: calls.append("mark")
        script_controller.get_action.side_effect = lambda _: calls.append("get_action") or _action(should_execute_action=False)
        script_controller.get_current_state.return_value = BilliardStatus.IDLE

        demo_orchestrator.step(_observation())

        assert calls == ["check", "mark", "get_action"]

    def test_training_orchestrator_has_no_downstream_to_check(
        self,
        training_orchestrator,
        script_controller: MagicMock,
        error_state: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=False)
        script_controller.get_current_state.return_value = BilliardStatus.IDLE

        training_orchestrator.step(_observation())

        error_state.mark_error.assert_not_called()


class TestStepErrorHandling:
    def test_downstream_exception_is_recorded_and_not_reraised(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        error_state: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.AIMING
        boom = RuntimeError("boom")
        demo_orchestrator._execute_aim = MagicMock(side_effect=boom)

        demo_orchestrator.step(_observation())  # 不應往外拋

        error_state.mark_error.assert_called_once_with(boom)
        assert error_state.has_error() is True
        assert error_state.get_last_exception() is boom

    def test_no_exception_leaves_error_state_untouched(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        ball_position_provider: MagicMock,
        error_state: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.RESET
        ball_position_provider.get_positions.return_value = {0: (0.0, 0.0)}

        demo_orchestrator.step(_observation())

        error_state.mark_error.assert_not_called()
        assert error_state.has_error() is False


class TestReset:
    def test_reset_clears_error_state_and_resets_script_controller(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        error_state: MagicMock,
    ):
        demo_orchestrator.reset()

        error_state.clear.assert_called_once_with()
        script_controller.reset.assert_called_once_with()

    def test_reset_clears_a_previously_recorded_error(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        error_state: MagicMock,
    ):
        error_state.mark_error(RuntimeError("boom"))
        assert error_state.has_error() is True

        demo_orchestrator.reset()

        assert error_state.has_error() is False
        assert error_state.get_last_exception() is None


class TestFullReset:
    def test_full_reset_also_racks_the_balls_and_homes_the_arm(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
        error_state: MagicMock,
        robot_arm: MagicMock,
    ):
        ball_position_provider.get_positions.return_value = {0: (0.1, 0.2)}

        demo_orchestrator.full_reset()

        error_state.clear.assert_called_once_with()
        script_controller.reset.assert_called_once_with()
        table_ball_set.reset.assert_called_once_with({0: (0.1, 0.2)})
        robot_arm.reset.assert_called_once_with()

    def test_full_reset_needs_no_action_dispatch(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
    ):
        """
        RESET 狀態本身的 Action 是 no-op（should_execute_action=False），重擺球
        只在 WAITING → RESET 那一個 tick 才會被帶出來；full_reset() 必須自己
        直接做，不能依賴 step() 分派。
        """
        ball_position_provider.get_positions.return_value = {0: (0.0, 0.0)}

        demo_orchestrator.full_reset()

        script_controller.get_action.assert_not_called()
        table_ball_set.reset.assert_called_once_with({0: (0.0, 0.0)})

    def test_training_full_reset_racks_the_balls_without_a_downstream(
        self,
        training_orchestrator,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
    ):
        ball_position_provider.get_positions.return_value = {0: (0.3, 0.4)}

        training_orchestrator.full_reset()

        table_ball_set.reset.assert_called_once_with({0: (0.3, 0.4)})


class TestResetBalls:
    def test_reset_balls_uses_position_provider_positions(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
    ):
        ball_position_provider.get_positions.return_value = {1: (0.1, 0.2)}

        demo_orchestrator._reset_balls()

        table_ball_set.reset.assert_called_once_with({1: (0.1, 0.2)})


class TestDemoTableOrchestrator:
    def test_reset_downstream_calls_robot_arm_reset(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        robot_arm: MagicMock,
    ):
        demo_orchestrator._reset_downstream()

        robot_arm.reset.assert_called_once_with()


class TestDemoTableOrchestratorExecuteAim:
    """`_execute_aim` 現在只做手臂型號無關的共用前置工作（table_z/ball_radius/
    cue_ball 計算、母球 teleport），實際瞄準邏輯全權委派給
    `RobotSwingStrategy.execute_aim()`（見 core/services/wam7_swing_strategy.py／
    ur3e_swing_strategy.py 的獨立測試，涵蓋 flat/bridge/infeasible 各分支）。"""

    def _setup_table(self, table_ball_set: MagicMock) -> None:
        table_ball_set.get_table_z.return_value = 0.0
        table_ball_set.DEFAULT_BALL_RADIUS = 0.028575

    def test_places_cue_ball_at_action_cue_ball_placement(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        table_ball_set: MagicMock,
        swing_strategy: MagicMock,
    ):
        """ModelController 的 policy 每一局自己決定母球擺位，Demo 端必須把這個
        決定真的 teleport 到球上（跟 Training 端 _apply_strike() 一致），否則
        swing_strategy.execute_aim() 會拿一個沒有球的座標當瞄準錨點。"""
        self._setup_table(table_ball_set)
        action = _action(should_execute_action=True)
        action.cue_ball_placement = [0.1, -0.9]

        demo_orchestrator._execute_aim(action)

        table_ball_set.place_ball.assert_called_once_with(0, 0.1, -0.9)

    def test_delegates_to_swing_strategy_with_table_z_ball_radius_and_cue_ball(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        table_ball_set: MagicMock,
        swing_strategy: MagicMock,
    ):
        self._setup_table(table_ball_set)
        action = _action(should_execute_action=True)
        action.cue_ball_placement = [0.1, -0.9]

        demo_orchestrator._execute_aim(action)

        swing_strategy.execute_aim.assert_called_once_with(action, (0.1, -0.9), 0.0, 0.028575)

    def test_propagates_swing_strategy_exception(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        table_ball_set: MagicMock,
        swing_strategy: MagicMock,
    ):
        self._setup_table(table_ball_set)
        swing_strategy.execute_aim.side_effect = ValueError("幾何無解")
        action = _action(should_execute_action=True)

        with pytest.raises(ValueError):
            demo_orchestrator._execute_aim(action)


class TestDemoTableOrchestratorExecuteStrike:
    """同樣只保留跟手臂型號無關的通用前置檢查（did_last_motion_timeout）與
    table_z/ball_radius/cue_ball 計算，實際揮桿邏輯委派給
    `RobotSwingStrategy.execute_strike()`（獨立測試見
    test_wam7_swing_strategy.py／test_ur3e_swing_strategy.py）。"""

    def _setup_table(self, table_ball_set: MagicMock, articulation_api: MagicMock) -> None:
        table_ball_set.get_table_z.return_value = 0.0
        table_ball_set.DEFAULT_BALL_RADIUS = 0.028575
        articulation_api.did_last_motion_timeout.return_value = False

    def test_delegates_to_swing_strategy_with_table_z_ball_radius_and_cue_ball(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        table_ball_set: MagicMock,
        articulation_api: MagicMock,
        swing_strategy: MagicMock,
    ):
        self._setup_table(table_ball_set, articulation_api)
        action = _action(should_execute_action=True)
        action.cue_ball_placement = [0.0, -0.9382125]

        demo_orchestrator._execute_strike(action)

        swing_strategy.execute_strike.assert_called_once_with(
            action, (0.0, -0.9382125), 0.0, 0.028575
        )

    def test_raises_without_calling_swing_strategy_when_aim_timed_out(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        table_ball_set: MagicMock,
        articulation_api: MagicMock,
        swing_strategy: MagicMock,
    ):
        self._setup_table(table_ball_set, articulation_api)
        articulation_api.did_last_motion_timeout.return_value = True
        action = _action(should_execute_action=True)

        with pytest.raises(RuntimeError):
            demo_orchestrator._execute_strike(action)

        swing_strategy.execute_strike.assert_not_called()

    def test_propagates_swing_strategy_exception(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        table_ball_set: MagicMock,
        articulation_api: MagicMock,
        swing_strategy: MagicMock,
    ):
        self._setup_table(table_ball_set, articulation_api)
        swing_strategy.execute_strike.side_effect = ValueError("幾何無解")
        action = _action(should_execute_action=True)

        with pytest.raises(ValueError):
            demo_orchestrator._execute_strike(action)

    def test_timeout_is_recorded_via_step_error_handling(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        articulation_api: MagicMock,
        error_state: MagicMock,
    ):
        # 整合既有的 TableOrchestrator.step() try/except 流程：_execute_strike
        # 因逾時拋出的例外不應該往外傳，而是被 error_state 記錄下來。
        self._setup_table(table_ball_set, articulation_api)
        articulation_api.did_last_motion_timeout.return_value = True
        action = _action(should_execute_action=True)
        script_controller.get_action.return_value = action
        script_controller.get_current_state.return_value = BilliardStatus.STRIKING

        demo_orchestrator.step(_observation())

        assert error_state.has_error() is True


class TestTrainingTableOrchestrator:
    def test_step_dispatches_reset_balls_only(
        self,
        training_orchestrator: TrainingTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.RESET
        ball_position_provider.get_positions.return_value = {0: (0.0, 0.0)}

        training_orchestrator.step(_observation())

        table_ball_set.reset.assert_called_once_with({0: (0.0, 0.0)})

    def test_reset_downstream_is_noop(
        self, training_orchestrator: TrainingTableOrchestrator
    ):
        training_orchestrator._reset_downstream()

    def test_execute_aim_is_noop(
        self, training_orchestrator: TrainingTableOrchestrator
    ):
        training_orchestrator._execute_aim(_action(should_execute_action=True))

    def test_execute_strike_calls_impulse_service_with_table_z(
        self,
        training_orchestrator: TrainingTableOrchestrator,
        impulse_striking_service: MagicMock,
        table_ball_set: MagicMock,
    ):
        table_ball_set.get_table_z.return_value = 0.75
        table_ball_set.get_table_x_y.return_value = (5.0, 3.0)
        action = _action(should_execute_action=True)

        training_orchestrator._execute_strike(action)

        impulse_striking_service.strike.assert_called_once_with(action, 5.0, 3.0, table_z=0.75)
