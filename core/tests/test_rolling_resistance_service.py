import math
from unittest.mock import MagicMock

import pytest

from core.services.rolling_resistance_service import (
    GRAVITY,
    ROLLING_FRICTION_COEFF,
    NEGLIGIBLE_SPEED_THRESHOLD,
    NEGLIGIBLE_SPIN_THRESHOLD,
    PHYSICS_DT,
    SPIN_DECAY_RATE,
    RollingResistanceService,
)

BALL_RADIUS = 0.03
BALL_PATH = "/World/Table_0/Balls/Ball_0"


def _batched_rigid_body_mock() -> MagicMock:
    """RigidBodyAPI 的測試替身。

    RollingResistanceService 已改成每個 tick 只做一次批次讀取
    （`get_velocities(paths)`，見 core/ports/rigid_body_api.py 的效能說明），
    但測試資料用「這顆球的線速度是多少、角速度是多少」描述最好讀，所以這裡
    把批次介面接回逐顆的 mock 屬性上，實際取值發生在 service 呼叫的當下。
    受測對象呼叫的仍然是批次介面。
    """
    api = MagicMock()
    api.get_velocities.side_effect = lambda paths: (
        [api.get_linear_velocity(path) for path in paths],
        [api.get_angular_velocity(path) for path in paths],
    )
    return api


class TestApplyLinearDecay:
    def test_apply_decays_linear_velocity_by_rolling_friction(self):
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.return_value = [1.0, 0.0, 0.0]
        # 純滾動狀態（無殘留自旋）：ω_roll = (-vy, vx, 0) / R
        rigid_body_api.get_angular_velocity.return_value = [0.0, 1.0 / BALL_RADIUS, 0.0]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        expected_delta_v = 0.01 * GRAVITY * PHYSICS_DT
        prim_path, linear, _ = rigid_body_api.set_velocities.call_args[0]
        assert prim_path == BALL_PATH
        assert linear[0] == pytest.approx(1.0 - expected_delta_v)
        assert linear[1] == pytest.approx(0.0)

    def test_apply_preserves_vertical_velocity_component(self):
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.return_value = [1.0, 0.0, -0.5]
        rigid_body_api.get_angular_velocity.return_value = [0.0, 1.0 / BALL_RADIUS, 0.0]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        _, linear, _ = rigid_body_api.set_velocities.call_args[0]
        assert linear[2] == pytest.approx(-0.5)

    def test_apply_does_not_touch_ball_when_only_vertical_residual_remains(self):
        """水平方向與自旋都只是沉降/接觸解算的數值雜訊量級（不是真的在滾動
        或自旋）時，即使垂直分量還帶著明顯大於 NEGLIGIBLE_SPEED_THRESHOLD
        的殘留（實測沉降/接觸解算殘留約 0.069 m/s），這個服務也完全不該再
        呼叫 set_velocities()——垂直分量從頭到尾不是滾動摩擦要處理的對象，
        不該被這裡寫入或清零。實測發現：只要每個 tick 持續呼叫
        set_velocities()（哪怕只是把水平分量原封不動寫回同樣的 0），這個
        顯式寫入本身就會讓 PhysX 永遠沒機會把球放進 sleep 狀態，導致垂直
        殘留被迫每個 tick 重新計算出類似大小的值，永遠卡在門檻附近不消失
        （見 GUI 實測回報：9 顆 rack 球卡在 vz≈0.0687 永久不動，導致
        is_ball_moving 永遠是 True，狀態機卡死在 IDLE）。純物理環境不受
        干擾時，這個殘留會在 0.4 秒內自然收斂到 0 並保持 sleep，因此正確
        做法是完全跳過寫入，把它交還給 PhysX 自己處理。"""
        rigid_body_api = _batched_rigid_body_mock()
        frozen_vz = 0.0687  # 實測回報的凍結值，刻意大於 NEGLIGIBLE_SPEED_THRESHOLD
        rigid_body_api.get_linear_velocity.return_value = [0.0, 0.0, frozen_vz]
        rigid_body_api.get_angular_velocity.return_value = [0.0, 0.0, 0.0]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        rigid_body_api.set_velocities.assert_not_called()

    def test_apply_does_not_touch_ball_with_only_settling_noise_on_horizontal_and_spin(self):
        """水平線速度與殘留自旋都只是沉降/多球接觸解算殘留的數值雜訊量級
        （實測約 1e-7~1e-5 m/s、1e-4~1e-3 rad/s），遠低於 NEGLIGIBLE_SPEED_
        THRESHOLD／NEGLIGIBLE_SPIN_THRESHOLD 這兩個視覺門檻，但不是精確的
        0——這種情況也該完全跳過寫入，不能因為兩者「都低於視覺門檻」就觸發
        一次夾到 0 的寫入（那樣做本身就會持續打斷 PhysX 的 sleep 判定，見
        test_apply_does_not_touch_ball_when_only_vertical_residual_remains
        的說明）。"""
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.return_value = [7.8e-6, 0.0, 0.06872]
        rigid_body_api.get_angular_velocity.return_value = [0.0007, 0.0003, 0.0]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        rigid_body_api.set_velocities.assert_not_called()


class TestApplyClamping:
    def test_apply_clamps_to_zero_when_delta_exceeds_speed(self):
        rigid_body_api = _batched_rigid_body_mock()
        # rolling_friction_coeff 刻意設大，讓這個 tick 該扣的量超過目前球速
        rigid_body_api.get_linear_velocity.return_value = [0.025, 0.0, 0.0]
        rigid_body_api.get_angular_velocity.return_value = [0.0, 0.025 / BALL_RADIUS, 0.0]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS, rolling_friction_coeff=1.0)

        service.apply([BALL_PATH])

        _, linear, angular = rigid_body_api.set_velocities.call_args[0]
        assert linear[0] == pytest.approx(0.0)
        assert linear[1] == pytest.approx(0.0)
        assert angular[1] == pytest.approx(0.0)

    def test_apply_clamps_linear_velocity_to_exact_zero_below_negligible_threshold(self):
        """速度低於視覺門檻時要真的夾到 0（不是放著不管），
        否則球會用門檻附近的殘留速度永遠移動下去（見 #203 後續回報的 bug）。"""
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.return_value = [NEGLIGIBLE_SPEED_THRESHOLD / 2, 0.0, 0.0]
        rigid_body_api.get_angular_velocity.return_value = [
            0.0,
            (NEGLIGIBLE_SPEED_THRESHOLD / 2) / BALL_RADIUS,
            0.0,
        ]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        rigid_body_api.set_velocities.assert_called_once()
        _, linear, angular = rigid_body_api.set_velocities.call_args[0]
        assert linear[0] == pytest.approx(0.0)
        assert linear[1] == pytest.approx(0.0)
        assert angular[1] == pytest.approx(0.0)

    def test_apply_skips_ball_already_fully_at_rest(self):
        """線速度與殘留自旋都已經完全是 0 時，不需要重複寫入同樣的 0。"""
        rigid_body_api = _batched_rigid_body_mock()
        rigid_body_api.get_linear_velocity.return_value = [0.0, 0.0, 0.0]
        rigid_body_api.get_angular_velocity.return_value = [0.0, 0.0, 0.0]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        rigid_body_api.set_velocities.assert_not_called()


class TestApplyAngularDecomposition:
    def test_apply_decays_residual_angular_velocity_by_spin_decay_rate(self):
        """殘留分量（加塞／side-spin）依球-呢絨自旋衰減率獨立衰減，
        不是像滾動分量的舊設計那樣完全不動。"""
        rigid_body_api = _batched_rigid_body_mock()
        vx, vy = 1.0, 0.0
        roll_wx_before = -vy / BALL_RADIUS
        roll_wy_before = vx / BALL_RADIUS
        residual = (2.0, 3.0, 5.0)  # 模擬加塞／side-spin 殘留分量
        rigid_body_api.get_linear_velocity.return_value = [vx, vy, 0.0]
        rigid_body_api.get_angular_velocity.return_value = [
            roll_wx_before + residual[0],
            roll_wy_before + residual[1],
            residual[2],
        ]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        _, linear, angular = rigid_body_api.set_velocities.call_args[0]
        new_vx, new_vy, _ = linear
        roll_wx_after = -new_vy / BALL_RADIUS
        roll_wy_after = new_vx / BALL_RADIUS
        new_residual = (
            angular[0] - roll_wx_after,
            angular[1] - roll_wy_after,
            angular[2],
        )

        residual_magnitude = math.sqrt(sum(c**2 for c in residual))
        expected_delta_w = SPIN_DECAY_RATE * PHYSICS_DT
        expected_scale = (residual_magnitude - expected_delta_w) / residual_magnitude
        assert new_residual[0] == pytest.approx(residual[0] * expected_scale)
        assert new_residual[1] == pytest.approx(residual[1] * expected_scale)
        assert new_residual[2] == pytest.approx(residual[2] * expected_scale)

    def test_apply_clamps_residual_angular_velocity_to_zero_below_negligible_threshold(self):
        rigid_body_api = _batched_rigid_body_mock()
        vx, vy = 1.0, 0.0
        roll_wx_before = -vy / BALL_RADIUS
        roll_wy_before = vx / BALL_RADIUS
        tiny_residual_z = NEGLIGIBLE_SPIN_THRESHOLD / 2
        rigid_body_api.get_linear_velocity.return_value = [vx, vy, 0.0]
        rigid_body_api.get_angular_velocity.return_value = [roll_wx_before, roll_wy_before, tiny_residual_z]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        _, _, angular = rigid_body_api.set_velocities.call_args[0]
        assert angular[2] == pytest.approx(0.0)

    def test_apply_scales_rolling_angular_component_with_linear_velocity(self):
        rigid_body_api = _batched_rigid_body_mock()
        vx, vy = 1.0, 0.0
        # 純滾動、無殘留分量
        rigid_body_api.get_linear_velocity.return_value = [vx, vy, 0.0]
        rigid_body_api.get_angular_velocity.return_value = [-vy / BALL_RADIUS, vx / BALL_RADIUS, 0.0]
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        service.apply([BALL_PATH])

        _, linear, angular = rigid_body_api.set_velocities.call_args[0]
        new_vx, new_vy, _ = linear
        # 無殘留分量時，衰減後仍應精確滿足滾動不打滑關係
        assert angular[0] == pytest.approx(-new_vy / BALL_RADIUS)
        assert angular[1] == pytest.approx(new_vx / BALL_RADIUS)


class TestApplyBatchesReads:
    """效能契約（見 core/ports/rigid_body_api.py）：整桌球的速度只能讀一次。
    逐顆版本每顆要兩次同步（線速度＋角速度各一次），10 顆就是 20 次。"""

    def test_reads_all_ball_velocities_in_a_single_batched_call(self):
        # Arrange
        rigid_body_api = MagicMock()
        ball_paths = [f"/World/Table_0/Balls/Ball_{i}" for i in range(10)]
        rigid_body_api.get_velocities.return_value = (
            [[1.0, 0.0, 0.0] for _ in ball_paths],
            [[0.0, 1.0 / BALL_RADIUS, 0.0] for _ in ball_paths],
        )
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        # Act
        service.apply(ball_paths)

        # Assert
        rigid_body_api.get_velocities.assert_called_once_with(ball_paths)
        rigid_body_api.get_linear_velocity.assert_not_called()
        rigid_body_api.get_angular_velocity.assert_not_called()

    def test_applies_each_ball_own_velocity_not_the_first_one(self):
        """批次化最容易寫錯的地方：把整批的第一筆值套到每顆球身上。"""
        # Arrange
        rigid_body_api = MagicMock()
        ball_paths = ["/World/Table_0/Balls/Ball_0", "/World/Table_0/Balls/Ball_1"]
        rigid_body_api.get_velocities.return_value = (
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 1.0 / BALL_RADIUS, 0.0], [0.0, 2.0 / BALL_RADIUS, 0.0]],
        )
        service = RollingResistanceService(rigid_body_api, ball_radius=BALL_RADIUS)

        # Act
        service.apply(ball_paths)

        # Assert
        expected_delta_v = ROLLING_FRICTION_COEFF * GRAVITY * PHYSICS_DT
        calls = rigid_body_api.set_velocities.call_args_list
        assert [call[0][0] for call in calls] == ball_paths
        assert calls[0][0][1][0] == pytest.approx(1.0 - expected_delta_v)
        assert calls[1][0][1][0] == pytest.approx(2.0 - expected_delta_v)
