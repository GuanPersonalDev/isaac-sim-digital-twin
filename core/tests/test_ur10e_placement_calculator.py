import pytest

from core.services import ur10e_placement_calculator


class TestComputeBasePosition:
    def test_flat_shot_along_positive_y_places_base_behind_wrist(self):
        wrist_position = (0.0, -2.093, 0.181)
        direction_unit = (0.0, -1.0, 0.0)
        table_z = 0.0

        base_position = ur10e_placement_calculator.compute_base_position(
            wrist_position, direction_unit, table_z
        )

        assert base_position == pytest.approx((0.0, -2.093 + 0.5, 0.0))

    def test_shot_along_positive_x_places_base_behind_wrist_on_x_axis(self):
        wrist_position = (1.2, 0.3, 0.1)
        direction_unit = (1.0, 0.0, 0.0)
        table_z = 0.0

        base_position = ur10e_placement_calculator.compute_base_position(
            wrist_position, direction_unit, table_z
        )

        assert base_position == pytest.approx((1.2 - 0.5, 0.3, 0.0))

    def test_base_z_matches_table_z_not_wrist_z(self):
        wrist_position = (0.0, 0.0, 0.5)
        direction_unit = (0.0, 1.0, 0.0)
        table_z = -0.6

        base_position = ur10e_placement_calculator.compute_base_position(
            wrist_position, direction_unit, table_z
        )

        assert base_position[2] == pytest.approx(-0.6)

    def test_tilted_direction_z_component_ignored_for_base_offset(self):
        """高架橋案例的 direction_unit 可能帶 Z 分量（cue_pose_calculator.
        compute_tilted_direction()），基座只依水平分量退開，Z 固定跟桌面
        同高，不隨 tilt 變化。"""
        wrist_position = (0.0, -1.0, 0.2)
        direction_flat = (0.0, -1.0, 0.0)
        direction_tilted = (0.0, -0.985, 0.172)
        table_z = 0.0

        base_flat = ur10e_placement_calculator.compute_base_position(
            wrist_position, direction_flat, table_z
        )
        base_tilted = ur10e_placement_calculator.compute_base_position(
            wrist_position, direction_tilted, table_z
        )

        assert base_flat[2] == pytest.approx(0.0)
        assert base_tilted[2] == pytest.approx(0.0)
        # 水平分量幾乎歸一化後方向接近，退開後的 X/Y 應該很接近（誤差來自
        # tilted 方向本身水平分量歸一化的些微差異）。
        assert base_flat[0] == pytest.approx(base_tilted[0], abs=0.02)
        assert base_flat[1] == pytest.approx(base_tilted[1], abs=0.02)

    def test_zero_horizontal_direction_keeps_wrist_xy(self):
        """direction_unit 水平分量為 0（例如純垂直方向，理論邊界案例）時
        不做除以零，基座 X/Y 直接等於 wrist 目標的 X/Y。"""
        wrist_position = (0.3, 0.4, 0.5)
        direction_unit = (0.0, 0.0, 1.0)
        table_z = 0.0

        base_position = ur10e_placement_calculator.compute_base_position(
            wrist_position, direction_unit, table_z
        )

        assert base_position == pytest.approx((0.3, 0.4, 0.0))
