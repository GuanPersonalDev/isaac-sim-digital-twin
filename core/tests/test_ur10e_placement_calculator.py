import numpy as np
import pytest

from core.services import cue_pose_calculator, ur10e_placement_calculator


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


class TestComputeRollMinimizingReorientation:
    def test_flat_shot_picks_roll_closest_to_current_orientation(self):
        """實際除錯發現的案例：cue_ball=(0.0,0.5)、shot_angle=0（flat）
        搭配 HOME 附近的實際朝向，roll_rad=0 需要接近 180 度翻轉，
        roll_rad=180 度只需要 90 度——這個函式應該找到後者附近的值。"""
        cue_ball = (0.0, 0.5)
        shot_angle_deg = 0.0
        table_z = 0.0
        ball_radius = 0.028575
        current_orientation = (
            -0.00024397906963713467, 0.0003263941325712949,
            0.7071066498756409, 0.7071066498756409,
        )
        base_position = (0.0, -1.35, 0.0)

        roll_rad = ur10e_placement_calculator.compute_roll_minimizing_reorientation(
            cue_ball, shot_angle_deg, table_z, ball_radius, [0.0, 0.0], current_orientation, base_position
        )

        assert roll_rad == pytest.approx(np.pi, abs=np.radians(5.0))

    def test_returned_roll_achieves_smaller_or_equal_angle_than_zero_roll(self):
        """不管起始姿態是什麼，搜尋出來的 roll_rad 對應的夾角都不應該比
        roll_rad=0（cue_pose_calculator 的預設）差。"""
        cue_ball = (0.0, 0.5)
        shot_angle_deg = 0.0
        table_z = 0.0
        ball_radius = 0.028575
        current_orientation = (0.6, 0.1, 0.7, 0.1)
        current_orientation = tuple(
            np.asarray(current_orientation) / np.linalg.norm(current_orientation)
        )
        base_position = (0.0, -1.35, 0.0)

        def _angle_for_roll(roll_rad: float) -> float:
            _, orientation, _, _ = cue_pose_calculator.compute_tilted_wrist_pose(
                cue_ball, shot_angle_deg, table_z, ball_radius, [0.0, 0.0], roll_rad=roll_rad
            )
            dot = float(np.clip(np.abs(np.dot(current_orientation, orientation)), -1.0, 1.0))
            return 2.0 * np.arccos(dot)

        roll_rad = ur10e_placement_calculator.compute_roll_minimizing_reorientation(
            cue_ball, shot_angle_deg, table_z, ball_radius, [0.0, 0.0], current_orientation, base_position
        )

        assert _angle_for_roll(roll_rad) <= _angle_for_roll(0.0) + 1e-6
