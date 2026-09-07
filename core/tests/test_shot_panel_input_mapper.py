import math

import pytest

from core.models.action_bounds import (
    POSITION_OFFSET_HORIZONTAL,
    POSITION_OFFSET_VERTICAL,
)
from core.models.manual_shot_bounds import (
    CUE_BALL_PLACEMENT_X,
    CUE_BALL_PLACEMENT_Y,
    MANUAL_SHOT_ANGLE,
)
from core.models.table_ball_set import TableBallSet
from core.services import cue_pose_calculator, shot_panel_input_mapper
from core.services.spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH

# 面板俯瞰圖的實際規劃尺寸（128×256），128/1.27 == 256/2.54，兩軸同尺度
# 不會把球畫成橢圓，見 hud-shot-control-panel-tech-design 計畫。測試沿用
# 同一組尺寸，但函式本身對任意 view_width_px/view_height_px 都成立。
_VIEW_WIDTH_PX = 128.0
_VIEW_HEIGHT_PX = 256.0


class TestOffsetFromCirclePixels:
    _CENTER_PX = 60.0
    _CENTER_PY = 60.0
    _RADIUS_PX = 60.0

    def test_marker_at_center_yields_zero_offset(self):
        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            self._CENTER_PX, self._CENTER_PY,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=1.0,
        )

        # Assert
        assert offset == pytest.approx([0.0, 0.0])

    def test_marker_above_center_is_positive_vertical(self):
        # 畫面往上（pixel y 變小）= position_offset[0] 為正 = 上塞，由
        # test_cue_pose_calculator.py:83 釘住的方向慣例。
        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            self._CENTER_PX, self._CENTER_PY - self._RADIUS_PX,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=1.0,
        )

        # Assert
        assert offset[0] == pytest.approx(POSITION_OFFSET_VERTICAL[1])
        assert offset[1] == pytest.approx(0.0)

    def test_marker_below_center_is_negative_vertical(self):
        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            self._CENTER_PX, self._CENTER_PY + self._RADIUS_PX,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=1.0,
        )

        # Assert
        assert offset[0] == pytest.approx(POSITION_OFFSET_VERTICAL[0])

    def test_marker_right_of_center_is_positive_horizontal(self):
        # 畫面往右 = position_offset[1] 為正 = 右塞。
        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            self._CENTER_PX + self._RADIUS_PX, self._CENTER_PY,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=1.0,
        )

        # Assert
        assert offset[1] == pytest.approx(POSITION_OFFSET_HORIZONTAL[1])
        assert offset[0] == pytest.approx(0.0)

    def test_marker_left_of_center_is_negative_horizontal(self):
        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            self._CENTER_PX - self._RADIUS_PX, self._CENTER_PY,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=1.0,
        )

        # Assert
        assert offset[1] == pytest.approx(POSITION_OFFSET_HORIZONTAL[0])

    def test_marker_on_circle_at_45_degrees(self):
        # Arrange
        delta = self._RADIUS_PX / math.sqrt(2)

        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            self._CENTER_PX + delta, self._CENTER_PY - delta,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=1.0,
        )

        # Assert：上塞與右塞各半，範數等於 POSITION_OFFSET_VERTICAL[1]。
        assert offset[0] == pytest.approx(offset[1])
        assert math.hypot(*offset) == pytest.approx(POSITION_OFFSET_VERTICAL[1])

    def test_direction_is_preserved_when_dragged_outside_the_circle(self):
        # #222 回歸測試：逐軸 clip 會轉向，拖到圓外必須保持方向、只縮長度。
        # Arrange
        pre_clip_horizontal, pre_clip_vertical = 2.0, -0.1
        marker_px = self._CENTER_PX + pre_clip_horizontal * self._RADIUS_PX
        marker_py = self._CENTER_PY - pre_clip_vertical * self._RADIUS_PX

        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            marker_px, marker_py,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=1.0,
        )

        # Assert
        expected_direction = math.atan2(pre_clip_vertical, pre_clip_horizontal)
        actual_direction = math.atan2(offset[0], offset[1])
        assert actual_direction == pytest.approx(expected_direction, abs=1e-9)
        # 且長度確實被裁到邊界（範數等於允許的最大物理範數）。
        assert math.hypot(*offset) == pytest.approx(POSITION_OFFSET_VERTICAL[1])

    @pytest.mark.parametrize(
        ("max_offset", "expected_norm_fraction"),
        [(0.0, 0.0), (0.6, 0.6), (1.0, 1.0)],
    )
    def test_max_offset_scales_the_clipped_norm(
        self, max_offset: float, expected_norm_fraction: float
    ):
        # Arrange：拖到圓周正上方，正規化範數恰為 1.0，觸不觸發裁切完全由
        # max_offset 決定。
        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            self._CENTER_PX, self._CENTER_PY - self._RADIUS_PX,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=max_offset,
        )

        # Assert
        assert math.hypot(*offset) == pytest.approx(
            POSITION_OFFSET_VERTICAL[1] * expected_norm_fraction
        )

    def test_non_positive_radius_raises(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.offset_from_circle_pixels(
                0.0, 0.0, 0.0, 0.0, 0.0, max_offset=1.0
            )

    def test_non_finite_marker_position_raises(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.offset_from_circle_pixels(
                math.nan, 0.0, 0.0, 0.0, 60.0, max_offset=1.0
            )


class TestCirclePixelsFromOffset:
    _CENTER_PX = 60.0
    _CENTER_PY = 60.0
    _RADIUS_PX = 60.0

    def test_zero_offset_returns_center(self):
        # Act
        px, py = shot_panel_input_mapper.circle_pixels_from_offset(
            (0.0, 0.0), self._CENTER_PX, self._CENTER_PY, self._RADIUS_PX
        )

        # Assert
        assert (px, py) == pytest.approx((self._CENTER_PX, self._CENTER_PY))

    def test_is_the_inverse_of_offset_from_circle_pixels(self):
        # Arrange：起點在圓內，不會觸發裁切，往返應精確互逆。
        marker_px, marker_py = 90.0, 20.0

        # Act
        offset = shot_panel_input_mapper.offset_from_circle_pixels(
            marker_px, marker_py,
            self._CENTER_PX, self._CENTER_PY,
            self._RADIUS_PX, max_offset=1.0,
        )
        recovered_px, recovered_py = shot_panel_input_mapper.circle_pixels_from_offset(
            offset, self._CENTER_PX, self._CENTER_PY, self._RADIUS_PX
        )

        # Assert
        assert (recovered_px, recovered_py) == pytest.approx((marker_px, marker_py))


class TestTopviewPixelTableConversion:
    def test_center_pixel_maps_to_table_origin(self):
        # Act
        x, y = shot_panel_input_mapper.table_xy_from_topview_pixels(
            _VIEW_WIDTH_PX / 2, _VIEW_HEIGHT_PX / 2, _VIEW_WIDTH_PX, _VIEW_HEIGHT_PX
        )

        # Assert
        assert (x, y) == pytest.approx((0.0, 0.0))

    def test_top_edge_is_positive_y_and_bottom_edge_is_negative_y(self):
        # y 反向：pixel y 往下增加，桌台座標 +Y 朝 foot end，畫面「上」對應
        # 桌台座標「+Y」，跟圓形選擇器的上塞方向慣例一致。
        # Act
        _, y_top = shot_panel_input_mapper.table_xy_from_topview_pixels(
            _VIEW_WIDTH_PX / 2, 0.0, _VIEW_WIDTH_PX, _VIEW_HEIGHT_PX
        )
        _, y_bottom = shot_panel_input_mapper.table_xy_from_topview_pixels(
            _VIEW_WIDTH_PX / 2, _VIEW_HEIGHT_PX, _VIEW_WIDTH_PX, _VIEW_HEIGHT_PX
        )

        # Assert
        assert y_top == pytest.approx(TABLE_LENGTH / 2)
        assert y_bottom == pytest.approx(-TABLE_LENGTH / 2)

    @pytest.mark.parametrize(
        ("px", "py", "expected_x", "expected_y"),
        [
            (0.0, 0.0, -TABLE_WIDTH / 2, TABLE_LENGTH / 2),
            (_VIEW_WIDTH_PX, 0.0, TABLE_WIDTH / 2, TABLE_LENGTH / 2),
            (0.0, _VIEW_HEIGHT_PX, -TABLE_WIDTH / 2, -TABLE_LENGTH / 2),
            (_VIEW_WIDTH_PX, _VIEW_HEIGHT_PX, TABLE_WIDTH / 2, -TABLE_LENGTH / 2),
        ],
    )
    def test_four_corners(
        self, px: float, py: float, expected_x: float, expected_y: float
    ):
        # Act
        x, y = shot_panel_input_mapper.table_xy_from_topview_pixels(
            px, py, _VIEW_WIDTH_PX, _VIEW_HEIGHT_PX
        )

        # Assert
        assert (x, y) == pytest.approx((expected_x, expected_y))

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (0.0, 0.0),
            (-TABLE_WIDTH / 2, TABLE_LENGTH / 2),
            (TABLE_WIDTH / 2, -TABLE_LENGTH / 2),
            (0.3, -0.9525),
        ],
    )
    def test_round_trip_through_pixels_and_back(self, x: float, y: float):
        # Act
        px, py = shot_panel_input_mapper.topview_pixels_from_table_xy(
            x, y, _VIEW_WIDTH_PX, _VIEW_HEIGHT_PX
        )
        recovered_x, recovered_y = shot_panel_input_mapper.table_xy_from_topview_pixels(
            px, py, _VIEW_WIDTH_PX, _VIEW_HEIGHT_PX
        )

        # Assert
        assert (recovered_x, recovered_y) == pytest.approx((x, y))

    def test_non_positive_view_size_raises(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.table_xy_from_topview_pixels(0.0, 0.0, 0.0, 100.0)


class TestClampCueBallPlacement:
    def test_value_within_bounds_is_unchanged(self):
        # Act
        x, y = shot_panel_input_mapper.clamp_cue_ball_placement(0.1, -0.9)

        # Assert
        assert (x, y) == pytest.approx((0.1, -0.9))

    def test_each_axis_is_clamped_independently(self):
        # 逐軸夾：跟偏移的圓形裁切刻意不同——擺位合法域是矩形，逐軸夾出來
        #仍是最近的合法點，不需要（也不應該）用圓形裁切的邏輯。
        # Act
        x, y = shot_panel_input_mapper.clamp_cue_ball_placement(10.0, -10.0)

        # Assert
        assert x == CUE_BALL_PLACEMENT_X[1]
        assert y == CUE_BALL_PLACEMENT_Y[0]

    def test_corner_case_clamps_to_the_nearest_legal_corner(self):
        # Act
        x, y = shot_panel_input_mapper.clamp_cue_ball_placement(
            CUE_BALL_PLACEMENT_X[1] + 1.0, CUE_BALL_PLACEMENT_Y[1] + 1.0
        )

        # Assert
        assert x == CUE_BALL_PLACEMENT_X[1]
        assert y == CUE_BALL_PLACEMENT_Y[1]

    def test_non_finite_input_raises(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.clamp_cue_ball_placement(math.nan, 0.0)


class TestShotAngleFromPoints:
    _CUE_BALL = (0.0, -0.9525)

    def test_straight_toward_positive_y_is_zero_degrees(self):
        # Act
        angle = shot_panel_input_mapper.shot_angle_from_points(
            self._CUE_BALL, (self._CUE_BALL[0], self._CUE_BALL[1] + 1.0)
        )

        # Assert
        assert angle == pytest.approx(0.0)

    def test_straight_toward_negative_x_is_positive_ninety(self):
        # "0° 朝桌台 +Y，正角朝 -X 增加"（action.py docstring）。
        # Act
        angle = shot_panel_input_mapper.shot_angle_from_points(
            self._CUE_BALL, (self._CUE_BALL[0] - 1.0, self._CUE_BALL[1])
        )

        # Assert
        assert angle == pytest.approx(90.0)

    def test_straight_toward_positive_x_is_negative_ninety(self):
        # Act
        angle = shot_panel_input_mapper.shot_angle_from_points(
            self._CUE_BALL, (self._CUE_BALL[0] + 1.0, self._CUE_BALL[1])
        )

        # Assert
        assert angle == pytest.approx(-90.0)

    def test_straight_toward_negative_y_pins_to_negative_180_not_positive(self):
        # 值域是半開區間 [-180, 180)，正對頭庫（母球正後方）回 -180.0 不是
        # +180.0——這是 atan2 對帶號零的既定行為，用 -(tx-cx) 而不是
        # (cx-tx) 就是為了保留這個號誌。
        # Act
        angle = shot_panel_input_mapper.shot_angle_from_points(
            self._CUE_BALL, (self._CUE_BALL[0], self._CUE_BALL[1] - 1.0)
        )

        # Assert
        assert angle == -180.0

    def test_is_translation_invariant(self):
        # Arrange
        shift = (5.0, -3.0)
        target = (self._CUE_BALL[0] - 0.4, self._CUE_BALL[1] + 0.6)

        # Act
        original = shot_panel_input_mapper.shot_angle_from_points(
            self._CUE_BALL, target
        )
        shifted = shot_panel_input_mapper.shot_angle_from_points(
            (self._CUE_BALL[0] + shift[0], self._CUE_BALL[1] + shift[1]),
            (target[0] + shift[0], target[1] + shift[1]),
        )

        # Assert
        assert shifted == pytest.approx(original)

    @pytest.mark.parametrize(
        "angle_deg",
        [-159.9, -90.0, -45.0, -1.0, 0.0, 1.0, 45.0, 90.0, 159.9],
    )
    def test_round_trips_with_compute_tilted_direction(self, angle_deg: float):
        # 最有價值的一條：把新函式直接釘在既有幾何上。shot_angle_from_points
        # 用 atan2(-(tx-cx), ty-cy)，跟 compute_tilted_direction() 的
        # (-sinθ, cosθ) 互為反函式。
        # Arrange
        direction = cue_pose_calculator.compute_tilted_direction(angle_deg, 0.0)
        cue_ball_xy = (0.3, -0.9)
        target_xy = (
            cue_ball_xy[0] + float(direction[0]),
            cue_ball_xy[1] + float(direction[1]),
        )

        # Act
        recovered = shot_panel_input_mapper.shot_angle_from_points(
            cue_ball_xy, target_xy
        )

        # Assert
        assert recovered == pytest.approx(angle_deg, abs=1e-7)

    def test_points_closer_than_ball_radius_raise(self):
        # Arrange
        epsilon = TableBallSet.DEFAULT_BALL_RADIUS * 0.5
        target = (self._CUE_BALL[0], self._CUE_BALL[1] + epsilon)

        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.shot_angle_from_points(self._CUE_BALL, target)

    def test_identical_points_raise(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.shot_angle_from_points(
                self._CUE_BALL, self._CUE_BALL
            )

    def test_non_finite_point_raises(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.shot_angle_from_points(
                self._CUE_BALL, (math.nan, 0.0)
            )


class TestClampManualShotAngle:
    def test_value_within_bounds_is_unchanged(self):
        # Assert
        assert shot_panel_input_mapper.clamp_manual_shot_angle(10.0) == 10.0

    def test_value_above_upper_bound_is_clamped(self):
        # Assert
        assert (
            shot_panel_input_mapper.clamp_manual_shot_angle(200.0)
            == MANUAL_SHOT_ANGLE[1]
        )

    def test_value_below_lower_bound_is_clamped(self):
        # Assert
        assert (
            shot_panel_input_mapper.clamp_manual_shot_angle(-200.0)
            == MANUAL_SHOT_ANGLE[0]
        )

    def test_non_finite_value_raises(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.clamp_manual_shot_angle(math.nan)


class TestAimLineEndpoint:
    def test_endpoint_direction_matches_compute_tilted_direction(self):
        # Arrange
        cue_ball_xy = (0.0, -0.9525)
        shot_angle_deg = 30.0

        # Act
        endpoint = shot_panel_input_mapper.aim_line_endpoint(
            cue_ball_xy, shot_angle_deg
        )

        # Assert
        direction = cue_pose_calculator.compute_tilted_direction(shot_angle_deg, 0.0)
        dx = endpoint[0] - cue_ball_xy[0]
        dy = endpoint[1] - cue_ball_xy[1]
        assert math.atan2(dx, dy) == pytest.approx(
            math.atan2(direction[0], direction[1])
        )

    def test_center_at_0_degrees_hits_foot_rail(self):
        # 0° 朝 +Y（compute_tilted_direction(0, 0) == (0, 1)），桌台中心
        # 沿 +Y 一路走到頭庫（foot rail），交點是 (0, +TABLE_LENGTH/2)。
        # Act
        endpoint = shot_panel_input_mapper.aim_line_endpoint((0.0, 0.0), 0.0)

        # Assert
        assert endpoint == pytest.approx((0.0, TABLE_LENGTH / 2.0))

    def test_center_at_90_degrees_hits_left_rail(self):
        # 正角朝 -X（compute_tilted_direction(90, 0) == (-1, 0)），交點是
        # (-TABLE_WIDTH/2, 0)。
        # Act
        endpoint = shot_panel_input_mapper.aim_line_endpoint((0.0, 0.0), 90.0)

        # Assert
        assert endpoint == pytest.approx((-TABLE_WIDTH / 2.0, 0.0))

    def test_center_at_negative_90_degrees_hits_right_rail(self):
        # 負角朝 +X（compute_tilted_direction(-90, 0) == (1, 0)），交點是
        # (+TABLE_WIDTH/2, 0)。
        # Act
        endpoint = shot_panel_input_mapper.aim_line_endpoint((0.0, 0.0), -90.0)

        # Assert
        assert endpoint == pytest.approx((TABLE_WIDTH / 2.0, 0.0))

    def test_center_at_180_degrees_hits_head_rail(self):
        # 180° 朝 -Y（compute_tilted_direction(180, 0) ≈ (0, -1)），交點是
        # (0, -TABLE_LENGTH/2)。
        # Act
        endpoint = shot_panel_input_mapper.aim_line_endpoint((0.0, 0.0), 180.0)

        # Assert
        assert endpoint == pytest.approx((0.0, -TABLE_LENGTH / 2.0))

    def test_kitchen_point_at_30_degrees_hits_side_rail(self):
        # 手算推導：cue_ball_xy = (0, -0.9525)，30°。
        # direction = compute_tilted_direction(30, 0) = (-sin30, cos30)
        #           = (-0.5, cos30)，dir_x < 0 → 候選邊界是左側 x =
        #           -TABLE_WIDTH/2 = -0.635。
        # t_x = (-0.635 - 0) / -0.5 = 1.27。
        # 另一軸驗證交點確實先撞 x 邊界：
        # t_y = (TABLE_LENGTH/2 - (-0.9525)) / cos30
        #     = (1.27 + 0.9525) / cos30 ≈ 2.566 > t_x = 1.27，
        # 所以射線先撞到 x 邊界，min(t_x, t_y) == t_x。
        # endpoint_x = -TABLE_WIDTH/2 = -0.635
        # endpoint_y = -0.9525 + t_x * cos30 = -0.9525 + 1.27 * cos30
        #            = -0.9525 + TABLE_WIDTH/2 / tan30（等價寫法，兩者
        #              相除得到同一個 t_x = TABLE_WIDTH/2 / sin30 = 1.27）
        # Arrange
        cue_ball_xy = (0.0, -0.9525)
        shot_angle_deg = 30.0

        # Act
        endpoint = shot_panel_input_mapper.aim_line_endpoint(
            cue_ball_xy, shot_angle_deg
        )

        # Assert
        expected_x = -TABLE_WIDTH / 2.0
        expected_y = cue_ball_xy[1] + (TABLE_WIDTH / 2.0) / math.tan(
            math.radians(shot_angle_deg)
        )
        assert endpoint == pytest.approx((expected_x, expected_y))

    def test_endpoint_lands_exactly_on_corner_when_aimed_there(self):
        # 正對角落的方向：兩軸同時到界，t_x == t_y。用
        # shot_angle_from_points() 反推瞄準角落所需的角度，驗證
        # aim_line_endpoint() 與它互為反函式的邊界情形。
        # Arrange
        cue_ball_xy = (0.0, 0.0)
        corner_xy = (TABLE_WIDTH / 2.0, TABLE_LENGTH / 2.0)
        shot_angle_deg = shot_panel_input_mapper.shot_angle_from_points(
            cue_ball_xy, corner_xy
        )

        # Act
        endpoint = shot_panel_input_mapper.aim_line_endpoint(
            cue_ball_xy, shot_angle_deg
        )

        # Assert
        assert endpoint == pytest.approx(corner_xy)

    def test_cue_ball_already_on_boundary_facing_outward_returns_itself(self):
        # 母球已在頭庫邊界上，0° 朝 +Y（繼續往外），t 應為 0——零長度，面板
        # 不畫線。
        # Arrange
        cue_ball_xy = (0.0, TABLE_LENGTH / 2.0)

        # Act
        endpoint = shot_panel_input_mapper.aim_line_endpoint(cue_ball_xy, 0.0)

        # Assert
        assert endpoint == pytest.approx(cue_ball_xy)

    def test_non_finite_angle_raises(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.aim_line_endpoint((0.0, 0.0), math.nan)


class TestIsWithinRadius:
    def test_point_at_target_is_within(self):
        # Assert
        assert shot_panel_input_mapper.is_within_radius(1.0, 1.0, 1.0, 1.0, 5.0)

    def test_point_exactly_on_boundary_is_within(self):
        # Assert
        assert shot_panel_input_mapper.is_within_radius(0.0, 5.0, 0.0, 0.0, 5.0)

    def test_point_outside_radius_is_not_within(self):
        # Assert
        assert not shot_panel_input_mapper.is_within_radius(0.0, 5.1, 0.0, 0.0, 5.0)

    def test_non_finite_point_raises(self):
        # Assert
        with pytest.raises(ValueError):
            shot_panel_input_mapper.is_within_radius(math.nan, 0.0, 0.0, 0.0, 5.0)
