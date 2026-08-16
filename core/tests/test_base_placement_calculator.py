import math

import pytest

from core.services.base_placement_calculator import (
    CANONICAL_REST_JOINTS,
    CUE_STICK_GRIP_TO_TIP,
    _LOCAL_TIP_HEIGHT,
    _LOCAL_TIP_RADIUS,
    compute_base_pose,
    compute_joint_targets,
    required_grip_position,
)

_BASE_YAW_JOINT_LIMIT = 2.6
"""wam_base_yaw_joint 的限位（rad），見 assets/barrett_wam/wam7.urdf。"""


class TestRequiredGripPosition:
    def test_aiming_straight_up_table(self):
        """0° 朝 +Y：握把應該退到母球正後方（-Y）。"""
        grip = required_grip_position(0.0, 0.0, 0.0)

        assert grip == pytest.approx((0.0, -CUE_STICK_GRIP_TO_TIP))

    def test_aiming_toward_negative_x(self):
        """90°：正角朝 -X，握把應該退到母球的 +X 側。"""
        grip = required_grip_position(0.0, 0.0, 90.0)

        assert grip == pytest.approx((CUE_STICK_GRIP_TO_TIP, 0.0))

    def test_aiming_toward_positive_x(self):
        """-90°：握把應該退到母球的 -X 側。"""
        grip = required_grip_position(0.0, 0.0, -90.0)

        assert grip == pytest.approx((-CUE_STICK_GRIP_TO_TIP, 0.0))

    @pytest.mark.parametrize(
        "cue_ball,target",
        [
            ((0.606425, -0.635), (0.0, 0.635)),  # Kitchen 近角落
            ((0.606425, -1.241425), (0.0, 0.635)),  # Kitchen 遠角落
        ],
    )
    def test_matches_reachability_doc_kitchen_corners(
        self, cue_ball: tuple[float, float], target: tuple[float, float]
    ):
        """對照 docs/issue-180-reachability-analysis.md 第九節的兩個代表角落。

        角度由母球與目標球的幾何現算（不寫死度數），握把需求位置則直接用
        「沿瞄準反方向退開 G」的向量算法獨立算一次，驗證角度換算與退開方向
        兩端接得起來。
        """
        angle_deg = math.degrees(
            math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1])
        )
        away_distance = math.dist(cue_ball, target)
        away_unit = (
            (cue_ball[0] - target[0]) / away_distance,
            (cue_ball[1] - target[1]) / away_distance,
        )
        expected_grip = (
            cue_ball[0] + CUE_STICK_GRIP_TO_TIP * away_unit[0],
            cue_ball[1] + CUE_STICK_GRIP_TO_TIP * away_unit[1],
        )

        grip = required_grip_position(cue_ball[0], cue_ball[1], angle_deg)

        assert grip == pytest.approx(expected_grip)


class TestComputeBasePose:
    def test_base_yaw_is_shot_angle_plus_quarter_turn(self):
        """base_yaw = θ + 90°（rad），見 scripts/probe_canonical_pose.py 的實測：
        base_yaw 每加 δ，桿尖方向角同步偏轉 δ（同向、1:1，非反向或其他比例）。
        """
        for shot_angle_deg in (-30.0, -10.0, 0.0, 10.0, 30.0):
            _, base_yaw_rad = compute_base_pose(0.0, 0.0, shot_angle_deg, table_z=0.0)

            assert base_yaw_rad == pytest.approx(
                math.radians(shot_angle_deg) + math.pi / 2.0
            )

    @pytest.mark.parametrize("shot_angle_deg", [-27.586, -30.0, 0.0, 27.586, 30.0])
    def test_base_yaw_stays_inside_joint_limit_for_fallback_b_cone(
        self, shot_angle_deg: float
    ):
        """fallback (b) 的瞄向球堆窄角錐（±30°，含接觸窗口的 ±27.586° legal aim）
        算出的 base_yaw 必須落在 wam_base_yaw_joint 的限位內，否則這支公式在
        legal 範圍內就已經算出不可行的關節目標。
        """
        _, base_yaw_rad = compute_base_pose(0.0, 0.0, shot_angle_deg, table_z=0.0)

        assert -_BASE_YAW_JOINT_LIMIT <= base_yaw_rad <= _BASE_YAW_JOINT_LIMIT

    def test_rotating_the_local_tip_offset_by_base_yaw_reproduces_the_grip_point(self):
        """round-trip 一致性檢查，不依賴公式本身的推導過程：

        把 base_yaw=0 時量到的桿尖本地向量（_LOCAL_TIP_RADIUS 方向沿
        (cos(base_yaw), sin(base_yaw)) 旋轉）加回 base_position，必須精確
        還原成 required_grip_position() 的握把需求點——這正是
        scripts/probe_canonical_pose.py 實際量測、且 align_prim_to_target
        會依賴的幾何關係，用來抓公式本身正負號寫反的問題。
        """
        cue_ball = (0.606425, -0.635)
        shot_angle_deg = 25.524

        base_position, base_yaw_rad = compute_base_pose(
            cue_ball[0], cue_ball[1], shot_angle_deg, table_z=0.0
        )
        expected_grip = required_grip_position(
            cue_ball[0], cue_ball[1], shot_angle_deg
        )

        reconstructed_tip_xy = (
            base_position[0] + _LOCAL_TIP_RADIUS * math.cos(base_yaw_rad),
            base_position[1] + _LOCAL_TIP_RADIUS * math.sin(base_yaw_rad),
        )

        assert reconstructed_tip_xy == pytest.approx(expected_grip)

    def test_base_height_cancels_the_measured_tip_offset(self):
        """base_z 要讓桿尖世界高度貼齊球心高度（桌面 + 一個球半徑）。"""
        table_z = 0.0
        ball_radius = 0.028575
        base_position, _ = compute_base_pose(
            0.0, 0.0, 0.0, table_z=table_z, ball_radius=ball_radius
        )

        tip_z = base_position[2] + _LOCAL_TIP_HEIGHT

        assert tip_z == pytest.approx(table_z + ball_radius)

    def test_base_height_is_independent_of_cue_ball_position_and_angle(self):
        """base_z 只跟 table_z、ball_radius 有關，跟母球位置／瞄準角無關。"""
        base_a, _ = compute_base_pose(0.0, 0.0, 0.0, table_z=0.0)
        base_b, _ = compute_base_pose(0.6, -1.2, 25.5, table_z=0.0)

        assert base_a[2] == pytest.approx(base_b[2])

    def test_base_height_shifts_with_table_z(self):
        base_a, _ = compute_base_pose(0.0, 0.0, 0.0, table_z=0.0)
        base_b, _ = compute_base_pose(0.0, 0.0, 0.0, table_z=0.5)

        assert base_b[2] - base_a[2] == pytest.approx(0.5)

    def test_custom_ball_radius_shifts_base_height(self):
        default_radius = 0.028575
        base_default, _ = compute_base_pose(
            0.0, 0.0, 0.0, table_z=0.0, ball_radius=default_radius
        )
        base_custom, _ = compute_base_pose(0.0, 0.0, 0.0, table_z=0.0, ball_radius=0.05)

        assert base_custom[2] - base_default[2] == pytest.approx(0.05 - default_radius)


class TestCanonicalRestJoints:
    def test_has_six_joints_excluding_base_yaw(self):
        """base_yaw 每球都變，不屬於固定姿態，其餘 6 個關節才是。"""
        assert len(CANONICAL_REST_JOINTS) == 6


class TestComputeJointTargets:
    def test_returns_seven_dof_in_urdf_order(self):
        """[base_yaw, *CANONICAL_REST_JOINTS]，對應 wam7.urdf 的 7 個關節順序。"""
        joint_targets = compute_joint_targets(0.0)

        assert len(joint_targets) == 7
        assert joint_targets[1:] == list(CANONICAL_REST_JOINTS)

    def test_base_yaw_matches_compute_base_pose(self):
        """跟 compute_base_pose() 用同一套 base_yaw 換算，兩者不能各算一次而漂移。"""
        shot_angle_deg = 12.3

        joint_targets = compute_joint_targets(shot_angle_deg)
        _, base_yaw_rad = compute_base_pose(0.0, 0.0, shot_angle_deg, table_z=0.0)

        assert joint_targets[0] == pytest.approx(base_yaw_rad)

    @pytest.mark.parametrize("shot_angle_deg", [-27.586, -30.0, 0.0, 27.586, 30.0])
    def test_base_yaw_component_stays_inside_joint_limit(self, shot_angle_deg: float):
        joint_targets = compute_joint_targets(shot_angle_deg)

        assert -_BASE_YAW_JOINT_LIMIT <= joint_targets[0] <= _BASE_YAW_JOINT_LIMIT
