import math

import numpy as np
import pytest

from core.services.ur3e_placement_calculator import (
    _FLAT_PLACEMENT,
    _PLACEMENT_LOOKUP_GRID,
    compute_bridge_base_position_and_joint_targets,
    compute_flat_base_position_and_joint_targets,
    lookup_placement_constants,
)


def _reconstruct_wrist_position(base_position, joint_targets, direction_local, local_tip_position):
    """`_solve_base_position_and_joint_targets()` 存在的理由是保證
    `base_position + Rz(required_pan) @ local_tip_position == target_wrist_
    position`——這個函式獨立重算這個等式左邊，拿來跟呼叫端給的
    target_wrist_position 比對，是這個模組最重要的正確性契約（沒有算對
    這個旋轉，第一版就曾經算出跟真實目標點差了一大截的 base_position）。
    """
    required_pan = joint_targets[0]
    cos_p, sin_p = math.cos(required_pan), math.sin(required_pan)
    local_tip = np.asarray(local_tip_position, dtype=float)
    rotated_tip = np.array([
        local_tip[0] * cos_p - local_tip[1] * sin_p,
        local_tip[0] * sin_p + local_tip[1] * cos_p,
        local_tip[2],
    ])
    return np.asarray(base_position, dtype=float) + rotated_tip


class TestLookupPlacementConstants:
    def test_exact_grid_point_returns_its_own_row(self):
        joints_pan0, direction_local, local_tip_position, speed_per_unit_omega = lookup_placement_constants(-0.635)

        expected = next(row for row in _PLACEMENT_LOOKUP_GRID if row[0] == -0.635)
        assert joints_pan0 == expected[1]
        assert direction_local == expected[2]
        assert local_tip_position == expected[3]
        assert speed_per_unit_omega == expected[4]

    def test_nearest_neighbor_for_off_grid_value(self):
        # -0.7 離 -0.635 比離 -0.9382125 近，應該回傳 -0.635 那一列。
        joints_pan0, _, _, _ = lookup_placement_constants(-0.7)

        expected = next(row for row in _PLACEMENT_LOOKUP_GRID if row[0] == -0.635)
        assert joints_pan0 == expected[1]


class TestComputeFlatBasePositionAndJointTargets:
    def test_wrist_target_is_exactly_reconstructed(self):
        target_wrist_position = (0.606425, -0.635, 0.157)

        base_position, joint_targets = compute_flat_base_position_and_joint_targets(
            target_wrist_position, shot_angle_deg=0.0
        )

        _, direction_local, local_tip_position, _ = _FLAT_PLACEMENT
        reconstructed = _reconstruct_wrist_position(
            base_position, joint_targets, direction_local, local_tip_position
        )
        assert reconstructed == pytest.approx(target_wrist_position, abs=1e-6)

    def test_joint_targets_length_matches_six_dof(self):
        base_position, joint_targets = compute_flat_base_position_and_joint_targets(
            (0.0, -0.635, 0.157), shot_angle_deg=0.0
        )

        assert len(joint_targets) == 6

    @pytest.mark.parametrize("shot_angle_deg", [-45.0, 0.0, 30.0, 90.0])
    def test_required_pan_reconstructs_target_for_any_shot_angle(self, shot_angle_deg):
        """不管瞄準角是多少，`shoulder_pan` 都應該解出讓桿尖剛好落在
        target_wrist_position 的角度——這是 shoulder_pan 扮演 WAM7
        `base_yaw` 角色（吸收任何瞄準角）的核心契約。"""
        target_wrist_position = (0.3, -0.9, 0.157)

        base_position, joint_targets = compute_flat_base_position_and_joint_targets(
            target_wrist_position, shot_angle_deg=shot_angle_deg
        )

        _, direction_local, local_tip_position, _ = _FLAT_PLACEMENT
        reconstructed = _reconstruct_wrist_position(
            base_position, joint_targets, direction_local, local_tip_position
        )
        assert reconstructed == pytest.approx(target_wrist_position, abs=1e-6)


class TestComputeBridgeBasePositionAndJointTargets:
    def test_wrist_target_is_exactly_reconstructed(self):
        target_wrist_position = (0.0, -1.979140646068706, 0.15421704545454545)
        target_direction = (0.0, 0.995659737828671, -0.09306818181818181)

        base_position, joint_targets = compute_bridge_base_position_and_joint_targets(
            target_wrist_position, target_direction, cue_ball_y=-0.635
        )

        _, direction_local, local_tip_position, _ = lookup_placement_constants(-0.635)
        reconstructed = _reconstruct_wrist_position(
            base_position, joint_targets, direction_local, local_tip_position
        )
        assert reconstructed == pytest.approx(target_wrist_position, abs=1e-6)

    def test_uses_looked_up_joints_as_base_of_joint_targets(self):
        target_wrist_position = (0.0, -1.979140646068706, 0.15421704545454545)
        target_direction = (0.0, 0.995659737828671, -0.09306818181818181)

        _, joint_targets = compute_bridge_base_position_and_joint_targets(
            target_wrist_position, target_direction, cue_ball_y=-0.635
        )

        joints_pan0, _, _, _ = lookup_placement_constants(-0.635)
        assert joint_targets[1:] == list(joints_pan0)


class TestComputeTargetElbowVelocity:
    def test_flat_scales_linearly_with_cue_ball_speed(self):
        from core.services.ur3e_placement_calculator import compute_flat_target_elbow_velocity

        v1 = compute_flat_target_elbow_velocity(1.0)
        v2 = compute_flat_target_elbow_velocity(2.0)

        assert v2 == pytest.approx(2 * v1)

    def test_bridge_uses_looked_up_speed_per_unit_omega(self):
        from core.services import swing_trajectory_calculator
        from core.services.ur3e_placement_calculator import compute_bridge_target_elbow_velocity

        cue_ball_speed = 1.995
        velocity = compute_bridge_target_elbow_velocity(cue_ball_speed, cue_ball_y=-0.635)

        _, _, _, speed_per_unit_omega = lookup_placement_constants(-0.635)
        expected = swing_trajectory_calculator.compute_required_tip_speed(cue_ball_speed) / speed_per_unit_omega
        assert velocity == pytest.approx(expected)
