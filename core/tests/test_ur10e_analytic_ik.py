import numpy as np
import pytest

from core.services import ur10e_analytic_ik as ik


class TestForwardInverseRoundTrip:
    @pytest.mark.parametrize(
        "joints",
        [
            (0.3931, 1.2479, 0.8661, -0.8633, -0.6278, 1.1736),
            (-1.5543, 1.0092, 0.9333, -0.1007, -0.6188, -0.6961),
            (1.2480, -0.9530, 2.1115, 0.3171, 1.7037, -0.3266),
        ],
    )
    def test_inverse_kinematics_recovers_original_joint_angles(self, joints):
        """對一組已知關節角算正向運動學，再解逆向運動學，回傳的 8 組候選
        解裡至少要有一組（模 2π）等於原始關節角——這是解析解正確性最直接
        的驗證，不依賴任何 Isaac Sim 元件（見
        scripts/verify_ur10e_analytic_ik.py 另外驗證跟 RMPflow 的座標系
        對應關係，這裡只驗證演算法本身）。"""
        position, rotation = ik.forward_kinematics(joints)

        solutions = ik.inverse_kinematics(position, rotation)

        assert len(solutions) > 0
        best_error = min(
            float(np.max(np.abs(np.mod(np.asarray(solution) - np.asarray(joints) + np.pi, 2 * np.pi) - np.pi)))
            for solution in solutions
        )
        assert best_error < 1e-6

    def test_every_returned_solution_reproduces_the_same_pose(self):
        """8 組解對應同一個末端位姿，任一組解重新算正向運動學都應該回到
        同一個 position/rotation（不是隨便選一組能用就好）。"""
        joints = (0.5, -1.0, 1.3, 0.2, 0.9, -0.4)
        position, rotation = ik.forward_kinematics(joints)

        solutions = ik.inverse_kinematics(position, rotation)

        assert len(solutions) >= 2
        for solution in solutions:
            solved_position, solved_rotation = ik.forward_kinematics(solution)
            assert solved_position == pytest.approx(position, abs=1e-8)
            assert solved_rotation == pytest.approx(rotation, abs=1e-8)


class TestHomeJointPositionsSitsAtWristSingularity:
    def test_home_wrist_2_is_exactly_zero_a_known_singularity(self):
        """`Ur10eRmpflowController._HOME_JOINT_POSITIONS`（[-0.0, -1.2, 1.1,
        0.0, 0.0, 0.0]）的 wrist_2=0，正好是 UR 系列手臂的手腕奇異點
        （wrist_1/wrist_3 兩軸共平面，theta6 數學上無定義）。這個測試把
        這個發現釘成迴歸測試——in-degenerate 姿態round-trip 本來就無法
        唯一解回原始關節角（theta4/theta6 只有和有意義，個別值不唯一），
        跟 test_inverse_kinematics_recovers_original_joint_angles() 刻意
        避開這個案例是同一個原因。"""
        home_joints = (-0.0, -1.2, 1.1, 0.0, 0.0, 0.0)

        assert ik.wrist_singularity_margin(home_joints) == pytest.approx(0.0, abs=1e-9)


class TestInverseKinematicsUnreachable:
    def test_target_too_far_away_returns_no_solutions(self):
        """腕心投影半徑比 d4 短會讓 theta1 無實數解——用一個明顯超出手臂
        可達範圍的目標（沿手臂完全打直方向再加十公尺）驗證回傳空清單，
        不是拋例外或回傳 NaN 混在結果裡。"""
        position = np.array([100.0, 100.0, 100.0])
        rotation = np.eye(3)

        solutions = ik.inverse_kinematics(position, rotation)

        assert solutions == []


class TestWristSingularityMargin:
    def test_zero_at_exact_wrist_2_singularity(self):
        joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert ik.wrist_singularity_margin(joints) == pytest.approx(0.0, abs=1e-9)

    def test_zero_at_wrist_2_equals_pi(self):
        joints = (0.0, 0.0, 0.0, 0.0, np.pi, 0.0)
        assert ik.wrist_singularity_margin(joints) == pytest.approx(0.0, abs=1e-9)

    def test_maximal_at_wrist_2_equals_half_pi(self):
        joints = (0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0)
        assert ik.wrist_singularity_margin(joints) == pytest.approx(1.0, abs=1e-9)

    def test_only_depends_on_wrist_2_component(self):
        margin_a = ik.wrist_singularity_margin((1.0, 2.0, 3.0, 4.0, 0.7, 5.0))
        margin_b = ik.wrist_singularity_margin((-9.0, 8.0, -7.0, 6.0, 0.7, -5.0))
        assert margin_a == pytest.approx(margin_b)


class TestIsaacFrameConversion:
    def test_isaac_to_dh_and_back_is_identity(self):
        """R_offset（見模組說明）是繞 Z 軸 180 度旋轉，自身互逆——來回轉
        兩次應該還原成原始值。"""
        position = np.array([0.3, -0.7, 0.5])
        rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        dh_position, dh_rotation = ik.isaac_to_dh_frame(position, rotation)
        roundtrip_position, roundtrip_rotation = ik.dh_to_isaac_frame(dh_position, dh_rotation)

        assert roundtrip_position == pytest.approx(position)
        assert roundtrip_rotation == pytest.approx(rotation)

    def test_conversion_negates_x_and_y_keeps_z(self):
        """見 scripts/verify_ur10e_analytic_ik.py 對 RMPflow 的批次數值
        驗證結果：R_offset=diag(-1,-1,1)。"""
        position = np.array([1.0, 2.0, 3.0])
        rotation = np.eye(3)

        converted_position, _ = ik.dh_to_isaac_frame(position, rotation)

        assert converted_position == pytest.approx([-1.0, -2.0, 3.0])
