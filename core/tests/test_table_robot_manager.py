from unittest.mock import MagicMock, call, patch

import pytest

from core.services.asset_utility import CUE_STICK_PATH


def _table_robot_manager_class():
    from core.models.table_robot_manager import TableRobotManager

    return TableRobotManager


@pytest.fixture
def fixed_joint_paths() -> dict[str, str]:
    return {
        "base": "/World/DemoTable",
        "cue_stick": "/World/DemoTable/CueStick",
        "end_effector": "/World/DemoTable/Robot/wrist_3_link",
        "joint": "/World/DemoTable/CueStick/FixedJointToRobot",
    }


@pytest.fixture
def fixed_joint_stage_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def articulation_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def fixed_joint_ur5_robot(fixed_joint_paths: dict[str, str]):
    with patch("core.models.table_robot_manager.UR5Robot") as mock_ur5_robot:
        mock_ur5_robot.get_end_effector_prim_path.return_value = (
            fixed_joint_paths["end_effector"]
        )
        yield mock_ur5_robot


class TestTableRobotManager:
    def test_table_robot_manager_creates_robot_with_offset_position(
        self, articulation_api: MagicMock
    ):
        stage_api = MagicMock()

        with patch("core.models.table_robot_manager.UR5Robot") as mock_ur5_robot:
            _table_robot_manager_class()(
                table_center=(2.0, 3.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=stage_api,
                articulation_api=articulation_api,
            )

            mock_ur5_robot.assert_called_once_with(
                "/World/DemoTable",
                stage_api,
                articulation_api,
                (3.5, 3.0, 0.0),
            )

    def test_table_robot_manager_get_robot_prim_path(self, articulation_api: MagicMock):
        with patch("core.models.table_robot_manager.UR5Robot") as mock_ur5_robot:
            mock_ur5_robot.get_prim_path.return_value = "/World/DemoTable/Robot"

            manager = _table_robot_manager_class()(
                table_center=(0.0, 0.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=MagicMock(),
                articulation_api=articulation_api,
            )

            assert manager.get_robot_prim_path() == "/World/DemoTable/Robot"
            mock_ur5_robot.get_prim_path.assert_called_once_with("/World/DemoTable")

    def test_table_robot_manager_destroy(self, articulation_api: MagicMock):
        with patch("core.models.table_robot_manager.UR5Robot"):
            manager = _table_robot_manager_class()(
                table_center=(0.0, 0.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=MagicMock(),
                articulation_api=articulation_api,
            )

            manager.destroy()

            assert manager._robot is None

    def test_table_robot_manager_creates_cue_stick_reference_prim(
        self, articulation_api: MagicMock
    ):
        stage_api = MagicMock()

        with patch("core.models.table_robot_manager.UR5Robot"):
            _table_robot_manager_class()(
                table_center=(2.0, 3.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=stage_api,
                articulation_api=articulation_api,
            )

            stage_api.create_reference_prim.assert_called_once_with(
                "/World/DemoTable/CueStick", CUE_STICK_PATH
            )

    def test_table_robot_manager_no_longer_sets_cue_stick_translate(
        self, articulation_api: MagicMock
    ):
        stage_api = MagicMock()

        with patch("core.models.table_robot_manager.UR5Robot"):
            _table_robot_manager_class()(
                table_center=(2.0, 3.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=stage_api,
                articulation_api=articulation_api,
            )

            stage_api.set_prim_translate.assert_not_called()

    def test_table_robot_manager_aligns_cue_stick_to_end_effector(
        self,
        fixed_joint_paths: dict[str, str],
        fixed_joint_stage_api: MagicMock,
        fixed_joint_ur5_robot: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
        )

        fixed_joint_stage_api.align_prim_to_target.assert_called_once_with(
            fixed_joint_paths["cue_stick"],
            fixed_joint_paths["end_effector"],
        )

    def test_table_robot_manager_filters_cue_stick_end_effector_collision(
        self,
        fixed_joint_paths: dict[str, str],
        fixed_joint_stage_api: MagicMock,
        fixed_joint_ur5_robot: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
        )

        fixed_joint_stage_api.filter_collision_pair.assert_called_once_with(
            fixed_joint_paths["cue_stick"],
            fixed_joint_paths["end_effector"],
        )

    def test_table_robot_manager_creates_fixed_joint(
        self,
        fixed_joint_paths: dict[str, str],
        fixed_joint_stage_api: MagicMock,
        fixed_joint_ur5_robot: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
        )

        fixed_joint_stage_api.create_fixed_joint.assert_called_once_with(
            fixed_joint_paths["joint"],
            fixed_joint_paths["cue_stick"],
            fixed_joint_paths["end_effector"],
        )

    def test_table_robot_manager_initializes_cue_stick_joint_in_order(
        self,
        fixed_joint_paths: dict[str, str],
        fixed_joint_stage_api: MagicMock,
        fixed_joint_ur5_robot: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
        )

        fixed_joint_stage_api.assert_has_calls(
            [
                call.create_reference_prim(
                    fixed_joint_paths["cue_stick"],
                    CUE_STICK_PATH,
                ),
                call.align_prim_to_target(
                    fixed_joint_paths["cue_stick"],
                    fixed_joint_paths["end_effector"],
                ),
                call.filter_collision_pair(
                    fixed_joint_paths["cue_stick"],
                    fixed_joint_paths["end_effector"],
                ),
                call.create_fixed_joint(
                    fixed_joint_paths["joint"],
                    fixed_joint_paths["cue_stick"],
                    fixed_joint_paths["end_effector"],
                ),
            ],
            any_order=False,
        )

    def test_table_robot_manager_get_cue_stick_prim_path(
        self, articulation_api: MagicMock
    ):
        with patch("core.models.table_robot_manager.UR5Robot"):
            manager = _table_robot_manager_class()(
                table_center=(0.0, 0.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=MagicMock(),
                articulation_api=articulation_api,
            )

            assert manager.get_cue_stick_prim_path() == "/World/DemoTable/CueStick"
