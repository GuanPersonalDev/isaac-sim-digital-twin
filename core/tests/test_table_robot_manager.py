from unittest.mock import MagicMock, call

import pytest

from core.models.ur10e_robot import UR10eRobot
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
def robot_arm_class() -> MagicMock:
    return MagicMock()


@pytest.fixture
def fixed_joint_robot_arm_class(robot_arm_class: MagicMock, fixed_joint_paths: dict[str, str]) -> MagicMock:
    robot_arm_class.get_end_effector_prim_path.return_value = fixed_joint_paths["end_effector"]
    return robot_arm_class


class TestTableRobotManager:
    def test_table_robot_manager_creates_robot_with_offset_position(
        self, articulation_api: MagicMock, robot_arm_class: MagicMock
    ):
        stage_api = MagicMock()

        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path="/World/DemoTable",
            stage_api=stage_api,
            articulation_api=articulation_api,
            robot_arm_class=robot_arm_class,
        )

        robot_arm_class.assert_called_once_with(
            "/World/DemoTable",
            stage_api,
            articulation_api,
            (3.5, 3.0, 0.0),
        )

    def test_table_robot_manager_get_robot_prim_path(
        self, articulation_api: MagicMock, robot_arm_class: MagicMock
    ):
        robot_arm_class.get_prim_path.return_value = "/World/DemoTable/Robot"

        manager = _table_robot_manager_class()(
            table_center=(0.0, 0.0, 0.0),
            base_path="/World/DemoTable",
            stage_api=MagicMock(),
            articulation_api=articulation_api,
            robot_arm_class=robot_arm_class,
        )

        assert manager.get_robot_prim_path() == "/World/DemoTable/Robot"
        robot_arm_class.get_prim_path.assert_called_once_with("/World/DemoTable")

    def test_table_robot_manager_destroy(
        self, articulation_api: MagicMock, robot_arm_class: MagicMock
    ):
        manager = _table_robot_manager_class()(
            table_center=(0.0, 0.0, 0.0),
            base_path="/World/DemoTable",
            stage_api=MagicMock(),
            articulation_api=articulation_api,
            robot_arm_class=robot_arm_class,
        )

        manager.destroy()

        assert manager._robot is None

    def test_table_robot_manager_destroy_removes_robot_and_cue_stick_prims(
        self, articulation_api: MagicMock, robot_arm_class: MagicMock
    ):
        stage_api = MagicMock()
        robot_arm_class.get_prim_path.return_value = "/World/DemoTable/Robot"

        manager = _table_robot_manager_class()(
            table_center=(0.0, 0.0, 0.0),
            base_path="/World/DemoTable",
            stage_api=stage_api,
            articulation_api=articulation_api,
            robot_arm_class=robot_arm_class,
        )
        stage_api.reset_mock()

        manager.destroy()

        stage_api.remove_prim.assert_any_call("/World/DemoTable/Robot")
        stage_api.remove_prim.assert_any_call("/World/DemoTable/CueStick")

    def test_table_robot_manager_destroy_does_not_remove_shared_base_path(
        self, articulation_api: MagicMock, robot_arm_class: MagicMock
    ):
        """
        base_path（如 /World/Table_Demo）跟 BilliardTable 共用，只有
        BilliardTable.destroy() 有資格移除它；TableRobotManager 只能移除
        自己的 Robot/CueStick 子路徑，否則會連桌子跟球一起誤刪。
        """
        stage_api = MagicMock()
        robot_arm_class.get_prim_path.return_value = "/World/DemoTable/Robot"

        manager = _table_robot_manager_class()(
            table_center=(0.0, 0.0, 0.0),
            base_path="/World/DemoTable",
            stage_api=stage_api,
            articulation_api=articulation_api,
            robot_arm_class=robot_arm_class,
        )
        stage_api.reset_mock()

        manager.destroy()

        removed_paths = [c.args[0] for c in stage_api.remove_prim.call_args_list]
        assert "/World/DemoTable" not in removed_paths

    def test_table_robot_manager_creates_cue_stick_reference_prim(
        self, articulation_api: MagicMock, robot_arm_class: MagicMock
    ):
        stage_api = MagicMock()

        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path="/World/DemoTable",
            stage_api=stage_api,
            articulation_api=articulation_api,
            robot_arm_class=robot_arm_class,
        )

        stage_api.create_reference_prim.assert_called_once_with(
            "/World/DemoTable/CueStick", CUE_STICK_PATH
        )

    def test_table_robot_manager_no_longer_sets_cue_stick_translate(
        self, articulation_api: MagicMock, robot_arm_class: MagicMock
    ):
        stage_api = MagicMock()

        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path="/World/DemoTable",
            stage_api=stage_api,
            articulation_api=articulation_api,
            robot_arm_class=robot_arm_class,
        )

        stage_api.set_prim_translate.assert_not_called()

    def test_table_robot_manager_aligns_cue_stick_to_end_effector(
        self,
        fixed_joint_paths: dict[str, str],
        fixed_joint_stage_api: MagicMock,
        fixed_joint_robot_arm_class: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
            robot_arm_class=fixed_joint_robot_arm_class,
        )

        fixed_joint_stage_api.align_prim_to_target.assert_called_once_with(
            fixed_joint_paths["cue_stick"],
            fixed_joint_paths["end_effector"],
        )

    def test_table_robot_manager_filters_cue_stick_end_effector_collision(
        self,
        fixed_joint_paths: dict[str, str],
        fixed_joint_stage_api: MagicMock,
        fixed_joint_robot_arm_class: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
            robot_arm_class=fixed_joint_robot_arm_class,
        )

        fixed_joint_stage_api.filter_collision_pair.assert_called_once_with(
            fixed_joint_paths["cue_stick"],
            fixed_joint_paths["end_effector"],
        )

    def test_table_robot_manager_creates_fixed_joint(
        self,
        fixed_joint_paths: dict[str, str],
        fixed_joint_stage_api: MagicMock,
        fixed_joint_robot_arm_class: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
            robot_arm_class=fixed_joint_robot_arm_class,
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
        fixed_joint_robot_arm_class: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
            robot_arm_class=fixed_joint_robot_arm_class,
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
        self, articulation_api: MagicMock, robot_arm_class: MagicMock
    ):
        manager = _table_robot_manager_class()(
            table_center=(0.0, 0.0, 0.0),
            base_path="/World/DemoTable",
            stage_api=MagicMock(),
            articulation_api=articulation_api,
            robot_arm_class=robot_arm_class,
        )

        assert manager.get_cue_stick_prim_path() == "/World/DemoTable/CueStick"


class TestTableRobotManagerUr10ePrismaticJoint:
    """UR10e 靠末端的線性滑軌關節（PrismaticJoint）出力，跟其餘手臂共用的
    `create_fixed_joint()` 路徑不同——見 UR10e 重新設計計畫決策 2/3、
    `TableRobotManager` 類別 docstring。"""

    def test_creates_prismatic_joint_not_fixed_joint(
        self, fixed_joint_paths: dict[str, str], fixed_joint_stage_api: MagicMock, articulation_api: MagicMock
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
            robot_arm_class=UR10eRobot,
        )

        fixed_joint_stage_api.create_fixed_joint.assert_not_called()
        fixed_joint_stage_api.create_prismatic_joint.assert_called_once_with(
            fixed_joint_paths["base"] + "/CueStick/CueSlideJoint",
            fixed_joint_paths["cue_stick"],
            fixed_joint_paths["base"] + "/Robot/wrist_3_link",
            axis="Y",
            drive_stiffness=100000.0,
            drive_damping=10000.0,
            drive_max_force=1000000.0,
        )

    def test_other_robot_arm_classes_still_create_fixed_joint(
        self,
        fixed_joint_paths: dict[str, str],
        fixed_joint_stage_api: MagicMock,
        fixed_joint_robot_arm_class: MagicMock,
        articulation_api: MagicMock,
    ):
        _table_robot_manager_class()(
            table_center=(2.0, 3.0, 0.0),
            base_path=fixed_joint_paths["base"],
            stage_api=fixed_joint_stage_api,
            articulation_api=articulation_api,
            robot_arm_class=fixed_joint_robot_arm_class,
        )

        fixed_joint_stage_api.create_prismatic_joint.assert_not_called()
        fixed_joint_stage_api.create_fixed_joint.assert_called_once()
