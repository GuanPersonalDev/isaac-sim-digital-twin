from unittest.mock import MagicMock, patch

from core.services.asset_utility import CUE_STICK_PATH


def _table_robot_manager_class():
    from core.models.table_robot_manager import TableRobotManager

    return TableRobotManager


class TestTableRobotManager:
    def test_table_robot_manager_creates_robot_with_offset_position(self):
        stage_api = MagicMock()

        with patch("core.models.table_robot_manager.UR5Robot") as mock_ur5_robot:
            _table_robot_manager_class()(
                table_center=(2.0, 3.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=stage_api,
            )

            mock_ur5_robot.assert_called_once_with(
                "/World/DemoTable",
                stage_api,
                (3.5, 3.0, 0.0),
            )

    def test_table_robot_manager_get_robot_prim_path(self):
        with patch("core.models.table_robot_manager.UR5Robot") as mock_ur5_robot:
            mock_ur5_robot.return_value.get_prim_path.return_value = (
                "/World/DemoTable/Robot"
            )

            manager = _table_robot_manager_class()(
                table_center=(0.0, 0.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=MagicMock(),
            )

            assert manager.get_robot_prim_path() == "/World/DemoTable/Robot"
            mock_ur5_robot.return_value.get_prim_path.assert_called_once_with()

    def test_table_robot_manager_destroy(self):
        with patch("core.models.table_robot_manager.UR5Robot") as mock_ur5_robot:
            manager = _table_robot_manager_class()(
                table_center=(0.0, 0.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=MagicMock(),
            )

            manager.destroy()

            assert manager._robot is None

    def test_table_robot_manager_creates_cue_stick_reference_prim(self):
        stage_api = MagicMock()

        with patch("core.models.table_robot_manager.UR5Robot"):
            _table_robot_manager_class()(
                table_center=(2.0, 3.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=stage_api,
            )

            stage_api.create_reference_prim.assert_called_once_with(
                "/World/DemoTable/CueStick", CUE_STICK_PATH
            )

    def test_table_robot_manager_sets_cue_stick_translate(self):
        stage_api = MagicMock()

        with patch("core.models.table_robot_manager.UR5Robot"):
            _table_robot_manager_class()(
                table_center=(2.0, 3.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=stage_api,
            )

            stage_api.set_prim_translate.assert_called_once_with(
                "/World/DemoTable/CueStick", 3.5, 3.0, 0.0
            )

    def test_table_robot_manager_get_cue_stick_prim_path(self):
        with patch("core.models.table_robot_manager.UR5Robot"):
            manager = _table_robot_manager_class()(
                table_center=(0.0, 0.0, 0.0),
                base_path="/World/DemoTable",
                stage_api=MagicMock(),
            )

            assert manager.get_cue_stick_prim_path() == "/World/DemoTable/CueStick"
