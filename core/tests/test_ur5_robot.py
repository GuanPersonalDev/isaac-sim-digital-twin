from unittest.mock import MagicMock


def _ur5_robot_class():
    from core.models.ur5_robot import UR5Robot

    return UR5Robot


def _ur5_path() -> str:
    from core.services.asset_utility import UR5_PATH

    return UR5_PATH


class TestUR5Robot:
    def test_init_creates_reference_prim_with_ur5_usd_path(self):
        stage_api = MagicMock()

        _ur5_robot_class()(
            base_path="/World/BilliardTable",
            stage_api=stage_api,
            articulation_api=MagicMock(),
            position=(1.5, 0.0, 0.0),
        )

        stage_api.create_reference_prim.assert_called_once_with(
            "/World/BilliardTable/Robot",
            _ur5_path(),
        )

    def test_init_sets_robot_world_position(self):
        stage_api = MagicMock()

        _ur5_robot_class()(
            base_path="/World/BilliardTable",
            stage_api=stage_api,
            articulation_api=MagicMock(),
            position=(1.5, 0.0, 0.0),
        )

        stage_api.set_prim_translate.assert_called_once_with(
            "/World/BilliardTable/Robot",
            1.5,
            0.0,
            0.0,
        )

    def test_get_prim_path_returns_robot_prim_path(self):
        ur5_robot_class = _ur5_robot_class()

        assert (
            ur5_robot_class.get_prim_path("/World/BilliardTable")
            == "/World/BilliardTable/Robot"
        )

    def test_get_end_effector_prim_path_returns_link_path(self):
        ur5_robot_class = _ur5_robot_class()

        assert ur5_robot_class.get_end_effector_prim_path("/World/BilliardTable") == (
            "/World/BilliardTable/Robot/" + ur5_robot_class._END_EFFECTOR_LINK_NAME
        )

    def test_reset_calls_move_to_home(self):
        articulation_api = MagicMock()
        robot = _ur5_robot_class()(
            base_path="/World/BilliardTable",
            stage_api=MagicMock(),
            articulation_api=articulation_api,
            position=(1.5, 0.0, 0.0),
        )

        robot.reset()

        articulation_api.move_to_home.assert_called_once_with()

    def test_is_reset_complete_returns_articulation_is_motion_complete(self):
        articulation_api = MagicMock()
        articulation_api.is_motion_complete.return_value = True
        robot = _ur5_robot_class()(
            base_path="/World/BilliardTable",
            stage_api=MagicMock(),
            articulation_api=articulation_api,
            position=(1.5, 0.0, 0.0),
        )

        assert robot.is_reset_complete() is True
        articulation_api.is_motion_complete.assert_called_once_with()
