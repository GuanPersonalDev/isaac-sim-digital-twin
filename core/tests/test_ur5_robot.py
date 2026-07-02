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
            position=(1.5, 0.0, 0.0),
        )

        stage_api.set_prim_translate.assert_called_once_with(
            "/World/BilliardTable/Robot",
            1.5,
            0.0,
            0.0,
        )

    def test_get_prim_path_returns_robot_prim_path(self):
        robot = _ur5_robot_class()(
            base_path="/World/BilliardTable",
            stage_api=MagicMock(),
            position=(1.5, 0.0, 0.0),
        )

        assert robot.get_prim_path() == "/World/BilliardTable/Robot"
