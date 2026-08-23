from core.models.pose_waypoint import PoseWaypoint


class TestPoseWaypoint:
    def test_fields_are_readable(self):
        waypoint = PoseWaypoint(
            position=[1.0, 2.0, 3.0],
            orientation=[1.0, 0.0, 0.0, 0.0],
            linear_velocity=[0.1, 0.2, 0.3],
            angular_velocity=[0.4, 0.5, 0.6],
        )

        assert waypoint.position == [1.0, 2.0, 3.0]
        assert waypoint.orientation == [1.0, 0.0, 0.0, 0.0]
        assert waypoint.linear_velocity == [0.1, 0.2, 0.3]
        assert waypoint.angular_velocity == [0.4, 0.5, 0.6]

    def test_velocity_defaults_to_zero(self):
        waypoint = PoseWaypoint(position=[0.0, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0])

        assert waypoint.linear_velocity == [0.0, 0.0, 0.0]
        assert waypoint.angular_velocity == [0.0, 0.0, 0.0]

    def test_default_velocity_instances_are_independent(self):
        # default_factory 必須是每個 instance 各自一份 list，不能共用同一個
        # 可變物件（dataclass 的經典陷阱）。
        a = PoseWaypoint(position=[0.0, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0])
        b = PoseWaypoint(position=[0.0, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0])

        a.linear_velocity.append(99.0)

        assert b.linear_velocity == [0.0, 0.0, 0.0]
