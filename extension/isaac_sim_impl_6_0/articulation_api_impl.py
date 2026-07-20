import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_motion_policy_config,
)

from core.ports.articulation_api import ArticulationAPI


class ArticulationAPIImpl(ArticulationAPI):
    POSITION_TOLERANCE = 0.001
    _MOTION_CALLBACK_NAME = "articulation_api_impl_step_motion"
    _HOME_CAPTURE_CALLBACK_NAME = "articulation_api_impl_capture_home"

    def __init__(
        self, world: World, robot_prim_path: str, end_effector_prim_path: str
    ) -> None:
        self._world = world
        self._robot_prim_path = robot_prim_path
        self._end_effector_prim_path = end_effector_prim_path

        self._articulation: SingleArticulation | None = None
        self._rmpflow: RmpFlow | None = None
        self._articulation_rmpflow: ArticulationMotionPolicy | None = None

        self._default_joint_positions = None
        self._home_position: np.ndarray | None = None
        self._target_position: np.ndarray | None = None
        self._tip_local_offset: np.ndarray | None = None
        self._motion_active = False

    def initialize(self) -> None:
        # 在 timeline play 之後呼叫
        self._articulation = SingleArticulation(self._robot_prim_path)
        self._articulation.initialize()
        self._default_joint_positions = self._articulation.get_joint_positions()

        rmp_config = load_supported_motion_policy_config("UR5", "RMPflow")
        self._rmpflow = RmpFlow(**rmp_config)
        self._articulation_rmpflow = ArticulationMotionPolicy(
            self._articulation, self._rmpflow
        )

        self._tip_local_offset = self._compute_tip_local_offset()
        self._world.add_physics_callback(
            self._HOME_CAPTURE_CALLBACK_NAME, self._capture_home_position_once
        )

    def _compute_tip_local_offset(self) -> np.ndarray:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._end_effector_prim_path)
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        local_range = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        min_pt = np.array(local_range.GetMin())
        max_pt = np.array(local_range.GetMax())
        axis_index = int(np.argmax(max_pt - min_pt))

        tip_local = np.zeros(3)
        tip_local[axis_index] = (
            max_pt[axis_index]
            if abs(max_pt[axis_index]) > abs(min_pt[axis_index])
            else min_pt[axis_index]
        )
        return tip_local

    def _capture_home_position_once(self, step_size: float) -> None:
        self._home_position = np.array(self.get_end_effector_position())
        self._world.remove_physics_callback(self._HOME_CAPTURE_CALLBACK_NAME)

    def move_to_pose(self, position: list[float], orientation: list[float]) -> None:
        target_position = np.array(position)
        target_orientation = np.array(orientation)

        self._rmpflow.set_end_effector_target(target_position, target_orientation)
        self._target_position = target_position
        self._start_motion()

    def _start_motion(self) -> None:
        if not self._motion_active:
            self._world.add_physics_callback(
                self._MOTION_CALLBACK_NAME, self._step_motion
            )
            self._motion_active = True

    def _step_motion(self, step_size: float) -> None:
        self._rmpflow.update_world()
        action = self._articulation_rmpflow.get_next_articulation_action(step_size)
        self._articulation.apply_action(action)
        if self.is_motion_complete():
            self._stop_motion()

    def _stop_motion(self) -> None:
        if self._motion_active:
            self._world.remove_physics_callback(self._MOTION_CALLBACK_NAME)
            self._motion_active = False

    def execute_strike(
        self, direction: list[float], distance: float, speed: float
    ) -> None:
        # TODO: 若要精準控制速度，需要調整為使用差動 IK (Articulation.get_jacobian_matrices())
        current_position = np.array(self.get_end_effector_position())
        direction_vector = np.array(direction)
        end_position = current_position + direction_vector * distance

        current_orientation = self._get_end_effector_world_orientation()
        self.move_to_pose(end_position.tolist(), current_orientation.tolist())

    def _get_end_effector_world_orientation(self) -> np.ndarray:
        world_matrix = self._get_world_matrix()
        quat = world_matrix.GetOrthonormalized().ExtractRotationQuat()
        real = quat.GetReal()
        imaginary = quat.GetImaginary()
        return np.array([real, imaginary[0], imaginary[1], imaginary[2]])

    def _get_world_matrix(self) -> Gf.Matrix4d:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._end_effector_prim_path)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return xform_cache.GetLocalToWorldTransform(prim)

    def move_to_home(self) -> None:
        action = ArticulationAction(joint_positions=self._default_joint_positions)
        self._articulation.apply_action(action)
        self._target_position = self._home_position
        self._start_motion()

    def get_end_effector_position(self) -> list[float]:
        if self._tip_local_offset is None:
            self._tip_local_offset = self._compute_tip_local_offset()

        world_matrix = self._get_world_matrix()
        tip_local_point = Gf.Vec3d(*self._tip_local_offset.tolist())
        tip_world_point = world_matrix.Transform(tip_local_point)
        return [tip_world_point[0], tip_world_point[1], tip_world_point[2]]

    def is_motion_complete(self) -> bool:
        if self._target_position is None:
            return True
        current_position = np.array(self.get_end_effector_position())
        error = np.linalg.norm(current_position - self._target_position)
        return bool(error < self.POSITION_TOLERANCE)

    def shutdown(self) -> None:
        #TODO: remove physics callback
        pass