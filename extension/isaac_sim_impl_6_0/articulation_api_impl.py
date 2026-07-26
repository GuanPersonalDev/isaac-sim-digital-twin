import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent

from core.ports.articulation_api import ArticulationAPI


class ArticulationAPIImpl(ArticulationAPI):
    """
    差動 IK（Jacobian-based）版本。RMPflow 需要為每一款機器人準備專屬的
    motion policy 設定檔（Isaac Sim 目前只內建 UR/Franka/Kawasaki 等少數
    型號，沒有 Barrett WAM），跟 RobotArm 抽象介面「換手臂不用大改程式碼」
    的精神衝突——換一款沒有現成設定檔的手臂，RMPflow 初始化就會直接找不到
    對應關節名稱而報錯（實測 Barrett WAM 換上後 KeyError: 'shoulder_pan_joint'）。

    差動 IK 只需要任意 articulation 都通用的 Jacobian，換手臂不需要任何
    額外設定檔。做法比照 scripts/measure_swing_speed.py 已實測驗證過的
    差動 IK 寫法：每個 physics tick 解一次
    q̇ = Jᵀ(JJᵀ+λ²I)⁻¹・twist（damped least squares 偽逆），twist 由
    「目前位姿→目標位姿」的位置/姿態誤差（P controller，供 move_to_pose／
    move_to_home 使用）或固定方向＋速度（供 execute_strike 使用）決定，
    再依關節限速裁切後下達 velocity target。

    末端執行器的世界位姿改用 RigidPrim（tensor API）讀取，不能用
    UsdGeom.XformCache 之類的 raw USD 讀法——關節是用新版 tensor-based
    Articulation 驅動的，它的模擬狀態走 Fabric，不會每個 tick 同步寫回
    classic USD stage 的 xformOp，raw USD 讀到的位置會是沒更新的殘留值
    （實測踩過：讀到 float32 max 等級的垃圾值）。這是跟本次工作階段稍早
    「tensor API/raw USD 混用造成資料不同步」同一類問題（見
    core/ports/rigid_body_api.py 的說明），這裡的修法同樣是讀寫都走同一
    條 tensor API 路徑。
    """

    # P controller 對穩態誤差有天生的殘留量（實測約 2mm），球半徑約 28.575mm，
    # 這個精度對擊球應用綽綽有餘，門檻沒必要比原本 RMPflow 版本的 1mm 更嚴格。
    POSITION_TOLERANCE = 0.005
    ORIENTATION_TOLERANCE = 0.02  # rad，四元數誤差角度
    POSITION_GAIN = 5.0
    ORIENTATION_GAIN = 5.0
    MAX_LINEAR_SPEED = 2.0  # m/s，P controller 位置誤差轉速度指令的上限
    MAX_ANGULAR_SPEED = 3.0  # rad/s
    DLS_LAMBDA = 0.05
    _MOTION_CALLBACK_NAME = "articulation_api_impl_step_motion"
    _HOME_CAPTURE_CALLBACK_NAME = "articulation_api_impl_capture_home"

    def __init__(
        self, robot_prim_path: str, end_effector_prim_path: str
    ) -> None:
        self._robot_prim_path = robot_prim_path
        self._end_effector_prim_path = end_effector_prim_path

        self._articulation: Articulation | None = None
        self._end_effector_rigid_prim: RigidPrim | None = None
        self._dof_limits: np.ndarray | None = None
        self._jac_link_index: int | None = None

        self._default_joint_positions: np.ndarray | None = None
        self._home_position: np.ndarray | None = None
        self._target_position: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        # execute_strike 用固定方向＋速度的 feed-forward twist，
        # 不是 move_to_pose/move_to_home 的位置誤差 P controller。
        self._strike_twist: np.ndarray | None = None
        # move_to_home 用 joint-space 位置控制直接讓 PhysX 關節驅動器插值，
        # 不需要（也不能，起始位置離目標可能很遠）跑 differential IK；
        # 這裡只需監控是否已到位，不必每個 tick 重新解 Jacobian。
        self._is_joint_space_motion = False
        self._tip_local_offset: np.ndarray | None = None
        self._motion_active = False
        self._step_motion_id: int | None = None
        # None 代表「尚未註冊」或「已觸發並清空」，供 cancel_pending_home_capture()
        # 判斷是否還需要取消
        self._capture_callback_id: int | None = None

    def initialize(self) -> None:
        # 在 timeline play 之後呼叫
        self._articulation = Articulation(paths=self._robot_prim_path)
        self._end_effector_rigid_prim = RigidPrim(paths=self._end_effector_prim_path)
        self._default_joint_positions = np.asarray(self._articulation.get_dof_positions())
        self._dof_limits = np.asarray(self._articulation.get_dof_max_velocities())[0]
        self._jac_link_index = self._resolve_end_effector_jacobian_index()

        self._tip_local_offset = self._compute_tip_local_offset()

        self._capture_callback_id = SimulationManager.register_callback(
            self._capture_home_position_once, event=SimulationEvent.PHYSICS_POST_STEP
        )

    def _resolve_end_effector_jacobian_index(self) -> int:
        end_effector_link_name = self._end_effector_prim_path.rsplit("/", 1)[-1]
        link_names = None
        for attr in ("link_names", "body_names"):
            if hasattr(self._articulation, attr):
                link_names = list(getattr(self._articulation, attr))
                break
        if link_names is None:
            raise RuntimeError("Articulation 沒有 link_names/body_names 屬性，無法定位末端執行器")
        link_index = link_names.index(end_effector_link_name)

        jacobians = np.asarray(self._articulation.get_jacobian_matrices().numpy())[0]
        # fixed-base articulation 的 Jacobian 不含 base link，
        # 見 scripts/measure_swing_speed.py 同一段邏輯的說明。
        if jacobians.shape[0] == len(link_names) - 1:
            return link_index - 1
        if jacobians.shape[0] == len(link_names):
            return link_index
        raise RuntimeError(
            f"Jacobian link 數 {jacobians.shape[0]} 與 link 名稱數 {len(link_names)} 對不上，"
            f"無法安全對應 {end_effector_link_name}"
        )

    def _compute_tip_local_offset(self) -> np.ndarray:
        """
        末端執行器 link 若本身就有幾何體（例如 UR5 的 wrist_3_link 那種
        實際的凸緣零件），用 local bounding box 沿最長軸找「離原點最遠的
        那一端」當作工具尖端；若這個 link 純粹是掛載參考點、沒有任何
        幾何體（例如 Barrett WAM 的 wam_wrist_palm_stump_link——球桿是靠
        align_prim_to_target 直接對齊到它的世界座標，不需要額外偏移量），
        USD 對空 bounding box 的慣例回傳值是無效的 min>max 範圍（實測數值
        落在 float32 極值等級），這種情況直接視為「原點本身就是尖端」，
        偏移量為 0，避免把這個無效值當成真的幾何尺寸用。
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._end_effector_prim_path)
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        local_range = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        min_pt = np.array(local_range.GetMin())
        max_pt = np.array(local_range.GetMax())

        if np.any(min_pt > max_pt):
            return np.zeros(3)

        axis_index = int(np.argmax(max_pt - min_pt))
        tip_local = np.zeros(3)
        tip_local[axis_index] = (
            max_pt[axis_index]
            if abs(max_pt[axis_index]) > abs(min_pt[axis_index])
            else min_pt[axis_index]
        )
        return tip_local

    def _capture_home_position_once(self, step_dt, context) -> None:
        self._home_position = np.array(self.get_end_effector_position())
        SimulationManager.deregister_callback(self._capture_callback_id)
        self._capture_callback_id = None

    def move_to_pose(self, position: list[float], orientation: list[float]) -> None:
        self._target_position = np.array(position)
        self._target_orientation = np.array(orientation)
        self._strike_twist = None
        self._is_joint_space_motion = False
        self._articulation.switch_dof_control_mode("velocity")
        self._start_motion()

    def _start_motion(self) -> None:
        if not self._motion_active:
            self._step_motion_id = SimulationManager.register_callback(
                self._step_motion, event=SimulationEvent.PHYSICS_POST_STEP
            )
            self._motion_active = True

    def _step_motion(self, step_dt, context) -> None:
        if not self._is_joint_space_motion:
            twist = (
                self._strike_twist
                if self._strike_twist is not None
                else self._compute_pose_tracking_twist()
            )

            jacobians = np.asarray(self._articulation.get_jacobian_matrices().numpy())[0]
            J = jacobians[self._jac_link_index]
            JJt = J @ J.T + (self.DLS_LAMBDA**2) * np.eye(6)
            qdot = J.T @ np.linalg.solve(JJt, twist)
            qdot = np.clip(qdot, -self._dof_limits, self._dof_limits)

            self._articulation.set_dof_velocity_targets(qdot[None, :])

        if self.is_motion_complete():
            self._stop_motion()

    def _compute_pose_tracking_twist(self) -> np.ndarray:
        current_position = np.array(self.get_end_effector_position())
        position_error = self._target_position - current_position
        linear_velocity = np.clip(
            self.POSITION_GAIN * position_error, -self.MAX_LINEAR_SPEED, self.MAX_LINEAR_SPEED
        )

        current_orientation = self._get_end_effector_world_orientation()
        angular_velocity = self.ORIENTATION_GAIN * self._orientation_error_to_angular_velocity(
            current_orientation, self._target_orientation
        )
        angular_velocity = np.clip(angular_velocity, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

        return np.concatenate([linear_velocity, angular_velocity])

    @staticmethod
    def _orientation_error_to_angular_velocity(
        current_wxyz: np.ndarray, target_wxyz: np.ndarray
    ) -> np.ndarray:
        """
        q_error = q_target * q_current⁻¹，小角度近似下角速度方向 ≈ 2 * q_error.xyz
        （q_error.w 為負時取反，走最短路徑）。
        """
        cw, cx, cy, cz = current_wxyz
        tw, tx, ty, tz = target_wxyz
        current_conj = np.array([cw, -cx, -cy, -cz])

        aw, ax, ay, az = tw, tx, ty, tz
        bw, bx, by, bz = current_conj
        q_error = np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ]
        )
        if q_error[0] < 0:
            q_error = -q_error
        return 2.0 * q_error[1:]

    def _stop_motion(self) -> None:
        if self._motion_active:
            if not self._is_joint_space_motion:
                self._articulation.set_dof_velocity_targets(
                    np.zeros((1, len(self._dof_limits)))
                )
            SimulationManager.deregister_callback(self._step_motion_id)
            self._motion_active = False

    def execute_strike(
        self, direction: list[float], distance: float, speed: float
    ) -> None:
        direction_unit = np.array(direction) / np.linalg.norm(direction)
        current_position = np.array(self.get_end_effector_position())
        self._target_position = current_position + direction_unit * distance
        self._target_orientation = self._get_end_effector_world_orientation()
        self._is_joint_space_motion = False
        self._articulation.switch_dof_control_mode("velocity")

        twist_unit = np.concatenate([direction_unit, np.zeros(3)])
        jacobians = np.asarray(self._articulation.get_jacobian_matrices().numpy())[0]
        J = jacobians[self._jac_link_index]
        JJt = J @ J.T + (self.DLS_LAMBDA**2) * np.eye(6)
        qdot_unit = J.T @ np.linalg.solve(JJt, twist_unit)
        # 依關節限速裁切 speed：若指定速度超出這個姿態下可行的最大值，
        # 退而求其次採用可行的最大值（不會反向、也不會讓某關節超速）。
        max_ratio = float(np.max(np.abs(qdot_unit) / self._dof_limits))
        feasible_speed = speed if max_ratio <= 1e-9 else min(speed, 1.0 / max_ratio)

        self._strike_twist = twist_unit * feasible_speed
        self._start_motion()

    def _get_end_effector_world_orientation(self) -> np.ndarray:
        _, orientations = self._end_effector_rigid_prim.get_world_poses()
        return np.array(orientations[0].list())

    @staticmethod
    def _rotate_vector_by_quat(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """四元數旋轉向量：v' = v + 2w(q_xyz×v) + 2(q_xyz×(q_xyz×v))"""
        w = quat_wxyz[0]
        q_xyz = quat_wxyz[1:]
        t = 2.0 * np.cross(q_xyz, vec)
        return vec + w * t + np.cross(q_xyz, t)

    def move_to_home(self) -> None:
        # joint-space 位置控制，交給 PhysX 關節驅動器自己插值到位，不需要
        # （起始位置離目標可能很遠，也不適合）跑 differential IK。
        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(self._default_joint_positions)
        self._target_position = self._home_position
        self._is_joint_space_motion = True
        self._start_motion()

    def get_end_effector_position(self) -> list[float]:
        if self._tip_local_offset is None:
            self._tip_local_offset = self._compute_tip_local_offset()

        positions, orientations = self._end_effector_rigid_prim.get_world_poses()
        position = np.array(positions[0].list())
        orientation = np.array(orientations[0].list())
        tip_world_point = position + self._rotate_vector_by_quat(orientation, self._tip_local_offset)
        return tip_world_point.tolist()

    def is_motion_complete(self) -> bool:
        if self._target_position is None:
            return True
        current_position = np.array(self.get_end_effector_position())
        error = np.linalg.norm(current_position - self._target_position)
        return bool(error < self.POSITION_TOLERANCE)

    def shutdown(self) -> None:
        pass

    def cancel_pending_home_capture(self) -> None:
        if self._capture_callback_id is not None:
            SimulationManager.deregister_callback(self._capture_callback_id)
            self._capture_callback_id = None
