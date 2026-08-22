import logging

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent

from core.ports.articulation_api import ArticulationAPI
from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP

logger = logging.getLogger(__name__)


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
    「目前位姿→目標位姿」的位置/姿態誤差（P controller）加上
    move_to_pose 的 feed-forward 速度決定，再依關節限速裁切後下達
    velocity target。

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
    POSITION_GAIN = 5.0
    ORIENTATION_GAIN = 5.0
    MAX_LINEAR_SPEED = 2.0  # m/s，P controller 位置誤差轉速度指令的上限
    MAX_ANGULAR_SPEED = 3.0  # rad/s
    DLS_LAMBDA = 0.05

    def __init__(
        self, robot_prim_path: str, end_effector_prim_path: str
    ) -> None:
        self._robot_prim_path = robot_prim_path
        self._end_effector_prim_path = end_effector_prim_path

        self._articulation: Articulation | None = None
        self._end_effector_rigid_prim: RigidPrim | None = None
        self._cue_stick_rigid_prim: RigidPrim | None = None
        self._dof_limits = np.empty(0, dtype=float)
        self._jac_link_index: int | None = None

        self._default_joint_positions: np.ndarray | None = None
        self._home_position: np.ndarray | None = None
        self._target_position: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        self._feedforward_twist = np.zeros(6)
        # move_to_home / move_to_joint_position 用 joint-space 位置控制
        # 直接讓 PhysX 關節驅動器插值，不需要（也不能，起始位置離目標
        # 可能很遠）跑 differential IK；這裡只需監控是否已到位，不必每個
        # tick 重新解 Jacobian。
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
        self._dof_limits = self._load_dof_max_velocities()
        self._jac_link_index = self._resolve_end_effector_jacobian_index()

        self._tip_local_offset = self._compute_tip_local_offset()

        cue_stick_prim_path = self._resolve_cue_stick_prim_path()
        self._cue_stick_rigid_prim = (
            RigidPrim(paths=cue_stick_prim_path) if cue_stick_prim_path is not None else None
        )

        self._capture_callback_id = SimulationManager.register_callback(
            self._capture_home_position_once, event=SimulationEvent.PHYSICS_POST_STEP
        )

    def _resolve_cue_stick_prim_path(self) -> str | None:
        """
        CueStick 跟 Robot 是同一個 base_path 底下的手足 prim（見
        TableRobotManager：`{base_path}/Robot`、`{base_path}/CueStick`），
        不是每個呼叫端都有掛球桿（例如 scripts/probe_base_reachability.py
        只建 Robot），所以要先確認 prim 真的存在才建立 RigidPrim，避免
        debug log 需求把沒掛球桿的呼叫端弄壞。
        """
        if not self._robot_prim_path.endswith("/Robot"):
            return None
        base_path = self._robot_prim_path[: -len("/Robot")]
        cue_stick_prim_path = base_path + "/CueStick"
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(cue_stick_prim_path).IsValid():
            return None
        return cue_stick_prim_path

    def _get_robot_prim_world_position(self) -> list[float]:
        """
        Robot prim 本身不是 physics 模擬的剛體，是靠 stage_api.set_prim_translate()
        設一次性的 classic USD xformOp（見 BarrettWamRobot.reposition()），不會
        每個 tick 被 Fabric 覆寫，用 raw USD 讀不會有本檔案 class docstring 提到
        的「tensor API/raw USD 不同步」問題，這裡讀的目的是回讀確認，不是讀
        physics 模擬狀態。
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._robot_prim_path)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        translation = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        return [translation[0], translation[1], translation[2]]

    def _get_cue_stick_world_pose(self) -> tuple[list[float], list[float]] | None:
        if self._cue_stick_rigid_prim is None:
            return None
        positions, orientations = self._cue_stick_rigid_prim.get_world_poses()
        return positions[0].list(), orientations[0].list()

    def _load_dof_max_velocities(self) -> np.ndarray:
        if self._articulation is None:
            raise RuntimeError("Articulation 尚未初始化，無法讀取關節限速")
        max_velocities = self._articulation.get_dof_max_velocities()
        if hasattr(max_velocities, "numpy"):
            max_velocities = max_velocities.numpy()
        limits = np.asarray(max_velocities, dtype=float)
        if limits.ndim == 2:
            limits = limits[0]
        if limits.ndim != 1 or limits.size == 0:
            raise RuntimeError(
                f"get_dof_max_velocities() 回傳 shape {np.asarray(max_velocities).shape}，"
                "無法當成單一 articulation 的關節限速"
            )
        return limits

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
        # _default_joint_positions 一起搬到這裡（跟 _home_position 同一個
        # PHYSICS_POST_STEP callback）擷取：get_dof_positions() 讀的是動態
        # 模擬狀態，在 initialize() 裡同步呼叫（physics 可能一步都還沒跑）
        # 拿到的值不可靠，跟 scripts/probe_palm_yaw_correction.py 除錯時
        # 踩到「剛建構的 Articulation 沒等 physics 穩定就讀，拿到垃圾值」
        # 是同一類問題。兩者原本不同步擷取，move_to_home() 會把關節開回一個
        # 不可靠的 _default_joint_positions，永遠碰不到用正確方式量到的
        # _home_position，是 RESET 狀態卡死、is_motion_complete() 恆為 False
        # 的根因。
        self._default_joint_positions = np.asarray(self._articulation.get_dof_positions())
        self._home_position = np.array(self.get_end_effector_position())
        SimulationManager.deregister_callback(self._capture_callback_id)
        self._capture_callback_id = None

    def move_to_pose(self, position: list[float], orientation: list[float], linear_velocity: list[float] = [0.0, 0.0, 0.0], angular_velocity: list[float] = [0.0, 0.0, 0.0]) -> None:
        self._target_position = np.array(position)
        self._target_orientation = np.array(orientation)
        self._feedforward_twist = np.concatenate([np.array(linear_velocity), np.array(angular_velocity)])
        self._is_joint_space_motion = False
        self._articulation.switch_dof_control_mode("velocity")
        self._start_motion()

    def _start_motion(self) -> None:
        if not self._motion_active:
            self._step_motion_id = SimulationManager.register_callback(
                self._step_motion, event=SimulationEvent.PHYSICS_POST_STEP
            )
            self._motion_active = True

    def _start_joint_space_motion(
        self,
        joint_positions: np.ndarray,
        target_end_effector_position: np.ndarray,
    ) -> None:
        # joint-space 位置控制，交給 PhysX 關節驅動器自己插值到位，不需要
        # （起始位置離目標可能很遠，也不適合）跑 differential IK。
        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(joint_positions)
        self._target_position = np.asarray(target_end_effector_position)
        self._is_joint_space_motion = True
        self._start_motion()

    def _step_motion(self, step_dt, context) -> None:
        if not self._is_joint_space_motion:
            twist = self._compute_pose_tracking_twist() + self._feedforward_twist

            jacobians = np.asarray(self._articulation.get_jacobian_matrices().numpy())[0]
            J = jacobians[self._jac_link_index]
            JJt = J @ J.T + (self.DLS_LAMBDA**2) * np.eye(6)
            qdot = J.T @ np.linalg.solve(JJt, twist)
            qdot = np.clip(qdot, -self._dof_limits, self._dof_limits)

            self._articulation.set_dof_velocity_targets(qdot[None, :])

        if self.is_motion_complete():
            actual_joint_positions = np.asarray(self._articulation.get_dof_positions())[0]
            end_effector_position = np.array(self.get_end_effector_position())
            cue_tip_offset = self._rotate_vector_by_quat(
                self._get_end_effector_world_orientation(),
                np.array([0.0, CUE_STICK_GRIP_TO_TIP, 0.0]),
            )
            cue_tip_position = end_effector_position + cue_tip_offset
            robot_prim_position = self._get_robot_prim_world_position()
            cue_stick_actual_pose = self._get_cue_stick_world_pose()
            logger.info(
                "[MOTION_COMPLETE] joint_positions(actual)=%s robot_prim_position=%s "
                "end_effector_position=%s cue_tip_position(computed)=%s "
                "cue_tip_offset_from_end_effector=%s "
                "cue_stick_actual_position=%s cue_stick_actual_orientation=%s",
                actual_joint_positions.tolist(), robot_prim_position,
                end_effector_position.tolist(), cue_tip_position.tolist(),
                cue_tip_offset.tolist(),
                cue_stick_actual_pose[0] if cue_stick_actual_pose else None,
                cue_stick_actual_pose[1] if cue_stick_actual_pose else None,
            )
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
        self._start_joint_space_motion(
            self._default_joint_positions, self._home_position
        )

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

    def move_to_joint_position(self, joint_positions: list[float], target_end_effector_position: list[float]) -> None:
        self._start_joint_space_motion(
            np.array([joint_positions]), np.array(target_end_effector_position)
        )