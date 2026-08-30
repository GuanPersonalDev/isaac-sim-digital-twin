import logging
import os

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom
from scipy.optimize import linprog

from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent

from core.ports.articulation_api import ArticulationAPI
from core.models.pose_waypoint import PoseWaypoint
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
    # move_to_pose()／move_through_poses() 過去零正式呼叫端，完成判定一直
    # 只看位置；這次要正式上線（高架橋 aim、揮桿），CUE_STICK_GRIP_TO_TIP=
    # 1.35m 的槓桿臂會把殘留朝向誤差放大好幾公分，需要一併檢查朝向（見
    # _is_current_target_converged()）。joint-space 動作不受影響，繼續只看
    # 位置。
    ORIENTATION_TOLERANCE = 0.02  # rad，四元數誤差角度
    # joint-space 動作過去的收斂判定只看末端 Cartesian 位置（POSITION_
    # TOLERANCE），只要末端「路過」目標位置就算完成，不保證 7 個關節個別都
    # 已經穩定在指定角度——多數 joint-space 呼叫端（move_to_home()／單獨的
    # move_to_joint_position()）用的是同一組固定終點，末端位置收斂跟關節
    # 收斂幾乎同時發生，這個落差長期沒被踩到。直到高架橋 Phase 0
    # （preceding_joint_targets）把「joint-space 剛收斂就立刻接著跑差動 IK」
    # 這個時序第一次真正端到端跑起來，才發現：末端位置提早判定收斂時，
    # wrist_pitch/palm_yaw 這類末端關節可能還在半路上，這個「尚未真正穩定」
    # 的起始姿態會讓後續差動 IK 走上不同路徑，足以導致某些案例改撞
    # shoulder_pitch 硬限位（同一組 safe_joint_targets，只因為交接時機不同
    # 就有完全不同的下游結果）。見
    # docs/issue-180-reachability-analysis.md 第十三節。
    JOINT_POSITION_TOLERANCE = 0.01  # rad，joint-space 動作額外要求全部關節都收斂
    POSITION_GAIN = 5.0
    ORIENTATION_GAIN = 5.0
    MAX_LINEAR_SPEED = 2.0  # m/s，P controller 位置誤差轉速度指令的上限
    MAX_ANGULAR_SPEED = 3.0  # rad/s
    DLS_LAMBDA = 0.05
    # 單一動作（含 move_through_poses() 裡的單一 waypoint 階段）超過這個步數
    # 仍未收斂，強制視為完成並標記 did_last_motion_timeout()——避免像
    # docs/issue-flat-case-residual-error.md 記錄的那兩個已知殘留誤差案例
    # 讓狀態機靜默卡死在 AIMING（永遠收斂不了、也永遠不會報錯）。
    MOTION_TIMEOUT_STEPS = 1000

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
        # joint-space 動作的目標關節角度（見 JOINT_POSITION_TOLERANCE），
        # 只在 _is_joint_space_motion=True 時有意義。
        self._target_joint_positions: np.ndarray | None = None
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
        # move_through_poses() 的 waypoint 佇列狀態：_pending_waypoints 是
        # 目前這次動作要依序播放的 Cartesian pose 目標；
        # _awaiting_waypoints_after_joint_motion 代表目前正在跑
        # preceding_joint_targets 的 joint-space 段，收斂後才切到
        # _pending_waypoints[0]；move_to_joint_position()/move_to_home() 這種
        # 非序列呼叫端會把 _pending_waypoints 清空，_step_motion() 的完成判定
        # 因此直接落到「最後一個 waypoint」分支，行為跟改動前一致。
        self._pending_waypoints: list[PoseWaypoint] = []
        self._waypoint_index: int = 0
        self._awaiting_waypoints_after_joint_motion: bool = False
        self._motion_step_count: int = 0
        self._did_last_motion_timeout: bool = False
        # None 代表「尚未註冊」或「已觸發並清空」，供 cancel_pending_home_capture()
        # 判斷是否還需要取消
        self._capture_callback_id: int | None = None
        # move_to_home() 在 home 姿態擷取完成前就被呼叫（Timeline PLAY 當下的
        # 完整 reset 會走到這條路徑），記下來等擷取完成後補做，見 move_to_home()
        self._pending_move_to_home: bool = False

        # move_swing() 的狀態：見該方法與 _step_swing_motion() 的說明
        # （docs/issue-180-reachability-analysis.md 第十六節）。
        # _awaiting_swing_after_backswing：正在跑後擺（一般 pose-tracking，
        # 姿態鎖死）子階段，收斂後才切到揮桿速度最優控制。
        self._awaiting_swing_after_backswing: bool = False
        self._is_swing_motion: bool = False
        self._swing_complete: bool = False
        self._swing_start: np.ndarray | None = None
        self._swing_direction: np.ndarray | None = None
        self._swing_total_distance: float = 0.0
        self._swing_end_position: np.ndarray | None = None
        self._swing_orientation: np.ndarray | None = None
        self._swing_orientation_gain: float = 1.0
        self._swing_max_angular_speed: float = 0.5

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

        if self._pending_move_to_home:
            self._pending_move_to_home = False
            self.move_to_home()

    def move_to_pose(self, position: list[float], orientation: list[float], linear_velocity: list[float] = [0.0, 0.0, 0.0], angular_velocity: list[float] = [0.0, 0.0, 0.0]) -> None:
        self.move_through_poses(
            [PoseWaypoint(position=position, orientation=orientation, linear_velocity=linear_velocity, angular_velocity=angular_velocity)]
        )

    def move_through_poses(
        self,
        waypoints: list[PoseWaypoint],
        preceding_joint_targets: tuple[list[float], list[float]] | None = None,
    ) -> None:
        """依序移動末端通過一串 Cartesian pose 目標，內部自我驅動、自我轉換
        階段，呼叫端只需要呼叫一次。只有走到最後一個 waypoint 才視為
        「動作完成」，is_motion_complete() 在整段序列播放期間持續回傳
        False，語意跟 move_to_pose() 一致。

        做法：waypoints 不可為空（raise ValueError）。self._pending_waypoints
        = list(waypoints)，self._waypoint_index = 0。

        preceding_joint_targets 不為 None：格式為
        (joint_positions, target_end_effector_position)，先呼叫
        self._start_joint_space_motion(...) 收斂到這組安全姿態（避開差動
        IK 在奇異點附近的失穩問題），self._awaiting_waypoints_after_joint_motion
        = True，收斂後由 _step_motion() 接著播放 self._pending_waypoints[0]。

        preceding_joint_targets 為 None：直接呼叫
        self._activate_pose_target(*waypoints[0] 的四個欄位)。
        """
        if waypoints is None or len(waypoints) == 0:
            raise ValueError("waypoints 不可為空")

        self._pending_waypoints = list(waypoints)
        self._waypoint_index = 0
        if preceding_joint_targets is not None:
            if len(preceding_joint_targets) != 2:
                raise ValueError("preceding_joint_targets 格式錯誤")
            if len(preceding_joint_targets[0]) != len(self._dof_limits):
                raise ValueError("preceding_joint_targets[0] 格式錯誤")
            if len(preceding_joint_targets[1]) != 3:
                raise ValueError("preceding_joint_targets[1] 格式錯誤")
            self._start_joint_space_motion(np.array([preceding_joint_targets[0]]), np.asarray(preceding_joint_targets[1]))
            self._awaiting_waypoints_after_joint_motion = True
        else:
            self._activate_pose_target(waypoints[0].position, waypoints[0].orientation, waypoints[0].linear_velocity, waypoints[0].angular_velocity)

    def _activate_pose_target(
        self, position: list[float], orientation: list[float], linear_velocity: list[float], angular_velocity: list[float]
    ) -> None:
        # 設定 self._target_position/_target_orientation/_feedforward_twist，
        # self._is_joint_space_motion = False，切到 velocity 控制模式
        # （self._articulation.switch_dof_control_mode("velocity")），重置
        # self._motion_step_count = 0、self._did_last_motion_timeout = False，
        # 呼叫 self._start_motion()。
        self._target_position = np.asarray(position)
        self._target_orientation = np.asarray(orientation)
        self._feedforward_twist = np.concatenate([np.asarray(linear_velocity), np.asarray(angular_velocity)])
        self._is_joint_space_motion = False
        self._is_swing_motion = False
        self._articulation.switch_dof_control_mode("velocity")
        self._motion_step_count = 0
        self._did_last_motion_timeout = False
        self._start_motion()

    def move_swing(
        self,
        backswing_position: list[float],
        orientation: list[float],
        swing_end_position: list[float],
        orientation_gain: float = 1.0,
        max_angular_speed: float = 0.5,
    ) -> None:
        """揮桿專用速度最優控制。先用一般 pose-tracking（姿態鎖死、
        `linear_velocity=[0,0,0]`）移動到 `backswing_position` 收斂，之後
        自動切換成揮桿模式：每個 physics tick 用線性規劃求「姿態修正角
        速度不超過 `max_angular_speed`（`orientation_gain` 控制修正力道）
        的前提下，沿直線方向最大化平移速度」的關節速度指令，直線移動到
        `swing_end_position`。呼叫端只需要呼叫一次，`is_motion_complete()`
        在後擺+揮桿全程持續回傳 False，語意跟 `move_through_poses()` 一致。

        背景：`_compute_pose_tracking_twist()` 的 P控制器+feedforward 為了
        把姿態鎖死在單一目標點，對某些 Kitchen 案例會讓可達平移速度大幅
        低於運動學理論上限（實測：完全鎖死姿態時最高只能到目標速度的
        ~50-90%，取消姿態約束後理論上限能覆蓋更高的目標速度）——原因是
        P控制器解的是「讓姿態誤差趨近 0」，不是「在容許範圍內最大化平移
        速度」，兩者是不同的最佳化目標，用同一組 DLS 偽逆求解自然無法
        同時達到兩者最優。這裡改用線性規劃直接針對「最大化揮桿方向速度」
        求解，姿態修正只是一個有限額度的約束，不是主要目標，讓擊球時桿身
        姿態盡量貼近目標朝向（不是完全鎖死，允許 `max_angular_speed` 內的
        自然修正，模擬真人揮桿桿身大致穩定但非絕對靜止的手感）而不是把
        平移速度硬吃掉。見 docs/issue-180-reachability-analysis.md 第十六
        節的線性規劃分析與實測數據。

        `orientation_gain`／`max_angular_speed`：跟 `ORIENTATION_GAIN`
        （目前 5.0）同單位但通常小很多，兩者共同決定姿態修正力道上限——
        `orientation_gain` 太大會跟一般 pose-tracking 一樣把速度吃光，
        太小則姿態可能持續緩慢漂移；`max_angular_speed` 直接封頂瞬時角
        速度，避免修正力道在誤差大時暴衝。

        ⚠️ `backswing_position`／`swing_end_position` 沿用
        `swing_trajectory_calculator.compute_swing_waypoints()` 既有慣例，
        是**腕部**（= end-effector 參考點）座標，不是桿尖座標——桿尖在
        `CUE_STICK_GRIP_TO_TIP`（1.35m）之外，姿態鎖死時「腕部走直線」
        跟「桿尖走直線」等價，但這裡姿態允許漂移，兩者不再等價：角速度
        會透過 `ω × tip_offset` 讓桿尖產生遠比腕部本身位移更大的側向
        偏移（見 docs/issue-180-reachability-analysis.md 第十六節——
        沒算這項時，線性規劃找到的「腕部方向最優」角速度反而讓桿尖越轉
        越偏，完全沒碰到球）。這個方法內部會依 `CUE_STICK_GRIP_TO_TIP`
        换算成桿尖座標，用桿尖的實際位置/速度做直線規劃與完成判定，
        呼叫端不需要自己處理這個轉換。
        """
        self._swing_orientation = np.asarray(orientation, dtype=float)
        nominal_tip_direction = self._rotate_vector_by_quat(self._swing_orientation, np.array([0.0, 1.0, 0.0]))
        self._swing_end_position = np.asarray(swing_end_position, dtype=float) + nominal_tip_direction * CUE_STICK_GRIP_TO_TIP
        self._swing_orientation_gain = orientation_gain
        self._swing_max_angular_speed = max_angular_speed
        self._is_swing_motion = False
        self._awaiting_swing_after_backswing = True
        # 清空舊的 waypoint 佇列狀態——如果呼叫端在這之前用過
        # move_through_poses()（例如 AIM 的高架橋序列），_pending_waypoints/
        # _waypoint_index 會留著舊值。揮桿完成後 _step_motion() 落到「還有
        # 沒播完的 waypoint 嗎」這個分支判斷時，沒清空就會誤把舊序列的下一
        # 個 waypoint 當成揮桿後的下一步去追，讓揮桿看起來遲遲沒有真正
        # 完成（實測：真正只需要 ~14 步的揮桿被拖到 57 步，姿態也跟著在
        # 那段多出來的時間裡亂飄——這是 docs/issue-180-reachability-
        # analysis.md 第十六節除錯過程中發現的另一個 bug，不是揮桿本身的
        # 問題）。
        self._pending_waypoints = []
        self._waypoint_index = 0
        self._awaiting_waypoints_after_joint_motion = False
        self._activate_pose_target(backswing_position, orientation, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def _step_swing_motion(self) -> None:
        current_wrist_position = np.array(self.get_end_effector_position())
        current_orientation = self._get_end_effector_world_orientation()
        current_tip_direction = self._rotate_vector_by_quat(current_orientation, np.array([0.0, 1.0, 0.0]))
        tip_offset = current_tip_direction * CUE_STICK_GRIP_TO_TIP
        current_tip_position = current_wrist_position + tip_offset

        # 沿方向投影出「桿尖」目前已走的距離（不是腕部，也不是跟終點的
        # 歐氏距離）——姿態修正過程中允許有限度的側向漂移，用投影距離
        # 判斷「有沒有走完全程」比較符合這裡的完成語意（側向漂移不該讓
        # 完成判定卡住），而且真正需要碰到球的是桿尖，不是腕部。
        traveled = float(np.dot(current_tip_position - self._swing_start, self._swing_direction))
        if traveled >= self._swing_total_distance:
            self._swing_complete = True
            self._articulation.set_dof_velocity_targets(np.zeros((1, len(self._dof_limits))))
            return

        # ⚠️ 第一版曾經把姿態修正做成「等式約束＝目前偏差的 P 修正量」
        # （Jang@qdot == orientation_gain * 目前姿態誤差），實測發現這其實
        # 等同又把角速度鎖回接近 0——揮桿一開始姿態誤差是 0（後擺剛收斂），
        # 這個等式約束因此一直逼近 0，`max_angular_speed` 給的額度完全沒被
        # 用上，平移速度依舊很慢（見 docs/issue-180-reachability-
        # analysis.md 第十六節）。改成**不等式箱型約束**：角速度可以在
        # `restore_bias ± max_angular_speed` 範圍內自由選擇，讓線性規劃
        # 真正能拿這個額度去換取平移速度，`restore_bias`（用
        # `orientation_gain` 縮放、且限制在額度一半內）只當一個溫和的
        # 回正偏置，避免姿態朝單一方向無止盡漂移，不是硬性鎖定。
        restore_bias = self._swing_orientation_gain * self._orientation_error_to_angular_velocity(
            current_orientation, self._swing_orientation
        )
        restore_bias = np.clip(
            restore_bias, -0.5 * self._swing_max_angular_speed, 0.5 * self._swing_max_angular_speed
        )

        jacobians = np.asarray(self._articulation.get_jacobian_matrices().numpy())[0]
        J = jacobians[self._jac_link_index]
        Jv = J[:3, :]
        Jang = J[3:, :]
        # ⚠️ 關鍵修正：`Jv` 是「腕部」的線性 Jacobian，不是桿尖的。桿尖在
        # `tip_offset`（CUE_STICK_GRIP_TO_TIP=1.35m）之外，角速度會透過
        # 剛體速度合成 v_tip = v_wrist + ω × tip_offset 讓桿尖產生遠比腕部
        # 本身位移更大的側向偏移——第一版沒算這項，線性規劃找到的「腕部
        # 方向最優」角速度反而讓桿尖越轉越偏，完全沒碰到球（實測：揮桿
        # 過程中桿尖到球的距離不減反增）。用桿尖的真正線性 Jacobian
        # `Jv_tip = Jv - skew(tip_offset) @ Jang` 取代，才是真正「讓桿尖
        # 沿揮桿方向最大化速度」的目標函式。
        Jv_tip = Jv - self._skew_matrix(tip_offset) @ Jang

        # ⚠️ 第二個關鍵修正：只優化「沿揮桿方向的速度」，完全沒有限制
        # 「垂直於揮桿方向的側向漂移」——目標函式只看 1D 投影進度
        # （`traveled`），線性規劃可以在完全不管側向誤差的情況下讓投影
        # 進度持續增加（沿線前進），同時桿尖在垂直方向越飄越遠，兩者不
        # 矛盾。實測：`traveled` 正常推進到完成，但桿尖到球的實際 3D
        # 距離不減反增，跟垂直分量的角速度沒被約束住直接對應。加一個
        # 側向位置回正項：把桿尖目前偏離「後擺→揮桿終點這條直線」的側向
        # 誤差，轉換成一個有限速度上限的修正目標，跟角速度的 restore_bias
        # 同一個做法——用不等式箱型約束（給線性規劃一點自由度，不是硬性
        # 鎖死），不是把側向速度完全交給最佳化自由發揮。
        lateral_gain = 5.0  # 跟 POSITION_GAIN 同量級，確保側向誤差被積極修正
        lateral_error = (current_tip_position - self._swing_start) - traveled * self._swing_direction
        lateral_restore_velocity = np.clip(-lateral_gain * lateral_error, -0.5, 0.5)
        lateral_tolerance = 0.1  # m/s，側向修正目標附近的自由度

        projection_perp = np.eye(3) - np.outer(self._swing_direction, self._swing_direction)
        Jv_tip_lateral = projection_perp @ Jv_tip

        c = self._swing_direction @ Jv_tip
        bounds = [(-limit, limit) for limit in self._dof_limits]
        A_ub = np.vstack([Jang, -Jang, Jv_tip_lateral, -Jv_tip_lateral])
        b_ub = np.concatenate([
            restore_bias + self._swing_max_angular_speed,
            -restore_bias + self._swing_max_angular_speed,
            lateral_restore_velocity + lateral_tolerance,
            -lateral_restore_velocity + lateral_tolerance,
        ])
        result = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if result.success:
            qdot = result.x
        else:
            # 理論上不該發生（箱型約束＋線性不等式對 7 個自由度來說通常
            # 有解）——真的求不到解時保守回傳零速度，不要下達沒驗證過的
            # 指令。
            logger.warning("move_swing linprog 求解失敗，本步維持不動")
            qdot = np.zeros(len(self._dof_limits))

        if os.environ.get("DEBUG_MOVE_SWING"):
            predicted_speed = float(c @ qdot)
            print(
                f"[move_swing DEBUG] qdot={np.round(qdot, 4).tolist()} predicted_speed={predicted_speed:.4f} "
                f"traveled={traveled:.4f}/{self._swing_total_distance:.4f} linprog_success={result.success}"
            )

        self._articulation.set_dof_velocity_targets(qdot[None, :])

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
        #
        # 如果上一個動作是 move_to_pose()（velocity 控制模式），關節上可能
        # 還留著非零的殘餘速度指令；切到 position 模式不會自動歸零，殘餘
        # 速度會跟新的 position drive 打架，讓末端在目標附近持續小幅震盪、
        # is_motion_complete() 的位置容許誤差永遠卡不進去（實測：同一個
        # 目標單獨呼叫可以在 <5mm 內收斂，緊接在一次 move_to_pose() 後面呼叫
        # 卻 1000 步都收斂不了）。切模式前先把速度目標歸零。
        if self._dof_limits.size > 0:
            self._articulation.set_dof_velocity_targets(np.zeros((1, len(self._dof_limits))))
        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(joint_positions)
        self._target_position = np.asarray(target_end_effector_position)
        self._target_joint_positions = np.asarray(joint_positions).reshape(-1)
        self._is_joint_space_motion = True
        self._is_swing_motion = False
        self._motion_step_count = 0
        self._did_last_motion_timeout = False
        self._start_motion()

    def _step_motion(self, step_dt, context) -> None:
        if self._is_swing_motion:
            self._step_swing_motion()
        elif not self._is_joint_space_motion:
            twist = self._compute_pose_tracking_twist() + self._feedforward_twist

            jacobians = np.asarray(self._articulation.get_jacobian_matrices().numpy())[0]
            J = jacobians[self._jac_link_index]
            JJt = J @ J.T + (self.DLS_LAMBDA**2) * np.eye(6)
            qdot = J.T @ np.linalg.solve(JJt, twist)
            qdot = np.clip(qdot, -self._dof_limits, self._dof_limits)

            self._articulation.set_dof_velocity_targets(qdot[None, :])

        self._motion_step_count += 1

        if self._motion_step_count > self.MOTION_TIMEOUT_STEPS and not self._is_current_target_converged():
            logger.warning("motion timeout: step_count: %d", self._motion_step_count)
            self._did_last_motion_timeout = True
            self._stop_motion()
            self._target_position = None
            self._target_joint_positions = None
            return

        if not self._is_current_target_converged():
            return

        if self._awaiting_swing_after_backswing:
            self._awaiting_swing_after_backswing = False
            self._is_swing_motion = True
            # _swing_start 用「真正量到的」桿尖位置（不是 move_swing() 呼叫
            # 時的 nominal 佔位符），比照既有的
            # compute_canonical_wrist_position() 那類「後擺剛收斂，用實測
            # 位置當下一階段起點」慣例。
            current_wrist_position = np.array(self.get_end_effector_position())
            current_orientation = self._get_end_effector_world_orientation()
            current_tip_direction = self._rotate_vector_by_quat(current_orientation, np.array([0.0, 1.0, 0.0]))
            self._swing_start = current_wrist_position + current_tip_direction * CUE_STICK_GRIP_TO_TIP
            delta = self._swing_end_position - self._swing_start
            self._swing_total_distance = float(np.linalg.norm(delta))
            self._swing_direction = delta / self._swing_total_distance
            self._swing_complete = False
            self._motion_step_count = 0
            return

        if self._awaiting_waypoints_after_joint_motion:
            self._awaiting_waypoints_after_joint_motion = False
            wp = self._pending_waypoints[0]
            self._activate_pose_target(wp.position, wp.orientation, wp.linear_velocity, wp.angular_velocity)
            return

        if self._waypoint_index + 1 < len(self._pending_waypoints):
            self._waypoint_index += 1
            wp = self._pending_waypoints[self._waypoint_index]
            self._activate_pose_target(wp.position, wp.orientation, wp.linear_velocity, wp.angular_velocity)
            return
        
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
    def _quat_error(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
        """q_error = q_target * q_current⁻¹（wxyz），q_error.w 為負時整體取反，
        走最短路徑。供 `_orientation_error_to_angular_velocity()`（角速度控制）
        與 `_is_current_target_converged()`（完成判定）共用。
        """
        w0, x0, y0, z0 = current_wxyz
        current_inv = np.array([w0, -x0, -y0, -z0])

        w1, x1, y1, z1 = target_wxyz
        w2, x2, y2, z2 = current_inv
        q_error = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
        if q_error[0] < 0:
            q_error = -q_error
        return q_error

    @staticmethod
    def _orientation_error_to_angular_velocity(
        current_wxyz: np.ndarray, target_wxyz: np.ndarray
    ) -> np.ndarray:
        """小角度近似下角速度方向 ≈ 2 * q_error.xyz。"""
        q_error = ArticulationAPIImpl._quat_error(current_wxyz, target_wxyz)
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

    @staticmethod
    def _skew_matrix(v: np.ndarray) -> np.ndarray:
        """向量叉積的反對稱矩陣表示：skew(v) @ x == v × x。供
        `_step_swing_motion()` 把「腕部」的線性 Jacobian 換算成「桿尖」的
        線性 Jacobian 用（剛體速度合成 v_tip = v_wrist + ω × tip_offset，
        寫成矩陣形式即 v_tip = (Jv - skew(tip_offset) @ Jang) @ qdot）。"""
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

    def move_to_home(self) -> None:
        self._pending_waypoints = []
        self._awaiting_waypoints_after_joint_motion = False
        if self._default_joint_positions is None or self._home_position is None:
            # initialize() 之後、第一個 physics step 之前就被呼叫：
            # _capture_home_position_once() 還沒把 home 姿態擷取下來，這時直接
            # 往下跑會拿 None 當 joint-space 目標。改成記下來，等擷取完成的
            # 同一個 callback 裡補做（那時場景才真的可讀）。
            self._pending_move_to_home = True
            return
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

    def get_end_effector_orientation(self) -> list[float]:
        return self._get_end_effector_world_orientation().tolist()

    def is_motion_complete(self) -> bool:
        return self._is_current_target_converged()

    def did_last_motion_timeout(self) -> bool:
        return self._did_last_motion_timeout

    def _is_current_target_converged(self) -> bool:
        # self._target_position is None：早退回傳 True（逾時後的清空狀態）。
        # 位置誤差 = ||get_end_effector_position() - self._target_position||，
        # >= POSITION_TOLERANCE：回傳 False。
        # self._is_joint_space_motion 為 True：只看位置，回傳 True。
        # 否則（pose-tracking 模式）：用 self._quat_error() 算姿態誤差角度
        # （2 * ||q_error.xyz||），< ORIENTATION_TOLERANCE 才算收斂。
        #
        # ⚠️ 2026-08-28 曾經在這裡加過「帶 feedforward 速度的 waypoint 放寬
        # 容許值」的修正（見 docs/issue-180-reachability-analysis.md 第十五
        # 節的穩態誤差公式），但用 scripts/verify_swing_trajectory.py 的
        # 真實桿尖速度量測（不是只看位置有沒有收斂）驗證後發現：那個放寬
        # 容許值的門檻剛好落在 P控制器+feedforward 的穩態平衡點，這個平衡
        # 點的物理意義是「合力趨近 0，關節速度也趨近 0」——也就是說系統會
        # 在那裡「宣告完成」，但桿尖當下幾乎是靜止的（實測 speed_error_
        # ratio≈0.98，實際速度只有該有速度的 ~2%），等於沒真正揮桿。這個
        # 修正已還原，STRIKE 隨揮終點的根因修法改成 move_swing()（見文件
        # 第十六節）：真正下令一個沿揮桿方向速度最優的關節速度指令，不是
        # 放寬既有 pose-tracking 的完成判定。
        #
        # move_swing() 的揮桿子階段用 self._swing_complete（由
        # _step_swing_motion() 依「沿方向投影出的移動距離」設定，不是靠
        # 這裡的 POSITION_TOLERANCE/ORIENTATION_TOLERANCE）判斷完成，
        # 必須在 self._target_position 的早退檢查之前處理——揮桿階段
        # self._target_position 還留著後擺子階段的舊值（不是 None），不
        # 特別處理會誤用下面的位置/姿態收斂邏輯。
        if self._is_swing_motion:
            return self._swing_complete

        # joint-space 動作的語意是「把關節開到指定角度」，末端世界位置是結果
        # 不是目標，收斂判定不能拿它當條件——move_to_home() 的
        # target_end_effector_position 是 _home_position，那是第一次 Play 的
        # 第一個 physics step 擷取的**世界**座標，而 _execute_aim() 每次擊球都
        # 會 robot_arm.reposition() 搬動基座。基座一搬走，關節即使完全回到
        # _default_joint_positions，末端世界位置也回不到舊的 _home_position，
        # 位置檢查永遠不過 → 每次 RESET 都跑滿 MOTION_TIMEOUT_STEPS 才逾時
        # 脫困（實測 console 重複出現 motion timeout: step_count: 1001）。
        # 位置/姿態容許值只對 Cartesian pose-tracking 有意義。
        if self._is_joint_space_motion:
            # 逾時善後（見 _step_motion()）會把目標清成 None，代表這次動作
            # 已經被中止，沒有還在等的目標
            if self._target_joint_positions is None:
                return True
            actual_joints = np.asarray(self._articulation.get_dof_positions())[0]
            joint_error = float(np.max(np.abs(actual_joints - self._target_joint_positions)))
            return joint_error < self.JOINT_POSITION_TOLERANCE

        if self._target_position is None:
            return True

        position_error = np.linalg.norm(np.array(self.get_end_effector_position()) - self._target_position)
        if position_error >= self.POSITION_TOLERANCE:
            return False

        current_orientation = self._get_end_effector_world_orientation()
        q_error = self._quat_error(current_orientation, self._target_orientation)
        orientation_error = 2.0 * np.linalg.norm(q_error[1:])
        return orientation_error < self.ORIENTATION_TOLERANCE

    def shutdown(self) -> None:
        pass

    def cancel_pending_home_capture(self) -> None:
        self._pending_move_to_home = False
        if self._capture_callback_id is not None:
            SimulationManager.deregister_callback(self._capture_callback_id)
            self._capture_callback_id = None

    def move_to_joint_position(self, joint_positions: list[float], target_end_effector_position: list[float]) -> None:
        self._pending_waypoints = []
        self._awaiting_waypoints_after_joint_motion = False
        self._start_joint_space_motion(
            np.array([joint_positions]), np.array(target_end_effector_position)
        )