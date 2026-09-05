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
from core.services import swing_trajectory_calculator
from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP

logger = logging.getLogger(__name__)


class ArticulationAPIImpl(ArticulationAPI):
    """
    差動 IK（Jacobian-based）版本，取代 RMPflow：RMPflow 需要為每一款機器人
    準備專屬 motion policy 設定檔，Isaac Sim 不內建 Barrett WAM，跟
    RobotArm「換手臂不用大改程式碼」的抽象介面精神衝突。差動 IK 只需要
    任意 articulation 都通用的 Jacobian，每個 physics tick 解一次
    q̇ = Jᵀ(JJᵀ+λ²I)⁻¹・twist（damped least squares 偽逆）。

    末端世界位姿一律用 RigidPrim（tensor API）讀取，不用 UsdGeom.XformCache
    這類 raw USD 讀法：關節由新版 tensor-based Articulation 驅動，模擬狀態
    走 Fabric，不會同步寫回 classic USD stage 的 xformOp（見
    core/ports/rigid_body_api.py 同一類 tensor/raw USD 不同步問題）。

    UR10e 走完全獨立的 RMPflow 路徑（見 _initialize_ur10e()），詳見
    docs/CHANGELOG.md。
    """

    # P controller 穩態殘留誤差約 2mm，球半徑 28.575mm，此精度對擊球應用
    # 已足夠，不需要比原本 RMPflow 版本的 1mm 更嚴格。
    POSITION_TOLERANCE = 0.005
    # 高架橋案例 CUE_STICK_GRIP_TO_TIP=1.35m 的槓桿臂會放大殘留朝向誤差，
    # 完成判定需一併檢查朝向（見 _is_current_target_converged()）；
    # joint-space 動作不受影響，繼續只看位置。
    ORIENTATION_TOLERANCE = 0.02  # rad，四元數誤差角度
    # joint-space 動作額外要求全部關節都收斂到位，不能只看末端 Cartesian
    # 位置是否「路過」目標——見 docs/issue-180-reachability-analysis.md
    # 第十三節。
    JOINT_POSITION_TOLERANCE = 0.01  # rad
    POSITION_GAIN = 5.0
    ORIENTATION_GAIN = 5.0
    MAX_LINEAR_SPEED = 2.0  # m/s，P controller 位置誤差轉速度指令的上限
    MAX_ANGULAR_SPEED = 3.0  # rad/s
    DLS_LAMBDA = 0.05
    # 超過這個步數仍未收斂就強制視為完成並標記 did_last_motion_timeout()，
    # 避免狀態機靜默卡死在 AIMING（見 docs/issue-flat-case-residual-error.md）。
    MOTION_TIMEOUT_STEPS = 1000
    # move_to_home() 回 home 前先垂直上移的安全距離：關節空間插值不保證
    # 桿尖走直線，可能讓桿尖橫掃過桌面撞到 RESET 剛擺好的球。跟
    # cue_pose_calculator.compute_elevated_bridge_waypoints() 的
    # safe_altitude_margin 同量級，足以清空庫邊高度。
    RESET_LIFT_CLEARANCE_M = 0.3

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
        # joint-space 動作的目標關節角度，只在 _is_joint_space_motion=True
        # 時有意義（見 JOINT_POSITION_TOLERANCE）。
        self._target_joint_positions: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        self._feedforward_twist = np.zeros(6)
        # move_to_home / move_to_joint_position 交給 PhysX 關節驅動器插值，
        # 不需要每個 tick 重新解 Jacobian。
        self._is_joint_space_motion = False
        self._tip_local_offset: np.ndarray | None = None
        self._motion_active = False
        self._step_motion_id: int | None = None
        # move_through_poses() 的 waypoint 佇列狀態：_pending_waypoints 是
        # 目前這次動作依序播放的 Cartesian pose 目標，
        # _awaiting_waypoints_after_joint_motion 代表正在跑
        # preceding_joint_targets 的 joint-space 段，收斂後才切到
        # _pending_waypoints[0]。
        self._pending_waypoints: list[PoseWaypoint] = []
        self._waypoint_index: int = 0
        self._awaiting_waypoints_after_joint_motion: bool = False
        # move_to_home() 先垂直上移一段安全距離，收斂後才切到關節空間回
        # home，見 move_to_home() 與 RESET_LIFT_CLEARANCE_M。
        self._awaiting_home_after_lift: bool = False
        self._motion_step_count: int = 0
        self._did_last_motion_timeout: bool = False
        # None 代表「尚未註冊」或「已觸發並清空」，供
        # cancel_pending_home_capture() 判斷是否還需要取消。
        self._capture_callback_id: int | None = None
        # move_to_home() 若在 home 姿態擷取完成前就被呼叫（Timeline PLAY
        # 當下的 reset 會走到這條路徑），記下來等擷取完成後補做。
        self._pending_move_to_home: bool = False

        # move_swing() 的狀態，見該方法與 _step_swing_motion() 的說明。
        # _awaiting_swing_after_backswing：正在跑後擺（pose-tracking，
        # 姿態鎖死）子階段，收斂後才切到揮桿速度最優控制。
        self._awaiting_swing_after_backswing: bool = False
        # 隨揮（swing_end_position）收斂後桿尖停在緊貼母球原始位置之後
        # 一點點的地方。RESET 會把母球瞬移回這個位置附近，桿尖必須在球被
        # 瞬移回去之前先撤離，因此撤離要接在揮桿本身完成後（move_swing()
        # 內部自動串接），不是接在 move_to_home() 前面——見
        # RESET_LIFT_CLEARANCE_M。
        self._awaiting_retreat_after_swing: bool = False
        self._is_swing_motion: bool = False
        self._swing_complete: bool = False
        self._swing_start: np.ndarray | None = None
        self._swing_direction: np.ndarray | None = None
        self._swing_total_distance: float = 0.0
        self._swing_end_position: np.ndarray | None = None
        self._swing_orientation: np.ndarray | None = None
        self._swing_orientation_gain: float = 1.0
        self._swing_max_angular_speed: float = 0.5

        # move_swing_elbow_pivot()（UR3e 專用揮桿控制）的狀態，跟
        # move_swing() 的 _is_swing_motion／
        # _awaiting_swing_after_backswing 一一對應，差別只在揮桿子階段的
        # 控制策略（單一關節 quintic，不是全關節 LP 最佳化）。揮桿完成後
        # 複用既有的 _awaiting_retreat_after_swing 撤離機制。
        self._awaiting_elbow_pivot_swing_after_backswing: bool = False
        self._is_elbow_pivot_swing_motion: bool = False
        self._elbow_pivot_dof_index: int = 0
        self._elbow_pivot_contact_joint_positions: np.ndarray | None = None
        self._elbow_pivot_target_velocity: float = 0.0
        self._elbow_pivot_quintic: tuple[float, float, float, float] | None = None
        self._elbow_pivot_elapsed_steps: int = 0
        self._elbow_pivot_complete: bool = False

        # UR10e 專用狀態（見 ur10e_rmpflow_controller.py／
        # ur10e_cue_slide_controller.py）：跟上面 WAM7/UR3e 的差動 IK／
        # elbow-pivot 狀態機完全獨立，_ur10e_mode 只在 initialize() 偵測到
        # dof_names 含 "CueSlideJoint" 時開啟。開啟後 move_to_pose()／
        # move_to_home()／is_motion_complete()／did_last_motion_timeout()
        # 這幾個共用方法會在最前面分流，不會執行到下面 WAM7/UR3e 的邏輯。
        self._ur10e_mode: bool = False
        self._ur10e_rmpflow_controller = None
        self._ur10e_linear_approach_controller = None
        self._ur10e_cue_slide_controller = None
        self._ur10e_active_controller = None
        self._ur10e_step_callback_id: int | None = None
        # move_to_pose() 的「退桿 → 移到安全中繼姿態（避障開）→ 逼近緩衝點
        # （避障開）→ 最終姿態（避障關）」四段式序列狀態，見該方法與
        # docs/CHANGELOG.md 的推導過程。
        self._ur10e_awaiting_arm_move_after_retract: bool = False
        self._ur10e_awaiting_final_approach_after_staging: bool = False
        self._ur10e_awaiting_final_short_leg_after_near_final: bool = False
        self._ur10e_pending_staging_target: tuple[list[float], list[float]] | None = None
        self._ur10e_pending_near_final_target: tuple[list[float], list[float]] | None = None
        self._ur10e_pending_arm_target: tuple[list[float], list[float]] | None = None
        # DEBUG_UR10E_AIM_PHASES 除錯用：分段計數各階段消耗的 step 數。
        self._ur10e_aim_phase_step_counter: int = 0
        self._ur10e_aim_final_approach_reported: bool = False
        # register_static_box_obstacle() 每次呼叫都要建立新的 USD prim
        # 路徑（VisualCuboid 不能共用同一個 prim_path 註冊兩個障礙物），
        # 用清單長度當流水號。
        self._ur10e_registered_obstacle_paths: list[str] = []

    def initialize(self) -> None:
        # 在 timeline play 之後呼叫
        self._articulation = Articulation(paths=self._robot_prim_path)
        self._end_effector_rigid_prim = RigidPrim(paths=self._end_effector_prim_path)

        dof_names = list(self._articulation.dof_names)
        self._ur10e_mode = "CueSlideJoint" in dof_names
        if self._ur10e_mode:
            self._initialize_ur10e()
            return

        self._dof_limits = self._load_dof_max_velocities()
        self._jac_link_index = self._resolve_end_effector_jacobian_index()

        self._tip_local_offset = self._compute_tip_local_offset()
        self._boost_wrist_gains_for_cue_stick_load()

        cue_stick_prim_path = self._resolve_cue_stick_prim_path()
        self._cue_stick_rigid_prim = (
            RigidPrim(paths=cue_stick_prim_path) if cue_stick_prim_path is not None else None
        )

        self._capture_callback_id = SimulationManager.register_callback(
            self._capture_home_position_once, event=SimulationEvent.PHYSICS_POST_STEP
        )

    def _initialize_ur10e(self) -> None:
        """UR10e 專屬初始化路徑，完全繞開上面 WAM7/UR3e 的差動 IK 設定與
        home-capture 機制——UR10e 用固定 HOME 關節角度，不是「USD 重新
        放進場景時的自然落點」。

        刻意不呼叫 _boost_wrist_gains_for_cue_stick_load()：那組極高增益
        （stiffness=1e15）是幫差動 IK 控制下的 WAM7/UR3e 補強用的，UR10e
        完全交給 RMPflow 驅動，套用會干擾 RMPflow 自己的 PD 追蹤動態。
        """
        from .ur10e_cue_slide_controller import Ur10eCueSlideController
        from .ur10e_linear_approach_controller import Ur10eLinearApproachController
        from .ur10e_rmpflow_controller import Ur10eRmpflowController

        self._ur10e_rmpflow_controller = Ur10eRmpflowController(
            self._articulation, self._end_effector_prim_path
        )
        self._ur10e_linear_approach_controller = Ur10eLinearApproachController(
            self._articulation, self._end_effector_prim_path
        )
        self._ur10e_cue_slide_controller = Ur10eCueSlideController(self._articulation)
        self._ur10e_step_callback_id = SimulationManager.register_callback(
            self._step_ur10e_motion, event=SimulationEvent.PHYSICS_POST_STEP
        )

    def _step_ur10e_motion(self, step_dt, context) -> None:
        if self._ur10e_awaiting_arm_move_after_retract:
            if not self._ur10e_cue_slide_controller.is_motion_complete():
                self._ur10e_cue_slide_controller.step(step_dt)
                return
            # 退桿完成（或逾時，best-effort 繼續）——切到 RMPflow 控制器
            # 先移到安全中繼姿態；退桿階段自己的逾時狀態不往外傳，這個
            # 序列只回報「平移到最終姿態」那一段的完成/逾時狀態。
            self._ur10e_awaiting_arm_move_after_retract = False
            self._ur10e_awaiting_final_approach_after_staging = True
            staging_position, staging_orientation = self._ur10e_pending_staging_target
            self._ur10e_pending_staging_target = None
            self._ur10e_active_controller = self._ur10e_rmpflow_controller
            self._ur10e_aim_phase_step_counter = 0
            self._ur10e_rmpflow_controller.move_to_pose(staging_position, staging_orientation)
            return

        if self._ur10e_awaiting_final_approach_after_staging:
            self._ur10e_aim_phase_step_counter += 1
            if not self._ur10e_rmpflow_controller.is_motion_complete():
                self._ur10e_rmpflow_controller.step(step_dt)
                return
            # 安全中繼姿態到位——平移到逼近緩衝點，方向跟中繼姿態相同，
            # 不需要再解一次重新定向。母球避障這段仍然開著，下一段才關閉。
            if os.environ.get("DEBUG_UR10E_AIM_PHASES"):
                print(
                    f"[aim_phases] STAGING 階段結束：耗費 step={self._ur10e_aim_phase_step_counter} "
                    f"timeout={self._ur10e_rmpflow_controller.did_last_motion_timeout()} "
                    f"目前 wrist 位置={self.get_end_effector_position()}",
                    flush=True,
                )
            self._ur10e_awaiting_final_approach_after_staging = False
            self._ur10e_awaiting_final_short_leg_after_near_final = True
            near_final_position, near_final_orientation = self._ur10e_pending_near_final_target
            self._ur10e_pending_near_final_target = None
            self._ur10e_aim_phase_step_counter = 0
            self._ur10e_rmpflow_controller.move_to_pose(near_final_position, near_final_orientation)
            return

        if self._ur10e_awaiting_final_short_leg_after_near_final:
            self._ur10e_aim_phase_step_counter += 1
            if not self._ur10e_rmpflow_controller.is_motion_complete():
                self._ur10e_rmpflow_controller.step(step_dt)
                return
            # 逼近緩衝點到位——剩下這一小段是純軸向平移、方向不變，改用
            # Lula 離線軌跡精確走完（見 Ur10eLinearApproachController）。
            # 桿子在 AIM 期間已經退到後擺位置，沿軸平移碰不到母球，安全性
            # 由幾何保證，不需要避障，也就沒有「母球同時是障礙物又是目的地」
            # 的矛盾。Lula 產不出軌跡時退回原本的 RMPflow 路徑（那條才需要
            # 停用母球避障）。
            if os.environ.get("DEBUG_UR10E_AIM_PHASES"):
                print(
                    f"[aim_phases] NEAR_FINAL 階段結束：耗費 step={self._ur10e_aim_phase_step_counter} "
                    f"timeout={self._ur10e_rmpflow_controller.did_last_motion_timeout()} "
                    f"目前 wrist 位置={self.get_end_effector_position()}",
                    flush=True,
                )
            self._ur10e_awaiting_final_short_leg_after_near_final = False
            position, orientation = self._ur10e_pending_arm_target
            self._ur10e_pending_arm_target = None
            self._ur10e_aim_phase_step_counter = 0
            self._ur10e_aim_final_approach_reported = False
            if self._ur10e_linear_approach_controller.move_to_pose(position, orientation):
                self._ur10e_active_controller = self._ur10e_linear_approach_controller
            else:
                self._ur10e_rmpflow_controller.disable_dynamic_obstacles()
                self._ur10e_active_controller = self._ur10e_rmpflow_controller
                self._ur10e_rmpflow_controller.move_to_pose(position, orientation)
            return

        if self._ur10e_active_controller is not None:
            debug_aim_phases = os.environ.get("DEBUG_UR10E_AIM_PHASES") and self._ur10e_active_controller is self._ur10e_rmpflow_controller
            if debug_aim_phases and not self._ur10e_rmpflow_controller.is_motion_complete():
                self._ur10e_aim_phase_step_counter += 1
            self._ur10e_active_controller.step(step_dt)
            if (
                debug_aim_phases
                and not self._ur10e_aim_final_approach_reported
                and self._ur10e_rmpflow_controller.is_motion_complete()
            ):
                self._ur10e_aim_final_approach_reported = True
                print(
                    f"[aim_phases] FINAL_APPROACH 階段結束：耗費 step={self._ur10e_aim_phase_step_counter} "
                    f"timeout={self._ur10e_rmpflow_controller.did_last_motion_timeout()} "
                    f"目前 wrist 位置={self.get_end_effector_position()}",
                    flush=True,
                )

    def set_robot_base_pose(
        self, base_position: list[float], base_orientation: list[float]
    ) -> None:
        if not self._ur10e_mode:
            return
        self._ur10e_rmpflow_controller.set_robot_base_pose(base_position, base_orientation)
        self._ur10e_linear_approach_controller.set_robot_base_position(base_position)

    def register_static_box_obstacle(self, center: list[float], size: list[float]) -> None:
        if not self._ur10e_mode:
            return
        # 用 VisualCuboid（純幾何+世界座標，沒有 RigidBodyAPI/CollisionAPI）
        # 而不是 FixedCuboid：後者會參與真實 PhysX 碰撞，讓這個純粹給
        # RMPflow 避障邏輯參考用的標記變成真的會撞的東西。
        from isaacsim.core.api.objects import VisualCuboid

        obstacle_path = f"/World/_RmpflowStaticObstacle_{len(self._ur10e_registered_obstacle_paths)}"
        self._ur10e_registered_obstacle_paths.append(obstacle_path)
        obstacle = VisualCuboid(
            prim_path=obstacle_path,
            position=np.asarray(center, dtype=float),
            scale=np.asarray(size, dtype=float),
            size=1.0,
            visible=False,
        )
        self._ur10e_rmpflow_controller.add_obstacle(obstacle, static=True)

    def register_dynamic_sphere_obstacle(self, prim_path: str, radius: float) -> None:
        if not self._ur10e_mode:
            return
        # 建立獨立的 DynamicSphere 障礙物 proxy，每個 tick 從母球真正的
        # RigidPrim 讀取最新世界座標同步過去（見
        # Ur10eRmpflowController.add_dynamic_sphere_obstacle()），不直接
        # 包既有母球 prim——母球實際的 Sphere geometry 在更深的子節點，
        # DynamicSphere 建構子會嚴格檢查 prim type，包不了。
        self._ur10e_rmpflow_controller.add_dynamic_sphere_obstacle(prim_path, radius)

    def move_cue_slide_stroke(
        self, backswing_position: float, target_velocity: float
    ) -> None:
        self._ur10e_active_controller = self._ur10e_cue_slide_controller
        self._ur10e_cue_slide_controller.move_stroke(backswing_position, target_velocity)

    def _resolve_cue_stick_prim_path(self) -> str | None:
        """CueStick 跟 Robot 是同一個 base_path 底下的手足 prim（見
        TableRobotManager：`{base_path}/Robot`、`{base_path}/CueStick`）。
        不是每個呼叫端都有掛球桿，先確認 prim 存在才建立 RigidPrim。"""
        if not self._robot_prim_path.endswith("/Robot"):
            return None
        base_path = self._robot_prim_path[: -len("/Robot")]
        cue_stick_prim_path = base_path + "/CueStick"
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(cue_stick_prim_path).IsValid():
            return None
        return cue_stick_prim_path

    def _get_robot_prim_world_position(self) -> list[float]:
        """Robot prim 本身不是 physics 模擬的剛體，是靠
        stage_api.set_prim_translate() 設一次性的 classic USD xformOp（見
        BarrettWamRobot.reposition()），不會每個 tick 被 Fabric 覆寫，讀
        raw USD 沒有本檔案 class docstring 提到的 tensor/USD 不同步問題。
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

    _WRIST_GAIN_BOOST_JOINT_NAME_SUBSTRINGS = ("wrist_1", "wrist_3")
    """UR3e 專用：`dof_names` 裡符合這些子字串的關節才會被加強，WAM7 的
    關節命名（`wam_*`）不會命中，等同對 WAM7 是 no-op。"""
    _WRIST_GAIN_STIFFNESS = 1e15
    _WRIST_GAIN_DAMPING = 1e5
    """沿用 Isaac Sim 官方 Gain Tuner extension「SET STIFF GAINS」慣例值
    （見 isaacsim.robot_setup.gain_tuner），不是自訂數值。"""
    _WRIST_GAIN_MAX_EFFORT_MULTIPLIER = 20.0

    def _boost_wrist_gains_for_cue_stick_load(self) -> None:
        """球桿透過 FixedJoint 剛性掛在腕部後，1.35m 力臂會在肘關節動態
        擺動時對 wrist_1/wrist_3（UR3e 扭矩容許值較小的兩個關節）產生額外
        反作用力矩，PD 增益/扭矩上限不夠會讓關節收斂不到目標角度（見
        NVIDIA 官方 Gain Tuner extension 文件同一類問題）。提高這兩個
        關節的增益與扭矩上限來補償，詳細除錯過程見 docs/CHANGELOG.md。

        只挑 wrist_1/wrist_3（wrist_2 收斂正常）：肘關節後續會切到
        velocity 模式（自動把 stiffness 歸零，這裡先調高不影響揮桿行為），
        其餘關節維持原廠數值。用 dof_names 子字串比對，對 WAM7 是 no-op。
        """
        if self._articulation is None:
            return
        dof_names = list(self._articulation.dof_names) if hasattr(self._articulation, "dof_names") else None
        if dof_names is None:
            return
        target_indices = [
            i for i, name in enumerate(dof_names)
            if any(sub in name.lower() for sub in self._WRIST_GAIN_BOOST_JOINT_NAME_SUBSTRINGS)
        ]
        if not target_indices:
            return

        stiffnesses, dampings = self._articulation.get_dof_gains()
        stiffnesses = np.asarray(stiffnesses.numpy() if hasattr(stiffnesses, "numpy") else stiffnesses, dtype=float)
        dampings = np.asarray(dampings.numpy() if hasattr(dampings, "numpy") else dampings, dtype=float)
        max_efforts = np.asarray(
            self._articulation.get_dof_max_efforts().numpy()
            if hasattr(self._articulation.get_dof_max_efforts(), "numpy")
            else self._articulation.get_dof_max_efforts(),
            dtype=float,
        )
        if stiffnesses.ndim == 2:
            stiffnesses, dampings, max_efforts = stiffnesses[0], dampings[0], max_efforts[0]

        for idx in target_indices:
            stiffnesses[idx] = self._WRIST_GAIN_STIFFNESS
            dampings[idx] = self._WRIST_GAIN_DAMPING
            max_efforts[idx] = max_efforts[idx] * self._WRIST_GAIN_MAX_EFFORT_MULTIPLIER

        logger.info(
            "wrist gain boost: joint indices=%s new_max_efforts=%s（見 _boost_wrist_gains_for_cue_stick_load docstring）",
            target_indices, [max_efforts[i] for i in target_indices],
        )
        self._articulation.set_dof_gains(stiffnesses[None, :], dampings[None, :])
        self._articulation.set_dof_max_efforts(max_efforts[None, :])

    def _compute_tip_local_offset(self) -> np.ndarray:
        """末端執行器 link 若本身有幾何體（例如 UR5 的 wrist_3_link 凸緣
        零件），用 local bounding box 沿最長軸找「離原點最遠的一端」當
        工具尖端；若這個 link 純粹是掛載參考點、沒有幾何體（例如 Barrett
        WAM 的 wam_wrist_palm_stump_link），USD 對空 bounding box 的慣例
        回傳值是無效的 min>max 範圍，這種情況直接視為「原點本身就是
        尖端」，偏移量為 0。
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
        # _default_joint_positions 跟 _home_position 用同一個
        # PHYSICS_POST_STEP callback 擷取：get_dof_positions() 若在
        # initialize() 同步呼叫（physics 可能還一步都沒跑）會拿到不可靠的
        # 值，必須等 physics 至少跑過一步才讀。
        self._default_joint_positions = np.asarray(self._articulation.get_dof_positions())
        self._home_position = np.array(self.get_end_effector_position())
        SimulationManager.deregister_callback(self._capture_callback_id)
        self._capture_callback_id = None

        if self._pending_move_to_home:
            self._pending_move_to_home = False
            self.move_to_home()

    # 沿用 STRIKE 揮桿用的 DEFAULT_BACKSWING_DISTANCE_M：退桿距離跟「手臂
    # 移動過程中桿子掃過的空間大小」互相拉扯（加大退桿距離不能單方面
    # 解決蹭球風險），推導過程見 docs/CHANGELOG.md。
    _UR10E_AIM_RETRACT_POSITION_M = -swing_trajectory_calculator.DEFAULT_BACKSWING_DISTANCE_M

    # STAGING 中繼姿態的偏移量：先移到方向與最終 AIM 目標相同、但沿桿軸
    # 往後退開這個距離的安全姿態（母球避障全程啟用），把「大幅重新定向」
    # 這個高風險動作留在遠離球的階段做；FINAL_APPROACH 只需沿同一軸向
    # 直線平移，不必再解一次 6-DOF 重新定向。數值來源與試過的替代方案見
    # docs/CHANGELOG.md。
    _UR10E_AIM_STAGING_OFFSET_M = CUE_STICK_GRIP_TO_TIP + 0.1

    # FINAL_APPROACH 拆成兩段（見 move_to_pose()）：先在避障開著的情況下
    # 逼近到只剩這個緩衝距離的「逼近緩衝點」，最後才關避障直線逼近剩下
    # 一小段。只需大於退桿距離 DEFAULT_BACKSWING_DISTANCE_M（0.15m）留
    # 餘裕即可，不需要跟 _UR10E_AIM_STAGING_OFFSET_M 一樣大。
    _UR10E_FINAL_APPROACH_SAFE_MARGIN_M = 0.2

    def move_to_pose(self, position: list[float], orientation: list[float], linear_velocity: list[float] = [0.0, 0.0, 0.0], angular_velocity: list[float] = [0.0, 0.0, 0.0]) -> None:
        if self._ur10e_mode:
            # linear_velocity/angular_velocity 沒有對應語意（RMPflow 是
            # 反應式收斂，不是 feed-forward 速度控制），UR10e 呼叫端不會
            # 傳非零值，這裡忽略。
            #
            # 移動手臂前先讓 CueSlideJoint 退到後擺位置，退到位才開始移動
            # 手臂——手臂定位全程桿尖都在安全距離外，避免桿尖貼球狀態下
            # 修正路徑蹭到球。實際序列由 _step_ur10e_motion() 的狀態機
            # 驅動：退桿 → 安全中繼姿態 → 逼近緩衝點 → 最終姿態，這裡只
            # 負責啟動＋算好各段目標存起來。
            #
            # 每次新的 move_to_pose() 呼叫都重新啟用動態障礙物——上一次
            # 呼叫的最終逼近階段可能停用過（見 disable_dynamic_obstacles()
            # 呼叫處），沒有重新啟用，安全中繼姿態這段大幅移動就不會避開
            # 母球。
            self._ur10e_rmpflow_controller.enable_dynamic_obstacles()
            approach_direction = self._rotate_vector_by_quat(
                np.asarray(orientation, dtype=float), np.array([0.0, 1.0, 0.0])
            )
            staging_position = (
                np.asarray(position, dtype=float) - approach_direction * self._UR10E_AIM_STAGING_OFFSET_M
            ).tolist()
            near_final_position = (
                np.asarray(position, dtype=float) - approach_direction * self._UR10E_FINAL_APPROACH_SAFE_MARGIN_M
            ).tolist()
            self._ur10e_pending_staging_target = (staging_position, orientation)
            self._ur10e_pending_near_final_target = (near_final_position, orientation)
            self._ur10e_pending_arm_target = (position, orientation)
            self._ur10e_awaiting_arm_move_after_retract = True
            self._ur10e_active_controller = self._ur10e_cue_slide_controller
            self._ur10e_cue_slide_controller.retract(self._UR10E_AIM_RETRACT_POSITION_M)
            return
        self.move_through_poses(
            [PoseWaypoint(position=position, orientation=orientation, linear_velocity=linear_velocity, angular_velocity=angular_velocity)]
        )

    def move_through_poses(
        self,
        waypoints: list[PoseWaypoint],
        preceding_joint_targets: tuple[list[float], list[float]] | None = None,
    ) -> None:
        """依序移動末端通過一串 Cartesian pose 目標，內部自我驅動、自我
        轉換階段，呼叫端只需要呼叫一次；只有走到最後一個 waypoint 才視為
        「動作完成」，語意跟 move_to_pose() 一致。

        preceding_joint_targets 不為 None 時，先收斂到這組安全的
        joint-space 姿態（避開差動 IK 在奇異點附近的失穩問題），再開始
        播放 waypoints。
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
        self._target_position = np.asarray(position)
        self._target_orientation = np.asarray(orientation)
        self._feedforward_twist = np.concatenate([np.asarray(linear_velocity), np.asarray(angular_velocity)])
        self._is_joint_space_motion = False
        self._is_swing_motion = False
        self._is_elbow_pivot_swing_motion = False
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
        """揮桿專用速度最優控制。先用一般 pose-tracking（姿態鎖死）移動到
        `backswing_position` 收斂，之後自動切換成揮桿模式：每個 physics
        tick 用線性規劃求「姿態修正角速度不超過 `max_angular_speed`（受
        `orientation_gain` 控制）的前提下，沿直線方向最大化桿尖平移速度」
        的關節速度指令。呼叫端只需呼叫一次，`is_motion_complete()` 在
        後擺+揮桿全程回傳 False。

        用線性規劃而非一般 pose-tracking 的原因：姿態鎖死的 P controller
        解的是「讓姿態誤差趨近 0」，跟「在容許範圍內最大化平移速度」是
        不同的最佳化目標，同一組 DLS 偽逆無法同時達到兩者最優（詳細分析
        與實測數據見 docs/issue-180-reachability-analysis.md 第十六節）。

        ⚠️ `backswing_position`／`swing_end_position` 是**腕部**座標（沿用
        `swing_trajectory_calculator.compute_swing_waypoints()` 慣例），
        不是桿尖座標。姿態允許在揮桿中漂移，「腕部走直線」與「桿尖走
        直線」不再等價（角速度會透過 `ω × tip_offset` 放大桿尖側向偏移），
        這個方法內部會依 `CUE_STICK_GRIP_TO_TIP` 換算成桿尖座標處理，
        呼叫端不需要自己轉換。
        """
        self._swing_orientation = np.asarray(orientation, dtype=float)
        nominal_tip_direction = self._rotate_vector_by_quat(self._swing_orientation, np.array([0.0, 1.0, 0.0]))
        self._swing_end_position = np.asarray(swing_end_position, dtype=float) + nominal_tip_direction * CUE_STICK_GRIP_TO_TIP
        self._swing_orientation_gain = orientation_gain
        self._swing_max_angular_speed = max_angular_speed
        self._is_swing_motion = False
        self._awaiting_swing_after_backswing = True
        self._awaiting_retreat_after_swing = True
        # 清空舊的 waypoint 佇列狀態：若呼叫端在此之前用過
        # move_through_poses()（例如高架橋 AIM 序列），沒清空的話
        # _step_motion() 會誤把舊序列的下一個 waypoint 當成揮桿後的下一步
        # 去追，讓揮桿遲遲無法真正完成。
        self._pending_waypoints = []
        self._waypoint_index = 0
        self._awaiting_waypoints_after_joint_motion = False
        self._activate_pose_target(backswing_position, orientation, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def move_swing_elbow_pivot(
        self,
        backswing_joint_positions: list[float],
        backswing_target_end_effector_position: list[float],
        contact_joint_positions: list[float],
        elbow_dof_index: int,
        target_elbow_velocity: float,
    ) -> None:
        """UR3e 專用揮桿控制：跟 `move_swing()` 是平行、互斥的兩套策略，
        不是它的變形。只讓 `elbow_dof_index` 這一個關節從 0 加速到
        `target_elbow_velocity`，其餘關節角速度固定為 0，即可達到目標
        桿尖速度（WAM7 需要 `move_swing()` 的全關節 LP 最佳化才能協調
        達成，UR3e 已驗證不需要，見 docs/CHANGELOG.md）。

        做法：先用 joint-space 收斂到 `backswing_joint_positions`；收斂後
        對 `elbow_dof_index` 解一段 quintic polynomial（起點=收斂當下
        實測角度、終點角度=`contact_joint_positions[elbow_dof_index]`、
        終點角速度=`target_elbow_velocity`），time-scaling 找最小可行 T，
        逐 tick 下達 q̇(t)（含重力補償）。完成後複用
        `_awaiting_retreat_after_swing` 撤離機制。

        ⚠️ `backswing_joint_positions`／`contact_joint_positions` 是完整
        6 個關節角度，只有 `elbow_dof_index` 那個分量在揮桿階段會變動，
        呼叫端需自行保證兩組角度除了 elbow 之外一致。

        ⚠️ 只驗證過「後擺→接觸」這段揮桿動作本身，未驗證「從目前姿態
        安全接近到 backswing_joint_positions」這一段（WAM7 為此設計了
        B1/B2/C1/C2 多階段安全接近序列，見
        `cue_pose_calculator.compute_elevated_bridge_waypoints()`，UR3e
        目前沒有對應機制）。
        """
        self._elbow_pivot_dof_index = elbow_dof_index
        self._elbow_pivot_contact_joint_positions = np.array(contact_joint_positions, dtype=float)
        self._elbow_pivot_target_velocity = float(target_elbow_velocity)
        self._is_elbow_pivot_swing_motion = False
        self._awaiting_elbow_pivot_swing_after_backswing = True
        self._awaiting_retreat_after_swing = True
        self._pending_waypoints = []
        self._waypoint_index = 0
        self._awaiting_waypoints_after_joint_motion = False
        self._start_joint_space_motion(
            np.array([backswing_joint_positions]), np.asarray(backswing_target_end_effector_position)
        )

    @staticmethod
    def _solve_quintic_coeffs(q0: float, q1: float, v1: float, T: float) -> tuple[float, float, float]:
        """單一關節 joint-space quintic：`q(0)=q0,q̇(0)=0,q̈(0)=0,q(T)=q1,
        q̇(T)=v1,q̈(T)=0`。回傳 `(c3,c4,c5)`（`c0=q0,c1=0,c2=0` 已知），
        跟 `scripts/test_ur3e_human_pose_swing_speed.py` 同一個公式。

        ⚠️ `v(T)=v1` 不會隨 `T` 縮放——`v1` 若已超過關節限速，加大 `T`
        救不了，呼叫端須自行確認 `target_elbow_velocity` 在馬達限速內。
        """
        A = np.array([
            [T ** 3, T ** 4, T ** 5],
            [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [6 * T, 12 * T ** 2, 20 * T ** 3],
        ])
        b = np.array([q1 - q0, v1, 0.0])
        c3, c4, c5 = np.linalg.solve(A, b)
        return float(c3), float(c4), float(c5)

    @staticmethod
    def _quintic_velocity(c3: float, c4: float, c5: float, t: float) -> float:
        return 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4

    @staticmethod
    def _peak_abs_quintic_velocity(c3: float, c4: float, c5: float, T: float, samples: int = 200) -> float:
        ts = np.linspace(0.0, T, samples)
        return max(abs(ArticulationAPIImpl._quintic_velocity(c3, c4, c5, t)) for t in ts)

    def _step_elbow_pivot_swing_motion(self) -> None:
        c3, c4, c5, T = self._elbow_pivot_quintic
        physics_dt = 1.0 / 60.0
        t = min(self._elbow_pivot_elapsed_steps * physics_dt, T)
        qdot = np.zeros(len(self._dof_limits))
        qdot[self._elbow_pivot_dof_index] = self._quintic_velocity(c3, c4, c5, t)
        self._apply_velocity_targets_with_gravity_compensation(qdot)
        self._elbow_pivot_elapsed_steps += 1
        if self._elbow_pivot_elapsed_steps * physics_dt >= T:
            self._elbow_pivot_complete = True

    def _apply_velocity_targets_with_gravity_compensation(self, qdot: np.ndarray) -> None:
        """下達關節角速度指令的同時疊加重力補償力矩前饋。

        `switch_dof_control_mode("velocity")` 只把 drive 的 stiffness
        歸零，damping 沿用 USD 內建值，PhysX 的 velocity-mode PD 只針對
        「目標速度 vs 目前速度」的誤差出力，跟重力力矩大小無關——若 USD
        裡的 damping 不夠大，速度目標=0 時關節會完全沒有力矩對抗重力。
        WAM7 的 URDF→USD 轉換工具剛好寫死了偏高的 damping（見
        `assets/barrett_wam/wam7/payloads/Physics/physics.usda`）意外
        夠用，換一支手臂或重新轉換 USD 都可能重演這個問題，詳見
        docs/CHANGELOG.md。

        做法：每個 tick 讀 `get_dof_gravity_compensation_forces()`（維持
        目前姿態靜止所需的重力補償力矩），用 `set_dof_efforts()` 疊加
        上去——這是額外的 actuation force，跟 velocity drive 的 PD 力矩
        相加，不需要切到 `"effort"` 模式（那樣會把 stiffness/damping 一併
        歸零）。`set_dof_efforts()` 官方文件要求每個 physics tick 重新
        呼叫，正好符合這個函式已經在每個 tick 被呼叫的慣例。

        ⚠️ 若場景走新版 Newton physics tensor backend，
        `get_dof_gravity_compensation_forces()` 目前是官方尚未實作的
        stub（回傳全 0），補償會退化成無作用；本專案走舊版 PhysX tensor
        backend，不受影響。

        ⚠️ 只在動作進行中（`_step_swing_motion()`／`_step_motion()`）
        生效，`_stop_motion()` 解除 callback 後不再每 tick 呼叫
        `set_dof_efforts()`，閒置持穩狀態不受這段補償影響。
        """
        self._articulation.set_dof_velocity_targets(qdot[None, :])
        gravity_compensation_forces = self._articulation.get_dof_gravity_compensation_forces()
        self._articulation.set_dof_efforts(gravity_compensation_forces)

    def _step_swing_motion(self) -> None:
        current_wrist_position = np.array(self.get_end_effector_position())
        current_orientation = self._get_end_effector_world_orientation()
        current_tip_direction = self._rotate_vector_by_quat(current_orientation, np.array([0.0, 1.0, 0.0]))
        tip_offset = current_tip_direction * CUE_STICK_GRIP_TO_TIP
        current_tip_position = current_wrist_position + tip_offset

        # 沿方向投影出桿尖已走的距離（不是腕部、也不是跟終點的歐氏距離）
        # ——姿態修正過程允許有限度側向漂移，完成判定看投影進度、且看的是
        # 桿尖（真正需要碰到球的部位）而非腕部。
        traveled = float(np.dot(current_tip_position - self._swing_start, self._swing_direction))
        if traveled >= self._swing_total_distance:
            self._swing_complete = True
            self._apply_velocity_targets_with_gravity_compensation(np.zeros(len(self._dof_limits)))
            return

        # 姿態修正用不等式箱型約束（角速度可在 restore_bias ±
        # max_angular_speed 內自由選擇），而非等式約束鎖定在目前 P
        # 修正量——等式約束會讓線性規劃找不到平移速度的自由度（見
        # docs/issue-180-reachability-analysis.md 第十六節）。
        # restore_bias 只當溫和回正偏置，避免姿態單向無止盡漂移。
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
        # `Jv` 是腕部的線性 Jacobian，桿尖在 `tip_offset` 之外，剛體速度
        # 合成 v_tip = v_wrist + ω × tip_offset，換算成桿尖的線性 Jacobian
        # `Jv_tip = Jv - skew(tip_offset) @ Jang`，才是真正「讓桿尖沿揮桿
        # 方向最大化速度」的目標函式。
        Jv_tip = Jv - self._skew_matrix(tip_offset) @ Jang

        # 只優化沿揮桿方向的速度會讓垂直方向的側向漂移不受控制（線性
        # 規劃可以在完全不管側向誤差下持續推進投影進度），因此加一個
        # 側向位置回正項：把桿尖偏離「後擺→揮桿終點」直線的側向誤差轉成
        # 有限速度上限的修正目標，跟角速度的 restore_bias 同一個不等式
        # 箱型約束做法。
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
            # 箱型約束＋線性不等式對 7 個自由度通常有解，理論上不該發生；
            # 真的求不到解時保守回傳零速度，不下達沒驗證過的指令。
            logger.warning("move_swing linprog 求解失敗，本步維持不動")
            qdot = np.zeros(len(self._dof_limits))

        if os.environ.get("DEBUG_MOVE_SWING"):
            predicted_speed = float(c @ qdot)
            print(
                f"[move_swing DEBUG] qdot={np.round(qdot, 4).tolist()} predicted_speed={predicted_speed:.4f} "
                f"traveled={traveled:.4f}/{self._swing_total_distance:.4f} linprog_success={result.success}"
            )

        self._apply_velocity_targets_with_gravity_compensation(qdot)

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
        # 切到 position 模式前先把速度目標歸零：若上一個動作是
        # move_to_pose()（velocity 模式），關節上可能還留著非零殘餘速度
        # 指令，切模式不會自動歸零，殘餘速度會跟新的 position drive 打架
        # 導致末端在目標附近持續小幅震盪、永遠收斂不進容許誤差。
        if self._dof_limits.size > 0:
            self._articulation.set_dof_velocity_targets(np.zeros((1, len(self._dof_limits))))
        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(joint_positions)
        self._target_position = np.asarray(target_end_effector_position)
        self._target_joint_positions = np.asarray(joint_positions).reshape(-1)
        self._is_joint_space_motion = True
        self._is_swing_motion = False
        self._is_elbow_pivot_swing_motion = False
        self._motion_step_count = 0
        self._did_last_motion_timeout = False
        self._start_motion()

    def _step_motion(self, step_dt, context) -> None:
        if self._is_swing_motion:
            self._step_swing_motion()
        elif self._is_elbow_pivot_swing_motion:
            self._step_elbow_pivot_swing_motion()
        elif not self._is_joint_space_motion:
            twist = self._compute_pose_tracking_twist() + self._feedforward_twist

            jacobians = np.asarray(self._articulation.get_jacobian_matrices().numpy())[0]
            J = jacobians[self._jac_link_index]
            JJt = J @ J.T + (self.DLS_LAMBDA**2) * np.eye(6)
            qdot = J.T @ np.linalg.solve(JJt, twist)
            qdot = np.clip(qdot, -self._dof_limits, self._dof_limits)

            self._apply_velocity_targets_with_gravity_compensation(qdot)

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
            # 用實測的桿尖位置（不是 move_swing() 呼叫時的 nominal
            # 佔位符）當後擺結束、揮桿開始的起點。
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

        if self._awaiting_elbow_pivot_swing_after_backswing:
            self._awaiting_elbow_pivot_swing_after_backswing = False
            self._is_elbow_pivot_swing_motion = True
            self._is_joint_space_motion = False
            # 用實測收斂到的肘關節角度當 quintic 起點（不是呼叫端傳入的
            # 後擺目標角度），JOINT_POSITION_TOLERANCE 容許的殘留誤差用
            # 實測值比用目標值更準。
            live_joints = np.asarray(self._articulation.get_dof_positions())[0]
            q0 = float(live_joints[self._elbow_pivot_dof_index])
            q1 = float(self._elbow_pivot_contact_joint_positions[self._elbow_pivot_dof_index])
            v1 = self._elbow_pivot_target_velocity
            elbow_limit = float(self._dof_limits[self._elbow_pivot_dof_index])
            T = max(abs(q1 - q0) / max(abs(v1), 1e-6), 0.05)
            c3 = c4 = c5 = 0.0
            for _ in range(50):
                c3, c4, c5 = self._solve_quintic_coeffs(q0, q1, v1, T)
                peak_velocity = self._peak_abs_quintic_velocity(c3, c4, c5, T)
                if peak_velocity <= elbow_limit + 1e-9:
                    break
                T *= (peak_velocity / elbow_limit) * 1.05
            else:
                logger.warning("move_swing_elbow_pivot: time-scaling 50 次仍未收斂，直接用目前的 T=%f", T)
            self._elbow_pivot_quintic = (c3, c4, c5, T)
            self._elbow_pivot_elapsed_steps = 0
            self._elbow_pivot_complete = False
            self._articulation.switch_dof_control_mode("velocity")
            self._motion_step_count = 0
            return

        if self._awaiting_retreat_after_swing and (self._is_swing_motion or self._is_elbow_pivot_swing_motion):
            # 揮桿剛收斂，桿尖停在緊貼母球原始位置之後一點點的地方（見
            # _awaiting_retreat_after_swing 欄位說明）。在回報「揮桿完成」
            # 之前先垂直上移 RESET_LIFT_CLEARANCE_M，避免 RESET 把母球
            # 瞬移回來時跟還停在原地的桿尖重疊。
            self._awaiting_retreat_after_swing = False
            self._is_swing_motion = False
            self._is_elbow_pivot_swing_motion = False
            current_position = np.array(self.get_end_effector_position())
            current_orientation = self._get_end_effector_world_orientation()
            retreat_position = current_position + np.array([0.0, 0.0, self.RESET_LIFT_CLEARANCE_M])
            self._activate_pose_target(
                retreat_position.tolist(), current_orientation.tolist(),
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            )
            return

        if self._awaiting_home_after_lift:
            self._awaiting_home_after_lift = False
            self._start_joint_space_motion(
                self._default_joint_positions, self._home_position
            )
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
        `_step_swing_motion()` 把腕部的線性 Jacobian 換算成桿尖的線性
        Jacobian 用。"""
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

    def move_to_home(self) -> None:
        if self._ur10e_mode:
            # 上一次 move_to_pose() 的最終逼近階段可能停用過動態障礙物，
            # RESET 回 HOME 這段大幅移動要重新啟用避障（重複呼叫是安全的
            # no-op）。
            self._ur10e_rmpflow_controller.enable_dynamic_obstacles()
            self._ur10e_active_controller = self._ur10e_rmpflow_controller
            self._ur10e_rmpflow_controller.move_to_home()
            return
        self._pending_waypoints = []
        self._awaiting_waypoints_after_joint_motion = False
        if self._default_joint_positions is None or self._home_position is None:
            # initialize() 之後、第一個 physics step 之前就被呼叫：
            # _capture_home_position_once() 還沒把 home 姿態擷取下來，記下來
            # 等擷取完成的同一個 callback 裡補做。
            self._pending_move_to_home = True
            return
        # 先垂直上移 RESET_LIFT_CLEARANCE_M，收斂後才切到關節空間回
        # home——關節空間插值不保證桿尖走直線，可能讓桿尖在回 home 的路上
        # 橫掃過桌面撞到 RESET 剛擺好的球。
        current_position = np.array(self.get_end_effector_position())
        current_orientation = self._get_end_effector_world_orientation()
        lift_position = current_position + np.array([0.0, 0.0, self.RESET_LIFT_CLEARANCE_M])
        self._awaiting_home_after_lift = True
        self._activate_pose_target(
            lift_position.tolist(), current_orientation.tolist(),
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        )

    def get_end_effector_position(self) -> list[float]:
        # UR10e 模式直接回傳 raw wrist_3_link 世界座標，不套用下面
        # _compute_tip_local_offset() 的偏移量——那是給 WAM7/UR3e 用
        # align_prim_to_target 掛接球桿的參考點設計的，跟 UR10e 透過
        # CueSlideJoint 掛接的方式無關，套用會對 AIM 收斂診斷加上一個
        # 無關的常數位移。
        if self._ur10e_mode:
            positions, _orientations = self._end_effector_rigid_prim.get_world_poses()
            return np.array(positions[0].list()).tolist()

        if self._tip_local_offset is None:
            self._tip_local_offset = self._compute_tip_local_offset()

        positions, orientations = self._end_effector_rigid_prim.get_world_poses()
        position = np.array(positions[0].list())
        orientation = np.array(orientations[0].list())
        tip_world_point = position + self._rotate_vector_by_quat(orientation, self._tip_local_offset)
        return tip_world_point.tolist()

    def get_end_effector_orientation(self) -> list[float]:
        return self._get_end_effector_world_orientation().tolist()

    def get_dof_positions_for_debug(self) -> list[float]:
        """僅供除錯用，不是 `ArticulationAPI` 正式介面：回傳目前所有關節
        角度，供 `billiard_digital_twin.py` 的 `BILLIARD_DEBUG_LOG_PATH`
        除錯 log 使用。"""
        if self._articulation is None:
            return []
        return np.asarray(self._articulation.get_dof_positions())[0].tolist()

    def is_motion_complete(self) -> bool:
        if self._ur10e_mode:
            # 退桿完成的那個 tick，_ur10e_active_controller 仍然是
            # cue_slide_controller（真正切到 rmpflow_controller 要等
            # _step_ur10e_motion() 下一次呼叫），不能只看它目前的狀態，
            # 否則呼叫端會在手臂目標從未真正送出前就判定整段動作完成。
            # 「安全中繼姿態→逼近緩衝點」「逼近緩衝點→最終姿態」這兩次
            # 交接同理。
            if self._ur10e_awaiting_arm_move_after_retract:
                return False
            if self._ur10e_awaiting_final_approach_after_staging:
                return False
            if self._ur10e_awaiting_final_short_leg_after_near_final:
                return False
            if self._ur10e_active_controller is None:
                return True
            return self._ur10e_active_controller.is_motion_complete()
        # `_on_tick`（驅動狀態機）跟 `_step_motion`（驅動實際換下一個
        # 子動作）是兩個各自獨立的 PHYSICS_POST_STEP callback，`_on_tick`
        # 註冊得早、每個 physics step 搶先執行，因此不能拿
        # `_is_current_target_converged()`（只看「當下這一小段」）當這裡
        # 的完成判定——否則某個中繼子動作剛好在這個 physics step 收斂時，
        # `_on_tick` 會搶先讀到「已收斂」，讓外部誤以為整個動作做完、
        # 狀態機提早跳下一個狀態，但 `_step_motion()` 根本還沒機會換到
        # 後面真正的目標。以下幾種情形只要還有排隊中的後續動作，就不算
        # 完成（見 docs/CHANGELOG.md 的除錯過程）：
        #   - _awaiting_waypoints_after_joint_motion：Phase 0 收斂後還要
        #     接 move_through_poses() 的 waypoints
        #   - _waypoint_index + 1 < len(_pending_waypoints)：目前不是
        #     最後一個 waypoint
        #   - _awaiting_swing_after_backswing／
        #     _awaiting_elbow_pivot_swing_after_backswing：後擺收斂後
        #     還要接真正的揮桿
        #   - _awaiting_home_after_lift：垂直上移收斂後還要接關節空間
        #     回 home
        #   - _awaiting_retreat_after_swing：揮桿收斂後還要接隨揮後的
        #     垂直撤離
        if self._motion_active and (
            self._awaiting_waypoints_after_joint_motion
            or self._waypoint_index + 1 < len(self._pending_waypoints)
            or self._awaiting_swing_after_backswing
            or self._awaiting_home_after_lift
            or self._awaiting_retreat_after_swing
            or self._awaiting_elbow_pivot_swing_after_backswing
        ):
            return False
        return self._is_current_target_converged()

    def did_last_motion_timeout(self) -> bool:
        if self._ur10e_mode:
            if self._ur10e_active_controller is None:
                return False
            return self._ur10e_active_controller.did_last_motion_timeout()
        return self._did_last_motion_timeout

    def _is_current_target_converged(self) -> bool:
        # move_swing() 的揮桿子階段用 self._swing_complete 判斷完成（由
        # _step_swing_motion() 依投影移動距離設定），必須在
        # self._target_position 早退檢查之前處理——揮桿階段
        # self._target_position 還留著後擺子階段的舊值。
        if self._is_swing_motion:
            return self._swing_complete

        # move_swing_elbow_pivot() 同理，用 self._elbow_pivot_complete
        # （由 quintic 的 T 是否跑完設定）判斷。
        if self._is_elbow_pivot_swing_motion:
            return self._elbow_pivot_complete

        # joint-space 動作的語意是「把關節開到指定角度」，末端世界位置是
        # 結果不是目標：_execute_aim() 每次擊球都會搬動基座，基座一搬走
        # 即使關節完全回到 _default_joint_positions，末端世界位置也回不到
        # 舊的 _home_position，位置檢查不能拿來當 joint-space 的收斂條件。
        if self._is_joint_space_motion:
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
