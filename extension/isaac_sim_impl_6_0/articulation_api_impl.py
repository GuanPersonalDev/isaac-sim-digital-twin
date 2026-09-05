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
    # move_to_home() 回 home 之前先垂直上移的安全距離——關節空間插值（PhysX
    # 自己算各關節的位置驅動路徑，不保證桿尖走直線）可能會讓桿尖在回到 home
    # 姿態的路上下降、橫掃過桌面，撞到 RESET 剛擺好的球（見 move_to_home()
    # 說明）。跟 cue_pose_calculator.compute_elevated_bridge_waypoints() 的
    # safe_altitude_margin 用同一個量級，足以清空庫邊高度。
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
        # move_to_home() 先垂直上移一段安全距離，收斂後才切到關節空間回 home——
        # 見 move_to_home() 與 RESET_LIFT_CLEARANCE_M 的說明。
        self._awaiting_home_after_lift: bool = False
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
        # 隨揮（swing_end_position）收斂後，桿尖停在緊貼母球原始位置之後
        # 一點點的地方（follow_through_distance 通常只有幾公分）。RESET 會
        # 把母球瞬移「回到」正是這個位置附近，若桿尖還停在那裡，球一擺回去
        # 兩者幾乎已經疊在一起，這時不管手臂接下來往哪個方向動都會碰到球
        # （實測：move_to_home() 前先垂直上移仍然在上移的第一步就撞到，因為
        # 起點本身就已經貼著球）。真正需要的是「隨揮一結束、球還沒被重新
        # 擺放之前，桿尖就先撤離」，所以這個垂直上移要接在揮桿本身完成後
        # （move_swing() 內部自動串接），不是接在 move_to_home() 前面——見
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

        # move_swing_elbow_pivot() 的狀態（UR3e 專用揮桿控制，見該方法
        # 說明）：跟 move_swing() 的關係跟 _is_swing_motion／
        # _awaiting_swing_after_backswing 對 _is_elbow_pivot_swing_motion／
        # _awaiting_elbow_pivot_swing_after_backswing 完全對應，只是揮桿
        # 子階段的控制策略不同（單一關節 quintic，不是全關節 LP 最佳化）。
        # 揮桿完成後複用既有的 _awaiting_retreat_after_swing 撤離機制。
        self._awaiting_elbow_pivot_swing_after_backswing: bool = False
        self._is_elbow_pivot_swing_motion: bool = False
        self._elbow_pivot_dof_index: int = 0
        self._elbow_pivot_contact_joint_positions: np.ndarray | None = None
        self._elbow_pivot_target_velocity: float = 0.0
        self._elbow_pivot_quintic: tuple[float, float, float, float] | None = None
        self._elbow_pivot_elapsed_steps: int = 0
        self._elbow_pivot_complete: bool = False

        # UR10e 專用狀態（見 UR10e 重新設計計畫決策 3/5，
        # extension/isaac_sim_impl_6_0/ur10e_rmpflow_controller.py／
        # ur10e_cue_slide_controller.py）：跟上面 WAM7/UR3e 的差動 IK／
        # elbow-pivot 狀態機完全獨立，_ur10e_mode 只在 initialize() 偵測到
        # dof_names 含 "CueSlideJoint" 時開啟，開啟後 move_to_pose()／
        # move_to_home()／is_motion_complete()／did_last_motion_timeout()
        # 這幾個共用方法會在最前面分流，不會執行到下面 WAM7/UR3e 的既有
        # 邏輯（刻意保持兩條路徑互不干擾，降低互相拖累出 regression 的風險）。
        self._ur10e_mode: bool = False
        self._ur10e_rmpflow_controller = None
        self._ur10e_cue_slide_controller = None
        self._ur10e_active_controller = None
        self._ur10e_step_callback_id: int | None = None
        # move_to_pose() 的「先退桿→移到安全中繼姿態→（避障開）平移到
        # 逼近緩衝點→（避障關）平移到最終姿態」四段式序列狀態（2026-09-05
        # 補充，見該方法說明）：_ur10e_awaiting_arm_move_after_retract=True
        # 期間 _ur10e_active_controller 指向 cue_slide_controller，退桿
        # 完成後先移到 _ur10e_pending_staging_target（安全中繼姿態，避障
        # 留在這段做），到位後切到 _ur10e_awaiting_final_approach_after_
        # staging 狀態，平移到 _ur10e_pending_near_final_target（逼近緩衝
        # 點，避障仍然開著），到位後再切到 _ur10e_awaiting_final_short_
        # leg_after_near_final 狀態，關掉母球避障、平移剩下一小段到
        # _ur10e_pending_arm_target（真正的最終姿態）。
        #
        # ⚠️ 2026-09-05 除錯記錄：原本只有「STAGING→最終姿態」兩段，最終
        # 逼近整段都停用母球避障（終點本來就緊貼母球，避障開著會卡死），
        # 但這代表整段（~1.45m）都沒有防護——實測踩過：桿身在抵達終點前
        # 的倒數第二個 waypoint（桿尖離球心僅 0.054m）擦到母球，把球撞出
        # 殘留速度，STRIKE 開始前球已經不在瞄準時的位置，命中率變成 0%。
        # 拆成兩段：先在避障開著的情況下逼近到只剩
        # _UR10E_FINAL_APPROACH_SAFE_MARGIN_M 的緩衝點，這段還有避障防護；
        # 只有最後這一小段才關避障直線逼近，大幅縮小「零防護」路徑的長度，
        # 且不影響已驗證能完全收斂的 STAGING 距離本身。
        self._ur10e_awaiting_arm_move_after_retract: bool = False
        self._ur10e_awaiting_final_approach_after_staging: bool = False
        self._ur10e_awaiting_final_short_leg_after_near_final: bool = False
        self._ur10e_pending_staging_target: tuple[list[float], list[float]] | None = None
        self._ur10e_pending_near_final_target: tuple[list[float], list[float]] | None = None
        self._ur10e_pending_arm_target: tuple[list[float], list[float]] | None = None
        # DEBUG_UR10E_AIM_PHASES 除錯用：分段計數兩階段 AIM 各自消耗的
        # step 數，用來判斷是 staging 那段吃光大部分預算、還是 final
        # approach 那段本身卡住（見 2026-09-05 obstacle 註冊調查）。
        self._ur10e_aim_phase_step_counter: int = 0
        self._ur10e_aim_final_approach_reported: bool = False
        # register_static_box_obstacle() 每次呼叫都要建立一個新的 USD prim
        # 路徑（FixedCuboid 不能共用同一個 prim_path 註冊兩個不同的障礙物），
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
        """UR10e 專屬初始化路徑，完全繞開上面 WAM7/UR3e 的差動 IK 設定
        （_load_dof_max_velocities／_resolve_end_effector_jacobian_index／
        _compute_tip_local_offset／_boost_wrist_gains_for_cue_stick_load／
        _capture_home_position_once 這一整套 home-capture 機制——決策 11
        明確排除，UR10e 用固定 HOME 關節角度，不是「USD 重新放進場景時的
        自然落點」）。

        ⚠️ 刻意不呼叫 _boost_wrist_gains_for_cue_stick_load()：那是幫
        WAM7/UR3e 的 wrist_1/wrist_3 關節在差動 IK 控制下補強增益用的，
        UR10e 完全交給 RMPflow 驅動，套用那組刻意設得極高的增益
        （stiffness=1e15）可能干擾 RMPflow 自己對各關節的 PD 追蹤動態——
        本次對話所有 UR10e 驗證（scripts/verify_ur10e_*.py／
        test_ur10e_actuator_swing_isolated.py）都是在完全沒有這個增益
        覆蓋的情況下跑的，套用會是沒驗證過的新變因。
        """
        from .ur10e_cue_slide_controller import Ur10eCueSlideController
        from .ur10e_rmpflow_controller import Ur10eRmpflowController

        self._ur10e_rmpflow_controller = Ur10eRmpflowController(
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
            # 退桿完成（或逾時，best-effort 繼續，不擋住整段動作）——切到
            # RMPflow 控制器先移到安全中繼姿態（見 move_to_pose() 2026-09-05
            # 補充），退桿階段自己的 did_last_motion_timeout 不往外傳（這個
            # 序列最終只回報「平移到最終姿態」那一段的完成/逾時狀態）。
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
            # 安全中繼姿態到位（或逾時，best-effort 繼續）——平移到逼近
            # 緩衝點，方向跟中繼姿態完全相同，只有位置沿同一個軸向逼近，
            # 不需要再解一次複雜的重新定向。母球避障這段仍然開著（見類別
            # 屬性區塊 2026-09-05 補充），只有再下一段才關閉。
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
            # 逼近緩衝點到位（或逾時，best-effort 繼續）——關掉母球避障，
            # 平移剩下一小段（_UR10E_FINAL_APPROACH_SAFE_MARGIN_M）到真正
            # 的最終姿態。這段終點本來就緊貼母球（AIM 的定義就是「桿尖
            # 對準球」），如果母球這時候還是啟用中的障礙物，等於同時要求
            # RMPflow「靠近」又「遠離」同一個目標，實測踩過：整段平移
            # 卡死，跑滿步數上限逾時。球檯（靜態）維持避障——手臂逼近球的
            # 最後一段不該撞到球檯，這個顧慮是分開的、需要繼續生效。
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
            self._ur10e_rmpflow_controller.disable_dynamic_obstacles()
            self._ur10e_aim_phase_step_counter = 0
            self._ur10e_aim_final_approach_reported = False
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

    def register_static_box_obstacle(self, center: list[float], size: list[float]) -> None:
        if not self._ur10e_mode:
            return
        # ⚠️ 2026-09-05 除錯記錄：一開始用 FixedCuboid——那是「靜態剛體＋
        # 真實 PhysX 碰撞」（見 isaacsim.core.api.objects.cuboid 的類別
        # 階層：FixedCuboid 繼承 VisualCuboid 並加上 RigidBodyAPI/
        # CollisionAPI），不是純粹給 RMPflow 內部避障邏輯參考用的幾何
        # 標記。實測踩過：這個障礙物箱體跟真正的球檯位置重疊，變成真的會
        # 撞的東西，CueStick 反覆跟它產生接觸事件。改用 VisualCuboid——
        # 純幾何+世界座標，沒有 RigidBodyAPI/CollisionAPI，不會參與真實
        # PhysX 碰撞反應，只提供 RMPflow 需要的形狀/位置資訊。
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
        # ⚠️ 2026-09-05 除錯記錄：一開始直接 DynamicSphere(prim_path=
        # 母球路徑) 想直接包一層現有的母球 prim，實測噴例外——
        # 「cannot be parsed as a Sphere object」。母球實際的 USD 結構
        # 不是頂層就是一個 UsdGeom.Sphere（真正的 Sphere geometry 在更深的
        # 子節點），DynamicSphere 的建構子對「包既有 prim」這個用法會嚴格
        # 檢查 prim type，包不了。改成建立一個全新、獨立的 DynamicSphere
        # 障礙物 proxy，每個 tick 從母球真正的 RigidPrim 讀取最新世界座標
        # 手動同步過去（見 Ur10eRmpflowController.add_dynamic_sphere_
        # obstacle()／_step_rmpflow() 的同步邏輯），不依賴「wrapper 直接
        # 讀到原始 prim」這個做不到的假設。
        self._ur10e_rmpflow_controller.add_dynamic_sphere_obstacle(prim_path, radius)

    def move_cue_slide_stroke(
        self, backswing_position: float, target_velocity: float
    ) -> None:
        self._ur10e_active_controller = self._ur10e_cue_slide_controller
        self._ur10e_cue_slide_controller.move_stroke(backswing_position, target_velocity)

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

    _WRIST_GAIN_BOOST_JOINT_NAME_SUBSTRINGS = ("wrist_1", "wrist_3")
    """UR3e 專用（見下方 docstring）——`dof_names` 裡符合這些子字串的關節
    才會被加強，WAM7 的關節命名（`wam_*`）完全不會命中，等同這個方法對
    WAM7 是 no-op，不需要另外用機器人類型分流。"""
    _WRIST_GAIN_STIFFNESS = 1e15
    _WRIST_GAIN_DAMPING = 1e5
    """沿用 Isaac Sim 官方 Gain Tuner extension 的「SET STIFF GAINS」慣例值
    （見 isaacsim.robot_setup.gain_tuner），這組數值是官方工具本身用來
    消除關節在負載下下垂/漂移的標準做法，不是這裡自己拍腦袋定的。"""
    _WRIST_GAIN_MAX_EFFORT_MULTIPLIER = 20.0

    def _boost_wrist_gains_for_cue_stick_load(self) -> None:
        """2026-09-02：真實 GUI 執行（`billiard_digital_twin.py` 換成
        UR3eRobot 之後）逐 tick log 顯示——`move_swing_elbow_pivot()` 進入
        「joint-space 移動到後擺姿態」這個子動作時，肘關節正常收斂，但
        `wrist_1`／`wrist_3` 兩個關節即使目標角度完全沒變（跟 AIM 階段
        收斂到的值相同），也會在肘關節做大幅度動態擺動的過程中被拖離目標
        （wrist_1 從 -0.6999 漂移到 -0.433、wrist_3 從 ~0 漂移到 0.072，
        之後卡住不動），導致這個 joint-space 子動作永遠收斂不了、1000 步
        後逾時，STRIKE 從此卡死在錯誤姿態、桿尖離母球 1.87m。

        UR3e 官方 USD 的關節 PD 增益／扭矩上限應該是針對「手臂自身負載」
        調的，沒有考慮到球桿透過 FixedJoint 剛性掛在腕部之後，1.35m 長的
        力臂在肘關節動態擺動時會對 wrist_1／wrist_3（UR3e 裡扭矩容許值
        較小的兩個關節）產生的額外反作用力矩——這是 PhysX 关节 drive 已知
        的限制類型（位置控制的合力＝stiffness×位置誤差＋damping×速度誤差，
        兩者都不夠大、或 max_effort 扭矩上限太低，都會讓關節在外部負載下
        收斂不到目標，見 NVIDIA 官方 Gain Tuner extension 文件同一類問題）。

        只挑 `wrist_1`／`wrist_3`（`_ELBOW_DOF_INDEX` 本身跟 `wrist_2` 在
        同一份 log 裡收斂正常，不需要跟著加強）：肘關節後續會切到 velocity
        模式（`switch_dof_control_mode('velocity')` 會自動把 stiffness
        歸零，這裡先調高也不影響 velocity 模式的揮桿行為），其餘關節維持
        原廠數值，改動範圍盡量小。用 `dof_names` 子字串比對（不是寫死
        index）自動只對命中的關節生效，對 WAM7（關節命名完全不同）是
        no-op，不需要另外用機器人類型分流。"""
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

    # ⚠️ 2026-09-05 除錯記錄：查證發現真正蹭到母球的是桿身（不是桿尖），
    # 原本試過把這裡的退桿距離從 DEFAULT_BACKSWING_DISTANCE_M（0.15m）
    # 加大到 CUE_STICK_GRIP_TO_TIP+安全邊際（1.45m），指望讓整根 1.35m
    # 長的桿身徹底撤出母球所在的軸向區間。**這個做法已實測證實會讓情況
    # 更糟並回退**：手臂從 RESET(HOME) 移動到 AIM 目標的過程中，退更遠
    # 的桿子拖著更長的尾巴（握把端變成拖在後面），跟著手臂一起掃過更大
    # 的空間，反而撞到地板（impulse=2.34）、撞到多顆原本沒事的球——比
    # 原本輕微蹭到一顆球（impulse=0.024）嚴重得多。也就是說「退桿完成後
    # 靜止不動時的安全距離」跟「手臂移動過程中桿子掃過的空間大小」是互相
    # 拉扯的兩個需求，不能只靠加大退桿距離單方面解決，兩者中間有取捨。
    #
    # 改回沿用 STRIKE 揮桿用的 DEFAULT_BACKSWING_DISTANCE_M，維持原本較
    # 保守（掃過範圍較小）的退桿距離，接受桿身蹭到球這個殘留風險——真正
    # 的修法應該是讓 RMPflow 的路徑規劃本身知道母球/球檯的存在（見
    # add_obstacle()/add_ground_plane()，目前 production 路徑完全沒有
    # 呼叫，是另一個獨立、範圍更大的問題），而不是在「桿子退多遠」這個
    # 參數上打轉。
    _UR10E_AIM_RETRACT_POSITION_M = -swing_trajectory_calculator.DEFAULT_BACKSWING_DISTANCE_M

    # ⚠️ 2026-09-05 除錯記錄：註冊母球/球檯為 RMPflow 障礙物
    # （register_static_box_obstacle()/register_dynamic_sphere_obstacle()）
    # 後，直接接觸問題解決了（CueStick-母球 impulse 從 0.024 降到 0），
    # 但單一長距離、大幅重新定向的 waypoint chain（從 HOME 直接規劃到
    # 貼近母球的最終 AIM 姿態）反而更容易在避障的複雜决策空間裡卡進
    # 錯誤的姿態分支（實測：方向誤差從 0.02rad 惡化到 0.86rad 等級，
    # wxyz 符號幾乎完全相反）。改成兩階段：
    # 1. 先移動到「安全中繼姿態」——方向跟最終 AIM 目標完全相同（避開
    #    奇異點的 roll 已經算好），位置沿桿軸方向往後退開
    #    _UR10E_AIM_STAGING_OFFSET_M——這段大幅移動＋避障留在方向還沒
    #    貼近球、犯錯本錢比較大的階段做，母球避障全程啟用。
    # 2. 從安全中繼姿態出發，只需要沿同一個軸向做一段直線平移到真正的
    #    最終位置，方向全程不變——這段動作 RMPflow 不需要再解一次複雜的
    #    6-DOF 重新定向，只是單純延同一個已知安全的方向逼近，穩定性遠
    #    高於「從 HOME 出發直接規劃到貼近球的姿態」這種大幅度＋高風險
    #    的單一移動。
    #
    # ⚠️ 2026-09-05 除錯記錄（第二輪）：這段最終逼近（第 2 步）的終點
    # 本來就緊貼母球，母球避障全程停用（見 move_to_pose() 下方的呼叫端
    # 說明），等於整段路徑都沒有防護。當時把 offset 設成
    # CUE_STICK_GRIP_TO_TIP+0.1（~1.45m）是為了在「安全中繼姿態」保證
    # 整根 1.35m 球桿都離球夠遠，但這個距離現在已經由 STAGING 階段主動
    # 的母球避障負責，不再需要靠這個 offset 本身撐住——offset 越大，
    # FINAL_APPROACH 這段「零防護」的直線逼近就越長，暴露在中途因
    # waypoint 收斂誤差（見 _MAX_WAYPOINT_STEP_M/_ORIENTATION_TOLERANCE_
    # RAD）被 1.35m 長桿身槓桿放大成擦碰的風險視窗也越大——實測踩過：
    # 桿尖在抵達終點前的第 18/19 段（距離終點僅一小段）擦到母球，把球
    # 撞出 0.1185m/s 殘留速度，STRIKE 開始前球已經不在瞄準時的位置，
    # 命中率變成 0%。
    #
    # ⚠️ 2026-09-05 除錯記錄（第三輪）：嘗試縮短這個距離想降低
    # FINAL_APPROACH 無防護直線的暴露長度，改成 0.3m 跟 0.6m 都讓 AIM
    # 整個收斂變差（0.3m：位置誤差 0.306m，比整段 FINAL_APPROACH 的
    # 移動距離本身還長；0.6m：結果數值幾乎跟 0.3m 一樣，對這個參數不
    # 敏感）——研判 STAGING 階段本身的避障規劃在終點離母球太近時會被
    # 過度擠壓，不是單純調小這個數字就能解決，需要更精細的方案（例如
    # 只在最後一小段才停用母球避障，而不是整段 FINAL_APPROACH 都停用）。
    # 目前先退回已驗證能完全收斂的原始值（1.45m，見上方 2026-09-05 第一輪
    # 記錄），中途擦碰問題留待後續用「分段停用避障」處理，不要犧牲已經
    # 驗證有效的 AIM 收斂去換一個還沒驗證足夠次數的縮短值。
    _UR10E_AIM_STAGING_OFFSET_M = CUE_STICK_GRIP_TO_TIP + 0.1

    # 2026-09-05 補充（第四輪，分段停用避障）：FINAL_APPROACH 拆成兩段——
    # 先在避障開著的情況下逼近到只剩這個緩衝距離的「逼近緩衝點」，最後
    # 才關避障直線逼近剩下的一小段。數值只需要大於退桿距離
    # DEFAULT_BACKSWING_DISTANCE_M（0.15m）留一點餘裕即可，不需要跟
    # _UR10E_AIM_STAGING_OFFSET_M 一樣大——那個是為了「大幅重新定向」
    # 保留犯錯空間，這裡只是單純直線平移的最後一小段。
    _UR10E_FINAL_APPROACH_SAFE_MARGIN_M = 0.2

    def move_to_pose(self, position: list[float], orientation: list[float], linear_velocity: list[float] = [0.0, 0.0, 0.0], angular_velocity: list[float] = [0.0, 0.0, 0.0]) -> None:
        if self._ur10e_mode:
            # linear_velocity/angular_velocity 沒有對應語意（RMPflow 是
            # 反應式收斂，不是 feed-forward 速度控制），UR10e 呼叫端
            # （Ur10eSwingStrategy）不會傳非零值，這裡忽略。
            #
            # 2026-09-04 補充：AIM 收斂後 CueSlideJoint 原本停在 q=0（桿尖
            # 貼著接觸點，見 Ur10eCueSlideController docstring），代表手臂
            # 這段移動／收尾差動 IK 修正全程都是在桿尖已經貼球的狀態下
            # 進行——只要修正路徑不是精準沿球桿軸，就有蹭到球的風險（實測
            # 踩過：STRIKE 開始前母球就已經有非零速度、STRIKE 出現兩次分開
            # 的碰撞事件，最終球速只有目標的 12%）。改成移動手臂前先讓
            # CueSlideJoint 退到後擺位置（見 _UR10E_AIM_RETRACT_POSITION_M），
            # 退到位才開始移動手臂——手臂定位全程桿尖都在安全距離外，STRIKE
            # 開始時桿子已經在後擺位置，move_cue_slide_stroke() 的退桿子
            # 階段會直接判定已收斂，接著才真正加速揮桿。實際序列由
            # _step_ur10e_motion() 的狀態機驅動（先跑 cue_slide_controller.
            # retract()，收斂後移到安全中繼姿態，再平移到真正的最終姿態，
            # 見類別 docstring 2026-09-05 補充），這裡只負責啟動＋算好中繼
            # 目標存起來。
            # 確保每次新的 move_to_pose() 呼叫都從「動態障礙物啟用」的狀態
            # 開始——上一次呼叫的最終逼近階段可能停用過（見下方
            # disable_dynamic_obstacles() 呼叫處的說明），重新啟用之後
            # 安全中繼姿態這段大幅移動才會真的避開母球。
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
        self._awaiting_retreat_after_swing = True
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

    def move_swing_elbow_pivot(
        self,
        backswing_joint_positions: list[float],
        backswing_target_end_effector_position: list[float],
        contact_joint_positions: list[float],
        elbow_dof_index: int,
        target_elbow_velocity: float,
    ) -> None:
        """UR3e 專用揮桿控制：跟 `move_swing()` 是平行、互斥的兩套揮桿策略，
        不是 `move_swing()` 的變形。

        `move_swing()` 對 WAM7 有效的原因是 WAM7 需要多個關節協調（線性
        規劃跨全部關節求解）才能達到目標桿尖速度。UR3e 已驗證不需要這樣：
        只讓 `elbow_dof_index` 這一個關節從 0 加速到 `target_elbow_
        velocity`，其餘關節角速度指令精確為 0，就足以達到目標桿尖速度
        （見 `scripts/test_ur3e_human_pose_swing_speed.py`／
        `scripts/test_elevated_bridge_ur3e_table.py` 的真實 quintic 軌跡
        執行驗證，分別達成 104.7%／96.1%），而且完全靜止的 base/肩關節
        更貼近人體揮桿手肘擺動的動作設計（見對話紀錄的人體化姿態討論）。
        對 UR3e 硬套 `move_swing()` 的全關節 LP 最佳化沒有驗證過會不會
        達到同樣的速度，不應該直接沿用。

        做法：先用跟 `move_to_joint_position()` 一樣的 joint-space 動作
        收斂到 `backswing_joint_positions`；收斂後（`_step_motion()` 的
        `_awaiting_elbow_pivot_swing_after_backswing` 分支）對
        `elbow_dof_index` 解一段 joint-space quintic polynomial（邊界
        條件：起點角度=收斂當下實測值、起點角速度/角加速度=0，終點角度=
        `contact_joint_positions[elbow_dof_index]`、終點角速度=
        `target_elbow_velocity`、終點角加速度=0），time-scaling 找最小
        可行 `T`（不超過關節馬達限速），逐 tick 下達 q̇(t)（其餘關節固定
        0），含 `_apply_velocity_targets_with_gravity_compensation()`
        重力補償。呼叫端只需要呼叫一次，`is_motion_complete()` 在後擺+
        揮桿全程持續回傳 False，語意跟 `move_swing()` 一致，完成後也複用
        同一個 `_awaiting_retreat_after_swing` 垂直撤離收尾機制。

        ⚠️ `backswing_joint_positions`／`contact_joint_positions` 是完整
        6 個關節角度（不是只有 elbow 那一個），呼叫端負責用
        `core/services/ur3e_placement_calculator.py` 算好整組關節目標
        （只有 `elbow_dof_index` 那個分量在揮桿階段會真的變動，其餘分量
        在後擺跟接觸姿態之間應該相同，這個方法不驗證這件事，呼叫端要自己
        保證兩組角度除了 elbow 之外一致，否則「其餘關節角速度固定為 0」
        會讓手臂卡在後擺姿態、到不了接觸姿態的其餘關節角）。

        ⚠️ 這個方法只驗證過「後擺→接觸」這段揮桿動作本身的速度與（在
        `scripts/test_elevated_bridge_ur3e_table.py` 的測試場景下）沒有
        撞到球檯，**沒有**驗證過「從手臂目前姿態安全接近到 `backswing_
        joint_positions`」這一段（WAM7 的高架橋案例為此設計了 B1/B2/C1/C2
        多階段 Cartesian 安全接近序列，見 `cue_pose_calculator.
        compute_elevated_bridge_waypoints()`，UR3e 目前沒有對應機制，直接
        用 joint-space 插值可能讓球桿沿途掃過球檯/球——呼叫端需要自行
        評估這個風險，或之後補上對應的安全接近機制）。
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
        q̇(T)=v1,q̈(T)=0`。回傳 `(c3,c4,c5)`（`c0=q0,c1=0,c2=0` 已知）。
        跟 `scripts/test_ur3e_human_pose_swing_speed.py` 同一個公式。

        ⚠️ `v(T)=v1` 這個邊界條件不會隨 `T` 縮放——如果 `v1` 本身就超過
        關節限制，加大 `T` 救不了（見該腳本同一個警告）。
        `move_swing_elbow_pivot()` 的呼叫端應該已經確認過
        `target_elbow_velocity` 在馬達限速內，這個函式本身不重複防呆。
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
        """下達關節角速度指令的同時，疊加重力補償力矩前饋（gravity
        compensation feedforward）。

        背景：`switch_dof_control_mode("velocity")`（`_activate_pose_target()`
        呼叫）只會把 drive 的 stiffness 歸零，damping 沿用 USD 內建值，
        **不會自動幫忙抗重力**——PhysX 的 velocity-mode PD 只針對「目標
        速度 vs 目前速度」的誤差出力，跟目前關節角度、重力力矩大小完全
        無關。如果 USD 裡 velocity-mode 的 damping 不夠大，速度目標=0 時
        關節就完全沒有力矩對抗重力，手臂會自由落體漂移。

        專案目前用的 WAM7 剛好沒踩到這個問題，但不是因為刻意處理過：
        URDF→USD 轉換工具幫每個關節寫死了一組偏高的 damping（見
        `assets/barrett_wam/wam7/payloads/Physics/physics.usda`，每個關節
        `drive:angular:physics:damping=174.53`），意外地夠抗重力，不是
        專案自己調過的值。這是一個結構性風險：換一支手臂（例如 UR3e，
        `scripts/test_ur3e_human_pose_swing_speed.py` 就真的在 isolated
        測試場景踩到過，達成率一度只有理論值的 10~55%，整支手臂在還沒
        開始揮桿前就先自由落體，量到的低速度是重力漂移的假象，不是
        姿態設計本身的問題）、或未來 WAM7 的 USD 被重新轉換出不同的
        damping 值，都可能重演同一個問題。

        做法：每個 physics tick 額外呼叫 `get_dof_gravity_compensation_forces()`
        讀出「維持目前姿態靜止所需要的重力補償力矩」，用 `set_dof_efforts()`
        疊加上去——這是標準機器人學做法，`set_dof_efforts()` 下達的是
        額外的 actuation force，跟 velocity drive 本身算出來的 PD 力矩在
        PhysX 內部是相加關係，不需要為此切到 `"effort"` 控制模式（那樣
        會把 stiffness/damping 一起歸零，反而失去 velocity-mode 原本的
        速度追蹤能力，見 `switch_dof_control_mode()` 的三種模式對照表）。
        `set_dof_efforts()` 官方文件明確標註「非常駐設定，必須每個
        physics tick 重新呼叫」，這正好符合這個函式已經在每個 tick 被
        呼叫的既有慣例，不需要額外的生命週期管理。

        ⚠️ 已知限制（Isaac Sim 6.0.0）：`get_dof_gravity_compensation_forces()`
        若場景用的是新版 Newton physics tensor backend，目前是官方尚未
        實作的 stub（回傳全 0），這個補償在該後端下會退化成無作用（不會
        報錯，只是補償力矩恆為 0，等同沒有這段程式碼）。這個專案目前走
        的是舊版 PhysX tensor backend（`isaacsim.core.experimental.prims.
        Articulation` 預設路徑），沒有這個限制；未來若切換 physics
        backend，需要重新確認這裡是否還有效。

        ⚠️ 已知範圍限制：這個補償只在「動作進行中」（`_step_swing_motion()`／
        `_step_motion()` 這兩個每 tick 被呼叫的方法）生效——`_stop_motion()`
        把驅動動作的 PHYSICS_POST_STEP callback 解除註冊之後，沒有任何
        程式碼會繼續每 tick 呼叫 `set_dof_efforts()`，動作完全停止、
        進入「閒置持穩」狀態時，重力補償會停止生效，回到只靠 damping
        硬撐的舊行為。這次的範圍是修正「動作進行中」（含揮桿）的重力
        漂移，不含「動作之間閒置等待」的持穩問題——後者如果之後也需要
        修，得另外設計一個不受單一動作生命週期綁定的常駐 callback。
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

        # 沿方向投影出「桿尖」目前已走的距離（不是腕部，也不是跟終點的
        # 歐氏距離）——姿態修正過程中允許有限度的側向漂移，用投影距離
        # 判斷「有沒有走完全程」比較符合這裡的完成語意（側向漂移不該讓
        # 完成判定卡住），而且真正需要碰到球的是桿尖，不是腕部。
        traveled = float(np.dot(current_tip_position - self._swing_start, self._swing_direction))
        if traveled >= self._swing_total_distance:
            self._swing_complete = True
            self._apply_velocity_targets_with_gravity_compensation(np.zeros(len(self._dof_limits)))
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

        if self._awaiting_elbow_pivot_swing_after_backswing:
            self._awaiting_elbow_pivot_swing_after_backswing = False
            self._is_elbow_pivot_swing_motion = True
            self._is_joint_space_motion = False
            # 用「真正收斂到的」肘關節角度當 quintic 起點（不是呼叫端傳入的
            # 後擺目標角度本身——跟 _awaiting_swing_after_backswing 用實測
            # 桿尖位置當下一階段起點同一個道理，JOINT_POSITION_TOLERANCE
            # 容許一點殘留誤差，用實測值比用目標值更準）。
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
            # 揮桿（_step_swing_motion() 驅動）本身剛收斂（_swing_complete=
            # True），桿尖停在緊貼母球原始位置之後一點點的地方
            # （follow_through_distance 通常只有幾公分）。見
            # _awaiting_retreat_after_swing 欄位說明：RESET 會把母球瞬移
            # 「回到」正是這個位置附近，若桿尖還停在那裡，球一擺回去兩者
            # 幾乎已經疊在一起。這裡在真正回報「揮桿完成」之前，先垂直上移
            # RESET_LIFT_CLEARANCE_M，用目前姿態當出發點、只改 Z、方向不變。
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
        `_step_swing_motion()` 把「腕部」的線性 Jacobian 換算成「桿尖」的
        線性 Jacobian 用（剛體速度合成 v_tip = v_wrist + ω × tip_offset，
        寫成矩陣形式即 v_tip = (Jv - skew(tip_offset) @ Jang) @ qdot）。"""
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

    def move_to_home(self) -> None:
        if self._ur10e_mode:
            # 上一次 move_to_pose() 的「最終逼近」階段可能停用過動態障礙物
            # （見該方法 2026-09-05 補充），RESET 回 HOME 這段大幅移動要
            # 重新啟用避障——重複呼叫 enable_dynamic_obstacles() 在已啟用
            # 的障礙物上是安全的 no-op，不需要額外判斷「上次是不是真的
            # 停用過」。
            self._ur10e_rmpflow_controller.enable_dynamic_obstacles()
            self._ur10e_active_controller = self._ur10e_rmpflow_controller
            self._ur10e_rmpflow_controller.move_to_home()
            return
        self._pending_waypoints = []
        self._awaiting_waypoints_after_joint_motion = False
        if self._default_joint_positions is None or self._home_position is None:
            # initialize() 之後、第一個 physics step 之前就被呼叫：
            # _capture_home_position_once() 還沒把 home 姿態擷取下來，這時直接
            # 往下跑會拿 None 當 joint-space 目標。改成記下來，等擷取完成的
            # 同一個 callback 裡補做（那時場景才真的可讀）。
            self._pending_move_to_home = True
            return
        # ⚠️ 2026-08-31：先垂直上移 RESET_LIFT_CLEARANCE_M，收斂後才切到關節
        # 空間回 home——關節空間插值（PhysX 自己算各關節的位置驅動路徑，不
        # 保證桿尖走直線）可能會讓桿尖在回到 home 姿態的路上下降、橫掃過
        # 桌面，撞到 RESET 剛擺好的球。用目前姿態當出發點，只改 Z、方向
        # 不變（純垂直平移，語意最單純，不需要額外算朝向）。
        current_position = np.array(self.get_end_effector_position())
        current_orientation = self._get_end_effector_world_orientation()
        lift_position = current_position + np.array([0.0, 0.0, self.RESET_LIFT_CLEARANCE_M])
        self._awaiting_home_after_lift = True
        self._activate_pose_target(
            lift_position.tolist(), current_orientation.tolist(),
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        )

    def get_end_effector_position(self) -> list[float]:
        # ⚠️ 2026-09-04 除錯記錄：UR10e 模式下這裡曾經誤用下面
        # _compute_tip_local_offset()（讀 end effector link 自身的 bounding
        # box，找「離原點最遠那一端」當工具尖端）——那是幫 WAM7/UR3e 找
        # 「球桿用 align_prim_to_target 掛接的實體參考點」設計的，UR10e 的
        # 球桿是透過 CueSlideJoint 掛在 wrist_3_link 之後，跟 wrist_3_link
        # 自己的 flange 幾何體完全無關；wrist_3_link 本身確實有真實幾何體
        # （不是空 bounding box），套用這個偏移量會加上一個跟 AIM 目標
        # （`cue_pose_calculator` 算出的是「wrist」位置，RMPflow／收尾差動
        # IK 追蹤的也是同一個 raw wrist_3_link 世界座標，兩者都沒有這個
        # 偏移量）無關的常數位移，讓 AIM 收斂診斷憑空多出約 5cm 的「假
        # 誤差」（實測：joint tracking gap 僅約 6e-5 rad，代表關節本身
        # 幾乎完美收斂到 RMPflow 目標，但套用這個偏移量量出來的末端位置
        # 卻跟目標差了 5.6cm——整段對不上）。UR10e 模式直接回傳 raw
        # wrist_3_link 世界座標，不套用這個只對 WAM7/UR3e 有意義的偏移量。
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
        """僅供除錯用，不是 `ArticulationAPI` 正式介面（不加進抽象 port）——
        回傳目前所有關節角度，供 `billiard_digital_twin.py` 的
        `BILLIARD_DEBUG_LOG_PATH` GUI 除錯 log 使用，讓 GUI 手動觀察之外
        也能拿到逐 tick 的關節角度數據。"""
        if self._articulation is None:
            return []
        return np.asarray(self._articulation.get_dof_positions())[0].tolist()

    def is_motion_complete(self) -> bool:
        if self._ur10e_mode:
            # ⚠️ 2026-09-04 除錯記錄：不能只看 _ur10e_active_controller 目前
            # 的狀態——退桿完成的那個 tick，_ur10e_active_controller 仍然是
            # cue_slide_controller（真正切到 rmpflow_controller、餵入手臂
            # 目標，要等 _step_ur10e_motion() 的下一次呼叫才會做），若這裡
            # 直接回傳 cue_slide_controller.is_motion_complete()==True，
            # 呼叫端的輪詢迴圈會在那個 tick 就判定「整段動作完成」提早跳出
            # ——手臂的移動指令永遠沒有機會被送出去（實測踩過：AIM 27 步
            # 就回報完成，量到的最終姿態幾乎還停在 HOME，因為 move_to_pose()
            # 從未真的被呼叫）。_ur10e_awaiting_arm_move_after_retract 為
            # True 這段期間，不管 cue_slide_controller 內部狀態如何，一律
            # 回報「尚未完成」，讓 _step_ur10e_motion() 有機會真正做完交接。
            if self._ur10e_awaiting_arm_move_after_retract:
                return False
            # 同一個道理套用在「安全中繼姿態→逼近緩衝點」跟「逼近緩衝點→
            # 最終姿態」這兩次交接：中繼姿態／緩衝點到位的那個 tick，
            # _ur10e_rmpflow_controller.is_motion_complete() 已經是 True，
            # 但下一段移動的指令要等 _step_ur10e_motion() 下一次呼叫才會
            # 真的送出去，這裡也要擋住提早判定完成。
            if self._ur10e_awaiting_final_approach_after_staging:
                return False
            if self._ur10e_awaiting_final_short_leg_after_near_final:
                return False
            if self._ur10e_active_controller is None:
                return True
            return self._ur10e_active_controller.is_motion_complete()
        # ⚠️ 2026-08-31（Demo 桌真實 GUI 執行才踩到，diagnose_move_swing.py
        # 這類單執行緒手動迴圈腳本測不出來）：這個方法是外部（ObservationBuilder
        # → 狀態機）唯一查詢「動作是否完成」的入口，跟 `_step_motion()` 內部
        # 用來判斷「目前這個子目標到了沒、該不該換下一個」的
        # `_is_current_target_converged()` 不能共用同一個判定——兩者語意不同：
        # 後者只看「當下這一小段」，前者必須看「一整串排隊中的子動作是否全部
        # 播完」。曾經誤把這個守門邏輯直接加進 `_is_current_target_converged()`
        # 本體，結果連 `_step_motion()` 自己判斷「這個 waypoint 到了、該換下
        # 一個」都被擋住，整條 waypoint 序列永遠卡在第一個——這裡改成只在
        # `is_motion_complete()` 這一層額外把關，不動 `_step_motion()` 依賴的
        # 內部判定。
        #
        # 根因：`_on_tick`（驅動狀態機）跟 `_step_motion`（驅動實際換下一個
        # 子動作）是兩個各自獨立註冊的 PHYSICS_POST_STEP callback，`_on_tick`
        # 註冊得早，每個 physics step 會搶先執行。PhysX 對 position-drive
        # joint 的實際求解發生在 callback 觸發之前，所以當某個「中繼子動作」
        # （move_through_poses() 的 Phase 0 joint-space 安全姿態、或
        # move_swing() 的後擺子階段）剛好在這個 physics step 收斂時，`_on_tick`
        # 會搶先讀到「目前子目標已收斂」，讓外部以為**整個**動作做完了、狀態機
        # 直接跳下一個狀態——但 `_step_motion()` 根本還沒機會把動作換到後面
        # 真正的目標（高架橋 waypoint／揮桿本身），手臂因此永遠卡在中繼姿態
        # （實測：球桿跟母球呈現不合理的角度，STRIKING 卻已經在執行）。這裡
        # 擋掉「還有排隊中的後續動作」這幾種情形，不管目前子目標本身有沒有
        # 收斂都不算完成：
        #   - _awaiting_waypoints_after_joint_motion：Phase 0 收斂後還要接
        #     move_through_poses() 的 waypoints
        #   - _waypoint_index + 1 < len(_pending_waypoints)：目前不是最後一個
        #     waypoint
        #   - _awaiting_swing_after_backswing：move_swing() 的後擺收斂後還要
        #     接真正的揮桿
        #   - _awaiting_home_after_lift：move_to_home() 的垂直上移子動作收斂
        #     後還要接關節空間回 home
        #   - _awaiting_retreat_after_swing：move_swing()／
        #     move_swing_elbow_pivot() 的揮桿本身收斂後還要接隨揮後的垂直
        #     撤離（見該欄位說明——揮桿完成時桿尖緊貼母球原始位置，必須在
        #     球被 RESET 瞬移回去之前先撤離）
        #   - _awaiting_elbow_pivot_swing_after_backswing：
        #     move_swing_elbow_pivot() 的後擺收斂後還要接真正的揮桿，跟
        #     _awaiting_swing_after_backswing 對 move_swing() 同一個道理
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

        # move_swing_elbow_pivot() 的揮桿子階段跟 move_swing() 同一個道理，
        # 用 self._elbow_pivot_complete（由 _step_elbow_pivot_swing_motion()
        # 依 quintic 的 T 是否已經跑完設定）判斷完成，不是位置/姿態容許值。
        if self._is_elbow_pivot_swing_motion:
            return self._elbow_pivot_complete

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