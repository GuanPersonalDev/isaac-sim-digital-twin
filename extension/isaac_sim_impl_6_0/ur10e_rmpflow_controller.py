import json
import logging
import os

import numpy as np

from core.services import ur10e_analytic_ik

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "rmpflow_config", "ur10e_cue", "rmpflow"
)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """四元數球面線性內插（wxyz），t in [0, 1]。q0/q1 反向（dot<0）時翻轉
    q1 走最短路徑，跟 Ur10eRmpflowController._is_current_waypoint_converged()
    的 abs(dot) 收斂判斷是同一個「正負號代表同一個旋轉」慣例。"""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def _quat_error(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    """q_error = q_target * q_current⁻¹（wxyz），q_error.w 為負時整體取反，
    走最短路徑。跟 ArticulationAPIImpl._quat_error() 同一套公式，複製一份
    避免這個模組反過來依賴 ArticulationAPIImpl 的私有方法。"""
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


def _orientation_error_to_angular_velocity(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    """四元數誤差轉角速度指令：q_error 的虛部乘 2 即為（小角度近似下的）
    修正角速度方向與大小，跟 ArticulationAPIImpl 差動 IK 用的公式一致。"""
    q_error = _quat_error(current_wxyz, target_wxyz)
    return 2.0 * q_error[1:]


def _load_rmp_flow(config_dir: str = _CONFIG_DIR):
    """依 config.json 的 relative_asset_paths 組出絕對路徑，建構 RmpFlow。
    不用官方 interface_config_loader.load_supported_motion_policy_config()
    ——那個函式只認官方內建的 motion_policy_configs/ 目錄結構，我們的設定檔
    放在專案自己的 assets/rmpflow_config/ur10e_cue/，手動組路徑即可。
    """
    from isaacsim.robot_motion.motion_generation import RmpFlow

    with open(os.path.join(config_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    relative_paths = config.pop("relative_asset_paths")
    for key, rel_path in relative_paths.items():
        config[key] = os.path.normpath(os.path.join(config_dir, rel_path))
    return RmpFlow(**config)


class Ur10eRmpflowController:
    """UR10e 手臂定位用的 RMPflow 包裝。

    只驅動 RMPflow 自己認得的 6 個手臂關節（見
    assets/rmpflow_config/ur10e_cue/rmpflow/ur10e_robot_description.yaml
    的 cspace），完全不知道也不觸碰 CueSlideJoint（第 7 個 DOF，掛在球桿
    跟 wrist_3_link 之間，見 TableRobotManager）——每次都先讀完整 7-DOF
    的當前關節位置當底，只覆寫 RMPflow 算出來的 6 個手臂關節分量，
    CueSlideJoint 維持原地不動，要移動滑軌關節得靠獨立的
    Ur10eCueSlideController（STRIKE 用，不經過 RMPflow）。

    刻意繞開 deprecated 的 ArticulationMotionPolicy/MotionPolicyController
    ——兩者的型別標註要求舊版 isaacsim.core.prims.SingleArticulation，跟
    專案現有的 isaacsim.core.experimental.prims.Articulation（batched/
    Warp array 架構）不相容，見 skills/isaac_sim_6_api_cache.md「RmpFlow」
    條目。直接用 RmpFlow 本身（純 numpy in/out）自己寫這層薄 adapter 對接。

    move_to_pose() 把大位移目標拆成一串中繼 waypoint 依序餵給 RMPflow
    （reactive RMP controller 對單一大位移容易卡在局部穩定點，是已知
    特性不是 bug）；最後一個 waypoint 收斂後若仍有殘留誤差，改用
    joint-space/差動 IK 收尾（見 _start_finishing_phase()）——推導過程見
    docs/CHANGELOG.md。
    """

    _MAX_WAYPOINT_STEP_M = 0.08
    # 中繼 waypoint 之間的方向變化上限：純位置距離決定 waypoint 數量時，
    # 在「位置距離小但需要旋轉的角度很大」的路段會讓單一 waypoint 內的
    # 方向變化量過大，RMPflow 卡住不收斂。30 度是保守值。
    _MAX_WAYPOINT_ROTATION_RAD = 0.5235987755982988  # 30 度
    _POSITION_TOLERANCE_M = 0.005
    _ORIENTATION_TOLERANCE_RAD = 0.02
    _MAX_STEPS_PER_WAYPOINT = 240

    # 最終姿態需要比中繼 waypoint 更嚴格的方向精度：球桿 1.35m
    # （CUE_STICK_GRIP_TO_TIP）的槓桿臂會把末端方向誤差等比放大，
    # 0.02rad 換算桿尖橫向誤差最壞可達 2.7cm，接近母球半徑（2.857cm）。
    # 只套用在「最終姿態是否已收斂」（_is_pose_converged()）與
    # joint-space 收尾（_step_joint_space_finish()），中繼 waypoint 跟
    # 差動 IK 保底路徑維持原本 0.02，避免拖慢每一段的收斂（見
    # docs/CHANGELOG.md 的實測數據）。
    _FINAL_ORIENTATION_TOLERANCE_RAD = 0.005

    # 收尾差動 IK 用的常數，數值沿用 ArticulationAPIImpl 既有 WAM7/UR3e
    # 差動 IK 的同名常數——同一個 codebase 已驗證過的收斂行為。
    _FINISH_POSITION_GAIN = 5.0
    _FINISH_ORIENTATION_GAIN = 5.0
    _FINISH_MAX_LINEAR_SPEED = 2.0  # m/s
    _FINISH_MAX_ANGULAR_SPEED = 3.0  # rad/s
    _FINISH_DLS_LAMBDA = 0.05
    _FINISH_MAX_STEPS = 240  # 跟 _MAX_STEPS_PER_WAYPOINT 同量級

    # 手動指定一組安全、離球檯足夠遠的固定 HOME 關節角度，不沿用「USD
    # 重新放進場景時自然落點當 HOME」的舊機制（WAM7/UR3e 的
    # _capture_home_position_once() 慣例）。沿用
    # assets/rmpflow_config/ur10e_cue/rmpflow/ur10e_robot_description.yaml
    # 的 default_q——Lula 官方替 UR10e 選定的 cspace 參考姿態（elbow
    # 彎起、遠離手臂完全打直的奇異點），不是隨便選的數字（見
    # docs/CHANGELOG.md 的 wrist_2 奇異點排查記錄）。
    _HOME_JOINT_POSITIONS = [-0.0, -1.2, 1.1, 0.0, 0.0, 0.0]

    def __init__(self, articulation, end_effector_prim_path: str) -> None:
        from isaacsim.core.experimental.prims import RigidPrim

        self._articulation = articulation
        self._end_effector_prim_path = end_effector_prim_path
        self._end_effector_rigid_prim = RigidPrim(paths=end_effector_prim_path)
        self._rmp_flow = _load_rmp_flow()

        # 拉高 solver iteration count：7-DOF 耦合鏈裡各關節 stiffness
        # 量級差異大（shoulder ~187k vs wrist/elbow boost 後 1e6）時，
        # 預設 iteration count 不足以讓 TGS 求解器正確收斂，會穩定收斂到
        # 一個數值上自洽但錯誤的解（跟本專案 WAM7 除錯
        # scripts/probe_first_case_residual_error.py 記錄過的同一類假影
        # 特徵）。128 是能跨過收斂門檻、同時不用 255 這種極端值的折衷值；
        # 代價是每個 waypoint 收斂變慢，已在呼叫端
        # （scripts/test_ur10e_table_flat.py 的 _MAX_STEPS_PER_AIM_ACTION）
        # 加大步數預算吸收，數據見 docs/CHANGELOG.md。
        self._articulation.set_solver_iteration_counts(128, 128)

        # 提前呼叫一次 switch_dof_control_mode()，用 USD 原始烘焙增益值
        # 把 Articulation 內部的「default gains」快取填好，避免這個快取
        # 在 _boost_gains_for_finish_once() 之後才第一次被填、記錄到 boost
        # 過的錯誤數值（見 docs/CHANGELOG.md 的 default gains 快取污染
        # 記錄）。
        self._articulation.switch_dof_control_mode("position")

        dof_names = list(self._articulation.dof_names)
        self._num_dofs = len(dof_names)
        self._active_joint_names = list(self._rmp_flow.get_active_joints())
        self._active_dof_indices = [dof_names.index(name) for name in self._active_joint_names]

        self._waypoints = []
        self._waypoint_index = 0
        self._steps_on_current_waypoint = 0
        self._motion_active = False
        self._did_last_motion_timeout = False
        self._last_active_position_targets: np.ndarray | None = None
        self._last_active_positions_before_step: np.ndarray | None = None
        # RMPflow 不管的 DOF（目前只有 CueSlideJoint）在整段手臂移動期間要
        # 維持的位置目標，在 move_to_pose() 當下擷取一次。不能每個 tick 拿
        # 「當下實際位置」當目標——那樣 PD 每個 tick 看到的誤差恆為 0，等於
        # 完全沒有回復力，球桿會被慣性/重力一路帶著漂移（實測 AIM 全程從
        # 退桿位置 -0.15 漂到 -0.11，安全間距少掉 4cm）。
        self._passive_dof_hold_targets: np.ndarray | None = None

        # 收尾差動 IK 狀態（見類別 docstring）。
        self._jac_link_index: int | None = None
        self._finishing_active = False
        self._finish_steps = 0
        self._finish_target_position: np.ndarray | None = None
        self._finish_target_orientation: np.ndarray | None = None
        # 收尾用 joint-space 精確目標的狀態（見 _start_finishing_phase()）。
        self._joint_finish_active = False
        self._joint_finish_target: np.ndarray | None = None
        # 呼叫端已知的關節角收尾目標，設了就跳過解析 IK 直接用（見
        # move_to_home()）。每次 move_to_pose() 都會清掉，不會外溢到下一段動作。
        self._joint_finish_override: np.ndarray | None = None
        self._gains_boosted_for_finish = False
        # set_robot_base_pose() 存下來的底座世界座標，供
        # _compute_analytic_finish_joint_target() 把世界座標目標轉成
        # ur10e_analytic_ik 需要的「機器人底座座標系」用——假設底座朝向
        # 固定是單位四元數（見 Ur10eSwingStrategy._BASE_ORIENTATION），
        # 只需要平移，不需要旋轉。
        self._base_position: np.ndarray | None = None
        # add_dynamic_sphere_obstacle() 註冊的 (來源 RigidPrim, 障礙物
        # proxy) 配對清單，_step_rmpflow() 每個 tick 都會用來同步位置。
        self._dynamic_obstacle_sources: list[tuple] = []
        # Lula 的 enable_obstacle()/disable_obstacle() 對「已經是目標
        # 狀態」會直接拋例外，不是安全的 no-op——自己追蹤目前狀態，見
        # enable_dynamic_obstacles()/disable_dynamic_obstacles()。新加入
        # 的障礙物預設是啟用狀態（Lula 的預設行為），初始值對應這個假設。
        self._dynamic_obstacles_enabled = True

    def set_robot_base_pose(self, base_position, base_orientation) -> None:
        """告訴 RMPflow 手臂底座目前在世界座標系的實際位姿。"""
        self._base_position = np.asarray(base_position, dtype=float)
        self._rmp_flow.set_robot_base_pose(
            self._base_position,
            np.asarray(base_orientation, dtype=float),
        )

    def move_to_pose(self, target_position, target_orientation) -> None:
        """開始一段移動：把位移拆成一串中繼 waypoint 依序餵給 RMPflow，
        方向也用 slerp 逐段內插（不是從第一段就鎖定最終方向）——若方向
        從第一段就直接設成最終目標，「位置移動量大＋方向本身需要旋轉」的
        目標（例如高架橋案例）容易讓 RMPflow 在離最終位置還很遠的中繼點
        就被迫同時追蹤最終方向，跟位置追蹤互相拉扯出局部穩定點。
        """
        # 擷取這段移動期間 RMPflow 不管的 DOF 要維持的位置（見
        # _passive_dof_hold_targets）。呼叫時機保證在退桿完成之後，擷取到的
        # 就是退桿位置本身。
        self._passive_dof_hold_targets = np.asarray(self._articulation.get_dof_positions())[0].copy()
        self._joint_finish_override = None

        current_position, current_orientation = self._end_effector_rigid_prim.get_world_poses()
        current_position = np.asarray(current_position[0], dtype=float)
        current_orientation = np.asarray(current_orientation[0], dtype=float)
        target_position = np.asarray(target_position, dtype=float)
        target_orientation = np.asarray(target_orientation, dtype=float)

        total_displacement = target_position - current_position
        distance = float(np.linalg.norm(total_displacement))
        position_segments = int(np.ceil(distance / self._MAX_WAYPOINT_STEP_M))

        dot = float(np.clip(np.abs(np.dot(current_orientation, target_orientation)), -1.0, 1.0))
        total_rotation = 2.0 * np.arccos(dot)
        rotation_segments = int(np.ceil(total_rotation / self._MAX_WAYPOINT_ROTATION_RAD))

        num_segments = max(1, position_segments, rotation_segments)

        self._waypoints = [
            (
                current_position + total_displacement * (i / num_segments),
                _slerp(current_orientation, target_orientation, i / num_segments),
            )
            for i in range(1, num_segments + 1)
        ]
        self._waypoint_index = 0
        self._motion_active = True
        self._did_last_motion_timeout = False
        self._activate_current_waypoint()

    def move_to_home(self) -> None:
        """回到 HOME 姿態，透過 RMPflow 導航（若有註冊障礙物會主動避開），
        不是直接 joint-space 瞬移——RESET/HOME 這段路徑一樣可能掃過球檯，
        需要跟 AIM 一樣的避障能力。

        用 RmpFlow.get_end_effector_pose(joint_positions) 算出 HOME 關節
        角度對應的世界座標末端位姿（已套用 set_robot_base_pose() 設定的
        目前底座位姿），不需要真的先把手臂瞬移過去才能量到。
        """
        home_position, home_orientation = self._compute_home_end_effector_pose()
        self.move_to_pose(home_position, home_orientation)
        # HOME 本來就是用關節角定義的（_HOME_JOINT_POSITIONS），收尾不需要
        # 也不應該再從 Cartesian 姿態反解一次 IK：HOME 的 wrist_2_joint=0 正好
        # 落在 UR 手腕奇異點上，解集合會退化，解析 IK 選不出可信的分支、差動
        # IK 在奇異點附近也收斂不了（實測 RESET 逾時：位置誤差 0.021m、方向
        # 誤差 0.086rad，還會讓 did_last_motion_timeout() 把整條 Demo 流程標成
        # 錯誤）。直接把已知的關節角當收尾目標，精確且完全繞開奇異點。
        # 必須在 move_to_pose() 之後設定——它會清掉這個覆寫值。
        self._joint_finish_override = np.array(self._HOME_JOINT_POSITIONS, dtype=float)

    def _compute_home_end_effector_pose(self):
        from scipy.spatial.transform import Rotation

        translation, rotation_matrix = self._rmp_flow.get_end_effector_pose(
            np.array(self._HOME_JOINT_POSITIONS, dtype=float)
        )
        quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.asarray(translation, dtype=float), quat_wxyz

    def is_motion_complete(self) -> bool:
        return not self._motion_active

    def did_last_motion_timeout(self) -> bool:
        return self._did_last_motion_timeout

    def step(self, frame_duration: float) -> None:
        if self._joint_finish_active:
            self._step_joint_space_finish()
            return

        if self._finishing_active:
            self._step_finish_ik()
            return

        if not self._motion_active:
            return

        self._step_rmpflow(frame_duration)
        self._steps_on_current_waypoint += 1

        converged = self._is_current_waypoint_converged()
        timed_out = self._steps_on_current_waypoint >= self._MAX_STEPS_PER_WAYPOINT
        if not (converged or timed_out):
            return

        if timed_out and not converged:
            self._did_last_motion_timeout = True

        if os.environ.get("DEBUG_UR10E_AIM_WAYPOINTS"):
            target_position, target_orientation = self._waypoints[self._waypoint_index]
            live_position, live_orientation = self._end_effector_rigid_prim.get_world_poses()
            live_position = np.asarray(live_position[0], dtype=float)
            live_orientation = np.asarray(live_orientation[0], dtype=float)
            position_error = float(np.linalg.norm(live_position - target_position))
            dot = float(np.clip(np.abs(np.dot(live_orientation, target_orientation)), -1.0, 1.0))
            orientation_error = 2.0 * np.arccos(dot)
            print(
                f"[aim_waypoints] waypoint={self._waypoint_index + 1}/{len(self._waypoints)} "
                f"steps={self._steps_on_current_waypoint} converged={converged} timed_out={timed_out} "
                f"position_error={position_error:.5f}m orientation_error={orientation_error:.5f}rad",
                flush=True,
            )

        self._waypoint_index += 1
        if self._waypoint_index >= len(self._waypoints):
            self._start_finishing_phase()
        else:
            self._activate_current_waypoint()

    def _start_finishing_phase(self) -> None:
        """最後一個 waypoint 收斂（或逾時）之後的收尾判斷：已經在容許
        誤差內就直接結束；否則切到 joint-space 或差動 IK 收尾，維持
        `_motion_active=True` 讓 step() 繼續被呼叫，只是改走
        `_step_joint_space_finish()`／`_step_finish_ik()` 分支。

        用 `self._waypoints[-1]` 搭配 `_is_pose_converged()`，不用
        `_is_current_waypoint_converged()`：呼叫端在呼叫這裡之前已把
        `_waypoint_index` 遞增到 `len(self._waypoints)`，是 out-of-range
        index（見 docs/CHANGELOG.md）。
        """
        target_position, target_orientation = self._waypoints[-1]
        target_position = np.asarray(target_position, dtype=float)
        target_orientation = np.asarray(target_orientation, dtype=float)

        live_position, live_orientation = self._end_effector_rigid_prim.get_world_poses()
        live_position = np.asarray(live_position[0], dtype=float)
        live_orientation = np.asarray(live_orientation[0], dtype=float)

        if self._is_pose_converged(live_position, live_orientation, target_position, target_orientation):
            self._motion_active = False
            return

        # 呼叫端已經知道要收到哪組關節角（HOME 是用關節角定義的）就直接用，
        # 不必從 Cartesian 姿態反解——見 move_to_home()。
        if self._joint_finish_override is not None:
            self._boost_gains_for_finish_once()
            self._joint_finish_target = self._joint_finish_override
            self._joint_finish_active = True
            self._finish_steps = 0
            return

        # 優先用 ur10e_analytic_ik 的 closed-form 解算出精確關節目標，
        # 直接 joint-space 收尾：PhysX position-mode drive 追蹤固定關節
        # 目標的精度已實測驗證（tracking gap ~6e-5 rad），不像 Cartesian
        # 差動 IK 那樣疊代逼近，不受運動學奇異點影響。只有 analytic IK
        # 真的找不到可達解（理論上不該發生，wrist 目標已是 RMPflow 自己
        # 收斂過的可達姿態）才退回原本的 Cartesian 差動 IK 當保底。
        joint_target = self._compute_analytic_finish_joint_target(target_position, target_orientation)
        if joint_target is not None:
            self._boost_gains_for_finish_once()
            self._joint_finish_target = joint_target
            self._joint_finish_active = True
            self._finish_steps = 0
            return

        logger.warning(
            "analytic IK 找不到可達解，退回 Cartesian 差動 IK 收尾（見 "
            "_compute_analytic_finish_joint_target() 說明，理論上不該發生）"
        )
        self._finish_target_position = target_position
        self._finish_target_orientation = target_orientation
        if self._jac_link_index is None:
            self._jac_link_index = self._resolve_jacobian_link_index()
        self._finishing_active = True
        self._finish_steps = 0

    def _compute_analytic_finish_joint_target(
        self, target_position: np.ndarray, target_orientation: np.ndarray
    ) -> np.ndarray | None:
        """把世界座標的 wrist 目標轉成 ur10e_analytic_ik 需要的機器人底座
        座標系，解出所有可達的 closed-form 關節解，回傳跟目前實際關節
        角度差距（wrap 到 ±π 後取最大分量）最小的一組——這組解在關節空間
        裡離目前姿態最近，最貼近 RMPflow 自己會收斂到的那個分支，走
        joint-space 直接過去只是「走完最後一小段」。`_base_position` 還
        沒設定（理論上不該發生，AIM 呼叫前一定會先 set_robot_base_pose()）
        或 analytic IK 找不到可達解時回傳 None，呼叫端負責退回差動 IK。
        """
        if self._base_position is None:
            return None

        position_in_base = target_position - self._base_position
        rotation_matrix = ur10e_analytic_ik.quat_wxyz_to_rotation_matrix(target_orientation)
        dh_position, dh_rotation = ur10e_analytic_ik.isaac_to_dh_frame(position_in_base, rotation_matrix)
        solutions = ur10e_analytic_ik.inverse_kinematics(dh_position, dh_rotation)
        if not solutions:
            return None

        current_positions = np.asarray(self._articulation.get_dof_positions())[0]
        current_active_positions = current_positions[self._active_dof_indices]

        def _wrapped_diff(solution: np.ndarray) -> np.ndarray:
            return np.mod(solution - current_active_positions + np.pi, 2.0 * np.pi) - np.pi

        best_solution = min(solutions, key=lambda s: float(np.max(np.abs(_wrapped_diff(s)))))
        best_solution_delta = float(np.max(np.abs(_wrapped_diff(best_solution))))

        # 靠近手腕奇異點（wrist_2=0，見 _HOME_JOINT_POSITIONS 說明）時，
        # closed-form IK 的解集合會退化成較少組解，這時「挑離目前姿態
        # 最近的分支」不可信（見 docs/CHANGELOG.md 的排查記錄）。RMPflow
        # 的 waypoint chain 在呼叫這裡之前理論上已把手臂帶到跟目標很接近
        # 的姿態，合理的解不該跟目前姿態差距過大——加一個寬鬆的合理性
        # 上限，超過就視為解集合不可信，回傳 None 讓呼叫端退回差動 IK
        # （DLS 疊代逼近即使在奇異點附近也只會產生平滑的小修正，不會整個
        # 跳到別的姿態）。
        _MAX_REASONABLE_FINISH_DELTA_RAD = 0.5
        if best_solution_delta > _MAX_REASONABLE_FINISH_DELTA_RAD:
            if os.environ.get("DEBUG_UR10E_FINISH_IK"):
                print(
                    f"[analytic_finish] 拒絕不合理的解：best_solution_delta={best_solution_delta:.5f}rad "
                    f"超過門檻 {_MAX_REASONABLE_FINISH_DELTA_RAD}rad（懷疑目前姿態靠近奇異點，"
                    f"n_solutions={len(solutions)} 已退化），回傳 None 退回差動 IK",
                    flush=True,
                )
            return None

        if os.environ.get("DEBUG_UR10E_FINISH_IK"):
            check_position, check_rotation = ur10e_analytic_ik.forward_kinematics(best_solution)
            check_position_isaac, check_rotation_isaac = ur10e_analytic_ik.dh_to_isaac_frame(
                check_position, check_rotation
            )
            position_check_error = float(np.linalg.norm(check_position_isaac - position_in_base))
            rotation_check_error = float(np.max(np.abs(check_rotation_isaac - rotation_matrix)))
            print(
                f"[analytic_finish] n_solutions={len(solutions)} best_solution={best_solution.tolist()} "
                f"FK驗證：position_check_error={position_check_error:.6f}m rotation_check_error={rotation_check_error:.6f}",
                flush=True,
            )

        # UR 關節範圍是 ±2π（連續旋轉），analytic IK 解出來的原始數值
        # （值域約 [-π,π]）可能跟目前實際關節角相差一整圈但「等效角度」
        # 很近；PhysX 的 position-mode drive 不會自動抄近路，只照給定的
        # 原始數值直接追。把每個關節分量平移成「離目前實際角度最近的
        # 等效角度」再回傳，讓 drive 走最短路徑（見 docs/CHANGELOG.md）。
        return current_active_positions + _wrapped_diff(best_solution)

    # 逐關節分別指定 (stiffness, damping, max_effort_multiplier)，不共用
    # 同一組數值：elbow 扛的下游慣量遠大於 wrist_1/wrist_3，同一組
    # damping 對 elbow 相對不足，收尾快速修正時會產生欠阻尼震盪並波及
    # 球桿撞到母球，因此把 elbow 的 damping 拉高到 5 倍（阻尼比
    # ζ=damping/(2*sqrt(stiffness*inertia))，慣量更大的關節需要更高
    # damping 才能拉回接近臨界阻尼，這是實測後的第一輪調整值）。數值來源
    # 與排查過程見 docs/CHANGELOG.md。
    _FINISH_GAIN_OVERRIDES = {
        "wrist_1": (1e6, 1e4, 20.0),
        "wrist_3": (1e6, 1e4, 20.0),
        "elbow": (1e6, 5e4, 20.0),
        "wrist_2": (1e6, 1e4, 20.0),
    }

    def _boost_gains_for_finish_once(self) -> None:
        """joint-space 收尾用的增益覆寫（沿用 ArticulationAPIImpl.
        _boost_wrist_gains_for_cue_stick_load() 已在 UR3e 驗證過的同一組
        數值），只在真的要進入 joint-space 收尾時才呼叫一次（見
        `_gains_boosted_for_finish`）。刻意不放進 `__init__()` 對整個 AIM
        過程常駐生效——RMPflow 自己的 waypoint chain 階段
        （`_step_rmpflow()`）用預設增益就能正常收斂，每 tick 重新給貼近
        目前值的新目標，等於變相用位置追蹤模擬速度追蹤，沒有這個穩態
        下垂問題，套用高增益是沒必要的新變因。排查過程見
        docs/CHANGELOG.md。
        """
        if self._gains_boosted_for_finish:
            return
        self._gains_boosted_for_finish = True

        dof_names = list(self._articulation.dof_names)
        target_overrides = {}
        for i, name in enumerate(dof_names):
            for sub, override in self._FINISH_GAIN_OVERRIDES.items():
                if sub in name.lower():
                    target_overrides[i] = override
                    break
        if not target_overrides:
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

        for idx, (stiffness, damping, max_effort_multiplier) in target_overrides.items():
            stiffnesses[idx] = stiffness
            dampings[idx] = damping
            max_efforts[idx] = max_efforts[idx] * max_effort_multiplier

        logger.info(
            "ur10e joint-space finish gain boost: joint indices=%s new_max_efforts=%s",
            list(target_overrides.keys()), [max_efforts[i] for i in target_overrides],
        )
        # update_default_gains=False：避免這個 boost 順便永久覆寫
        # Articulation 內部快取的「default gains」，污染稍後差動 IK 收尾
        # 切到 velocity 模式時引用的阻尼值（見 docs/CHANGELOG.md 的
        # default gains 快取污染記錄）。
        self._articulation.set_dof_gains(
            stiffnesses[None, :], dampings[None, :], update_default_gains=False
        )
        self._articulation.set_dof_max_efforts(max_efforts[None, :])

    def _step_joint_space_finish(self) -> None:
        """把 `_compute_analytic_finish_joint_target()` 算出的精確關節角
        當 joint-space 目標，交給 PhysX 關節驅動器自己插值到位——跟
        `ArticulationAPIImpl._start_joint_space_motion()` 同一個精神：
        起點離目標很近，不需要再跑一次差動 IK。收斂判定直接沿用弧度制
        `_FINAL_ORIENTATION_TOLERANCE_RAD` 當關節角度容許值。"""
        full_position_targets = self._passive_dof_hold_targets.copy()
        full_position_targets[self._active_dof_indices] = self._joint_finish_target
        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(full_position_targets[None, :])

        if os.environ.get("DEBUG_UR10E_FINISH_IK") and self._finish_steps == 0:
            stiffnesses, dampings = self._articulation.get_dof_gains()
            stiffnesses = np.asarray(stiffnesses.numpy() if hasattr(stiffnesses, "numpy") else stiffnesses, dtype=float)
            dampings = np.asarray(dampings.numpy() if hasattr(dampings, "numpy") else dampings, dtype=float)
            max_efforts = self._articulation.get_dof_max_efforts()
            max_efforts = np.asarray(max_efforts.numpy() if hasattr(max_efforts, "numpy") else max_efforts, dtype=float)
            print(
                f"[joint_space_finish DIAG] 剛套用 position target 之後讀回的 gains："
                f"stiffness={stiffnesses.tolist()} damping={dampings.tolist()} max_effort={max_efforts.tolist()}",
                flush=True,
            )
        # 疊加重力補償力矩前饋（跟 ArticulationAPIImpl.
        # _apply_velocity_targets_with_gravity_compensation() 同一個
        # 理由）：球桿透過 CueSlideJoint 掛在 wrist_3_link 之後的槓桿臂
        # 重力力矩，會讓 position-mode 穩定卡在「stiffness×殘留誤差＝
        # 重力力矩」的平衡點，需要重力補償讓 stiffness 項只修正真正的
        # 追蹤誤差。
        gravity_compensation_forces = self._articulation.get_dof_gravity_compensation_forces()
        self._articulation.set_dof_efforts(gravity_compensation_forces)

        self._finish_steps += 1
        current_positions = np.asarray(self._articulation.get_dof_positions())[0]
        current_active_positions = current_positions[self._active_dof_indices]
        joint_error = float(np.max(np.abs(current_active_positions - self._joint_finish_target)))
        converged = joint_error <= self._FINAL_ORIENTATION_TOLERANCE_RAD
        timed_out = self._finish_steps >= self._FINISH_MAX_STEPS

        if os.environ.get("DEBUG_UR10E_FINISH_IK"):
            print(
                f"[joint_space_finish] step={self._finish_steps} joint_error={joint_error:.5f}rad "
                f"current={current_active_positions.tolist()} target={self._joint_finish_target.tolist()}",
                flush=True,
            )

        if not (converged or timed_out):
            return

        if not converged:
            per_joint_error = np.abs(current_active_positions - self._joint_finish_target)
            per_joint_breakdown = ", ".join(
                f"{name}={error:.5f}rad" for name, error in zip(self._active_joint_names, per_joint_error)
            )
            # 逐關節分解方便定位是哪個關節拖慢收斂（見 _FINISH_GAIN_OVERRIDES
            # 說明：不同關節負載差異大，不能片面猜測要對哪個關節加處理）。
            logger.warning(
                "joint_space_finish 逾時未收斂（%d 步）：關節角最大誤差=%.5frad（容許%.5frad）"
                "，逐關節分解：%s",
                self._finish_steps, joint_error, self._FINAL_ORIENTATION_TOLERANCE_RAD, per_joint_breakdown,
            )
        self._did_last_motion_timeout = not converged
        self._joint_finish_active = False
        self._motion_active = False

    def _resolve_jacobian_link_index(self) -> int:
        """跟 ArticulationAPIImpl._resolve_end_effector_jacobian_index()
        同一套邏輯（fixed-base articulation 的 Jacobian 不含 base link），
        複製一份避免這個模組反過來依賴 ArticulationAPIImpl 的私有方法。"""
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
        if jacobians.shape[0] == len(link_names) - 1:
            return link_index - 1
        if jacobians.shape[0] == len(link_names):
            return link_index
        raise RuntimeError(
            f"Jacobian link 數 {jacobians.shape[0]} 與 link 名稱數 {len(link_names)} 對不上，"
            f"無法安全對應 {end_effector_link_name}"
        )

    def _step_finish_ik(self) -> None:
        """DLS 差動 IK 收尾一個 physics tick。只驅動 RMPflow 認得的 6 個
        手臂關節（`_active_dof_indices`），CueSlideJoint 的速度目標固定為
        0（原地保持，跟 `_step_rmpflow()` 對這個 DOF 的處理精神一致）。
        沿用 `_apply_velocity_targets_with_gravity_compensation()` 同一個
        重力補償理由：velocity-mode PD 不會自動抗重力。"""
        live_position, live_orientation = self._end_effector_rigid_prim.get_world_poses()
        live_position = np.asarray(live_position[0], dtype=float)
        live_orientation = np.asarray(live_orientation[0], dtype=float)

        position_error, orientation_error = self._pose_error_components(
            live_position, live_orientation, self._finish_target_position, self._finish_target_orientation
        )
        position_ok = position_error <= self._POSITION_TOLERANCE_M
        orientation_ok = orientation_error <= self._ORIENTATION_TOLERANCE_RAD
        converged = position_ok and orientation_ok

        if os.environ.get("DEBUG_UR10E_FINISH_IK"):
            print(
                f"[finish_ik] step={self._finish_steps} "
                f"position_error={position_error:.5f}m(ok={position_ok}) "
                f"orientation_error={orientation_error:.5f}rad(ok={orientation_ok})",
                flush=True,
            )

        self._finish_steps += 1
        if converged or self._finish_steps >= self._FINISH_MAX_STEPS:
            if not converged:
                # 明確記錄是位置、方向、還是兩者都沒到容許值內——方向誤差
                # 沒收斂會被 1.35m 長的球桿放大成數公分等級的桿尖偏移，
                # 只留一個布林值看不出這個區別（見 _pose_error_components()）。
                logger.warning(
                    "finish_ik 逾時未收斂（%d 步）：position_error=%.5fm（容許%.5fm，%s）"
                    " orientation_error=%.5frad（容許%.5frad，%s）",
                    self._finish_steps, position_error, self._POSITION_TOLERANCE_M,
                    "OK" if position_ok else "未達標",
                    orientation_error, self._ORIENTATION_TOLERANCE_RAD,
                    "OK" if orientation_ok else "未達標",
                )
            # 收斂/逾時當下先明確下達一次全零速度指令止住殘留漂移——
            # velocity-mode drive 會持續套用上一次的非零角速度指令直到
            # 有新指令覆寫，不歸零會讓手臂在收斂後繼續照原速度漂移（比照
            # ArticulationAPIImpl._stop_motion() 的做法）。
            self._articulation.set_dof_velocity_targets(np.zeros((1, self._num_dofs)))
            self._did_last_motion_timeout = not converged
            self._finishing_active = False
            self._motion_active = False
            return

        position_error = self._finish_target_position - live_position
        linear_velocity = np.clip(
            self._FINISH_POSITION_GAIN * position_error,
            -self._FINISH_MAX_LINEAR_SPEED, self._FINISH_MAX_LINEAR_SPEED,
        )
        angular_velocity = self._FINISH_ORIENTATION_GAIN * _orientation_error_to_angular_velocity(
            live_orientation, self._finish_target_orientation
        )
        angular_velocity = np.clip(
            angular_velocity, -self._FINISH_MAX_ANGULAR_SPEED, self._FINISH_MAX_ANGULAR_SPEED
        )
        twist = np.concatenate([linear_velocity, angular_velocity])

        jacobians = np.asarray(self._articulation.get_jacobian_matrices().numpy())[0]
        jacobian_full = jacobians[self._jac_link_index]
        jacobian_active = jacobian_full[:, self._active_dof_indices]
        jjt = jacobian_active @ jacobian_active.T + (self._FINISH_DLS_LAMBDA ** 2) * np.eye(6)
        qdot_active = jacobian_active.T @ np.linalg.solve(jjt, twist)

        # 印出 Jacobian 奇異值當奇異點的直接證據：最小奇異值遠小於
        # _FINISH_DLS_LAMBDA 代表 DLS 在某個方向上把修正量壓到幾乎為零。
        if os.environ.get("DEBUG_UR10E_FINISH_IK") and self._finish_steps % 20 == 0:
            singular_values = np.linalg.svd(jacobian_active, compute_uv=False)
            print(
                f"[finish_ik SVD] step={self._finish_steps} singular_values={np.round(singular_values, 5).tolist()} "
                f"twist={np.round(twist, 5).tolist()} qdot_active={np.round(qdot_active, 5).tolist()}",
                flush=True,
            )

        full_velocity_targets = np.zeros(jacobian_full.shape[1])
        full_velocity_targets[self._active_dof_indices] = qdot_active

        # 只把 RMPflow 管的 6 個手臂關節切到 velocity 模式：不限定
        # dof_indices 會連 CueSlideJoint 的 stiffness 也一起歸零，球桿在收尾
        # 期間失去位置保持、被慣性帶著滑動（跟 Ur10eCueSlideController.
        # _step_strike() 修過的同一類問題）。
        self._articulation.switch_dof_control_mode("velocity", dof_indices=self._active_dof_indices)
        self._articulation.set_dof_velocity_targets(full_velocity_targets[None, :])
        gravity_compensation_forces = self._articulation.get_dof_gravity_compensation_forces()
        self._articulation.set_dof_efforts(gravity_compensation_forces)

    @staticmethod
    def _pose_error_components(
        live_position: np.ndarray, live_orientation: np.ndarray,
        target_position: np.ndarray, target_orientation: np.ndarray,
    ) -> tuple[float, float]:
        """回傳 (position_error_m, orientation_error_rad)，拆成兩個獨立
        分量供 `_is_pose_converged()`／`_step_finish_ik()` 共用，也讓
        逾時當下能明確記錄是哪一項沒收斂（單看位置誤差會漏掉被 1.35m
        長球桿放大的方向誤差）。"""
        position_error = float(np.linalg.norm(live_position - target_position))
        dot = float(np.clip(np.abs(np.dot(live_orientation, target_orientation)), -1.0, 1.0))
        orientation_error = 2.0 * np.arccos(dot)
        return position_error, orientation_error

    def _is_pose_converged(
        self, live_position: np.ndarray, live_orientation: np.ndarray,
        target_position: np.ndarray, target_orientation: np.ndarray,
    ) -> bool:
        position_error, orientation_error = self._pose_error_components(
            live_position, live_orientation, target_position, target_orientation
        )
        return (
            position_error <= self._POSITION_TOLERANCE_M
            and orientation_error <= self._FINAL_ORIENTATION_TOLERANCE_RAD
        )

    def _activate_current_waypoint(self) -> None:
        position, orientation = self._waypoints[self._waypoint_index]
        self._set_end_effector_target(position, orientation)
        self._steps_on_current_waypoint = 0

    def _is_current_waypoint_converged(self) -> bool:
        target_position, target_orientation = self._waypoints[self._waypoint_index]
        live_position, live_orientation = self._end_effector_rigid_prim.get_world_poses()
        live_position = np.asarray(live_position[0])
        live_orientation = np.asarray(live_orientation[0])

        position_error = float(np.linalg.norm(live_position - target_position))
        if position_error > self._POSITION_TOLERANCE_M:
            return False

        dot = float(np.clip(np.abs(np.dot(live_orientation, target_orientation)), -1.0, 1.0))
        orientation_error = 2.0 * np.arccos(dot)
        return orientation_error <= self._ORIENTATION_TOLERANCE_RAD

    def _set_end_effector_target(self, target_position, target_orientation) -> None:
        self._rmp_flow.set_end_effector_target(
            np.asarray(target_position, dtype=float),
            np.asarray(target_orientation, dtype=float) if target_orientation is not None else None,
        )

    def update_world(self) -> None:
        self._rmp_flow.update_world()

    def add_ground_plane(self, ground_plane) -> None:
        self._rmp_flow.add_ground_plane(ground_plane)

    def add_obstacle(self, obstacle, static: bool = True) -> None:
        self._rmp_flow.add_obstacle(obstacle, static=static)

    def disable_dynamic_obstacles(self) -> None:
        """暫時停用所有動態障礙物（目前只有追蹤母球的 proxy，見
        `add_dynamic_sphere_obstacle()`）的避障效果——兩階段 AIM 的「最後
        平移到最終姿態」這一段，目的地本來就緊貼在母球旁邊，這時候還讓
        RMPflow 主動避開母球，等於同時要求「靠近」跟「遠離」同一個目標
        （見 docs/CHANGELOG.md 的排查記錄）。球檯（靜態障礙物）不受影響。
        呼叫端負責在安全時機（下一次 AIM 開始前）呼叫
        `enable_dynamic_obstacles()` 重新啟用。

        Lula 的 `enable_obstacle()`/`disable_obstacle()` 對「已經是目標
        狀態」的障礙物會直接拋例外，不是安全的 no-op，必須自己追蹤目前
        狀態（見 `_dynamic_obstacles_enabled`）。"""
        if not self._dynamic_obstacles_enabled:
            return
        self._dynamic_obstacles_enabled = False
        for _source_rigid_prim, obstacle in self._dynamic_obstacle_sources:
            self._rmp_flow.disable_obstacle(obstacle)

    def enable_dynamic_obstacles(self) -> None:
        if self._dynamic_obstacles_enabled:
            return
        self._dynamic_obstacles_enabled = True
        for _source_rigid_prim, obstacle in self._dynamic_obstacle_sources:
            self._rmp_flow.enable_obstacle(obstacle)

    def add_dynamic_sphere_obstacle(self, prim_path: str, radius: float) -> None:
        """建立一個獨立的 VisualSphere 障礙物 proxy，每個 physics tick
        （見 `_step_rmpflow()`）從 `prim_path` 對應的真實 RigidPrim 讀取
        最新世界座標同步過去，讓 RMPflow 看到的障礙物位置跟蹤真實物體
        （例如母球）目前所在的位置，不是註冊當下的固定快照。

        用 VisualSphere（純幾何，沒有 RigidBodyAPI/CollisionAPI）而不是
        DynamicSphere：後者是真實剛體，會跟完全重疊的真正母球互撞產生
        物理作用力；也不能直接包既有母球 prim（母球的 Sphere geometry
        不在頂層，DynamicSphere 建構子檢查 prim type 會直接噴例外），
        因此建一個全新獨立的障礙物 prim，見 docs/CHANGELOG.md。
        """
        from isaacsim.core.api.objects import VisualSphere
        from isaacsim.core.experimental.prims import RigidPrim as _RigidPrim

        source_rigid_prim = _RigidPrim(paths=prim_path)
        position, _orientation = source_rigid_prim.get_world_poses()
        obstacle_path = f"/World/_RmpflowDynamicObstacle_{len(self._dynamic_obstacle_sources)}"
        obstacle = VisualSphere(
            prim_path=obstacle_path,
            position=np.asarray(position[0], dtype=float),
            radius=radius,
            visible=False,
        )
        self._dynamic_obstacle_sources.append((source_rigid_prim, obstacle))
        self.add_obstacle(obstacle, static=False)

    def _sync_dynamic_obstacles(self) -> None:
        for source_rigid_prim, obstacle in self._dynamic_obstacle_sources:
            position, _orientation = source_rigid_prim.get_world_poses()
            obstacle.set_world_pose(position=np.asarray(position[0], dtype=float))

    def _step_rmpflow(self, frame_duration: float) -> None:
        # 每個 tick 先同步動態障礙物（母球）的最新世界座標，再呼叫
        # update_world() 讓 RMPflow 內部快取讀到更新。靜態障礙物（球檯）
        # 不需要每 tick 更新，但一起做掉比額外維護條件邏輯簡單。
        #
        # ⚠️ 只覆蓋 RMPflow waypoint chain 本身。AIM 收尾階段
        # （_step_joint_space_finish()／_step_finish_ik()）不呼叫
        # RMPflow，這裡註冊的障礙物對收尾那一小段沒有避障作用。
        self._sync_dynamic_obstacles()
        self._rmp_flow.update_world()

        positions = np.asarray(self._articulation.get_dof_positions())[0]
        velocities = np.asarray(self._articulation.get_dof_velocities())[0]
        active_positions = positions[self._active_dof_indices]
        active_velocities = velocities[self._active_dof_indices]

        position_targets, _velocity_targets = self._rmp_flow.compute_joint_targets(
            active_positions,
            active_velocities,
            np.array([]),
            np.array([]),
            frame_duration,
        )

        full_position_targets = self._passive_dof_hold_targets.copy()
        full_position_targets[self._active_dof_indices] = position_targets
        self._last_active_position_targets = position_targets.copy()
        self._last_active_positions_before_step = active_positions.copy()

        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(full_position_targets[None, :])
