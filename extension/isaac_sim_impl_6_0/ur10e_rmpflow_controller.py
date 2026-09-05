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
    放在專案自己的 assets/rmpflow_config/ur10e_cue/，手動組路徑即可（見
    RmpFlow.__init__() 只需要三個檔案的絕對路徑加 end_effector_frame_name，
    對路徑來源沒有額外假設）。
    """
    from isaacsim.robot_motion.motion_generation import RmpFlow

    with open(os.path.join(config_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    relative_paths = config.pop("relative_asset_paths")
    for key, rel_path in relative_paths.items():
        config[key] = os.path.normpath(os.path.join(config_dir, rel_path))
    return RmpFlow(**config)


class Ur10eRmpflowController:
    """UR10e 手臂定位用的 RMPflow 包裝（見 UR10e 重新設計計畫決策 3/5）。

    只驅動 RMPflow 自己認得的 6 個手臂關節（shoulder_pan_joint 等，見
    assets/rmpflow_config/ur10e_cue/rmpflow/ur10e_robot_description.yaml
    的 cspace），完全不知道也不觸碰 CueSlideJoint（第 7 個 DOF，掛在
    球桿跟 wrist_3_link 之間，見 TableRobotManager）——每次都先讀完整
    7-DOF 的當前關節位置當底，只覆寫 RMPflow 算出來的 6 個手臂關節分量，
    CueSlideJoint 的目標值維持「當下實際位置」，等於什麼都不做（保持
    原地），需要移動滑軌關節要靠另一個獨立的控制器（STRIKE 用，不經過
    RMPflow，見計畫決策 5）。

    刻意繞開 deprecated 的 ArticulationMotionPolicy/MotionPolicyController
    ——兩者的型別標註要求舊版 isaacsim.core.prims.SingleArticulation，跟
    專案現有的 isaacsim.core.experimental.prims.Articulation（batched/
    Warp array 架構）不相容。直接使用 RmpFlow 本身（純 numpy in/out，
    不依賴任何 Articulation wrapper 類別），自己寫這層薄 adapter 對接，見
    skills/isaac_sim_6_api_cache.md「RmpFlow」條目 Q6 的原始碼查證結論。

    2026-09-03 實測發現（見 scripts/verify_ur10e_rmpflow_reach.py 開頭
    註解）：RMPflow 對「一次給一個很大的末端目標位移」（約 30cm 量級的
    對角線跳躍）會卡在局部穩定點，殘留誤差可達 0.1m 以上、長時間不再收斂
    ——這是 reactive RMP controller 的已知特性（多個 RMP 分量互相拉扯），
    不是 bug。move_to_pose() 因此把大位移目標拆成一串位移量不超過
    _MAX_WAYPOINT_STEP_M（實測驗證過穩定收斂的量級）的中繼 waypoint，
    依序餵給 RMPflow，每段收斂（或逾時）才切下一段——精神上跟 WAM7 舊架構
    「Phase 0 安全姿態加 Cartesian waypoint 序列」類似，差別是這裡每一段
    都交給 RMPflow 自己導航加避障，不是差動 IK。

    2026-09-04 補充（收尾差動 IK）：`scripts/test_ur10e_table_flat.py` 的
    診斷發現，flat 案例走完整段 waypoint 後仍殘留 5.6cm 誤差，但 RMPflow
    算出的關節目標跟 PhysX 實際量到的關節位置幾乎完全吻合（tracking gap
    僅約 6e-5 rad）——代表這不是 joint drive 追不上目標（那種情況才需要
    比照 ArticulationAPIImpl._boost_wrist_gains_for_cue_stick_load() 補強
    gain），而是 RMPflow 這個 reactive controller 本身在這個姿態附近的
    計算殘留（多個 RMP 分量互相拉扯出的穩態偏差，NVIDIA 官方論壇也有
    相同回報：forums.developer.nvidia.com/t/imprecise-control-via-
    rmpflow/253139）。對照 ArticulationAPIImpl 既有 WAM7/UR3e 差動 IK 的
    做法，在最後一個 waypoint 收斂（或逾時）之後，若仍未進入容許誤差，
    改用同一套 DLS（damped least squares）差動 IK 再收尾（見
    _step_finish_ik()）——此時手臂已經很接近目標姿態，不會像大位移那樣
    掃過球檯，跳過 RMPflow 的避障不是新風險。這個收尾只在最終目標未收斂
    時才會啟動，其餘情況（例如提早收斂的 waypoint）完全不受影響。
    """

    _MAX_WAYPOINT_STEP_M = 0.08
    # 中繼 waypoint 之間的方向變化上限（見 move_to_pose() 2026-09-03
    # 補充：純位置距離決定 waypoint 數量，在「位置距離小但需要旋轉的角度
    # 很大」的路段（例如目標姿態接近起始姿態的正反面、需要接近 180 度
    # 翻轉）會讓單一 waypoint 內的方向變化量過大，RMPflow 卡住不收斂）。
    # 30 度是保守值，180 度的翻轉至少會拆成 6 段。
    _MAX_WAYPOINT_ROTATION_RAD = 0.5235987755982988  # 30 度
    _POSITION_TOLERANCE_M = 0.005
    _ORIENTATION_TOLERANCE_RAD = 0.02
    _MAX_STEPS_PER_WAYPOINT = 240

    # ⚠️ 2026-09-04 除錯記錄：一開始把 _ORIENTATION_TOLERANCE_RAD 整體
    # 從 0.02 收緊到 0.005，結果 AIM 直接崩潰（3379 步逾時，位置誤差
    # 0.408m、方向誤差 0.86rad，完全是另一個姿態）——因為這個常數同時被
    # RMPflow waypoint chain 每一段中繼點的收斂判定（_is_current_waypoint_
    # converged()）沿用，中繼點容許值收太緊會讓每一段都逼近 240 步逾時
    # 上限，累積誤差整段路徑跑歪。真正需要收緊的只有「最終姿態」精度：
    # 這裡的球桿有 1.35m 長（CUE_STICK_GRIP_TO_TIP），末端方向誤差會被
    # 這根槓桿臂等比放大——0.02rad 換算桿尖橫向誤差最壞可達 1.35*0.02≈
    # 2.7cm，幾乎等於母球半徑（2.857cm）。實測踩過：AIM 方向誤差
    # 0.00933rad（在 0.02 容許值內，判定「收斂成功」）換算桿尖偏移約
    # 1.26cm，跟 STRIKE 實測 miss 向量的橫向分量（約 1.2cm）吻合，這個
    # 偏移小於「球桿半徑+母球半徑」，導致 STRIKE 階段球桿的圓柱形桿身
    # （不是桿尖）貼著母球側面蹭過去，衝量沒有正面轉移，達成率只有
    # 42%。改成只在「最終姿態是否已收斂／要不要進入收尾」
    # （_is_pose_converged()）與「joint-space 收尾本身的收斂判定」
    # （_step_joint_space_finish()）套用這個更緊的門檻，中繼 waypoint
    # 跟差動 IK 保底路徑維持原本 0.02，不受影響。
    _FINAL_ORIENTATION_TOLERANCE_RAD = 0.005

    # 收尾差動 IK 用的常數，數值沿用 ArticulationAPIImpl 既有 WAM7/UR3e
    # 差動 IK 的同名常數（POSITION_GAIN/ORIENTATION_GAIN/DLS_LAMBDA 等）
    # ——同一個 codebase 已經實測驗證過的收斂行為，不是另外拍腦袋調的。
    _FINISH_POSITION_GAIN = 5.0
    _FINISH_ORIENTATION_GAIN = 5.0
    _FINISH_MAX_LINEAR_SPEED = 2.0  # m/s
    _FINISH_MAX_ANGULAR_SPEED = 3.0  # rad/s
    _FINISH_DLS_LAMBDA = 0.05
    _FINISH_MAX_STEPS = 240  # 跟 _MAX_STEPS_PER_WAYPOINT 同量級

    # UR10e 重新設計計畫決策 11：手動指定一組安全、離球檯足夠遠的固定
    # HOME 關節角度，不沿用「USD 重新放進場景時自然落點當 HOME」的舊
    # 機制（那是 WAM7/UR3e 的 _capture_home_position_once() 慣例）。直接
    # 沿用 assets/rmpflow_config/ur10e_cue/rmpflow/ur10e_robot_description.yaml
    # 的 default_q——這是 Lula 官方替 UR10e 選定的 cspace 參考姿態
    # （elbow 彎起、遠離手臂完全打直的奇異點），不是隨便選的數字。
    #
    # ⚠️ 2026-09-03 除錯記錄：這組 default_q 的 wrist_2_joint=0，懷疑落在
    # UR 家族手臂的手腕奇異點（wrist_2=0 時 wrist_1／wrist_3 兩軸平行/
    # 耦合）附近，可能是某些 AIM 目標（尤其 flat 案例）從 HOME 出發會卡在
    # 局部穩定點的原因之一。實測把 wrist_2 改成 π/2（遠離這個值）之後，
    # HOME 本身跟後續 AIM 反而都變得更難收斂（HOME 自己開始逾時、AIM
    # 殘留誤差從 0.16m 惡化到 0.20m），已改回原始 default_q——這個假設
    # 沒有被證實，維持官方原值，問題根因仍待查。
    _HOME_JOINT_POSITIONS = [-0.0, -1.2, 1.1, 0.0, 0.0, 0.0]

    def __init__(self, articulation, end_effector_prim_path: str) -> None:
        from isaacsim.core.experimental.prims import RigidPrim

        self._articulation = articulation
        self._end_effector_prim_path = end_effector_prim_path
        self._end_effector_rigid_prim = RigidPrim(paths=end_effector_prim_path)
        self._rmp_flow = _load_rmp_flow()

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

        # 收尾差動 IK 狀態（見類別 docstring 2026-09-04 補充）。
        self._jac_link_index: int | None = None
        self._finishing_active = False
        self._finish_steps = 0
        self._finish_target_position: np.ndarray | None = None
        self._finish_target_orientation: np.ndarray | None = None
        # 收尾用 joint-space 精確目標的狀態（見 _start_finishing_phase()
        # 2026-09-04 補充）。
        self._joint_finish_active = False
        self._joint_finish_target: np.ndarray | None = None
        self._gains_boosted_for_finish = False
        # set_robot_base_pose() 存下來的底座世界座標，供
        # _compute_analytic_finish_joint_target() 把世界座標目標轉成
        # ur10e_analytic_ik 需要的「機器人底座座標系」用——假設底座朝向
        # 固定是單位四元數（見 Ur10eSwingStrategy._BASE_ORIENTATION），
        # 只需要平移，不需要旋轉。
        self._base_position: np.ndarray | None = None

    def set_robot_base_pose(self, base_position, base_orientation) -> None:
        """告訴 RMPflow 手臂底座目前在世界座標系的實際位姿。"""
        self._base_position = np.asarray(base_position, dtype=float)
        self._rmp_flow.set_robot_base_pose(
            self._base_position,
            np.asarray(base_orientation, dtype=float),
        )

    def move_to_pose(self, target_position, target_orientation) -> None:
        """開始一段移動：把位移拆成一串中繼 waypoint 依序餵給 RMPflow，
        方向也跟著用 slerp 內插（不是從第一段就鎖定最終方向）。

        2026-09-03 實測發現：只內插位置、方向從第一段就直接設成最終目標，
        對「位置移動量大＋方向本身需要旋轉」（例如高架橋案例的傾斜姿態）
        的真實 AIM 目標會卡住不收斂——研判是 RMPflow 被迫在離最終位置還很
        遠的中繼點就同時追蹤最終方向，跟位置追蹤互相拉扯出局部穩定點。
        方向也跟著逐段內插後，每個中繼點的方向目標都貼近該中繼點「應該」
        呈現的姿態，兩者不再互相打架。
        """
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
        """回到 HOME 姿態（decision 11 的固定關節角度），透過 RMPflow 導航
        （若有註冊障礙物，會主動避開，見 add_ground_plane()/add_obstacle()），
        不是直接 joint-space 瞬移過去——decision 5：「所有手臂移動都用
        RMPflow」，RESET/HOME 這段路徑一樣可能掃過球檯，需要跟 AIM 一樣的
        避障能力。

        用 RmpFlow.get_end_effector_pose(joint_positions) 算 HOME 關節角度
        對應的世界座標末端位姿（已經套用 set_robot_base_pose() 設定的目前
        底座位姿，見該方法官方 docstring：「transformed into world
        coordinates based on the believed position of the robot base」），
        不需要真的先把手臂瞬移過去才能量到——這樣可以在手臂還在別的姿態時
        就先算出 HOME 對應的世界座標目標，交給 move_to_pose() 導航過去。
        """
        home_position, home_orientation = self._compute_home_end_effector_pose()
        self.move_to_pose(home_position, home_orientation)

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

        self._waypoint_index += 1
        if self._waypoint_index >= len(self._waypoints):
            self._start_finishing_phase()
        else:
            self._activate_current_waypoint()

    def _start_finishing_phase(self) -> None:
        """最後一個 waypoint 收斂（或逾時）之後的收尾判斷：已經在容許
        誤差內就直接結束；否則切到差動 IK 收尾（見類別 docstring
        2026-09-04 補充），維持 `_motion_active=True` 讓 step() 繼續被
        呼叫，只是改走 `_step_finish_ik()` 這個分支。

        ⚠️ 不能用 `_is_current_waypoint_converged()`——那個方法讀
        `self._waypoints[self._waypoint_index]`，但呼叫端（`step()`）在
        呼叫這裡之前已經把 `_waypoint_index` 遞增到 `len(self._waypoints)`
        （迴圈結束的信號），會是 out-of-range index（實測踩過：
        IndexError，PHYSICS_POST_STEP callback 內的例外被 Kit 印出但
        不會讓整個 SimulationApp 崩潰，導致 AIM/RESET 兩段都靜默卡死到
        `_MAX_STEPS_PER_ACTION` 逾時，而不是真的收斂或報錯）。改成直接用
        `self._waypoints[-1]`（最終目標，跟 out-of-range 那個 index 無關）
        搭配 `_is_pose_converged()`。
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

        # 2026-09-04 補充：優先用 ur10e_analytic_ik 的 closed-form 解算出
        # 精確關節目標，直接 joint-space 收尾——PhysX position-mode drive
        # 追蹤固定關節目標的精度已經實測驗證過（tracking gap ~6e-5 rad，
        # 見 scripts/test_ur10e_table_flat.py 稍早的診斷），不像 Cartesian
        # 差動 IK 那樣疊代逼近，不會受 Jacobian 在某些姿態附近病態
        # （運動學奇異點）影響——實測發現差動 IK 收尾在某些姿態會卡在
        # 0.0294rad 的方向誤差不再收斂，換算成 1.35m 長球桿的桿尖偏移
        # 超過球半徑，是這個「精確關節收尾」要解決的問題。只有 analytic
        # IK 真的找不到可達解（理論上不該發生，wrist 目標已經是 RMPflow
        # 自己收斂過的可達姿態）才退回原本的 Cartesian 差動 IK 當保底。
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
        joint-space 直接過去只是「走完最後一小段」，不是憑空跳到任意
        姿態。`_base_position` 還沒設定（理論上不該發生，AIM 呼叫前一定
        會先 set_robot_base_pose()）或 analytic IK 找不到可達解時回傳
        None，呼叫端負責退回差動 IK。"""
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

        # ⚠️ 2026-09-04 除錯記錄：收緊 _FINAL_ORIENTATION_TOLERANCE_RAD 後
        # 才踩到的新問題——move_to_home() 走的也是同一條收尾路徑，而 HOME
        # 姿態（_HOME_JOINT_POSITIONS）的 wrist_2_joint=0 正好卡在 UR
        # 家族手臂的手腕奇異點上（見類別 docstring 2026-09-03 補充）。在
        # 奇異點附近，closed-form IK 的解集合會退化（實測：只解出 4 組，
        # 不是滿額的 8 組），這時候「挑離目前姿態最近的分支」完全不可信
        # ——實測踩過：目前關節角幾乎正好在 HOME，選出來的「最近」分支卻
        # 離目前姿態達 2.7rad，把手臂拖去了完全錯誤的姿態。RMPflow 的
        # waypoint chain 在呼叫這裡之前，理論上已經把手臂帶到跟目標很接近
        # 的姿態（這個收尾機制的設計前提本來就是「走完最後一小段」），
        # 所以合理的解不該跟目前姿態差距過大——這裡加一個寬鬆的合理性
        # 上限（遠大於任何正常收尾correction，但遠小於「跳到完全不同分支」
        # 的量級），一旦挑出來的最近分支仍然差距過大，就視為「這個解集合
        # 不可信」，回傳 None 讓呼叫端退回原本的 Cartesian 差動 IK 保底
        # 路徑（DLS 疊代逼近，即使在奇異點附近也只會產生平滑的小修正，
        # 不會像 closed-form 分支選擇這樣整個跳到別的姿態）。
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

        # ⚠️ 2026-09-04 除錯記錄：只挑「wrap 後距離最近」的解還不夠——
        # UR 關節範圍是 ±2π（連續旋轉，不是 ±π 就繞回來），analytic IK
        # 解出來的原始數值（arctan2/arccos 的值域大約落在 [-π,π]）可能
        # 跟目前實際關節角相差了一整圈但「等效角度」很近（例如目前關節
        # 角實際存的是 4.5 rad，解析解算出等效的 -1.78 rad，wrap 後距離
        # 很近，但兩個原始數值差了快 2π）。PhysX 的 position-mode drive
        # 不會自動幫關節抄近路，只會照 set_dof_position_targets() 給的
        # 原始數值直接追——實測踩過：joint_space_finish 240 步還收斂不了
        # （最大誤差 0.033rad），懷疑就是走了遠路。這裡把每個關節分量都
        # 平移成「離目前實際角度最近的等效角度」再回傳，讓 drive 真的走
        # 最短路徑。
        return current_active_positions + _wrapped_diff(best_solution)

    _FINISH_GAIN_BOOST_JOINT_NAME_SUBSTRINGS = ("wrist_1", "wrist_3", "elbow")
    _FINISH_GAIN_STIFFNESS = 1e6
    _FINISH_GAIN_DAMPING = 1e4
    _FINISH_GAIN_MAX_EFFORT_MULTIPLIER = 20.0
    """⚠️ 2026-09-04 除錯記錄：一開始照抄 ArticulationAPIImpl.
    _boost_wrist_gains_for_cue_stick_load()（UR3e 驗證過的同一組常數，
    stiffness=1e15/damping=1e5），結果 wrist_1_joint 在 joint-space 收尾
    完全卡住不動（240 步、1120N·m 飽和力矩幾乎無效）。逐一排除碰撞（全
    連桿 contact reporting 確認零接觸）、關節極限（差六圈以上）、drive
    type（確認是 "force"，不是 "none"）、gains 寫入沒生效（讀原始碼確認
    update_default_gains 預設 True，boost 有正確持續生效）之後，靠 A/B
    對照測試（verify_ur10e_arm_table_collision.py 的環境變數覆寫）發現：
    wrist_1_joint 原始 baked stiffness 只有約 72,662，1e15 是這個值的
    一百多億倍，跟同一條運動鏈上其他關節（例如 CueSlideJoint 的
    max_effort=1e6）的量級差距過大，讓 PhysX 的 TGS 迭代求解器對這個最
    僵硬的關節反而欠收斂——數值上病態，不是真的「增益不夠」。改成
    1e6/1e4（約為原始值的 14 倍，遠比 UR3e 那組「1e15/1e5」溫和）之後，
    實測 2 個 physics tick 就收斂（joint_error 從 0.033rad 降到
    0.0096rad，容許值 0.02rad）。UR3e 跟 UR10e 兩邊的下游負載結構不同，
    同一組「越硬越好」的增益常數不能直接照搬，這是這次除錯的教訓。

    2026-09-05 補充（收緊 _FINAL_ORIENTATION_TOLERANCE_RAD 到 0.005 之後
    才浮現）：先查證網路上對這類「PD 位置控制器收斂不到目標」問題的建議
    做法（Isaac Sim 官方 Gain Tuner 文件、PhysX 官方文件、PD+feedforward
    控制文獻），結論一致：**應該針對個別關節依其負載分別調整增益，不是
    對全部關節套用同一組數值**（官方原話：肩部/手肘關節承受下游全部連桿
    的重力力矩需要較高增益，手腕關節較輕、低增益即可）。逐關節 log 證實
    這個方向：逾時當下 elbow_joint 誤差 0.02205rad（遠高於其他關節），
    wrist_1_joint 0.00504rad（剛好卡在門檻邊緣），其餘 4 個關節都遠低於
    容許值——不是「全部關節都不夠力」，是 elbow_joint 這個原本沒被列入
    boost 名單、扛著整條下游手臂重力力矩的關節不夠力。把 "elbow" 加進
    boost 名單（沿用同一組 1e6/1e4/20x 數值，不是另外調一組——這個量級
    已經證實對高負載關節有效，沒有先驗理由 elbow 需要不同數值，之後如果
    實測顯示不夠再個別調整）解決，不是不分青紅皂白把全部 6 個關節一起
    升到極端值。"""

    def _boost_gains_for_finish_once(self) -> None:
        """2026-09-04 除錯記錄：joint-space 收尾（_step_joint_space_finish()）
        第一版只加了重力補償力矩前饋，逐 tick log 顯示 wrist_1_joint 幾乎
        沒有改善，仍然穩定卡在離目標 0.033rad 的地方——代表問題不只是
        重力矩本身，是預設 PD 增益（stiffness/damping/max_effort）對這個
        負載來說太軟，不管有沒有額外的重力補償力矩，stiffness 項都不夠
        力氣把殘留誤差壓到收斂容許值內。改用跟 ArticulationAPIImpl.
        _boost_wrist_gains_for_cue_stick_load()（UR3e 上驗證過的同一個
        問題／同一個修法）完全相同的增益數值。

        故意只在真的要進入 joint-space 收尾時才呼叫（第一次呼叫後用
        `_gains_boosted_for_finish` 擋掉後續重複呼叫），刻意不放進
        `__init__()` 對整個 AIM 過程常駐生效——RMPflow 自己的 waypoint
        chain 階段（`_step_rmpflow()`）已經實測驗證過用預設增益就能正常
        收斂（每 tick 重新給貼近目前值的新目標，等於變相用位置追蹤模擬
        速度追蹤，沒有這個穩態下垂問題），套用未驗證過的高增益去干擾
        那段是沒必要的新變因。
        """
        if self._gains_boosted_for_finish:
            return
        self._gains_boosted_for_finish = True

        dof_names = list(self._articulation.dof_names)
        target_indices = [
            i for i, name in enumerate(dof_names)
            if any(sub in name.lower() for sub in self._FINISH_GAIN_BOOST_JOINT_NAME_SUBSTRINGS)
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
            stiffnesses[idx] = self._FINISH_GAIN_STIFFNESS
            dampings[idx] = self._FINISH_GAIN_DAMPING
            max_efforts[idx] = max_efforts[idx] * self._FINISH_GAIN_MAX_EFFORT_MULTIPLIER

        logger.info(
            "ur10e joint-space finish wrist gain boost: joint indices=%s new_max_efforts=%s",
            target_indices, [max_efforts[i] for i in target_indices],
        )
        self._articulation.set_dof_gains(stiffnesses[None, :], dampings[None, :])
        self._articulation.set_dof_max_efforts(max_efforts[None, :])

    def _step_joint_space_finish(self) -> None:
        """把 `_compute_analytic_finish_joint_target()` 算出的精確關節角
        當 joint-space 目標，交給 PhysX 關節驅動器自己插值到位——跟
        `ArticulationAPIImpl._start_joint_space_motion()`（WAM7/UR3e 的
        joint-space 移動）同一個精神：起點離目標很近，不需要（也不適合）
        再跑一次差動 IK。收斂判定看六個關節角度是不是都進到容許值內——
        這裡直接沿用弧度制 `_FINAL_ORIENTATION_TOLERANCE_RAD`（跟末端
        姿態的最終精度門檻共用同一個值，見類別常數說明）當關節角度容許
        值，量級相符（都是「小角度」等級的收斂判定）。"""
        full_position_targets = np.asarray(self._articulation.get_dof_positions())[0].copy()
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
        # ⚠️ 2026-09-04 除錯記錄：第一版沒加重力補償，逐 tick log 顯示
        # wrist_1_joint 穩定卡在離目標 0.033rad 的地方完全不動（其餘 5 個
        # 關節都準確追到 1e-4rad 等級）——這正是 ArticulationAPIImpl.
        # _boost_wrist_gains_for_cue_stick_load() 修過的同一類問題（球桿
        # 透過 CueSlideJoint 掛在 wrist_3_link 之後的槓桿臂重力力矩，超過
        # wrist_1/wrist_3 預設 PD 增益能扛住的範圍，position-mode 下穩定
        # 卡在「stiffness×殘留誤差＝重力力矩」的平衡點）。UR10e 的
        # _step_rmpflow() 之所以沒踩到，是因為 RMPflow 每個 tick 都重新
        # 給一個貼近目前值的新目標，等於變相用位置追蹤模擬速度追蹤，
        # 掩蓋了這個穩態誤差；這裡是固定目標長時間 hold，穩態下垂才會
        # 顯現。加上重力補償力矩前饋（跟 _step_strike()／差動 IK 收尾
        # 同一個做法），讓 stiffness 項只需要修正真正的追蹤誤差，不用
        # 同時對抗重力。
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
            # 2026-09-04 除錯記錄：先前只印 max_abs，看不出「哪一個關節」
            # 拖慢了收斂——wrist gain boost 只套用在 wrist_1/wrist_3（見
            # _FINISH_GAIN_BOOST_JOINT_NAME_SUBSTRINGS），逾時代表殘留誤差
            # 可能落在沒被 boost 到的關節上（wrist_2，或 shoulder/elbow
            # 端），需要逐關節數據才知道該對哪個關節加處理，不能片面猜測
            # 「全部關節一起 boost」（官方 Gain Tuner 文件建議依個別關節
            # 負載分別調整，不建議齊頭式套用同一組增益）。
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
        重力補償理由：velocity-mode PD 不會自動抗重力，見
        ArticulationAPIImpl 該方法的說明。"""
        live_position, live_orientation = self._end_effector_rigid_prim.get_world_poses()
        live_position = np.asarray(live_position[0], dtype=float)
        live_orientation = np.asarray(live_orientation[0], dtype=float)

        position_error, orientation_error = self._pose_error_components(
            live_position, live_orientation, self._finish_target_position, self._finish_target_orientation
        )
        position_ok = position_error <= self._POSITION_TOLERANCE_M
        orientation_ok = orientation_error <= self._ORIENTATION_TOLERANCE_RAD
        converged = position_ok and orientation_ok

        # 2026-09-04 補充：逐 tick 記錄位置/方向誤差的收斂趨勢（收斂中/
        # 卡住不動/來回震盪），只在 DEBUG_UR10E_FINISH_IK 環境變數開啟時
        # 印出，避免正常執行時洗版——用法跟既有 DEBUG_MOVE_SWING 慣例一致
        # （見 ArticulationAPIImpl.move_swing()）。
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
                # 逾時未收斂——明確記錄是位置、方向、還是兩者都沒到容許值
                # 內，不要只留一個「did_last_motion_timeout=True」的布林值。
                # 見 _pose_error_components() docstring：AIM 單看位置誤差
                # 曾經誤判為「已收斂」，但方向誤差沒收斂會被 1.35m 長的
                # 球桿放大成數公分等級的桿尖偏移，讓 STRIKE 完全打不到球。
                logger.warning(
                    "finish_ik 逾時未收斂（%d 步）：position_error=%.5fm（容許%.5fm，%s）"
                    " orientation_error=%.5frad（容許%.5frad，%s）",
                    self._finish_steps, position_error, self._POSITION_TOLERANCE_M,
                    "OK" if position_ok else "未達標",
                    orientation_error, self._ORIENTATION_TOLERANCE_RAD,
                    "OK" if orientation_ok else "未達標",
                )
            # ⚠️ 2026-09-04 除錯記錄：第一版在這裡直接 return，沒有歸零
            # velocity target——velocity-mode drive 會持續套用「上一次」
            # 下達的非零角速度指令，直到有新指令覆寫為止。收斂判定成立的
            # 那一刻其實還帶著非零殘留速度，若不歸零，手臂會在收斂之後
            # 繼續照原速度漂移（實測：STRIKE 開始前母球速度就已經非零，
            # 代表收斂後手臂漂移的桿子先撞到了球）。比照 ArticulationAPIImpl.
            # _stop_motion() 停止差動 IK 動作時的做法，收斂/逾時當下先明確
            # 下達一次全零速度指令止住殘留漂移。
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

        full_velocity_targets = np.zeros(jacobian_full.shape[1])
        full_velocity_targets[self._active_dof_indices] = qdot_active

        self._articulation.switch_dof_control_mode("velocity")
        self._articulation.set_dof_velocity_targets(full_velocity_targets[None, :])
        gravity_compensation_forces = self._articulation.get_dof_gravity_compensation_forces()
        self._articulation.set_dof_efforts(gravity_compensation_forces)

    @staticmethod
    def _pose_error_components(
        live_position: np.ndarray, live_orientation: np.ndarray,
        target_position: np.ndarray, target_orientation: np.ndarray,
    ) -> tuple[float, float]:
        """回傳 (position_error_m, orientation_error_rad)——拆成兩個獨立
        分量，供 `_is_pose_converged()`／`_step_finish_ik()` 共用，也讓
        逾時當下能明確記錄「是哪一項沒收斂」（見 2026-09-04 除錯記錄：
        `scripts/test_ur10e_table_flat.py` 的桿尖-母球距離診斷發現，AIM
        單看位置誤差「看起來合格」，實際上是方向誤差沒收斂，被 1.35m 長
        的球桿槓桿放大成 4cm+ 的桿尖偏移——只回報一個布林值「有沒有收斂」
        看不出是哪個分量出問題）。"""
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

    def _step_rmpflow(self, frame_duration: float) -> None:
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

        full_position_targets = positions.copy()
        full_position_targets[self._active_dof_indices] = position_targets
        self._last_active_position_targets = position_targets.copy()
        self._last_active_positions_before_step = active_positions.copy()

        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(full_position_targets[None, :])
