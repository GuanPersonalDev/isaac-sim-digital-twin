import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "rmpflow_config", "ur10e_cue", "rmpflow"
)


def _load_config(config_dir: str = _CONFIG_DIR) -> tuple[str, str, str]:
    """回傳 (robot_description_path, urdf_path, end_effector_frame_name)，
    跟 Ur10eRmpflowController 讀同一份 config.json，確保兩個控制器用的是
    同一套運動學模型。"""
    with open(os.path.join(config_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    relative_paths = config["relative_asset_paths"]
    robot_description_path = os.path.normpath(
        os.path.join(config_dir, relative_paths["robot_description_path"])
    )
    urdf_path = os.path.normpath(os.path.join(config_dir, relative_paths["urdf_path"]))
    return robot_description_path, urdf_path, config["end_effector_frame_name"]


class Ur10eLinearApproachController:
    """用 Lula 任務空間軌跡產生器走 AIM 最後一段沿球桿軸的直線逼近。

    跟 Ur10eRmpflowController 是平行、互斥的兩套手臂控制器，介面刻意一致
    （move_to_pose／step／is_motion_complete／did_last_motion_timeout），
    ArticulationAPIImpl 只要把 _ur10e_active_controller 指過來就能換手。

    為什麼最後一段不繼續用 RMPflow：這段是純軸向平移、方向全程不變，路徑
    本身早就確定，不需要反應式規劃。RMPflow 是 reactive controller，每個中繼
    waypoint 都要賭收斂步數、末端還有殘留誤差要另外用解析 IK／差動 IK 收尾補
    （見 Ur10eRmpflowController._start_finishing_phase()）。Lula 直接把這條
    直線離線轉成時間最優的關節空間軌跡，逐 tick 播放即可——
    scripts/verify_ur10e_linear_approach_trajectory.py 實測：0.2m 的逼近
    17 個 tick 播完，終點 FK 誤差 1e-6 m、全程側向偏離 3e-4 m，對照 RMPflow
    同一段要 934 個 tick 且仍有殘留誤差。

    也順帶消掉「母球同時是障礙物又是目的地」這個矛盾：這段完全不經過
    RMPflow，不需要先 disable_dynamic_obstacles() 再記得重新啟用。桿子在
    AIM 期間已經退到後擺位置（見 ArticulationAPIImpl._UR10E_AIM_RETRACT_
    POSITION_M），沿軸平移不會碰到母球，安全性由幾何保證而不是由避障保證。

    ⚠️ 適用範圍有限：`move_to_pose()` 在起點/終點連線運動學上不可行時會回傳
    False（Lula 回傳 None），呼叫端必須準備退回 RMPflow。實測逼近距離超過
    約 0.3m 就會失敗——起點會落到機器人底座後方，直線路徑等於要求手腕穿過
    機器人本體（見上述驗證腳本的距離掃描）。
    """

    # 軌跡播完之後繼續 hold 最終關節目標、等 PhysX 追上的步數上限。
    _SETTLE_MAX_STEPS = 120
    _POSITION_TOLERANCE_M = 0.005
    _ORIENTATION_TOLERANCE_RAD = 0.005
    # 軌跡起點關節角跟手臂當下關節角容許的最大差距。IK 有多組解，即使餵了
    # 種子也不保證一定收在同一分支；差距過大代表 Lula 選到別的分支，照著播
    # 會讓手臂先大幅甩到另一個構型再逼近（實測踩過：終點誤差 0.62m／1.02rad，
    # 母球達成率 0%）。跟 Ur10eRmpflowController._compute_analytic_finish_
    # joint_target() 的 _MAX_REASONABLE_FINISH_DELTA_RAD 同一個防呆模式。
    _MAX_START_DEVIATION_RAD = 0.05

    def __init__(self, articulation, end_effector_prim_path: str) -> None:
        import lula
        from isaacsim.core.experimental.prims import RigidPrim

        robot_description_path, urdf_path, end_effector_frame = _load_config()
        self._articulation = articulation
        self._end_effector_rigid_prim = RigidPrim(paths=end_effector_prim_path)
        self._end_effector_frame = end_effector_frame

        # 直接用 lula，不透過 LulaTaskSpaceTrajectoryGenerator 包裝：那層的
        # compute_task_space_trajectory_from_points() 不讓呼叫端指定 IK 種子，
        # 只能用預設種子（robot description 的 default_q），對「手臂已經在某個
        # 特定構型、只想沿直線再走一小段」這個用法會選到別的 IK 分支。
        robot_description = lula.load_robot(robot_description_path, urdf_path)
        self._kinematics = robot_description.kinematics()
        self._c_space_trajectory_generator = lula.create_c_space_trajectory_generator(self._kinematics)
        self._path_conversion_config = lula.TaskSpacePathConversionConfig()

        dof_names = list(self._articulation.dof_names)
        self._num_dofs = len(dof_names)
        self._active_joint_names = [
            self._kinematics.c_space_coord_name(i)
            for i in range(self._kinematics.num_c_space_coords())
        ]
        self._active_dof_indices = [dof_names.index(name) for name in self._active_joint_names]

        # Lula 的軌跡產生器吃機器人底座座標系，沒有 RmpFlow.set_robot_base_pose()
        # 那種 setter，世界座標目標要自己扣掉底座平移再餵進去。底座朝向固定是
        # 單位四元數（見 Ur10eSwingStrategy._BASE_ORIENTATION），不需要旋轉。
        self._base_position: np.ndarray | None = None

        self._trajectory = None
        self._elapsed_time = 0.0
        self._settle_steps = 0
        self._motion_active = False
        self._did_last_motion_timeout = False
        self._target_position: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        # 跟 Ur10eRmpflowController 同一個理由：不能每個 tick 拿 CueSlideJoint
        # 當下的實際位置當它的目標，那樣 PD 誤差恆為 0，球桿會被慣性帶著漂移。
        self._passive_dof_hold_targets: np.ndarray | None = None

    def set_robot_base_position(self, base_position) -> None:
        self._base_position = np.asarray(base_position, dtype=float)

    def move_to_pose(self, target_position, target_orientation) -> bool:
        """從目前末端位姿沿直線走到目標位姿。回傳 False 代表這條直線走不了
        （產不出軌跡，或產出的軌跡起點落在別的 IK 分支），呼叫端要自己退回
        RMPflow。"""
        import lula
        from isaacsim.robot_motion.motion_generation.lula.utils import get_pose3

        target_position = np.asarray(target_position, dtype=float)
        target_orientation = np.asarray(target_orientation, dtype=float)

        live_position, live_orientation = self._end_effector_rigid_prim.get_world_poses()
        live_position = np.asarray(live_position[0], dtype=float)
        live_orientation = np.asarray(live_orientation[0], dtype=float)
        current_active_positions = np.asarray(self._articulation.get_dof_positions())[0][
            self._active_dof_indices
        ]

        path_spec = lula.create_task_space_path_spec(
            get_pose3(live_position - self._base_position, rot_quat=live_orientation)
        )
        path_spec.add_linear_path(
            get_pose3(target_position - self._base_position, rot_quat=target_orientation)
        )

        # 用手臂當下的關節角當 IK 種子，讓解出來的 c-space 路徑留在目前分支。
        ik_config = lula.CyclicCoordDescentIkConfig()
        ik_config.cspace_seeds = [current_active_positions]

        c_space_path = lula.convert_task_space_path_spec_to_c_space(
            path_spec, self._kinematics, self._end_effector_frame,
            self._path_conversion_config, ik_config,
        )
        if c_space_path is None:
            logger.warning("Lula 無法把直線逼近轉成 c-space 路徑，退回 RMPflow")
            return False

        trajectory = self._c_space_trajectory_generator.generate_trajectory(c_space_path.waypoints())
        if trajectory is None:
            logger.warning("Lula 產不出直線逼近軌跡，退回 RMPflow")
            return False

        start_targets = np.asarray(trajectory.eval(trajectory.domain().lower, 0), dtype=float)
        start_deviation = float(
            np.max(np.abs(np.mod(start_targets - current_active_positions + np.pi, 2.0 * np.pi) - np.pi))
        )
        if start_deviation > self._MAX_START_DEVIATION_RAD:
            logger.warning(
                "直線逼近軌跡的起點關節角離手臂當下姿態 %.5frad（上限 %.5frad），"
                "研判 IK 選到別的分支，退回 RMPflow",
                start_deviation, self._MAX_START_DEVIATION_RAD,
            )
            return False

        self._trajectory = trajectory
        self._elapsed_time = 0.0
        self._settle_steps = 0
        self._motion_active = True
        self._did_last_motion_timeout = False
        self._target_position = target_position
        self._target_orientation = target_orientation
        self._passive_dof_hold_targets = np.asarray(self._articulation.get_dof_positions())[0].copy()
        return True

    def is_motion_complete(self) -> bool:
        return not self._motion_active

    def did_last_motion_timeout(self) -> bool:
        return self._did_last_motion_timeout

    def step(self, frame_duration: float) -> None:
        if not self._motion_active:
            return

        domain = self._trajectory.domain()
        sample_time = min(domain.lower + self._elapsed_time, domain.upper)
        joint_positions = np.asarray(self._trajectory.eval(sample_time, 0), dtype=float)

        full_position_targets = self._passive_dof_hold_targets.copy()
        full_position_targets[self._active_dof_indices] = joint_positions
        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(full_position_targets[None, :])

        # velocity-mode PD 不會自動抗重力，position-mode 則會停在「stiffness×
        # 殘留誤差＝重力力矩」的平衡點——球桿掛在 wrist_3_link 之後的槓桿臂
        # 重力矩夠大，需要前饋補償（跟 Ur10eRmpflowController.
        # _step_joint_space_finish() 同一個做法）。
        gravity_compensation_forces = self._articulation.get_dof_gravity_compensation_forces()
        self._articulation.set_dof_efforts(gravity_compensation_forces)

        self._elapsed_time += frame_duration
        if domain.lower + self._elapsed_time < domain.upper:
            return

        # 軌跡本身播完了，繼續 hold 最終關節目標等 PhysX 追上——軌跡是運動學
        # 上精確的，但 joint drive 追蹤有自己的延遲。
        self._settle_steps += 1
        if self._is_converged():
            self._motion_active = False
            return
        if self._settle_steps >= self._SETTLE_MAX_STEPS:
            live_position, live_orientation = self._end_effector_rigid_prim.get_world_poses()
            position_error, orientation_error = self._pose_error(
                np.asarray(live_position[0], dtype=float),
                np.asarray(live_orientation[0], dtype=float),
            )
            logger.warning(
                "直線逼近 settle 逾時（%d 步）：position_error=%.5fm（容許%.5fm）"
                " orientation_error=%.5frad（容許%.5frad）",
                self._settle_steps, position_error, self._POSITION_TOLERANCE_M,
                orientation_error, self._ORIENTATION_TOLERANCE_RAD,
            )
            self._did_last_motion_timeout = True
            self._motion_active = False

    def _is_converged(self) -> bool:
        live_position, live_orientation = self._end_effector_rigid_prim.get_world_poses()
        position_error, orientation_error = self._pose_error(
            np.asarray(live_position[0], dtype=float),
            np.asarray(live_orientation[0], dtype=float),
        )
        return (
            position_error <= self._POSITION_TOLERANCE_M
            and orientation_error <= self._ORIENTATION_TOLERANCE_RAD
        )

    def _pose_error(self, live_position: np.ndarray, live_orientation: np.ndarray) -> tuple[float, float]:
        position_error = float(np.linalg.norm(live_position - self._target_position))
        dot = float(np.clip(np.abs(np.dot(live_orientation, self._target_orientation)), -1.0, 1.0))
        return position_error, 2.0 * np.arccos(dot)
