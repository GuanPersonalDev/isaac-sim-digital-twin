import json
import os

import numpy as np

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
    """

    _MAX_WAYPOINT_STEP_M = 0.08
    _POSITION_TOLERANCE_M = 0.005
    _ORIENTATION_TOLERANCE_RAD = 0.02
    _MAX_STEPS_PER_WAYPOINT = 240

    def __init__(self, articulation, end_effector_prim_path: str) -> None:
        from isaacsim.core.experimental.prims import RigidPrim

        self._articulation = articulation
        self._end_effector_rigid_prim = RigidPrim(paths=end_effector_prim_path)
        self._rmp_flow = _load_rmp_flow()

        dof_names = list(self._articulation.dof_names)
        self._active_joint_names = list(self._rmp_flow.get_active_joints())
        self._active_dof_indices = [dof_names.index(name) for name in self._active_joint_names]

        self._waypoints = []
        self._waypoint_index = 0
        self._steps_on_current_waypoint = 0
        self._motion_active = False
        self._did_last_motion_timeout = False

    def set_robot_base_pose(self, base_position, base_orientation) -> None:
        """告訴 RMPflow 手臂底座目前在世界座標系的實際位姿。"""
        self._rmp_flow.set_robot_base_pose(
            np.asarray(base_position, dtype=float),
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
        num_segments = max(1, int(np.ceil(distance / self._MAX_WAYPOINT_STEP_M)))

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

    def is_motion_complete(self) -> bool:
        return not self._motion_active

    def did_last_motion_timeout(self) -> bool:
        return self._did_last_motion_timeout

    def step(self, frame_duration: float) -> None:
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
            self._motion_active = False
        else:
            self._activate_current_waypoint()

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

        self._articulation.switch_dof_control_mode("position")
        self._articulation.set_dof_position_targets(full_position_targets[None, :])
