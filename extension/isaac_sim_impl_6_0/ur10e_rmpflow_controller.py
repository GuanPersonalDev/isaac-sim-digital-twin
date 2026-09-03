import json
import os

import numpy as np

_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "rmpflow_config", "ur10e_cue", "rmpflow"
)


def _load_rmp_flow(config_dir: str = _CONFIG_DIR):
    """依 config.json 的 relative_asset_paths 組出絕對路徑，建構 RmpFlow。

    不用官方 `interface_config_loader.load_supported_motion_policy_config()`
    ——那個函式只認官方內建的 `motion_policy_configs/` 目錄結構，我們的設定檔
    放在專案自己的 `assets/rmpflow_config/ur10e_cue/`，手動組路徑即可（見
    `RmpFlow.__init__()` 只需要三個檔案的絕對路徑＋`end_effector_frame_name`，
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

    只驅動 RMPflow 自己認得的 6 個手臂關節（`shoulder_pan_joint` 等，見
    `assets/rmpflow_config/ur10e_cue/rmpflow/ur10e_robot_description.yaml`
    的 `cspace`），完全不知道／不觸碰 `CueSlideJoint`（第 7 個 DOF，掛在
    球桿跟 `wrist_3_link` 之間，見 `TableRobotManager`）——`step()` 每次都
    先讀完整 7-DOF 的當前關節位置當底，只覆寫 RMPflow 算出來的 6 個手臂
    關節分量，`CueSlideJoint` 的目標值維持「當下實際位置」，等於什麼都不做
    （保持原地），需要移動滑軌關節要靠另一個獨立的控制器（STRIKE 用，
    不經過 RMPflow，見計畫決策 5）。

    刻意繞開 deprecated 的 `ArticulationMotionPolicy`/`MotionPolicyController`
    ——兩者的型別標註要求舊版 `isaacsim.core.prims.SingleArticulation`，跟
    專案現有的 `isaacsim.core.experimental.prims.Articulation`（batched／
    Warp array 架構）不相容。直接使用 `RmpFlow` 本身（純 numpy in/out，
    不依賴任何 Articulation wrapper 類別），自己寫這層薄 adapter 對接，見
    `skills/isaac_sim_6_api_cache.md`「RmpFlow」條目 Q6 的原始碼查證結論。
    """

    def __init__(self, articulation) -> None:
        self._articulation = articulation
        self._rmp_flow = _load_rmp_flow()

        dof_names = list(self._articulation.dof_names)
        self._active_joint_names = list(self._rmp_flow.get_active_joints())
        # RMPflow 的 6 個手臂關節在 7-DOF 完整陣列裡的 index——用名稱比對，
        # 不寫死順序（見 RmpFlow 的 get_active_joints() 是名稱查詢介面，跟
        # 底層 dof_names 的實際排序無關，這個對應關係只需要算一次）。
        self._active_dof_indices = [dof_names.index(name) for name in self._active_joint_names]

    def set_robot_base_pose(
        self, base_position: list[float], base_orientation: list[float]
    ) -> None:
        """告訴 RMPflow 手臂底座目前在世界座標系的實際位姿——RMPflow 內部
        運動學模型預設假設底座在世界原點（identity pose），我們的
        UR10e 透過 `RobotArm.reposition()` 被搬到球檯旁（`TableRobotManager.
        _ROBOT_OFFSET_FROM_TABLE_CENTER`），沒呼叫這個方法的話，RMPflow
        算出的末端目標會系統性偏移整個底座位移量（實測：X/Z 大致收斂，
        Y 卡在遠離目標約 0.25m 處收斂不下去，正是底座偏移量級）。

        base_orientation: [qw, qx, qy, qz]，跟 set_end_effector_target()
        同一組四元數慣例。robot_arm.reposition() 每次搬動底座後都要重新
        呼叫這個方法，跟目標末端位姿無關（AIM 每次瞄準都會呼叫
        reposition()，見 core/models/table_robot_manager.py／
        core/services/table_orchestrator.py）。
        """
        self._rmp_flow.set_robot_base_pose(
            np.asarray(base_position, dtype=float),
            np.asarray(base_orientation, dtype=float),
        )

    def set_end_effector_target(
        self, target_position: list[float], target_orientation: list[float] | None = None
    ) -> None:
        """設定 RMPflow 的末端執行器目標（wrist_3_link 的世界位姿）。

        target_orientation: [qw, qx, qy, qz]，跟專案其餘介面一致；RmpFlow
        底層用同樣的四元數慣例（wxyz），不需要轉換。

        ⚠️ 一定要轉成 numpy array 再傳——RmpFlow 內部
        （`lula/motion_policies.py` `set_end_effector_target()`）直接寫
        `target_position * self._meters_per_unit`，傳純 Python list 進去
        會是「list 乘 float」，丟 TypeError；這個例外沒有被 Kit 的例外處理
        機制印出 traceback，只會讓整個 SimulationApp 悄悄關閉、exit code
        0（實測踩過：看起來像是卡住或 native crash，其實只是一般 Python
        TypeError 沒被印出來）。
        """
        orientation_arg = (
            np.asarray(target_orientation, dtype=float) if target_orientation is not None else None
        )
        self._rmp_flow.set_end_effector_target(
            np.asarray(target_position, dtype=float), orientation_arg
        )

    def update_world(self) -> None:
        """更新已註冊障礙物（見 add_ground_plane()/add_obstacle()）在 RMPflow
        world 內的姿態——只有動態障礙物（static=False）才需要每次呼叫，
        本專案的球檯/地板全部是 static=True，理論上呼叫這個是 no-op，但
        保留呼叫點方便未來若真的需要動態障礙物時不用改呼叫端邏輯。"""
        self._rmp_flow.update_world()

    def add_ground_plane(self, ground_plane) -> None:
        """註冊地板障礙物，ground_plane 必須是
        `isaacsim.core.api.objects.GroundPlane`（舊版 API wrapper，不是
        prim path 字串，也不是新版 experimental prims 物件——見
        skills/isaac_sim_6_api_cache.md「RmpFlow」條目 Q4 的原始碼查證）。"""
        self._rmp_flow.add_ground_plane(ground_plane)

    def add_obstacle(self, obstacle, static: bool = True) -> None:
        """註冊障礙物（球檯等），obstacle 必須是
        `isaacsim.core.api.objects` 底下的 FixedCuboid/VisualCuboid/
        DynamicCuboid 等舊版 API wrapper（同上 Q4）。"""
        self._rmp_flow.add_obstacle(obstacle, static=static)

    def step(self, frame_duration: float) -> None:
        """每個 physics tick 呼叫一次：讀當前 7-DOF 關節狀態、篩出 RMPflow
        的 6 個活動關節、算出新的關節位置目標、寫回完整 7-DOF 陣列（第 7
        個 DOF 維持原地，見類別 docstring），套用到 articulation。"""
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

