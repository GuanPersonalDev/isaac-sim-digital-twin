"""scripts/verify_ur10e_linear_approach_trajectory.py — 驗證 Lula 能不能為
UR10e 產出「沿球桿軸的精確直線逼近」軌跡，並找出可行的逼近距離上限。

背景：AIM 的最後一段（逼近緩衝點→最終瞄準姿態）是一條純軸向直線，方向全程
不變。交給 RMPflow 反應式追蹤的代價是每段中繼 waypoint 都要賭 240 步收斂、
末端還有殘留誤差要靠解析 IK／差動 IK 收尾補。Lula 直接把這條直線離線轉成
時間最優的關節空間軌跡，沒有反應式收斂的不確定性。

這支不跑物理、不動手臂，只驗證軌跡產生本身，跟
extension/isaac_sim_impl_6_0/ur10e_linear_approach_controller.py 走同一條
API 路徑（含 IK 種子），確認三件事：
1. 能用專案既有的 ur10e_robot_description.yaml + ur10e.urdf 產出軌跡
2. 哪些逼近距離可行——距離太大時起點會落到機器人底座後方，直線路徑等於
   要求手腕穿過機器人本體
3. 產出的軌跡真的走直線（沿路取樣做 FK 量側向偏離），終點夠精確

⚠️ 兩個容易踩的點：
- Lula 的軌跡產生器吃**機器人底座座標系**，不是世界座標（沒有
  RmpFlow.set_robot_base_pose() 的等價 setter）。底座朝向固定為單位四元數
  （見 Ur10eSwingStrategy._BASE_ORIENTATION），世界→底座只需扣掉平移。
- 一定要餵 IK 種子。LulaTaskSpaceTrajectoryGenerator.compute_task_space_
  trajectory_from_points() 不讓呼叫端指定種子，會用 robot description 的
  default_q，對「手臂已經在某個構型、只想沿直線再走一小段」這個用法會選到
  別的 IK 分支——實測踩過：手臂先大幅甩到另一個構型，終點誤差 0.62m／
  1.02rad，母球達成率 0%。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_linear_approach_trajectory.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CONFIG_DIR = os.path.join(
    _PROJECT_ROOT, "assets", "rmpflow_config", "ur10e_cue", "rmpflow"
)
_ROBOT_DESCRIPTION_PATH = os.path.join(_CONFIG_DIR, "ur10e_robot_description.yaml")
_URDF_PATH = os.path.normpath(os.path.join(_CONFIG_DIR, "..", "ur10e.urdf"))
_END_EFFECTOR_FRAME = "wrist_3_link"

# 跟 scripts/test_ur10e_table_flat.py 同一個 flat 測試案例
_CUE_BALL = (0.0, 0.5)
_SHOT_ANGLE_DEG = 0.0
_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575

_APPROACH_OFFSETS_M = (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.45)


def _run() -> None:
    import numpy as np
    import lula
    from isaacsim.robot_motion.motion_generation.lula.utils import get_pose3

    from core.services import cue_pose_calculator, ur10e_analytic_ik, ur10e_placement_calculator

    robot_description = lula.load_robot(_ROBOT_DESCRIPTION_PATH, _URDF_PATH)
    kinematics = robot_description.kinematics()
    c_space_generator = lula.create_c_space_trajectory_generator(kinematics)
    conversion_config = lula.TaskSpacePathConversionConfig()

    joint_names = [
        kinematics.c_space_coord_name(i) for i in range(kinematics.num_c_space_coords())
    ]
    print(f"[linear] 載入成功，c-space 關節={joint_names}")
    if _END_EFFECTOR_FRAME not in kinematics.frame_names():
        print(f"[linear] FAIL：URDF 裡找不到 frame {_END_EFFECTOR_FRAME}")
        return

    wrist_position, wrist_orientation, tilt_rad, _crossing = (
        cue_pose_calculator.compute_tilted_wrist_pose(
            _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS
        )
    )
    if tilt_rad is None:
        print("[linear] FAIL：測試案例本身幾何無解")
        return
    direction_unit = np.asarray(
        cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad), dtype=float
    )
    base_position = np.asarray(
        ur10e_placement_calculator.compute_base_position(
            tuple(wrist_position), tuple(direction_unit), _TABLE_Z
        ),
        dtype=float,
    )

    # 世界→底座：底座朝向固定為單位四元數，只需要扣掉平移
    final_base = np.asarray(wrist_position, dtype=float) - base_position
    orientation = np.asarray(wrist_orientation, dtype=float)
    print(f"[linear] base_position={base_position.tolist()}")
    print(f"[linear] 最終瞄準姿態（底座座標）={np.round(final_base, 5).tolist()}")

    # 用專案自己的解析 IK 取一組終點姿態的關節解當種子——真實執行時手臂已經
    # 被 RMPflow 帶到逼近緩衝點，種子是「手臂當下的實際關節角」，這裡沒有
    # 活的手臂，用終點解析解代表同一個分支。
    rotation_matrix = ur10e_analytic_ik.quat_wxyz_to_rotation_matrix(orientation)
    dh_position, dh_rotation = ur10e_analytic_ik.isaac_to_dh_frame(final_base, rotation_matrix)
    solutions = ur10e_analytic_ik.inverse_kinematics(dh_position, dh_rotation)
    if not solutions:
        print("[linear] FAIL：終點姿態解不出 IK，取不到種子")
        return
    seed = np.asarray(solutions[0], dtype=float)
    print(f"[linear] IK 種子={np.round(seed, 4).tolist()}（共 {len(solutions)} 組解）")

    feasible = 0
    accurate = 0
    for offset in _APPROACH_OFFSETS_M:
        start_base = final_base - direction_unit * offset

        path_spec = lula.create_task_space_path_spec(get_pose3(start_base, rot_quat=orientation))
        path_spec.add_linear_path(get_pose3(final_base, rot_quat=orientation))

        ik_config = lula.CyclicCoordDescentIkConfig()
        ik_config.cspace_seeds = [seed]

        c_space_path = lula.convert_task_space_path_spec_to_c_space(
            path_spec, kinematics, _END_EFFECTOR_FRAME, conversion_config, ik_config
        )
        trajectory = (
            None if c_space_path is None
            else c_space_generator.generate_trajectory(c_space_path.waypoints())
        )
        if trajectory is None:
            print(f"[linear] offset={offset:.2f}m 起點={np.round(start_base, 4).tolist()} → 不可行")
            continue

        feasible += 1
        domain = trajectory.domain()
        duration = domain.upper - domain.lower

        max_lateral = 0.0
        samples = 25
        for i in range(samples + 1):
            t = domain.lower + duration * i / samples
            fk_position = np.asarray(
                kinematics.position(np.asarray(trajectory.eval(t, 0), dtype=float), _END_EFFECTOR_FRAME),
                dtype=float,
            )
            along = float(np.dot(fk_position - start_base, direction_unit))
            max_lateral = max(
                max_lateral,
                float(np.linalg.norm((fk_position - start_base) - along * direction_unit)),
            )

        end_targets = np.asarray(trajectory.eval(domain.upper, 0), dtype=float)
        end_fk = np.asarray(kinematics.position(end_targets, _END_EFFECTOR_FRAME), dtype=float)
        endpoint_error = float(np.linalg.norm(end_fk - final_base))

        start_targets = np.asarray(trajectory.eval(domain.lower, 0), dtype=float)
        start_fk = np.asarray(kinematics.position(start_targets, _END_EFFECTOR_FRAME), dtype=float)
        start_error = float(np.linalg.norm(start_fk - start_base))

        is_accurate = endpoint_error <= 0.005 and max_lateral <= 0.005 and start_error <= 0.005
        accurate += int(is_accurate)
        print(f"[linear] offset={offset:.2f}m 起點={np.round(start_base, 4).tolist()} → OK  "
              f"duration={duration:.4f}s（{duration * 60:.0f} tick @60Hz）  "
              f"起點FK誤差={start_error:.6f}m  終點FK誤差={endpoint_error:.6f}m  "
              f"最大側向偏離={max_lateral:.6f}m  {'精確' if is_accurate else '⚠️ 不夠精確'}")

    print(f"[linear] 可產生軌跡 {feasible}/{len(_APPROACH_OFFSETS_M)}，其中精確 {accurate} 個")
    if accurate:
        print("[linear] PASS：Lula 能為 UR10e 產出精確的直線逼近軌跡")
    else:
        print("[linear] FAIL：沒有任何逼近距離產出夠精確的直線軌跡")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
