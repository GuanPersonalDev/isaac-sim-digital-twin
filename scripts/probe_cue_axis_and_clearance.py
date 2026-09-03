"""
scripts/probe_cue_axis_and_clearance.py — 兩件事一次量清楚：

(1) **球桿實體的軸向到底是末端執行器的哪一軸**。`ball_stick.usda` 的 Cylinder
    是 `axis="Y"`、local Y 從 -0.15 到 +1.35；而 `align_prim_to_target()` 是把
    球桿的世界變換直接設成 end effector 的世界變換，所以球桿實體沿的是 ee 的
    **Y 軸**。但 `search_ur3e_placement_constants.py`／`ur3e_placement_
    calculator.py`／`test_*_ur3e_table.py` 的解析式一律用
    `cue_local_axis=[0,0,1]`（ee 的 **Z 軸**）算桿尖。兩者若不一致，先前所有
    「解析算出的 base_position 跟 GUI Property 面板吻合」只證明兩段程式碼用
    同一套假設、彼此自洽，並沒有證明跟物理一致。這支腳本把解析桿尖與實體
    球桿的世界座標並排印出來直接比對。

(2) **後擺過程球桿實體的最低點**，對照已量到的地板高度
    （`billiard_env.usda` 的 SimpleRoom/GroundPlane translate z=-0.7695，
    Towel_Room01_floor_bottom 上緣 -0.76957），確認 2026-09-02 GUI 重跑觀察到
    的「tick 75 撞地板後整支手臂凍結」在幾何上的成因，並產出把「後擺球桿地板
    淨空」寫成搜尋限制式所需要的數據。

重現的是 GUI 實際卡住的那一組：cue_ball=(-0.036, -0.752)（tilt≈6.50°，屬
高架橋案例），走跟 `table_orchestrator._execute_aim_ur3e()` 完全一樣的正式
程式碼路徑算 base_position/joint_targets。

跑法（需要 Isaac Sim headless）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" -u scripts/probe_cue_axis_and_clearance.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_UR3E_PATH = "Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_CUE_BALL = (-0.036, -0.752)
_SHOT_ANGLE_DEG = 0.0
_FLOOR_Z = -0.7695
_BACKSWING_DEG = 30.0
_SWEEP_SAMPLES = 13


def _rotate_vector_by_quat(quat_wxyz, vec):
    import numpy as np
    w = quat_wxyz[0]
    q_xyz = np.asarray(quat_wxyz[1:], dtype=float)
    t = 2.0 * np.cross(q_xyz, vec)
    return np.asarray(vec, dtype=float) + w * t + np.cross(q_xyz, t)


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, UsdGeom, Usd

    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from core.models.billiard_table import BilliardTable
    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    from core.services.asset_utility import CUE_STICK_PATH
    from core.services import cue_pose_calculator, ur3e_placement_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    BilliardTable(
        "/World/ProbeTable", stage_api, MaterialAPIImpl(), RigidBodyAPIImpl(), (0.0, 0.0)
    )

    wrist, _orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS,
    )
    target_direction = cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad)
    print(f"[probe] cue_ball={_CUE_BALL} tilt={np.degrees(tilt_rad):.2f}deg crossing={crossing}")
    print(f"[probe] target_wrist={list(wrist)}  target_direction={target_direction.tolist()}")

    if tilt_rad <= 1e-6:
        base_position, joint_targets = ur3e_placement_calculator.compute_flat_base_position_and_joint_targets(
            tuple(wrist), _SHOT_ANGLE_DEG
        )
    else:
        base_position, joint_targets = ur3e_placement_calculator.compute_bridge_base_position_and_joint_targets(
            tuple(wrist), tuple(target_direction), _CUE_BALL[1]
        )
    print(f"[probe] base_position={base_position}")
    print(f"[probe] joint_targets={joint_targets}")
    print(f"[probe] floor_z={_FLOOR_Z} (base is {base_position[2] - _FLOOR_Z:+.4f} m above floor)")

    robot_base_path = "/World/ProbeRobot"
    robot_prim_path = robot_base_path + "/Robot"
    end_effector_prim_path = robot_prim_path + "/wrist_3_link"
    cue_stick_prim_path = robot_base_path + "/CueStick"

    stage_api.create_reference_prim(robot_prim_path, _UR3E_PATH)
    stage_api.set_prim_translate(robot_prim_path, *base_position)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=robot_prim_path)
    for _ in range(5):
        simulation_app.update()

    num_joints = np.asarray(articulation.get_dof_positions()).reshape(-1).size
    targets = np.asarray(joint_targets, dtype=float)[:num_joints]

    def _set_joints(joints):
        articulation.set_dof_positions(np.asarray(joints, dtype=float)[None, :])
        articulation.set_dof_velocities(np.zeros((1, num_joints)))
        for _ in range(3):
            simulation_app.update()

    _set_joints(targets)

    # 球桿照 TableRobotManager 的做法掛上去（同樣四個呼叫、同樣順序）
    stage_api.create_reference_prim(cue_stick_prim_path, CUE_STICK_PATH)
    stage_api.align_prim_to_target(cue_stick_prim_path, end_effector_prim_path)
    stage_api.filter_collision_pair(cue_stick_prim_path, end_effector_prim_path)
    stage_api.create_fixed_joint(
        cue_stick_prim_path + "/FixedJointToRobot", cue_stick_prim_path, end_effector_prim_path
    )
    for _ in range(5):
        simulation_app.update()

    end_effector_rigid_prim = RigidPrim(paths=end_effector_prim_path)
    cue_prim = stage.GetPrimAtPath(cue_stick_prim_path)

    def _cue_world_bounds():
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeWorldBound(cue_prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        return np.array(rng.GetMin()), np.array(rng.GetMax())

    def _report(label):
        pos, orient = end_effector_rigid_prim.get_world_poses()
        ee_pos = np.asarray(pos[0], dtype=float)
        ee_quat = np.asarray(orient[0], dtype=float)
        axis_z = _rotate_vector_by_quat(ee_quat, [0.0, 0.0, 1.0])
        axis_y = _rotate_vector_by_quat(ee_quat, [0.0, 1.0, 0.0])
        tip_z_axis = ee_pos + CUE_STICK_GRIP_TO_TIP * axis_z
        tip_y_axis = ee_pos + CUE_STICK_GRIP_TO_TIP * axis_y
        bounds = _cue_world_bounds()
        print(f"--- {label} ---")
        print(f"  ee_pos={np.round(ee_pos, 4).tolist()}")
        print(f"  analytic tip via ee Z axis (current code assumption)={np.round(tip_z_axis, 4).tolist()}")
        print(f"  analytic tip via ee Y axis (cue asset axis)         ={np.round(tip_y_axis, 4).tolist()}")
        if bounds is not None:
            lo, hi = bounds
            print(f"  physical cue world bbox: min={np.round(lo, 4).tolist()} max={np.round(hi, 4).tolist()}")
            print(f"  physical cue lowest z={lo[2]:+.4f}  clearance above floor {lo[2] - _FLOOR_Z:+.4f} m")

    _report("AIM target pose (joint_targets)")

    print("")
    print(f"=== backswing sweep (elbow rotates {_BACKSWING_DEG} deg back from target) ===")
    elbow_index = ur3e_placement_calculator.UR3E_ELBOW_DOF_INDEX
    backswing_rad = np.radians(_BACKSWING_DEG)
    worst = None
    for i in range(_SWEEP_SAMPLES):
        frac = i / (_SWEEP_SAMPLES - 1)
        joints = targets.copy()
        joints[elbow_index] = targets[elbow_index] - backswing_rad * frac
        _set_joints(joints)
        bounds = _cue_world_bounds()
        if bounds is None:
            continue
        lo, _hi = bounds
        clearance = lo[2] - _FLOOR_Z
        flag = "  <<< HITS FLOOR" if clearance <= 0 else ""
        print(f"  elbow={joints[elbow_index]:+.4f} (backswing {np.degrees(backswing_rad * frac):5.1f} deg)  "
              f"cue lowest z={lo[2]:+.4f}  clearance={clearance:+.4f}{flag}")
        if worst is None or clearance < worst[1]:
            worst = (joints[elbow_index], clearance)

    print("")
    print(f"[probe] worst clearance over backswing={worst[1]:+.4f} m (elbow={worst[0]:+.4f})")


if __name__ == "__main__":
    import traceback

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
