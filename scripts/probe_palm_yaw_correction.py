"""
scripts/probe_palm_yaw_correction.py — 修正 CANONICAL_REST_JOINTS 的手腕姿態：
base_yaw=0 時，網格搜尋 (wrist_pitch, palm_yaw) 組合，找出能同時滿足
「球桿水平指向角=0°（跟手腕徑向參考方向 +X 一致）」「離水平面傾斜角最小」
「手腕位置維持在 (_LOCAL_TIP_RADIUS, 0, _LOCAL_TIP_HEIGHT) 不變」三個條件的姿態。

背景見 core/services/base_placement_calculator.py docstring「曾經誤判」段落：
CANONICAL_REST_JOINTS（wrist_pitch=palm_yaw=0）在 base_yaw=0 時手腕落在 +X，
但球桿實際指向 +Y（82° 附近），差了快 90 度。單獨調 palm_yaw ±90° 測過，
指向角可以接近 0，但會引入嚴重傾斜（這組姿態下 palm_yaw 的旋轉軸沒有對齊
世界垂直軸，指向角跟傾斜角被耦合在一起），所以需要兩個關節一起搜尋。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_palm_yaw_correction.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

_WRIST_PITCH_DEG_STEPS = (-32,)
_PALM_YAW_DEG_STEPS = (86,)
_GRID = [
    (math.radians(wp), math.radians(py))
    for wp in _WRIST_PITCH_DEG_STEPS
    for py in _PALM_YAW_DEG_STEPS
]


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.services.asset_utility import CUE_STICK_PATH
    from core.services.base_placement_calculator import (
        CANONICAL_REST_JOINTS,
        CUE_STICK_GRIP_TO_TIP,
        _LOCAL_TIP_RADIUS,
        _LOCAL_TIP_HEIGHT,
    )

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    timeline = omni.timeline.get_timeline_interface()

    def _rotate(quat_wxyz, vec):
        w = quat_wxyz[0]
        qxyz = quat_wxyz[1:]
        t = 2.0 * np.cross(qxyz, vec)
        return vec + w * t + np.cross(qxyz, t)

    results = []

    for index, (wrist_pitch, palm_yaw) in enumerate(_GRID):
        label = f"wp{math.degrees(wrist_pitch):+.0f}_py{math.degrees(palm_yaw):+.0f}"
        base_offset = np.array([index * 10.0, 0.0, 0.0])
        base_path = f"/World/Grid_{index}"
        robot = BarrettWamRobot(base_path, stage_api, articulation_api=None, position=tuple(base_offset.tolist()))
        robot_prim_path = BarrettWamRobot.get_prim_path(base_path)
        end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(base_path)

        cue_stick_path = base_path + "/CueStick"
        stage_api.create_reference_prim(cue_stick_path, CUE_STICK_PATH)

        # 對已在 Play 中的 timeline 再次 play() 不會讓新建 prim 正確被 tensor
        # articulation view 接手，必須先 stop 再 play 強制重新初始化。
        timeline.stop()
        for _ in range(5):
            simulation_app.update()
        timeline.play()
        for _ in range(5):
            simulation_app.update()

        articulation = Articulation(paths=robot_prim_path)
        articulation.switch_dof_control_mode("position")

        # 先到未修改過的 canonical rest pose 完全穩定，再單獨改 wrist_pitch/palm_yaw，
        # 避免從初始姿態同時對好幾個關節下一次大位移目標引發不必要的動態震盪。
        q_rest = [0.0, *CANONICAL_REST_JOINTS]
        articulation.set_dof_position_targets(np.array([q_rest]))
        for _ in range(150):
            simulation_app.update()

        shoulder_pitch, shoulder_yaw, elbow_pitch, _wy0, _wp0, _py0 = CANONICAL_REST_JOINTS
        q_target = [0.0, shoulder_pitch, shoulder_yaw, elbow_pitch, 0.0, wrist_pitch, palm_yaw]
        articulation.set_dof_position_targets(np.array([q_target]))
        for _ in range(300):
            simulation_app.update()

        end_effector = RigidPrim(paths=end_effector_prim_path)
        positions, orientations = end_effector.get_world_poses()
        wrist_world = np.array(positions[0].list()) - base_offset

        stage_api.align_prim_to_target(cue_stick_path, end_effector_prim_path)
        stage_api.filter_collision_pair(cue_stick_path, end_effector_prim_path)
        joint_path = cue_stick_path + "/FixedJointToRobot"
        stage_api.create_fixed_joint(joint_path, cue_stick_path, end_effector_prim_path)
        for _ in range(20):
            simulation_app.update()

        cue_stick = RigidPrim(paths=cue_stick_path)
        _cue_positions, cue_orientations = cue_stick.get_world_poses()
        cue_orient = np.array(cue_orientations[0].list())
        cue_tip_offset = _rotate(cue_orient, np.array([0.0, CUE_STICK_GRIP_TO_TIP, 0.0]))

        heading_deg = math.degrees(math.atan2(cue_tip_offset[1], cue_tip_offset[0]))
        horizontal_len = float(np.hypot(cue_tip_offset[0], cue_tip_offset[1]))
        tilt_deg = math.degrees(math.atan2(abs(cue_tip_offset[2]), horizontal_len))
        wrist_target = np.array([_LOCAL_TIP_RADIUS, 0.0, _LOCAL_TIP_HEIGHT])
        wrist_error_m = float(np.linalg.norm(wrist_world - wrist_target))
        print(f"  wrist_world_position(precise)={wrist_world.tolist()}")
        print(f"  cue_tip_offset_from_wrist(precise)={cue_tip_offset.tolist()}")
        actual_q_final = np.asarray(articulation.get_dof_positions())[0]
        print(f"  actual_dof_positions(final)={actual_q_final.tolist()}")

        results.append(
            {
                "label": label,
                "wrist_pitch_deg": math.degrees(wrist_pitch),
                "palm_yaw_deg": math.degrees(palm_yaw),
                "heading_deg": heading_deg,
                "tilt_deg": tilt_deg,
                "wrist_error_m": wrist_error_m,
            }
        )
        print(
            f"{label}: heading={heading_deg:+7.2f} deg  tilt={tilt_deg:6.2f} deg  "
            f"wrist_error={wrist_error_m:.4f} m"
        )

    print("\n=== TOP 5（heading 越接近 0、tilt 越小、wrist_error 越小越好，等權重排序）===")
    results.sort(key=lambda r: abs(r["heading_deg"]) + r["tilt_deg"] * 2.0 + r["wrist_error_m"] * 1000.0)
    for r in results[:5]:
        print(
            f"  wrist_pitch={r['wrist_pitch_deg']:+.0f}°  palm_yaw={r['palm_yaw_deg']:+.0f}°  "
            f"heading={r['heading_deg']:+7.2f}°  tilt={r['tilt_deg']:6.2f}°  "
            f"wrist_error={r['wrist_error_m']:.4f} m"
        )


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
