"""
scripts/test_ur10e_actuator_swing_isolated.py - UR10e+專用出力結構重新設計
計畫步驟 4：推桿機構（不含手臂）。

沿用 scripts/test_ur3e_human_pose_swing_speed.py 已驗證過的方法論
（quintic velocity profile + gravity compensation，逐 tick 記錄歷史、
事後從紀錄裡挑「關節角度最接近目標」那一個 tick 當真正的接觸瞬間量測點，
避免衝過頭造成的假象），改成驗證 Ur10eCueSlideController 驅動
CueSlideJoint（1 個線性 DOF）單獨能不能達到目標桿尖速度。

跟 UR3e 的 elbow-pivot 案例不同：UR10e 的滑軌關節軸向就是球桿軸向，
桿尖速度＝滑軌關節線速度，不需要槓桿臂（CUE_STICK_GRIP_TO_TIP）換算，
量測時直接讀 CueStick 剛體本身的線速度即可（純平移，無轉動分量）。

不含球檯、不含手臂移動——手臂固定在 TableRobotManager 建構時的初始姿態，
只驗證推桿機構本身。

2026-09-03 除錯記錄：第一版執行後 CueSlideJoint 位置在幾秒內飄到
1000+ 公尺外，看起來完全失控。根本原因：新建立的 PrismaticJoint（見
core/models/table_robot_manager.py 的 UR10e 分支）預設沒有任何 PhysX
drive（跟其餘手臂關節不同——那些是官方 URDF 轉換出來的，本來就帶
drive），Articulation.set_dof_position_targets()/set_dof_velocity_targets()
對這個 DOF 完全沒有作用力可循，joint 在無約束下自由漂移。修法：
core/ports/stage_api.py create_prismatic_joint() 新增
drive_stiffness/drive_damping/drive_max_force 參數，TableRobotManager
建立 CueSlideJoint 時明確傳入非零增益（UsdPhysics.DriveAPI，instance
名稱 "linear"）。修好後：後擺 28 步收斂到 -0.14853（目標 -0.15），
揮桿量到接觸瞬間桿尖總速度 1.7939 m/s，對 required_tip_speed=1.5116 m/s
達成率 118.7%，PASS（見下方沿用 test_ur3e_human_pose_swing_speed.py 的
「事後從歷史紀錄挑最接近目標的 tick」方法論，避免衝過頭造成的假象）。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_ur10e_actuator_swing_isolated.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CUE_BALL_SPEED = 1.995
_PHYSICS_DT = 1.0 / 60.0
_SETTLE_STEPS = 30
_EXTRA_STEPS_AFTER_T = 30


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.ur10e_cue_slide_controller import (
        Ur10eCueSlideController,
        _quintic_velocity,
    )

    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services import swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    base_path = "/World/Ur10eActuatorIsolatedTest"
    robot_manager = TableRobotManager(
        (0.0, 0.0, 0.0), base_path, stage_api, None, UR10eRobot,
    )
    robot_prim_path = robot_manager.get_robot_prim_path()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    print(f"[actuator] robot_prim_path={robot_prim_path}")
    print(f"[actuator] cue_stick_prim_path={cue_stick_prim_path}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=robot_prim_path)
    for _ in range(5):
        simulation_app.update()

    dof_names = list(articulation.dof_names)
    slide_dof_index = dof_names.index("CueSlideJoint")
    print(f"[actuator] dof_names={dof_names}  slide_dof_index={slide_dof_index}")

    cue_stick_rigid_prim = RigidPrim(paths=cue_stick_prim_path)

    initial_slide_position = float(np.asarray(articulation.get_dof_positions())[0][slide_dof_index])
    print(f"[actuator] 初始 CueSlideJoint 位置={initial_slide_position:.5f}（預期接近 0，見 align_prim_to_target 慣例）")

    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    backswing_position = -swing_trajectory_calculator.DEFAULT_BACKSWING_DISTANCE_M
    print(f"[actuator] cue_ball_speed={_CUE_BALL_SPEED}  required_tip_speed={required_tip_speed:.4f} m/s")
    print(f"[actuator] backswing_position={backswing_position:.4f} m")

    controller = Ur10eCueSlideController(articulation)
    controller.move_stroke(backswing_position, required_tip_speed)

    # ---- Phase 1：退到後擺位置 ----
    backswing_steps = 0
    while controller._phase == "backswing" and backswing_steps < 300:
        controller.step(_PHYSICS_DT)
        simulation_app.update()
        backswing_steps += 1
    live_slide_after_backswing = float(np.asarray(articulation.get_dof_positions())[0][slide_dof_index])
    print(f"[actuator] 後擺 {backswing_steps} 步後 CueSlideJoint 位置={live_slide_after_backswing:.5f}（目標 {backswing_position:.5f}）")

    if controller._phase != "strike":
        print("[actuator] FAIL：後擺階段沒有正常切到 strike 階段")
        return

    c3, c4, c5, T = controller._quintic
    print(f"[actuator] quintic T={T:.4f}s")

    # ---- Phase 2：揮桿，逐 tick 記錄歷史，事後挑「滑軌位置最接近 0（接觸點）」那一個 tick ----
    num_steps = int(T / _PHYSICS_DT) + _EXTRA_STEPS_AFTER_T
    history = []
    for step in range(num_steps):
        t = min(step * _PHYSICS_DT, T)
        # 揮桿收斂的那一 tick，控制器會在 step() 裡就切進 post_strike_retract
        # （切回 position 模式、位置目標指向後擺位置），這一 tick 的物理是
        # 「開始煞車」而不是揮桿，桿尖速度不能拿來當接觸速度。只採計 step()
        # 前後都還在 strike 階段的 tick。
        phase_before = controller._phase
        controller.step(_PHYSICS_DT)
        simulation_app.update()

        live_slide_position = float(np.asarray(articulation.get_dof_positions())[0][slide_dof_index])
        tip_velocity, _ = cue_stick_rigid_prim.get_velocities()
        tip_velocity = np.asarray(tip_velocity[0])

        position_error = abs(live_slide_position - 0.0)
        if phase_before == "strike" and controller._phase == "strike":
            history.append((position_error, tip_velocity.copy(), live_slide_position))

        if step % 10 == 0 or step >= num_steps - 5:
            qdot_ref = _quintic_velocity(c3, c4, c5, t)
            print(f"[actuator] step={step} t={t:.3f} qdot_ref={qdot_ref:.4f} slide_position={live_slide_position:.5f} position_error={position_error:.5f}")

    history.sort(key=lambda h: h[0])
    position_error_at_contact, tip_velocity_at_contact, slide_position_at_contact = history[0]

    tip_speed_total = float(np.linalg.norm(tip_velocity_at_contact))
    print(f"[actuator] 接觸瞬間 slide_position={slide_position_at_contact:.5f}（誤差 {position_error_at_contact:.5f}，挑全程最接近 0 的 tick）")
    print(f"[actuator] 接觸瞬間桿尖速度向量={tip_velocity_at_contact.tolist()}")
    print(f"[actuator] 接觸瞬間桿尖總速度={tip_speed_total:.4f} m/s")

    # 達成率用「沿球桿軸向（grip→tip，= CueStick 本地 +Y）的帶號投影」，不是
    # 速度向量長度：長度是純量，方向裝反也會過關——CueSlideJoint 的 body0/
    # body1 順序寫反時，推桿實際是往後退，這個測試卻仍回報 118.7% PASS，
    # 直到真實球檯測試才發現桿子是從母球另一側往回抽。
    _, cue_orientation = cue_stick_rigid_prim.get_world_poses()
    cue_orientation = np.asarray(cue_orientation[0], dtype=float)
    w, x, y, z = cue_orientation
    q_xyz = np.array([x, y, z])
    local_axis = np.array([0.0, 1.0, 0.0])
    t = 2.0 * np.cross(q_xyz, local_axis)
    cue_axis_world = local_axis + w * t + np.cross(q_xyz, t)

    forward_speed = float(np.dot(tip_velocity_at_contact, cue_axis_world))
    print(f"[actuator] 球桿軸向（世界座標，grip→tip）={cue_axis_world.tolist()}")
    print(f"[actuator] 沿軸向帶號速度={forward_speed:.4f} m/s"
          f"（負值代表推桿方向裝反）")
    print(f"[actuator] required_tip_speed={required_tip_speed:.4f} m/s  達成率={100 * forward_speed / required_tip_speed:.1f}%")

    speed_pass = forward_speed >= 0.9 * required_tip_speed
    if speed_pass:
        print("[actuator] PASS：推桿機構單獨達成 >=90% 目標桿尖速度，且方向正確")
    elif forward_speed < 0:
        print("[actuator] FAIL：推桿方向相反——桿尖朝遠離母球的方向移動，"
              "檢查 create_prismatic_joint() 的 body0/body1 順序")
    else:
        print("[actuator] FAIL：推桿機構單獨未達 90% 目標桿尖速度")

    # ---- Phase 3：揮桿後沿原軸縮回（決策 5 的 post_strike_retract 階段） ----
    # 不縮回的話球桿會停在 q≈0（母球原本待的位置），母球撞球堆彈回來會再撞
    # 上球桿，構成二次擊球。這裡驗證縮回本身能不能在時限內走完。
    print("[actuator] --- Phase 3：揮桿後縮回 ---")
    retract_steps = 0
    while not controller.is_motion_complete() and retract_steps < 400:
        controller.step(_PHYSICS_DT)
        simulation_app.update()
        retract_steps += 1

        if retract_steps % 10 == 0 or retract_steps <= 3:
            live_slide_position = float(np.asarray(articulation.get_dof_positions())[0][slide_dof_index])
            live_target = float(np.asarray(articulation.get_dof_position_targets())[0][slide_dof_index])
            stiffnesses, dampings = articulation.get_dof_gains()
            live_stiffness = float(np.asarray(stiffnesses)[0][slide_dof_index])
            live_damping = float(np.asarray(dampings)[0][slide_dof_index])
            live_effort = float(np.asarray(articulation.get_dof_efforts())[0][slide_dof_index])
            print(f"[actuator] retract step={retract_steps} phase={controller._phase} "
                  f"q={live_slide_position:.5f} 位置目標={live_target:.5f} "
                  f"stiffness={live_stiffness:.4g} damping={live_damping:.4g} effort={live_effort:.4g}")

    final_slide_position = float(np.asarray(articulation.get_dof_positions())[0][slide_dof_index])
    print(f"[actuator] 縮回 {retract_steps} 步後 CueSlideJoint 位置={final_slide_position:.5f}"
          f"（目標 {backswing_position:.5f}）did_last_motion_timeout={controller.did_last_motion_timeout()}")

    retract_pass = abs(final_slide_position - backswing_position) <= 0.005
    if retract_pass:
        print("[actuator] PASS：揮桿後球桿沿原軸縮回後擺位置")
    else:
        print("[actuator] FAIL：揮桿後球桿沒有縮回後擺位置，球桿仍擋在母球原位")

    print(f"[actuator] 總結：推桿速度 {'PASS' if speed_pass else 'FAIL'}  縮回 {'PASS' if retract_pass else 'FAIL'}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
