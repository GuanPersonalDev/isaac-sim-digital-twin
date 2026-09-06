"""
scripts/diagnose_gui_joint_warning.py — 一次性診斷：使用者在真實 GUI 執行
`billiard_digital_twin` extension 時，log 出現兩筆
「Updating joint local poses in articulations is not supported after
simulation start (Joint .../CueSlideJoint)」警告，且手臂完全不動。

headless 驗收腳本（test_ur10e_table_flat.py 等）從未呼叫
`SimulationManager.setup_simulation()`，且是「先建 TableRobotManager（含
CueSlideJoint）→ 才 timeline.play()」；production 的
`BilliardExtension._billiard_init()` 呼叫順序不同：
`SimulationManager.setup_simulation(dt=1/60)` → 建 Training 桌 →
建 Demo 桌（含 CueSlideJoint）→ 都還在 timeline.play() 之前。

這支腳本複製 production 的呼叫順序（headless），檢查：
1. 一樣的警告會不會重現
2. CueSlideJoint 的 body0/body1/axis/drive 屬性最終有沒有被寫壞
3. 接上 Ur10eCueSlideController 之後，這個關節能不能正常回應位置/速度指令
   （直接複製 test_ur10e_actuator_swing_isolated.py 的驗證方式）

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/diagnose_gui_joint_warning.py

結論（2026-09-06）：警告本身在這個 headless 環境沒有重現（只在真實
isaacsim.exp.full.kit GUI 才出現，懷疑是該 profile 的 eager 物理解析跟
輕量 SimulationApp 行為不同），但這支腳本查證出的三件事都是好消息：
CueSlideJoint 的 body0/body1/axis/drive 屬性完全正確，關節本身正常回應
指令。真正讓「手臂完全不動」的根因後來在
scripts/diagnose_production_tick.py 抓到，是完全不同的 bug（見
docs/CHANGELOG.md「did_last_motion_timeout() 對 UR10e 提早誤判逾時」）。
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaacsim.core.simulation_manager import SimulationManager

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    # === 複製 production 的呼叫順序 ===
    print("[diag] SimulationManager.setup_simulation(dt=1/60) ...")
    SimulationManager.setup_simulation(dt=1 / 60)

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    # 先建一張「Training 桌」（跟 production _on_training_toggle() 一樣，
    # 沒有機器人，純球檯+球——這一步在 production 會在 Demo 桌之前執行）。
    print("[diag] 建立 Training 桌（無機器人）...")
    training_table = BilliardTable("/World/Table_0", stage_api, material_api, rigid_body_api, (2.6, 2.6))

    # 再建 Demo 桌 + TableRobotManager（這裡會建立 CueSlideJoint）。
    print("[diag] 建立 Demo 桌 + TableRobotManager（建立 CueSlideJoint）...")
    demo_table_path = "/World/Table_Demo"
    demo_table = BilliardTable(demo_table_path, stage_api, material_api, rigid_body_api, (0, 0))

    robot_prim_path = UR10eRobot.get_prim_path(demo_table_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(demo_table_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    robot_manager = TableRobotManager(
        demo_table.get_table_center(), demo_table_path, stage_api, articulation_api, UR10eRobot,
    )
    robot_arm = robot_manager.get_robot()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()

    # 這裡才 Play——跟 production 一樣，joint 是在 Play **之前**建立的。
    print("[diag] timeline.play() ...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    demo_table.get_table_ball_set().place_ball(0, 0.0, 0.5)
    for _ in range(5):
        simulation_app.update()

    # === 查證 1：CueSlideJoint 的 USD 屬性有沒有被寫壞 ===
    joint_prim = stage.GetPrimAtPath(cue_stick_prim_path + "/CueSlideJoint")
    print(f"[diag] CueSlideJoint prim 存在={joint_prim.IsValid()}")
    joint = UsdPhysics.PrismaticJoint(joint_prim)
    print(f"[diag] Body0Rel targets={joint.GetBody0Rel().GetTargets()}")
    print(f"[diag] Body1Rel targets={joint.GetBody1Rel().GetTargets()}")
    print(f"[diag] AxisAttr={joint.GetAxisAttr().Get()}")
    print(f"[diag] LocalPos0Attr={joint.GetLocalPos0Attr().Get()}")
    print(f"[diag] LocalRot0Attr={joint.GetLocalRot0Attr().Get()}")
    print(f"[diag] LocalPos1Attr={joint.GetLocalPos1Attr().Get()}")
    print(f"[diag] LocalRot1Attr={joint.GetLocalRot1Attr().Get()}")
    drive_api = UsdPhysics.DriveAPI(joint_prim, "linear")
    print(f"[diag] Drive stiffness={drive_api.GetStiffnessAttr().Get()} "
          f"damping={drive_api.GetDampingAttr().Get()} "
          f"maxForce={drive_api.GetMaxForceAttr().Get()}")

    # === 查證 2：articulation_api.initialize() 之後這個關節能不能正常動 ===
    print("[diag] articulation_api.initialize() ...")
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    dof_names = list(articulation_api._articulation.dof_names)
    print(f"[diag] dof_names={dof_names}")
    slide_dof_index = dof_names.index("CueSlideJoint")
    initial_slide_position = float(
        np.asarray(articulation_api._articulation.get_dof_positions())[0][slide_dof_index]
    )
    print(f"[diag] 初始 CueSlideJoint 位置={initial_slide_position:.5f}（預期接近 0）")

    # 直接下一個位置指令，看關節真的會不會動。
    target = -0.15
    positions = np.asarray(articulation_api._articulation.get_dof_positions())[0].copy()
    positions[slide_dof_index] = target
    articulation_api._articulation.switch_dof_control_mode("position")
    articulation_api._articulation.set_dof_position_targets(positions[None, :])

    moved = False
    for step in range(180):
        simulation_app.update()
        current = float(
            np.asarray(articulation_api._articulation.get_dof_positions())[0][slide_dof_index]
        )
        if step % 30 == 0:
            print(f"[diag] step={step} CueSlideJoint 位置={current:.5f}（目標 {target}）")
        if abs(current - target) <= 0.005:
            moved = True
            print(f"[diag] step={step} 已收斂到目標")
            break

    if moved:
        print("[diag] PASS：CueSlideJoint 屬性正常、位置指令生效——照 production 順序建立也沒問題")
    else:
        print("[diag] FAIL：CueSlideJoint 沒有回應位置指令，照 production 順序建立會讓關節壞掉")

    # === 查證 3：手臂 6 個關節（不是 CueSlideJoint）能不能動——對照使用者
    # 回報的「手臂不會動」，這個現象也可能跟 CueSlideJoint 無關，是手臂
    # 本體完全沒收到任何指令。===
    print("[diag] 呼叫 RobotArm.reset()（模擬正式 RESET）...")
    robot_arm.reset()
    reset_steps = 0
    while not robot_arm.is_reset_complete() and reset_steps < 4000:
        simulation_app.update()
        reset_steps += 1
    print(f"[diag] RESET steps={reset_steps} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")
    arm_dof_positions = np.asarray(articulation_api._articulation.get_dof_positions())[0]
    print(f"[diag] RESET 後六個手臂關節角度={arm_dof_positions[:6].tolist()}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[diag] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
