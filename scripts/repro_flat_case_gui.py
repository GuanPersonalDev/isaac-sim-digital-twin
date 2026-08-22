"""
scripts/repro_flat_case_gui.py — 在 Isaac Sim GUI（非 headless）裡重現
flat 案例殘留誤差問題，方便用視窗內建的 Physics Debug 工具即時觀察卡住
當下的關節受力/接觸狀態，而不是繼續 headless 腳本盲測參數。

用法（可切換 _CUE_BALL 選兩個已知失敗案例之一，或改成 (0.0, 0.4) 當作
「正常收斂」的對照組）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/repro_flat_case_gui.py

跑起來之後視窗不會自動關閉，會停在收斂/卡住的狀態，讓你在 GUI 裡自由操作
（暫停、單步、開 Physics Debug 視覺化）。要結束時直接關視窗即可。

建議在 GUI 裡檢查的項目（詳見 docs/issue-flat-case-residual-error.md
「建議後續方向」）：
  1. Window > Physics > Physics Inspector 或 Debug 面板打開，觀察
     wam_wrist_yaw_joint / wam_shoulder_pitch_joint 的即時 drive force、
     drive error。
  2. 開 Joint 視覺化（Physics Debug Visualization 裡的 Joints），看有沒有
     震盪（跳動的箭頭/彩色標記），還是真的靜止不動。
  3. 檢查 wam_wrist_palm_stump_link 的質量（mass=0.000001，幾乎是零質量的
     「假 link」，只用來當掛載點）跟上游 wam_upper_arm_link（mass=2.2）/
     wam_forearm_link（mass=0.5）之間的質量比是否被 PhysX 判定為
     ill-conditioned（Isaac Sim 的 Physics Debug 有時會在 console 印出
     mass ratio 過大的警告）——這是排除清單裡還沒測過的新假設。
  4. 打開 Contact Visualization，即使 physics_api 的 contact reporting
     沒偵測到碰撞事件，也直接用肉眼確認球桿/手臂有沒有跟環境或自己
     有微幅穿模（interpenetration）但因為太輕微沒觸發 contact report
     threshold。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

_TARGET_BALL = (0.0, 0.635)
# 已知失敗案例：(-0.25, -0.1) 或 (0.25, -0.1)；(0.0, 0.4) 是正常收斂的對照組。
_CUE_BALL = (-0.25, -0.1)


def _shot_angle_deg(cue_ball, target):
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.services.base_placement_calculator import CANONICAL_REST_JOINTS, compute_base_pose

    from scripts.scan_elevated_bridge_approach import compute_tilted_wrist_pose

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/ReproTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_z = 0.0
    ball_radius = 0.028575

    robot_prim_path = BarrettWamRobot.get_prim_path(table_base_path)
    end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, BarrettWamRobot
    )
    robot = robot_manager.get_robot()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    angle_deg = _shot_angle_deg(_CUE_BALL, _TARGET_BALL)
    base_position, base_yaw_rad = compute_base_pose(_CUE_BALL[0], _CUE_BALL[1], angle_deg, table_z=table_z)
    robot.reposition(base_position)
    for _ in range(30):
        simulation_app.update()

    wrist0, orientation0, tilt0, crossing0 = compute_tilted_wrist_pose(
        _CUE_BALL, angle_deg, table_z, ball_radius, roll_rad=0.0
    )
    joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
    print(f"cue_ball={_CUE_BALL}  base_position={base_position}  base_yaw_rad={base_yaw_rad:.4f}")
    print(f"joint_targets={joint_targets}")
    print(f"wrist0（末端目標，世界座標）={wrist0.tolist()}")
    print(f"機器人 prim 路徑：{robot_prim_path}（可在 Stage 視窗展開檢查各 link）")
    print("開始驅動手臂到目標關節角，請在視窗裡打開 Physics Debug 面板觀察...")

    articulation_api.move_to_joint_position(joint_targets, wrist0.tolist())

    # 不自動關閉：持續推進模擬，讓使用者在 GUI 裡即時觀察卡住當下的狀態。
    # 每 60 步（約 1 秒）印一次目前誤差跟關節角，方便對照 GUI 上看到的數值。
    step = 0
    while simulation_app.is_running():
        simulation_app.update()
        if step % 60 == 0:
            current = np.array(articulation_api.get_end_effector_position())
            err = float(np.linalg.norm(current - wrist0))
            joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
            complete = articulation_api.is_motion_complete()
            print(f"step={step:5d} err={err*1000:.2f}mm complete={complete} joints={np.round(joints, 4).tolist()}")
        step += 1


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    try:
        _run()
    finally:
        simulation_app.close()
