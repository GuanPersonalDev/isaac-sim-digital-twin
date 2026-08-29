"""
scripts/minimal_repro_cue_impact.py — 最小重現案例：完全不接機器人/
articulation，只用一個獨立的自由剛體圓柱（跟 assets/ball_stick.usda 的
Cylinder 同樣尺寸：radius=0.01, length=1.5），直接用
`RigidPrim.set_velocities()` 給定線速度去撞一顆真實的母球，量測碰撞
衝量／母球速度有沒有正常反應。

背景：docs/issue-180-reachability-analysis.md 第十六節記錄
`ArticulationAPIImpl.move_swing()` 的桿尖已經確認幾何上有精準逼近母球
（12.6mm，遠小於球半徑），姿態也確認跟球桿剛體完全同步，但整個高速揮桿
過程 PhysX 從未觸發任何新的碰撞事件（唯一一次接觸發生在 AIM 收斂時的
緩慢碰觸，不是揮桿本身）。已排除的假說：solver iteration count、關節
drive stiffness、CCD、姿態不同步。這支腳本要隔離「桿子掛在受驅動
articulation 上（FixedJoint+高 stiffness 關節鏈）」是不是真正變因——
如果一個完全自由、不受任何 articulation 牽制的剛體用同樣速度撞球也一樣
量不到衝量，代表問題出在更基礎的 PhysX 碰撞/材質設定，不是 articulation
特有的行為。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/minimal_repro_cue_impact.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_BALL_RADIUS = 0.028575
_CYLINDER_RADIUS = 0.01
_CYLINDER_LENGTH = 1.5
_APPROACH_SPEED = float(os.environ.get("REPRO_SPEED", "1.0"))
_MAX_STEPS = 300


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf
    from isaacsim.core.experimental.prims import RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from core.models.billiard_table import BilliardTable
    from core.models.contact_event import ContactEvent
    from core.services.asset_utility import CUE_STICK_PATH

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/MinimalReproTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    ball_xy = (0.0, 0.0)
    ball_positions = {i: (5.0 + i * 0.2, 5.0) for i in range(10)}
    ball_positions[0] = ball_xy
    table.get_table_ball_set().build(ball_positions)
    ball_prim_path = table.get_table_ball_set().get_ball_prim_paths()[0]

    # 獨立的自由剛體圓柱：直接參照 ball_stick.usda 的 CueStick（跟真實球桿
    # 完全同一份幾何/質量設定），但**不**呼叫 align_prim_to_target／
    # create_fixed_joint，不掛任何機器人，是完全自由的剛體。
    free_cue_path = "/World/FreeCueStick"
    stage_api.create_reference_prim(free_cue_path, CUE_STICK_PATH)
    ball_center_z = 0.0 + _BALL_RADIUS
    # 起始位置：桿尖（局部 +Y 方向 1.35m 處）對準球心正上方一小段距離，
    # 沿 -Y 方向移動撞向球。圓柱體原點需要位在球心後方
    # 1.35 + 一段緩衝距離。
    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    approach_distance = 0.3
    start_origin_y = ball_center_z * 0 + (0.0 - (CUE_STICK_GRIP_TO_TIP + approach_distance))
    # 沿 +Y 移動撞向 Y=0 處的球（局部 +Y 是桿尖方向，讓圓柱體 orientation
    # 維持單位四元數，桿尖自然指向 +Y）。
    stage_api.set_prim_translate(free_cue_path, 0.0, start_origin_y, ball_center_z)

    contacts: list[ContactEvent] = []
    physics_api.enable_contact_reporting(ball_prim_path)
    physics_api.enable_contact_reporting(free_cue_path)
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    free_cue_rigid_prim = RigidPrim(paths=free_cue_path)
    ball_rigid_prim = RigidPrim(paths=ball_prim_path)

    positions, orientations = free_cue_rigid_prim.get_world_poses()
    print(f"[repro] free_cue initial pos={positions[0].list()} orient={orientations[0].list()}")

    free_cue_rigid_prim.set_velocities(
        linear_velocities=[[0.0, _APPROACH_SPEED, 0.0]],
        angular_velocities=[[0.0, 0.0, 0.0]],
    )

    max_ball_speed = 0.0
    max_ball_speed_step = -1
    for step in range(_MAX_STEPS):
        simulation_app.update()
        # 每步都重新下達速度指令（跟差動 IK 的 set_dof_velocity_targets()
        # 邏輯一致：velocity target 需要持續下達，不是一次性衝量）。
        free_cue_rigid_prim.set_velocities(
            linear_velocities=[[0.0, _APPROACH_SPEED, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
        )
        linear_vel, _angular_vel = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(linear_vel)[0]))
        if ball_speed > max_ball_speed:
            max_ball_speed = ball_speed
            max_ball_speed_step = step
        if step % 20 == 0:
            cue_pos, _cue_orient = free_cue_rigid_prim.get_world_poses()
            print(f"[repro] step={step} cue_origin_y={cue_pos[0].list()[1]:.4f} ball_speed={ball_speed:.4f}")

    print(f"[repro] max_ball_speed={max_ball_speed:.4f} m/s at step={max_ball_speed_step}")
    print(f"[repro] contact_events_count={len(contacts)}")
    for i, c in enumerate(contacts):
        print(f"[repro] contact[{i}]: {c.collider_path_a} <-> {c.collider_path_b}  impulse={c.impulse}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
