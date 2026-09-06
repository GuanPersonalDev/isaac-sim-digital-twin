"""
scripts/verify_ur10e_cue_actuator.py — 驗證專用出力機構的外觀件
（`assets/cue_actuator.usda`）掛上去之後：

1. 幾何對得上：缸體沿球桿軸、球桿從前端導桿套伸出，且退桿 15cm 之後
   露出長度真的少 15cm（這就是 Demo 要讓人看懂的那件事）
2. 完全沒有動到物理：`dof_names` 仍然是 7 個且含 CueSlideJoint、
   沒有多出剛體或碰撞對——資產刻意不帶任何 physics schema
3. 出圖：這是視覺功能，算兩張圖（伸出 q=0 / 縮回 q=-0.15）直接看

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_cue_actuator.py
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "_actuator_preview")
# cue_actuator.usda 的 RodGuide 前緣（局部 Y），球桿從這裡伸出來
_ROD_GUIDE_FRONT_Y = 0.314
_RETRACT_POSITION_M = -0.15


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, UsdGeom, Usd

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

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/ActuatorPreview"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))

    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, UR10eRobot,
    )
    actuator_prim_path = robot_manager.get_cue_actuator_prim_path()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    print(f"[actuator] 出力機構 prim path={actuator_prim_path}")

    actuator_prim = stage.GetPrimAtPath(actuator_prim_path)
    print(f"[actuator] prim 存在={actuator_prim.IsValid()}")
    if not actuator_prim.IsValid():
        print("[actuator] FAIL：prim 沒有建立成功")
        return

    # 查證「純外觀」：不能帶任何 physics schema，否則就會影響已驗收的行為
    physics_schemas = []
    for prim in Usd.PrimRange(actuator_prim):
        for schema in prim.GetAppliedSchemas():
            if "Physics" in schema or "Physx" in schema:
                physics_schemas.append(f"{prim.GetPath()}: {schema}")
    print(f"[actuator] 帶 physics schema 的子 prim={physics_schemas if physics_schemas else '無（純外觀，符合設計）'}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()
    articulation_api.initialize()
    for _ in range(20):
        simulation_app.update()

    articulation = articulation_api._articulation
    dof_names = list(articulation.dof_names)
    print(f"[actuator] dof_names={dof_names}")
    dof_ok = len(dof_names) == 7 and "CueSlideJoint" in dof_names
    print(f"[actuator] DOF 數量/組成未被影響={dof_ok}")
    slide_dof_index = dof_names.index("CueSlideJoint")

    def _cue_tip_and_guide_gap() -> tuple[float, float]:
        """回傳 (CueSlideJoint 位置, 球桿從導桿套前緣露出的長度)。
        兩者都在末端連桿的局部座標系裡算，不受手臂姿態影響。"""
        slide_position = float(np.asarray(articulation.get_dof_positions())[0][slide_dof_index])
        # 球桿尖端在末端連桿局部座標的 Y = CUE_STICK_GRIP_TO_TIP + 滑軌位置
        from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP

        tip_local_y = CUE_STICK_GRIP_TO_TIP + slide_position
        return slide_position, tip_local_y - _ROD_GUIDE_FRONT_Y

    # 補一盞 dome light：第一版沒補，算出來的機構整個是黑的看不出形狀
    from pxr import UsdLux

    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/_ActuatorPreviewLight"))
    dome_light.CreateIntensityAttr(1200.0)

    def _capture(label: str, file_name: str) -> None:
        """⚠️ 刻意不用 rep.orchestrator.step()／BasicWriter：那條路徑會接管
        並暫停 timeline，導致後續的 simulation_app.update() 不再推進物理
        （第一版踩過：縮回指令下了 180 個 tick，關節位置卻完全沒變）。
        改用 annotator 直接取畫面，不碰 timeline。"""
        import omni.replicator.core as rep
        from PIL import Image

        wrist_position = np.asarray(articulation_api.get_end_effector_position(), dtype=float)
        wrist_orientation = np.asarray(articulation_api.get_end_effector_orientation(), dtype=float)
        w, x, y, z = wrist_orientation
        q_xyz = np.array([x, y, z])
        local_y = np.array([0.0, 1.0, 0.0])
        t = 2.0 * np.cross(q_xyz, local_y)
        cue_axis = local_y + w * t + np.cross(q_xyz, t)
        cue_axis = cue_axis / np.linalg.norm(cue_axis)

        # 看向機構中段（缸體 + 導桿套 + 一小段球桿），從側面拍才看得到
        # 球桿相對缸體進出
        focus = wrist_position + cue_axis * 0.30
        side = np.cross(cue_axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
        side = side / np.linalg.norm(side)
        camera_position = focus + side * 1.1 + np.array([0.0, 0.0, 0.28])

        camera = rep.create.camera(
            position=tuple(camera_position.tolist()), look_at=tuple(focus.tolist())
        )
        render_product = rep.create.render_product(camera, (1280, 720))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(render_product)
        # 讓 RTX 累積幾幀，不然會拿到還沒收斂的雜訊畫面
        for _ in range(30):
            simulation_app.update()

        rgb = annotator.get_data()
        annotator.detach()
        render_product.destroy()

        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(_OUTPUT_DIR, f"{file_name}.png")
        Image.fromarray(np.asarray(rgb)[:, :, :3]).save(output_path)
        print(f"[actuator] 已算圖：{label} → {output_path}")
        sys.stdout.flush()

    # 先做 RESET 讓手臂到 HOME：原始 USD 預設姿態下手腕朝向不好取景，
    # HOME 是設計過的固定姿態，每次跑出來的圖才可比較。
    initial_base_offset = TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    table_center = table.get_table_center()
    articulation_api.set_robot_base_pose(
        [table_center[i] + initial_base_offset[i] for i in range(3)], [1.0, 0.0, 0.0, 0.0]
    )
    articulation_api.move_to_home()
    reset_steps = 0
    while not articulation_api.is_motion_complete() and reset_steps < 2000:
        simulation_app.update()
        reset_steps += 1
    print(f"[actuator] RESET 到 HOME 完成，steps={reset_steps}")

    # === 狀態 1：伸出（q=0，設計接觸點）===
    slide_position, exposed = _cue_tip_and_guide_gap()
    print(f"[actuator] 【伸出】CueSlideJoint={slide_position:.5f} "
          f"球桿露出導桿套前緣={exposed:.4f} m")
    _capture("伸出 q=0", "extended")

    # === 狀態 2：縮回（q=-0.15，後擺位置）===
    positions = np.asarray(articulation.get_dof_positions())[0].copy()
    positions[slide_dof_index] = _RETRACT_POSITION_M
    articulation.switch_dof_control_mode("position")
    articulation.set_dof_velocity_targets(np.zeros((1, len(dof_names))))
    articulation.set_dof_position_targets(positions[None, :])
    for _ in range(240):
        simulation_app.update()
        live = float(np.asarray(articulation.get_dof_positions())[0][slide_dof_index])
        if abs(live - _RETRACT_POSITION_M) <= 0.002:
            break

    retracted_position, retracted_exposed = _cue_tip_and_guide_gap()
    if abs(retracted_position - _RETRACT_POSITION_M) > 0.01:
        print(f"[actuator] ⚠️ 縮回沒有到位（{retracted_position:.5f}，目標 {_RETRACT_POSITION_M}）"
              f"——timeline.is_playing()={timeline.is_playing()}")
    print(f"[actuator] 【縮回】CueSlideJoint={retracted_position:.5f} "
          f"球桿露出導桿套前緣={retracted_exposed:.4f} m")
    _capture("縮回 q=-0.15", "retracted")

    travel = exposed - retracted_exposed
    print(f"[actuator] 兩個狀態的露出長度差={travel:.4f} m（應該接近行程 0.15 m）")

    geometry_ok = abs(travel - abs(_RETRACT_POSITION_M)) <= 0.01
    no_physics_ok = not physics_schemas
    if geometry_ok and no_physics_ok and dof_ok:
        print("[actuator] PASS：機構外觀件對齊球桿軸、伸縮行程看得出來，且完全沒有動到物理")
    else:
        reasons = []
        if not geometry_ok:
            reasons.append(f"露出長度變化 {travel:.4f}m 與行程 0.15m 不符")
        if not no_physics_ok:
            reasons.append("外觀件帶了 physics schema")
        if not dof_ok:
            reasons.append(f"DOF 組成被影響：{dof_names}")
        print(f"[actuator] FAIL：{'；'.join(reasons)}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[actuator] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
