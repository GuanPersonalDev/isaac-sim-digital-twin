"""
scripts/diagnose_move_swing.py — 驗證新實作的 `ArticulationAPIImpl.
move_swing()`（揮桿專用速度最優控制，見 docs/issue-180-reachability-
analysis.md 第十六節）：對最難的 Kitchen 案例（y=-0.9382125，24 個 roll
候選裡沒有任何一個在「姿態完全鎖死」下能達到所需速度）用真實母球物理
速度（不透過任何軟體完成判定）驗收，取代舊的 compute_swing_waypoints()
+ move_through_poses() 兩段式呼叫。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/diagnose_move_swing.py
"""

import logging
import math
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_CUE_BALL = (
    float(os.environ.get("DIAG_CUE_BALL_X", "0.0")),
    float(os.environ.get("DIAG_CUE_BALL_Y", "-0.9382125")),
)
_SHOT_ANGLE_DEG = float(os.environ.get("DIAG_SHOT_ANGLE_DEG", "0.0"))
_POSITION_OFFSET = [
    float(os.environ.get("DIAG_POSITION_OFFSET_V", "0.0")),
    float(os.environ.get("DIAG_POSITION_OFFSET_H", "0.0")),
]
_CUE_BALL_SPEED = 1.995
_AIM_MAX_STEPS = 4000
_SWING_MAX_STEPS = 400
_BACKSWING_DISTANCE_M = 0.15
_ORIENTATION_GAIN = float(os.environ.get("SWING_ORIENTATION_GAIN", "1.0"))
_MAX_ANGULAR_SPEED = float(os.environ.get("SWING_MAX_ANGULAR_SPEED", "1.0"))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd
    from isaacsim.core.experimental.prims import RigidPrim

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("isaac_sim_impl_6_0.articulation_api_impl").setLevel(logging.DEBUG)

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.contact_event import ContactEvent
    from core.services.base_placement_calculator import (
        CANONICAL_FLAT_ORIENTATION, CANONICAL_REST_JOINTS, compute_base_pose,
        compute_canonical_wrist_position,
    )
    from core.services import cue_pose_calculator, swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    # ⚠️ 實驗性檢查：PhysX TGS 求解器預設只在每個 frame 開頭套用一次
    # 外力（含 gravity），`physxScene:enableExternalForcesEveryIteration`
    # 打開後會在每個 TGS 內部子步都按比例套用外力／articulation joint
    # efforts，理論上能改善「快速通過的碰撞在單一子步內來不及被求解器
    # 正確吸收」這類問題。用環境變數 SWING_EXTERNAL_FORCES_EVERY_ITER
    # 測試。
    if os.environ.get("SWING_EXTERNAL_FORCES_EVERY_ITER"):
        from pxr import PhysxSchema
        scene_prim = stage.GetPrimAtPath("/PhysicsScene")
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        attr = physx_scene_api.GetEnableExternalForcesEveryIterationAttr()
        if not attr:
            attr = physx_scene_api.CreateEnableExternalForcesEveryIterationAttr()
        print(f"[diag] enableExternalForcesEveryIteration BEFORE={attr.Get()}")
        attr.Set(True)
        print(f"[diag] enableExternalForcesEveryIteration AFTER={attr.Get()}")

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/DiagnoseMoveSwingTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    ball_positions = {i: (5.0 + i * 0.2, 5.0) for i in range(10)}
    ball_positions[0] = _CUE_BALL
    table.get_table_ball_set().build(ball_positions)

    robot_prim_path = BarrettWamRobot.get_prim_path(table_base_path)
    end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, BarrettWamRobot
    )
    robot = robot_manager.get_robot()

    ball_prim_path = table.get_table_ball_set().get_ball_prim_paths()[0]
    ball_rigid_prim = RigidPrim(paths=ball_prim_path)

    # 直接查詢執行時的碰撞相關 USD attribute，確認球桿的 Cylinder／母球
    # 有沒有真的開啟碰撞、有沒有被意外過濾。
    cue_stick_cylinder_path = robot_manager.get_cue_stick_prim_path() + "/Cylinder"
    cylinder_prim = stage.GetPrimAtPath(cue_stick_cylinder_path)
    ball_prim_for_check = stage.GetPrimAtPath(ball_prim_path + "/Ball")
    for label, prim in [("CueStick/Cylinder", cylinder_prim), ("Ball", ball_prim_for_check)]:
        if not prim.IsValid():
            print(f"[diag] {label} prim 不存在：{prim.GetPath()}")
            continue
        collision_enabled_attr = prim.GetAttribute("physics:collisionEnabled")
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
        filtered_pairs_rel = prim.GetRelationship("physics:filteredPairs")
        filtered_targets = filtered_pairs_rel.GetTargets() if filtered_pairs_rel else []
        print(
            f"[diag] {label}: path={prim.GetPath()} has_CollisionAPI={has_collision_api} "
            f"collisionEnabled={collision_enabled_attr.Get() if collision_enabled_attr and collision_enabled_attr.IsValid() else 'N/A(預設True)'} "
            f"filteredPairs={filtered_targets}"
        )

    # ⚠️ 嚴謹的真實幾何重疊檢查（不是用「桿尖單點 vs 球心」的近似公式）：
    # 之前的 tip_to_ball 只拿「腕部姿態推算出的桿尖點」跟球心比較，還跟
    # 球半徑比而已，完全沒扣掉 Cylinder 自己的半徑，也沒有考慮真正最近
    # 的碰撞點可能不是桿尖端點、而是桿身上更靠近球心的某一點。這裡改用
    # 「直接從 USD 讀 Cylinder 的真實半徑/高度/軸向」＋「用 CueStick
    # 剛體自己每步回報的真實世界姿態」重建整條桿身的世界座標線段，
    # 再算「球心到這條線段的最近距離」減去「兩者半徑和」，才是真正的
    # 表面間距（負值＝真的幾何重疊）。
    from pxr import Gf, UsdGeom
    _cylinder_geom = UsdGeom.Cylinder(cylinder_prim)
    _cylinder_radius = _cylinder_geom.GetRadiusAttr().Get()
    _cylinder_height = _cylinder_geom.GetHeightAttr().Get()
    _cylinder_axis_char = _cylinder_geom.GetAxisAttr().Get() or "Z"
    _axis_local_vec = {
        "X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1),
    }[_cylinder_axis_char]
    _xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    _cyl_local_matrix, _ = _xform_cache.GetLocalTransformation(cylinder_prim)
    _cylinder_center_in_cuestick = np.array(_cyl_local_matrix.Transform(Gf.Vec3d(0, 0, 0)))
    _cylinder_axis_in_cuestick = np.array(_cyl_local_matrix.TransformDir(_axis_local_vec))
    _cylinder_axis_in_cuestick /= np.linalg.norm(_cylinder_axis_in_cuestick)
    _ball_geom = UsdGeom.Sphere(ball_prim_for_check)
    _ball_radius_real = _ball_geom.GetRadiusAttr().Get()
    print(
        f"[diag] 真實幾何讀值：cylinder_radius={_cylinder_radius} cylinder_height={_cylinder_height} "
        f"cylinder_axis={_cylinder_axis_char} cylinder_center_in_cuestick={_cylinder_center_in_cuestick.tolist()} "
        f"ball_radius_real={_ball_radius_real}（設計常數 _BALL_RADIUS={_BALL_RADIUS}）"
    )

    def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        t = float(np.dot(p - a, ab) / np.dot(ab, ab))
        t = max(0.0, min(1.0, t))
        return a + t * ab

    def _real_surface_gap(cue_stick_pos: np.ndarray, cue_stick_orient: np.ndarray, ball_pos: np.ndarray, debug: bool = False) -> float:
        """回傳球心到 Cylinder 表面的真實最短距離（世界座標，用即時姿態
        重建，負值代表真的幾何重疊），取代單點近似公式。"""
        world_center = cue_stick_pos + articulation_api._rotate_vector_by_quat(
            cue_stick_orient, _cylinder_center_in_cuestick
        )
        world_axis = articulation_api._rotate_vector_by_quat(cue_stick_orient, _cylinder_axis_in_cuestick)
        half_height = _cylinder_height / 2.0
        endpoint_a = world_center - half_height * world_axis
        endpoint_b = world_center + half_height * world_axis
        closest_pt = _closest_point_on_segment(ball_pos, endpoint_a, endpoint_b)
        center_distance = float(np.linalg.norm(closest_pt - ball_pos))
        if debug:
            print(
                f"[diag]   _real_surface_gap DEBUG: world_center={world_center.tolist()} "
                f"world_axis={world_axis.tolist()} endpoint_a={endpoint_a.tolist()} "
                f"endpoint_b={endpoint_b.tolist()} closest_pt={closest_pt.tolist()} "
                f"center_distance={center_distance}"
            )
        return center_distance - (_cylinder_radius + _ball_radius_real)

    # ⚠️ 實驗性檢查：PhysX articulation 用 reduced-coordinates 求解，
    # `table_robot_manager.py` 的 create_fixed_joint() 從未設定
    # `excludeFromArticulation`——官方文件（Omni Physics「Exclude Joints
    # from Articulations」章節）描述的案例正是「用 FixedJoint 把
    # swappable manipulator 接到機器人手臂」，範例程式碼是
    # `fixedJoint.CreateExcludeFromArticulationAttr(True)`，但沒有明確
    # 說明「不排除」時，一個原本是一般剛體（沒有 ArticulationRootAPI）
    # 的外部物件被 FixedJoint 接上 articulation link 會發生什麼——懷疑
    # PhysX 預設會把它併入 articulation 的 reduced-coordinate 系統當
    # 隱含的額外 link，導致外部碰撞力的處理路徑跟一般剛體不同。用環境
    # 變數 SWING_EXCLUDE_FROM_ARTICULATION（"true"/"false"）測試設定
    # 這個 attribute 會不會修好零衝量問題，要在 timeline.play() 之前
    # 設定（跟 FixedJoint 建立同時，這裡用 USD attribute 直接補設）。
    exclude_override = os.environ.get("SWING_EXCLUDE_FROM_ARTICULATION")
    if exclude_override is not None:
        from pxr import Usd
        cue_stick_joint_path = robot_manager.get_cue_stick_prim_path() + "/FixedJointToRobot"
        joint_prim = stage.GetPrimAtPath(cue_stick_joint_path)
        if not joint_prim.IsValid():
            print(f"[diag] WARNING: 找不到 FixedJoint prim {cue_stick_joint_path}")
        else:
            joint = UsdPhysics.Joint(joint_prim)
            exclude_attr = joint.CreateExcludeFromArticulationAttr()
            print(f"[diag] excludeFromArticulation BEFORE={exclude_attr.Get()}")
            exclude_attr.Set(exclude_override.lower() == "true")
            print(f"[diag] excludeFromArticulation AFTER={exclude_attr.Get()}")

            if exclude_override.lower() == "true":
                # excludeFromArticulation=True 後，articulation 內部原本
                # 隱含排除相鄰 link 自碰撞的機制不再適用於 CueStick，
                # filter_collision_pair() 原本只排除了跟末端執行器的碰撞，
                # 沒涵蓋其他連桿——這裡補上跟機器人「全部」剛體連桿的碰撞
                # 過濾，避免 CueStick 跟手臂本體打架。
                cue_stick_path = robot_manager.get_cue_stick_prim_path()
                filtered_count = 0
                for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                        stage_api.filter_collision_pair(cue_stick_path, str(prim.GetPath()))
                        filtered_count += 1
                print(f"[diag] 已對 CueStick 補上跟機器人 {filtered_count} 個剛體/碰撞 prim 的過濾")

    # ⚠️ 實驗性檢查：CCD（Continuous Collision Detection）在整個專案裡
    # 從未被設定過（預設關閉）。球桿 Cylinder 半徑只有 0.01m（很細）、
    # 揮桿速度快，標準離散碰撞偵測（只在時間步起訖點取樣）對這種細長、
    # 高速的物體容易「穿透而過」——幾何上算出來有重疊（廣相位偵測到，
    # ContactEvent 因此觸發），但窄相位求解器可能因為沒有真正的滲透深度
    # 而算出零衝量。用環境變數 SWING_ENABLE_CCD=1 測試開啟 CCD 會不會
    # 修好這個問題，要在 timeline.play() 之前設定。
    if os.environ.get("SWING_ENABLE_CCD"):
        cue_stick_rigid_prim_for_ccd = stage.GetPrimAtPath(robot_manager.get_cue_stick_prim_path())
        ball_rigid_prim_for_ccd = stage.GetPrimAtPath(ball_prim_path)
        for label, prim in [("CueStick", cue_stick_rigid_prim_for_ccd), ("Ball", ball_rigid_prim_for_ccd)]:
            ccd_attr = prim.GetAttribute("physxRigidBody:enableCCD")
            if not ccd_attr or not ccd_attr.IsValid():
                print(f"[diag] {label} 沒有 physxRigidBody:enableCCD attribute，跳過")
                continue
            print(f"[diag] {label} CCD BEFORE={ccd_attr.Get()}")
            ccd_attr.Set(True)
            print(f"[diag] {label} CCD AFTER={ccd_attr.Get()}")

    # ⚠️ 假說 9：CueStick/Cylinder（半徑僅 0.01m）跟 Ball 都沒有明確設定
    # `physxCollision:contactOffset`／`restOffset`，用的是 PhysX 場景
    # 預設值。官方文件描述 speculative CCD 的機制是「放大 contact offset
    # 依物體運動量提前抓到接觸」——揮桿桿尖單步移動量（4+ rad/s 角速度
    # ×1.35m 槓桿臂，1/60s 時間步）可能遠大於預設 contactOffset margin，
    # 導致離散/speculative 偵測都在單一時間步內完全跳過球。用環境變數
    # SWING_CONTACT_OFFSET（公尺）覆寫 CueStick Cylinder 與 Ball 的
    # `physxCollision:contactOffset`（`restOffset` 保持不動，只放大
    # margin，不改變實際幾何邊界），要在 timeline.play() 之前設定。
    contact_offset_override = os.environ.get("SWING_CONTACT_OFFSET")
    if contact_offset_override is not None:
        from pxr import PhysxSchema
        cue_stick_cylinder_prim_for_offset = stage.GetPrimAtPath(
            robot_manager.get_cue_stick_prim_path() + "/Cylinder"
        )
        ball_prim_for_offset = stage.GetPrimAtPath(ball_prim_path + "/Ball")
        for label, prim in [("CueStick/Cylinder", cue_stick_cylinder_prim_for_offset), ("Ball", ball_prim_for_offset)]:
            if not prim.IsValid():
                print(f"[diag] {label} prim 不存在，跳過 contactOffset 覆寫")
                continue
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            offset_attr = physx_collision_api.GetContactOffsetAttr()
            if not offset_attr:
                offset_attr = physx_collision_api.CreateContactOffsetAttr()
            print(f"[diag] {label} contactOffset BEFORE={offset_attr.Get()}")
            offset_attr.Set(float(contact_offset_override))
            print(f"[diag] {label} contactOffset AFTER={offset_attr.Get()}")

    contacts: list[tuple[int, ContactEvent]] = []
    _step_counter = {"value": -1}  # AIM 開始前是 -1，之後由主迴圈更新
    physics_api.enable_contact_reporting(ball_prim_path)
    physics_api.enable_contact_reporting(robot_manager.get_cue_stick_prim_path())
    physics_api.subscribe_contact_events(lambda e: contacts.append((_step_counter["value"], e)))

    # ⚠️ #182：Ball／CueStick 兩個 RigidBody（assets/ball_template.usda、
    # assets/ball_stick.usda）從未明確設定過 physxRigidBody:solver
    # PositionIterationCount／solverVelocityIterationCount，用的是 PhysX
    # 場景層級預設值。真實桿尖接觸時間僅約 1-2ms，遠短於 60Hz 的 16.7ms
    # timestep，懷疑求解器在單一 substep 內來不及疊代收斂到正確的接觸
    # 衝量/反彈方向，尤其是偏移擊球（spin）的情況。用環境變數
    # SWING_RIGID_BODY_ITERATIONS="position,velocity"（例如 "32,8"）覆寫
    # Ball 與 CueStick 自己的 solver iteration count（跟上面 articulation
    # 專用的 SWING_VELOCITY_ITERATIONS 是不同的 prim，articulation 那個管
    # 手臂關節，這個管球與桿身的碰撞求解），一樣要在 timeline.play() 之前
    # 設定。api-lookup 查證結果：這個 attribute 只在單一 physics step 內部
    # 生效，不會像調整 physicsScene dt 一樣影響 PHYSICS_POST_STEP 觸發
    # 頻率，因此不會動到 RollingResistanceService 的固定 PHYSICS_DT 假設。
    rigid_body_iterations_override = os.environ.get("SWING_RIGID_BODY_ITERATIONS")
    if rigid_body_iterations_override is not None:
        from pxr import PhysxSchema
        pos_str, vel_str = rigid_body_iterations_override.split(",")
        cue_stick_rigid_prim_for_iter = stage.GetPrimAtPath(robot_manager.get_cue_stick_prim_path())
        ball_rigid_prim_for_iter = stage.GetPrimAtPath(ball_prim_path)
        for label, prim in [("CueStick", cue_stick_rigid_prim_for_iter), ("Ball", ball_rigid_prim_for_iter)]:
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            pos_attr = physx_rb_api.GetSolverPositionIterationCountAttr()
            if not pos_attr:
                pos_attr = physx_rb_api.CreateSolverPositionIterationCountAttr()
            vel_attr = physx_rb_api.GetSolverVelocityIterationCountAttr()
            if not vel_attr:
                vel_attr = physx_rb_api.CreateSolverVelocityIterationCountAttr()
            print(f"[diag] {label} rigid body iterations BEFORE: position={pos_attr.Get()} velocity={vel_attr.Get()}")
            pos_attr.Set(int(pos_str))
            vel_attr.Set(int(vel_str))
            print(f"[diag] {label} rigid body iterations AFTER: position={pos_attr.Get()} velocity={vel_attr.Get()}")

    # ⚠️ assets/barrett_wam/wam7/payloads/Physics/physics.usda 把
    # solverPositionIterationCount／solverVelocityIterationCount 都設成
    # 255（PhysX 上限），是之前修正 move_to_joint_position() 大幅度
    # joint-space 移動殘留誤差用的（見 scripts/probe_solver_iterations.py），
    # 但也正好對應到整個 session 反覆出現的「more than 4 velocity
    # iterations being added to a TGS scene」警告——懷疑跟這次的零衝量
    # 問題有關（TGS 求解器正常建議低 velocity iteration，Isaac Sim 預設
    # 是 1）。這裡在 timeline.play() 之前只覆寫 velocity 迭代次數（保留
    # position=255 不動，避免重新踩到原本的 joint-space 收斂問題），
    # 用環境變數 SWING_VELOCITY_ITERATIONS 方便快速測試不同值。必須在
    # timeline.play()／Articulation tensor view 建立之前設定，PhysX 只在
    # cook articulation 當下讀取一次，不是每個 tick 動態生效。
    velocity_iterations_override = os.environ.get("SWING_VELOCITY_ITERATIONS")
    if velocity_iterations_override is not None:
        from pxr import Usd
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        articulation_root_prim = None
        for prim in Usd.PrimRange(robot_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_root_prim = prim
                break
        if articulation_root_prim is None:
            print("[diag] WARNING: 找不到 ArticulationRootAPI prim，無法覆寫 solver iteration")
        else:
            vel_attr = articulation_root_prim.GetAttribute("physxArticulation:solverVelocityIterationCount")
            pos_attr = articulation_root_prim.GetAttribute("physxArticulation:solverPositionIterationCount")
            print(f"[diag] solver iteration BEFORE: position={pos_attr.Get()} velocity={vel_attr.Get()}")
            vel_attr.Set(int(velocity_iterations_override))
            print(f"[diag] solver iteration AFTER: position={pos_attr.Get()} velocity={vel_attr.Get()}")

    # ⚠️ 實驗性檢查：每個關節的 PD 驅動 stiffness=1745.3292（位置增益）
    # 非常高（見 assets/barrett_wam/wam7/payloads/Physics/physics.usda），
    # 懷疑即使在 velocity 控制模式下，若 targetPosition 沒有隨當下實際
    # 位置持續更新，殘留的位置誤差乘上這麼高的 stiffness 會產生巨大修正力，
    # 讓手臂在碰撞瞬間表現得像無限剛性的牆，吸收掉碰撞衝量而不傳給球。
    # 用環境變數 SWING_JOINT_STIFFNESS 測試調低/歸零這個值會不會修好
    # 零衝量問題，同樣要在 timeline.play() 之前設定。
    joint_stiffness_override = os.environ.get("SWING_JOINT_STIFFNESS")
    if joint_stiffness_override is not None:
        from pxr import Usd
        physics_root = stage.GetPrimAtPath(robot_prim_path)
        joint_count = 0
        for prim in Usd.PrimRange(physics_root):
            stiffness_attr = prim.GetAttribute("drive:angular:physics:stiffness")
            if stiffness_attr and stiffness_attr.IsValid():
                stiffness_attr.Set(float(joint_stiffness_override))
                joint_count += 1
        print(f"[diag] 已把 {joint_count} 個關節的 drive stiffness 改成 {joint_stiffness_override}")

    # ⚠️ 假說 8（PhysX 官方文件「Articulation Drive Stability」章節，
    # https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html）：
    # PhysX 用 Gauss-Seidel 迭代求解耦合約束時，「最後被解算的約束會贏」——
    # move_swing() 走 velocity-drive（damping=174.53293、type="force"，
    # 且沒有設定 maxForce，預設無上限），這代表 joint drive 本身就是一個
    # 可以輸出無限力的硬約束，很可能在同一個 solver sub-step 內蓋掉 contact
    # 算出的碰撞衝量，讓 joint 速度被強制拉回 drive 的目標值，表現得像
    # 完全無視那次碰撞。文件建議的修法是「降低 drive 的 stiffness、
    # damping、maxForce」或「縮小 timestep 讓 drive target 不會在單一
    # solver step 內就被達成」。假說 2（stiffness 1745→0）已確認在
    # velocity-drive 模式下無效——stiffness 是 position 誤差的增益，
    # 這裡沒有用到；真正對應的增益是 damping（velocity 誤差的增益）跟
    # maxForce（目前未設，預設無限）都還沒測過。用環境變數
    # SWING_JOINT_DAMPING（覆寫全部關節 damping）與 SWING_JOINT_MAX_FORCE
    # （設定全部關節 drive:angular:physics:maxForce 上限）分別測試。
    joint_damping_override = os.environ.get("SWING_JOINT_DAMPING")
    if joint_damping_override is not None:
        from pxr import Usd
        physics_root = stage.GetPrimAtPath(robot_prim_path)
        joint_count = 0
        for prim in Usd.PrimRange(physics_root):
            damping_attr = prim.GetAttribute("drive:angular:physics:damping")
            if damping_attr and damping_attr.IsValid():
                damping_attr.Set(float(joint_damping_override))
                joint_count += 1
        print(f"[diag] 已把 {joint_count} 個關節的 drive damping 改成 {joint_damping_override}")

    joint_max_force_override = os.environ.get("SWING_JOINT_MAX_FORCE")
    if joint_max_force_override is not None:
        from pxr import Usd
        physics_root = stage.GetPrimAtPath(robot_prim_path)
        joint_count = 0
        for prim in Usd.PrimRange(physics_root):
            damping_attr = prim.GetAttribute("drive:angular:physics:damping")
            if not (damping_attr and damping_attr.IsValid()):
                continue
            drive_prim = damping_attr.GetPrim()
            max_force_attr = drive_prim.GetAttribute("drive:angular:physics:maxForce")
            if not max_force_attr or not max_force_attr.IsValid():
                from pxr import UsdPhysics as _UsdPhysics
                drive_api = _UsdPhysics.DriveAPI.Get(drive_prim, "angular")
                max_force_attr = drive_api.CreateMaxForceAttr()
            max_force_attr.Set(float(joint_max_force_override))
            joint_count += 1
        print(f"[diag] 已把 {joint_count} 個關節的 drive maxForce 設為 {joint_max_force_override}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    _RESET_JOINTS = np.array([[0.0, *CANONICAL_REST_JOINTS]])
    articulation_api._articulation.set_dof_positions(_RESET_JOINTS)
    articulation_api._articulation.set_dof_velocities(np.zeros((1, 7)))
    for _ in range(10):
        simulation_app.update()

    base_position, base_yaw_rad = compute_base_pose(_CUE_BALL[0], _CUE_BALL[1], _SHOT_ANGLE_DEG, _TABLE_Z)
    robot.reposition(base_position)
    for _ in range(30):
        simulation_app.update()

    roll_rad = cue_pose_calculator.lookup_roll_rad(_CUE_BALL)
    print(f"[diag] roll_deg={math.degrees(roll_rad):.1f}")
    safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
    safe_target_position = list(compute_canonical_wrist_position(base_position, 0.0))
    # ⚠️ 2026-09-01：預設改用查表值（IK 可達邊界法反推，見 cue_pose_
    # calculator.lookup_backswing_distance_m() 說明），跟正式 _execute_aim()
    # 一致；env var 仍可覆寫做實驗性測試。
    _default_backswing_distance_m = cue_pose_calculator.lookup_backswing_distance_m(_CUE_BALL)
    _aim_backswing_distance_m = float(
        os.environ.get("AIM_BACKSWING_DISTANCE_M", str(_default_backswing_distance_m))
    )
    print(f"[diag] backswing_distance_m={_aim_backswing_distance_m}")
    print(f"[diag] position_offset={_POSITION_OFFSET}")
    bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
        safe_target_position, list(CANONICAL_FLAT_ORIENTATION),
        _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS, position_offset=_POSITION_OFFSET, roll_rad=roll_rad,
        backswing_distance_m=_aim_backswing_distance_m,
    )
    articulation_api.move_through_poses(
        bridge_waypoints, preceding_joint_targets=(safe_joint_targets, safe_target_position)
    )
    wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS, _POSITION_OFFSET, roll_rad=roll_rad
    )
    # ⚠️ 追蹤 AIM 收斂過程中母球的即時位置/速度——之前只在揮桿迴圈量測，
    # 發現揮桿一開始球就已經有 0.27m/s 殘留速度，懷疑母球在 AIM 收斂
    # 過程中就已經被輕微撞動、正在遠離原始擺放位置滾動，導致揮桿瞄準的
    # 是「球原本的位置」而不是「球現在的真實位置」，桿尖因此打空。
    _ball_nominal_xy = np.array(_CUE_BALL)
    _aim_ball_drift_logged = False
    for step in range(_AIM_MAX_STEPS):
        _step_counter["value"] = f"aim:{step}"
        simulation_app.update()
        aim_ball_positions, _aim_ball_orients = ball_rigid_prim.get_world_poses()
        aim_ball_pos = np.asarray(aim_ball_positions[0])
        aim_ball_linear_vel, _aim_ball_ang_vel = ball_rigid_prim.get_velocities()
        aim_ball_speed = float(np.linalg.norm(np.asarray(aim_ball_linear_vel)[0]))
        aim_drift = float(np.linalg.norm(aim_ball_pos[:2] - _ball_nominal_xy))
        if aim_drift > 0.005 and not _aim_ball_drift_logged:
            print(
                f"[diag] AIM 階段偵測到母球開始偏離原始位置：step={step} "
                f"ball_pos={aim_ball_pos.tolist()} drift={aim_drift:.4f}m ball_speed={aim_ball_speed:.4f}"
            )
            _aim_ball_drift_logged = True
        # 撞球發生在 aim:367 附近，追蹤前後每一步的真實桿身-球表面間距，
        # 確認是「逼近時真的貼到球」還是「收斂震盪讓桿身瞬間掃過球」。
        if 355 <= step <= 380:
            _aim_cue_pose = articulation_api._get_cue_stick_world_pose()
            if _aim_cue_pose is not None:
                _aim_gap = _real_surface_gap(
                    np.array(_aim_cue_pose[0]), np.array(_aim_cue_pose[1]), aim_ball_pos
                )
            else:
                _aim_gap = None
            print(
                f"[diag] aim_step={step} real_surface_gap={_aim_gap} ball_speed={aim_ball_speed:.4f} "
                f"ball_pos={np.round(aim_ball_pos,4).tolist()} is_complete={articulation_api.is_motion_complete()}"
            )
        elif step % 50 == 0:
            print(
                f"[diag] aim_step={step} ball_pos={np.round(aim_ball_pos,4).tolist()} "
                f"ball_speed={aim_ball_speed:.4f} drift_from_nominal={aim_drift:.4f}"
            )
        if articulation_api.is_motion_complete():
            break
    print(f"[diag] AIM done at step={step}  timed_out={articulation_api.did_last_motion_timeout()}")
    _final_aim_ball_positions, _ = ball_rigid_prim.get_world_poses()
    print(
        f"[diag] AIM 結束時母球位置={np.asarray(_final_aim_ball_positions[0]).tolist()}  "
        f"距原始擺放({_CUE_BALL})偏移={float(np.linalg.norm(np.asarray(_final_aim_ball_positions[0])[:2] - _ball_nominal_xy)):.4f}m"
    )

    direction_unit = np.array(cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad))
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    # ⚠️ 實驗性：用 SWING_FOLLOW_THROUGH_M 覆寫隨揮距離，測試「維持重疊
    # 接觸的時間夠不夠長」這個假說——最小重現案例（自由剛體）用了遠比
    # 這裡長的持續接觸距離/時間才觀察到真實動量轉移，懷疑正式設計的
    # follow_through_distance（只有 0.01-0.06m）太短，桿尖通過球之後
    # 立刻停下，沒有足夠時間累積衝量。
    follow_through_override = os.environ.get("SWING_FOLLOW_THROUGH_M")
    if follow_through_override is not None:
        follow_through_distance = float(follow_through_override)
        print(f"[diag] follow_through_distance 覆寫為 {follow_through_distance}")
    else:
        follow_through_distance = swing_trajectory_calculator.compute_follow_through_distance(required_tip_speed)
    contact_position = np.array(wrist)
    # ⚠️ 2026-09-01：跟正式 _execute_strike() 一致，STRIKE 後擺起點必須用
    # AIM 收斂終點同一個距離（_aim_backswing_distance_m），不是這支腳本
    # 自己另外定義的 _BACKSWING_DISTANCE_M 常數（那是舊的 0.15 寫死值）。
    backswing_position = swing_trajectory_calculator.compute_backswing_position(
        contact_position, direction_unit, _aim_backswing_distance_m
    )
    follow_through_position = contact_position + follow_through_distance * direction_unit

    print(f"[diag] required_tip_speed={required_tip_speed:.4f}  orientation_gain={_ORIENTATION_GAIN}  max_angular_speed={_MAX_ANGULAR_SPEED}")
    print(f"[diag] backswing={backswing_position.tolist()}  swing_end={follow_through_position.tolist()}")

    articulation_api.move_swing(
        backswing_position.tolist(), orientation.tolist(), follow_through_position.tolist(),
        orientation_gain=_ORIENTATION_GAIN, max_angular_speed=_MAX_ANGULAR_SPEED,
    )

    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    # 揮桿允許姿態漂移，不能用固定的 nominal direction_unit 當桿尖偏移量
    # （1.35m 槓桿臂，角度誤差會被放大成很大的位置誤差）——跟
    # ArticulationAPIImpl._rotate_vector_by_quat() 同一套做法，用「當下
    # 實際姿態」把桿身局部 +Y 軸（direction_unit 定義時的參考軸，見
    # cue_pose_calculator.compute_tilted_wrist_pose() 的 _shortest_arc_quat
    # 呼叫）轉到世界座標，才是桿尖真正的即時方向。
    local_y_axis = np.array([0.0, 1.0, 0.0])
    ball_center = np.array([_CUE_BALL[0], _CUE_BALL[1], _TABLE_Z + _BALL_RADIUS])
    min_tip_to_ball = float("inf")
    min_tip_to_ball_step = -1
    min_real_gap = float("inf")
    min_real_gap_step = -1
    any_real_overlap = False
    max_ball_speed = 0.0
    max_ball_speed_step = -1
    max_orient_err_deg = 0.0
    for step in range(_SWING_MAX_STEPS):
        _step_counter["value"] = f"swing:{step}(is_swing_motion={articulation_api._is_swing_motion})"
        simulation_app.update()
        linear_vel, _angular_vel = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(linear_vel)[0]))
        if ball_speed > max_ball_speed:
            max_ball_speed = ball_speed
            max_ball_speed_step = step
        current_orientation_for_tip = articulation_api._get_end_effector_world_orientation()
        current_tip_direction = articulation_api._rotate_vector_by_quat(current_orientation_for_tip, local_y_axis)
        tip_position = np.array(articulation_api.get_end_effector_position()) + current_tip_direction * CUE_STICK_GRIP_TO_TIP
        tip_to_ball = float(np.linalg.norm(tip_position - ball_center))
        if tip_to_ball < min_tip_to_ball:
            min_tip_to_ball = tip_to_ball
            min_tip_to_ball_step = step
        current_orientation = articulation_api._get_end_effector_world_orientation()
        q_error = articulation_api._quat_error(current_orientation, np.asarray(orientation))
        orient_err_deg = math.degrees(2.0 * np.linalg.norm(q_error[1:]))
        max_orient_err_deg = max(max_orient_err_deg, orient_err_deg)
        joint_velocities = np.asarray(articulation_api._articulation.get_dof_velocities())[0]

        # 每一步都用「CueStick 剛體自己回報的即時世界姿態」＋「球自己
        # 即時世界位置」（不是靜態的 nominal ball_center）重建真實幾何，
        # 算出球心到 Cylinder 表面的真實最短距離。
        cue_stick_pose = articulation_api._get_cue_stick_world_pose()
        cue_stick_pos = cue_stick_pose[0] if cue_stick_pose else None
        cue_stick_orient = cue_stick_pose[1] if cue_stick_pose else None
        ball_positions_now, _ball_orients_now = ball_rigid_prim.get_world_poses()
        ball_pos_now = np.asarray(ball_positions_now[0])
        if cue_stick_pos is not None and cue_stick_orient is not None:
            real_gap = _real_surface_gap(
                np.array(cue_stick_pos), np.array(cue_stick_orient), ball_pos_now,
                debug=(35 <= step <= 65 and step % 5 == 0),
            )
            if real_gap < min_real_gap:
                min_real_gap = real_gap
                min_real_gap_step = step
            if real_gap < 0.0:
                any_real_overlap = True
        else:
            real_gap = None

        if step % 2 == 0 or (35 <= step <= 65):
            wrist_orient = current_orientation.tolist()
            # 桿尖用「球桿自己回報的姿態」重算一次，跟用「腕部姿態」算出來的
            # 版本對照——如果兩者算出來的桿尖位置有落差，代表 FixedJoint
            # 兩端的姿態沒有像位置一樣完全同步，是真正碰不到球的原因。
            if cue_stick_orient is not None:
                cue_stick_tip_dir = articulation_api._rotate_vector_by_quat(np.array(cue_stick_orient), local_y_axis)
                cue_stick_tip_pos = np.array(cue_stick_pos) + cue_stick_tip_dir * CUE_STICK_GRIP_TO_TIP
                tip_to_ball_via_cuestick = float(np.linalg.norm(cue_stick_tip_pos - ball_center))
            else:
                tip_to_ball_via_cuestick = None
            print(
                f"[diag] step={step} ball_speed={ball_speed:.4f} tip_to_ball={tip_to_ball:.4f} "
                f"tip_to_ball_via_cuestick={tip_to_ball_via_cuestick} real_surface_gap={real_gap} "
                f"orient_err_deg={orient_err_deg:.2f} "
                f"is_complete={articulation_api.is_motion_complete()} is_swing_motion={articulation_api._is_swing_motion} "
                f"joint_vel_norm={float(np.linalg.norm(joint_velocities)):.3f}\n"
                f"    wrist_pos={np.round(np.array(articulation_api.get_end_effector_position()),4).tolist()} wrist_orient={np.round(wrist_orient,4).tolist()}\n"
                f"    cue_stick_pos={np.round(cue_stick_pos,4).tolist() if cue_stick_pos else None} cue_stick_orient={np.round(cue_stick_orient,4).tolist() if cue_stick_orient is not None else None}\n"
                f"    ball_pos_now={np.round(ball_pos_now,4).tolist()}"
            )
        if articulation_api.is_motion_complete():
            print(f"[diag] swing complete at step={step}")
            break
    else:
        print(f"[diag] swing EXHAUSTED {_SWING_MAX_STEPS} steps without completing")

    print(
        f"[diag] real_surface_gap 統計：min={min_real_gap:.6f} at step={min_real_gap_step}  "
        f"any_real_overlap(gap<0)={any_real_overlap}"
    )

    # 收尾：多跑幾步讓碰撞完全結算。
    for i in range(60):
        _step_counter["value"] = f"settle:{i}"
        simulation_app.update()
        linear_vel, _angular_vel = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(linear_vel)[0]))
        if ball_speed > max_ball_speed:
            max_ball_speed = ball_speed
            max_ball_speed_step = -2

    print(f"[diag] contact_events_count={len(contacts)}")
    for i, (phase, c) in enumerate(contacts):
        print(f"[diag] contact[{i}] phase={phase}: {c.collider_path_a} <-> {c.collider_path_b}  impulse={c.impulse}")

    print(f"[diag] max_ball_speed={max_ball_speed:.4f} m/s at step={max_ball_speed_step}  required_tip_speed={required_tip_speed:.4f}")
    print(f"[diag] max_orient_err_deg={max_orient_err_deg:.2f}")
    print(f"[diag] min_tip_to_ball={min_tip_to_ball:.4f} at step={min_tip_to_ball_step}  ball_radius={_BALL_RADIUS}")

    # ⚠️ 驗證 2026-08-31 修復：move_to_home() 應該先垂直上移（RESET_LIFT_
    # CLEARANCE_M）再回 home，不該讓桿尖在關節空間插值路上下降、橫掃過桌面
    # 撞到 RESET 剛擺好的球。追蹤桿尖 Z 軌跡確認有先升高，同時確認這段過程
    # 母球沒有被再次撞動。
    #
    # ⚠️ 這裡必須先把母球瞬移回一個「已經靜止」的位置（比照正式流程
    # TableBallSet.reset() 的語意），不能直接沿用揮桿剛結束時還在滾動中的
    # 球——否則球自己的殘留動量會被誤判成「被手臂撞到」，量不出真正的結果。
    rigid_body_api.set_position(ball_prim_path, _CUE_BALL[0], _CUE_BALL[1], _TABLE_Z + _BALL_RADIUS)
    rigid_body_api.set_velocities(ball_prim_path, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    for _ in range(5):
        simulation_app.update()
    _pre_reset_ball_pos, _ = ball_rigid_prim.get_world_poses()
    _pre_reset_ball_xy = np.asarray(_pre_reset_ball_pos[0])[:2].copy()
    _start_tip_z = articulation_api.get_end_effector_position()[2]
    _max_tip_z_during_reset = _start_tip_z
    _reset_ball_disturbed = False
    articulation_api.move_to_home()
    _reset_step = 0
    _disturb_step = None
    _disturb_tip_z = None
    _disturb_phase = None
    for _reset_step in range(_SWING_MAX_STEPS):
        _step_counter["value"] = f"reset:{_reset_step}"
        simulation_app.update()
        _tip_z = articulation_api.get_end_effector_position()[2]
        _max_tip_z_during_reset = max(_max_tip_z_during_reset, _tip_z)
        _cur_ball_pos, _ = ball_rigid_prim.get_world_poses()
        _cur_ball_xy = np.asarray(_cur_ball_pos[0])[:2]
        if float(np.linalg.norm(_cur_ball_xy - _pre_reset_ball_xy)) > 0.005 and _disturb_step is None:
            _reset_ball_disturbed = True
            _disturb_step = _reset_step
            _disturb_tip_z = _tip_z
            _disturb_phase = "lift" if getattr(articulation_api, "_awaiting_home_after_lift", False) else "joint_space_home"
        if articulation_api.is_motion_complete():
            break
    print(f"[diag] reset done at step={_reset_step} timed_out={articulation_api.did_last_motion_timeout()}")
    print(
        f"[diag] reset start_tip_z={_start_tip_z:.4f} "
        f"max_tip_z_during_reset={_max_tip_z_during_reset:.4f} final_tip_z={_tip_z:.4f}"
    )
    print(f"[diag] reset_ball_disturbed={_reset_ball_disturbed}")
    if _reset_ball_disturbed:
        print(f"[diag] reset_disturb_step={_disturb_step} disturb_tip_z={_disturb_tip_z:.4f} disturb_phase={_disturb_phase}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    # DIAG_HEADLESS=0 開 GUI 視窗；GUI 模式下跑完不會立刻關閉，會停留讓
    # 使用者肉眼確認，直到手動關閉視窗為止（比照 repro_flat_case_gui.py
    # 的 while simulation_app.is_running() 手法）。
    _headless = os.environ.get("DIAG_HEADLESS", "1") != "0"
    simulation_app = SimulationApp({"headless": _headless})
    try:
        _run()
        if not _headless:
            print("[diag] 執行完畢，視窗保留中，關閉視窗以結束程式。")
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        simulation_app.close()
