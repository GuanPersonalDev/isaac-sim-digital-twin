"""
scripts/search_canonical_pose_candidates.py — Phase 1（重新設計 CANONICAL_REST_JOINTS）：
對 (shoulder_pitch, elbow_pitch) 候選網格做碰撞篩選＋高架橋抬高餘裕實測，用真實
`cue_pose_calculator.compute_elevated_bridge_waypoints()` +
`ArticulationAPIImpl.move_through_poses()` 管線，為重新設計
`base_placement_calculator.CANONICAL_REST_JOINTS` 挑選候選姿態。

見 C:\\Users\\Kuan\\.claude\\plans\\ancient-skipping-wand.md（重新設計
CANONICAL_REST_JOINTS：讓高架橋姿態涵蓋真實 Kitchen 邊界）。

## 根因

`docs/issue-180-reachability-analysis.md` 第十二節：目前 `shoulder_pitch=1.9`
只留 0.085 rad（≈4.87°）限位餘裕（硬限位 1.985），但真實 Kitchen 母球邊界代表點
（`action_bounds.CUE_BALL_PLACEMENT_X/Y`）需要的抬高角換算成腕部爬升高度是
12.6cm~66.7cm（`ΔZ ≈ CUE_STICK_GRIP_TO_TIP(1.35m) × sin(tilt_rad)`），遠超過
0.085 rad 能提供的量（估計只有 ~7-8cm）。`elbow_pitch=1.8` 目前只用了朝上限
（π）行程的 57%，還有 76.85°（1.34 rad）完全沒用到，是唯一有大量未用空間的
關節。

⚠️ 差動 IK（`ArticulationAPIImpl._step_motion()`）是無加權 DLS 偽逆
（`q̇=Jᵀ(JJᵀ+λ²I)⁻¹twist`），沒有 per-joint 權重，物理上偏好槓桿臂最長的
`shoulder_pitch`——不能假設「加大 `elbow_pitch`」就會自動被差動 IK 用上分擔
負擔，必須用這支腳本的實測結果驗證，不能只看解析 FK 判斷。

⚠️ `wrist_pitch`/`palm_yaw` 這個階段固定用 0.0 佔位（不是最終值，Phase 2 才
精修），這裡只篩「shoulder_pitch 全程會不會撞限位」，跟手腕最後兩個關節的
指向精修無關。

⚠️ Phase 1 用現行（尚未針對候選重新量測）的 `_LOCAL_TIP_RADIUS`/
`_LOCAL_TIP_HEIGHT` 呼叫 `compute_base_pose()` 決定基座位置——這對不同候選
姿態的實際腕部落點會有些微系統性偏移，但抬高需求本身（`compute_required_
tilt_rad()`）只跟握把→母球連線的幾何有關，不受這個偏移影響，Phase 1 只是
粗篩「量級對不對」，精確結果留給 Phase 5（常數重新量測後）驗證。

## 用法

先跑 flat 合法性篩選（快，篩掉明顯自我碰撞/環境碰撞的候選）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_canonical_pose_candidates.py flat

再對倖存候選跑高架橋餘裕實測（把 `_BRIDGE_STAGE_CANDIDATES` 改成 flat 階段的
倖存名單後執行，慢，一次背景執行有 10 分鐘上限，必要時分批跑）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_canonical_pose_candidates.py bridge
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

# WAM7 關節限位（見 assets/barrett_wam/wam7.urdf）。
_SHOULDER_PITCH_LIMIT = 1.985
_ELBOW_PITCH_LOWER_LIMIT = -0.9
_ELBOW_PITCH_UPPER_LIMIT = 3.14159

# Phase 1 粗篩網格：(shoulder_pitch, elbow_pitch)，1.9/1.8 是現行對照組。
_FLAT_STAGE_CANDIDATES = [
    (sp, ep)
    for sp in (1.9, 1.7, 1.5, 1.3, 1.1)
    for ep in (1.8, 2.1, 2.4, 2.7)
]

# flat 階段跑完、人工檢視結果後，把倖存（status=OK）的候選填進這裡再跑
# bridge 階段——刻意分開兩個階段的候選名單，避免對已經在 flat 就出局的候選
# 浪費最耗時的高架橋實測。
_BRIDGE_STAGE_CANDIDATES = [
    (1.9, 1.8),  # 現行對照組：測試「重設計 waypoint 順序」這個改動本身（不動
                 # shoulder_pitch/elbow_pitch）能不能單獨解決 wrist_yaw/wrist_pitch 卡死問題
]

_SHOULDER_PITCH_MARGIN_TARGET = 0.15  # rad，比現行 0.085 更保守的門檻
_ELBOW_PITCH_MARGIN_TARGET = 0.2  # rad

_TARGET_BALL = (0.0, 0.635)
_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575

# Kitchen 邊界最嚴苛＋次嚴苛代表點（見 docs/issue-180-reachability-analysis.md
# 第十二節算出的表格）：(cue_ball_xy, 對應的 tilt 大約落在 9.91°~29.61°)。
_BRIDGE_TEST_CASES = [
    (0.0, -0.9382125),  # 隔離診斷：已知手動兩步驟量測法 roll=15° 成功，測打包呼叫法是否也成功
]

# 每個 base_yaw／Kitchen 母球位置代表點都跑一次 flat 合法性檢查，不是只測
# base_yaw=0——高架橋 Phase 0 一律用 base_yaw=0（安全起點，跟瞄準角無關，
# 見 table_orchestrator.py 的既有設計），所以 flat 合法性只需要驗證 base_yaw=0
# 這一組姿態本身，不需要對每個瞄準角各測一次。


def _shot_angle_deg(cue_ball, target):
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _candidate_joint_targets(shoulder_pitch, elbow_pitch):
    # [base_yaw, shoulder_pitch, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_pitch, palm_yaw]
    #
    # ⚠️ 第一輪診斷用 wrist_pitch=palm_yaw=0.0 當佔位，結果發現卡住的不是
    # shoulder_pitch（還有 0.3rad 餘裕），是 wrist_yaw（卡在上限 1.25）／
    # wrist_pitch（卡在上限 1.5707）——但那組 0/0 起點本身就是未校正的干擾
    # 變因（現行 CANONICAL_REST_JOINTS 的 wrist_pitch/palm_yaw 是靠
    # probe_palm_yaw_correction.py 網格搜尋出來的 -0.5585/1.5010，不是隨便
    # 挑的）。改用現行已校正值當 wrist_pitch/palm_yaw 起點，才能乾淨隔離
    # shoulder_pitch/elbow_pitch 這兩個變數的效果，不跟「wrist 起點沒校正」
    # 這個獨立問題混在一起。shoulder_yaw/wrist_yaw 維持 0.0（不在這次搜尋
    # 範圍內）。
    return [0.0, shoulder_pitch, 0.0, elbow_pitch, 0.0, -0.5585, 1.5010]


def _run_flat_stage() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd, UsdPhysics as _UsdPhysics

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.contact_event import ContactEvent

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/CanonicalSearchTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))

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

    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(_UsdPhysics.RigidBodyAPI):
            physics_api.enable_contact_reporting(prim.GetPath().pathString)

    contacts: list[ContactEvent] = []
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    def _is_self_path(p: str) -> bool:
        return p.startswith(robot_prim_path) or p.startswith(cue_stick_prim_path)

    print(f"{'shoulder_pitch':>15} {'elbow_pitch':>12} {'status':>10} {'tip_height':>11} {'tip_radius':>11} {'partners'}")
    results = []
    for shoulder_pitch, elbow_pitch in _FLAT_STAGE_CANDIDATES:
        contacts.clear()
        robot.reposition((0.0, 0.0, 0.0))
        for _ in range(10):
            simulation_app.update()
        joint_targets = _candidate_joint_targets(shoulder_pitch, elbow_pitch)
        articulation_api.move_to_joint_position(joint_targets, articulation_api.get_end_effector_position())
        for _ in range(150):
            simulation_app.update()

        tip_world = np.array(articulation_api.get_end_effector_position())
        blocking_partners = sorted(
            {c.collider_path_b for c in contacts if not _is_self_path(c.collider_path_b) and "Surface" not in c.collider_path_b}
            | {c.collider_path_a for c in contacts if not _is_self_path(c.collider_path_a) and "Surface" not in c.collider_path_a}
        )
        status = "COLLISION" if blocking_partners else "OK"
        print(
            f"{shoulder_pitch:>15.3f} {elbow_pitch:>12.3f} {status:>10} "
            f"{tip_world[2]:>11.4f} {float(np.hypot(tip_world[0], tip_world[1])):>11.4f} "
            f"{blocking_partners if blocking_partners else ''}"
        )
        results.append((shoulder_pitch, elbow_pitch, status))

    n_ok = sum(1 for *_r, status in results if status == "OK")
    print(f"\n=== FLAT STAGE SUMMARY: total={len(results)}  OK={n_ok}  COLLISION={len(results) - n_ok} ===")
    print("OK candidates:", [(sp, ep) for sp, ep, status in results if status == "OK"])

    physics_api.unsubscribe_contact_events()


def _run_bridge_stage() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd, UsdPhysics as _UsdPhysics

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.contact_event import ContactEvent
    from core.services.base_placement_calculator import compute_base_pose
    from core.services import cue_pose_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/CanonicalSearchBridgeTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))

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

    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(_UsdPhysics.RigidBodyAPI):
            physics_api.enable_contact_reporting(prim.GetPath().pathString)

    contacts: list[ContactEvent] = []
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    def _is_self_path(p: str) -> bool:
        return p.startswith(robot_prim_path) or p.startswith(cue_stick_prim_path)

    def _run_case(shoulder_pitch, elbow_pitch, cue_ball, roll_rad=0.0, max_steps=1200, use_bundled_call=False):
        angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
        base_position, _base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], angle_deg, _TABLE_Z)

        contacts.clear()
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        safe_joint_targets = _candidate_joint_targets(shoulder_pitch, elbow_pitch)

        if use_bundled_call:
            # 診斷用：完全比照正式程式碼 table_orchestrator._execute_aim()
            # 的做法——不手動分兩步，用 compute_canonical_wrist_position()
            # 算 Phase 0 的分析目標位置＋單位四元數當分析目標朝向，一次呼叫
            # move_through_poses(waypoints, preceding_joint_targets=...) 讓
            # ArticulationAPIImpl 內部自己驅動 Phase 0→B1→B2→C1→C2。用來比對
            # 這支腳本手動兩步驟量測法（下面的 else 分支）跟正式程式碼的
            # 打包呼叫法，結果是否一致。
            from core.services.base_placement_calculator import (
                compute_canonical_wrist_position, CANONICAL_FLAT_ORIENTATION,
            )
            safe_target_position = list(compute_canonical_wrist_position(base_position, 0.0))
            safe_orientation = list(CANONICAL_FLAT_ORIENTATION)
            waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
                safe_target_position, safe_orientation, cue_ball, angle_deg, _TABLE_Z, _BALL_RADIUS,
                roll_rad=roll_rad,
            )
            if waypoints is None:
                return {"status": "GEOMETRICALLY_INFEASIBLE"}
            articulation_api.move_through_poses(
                waypoints, preceding_joint_targets=(safe_joint_targets, safe_target_position)
            )
        else:
            # Phase 0：joint-space 帶到候選的 flat 姿態（base_yaw=0，是高架橋
            # Phase 0 一貫的安全起點，見 table_orchestrator.py 既有設計）。不靠
            # compute_canonical_wrist_position()（那要用尚未針對這個候選重新量
            # 測的 _LOCAL_TIP_RADIUS/_LOCAL_TIP_HEIGHT，會不準）——改成固定步數
            # 穩定後直接讀真實 FK 位置，完全不依賴任何預先量測的常數。
            articulation_api.move_to_joint_position(safe_joint_targets, articulation_api.get_end_effector_position())
            for _ in range(200):
                simulation_app.update()
            contacts.clear()

            start_position = np.array(articulation_api.get_end_effector_position()).tolist()
            start_orientation = np.array(articulation_api.get_end_effector_orientation()).tolist()
            print(f"    [measured] start_position={start_position} start_orientation={start_orientation}")

            waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
                start_position, start_orientation, cue_ball, angle_deg, _TABLE_Z, _BALL_RADIUS,
                roll_rad=roll_rad,
            )
            if waypoints is None:
                return {"status": "GEOMETRICALLY_INFEASIBLE"}
            articulation_api.move_through_poses(waypoints)

        max_shoulder_pitch = float(np.asarray(articulation_api._articulation.get_dof_positions())[0][1])
        settled_step = None
        contacts.clear()
        current_waypoint_index = articulation_api._waypoint_index
        # B1(0)/B2(1)/C1 中繼點(2..2+rotate_steps-1)/C2(最後一個) —— waypoint
        # 數量隨 rotate_steps 變動，用 defaultdict 而不是寫死 4 個 key。
        import collections
        waypoint_partners: dict = collections.defaultdict(set)
        for step in range(max_steps):
            simulation_app.update()
            current_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
            max_shoulder_pitch = max(max_shoulder_pitch, float(current_joints[1]))
            if articulation_api._waypoint_index != current_waypoint_index:
                # waypoint 剛切換：把這段期間累積的碰撞歸給剛結束的那個 waypoint，
                # 才能知道到底是 B1（爬升）、B2（平移）、C1（轉向）還是 C2（下降）
                # 撞到東西，不是只知道「整段序列裡有撞」。
                partners = {c.collider_path_b for c in contacts if not _is_self_path(c.collider_path_b) and "Surface" not in c.collider_path_b}
                partners |= {c.collider_path_a for c in contacts if not _is_self_path(c.collider_path_a) and "Surface" not in c.collider_path_a}
                waypoint_partners[current_waypoint_index] |= partners
                contacts.clear()
                current_waypoint_index = articulation_api._waypoint_index
            if step % 200 == 0:
                err = (
                    float(np.linalg.norm(np.array(articulation_api.get_end_effector_position()) - articulation_api._target_position))
                    if articulation_api._target_position is not None else -1
                )
                print(
                    f"    step={step} waypoint_index={articulation_api._waypoint_index} "
                    f"is_joint_space={articulation_api._is_joint_space_motion} err={err:.5f} "
                    f"joints={np.round(current_joints, 4).tolist()}"
                )
            if articulation_api.is_motion_complete():
                settled_step = step
                break
        # 收尾：把最後一個 waypoint 期間累積的碰撞也歸進去。
        partners = {c.collider_path_b for c in contacts if not _is_self_path(c.collider_path_b) and "Surface" not in c.collider_path_b}
        partners |= {c.collider_path_a for c in contacts if not _is_self_path(c.collider_path_a) and "Surface" not in c.collider_path_a}
        waypoint_partners[current_waypoint_index] |= partners

        def _waypoint_name(idx: int) -> str:
            if idx == 0:
                return "B1(climb)"
            if idx == 1:
                return "B2(translate)"
            if idx == len(waypoints) - 1:
                return "C2(descend)"
            return f"C1(rotate step {idx - 1}/{len(waypoints) - 3})"

        for idx, partners in waypoint_partners.items():
            if partners:
                print(f"    COLLISION during {_waypoint_name(idx)}: {sorted(partners)}")
        final_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
        timed_out = articulation_api.did_last_motion_timeout()
        print(f"    FINAL joints={np.round(final_joints, 4).tolist()} waypoint_index={articulation_api._waypoint_index}")

        blocking_partners = sorted(set().union(*waypoint_partners.values()))
        collided = len(blocking_partners) > 0
        shoulder_pitch_margin = _SHOULDER_PITCH_LIMIT - max_shoulder_pitch
        status = "COLLISION" if collided else ("TIMEOUT" if timed_out else "OK")
        return {
            "status": status,
            "settled_step": settled_step,
            "max_shoulder_pitch": max_shoulder_pitch,
            "shoulder_pitch_margin": shoulder_pitch_margin,
            "partners": blocking_partners,
        }

    # 撞到球檯的案例，逐一嘗試 roll 候選值（比照 scan_elevated_bridge_
    # approach.py 既有做法），第一個成功（status=OK）的就採用，找不到就回報
    # 全部候選的結果。roll_rad 只影響球桿繞自身軸的姿態，不影響擊球結果，
    # 純粹用來閃避特定關節配置卡住/碰撞的問題。
    _ROLL_CANDIDATES_DEG = (15,)  # 診斷用：已知手動兩步驟量測法在這個 roll 下成功，這輪改用打包呼叫法(use_bundled_call=True)比對

    print(f"{'shoulder_pitch':>15} {'elbow_pitch':>12} {'cue_ball':>22} {'roll_deg':>9} {'status':>16} {'max_sp':>8} {'sp_margin':>10}")
    all_results = []
    for shoulder_pitch, elbow_pitch in _BRIDGE_STAGE_CANDIDATES:
        for cue_ball in _BRIDGE_TEST_CASES:
            chosen_result = None
            chosen_roll_deg = None
            for roll_deg in _ROLL_CANDIDATES_DEG:
                result = _run_case(shoulder_pitch, elbow_pitch, cue_ball, roll_rad=math.radians(roll_deg), use_bundled_call=True)
                print(
                    f"{shoulder_pitch:>15.3f} {elbow_pitch:>12.3f} {str(cue_ball):>22} {roll_deg:>9} "
                    f"{result['status']:>16} "
                    f"{result.get('max_shoulder_pitch', float('nan')):>8.4f} "
                    f"{result.get('shoulder_pitch_margin', float('nan')):>10.4f}"
                    + (f"  partners={result['partners']}" if result.get("partners") else "")
                )
                if result["status"] == "OK":
                    chosen_result = result
                    chosen_roll_deg = roll_deg
                    break
                if chosen_result is None:
                    chosen_result = result
                    chosen_roll_deg = roll_deg
            print(f"  -> chosen roll_deg={chosen_roll_deg} status={chosen_result['status']}")
            all_results.append((shoulder_pitch, elbow_pitch, cue_ball, chosen_result))

    print("\n=== BRIDGE STAGE SUMMARY ===")
    for shoulder_pitch, elbow_pitch in _BRIDGE_STAGE_CANDIDATES:
        case_results = [r for sp, ep, cb, r in all_results if sp == shoulder_pitch and ep == elbow_pitch]
        n_ok = sum(1 for r in case_results if r["status"] == "OK")
        min_margin = min(
            (r["shoulder_pitch_margin"] for r in case_results if "shoulder_pitch_margin" in r),
            default=float("nan"),
        )
        meets_target = n_ok == len(case_results) and min_margin >= _SHOULDER_PITCH_MARGIN_TARGET
        print(
            f"  shoulder_pitch={shoulder_pitch:.3f} elbow_pitch={elbow_pitch:.3f}: "
            f"{n_ok}/{len(case_results)} OK, min_shoulder_pitch_margin={min_margin:.4f} rad "
            f"({'PASS' if meets_target else 'FAIL'} target>={_SHOULDER_PITCH_MARGIN_TARGET})"
        )

    physics_api.unsubscribe_contact_events()


if __name__ == "__main__":
    from isaacsim import SimulationApp

    stage_arg = sys.argv[1] if len(sys.argv) > 1 else "flat"
    simulation_app = SimulationApp({"headless": True})
    try:
        if stage_arg == "flat":
            _run_flat_stage()
        elif stage_arg == "bridge":
            _run_bridge_stage()
        else:
            print(f"未知階段 '{stage_arg}'，用 'flat' 或 'bridge'")
    finally:
        simulation_app.close()
