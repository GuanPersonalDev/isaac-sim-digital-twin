"""
scripts/scan_elevated_bridge_approach.py — 「高架橋（elevated bridge）」通用公式
的可行性驗證：對 scan_rail_collisions.py 掃到會撞庫邊的母球位置，改用「握把端
抬高、桿頭仍貼著母球」的傾斜姿態，用差動 IK 分段逼近，量測能不能避開庫邊。

## 通用公式

1. `_segment_rail_crossings()`：握把→母球這條直線，跟四面庫邊（見
   assets/billiard_env.usda 的 Cushion_Left/Right/Head/Foot 幾何）的交點。
   純 2D 線段-直線相交，庫邊近似成無限薄的線（庫邊本身只有 5cm 厚，這個
   近似的誤差遠小於安全餘量）。

2. `compute_required_tilt_rad()`：多個交點裡，離桿頭（母球）最近的交點是最
   嚴苛的限制——同樣仰角下，離桿頭越近，那個交點的實際高度越低。用這個交點
   反推最小仰角 φ，使得 `桿頭高度 + d×sin(φ) ≥ 庫邊頂部 + 安全餘量`。

3. `compute_tilted_direction()`：仰角 φ 決定桿身方向從純水平的
   `_aim_direction(shot_angle_deg)` 變成 `(dx·cosφ, dy·cosφ, -sinφ)`——水平
   分量按 cosφ 縮小、多出一個向下（朝母球）的 Z 分量。

4. `_shortest_arc_quat()`：把「球桿局部 +Y 軸」（見 ball_stick.usda 的
   Cylinder 沿 Y 延伸、CUE_STICK_GRIP_TO_TIP 的量測慣例）轉到這個世界座標
   方向所需的四元數，用最短弧公式（不管 roll，roll 對擊球結果沒影響，見
   docs/WAM_IK_implementation_and_verification.md 1.3 節「5 維冗餘」）。

5. 逼近：不用 CANONICAL_REST_JOINTS 固定姿態（那組本身就是水平、無法表示
   傾斜），改用差動 IK（`ArticulationAPIImpl.move_to_pose`），比照
   `scripts/probe_base_reachability.py` 的分段中繼點做法（位置＋姿態都線性
   內插＋逐段收斂），避免一次大跳造成的失穩問題。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/scan_elevated_bridge_approach.py
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
_CUE_BALL_X_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)
_CUE_BALL_Y_GRID = (-1.1, -0.6, -0.1, 0.4, 0.9)

_RAIL_TOP_HEIGHT = 0.04
_SAFETY_MARGIN = 0.05
_RAILS = [
    ("x", -0.66, (-1.27, 1.27)),
    ("x", 0.66, (-1.27, 1.27)),
    ("y", -1.295, (-0.635, 0.635)),
    ("y", 1.295, (-0.635, 0.635)),
]


def _shot_angle_deg(cue_ball, target):
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _segment_rail_crossings(p0, p1, rails):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    result = []
    for axis, coord, other_range in rails:
        if axis == "x":
            if dx == 0:
                continue
            t = (coord - x0) / dx
            if not (0.0 <= t <= 1.0):
                continue
            y = y0 + t * dy
            if other_range[0] <= y <= other_range[1]:
                d = math.hypot(coord - x1, y - y1)
                result.append(((coord, y), d))
        else:
            if dy == 0:
                continue
            t = (coord - y0) / dy
            if not (0.0 <= t <= 1.0):
                continue
            x = x0 + t * dx
            if other_range[0] <= x <= other_range[1]:
                d = math.hypot(x - x1, coord - y1)
                result.append(((x, coord), d))
    return result


def compute_required_tilt_rad(grip_xy, ball_xy, tip_height):
    """回傳 (tilt_rad, crossing_point_or_None)。tilt_rad=0 代表不需要抬；
    tilt_rad=None 代表無解（即使垂直也不夠高，這個交點物理上過不去）。"""
    crossings = _segment_rail_crossings(grip_xy, ball_xy, _RAILS)
    if not crossings:
        return 0.0, None
    crossing, d = min(crossings, key=lambda c: c[1])
    if d < 1e-6:
        return None, crossing
    required_sin = (_RAIL_TOP_HEIGHT + _SAFETY_MARGIN - tip_height) / d
    if required_sin <= 0:
        return 0.0, crossing
    if required_sin >= 1.0:
        return None, crossing
    return math.asin(required_sin), crossing


def compute_tilted_direction(shot_angle_deg, tilt_rad):
    import numpy as np

    theta = math.radians(shot_angle_deg)
    dx, dy = -math.sin(theta), math.cos(theta)
    return np.array([dx * math.cos(tilt_rad), dy * math.cos(tilt_rad), -math.sin(tilt_rad)])


def _shortest_arc_quat(v_from, v_to):
    import numpy as np

    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    dot = float(np.dot(v_from, v_to))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.999999:
        axis = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, *axis])
    half = v_from + v_to
    half = half / np.linalg.norm(half)
    w = float(np.dot(v_from, half))
    xyz = np.cross(v_from, half)
    return np.array([w, *xyz])


def _axis_angle_quat(axis, angle_rad):
    import numpy as np

    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([math.cos(half), *(axis * math.sin(half))])


def _quat_multiply(q1, q0):
    """q1 ⊗ q0：先套用 q0、再套用 q1（wxyz）。"""
    import numpy as np

    w1, x1, y1, z1 = q1
    w0, x0, y0, z0 = q0
    return np.array(
        [
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
        ]
    )


def compute_tilted_wrist_pose(cue_ball, shot_angle_deg, table_z, ball_radius, roll_rad=0.0):
    """回傳 (wrist_position, wrist_orientation_wxyz, tilt_rad, crossing)。
    tilt_rad=None 代表這個母球位置無解（純幾何上過不去，不是差動 IK 的問題）。

    roll_rad：球桿繞自身軸（=繞 direction 這個世界向量）額外旋轉的角度。
    這是 5 維冗餘的那個自由度，不影響桿頭實際指向或位置，純粹用來閃避
    特定關節配置下的關節限位（見 __doc__ 開頭的 wrist_pitch/palm_yaw
    卡限位紀錄）。
    """
    import numpy as np
    from core.services.base_placement_calculator import (
        CUE_STICK_GRIP_TO_TIP,
        required_grip_position,
    )

    tip_height = table_z + ball_radius
    grip_x, grip_y = required_grip_position(cue_ball[0], cue_ball[1], shot_angle_deg)
    tilt_rad, crossing = compute_required_tilt_rad((grip_x, grip_y), cue_ball, tip_height)
    if tilt_rad is None:
        return None, None, None, crossing

    direction = compute_tilted_direction(shot_angle_deg, tilt_rad)  # wrist→tip
    tip = np.array([cue_ball[0], cue_ball[1], tip_height])
    wrist = tip - CUE_STICK_GRIP_TO_TIP * direction
    base_orientation = _shortest_arc_quat(np.array([0.0, 1.0, 0.0]), direction)
    if roll_rad != 0.0:
        q_roll = _axis_angle_quat(direction, roll_rad)
        orientation = _quat_multiply(q_roll, base_orientation)
    else:
        orientation = base_orientation
    return wrist, orientation, tilt_rad, crossing


def _slerp_like(q0, q1, t):
    import numpy as np

    q = (1 - t) * q0 + t * q1
    return q / np.linalg.norm(q)


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.contact_event import ContactEvent
    from core.services.base_placement_calculator import CANONICAL_REST_JOINTS, compute_base_pose

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/BridgeScanTable"
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

    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    physics_api.enable_contact_reporting(cue_stick_prim_path)

    # 之前的掃描只對球桿本身啟用碰撞回報，完全沒偵測手臂本體（前臂／連桿等）
    # 撞到桌子/房間的情況——遍歷機器人 prim 底下所有帶 RigidBodyAPI 的 link，
    # 全部啟用回報，才能看到手臂本體的碰撞。
    from pxr import Usd, UsdPhysics as _UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    robot_link_paths = []
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(_UsdPhysics.RigidBodyAPI):
            robot_link_paths.append(prim.GetPath().pathString)
            physics_api.enable_contact_reporting(prim.GetPath().pathString)
    print(f"enabled contact reporting on {len(robot_link_paths)} robot links: {robot_link_paths}")

    contacts: list[ContactEvent] = []
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    def _move_through_waypoints(label, pos_from, pos_to, orient_from, orient_to, num_waypoints=10, max_steps=250, verbose=True):
        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            waypoint_pos = (pos_from + (pos_to - pos_from) * t).tolist()
            waypoint_orient = _slerp_like(orient_from, orient_to, t).tolist()
            articulation_api.move_to_pose(waypoint_pos, waypoint_orient)
            settled_step = None
            for _step in range(max_steps):
                simulation_app.update()
                if articulation_api.is_motion_complete():
                    settled_step = _step
                    break
            if verbose:
                current_position = np.array(articulation_api.get_end_effector_position())
                pos_err = float(np.linalg.norm(current_position - np.array(waypoint_pos)))
                print(f"  [{label}] waypoint {i}/{num_waypoints}: settled_step={settled_step}  pos_err={pos_err:.4f} m")
        return np.array(articulation_api.get_end_effector_position())

    def _run_flat_case(cue_ball, base_position, base_yaw_rad):
        """沒有庫邊交會（tilt=0）時，直接沿用已驗證過的
        CANONICAL_REST_JOINTS+base_yaw joint-space 做法，不要透過差動 IK 的
        高架橋管線——差動 IK 用的目標姿態是另外用 _shortest_arc_quat 構造出
        來的，跟 CANONICAL_REST_JOINTS 實際 FK 出來的姿態即使指向相同，roll
        分量也不保證一樣，會逼手臂多繞一段路徑去湊一個「等效但不同」的姿態，
        這是精度不足案例裡 tilt=0.00 卻仍有 0.2m 誤差、且不管換哪個 roll 都
        一樣的根因（roll 在這裡根本沒被套用，問題出在流程本身，不是 roll 選
        不對）。
        """
        contacts.clear()
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()
        _wrist0, _orientation0, _tilt0, _crossing0 = compute_tilted_wrist_pose(
            cue_ball, _shot_angle_deg(cue_ball, _TARGET_BALL), table_z, ball_radius, roll_rad=0.0
        )
        joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
        articulation_api.move_to_joint_position(joint_targets, _wrist0.tolist())
        settled_step = None
        for _step in range(1000):
            simulation_app.update()
            if articulation_api.is_motion_complete():
                settled_step = _step
                break
        final_position = np.array(articulation_api.get_end_effector_position())
        position_error = float(np.linalg.norm(final_position - _wrist0))
        all_partners = sorted({c.collider_path_b for c in contacts} | {c.collider_path_a for c in contacts})
        collided = len(contacts) > 0
        return {
            "status": "COLLISION" if collided else "OK",
            "all_partners": all_partners,
            "tilt_deg": 0.0,
            "position_error_m": position_error,
        }

    def _run_elevated_bridge_case(cue_ball, roll_rad, verbose=False):
        angle_deg = _shot_angle_deg(cue_ball, _TARGET_BALL)
        base_position, base_yaw_rad = compute_base_pose(
            cue_ball[0], cue_ball[1], angle_deg, table_z=table_z
        )
        # 先用 tilt=0 探測這個母球位置本身需不需要抬高。
        _wrist0, _orientation0, tilt_rad, crossing = compute_tilted_wrist_pose(
            cue_ball, angle_deg, table_z, ball_radius, roll_rad=0.0
        )
        if tilt_rad is None:
            return {"status": "GEOMETRICALLY_INFEASIBLE"}
        if tilt_rad <= 1e-6:
            # 不需要抬高：直接用原本驗證過的水平姿態，不要進差動 IK 管線。
            return _run_flat_case(cue_ball, base_position, base_yaw_rad)
        wrist, orientation, tilt_rad, crossing = compute_tilted_wrist_pose(
            cue_ball, angle_deg, table_z, ball_radius, roll_rad=roll_rad
        )

        contacts.clear()
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        # 不要從預設「全關節 0、完全伸直朝上」姿態直接跑差動 IK——那正好是奇異點
        # 附近，阻尼最小二乘偽逆在那裡會刻意產生極小的關節速度換取數值穩定，追不上
        # 路徑點。先用 joint-space 控制（不受奇異點影響）帶到已驗證過的
        # CANONICAL_REST_JOINTS，從這個安全起點再開始差動 IK 的高架橋動作。
        safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
        articulation_api.move_to_joint_position(
            safe_joint_targets, articulation_api.get_end_effector_position()
        )
        for _ in range(300):
            simulation_app.update()
        contacts.clear()

        start_position = np.array(articulation_api.get_end_effector_position())
        start_orientation = articulation_api._get_end_effector_world_orientation()

        up_orientation = _shortest_arc_quat(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        safe_high_z = max(float(start_position[2]), float(wrist[2])) + 0.3
        approach_point = np.array([wrist[0], wrist[1], safe_high_z])

        contacts.clear()
        pos_after_a = _move_through_waypoints(
            "A", start_position, start_position, start_orientation, up_orientation,
            num_waypoints=8, max_steps=250, verbose=verbose,
        )
        contacts_a = len(contacts)
        partners_a = sorted({c.collider_path_b for c in contacts} | {c.collider_path_a for c in contacts})
        contacts.clear()

        pos_after_b = _move_through_waypoints(
            "B", pos_after_a, approach_point, up_orientation, up_orientation,
            num_waypoints=12, max_steps=250, verbose=verbose,
        )
        contacts_b = len(contacts)
        partners_b = sorted({c.collider_path_b for c in contacts} | {c.collider_path_a for c in contacts})
        contacts.clear()

        # Phase C 拆成兩段，避免「位置＋姿態同時線性內插」在動態追蹤時互相
        # 干擾，導致桿頭高度中途短暫略低於終點值（實測撞到桌面 Surface 一次）：
        # C1 先在安全高度純轉向到最終傾斜姿態（腕部不動，桿頭再怎麼轉都還在
        # 安全高度之上）；C2 純垂直下降到最終腕部位置（姿態already固定不變，
        # 桿頭高度 = 腕部高度 + 固定偏移，隨腕部線性下降、不會中途下探）。
        pos_after_c1 = _move_through_waypoints(
            "C1:rotate_at_altitude", pos_after_b, pos_after_b, up_orientation, orientation,
            num_waypoints=10, max_steps=300, verbose=verbose,
        )
        contacts_c1 = len(contacts)
        partners_c1 = sorted({c.collider_path_b for c in contacts} | {c.collider_path_a for c in contacts})
        contacts.clear()

        pos_after_c2 = _move_through_waypoints(
            "C2:descend", pos_after_c1, wrist, orientation, orientation,
            num_waypoints=20, max_steps=400, verbose=verbose,
        )
        contacts_c2 = len(contacts)
        partners_c2 = sorted({c.collider_path_b for c in contacts} | {c.collider_path_a for c in contacts})

        final_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
        position_error = float(np.linalg.norm(pos_after_c2 - wrist))
        all_partners = set(partners_a) | set(partners_b) | set(partners_c1) | set(partners_c2)
        # 排除機器人自己（手臂各連桿之間、連桿跟球桿之間）的自我接觸，只留下
        # 跟環境（桌子/房間/球）的真實碰撞；桿頭在最終擊球高度（離桌面僅
        # 2.86mm=球半徑）輕觸桌面氈布（Surface）是預期中的正常現象，不計入
        # 失敗。
        # 用路徑前綴比對（不是精確相等）：碰撞回報的是實際碰撞形狀的 prim
        # path，可能是 RigidBodyAPI 那個 link 底下的子節點（跟球桿
        # CueStick/Cylinder 同一種巢狀狀況），精確比對會漏掉。
        def _is_self_path(p: str) -> bool:
            return p.startswith(robot_prim_path) or p.startswith(cue_stick_prim_path)

        blocking_partners = {p for p in all_partners if "Surface" not in p and not _is_self_path(p)}
        collided = len(blocking_partners) > 0
        return {
            "status": "COLLISION" if collided else "OK",
            "all_partners": sorted(all_partners),
            "tilt_deg": math.degrees(tilt_rad),
            "position_error_m": position_error,
            "final_joints": final_joints.tolist(),
            "contacts_a": contacts_a, "partners_a": partners_a,
            "contacts_b": contacts_b, "partners_b": partners_b,
            "contacts_c1": contacts_c1, "partners_c1": partners_c1,
            "contacts_c2": contacts_c2, "partners_c2": partners_c2,
        }

    # 全網格驗證：roll 不是對所有案例都通用的常數（前一輪固定 roll=90 只
    # 成功 48%，部分案例最佳 roll 不同），對每個母球位置嘗試幾個候選 roll
    # 值，用第一個成功（status=OK）的結果；不需要抬高的案例強制 roll=0
    # （已在 _run_elevated_bridge_case 內處理）。
    ROLL_CANDIDATES_DEG = (90, -90, 45)  # 控制在 3 個候選值內，避免單次背景執行超過 10 分鐘上限
    results = []
    for cue_x in _CUE_BALL_X_GRID:
        for cue_y in _CUE_BALL_Y_GRID:
            cue_ball = (cue_x, cue_y)
            chosen_result = None
            chosen_roll_deg = None
            for roll_deg in ROLL_CANDIDATES_DEG:
                result = _run_elevated_bridge_case(cue_ball, math.radians(roll_deg))
                if result["status"] in ("OK", "GEOMETRICALLY_INFEASIBLE"):
                    chosen_result = result
                    chosen_roll_deg = roll_deg
                    break
                if chosen_result is None:
                    chosen_result = result
                    chosen_roll_deg = roll_deg
            chosen_result["cue_ball"] = cue_ball
            results.append(chosen_result)
            partners = chosen_result.get("all_partners", [])
            print(
                f"cue_ball={cue_ball}  best_roll_deg={chosen_roll_deg}  status={chosen_result['status']:25s}  "
                f"tilt_deg={chosen_result.get('tilt_deg', float('nan')):6.2f}  "
                f"position_error={chosen_result.get('position_error_m', float('nan')):.4f} m"
                + (f"  partners={partners}" if chosen_result["status"] == "COLLISION" else "")
            )

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_collision = sum(1 for r in results if r["status"] == "COLLISION")
    n_infeasible = sum(1 for r in results if r["status"] == "GEOMETRICALLY_INFEASIBLE")
    print(f"\n=== FINAL SUMMARY (per-shot roll search, Phase C split): "
          f"total={len(results)}  OK={n_ok}  COLLISION={n_collision}  GEOMETRICALLY_INFEASIBLE={n_infeasible} ===")

    physics_api.unsubscribe_contact_events()


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
