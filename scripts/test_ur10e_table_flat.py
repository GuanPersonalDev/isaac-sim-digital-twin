"""
scripts/test_ur10e_table_flat.py - UR10e+專用出力結構重新設計計畫
步驟 6：整合進 table_orchestrator.py 後的真實驗證（flat 案例）。

直接呼叫**正式生產程式碼**（Ur10eSwingStrategy.execute_aim()/
execute_strike()，透過真正的 ArticulationAPIImpl，不是各自獨立測試用的
Ur10eRmpflowController/Ur10eCueSlideController 直接呼叫）驗證整合後的
完整 AIM->STRIKE 流程：手臂能不能真的把母球打出去，達成合理比例的目標
球速。

跟稍早的 scripts/verify_ur10e_rmpflow_aim.py 差異：那支只驗證手臂能不能
收斂到 AIM 目標姿態；這支往下多驗證 STRIKE（滑軌關節出力）跟真實碰撞
（母球有沒有被打到、達成率多少），且全程走
core/services/ur10e_swing_strategy.py 的正式程式碼路徑（robot_arm.
reposition()/articulation_api.set_robot_base_pose()/move_to_pose()/
move_cue_slide_stroke()），不是繞過 Strategy 直接操作底層 controller。

先只測 flat 案例（cue_ball 選在遠離庫邊、tilt_rad<=1e-6 的位置），高架橋
案例留給步驟 8。

⚠️ 2026-09-03 目前狀態：FAIL，深入 root cause 調查記錄如下。
cue_ball=(0.0, 0.5)（flat，tilt_rad=0）走完整 Ur10eSwingStrategy.
execute_aim() 流程後，AIM 卡在殘留誤差不收斂。

已排除的假設（逐一實測驗證過）：
- ArticulationAPIImpl 包裝層本身的問題——用另一支診斷腳本直接呼叫
  Ur10eRmpflowController（繞開 ArticulationAPIImpl）測同一個 cue_ball，
  結果幾乎一樣（0.69m 殘留誤差），排除是步驟 6 整合引入的 bug。
- reposition() 時機（模擬已在跑時才呼叫 vs. 播放前呼叫）——用另一支
  診斷腳本直接比對 Articulation.get_world_poses()，確認 reposition()
  無論何時呼叫都會立即正確反映到 tensor-based Articulation 的 physics
  root transform，這個環節沒有問題。
- PHYSICS_POST_STEP callback 的 step_dt／呼叫頻率——加了計數器＋數值
  log 確認每個 physics tick 正好呼叫一次、step_dt 穩定為 1/60，正常。
- waypoint 純位置距離決定段數，沒有依旋轉角度加開更多段——加了
  _MAX_WAYPOINT_ROTATION_RAD 補上這個機制（見
  Ur10eRmpflowController.move_to_pose()），對這個案例沒有實質改善
  （本來的段數就已經被位置距離撐夠了），但保留下來當一般性強化。
- HOME 的 wrist_2_joint=0 是不是踩到 UR 家族手臂的手腕奇異點——實測把
  wrist_2 從 0 改到 π/2，結果反而更差（HOME 本身開始逾時、AIM 殘留誤差
  從 0.16m 惡化到 0.20m），已改回官方原始 default_q，這個假設沒有被
  證實。

有實質幫助但沒有完全解決的發現：raw USD 預設姿態（沒走過 RESET）到某些
AIM 目標需要接近 180 度的姿態翻轉，會讓 RMPflow 卡在很差的殘留誤差
（0.62m）；改成先呼叫 move_to_home()（模擬正式流程的 RESET 階段，
production 本來就是 RESET→AIM，不會從 raw 預設姿態直接跳 AIM）之後，
殘留誤差降到 0.16m——明顯更好，但仍未達到容許誤差。

具體診斷證據：逐 waypoint 記錄過六個關節角度，flat 案例卡住時
wrist_1_joint 在短短幾個 waypoint 內從接近 0 衝到超過 -π（-3.5+ rad）
才折返，elbow_joint 也在某個 waypoint 之後由遞增轉為遞減——這個「先衝
過頭、方向反轉、卡住不動」的模式，指向手臂被迫做過大的姿態翻轉。

真正找到的根因：cue_pose_calculator.compute_tilted_wrist_pose() 的
roll_rad（球桿繞自身軸的冗餘自由度，不影響桿頭實際指向）預設用 0（最短
弧慣例），但這個選擇剛好讓 flat 案例的目標姿態跟 HOME 附近的實際朝向
接近正反面，需要接近 180 度翻轉。改用
ur10e_placement_calculator.compute_roll_minimizing_reorientation()
搜尋讓翻轉角度最小的 roll_rad 後（同一個指向改用另一個滾動角表示），
所需翻轉角度從 180 度降到 90 度，AIM 殘留誤差從 0.16m（HOME-first 版）
大幅降到 0.056m——已經是 10 倍以上的改善，但仍未收斂到 1cm 容許值內，
研判是接近舒適姿態邊界的最後一段路徑本身還有殘留的局部穩定點，需要更
進一步的策略（例如收斂到接近後改用差動 IK 之類的精確控制器收尾）才能
完全解決，留給後續處理。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_ur10e_table_flat.py
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CUE_BALL = (0.0, 0.5)
_SHOT_ANGLE_DEG = 0.0
_CUE_BALL_SPEED = 1.995
_PHYSICS_DT = 1.0 / 60.0
_MAX_STEPS_PER_ACTION = 4000
# 2026-09-05 補充：兩階段 AIM（先到安全中繼姿態，再平移到最終姿態，見
# ArticulationAPIImpl.move_to_pose()）等於把 AIM 移動距離跟 waypoint
# 數量大致加倍，4000 步的舊上限不夠讓整段流程真的跑完（實測踩過：AIM
# 卡在剛好 4000 步逾時，量到的姿態離目標 1m+，但那其實是「還沒跑完」，
# 不是真的收斂失敗）。AIM 專用一個更寬裕的上限，RESET／STRIKE 不需要
# 這麼多，維持原本的 _MAX_STEPS_PER_ACTION。
# 2026-09-05 補充：solver iteration count 從預設拉高到 128（見
# ur10e_rmpflow_controller.py 同日補充，修正 wrist_2_joint 耦合動力學
# 殘留誤差）之後，每個 waypoint 收斂明顯變慢（STAGING 單一階段就要
# 4000+ 步），8000 步的上限不夠讓兩階段 AIM 真的跑完 FINAL_APPROACH。
_MAX_STEPS_PER_AIM_ACTION = 20000


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd, UsdGeom

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.ur10e_cue_slide_controller import _quintic_velocity

    from core.models.action import Action
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services import cue_pose_calculator, swing_trajectory_calculator
    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    from core.services.spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH
    from core.services.ur10e_swing_strategy import Ur10eSwingStrategy
    from isaacsim.core.experimental.prims import RigidPrim

    def _compute_bbox_tip_local_offset(stage, prim_path: str) -> np.ndarray:
        """跟 ArticulationAPIImpl._compute_tip_local_offset() 同一套邏輯：
        讀 prim 自己的 local bounding box，沿最長軸找「離原點最遠那一端」
        當尖端局部座標——直接量 CueStick 本身的真實幾何尖端，不透過
        CUE_STICK_GRIP_TO_TIP 假設值，用來交叉驗證那個常數對不對得上
        UR10e 實際掛接的桿身幾何（見本檔案 STRIKE 診斷）。"""
        prim = stage.GetPrimAtPath(prim_path)
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        local_range = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        min_pt = np.array(local_range.GetMin())
        max_pt = np.array(local_range.GetMax())
        if np.any(min_pt > max_pt):
            return np.zeros(3)
        axis_index = int(np.argmax(max_pt - min_pt))
        tip_local = np.zeros(3)
        tip_local[axis_index] = (
            max_pt[axis_index] if abs(max_pt[axis_index]) > abs(min_pt[axis_index]) else min_pt[axis_index]
        )
        return tip_local

    def _rotate_vector_by_quat(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
        w = quat_wxyz[0]
        q_xyz = quat_wxyz[1:]
        t = 2.0 * np.cross(q_xyz, vec)
        return vec + w * t + np.cross(q_xyz, t)

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/TestUr10eTableFlat"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS

    wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, table_z, ball_radius,
    )
    if tilt_rad is None:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 幾何無解")
    print(f"[flat] cue_ball={_CUE_BALL}  tilt_rad={tilt_rad:.6f}（應該 <= 1e-6，flat 案例）")
    if tilt_rad > 1e-6:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 不是 flat 案例（tilt_rad={tilt_rad}），改別的座標")

    print("[flat] 建立 ArticulationAPIImpl ...")
    sys.stdout.flush()
    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    print("[flat] 建立 TableRobotManager ...")
    sys.stdout.flush()
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, UR10eRobot,
    )
    robot_arm = robot_manager.get_robot()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    ball_prim_path = table_ball_set.get_ball_prim_paths()[0]

    print("[flat] timeline.play() ...")
    sys.stdout.flush()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    # ⚠️ place_ball() 必須在 timeline.play() 之後才能呼叫——球的
    # xformOp:orient 屬性要等 physics/timeline 先初始化過一次才會存在
    # （實測踩過：AssertionError: Undefined 'xformOp:orient' property）。
    # production 路徑（DemoTableOrchestrator._execute_aim()）本來就是在
    # 模擬進行中呼叫，這裡改成跟正式路徑一致的時序，不是在場景剛建好、
    # timeline 還沒 play 的時候呼叫。
    print("[flat] place_ball ...")
    sys.stdout.flush()
    table_ball_set.place_ball(0, _CUE_BALL[0], _CUE_BALL[1])
    for _ in range(5):
        simulation_app.update()

    print("[flat] articulation_api.initialize() ...")
    sys.stdout.flush()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()
    print("[flat] initialize() 完成")
    sys.stdout.flush()

    # 2026-09-05 補充：把球檯跟母球註冊為 RMPflow 障礙物——今天稍早查證
    # 過 production 路徑從未呼叫過 add_obstacle()/add_ground_plane()，
    # 也直接證實 AIM 收尾過程中桿身（不是桿尖）真的蹭到母球。球檯用固定
    # 方塊（跟 scripts/verify_ur10e_home_pose.py 同一套尺寸慣例：
    # spread_score_calculator.TABLE_WIDTH/TABLE_LENGTH，不是整個
    # billiard_env 參照的 bbox——那個含整個房間，量到 10x10x10m 會把手臂
    # 吞進去），母球用會持續追蹤最新世界座標的動態球體（不是註冊當下的
    # 固定快照，球會被打去別的地方）。
    # 2026-09-05 除錯記錄：原本把方塊放在 table_z 之上（[table_z,
    # table_z+0.15]），逐 waypoint 診斷（DEBUG_UR10E_AIM_WAYPOINTS）顯示
    # FINAL_APPROACH 階段全部 19 個 waypoint 無一收斂、方向誤差從 0.04rad
    # 一路發散到 1.31rad——桿尖擊球所在高度正是 table_z+ball_radius
    # （約 table_z+0.03m），完全落在這個「障礙物」範圍內，等於整段最終
    # 逼近都在跟自己要抵達的高度打架。改成放在 table_z 之下（代表球檯桌面
    # 以下的實體結構：石板/桌腳/桌框），讓桌面正上方（球桿實際需要操作的
    # 空間）保持淨空。
    _TABLE_OBSTACLE_HEIGHT_M = 0.15
    table_center = table.get_table_center()
    table_obstacle_center = [table_center[0], table_center[1], table_z - _TABLE_OBSTACLE_HEIGHT_M / 2.0]
    articulation_api.register_static_box_obstacle(
        table_obstacle_center, [TABLE_WIDTH, TABLE_LENGTH, _TABLE_OBSTACLE_HEIGHT_M]
    )
    articulation_api.register_dynamic_sphere_obstacle(ball_prim_path, ball_radius)
    print(f"[flat] 已註冊 RMPflow 障礙物：球檯 center={table_obstacle_center} size=({TABLE_WIDTH},{TABLE_LENGTH},{_TABLE_OBSTACLE_HEIGHT_M})  母球 prim={ball_prim_path} radius={ball_radius}")

    ball_rigid_prim = RigidPrim(paths=ball_prim_path)
    cue_stick_rigid_prim = RigidPrim(paths=cue_stick_prim_path)
    cue_tip_bbox_local_offset = _compute_bbox_tip_local_offset(stage, cue_stick_prim_path)
    print(f"[flat] CueStick bbox 量測尖端局部座標={cue_tip_bbox_local_offset.tolist()}"
          f"（跟 CUE_STICK_GRIP_TO_TIP={CUE_STICK_GRIP_TO_TIP} 沿 +Y 的假設值比較用）")

    # 2026-09-05 補充：查證 cue_pose_calculator 純數學計算完全自洽（roll_rad
    # 從 0 到 360 度測過，桿尖預測值跟母球中心誤差都是 0），懷疑落差出在
    # align_prim_to_target() 設好的初始 XformOp 跟 PrismaticJoint 實際物理
    # 約束（body0/body1 各自原點對齊，見 create_prismatic_joint() 沒有
    # 額外設 localPos/localRot）兩者之間，在 timeline.play() 之後是否真的
    # 完全重合——直接量測 wrist_3_link（articulation_api 的 end effector）
    # 跟 CueStick 剛好都還在 q=0（原始生成姿態，還沒被任何 retract/AIM
    # 動過）時的世界座標位姿，看兩者是否真的一致。
    raw_wrist_position = np.asarray(articulation_api.get_end_effector_position())
    raw_wrist_orientation = np.asarray(articulation_api.get_end_effector_orientation())
    raw_cue_position, raw_cue_orientation = cue_stick_rigid_prim.get_world_poses()
    raw_cue_position = np.asarray(raw_cue_position[0], dtype=float)
    raw_cue_orientation = np.asarray(raw_cue_orientation[0], dtype=float)
    print(f"[flat] 【原始生成姿態，尚未 RESET/AIM】wrist_3_link 位置={raw_wrist_position.tolist()} 方向={raw_wrist_orientation.tolist()}")
    print(f"[flat] 【原始生成姿態，尚未 RESET/AIM】CueStick    位置={raw_cue_position.tolist()} 方向={raw_cue_orientation.tolist()}")
    print(f"[flat] 位置差={ (raw_cue_position - raw_wrist_position).tolist() } "
          f"norm={np.linalg.norm(raw_cue_position - raw_wrist_position):.6f}m")
    orientation_dot = float(np.clip(np.abs(np.dot(raw_wrist_orientation, raw_cue_orientation)), -1.0, 1.0))
    print(f"[flat] 方向夾角={2.0*np.arccos(orientation_dot):.6f}rad")

    # 2026-09-05 補充：直接查證球桿（包含桿身，不只桿尖）有沒有真的碰到
    # 母球——enable_contact_reporting(cue_stick_prim_path) 涵蓋整個
    # CueStick prim（含其 Cylinder collider），任何部位碰到球都會產生
    # CONTACT_FOUND 事件，不是只有桿尖端點。之前只檢查 impulse>0 的事件，
    # 這裡額外記錄「目前處於哪個階段」，並印出全部事件（含 impulse=0），
    # 排除「衝量太小被當成雜訊濾掉」的可能性。
    contacts = []
    current_phase_for_contacts = {"name": "SETUP"}
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    physics_api.enable_contact_reporting(ball_prim_path)
    # 2026-09-05 補充：先前只對 CueStick／母球啟用碰撞回報，手臂本體
    # （forearm/wrist_1~3_link）完全沒有可見度——使用者提出的假設「AIM
    # 過程中會不會是手臂本體（不是球桿）撞到庫邊」在這之前根本測不出來。
    # 移動障礙方塊到 table_z 之下之後（見前面 register_static_box_obstacle
    # 的說明），球檯庫邊（真實高於桌面的實體結構）已經不在避障範圍內，
    # 需要直接用碰撞回報驗證手臂本體有沒有真的撞上去，不能只憑理論推測。
    arm_link_prim_paths = [
        f"{robot_prim_path}/forearm_link",
        f"{robot_prim_path}/wrist_1_link",
        f"{robot_prim_path}/wrist_2_link",
        f"{robot_prim_path}/wrist_3_link",
    ]
    for arm_link_prim_path in arm_link_prim_paths:
        physics_api.enable_contact_reporting(arm_link_prim_path)
    physics_api.subscribe_contact_events(
        lambda e: contacts.append((current_phase_for_contacts["name"], e))
    )

    strategy = Ur10eSwingStrategy(robot_arm, articulation_api)
    action = Action(
        cue_ball_speed=_CUE_BALL_SPEED,
        shot_angle=_SHOT_ANGLE_DEG,
        position_offset=[0.0, 0.0],
        cue_ball_placement=list(_CUE_BALL),
        should_execute_action=True,
    )

    def _measure_tip_to_ball():
        """回傳 (tip_world, ball_world, distance)——桿尖世界座標用 bbox
        量測到的真實幾何尖端（見 _compute_bbox_tip_local_offset()），不是
        CUE_STICK_GRIP_TO_TIP 假設值。"""
        cue_position, cue_orientation = cue_stick_rigid_prim.get_world_poses()
        cue_position = np.asarray(cue_position[0], dtype=float)
        cue_orientation = np.asarray(cue_orientation[0], dtype=float)
        tip_world = cue_position + _rotate_vector_by_quat(cue_orientation, cue_tip_bbox_local_offset)

        ball_position, _ball_orientation = ball_rigid_prim.get_world_poses()
        ball_position = np.asarray(ball_position[0], dtype=float)

        distance = float(np.linalg.norm(tip_world - ball_position))
        return tip_world, ball_position, distance

    def _run_until_complete(label: str) -> int:
        step = 0
        while not articulation_api.is_motion_complete() and step < _MAX_STEPS_PER_ACTION:
            simulation_app.update()
            step += 1
        print(f"[flat] {label} 完成，steps={step} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")
        return step

    # ⚠️ 2026-09-03 除錯發現：raw USD 預設姿態（沒走過 RESET）到某些 AIM
    # 目標需要接近 180 度的姿態翻轉，會讓 RMPflow 卡住不收斂（flat 案例
    # 實測卡在 0.62m）；真正的 production 流程是 RESET（回 HOME）→ AIM，
    # 不會從 raw 預設姿態直接跳 AIM。這裡先呼叫 move_to_home()，模擬
    # 正式流程的 RESET 階段，再執行 AIM。
    # move_to_home() 需要 RMPflow 先知道目前底座的真實世界位姿——這時候
    # 手臂還在 TableRobotManager 建構時的初始固定偏移位置
    # （TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER），還沒被
    # execute_aim() 的 per-shot reposition() 移動過，先同步這個初始位置。
    initial_base_offset = TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    initial_base_position = [
        table.get_table_center()[0] + initial_base_offset[0],
        table.get_table_center()[1] + initial_base_offset[1],
        table.get_table_center()[2] + initial_base_offset[2],
    ]
    articulation_api.set_robot_base_pose(initial_base_position, [1.0, 0.0, 0.0, 0.0])

    print("[flat] 呼叫 articulation_api.move_to_home()（模擬 RESET 階段）...")
    current_phase_for_contacts["name"] = "RESET"
    articulation_api.move_to_home()
    _run_until_complete("RESET(HOME)")

    # 2026-09-05 補充：STRIKE 開始前母球速度非零（不是預期的 0），代表
    # AIM 收尾過程中球被短暫碰到，即使「最終停止姿態」量起來離球有安全
    # 距離。逐 tick 監控母球速度，抓到第一次變成非零的那個 tick，記錄
    # 當下 RMPflow 控制器內部處於哪個階段（哪個 waypoint／joint-space
    # 收尾／差動 IK 收尾）跟桿尖-母球距離，藉此定位是哪一段動作造成的。
    rmp_ctrl_for_bump_diag = articulation_api._ur10e_rmpflow_controller

    def _run_aim_watch_ball_bump() -> int:
        step = 0
        bump_reported = False
        min_distance_during_finish = float("inf")
        finish_tick_count = 0
        while not articulation_api.is_motion_complete() and step < _MAX_STEPS_PER_AIM_ACTION:
            simulation_app.update()
            step += 1

            # 2026-09-05 補充：先前只在「偵測到非零速度」那一刻量一次距離，
            # 沒有連續追蹤 joint_finish 整段過程——如果桿尖在收尾修正的
            # 「過程中」（不是最終停止姿態）曾經短暫掃進球體範圍，只看
            # 起點/終點量不到。這裡改成 joint_finish_active 期間每個 tick
            # 都量距離，找出整段收尾過程真正最接近的瞬間。
            if rmp_ctrl_for_bump_diag._joint_finish_active:
                finish_tick_count += 1
                _, _, finish_distance = _measure_tip_to_ball()
                if finish_distance < min_distance_during_finish:
                    min_distance_during_finish = finish_distance

            ball_velocity, _ = ball_rigid_prim.get_velocities()
            ball_speed = float(np.linalg.norm(np.asarray(ball_velocity[0], dtype=float)))
            if not bump_reported and ball_speed > 1e-4:
                bump_reported = True
                tip_world, ball_world, distance = _measure_tip_to_ball()
                print(
                    f"[flat] AIM step={step} 偵測到母球第一次非零速度={ball_speed:.6f}m/s——"
                    f"waypoint_index={rmp_ctrl_for_bump_diag._waypoint_index}/"
                    f"{len(rmp_ctrl_for_bump_diag._waypoints)} "
                    f"finishing_active={rmp_ctrl_for_bump_diag._finishing_active} "
                    f"joint_finish_active={rmp_ctrl_for_bump_diag._joint_finish_active} "
                    f"桿尖-母球距離={distance:.5f}m"
                )
        print(f"[flat] AIM 完成，steps={step} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")
        print(
            f"[flat] joint_finish 全程（{finish_tick_count} tick）桿尖-母球最小距離="
            f"{min_distance_during_finish:.5f}m（球半徑={ball_radius:.5f}m，距離<=球半徑代表過程中真的掃進球體範圍）"
        )
        return step

    print("[flat] 呼叫 Ur10eSwingStrategy.execute_aim() ...")
    current_phase_for_contacts["name"] = "AIM"
    strategy.execute_aim(action, tuple(_CUE_BALL), table_z, ball_radius)
    _run_aim_watch_ball_bump()

    # 2026-09-05 補充：直接印出目前為止（RESET+AIM 全程）記錄到的「所有」
    # CueStick 相關事件，不篩選 impulse>0——查證使用者提出的假設「會不會
    # 是桿身（不是桿尖）碰到母球」：enable_contact_reporting(cue_stick_
    # prim_path) 涵蓋整個 CueStick prim（含 Cylinder collider），球桿
    # 任何部位碰到球都會產生事件，不會因為衝量太小被濾掉（CONTACT_FOUND
    # 事件本身跟 impulse 大小無關，只要有偵測到接觸就會觸發，impulse 只是
    # 附帶資訊）。
    cue_related_contacts_so_far = [
        (phase_name, c) for phase_name, c in contacts
        if cue_stick_prim_path in (c.actor_path_a, c.actor_path_b)
    ]
    print(f"[flat] RESET+AIM 全程，CueStick 相關事件（不篩選 impulse）共 {len(cue_related_contacts_so_far)} 筆：")
    for phase_name, c in cue_related_contacts_so_far:
        print(f"[flat]   phase={phase_name} a={c.actor_path_a} b={c.actor_path_b} impulse={c.impulse}")
    if not cue_related_contacts_so_far:
        print("[flat]   （完全沒有記錄到任何 CueStick 相關事件——母球的擾動確認跟球桿無關）")

    # 2026-09-05 補充：查證「手臂本體（不是球桿）撞到庫邊」的假設——
    # 手臂避障方塊移到 table_z 之下之後，真實庫邊（高於桌面的實體結構）
    # 不再被 RMPflow 避障涵蓋，需要直接看 forearm/wrist_1~3_link 有沒有
    # 真的撞上球檯任何部位（不篩選 impulse，理由同上）。
    arm_related_contacts_so_far = [
        (phase_name, c) for phase_name, c in contacts
        if any(p in (c.actor_path_a, c.actor_path_b) for p in arm_link_prim_paths)
    ]
    print(f"[flat] RESET+AIM 全程，手臂本體（forearm/wrist_1~3_link）相關事件（不篩選 impulse）共 {len(arm_related_contacts_so_far)} 筆：")
    for phase_name, c in arm_related_contacts_so_far:
        print(f"[flat]   phase={phase_name} a={c.actor_path_a} b={c.actor_path_b} impulse={c.impulse}")
    if not arm_related_contacts_so_far:
        print("[flat]   （完全沒有記錄到任何手臂本體相關事件）")

    # ⚠️ 2026-09-04 除錯記錄：之前這裡只檢查位置誤差，讓一個實際上方向
    # 沒收斂的姿態被誤判成「AIM 成功」——1.35m 長的球桿會把沒被抓到的
    # 方向誤差在桿尖處放大成數公分等級的偏移（見
    # scripts_diag 桿尖-母球距離診斷）。現在兩者都檢查。
    # `wrist_position`（top-level 變數）用 roll_rad=0 算的，但 roll 不影響
    # 位置（見 cue_pose_calculator.compute_tilted_wrist_pose() docstring），
    # 位置比較仍然有效；方向則必須跟 execute_aim() 內部實際用的
    # roll-optimized 目標比（直接讀 controller 記錄的最終 waypoint，不是
    # 重算一次，避免跟正式路徑的計算邏輯不同步）。
    live_wrist_position = np.asarray(articulation_api.get_end_effector_position())
    live_wrist_orientation = np.asarray(articulation_api.get_end_effector_orientation())
    aim_position_error = float(np.linalg.norm(live_wrist_position - np.asarray(wrist_position)))

    rmp_ctrl_for_target = articulation_api._ur10e_rmpflow_controller
    final_target_position, final_target_orientation = rmp_ctrl_for_target._waypoints[-1]
    orientation_dot = float(np.clip(np.abs(np.dot(live_wrist_orientation, final_target_orientation)), -1.0, 1.0))
    aim_orientation_error = 2.0 * np.arccos(orientation_dot)

    print(f"[flat] AIM 收斂後 wrist 位置={live_wrist_position.tolist()}  位置誤差={aim_position_error:.5f} m（容許 0.01 m）")
    print(f"[flat] AIM 收斂後 wrist 方向={live_wrist_orientation.tolist()}  目標方向={np.asarray(final_target_orientation).tolist()}"
          f"  方向誤差={aim_orientation_error:.5f} rad（容許 0.02 rad，約 1.15 度）")
    print(f"[flat] AIM 判定：位置{'OK' if aim_position_error <= 0.01 else '未達標'} "
          f"方向{'OK' if aim_orientation_error <= 0.02 else '未達標'}")

    # 診斷：RMPflow 最後一次算出的關節目標 vs 實際量到的關節位置——
    # 用來分辨殘留誤差是「PhysX joint drive 追不上 RMPflow 給的目標」
    # （tracking gap，可靠 stiffness/damping 補強修，見 UR3e
    # _boost_wrist_gains_for_cue_stick_load() 先例）還是「RMPflow 自己算出
    # 的關節目標，即使完美追上也不對應期望的末端位姿」（RMPflow 本身的
    # reactive 殘留，需要換精確控制器收尾）。
    rmp_ctrl = articulation_api._ur10e_rmpflow_controller
    last_targets = rmp_ctrl._last_active_position_targets
    if last_targets is not None:
        current_full_positions = np.asarray(articulation_api._articulation.get_dof_positions())[0]
        current_active_positions = current_full_positions[rmp_ctrl._active_dof_indices]
        joint_gap = current_active_positions - last_targets
        print(f"[flat] active_joint_names={rmp_ctrl._active_joint_names}")
        print(f"[flat] RMPflow 最後關節目標={last_targets.tolist()}")
        print(f"[flat] 實際量到的關節位置    ={current_active_positions.tolist()}")
        print(f"[flat] 關節 tracking gap (實際-目標)={joint_gap.tolist()}  max_abs_gap={np.max(np.abs(joint_gap)):.6f} rad")

    ball_velocity_before, _ = ball_rigid_prim.get_velocities()
    print(f"[flat] STRIKE 前母球速度={np.asarray(ball_velocity_before[0]).tolist()}（應接近 0）")

    tip_world, ball_world, distance = _measure_tip_to_ball()
    print(f"[flat] STRIKE 開始前（桿子已退到後擺位置）桿尖={tip_world.tolist()} "
          f"母球={ball_world.tolist()} 距離={distance:.5f} m（miss 向量={( tip_world - ball_world).tolist()}）")

    print("[flat] 呼叫 Ur10eSwingStrategy.execute_strike() ...")
    current_phase_for_contacts["name"] = "STRIKE"
    strategy.execute_strike(action, tuple(_CUE_BALL), table_z, ball_radius)

    # 診斷：STRIKE 全程逐 tick 記錄桿尖(bbox量測)跟母球的距離，找出整段
    # 行程中最接近的瞬間跟距離／miss 向量——藉此判斷「完全沒打到球」是
    # 系統性的幾何偏移（q=0 時桿尖本來就沒有真的對準球心，miss 向量在
    # 垂直於揮桿方向上有固定分量）還是 AIM 殘留誤差造成的偶發性偏移。
    #
    # 2026-09-04 補充：42% 達成率（不是 0%）顯示桿子有真的碰到球，但
    # 動量沒轉移完全——懷疑是球的實際位置比 CueSlideJoint quintic 設計的
    # q=0（v=target_velocity）接觸點更靠近後擺起點，桿子在還沒加速到全速
    # 前就先碰到球。額外逐 tick 記錄 CueSlideJoint 實際量到的位置/速度，
    # 抓到碰撞事件發生的那個 tick，比對當下的實際速度 vs 目標桿尖速度，
    # 直接驗證這個假設。
    slide_dof_index = articulation_api._ur10e_cue_slide_controller._slide_dof_index
    min_distance = float("inf")
    min_distance_step = -1
    min_distance_tip = None
    min_distance_ball = None
    # STRIKE 這段迴圈現在也涵蓋「揮桿後沿原軸縮回」（Ur10eCueSlideController
    # 的 post_strike_retract 階段，決策 5），迴圈結束時母球早就撞上球堆、
    # 速度已經不是球桿賦予的值了。達成率改用整段 STRIKE 觀察到的**最大**
    # 母球速度——球桿接觸結束後只會因摩擦/碰撞遞減，峰值就是球桿實際傳遞
    # 出去的速度。
    peak_ball_speed = 0.0
    strike_steps = 0
    contacts_seen_before_loop = len(contacts)
    while not articulation_api.is_motion_complete() and strike_steps < _MAX_STEPS_PER_ACTION:
        contacts_before_tick = len(contacts)
        simulation_app.update()
        strike_steps += 1

        tip_world, ball_world, distance = _measure_tip_to_ball()
        if distance < min_distance:
            min_distance = distance
            min_distance_step = strike_steps
            min_distance_tip = tip_world
            min_distance_ball = ball_world

        slide_positions = np.asarray(articulation_api._articulation.get_dof_positions())[0]
        slide_velocities = np.asarray(articulation_api._articulation.get_dof_velocities())[0]
        slide_position = float(slide_positions[slide_dof_index])
        slide_velocity = float(slide_velocities[slide_dof_index])
        ball_velocity_now, _ = ball_rigid_prim.get_velocities()
        ball_speed_now = float(np.linalg.norm(np.asarray(ball_velocity_now[0], dtype=float)))
        peak_ball_speed = max(peak_ball_speed, ball_speed_now)

        # 2026-09-05 補充：懷疑 _step_strike() 用「經過 T 秒」（開放迴路
        # 計時）判定揮桿完成，而不是「q 真的到 0」——quintic 邊界條件保證
        # 的是「指令位置」q_command(T)=0，不保證 PhysX 的 velocity-mode PD
        # 在這麼短時間內（觀察到整個 STRIKE 只有 17~24 tick）真的追上這麼
        # 快速變化的速度指令。逐 tick 比對「這一刻應該下達的指令速度」
        # （直接用同一組 quintic 係數重算）跟「實際量到的速度」，量化
        # tracking lag 有多大、有沒有持續存在。
        cue_slide_ctrl = articulation_api._ur10e_cue_slide_controller
        if cue_slide_ctrl._phase == "strike" and cue_slide_ctrl._quintic is not None:
            c3, c4, c5, T = cue_slide_ctrl._quintic
            t = min((cue_slide_ctrl._elapsed_strike_steps - 1) * _PHYSICS_DT, T)
            commanded_velocity = _quintic_velocity(c3, c4, c5, t)
            velocity_lag = commanded_velocity - slide_velocity
        else:
            commanded_velocity = None
            velocity_lag = None

        print(
            f"[flat] STRIKE step={strike_steps} 桿尖={tip_world.tolist()} 母球={ball_world.tolist()} "
            f"距離={distance:.5f}m CueSlideJoint位置={slide_position:.5f} 實際速度={slide_velocity:.5f}m/s "
            f"指令速度={commanded_velocity if commanded_velocity is None else f'{commanded_velocity:.5f}'}m/s "
            f"lag={velocity_lag if velocity_lag is None else f'{velocity_lag:.5f}'}m/s "
            f"母球速度={ball_speed_now:.5f}m/s"
        )

        if len(contacts) > contacts_before_tick:
            new_events = contacts[contacts_before_tick:]
            print(
                f"[flat] STRIKE step={strike_steps} 偵測到新碰撞事件 {len(new_events)} 筆："
                f"CueSlideJoint 當下位置={slide_position:.5f}（q=0 為設計接觸點） "
                f"當下速度={slide_velocity:.5f}m/s（目標桿尖速度={swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED):.5f}m/s） "
                f"桿尖-母球距離={distance:.5f}m"
            )
            for phase_name, c in new_events:
                print(f"[flat]     phase={phase_name} a={c.actor_path_a} b={c.actor_path_b} impulse={c.impulse}")

    print(f"[flat] STRIKE 完成，steps={strike_steps} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")
    print(f"[flat] STRIKE 全程桿尖離母球最近的瞬間：step={min_distance_step}  距離={min_distance:.5f} m"
          f"（球半徑={ball_radius:.5f} m，距離 <= 球半徑代表桿尖曾經進入球體範圍）")
    if min_distance_tip is not None:
        miss_vector = min_distance_tip - min_distance_ball
        print(f"[flat]   當下桿尖={min_distance_tip.tolist()}  母球={min_distance_ball.tolist()}  miss 向量（桿尖-母球）={miss_vector.tolist()}")

    # 揮桿完成後再多跑幾步，讓球的速度穩定下來（衝量剛作用完那一瞬間的
    # get_velocities() 可能還沒反映完整動量轉移）。
    for _ in range(10):
        simulation_app.update()

    ball_velocity_after, _ = ball_rigid_prim.get_velocities()
    ball_velocity_after = np.asarray(ball_velocity_after[0])
    ball_speed_after = float(np.linalg.norm(ball_velocity_after))
    print(f"[flat] STRIKE 後母球速度向量={ball_velocity_after.tolist()}")
    print(f"[flat] STRIKE 後母球速度={ball_speed_after:.4f} m/s（此時母球已撞過球堆，僅供參考）")
    achievement = peak_ball_speed / _CUE_BALL_SPEED
    print(f"[flat] STRIKE 全程母球速度峰值={peak_ball_speed:.4f} m/s  目標母球速度={_CUE_BALL_SPEED} m/s  達成率={100 * achievement:.1f}%")

    ball_contacts = [
        (phase_name, c) for phase_name, c in contacts
        if ball_prim_path in (c.actor_path_a, c.actor_path_b) and c.impulse > 0.0
    ]
    print(f"[flat] 母球碰撞事件數（impulse>0）={len(ball_contacts)}")
    for phase_name, c in ball_contacts:
        print(f"[flat]   CONTACT phase={phase_name} a={c.actor_path_a} b={c.actor_path_b} impulse={c.impulse}")

    # 決策 7 的「母球碰撞事件數恰好 1 次」指的是**球桿**碰到母球恰好一次
    # （沒有蹭到、沒有二次擊球）；母球撞上球堆（Ball_1）是這一擊預期中的
    # 結果，不算違規，所以只篩球桿相關的事件來判定。
    cue_stick_ball_contacts = [
        (phase_name, c) for phase_name, c in ball_contacts
        if cue_stick_prim_path in (c.actor_path_a, c.actor_path_b)
    ]
    print(f"[flat] 其中球桿-母球碰撞事件數={len(cue_stick_ball_contacts)}（決策 7 要求恰好 1 次）")

    speed_ok = achievement >= 0.9
    contact_ok = len(cue_stick_ball_contacts) == 1
    if speed_ok and contact_ok:
        print("[flat] PASS：達成率 >=90% 且球桿只碰到母球一次")
    else:
        reasons = []
        if not speed_ok:
            reasons.append(f"達成率 {100 * achievement:.1f}% < 90%")
        if not contact_ok:
            reasons.append(f"球桿-母球碰撞 {len(cue_stick_ball_contacts)} 次（應為 1 次）")
        print(f"[flat] FAIL：{'；'.join(reasons)}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[flat] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
