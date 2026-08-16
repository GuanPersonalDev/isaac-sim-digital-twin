"""
scripts/measure_swing_speed.py — Issue #176 空揮測速

用法（獨立執行）：
    python.bat scripts/measure_swing_speed.py

也可透過 Tool Menu Registry（extension/ui/tool_menu_registry.py）在 Kit 主選單
「Tools > Billiard/...」點擊執行，此時共用目前 Kit session 已開啟的 stage。

不 import core/ 任何模組，直接用原生 API（見 docs/task-176-swing-speed-spec.md 第 3 節決議）。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

# 必須跟 extension/billiard_digital_twin/billiard_digital_twin.py 用同一種 import
# 路徑（把 extension/ 本身加進 sys.path，import 成 "ui.tool_menu_registry"），
# 否則同一支檔案會被當成兩個不同模組載入，各自有獨立的 _REGISTERED_TOOLS 清單，
# decorator 註冊的內容跟 discover_and_register 讀到的會對不上。
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
import omni.usd

from ui.tool_menu_registry import tool_menu_item

# 注意：不要在檔案最上層 import isaacsim.* 底下「本身也是獨立 Kit extension」的
# 子模組（例如 isaacsim.core.experimental.prims、isaacsim.storage.native）。
# discover_and_register 會在 extension on_startup 當下把整支檔案 import 一次
# 以觸發 decorator 註冊；若這些模組在檔案最上層被 import，會在其底層 DLL
# （例如 isaacsim.core.simulation_manager 的 _simulation_manager）尚未載入
# 完成時就強制觸發 import，導致 "DLL load failed while importing
# _simulation_manager" 之類的錯誤。這些重量級模組請延後到工具函式「真正
# 執行」時才 import（見 _load_ur5/ check_joint_limits 內的 import）。
# omni.usd 與 pxr 屬於 Kit/USD 基礎綁定，開機當下就可用，可放檔案最上層。

UR5_ASSET_PATH = "Isaac/Robots/UniversalRobots/ur5/ur5.usd"
UR5_PRIM_PATH = "/World/ur5"
REAL_ROBOT_LIMIT_DEG_S = 180.0

CUE_STICK_ASSET_PATH = os.path.join(_PROJECT_ROOT, "assets", "ball_stick.usda")
CUE_STICK_PRIM_PATH = "/World/CueStick"
FIXED_JOINT_PATH = CUE_STICK_PRIM_PATH + "/FixedJointToRobot"
END_EFFECTOR_LINK_NAME = "wrist_3_link"

PHYSICS_DT = 1.0 / 60.0
SWING_DURATION_S = 2.0  # 每輪推桿最長時間（撞到奇異點/關節極限會提前結束）
NUM_RUNS = 3  # 規格書 5.3：相同軌跡至少執行 3 次
RESET_SETTLE_STEPS = 120  # 每輪之間走回起始姿態並讓系統穩定的 step 數（兼作規格書 5.3 的暫態排除）

# 推桿起始姿態（deg）。目標：手臂伸展、球桿接近水平（平行 x-y 平面）。
# ⚠️ 這組數值是依 UR5 慣例姿態的初始猜測，第一次執行時工具會印出球桿的
# 實際傾角，若偏離水平太多（工具會直接報錯提示），請依印出的資訊微調這裡。
# 姿態設計意圖：「拉桿蓄力」摺疊姿——上臂向後下方收（X 分量 = -X）、前臂與
# 球桿水平向前（+X），球桿高度不高於肩/肘旋轉軸。這樣往 +X 前推時手肘有
# 完整的展開行程（不會一開始就頂到工作空間邊界，導致 IK 只能往 +Z 亂抬），
# 之後實際擺位時把手臂整體架高，球桿上下也有更多姿態調整空間。
# 實測記錄（符號慣例）：
# - shoulder_lift：負值抬高上臂、正值壓低；+45 = 向下前方、+90 = 正下方、
#   +135 = 向下後方（X 分量轉為 -X）。
# - elbow 與 lift 互為反號時前臂維持水平 +X（net pitch = 0）。
# - 球桿的水平指向由 wrist_1 控制：-90 與 +90 都是水平但方向相反
#   （-90 = -X、+90 = +X）；球桿指向必須與「前臂」同向（+X），
#   且前臂需與上臂反向摺疊，桿身才不會跨過手臂本體穿模。
STARTING_POSTURE_DEG = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": 135.0,
    "elbow_joint": -135.0,
    "wrist_1_joint": 90.0,
    "wrist_2_joint": 90.0,
    "wrist_3_joint": 0.0,
}
STICK_MAX_TILT_DEG = 20.0  # 球桿相對水平面的容許傾角，超過就報錯要求調姿態
DLS_LAMBDA = 0.05  # 差動 IK damped least squares 的阻尼項，抑制奇異點附近的爆速
MIN_SPEED_SCALE = 0.05  # 可行速度縮放低於此值（接近奇異點/工作空間邊緣）就提前結束該輪
ELBOW_STOP_DEG = 20.0  # 手肘角度絕對值小於此值（手臂接近打直＝奇異點）就提前結束，避免抖動


def _load_ur5():
    from isaacsim.storage.native import get_assets_root_path

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(UR5_PRIM_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(UR5_PRIM_PATH)
        resolved_path = get_assets_root_path() + "/" + UR5_ASSET_PATH
        prim.GetReferences().AddReference(resolved_path)
    return stage, prim


def _list_revolute_joints(stage, root_path: str):
    joints = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        joints.append(prim)
    return joints


# --- 以下三個函式照抄 extension/isaac_sim_impl_6_0/stage_api_impl.py 對應方法的
# 內部邏輯（align_prim_to_target / filter_collision_pair / create_fixed_joint），
# 依 docs/task-176-swing-speed-spec.md 第 3 節決議：standalone script 不 import
# core/ 或 extension/isaac_sim_impl_6_0/，直接用同一批原生 API 自己重寫一份。


def _align_prim_to_target(stage, prim_path: str, target_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    target_prim = stage.GetPrimAtPath(target_path)

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    target_world = xform_cache.GetLocalToWorldTransform(target_prim)
    prim_parent_world = xform_cache.GetParentToWorldTransform(prim)
    prim_local = target_world * prim_parent_world.GetInverse()

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(prim_local)


def _filter_collision_pair(stage, prim0_path: str, prim1_path: str) -> None:
    prim0 = stage.GetPrimAtPath(prim0_path)
    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(prim0)
    filtered_pairs.CreateFilteredPairsRel().AddTarget(Sdf.Path(prim1_path))


def _create_fixed_joint(stage, joint_path: str, body0_path: str, body1_path: str) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])


def _disable_gravity_recursive(root_prim) -> None:
    """對 root_prim 底下每一個具備 RigidBodyAPI 的 prim 個別停用重力。
    disableGravity 是 per-body 屬性，不會沿 Articulation 樹自動 cascade，
    也不會動 Physics Scene 的全域重力設定，不影響同一 stage 裡的其他物件
    （見 skills/isaac_sim_6_api_cache.md「In-process 手動物理模擬」章節）。"""
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rb.CreateDisableGravityAttr().Set(True)


def _compute_com_to_tip_local(stage, prim_path: str):
    """回傳 (r_local, axis_name, stick_length)：從 COM 到桿尖的 local-space 向量。

    假設（未經 Isaac Sim 實測驗證，第一次執行時務必核對印出的 stick_length
    是否符合實際球桿尺寸）：
      (a) 球桿 local 原點＝握把端（align_prim_to_target 後與 wrist_3_link 世界座標重合）
      (b) COM 約在幾何中心（質量均勻分布的簡化假設）
      (c) 桿尖＝ local bounding box 沿最長軸、離 local 原點較遠的一端
    用 UsdGeom.BBoxCache 的 local bound，不受目前世界座標 transform 影響，
    重複呼叫（例如 Step 3 每一輪）結果一致。
    """
    prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    # 用 ComputeUntransformedBound（不含 prim 自身 transform 的真正 local bound），
    # 不能用 ComputeLocalBound——那是 parent 空間的 bound（含 prim 自身 transform），
    # 對齊手臂後 transform 改變會讓軸向偵測跟著變（實測踩過：Step 2 印 Y 軸、
    # Step 3 印 Z 軸），而且拿它再乘 local-to-world 會重複套用旋轉。
    local_range = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    min_pt = np.array(local_range.GetMin())
    max_pt = np.array(local_range.GetMax())
    sizes = max_pt - min_pt

    axis_index = int(np.argmax(sizes))
    axis_names = ["X", "Y", "Z"]
    stick_length = float(sizes[axis_index])

    com_local = (min_pt + max_pt) / 2.0
    tip_val = (
        max_pt[axis_index] if abs(max_pt[axis_index]) > abs(min_pt[axis_index]) else min_pt[axis_index]
    )
    tip_local = com_local.copy()
    tip_local[axis_index] = tip_val

    r_local = tip_local - com_local
    return r_local, axis_names[axis_index], stick_length


FOREARM_LINK_NAME = "forearm_link"  # 前臂 link 的 frame 原點即手肘關節位置


def _stick_pose_report(stage, r_local) -> tuple[np.ndarray, float, float]:
    """回傳 (axis_unit, tilt_deg, forearm_dot)：球桿軸向單位向量（world）、
    相對水平面的傾角（deg）、與「前臂方向（手肘→手腕）水平投影」的內積。
    forearm_dot <= 0 代表球桿指向與前臂相反、會反向跨過手臂本體（穿模），
    姿態不可用。

    （最初用「基座→手腕」當朝外基準，但摺疊蓄力姿下手腕會收回到基座正下方
    附近，該向量趨近於零而失真；實測出的穿模規則是「球桿必須與前臂同向」，
    所以改用前臂方向當基準。）"""
    axis_world = _local_vector_to_world(stage, CUE_STICK_PRIM_PATH, r_local)
    axis_unit = axis_world / np.linalg.norm(axis_world)
    tilt_deg = float(np.rad2deg(np.arcsin(abs(axis_unit[2]))))

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    wrist_prim = stage.GetPrimAtPath(f"{UR5_PRIM_PATH}/{END_EFFECTOR_LINK_NAME}")
    elbow_prim = stage.GetPrimAtPath(f"{UR5_PRIM_PATH}/{FOREARM_LINK_NAME}")
    if not elbow_prim.IsValid():
        raise RuntimeError(f"找不到 {FOREARM_LINK_NAME}，檢查 UR5 asset 的 link 命名")
    wrist_pos = np.array(xform_cache.GetLocalToWorldTransform(wrist_prim).ExtractTranslation())
    elbow_pos = np.array(xform_cache.GetLocalToWorldTransform(elbow_prim).ExtractTranslation())
    forearm = wrist_pos - elbow_pos
    forearm[2] = 0.0
    forearm_norm = np.linalg.norm(forearm)
    if forearm_norm < 1e-6:
        return axis_unit, tilt_deg, 0.0
    forearm /= forearm_norm

    axis_h = axis_unit.copy()
    axis_h[2] = 0.0
    axis_h_norm = np.linalg.norm(axis_h)
    forearm_dot = float(np.dot(axis_h / axis_h_norm, forearm)) if axis_h_norm > 1e-6 else 0.0
    return axis_unit, tilt_deg, forearm_dot


def _local_vector_to_world(stage, prim_path: str, vec_local) -> np.ndarray:
    """把方向向量（非座標點）從 prim 的 local space 轉到目前世界座標系，
    用 Gf.Matrix4d.TransformDir 排除平移分量。"""
    prim = stage.GetPrimAtPath(prim_path)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_matrix = xform_cache.GetLocalToWorldTransform(prim)
    gf_vec = Gf.Vec3d(float(vec_local[0]), float(vec_local[1]), float(vec_local[2]))
    world_vec = world_matrix.TransformDir(gf_vec)
    return np.array([world_vec[0], world_vec[1], world_vec[2]])


@tool_menu_item("Billiard/Measure Swing Speed - 2. Attach Cue Stick")
def attach_cue_stick_for_swing_test():
    """規格書 Step 2：UR5 + 球桿 + Fixed Joint，關重力（僅限這組 prim），不載撞球桌。"""
    stage, _ = _load_ur5()
    end_effector_path = f"{UR5_PRIM_PATH}/{END_EFFECTOR_LINK_NAME}"

    cue_stick_prim = stage.GetPrimAtPath(CUE_STICK_PRIM_PATH)
    if not cue_stick_prim.IsValid():
        cue_stick_prim = stage.DefinePrim(CUE_STICK_PRIM_PATH)
        cue_stick_prim.GetReferences().AddReference(CUE_STICK_ASSET_PATH)

    r_local, axis_name, stick_length = _compute_com_to_tip_local(stage, CUE_STICK_PRIM_PATH)
    print(f"[SwingTest] 偵測到球桿沿 {axis_name} 軸，長度約 {stick_length:.3f} m（COM->桿尖 local={r_local}）")
    print("[SwingTest] 請自行核對這個長度是否符合實際球桿尺寸，明顯不合理請檢查 asset 幾何或改用其他量測方式。")

    _align_prim_to_target(stage, CUE_STICK_PRIM_PATH, end_effector_path)
    _filter_collision_pair(stage, CUE_STICK_PRIM_PATH, end_effector_path)

    joint_prim = stage.GetPrimAtPath(FIXED_JOINT_PATH)
    if not joint_prim.IsValid():
        _create_fixed_joint(stage, FIXED_JOINT_PATH, CUE_STICK_PRIM_PATH, end_effector_path)

    if cue_stick_prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(cue_stick_prim)
        mass = mass_api.GetMassAttr().Get()
        print(f"[SwingTest] 球桿質量 = {mass} kg（預期 0.5 kg）")
    else:
        print("[SwingTest] 警告：球桿 prim 沒有 MassAPI，無法確認質量")

    _disable_gravity_recursive(stage.GetPrimAtPath(UR5_PRIM_PATH))
    _disable_gravity_recursive(cue_stick_prim)

    print("[SwingTest] Step 2 完成：UR5 + 球桿 + Fixed Joint 已就緒，重力已針對這組 prim 個別停用。")
    print("[SwingTest] 提醒：全速揮桿時若 Fixed Joint 出現爆震，優先調 solver iteration，不要降速測。")


@tool_menu_item("Billiard/Measure Swing Speed - 1. Check Joint Limits")
def check_joint_limits():
    """讀取 UR5 各關節 velocity/effort limit，對照實機 ±180 deg/s，輸出逐關節表格。"""
    stage, _ = _load_ur5()

    joints = _list_revolute_joints(stage, UR5_PRIM_PATH)
    if not joints:
        raise RuntimeError(f"在 {UR5_PRIM_PATH} 下找不到 PhysicsRevoluteJoint，檢查 asset 路徑或 prim 結構")

    print(f"{'Joint':30s} {'USD maxVel(deg/s)':>20s} {'maxForce(N*m)':>15s} {'>180deg/s?':>12s}")
    print("-" * 80)

    rows = []
    for joint_prim in joints:
        physx_joint = PhysxSchema.PhysxJointAPI(joint_prim)
        max_vel_attr = physx_joint.GetMaxJointVelocityAttr()
        max_vel_deg_s = max_vel_attr.Get() if max_vel_attr and max_vel_attr.HasValue() else None

        drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        max_force_nm = drive_api.GetMaxForceAttr().Get() if drive_api else None

        exceeds = (
            "YES - 需覆寫"
            if (max_vel_deg_s is not None and max_vel_deg_s > REAL_ROBOT_LIMIT_DEG_S)
            else "OK"
        )
        name = joint_prim.GetName()
        print(f"{name:30s} {str(max_vel_deg_s):>20s} {str(max_force_nm):>15s} {exceeds:>12s}")
        rows.append((name, max_vel_deg_s, max_force_nm, exceeds))

    from isaacsim.core.experimental.prims import Articulation

    print("\n--- Core API 交叉驗證（rad/s -> deg/s）---")
    ur5 = Articulation(paths=UR5_PRIM_PATH)
    dof_names = ur5.dof_names
    max_vel_rad_s = np.asarray(ur5.get_dof_max_velocities())
    max_vel_deg_s_from_core = np.rad2deg(max_vel_rad_s)
    for dof_name, v_deg in zip(dof_names, max_vel_deg_s_from_core.flatten()):
        print(f"{dof_name:30s} core_api={v_deg:.2f} deg/s")

    return rows


async def _run_swing_test_async():
    """Step 3 的實際量測主體（coroutine）。

    以一般 `omni.timeline` Play 驅動物理，每個 frame 用
    `await omni.kit.app.get_app().next_update_async()` 等待 Kit 自己的下一個
    update（比照 extension/ui/debug_menu.py 的 _dock_to_viewport 寫法），
    讓 Kit 的 frame loop 正常運轉、viewport 逐幀重繪，看得到揮桿過程。

    ⚠️ 絕對不要改成在選單 callback 裡同步呼叫 `app.update()` 逐步推進——
    callback 本身就在一個 update frame 裡執行，同步 pump 新 frame 會造成
    渲染指令流重入（"call cmdEnd before calling cmdBegin again"、大量
    VkResult: NOT_READY），最後 Kit 直接 crash（實測踩過）。

    timeline.play() 是全域的：本函式假設目前 stage 只有測速用的 UR5 + 球桿，
    沒有其他正在使用的場景物件；若有，請先清掉或另開新 stage。
    """
    import omni.timeline
    import omni.kit.app
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError("目前 timeline 已經在 Play，請先 Stop 再執行本工具。")

    stage = omni.usd.get_context().get_stage()
    cue_stick_prim = stage.GetPrimAtPath(CUE_STICK_PRIM_PATH)
    joint_prim = stage.GetPrimAtPath(FIXED_JOINT_PATH)
    if not cue_stick_prim.IsValid() or not joint_prim.IsValid():
        raise RuntimeError("找不到球桿或 Fixed Joint，請先執行 Step 2（Attach Cue Stick）。")

    r_local, axis_name, stick_length = _compute_com_to_tip_local(stage, CUE_STICK_PRIM_PATH)

    app = omni.kit.app.get_app()
    total_steps = int(SWING_DURATION_S / PHYSICS_DT)

    timeline.play()
    try:
        # 讓 Play 完整初始化幾個 frame（tensor view、articulation view 等）再開始讀寫
        for _ in range(5):
            await app.next_update_async()

        ur5 = Articulation(paths=UR5_PRIM_PATH)
        cue_stick = RigidPrim(paths=CUE_STICK_PRIM_PATH)

        dof_names = list(ur5.dof_names)
        missing = [n for n in STARTING_POSTURE_DEG if n not in dof_names]
        if missing:
            raise RuntimeError(f"STARTING_POSTURE_DEG 有找不到的關節 {missing}，可用關節：{dof_names}")
        posture = np.array(
            [[np.deg2rad(STARTING_POSTURE_DEG[name]) for name in dof_names]]
        )
        dof_limits = np.asarray(ur5.get_dof_max_velocities())[0]  # rad/s，shape (n,)

        # 解析 wrist_3_link 在 Jacobian link 維度的 index。
        # fixed-base articulation 的 Jacobian 不含 base link（PhysX tensor API
        # 的已知行為），所以當 Jacobian link 數 == link 總數 - 1 時 index 要 -1。
        link_names = None
        for attr in ("link_names", "body_names"):
            if hasattr(ur5, attr):
                link_names = list(getattr(ur5, attr))
                break
        if link_names is None:
            candidates = [a for a in dir(ur5) if "link" in a.lower() or "body" in a.lower()]
            raise RuntimeError(f"找不到 link 名稱清單屬性，候選：{candidates}")

        jacobians = np.asarray(ur5.get_jacobian_matrices().numpy())[0]  # (num_links, 6, n)
        wrist_idx = link_names.index(END_EFFECTOR_LINK_NAME)
        if jacobians.shape[0] == len(link_names) - 1:
            jac_idx = wrist_idx - 1  # fixed-base：Jacobian 不含 base link
        elif jacobians.shape[0] == len(link_names):
            jac_idx = wrist_idx
        else:
            raise RuntimeError(
                f"Jacobian link 數 {jacobians.shape[0]} 與 link 名稱數 {len(link_names)} 對不上，"
                f"無法安全對應 {END_EFFECTOR_LINK_NAME}，link_names={link_names}"
            )
        print(f"[SwingTest] Jacobian shape={jacobians.shape}，{END_EFFECTOR_LINK_NAME} 對應 index={jac_idx}")

        peak_speeds = []
        push_dir = None
        for run in range(NUM_RUNS):
            # 1) 位置控制走到固定的起始姿態（同時兼作暫態排除的穩定期）
            ur5.switch_dof_control_mode("position")
            ur5.set_dof_position_targets(posture)
            for _ in range(RESET_SETTLE_STEPS):
                await app.next_update_async()

            # 2) 推桿方向 = 球桿當下軸向在 x-y 平面上的投影（沿桿身水平前推）。
            #    第一輪算好之後固定沿用，確保三輪是同一條軌跡。
            if push_dir is None:
                axis_unit, tilt_deg, forearm_dot = _stick_pose_report(stage, r_local)
                print(f"[SwingTest] 起始姿態下球桿軸向（world）={np.round(axis_unit, 3)}，"
                      f"相對水平面傾角 {tilt_deg:.1f} 度，與前臂同向分量 {forearm_dot:.2f}")
                if tilt_deg > STICK_MAX_TILT_DEG:
                    raise RuntimeError(
                        f"球桿傾角 {tilt_deg:.1f} 度超過容許值 {STICK_MAX_TILT_DEG} 度，"
                        "不符合「平行 x-y 平面」前提，請調整 STARTING_POSTURE_DEG 後重試。"
                    )
                if forearm_dot <= 0.0:
                    raise RuntimeError(
                        f"球桿指向與前臂相反（同向分量 {forearm_dot:.2f} <= 0，會跨過手臂本體穿模），"
                        "請調整 STARTING_POSTURE_DEG（通常翻轉 wrist_1_joint 的正負號即可）後重試。"
                    )
                horizontal = axis_unit.copy()
                horizontal[2] = 0.0
                push_dir = horizontal / np.linalg.norm(horizontal)
                print(f"[SwingTest] 推桿方向（world，單一軸向）={np.round(push_dir, 3)}")

            # 3) 差動 IK 直線推桿：每 frame 解 q̇ = J⁺·[v_dir; 0]（角速度=0，
            #    球桿保持姿態平移），縮放 q̇ 讓最緊的關節剛好貼在速度上限——
            #    量到的就是「關節限制下沿此軸向的最大可行桿速」。
            ur5.switch_dof_control_mode("velocity")

            twist = np.concatenate([push_dir, np.zeros(3)])  # [vx,vy,vz, wx,wy,wz]
            elbow_idx = dof_names.index("elbow_joint")
            peak = 0.0
            peak_along = 0.0
            stroke = 0.0
            steps_run = 0
            stop_reason = ""
            for _ in range(total_steps):
                # 手臂接近打直（手肘角度趨近 0）＝運動學奇異點，DLS 解會開始
                # 抖動且速度峰值早已過，提前結束該輪（實測踩過打直瞬間的抖動）
                elbow_deg = abs(float(np.rad2deg(
                    np.asarray(ur5.get_dof_positions(dof_indices=[elbow_idx]))[0, 0]
                )))
                if elbow_deg < ELBOW_STOP_DEG:
                    stop_reason = f"手肘 {elbow_deg:.0f} 度接近打直（奇異點）"
                    break
                J = np.asarray(ur5.get_jacobian_matrices().numpy())[0][jac_idx]  # (6, n)
                # damped least squares：q̇_unit = Jᵀ(JJᵀ + λ²I)⁻¹ · twist（1 m/s 基準）
                JJt = J @ J.T + (DLS_LAMBDA**2) * np.eye(6)
                qdot_unit = J.T @ np.linalg.solve(JJt, twist)
                max_ratio = float(np.max(np.abs(qdot_unit) / dof_limits))
                if max_ratio <= 1e-9:
                    stop_reason = "IK 解退化"
                    break
                scale = 1.0 / max_ratio  # 此姿態下沿 push_dir 的最大可行線速度 (m/s)
                if scale < MIN_SPEED_SCALE:
                    stop_reason = "可行速度趨近於零（工作空間邊緣）"
                    break
                ur5.set_dof_velocity_targets((qdot_unit * scale)[None, :])

                await app.next_update_async()
                steps_run += 1

                linear_vel, angular_vel = cue_stick.get_velocities()
                v_com = linear_vel.numpy()[0]
                omega = angular_vel.numpy()[0]
                r_world = _local_vector_to_world(stage, CUE_STICK_PRIM_PATH, r_local)
                v_tip = v_com + np.cross(omega, r_world)
                along = float(np.dot(v_tip, push_dir))  # 沿推桿軸向的分量（採用值看這個）
                peak = max(peak, float(np.linalg.norm(v_tip)))
                peak_along = max(peak_along, along)
                stroke += max(along, 0.0) * PHYSICS_DT

            ur5.set_dof_velocity_targets(np.zeros((1, len(dof_names))))
            peak_speeds.append(peak_along)
            print(
                f"[SwingTest] Run {run + 1}/{NUM_RUNS}：沿軸向峰值 = {peak_along:.3f} m/s"
                f"（|v_tip| 峰值 {peak:.3f} m/s），行程 {stroke:.3f} m，"
                f"執行 {steps_run}/{total_steps} 步"
                + (f"（提前結束：{stop_reason}）" if stop_reason else "")
            )
    finally:
        # Stop 會把場景重置回這次 Play 開始前的狀態，不需手動歸位。
        timeline.stop()

    adopted = max(peak_speeds)
    print(f"[SwingTest] 三次量測沿軸向峰值：{[round(v, 3) for v in peak_speeds]}")
    print(f"[SwingTest] 採用值（最大）＝ {adopted:.3f} m/s")
    print(
        f"[SwingTest] 軌跡參數：直線推桿（差動 IK，球桿平行 x-y 平面、姿態保持不旋轉），"
        f"起始姿態(deg)={STARTING_POSTURE_DEG}，推桿方向={np.round(push_dir, 3) if push_dir is not None else None}，"
        f"DLS λ={DLS_LAMBDA}，最長行程時間={SWING_DURATION_S}s，"
        f"桿長估計={stick_length:.3f}m（沿{axis_name}軸）"
    )
    return peak_speeds, adopted


async def _print_errors(coro, tag: str):
    """coroutine 的例外若留給 asyncio 預設 logger，會在 Kit console 變成亂碼
    （編碼問題），統一在這裡攔下來用 print 印出可讀訊息。"""
    try:
        return await coro
    except Exception as exc:
        print(f"[{tag}] 執行失敗：{exc}")


async def _pose_preview_async():
    import omni.timeline
    import omni.kit.app
    from isaacsim.core.experimental.prims import Articulation

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError("目前 timeline 已經在 Play，請先 Stop 再執行本工具。")

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(CUE_STICK_PRIM_PATH).IsValid():
        raise RuntimeError("找不到球桿，請先執行 Step 2（Attach Cue Stick）。")
    r_local, _, _ = _compute_com_to_tip_local(stage, CUE_STICK_PRIM_PATH)

    app = omni.kit.app.get_app()
    timeline.play()
    try:
        for _ in range(5):
            await app.next_update_async()
        ur5 = Articulation(paths=UR5_PRIM_PATH)
        dof_names = list(ur5.dof_names)
        posture = np.array([[np.deg2rad(STARTING_POSTURE_DEG[n]) for n in dof_names]])
        ur5.switch_dof_control_mode("position")
        ur5.set_dof_position_targets(posture)
        for _ in range(RESET_SETTLE_STEPS):
            await app.next_update_async()

        axis_unit, tilt_deg, forearm_dot = _stick_pose_report(stage, r_local)
        verdict = "OK 可用" if (tilt_deg <= STICK_MAX_TILT_DEG and forearm_dot > 0.0) else "不可用"
        print(f"[PosePreview] 球桿軸向={np.round(axis_unit, 3)}，傾角 {tilt_deg:.1f} 度，"
              f"與前臂同向分量 {forearm_dot:.2f} → {verdict}")
        if forearm_dot <= 0.0:
            print("[PosePreview] 球桿指向與前臂相反（會跨過手臂穿模），建議翻轉 wrist_1_joint 正負號。")
        if tilt_deg > STICK_MAX_TILT_DEG:
            print("[PosePreview] 球桿不夠水平，建議調整 wrist_1/shoulder_lift/elbow。")
        # 停留 3 秒讓你目視檢查姿態與有無穿模，之後 stop 會重置回原狀
        for _ in range(180):
            await app.next_update_async()
    finally:
        timeline.stop()


@tool_menu_item("Billiard/Measure Swing Speed - 2.5 Pose Preview")
def preview_starting_posture():
    """把手臂擺到 STARTING_POSTURE_DEG 並停留 3 秒供目視檢查（有無穿模、
    球桿是否水平朝外），同時在 Console 印出傾角/朝外分量判定。
    調整姿態常數後，重新啟用 extension 讓 scripts 重新載入再預覽。"""
    import asyncio

    asyncio.ensure_future(_print_errors(_pose_preview_async(), "PosePreview"))
    print("[PosePreview] 擺姿預覽已開始，請看 viewport 與 Console。")


@tool_menu_item("Billiard/Measure Swing Speed - 3. Run Swing Test")
def measure_swing_peak_speed():
    """規格書 Step 3（修訂版）：差動 IK 直線推桿 ×NUM_RUNS 次，量測球桿桿尖
    沿水平單一軸向前推的峰值線速度（v_tip = v_com + ω × r 投影到推桿軸向）。

    需先執行過 Step 2（attach_cue_stick_for_swing_test）。點擊後立即返回，
    量測以 async coroutine 在背景逐 frame 執行（可在 viewport 看到推桿過程），
    結果印在 Console。
    """
    import asyncio

    asyncio.ensure_future(_print_errors(_run_swing_test_async(), "SwingTest"))
    print("[SwingTest] 量測已開始（背景執行），請看 viewport 推桿過程，結果會印在 Console。")


if __name__ == "__main__":
    import asyncio

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    check_joint_limits()
    attach_cue_stick_for_swing_test()
    # 獨立執行模式：排程 coroutine 後持續 pump app update，直到量測跑完
    # （next_update_async 要靠 app update 推進，不能用 run_until_complete 空等）
    _future = asyncio.ensure_future(_run_swing_test_async())
    while not _future.done():
        simulation_app.update()
    _future.result()  # 若 coroutine 內有例外，在這裡重新拋出
    simulation_app.close()
