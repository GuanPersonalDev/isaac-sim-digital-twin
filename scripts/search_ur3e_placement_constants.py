"""
scripts/search_ur3e_placement_constants.py — 對指定的 `tilt_rad` 目標角度，
用 `scripts/test_elevated_bridge_ur3e_table.py` 已驗證過的兩階段搜尋方法論
（Stage 1：pan=0 時掃 shoulder_lift/elbow/wrist1/wrist2，篩高度合理+Z傾斜
對齊；Stage 2：解析算 shoulder_pan），找出可以離線量測、寫死成常數的
「pan=0 姿態＋direction_local＋local_tip_position」，供
`core/services/ur3e_placement_calculator.py` 當查表常數用。

跟 `test_elevated_bridge_ur3e_table.py` 的差異：這支腳本只做「搜姿態」，
不建球檯、不跑揮桿執行驗證（那些留給個別案例的真實驗證腳本），純粹是給
`ur3e_placement_calculator.py` 產生查表常數的離線工具，對應
`scripts/probe_canonical_pose.py`／`scripts/probe_palm_yaw_correction.py`
之於 WAM7 `base_placement_calculator.py` 常數的角色。

`tilt_rad=0`（flat 案例）時，target_direction 是純水平方向，這組結果可以
當成「canonical」姿態沿用給所有 flat shot（跟 WAM7 的 `CANONICAL_REST_
JOINTS` 對任何 shot_angle 都通用同一個道理——shoulder_pan 負責吸收
shot_angle，跟 WAM7 的 base_yaw 一樣）。

## 2026-09-02 幾何修正（會讓先前產生的常數全部作廢，必須重跑）

GUI 重跑確認揮桿卡住的根因是球桿後擺撞地板之後，往回追查發現這支腳本（以及
2026-09-01 同批寫的 `test_*_ur3e_*.py`／`design_human_like_ur3e_pose.py`）
的幾何有兩個複合錯誤，`scripts/probe_cue_axis_and_clearance.py` 實測確認：

1. **球桿軸向用錯軸**：用了 ee 的 Z 軸，但球桿實體沿的是 ee 的 **Y 軸**
   （見 `_CUE_LOCAL_AXIS` 說明）。解析桿尖與實體球桿差約 1.3m。
2. **1.35m 偏移加在錯的一端**：`_solve_base_position_and_joint_targets()`
   是把儲存的 local 點平移到 `target_wrist_position`，而
   `cue_pose_calculator` 的 `wrist = contact - CUE_STICK_GRIP_TO_TIP *
   direction` 本來就是**握把**目標；舊版卻存「已加 1.35m 偏移的桿尖」，等於
   多算一次 1.35m。改成儲存 `local_grip_position`（ee 本身，不加偏移）。

連帶新增兩個先前完全沒有的限制式：球桿**軸向**本身的傾斜對齊（舊版只約束
桿尖速度方向，等於容許「球桿指著地板、桿尖卻橫著掃」的候選），以及後擺全程
的球桿地板淨空（Stage 1.5）。

跑法（需要 Isaac Sim headless）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 SEARCH_TILT_DEG=0.0 \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_ur3e_placement_constants.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_UR3E_PATH = "Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
_TILT_DEG = float(os.environ.get("SEARCH_TILT_DEG", "0.0"))
_CUE_BALL_SPEED = 1.995
_ELBOW_DOF_INDEX = 2

_ULTRA_FAST_GRID = os.environ.get("SEARCH_ULTRA_FAST_GRID") == "1"
"""2026-09-02：放寬純肘關節限制、改成加權多關節揮桿解之後，先用比
`SEARCH_FAST_GRID` 更粗的網格（3×5×2×4×4=480 組）快速確認「有沒有可行解」
這個是非題，能在合理時間內跑完；確認有解之後再花時間跑 `SEARCH_FAST_GRID`
甚至完整網格去找品質更好的候選，不要一開始就賭一個要跑數小時的網格結果是
「無解」（2026-09-02 稍早那次单關節版本就是這樣，四千七百多組跑了近 4
小時才確認全滅）。"""
_FAST_GRID = os.environ.get("SEARCH_FAST_GRID") == "1" or _ULTRA_FAST_GRID
"""2026-09-02：這台機器這次執行時遇到跟 Isaac Sim 無關的環境問題（RDP
連線中斷/恢復期間，長時間執行的 `simulation_app.update()` 迴圈會間歇性
卡住數十秒到數分鐘，讓原本 7×15×5×13=6825 組網格的搜尋總耗時從預期的
~9 分鐘暴增到 40 分鐘還沒跑完），加這個環境變數當退路——縮回接近原始
7×8×3×7=1176 組網格（見 test_elevated_bridge_ur3e_table.py 註解，這個
較小的網格量級過去驗證過能在正常環境下 ~90s 內跑完），犧牲一些解析度
換取能在環境不穩定時也跑得完，不是永久改小網格。"""
if _ULTRA_FAST_GRID:
    _SHOULDER_LIFT_CANDIDATES = [-2.4, -1.6, -0.8]
    _ELBOW_CANDIDATES = [-2.4, -1.2, 0.0, 1.2, 2.4]
    _WRIST1_CANDIDATES = [-1.3, -0.7]
    _WRIST2_CANDIDATES = [-1.5, -0.5, 0.5, 1.5]
elif _FAST_GRID:
    _SHOULDER_LIFT_CANDIDATES = [-2.8, -2.0, -1.2, -0.4]
    _ELBOW_CANDIDATES = [-2.8, -2.0, -1.2, -0.4, 0.4, 1.2, 2.0, 2.8]
    _WRIST1_CANDIDATES = [-1.6, -1.0, -0.4]
    _WRIST2_CANDIDATES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
else:
    _SHOULDER_LIFT_CANDIDATES = [-2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4]
    _ELBOW_CANDIDATES = [-2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    _WRIST1_CANDIDATES = [-1.6, -1.3, -1.0, -0.7, -0.4]
    _WRIST2_CANDIDATES = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
if _ULTRA_FAST_GRID:
    _WRIST3_CANDIDATES = [-2.4, -1.2, 0.0, 1.2, 2.4]
elif _FAST_GRID:
    _WRIST3_CANDIDATES = [-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4]
else:
    _WRIST3_CANDIDATES = [-2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
"""2026-09-02：`wrist_3` 從固定 0 改成一起掃。球桿軸向是 ee 的 **Y** 軸
（見 `_CUE_LOCAL_AXIS`），而 `wrist_3` 正是繞 ee 的工具軸自轉的關節——它一轉
球桿就在一個圓錐面上掃，也就是說 `wrist_3` 直接決定球桿指向哪裡，是這個幾何
下的關鍵自由度。舊版把它固定在 0，是建立在「球桿沿 ee 的 Z 軸（＝工具軸）、
繞工具軸自轉不影響指向」這個錯誤假設上；改用正確軸向後固定 0 會讓整個網格
無解（實測：672 組全滅）。"""

_MAX_Z_TILT_ERROR = 0.04
_MIN_ALIGNMENT = 0.97

_PREFILTER_Z_TILT_ERROR = 0.35
_PREFILTER_AXIS_SWING_ALIGNMENT = 0.80
_REFINE_SHORTLIST = 20
_REFINE_MAX_ITER = 600
"""2026-09-02：改成「粗網格挑基準點 → 連續局部精修」，不再指望網格自己踩中解。

修正球桿軸向之後，Stage 1 要同時把**兩個**獨立的方向 Z 分量（球桿軸向、桿尖
速度方向）壓進 ±`_MAX_Z_TILT_ERROR`(0.04)，而網格步長是 0.4~0.8 rad——實測
4704 組全滅，且淘汰分佈（base_height 2997、axis_z_tilt 1005、swing_z_tilt
206、singular 496，加起來剛好等於總數）顯示這是解析度問題，不是不可行：把
網格加密到能踩中 0.04 容差需要十萬組以上，每組都要 `_set_pose()` 三次
simulation update，時間上不可行。

所以 Stage 1 的方向類條件改成寬鬆「預篩」（`_PREFILTER_*`，只負責挑出方向
大致對的基準點），真正收斂到 `_MAX_Z_TILT_ERROR`/`_MIN_ALIGNMENT` 交給
Stage 1.4 的 Nelder-Mead 連續精修。基座高度窗口維持硬性（那是結構性條件，
不是可以靠微調收斂的連續量）。

⚠️ 2026-09-01 曾經有一次 scipy 精修踩坑（見對話紀錄踩坑第 (3) 點）：精修出來
的解只檢查了桿尖高度、沒檢查地板淨空，實際上手臂連桿撞進地板（impulse
62~217）。這次不同的是精修**之後**還要過 Stage 1.5 的後擺地板淨空硬檢查，
精修目標函式本身也不涉及淨空（淨空要 13 次 `_set_pose()`，放進目標函式會
讓每次迭代成本暴增），是「先精修方向、再驗淨空」的兩段式，不是把安全條件
塞進目標函式賭它會滿足。"""

_CUE_LOCAL_AXIS = (0.0, 1.0, 0.0)
"""2026-09-02 修正：球桿實體的軸向是 end effector 的 **Y 軸**，不是 Z 軸。

`assets/ball_stick.usda` 的 Cylinder 是 `axis="Y"`，而
`StageAPIImpl.align_prim_to_target()` 把球桿的世界變換**直接設成 end
effector 的世界變換**，所以球桿實體沿的是 ee 的 local Y。專案既有慣例本來
就是 Y——`ArticulationAPIImpl.move_swing()` 用 `[0,1,0]`、WAM7 常數的量測
來源 `scripts/probe_palm_yaw_correction.py` 用 `[0, CUE_STICK_GRIP_TO_TIP,
0]`、`cue_pose_calculator._shortest_arc_quat()` 註解寫明「從 +Y 轉到」。
2026-09-01 這條 UR3e 線的搜尋/驗證腳本一律誤用了 `[0,0,1]`（ee 的 Z 軸），
`scripts/probe_cue_axis_and_clearance.py` 實測對照確認：解析桿尖算在
(-0.037,-2.098,0.182)，實體球桿 bbox 卻落在 (1.25~1.28, -3.26~-1.84,
-0.61~-0.07)，兩者差約 1.3m，完全不是同一條線。"""

_CUE_BUTT_LOCAL_OFFSET_M = -0.15
"""球桿握把端（butt）相對 ee 的偏移量：`ball_stick.usda` 的 Cylinder
height=1.5、沿 local Y、`xformOp:translate=(0, 0.6, 0)`，所以圓柱在 local Y
上從 -0.15 延伸到 +1.35（=`CUE_STICK_GRIP_TO_TIP`）。做地板淨空檢查時球桿
本體要整段檢查，不能只看桿尖那一端。"""

_CUE_RADIUS_M = 0.01

_FLOOR_Z_WORLD = -0.7695
"""`assets/billiard_env.usda` 的 SimpleRoom/GroundPlane `xformOp:translate`
z 值（球檯本身放在 z=0，所以這就是世界座標）。`scripts/probe_floor_
geometry.py` 實測 Towel_Room01_floor_bottom 上緣 -0.76957 與此一致，取
較高（較保守）的那個當地板高度。2026-09-01 的踩坑紀錄第 (4) 點當時放棄
「連桿地板淨空」限制式的理由是「不知道真實地板的精確世界座標，沒有可靠的
解析代理」——這個未知量現在量出來了。"""

_CONTACT_Z_WORLD = 0.028575
"""桿尖擊球點的世界高度＝檯面 z(0.0)＋球半徑。搜尋是在「基座放世界原點」的
local frame 做的，靠這個值把 local frame 換算回世界高度：正式部署時基座會被
平移到讓桿尖落在擊球點，所以 `base_z = _CONTACT_Z_WORLD - local_tip_z`。"""

_CONTACT_TO_FLOOR_M = _CONTACT_Z_WORLD - _FLOOR_Z_WORLD

_MIN_FLOOR_CLEARANCE_M = 0.05
"""後擺全程球桿本體離地板的最小容許淨空。2026-09-02 GUI 重跑實測：出事的
那組候選在 AIM 姿態就只剩 0.163m，後擺 7.5° 就觸地並讓整支手臂凍結
（`scripts/probe_cue_axis_and_clearance.py`），所以這個限制式必須涵蓋整段
後擺軌跡，不是只檢查靜態 AIM 姿態。"""

_MIN_BASE_HEIGHT_ABOVE_FLOOR_M = 0.0
_MAX_BASE_HEIGHT_ABOVE_FLOOR_M = 0.80
"""基座相對地板的高度窗口，取代舊的 `_MIN_TIP_HEIGHT_M`/`_MAX_TIP_HEIGHT_M`
（那兩個是「桿尖相對基座」的高度，跟「手臂會不會陷進地板」只有間接關係）。

下界 0：基座不可以低於地板（2026-09-01 踩坑第 (1) 點：基座被放到地板下
1.7m，整支手臂陷進地板物理直接爆掉）。

上界 0.80 ≈ 檯面相對地板的高度（`_CONTACT_TO_FLOOR_M`=0.798）：再高就代表
基座比擊球面還高，對「從側邊伸過去擊球」不合理。第一版設 0.40 太緊，實測
4704 組裡有 3599 組（76%）死在這一關——這反映一個實體事實：**檯面比地板高
0.798m，而 UR3e 是小型手臂（垂直可達約 0.74m），站在地板上根本搆不到檯面，
必須架在台座上**。台座高度是自由設計變數（`RobotArm.reposition()` 只設
translate，場景裡沒有台座實體幾何），所以這裡放寬讓搜尋自己決定，真正的
安全把關交給地板淨空限制式。"""

_BACKSWING_DEG = 30.0
_CLEARANCE_SAMPLES = 13
_CLEARANCE_SHORTLIST = 30
"""地板淨空檢查要真的把姿態擺出來量（每個取樣點一次 `_set_pose()`），成本
遠高於 Stage 1 的解析篩選，所以只對 Stage 1 分數前段的候選做。取 30 而不是
舊版 Stage 2 的 10，是因為淨空是新加的硬限制、可能刷掉不少原本的前段候選。"""
_MIN_AXIS_SWING_ALIGNMENT = 0.97
"""球桿軸向與「揮桿給桿尖的速度方向」的夾角餘弦下限。這兩個是不同的向量
（軸向是球桿指哪裡，速度方向是桿尖往哪裡走），但一個正常的推桿動作兩者
必須幾乎平行，否則就是「球桿指著球、桿尖卻往旁邊掃」。舊版只檢查速度
方向，完全沒有約束球桿軸向本身，是這次會選到「球桿指向地板」候選的原因
之一。"""

# ---- 2026-09-02：放寬「純肘關節驅動」限制 ----
# 用修正過的球桿軸向（Y 軸）重跑搜尋後，Stage 1 網格 4704 組全滅，且通過前面
# 篩選的候選裡最佳 axis_swing_alignment 只有 0.074（幾乎垂直，不是平行）——
# 診斷結果明確顯示這不是網格解析度問題：純肘關節轉動＋球桿剛性掛載在腕部，
# 這個自由度組合在修正後的幾何下，原則上就搆不出「桿尖沿著球桿軸向前推」
# 這個動作模式。加密網格或連續精修都不會改變這個結構性結論。
#
# 使用者決定放寬「其餘關節在揮桿全程必須完全靜止」這個限制，允許少數幾個
# 額外關節一起參與、共同決定揮桿方向，但**不是**回到 WAM7 `move_swing()`
# 那種全關節線性規劃（那樣就完全不是「肘關節為主」的人體化動作了）——做法
# 是加權最小範數解（weighted pseudo-inverse）：只解「桿尖線速度方向」這個
# 3 維目標（不是 WAM7 早期踩過坑的 6 維位置+朝向目標——6-DOF 非冗餘臂對 6
# 維目標沒有零空間可以讓權重矩陣發揮，但對 3 維純線速度目標，6 個關節
# （其中 shoulder_pan 另外用解析解，不參與這裡）還有 3 維零空間，加權矩陣
# 可以在這個零空間裡把解「拉」向偏好肘關節出力的方向）。
_SWING_DOF_INDICES = [1, 2, 3, 4, 5]
"""參與揮桿的關節在 [pan, shoulder_lift, elbow, wrist1, wrist2, wrist3] 裡
的索引——不含 pan（index 0）：pan 只負責解析對齊瞄準角的 XY 方向，跟揮桿本身
的動力學無關，這個既有分工不變。"""
_SWING_COST_WEIGHTS = [100.0, 1.0, 5.0, 5.0, 5.0]
"""跟 `_SWING_DOF_INDICES` 一一對應（`[shoulder_lift, elbow, wrist1, wrist2,
wrist3]`）的使用代價：肘關節=1（最鼓勵動）、wrist1/2/3=5（次要，人體揮桿
手腕本來就會有一點跟隨動作）、shoulder_lift=100（重懲罰，盡量不要整條上臂
一起擺動，維持跟原設計最接近的「主要看得出來是手肘在動」）。純 Python list
（不是 np.array）——頂層還沒 import numpy，用到的地方（`_solve_weighted_
swing_qdot_unit()`）自己轉型。"""
_SWING_DLS_LAMBDA = 1e-3
"""阻尼最小平方法的阻尼係數，避免候選姿態接近奇異點時解爆掉——跟
`ArticulationAPIImpl.DLS_LAMBDA`（WAM7 `move_swing()` 用的同一個機制）同一個
數量級，不是新發明的參數。"""


def _solve_weighted_swing_qdot_unit(Jv_tip, target_direction, cost_weights):
    """給定桿尖線性 Jacobian（`Jv_tip`，3×6）跟目標方向（單位向量），解出
    `_SWING_DOF_INDICES` 這幾個關節的加權最小範數 `qdot`，使得
    `Jv_tip[:, _SWING_DOF_INDICES] @ qdot_unit` 盡量對齊 `target_direction`
    （不保證完全平行——阻尼項會留一點殘差，呼叫端要用實際算出來的方向重新
    正規化，不能假設等於 `target_direction`）。

    `cost_weights`：`_SWING_DOF_INDICES` 對應位置的「使用代價」，數值越大
    代表這個關節越不被鼓勵參與（懲罰越重）——肘關節給最小代價（優先使用），
    `shoulder_lift` 給最大代價（盡量不動整條上臂）。標準加權最小範數公式：
    `qdot = W⁻¹Jᵀ(JW⁻¹Jᵀ+λ²I)⁻¹v_target`。
    """
    import numpy as np
    J_sub = Jv_tip[:, _SWING_DOF_INDICES]  # 3x5
    w_inv = np.diag(1.0 / np.asarray(cost_weights, dtype=float))
    JWJt = J_sub @ w_inv @ J_sub.T + (_SWING_DLS_LAMBDA ** 2) * np.eye(3)
    qdot_sub = w_inv @ J_sub.T @ np.linalg.solve(JWJt, target_direction)
    return qdot_sub


def _skew_matrix(v):
    import numpy as np
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _rotate_vector_by_quat(quat_wxyz, vec):
    import numpy as np
    w = quat_wxyz[0]
    q_xyz = quat_wxyz[1:]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    from core.services import swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    robot_prim_path = "/World/SearchUR3ePlacementConstants/Robot"
    stage_api.create_reference_prim(robot_prim_path, _UR3E_PATH)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=robot_prim_path)
    for _ in range(5):
        simulation_app.update()

    dof_max_velocities = np.asarray(articulation.get_dof_max_velocities())
    if hasattr(dof_max_velocities, "numpy"):
        dof_max_velocities = dof_max_velocities.numpy()
    dof_max_velocities = np.asarray(dof_max_velocities, dtype=float).reshape(-1)
    num_joints = dof_max_velocities.size

    lower_limits, upper_limits = articulation.get_dof_limits()
    lower_limits = np.asarray(lower_limits.numpy() if hasattr(lower_limits, "numpy") else lower_limits, dtype=float).reshape(-1)
    upper_limits = np.asarray(upper_limits.numpy() if hasattr(upper_limits, "numpy") else upper_limits, dtype=float).reshape(-1)
    joint_mid = (lower_limits + upper_limits) / 2.0
    joint_half_range = (upper_limits - lower_limits) / 2.0

    dof_names = list(articulation.dof_names) if hasattr(articulation, "dof_names") else None
    elbow_dof_index = (
        next((i for i, n in enumerate(dof_names) if "elbow" in n.lower()), _ELBOW_DOF_INDEX)
        if dof_names is not None else _ELBOW_DOF_INDEX
    )

    link_names = None
    for attr in ("link_names", "body_names"):
        if hasattr(articulation, attr):
            link_names = list(getattr(articulation, attr))
            break
    end_effector_link_name = "wrist_3_link"
    idx = link_names.index(end_effector_link_name)
    jac_probe = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
    if jac_probe.shape[0] == len(link_names) - 1:
        jac_link_index = idx - 1
    elif jac_probe.shape[0] == len(link_names):
        jac_link_index = idx
    else:
        raise RuntimeError(f"Jacobian link 數 {jac_probe.shape[0]} 與 link 名稱數 {len(link_names)} 對不上")

    end_effector_rigid_prim = RigidPrim(paths=f"{robot_prim_path}/{end_effector_link_name}")
    elbow_limit = dof_max_velocities[elbow_dof_index]

    def _set_pose(joints):
        articulation.set_dof_positions(joints[None, :])
        articulation.set_dof_velocities(np.zeros((1, num_joints)))
        for _ in range(3):
            simulation_app.update()

    tilt_rad = np.radians(_TILT_DEG)
    # target_direction：跟 cue_pose_calculator.compute_tilted_direction(0, tilt_rad)
    # 完全一致的公式（shot_angle_deg=0：dx=0,dy=1），這裡直接展開避免多繞一層
    # import（這支腳本只需要方向本身，不需要完整的球檯幾何）。
    target_direction = np.array([0.0, np.cos(tilt_rad), -np.sin(tilt_rad)])
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    print(f"[search] tilt_deg={_TILT_DEG}  target_direction={target_direction.tolist()}  required_tip_speed={required_tip_speed:.4f}")

    print("=== Stage 1：掃 shoulder_lift/elbow/wrist1/wrist2（pan=0） ===")
    stage1_candidates = []
    # 每層篩選各自的淘汰計數——網格全滅時要能一眼看出是卡在哪一個
    # 限制式（2026-09-02 加上正確軸向後第一次跑就是 672 組全滅，
    # 沒有這個計數只能靠猜）。
    rejects = dict.fromkeys(
        ("joint_limits", "singular", "base_height", "axis_z_tilt",
         "degenerate_speed", "swing_z_tilt", "axis_swing_alignment"), 0)
    # 診斷量：整個網格裡「球桿軸向 vs 純肘關節轉動的桿尖速度方向」的最佳
    # 平行度。連續三次網格全滅之後加的——如果這個值從來就靠近 0（垂直），
    # 代表問題不是網格不夠密，而是這個幾何下純肘關節轉動根本不可能讓桿尖
    # 沿著球桿走，加密網格再久也不會有解。
    best_alignment_seen = -2.0
    best_alignment_joints = None
    total_grid = (len(_SHOULDER_LIFT_CANDIDATES) * len(_ELBOW_CANDIDATES) * len(_WRIST1_CANDIDATES)
                  * len(_WRIST2_CANDIDATES) * len(_WRIST3_CANDIDATES))
    for shoulder_lift in _SHOULDER_LIFT_CANDIDATES:
        for elbow in _ELBOW_CANDIDATES:
            for wrist1 in _WRIST1_CANDIDATES:
                for wrist2 in _WRIST2_CANDIDATES:
                    for wrist3 in _WRIST3_CANDIDATES:
                        joints = np.array([0.0, shoulder_lift, elbow, wrist1, wrist2, wrist3])[:num_joints]
                        if np.any(joints < lower_limits) or np.any(joints > upper_limits):
                            rejects["joint_limits"] += 1
                            continue

                        _set_pose(joints)
                        jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
                        J = jac_all[jac_link_index]
                        singular_values = np.linalg.svd(J, compute_uv=False)
                        if singular_values.min() < 1e-4:
                            rejects["singular"] += 1
                            continue

                        ee_pos, ee_orient = end_effector_rigid_prim.get_world_poses()
                        local_grip_position = np.asarray(ee_pos[0], dtype=float)
                        ee_orient = np.asarray(ee_orient[0])
                        cue_axis = _rotate_vector_by_quat(ee_orient, np.array(_CUE_LOCAL_AXIS))
                        cue_axis = cue_axis / np.linalg.norm(cue_axis)
                        tip_offset = CUE_STICK_GRIP_TO_TIP * cue_axis
                        local_tip_position = local_grip_position + tip_offset

                        # 基座高度：正式部署時基座被平移到讓桿尖落在擊球點，
                        # 所以 base_z = _CONTACT_Z_WORLD - local_tip_z，相對地板的
                        # 高度就是 _CONTACT_TO_FLOOR_M - local_tip_z。
                        base_height_above_floor = _CONTACT_TO_FLOOR_M - float(local_tip_position[2])
                        if not (_MIN_BASE_HEIGHT_ABOVE_FLOOR_M
                                <= base_height_above_floor
                                <= _MAX_BASE_HEIGHT_ABOVE_FLOOR_M):
                            rejects["base_height"] += 1
                            continue

                        # 球桿軸向本身的傾斜必須對得上目標（繞基座 Z 軸的
                        # shoulder_pan 只改 XY 方向角、不改 Z 分量，所以 Z 分量
                        # 在 pan=0 就可以先篩）。舊版只篩速度方向、沒篩軸向。
                        axis_z_tilt_error = abs(float(cue_axis[2]) - float(target_direction[2]))
                        if axis_z_tilt_error > _PREFILTER_Z_TILT_ERROR:
                            rejects["axis_z_tilt"] += 1
                            continue

                        Jv = J[:3, :]
                        Jang = J[3:, :]
                        Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang
                        # 2026-09-02：放寬純肘關節限制——不再只取 elbow 那一欄
                        # Jacobian，改成對 `_SWING_DOF_INDICES` 這幾個關節解加權
                        # 最小範數 qdot（見 `_solve_weighted_swing_qdot_unit()`
                        # docstring），讓解出來的方向本來就偏向對齊
                        # `target_direction`，不是像舊版那樣先算出 elbow 自己的
                        # 固有方向、事後才檢查有沒有剛好對上。
                        qdot_unit_sub = _solve_weighted_swing_qdot_unit(
                            Jv_tip, target_direction, _SWING_COST_WEIGHTS
                        )
                        tip_velocity_unit = Jv_tip[:, _SWING_DOF_INDICES] @ qdot_unit_sub
                        speed_per_unit_scale = float(np.linalg.norm(tip_velocity_unit))
                        if speed_per_unit_scale < 1e-6:
                            rejects["degenerate_speed"] += 1
                            continue
                        direction_local = tip_velocity_unit / speed_per_unit_scale

                        z_tilt_error = abs(float(direction_local[2]) - float(target_direction[2]))
                        if z_tilt_error > _PREFILTER_Z_TILT_ERROR:
                            rejects["swing_z_tilt"] += 1
                            continue

                        # 推桿動作：桿尖必須沿著球桿指的方向走，不能橫掃。
                        axis_swing_alignment = float(np.dot(cue_axis, direction_local))
                        if axis_swing_alignment > best_alignment_seen:
                            best_alignment_seen = axis_swing_alignment
                            best_alignment_joints = joints.copy()
                        if axis_swing_alignment < _PREFILTER_AXIS_SWING_ALIGNMENT:
                            rejects["axis_swing_alignment"] += 1
                            continue

                        # scale_needed：把 qdot_unit_sub 放大到讓桿尖沿
                        # target_direction 的分量恰好等於 required_tip_speed 的
                        # 倍率；qdot_actual = scale_needed * qdot_unit_sub 才是
                        # 揮桿真正要下達的關節角速度（每個活動關節分開檢查各自
                        # 的轉速上限，不是只看肘關節）。
                        along_target = float(np.dot(tip_velocity_unit, target_direction))
                        if along_target < 1e-6:
                            rejects["degenerate_speed"] += 1
                            continue
                        scale_needed_est = required_tip_speed / along_target
                        qdot_actual_est = scale_needed_est * qdot_unit_sub
                        per_joint_margin = np.abs(qdot_actual_est) / dof_max_velocities[_SWING_DOF_INDICES]
                        elbow_margin_ratio_est = float(np.max(per_joint_margin))
                        range_deviation = float(np.max(np.abs((joints - joint_mid) / joint_half_range)))
                        # 2026-09-02 新增：base_distance_xy——這組姿態 pan=0 時
                        # local_tip_position 的 XY 分量大小，等於平移基座之後
                        # 「基座離目標接觸點」的水平距離（繞 Z 軸的 shoulder_pan
                        # 旋轉只改 XY 方向角，不改量值，見 Stage 2 說明），跟
                        # `_solve_base_position_and_joint_targets()` 裡
                        # `base_position = target_wrist_position - rotated_tip`
                        # 是同一個量。原本只用桿尖高度／Z 傾斜／餘裕排序，選到
                        # 過的候選（已驗證 96.1% 達成率那組）這個值高達 1.55m——
                        # 高度需求主要靠球桿（CUE_STICK_GRIP_TO_TIP=1.35m）以
                        # 接近水平的角度伸出去達成，不是靠手臂本身伸展，導致
                        # 真實場景裡機器人本體視覺上離球檯異常遠。這裡加進評分，
                        # 優先選手臂本身伸展、基座離目標點較近的候選。
                        base_distance_xy = float(np.linalg.norm(local_grip_position[:2]))

                        stage1_candidates.append({
                            "joints": joints.copy(),
                            "cue_axis": cue_axis,
                            "direction_local": direction_local,
                            "qdot_unit_sub": qdot_unit_sub.copy(),
                            "local_grip_position": local_grip_position,
                            "local_tip_position": local_tip_position,
                            "base_height_above_floor": base_height_above_floor,
                            "axis_swing_alignment": axis_swing_alignment,
                            "z_tilt_error": z_tilt_error,
                            "range_deviation": range_deviation,
                            "elbow_margin_ratio_est": elbow_margin_ratio_est,
                            "base_distance_xy": base_distance_xy,
                        })

    print(f"[search] Stage 1 網格總候選數={total_grid}  高度合理+Z傾斜對齊候選數={len(stage1_candidates)}")
    print(f"[search] Stage 1 各層淘汰數：{rejects}")
    print(f"[search] 診斷：通過前面各層預篩的姿態裡最佳 axis_swing_alignment={best_alignment_seen:.5f}"
          f"  joints={np.round(best_alignment_joints, 4).tolist() if best_alignment_joints is not None else None}")
    print("[search] 診斷說明：這個值代表『球桿指的方向』與『加權揮桿解（偏好肘關節，"
          "見 _SWING_COST_WEIGHTS）給桿尖的方向』最多能多平行。靠近 1 才代表推桿"
          "動作成立；靠近 0 代表桿尖是橫著掃過去，不是沿著球桿推——那是幾何結構"
          "問題，加密網格救不了。"
          "⚠️ 這個統計量只涵蓋『有走到這一層』的姿態（前面 base_height／"
          "axis_z_tilt／swing_z_tilt 預篩淘汰掉的不計入），不是整個網格的最大值。")
    if not stage1_candidates:
        print("[search] [WARN] 沒有任何候選通過 Stage 1 篩選，找不到解")
        return

    max_range_dev_1 = max(c["range_deviation"] for c in stage1_candidates) or 1.0
    max_z_err = max(c["z_tilt_error"] for c in stage1_candidates) or 1.0
    max_margin_est = max(c["elbow_margin_ratio_est"] for c in stage1_candidates) or 1.0
    max_base_dist = max(c["base_distance_xy"] for c in stage1_candidates) or 1.0
    for c in stage1_candidates:
        c["stage1_score"] = (
            (c["z_tilt_error"] / max_z_err) * 5.0
            + (c["range_deviation"] / max_range_dev_1)
            + (c["elbow_margin_ratio_est"] / max_margin_est) * 5.0
            + (c["base_distance_xy"] / max_base_dist) * 3.0
        )
    stage1_candidates.sort(key=lambda c: c["stage1_score"])

    print("[search] Stage 1 前 10 名：")
    for rank, c in enumerate(stage1_candidates[:10], start=1):
        print(f"  #{rank} score={c['stage1_score']:.4f}  z_tilt_error={c['z_tilt_error']:.5f}  "
              f"base_height={c['base_height_above_floor']:.4f}  align={c['axis_swing_alignment']:.4f}  "
              f"margin={c['elbow_margin_ratio_est']:.4f}  base_distance_xy={c['base_distance_xy']:.4f}  "
              f"joints={np.round(c['joints'], 4).tolist()}")

    # 2026-09-02 調整：精修種子改成按 margin_ratio 由小到大挑，不是沿用
    # Stage 1 的綜合分數排序——綜合分數把 z_tilt_error/base_distance_xy 也
    # 混進去，可能選到 alignment/距離很漂亮但 margin 高達 7~8 倍的候選當
    # 種子，Nelder-Mead 從那種起點出發，要把 margin 從 7~8 壓到 1 需要走
    # 很遠的路，容易把 alignment 一起拖壞（見 `_refine_cost()` 的說明）。
    # margin 越接近 1，精修要走的路越短，越不容易在收斂前就先把 alignment
    # 犧牲掉——直接針對這個「最難達成的硬條件」挑種子。
    margin_sorted_candidates = sorted(stage1_candidates, key=lambda c: c["elbow_margin_ratio_est"])
    print(f"[search] 精修種子改用 margin_ratio 排序（最小 5 個）：" + ", ".join(
        f"{c['elbow_margin_ratio_est']:.4f}" for c in margin_sorted_candidates[:5]
    ))

    print("")
    print(f"=== Stage 1.4：連續局部精修（前 {_REFINE_SHORTLIST} 名，依 margin_ratio 排序，Nelder-Mead）===")

    def _measure(joints):
        """把姿態擺出來，回傳這組關節角的所有幾何量。Stage 1 的內嵌計算是同一
        套公式，這裡抽成函式給精修的目標函式反覆呼叫。"""
        _set_pose(joints)
        jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
        J = jac_all[jac_link_index]
        ee_pos, ee_orient = end_effector_rigid_prim.get_world_poses()
        grip = np.asarray(ee_pos[0], dtype=float)
        axis = _rotate_vector_by_quat(np.asarray(ee_orient[0]), np.array(_CUE_LOCAL_AXIS))
        axis = axis / np.linalg.norm(axis)
        tip_offset = CUE_STICK_GRIP_TO_TIP * axis
        tip = grip + tip_offset
        base_height = _CONTACT_TO_FLOOR_M - float(tip[2])
        Jv_tip = J[:3, :] - _skew_matrix(tip_offset) @ J[3:, :]
        qdot_unit_sub = _solve_weighted_swing_qdot_unit(Jv_tip, target_direction, _SWING_COST_WEIGHTS)
        tip_velocity_unit = Jv_tip[:, _SWING_DOF_INDICES] @ qdot_unit_sub
        speed = float(np.linalg.norm(tip_velocity_unit))
        if speed < 1e-9:
            return None
        swing_dir = tip_velocity_unit / speed
        along = float(np.dot(tip_velocity_unit, target_direction))
        if along > 1e-9:
            scale_needed = required_tip_speed / along
            per_joint_margin = np.abs(scale_needed * qdot_unit_sub) / dof_max_velocities[_SWING_DOF_INDICES]
            margin_ratio = float(np.max(per_joint_margin))
        else:
            margin_ratio = 1e9
        return {
            "joints": np.asarray(joints, dtype=float).copy(),
            "cue_axis": axis,
            "direction_local": swing_dir,
            "qdot_unit_sub": qdot_unit_sub.copy(),
            "local_grip_position": grip,
            "local_tip_position": tip,
            "base_height_above_floor": base_height,
            "axis_z_tilt_error": abs(float(axis[2]) - float(target_direction[2])),
            "z_tilt_error": abs(float(swing_dir[2]) - float(target_direction[2])),
            "axis_swing_alignment": float(np.dot(axis, swing_dir)),
            "along_speed_per_unit_omega": along,
            "elbow_margin_ratio_est": margin_ratio,
        }

    def _refine_cost(free_joints):
        """精修目標函式。只放「靠微調關節角可以連續收斂」的量：兩個方向 Z
        分量誤差、軸向與揮桿方向的平行度、基座高度窗口、肘關節轉速餘裕。
        地板淨空刻意不放進來（見 `_REFINE_MAX_ITER` 說明）。"""
        joints = np.concatenate(([0.0], np.asarray(free_joints, dtype=float)))[:num_joints]
        over = np.maximum(0.0, joints - upper_limits) + np.maximum(0.0, lower_limits - joints)
        limit_penalty = float(np.sum(over ** 2)) * 1e3
        if limit_penalty > 0:
            return 1e3 + limit_penalty
        m = _measure(joints)
        if m is None:
            return 1e6
        height_violation = (
            max(0.0, _MIN_BASE_HEIGHT_ABOVE_FLOOR_M - m["base_height_above_floor"])
            + max(0.0, m["base_height_above_floor"] - _MAX_BASE_HEIGHT_ABOVE_FLOOR_M)
        )
        margin_violation = max(0.0, m["elbow_margin_ratio_est"] - 1.0)
        # 2026-09-02 調整：放寬純肘關節限制後，Stage 1 種子點的起始 margin
        # 常常落在 6~8（需要的角速度是上限的 6~8 倍），舊公式
        # `(margin_violation*10)**2` 在這個起始值下高達 3600~4900，把
        # alignment 項（最多 ~1）徹底壓過去——實測精修出來的候選 alignment
        # 掉到 0.43~0.56（優化器完全不管 alignment，只顧著壓 margin），
        # margin 卻也只壓到 1.002 附近就卡住（Nelder-Mead 是局部方法，起點
        # 離可行域太遠時，走的路徑會把關節角拖到一個 margin 剛好壓線、但
        # alignment 很差的局部最佳解，回不去了）。
        #
        # 改成不乘 10、也不在乘 10 之後才平方——margin_violation=6 時項數
        # 從 3600 降到 36，跟 alignment 項（*5.0 之後最多 ~5）數量級接近，
        # 讓兩者在整段精修過程都有影響力，不是「先無視 alignment 拚死壓
        # margin，壓到底才回頭」這種容易卡進壞局部最佳解的行為。margin 越
        # 接近可行（violation 越小）時這一項會二次方地快速縮小，最後階段
        # 仍然是 margin 主導收斂到 <=1，不影響原本「margin 是硬條件」這件
        # 事本身。
        return (
            m["axis_z_tilt_error"] ** 2
            + m["z_tilt_error"] ** 2
            + max(0.0, _MIN_AXIS_SWING_ALIGNMENT - m["axis_swing_alignment"]) ** 2 * 5.0
            + (height_violation * 10.0) ** 2
            + margin_violation ** 2
        )

    from scipy.optimize import minimize

    refined_candidates = []
    for rank, c in enumerate(margin_sorted_candidates[:_REFINE_SHORTLIST], start=1):
        result = minimize(
            _refine_cost,
            c["joints"][1:num_joints],
            method="Nelder-Mead",
            options={"maxiter": _REFINE_MAX_ITER, "xatol": 1e-5, "fatol": 1e-10},
        )
        joints = np.concatenate(([0.0], np.asarray(result.x, dtype=float)))[:num_joints]
        if np.any(joints < lower_limits) or np.any(joints > upper_limits):
            print(f"  #{rank} 精修結果超出關節限制，捨棄")
            continue
        m = _measure(joints)
        if m is None:
            continue
        passed = (
            m["axis_z_tilt_error"] <= _MAX_Z_TILT_ERROR
            and m["z_tilt_error"] <= _MAX_Z_TILT_ERROR
            and m["axis_swing_alignment"] >= _MIN_AXIS_SWING_ALIGNMENT
            and _MIN_BASE_HEIGHT_ABOVE_FLOOR_M <= m["base_height_above_floor"] <= _MAX_BASE_HEIGHT_ABOVE_FLOOR_M
            and m["elbow_margin_ratio_est"] <= 1.0
        )
        print(f"  #{rank} cost={result.fun:.3e}  axis_z_err={m['axis_z_tilt_error']:.5f}  "
              f"swing_z_err={m['z_tilt_error']:.5f}  align={m['axis_swing_alignment']:.5f}  "
              f"base_height={m['base_height_above_floor']:.4f}  margin={m['elbow_margin_ratio_est']:.4f}  "
              f"{'OK' if passed else '未通過硬條件'}")
        if passed:
            m["base_distance_xy"] = float(np.linalg.norm(m["local_grip_position"][:2]))
            refined_candidates.append(m)

    if not refined_candidates:
        print("[search] [WARN] 精修後沒有任何候選通過硬條件，找不到解")
        return

    refined_candidates.sort(key=lambda m: (m["elbow_margin_ratio_est"], m["base_distance_xy"]))
    stage1_candidates = refined_candidates
    print(f"[search] 精修後通過硬條件的候選數={len(refined_candidates)}")

    print("")
    print(f"=== Stage 1.5：後擺 {_BACKSWING_DEG}° 全程球桿地板淨空（前 {_CLEARANCE_SHORTLIST} 名）===")

    def _min_backswing_clearance(candidate):
        """把候選姿態的肘關節從 AIM 目標角往後擺方向掃一遍，回傳整段軌跡裡
        球桿本體離地板的最小淨空（公尺，負值代表插進地板）。

        座標換算：搜尋是在「基座放世界原點」的 local frame 做的，正式部署時
        基座會被平移到讓桿尖落在擊球點，所以 local frame 的某點 p 對應的
        世界高度是 `_CONTACT_Z_WORLD - tip_z_aim + p_z`，離地板的淨空就是
        `p_z - tip_z_aim + _CONTACT_TO_FLOOR_M`。`tip_z_aim` 用 AIM 姿態
        （後擺前）的桿尖高度，因為基座位置是由 AIM 姿態決定、整段動作固定
        不變。

        球桿本體整段都要檢查（`_CUE_BUTT_LOCAL_OFFSET_M` 到
        `CUE_STICK_GRIP_TO_TIP`），不是只看桿尖那一端——2026-09-02 實測
        出事時最低點確實不一定落在桿尖。手臂本身的連桿不在這個檢查範圍內
        （2026-09-01 踩坑第 (4) 點：對中段連桿套桿尖高度門檻會把已驗證安全
        的候選誤殺），仍然沿用「Stage 2 選出最終候選後靠真實場景 settle
        階段的碰撞回報把關」。"""
        tip_z_aim = float(candidate["local_tip_position"][2])
        backswing_rad = np.radians(_BACKSWING_DEG)
        # 2026-09-02：放寬純肘關節限制後，後擺不再只轉肘關節——把
        # `qdot_unit_sub`（Stage 1/1.4 解出來、偏好肘關節但允許其他關節參與
        # 的加權方向）正規化成「肘關節分量＝1」，讓肘關節仍然轉滿
        # `_BACKSWING_DEG`、其餘活動關節依同一個比例跟著後擺，維持跟正式
        # 揮桿執行時同一個相對運動方向，不是只憑肘關節單獨掃。
        elbow_sub_index = _SWING_DOF_INDICES.index(elbow_dof_index)
        qdot_unit_sub = candidate["qdot_unit_sub"]
        if abs(qdot_unit_sub[elbow_sub_index]) < 1e-9:
            return None
        backswing_direction_sub = qdot_unit_sub / qdot_unit_sub[elbow_sub_index]
        worst = None
        for i in range(_CLEARANCE_SAMPLES):
            frac = i / (_CLEARANCE_SAMPLES - 1)
            joints = candidate["joints"].copy()
            joints[_SWING_DOF_INDICES] = joints[_SWING_DOF_INDICES] - backswing_rad * frac * backswing_direction_sub
            if np.any(joints < lower_limits) or np.any(joints > upper_limits):
                return None
            _set_pose(joints)
            ee_pos, ee_orient = end_effector_rigid_prim.get_world_poses()
            grip = np.asarray(ee_pos[0], dtype=float)
            axis = _rotate_vector_by_quat(np.asarray(ee_orient[0]), np.array(_CUE_LOCAL_AXIS))
            axis = axis / np.linalg.norm(axis)
            cue_lowest_z = min(
                grip[2] + _CUE_BUTT_LOCAL_OFFSET_M * axis[2],
                grip[2] + CUE_STICK_GRIP_TO_TIP * axis[2],
            ) - _CUE_RADIUS_M
            clearance = cue_lowest_z - tip_z_aim + _CONTACT_TO_FLOOR_M
            if worst is None or clearance < worst:
                worst = clearance
        return worst

    clearance_passed = []
    for rank, c in enumerate(stage1_candidates[:_CLEARANCE_SHORTLIST], start=1):
        clearance = _min_backswing_clearance(c)
        if clearance is None:
            print(f"  #{rank} 後擺會超出關節限制，跳過")
            continue
        c["min_backswing_clearance"] = clearance
        verdict = "OK" if clearance >= _MIN_FLOOR_CLEARANCE_M else "撞地板/淨空不足"
        print(f"  #{rank} min_clearance={clearance:+.4f} m  {verdict}")
        if clearance >= _MIN_FLOOR_CLEARANCE_M:
            clearance_passed.append(c)

    if not clearance_passed:
        print("[search] [WARN] 前段候選沒有任何一組能在後擺全程維持地板淨空，找不到解")
        return
    stage1_candidates = clearance_passed

    print("")
    print("=== Stage 2：解析解 shoulder_pan，確認對齊+可行性 ===")
    best = None
    for c in stage1_candidates:
        # required_pan 用**球桿軸向**對齊目標方向，不是用速度方向——決定
        # 「桿尖有沒有落在母球上」的是球桿指哪裡（`base_position` 由
        # `local_grip_position` 平移而來），速度方向只要跟軸向幾乎平行即可
        # （Stage 1 的 `_MIN_AXIS_SWING_ALIGNMENT` 已經保證），下面再驗一次。
        cue_axis = c["cue_axis"]
        angle_current_xy = float(np.arctan2(cue_axis[1], cue_axis[0]))
        angle_target_xy = float(np.arctan2(target_direction[1], target_direction[0]))
        required_pan = angle_target_xy - angle_current_xy
        required_pan = (required_pan + np.pi) % (2 * np.pi) - np.pi

        joints = c["joints"].copy()
        joints[0] = required_pan
        if np.any(joints < lower_limits) or np.any(joints > upper_limits):
            print(f"[search] required_pan={np.degrees(required_pan):.2f}° 超出 shoulder_pan 關節限制，跳過")
            continue
        _set_pose(joints)

        jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
        J = jac_all[jac_link_index]
        ee_pos, ee_orient = end_effector_rigid_prim.get_world_poses()
        ee_orient = np.asarray(ee_orient[0])
        cue_axis_after_pan = _rotate_vector_by_quat(ee_orient, np.array(_CUE_LOCAL_AXIS))
        cue_axis_after_pan = cue_axis_after_pan / np.linalg.norm(cue_axis_after_pan)
        tip_offset = CUE_STICK_GRIP_TO_TIP * cue_axis_after_pan

        axis_alignment = float(np.dot(cue_axis_after_pan, target_direction))
        if axis_alignment < _MIN_ALIGNMENT:
            print(f"[search] required_pan={np.degrees(required_pan):.2f}°  axis_alignment={axis_alignment:.5f}  球桿軸向未對齊，跳過")
            continue

        Jv = J[:3, :]
        Jang = J[3:, :]
        Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang
        # 2026-09-02：跟 Stage 1/1.4 用同一套加權揮桿解，不是只看 elbow 那欄。
        qdot_unit_sub = _solve_weighted_swing_qdot_unit(Jv_tip, target_direction, _SWING_COST_WEIGHTS)
        tip_velocity_unit = Jv_tip[:, _SWING_DOF_INDICES] @ qdot_unit_sub
        speed_per_unit_scale = float(np.linalg.norm(tip_velocity_unit))
        along_target = float(np.dot(tip_velocity_unit, target_direction))
        alignment = along_target / speed_per_unit_scale if speed_per_unit_scale > 1e-6 else -1.0

        if alignment < _MIN_ALIGNMENT:
            print(f"[search] required_pan={np.degrees(required_pan):.2f}°  alignment={alignment:.5f}  未達門檻，跳過")
            continue

        scale_needed = required_tip_speed / along_target
        qdot_actual = scale_needed * qdot_unit_sub
        per_joint_margin = np.abs(qdot_actual) / dof_max_velocities[_SWING_DOF_INDICES]
        margin_ratio = float(np.max(per_joint_margin))
        feasible = bool(margin_ratio <= 1.0 + 1e-9)
        print(f"[search] required_pan={np.degrees(required_pan):.2f}°  axis_alignment={axis_alignment:.5f}  "
              f"alignment={alignment:.5f}  margin_ratio={margin_ratio:.4f}  "
              f"qdot_actual={np.round(qdot_actual, 4).tolist()}  feasible={feasible}")
        if not feasible:
            continue

        best = {
            "joints_pan0": c["joints"],
            "cue_axis": c["cue_axis"],
            "direction_local": c["direction_local"],
            "local_grip_position": c["local_grip_position"],
            "base_height_above_floor": c["base_height_above_floor"],
            "min_backswing_clearance": c["min_backswing_clearance"],
            "speed_per_unit_scale": along_target,
            "qdot_unit_sub": qdot_unit_sub,
            "margin_ratio": margin_ratio,
        }
        break

    if best is None:
        print("[search] [WARN] 通過地板淨空的候選都無法在 Stage 2 通過對齊/可行性檢查")
        return

    print("")
    print("=== 結果：可直接貼進 ur3e_placement_calculator.py 的常數 ===")
    joints_pan0 = best["joints_pan0"]
    print(f"tilt_deg={_TILT_DEG}")
    print(f"joints_pan0 (shoulder_lift, elbow, wrist1, wrist2, wrist3) = {tuple(round(float(v), 6) for v in joints_pan0[1:])}")
    print(f"cue_axis_local (pan=0 時球桿軸向，供解 required_pan 用) = {tuple(round(float(v), 6) for v in best['cue_axis'])}")
    print(f"direction_local (pan=0 時加權揮桿解給桿尖的速度方向) = {tuple(round(float(v), 6) for v in best['direction_local'])}")
    print(f"local_grip_position (pan=0、基座在原點時的**握把**世界座標) = {tuple(round(float(v), 6) for v in best['local_grip_position'])}")
    print(f"speed_per_unit_scale = {best['speed_per_unit_scale']:.10f}  "
          f"（qdot_unit_sub 整組乘上 scale 之後，桿尖沿 target_direction 的速度分量）")
    print(f"swing_dof_indices (對應 [pan,shoulder_lift,elbow,wrist1,wrist2,wrist3] 的索引) = {_SWING_DOF_INDICES}")
    print(f"qdot_unit_sub (對應 swing_dof_indices 的加權關節角速度方向，乘上 scale_needed 後才是實際下達的 qdot) = "
          f"{tuple(round(float(v), 6) for v in best['qdot_unit_sub'])}")
    print(f"base_height_above_floor = {best['base_height_above_floor']:.4f} m")
    print(f"min_backswing_clearance = {best['min_backswing_clearance']:.4f} m")
    print(f"base_distance_xy (基座離目標握把點的水平距離) = {float(np.linalg.norm(best['local_grip_position'][:2])):.4f} m")
    print(f"margin_ratio (所有活動關節裡最吃緊的那個，需求/上限) = {best['margin_ratio']:.6f}")
    print("")
    print("⚠️ 這組結果現在是「多關節加權揮桿」，不是單純肘關節——`ur3e_placement_")
    print("calculator.py`／`ArticulationAPIImpl.move_swing_elbow_pivot()` 目前都是")
    print("寫死只驅動一個關節（elbow）的量身設計，要真的採用這組結果需要先把production")
    print("那兩處也改成能同時驅動 swing_dof_indices 這幾個關節（用同一個 qdot_unit_sub")
    print("方向、同一條純量 quintic 曲線縮放），這是接下來要做的實作，這支腳本只負責")
    print("證實「放寬限制後有沒有可行解」。")


if __name__ == "__main__":
    import traceback

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
