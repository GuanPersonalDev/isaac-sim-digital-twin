"""
scripts/measure_rolling_friction.py — Issue #81 滾動摩擦／扭轉阻尼參數量測

用法：透過 Tool Menu Registry 在 Kit 主選單「Tools > Billiard/Rolling Friction
Test - Run」點擊執行。前置條件：
  - Table_0（訓練桌）已經在場景中建好（extension 的 _billiard_init 已跑過）
  - timeline 目前正在 Play
  - 建議先把 Debug Menu 的「Training」開關關閉，避免狀態機（ScriptController）
    同時把球 RESET 回起始位置，干擾這裡的量測

不需要手動把其他球移開或關掉：工具執行時會先把母球傳送到桌面中央
（球堆固定放在 +Y 那一側，見 core/services/break_shot_position_provider.py
的 BREAK_SHOT_POSITIONS），往 -Y 方向（遠離球堆）滾動，並持續監控球的位置，
一旦接近桌緣護欄就提前結束取樣，避免撞到球堆或護欄污染量測數據。

不 import core/ 任何模組，直接用原生 API（比照 scripts/measure_swing_speed.py
的既有慣例，見該檔案開頭說明）。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

# 理由同 scripts/measure_swing_speed.py：把 extension/ 加進 sys.path、以
# "ui.tool_menu_registry" 這個模組名稱 import，避免跟 extension 本體對同一支
# 檔案產生兩份不同模組、各自獨立的 _REGISTERED_TOOLS 清單。
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import omni.usd

from ui.tool_menu_registry import tool_menu_item

# 注意：不要在檔案最上層 import isaacsim.* 底下「本身也是獨立 Kit extension」的
# 子模組（例如 isaacsim.core.experimental.prims）。discover_and_register 會在
# extension on_startup 當下把整支檔案 import 一次以觸發 decorator 註冊；若這些
# 模組在檔案最上層被 import，可能在其底層 DLL 尚未載入完成時就被強制觸發，
# 導致 "DLL load failed" 之類的錯誤（見 skills/isaac_sim_6_api_cache.md 對應
# 章節）。這些模組延後到函式「真正執行」時才 import。

BALL_PRIM_PATH = "/World/Table_0/Balls/Ball_0"
BALL_RADIUS = 0.028575  # m，撞球半徑，跟 core/services/break_shot_position_provider.py 的 _D/2 一致

# 桌面（Surface）尺寸見 assets/billiard_env.usda：scale=(1.27, 2.54, 0.05)，
# 半寬 0.635m、半長 1.27m。球堆固定在 +Y 側（BREAK_SHOT_POSITIONS 的 y 全是正值，
# 最遠到 _FOOT + 4*_ROW ≈ 0.83），母球測試起點選在桌面中央 (0, 0)，往 -Y（遠離
# 球堆的方向）滾動，-Y 側到護欄內緣淨空約 1.27m，扣安全邊界後留給滾動的距離。
TEST_START_XY = (0.0, 0.0)
ROLL_DIRECTION_XY = (0.0, -1.0)  # 單位向量，遠離球堆的方向
Y_SAFETY_BOUND = -1.1  # 球的 y 座標低於這個值（接近 -Y 護欄）就提前結束取樣

INITIAL_SPEED = 0.4  # m/s，測試用初速。直接給「已經在純滾動狀態」的初始條件
# （線速度 + 對應角速度同時給），刻意跳過滑動轉滾動的過渡階段，只單純量測
# 「純滾動之後」的減速表現，避免兩種摩擦效應（slide friction 與 rolling
# resistance）疊在一起分不清楚。速度刻意選得比較保守：即使目前
# torsionalPatchRadius 還沒調（滾動阻力接近 0），球在達到 Y_SAFETY_BOUND
# 前的滾動距離／時間也還在合理量測範圍內，不會太快逼近護欄。
PHYSICS_DT = 1.0 / 60.0
MAX_DURATION_S = 10.0
STOP_SPEED_THRESHOLD = 0.02  # m/s，球速低於此值視為已停止，提前結束取樣

TARGET_ROLLING_FRICTION_COEFF = 0.01  # 撞球公認滾動摩擦係數量級（0.005~0.015 中間值）
GRAVITY = 9.81


async def _run_rolling_friction_test_async():
    import omni.timeline
    import omni.kit.app
    from isaacsim.core.experimental.prims import RigidPrim

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        raise RuntimeError("目前 timeline 沒有在 Play，請先按 Play 讓 Table_0 的物理場景初始化後再執行本工具。")

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(BALL_PRIM_PATH).IsValid():
        raise RuntimeError(f"找不到 {BALL_PRIM_PATH}，確認 Table_0 已經建好（stage 已開啟、extension 已初始化）。")

    ball = RigidPrim(paths=BALL_PRIM_PATH)

    app = omni.kit.app.get_app()

    # 先把球傳送到桌面中央、遠離球堆的安全起點，並歸零殘留速度，確保每次
    # 測試都是同樣乾淨的初始條件，不受球先前被打去哪裡影響。
    start_x, start_y = TEST_START_XY
    start_position = np.array([[start_x, start_y, BALL_RADIUS]])
    ball.set_world_poses(positions=start_position)
    ball.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
    await app.next_update_async()  # 讓傳送/歸零速度先生效一個 frame

    # 純滾動（無滑動）初始條件：球沿 ROLL_DIRECTION_XY 方向水平前進，接觸點
    # （球心正下方，r_contact = (0, 0, -R)）的速度必須為 0（跟靜止桌面速度
    # 一致）：v_contact = v_center + ω × r_contact = 0。
    # 代入 v_center=(vx, vy, 0)、r_contact=(0, 0, -R) 解出 ω=(-vy/R, vx/R, 0)。
    dir_x, dir_y = ROLL_DIRECTION_XY
    vx, vy = dir_x * INITIAL_SPEED, dir_y * INITIAL_SPEED
    linear_velocity = np.array([[vx, vy, 0.0]])
    angular_velocity = np.array([[-vy / BALL_RADIUS, vx / BALL_RADIUS, 0.0]])
    ball.set_velocities(linear_velocity, angular_velocity)
    await app.next_update_async()  # 讓剛設定的速度先生效一個 frame

    times: list[float] = []
    speeds: list[float] = []
    t = 0.0
    total_steps = int(MAX_DURATION_S / PHYSICS_DT)
    for _ in range(total_steps):
        linear, _ = ball.get_velocities()
        v = linear.numpy()[0]
        speed = float(np.linalg.norm(v[:2]))  # 只看水平面（xy）速度，忽略垂直方向的沉降雜訊
        times.append(t)
        speeds.append(speed)

        positions, _ = ball.get_world_poses()
        ball_y = float(positions.numpy()[0][1])
        if ball_y < Y_SAFETY_BOUND:
            print(
                f"[RollingFrictionTest] 球的 y 座標到達 {ball_y:.3f}（低於安全邊界 {Y_SAFETY_BOUND}，接近護欄），"
                f"提前結束取樣，t={t:.2f}s"
            )
            break
        if speed < STOP_SPEED_THRESHOLD:
            print(
                f"[RollingFrictionTest] 速度降到 {speed:.4f} m/s（低於門檻 {STOP_SPEED_THRESHOLD}），"
                f"提前結束取樣，t={t:.2f}s"
            )
            break
        await app.next_update_async()
        t += PHYSICS_DT

    times_arr = np.array(times)
    speeds_arr = np.array(speeds)

    if len(times_arr) < 5:
        print(f"[RollingFrictionTest] 取樣點太少（{len(times_arr)} 筆），量測失敗，請確認球有沒有正確設定初速、"
              f"或是不是撞到護欄/其他球中斷了量測。")
        return

    # 線性迴歸：speed = intercept + slope * t，slope 應為負值，減速度 = -slope
    slope, _intercept = np.polyfit(times_arr, speeds_arr, 1)
    measured_deceleration = -float(slope)
    target_deceleration = TARGET_ROLLING_FRICTION_COEFF * GRAVITY

    print(f"[RollingFrictionTest] 取樣 {len(times_arr)} 筆，時間範圍 0 ~ {times_arr[-1]:.2f}s")
    print(f"[RollingFrictionTest] 初速={speeds_arr[0]:.3f} m/s，末速={speeds_arr[-1]:.3f} m/s")
    print(f"[RollingFrictionTest] 量測減速度 = {measured_deceleration:.4f} m/s²")
    print(f"[RollingFrictionTest] 目標減速度（μ_r={TARGET_ROLLING_FRICTION_COEFF}，真實撞球滾動摩擦係數量級）"
          f" = {target_deceleration:.4f} m/s²")

    if target_deceleration > 1e-9:
        ratio = measured_deceleration / target_deceleration
        print(f"[RollingFrictionTest] 比值（量測/目標）= {ratio:.2f}")
        if ratio > 1.2:
            print("[RollingFrictionTest] 減速太快 → torsionalPatchRadius 目前設太大，建議調小。")
        elif ratio < 0.8:
            print("[RollingFrictionTest] 減速太慢 → torsionalPatchRadius 目前設太小（或為 0／關閉），建議調大。")
        else:
            print("[RollingFrictionTest] 減速幅度接近目標範圍，數值合理。")
    else:
        print("[RollingFrictionTest] 目標減速度為 0，無法算比值（檢查 TARGET_ROLLING_FRICTION_COEFF 設定）。")


async def _print_errors(coro, tag: str):
    """coroutine 的例外若留給 asyncio 預設 logger，會在 Kit console 變成亂碼
    （編碼問題），統一在這裡攔下來用 print 印出可讀訊息。"""
    try:
        return await coro
    except Exception as exc:
        print(f"[{tag}] 執行失敗：{exc}")


@tool_menu_item("Billiard/Rolling Friction Test - Run")
def run_rolling_friction_test():
    """Issue #81：給 Table_0 母球一個已知的純滾動初速，量測滾動減速度，
    跟真實撞球滾動摩擦係數（μ_r≈0.01）換算出的目標減速度比對，協助調校
    assets/ball_template.usda 的 physxCollision:torsionalPatchRadius /
    minTorsionalPatchRadius。

    前置條件：Table_0 已建好、目前 timeline 正在 Play。點擊後立即返回，
    量測以 async coroutine 在背景逐 frame 執行，結果印在 Console。
    """
    import asyncio

    asyncio.ensure_future(_print_errors(_run_rolling_friction_test_async(), "RollingFrictionTest"))
    print("[RollingFrictionTest] 量測已開始（背景執行），請看 Console 結果。")
