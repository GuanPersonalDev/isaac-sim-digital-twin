
ACTION_DIM = 6

# RL 訓練時 policy 會依序取得，因此順序不可更動
# 每一項都是物理域（真實單位）的 (low, high)。

# 母球擺位，桌台相對座標（m）；世界座標需先扣除桌台 XY。
# X：桌面半寬 0.635 扣一顆球半徑 0.028575，得球心安全範圍。
CUE_BALL_PLACEMENT_X = (-0.606425, 0.606425)
# Y：Kitchen 的球心範圍。下界為 -1.27 + 球半徑；上界即 head string
# （Y = -0.635），母球球心可壓在線上。
CUE_BALL_PLACEMENT_Y = (-1.241425, -0.635)

# 擊球方向角（degree）。0° 朝桌台 +Y，正角朝 -X 增加。
#
# ⚠️ 物理域是半開區間 [-180, 180)，但 gymnasium 的 Box 只能表達閉區間，因此
# high 記為 180.0，視為「週期的另一端」而非可達上限。反正規化的尾端必須統一
# 折回 [-180, 180)（`rl_action_decoder._wrap_angle`），否則 +1 會還原成
# 180.0——與 -180.0 是同一個方向卻是不同數值。
#
# 端點選在 (-180, 180) 而非 (0, 360)：讓不連續點落在 ±180°（背對球堆），
# Gaussian policy 初始輸出集中的 normalized 0 對應開球最佳方向 0°（正對
# 球堆），推導過程見 docs/CHANGELOG.md（#231）。
#
# 🔴 Milestone A 期間收窄為 ±30°（#231 問題 2，2026-08-11，訓練信號密度
# 不足，見 docs/CHANGELOG.md 的 PPO 實測數據）。**Milestone B 之前必須改回
# (-180, 180) 並重訓**——走位球要能瞄任意方向。
#
# 30 這個值不是拍腦袋：母球從任何合法擺位瞄準 1 號球所需的最大角度是
# ±25.524°（kitchen 兩個 head string 角落），加上接觸窗口 ±2.062° 共
# ±27.586°，30 留 2.4° 餘裕。由
# `test_shot_angle_covers_every_legal_aim_at_the_one_ball` 從幾何現算釘住。
#
# ⚠️ 收窄之後區間不再涵蓋整圈，`[-30, 30]` 是閉區間而非半開——兩個端點是
#    不同方向，不該互相折回。`rl_action_decoder._wrap_angle()` 以區間中心
#    為錨折回，涵蓋整圈與收窄兩種情形都正確。
# ⚠️ `normalize_action()` 對超出可表達範圍的角度會拋 ValueError，刻意如此：
#    Milestone B 改回整圈時，任何殘留假設會大聲失敗而不是靜默算出越界值。
SHOT_ANGLE = (-30.0, 30.0)

# 母球目標初速（m/s），不是球桿桿尖速度。
#
# 上限來源：實測 Barrett WAM 差動 IK 桿尖峰值速度 2.5302 m/s，套用真實撞球
# 動量傳遞公式 v_ball = v_cue×(1+e)×M桿/(M桿+m球)（球桿 0.5kg、母球 0.163kg、
# 皮革頭恢復係數 e=0.75）換算得 3.3392 m/s。提高上界屬於手臂能力問題（桿尖
# 姿態優化，見 #176），不在本檔處理。
#
# 訓練桌（impulse strike）路徑把這個上限直接當成母球初速（見
# core/services/impulse_striking_service.py 的 compute_cue_ball_velocities，
# 1:1 直接賦值，不再套一次動量轉換）。
#
# 下限只約束 RL Action；執行期 no-op Action 可用 0.0（#110）。
#
# 0.65 的來源（#123）：純滾動水平減速度 μg=0.0981 m/s²，母球從 kitchen 最遠處
# 到 1 號球需要至少 0.5974 m/s，低於這個速度停下來之前根本碰不到球堆，
# 正規化域低端會是死區。取 0.65 讓任何合法擺位都還碰得到球堆（最差
# 0.256 m/s 抵達）。上界本身仍是策略瓶頸（PhysX 實測速度掃描顯示 spread 沒有
# 飽和、彈性仍達 1.36），推導數據見 docs/CHANGELOG.md。
CUE_BALL_SPEED = (0.65, 3.3392)

# 上下（索引 4）／左右（索引 5）擊球偏移，單位是**球半徑比例**，不是公尺。
# ±0.5R 已接近撞球物理的 miscue limit（約 0.5R，超過即滑桿）。
#
# ⚠️ 本檔全部是物理域，偏移量的 clamp 不在這把尺上執行：
#
#     policy 輸出（正規化域 [-1, 1]）
#           ↓ clamp_position_offset(offset, max_offset)   ← 在這裡
#           ↓ 反正規化 × 本檔的 ±0.5
#       Action.position_offset（物理域 ±0.5R）
#
# `max_offset ∈ [0, 1]` 的語意是「可用偏移能力的比例」，與正規化域的
# [-1, 1] 同一把尺，`hypot(offset)` 與它才能直接比較。若改成先反正規化再
# clamp，物理域的最大範數只有 hypot(0.5, 0.5) ≈ 0.707，
# `max_offset ∈ (0.707, 1.0]` 會整段變成死區，policy 分不出 0.8 與 1.0
# （見 #222）。
#
# Box 的上下限維持 ±0.5（物理域）／±1（正規化域），不可寫成 [0, 0.5]，
# 否則失去負方向偏移。
POSITION_OFFSET_VERTICAL = (-0.5, 0.5)
POSITION_OFFSET_HORIZONTAL = (-0.5, 0.5)

ACTION_BOUNDS = (
    CUE_BALL_PLACEMENT_X,
    CUE_BALL_PLACEMENT_Y,
    SHOT_ANGLE,
    CUE_BALL_SPEED,
    POSITION_OFFSET_VERTICAL,
    POSITION_OFFSET_HORIZONTAL,
)

ACTION_LOW = [low for low, _ in ACTION_BOUNDS]
ACTION_HIGH = [high for _, high in ACTION_BOUNDS]

# 中點式反正規化的兩個係數（見 `rl_action_decoder._denormalize`）。
#
# 需要換算尺度的呼叫端一律引用這兩個常數，**不要**自己從 ACTION_BOUNDS 重算
# (high ± low) / 2——那就是第二份實作，而換算漂移不會報錯（#228）。
ACTION_CENTER = [(low + high) / 2.0 for low, high in ACTION_BOUNDS]
ACTION_HALF_SPAN = [(high - low) / 2.0 for low, high in ACTION_BOUNDS]