
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
# ⚠️ 物理域是半開區間 [0, 360)，但 gymnasium 的 Box 只能表達閉區間，因此
# high 記為 360.0，把 360 視為「週期」而非可達上限。反正規化的尾端必須
# 統一做 `angle % 360.0` 把值收回 [0, 360)（#225 負責），否則正規化域的
# +1 會還原成 360.0 —— 與 0.0 是同一個方向卻是不同數值，兩端各自處理就
# 會出現只在邊界重現的不一致。
SHOT_ANGLE = (0.0, 360.0)

# 母球目標初速（m/s），不是球桿桿尖速度。
#
# 上限數值來源：2026-07-26 換裝 Barrett WAM + 差動 IK 後實測桿尖峰值速度
# 2.5302 m/s（預設姿態，見 docs/phase3-task-breakdown.md 出桿速度範圍列的
# 更新說明），套用真實撞球動量傳遞公式
# v_ball = v_cue×(1+e)×M桿/(M桿+m球)（球桿 0.5kg、母球 0.163kg、皮革頭
# 恢復係數 e=0.75，Dr. Dave Pool Info 引用範圍 0.71–0.75）換算得
# 2.5302×1.75×0.5/0.663 ≈ 3.3392 m/s。#176 UR5 的 1.313 m/s 已淘汰。
#
# 訓練桌（impulse strike）路徑把這個上限直接當成母球初速（見
# core/services/impulse_striking_service.py 的 compute_cue_ball_velocities，
# 目前 1:1 直接賦值，沒有再套一次動量轉換）。
#
# 下限只約束 RL Action；執行期 no-op Action 可用 0.0（#110）。
#
# 0.65 的來源（#123，2026-08-10）：原本的 0.5 有一段「母球滾不到球堆」的死區。
# 訓練端每個 tick 把角速度寫成純滾動分量（rolling_resistance_service），球永不
# 滑動，水平減速度就只有 μg = 0.01×9.81 = 0.0981 m/s²，因此母球的可滾行程是
# v²/(2×0.0981)。母球從 kitchen 最遠處（Y = -1.241425）到 1 號球（Y = 0.635）
# 要走 1.8193 m（扣掉接觸時的 2R），至少需要 0.5974 m/s——0.5 m/s 在停下來之前
# 根本碰不到球堆，正規化域低端整段是死區（policy 分不出裡面的差別）。
#
#   起點            行程      v_min    0.50 抵達速度   0.65 抵達速度
#   head string   1.2129 m   0.4878     0.1097          0.4296
#   break spot    1.5304 m   0.5480     0.0000          0.3496
#   kitchen 最遠   1.8193 m   0.5974     0.0000          0.2560
#
# 取 0.65 是讓**任何**合法擺位都還碰得到球堆（最差 0.256 m/s）。這個下限本身
# 仍然打不散球，但至少每個動作都有物理後果。
#
# ⚠️ 2026-08-11 修正：這裡原本寫「spread 要到約 1.8 m/s 才飽和」，那個數字來自
#    一個被 RunPod 實測推翻的 2D 模型，**已證實錯誤**。真實 PhysX 的速度掃描
#    （各 500+ 筆 first_contact == 1）顯示完全沒有飽和：
#
#      1.3223 m/s  spread 0.01349   legal break  0.0%
#      1.9946 m/s  spread 0.01798   legal break  0.0%
#      2.6669 m/s  spread 0.03451   legal break 28.7%
#      3.3392 m/s  spread 0.04264   legal break 44.8%
#
#    扣掉 rack 基準後的彈性 d ln(spread-rack)/d ln(v) 在上界處仍有 1.36
#    （飽和的話會趨近 0），也就是**上界 3.3392 本身才是瓶頸**。
#
#    連帶的事實：低於約 2.6 m/s 的整段是策略上的嚴格劣勢——spread 更差，而且
#    legal break 掛零（4 顆碰顆星／有球進袋一個都達不到），保證吃 -0.5。所以
#    policy 會學成永遠輸出速度上界，這一維不會有有意義的策略。下界從 0.5 提到
#    0.65 消掉的只是「碰不到球堆」那一小段，不代表 0.65 以上就都可用。
#
#    提高上界屬於手臂能力問題（桿尖姿態優化，見 #176），不在本檔處理。
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