"""手動擊球參數面板（#115）像素↔物理量換算的純函式集合。

統一慣例：**像素座標一律是 widget local**（左上原點、y 往下增加），由
`extension/ui/hud_panel.py` 負責把 `set_mouse_*_fn` 拿到的座標換算成這裡
吃的座標；本檔完全不知道螢幕座標系或 omni.ui 的存在。

**必須重用既有實作，不要自己重寫**：
- 偏移裁切固定走 `position_offset_limiter.clamp_position_offset()`（圓形
  裁切）——逐軸 clip 會轉向，而偏移方向就是加旋方向（#222）。
- 正規化／反正規化固定走 `rl_action_decoder.normalize_axis()` /
  `denormalize_axis()`，不自己乘 0.5 或除以 half_span——兩個方向都公開
  是為了對稱，避免下一個人只看到 `denormalize_axis` 是公開的，就自己重算
  正規化那一半（換算漂移不會報錯，#228）。
- 偏移的處理順序固定 **正規化 → 圓形裁切 → 反正規化**，跟
  `decode_rl_action()` 同一條順序，理由同上。

**方向慣例**（跟 `cue_pose_calculator.compute_contact_point()` 對齊，由
`test_cue_pose_calculator.py:83` 釘住）：畫面往上 = `position_offset[0]`
為正 = 上塞；畫面往右 = `position_offset[1]` 為正 = 右塞。**圓不隨
`shot_angle` 旋轉**——`compute_contact_point()` 的 `e_up`/`e_side` 是以
桿身方向現算的正交基，這個圓代表的是「射手視角」的球面（永遠上下左右），
不是「桿身視角」，旋轉圓面反而會讓使用者的直覺跟實際偏移方向對不上。

俯瞰圖的 Y 軸同理反向：pixel y 往下增加，但桌台座標 +Y 朝 foot end（見
`docs/architecture-spec.md` 的座標系規範），畫面「上」對應桌台「+Y」，跟
偏移圓的上塞方向慣例一致，兩者不是巧合——都是「畫面往上＝正方向」這條
單一直覺的兩處體現。

擺位的合法域是矩形（`clamp_cue_ball_placement`），偏移的合法域是圓
（`clamp_position_offset`）：這是刻意不同的兩種裁切，逐軸夾矩形依然是
最近的合法點，逐軸夾圓形會轉向，**不要把兩者「統一」成同一套邏輯**。
"""

import math
from collections.abc import Sequence

from ..models.action_bounds import ACTION_DIM
from ..models.manual_shot_bounds import (
    CUE_BALL_PLACEMENT_X,
    CUE_BALL_PLACEMENT_Y,
    MANUAL_SHOT_ANGLE,
)
from ..models.table_ball_set import TableBallSet
from .cue_pose_calculator import compute_tilted_direction
from .numeric_validation import validate_2d_value, validate_finite_number
from .position_offset_limiter import clamp_position_offset
from .rl_action_decoder import denormalize_axis, normalize_axis
from .spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH

# 跟 rl_action_decoder.py 同一個慣例：用 range(ACTION_DIM) 拆出索引，而不是
# 寫死 4、5——欄位順序是 #110 的契約，這裡只是借用同一份契約，不重新定義。
_, _, _, _, _OFFSET_VERTICAL_INDEX, _OFFSET_HORIZONTAL_INDEX = range(ACTION_DIM)

# 兩點距離小於一顆球半徑就視為方向未定義（貼在同一個點上，atan2 數值上會
# 抓到雜訊方向而不是使用者的意圖）。引用單一來源常數，不寫死 0.028575。
_MIN_AIM_DISTANCE_M = TableBallSet.DEFAULT_BALL_RADIUS

# 桌面矩形邊界（桌台相對座標），供 aim_line_endpoint() 的射線／矩形求交
# 使用。一律從 spread_score_calculator 的單一來源算出，不得寫死 1.27/2.54。
_TABLE_HALF_WIDTH_M = TABLE_WIDTH / 2.0
_TABLE_HALF_LENGTH_M = TABLE_LENGTH / 2.0


def _require_positive(value: float, field_name: str) -> float:
    numeric_value = validate_finite_number(value, field_name)
    if numeric_value <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return numeric_value


def offset_from_circle_pixels(
    marker_px: float,
    marker_py: float,
    center_px: float,
    center_py: float,
    radius_px: float,
    max_offset: float,
) -> list[float]:
    """圓形擊球點選擇器的標記像素位置 → `Action.position_offset`（物理域）。

    處理順序固定 正規化 → 圓形裁切 → 反正規化，跟 `decode_rl_action()`
    完全一致；`max_offset` 語意也相同（可用偏移能力的比例，`[0, 1]`），
    圓形裁切前後都在正規化域比較，物理域才不會出現死區（#222 的教訓）。
    """
    marker_px = validate_finite_number(marker_px, "marker_px")
    marker_py = validate_finite_number(marker_py, "marker_py")
    center_px = validate_finite_number(center_px, "center_px")
    center_py = validate_finite_number(center_py, "center_py")
    radius_px = _require_positive(radius_px, "radius_px")

    # 正規化：pixel delta 除以半徑轉成跟 decode_rl_action() 同尺度的
    # [-1, 1] 域。y 分量要反號——pixel y 往下增加，但畫面往上才是上塞。
    normalized_vertical = (center_py - marker_py) / radius_px
    normalized_horizontal = (marker_px - center_px) / radius_px

    normalized_vertical, normalized_horizontal = clamp_position_offset(
        [normalized_vertical, normalized_horizontal], max_offset
    )

    return [
        denormalize_axis(normalized_vertical, _OFFSET_VERTICAL_INDEX),
        denormalize_axis(normalized_horizontal, _OFFSET_HORIZONTAL_INDEX),
    ]


def circle_pixels_from_offset(
    position_offset: Sequence[float],
    center_px: float,
    center_py: float,
    radius_px: float,
) -> tuple[float, float]:
    """`offset_from_circle_pixels()` 的反方向：物理偏移 → 標記像素位置，
    用來在面板重繪時把常駐 controller 目前的參數畫回圓上（例如切回手動
    模式、或 Timeline Stop→Play 之後恢復畫面）。

    參數用 `Sequence[float]` 而非 `list[float]`：呼叫端常常是
    `ManualShotParameters.position_offset`（`tuple`，因為那個 dataclass
    是 frozen），硬性要求 `list` 只會逼呼叫端多包一層轉型，比照
    `numeric_validation.validate_2d_value()` 的作法放寬成序列。

    換算固定走 `rl_action_decoder.normalize_axis()`（`denormalize_axis()` 的
    反函式），不自己重算一次除法——即使偏移兩軸的 center 恰為 0、數學上
    跟直接除以 half_span 等價，這裡仍然只引用單一來源，避免換算漂移
    （#228）。
    """
    offset_vertical, offset_horizontal = validate_2d_value(
        position_offset, "position_offset"
    )
    center_px = validate_finite_number(center_px, "center_px")
    center_py = validate_finite_number(center_py, "center_py")
    radius_px = _require_positive(radius_px, "radius_px")

    normalized_vertical = normalize_axis(offset_vertical, _OFFSET_VERTICAL_INDEX)
    normalized_horizontal = normalize_axis(
        offset_horizontal, _OFFSET_HORIZONTAL_INDEX
    )

    marker_px = center_px + normalized_horizontal * radius_px
    marker_py = center_py - normalized_vertical * radius_px
    return marker_px, marker_py


def table_xy_from_topview_pixels(
    px: float, py: float, view_width_px: float, view_height_px: float
) -> tuple[float, float]:
    """俯瞰圖像素座標 → 桌台相對座標（m）。

    畫布中心對應桌台原點，X 軸方向與畫面一致（右為正），Y 軸反向（畫面
    往上＝桌台 +Y，見本檔案級 docstring）。桌面尺寸一律取自
    `spread_score_calculator.TABLE_WIDTH/TABLE_LENGTH`，不得硬編碼。
    """
    px = validate_finite_number(px, "px")
    py = validate_finite_number(py, "py")
    view_width_px = _require_positive(view_width_px, "view_width_px")
    view_height_px = _require_positive(view_height_px, "view_height_px")

    x = (px / view_width_px - 0.5) * TABLE_WIDTH
    y = (0.5 - py / view_height_px) * TABLE_LENGTH
    return x, y


def topview_pixels_from_table_xy(
    x: float, y: float, view_width_px: float, view_height_px: float
) -> tuple[float, float]:
    """`table_xy_from_topview_pixels()` 的反方向，用來把桌面上的固定幾何
    （Kitchen 合法區、袋口、開球球堆、瞄準線）畫回俯瞰圖畫布。
    """
    x = validate_finite_number(x, "x")
    y = validate_finite_number(y, "y")
    view_width_px = _require_positive(view_width_px, "view_width_px")
    view_height_px = _require_positive(view_height_px, "view_height_px")

    px = (x / TABLE_WIDTH + 0.5) * view_width_px
    py = (0.5 - y / TABLE_LENGTH) * view_height_px
    return px, py


def clamp_cue_ball_placement(x: float, y: float) -> tuple[float, float]:
    """把母球擺位逐軸夾回合法矩形域（Kitchen 範圍）內。

    這跟偏移的圓形裁切刻意不同：擺位合法域是矩形，逐軸夾出來仍然是矩形內
    離原點最近的合法點；偏移合法域是圓，逐軸夾會改變方向（#222）。兩者是
    不同形狀的裁切問題，不要為了程式碼看起來一致而合併成同一套邏輯。
    """
    x = validate_finite_number(x, "x")
    y = validate_finite_number(y, "y")

    clamped_x = min(max(x, CUE_BALL_PLACEMENT_X[0]), CUE_BALL_PLACEMENT_X[1])
    clamped_y = min(max(y, CUE_BALL_PLACEMENT_Y[0]), CUE_BALL_PLACEMENT_Y[1])
    return clamped_x, clamped_y


def shot_angle_from_points(
    cue_ball_xy: tuple[float, float], target_xy: tuple[float, float]
) -> float:
    """由母球位置瞄準俯瞰圖上任一點，反推 `shot_angle`（degree）。

    公式 `degrees(atan2(-(tx-cx), ty-cy))` 與
    `cue_pose_calculator.compute_tilted_direction()` 的
    `(-sinθ, cosθ)` 互為反函式（見本模組測試的往返驗證）。值域是半開區間
    `[-180, 180)`——正對頭庫（母球正後方）回 `-180.0` 不是 `+180.0`，這是
    `-(tx - cx)` 在 `tx == cx` 時保留負零號、`atan2` 對帶號零有既定行為
    的結果，寫成 `cx - tx` 或先取絕對值都會弄丟這個號誌，不要「化簡」。

    兩點距離小於一顆球半徑時方向在數值上未定義（貼在同一點，atan2 只會
    放大浮點雜訊），拋 `ValueError` 而不是回傳一個沒有意義的角度。
    """
    cue_x, cue_y = validate_2d_value(cue_ball_xy, "cue_ball_xy")
    target_x, target_y = validate_2d_value(target_xy, "target_xy")

    dx = target_x - cue_x
    dy = target_y - cue_y
    if math.hypot(dx, dy) < _MIN_AIM_DISTANCE_M:
        raise ValueError(
            "target_xy 與 cue_ball_xy 距離小於一顆球半徑，瞄準方向未定義"
        )

    return math.degrees(math.atan2(-dx, dy))


def clamp_manual_shot_angle(angle_deg: float) -> float:
    """把任意角度夾進 `manual_shot_bounds.MANUAL_SHOT_ANGLE` 的安全區。

    面板端的最後一道防線：`shot_angle_from_points()` 算出來的角度理論上
    涵蓋整個 `[-180, 180)`，但手動面板受限於機械臂基座碰撞（見
    `manual_shot_bounds.py` 的推導），必須在這裡收窄到 ±160°。
    """
    angle_deg = validate_finite_number(angle_deg, "angle_deg")
    low, high = MANUAL_SHOT_ANGLE
    return min(max(angle_deg, low), high)


def aim_line_endpoint(
    cue_ball_xy: tuple[float, float], shot_angle_deg: float
) -> tuple[float, float]:
    """由母球位置與 `shot_angle` 算出瞄準線的另一端點：從母球沿擊球方向
    射出的射線，與桌面矩形邊界（桌台相對座標，`x ∈ [-TABLE_WIDTH/2,
    +TABLE_WIDTH/2]`、`y ∈ [-TABLE_LENGTH/2, +TABLE_LENGTH/2]`）的交點。

    沿用 `compute_tilted_direction(shot_angle_deg, 0.0)`（水平桿身、
    tilt=0）取方向——跟 `shot_angle_from_points()` 用同一份幾何定義，兩者
    互為反函式。

    標準的射線／矩形求交（slab method）：母球必然在矩形內（合法擺位域是
    矩形的子集），所以對每一軸只有「方向分量指向的那一側邊界」在射線前方
    （t > 0），逐軸算出走到那側邊界要的參數 t，取其中最小的正 t 就是先撞
    到的那面邊界——取最小而非最大，因為矩形內任一射線一定先撞到較近的
    那一側。方向分量為 0 的軸不會撞到那一側的任何邊界（射線跟那兩條邊平
    行），要跳過以避免除以零。

    母球恰好在邊界上且方向朝外時，撞到邊界的 t 是 0，回傳母球本身
    （零長度，面板不畫線）——這不是特例，是同一條公式在邊界上的自然結果。
    """
    cue_x, cue_y = validate_2d_value(cue_ball_xy, "cue_ball_xy")
    angle_deg = validate_finite_number(shot_angle_deg, "shot_angle_deg")

    direction = compute_tilted_direction(angle_deg, 0.0)
    dir_x, dir_y = float(direction[0]), float(direction[1])

    candidate_t: list[float] = []
    if dir_x > 0.0:
        candidate_t.append((_TABLE_HALF_WIDTH_M - cue_x) / dir_x)
    elif dir_x < 0.0:
        candidate_t.append((-_TABLE_HALF_WIDTH_M - cue_x) / dir_x)
    if dir_y > 0.0:
        candidate_t.append((_TABLE_HALF_LENGTH_M - cue_y) / dir_y)
    elif dir_y < 0.0:
        candidate_t.append((-_TABLE_HALF_LENGTH_M - cue_y) / dir_y)

    # direction 恆為單位向量（compute_tilted_direction 的 tilt=0 分量是
    # (-sinθ, cosθ)），兩軸不可能同時為 0，candidate_t 必然非空。
    t = max(min(candidate_t), 0.0)

    return (cue_x + t * dir_x, cue_y + t * dir_y)


def is_within_radius(
    px: float, py: float, target_px: float, target_py: float, radius_px: float
) -> bool:
    """判斷像素座標 `(px, py)` 是否落在 `(target_px, target_py)` 的命中半徑
    內，用來分辨俯瞰圖上「按在母球命中半徑內 → 拖擺位」還是「按在別處 →
    定角度」，不需要另外的模式切換鈕。
    """
    px = validate_finite_number(px, "px")
    py = validate_finite_number(py, "py")
    target_px = validate_finite_number(target_px, "target_px")
    target_py = validate_finite_number(target_py, "target_py")
    radius_px = validate_finite_number(radius_px, "radius_px")

    return math.hypot(px - target_px, py - target_py) <= radius_px
