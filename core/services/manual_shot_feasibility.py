"""手動擊球參數面板（#115）的可行性閘門：判斷「按下擊球鈕真的打得出去嗎」。

## 為什麼只判斷「幾何無解」這一種原因

`Ur10eSwingStrategy.execute_aim()` 目前有兩種已知會失敗的情境：

1. **基座 teleport 進球桌中央**——`base = cue_ball − 2.15 × (−sinθ, cosθ)`
   在 `|θ|` 過大時解出來的基座落在桌台範圍內。這一種已經在
   `manual_shot_bounds.MANUAL_SHOT_ANGLE = (-160.0, 160.0)` 從源頭排除
   （推導見該檔案級 docstring），本模組**不重複判斷**。
2. **幾何無解**——`cue_pose_calculator.compute_tilted_wrist_pose()` 回傳
   `tilt_rad is None`：即使把球桿垂直抬到最高也閃不過庫邊，跟角度範圍
   無關，是母球位置＋角度＋偏移量的組合造成的純幾何問題。這一種角度
   上限管不到，是本模組存在的唯一理由。

若未來 Milestone B（#232）把角度放寬回 ±180，第 1 種失敗會重新出現，
屆時本模組需要補上基座位置檢查，`ManualShotFeasibility.reason` 會多一個
`"base_inside_table"`。現在先不做——YAGNI，而且提早加會在沒有真實需求
的情況下猜錯介面形狀。

## `reason` 為什麼是機器可讀字串，不是給人看的訊息

`core/` 不做 UI 呈現（見 `docs/architecture-spec.md` 的分層規則）：中文
文案、要不要顯示原因、顯示成什麼措辭，都是面板（`extension/ui/hud_panel.py`）
的職責。這裡只負責把「`compute_tilted_wrist_pose()` 拋回 None」翻譯成一個
穩定、可比對的字串，面板端用 `if feasibility.reason == "geometry_unsolvable"`
之類的方式決定文案，兩層職責不糾纏在一起。

## 不自己重算幾何

本模組完全委派給 `cue_pose_calculator.compute_tilted_wrist_pose()`——那是
「母球位置＋角度＋偏移量能不能打出這一桿」的幾何單一事實來源
（`Ur10eSwingStrategy.execute_aim()`/`execute_strike()` 也是呼叫同一支函式）。
這裡不重新推導庫邊碰撞或抬桿角度，只是把回傳值翻譯成可行性結果物件。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from . import cue_pose_calculator
from .numeric_validation import validate_2d_value, validate_finite_number
from ..models.manual_shot_parameters import ManualShotParameters

# 面板端拿這個字串跟 ManualShotFeasibility.reason 比對，不是給人看的訊息
# （見檔案級 docstring 的「為什麼」）。
REASON_GEOMETRY_UNSOLVABLE = "geometry_unsolvable"


@dataclass(frozen=True)
class ManualShotFeasibility:
    """一次可行性判斷的結果。`reason` 可行時固定是空字串——面板靠
    `reason == ""` 判斷是否顯示紅線/擋鈕，不是靠 `is_feasible`（兩者永遠
    一致，但空字串比 bool 更適合直接串進 log／debug 顯示）。
    """

    is_feasible: bool
    reason: str  # "" | "geometry_unsolvable"


def evaluate_manual_shot(
    cue_ball_xy: tuple[float, float],
    shot_angle_deg: float,
    table_z: float,
    ball_radius: float,
    position_offset: Sequence[float],
) -> ManualShotFeasibility:
    """判斷這一組手動擊球參數幾何上打不打得出去。

    直接呼叫 `cue_pose_calculator.compute_tilted_wrist_pose()`；`tilt_rad`
    是 `None` 代表無解，其餘情況（`tilt_rad == 0` 或正值，代表不需要／需要
    抬桿即可避開庫邊）都算可行。角度是否落在安全區由呼叫端事先透過
    `ManualShotParameters.__post_init__()` 保證（角度那一類失敗不在這裡
    判斷，理由見檔案級 docstring）。
    """
    cue_ball = validate_2d_value(cue_ball_xy, "cue_ball_xy")
    angle = validate_finite_number(shot_angle_deg, "shot_angle_deg")
    z = validate_finite_number(table_z, "table_z")
    radius = validate_finite_number(ball_radius, "ball_radius")
    offset = validate_2d_value(position_offset, "position_offset")

    _, _, tilt_rad, _ = cue_pose_calculator.compute_tilted_wrist_pose(
        cue_ball, angle, z, radius, list(offset)
    )

    if tilt_rad is None:
        return ManualShotFeasibility(is_feasible=False, reason=REASON_GEOMETRY_UNSOLVABLE)
    return ManualShotFeasibility(is_feasible=True, reason="")


def evaluate_manual_shot_parameters(
    parameters: ManualShotParameters, table_z: float, ball_radius: float
) -> ManualShotFeasibility:
    """給面板用的方便版本：面板手上拿到的是整包 `ManualShotParameters`
    （`get_parameters()` 的回傳型別），不想在呼叫端逐欄位拆開再組回
    `evaluate_manual_shot()` 的參數列表。純粹是欄位轉發，不重複驗證——
    `ManualShotParameters` 建構時已經驗證過四項參數本身合法，這裡只是把
    它們攤開餵給底層版本。
    """
    return evaluate_manual_shot(
        parameters.cue_ball_placement,
        parameters.shot_angle,
        table_z,
        ball_radius,
        parameters.position_offset,
    )
