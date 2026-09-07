"""手動擊球參數面板（#115）的參數容器。

`ManualShotParameters` 是 UI 執行緒與 physics 執行緒之間交換擊球參數的
唯一資料格式。`frozen=True` + tuple 欄位是刻意的執行緒安全設計，不是
隨手選的型別：

- UI 端的寫入模式是「整包算好再一次賦值」——拖曳/輸入事件在 UI 執行緒
  裡各自更新暫存值，全部合法之後才建構一個新的 `ManualShotParameters`
  蓋掉舊的（見 `ManualController.set_parameters()`）。
- physics 端的讀取模式是「一次讀出整包」——`_idle_state_action_result()`
  只在進入 AIMING 的那一刻讀一次。

兩邊都不會讀到「角度已更新但速度還沒」這種中間態：因為欄位是 frozen，
唯一能看到的狀態只有「換之前的完整一份」或「換之後的完整一份」，不存在
逐欄位修改途中被另一個執行緒讀到一半的問題。這比對 4 個欄位分別上鎖便宜
得多，也不需要鎖。
"""

from dataclasses import dataclass

from .action import Action
from .action_bounds import (
    CUE_BALL_SPEED,
    POSITION_OFFSET_HORIZONTAL,
    POSITION_OFFSET_VERTICAL,
)
from .manual_shot_bounds import (
    CUE_BALL_PLACEMENT_X,
    CUE_BALL_PLACEMENT_Y,
    MANUAL_SHOT_ANGLE,
)
from ..services.break_shot_position_provider import BREAK_SHOT_POSITIONS
from ..services.numeric_validation import validate_2d_value, validate_finite_number


def _validate_range(value: float, bounds: tuple[float, float], field_name: str) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{field_name}={value} 超出合法範圍 [{low}, {high}]")


@dataclass(frozen=True)
class ManualShotParameters:
    """手動擊球的四項參數：擺位、角度、初速、偏移。

    邊界值全部引用既有常數（擺位/速度/偏移三項是物理能力邊界，角度是
    `manual_shot_bounds.MANUAL_SHOT_ANGLE` 定義的安全區，見該檔案的
    「兩把尺」說明），本類別不重新定義任何數值。

    `__post_init__` 建構即驗證：越界直接拋 `ValueError`，不靜默夾住——夾
    住等於偷偷把使用者拖出去的值改成別的值，UI 顯示的讀數會跟實際套用的
    參數對不上。呼叫端（`shot_panel_input_mapper` 的裁切函式）有責任在
    建構前先把值夾進合法域，這裡只負責把「還是越界」這件事情大聲擋下來。
    """

    cue_ball_placement: tuple[float, float]
    shot_angle: float
    cue_ball_speed: float
    position_offset: tuple[float, float]

    def __post_init__(self) -> None:
        placement_x, placement_y = validate_2d_value(
            self.cue_ball_placement, "cue_ball_placement"
        )
        _validate_range(placement_x, CUE_BALL_PLACEMENT_X, "cue_ball_placement[0]")
        _validate_range(placement_y, CUE_BALL_PLACEMENT_Y, "cue_ball_placement[1]")

        shot_angle = validate_finite_number(self.shot_angle, "shot_angle")
        _validate_range(shot_angle, MANUAL_SHOT_ANGLE, "shot_angle")

        speed = validate_finite_number(self.cue_ball_speed, "cue_ball_speed")
        _validate_range(speed, CUE_BALL_SPEED, "cue_ball_speed")

        offset_v, offset_h = validate_2d_value(self.position_offset, "position_offset")
        _validate_range(offset_v, POSITION_OFFSET_VERTICAL, "position_offset[0]")
        _validate_range(offset_h, POSITION_OFFSET_HORIZONTAL, "position_offset[1]")

    @staticmethod
    def default() -> "ManualShotParameters":
        """切到手動模式、什麼都不調就按擊球，要打出跟現行 `ScriptController`
        同一球——開球擺位、最大初速、零角度零偏移，全部引用單一來源常數，
        不得寫死數值。
        """
        return ManualShotParameters(
            cue_ball_placement=BREAK_SHOT_POSITIONS[0],
            shot_angle=0.0,
            cue_ball_speed=CUE_BALL_SPEED[1],
            position_offset=(0.0, 0.0),
        )

    def to_action(self, should_execute_action: bool) -> Action:
        """轉成執行期 `Action`，**每次呼叫都回傳全新物件**。

        `Action` 是 mutable dataclass，下游會直接改它（見
        `ModelController._aiming_state_action_result()` 的說明：AIM 與
        STRIKE 兩次分派各自需要一份不受另一次影響的 Action）。若在這裡
        快取單一 instance 重複回傳，AIM 用過的 Action 被下游改動後，
        STRIKE 讀到的就不再是原本的參數，兩次分派會互相污染。
        """
        return Action(
            cue_ball_placement=list(self.cue_ball_placement),
            shot_angle=self.shot_angle,
            cue_ball_speed=self.cue_ball_speed,
            position_offset=list(self.position_offset),
            should_execute_action=should_execute_action,
        )
