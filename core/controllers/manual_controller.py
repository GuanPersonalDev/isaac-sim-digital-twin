import logging

from .billiard_state_machine_controller import BilliardStateMachineController
from ..models.action import Action
from ..models.billiard_state import BilliardStatus
from ..models.manual_shot_parameters import ManualShotParameters
from ..models.observation import Observation

logger = logging.getLogger(__name__)


class ManualController(BilliardStateMachineController):
    """由使用者透過 HUD 面板（#115）手動決定六維擊球參數，按下「擊球」鈕才
    出一桿一次，不像 `ScriptController` 那樣自動循環。

    只覆寫 `_idle_state_action_result()`、`_aiming_state_action_result()`、
    `_on_reset()` 這三個基底類別明確開放的擴充點，其餘 4 個狀態轉換（含
    STRIKING/WAITING/RESET 的條件與 no-op Action 格式）沿用基底類別，是
    `ScriptController` 與 `ModelController` 共用的契約，本類別不例外。

    **不呼叫 `decode_rl_action()`/`normalize_action()`**：參數來自使用者
    拖曳/輸入，已經在 `shot_panel_input_mapper` 的各個裁切函式夾過邊界，
    `ManualShotParameters.__post_init__` 是建構時的最後一道驗證（越界拋
    `ValueError`，不靜默夾住）。角度用的尺是 `manual_shot_bounds.
    MANUAL_SHOT_ANGLE`（±160°，機械臂基座碰撞的安全區），不是
    `action_bounds.SHOT_ANGLE`（±30°，那是 Milestone A 為了訓練信號密度
    收窄的 RL 動作空間，跟手動面板的物理限制無關，見
    `manual_shot_bounds.py` 檔案級 docstring 的「兩把尺」說明）。

    **為什麼沒按鈕就會永遠停在 IDLE**：`BilliardStateMachineController` 的
    5 個狀態轉換裡，只有 IDLE -> AIMING 的條件由子類別決定。
    `ScriptController` 那裡的條件永遠成立（開球後自動循環）；本類別多一個
    「有排隊請求」的前提（`is_shot_pending()`）。一旦條件成立進了 AIMING，
    AIMING -> STRIKING -> WAITING -> RESET -> IDLE 就照基底類別的邏輯自己
    跑完，回到 IDLE 時因為請求已經被消費，不會再自動觸發下一次——不需要、
    也不應該去動基底類別。

    **為什麼 AIM 與 STRIKE 要讀同一份快照 `_pending`**：兩次分派之間（AIM
    發生在 IDLE -> AIMING、STRIKE 發生在 AIMING -> STRIKING）使用者可能已
    經在瞄準動畫播放期間改了面板上的值。若各自去讀 `self._parameters` 這個
    「即時值」，會變成「照舊角度瞄準、照新速度打」；改擺位更糟——AIM 已經
    把母球 teleport 到舊位置，STRIKE 卻對著空氣揮桿。做法與
    `ModelController` 快取 `_cached_raw_action` 是同一個理由：一次決策，
    兩次分派共用同一份輸出。

    **為什麼用序號而不是 bool 旗標**：`request_shot()` 由 UI 執行緒呼叫，
    `_idle_state_action_result()` 由 physics 執行緒呼叫，兩者沒有鎖保護。
    若用 `bool` 旗標，消費端勢必要寫成
    `pending = self._flag; self._flag = False`（read-modify-write），這
    段序列不是原子操作，兩個執行緒交錯時可能漏掉一次請求或誤判成兩次。
    序號法讓每個欄位只有唯一寫者：`_requested_seq` 只被 UI 執行緒寫、
    `_handled_seq` 只被 physics 執行緒寫，兩邊都只讀對方的欄位，不存在
    lost update；`is_shot_pending()` 只是比較兩個整數，讀到任一方「寫到
    一半」的中間狀態也不影響正確性（int 賦值本身是原子的）。連按多次
    只會讓 `_requested_seq` 多加幾次，消費時一次性追平成同一個值，等效於
    合併成一次擊球——這是刻意的設計，不做擊球佇列。

    **為什麼不加 `threading.Lock`**：physics callback 每個 tick 都會經過
    `get_action()`，若在這條熱路徑上取鎖，是拿一筆確定會發生的效能成本去
    換一個原本就不存在的競態（上面序號法已經是無鎖安全的）。
    """

    def __init__(self, parameters: ManualShotParameters | None = None) -> None:
        super().__init__()
        self._parameters = parameters or ManualShotParameters.default()
        self._requested_seq = 0  # 只有 UI 執行緒寫
        self._handled_seq = 0  # 只有 physics 執行緒寫
        self._pending: ManualShotParameters | None = None

    # ------------------------------------------------------------------
    # UI 執行緒入口
    # ------------------------------------------------------------------

    def set_parameters(self, parameters: ManualShotParameters) -> None:
        """UI 執行緒呼叫，整包覆蓋目前的參數。

        `ManualShotParameters` 是 frozen dataclass，這裡是單一物件參照的
        賦值（原子操作），不會讓 physics 執行緒讀到某個欄位換了、其他欄位
        還沒換的中間態，詳見 `manual_shot_parameters.py` 的檔案級 docstring。
        """
        self._parameters = parameters

    def get_parameters(self) -> ManualShotParameters:
        """回傳目前的參數，供面板重繪（例如切回手動模式、或 Timeline
        Stop->Play 之後恢復畫面上的讀數）。
        """
        return self._parameters

    def request_shot(self) -> None:
        """UI 執行緒呼叫，對應「擊球」按鈕按下。"""
        self._requested_seq += 1

    def is_shot_pending(self) -> bool:
        """是否還有尚未被 physics 執行緒消費的擊球請求。"""
        return self._requested_seq != self._handled_seq

    # ------------------------------------------------------------------
    # physics 執行緒：狀態機擴充點
    # ------------------------------------------------------------------

    def _idle_state_action_result(self, observation: Observation) -> Action:
        """球已擺好且靜止、且有排隊請求時才進 AIMING；否則永遠停在 IDLE。

        呼應時序坑：`TableOrchestrator.step()` 先呼叫 `get_action()`（本
        方法內部已改狀態），**再**讀 `get_current_state()` 去分派
        `_execute_aim()`。也就是說 AIMING 消費的其實是這一次 IDLE handler
        回傳的 Action，`cue_ball_placement`/`shot_angle`/`position_offset`
        必須在這裡就填好——`ScriptController._idle_state_action_result()`
        沒填 `cue_ball_placement`（沿用 `_generate_action_result()` 的
        `[0, 0]` 桌台中心）是既有 bug，本類別改用
        `ManualShotParameters.to_action()` 一次填滿四項，不重演那個問題。

        `self._parameters` 只讀這一次並存進 `self._pending`：AIMING ->
        STRIKING 那次分派要重用同一份，理由見類別 docstring。
        """
        if not self.is_shot_pending():
            return self._generate_action_result()
        if not observation.is_init_state or observation.is_ball_moving:
            return self._generate_action_result()

        parameters = self._parameters  # 只讀這一次，鎖定這次擊球的快照
        self._handled_seq = self._requested_seq  # 消費請求，連按合併為一次
        self._pending = parameters
        self._change_state(BilliardStatus.AIMING)
        return parameters.to_action(should_execute_action=True)

    def _aiming_state_action_result(self, observation: Observation) -> Action:
        """瞄準動作完成後進 STRIKING，回傳的 Action 用 IDLE handler 存下的
        快照重新轉換，不讀 `self._parameters`——理由見類別 docstring 的
        「同一份快照」段落。

        `_pending is None` 代表時序被破壞（例如外部直接把狀態撥到
        AIMING、繞過了 IDLE handler），走 `_enter_error_state()` 而不是
        對 `None` 呼叫 `to_action()` 炸出 `AttributeError`。
        """
        if not observation.is_motion_complete:
            return self._generate_action_result()

        if self._pending is None:
            return self._enter_error_state(ValueError("ManualController 時序被破壞"))

        self._change_state(BilliardStatus.STRIKING)
        return self._pending.to_action(should_execute_action=True)

    def _enter_error_state(self, exception: Exception) -> Action:
        """吸收時序例外，不重新拋出。

        跟 `ModelController._enter_error_state()` 同一個理由：
        `TableOrchestrator.step()`/`TableRuntime.tick()` 都沒有包住
        `get_action()`，例外從這裡穿透出去會直接中斷 physics callback，
        一次打死所有桌子的 tick loop。復原走 `TableOrchestrator.reset()`。
        """
        logger.exception("ManualController 時序錯誤", exc_info=exception)
        self._change_state(BilliardStatus.ERROR)
        return self._generate_action_result()

    def _on_reset(self) -> None:
        """丟掉排隊中的請求，但保留參數本身。

        `_handled_seq` 追平 `_requested_seq`：reset 期間（例如 ERROR 後
        使用者按「重設球局」）發生過的請求視為作廢，不應該在下一次進
        IDLE 時被誤判成「還有請求待處理」而立刻自動出桿——使用者這時候
        通常還沒看過重設後的畫面，不該有一桿在背後排隊。

        `_pending` 一併清空：避免下一次不小心以未經 IDLE handler 消費的
        舊快照進了 AIMING（見 `_aiming_state_action_result()` 的防呆）。

        `self._parameters` 不動——面板上已經調好的值（力道、角度等）是
        使用者的工作成果，reset 不代表放棄這些設定，只代表放棄「已經按下
        但還沒打出去」的那一次請求。
        """
        self._handled_seq = self._requested_seq
        self._pending = None
