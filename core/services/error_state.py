import logging
import threading

logger = logging.getLogger(__name__)

class ErrorState:
    """
    集中記錄下游動作執行時發生的例外，讓呼叫端可以「不重新拋出」，
    避免一張桌子的錯誤讓共用 tick loop 的其他桌子跟著中斷；
    同時保留可見性（完整 log + get_last_exception() 事後查詢）。
    每張桌子（Demo/Training）各自持有獨立的 instance，不與其他桌子共用。

    mark_error()/clear() 可能分別來自不同執行緒觸發
    （例如 physics callback 呼叫 step() vs UI 觸發 reset()），
    以 Lock 保護 _has_error/_last_exception 的讀寫，避免交錯寫入。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._has_error = False
        self._last_exception: Exception | None = None

    def mark_error(self, exception: Exception) -> None:
        """記錄例外但不重新拋出，呼叫端（TableOrchestrator.step()）藉此吸收下游動作的例外。"""
        logger.exception("Exception happened", exc_info=exception)
        with self._lock:
            self._has_error = True
            self._last_exception = exception

    def has_error(self) -> bool:
        with self._lock:
            return self._has_error

    def get_last_exception(self) -> Exception | None:
        with self._lock:
            return self._last_exception

    def clear(self) -> None:
        """
        必須與 ScriptController.reset() 同時發生：
        ScriptController.get_action() 判斷 has_error 優先於 current_state，
        只清一邊會讓狀態機瞬間又跳回 ERROR。見 TableOrchestrator.reset()。
        """
        with self._lock:
            self._has_error = False
            self._last_exception = None