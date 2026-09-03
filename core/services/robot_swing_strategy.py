from abc import ABC, abstractmethod

from ..models.action import Action
from ..models.robot_arm import RobotArm
from ..ports.articulation_api import ArticulationAPI


class RobotSwingStrategy(ABC):
    """
    手臂瞄準/揮桿的策略介面。

    取代原本 DemoTableOrchestrator._execute_aim()/_execute_strike() 裡用
    isinstance(self._robot_arm, UR3eRobot) 手動判斷分流的做法——策略物件在
    建構時就依 robot_arm 型別決定好（見 create_swing_strategy_for()），
    DemoTableOrchestrator 執行時只透過這個抽象介面呼叫，不需要知道實際是
    哪一款手臂。

    每個具體策略（Wam7SwingStrategy／Ur3eSwingStrategy／未來的
    Ur10eSwingStrategy）各自封裝一套完全不同的瞄準/揮桿演算法，彼此互不
    相依。
    """

    @abstractmethod
    def execute_aim(
        self,
        action: Action,
        cue_ball: tuple[float, float],
        table_z: float,
        ball_radius: float,
    ) -> None:
        """
        AIMING 狀態下游動作：把手臂移動到瞄準姿態。

        對應 DemoTableOrchestrator._execute_aim() 目前 isinstance 判斷之後
        的那一段（WAM7 走 compute_base_pose()+cue_pose_calculator 那條路徑，
        UR3e 走 _execute_aim_ur3e() 那條路徑），呼叫端已經先把
        table_z/ball_radius/cue_ball 算好、母球也已經瞬移到位，這裡只需要
        專心處理手臂本身的移動。
        """
        ...

    @abstractmethod
    def execute_strike(
        self,
        action: Action,
        cue_ball: tuple[float, float],
        table_z: float,
        ball_radius: float,
    ) -> None:
        """
        STRIKING 狀態下游動作：執行揮桿。

        對應 DemoTableOrchestrator._execute_strike() 目前 isinstance 判斷
        之後的那一段（WAM7 走 move_swing() 那條路徑，UR3e 走
        _execute_strike_ur3e() 那條路徑）。呼叫端已經先做過
        did_last_motion_timeout() 這類跟手臂型號無關的通用檢查，這裡只需要
        專心處理揮桿本身。
        """
        ...


def create_swing_strategy_for(
    robot_arm: RobotArm, articulation_api: ArticulationAPI
) -> RobotSwingStrategy:
    """
    依 robot_arm 的實際型別挑選對應的 RobotSwingStrategy 實作。

    這是整個專案裡唯一一處對手臂型別做判斷的地方——取代原本散落在
    DemoTableOrchestrator._execute_aim()/_execute_strike() 兩處、且每個
    tick 都要重新判斷一次的 isinstance 分支，改成物件建構時判斷一次就好
    （見 billiard_digital_twin.py::_build_demo_session()）。

    刻意放在 core/services/（不是 RobotArm 的方法）：core/models/ 不該
    反過來依賴 core/services/，見對話紀錄的分層取捨討論。未來新增
    Ur10eSwingStrategy 時，在這裡多加一個 elif 分支即可。
    """
    # 延遲 import：避免這個模組被載入時，UR3eRobot／Ur3eSwingStrategy 這些
    # 目前已停用但保留的程式碼一定要跟著載入（見 Ur3eSwingStrategy
    # docstring 的「暫緩維護」說明），也避免跟 core/models/ur3e_robot.py
    # 之間形成不必要的模組層級耦合。
    from ..models.ur3e_robot import UR3eRobot
    from .ur3e_swing_strategy import Ur3eSwingStrategy
    from .wam7_swing_strategy import Wam7SwingStrategy

    if isinstance(robot_arm, UR3eRobot):
        return Ur3eSwingStrategy(robot_arm, articulation_api)
    return Wam7SwingStrategy(robot_arm, articulation_api)
