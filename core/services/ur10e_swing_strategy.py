from ..models.action import Action
from ..models.robot_arm import RobotArm
from ..ports.articulation_api import ArticulationAPI
from . import cue_pose_calculator, swing_trajectory_calculator, ur10e_placement_calculator
from .robot_swing_strategy import RobotSwingStrategy


class Ur10eSwingStrategy(RobotSwingStrategy):
    """UR10e 專屬的瞄準/揮桿策略（見 UR10e 重新設計計畫，決策 3-5）。

    跟 Wam7SwingStrategy／Ur3eSwingStrategy 的根本差異：手臂本身只負責
    「瞄準定位」（透過 RMPflow 反應式收斂＋避障，見
    extension/isaac_sim_impl_6_0/ur10e_rmpflow_controller.py），實際
    出力（揮桿）完全由末端的線性滑軌關節（CueSlideJoint）負責，不靠手臂
    關節角速度——這正是這次重新設計要解決的問題：UR3e 的加權多關節驅動
    在正確幾何下有結構性的 manipulability ellipsoid 限制，margin
    （可達裕度）與 align（軸向對齊）無法同時達標；UR10e＋滑軌關節設計
    讓對齊永遠是 1.0，徹底繞開這個問題。

    這個類別只透過 core/ports/articulation_api.py 的抽象方法跟引擎互動
    （set_robot_base_pose()／move_to_pose()／move_cue_slide_stroke()），
    不直接依賴任何 Isaac Sim API——RMPflow／滑軌關節的實際控制邏輯都在
    extension/isaac_sim_impl_6_0/ArticulationAPIImpl 內部分流處理，見該
    類別 _ur10e_mode 相關方法。
    """

    _BACKSWING_DISTANCE_M = swing_trajectory_calculator.DEFAULT_BACKSWING_DISTANCE_M
    _BASE_ORIENTATION = [1.0, 0.0, 0.0, 0.0]

    def __init__(self, robot_arm: RobotArm, articulation_api: ArticulationAPI) -> None:
        self._robot_arm = robot_arm
        self._articulation_api = articulation_api

    def execute_aim(
        self,
        action: Action,
        cue_ball: tuple[float, float],
        table_z: float,
        ball_radius: float,
    ) -> None:
        # 跟 WAM7/UR3e 一樣沿用 cue_pose_calculator.py 既有邏輯算 wrist
        # 目標（decision 4：「完全沿用...不用改」）。差異在基座計算：
        # WAM7 用 compute_base_pose()、UR3e 用 ur3e_placement_calculator，
        # 兩者都是為了讓「特定關節組合」能沿方向出力而設計的精密解；UR10e
        # 靠 RMPflow 解完整 6-DOF IK，基座只需要落在舒適可達範圍內即可
        # （見 ur10e_placement_calculator.py 模組說明——decision 4 原本
        # 假設固定基座位置夠用，實測發現對某些母球位置距離目標遠達 2.6m、
        # 超過 UR10e 1.3m 可達距離，因此改回 per-shot 重新計算，但比
        # WAM7/UR3e 簡單很多）。
        wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset
        )
        if tilt_rad is None:
            raise ValueError("幾何無解（即使垂直抬高也無法閃避庫邊）")

        direction_unit = cue_pose_calculator.compute_tilted_direction(action.shot_angle, tilt_rad)
        base_position = ur10e_placement_calculator.compute_base_position(
            tuple(wrist_position), tuple(direction_unit), table_z
        )

        self._robot_arm.reposition(base_position)
        # reposition() 只搬動 USD prim，RMPflow 的內部運動學模型需要另外
        # 被告知目前的底座世界位姿才能算對世界座標目標（見
        # ArticulationAPI.set_robot_base_pose() docstring）。
        self._articulation_api.set_robot_base_pose(list(base_position), self._BASE_ORIENTATION)
        self._articulation_api.move_to_pose(list(wrist_position), list(wrist_orientation))

    def execute_strike(
        self,
        action: Action,
        cue_ball: tuple[float, float],
        table_z: float,
        ball_radius: float,
    ) -> None:
        # 跟 WAM7/UR3e 的揮桿方法不同：不需要重新算 wrist/方向幾何——
        # 決策 4：滑軌關節軸向＝球桿軸向＝桿尖速度方向，AIM 收斂後手臂
        # 完全靜止，桿尖速度 100% 來自滑軌關節本身的線速度，不需要
        # 雅可比矩陣／槓桿臂換算，也不需要像 WAM7 高架橋案例那樣查表找
        # roll_rad／backswing_distance（那些都是為了閃避手臂本體撞庫邊
        # 才需要的自由度，UR10e 推桿時手臂本身不動，不會有這個問題）。
        required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(action.cue_ball_speed)
        self._articulation_api.move_cue_slide_stroke(-self._BACKSWING_DISTANCE_M, required_tip_speed)
