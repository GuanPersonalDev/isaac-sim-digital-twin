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
        # 目標。差異在基座計算：WAM7/UR3e 是為了讓「特定關節組合」能沿
        # 方向出力設計的精密解；UR10e 靠 RMPflow 解完整 6-DOF IK，基座
        # 只需要落在舒適可達範圍內即可，per-shot 重新計算但比 WAM7/UR3e
        # 簡單很多（見 ur10e_placement_calculator.py 模組說明）。
        wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset
        )
        if tilt_rad is None:
            raise ValueError("幾何無解（即使垂直抬高也無法閃避庫邊）")

        # base_position 只由 wrist 位置／方向決定，roll_rad（球桿繞自身軸
        # 的冗餘自由度）不影響位置，因此可以在搜尋 roll_rad 之前先算好、
        # 傳給下面的搜尋函式用——不用等 roll_rad 決定之後才算，也不用為了
        # 搜尋而重算好幾次。
        direction_unit = cue_pose_calculator.compute_tilted_direction(action.shot_angle, tilt_rad)
        base_position = ur10e_placement_calculator.compute_base_position(
            tuple(wrist_position), tuple(direction_unit), table_z
        )

        # roll_rad 是球桿繞自身軸的冗餘自由度（不影響桿頭實際指向或
        # 位置），固定用預設 roll_rad=0 算出來的姿態，對某些目前姿態
        # （尤其從 HOME 出發的 flat 案例）可能剛好跟目前姿態接近正反面，
        # 讓 RMPflow 被迫做接近 180 度的姿態翻轉，反應式求解容易卡在局部
        # 穩定點；也可能逼近 UR10e 手腕的運動學奇異點導致收斂不了。用
        # compute_roll_minimizing_reorientation() 搜尋「翻轉角度最小、且
        # 離奇異點夠遠」的 roll_rad（用 ur10e_analytic_ik 的 closed-form
        # 逆向運動學評估每個候選離奇異點多遠，實測數據見
        # docs/CHANGELOG.md）。
        current_orientation = self._articulation_api.get_end_effector_orientation()
        roll_rad = ur10e_placement_calculator.compute_roll_minimizing_reorientation(
            cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset,
            tuple(current_orientation), base_position,
        )
        wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, action.shot_angle, table_z, ball_radius, action.position_offset, roll_rad=roll_rad
        )
        if tilt_rad is None:
            raise ValueError("幾何無解（即使垂直抬高也無法閃避庫邊）")

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
        # 用滑軌機構專用的實測校準版本，不是動量傳遞理論值——理論公式假設
        # 球桿是自由的 0.5kg 物體，但滑軌關節在撞擊瞬間是被 drive 硬撐住的
        # （見 compute_required_tip_speed_for_cue_slide() 說明）。
        required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed_for_cue_slide(
            action.cue_ball_speed
        )
        self._articulation_api.move_cue_slide_stroke(-self._BACKSWING_DISTANCE_M, required_tip_speed)
