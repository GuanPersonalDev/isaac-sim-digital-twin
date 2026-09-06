from ..ports.stage_api import StageAPI
from ..ports.articulation_api import ArticulationAPI
from .robot_arm import RobotArm
from .ur10e_robot import UR10eRobot
from ..services.asset_utility import CUE_ACTUATOR_PATH, CUE_STICK_PATH


class TableRobotManager:
    """
    掛載手臂＋球桿的通用流程，實際掛哪一款手臂由呼叫端傳入的
    robot_arm_class 決定（見 RobotArm 抽象介面），本類別不依賴任何
    特定手臂的具體實作——唯一例外是 UR10eRobot：球桿跟末端執行器之間改用
    `create_prismatic_joint()`（線性滑軌，見 UR10e 重新設計計畫決策 2/3），
    取代其餘手臂共用的 `create_fixed_joint()`，因為 UR10e 靠這個滑軌關節
    本身的線速度出力，不是手臂關節角速度。UR10e 另外會在手腕上掛一個
    出力機構的**外觀件**（`assets/cue_actuator.usda`），讓球桿的平移在
    Demo 畫面上看得懂，見建構子內的說明。
    """

    # 球桿沿自身軸向（= end effector 的 local Y，見 ball_stick.usda 的
    # Cylinder axis="Y" 與 cue_pose_calculator.py 的 _CUE_LOCAL_AXIS）前後
    # 滑動，跟推桿方向一致。關節 DOF 的正方向＝球桿往桿尖方向（朝母球）
    # 伸出，負值＝退桿；這個符號由 create_prismatic_joint() 的 body0/body1
    # 順序決定，見下方呼叫端說明。
    _CUE_SLIDE_JOINT_AXIS = "Y"
    # 新建立的 PrismaticJoint 預設沒有 drive（跟其餘手臂關節不同——那些是
    # URDF 轉換來的，本來就帶 drive），沒有這組增益 set_dof_position_
    # targets()/set_dof_velocity_targets() 對這個 DOF 完全沒有作用力，實測
    # 踩過：關節位置幾秒內飄到 1000+ 公尺外（見
    # extension/isaac_sim_impl_6_0/ur10e_cue_slide_controller.py 開發過程
    # 的除錯記錄）。數值刻意選得很寬裕（球桿質量很輕，不會是效能瓶頸）。
    _CUE_SLIDE_JOINT_DRIVE_STIFFNESS = 1.0e5
    _CUE_SLIDE_JOINT_DRIVE_DAMPING = 1.0e4
    _CUE_SLIDE_JOINT_DRIVE_MAX_FORCE = 1.0e6

    _ROBOT_OFFSET_FROM_TABLE_CENTER = (1.5, 0.0, 0.0)

    def __init__(
        self,
        table_center: tuple[float, float, float],
        base_path: str,
        stage_api: StageAPI,
        articulation_api: ArticulationAPI,
        robot_arm_class: type[RobotArm],
    ) -> None:
        world_position = (
            table_center[0] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[0],
            table_center[1] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[1],
            table_center[2] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[2],
        )
        self._robot_base_path = base_path
        self._initial_robot_base_position = world_position
        self._robot_arm_class = robot_arm_class
        self._stage_api = stage_api
        self._robot = robot_arm_class(base_path, stage_api, articulation_api, world_position)
        self._cue_stick_prim_path = base_path + "/CueStick"
        self._cue_actuator_prim_path: str | None = None
        stage_api.create_reference_prim(self._cue_stick_prim_path, CUE_STICK_PATH)
        end_effector_path = robot_arm_class.get_end_effector_prim_path(base_path)

        stage_api.align_prim_to_target(self._cue_stick_prim_path, end_effector_path)
        stage_api.filter_collision_pair(self._cue_stick_prim_path, end_effector_path)

        if robot_arm_class is UR10eRobot:
            joint_path = self._cue_stick_prim_path + "/CueSlideJoint"
            # body0=手臂末端（父）、body1=球桿（子）。UsdPhysics 的 DOF 量的是
            # 「body1 相對 body0 沿 axis 的位移」，順序寫反會讓 DOF 正方向變成
            # 球桿往後退，跟 Ur10eCueSlideController「負值＝退桿」的約定相反。
            stage_api.create_prismatic_joint(
                joint_path, end_effector_path, self._cue_stick_prim_path,
                axis=self._CUE_SLIDE_JOINT_AXIS,
                drive_stiffness=self._CUE_SLIDE_JOINT_DRIVE_STIFFNESS,
                drive_damping=self._CUE_SLIDE_JOINT_DRIVE_DAMPING,
                drive_max_force=self._CUE_SLIDE_JOINT_DRIVE_MAX_FORCE,
            )
            # 專用出力機構的外觀件：球桿沿滑軌平移在物理上完全正確，但畫面上
            # 球桿是憑空從手腕伸出來、中間沒有看得見的機構，Demo 時看不出來
            # 為什麼它可以平移。掛一個致動器缸體讓球桿看起來是從缸體前端伸縮
            # 的活塞桿。
            #
            # 掛在 end effector **底下**（不是 base_path 底下）：靠 USD
            # transform 繼承跟著手腕走，不需要額外的關節。資產本身刻意不帶
            # 任何 physics schema（純外觀），所以不會多出剛體質量、碰撞對或
            # articulation 連桿——`dof_names` 仍然是 7 個，
            # `Ur10eCueSlideController` 的 `dof_names.index("CueSlideJoint")`
            # 不受影響。
            self._cue_actuator_prim_path = end_effector_path + "/CueActuator"
            stage_api.create_reference_prim(self._cue_actuator_prim_path, CUE_ACTUATOR_PATH)
        else:
            joint_path = self._cue_stick_prim_path + "/FixedJointToRobot"
            stage_api.create_fixed_joint(
                joint_path, self._cue_stick_prim_path, end_effector_path
            )

    def get_cue_stick_prim_path(self) -> str:
        return self._cue_stick_prim_path

    def get_cue_actuator_prim_path(self) -> str | None:
        """專用出力機構外觀件的 prim path；只有 UR10e 有，其餘手臂回傳
        None（它們的球桿是用 fixed joint 固定的，沒有滑軌機構可展示）。"""
        return self._cue_actuator_prim_path

    def get_robot_prim_path(self) -> str:
        return self._robot_arm_class.get_prim_path(self._robot_base_path)

    def get_initial_robot_base_position(self) -> tuple[float, float, float]:
        """建構時擺放手臂底座的世界座標（`_ROBOT_OFFSET_FROM_TABLE_CENTER`
        相對球檯中心的固定偏移）。

        刻意叫 initial 而不是 current：UR10e 每次瞄準都會用
        `RobotArm.reposition()` per-shot 重新擺放底座（見
        `core/services/ur10e_placement_calculator.py`），之後這個值就不再是
        目前位置。用途只有一個——第一次 RESET 之前，要先讓 RMPflow 知道
        底座在哪（`ArticulationAPI.set_robot_base_pose()`），那時還沒有任何
        `reposition()` 發生過。
        """
        return self._initial_robot_base_position

    def get_robot(self) -> RobotArm | None:
        return self._robot

    def destroy(self) -> None:
        # 不移除 self._robot_base_path 本身——它跟 BilliardTable 共用同一個
        # base_path（例如 /World/Table_Demo），只有 BilliardTable.destroy()
        # 有資格移除整個 base_path，這裡只移除自己掛載的 Robot/CueStick 子路徑。
        self._stage_api.remove_prim(
            self._robot_arm_class.get_prim_path(self._robot_base_path)
        )
        self._stage_api.remove_prim(self._cue_stick_prim_path)
        self._robot = None
