from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


class StageAPI(ABC):
    """
    USD Stage 的查詢介面
    """

    @abstractmethod
    def get_stage(self):
        """
        取得stage
        """
        ...

    @abstractmethod
    def prim_exists(self, prim_path: str) -> bool:
        """
        指定路徑的 prim 是否存在
        """
        ...

    @abstractmethod
    def get_child_prim_paths(self, parent_prim_path: str) -> list[str]:
        """
        回傳指定路徑的 prim 底下的所有子 prim 的路徑
        """
        ...

    @abstractmethod
    def create_reference_prim(self, prim_path: str, asset_path: str) -> Usd.Prim:
        """
        從 assets 引用 prim 生成進 viewport 中, 回傳建立完成的prim
        """
        ...

    @abstractmethod
    def set_visibility(self, prim_path: str, visible: bool) -> None:
        """
        prim 的可視化設定
        """
        ...

    @abstractmethod
    def get_prim_at_path(self, prim_path: str) -> Usd.Prim:
        """
        透過 prim_path 取得 prim
        """
        ...

    @abstractmethod
    def set_prim_translate(self, prim_path: str, x: float, y: float, z: float) -> None:
        """
        設定指定 prim_path 的 XYZ 位移。
        """
        ...

    @abstractmethod
    def get_prim_sides(self, prim_path: str) -> tuple[float, float, float]:
        """
        取得三邊長
        """
        ...

    @abstractmethod
    def create_fixed_joint(
        self, joint_path: str, body0_path: str, body1_path: str
    ) -> None:
        """
        在 joint_path 建立 Fixed Joint Prim, 將 body0_path 與 body1_path 兩端固定連接
        """
        ...

    @abstractmethod
    def create_prismatic_joint(
        self,
        joint_path: str,
        body0_path: str,
        body1_path: str,
        axis: str = "Y",
        lower_limit: float | None = None,
        upper_limit: float | None = None,
        drive_stiffness: float = 0.0,
        drive_damping: float = 0.0,
        drive_max_force: float = 0.0,
    ) -> None:
        """
        在 joint_path 建立 Prismatic Joint Prim，讓 body0_path 與 body1_path
        兩端只能沿 axis（"X"/"Y"/"Z"，body1 的本地座標系）相對滑動——取代
        `create_fixed_joint()` 給 UR10e＋專用出力機構用（見 UR10e 重新設計
        計畫決策 2/3）：球桿沿自身軸向在這個關節上前後滑動產生揮桿速度，
        而不是靠手臂關節本身的角速度。

        lower_limit／upper_limit（單位：公尺，相對兩端初始重合位置的偏移）
        皆為 None 時不設限（PhysX 預設無限制）。

        ⚠️ drive_stiffness／drive_damping／drive_max_force 預設 0 只建立
        joint 的幾何約束（body0/body1/axis），不會有 PD 驅動——
        `Articulation.set_dof_position_targets()`/`set_dof_velocity_targets()`
        對這個 DOF 完全沒有作用力可循（實測踩過：關節位置在幾秒內飄到
        1000+ 公尺外，`switch_dof_control_mode()` 讀到的「預設增益」本身
        就是 0）。UR10e 的 CueSlideJoint 是全新建立的 joint（不像其餘手臂
        關節是從官方 URDF 轉換來的、本來就帶 drive），呼叫端必須明確傳入
        非零的 drive_stiffness/drive_damping/drive_max_force 才能讓
        position/velocity 控制生效（見 TableRobotManager 的呼叫端）。
        """
        ...

    @abstractmethod
    def align_prim_to_target(self, prim_path: str, target_path: str) -> None:
        """
        將 prim_path 的 World Transform 對齊 target_path
        """
        ...
        
    @abstractmethod
    def filter_collision_pair(self, prim0_path: str, prim1_path: str) -> None:
        """
        停用兩個 rigidbody 間的碰撞
        """
        ...

    @abstractmethod
    def remove_prim(self, prim_path: str) -> None:
        """
        移除指定路徑的 prim 及其所有子節點（含所有 local layer 的 spec）。
        """
        ...
