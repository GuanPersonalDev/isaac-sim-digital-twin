from ..ports.rigid_body_api import RigidBodyAPI
import logging

logger = logging.getLogger(__name__)

class BallMotionMonitor:
    SPEED_THRESHOLD = 0.001
    
    def __init__(self, rigid_body_api: RigidBodyAPI, ball_prim_paths: list[str]) -> None:
        self._rigid_body_api = rigid_body_api
        self._ball_prim_paths = ball_prim_paths
        
    def is_any_ball_moving(self) -> bool:
        # 批次讀取取代「逐顆讀、發現有球在動就提前 return」：提前 return 只有在
        # 球真的在滾的時候省得到，但絕大多數 tick（RESET／AIM 期間）球都是靜止
        # 的，那時逐顆版本必定跑滿 10 次同步。批次版本任何情況都只有 1 次。
        # 見 core/ports/rigid_body_api.py 的效能說明。
        try:
            linear_velocities, _ = self._rigid_body_api.get_velocities(self._ball_prim_paths)
        except Exception as e:
            logger.error(f"get {self._ball_prim_paths} velocity fail : {e}")
            raise
        for vx, vy, vz in linear_velocities:
            if vx**2 + vy**2 + vz**2 >= self.SPEED_THRESHOLD ** 2:
                return True
        return False
