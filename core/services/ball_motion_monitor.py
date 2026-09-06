from ..ports.rigid_body_api import RigidBodyAPI
import logging

logger = logging.getLogger(__name__)

class BallMotionMonitor:
    SPEED_THRESHOLD = 0.001
    
    def __init__(self, rigid_body_api: RigidBodyAPI, ball_prim_paths: list[str]) -> None:
        self._rigid_body_api = rigid_body_api
        self._ball_prim_paths = ball_prim_paths
        
    def is_any_ball_moving(self) -> bool:
        # 批次讀取，不逐顆呼叫並提前 return——批次版本讀一次的成本，不會因為
        # 提前找到會動的球而變低，逐顆版本反而在最常見的「全部靜止」情況下
        # 最慢。見 docs/CHANGELOG.md「GUI FPS 調校」一節。
        try:
            linear_velocities, _ = self._rigid_body_api.get_velocities(self._ball_prim_paths)
        except Exception as e:
            logger.error(f"get {self._ball_prim_paths} velocity fail : {e}")
            raise
        for vx, vy, vz in linear_velocities:
            if vx**2 + vy**2 + vz**2 >= self.SPEED_THRESHOLD ** 2:
                return True
        return False
