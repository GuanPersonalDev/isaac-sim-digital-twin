from ..ports.rigid_body_api import RigidBodyAPI
import logging

logger = logging.getLogger(__name__)

class BallMotionMonitor:
    SPEED_THRESHOLD = 0.001
    
    def __init__(self, rigid_body_api: RigidBodyAPI, ball_prim_paths: list[str]) -> None:
        self._rigid_body_api = rigid_body_api
        self._ball_prim_paths = ball_prim_paths
        
    def is_any_ball_moving(self) -> bool:
        for prim_path in self._ball_prim_paths:
            try:
                vx, vy, vz = self._rigid_body_api.get_linear_velocity(prim_path)
                if vx**2 + vy**2 + vz**2 >= self.SPEED_THRESHOLD ** 2:
                    return True
            except Exception as e:
                logger.error(f"get {prim_path} velocity fail : {e}")
                raise
        return False
