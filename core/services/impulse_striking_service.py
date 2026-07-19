import math

from ..models.action import Action
from ..ports.rigid_body_api import RigidBodyAPI
from ..ports.stage_api import StageAPI

def compute_cue_ball_velocities(action: Action, ball_radius: float, spin_efficiency: float = 0.8) -> tuple[list[float], list[float]]:
    """
    由 Action 反推出初速度與角速度
    """
    theta = math.radians(action.shot_angle)
    forward = (-math.sin(theta), math.cos(theta), 0.0)
    side = (math.cos(theta), math.sin(theta), 0.0)
    
    speed = action.cue_speed
    a = action.position_offset[0] * ball_radius
    b = action.position_offset[1] * ball_radius
    
    linear_velocity = [speed * forward[0], speed * forward[1], speed * forward[2]]
    
    k = spin_efficiency * 5.0 * speed / (2 * ball_radius**2)
    angular_velocity = [
        -k * a * side[0],
        -k * a * side[1],
        k * b
    ]
    
    return linear_velocity, angular_velocity

class ImpulseStrikingService:
    
    def __init__(self, stage_api: StageAPI, rigid_body_api: RigidBodyAPI, cue_ball_prim: str, ball_radius: float, spin_efficiency: float = 0.8) -> None:
        self._stage_api = stage_api
        self._rigid_body_api = rigid_body_api
        self._cue_ball_prim = cue_ball_prim
        self._ball_radius = ball_radius
        self._spin_efficiency = spin_efficiency
    
    def strike(self, action: Action, table_z: float) -> None:
        x, y = action.cue_ball_placement
        self._stage_api.set_prim_translate(self._cue_ball_prim, x, y, table_z)
        
        linear_velocity, angular_velocity = compute_cue_ball_velocities(action, self._ball_radius, self._spin_efficiency)
        
        self._rigid_body_api.set_velocities(self._cue_ball_prim, linear_velocity, angular_velocity)