from __future__ import annotations

import numpy as np

from isaacsim.core.prims import RigidPrim

from core.ports.rigid_body_api import RigidBodyAPI

class RigidBodyAPIImpl(RigidBodyAPI):
    
    def __init__(self):
        self._rigid_prims: dict[str, RigidPrim] = {}
        
    def _get_rigid_prim(self, prim_path: str) -> RigidPrim:
        rigid_prim = self._rigid_prims.get(prim_path)
        if rigid_prim is None:
            rigid_prim = RigidPrim(prim_paths_expr=prim_path)
            rigid_prim.initialize()
            self._rigid_prims[prim_path] = rigid_prim
        return rigid_prim
    
    def get_position(self, prim_path: str) -> list[float]:
        """
        回傳世界座標 (x, y, z) (m)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        positions, _ = rigid_prim.get_world_poses()
        return positions[0].tolist()

    def get_linear_velocity(self, prim_path: str) -> list[float]:
        """
        回傳速度 (vx, vy, vz) (m/s)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        return rigid_prim.get_linear_velocities()[0].tolist()

    def get_angular_velocity(self, prim_path: str) -> list[float]:
        """
        回傳角速度 (wx, wy, wz) (rad/s)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        return rigid_prim.get_angular_velocities()[0].tolist()

    def set_velocities(self, prim_path: str, linear_velocity: list[float], angular_velocity: list[float]) -> None:
        """
        設定 Rigidbody 速度
        (vx, vy, vz) (m/s)
        (wx, wy, wz) (rad/s)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        rigid_prim.set_linear_velocities(np.array([linear_velocity]))
        rigid_prim.set_angular_velocities(np.array([angular_velocity]))