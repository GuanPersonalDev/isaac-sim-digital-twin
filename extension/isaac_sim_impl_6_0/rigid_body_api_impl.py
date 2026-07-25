from __future__ import annotations

import numpy as np

from isaacsim.core.experimental.prims import RigidPrim

from core.ports.rigid_body_api import RigidBodyAPI

class RigidBodyAPIImpl(RigidBodyAPI):
    
    def __init__(self):
        self._rigid_prims: dict[str, RigidPrim] = {}
        
    def _get_rigid_prim(self, prim_path: str) -> RigidPrim:
        rigid_prim = self._rigid_prims.get(prim_path)
        if rigid_prim is None:
            rigid_prim = RigidPrim(paths=prim_path)
            self._rigid_prims[prim_path] = rigid_prim
        return rigid_prim
    
    def get_position(self, prim_path: str) -> list[float]:
        """
        回傳世界座標 (x, y, z) (m)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        positions, _ = rigid_prim.get_world_poses()
        return positions[0].list()

    def get_linear_velocity(self, prim_path: str) -> list[float]:
        """
        回傳速度 (vx, vy, vz) (m/s)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        linear, _ = rigid_prim.get_velocities()
        return linear[0].list()

    def get_angular_velocity(self, prim_path: str) -> list[float]:
        """
        回傳角速度 (wx, wy, wz) (rad/s)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        _ , angular = rigid_prim.get_velocities()
        return angular[0].list()

    def set_velocities(self, prim_path: str, linear_velocity: list[float], angular_velocity: list[float]) -> None:
        """
        設定 Rigidbody 速度
        (vx, vy, vz) (m/s)
        (wx, wy, wz) (rad/s)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        rigid_prim.set_velocities(linear_velocity, angular_velocity)