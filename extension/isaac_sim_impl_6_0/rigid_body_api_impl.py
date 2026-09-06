from __future__ import annotations

import numpy as np

from isaacsim.core.experimental.prims import RigidPrim

from core.ports.rigid_body_api import RigidBodyAPI

class RigidBodyAPIImpl(RigidBodyAPI):
    
    def __init__(self):
        self._rigid_prims: dict[str, RigidPrim] = {}
        self._batch_prims: dict[tuple[str, ...], RigidPrim] = {}

    def _get_rigid_prim(self, prim_path: str) -> RigidPrim:
        rigid_prim = self._rigid_prims.get(prim_path)
        if rigid_prim is None:
            rigid_prim = RigidPrim(paths=prim_path)
            self._rigid_prims[prim_path] = rigid_prim
        return rigid_prim

    def _get_batch_prim(self, prim_paths: list[str]) -> RigidPrim:
        """RigidPrim 可以一次包住多個 prim（paths: str | list[str]），對整批
        prim 只做一次 tensor 讀取。view 依 path 組合快取起來重用，建立 view
        本身有成本，不能每次呼叫都重建。"""
        key = tuple(prim_paths)
        batch_prim = self._batch_prims.get(key)
        if batch_prim is None:
            batch_prim = RigidPrim(paths=list(prim_paths))
            self._batch_prims[key] = batch_prim
        return batch_prim

    def get_position(self, prim_path: str) -> list[float]:
        """
        回傳世界座標 (x, y, z) (m)
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        positions, _ = rigid_prim.get_world_poses()
        return positions[0].list()

    def get_positions(self, prim_paths: list[str]) -> list[list[float]]:
        """
        一次回傳多個 prim 的世界座標，順序與 prim_paths 相同（見
        core/ports/rigid_body_api.py 的效能說明）。
        """
        if not prim_paths:
            return []
        batch_prim = self._get_batch_prim(prim_paths)
        positions, _ = batch_prim.get_world_poses()
        return positions.numpy().tolist()

    def get_velocities(
        self, prim_paths: list[str]
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        一次回傳多個 prim 的 (線速度清單, 角速度清單)，順序與 prim_paths 相同。
        一次 RigidPrim.get_velocities() 同時拿到兩者，不需要讀兩遍。
        """
        if not prim_paths:
            return [], []
        batch_prim = self._get_batch_prim(prim_paths)
        linear, angular = batch_prim.get_velocities()
        return linear.numpy().tolist(), angular.numpy().tolist()

    def set_position(self, prim_path: str, x: float, y: float, z: float) -> None:
        """
        Teleport 設定世界座標 (x, y, z) (m)。用 RigidPrim.set_world_poses()
        （tensor API），不能改用 StageAPI 的 raw xform op——見
        core/ports/rigid_body_api.py 的說明，兩條路徑混用會讓後續
        set_velocities() 靜默失效。
        """
        rigid_prim = self._get_rigid_prim(prim_path)
        rigid_prim.set_world_poses(positions=np.array([[x, y, z]]))

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