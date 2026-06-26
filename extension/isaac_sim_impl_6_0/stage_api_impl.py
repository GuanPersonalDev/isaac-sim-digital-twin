import omni.usd
from pxr import Gf, Usd, UsdGeom
import isaacsim.core.utils.bounds as bounds_util

from core.ports.stage_api import StageAPI


class StageAPIImpl(StageAPI):
    def get_stage(self):
        return omni.usd.get_context().get_stage()

    def prim_exists(self, prim_path: str) -> bool:
        return self._get_prim(prim_path).IsValid()

    def _get_prim(self, prim_path):
        return self.get_stage().GetPrimAtPath(prim_path)

    def get_child_prim_paths(self, parent_prim_path: str) -> list[str]:
        prim = self._get_prim(parent_prim_path)
        return [child.GetPath().pathString for child in prim.GetChildren()]

    def create_reference_prim(self, prim_path: str, asset_path: str) -> Usd.Prim:
        stage = self.get_stage()
        prim = stage.DefinePrim(prim_path)
        prim.GetReferences().AddReference(asset_path)
        return prim

    def set_visibility(self, prim_path: str, visible: bool) -> None:
        prim = self._get_prim(prim_path)
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def get_prim_at_path(self, prim_path: str) -> Usd.Prim:
        return self.get_stage().GetPrimAtPath(prim_path)

    def set_prim_translate(self, prim_path: str, x: float, y: float, z: float) -> None:
        prim = self._get_prim(prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

    def get_prim_sides(self, prim_path: str) -> tuple[float, float, float]:
        cache = bounds_util.create_bbox_cache()

        aabb = bounds_util.compute_aabb(cache, prim_path=prim_path)

        x_size = aabb[3] - aabb[0]
        y_size = aabb[4] - aabb[1]
        z_size = aabb[5] - aabb[2]

        return (x_size, y_size, z_size)
