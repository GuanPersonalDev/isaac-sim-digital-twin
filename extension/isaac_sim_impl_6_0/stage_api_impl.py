import os

import omni.kit.undo
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
import isaacsim.core.utils.bounds as bounds_util
import isaacsim.core.utils.prims as prims_util
from isaacsim.storage.native import get_assets_root_path

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
        resolved_path = asset_path

        # 雲端資源的路徑修正
        if not os.path.isabs(asset_path) and not asset_path.startswith("omniverse://"):
            resolved_path = get_assets_root_path() + "/" + asset_path
        prim.GetReferences().AddReference(resolved_path)
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

    def create_fixed_joint(
        self, joint_path: str, body0_path: str, body1_path: str
    ) -> None:
        joint = UsdPhysics.FixedJoint.Define(self.get_stage(), joint_path)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])

    def create_prismatic_joint(
        self,
        joint_path: str,
        body0_path: str,
        body1_path: str,
        axis: str = "Y",
        lower_limit: float | None = None,
        upper_limit: float | None = None,
    ) -> None:
        joint = UsdPhysics.PrismaticJoint.Define(self.get_stage(), joint_path)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])
        joint.CreateAxisAttr().Set(axis)
        if lower_limit is not None:
            joint.CreateLowerLimitAttr().Set(lower_limit)
        if upper_limit is not None:
            joint.CreateUpperLimitAttr().Set(upper_limit)

    def align_prim_to_target(self, prim_path: str, target_path: str) -> None:
        prim = self._get_prim(prim_path)
        target_prim = self._get_prim(target_path)
        
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        target_world = xform_cache.GetLocalToWorldTransform(target_prim)
        prim_parent_world = xform_cache.GetParentToWorldTransform(prim)
        
        prim_local = target_world * prim_parent_world.GetInverse()
        
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(prim_local)


    def filter_collision_pair(self, prim0_path: str, prim1_path: str) -> None:
        prim0 = self._get_prim(prim0_path)

        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(prim0)
        filtered_pairs.CreateFilteredPairsRel().AddTarget(Sdf.Path(prim1_path))

    def remove_prim(self, prim_path: str) -> None:
        # DeletePrimsCommand（而非 stage.RemovePrim）：destructive=True 預設值
        # 確保所有 local layer 的 prim spec 都被清除，同路徑才能立即重建；
        # 社群回報 stage.RemovePrim 有時候刪不乾淨、留下 over spec 殘影。
        # 包在 omni.kit.undo.disabled() 內：這類刪除通常伴隨對應 core 物件
        # 的 destroy()，若使用者事後 Ctrl+Z 復原 USD 場景，物件已經沒了會
        # 造成狀態不一致；用 disabled() 只讓這次刪除不進 undo stack，不影響
        # 使用者在別處操作的歷史（相對於整個清空 undo stack 更精準）。
        with omni.kit.undo.disabled():
            prims_util.delete_prim(prim_path)
