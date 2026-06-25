import omni.usd
from pxr import Gf, Sdf, UsdShade

from core.ports.material_api import MaterialAPI


class MaterialAPIImpl(MaterialAPI):
    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def apply_preview_surface(
        self, prim_path: str, r: float, g: float, b: float
    ) -> None:
        stage = self._get_stage()
        mat_path = prim_path + "/Mat"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(r, g, b)
        )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        prim = stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def apply_mdl_shader(
        self,
        prim_path: str,
        mdl_path: str,
        identifier: str,
        stripe_color: tuple[float, float, float],
        base_color: tuple[float, float, float],
    ) -> None:
        stage = self._get_stage()
        mat_path = prim_path + "/Mat"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath(mdl_path), "mdl")
        shader.SetSourceAssetSubIdentifier(identifier, "mdl")
        shader.CreateInput("stripe_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*stripe_color)
        )
        shader.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*base_color)
        )
        material.CreateSurfaceOutput("mdl").ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        material.CreateDisplacementOutput("mdl").ConnectToSource(
            shader.ConnectableAPI(), "displacement"
        )
        material.CreateVolumeOutput("mdl").ConnectToSource(
            shader.ConnectableAPI(), "volume"
        )
        prim = stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI(prim).Bind(material)
