"""
PhysicsScene 的規模相關設定。

放在實作層（不是 core/）是因為內容全部是 PhysX/USD 專屬 schema；獨立成一個
模組是因為有三個呼叫端必須用**完全相同**的設定：GUI extension
（`billiard_digital_twin.py`）與兩支真實球檯驗收腳本
（`scripts/test_ur10e_table_flat.py`／`test_ur10e_table_bridge.py`）。驗收
腳本如果跑在跟 GUI 不同的物理設定下，「驗收通過」就不能代表 GUI 的行為。
"""

from pxr import PhysxSchema, Usd, UsdPhysics


def configure_physics_scene_for_demo_scale(stage: Usd.Stage) -> None:
    """把場上所有 PhysicsScene 調成適合 Demo 規模（十幾個剛體）的設定：關閉
    GPU dynamics、broadphase 改用 MBP。數據與量測方法見 docs/CHANGELOG.md
    「GUI FPS 調校」一節。

    ⚠️ 只對 Demo 規模成立。RL 訓練環境（`rl_task/billiard_rl/`）是上萬個
    剛體的量級，GPU 物理在那裡才會贏，訓練端不該套用本函式。
    """
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
        physx_scene.CreateEnableGPUDynamicsAttr().Set(False)
        physx_scene.CreateBroadphaseTypeAttr().Set("MBP")
