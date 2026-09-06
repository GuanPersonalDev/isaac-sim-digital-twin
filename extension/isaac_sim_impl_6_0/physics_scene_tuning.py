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
    """把場上所有 PhysicsScene 調成適合 Demo 規模（十幾個剛體）的設定。

    這個場景只有 18 個剛體（10 顆球＋球桿＋手臂連桿）跟 1 個 articulation，
    GPU 物理管線的固定開銷（kernel launch、GPU 記憶體同步，以及每次
    tensor 讀取都要 GPU→CPU 搬一次）在這個量級是純虧損：

        scripts/benchmark_gui_frametime.py，單張 Demo 桌、RTX 4090、600 frame
        GPU dynamics 開：PhysX Update 20.25ms → 30.59 FPS
        GPU dynamics 關：PhysX Update  9.33ms → 40.56 FPS

    也就是每 frame 白花約 11ms。broadphase 一併從 GPU 改成 MBP，GPU
    broadphase 在沒有 GPU dynamics 的情況下沒有意義。

    ⚠️ 這個判斷**只對 Demo 規模成立**。RL 訓練環境（`rl_task/billiard_rl/`）
    是 1024 個平行 env、上萬個剛體，那個量級 GPU 物理才會贏，訓練端不該套用
    本函式。
    """
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
        physx_scene.CreateEnableGPUDynamicsAttr().Set(False)
        physx_scene.CreateBroadphaseTypeAttr().Set("MBP")
