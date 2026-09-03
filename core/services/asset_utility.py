import os

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
BALL_TEMPLATE_PATH = os.path.join(ASSET_DIR, "ball_template.usda")
CUE_STICK_PATH = os.path.join(ASSET_DIR, "ball_stick.usda")
STRIPE_MDL_PATH = os.path.join(ASSET_DIR, "materials", "stripe_ball.mdl")
TABLE_PATH = os.path.join(ASSET_DIR, "billiard_env.usda")
# 訓練專用：billiard_env.usda 去掉 SimpleRoom（地板、四面牆、Towel_Room01_* 裝飾件）
# 後的版本，只留桌台與材質。階層與 POCKET_RELATIVE_PATH 完全相同，兩者可互換。
#
# 訓練場景不需要房間——observation 是 21 維純球位（#222），policy 看不到房間，
# 而球不會離開桌面所以連碰撞都用不到。但 RL 環境會複製到 1024 份，房間的
# triangle mesh／convex hull 碰撞體全部會進 PhysX broadphase，且房間的地板會與
# InteractiveScene 自己的 ground plane 重疊。
#
# 另一個效益是載入時間：9 MB 的 ASCII USD 縮到 26 KB。
# Demo 端仍用 TABLE_PATH，維持「桌子在大房間裡」的場景設計（#121 A-3）。
TRAINING_TABLE_PATH = os.path.join(ASSET_DIR, "billiard_table_only.usda")
UR5_PATH = "Isaac/Robots/UniversalRobots/ur5/ur5.usd"
UR3E_PATH = "Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
UR10E_PATH = "Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
BARRETT_WAM_PATH = os.path.join(ASSET_DIR, "barrett_wam", "wam7", "wam7.usda")
STRIPE_IDENTIFIER = "stripe_material"

# billiard_env.usda 內 6 個球袋 Cylinder 相對於 table 參照 prim 的路徑
# （defaultPrim="Root"，參照後 Root 的子層直接掛在 table_prim_path 底下）。
POCKET_RELATIVE_PATH = "billiard_env_unit/BilliardEnv/BilliardTable"
POCKET_NAMES = [
    "Pocket_HeadLeft",
    "Pocket_HeadRight",
    "Pocket_SideLeft",
    "Pocket_SideRight",
    "Pocket_FootLeft",
    "Pocket_FootRight",
]
