import os

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
BALL_TEMPLATE_PATH = os.path.join(ASSET_DIR, "ball_template.usda")
CUE_STICK_PATH = os.path.join(ASSET_DIR, "ball_stick.usd")
STRIPE_MDL_PATH = os.path.join(ASSET_DIR, "materials", "stripe_ball.mdl")
TABLE_PATH = os.path.join(ASSET_DIR, "billiard_env.usda")
UR5_PATH = "Isaac/Robots/UniversalRobots/ur5/ur5.usd"
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
