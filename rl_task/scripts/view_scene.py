# Copyright (c) 2026 GuanPersonalDev
"""A-3 場景目視確認：只建 InteractiveScene，不經過 Manager。

`BilliardRlSceneCfg` 完成（A-1／A-2）但 MDP terms 還是空的（B-1~B-5），
所以 `ManagerBasedRLEnv` 實例化不了——`gym.make()` 與 `run_train.sh` 都還不能用。
這支腳本繞過 Manager 直接建場景，讓 A-1／A-2 可以在 B 組之前先驗收。

用法（pod 上，在 repo root 執行）::

    /workspace/IsaacLab/isaaclab.sh -p rl_task/scripts/view_scene.py --num_envs 4 --viz viser

`--viz viser` 會起一個 web server 並在啟動訊息印出網址（pod 沒有顯示器，只能走這條）。
SSH 連線時加 ``-L <port>:localhost:<port>`` 把 port 通出來才看得到。

B 組完成後這支腳本仍然有用——場景改動時可以單獨驗證，不必連 MDP 一起跑。
"""

import argparse

from isaaclab.app import AppLauncher

# ⚠️ AppLauncher 必須在任何 isaaclab.* / billiard_rl.* 匯入之前執行完。
#    那些模組會拉進 omni.*，而 omni 只有在 Kit app 啟動後才存在。這也是
#    Isaac Lab 所有腳本都長成「import 分兩段」的原因。順序寫錯會得到
#    看起來莫名其妙的 ModuleNotFoundError: omni。
parser = argparse.ArgumentParser(description="撞球場景目視確認（#121 A-3）")
parser.add_argument("--num_envs", type=int, default=4, help="要生成的子環境數量")
parser.add_argument("--env_spacing", type=float, default=4.0, help="子環境原點間距（公尺）")
# 附加 Isaac Lab 的通用參數：--headless / --viz / --device / --livestream 等
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下才能匯入 Isaac Lab 與本專案的模組。"""

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.billiard_rl_env_cfg import (  # noqa: E402
    BilliardRlSceneCfg,
)


def main() -> None:
    """建場景後持續 step，讓人用眼睛確認擺位。"""
    # 物理步長必須跟 env cfg 一致（1/60）。core/services/rolling_resistance_service.py
    # 把 PHYSICS_DT = 1/60 寫死成模組常數，用別的值滾動阻力會算錯且不會報錯（#121 B-6）。
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))

    # 第一個參數是相機位置，第二個是注視點。拉遠一點才看得到多個子環境；
    # 要細看單一張桌子的菱形排列時用 --num_envs 1 再自行調近。
    sim.set_camera_view([6.0, -6.0, 6.0], [0.0, 0.0, 0.0])

    scene_cfg = BilliardRlSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[view_scene] 場景建立完成")
    # A-2 的換算對照表全靠這個張量，先看一眼實際數值。形狀是 (num_envs, 3)。
    print(f"[view_scene] env_origins =\n{scene.env_origins}")

    # 開球擺位的球本來就是靜止的，不需要施加動作或做 reset——單純讓物理跑著，
    # 反而更能看出「球會不會自己滑動」這類物理設定問題。
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
