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
from core.models.table_ball_set import TableBallSet  # noqa: E402


def _report_scene(scene: InteractiveScene) -> None:
    """印出 A-3 的四項檢查所需的數值。

    RunPod 的 proxy SSH（``<id>@ssh.runpod.io``）不支援 port forwarding，
    在沒有 public IP 的 pod 上連不到 viser 的 web 介面。這裡改用數值驗證——
    而且比肉眼可靠：擺位差幾公釐看不出來，數字看得出來。

    （要用 viser 的話，把 8080 加進 pod 的 Expose HTTP Ports，
    RunPod 會給 https://<pod-id>-8080.proxy.runpod.net，不需要 tunnel。）
    """
    balls = scene["balls"]

    # body_link_pos_w 形狀 (num_envs, 10, 3)，world frame。
    # object_pos_w 是 deprecated 名稱，Isaac Lab 4.0 移除，不要用。
    pos = balls.data.body_link_pos_w.torch

    # A-2 的核心換算：env_origins 是 (num_envs, 3)，要 unsqueeze(1) 補出
    # object 維度才廣播得到 10 顆球。B-1 讀 observation 時照抄這一行。
    rel = pos - scene.env_origins.unsqueeze(1)

    print(f"[view_scene] env_origins =\n{scene.env_origins}")
    print(f"[view_scene] 球心 z（應全為 {TableBallSet.DEFAULT_BALL_RADIUS}）=\n{pos[:, :, 2]}")
    print(f"[view_scene] env 0 的桌台相對座標（對照 BREAK_SHOT_POSITIONS）=\n{rel[0]}")
    # 各環境的相對座標應該完全一致——不一致就代表有球跑到別的環境的桌上，
    # 或 cloner 沒有正確套用 env origin。這是 A-2 唯一的實質驗收。
    print(f"[view_scene] 各 env 相對座標最大差異（應接近 0）= {(rel - rel[0]).abs().max()}")


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
    _report_scene(scene)

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
