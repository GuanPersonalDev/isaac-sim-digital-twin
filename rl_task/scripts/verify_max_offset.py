# Copyright (c) 2026 GuanPersonalDev
"""#122：條件變數 `max_offset` 的端到端驗證（只能在 pod 上跑）。

`rl_task/tests/` 的對拍測試驗得到 `encode_ball_positions()` 吃逐 env 張量的
行為，但驗不到取樣本身——那需要 `ManagerBasedRLEnv` 實例，也就需要 Kit app。
本腳本補上那一段。

用法（pod 上，在 repo root 執行）::

    /workspace/IsaacLab/isaaclab.sh -p rl_task/scripts/verify_max_offset.py --headless

六項檢查，任何一項失敗都會 raise 並印出實際數值：

1. `get_term("strike")` —— ObsTerm 的 `action_term_name` 字串對得上 ActionsCfg
   的欄位名。這是 #122 新引入的耦合，本機沒有任何東西擋得住打錯
2. obs 第 21 維 == ActionTerm 的 buffer —— 「只有一份 buffer」的實質驗收
3. 取樣值落在 `max_offset_range` 內
4. 逐局重新取樣（跨多次 reset，每個 env 的值都要變動）
5. 部分 env reset 只動到那些 env —— 訓練時的實際路徑（`env_ids` 是索引張量
   而非 None），與全體 reset 走的是 `_resample_max_offset` 的不同分支
6. 實際生效的裁切半徑 == buffer —— 透過母球角速度反推，見 `_check_strike_uses_buffer`

檢查 6 是唯一驗得到「B-2 擊球路徑真的用了逐 env 值」的方法。前五項都只證明
obs 端正確；就算 `_apply_strike()` 退回用單一 cfg 值，1~5 也會全過。
"""

import argparse

from isaaclab.app import AppLauncher

# ⚠️ AppLauncher 必須在任何 isaaclab.* / billiard_rl.* 匯入之前執行完
#    （理由見 view_scene.py 的同位置註解）。
parser = argparse.ArgumentParser(description="max_offset 條件變數驗證（#122）")
parser.add_argument("--num_envs", type=int, default=8, help="子環境數量，至少 4")
parser.add_argument("--trials", type=int, default=6, help="重複 reset 的次數")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下才能匯入 Isaac Lab 與本專案的模組。"""

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.billiard_rl_env_cfg import (  # noqa: E402
    TRAINING_MAX_OFFSET_RANGE,
    BilliardRlEnvCfg,
)
from core.models.action_bounds import ACTION_DIM  # noqa: E402

_OFFSET_COLUMN = 20
"""21 維 observation 裡 max_offset 的索引（最後一格）。"""


def _check_term_lookup(env: ManagerBasedRLEnv):
    """檢查 1：ObsTerm 指名的 action term 真的存在。"""
    term = env.action_manager.get_term("strike")
    print(f"[1] get_term('strike') OK：{type(term).__name__}")
    if not hasattr(term, "max_offset"):
        raise AssertionError("ActionTerm 沒有 max_offset property")
    return term


def _check_obs_matches_buffer(env, term, trials: int) -> torch.Tensor:
    """檢查 2/3/4：obs 與 buffer 同源、值在範圍內、逐局重新取樣。

    回傳 `(trials, num_envs)` 的取樣紀錄。
    """
    low, high = TRAINING_MAX_OFFSET_RANGE
    seen = []

    for trial in range(trials):
        obs, _ = env.reset()
        policy_obs = obs["policy"]
        buffer = term.max_offset

        # 檢查 2：policy 看到的條件值必須就是 ActionTerm 手上那一份。
        # 不相等代表兩端又各存了一份——那正是 #122 改動要根除的狀況。
        column = policy_obs[:, _OFFSET_COLUMN]
        if not torch.allclose(column, buffer.to(column.dtype), atol=1e-6):
            raise AssertionError(
                f"trial {trial}：obs 第 21 維與 buffer 不符\n"
                f"  obs    = {column}\n"
                f"  buffer = {buffer}"
            )

        # 檢查 3：取樣值不得越界。validate_max_offset 只在建構時擋 cfg，
        # 取樣公式本身算錯（例如 span 用成 high 而不是 high-low）擋不住。
        if not bool(((buffer >= low - 1e-6) & (buffer <= high + 1e-6)).all()):
            raise AssertionError(f"trial {trial}：取樣值越界 {buffer}")

        seen.append(buffer.clone())
        print(f"[2/3] trial {trial}：obs == buffer，值域 OK，max_offset = {buffer}")

    stacked = torch.stack(seen)  # (trials, num_envs)

    # 檢查 4：這是 #122 的核心。退回定值時 1~3 全部照過，只有這一項會擋下來。
    spread = stacked.max(dim=0).values - stacked.min(dim=0).values
    if not bool((spread > 1e-6).all()):
        raise AssertionError(
            f"有 env 的 max_offset 跨 {trials} 局完全沒變動——條件變數退化成常數\n"
            f"  逐 env 變動幅度 = {spread}"
        )
    print(f"[4] 逐局重新取樣 OK：各 env 跨 {trials} 局的變動幅度 = {spread}")

    return stacked


def _check_partial_reset(env, term):
    """檢查 5：只 reset 部分 env 時，其餘 env 的條件值不得改變。

    這是訓練時的實際路徑——`_reset_idx()` 傳進來的是「這一步終止的那些 env」
    的索引張量，不是 None。`_resample_max_offset` 對 slice 與索引張量走的是
    不同分支，全體 reset 過了不代表這條也過。

    寫錯的典型後果：整批 env 被連坐重抽，還沒結束的 episode 中途換了條件值，
    policy 這一局前半看到的條件與後半不同——不報錯，只是學不起來。
    """
    before = term.max_offset.clone()
    target = torch.tensor([0, 2], device=before.device)

    term.reset(target)
    after = term.max_offset

    untouched = [i for i in range(before.shape[0]) if i not in (0, 2)]
    if not torch.equal(before[untouched], after[untouched]):
        raise AssertionError(
            f"部分 reset 波及到其他 env\n  before = {before}\n  after  = {after}"
        )
    if bool((before[target] == after[target]).all()):
        # 理論上有機率碰巧抽到同值，但兩個 env 同時碰巧的機率可忽略。
        raise AssertionError(f"指定的 env 沒有被重新取樣：{before[target]} → {after[target]}")

    print(f"[5] 部分 reset OK：env 0/2 重抽 {before[target]} → {after[target]}，其餘不變")


def _check_strike_uses_buffer(env, term):
    """檢查 6：擊球路徑實際生效的裁切半徑 == buffer。

    前五項都只驗 observation 端。`_apply_strike()` 若退回用單一 cfg 值，
    前五項會全部照過——policy 看到逐 env 的條件，實際卻全部用同一個半徑裁切。

    驗法：送一個偏移量遠超上限的動作（正規化域 `(1, 1)`，模長 √2 > 任何
    `max_offset`），圓形裁切後的模長會**精確等於**該 env 的 `max_offset`。
    母球角速度正比於物理域偏移量，所以 `|角速度|` 對 `max_offset` 做線性
    回歸應該得到高 R²。前四維全給 0（正規化域中點）確保除了偏移量以外的
    條件完全相同。

    ⚠️ 判準是**線性回歸**，不是比值 `|角速度| / max_offset` 為常數。後者是
    這支腳本第一版的寫法，2026-08-09 在 pod 上實測失敗（相對離散度 1.49），
    但那是判準錯，不是接線錯：

        斜率 64.23   截距 2.98 rad/s   R² 0.9972

    截距來自母球被賦速的那一個 tick 內、桌面摩擦誘導的自旋——母球寫入時
    沒有滾動自旋，滑動摩擦把它往純滾動拖，角加速度 `α = 2.5·μ·g/r`，
    μ≈0.2 時一個 tick 恰好產生 2.86 rad/s，與量到的截距吻合。這一項與
    `max_offset` **無關，是加性的**，比值判準因此在小 `max_offset` 上失效：
    `max_offset=0.0192` 的 env 偏移自旋只有 ~1.2 rad/s，摩擦貢獻比訊號本身
    還大，比值飆到 201。

    真正要抓的錯誤長得完全不同：`_apply_strike()` 若退回用單一 cfg 半徑，
    角速度會與 `max_offset` **無關**——各 env 幾乎同值，R² ≈ 0。線性回歸
    對加性偏移免疫，又抓得住這個，所以是對的判準。

    `decimation` 暫時改成 1：擊球寫在 `apply_actions()` 的最後一步（先衰減、
    後擊球），所以一個 tick 之後讀到的就是剛寫入的速度。維持 60 的話中間 59
    個 tick 的滾動阻力衰減會破壞線性關係。
    """
    balls = env.scene["balls"]
    cue_index = list(balls.body_names).index("ball_0")

    env.reset()
    buffer = term.max_offset.clone()

    action = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)
    # 上下／左右偏移都給滿檔，模長 √2 ≈ 1.414，恆大於 max_offset（≤ 1.0），
    # 所以一定會被裁切，裁切後模長就是 max_offset 本身。
    action[:, 4] = 1.0
    action[:, 5] = 1.0

    env.step(action)

    ang_vel = balls.data.body_com_ang_vel_w.torch[:, cue_index]
    magnitude = torch.linalg.vector_norm(ang_vel, dim=-1)

    print(f"[6] max_offset = {buffer}")
    print(f"[6] 母球角速度 = {magnitude}")

    # 回歸需要 x 有足夠分散度才有意義。取樣是隨機的，極端情況下八個 env
    # 可能全擠在一起——那時候什麼都證明不了，直接要求重跑。
    x = buffer.double()
    y = magnitude.double()
    if float(x.max() - x.min()) < 0.3:
        print("[6] 跳過：這批 max_offset 分散度不足，回歸沒有意義，請重跑")
        return

    mean_x, mean_y = x.mean(), y.mean()
    sxx = ((x - mean_x) ** 2).sum()
    syy = ((y - mean_y) ** 2).sum()
    sxy = ((x - mean_x) * (y - mean_y)).sum()

    # syy ≈ 0 就是「全部 env 用同一個半徑」的徵狀：角速度根本不隨條件變動。
    # 這一支要在算 R² 之前擋，否則除零。
    if float(syy) < 1e-6:
        raise AssertionError(
            f"母球角速度在各 env 之間幾乎不變（{magnitude}）——擊球路徑沒有"
            "使用逐 env 的裁切半徑"
        )

    slope = float(sxy / sxx)
    intercept = float(mean_y - sxy / sxx * mean_x)
    r_squared = float(sxy * sxy / (sxx * syy))

    print(
        f"[6] 線性回歸：斜率 {slope:.3f}，截距 {intercept:.4f} rad/s，"
        f"R² {r_squared:.6f}"
    )

    if slope <= 0.0:
        raise AssertionError(f"角速度未隨 max_offset 上升（斜率 {slope:.3f}）")
    if r_squared < 0.98:
        raise AssertionError(
            f"角速度與 max_offset 的線性關係不成立（R² {r_squared:.6f}）——"
            "擊球路徑可能沒有使用逐 env 的裁切半徑"
        )

    # 截距只報告不設限：它是那一個 tick 的桌面摩擦誘導自旋（docstring 有推導），
    # 大小隨 μ 與 dt 而定，不是正確性指標。明顯偏離 ~3 rad/s 時值得看一眼
    # 物理參數有沒有被動過，但不該擋下這項檢查。
    print("[6] 擊球路徑使用逐 env 裁切半徑 OK")


def main() -> None:
    if args_cli.num_envs < 4:
        raise SystemExit("--num_envs 至少要 4（檢查 5 需要留下未被 reset 的 env）")

    cfg = BilliardRlEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    # 檢查 6 需要「擊球後只走一個 physics tick」，理由見該函式 docstring。
    # 這是驗證腳本的本地覆寫，不影響產品 cfg 的 60。
    cfg.decimation = 1

    env = ManagerBasedRLEnv(cfg)
    print(f"[verify] 環境建立完成：num_envs={env.num_envs}, device={env.device}")
    print(f"[verify] TRAINING_MAX_OFFSET_RANGE = {TRAINING_MAX_OFFSET_RANGE}")

    try:
        term = _check_term_lookup(env)
        _check_obs_matches_buffer(env, term, args_cli.trials)
        _check_partial_reset(env, term)
        _check_strike_uses_buffer(env, term)
        print("\n[verify] 六項檢查全數通過 ✅")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
