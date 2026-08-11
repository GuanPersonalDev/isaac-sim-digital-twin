# Copyright (c) 2026 GuanPersonalDev
"""B-4 測試：球靜止終止判定（#121 B-4）。

===========================================================================
⚠️ 這個目錄**不在 pre-commit 閘門內**，必須手動在 pod 上跑。理由與跑法見
   test_mdp_observations.py 的檔頭。
===========================================================================

    cd /workspace/isaac-sim-digital-twin
    /workspace/IsaacLab/isaaclab.sh -p -m pytest rl_task/tests/ -q

判定本身只有兩行，但**錯了不會報錯**，只會安靜地改變 episode 的長度：

- 漏掉 `struck` → 開球前就終止，episode 長度 1 步
- 門檻寫死而不是取 `BallMotionMonitor.SPEED_THRESHOLD` → 與 Demo 端的
  「球停了沒」用不同的尺
- `.all(dim=1)` 寫成 `.any()` → 只要有一顆球停就終止

三種都會讓訓練跑得動而學不起來，所以逐項釘死。
"""

import pytest

pytest.importorskip("torch", reason="本機無 torch，本檔只在 pod 上跑")

import torch  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.mdp.rewards import (  # noqa: E402
    evaluate_shot,
)
from billiard_rl.tasks.manager_based.billiard_rl.mdp.terminations import (  # noqa: E402
    balls_at_rest_mask,
    break_foul_decided_mask,
)
from core.ports.rigid_body_api import RigidBodyAPI  # noqa: E402
from core.services.ball_motion_monitor import BallMotionMonitor  # noqa: E402
from core.services.break_foul_evaluator import (  # noqa: E402
    FIRST_CONTACT_FOUL_PENALTY,
)
from core.services.break_shot_position_provider import (  # noqa: E402
    BREAK_SHOT_POSITIONS,
)

_THRESHOLD = BallMotionMonitor.SPEED_THRESHOLD
_BALL_COUNT = 10


class _VelocityOnlyRigidBodyAPI(RigidBodyAPI):
    """只回傳線速度的假 RigidBodyAPI，給 BallMotionMonitor 對拍用。"""

    def __init__(self, linear: dict[str, list[float]]):
        self._linear = linear

    def get_position(self, prim_path: str) -> list[float]:
        raise AssertionError("靜止判定不該讀取位置")

    def set_position(self, prim_path: str, x: float, y: float, z: float) -> None:
        raise AssertionError("靜止判定不該寫入位置")

    def get_linear_velocity(self, prim_path: str) -> list[float]:
        return list(self._linear[prim_path])

    def get_angular_velocity(self, prim_path: str) -> list[float]:
        raise AssertionError("BallMotionMonitor 只看線速度")

    def set_velocities(self, prim_path, linear_velocity, angular_velocity) -> None:
        raise AssertionError("靜止判定不該寫入速度")


def _struck(num_envs: int, value: bool = True) -> torch.Tensor:
    return torch.full((num_envs,), value, dtype=torch.bool)


def test_all_at_rest_and_struck_terminates():
    lin = torch.zeros(2, _BALL_COUNT, 3, dtype=torch.float64)

    assert balls_at_rest_mask(lin, _struck(2)).all()


def test_one_moving_ball_blocks_termination():
    """`.all(dim=1)` 的規格：一顆球還在動就不算落定。"""
    lin = torch.zeros(1, _BALL_COUNT, 3, dtype=torch.float64)
    lin[0, 7, 0] = _THRESHOLD * 10.0

    assert not bool(balls_at_rest_mask(lin, _struck(1))[0])


def test_not_struck_blocks_termination():
    """開球前所有球本來就是靜止的，不得判定終止。

    這是整個 B-4 最容易漏、後果最嚴重的一項：漏掉 struck 的話每個 episode
    都是 1 步就結束，reward 是開球擺位本身的分數，policy 完全學不到擊球。
    """
    lin = torch.zeros(3, _BALL_COUNT, 3, dtype=torch.float64)

    assert not balls_at_rest_mask(lin, _struck(3, value=False)).any()


def test_threshold_is_exclusive():
    """速度恰好等於門檻時**不算**靜止（判定式用 `<` 不是 `<=`）。

    與 BallMotionMonitor.is_any_ball_moving() 的 `>=` 互補，兩邊在邊界上
    必須是同一個決定。
    """
    lin = torch.zeros(1, _BALL_COUNT, 3, dtype=torch.float64)
    lin[0, 0, 0] = _THRESHOLD

    assert not bool(balls_at_rest_mask(lin, _struck(1))[0])

    lin[0, 0, 0] = _THRESHOLD * 0.999
    assert bool(balls_at_rest_mask(lin, _struck(1))[0])


def test_uses_3d_speed_not_horizontal_only():
    """垂直速度也要算進去。

    B-6 的衰減公式讓 vz 原封不動傳遞，所以「水平停了但還在彈跳」的球必須
    擋住終止——這正是 B-6 要在整個 env 安靜後停止寫入的理由。
    """
    lin = torch.zeros(1, _BALL_COUNT, 3, dtype=torch.float64)
    lin[0, 3, 2] = _THRESHOLD * 100.0  # 只有 vz

    assert not bool(balls_at_rest_mask(lin, _struck(1))[0])


def test_matches_ball_motion_monitor():
    """與 Demo 端的 `BallMotionMonitor` 對拍：同一組速度必須得到同一個決定。

    兩邊用不同的實作（逐 prim 的 Python 迴圈 vs 張量），但門檻是同一個常數，
    邊界行為也必須一致。
    """
    torch.manual_seed(0)
    cases = [
        torch.zeros(1, _BALL_COUNT, 3, dtype=torch.float64),
        # 每個分量都低於門檻，模長也低於 → 靜止
        torch.full((1, _BALL_COUNT, 3), _THRESHOLD * 0.4, dtype=torch.float64),
        # 每個分量都低於門檻，但模長 (0.7√3 = 1.21×) 超過 → 移動。
        # 這一格抓的是「逐分量比較」而不是「比模長」的實作錯誤。
        torch.full((1, _BALL_COUNT, 3), _THRESHOLD * 0.7, dtype=torch.float64),
        torch.rand(1, _BALL_COUNT, 3, dtype=torch.float64) * _THRESHOLD * 2.0,
        torch.rand(1, _BALL_COUNT, 3, dtype=torch.float64),
    ]

    for lin in cases:
        prim_paths = [f"/ball{b}" for b in range(_BALL_COUNT)]
        api = _VelocityOnlyRigidBodyAPI(
            {f"/ball{b}": [float(v) for v in lin[0, b]] for b in range(_BALL_COUNT)}
        )
        monitor = BallMotionMonitor(api, prim_paths)

        expected = not monitor.is_any_ball_moving()
        actual = bool(balls_at_rest_mask(lin, _struck(1))[0])

        assert actual == expected, f"速度 {lin[0].tolist()} 兩邊判定不一致"


def test_per_env_independence():
    """各 env 的判定互不影響。"""
    lin = torch.zeros(3, _BALL_COUNT, 3, dtype=torch.float64)
    lin[1, 4, 1] = _THRESHOLD * 50.0  # 只有 env 1 有球在動

    result = balls_at_rest_mask(lin, _struck(3))

    assert result.tolist() == [True, False, True]


##
# 開球犯規提前終止（#123 review 第 4 點）
##
#
# 這一段的風險與 balls_at_rest 同類——錯了不報錯，只會安靜改變 episode 長度
# 或讓罰分消失：
#
# - 漏掉 `first_contact != 1` 的條件 → 合法開球也被提前終止，reward 是飛行
#   途中的隨機構型
# - 把 `first_contact == -1` 也算進來 → 「還沒碰到」被當成「永遠不會碰到」，
#   每一局都在第 1 步就終止
# - 漏掉 `struck` → 開球前就終止
#
# 另外，本 term 與 mdp/rewards.py 的結算 gate 是綁在一起的：提前終止但不放行
# reward 結算的話，-1.5 永遠不會被支付，policy 會學到「隨便亂打可以免費跳過
# 這一局」。那條耦合由本檔最後一個測試跨模組釘住。


def _first_contact(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


@pytest.mark.parametrize("ball_id", [2, 3, 4, 5, 6, 7, 8, 9])
def test_first_contact_other_than_one_terminates_immediately(ball_id: int):
    """首次接觸確定且不是 1 號球 → 結果已定，不必等球停。"""
    assert bool(
        break_foul_decided_mask(_first_contact([ball_id]), _struck(1))[0]
    )


def test_first_contact_one_does_not_terminate():
    """合法開球必須等落定——提前終止的話 spread 會取到飛行途中的構型。"""
    assert not bool(break_foul_decided_mask(_first_contact([1]), _struck(1))[0])


def test_no_contact_yet_does_not_terminate():
    """-1 同時代表「還沒碰到」與「整局都不會碰到」，分不出來就只能等落定。

    這一項是本 term 最容易寫錯的地方：把 -1 也當成犯規的話，每一局都會在
    母球還沒滾到球堆之前就終止。
    """
    assert not bool(break_foul_decided_mask(_first_contact([-1]), _struck(1))[0])


def test_not_struck_blocks_early_termination():
    """開球前 first_contact 是 -1，但即使殘留了舊值也不得終止。"""
    assert not break_foul_decided_mask(
        _first_contact([5, 7]), _struck(2, value=False)
    ).any()


def test_break_foul_per_env_independence():
    result = break_foul_decided_mask(_first_contact([1, 5, -1, 9]), _struck(4))

    assert result.tolist() == [False, True, False, True]


def test_early_termination_and_settle_are_mutually_exclusive_conditions():
    """兩個 DoneTerm 各管各的，不該互相取代。

    合法開球（first_contact == 1）只由 balls_at_rest 決定；犯規局即使球全停
    也仍然滿足 break_foul_decided——兩者同時為真是允許的（reward gate 取
    聯集），要防的是「犯規局只靠落定才結算」那種漏放行。
    """
    lin = torch.zeros(2, _BALL_COUNT, 3, dtype=torch.float64)
    lin[0, 2, 0] = _THRESHOLD * 10.0  # env 0 球還在動
    first_contact = _first_contact([5, 1])

    at_rest = balls_at_rest_mask(lin, _struck(2))
    foul_decided = break_foul_decided_mask(first_contact, _struck(2))

    # env 0：球還在動但犯規已定 → 只有 foul_decided 抓得到
    assert not bool(at_rest[0])
    assert bool(foul_decided[0])
    # env 1：合法開球且球全停 → 只有 at_rest 抓得到
    assert bool(at_rest[1])
    assert not bool(foul_decided[1])


@pytest.mark.parametrize("ball_id", [2, 5, 9])
def test_early_terminated_env_still_gets_paid_the_foul_penalty(ball_id: int):
    """跨模組耦合：被提前終止的 env 必須照樣拿到 -1.5。

    `TerminationsCfg.break_foul` 與 `mdp/rewards.py` 的結算 gate 是兩處各自
    寫的判斷，任何一邊改動都可能讓另一邊失效，而失效不會報錯——policy 只會
    安靜地學到「隨便亂打可以免費跳過這一局」。

    同時釘住「球還在動也算得出正確答案」：這裡刻意餵**部分**的進袋／顆星
    資料（模擬球還在飛時的追蹤狀態），evaluate_break_foul 在 first_contact
    != 1 的分支會 short-circuit、完全不看那兩項，所以結果仍然正確。
    """
    mid_flight_xy = [(0.1 * i, -0.2 * i) for i in range(_BALL_COUNT)]
    partial_pocket_index = [-1] * _BALL_COUNT
    partial_rail_contacted = [False] * _BALL_COUNT
    partial_rail_contacted[3] = True  # 追蹤到一半

    components = evaluate_shot(
        mid_flight_xy,
        partial_pocket_index,
        partial_rail_contacted,
        ball_id,
    )

    assert components == {
        "spread": 0.0,
        "cue_scratch": 0.0,
        "foul": FIRST_CONTACT_FOUL_PENALTY,
        "nine_ball": 0.0,
    }


def test_legal_break_still_computes_the_real_spread():
    """短路只能作用在 should_reset 分支，合法開球的 spread 不得被抹成 0。

    `evaluate_shot()` 為了省掉會被丟掉的凸包計算而延後了 spread 的計算
    （#123 review），這條確認延後沒有波及正常路徑。
    """
    settled_xy = [BREAK_SHOT_POSITIONS[ball_id] for ball_id in range(_BALL_COUNT)]
    settled_xy[4] = (0.4, 0.9)  # 讓球群不是原始 rack，spread 才會明顯非零

    components = evaluate_shot(
        settled_xy,
        [-1] * _BALL_COUNT,
        [True, True, True, True, True] + [False] * (_BALL_COUNT - 5),
        1,
    )

    assert components["foul"] == 0.0
    assert components["spread"] != 0.0
