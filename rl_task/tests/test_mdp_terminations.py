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

from billiard_rl.tasks.manager_based.billiard_rl.mdp.terminations import (  # noqa: E402
    balls_at_rest_mask,
)
from core.ports.rigid_body_api import RigidBodyAPI  # noqa: E402
from core.services.ball_motion_monitor import BallMotionMonitor  # noqa: E402

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
