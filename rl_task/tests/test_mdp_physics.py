# Copyright (c) 2026 GuanPersonalDev
"""B-6 對拍測試：torch 向量化的滾動阻力 vs core 的 RollingResistanceService。

===========================================================================
⚠️ 這個目錄**不在 pre-commit 閘門內**，必須手動在 pod 上跑。理由與跑法見
   test_mdp_observations.py 的檔頭。
===========================================================================

    cd /workspace/isaac-sim-digital-twin
    /workspace/IsaacLab/isaaclab.sh -p -m pytest rl_task/tests/ -q

四個分支必須全部涵蓋，隨機值幾乎不會落進門檻附近：

  1. 正常滾動      v_h >> 0.02、residual >> 0.1     → 兩邊逐元素相同
  2. 水平夾停      delta_v <= v_h < 0.02            → 兩邊都夾到 0
  3. 自旋夾停      delta_w <= residual < 0.1        → 兩邊都夾到 0
  4. 雜訊跳過      v_h 與 residual 都 < 0.005       → **兩邊刻意不同**

第 4 格是唯一的行為差異，斷言寫成明確的差異而不是 allclose：core 完全不寫入
（把收斂交還 PhysX sleep），torch 版算得出新值但由呼叫端決定不寫。若哪天有人
「修好」這一格讓兩邊一致，B-4 的球靜止終止會跟著壞掉——所以這裡要擋住。
"""

import pytest

pytest.importorskip("torch", reason="本機無 torch，本檔只在 pod 上跑")

import torch  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.mdp.physics import (  # noqa: E402
    decay_velocities,
)
from core.ports.rigid_body_api import RigidBodyAPI  # noqa: E402
from core.services.rolling_resistance_service import (  # noqa: E402
    GRAVITY,
    NEGLIGIBLE_SPEED_THRESHOLD,
    NEGLIGIBLE_SPIN_THRESHOLD,
    PHYSICS_DT,
    ROLLING_FRICTION_COEFF,
    SETTLING_NOISE_CEILING,
    SPIN_DECAY_RATE,
    RollingResistanceService,
)

_RADIUS = 0.028575
_ATOL = 1e-6

_DELTA_V = ROLLING_FRICTION_COEFF * GRAVITY * PHYSICS_DT  # 0.001635 m/s / tick
_DELTA_W = SPIN_DECAY_RATE * PHYSICS_DT  # 0.16667 rad/s / tick


class _RecordingRigidBodyAPI(RigidBodyAPI):
    """只餵速度、記錄寫入的假 RigidBodyAPI。

    位置相關的方法在本測試用不到——被呼叫到就代表 RollingResistanceService
    的行為跟預期不同，所以實作成直接失敗而不是回傳假資料。
    """

    def get_position(self, prim_path: str) -> list[float]:
        raise AssertionError("滾動阻力不該讀取位置")

    def set_position(self, prim_path: str, x: float, y: float, z: float) -> None:
        raise AssertionError("滾動阻力不該寫入位置")

    def __init__(self, linear: dict[str, list[float]], angular: dict[str, list[float]]):
        self._linear = linear
        self._angular = angular
        self.writes: dict[str, tuple[list[float], list[float]]] = {}

    def get_linear_velocity(self, prim_path: str) -> list[float]:
        return list(self._linear[prim_path])

    def get_angular_velocity(self, prim_path: str) -> list[float]:
        return list(self._angular[prim_path])

    def set_velocities(
        self, prim_path: str, linear_velocity: list[float], angular_velocity: list[float]
    ) -> None:
        self.writes[prim_path] = (list(linear_velocity), list(angular_velocity))


def _reference(lin: torch.Tensor, ang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, set[str]]:
    """逐球呼叫 core 的服務，回傳 (new_lin, new_ang, 有被寫入的 prim 集合)。

    沒有被寫入的球，回傳值填原速度——這樣呼叫端可以用同一個張量比對「有寫入」
    的部分，再單獨檢查跳過集合。
    """
    num_envs, num_balls = lin.shape[0], lin.shape[1]
    prim_paths = [f"/env{e}/ball{b}" for e in range(num_envs) for b in range(num_balls)]
    linear = {
        f"/env{e}/ball{b}": [float(v) for v in lin[e, b]]
        for e in range(num_envs)
        for b in range(num_balls)
    }
    angular = {
        f"/env{e}/ball{b}": [float(v) for v in ang[e, b]]
        for e in range(num_envs)
        for b in range(num_balls)
    }

    api = _RecordingRigidBodyAPI(linear, angular)
    RollingResistanceService(api, _RADIUS).apply(prim_paths)

    new_lin = lin.clone()
    new_ang = ang.clone()
    for e in range(num_envs):
        for b in range(num_balls):
            written = api.writes.get(f"/env{e}/ball{b}")
            if written is None:
                continue
            new_lin[e, b] = torch.tensor(written[0], dtype=lin.dtype)
            new_ang[e, b] = torch.tensor(written[1], dtype=ang.dtype)
    return new_lin, new_ang, set(api.writes.keys())


def _rolling(speed: float, spin_z: float) -> tuple[list[float], list[float]]:
    """造一顆「純滾動 + 指定側旋」的球：角速度 = n̂×v/R 再加上 z 軸殘留。"""
    vx, vy = speed, 0.0
    return [vx, vy, 0.0], [-vy / _RADIUS, vx / _RADIUS, spin_z]


def test_normal_rolling_matches_core():
    """分支 1：正常滾動，兩邊逐元素相同。"""
    lin_row, ang_row = _rolling(speed=1.5, spin_z=5.0)
    lin = torch.tensor([[lin_row]], dtype=torch.float64)
    ang = torch.tensor([[ang_row]], dtype=torch.float64)

    new_lin, new_ang, is_noise = decay_velocities(lin, ang, _RADIUS)
    ref_lin, ref_ang, written = _reference(lin, ang)

    assert not bool(is_noise[0, 0])
    assert written == {"/env0/ball0"}
    assert torch.allclose(new_lin, ref_lin, atol=_ATOL)
    assert torch.allclose(new_ang, ref_ang, atol=_ATOL)
    # 確實有在減速，不是原封不動抄回來
    assert new_lin[0, 0, :2].norm() < lin[0, 0, :2].norm()


def test_horizontal_clamp_matches_core():
    """分支 2：水平速度落在 delta_v 與視覺門檻之間，兩邊都夾到 0。"""
    speed = (_DELTA_V + NEGLIGIBLE_SPEED_THRESHOLD) / 2.0
    assert _DELTA_V < speed < NEGLIGIBLE_SPEED_THRESHOLD
    lin_row, ang_row = _rolling(speed=speed, spin_z=5.0)  # 側旋夠大，只測水平分支
    lin = torch.tensor([[lin_row]], dtype=torch.float64)
    ang = torch.tensor([[ang_row]], dtype=torch.float64)

    new_lin, new_ang, is_noise = decay_velocities(lin, ang, _RADIUS)
    ref_lin, ref_ang, written = _reference(lin, ang)

    assert not bool(is_noise[0, 0])  # 側旋 5.0 遠高於雜訊門檻
    assert written == {"/env0/ball0"}
    assert torch.allclose(new_lin[0, 0, :2], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(new_lin, ref_lin, atol=_ATOL)
    assert torch.allclose(new_ang, ref_ang, atol=_ATOL)


def test_spin_clamp_matches_core():
    """分支 3：殘留自旋落在 delta_w 與視覺門檻之間，兩邊都夾到 0。"""
    spin = (_DELTA_W + NEGLIGIBLE_SPIN_THRESHOLD) / 2.0
    assert _DELTA_W < spin < NEGLIGIBLE_SPIN_THRESHOLD
    lin_row, ang_row = _rolling(speed=1.5, spin_z=spin)  # 水平夠快，只測自旋分支
    lin = torch.tensor([[lin_row]], dtype=torch.float64)
    ang = torch.tensor([[ang_row]], dtype=torch.float64)

    new_lin, new_ang, is_noise = decay_velocities(lin, ang, _RADIUS)
    ref_lin, ref_ang, written = _reference(lin, ang)

    assert not bool(is_noise[0, 0])
    assert written == {"/env0/ball0"}
    # 殘留自旋被夾到 0，只剩衰減後的滾動分量（z 分量全部來自殘留，故為 0）
    assert abs(float(new_ang[0, 0, 2])) < _ATOL
    assert torch.allclose(new_lin, ref_lin, atol=_ATOL)
    assert torch.allclose(new_ang, ref_ang, atol=_ATOL)


def test_settling_noise_is_flagged_and_core_skips_write():
    """分支 4：**唯一刻意的行為差異**，斷言差異本身。

    core 對這種球完全不呼叫 set_velocities()，把收斂交還給 PhysX 的 sleep
    機制。torch 版算得出新值，但 is_settling_noise 會是 True，由
    BilliardStrikeAction._apply_rolling_resistance() 決定不寫入該 env。

    ⚠️ 不要把這一格改成 allclose 讓兩邊「一致」。core 的跳過是刻意的——
       持續寫入會讓球永遠 sleep 不了，vz 被反覆重新注入，B-4 的球靜止終止
       就永遠不成立。
    """
    tiny = SETTLING_NOISE_CEILING / 10.0
    lin = torch.tensor([[[tiny, 0.0, 0.0]]], dtype=torch.float64)
    ang = torch.tensor([[[0.0, 0.0, tiny]]], dtype=torch.float64)

    _, _, is_noise = decay_velocities(lin, ang, _RADIUS)
    _, _, written = _reference(lin, ang)

    assert bool(is_noise[0, 0]), "torch 版必須把這顆球標記為雜訊"
    assert written == set(), "core 必須完全不寫入這顆球"


def test_noise_flag_drives_per_env_skip_decision():
    """`is_settling_noise` 的 per-env 聚合語意：全靜止才跳過整個 env。

    這是 _apply_rolling_resistance() 的 `~is_noise.all(dim=1)` 那一行的規格。
    env 0 全部是雜訊 → 應跳過；env 1 有一顆還在滾 → 整個 env 都要寫。
    """
    tiny = SETTLING_NOISE_CEILING / 10.0
    quiet_lin = [tiny, 0.0, 0.0]
    quiet_ang = [0.0, 0.0, tiny]
    rolling_lin, rolling_ang = _rolling(speed=1.5, spin_z=5.0)

    lin = torch.tensor(
        [[quiet_lin, quiet_lin], [quiet_lin, rolling_lin]], dtype=torch.float64
    )
    ang = torch.tensor(
        [[quiet_ang, quiet_ang], [quiet_ang, rolling_ang]], dtype=torch.float64
    )

    _, _, is_noise = decay_velocities(lin, ang, _RADIUS)
    active = ~is_noise.all(dim=1)

    assert not bool(active[0]), "全部靜止的 env 應該完全不寫入"
    assert bool(active[1]), "還有球在滾的 env 必須寫入"


def test_batched_random_matches_core():
    """多 env 多球的隨機批次，確認向量化沒有把維度接錯。

    隨機值幾乎不會落進門檻附近，所以這一項不能取代上面四個分支——它抓的是
    廣播、reshape、索引這類形狀錯誤。
    """
    torch.manual_seed(0)
    lin = torch.rand(3, 10, 3, dtype=torch.float64) * 2.0 - 1.0
    ang = torch.rand(3, 10, 3, dtype=torch.float64) * 40.0 - 20.0

    new_lin, new_ang, is_noise = decay_velocities(lin, ang, _RADIUS)
    ref_lin, ref_ang, _ = _reference(lin, ang)

    # 隨機值的量級遠高於雜訊門檻，應該沒有任何一顆被標記
    assert not bool(is_noise.any())
    assert torch.allclose(new_lin, ref_lin, atol=_ATOL)
    assert torch.allclose(new_ang, ref_ang, atol=_ATOL)


def test_vertical_velocity_is_passed_through():
    """vz 原封不動傳遞，與 core 一致。

    這是 B-4 為什麼需要「整個 env 安靜才停止寫入」的根本原因：持續寫入等於
    持續把舊的 vz 重新注入，而 B-4 檢查的是完整 3D 速度模長。
    """
    lin = torch.tensor([[[1.5, 0.0, 0.37]]], dtype=torch.float64)
    ang = torch.tensor([[[0.0, 1.5 / _RADIUS, 5.0]]], dtype=torch.float64)

    new_lin, _, _ = decay_velocities(lin, ang, _RADIUS)

    assert float(new_lin[0, 0, 2]) == 0.37


def test_zero_velocity_does_not_produce_nan():
    """完全靜止的球不得產生 NaN。

    Python 版靠 `or` 短路避開 0/0；torch.where 沒有短路，兩個分支都會算，
    所以除法的分母必須 clamp。這一項就是在守 physics.py 的 _EPS。
    """
    lin = torch.zeros(2, 10, 3, dtype=torch.float64)
    ang = torch.zeros(2, 10, 3, dtype=torch.float64)

    new_lin, new_ang, is_noise = decay_velocities(lin, ang, _RADIUS)

    assert not bool(torch.isnan(new_lin).any())
    assert not bool(torch.isnan(new_ang).any())
    assert bool(is_noise.all())
