# Copyright (c) 2026 GuanPersonalDev
"""B-1 對拍測試：torch 向量化版 vs core 的純 Python 版（#121 B-1）。

===========================================================================
⚠️ 這個目錄**不在 pre-commit 閘門內**，必須手動在 pod 上跑。
===========================================================================

原因有兩個，都不是刻意跳過測試：

1. 本機沒有安裝 torch（Isaac Sim 只裝在 RunPod 的 pod 上）
2. `.git/hooks/pre-commit` 寫死 `pytest core/tests/`，`pytest.ini` 的
   `testpaths` 也只有 `core/tests`——所以裸跑 `pytest` **不會**收集到本檔

改動 `mdp/observations.py` 後在 pod 上跑：

    cd /workspace/isaac-sim-digital-twin
    /workspace/IsaacLab/isaaclab.sh -p -m pytest rl_task/tests/ -q

（`pytest` 沒有的話：`/workspace/IsaacLab/isaaclab.sh -p -m pip install pytest`）

本檔不啟動 Kit app、不建場景，也不 import isaaclab。`encode_ball_positions()`
是純張量函式；`ball_positions()` 是 ObsTerm 入口，用 stub 的 scene /
action_manager 餵已知球位就能驗（#228）。跑一次約 1 秒，不需要 GPU。

---

對拍的意義：`core` 的純 Python 版是 Demo 端實際在用的實作，也是本專案對
「21 維該長什麼樣」的唯一權威定義（#222）。訓練端因為效能不能重用它，
只能重寫；重寫就會漂移。`RL_BALL_ORDER` 常數共用擋得住欄位順序錯位，
擋不住換算語意變了——那要靠逐元素比對。
"""

import pytest

# 本機沒有 torch，collection 階段就整檔跳過（不是失敗）。
# 這行必須在 import torch 之前——skip 一旦觸發，後面的 import 不會執行。
# 不寫成 `torch = pytest.importorskip(...)` 是因為那會讓 torch 變成變數，
# 型別標註裡的 torch.Tensor 會被靜態檢查判為非法。
pytest.importorskip("torch", reason="本機無 torch，本檔只在 pod 上跑")

import torch  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.mdp.observations import (  # noqa: E402
    ball_positions,
    encode_ball_positions,
)
from core.models.observation import Observation  # noqa: E402
from core.services.break_shot_position_provider import (  # noqa: E402
    BreakShotPositionProvider,
)
from core.services.rl_observation_encoder import encode_rl_observation  # noqa: E402

# core 回傳 Python float（float64），Isaac Lab 的張量是 float32。
# 容差取 float32 的有效位數，不要用 allclose 預設的 1e-8——會假性失敗。
_ATOL = 1e-5

_BALL_COUNT = 10


def _reference(
    pos_w: torch.Tensor, origins: torch.Tensor, max_offset: float | list[float]
) -> torch.Tensor:
    """逐 env 呼叫 core 的純 Python 版，組成 (N, 21) 的 golden reference。

    `max_offset` 給 list 時是逐 env 的值——core 的純 Python 版本來就是單筆
    介面，per-env 條件值（#122）在這裡天然表達得出來。
    """
    offsets: list[float] = (
        list(max_offset)
        if isinstance(max_offset, list)
        else [float(max_offset)] * int(pos_w.shape[0])
    )
    rows = []
    for env_idx in range(pos_w.shape[0]):
        ball_positions = [
            [float(pos_w[env_idx, ball, 0]), float(pos_w[env_idx, ball, 1])]
            for ball in range(_BALL_COUNT)
        ]
        observation = Observation(
            ball_positions=ball_positions,
            # encoder 不讀這欄，但 dataclass 必填
            cue_ball_position=ball_positions[0],
            is_init_state=True,
            is_ball_moving=False,
            is_motion_complete=True,
            has_error=False,
        )
        # core 減的是桌台世界 XY；訓練場景裡桌台就落在子環境原點上（A-1 的
        # table 沒有給 init_state），所以這裡代入 env_origins。
        table_position = (float(origins[env_idx, 0]), float(origins[env_idx, 1]))
        rows.append(
            encode_rl_observation(observation, table_position, offsets[env_idx])
        )
    return torch.tensor(rows, dtype=torch.float64)


def test_matches_core_encoder_on_random_positions():
    """隨機球位 + 隨機 env 原點，逐元素對拍。"""
    torch.manual_seed(0)
    pos_w = torch.rand(8, _BALL_COUNT, 3, dtype=torch.float32) * 2.0 - 1.0
    origins = torch.rand(8, 3, dtype=torch.float32) * 10.0 - 5.0

    actual = encode_ball_positions(pos_w, origins, max_offset=0.7)

    assert actual.shape == (8, 21)
    assert torch.allclose(actual.double(), _reference(pos_w, origins, 0.7), atol=_ATOL)


def test_matches_core_encoder_on_break_shot_layout():
    """實際的開球擺位（不是隨機值）也要對得上。

    隨機值的數量級與正負分布跟真實球位不同，單靠上一個測試可能漏掉只在
    特定範圍才出現的問題。
    """
    positions = BreakShotPositionProvider().get_positions()
    pos_w = torch.zeros(1, _BALL_COUNT, 3, dtype=torch.float32)
    for ball_id, (x, y) in sorted(positions.items()):
        pos_w[0, ball_id, 0] = x
        pos_w[0, ball_id, 1] = y
    origins = torch.zeros(1, 3, dtype=torch.float32)

    actual = encode_ball_positions(pos_w, origins, max_offset=1.0)

    assert torch.allclose(actual.double(), _reference(pos_w, origins, 1.0), atol=_ATOL)


def test_cue_ball_is_last_and_max_offset_is_tail():
    """欄位順序的直接斷言：母球在第 19~20 格，max_offset 在第 21 格。

    對拍測試抓得到「只有一邊改錯」的情況，抓不到「兩邊一起改錯」。
    這一項釘死規格本身。
    """
    pos_w = torch.zeros(1, _BALL_COUNT, 3, dtype=torch.float32)
    # 給每顆球一個可辨識的值：ball_i 的 X = i
    for ball in range(_BALL_COUNT):
        pos_w[0, ball, 0] = float(ball)
    origins = torch.zeros(1, 3, dtype=torch.float32)

    encoded = encode_ball_positions(pos_w, origins, max_offset=0.25)

    # 前 9 組（index 0~17）依序是 ball_1 … ball_9
    assert [float(encoded[0, i * 2]) for i in range(9)] == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
    ]
    # 第 10 組（index 18~19）是母球 ball_0
    assert float(encoded[0, 18]) == 0.0
    # 第 21 格（index 20）是 max_offset
    assert float(encoded[0, 20]) == 0.25


def test_env_origins_are_subtracted_per_env():
    """同樣的桌台相對擺位放在不同 env 原點上，編碼結果必須完全相同。

    這是 A-2 換算在 B-1 這一端的實質驗收——漏減 env_origins 不會報錯，
    只會讓 policy 把桌子的擺放位置誤當成球局特徵，訓練跑得動但學不起來。
    """
    torch.manual_seed(1)
    layout = torch.rand(1, _BALL_COUNT, 3, dtype=torch.float32)
    origins = torch.tensor(
        [[0.0, 0.0, 0.0], [4.0, -4.0, 0.0], [-8.0, 12.0, 0.0]], dtype=torch.float32
    )
    # 三個 env 用同一份相對擺位，各自平移到自己的原點
    pos_w = layout.repeat(3, 1, 1) + origins[:, None, :]

    encoded = encode_ball_positions(pos_w, origins, max_offset=1.0)

    assert torch.allclose(encoded[0], encoded[1], atol=_ATOL)
    assert torch.allclose(encoded[0], encoded[2], atol=_ATOL)


def test_preserves_device_and_dtype():
    """輸出的 device / dtype 必須跟著輸入走。

    max_offset 那一格是用 torch.full 造出來的，寫死 dtype 或 device 會在
    --device cpu 或未來改用 float64 時炸掉（cat 要求兩者一致）。
    """
    pos_w = torch.zeros(2, _BALL_COUNT, 3, dtype=torch.float64)
    origins = torch.zeros(2, 3, dtype=torch.float64)

    encoded = encode_ball_positions(pos_w, origins, max_offset=0.5)

    assert encoded.dtype == torch.float64
    assert encoded.device == pos_w.device


##
# per-env 條件變數（#122）
##
# 以下四項驗的是「第 21 維可以逐 env 不同」這件事本身。取樣行為（每 episode
# 重新抽、整局固定）在 `BilliardStrikeAction.reset()`，需要 ManagerBasedRLEnv
# 實例才驗得了，屬 pod 上的 D-2 範圍，不在本檔。


def test_accepts_per_env_max_offset_tensor():
    """逐 env 的 max_offset：第 21 格必須是各 env 自己的值，與 core 對拍。

    #122 之前這裡是全場同一個常數。改成逐局取樣後，若哪天不小心退回
    `max_offset[0]` 之類的寫法，全部 env 會共用第 0 個 env 的條件值——
    policy 看到的條件與實際生效的裁切半徑對不上，而且完全不報錯。
    """
    torch.manual_seed(2)
    pos_w = torch.rand(4, _BALL_COUNT, 3, dtype=torch.float32) * 2.0 - 1.0
    origins = torch.rand(4, 3, dtype=torch.float32) * 10.0 - 5.0
    offsets = [0.0, 0.25, 0.75, 1.0]

    actual = encode_ball_positions(
        pos_w, origins, torch.tensor(offsets, dtype=torch.float32)
    )

    assert actual.shape == (4, 21)
    assert [float(actual[i, 20]) for i in range(4)] == offsets
    assert torch.allclose(actual.double(), _reference(pos_w, origins, offsets), atol=_ATOL)


def test_tensor_path_matches_float_path():
    """同一個值走張量路徑與 float 路徑，結果必須完全相同。

    兩條路徑是刻意並存的（訓練端逐 env、Demo 端與對拍單值），並存就會漂移。
    """
    torch.manual_seed(3)
    pos_w = torch.rand(3, _BALL_COUNT, 3, dtype=torch.float32)
    origins = torch.zeros(3, 3, dtype=torch.float32)

    via_float = encode_ball_positions(pos_w, origins, 0.4)
    via_tensor = encode_ball_positions(
        pos_w, origins, torch.full((3,), 0.4, dtype=torch.float32)
    )

    assert torch.equal(via_float, via_tensor)


def test_max_offset_tensor_is_cast_to_flat_dtype():
    """條件值的 dtype 由球位決定，不是由 buffer 決定。

    ActionTerm 的 buffer 是 float32，球位在 `--device cpu` 或未來改精度時
    可能是 float64。`torch.cat` 要求兩者一致，不轉會直接丟例外。
    """
    pos_w = torch.zeros(2, _BALL_COUNT, 3, dtype=torch.float64)
    origins = torch.zeros(2, 3, dtype=torch.float64)

    encoded = encode_ball_positions(
        pos_w, origins, torch.tensor([0.3, 0.6], dtype=torch.float32)
    )

    assert encoded.dtype == torch.float64
    assert float(encoded[1, 20]) == pytest.approx(0.6)


def test_accepts_column_shaped_max_offset():
    """(N, 1) 與 (N,) 兩種形狀都要吃得下。

    `reshape(-1, 1)` 而不是 `unsqueeze(-1)` 就是為了這個——呼叫端若先做過
    一次 unsqueeze，unsqueeze 版會變成 (N, 1, 1)，cat 報維度錯。
    """
    pos_w = torch.zeros(2, _BALL_COUNT, 3, dtype=torch.float32)
    origins = torch.zeros(2, 3, dtype=torch.float32)
    offsets = torch.tensor([0.2, 0.8], dtype=torch.float32)

    flat = encode_ball_positions(pos_w, origins, offsets)
    column = encode_ball_positions(pos_w, origins, offsets.reshape(-1, 1))

    assert torch.equal(flat, column)


##
# ObsTerm 入口（#228）
##
# 上面的對拍只涵蓋 `encode_ball_positions()`。訓練端實際被 ObsTerm 呼叫的是
# `ball_positions()`，它在共用函式之外還多做兩件取值：
#
#   1. `balls.data.body_link_pos_w.torch`（不是 deprecated 的 `object_pos_w`）
#   2. `env.action_manager.get_term(...).max_offset`（權威 buffer，#122）
#
# 這兩行取錯不會報錯——`object_pos_w` 在 3.0 還在、只是標記 deprecated；
# `max_offset[0]` 廣播給全部 env 形狀也對。對拍測不到入口，所以這裡用 stub
# 把取值接起來。不建 ManagerBasedRLEnv、不 import isaaclab。


class _WarpTensor:
    """Isaac Lab 3.0 的 data property 回傳 warp 包裝，必須再取 `.torch`。"""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.torch = tensor


class _BallData:
    def __init__(self, pos_w: torch.Tensor, decoy_pos_w: torch.Tensor) -> None:
        self.body_link_pos_w = _WarpTensor(pos_w)
        # 故意給不同的值：入口若退回 object_pos_w，對拍會失敗。
        self.object_pos_w = _WarpTensor(decoy_pos_w)


class _Scene(dict):
    def __init__(self, balls: object, env_origins: torch.Tensor) -> None:
        super().__init__(balls=balls)
        self.env_origins = env_origins


class _ActionManager:
    def __init__(self, terms: dict[str, object]) -> None:
        self._terms = terms

    def get_term(self, name: str) -> object:
        return self._terms[name]


class _StubEnv:
    """只實作 `ball_positions()` 會碰到的屬性，形狀與真實 env 對齊。"""

    def __init__(
        self,
        pos_w: torch.Tensor,
        origins: torch.Tensor,
        max_offset: torch.Tensor,
    ) -> None:
        decoy = pos_w + 99.0
        self.scene = _Scene(type("Balls", (), {"data": _BallData(pos_w, decoy)})(), origins)
        self.action_manager = _ActionManager(
            {"strike": type("Term", (), {"max_offset": max_offset})()}
        )


def test_obs_term_entry_matches_encoder_and_core():
    """入口輸出必須同時等於向量化編碼與 core 的純 Python 版。

    這是 #228 訓練端 observation 路徑的最後一節：取值 → 向量化編碼 →
    與 Demo 端同一份 `encode_rl_observation()` 對拍。
    """
    torch.manual_seed(4)
    pos_w = torch.rand(4, _BALL_COUNT, 3, dtype=torch.float32) * 2.0 - 1.0
    origins = torch.rand(4, 3, dtype=torch.float32) * 10.0 - 5.0
    offsets = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float32)
    env = _StubEnv(pos_w, origins, offsets)

    actual = ball_positions(env)

    assert actual.shape == (4, 21)
    assert torch.equal(actual, encode_ball_positions(pos_w, origins, offsets))
    assert torch.allclose(
        actual.double(), _reference(pos_w, origins, offsets.tolist()), atol=_ATOL
    )


def test_obs_term_reads_per_env_max_offset_from_the_action_term():
    """第 21 格必須是 ActionTerm 上那份 buffer，不是函式自己另取的值。

    退回 `max_offset[0]` 或模組常數不會報錯，policy 看到的條件與實際裁切
    半徑就對不上（#122）。
    """
    pos_w = torch.zeros(3, _BALL_COUNT, 3, dtype=torch.float32)
    origins = torch.zeros(3, 3, dtype=torch.float32)
    offsets = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32)

    encoded = ball_positions(_StubEnv(pos_w, origins, offsets))

    assert [float(encoded[i, 20]) for i in range(3)] == [0.1, 0.4, 0.9]


def test_obs_term_reads_body_link_pos_w_not_object_pos_w():
    """必須走 `body_link_pos_w.torch`。`object_pos_w` 在 3.0 還在，退回去不報錯。"""
    torch.manual_seed(5)
    pos_w = torch.rand(2, _BALL_COUNT, 3, dtype=torch.float32)
    origins = torch.zeros(2, 3, dtype=torch.float32)
    offsets = torch.full((2,), 0.5, dtype=torch.float32)
    env = _StubEnv(pos_w, origins, offsets)

    encoded = ball_positions(env)
    via_body = encode_ball_positions(pos_w, origins, offsets)
    via_object = encode_ball_positions(pos_w + 99.0, origins, offsets)

    assert torch.equal(encoded, via_body)
    assert not torch.allclose(encoded, via_object, atol=_ATOL)


def test_obs_term_wrong_action_term_name_raises():
    """名字打錯必須立刻 KeyError，不能靜默降級成常數或全 0。"""
    env = _StubEnv(
        torch.zeros(1, _BALL_COUNT, 3),
        torch.zeros(1, 3),
        torch.zeros(1),
    )

    with pytest.raises(KeyError):
        ball_positions(env, action_term_name="not_strike")


def test_obs_term_wrong_asset_name_raises():
    """asset 名字打錯必須立刻 KeyError，不能靜默讀到別的剛體。"""
    env = _StubEnv(
        torch.zeros(1, _BALL_COUNT, 3),
        torch.zeros(1, 3),
        torch.zeros(1),
    )

    with pytest.raises(KeyError):
        ball_positions(env, asset_name="not_balls")
