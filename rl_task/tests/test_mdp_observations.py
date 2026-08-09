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

本檔不啟動 Kit app、不建場景，也不 import isaaclab——`encode_ball_positions()`
是純張量函式，餵造出來的資料就能驗。跑一次約 1 秒，不需要 GPU。

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
    pos_w: torch.Tensor, origins: torch.Tensor, max_offset: float
) -> torch.Tensor:
    """逐 env 呼叫 core 的純 Python 版，組成 (N, 21) 的 golden reference。"""
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
        rows.append(encode_rl_observation(observation, table_position, max_offset))
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
