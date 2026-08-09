# Copyright (c) 2026 GuanPersonalDev
"""B-5 測試：開球擺位的重置（#121 B-5）。

===========================================================================
⚠️ 這個目錄**不在 pre-commit 閘門內**，必須手動在 pod 上跑。理由與跑法見
   test_mdp_observations.py 的檔頭。
===========================================================================

    cd /workspace/isaac-sim-digital-twin
    /workspace/IsaacLab/isaaclab.sh -p -m pytest rl_task/tests/ -q

最後一個測試把 B-5 與 B-1 串起來——reset 產生的世界座標餵進 observation 編碼器
應該剛好還原成 BREAK_SHOT_POSITIONS。那同時驗了兩件事：擺位對、以及 A-2 的
「reset 加 env_origins、讀 observation 減 env_origins」這組換算真的互為反向。
"""

import pytest

pytest.importorskip("torch", reason="本機無 torch，本檔只在 pod 上跑")

import torch  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.mdp.events import (  # noqa: E402
    break_shot_positions,
)
from billiard_rl.tasks.manager_based.billiard_rl.mdp.observations import (  # noqa: E402
    encode_ball_positions,
)
from core.models.table_ball_set import TableBallSet  # noqa: E402
from core.services.break_shot_position_provider import (  # noqa: E402
    BREAK_SHOT_POSITIONS,
)

_RADIUS = TableBallSet.DEFAULT_BALL_RADIUS
_ATOL = 1e-6
_BALL_COUNT = 10

_ORIGINS = torch.tensor(
    [[2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, -2.0, 0.0], [-2.0, 2.0, 0.0]],
    dtype=torch.float64,
)


def test_shape_and_ball_order():
    """object 維度順序必須是 ball_0…ball_9，與 collection 的插入順序一致。

    錯位不會報錯，只是每一局都從錯的盤面開始——例如 3 號球被擺到 7 號球的位置。
    """
    positions = break_shot_positions(_ORIGINS, _RADIUS)

    assert positions.shape == (4, _BALL_COUNT, 3)

    relative = positions - _ORIGINS[:, None, :]
    for ball_id, (x, y) in BREAK_SHOT_POSITIONS.items():
        assert abs(float(relative[0, ball_id, 0]) - x) < _ATOL
        assert abs(float(relative[0, ball_id, 1]) - y) < _ATOL


def test_cue_ball_is_index_zero_in_kitchen():
    """母球（ball_0）在 kitchen，不是在球堆裡。"""
    relative = break_shot_positions(_ORIGINS, _RADIUS) - _ORIGINS[:, None, :]

    assert abs(float(relative[0, 0, 0])) < _ATOL
    assert float(relative[0, 0, 1]) == pytest.approx(-0.9525, abs=_ATOL)
    # 其餘 9 顆都在球堆側（y > 0）
    assert bool((relative[0, 1:, 1] > 0).all())


def test_ball_height_is_radius_above_env_origin():
    """球心高度 = 環境原點 z + 球半徑。寫錯會穿模或浮空。"""
    origins = _ORIGINS.clone()
    origins[1, 2] = 0.5  # 故意抬高一個環境，確認 z 有跟著走

    positions = break_shot_positions(origins, _RADIUS)

    assert torch.allclose(
        positions[:, :, 2],
        (origins[:, 2:3] + _RADIUS).expand(-1, _BALL_COUNT),
        atol=_ATOL,
    )


def test_all_envs_get_identical_relative_layout():
    """每個 env 的相對擺位完全相同——固定開球盤面的定義。

    評估場景必須完全固定（training/README.md〈Demo 素材〉），這裡不得有任何
    隨機擾動。
    """
    relative = break_shot_positions(_ORIGINS, _RADIUS) - _ORIGINS[:, None, :]

    assert torch.allclose(relative, relative[0].expand_as(relative), atol=_ATOL)


def test_roundtrip_through_observation_encoder():
    """B-5 → B-1 串接：reset 的世界座標編碼後應還原成 BREAK_SHOT_POSITIONS。

    A-2 的換算表要求 reset 時「加」env_origins、讀 observation 時「減」，
    這一項直接驗兩者互為反向。任一邊漏掉或加錯方向，這裡就對不上。
    """
    positions = break_shot_positions(_ORIGINS, _RADIUS)

    encoded = encode_ball_positions(positions, _ORIGINS, max_offset=1.0)

    assert encoded.shape == (4, 21)
    # 21 維的球序是 1~9 號球在前、母球最後
    expected: list[float] = []
    for ball_id in list(range(1, 10)) + [0]:
        expected.extend(BREAK_SHOT_POSITIONS[ball_id])
    expected.append(1.0)

    expected_tensor = torch.tensor(expected, dtype=torch.float64)
    for env_idx in range(_ORIGINS.shape[0]):
        assert torch.allclose(encoded[env_idx], expected_tensor, atol=_ATOL)


def test_preserves_device_and_dtype():
    """輸出跟著 env_origins 的 device / dtype 走。"""
    origins = _ORIGINS.to(torch.float32)

    positions = break_shot_positions(origins, _RADIUS)

    assert positions.dtype == torch.float32
    assert positions.device == origins.device
