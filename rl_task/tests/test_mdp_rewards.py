# Copyright (c) 2026 GuanPersonalDev
"""B-3 測試：擊球事件偵測與 reward 分項（#121 B-3）。

===========================================================================
⚠️ 這個目錄**不在 pre-commit 閘門內**，必須手動在 pod 上跑。理由與跑法見
   test_mdp_observations.py 的檔頭。
===========================================================================

    cd /workspace/isaac-sim-digital-twin
    /workspace/IsaacLab/isaaclab.sh -p -m pytest rl_task/tests/ -q

最重要的一項是 `test_decomposition_sums_to_core_reward`：四個 RewTerm 是為了
TensorBoard 分項而拆的，但數值權威在 `core.services.reward_service`。拆解一旦
與 `calculate_reward()` 對不上，TensorBoard 上的分項會與實際回報不一致——
policy 照那個實際回報學，人卻照分項看，查起來會非常痛。
"""

import itertools

import pytest

pytest.importorskip("torch", reason="本機無 torch，本檔只在 pod 上跑")

import torch  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.mdp.rewards import (  # noqa: E402
    decompose_reward,
    evaluate_shot,
)
from billiard_rl.tasks.manager_based.billiard_rl.mdp.shot_tracking import (  # noqa: E402
    detect_cue_contact,
    detect_pocketed,
    detect_rail_contact,
    update_first_contact,
)
from core.models.shot_result import ShotResult  # noqa: E402
from core.models.table_ball_set import TableBallSet  # noqa: E402
from core.services.break_foul_evaluator import evaluate_break_foul  # noqa: E402
from core.services.break_shot_position_provider import (  # noqa: E402
    BREAK_SHOT_POSITIONS,
)
from core.services.pocket_geometry import (  # noqa: E402
    POCKET_POSITIONS,
    POCKET_RADIUS,
    rail_limits,
)
from core.services.reward_service import calculate_reward  # noqa: E402

_RADIUS = TableBallSet.DEFAULT_BALL_RADIUS
_POCKET_XY = list(POCKET_POSITIONS.values())
_BALL_COUNT = 10


##
# B-3a：事件偵測
##


def test_detect_pocketed_flags_ball_inside_pocket_radius():
    ball_xy = torch.zeros(1, _BALL_COUNT, 2, dtype=torch.float64)
    # ball_3 放在 Pocket_SideRight 正中央，ball_4 放在剛好超出半徑的地方
    ball_xy[0, 3] = torch.tensor(_POCKET_XY[3], dtype=torch.float64)
    ball_xy[0, 4] = torch.tensor(_POCKET_XY[3], dtype=torch.float64)
    ball_xy[0, 4, 1] += POCKET_RADIUS * 1.01

    is_pocketed, nearest = detect_pocketed(ball_xy)

    assert bool(is_pocketed[0, 3])
    assert not bool(is_pocketed[0, 4])
    assert int(nearest[0, 3]) == 3


def test_detect_pocketed_picks_the_actual_pocket():
    """袋口索引要對——B-3b 用它替進袋球代入座標，錯了散開分數就錯。"""
    ball_xy = torch.zeros(1, _BALL_COUNT, 2, dtype=torch.float64)
    for pocket_id in range(len(_POCKET_XY)):
        ball_xy[0, 0] = torch.tensor(_POCKET_XY[pocket_id], dtype=torch.float64)
        is_pocketed, nearest = detect_pocketed(ball_xy)
        assert bool(is_pocketed[0, 0])
        assert int(nearest[0, 0]) == pocket_id


def test_break_shot_layout_has_no_ball_in_a_pocket():
    """開球擺位不得有任何球被誤判進袋，否則第一個 tick 就全錯。"""
    ball_xy = torch.tensor(
        [[BREAK_SHOT_POSITIONS[b] for b in sorted(BREAK_SHOT_POSITIONS)]],
        dtype=torch.float64,
    )

    is_pocketed, _ = detect_pocketed(ball_xy)

    assert not bool(is_pocketed.any())


def test_detect_rail_contact_uses_ball_surface_not_centre():
    x_limit, y_limit = rail_limits(_RADIUS)
    ball_xy = torch.zeros(1, _BALL_COUNT, 2, dtype=torch.float64)
    ball_xy[0, 1, 0] = x_limit
    ball_xy[0, 2, 1] = -y_limit
    ball_xy[0, 3, 0] = x_limit * 0.99

    contacted = detect_rail_contact(ball_xy, _RADIUS)

    assert bool(contacted[0, 1])
    assert bool(contacted[0, 2])
    assert not bool(contacted[0, 3])


def test_break_shot_layout_has_no_rail_contact():
    ball_xy = torch.tensor(
        [[BREAK_SHOT_POSITIONS[b] for b in sorted(BREAK_SHOT_POSITIONS)]],
        dtype=torch.float64,
    )

    assert not bool(detect_rail_contact(ball_xy, _RADIUS).any())


def _layout(near_cue: dict[int, float] | None = None) -> torch.Tensor:
    """以開球擺位為底，把指定的球移到母球正前方指定距離處。

    ⚠️ 不要用 `torch.zeros((1, 10, 2))` 當底——那會讓 9 顆號碼球全部疊在母球
    身上（球心距離 0 < 2r），每一顆都被判定為接觸中。距離判定的測試必須從
    「球彼此分開」的局面出發。
    """
    xy = torch.tensor(
        [[BREAK_SHOT_POSITIONS[b] for b in sorted(BREAK_SHOT_POSITIONS)]],
        dtype=torch.float64,
    )
    cue = xy[0, 0].clone()
    for ball_id, distance in (near_cue or {}).items():
        xy[0, ball_id] = cue + torch.tensor([distance, 0.0], dtype=torch.float64)
    return xy


def test_detect_cue_contact_excludes_cue_ball_itself():
    ball_xy = _layout({5: 2.0 * _RADIUS})  # 剛好接觸

    touching = detect_cue_contact(ball_xy, _RADIUS)

    assert not bool(touching[0, 0]), "母球不能算成自己碰到自己"
    assert bool(touching[0, 5])
    assert int(touching.sum()) == 1, "球堆裡的球離母球 1.5 m 以上，不該被算進來"


def test_update_first_contact_is_sticky_and_picks_nearest():
    ball_xy = _layout({7: 2.0 * _RADIUS, 2: 1.5 * _RADIUS})  # 2 號球更近

    first = torch.full((1,), -1, dtype=torch.long)
    first = update_first_contact(first, ball_xy, _RADIUS)
    assert int(first[0]) == 2

    # 之後 1 號球也碰到了，而且更近——記錄不得被覆寫
    ball_xy = _layout({7: 2.0 * _RADIUS, 2: 1.5 * _RADIUS, 1: 0.5 * _RADIUS})
    first = update_first_contact(first, ball_xy, _RADIUS)
    assert int(first[0]) == 2


def test_update_first_contact_stays_unset_without_contact():
    ball_xy = torch.zeros(1, _BALL_COUNT, 2, dtype=torch.float64)
    for ball in range(1, _BALL_COUNT):
        ball_xy[0, ball, 0] = 0.5 * ball  # 全部離母球很遠

    first = update_first_contact(torch.full((1,), -1, dtype=torch.long), ball_xy, _RADIUS)

    assert int(first[0]) == -1


##
# B-3b／B-3c：reward 分項
##


def test_decomposition_sums_to_core_reward():
    """**最重要的一項**：四項之和必須恆等於 `calculate_reward()`。

    窮舉 reward 鏈的所有分支組合：母球進袋 × 9 號球進袋 × 三種犯規狀態。
    """
    foul_cases = [
        # (first_contact, pocketed_object_balls, rail_contacted) → 三種 BreakFoulResult
        (1, {3}, set()),  # 合法：有進袋
        (1, set(), {1, 2, 3, 4}),  # 合法：4 顆碰顆星
        (1, set(), {1, 2}),  # -0.5 未達 4 顆
        (5, set(), {1, 2, 3, 4}),  # -1.5 首次接觸不是 1 號球，且 should_reset
        (None, set(), set()),  # -1.5 整局沒碰到任何球
    ]
    for (first, pocketed, rails), cue_pocketed, nine_pocketed, spread_score in (
        itertools.product(foul_cases, (False, True), (False, True), (0.0, 0.37, 1.0))
    ):
        break_foul_result = evaluate_break_foul(first, pocketed, rails)
        shot_result = ShotResult(
            final_ball_positions=[[0.0, 0.0]] * _BALL_COUNT,
            cue_ball_pocketed=cue_pocketed,
            nine_ball_pocketed=nine_pocketed,
            spread_score=spread_score,
        )

        components = decompose_reward(shot_result, break_foul_result)
        expected = calculate_reward(shot_result, break_foul_result)

        assert sum(components.values()) == pytest.approx(expected, abs=1e-9), (
            f"分解與 core 不一致：first={first} pocketed={pocketed} rails={rails} "
            f"cue={cue_pocketed} nine={nine_pocketed} spread={spread_score} "
            f"→ {components} vs {expected}"
        )


def test_decomposition_zeroes_everything_on_reset_foul():
    """`should_reset` 時 core 只回傳罰分，其餘三項必須是 0（不是「剛好抵消」）。"""
    break_foul_result = evaluate_break_foul(5, set(), {1, 2, 3, 4})
    assert break_foul_result.should_reset

    components = decompose_reward(
        ShotResult(
            final_ball_positions=[[0.0, 0.0]] * _BALL_COUNT,
            cue_ball_pocketed=True,
            nine_ball_pocketed=True,
            spread_score=1.0,
        ),
        break_foul_result,
    )

    assert components["spread"] == 0.0
    assert components["cue_scratch"] == 0.0
    assert components["nine_ball"] == 0.0
    assert components["foul"] == break_foul_result.penalty


def test_evaluate_shot_uses_pocket_coordinates_for_pocketed_balls():
    """進袋球必須以**袋口座標**進入散開分數，不是它在檯面上的殘留位置。

    這是 calculate_spread_score() docstring 明文要求的呼叫端責任，也是訓練端
    不需要把球搬離檯面的原因。
    """
    ball_xy = [BREAK_SHOT_POSITIONS[b] for b in sorted(BREAK_SHOT_POSITIONS)]
    pocket_index = [-1] * _BALL_COUNT
    rail = [True] * _BALL_COUNT

    baseline = evaluate_shot(ball_xy, pocket_index, rail, first_contact=1)

    # 3 號球進了 Pocket_FootLeft（索引 4），但它在檯面上的座標沒變
    pocket_index[3] = 4
    pocketed = evaluate_shot(ball_xy, pocket_index, rail, first_contact=1)

    assert pocketed["spread"] != baseline["spread"], (
        "進袋球若沒有被代換成袋口座標，散開分數不會改變"
    )


def test_evaluate_shot_on_break_layout_is_a_valid_reward():
    """未擊球的開球擺位也要算得出合法 reward（不得丟例外）。"""
    ball_xy = [BREAK_SHOT_POSITIONS[b] for b in sorted(BREAK_SHOT_POSITIONS)]

    components = evaluate_shot(
        ball_xy, [-1] * _BALL_COUNT, [False] * _BALL_COUNT, first_contact=-1
    )

    # 沒碰到任何球 → 首次接觸不是 1 號球 → -1.5 且重置
    assert components["foul"] == pytest.approx(-1.5)
    assert components["spread"] == 0.0


def test_evaluate_shot_flags_cue_and_nine_ball_pockets():
    ball_xy = [BREAK_SHOT_POSITIONS[b] for b in sorted(BREAK_SHOT_POSITIONS)]
    pocket_index = [-1] * _BALL_COUNT
    pocket_index[0] = 2  # 母球落袋
    rail = [True] * _BALL_COUNT

    components = evaluate_shot(ball_xy, pocket_index, rail, first_contact=1)

    assert components["cue_scratch"] < 0.0
    # 母球進袋 → 9 號球獎勵歸零
    pocket_index[9] = 5
    with_nine = evaluate_shot(ball_xy, pocket_index, rail, first_contact=1)
    assert with_nine["nine_ball"] == 0.0
