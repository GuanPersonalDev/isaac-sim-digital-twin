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
import math

import pytest

pytest.importorskip("torch", reason="本機無 torch，本檔只在 pod 上跑")

import torch  # noqa: E402

from billiard_rl.tasks.manager_based.billiard_rl.mdp.rewards import (  # noqa: E402
    decompose_reward,
    evaluate_shot,
)
from billiard_rl.tasks.manager_based.billiard_rl.mdp.shot_tracking import (  # noqa: E402
    ONE_BALL_INDEX,
    detect_cue_contact,
    detect_pocketed,
    detect_rail_contact,
    update_closest_approach,
    update_first_contact,
)
from core.models.shot_result import ShotResult  # noqa: E402
from core.models.table_ball_set import TableBallSet  # noqa: E402
from core.services.aim_shaping_calculator import (  # noqa: E402
    AIM_REFERENCE_GAP,
    AIM_REWARD_SCALE,
)
from core.services.break_foul_evaluator import (  # noqa: E402
    NO_CONTACT_FOUL_PENALTY,
    evaluate_break_foul,
)
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
# #124：dense shaping 的距離追蹤
##


def _unset(n: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """`(closest_approach, first_contact)` 的初值。"""
    return (
        torch.full((n,), float("inf"), dtype=torch.float64),
        torch.full((n,), -1, dtype=torch.long),
    )


def test_closest_approach_measures_surface_gap_not_centre_distance():
    ball_xy = _layout({ONE_BALL_INDEX: 4.0 * _RADIUS})
    closest, first = _unset()

    closest = update_closest_approach(closest, ball_xy, _RADIUS, first)

    # 球心距 4r，表面間距 4r - 2r = 2r
    assert float(closest[0]) == pytest.approx(2.0 * _RADIUS)


def test_closest_approach_keeps_the_minimum_over_the_episode():
    closest, first = _unset()

    for gap in (1.0, 0.3, 0.7):
        ball_xy = _layout({ONE_BALL_INDEX: gap + 2.0 * _RADIUS})
        closest = update_closest_approach(closest, ball_xy, _RADIUS, first)

    assert float(closest[0]) == pytest.approx(0.3), "0.7 不該覆寫掉更近的 0.3"


def test_closest_approach_floors_at_zero_on_contact():
    """PhysX 的接觸解算容許輕微重疊，間距不得跑成負值。

    0.0 的語意固定是「碰到了」＝塑形滿分，負值會讓 `closest_approach_to_reward()`
    直接拋 ValueError。
    """
    ball_xy = _layout({ONE_BALL_INDEX: 1.5 * _RADIUS})  # 重疊
    closest, first = _unset()

    closest = update_closest_approach(closest, ball_xy, _RADIUS, first)

    assert float(closest[0]) == 0.0


def test_closest_approach_freezes_after_any_contact():
    """碰到球之後就定案——散開的 1 號球滾過母球旁邊不得補發塑形分。"""
    ball_xy = _layout({ONE_BALL_INDEX: 0.5 + 2.0 * _RADIUS})
    closest, _ = _unset()
    contacted = torch.full((1,), 5, dtype=torch.long)  # 已經碰到 5 號球

    closest = update_closest_approach(closest, ball_xy, _RADIUS, contacted)

    assert float(closest[0]) == float("inf")


def test_closest_approach_records_the_contact_tick_itself():
    """命中 1 號球的那個 tick 必須拿得到 ~0。

    `actions.py` 的呼叫順序保證這個 tick 的 `first_contact` 還是 -1；順序反過來
    的話，真正打中的那一局反而拿不到塑形滿分。
    """
    ball_xy = _layout({ONE_BALL_INDEX: 2.0 * _RADIUS})
    closest, first = _unset()

    closest = update_closest_approach(closest, ball_xy, _RADIUS, first)
    first = update_first_contact(first, ball_xy, _RADIUS)

    assert int(first[0]) == ONE_BALL_INDEX
    assert float(closest[0]) == pytest.approx(0.0, abs=1e-12)


def test_closest_approach_is_per_env():
    closest, first = _unset(2)
    ball_xy = torch.cat(
        [
            _layout({ONE_BALL_INDEX: 0.2 + 2.0 * _RADIUS}),
            _layout({ONE_BALL_INDEX: 0.8 + 2.0 * _RADIUS}),
        ]
    )

    closest = update_closest_approach(closest, ball_xy, _RADIUS, first)

    assert float(closest[0]) == pytest.approx(0.2)
    assert float(closest[1]) == pytest.approx(0.8)


##
# B-3b／B-3c：reward 分項
##


def test_decomposition_sums_to_core_reward():
    """**最重要的一項**：五項之和必須恆等於 `calculate_reward()`。

    窮舉 reward 鏈的所有分支組合：母球進袋 × 9 號球進袋 × 四種犯規狀態
    × 塑形距離（含跨過 `should_reset` 分支的那一條）。
    """
    foul_cases = [
        # (first_contact, pocketed_object_balls, rail_contacted) → 四種 BreakFoulResult
        (1, {3}, set()),  # 合法：有進袋
        (1, set(), {1, 2, 3, 4}),  # 合法：4 顆碰顆星
        (1, set(), {1, 2}),  # -0.5 未達 4 顆
        (5, set(), {1, 2, 3, 4}),  # -1.5 首次接觸不是 1 號球，且 should_reset
        (None, set(), set()),  # -2.0 整局沒碰到任何球（#124）
    ]
    # 塑形距離也要進窮舉：aim 是唯一跨過 should_reset 分支的項目，只測 inf
    # 的話那條路徑上的分解錯誤永遠不會被抓到。
    approaches = (0.0, AIM_REFERENCE_GAP / 3.0, AIM_REFERENCE_GAP, math.inf)
    for (
        (first, pocketed, rails),
        cue_pocketed,
        nine_pocketed,
        spread_score,
        approach,
    ) in itertools.product(
        foul_cases, (False, True), (False, True), (0.0, 0.37, 1.0), approaches
    ):
        break_foul_result = evaluate_break_foul(first, pocketed, rails)
        shot_result = ShotResult(
            final_ball_positions=[[0.0, 0.0]] * _BALL_COUNT,
            cue_ball_pocketed=cue_pocketed,
            nine_ball_pocketed=nine_pocketed,
            spread_score=spread_score,
            closest_approach=approach,
        )

        components = decompose_reward(shot_result, break_foul_result)
        expected = calculate_reward(shot_result, break_foul_result)

        assert sum(components.values()) == pytest.approx(expected, abs=1e-9), (
            f"分解與 core 不一致：first={first} pocketed={pocketed} rails={rails} "
            f"cue={cue_pocketed} nine={nine_pocketed} spread={spread_score} "
            f"approach={approach} → {components} vs {expected}"
        )


def test_decomposition_zeroes_everything_on_reset_foul():
    """`should_reset` 時 core 只回傳罰分（+塑形），其餘三項必須是 0。

    「是 0」而不是「剛好抵消」——抵消的話換一組輸入就會露餡。
    """
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
    # 沒帶 closest_approach → inf → 塑形 0 分，維持改動前的數值。
    assert components["aim"] == 0.0


def test_aim_is_the_only_term_paid_on_a_reset_foul():
    """#124 的核心：塑形必須跨過 `should_reset` 分支。

    訓練初期壓倒性多數的 episode 都走這個分支，塑形若跟其他四項一樣被吃掉
    就等於沒加——而且完全不報錯，只是 policy 學不動。
    """
    break_foul_result = evaluate_break_foul(None, set(), set())
    assert break_foul_result.should_reset

    components = decompose_reward(
        ShotResult(
            final_ball_positions=[[0.0, 0.0]] * _BALL_COUNT,
            cue_ball_pocketed=False,
            nine_ball_pocketed=False,
            spread_score=1.0,
            closest_approach=0.0,
        ),
        break_foul_result,
    )

    assert components["aim"] == pytest.approx(AIM_REWARD_SCALE)
    assert components["spread"] == 0.0
    assert components["cue_scratch"] == 0.0
    assert components["nine_ball"] == 0.0


def test_missing_every_ball_is_worse_than_hitting_the_wrong_one():
    """犯規階梯必須嚴格遞增，包含塑形之後也一樣（#124）。

    第一輪訓練就是踩在兩者同為 -1.5 的那片平原上收斂到亂打。
    """
    positions = [[0.0, 0.0]] * _BALL_COUNT

    best_miss = calculate_reward(
        ShotResult(positions, False, False, 0.0, closest_approach=0.0),
        evaluate_break_foul(None, set(), set()),
    )
    worst_wrong_contact = calculate_reward(
        ShotResult(positions, False, False, 0.0, closest_approach=math.inf),
        evaluate_break_foul(5, set(), {1, 2, 3, 4}),
    )

    assert best_miss < worst_wrong_contact


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

    # 沒碰到任何球 → -2.0 且重置（#124 起與「碰到錯球」的 -1.5 分開）
    assert components["foul"] == pytest.approx(NO_CONTACT_FOUL_PENALTY)
    assert components["spread"] == 0.0
    # closest_approach 沒帶 → inf → 塑形 0 分
    assert components["aim"] == 0.0


def test_evaluate_shot_passes_closest_approach_through_to_the_aim_term():
    ball_xy = [BREAK_SHOT_POSITIONS[b] for b in sorted(BREAK_SHOT_POSITIONS)]

    components = evaluate_shot(
        ball_xy,
        [-1] * _BALL_COUNT,
        [False] * _BALL_COUNT,
        first_contact=-1,
        closest_approach=0.0,
    )

    assert components["aim"] == pytest.approx(AIM_REWARD_SCALE)


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
