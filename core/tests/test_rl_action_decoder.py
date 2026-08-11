import math

import pytest

from core.models.action import Action
from core.models.action_bounds import (
    ACTION_BOUNDS,
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    SHOT_ANGLE,
)
from core.services.rl_action_decoder import decode_rl_action, normalize_action


_PLACEMENT_X, _PLACEMENT_Y, _SHOT_ANGLE, _SPEED, _OFFSET_V, _OFFSET_H = range(
    ACTION_DIM
)
_ANGLE_PERIOD = 360.0

# #110 的 6 維索引表，附上該欄位在 Action 上的讀取路徑。索引錯位時每個數字
# 仍然落在合法範圍內，Action 組得出來、物理跑得動、結果全錯，只能靠這張表
# 把「第 N 維對應哪個欄位」釘死。期望值一律由 ACTION_BOUNDS 推導，不在測試
# 裡重寫字面數值（#114 單一來源）。
_FIELD_READERS = (
    ("placement_x", _PLACEMENT_X, lambda action: action.cue_ball_placement[0]),
    ("placement_y", _PLACEMENT_Y, lambda action: action.cue_ball_placement[1]),
    ("shot_angle", _SHOT_ANGLE, lambda action: action.shot_angle),
    ("cue_ball_speed", _SPEED, lambda action: action.cue_ball_speed),
    ("offset_vertical", _OFFSET_V, lambda action: action.position_offset[0]),
    ("offset_horizontal", _OFFSET_H, lambda action: action.position_offset[1]),
)

_FIELD_PARAMS = tuple(
    pytest.param(index, reader, id=test_id)
    for test_id, index, reader in _FIELD_READERS
)


@pytest.fixture(params=[math.nan, math.inf, -math.inf])
def non_finite_value(request) -> float:
    return request.param


@pytest.fixture(params=[True, "invalid", None])
def non_numeric_value(request):
    return request.param


class TestFieldOrder:
    @pytest.mark.parametrize(("index", "reader"), _FIELD_PARAMS)
    def test_plus_one_lifts_only_its_own_field_to_the_upper_bound(
        self, index: int, reader
    ):
        # Act
        action = decode_rl_action(_unit_vector(index, 1.0), 1.0)

        # Assert
        assert reader(action) == pytest.approx(_expected_endpoint(index, 1.0))
        _assert_other_fields_stay_at_midpoint(action, index)

    @pytest.mark.parametrize(("index", "reader"), _FIELD_PARAMS)
    def test_minus_one_drops_only_its_own_field_to_the_lower_bound(
        self, index: int, reader
    ):
        # Act
        action = decode_rl_action(_unit_vector(index, -1.0), 1.0)

        # Assert
        assert reader(action) == pytest.approx(_expected_endpoint(index, -1.0))
        _assert_other_fields_stay_at_midpoint(action, index)

    def test_offset_axes_are_not_swapped(self):
        # 兩維偏移的上下限完全相同，只比對數值抓不到互換；分別送單軸值才
        # 能確認 index 4 進的是上下、index 5 進的是左右。
        # Act
        vertical_only = decode_rl_action(_unit_vector(_OFFSET_V, 1.0), 1.0)
        horizontal_only = decode_rl_action(_unit_vector(_OFFSET_H, 1.0), 1.0)

        # Assert
        assert vertical_only.position_offset[1] == pytest.approx(0.0)
        assert horizontal_only.position_offset[0] == pytest.approx(0.0)

    def test_zero_vector_lands_on_every_midpoint(self):
        # Act
        action = decode_rl_action([0.0] * ACTION_DIM, 1.0)

        # Assert
        for _, index, reader in _FIELD_READERS:
            assert reader(action) == pytest.approx(_midpoint(index))


class TestDenormalization:
    def test_bounds_come_from_action_bounds(self):
        # 反正規化的上下限必須全部取自 #114 的單一來源，實作中不得硬編碼。
        # Act
        low_ends = [
            reader(decode_rl_action(_unit_vector(index, -1.0), 1.0))
            for _, index, reader in _FIELD_READERS
        ]
        high_ends = [
            reader(decode_rl_action(_unit_vector(index, 1.0), 1.0))
            for _, index, reader in _FIELD_READERS
        ]

        # Assert
        assert low_ends == pytest.approx(
            [_expected_endpoint(index, -1.0) for index in range(ACTION_DIM)]
        )
        assert high_ends == pytest.approx(
            [_expected_endpoint(index, 1.0) for index in range(ACTION_DIM)]
        )

    @pytest.mark.parametrize(("index", "reader"), _FIELD_PARAMS)
    def test_values_above_one_are_capped_at_the_upper_bound(
        self, index: int, reader
    ):
        # Act
        capped = decode_rl_action(_unit_vector(index, 2.0), 1.0)
        at_limit = decode_rl_action(_unit_vector(index, 1.0), 1.0)

        # Assert
        assert reader(capped) == pytest.approx(reader(at_limit))

    @pytest.mark.parametrize(("index", "reader"), _FIELD_PARAMS)
    def test_values_below_minus_one_are_capped_at_the_lower_bound(
        self, index: int, reader
    ):
        # Act
        capped = decode_rl_action(_unit_vector(index, -3.0), 1.0)
        at_limit = decode_rl_action(_unit_vector(index, -1.0), 1.0)

        # Assert
        assert reader(capped) == pytest.approx(reader(at_limit))

    def test_offset_denormalization_is_pure_scaling(self):
        # 偏移兩維的中點恰為 0，中點式反正規化退化成純等比縮放；一旦混入
        # 平移，圓形 clamp 保住的方向就會在物理域被扭掉（#222）。
        # Act
        positive = decode_rl_action(_unit_vector(_OFFSET_V, 0.3), 1.0)
        negative = decode_rl_action(_unit_vector(_OFFSET_V, -0.3), 1.0)

        # Assert
        assert positive.position_offset[0] == pytest.approx(
            -negative.position_offset[0]
        )
        assert decode_rl_action([0.0] * ACTION_DIM, 1.0).position_offset == [
            0.0,
            0.0,
        ]


class TestShotAngleWrap:
    def test_upper_bound_wraps_to_the_lower_bound(self):
        # 物理域是半開區間 [-180, 180)。ACTION_BOUNDS 的 high 記為 180.0，
        # 必須由反正規化尾端折回，否則 +1 會還原成 180.0 —— 與 -180.0 同方向
        # 卻是不同數值，兩端各自處理就會在邊界靜默不一致。
        # Act
        action = decode_rl_action(_unit_vector(_SHOT_ANGLE, 1.0), 1.0)

        # Assert
        assert action.shot_angle == pytest.approx(SHOT_ANGLE[0])

    def test_value_just_below_the_upper_bound_is_not_wrapped(self):
        # Act
        action = decode_rl_action(_unit_vector(_SHOT_ANGLE, 0.99), 1.0)

        # Assert
        assert action.shot_angle == pytest.approx(SHOT_ANGLE[1] * 0.99)

    @pytest.mark.parametrize("normalized", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_result_always_stays_in_the_half_open_interval(
        self, normalized: float
    ):
        # Act
        action = decode_rl_action(_unit_vector(_SHOT_ANGLE, normalized), 1.0)

        # Assert
        assert SHOT_ANGLE[0] <= action.shot_angle < SHOT_ANGLE[0] + _ANGLE_PERIOD

    def test_neutral_action_aims_straight_at_the_rack(self):
        # #231 的核心：Gaussian policy 的初始輸出集中在 normalized 0，那個值
        # 必須對應「正對球堆」（0° 朝桌台 +Y）。舊區間 (0, 360) 會還原成
        # 180°——未訓練的 policy 預設把母球往 kitchen 底庫打。
        # Act
        action = decode_rl_action(_unit_vector(_SHOT_ANGLE, 0.0), 1.0)

        # Assert
        assert action.shot_angle == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("physical_angle", "equivalent_angle"),
        [(270.0, -90.0), (359.0, -1.0), (180.0, -180.0), (-370.0, -10.0)],
    )
    def test_out_of_interval_angles_fold_to_the_same_direction(
        self, physical_angle: float, equivalent_angle: float
    ):
        # 週期固定是 360，不是區間寬度。normalize_action 是公開介面，外部
        # 傳進來的角度不保證已經折過。
        # Assert
        assert normalize_action(
            _action(shot_angle=physical_angle)
        ) == pytest.approx(normalize_action(_action(shot_angle=equivalent_angle)))


class TestClampRunsBeforeDenormalization:
    def test_max_offset_above_the_physical_diagonal_is_not_a_dead_zone(self):
        # 順序防迴歸（本 Issue 完成標準）：若改成先反正規化再 clamp，物理域
        # 的最大範數只有 hypot(0.5, 0.5) ≈ 0.707，max_offset ∈ (0.707, 1.0]
        # 會整段變成死區，policy 分不出 0.8 與 1.0。
        # Arrange
        corner = _offset_vector(1.0, 1.0)

        # Act
        limited = decode_rl_action(corner, 0.8)
        full = decode_rl_action(corner, 1.0)

        # Assert
        assert math.hypot(*limited.position_offset) == pytest.approx(0.4)
        assert math.hypot(*full.position_offset) == pytest.approx(0.5)

    def test_limit_applies_on_the_normalized_scale(self):
        # max_offset = 0.5 是「可用偏移能力的一半」，在正規化域裁到範數
        # 0.5，反正規化 ×0.5 後物理域範數為 0.25。若順序對調會得到 0.5。
        # Act
        action = decode_rl_action(_offset_vector(1.0, 1.0), 0.5)

        # Assert
        assert math.hypot(*action.position_offset) == pytest.approx(0.25)

    def test_out_of_range_offset_keeps_its_direction(self):
        # 分軸截斷防迴歸：逐軸 clip 會把 [2.0, 0.01] 壓成 [1.0, 0.01]，方向
        # 整個歪掉；圓形裁切只縮長度。
        # Arrange
        raw_vertical, raw_horizontal = 2.0, 0.01

        # Act
        action = decode_rl_action(
            _offset_vector(raw_vertical, raw_horizontal), 1.0
        )

        # Assert
        assert math.atan2(*reversed(action.position_offset)) == pytest.approx(
            math.atan2(raw_horizontal, raw_vertical)
        )

    def test_zero_capability_collapses_the_offset(self):
        # Act
        action = decode_rl_action(_offset_vector(1.0, -1.0), 0.0)

        # Assert
        assert action.position_offset == [0.0, 0.0]


class TestShouldExecuteAction:
    def test_defaults_to_true(self):
        # Act
        action = decode_rl_action([0.0] * ACTION_DIM, 1.0)

        # Assert
        assert action.should_execute_action is True

    def test_is_taken_from_the_argument_not_the_vector(self):
        # Act
        action = decode_rl_action(
            [0.0] * ACTION_DIM, 1.0, should_execute_action=False
        )

        # Assert
        assert action.should_execute_action is False

    def test_seventh_dimension_is_rejected(self):
        # should_execute_action 由 Controller 的狀態轉換產生，不是第 7 維
        # 模型輸出；多送一維必須擋下來而不是默默吃掉。
        # Assert
        with pytest.raises(ValueError):
            decode_rl_action([0.0] * (ACTION_DIM + 1), 1.0)


class TestDecodeInputValidation:
    @pytest.mark.parametrize("length", [0, 5, 7, 21])
    def test_rejects_wrong_length(self, length: int):
        # Assert
        with pytest.raises(ValueError):
            decode_rl_action([0.0] * length, 1.0)

    @pytest.mark.parametrize("index", range(ACTION_DIM))
    def test_rejects_non_finite_value_and_names_the_index(
        self, index: int, non_finite_value: float
    ):
        # Assert
        with pytest.raises(ValueError, match=rf"raw_action\[{index}\]"):
            decode_rl_action(_unit_vector(index, non_finite_value), 1.0)

    @pytest.mark.parametrize("index", range(ACTION_DIM))
    def test_rejects_non_numeric_value(self, index: int, non_numeric_value):
        # Arrange
        raw_action = [0.0] * ACTION_DIM
        raw_action[index] = non_numeric_value

        # Assert
        with pytest.raises(ValueError):
            decode_rl_action(raw_action, 1.0)

    @pytest.mark.parametrize("max_offset", [-0.1, 1.1, math.nan])
    def test_rejects_invalid_max_offset(self, max_offset: float):
        # Assert
        with pytest.raises(ValueError):
            decode_rl_action([0.0] * ACTION_DIM, max_offset)

    @pytest.mark.parametrize("max_offset", [0.0, 0.5, 1.0])
    def test_accepts_max_offset_boundaries(self, max_offset: float):
        # Act
        action = decode_rl_action([0.0] * ACTION_DIM, max_offset)

        # Assert
        assert isinstance(action, Action)


class TestRoundTrip:
    def test_normalized_vector_survives_decode_then_normalize(self):
        # Arrange
        # 未觸發 clip 與 clamp 的一般值：偏移範數 0.224 遠小於 max_offset。
        raw_action = [0.3, -0.4, 0.25, 0.6, 0.2, -0.1]

        # Act
        recovered = normalize_action(decode_rl_action(raw_action, 1.0))

        # Assert
        assert recovered == pytest.approx(raw_action)

    def test_physical_action_survives_normalize_then_decode(self):
        # Arrange
        action = _action()

        # Act
        restored = decode_rl_action(normalize_action(action), 1.0)

        # Assert
        assert restored.cue_ball_placement == pytest.approx(
            action.cue_ball_placement
        )
        assert restored.shot_angle == pytest.approx(action.shot_angle)
        assert restored.cue_ball_speed == pytest.approx(action.cue_ball_speed)
        assert restored.position_offset == pytest.approx(action.position_offset)

    def test_angle_upper_bound_round_trips_to_the_same_direction(self):
        # +1 還原成 -180°（= +180°，背對球堆），再正規化回去是 -1 —— 數值
        # 不同、方向相同。往返在這一維只能斷言方向等價。
        # Act
        action = decode_rl_action(_unit_vector(_SHOT_ANGLE, 1.0), 1.0)
        recovered = normalize_action(action)

        # Assert
        assert action.shot_angle == pytest.approx(SHOT_ANGLE[0])
        assert recovered[_SHOT_ANGLE] == pytest.approx(-1.0)


class TestNormalizeActionInputContract:
    """`normalize_action` 是公開函式，呼叫者不保證 Action 來自 decode。

    以下三項依「應有行為」撰寫，對應 core-review 的必須修正項目 1 與 2。
    """

    def test_rejects_placement_with_wrong_length(self):
        # core-review finding 1：目前會拋 IndexError 而非帶欄位名的
        # ValueError，非法資料的根因無法從例外訊息回溯。
        # Assert
        with pytest.raises(ValueError):
            normalize_action(_action(cue_ball_placement=[0.0]))

    def test_rejects_non_finite_physical_value(self, non_finite_value: float):
        # core-review finding 1：目前 NaN/Inf 會直接除過去靜默傳播，訓練端
        # 拿到 NaN action 後極難回溯。
        # Assert
        with pytest.raises(ValueError):
            normalize_action(_action(cue_ball_speed=non_finite_value))

    @pytest.mark.parametrize("shot_angle", [360.0, 370.0, -10.0, 270.0, -270.0])
    def test_wraps_shot_angle_into_the_normalized_domain(
        self, shot_angle: float
    ):
        # core-review finding 2：decode 尾端有折回，反向沒有。傳入 370.0
        # 會算出越界值，下游若做 Box.contains() 檢查會是未定義行為。
        # #231 之後區間變成 [-180, 180)，270.0 是新增的越界案例——若沿用
        # 舊的 `% 360` 會得到 270 → 正規化 1.5。
        # Act
        recovered = normalize_action(_action(shot_angle=shot_angle))

        # Assert
        assert -1.0 <= recovered[_SHOT_ANGLE] <= 1.0


def _unit_vector(index: int, value: float) -> list[float]:
    raw_action: list[float] = [0.0] * ACTION_DIM
    raw_action[index] = value
    return raw_action


def _offset_vector(vertical: float, horizontal: float) -> list[float]:
    raw_action: list[float] = [0.0] * ACTION_DIM
    raw_action[_OFFSET_V] = vertical
    raw_action[_OFFSET_H] = horizontal
    return raw_action


def _midpoint(index: int) -> float:
    low, high = ACTION_BOUNDS[index]
    return (low + high) / 2.0


def _expected_endpoint(index: int, sign: float) -> float:
    bound = ACTION_HIGH[index] if sign > 0 else ACTION_LOW[index]
    if index != _SHOT_ANGLE:
        return bound
    # 角度是半開區間 [SHOT_ANGLE[0], SHOT_ANGLE[0] + 360)，兩個端點是同一個
    # 方向，都會被折到下界那一圈。不能寫成 `bound % 360`——Python 的 % 對
    # 負數取正餘數，-180 會變成 +180，正好折反邊。
    return (bound - SHOT_ANGLE[0]) % _ANGLE_PERIOD + SHOT_ANGLE[0]


def _assert_other_fields_stay_at_midpoint(action: Action, index: int) -> None:
    for _, other_index, reader in _FIELD_READERS:
        if other_index == index:
            continue
        assert reader(action) == pytest.approx(_midpoint(other_index)), (
            f"index {index} 的值外洩到 index {other_index}"
        )


def _action(**overrides) -> Action:
    fields = {
        "cue_ball_placement": [
            _midpoint(_PLACEMENT_X),
            _midpoint(_PLACEMENT_Y),
        ],
        "shot_angle": _midpoint(_SHOT_ANGLE),
        "cue_ball_speed": _midpoint(_SPEED),
        "position_offset": [0.2, -0.1],
        "should_execute_action": True,
    }
    fields.update(overrides)
    return Action(**fields)
