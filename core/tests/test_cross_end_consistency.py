"""#228：訓練端與 Demo 端走的是同一份 `core/` 組裝／還原邏輯。

本檔驗的不是「兩個函式算出同樣的數字」，而是**兩端真的走同一條路**。維度或
欄位順序不一致時 policy 不會報錯，只會安靜輸出垃圾動作，所以一致性必須有測試
擋著，不能只靠 code review。

訓練端（`rl_task/.../mdp/actions.py` 的 `BilliardStrikeAction`）在本機 import
不進來——它繼承 `isaaclab.managers.ActionTerm`，而本機沒有 Isaac Lab，也沒有
torch。因此兩端用不同手段驗：

    Demo 端      實際跑 ModelController 的狀態機，攔截它對 core 的呼叫
    訓練端       用 AST 讀原始碼，斷言呼叫點的形式與順序

兩者合起來才成立：Demo 端證明「輸出就是 `decode_rl_action(raw, offset, True)`
本身，沒有多做也沒有少做」，AST 證明「訓練端寫的是同一個呼叫」，所以同一組
輸入必然得到同一個 `Action`。

21 維 observation 的**數值**對拍不在本檔——訓練端是 torch 向量化的另一份實作，
比對需要 torch，見 `rl_task/tests/test_mdp_observations.py`（只能在 pod 上跑）。
本檔負責的是那份實作與 `core` 之間的靜態綁定（欄位順序常數、無重複實作），
以及 Demo 端確實把編碼工作交給了 `core`。
"""

import ast
from pathlib import Path

import pytest

from core.controllers import model_controller as model_controller_module
from core.controllers.model_controller import ModelController
from core.models.action import Action
from core.models.action_bounds import ACTION_DIM
from core.models.observation import Observation
from core.ports.policy_port import PolicyPort
from core.services.rl_action_decoder import decode_rl_action
from core.services.rl_observation_encoder import RL_BALL_ORDER


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_DECODER = _REPO_ROOT / "core" / "services" / "rl_action_decoder.py"
_CORE_ENCODER = _REPO_ROOT / "core" / "services" / "rl_observation_encoder.py"
_TRAINING_MDP = (
    _REPO_ROOT
    / "rl_task"
    / "billiard_rl"
    / "tasks"
    / "manager_based"
    / "billiard_rl"
    / "mdp"
)
_TRAINING_ACTION_TERM = _TRAINING_MDP / "actions.py"
_TRAINING_OBS_TERM = _TRAINING_MDP / "observations.py"
_DEMO_CONTROLLER = _REPO_ROOT / "core" / "controllers" / "model_controller.py"

_TABLE_POSITION = (1.5, -2.0)
_MAX_OFFSET = 0.6
_BALL_COUNT = 10


class _FakePolicy(PolicyPort):
    """回傳固定輸出，並記錄每次收到的觀測。"""

    def __init__(self, output: list[float]) -> None:
        self.output = output
        self.calls: list[list[float]] = []

    def infer(self, observation: list[float]) -> list[float]:
        self.calls.append(list(observation))
        return list(self.output)


def _observation(
    is_init_state: bool = False,
    is_ball_moving: bool = False,
    is_motion_complete: bool = False,
) -> Observation:
    """球 i 的桌台相對座標固定為 (0.01i, -0.02i)，回傳的是世界座標。"""
    table_x, table_y = _TABLE_POSITION
    ball_positions = [
        [table_x + ball_id * 0.01, table_y - ball_id * 0.02, 0.028575]
        for ball_id in range(_BALL_COUNT)
    ]
    return Observation(
        ball_positions=ball_positions,
        cue_ball_position=ball_positions[0],
        is_init_state=is_init_state,
        is_ball_moving=is_ball_moving,
        is_motion_complete=is_motion_complete,
        has_error=False,
    )


def _demo_end_action(raw_action: list[float], max_offset: float) -> Action:
    """跑完 Demo 端的完整路徑，回傳 IDLE → AIMING 那次推論產生的 `Action`。"""
    controller = ModelController(
        _FakePolicy(raw_action), _TABLE_POSITION, max_offset
    )
    controller.get_action(_observation(is_motion_complete=True))
    return controller.get_action(_observation(is_init_state=True))


def _training_end_action(raw_action: list[float], max_offset: float) -> Action:
    """訓練端 `_apply_strike()` 的逐 env 還原呼叫。

    這一行的形式（兩個位置參數 + `should_execute_action=True` 字面值）由
    `TestTrainingEndCallSite` 對著 `actions.py` 的 AST 釘死，所以這裡的轉寫
    不會與訓練端漂移——漂移的話那組測試會先失敗。
    """
    return decode_rl_action(raw_action, max_offset, should_execute_action=True)


##
# AST 工具
##


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in ast.walk(node):
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                return item
    raise AssertionError(f"找不到 {class_name}.{method_name}()")


def _calls(node: ast.AST, func_name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == func_name
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"呼叫缺少關鍵字引數 {name}")


def _imported_from(tree: ast.Module, module: str) -> set[str]:
    """`module` 用原始寫法，相對匯入含前綴點（例如 `..services.foo`）。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if "." * node.level + (node.module or "") == module:
            names |= {alias.name for alias in node.names}
    return names


_SKIPPED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".vscode",
    ".cursor",
    "assets",
}


def _python_files() -> list[Path]:
    return [
        path
        for path in _REPO_ROOT.rglob("*.py")
        if not _SKIPPED_DIRS & set(path.relative_to(_REPO_ROOT).parts)
    ]


def _outside_core() -> list[Path]:
    return [
        path
        for path in _python_files()
        if path.relative_to(_REPO_ROOT).parts[0] != "core"
    ]


def _where(paths: list[Path]) -> list[str]:
    """失敗訊息用的相對路徑，絕對路徑在 CI log 裡讀不出重點。"""
    return sorted(str(path.relative_to(_REPO_ROOT)) for path in paths)


##
# Demo 端：實際執行，攔截它對 core 的呼叫
##


class TestDemoEndDelegatesToCore:
    def test_policy_receives_exactly_the_shared_encoders_output(self, monkeypatch):
        # 「送進 policy 的是 21 維桌台相對向量」已有 test_model_controller
        # 驗過數值；這裡驗的是**那個值出自共用函式**，而不是 Controller 自己
        # 湊出一個剛好長得一樣的向量。
        # Arrange
        recorded: list[tuple] = []
        real_encode = model_controller_module.encode_rl_observation

        def _spy(observation, table_position, max_offset):
            recorded.append((observation, table_position, max_offset))
            return real_encode(observation, table_position, max_offset)

        monkeypatch.setattr(model_controller_module, "encode_rl_observation", _spy)
        policy = _FakePolicy([0.0] * ACTION_DIM)
        controller = ModelController(policy, _TABLE_POSITION, _MAX_OFFSET)
        controller.get_action(_observation(is_motion_complete=True))
        observation = _observation(is_init_state=True)

        # Act
        controller.get_action(observation)

        # Assert
        assert len(recorded) == 1
        assert recorded[0][0] is observation
        assert recorded[0][1] == _TABLE_POSITION
        assert recorded[0][2] == _MAX_OFFSET
        assert policy.calls[0] == real_encode(
            observation, _TABLE_POSITION, _MAX_OFFSET
        )

    def test_decode_is_called_with_the_policy_output_and_the_same_max_offset(
        self, monkeypatch
    ):
        # max_offset 一值兩用：第 21 維的條件值與圓形裁切半徑必須是同一個數。
        # 傳成兩個不同的值不會報錯，policy 看到的條件與實際生效的裁切就對不上。
        # Arrange
        recorded: list[tuple] = []
        real_decode = model_controller_module.decode_rl_action

        def _spy(raw_action, max_offset, should_execute_action=True):
            recorded.append((raw_action, max_offset, should_execute_action))
            return real_decode(raw_action, max_offset, should_execute_action)

        monkeypatch.setattr(model_controller_module, "decode_rl_action", _spy)
        raw_action = [0.3, -0.4, 0.5, 0.9, 0.7, -0.2]

        # Act
        _demo_end_action(raw_action, _MAX_OFFSET)

        # Assert
        assert recorded == [(raw_action, _MAX_OFFSET, True)]

    def test_striking_repeats_the_identical_decode_call(self, monkeypatch):
        # AIMING → STRIKING 重新 decode 一次（Action 是 mutable dataclass，
        # 兩次分派不能共用同一個 instance）。重新 decode 的引數必須與 IDLE
        # 那次完全相同，否則手臂瞄的方向與實際擊出的方向會不一致。
        # Arrange
        recorded: list[tuple] = []
        real_decode = model_controller_module.decode_rl_action

        def _spy(raw_action, max_offset, should_execute_action=True):
            recorded.append((list(raw_action), max_offset, should_execute_action))
            return real_decode(raw_action, max_offset, should_execute_action)

        monkeypatch.setattr(model_controller_module, "decode_rl_action", _spy)
        raw_action = [0.3, -0.4, 0.5, 0.9, 0.7, -0.2]
        controller = ModelController(
            _FakePolicy(raw_action), _TABLE_POSITION, _MAX_OFFSET
        )
        controller.get_action(_observation(is_motion_complete=True))
        aiming = controller.get_action(_observation(is_init_state=True))

        # Act
        striking = controller.get_action(_observation(is_motion_complete=True))

        # Assert
        assert len(recorded) == 2
        assert recorded[0] == recorded[1]
        assert aiming == striking
        assert aiming is not striking


##
# 兩端輸出一致
##


class TestBothEndsProduceTheSameAction:
    @pytest.mark.parametrize(
        "raw_action",
        [
            pytest.param([0.0] * ACTION_DIM, id="動作空間中心"),
            pytest.param([0.3, -0.4, 0.5, 0.9, 0.7, -0.2], id="一般值"),
            pytest.param([1.0] * ACTION_DIM, id="全上界"),
            pytest.param([-1.0] * ACTION_DIM, id="全下界"),
            # policy 的高斯取樣本來就會溢出 tanh 值域，越界是正常情形；
            # 兩端必須以同樣方式收斂，clip 差一步就是不同的球路。
            pytest.param([9.0, -9.0, 9.0, -9.0, 2.0, 2.0], id="越界"),
            # 偏移兩維同時飽和：走圓形裁切而不是逐軸 clip 的分歧點。
            pytest.param([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], id="偏移飽和"),
        ],
    )
    @pytest.mark.parametrize("max_offset", [0.0, 0.6, 1.0])
    def test_identical_action_for_the_same_policy_output(
        self, raw_action: list[float], max_offset: float
    ):
        # Act
        demo = _demo_end_action(raw_action, max_offset)
        training = _training_end_action(raw_action, max_offset)

        # Assert：同一份函式、同一組引數，浮點也必須位元相同，不用 approx
        assert demo == training

    def test_demo_end_does_not_post_process_the_decoded_action(self):
        # 上一項若兩端都壞成同樣的樣子仍會通過。這一項另外釘住「Demo 端回傳的
        # 就是 decode 的產物本身」——狀態機不得在回傳前補刀（例如自己再夾一次
        # 速度、或把 should_execute_action 改寫成模型的第 7 維輸出）。
        # Arrange
        raw_action = [0.3, -0.4, 0.5, 0.9, 0.7, -0.2]

        # Act
        action = _demo_end_action(raw_action, _MAX_OFFSET)

        # Assert
        expected = decode_rl_action(raw_action, _MAX_OFFSET, True)
        assert action.cue_ball_placement == expected.cue_ball_placement
        assert action.shot_angle == expected.shot_angle
        assert action.cue_ball_speed == expected.cue_ball_speed
        assert action.position_offset == expected.position_offset
        assert action.should_execute_action is True


##
# 訓練端：AST 靜態檢查呼叫點
##


class TestTrainingEndCallSite:
    @pytest.fixture
    def apply_strike(self) -> ast.FunctionDef:
        return _method(
            _parse(_TRAINING_ACTION_TERM), "BilliardStrikeAction", "_apply_strike"
        )

    def test_imports_the_shared_decoder(self):
        # Assert
        assert "decode_rl_action" in _imported_from(
            _parse(_TRAINING_ACTION_TERM), "core.services.rl_action_decoder"
        )

    def test_decodes_once_with_two_positional_arguments(self, apply_strike):
        # 兩個位置引數是逐 row 的 raw action 與**逐 env** 的裁切半徑（#122）。
        # 退回 `self.cfg.max_offset` 之類的常數會讓全部 env 共用同一個半徑，
        # 而 policy 看到的條件值仍是逐 env 的——完全不報錯的錯誤。
        # Act
        calls = _calls(apply_strike, "decode_rl_action")

        # Assert
        assert len(calls) == 1
        assert [type(arg) for arg in calls[0].args] == [ast.Name, ast.Name]

    def test_should_execute_action_is_a_literal_not_a_model_output(
        self, apply_strike
    ):
        # 它是 Demo 端狀態機的控制旗標，不是第 7 維模型輸出。若哪天改成從
        # raw action 取值，兩端的語意就分岔了（訓練端恆為 True）。
        # Act
        value = _keyword(_calls(apply_strike, "decode_rl_action")[0], "should_execute_action")

        # Assert
        assert isinstance(value, ast.Constant)
        assert value.value is True

    def test_clamp_happens_before_the_physics_conversion(self, apply_strike):
        # clamp 關在 decode_rl_action() 內部、反正規化之前（順序本身由
        # test_rl_action_decoder.TestClampRunsBeforeDenormalization 釘住），
        # 所以只要 decode 排在算速度之前，訓練端就滿足「policy 輸出後、進物理前
        # 裁切」。順序對調會讓未裁切的偏移量直接變成球的自旋。
        # Act
        decode = _calls(apply_strike, "decode_rl_action")[0]
        convert = _calls(apply_strike, "compute_cue_ball_velocities")[0]

        # Assert
        assert decode.lineno < convert.lineno

    def test_the_physics_conversion_consumes_the_decoded_action(self, apply_strike):
        # 只驗行號會漏掉「decode 了但拿別的東西去算速度」。這裡順著變數名接起來。
        # Arrange
        decode = _calls(apply_strike, "decode_rl_action")[0]
        assignment = next(
            node
            for node in ast.walk(apply_strike)
            if isinstance(node, ast.Assign) and node.value is decode
        )
        target = assignment.targets[0]
        assert isinstance(target, ast.Name)

        # Act
        convert = _calls(apply_strike, "compute_cue_ball_velocities")[0]

        # Assert
        assert isinstance(convert.args[0], ast.Name)
        assert convert.args[0].id == target.id


##
# 全庫只有一份實作
##


class TestSingleImplementation:
    @pytest.mark.parametrize(
        ("function_name", "owner"),
        [
            ("decode_rl_action", _CORE_DECODER),
            ("normalize_action", _CORE_DECODER),
            ("encode_rl_observation", _CORE_ENCODER),
        ],
    )
    def test_shared_function_is_defined_exactly_once(
        self, function_name: str, owner: Path
    ):
        # Act
        definitions = [
            path
            for path in _python_files()
            if any(
                isinstance(node, ast.FunctionDef) and node.name == function_name
                for node in ast.walk(_parse(path))
            )
        ]

        # Assert
        assert _where(definitions) == _where([owner])

    def test_ball_order_constant_is_defined_only_in_core(self):
        # Act
        definitions = [
            path
            for path in _python_files()
            if any(
                isinstance(node, ast.Name)
                and node.id == "RL_BALL_ORDER"
                and isinstance(node.ctx, ast.Store)
                for node in ast.walk(_parse(path))
            )
        ]

        # Assert
        assert _where(definitions) == _where([_CORE_ENCODER])

    def test_ball_order_is_never_written_out_as_a_literal(self):
        # 常數共用擋得住的就是欄位順序錯位這一層。有人手抄 [1,…,9,0] 就等於
        # 開了第二份順序定義，之後只會有一邊跟著改。
        # Arrange
        expected = list(RL_BALL_ORDER)

        def _is_ball_order(node: ast.AST) -> bool:
            if not isinstance(node, (ast.List, ast.Tuple)):
                return False
            values = [
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, int)
                and not isinstance(element.value, bool)
            ]
            return values == expected

        # Act
        offenders = [
            path
            for path in _python_files()
            if path != _CORE_ENCODER
            and any(_is_ball_order(node) for node in ast.walk(_parse(path)))
        ]

        # Assert
        assert _where(offenders) == []

    def test_action_bounds_are_not_re_derived_outside_core(self):
        # 反正規化要用到上下限，所以「core 以外沒有人索引 ACTION_BOUNDS」就
        # 涵蓋了「core 以外沒有第二份換算」。需要換算的呼叫端請用
        # decode_rl_action() / normalize_action()，需要尺度的請用
        # ACTION_CENTER / ACTION_HALF_SPAN / ACTION_LOW / ACTION_HIGH。
        # Act
        offenders = [
            f"{path.relative_to(_REPO_ROOT)}:{node.lineno}"
            for path in _outside_core()
            for node in ast.walk(_parse(path))
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ACTION_BOUNDS"
        ]

        # Assert
        assert sorted(offenders) == []

    def test_both_ends_import_the_shared_layer(self):
        # 完成標準的「兩端皆為 import」：Demo 端兩個函式都用，訓練端 action
        # 用同一份函式、observation 用同一份欄位順序常數。
        # Act
        demo = _imported_from(
            _parse(_DEMO_CONTROLLER), "..services.rl_action_decoder"
        ) | _imported_from(_parse(_DEMO_CONTROLLER), "..services.rl_observation_encoder")
        training_action = _imported_from(
            _parse(_TRAINING_ACTION_TERM), "core.services.rl_action_decoder"
        )
        training_observation = _imported_from(
            _parse(_TRAINING_OBS_TERM), "core.services.rl_observation_encoder"
        )

        # Assert
        assert {"decode_rl_action", "encode_rl_observation"} <= demo
        assert "decode_rl_action" in training_action
        assert "RL_BALL_ORDER" in training_observation
