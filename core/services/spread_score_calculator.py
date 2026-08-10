import math

# 9-ball 標準桌台尺寸（見 core/services/break_shot_position_provider.py 的
# _FOOT = 0.635 = 1/4 桌長反推：桌長 2.54m，桌寬取半桌長 1.27m）。
TABLE_LENGTH = 2.54
TABLE_WIDTH = 1.27
_TABLE_AREA = TABLE_LENGTH * TABLE_WIDTH
_TABLE_DIAGONAL = math.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)

# --- reward 用的重新正規化（#123，2026-08-10）---------------------------------
#
# `calculate_spread_score()` 的除數是整張桌（面積 3.2258、對角線 2.8398），
# 但物理上真正可達的區間只有 0.012 ~ 0.34 —— 整個 reward 項被壓在 1/3 個單位
# 裡，而對面是 `cue_scratch = -3.5` / `nine_ball = +3.0`。policy 會收斂到
# 「不犯規、不母球落袋」然後對散開程度完全無感（#123 的實測：800 局的
# spread 全部落在 0.012~0.026）。
#
# ⚠️ 只做「減掉 rack 基準」是**無效的**：`mdp/events.py` 的 reset 完全決定性，
#    每局都套同一組 BREAK_SHOT_POSITIONS，`SPREAD_RACK` 因此是常數。減常數不
#    改變梯度，PPO 的 advantage normalization 還會直接把它吃掉。真正壓住訊號
#    的是除數，所以差分之後必須重新正規化。
#
# 開球擺位（BREAK_SHOT_POSITIONS 的 1~9 號球）本身的分數。硬寫數值而不 import
# break_shot_position_provider 是為了不讓 calculator 反向依賴 provider；
# `test_spread_score_calculator.py` 有一條對拍測試釘住這個值，擺位改了會被擋下。
SPREAD_RACK = 0.01181600734141254
# 「9 顆球均勻散滿全桌」的期望分數，作為 reward = SPREAD_REWARD_SCALE 的錨點。
# 來源：#123 的 Monte Carlo（3000 組合法隨機擺位平均 0.2516）。
#
# ⚠️ 這個數是用**不含袋口**的模擬算的。實際訓練環境會把進袋球代入袋口座標
#    （袋口在四角與長邊中點，會把凸包撐大），分布可能偏移。開跑後請用實際
#    rollout 重新量一次；偏移超過 ±20% 就以實測值取代（#123 第 6 節）。
SPREAD_REF = 0.2516
# 縮放後「散滿全桌」值多少 reward。取 2.5 的理由：物理上可達的最佳開球
# （#123 模擬的 0.342）換算後是 +3.44，**剛好不超過母球落袋的 -3.5**——
# 再好的散開也不該蓋過一次 scratch。
SPREAD_REWARD_SCALE = 2.5


def spread_score_to_reward(spread_score: float) -> float:
    """`calculate_spread_score()` 的 0~1 分數 → reward 項的實際數值。

    對應關係（`SPREAD_REWARD_SCALE = 2.5`）：

        rack（球完全沒動）      0.0118 →  0.00
        中等散開（約 1/4 桌面） 0.0775 → +0.68
        良好散開（約 1/2 桌面） 0.1420 → +1.36
        滿速開球（模擬平均）    0.2160 → +2.13
        散滿全桌（錨點）        0.2516 → +2.50
        模擬中的最佳開球        0.3420 → +3.44

    這是 reward 的定義本身，不是 `RewTerm.weight`——四個 RewTerm 的 weight 一律
    維持 1.0。權重若寫在 RewTerm 上，`core.services.reward_service.calculate_reward()`
    就不再等於 policy 實際收到的 reward，`test_mdp_rewards.py` 的
    `test_decomposition_sums_to_core_reward` 那條護欄會失去意義（它比的是未加權的
    分項和）。
    """
    return SPREAD_REWARD_SCALE * (spread_score - SPREAD_RACK) / (SPREAD_REF - SPREAD_RACK)


def calculate_spread_score(
    ball_positions: dict[int, tuple[float, float]],
    pocketed_ball_ids: set[int],
) -> float:
    """
    計算 9 顆號碼球（ball_id 1~9）的散開程度，見
    docs/phase3-task-breakdown.md 的 Reward Function 表：
    凸包面積×0.5（進袋球以袋口座標納入，維持 9 點不退化）
    + 檯面上球平均最近鄰距離×0.5（進袋球排除），各自正規化到 0.0~1.0。

    ball_positions：9 顆號碼球的 (x, y) 位置，key 須為 1~9。進袋球一樣要
    給一筆資料，value 用該球進的那個袋口座標代入（呼叫端負責決定，這個
    函式本身不知道桌面/袋口在哪）。
    pocketed_ball_ids：已進袋的 ball_id 集合，只影響最近鄰距離這一項的
    計算對象，不影響凸包面積（凸包永遠用全部 9 個點）。
    """
    if set(ball_positions.keys()) != set(range(1, 10)):
        raise ValueError(
            f"ball_positions 須包含 key 1-9，實際收到：{sorted(ball_positions.keys())}"
        )

    normalized_area = _normalized_hull_area(list(ball_positions.values()))
    normalized_distance = _normalized_avg_nearest_neighbor_distance(
        ball_positions, pocketed_ball_ids
    )
    return 0.5 * normalized_area + 0.5 * normalized_distance


def _normalized_hull_area(points: list[tuple[float, float]]) -> float:
    area = _convex_hull_area(points)
    return min(area / _TABLE_AREA, 1.0)


def _normalized_avg_nearest_neighbor_distance(
    ball_positions: dict[int, tuple[float, float]], pocketed_ball_ids: set[int]
) -> float:
    remaining = [
        pos for ball_id, pos in ball_positions.items() if ball_id not in pocketed_ball_ids
    ]
    # 少於 2 顆球就沒有「最近鄰」數學上可以定義，視為滿分（見設計討論：
    # 這種情況代表幾乎清桌，屬於極端成功的開球，不該被這項技術性缺陷扣分，
    # 且此時 Reward Function 已由進袋獎懲項主導，散開分數權重相對次要）。
    if len(remaining) < 2:
        return 1.0

    avg_distance = _average_nearest_neighbor_distance(remaining)
    return min(avg_distance / _TABLE_DIAGONAL, 1.0)


def _average_nearest_neighbor_distance(points: list[tuple[float, float]]) -> float:
    nearest_distances = [
        min(math.dist(p, q) for j, q in enumerate(points) if j != i)
        for i, p in enumerate(points)
    ]
    return sum(nearest_distances) / len(nearest_distances)


def _convex_hull_area(points: list[tuple[float, float]]) -> float:
    return _polygon_area(_convex_hull(points))


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain，回傳凸包頂點（逆時針），點數 <=2 時退化。"""
    unique_sorted_points = sorted(set(points))
    if len(unique_sorted_points) <= 2:
        return unique_sorted_points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique_sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(unique_sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    total = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0
