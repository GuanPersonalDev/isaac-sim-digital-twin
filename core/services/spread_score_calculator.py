import math

# 9-ball 標準桌台尺寸（見 core/services/break_shot_position_provider.py 的
# _FOOT = 0.635 = 1/4 桌長反推：桌長 2.54m，桌寬取半桌長 1.27m）。
TABLE_LENGTH = 2.54
TABLE_WIDTH = 1.27
_TABLE_AREA = TABLE_LENGTH * TABLE_WIDTH
_TABLE_DIAGONAL = math.sqrt(TABLE_LENGTH**2 + TABLE_WIDTH**2)


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
