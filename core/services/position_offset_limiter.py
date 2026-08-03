import math

from .numeric_validation import validate_finite_number, validate_max_offset

def clamp_position_offset(position_offset: list[float], max_offset: float) -> list[float]:
    """
    將偏移向量裁進半徑 max_offset 的圓內，保持原方向、只縮長度。
    
    position_offset: [x, y]，表示偏移向量 [-1, -1] <= position_offset <= [1, 1]
    max_offset: 最大偏移量，0.0 <= max_offset <= 1.0

    """
    if len(position_offset) != 2:
        raise ValueError("position_offset must be a list of length 2")
    
    offset_x = validate_finite_number(position_offset[0], "position_offset[0]")
    offset_y = validate_finite_number(position_offset[1], "position_offset[1]")
    limit = validate_max_offset(max_offset)
    
    norm = math.hypot(offset_x, offset_y)
    if norm <= limit:
        return [offset_x, offset_y]
    scale = limit / norm
    
    return [offset_x * scale, offset_y * scale]