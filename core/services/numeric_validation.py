import math
from numbers import Real

def validate_finite_number(
    value: object,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real number")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite")

    return numeric_value

def validate_max_offset(value: float) -> float:
    numeric_value = validate_finite_number(value, "max_offset")
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError("max_offset must be in the range [0.0, 1.0]")
    return numeric_value

def validate_2d_value(
    values: list[float],
    field_name: str,
) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError(f"{field_name} elements counts is less than 2")

    return (
        validate_finite_number(values[0], f"{field_name}[0]"),
        validate_finite_number(values[1], f"{field_name}[1]"),
    )