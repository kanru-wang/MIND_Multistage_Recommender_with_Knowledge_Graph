from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any


def float_slug(value: float) -> str:
    """Return a stable, filesystem-safe slug for a float."""
    mantissa, exponent_text = f"{float(value):.12e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    exponent = int(exponent_text)
    exponent_sign = "m" if exponent < 0 else ""
    return f"{mantissa}e{exponent_sign}{abs(exponent):02d}"


def best_result(
    results: list[dict[str, Any]],
    metric_path: Sequence[str],
    *,
    eligible_statuses: Collection[str] | None = None,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_value = float("-inf")
    for result in results:
        if eligible_statuses is not None and result.get("status") not in eligible_statuses:
            continue
        current: Any = result
        for key in metric_path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is None:
            continue
        value = float(current)
        if value > best_value:
            best = result
            best_value = value
    return best
