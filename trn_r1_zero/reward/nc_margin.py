"""Reward function variant that scales node-classification rewards by hardness scores.

This module wraps the existing nc reward with a multiplicative hardness weight.
The score is expected to be pre-computed (e.g., margin gain) and passed via
`extra_info["hardness_score"]`. A `hardness_scale` parameter allows tuning the
impact of the hardness weight; it defaults to 10.0.
"""
from __future__ import annotations

from typing import Any, Dict

if __package__ is None or __package__ == "":  # pragma: no cover - allow running as script
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from .nc import compute_score as base_compute_score
except ImportError:  # pragma: no cover - fallback when run outside package context
    from trn_r1_zero.reward.nc import compute_score as base_compute_score


def _extract_hardness(extra_info: Dict[str, Any] | None) -> float | None:
    if not extra_info:
        return None
    if "hardness_score" in extra_info:
        value = extra_info.get("hardness_score")
    else:
        value = extra_info.get("margin_gain")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    format_score: float = 0.1,
    score: float = 1.0,
    hardness_scale: float = 10.0,
):
    """Compute reward and apply hardness-based scaling to the final score."""

    base_result = base_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        format_score=format_score,
        score=score,
    )
    base_total = base_result.get("score", 0.0)
    hardness = _extract_hardness(extra_info)

    if hardness is None:
        print(
            "[nc_margin] Missing hardness_score in extra_info:",
            extra_info,
        )
        raise ValueError(
            "nc_margin.compute_score requires 'hardness_score' in extra_info; received sample without it."
        )

    multiplier = hardness * hardness_scale
    hardness_weight = hardness
    effective_scale = hardness_scale

    result = dict(base_result)
    result["base_score"] = base_total
    result["hardness_weight"] = hardness_weight
    result["hardness_scale"] = effective_scale
    result["score"] = base_total * multiplier
    return result
