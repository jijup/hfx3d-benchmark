"""Unified scoring primitives for the simple refinement method.

Conventions
-----------
* Every score is in [0, 1].
* 0.5 is the neutral / unknown value. Missing or unreliable evidence returns 0.5
  and must never be interpreted as strong negative evidence.
* Distance-based scores are normalised by local point spacing (scale free).
* Scale-dependent features use percentile-rank normalisation inside the same
  building and semantic class; a class with a single instance returns neutral.

Fusion
------
Single, parameter-free odds fusion used by every condition:

    positive = P * G * C
    negative = (1-P) * (1-G) * (1-C)
    conf     = positive / (positive + negative)

With G = C = 0.5 this reduces exactly to conf = P, which is how
``condition: always`` (pure semantic prior) is handled without a special path.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

NEUTRAL = 0.5


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def odds_fusion(prior: float, geometry: float, context: float, odds_epsilon: float) -> float:
    """Parameter-free fusion of P, G, C using the odds-ratio form."""
    prior = clip01(prior)
    geometry = clip01(geometry)
    context = clip01(context)
    positive = prior * geometry * context
    negative = (1.0 - prior) * (1.0 - geometry) * (1.0 - context)
    denominator = positive + negative
    if denominator > 0.0:
        # Keep the paper equation exact whenever it is mathematically defined.
        # Flooring a small, valid denominator changes conflicting near-0/1
        # evidence from an approximately neutral result into a near-zero one.
        return clip01(positive / denominator)

    # Exact contradictory endpoints can produce 0/0. Only this undefined
    # boundary case uses epsilon, via the equivalent sum of clipped logits.
    epsilon = float(odds_epsilon)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("odds_epsilon must be in (0, 0.5)")
    values = [min(1.0 - epsilon, max(epsilon, value)) for value in (prior, geometry, context)]
    log_odds = sum(math.log(value) - math.log1p(-value) for value in values)
    if log_odds >= 0.0:
        return clip01(1.0 / (1.0 + math.exp(-log_odds)))
    exponential = math.exp(log_odds)
    return clip01(exponential / (1.0 + exponential))


def geometric_mean(scores: Sequence[float]) -> float:
    """All evidence must hold simultaneously. No weights, no magic constants."""
    scores = [clip01(s) for s in scores]
    if not scores:
        return NEUTRAL
    product = 1.0
    for score in scores:
        product *= score
    return float(product ** (1.0 / len(scores)))


def max_merge(scores: Sequence[float]) -> float:
    """Alternative physical mechanisms; the strongest one wins."""
    scores = [clip01(s) for s in scores]
    if not scores:
        return NEUTRAL
    return float(max(scores))


def availability_aware_max_merge(scores: Sequence[tuple[float, bool]]) -> tuple[float, bool]:
    """Merge alternative mechanisms without treating unknown as a score floor.

    Each item is ``(score, informative)``.  Unavailable branches are excluded;
    if every branch is unavailable the result is neutral and uninformative.
    """
    measured = [clip01(score) for score, informative in scores if informative]
    if not measured:
        return NEUTRAL, False
    return float(max(measured)), True


def availability_aware_geometric_mean(scores: Sequence[tuple[float, bool]]) -> tuple[float, bool]:
    """Conjunctive merge over measured branches only; unknown stays neutral."""
    measured = [clip01(score) for score, informative in scores if informative]
    if not measured:
        return NEUTRAL, False
    return geometric_mean(measured), True


def proximity(distance: float, spacing: float, epsilon: float) -> float:
    """Continuous proximity: d=0 -> 1, d~spacing -> 0.5, d large -> ~0."""
    spacing = max(float(spacing), epsilon)
    return clip01(spacing / (spacing + max(float(distance), 0.0)))


def vertical_gap_score(gap: float, spacing: float, epsilon: float) -> float:
    """How tightly two instances are stacked vertically (gap >= 0)."""
    return proximity(max(float(gap), 0.0), spacing, epsilon)


def overlap_min_extent(
    src_min: np.ndarray,
    src_max: np.ndarray,
    tgt_min: np.ndarray,
    tgt_max: np.ndarray,
    axes: Sequence[int],
    epsilon: float,
) -> float:
    """Overlap normalized by the smaller extent (alignment on that projection)."""
    values = []
    for axis in axes:
        intersection = max(0.0, min(src_max[axis], tgt_max[axis]) - max(src_min[axis], tgt_min[axis]))
        denominator = min(src_max[axis] - src_min[axis], tgt_max[axis] - tgt_min[axis])
        values.append(clip01(intersection / max(denominator, epsilon)))
    return float(math.prod(values)) if values else NEUTRAL


def rank_01(value: float, values: Sequence[float]) -> float:
    """Percentile rank in [0,1]; the median maps to ~0.5 (neutral).

    Diagnostic note: this is a RELATIVE score inside one building and class.
    A high rank does not by itself prove that the absolute geometry supports
    the function (e.g. a building whose windows are all flat still has a
    top-ranked window). Always report the raw measured value alongside the
    rank. A single instance or all-equal values return neutral 0.5 (no
    ordering information).
    """
    values = [float(v) for v in values]
    if len(values) < 2:
        return NEUTRAL
    less = sum(1 for v in values if v < float(value))
    equal = sum(1 for v in values if v == float(value))
    return clip01((less + 0.5 * equal) / len(values))


def building_scale_normalize(value: float, scale: float, epsilon: float) -> float:
    """Continuous ramp value/scale clipped to [0,1] (scale-free against building size)."""
    return clip01(float(value) / max(float(scale), epsilon))


def angle_inverse_ramp(slope_degrees: float, full_scale_degrees: float) -> float:
    """Low slope -> high walkable score: 0° -> 1, full_scale -> 0."""
    return clip01(1.0 - max(float(slope_degrees), 0.0) / max(float(full_scale_degrees), 1e-9))


def angle_ramp(slope_degrees: float, full_scale_degrees: float) -> float:
    """Slope magnitude as positive evidence: 0° -> 0, full_scale -> 1."""
    return clip01(max(float(slope_degrees), 0.0) / max(float(full_scale_degrees), 1e-9))
