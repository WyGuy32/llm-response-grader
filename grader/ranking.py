"""Turning ratings into a ranking of systems.

Two views, because they can disagree:

* Absolute: the mean weighted rubric score each system earned.
* Relative: head-to-head preferences, fitted with a Bradley-Terry model so
  systems that never met directly can still be compared through common
  opponents.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Comparison:
    """One judgment of `first` vs `second`: 1.0 = first won, 0.5 = tie, 0.0 = second won."""

    first: str
    second: str
    outcome: float


def bradley_terry(comparisons: Sequence[Comparison], prior: float = 0.5, iterations: int = 1000) -> dict[str, float]:
    """Fit Bradley-Terry strengths with Hunter's MM algorithm.

    P(i beats j) = s_i / (s_i + s_j). `prior` adds that many virtual wins for
    each side of every pair, a light regularizer that keeps the fit finite
    when a system won (or lost) every comparison. Strengths are normalized
    to a geometric mean of 1.
    """
    systems = sorted({c.first for c in comparisons} | {c.second for c in comparisons})
    if len(systems) < 2:
        raise ValueError("need at least two systems")
    wins: dict[str, float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    for c in comparisons:
        wins[c.first] += c.outcome
        wins[c.second] += 1 - c.outcome
        games[c.first, c.second] += 1
        games[c.second, c.first] += 1
    for i in systems:
        for j in systems:
            if i != j:
                wins[i] += prior
                games[i, j] += 2 * prior

    strength = {s: 1.0 for s in systems}
    for _ in range(iterations):
        updated = {}
        for i in systems:
            denom = sum(games[i, j] / (strength[i] + strength[j]) for j in systems if j != i)
            updated[i] = wins[i] / denom
        log_mean = sum(math.log(v) for v in updated.values()) / len(updated)
        updated = {s: v / math.exp(log_mean) for s, v in updated.items()}
        converged = max(abs(updated[s] - strength[s]) for s in systems) < 1e-12
        strength = updated
        if converged:
            break
    return strength


def win_probability(strength: dict[str, float], a: str, b: str) -> float:
    return strength[a] / (strength[a] + strength[b])


def elo_scale(strength: dict[str, float], anchor: float = 1000.0) -> dict[str, float]:
    """Express strengths on the familiar Elo scale (400 points = 10:1 odds)."""
    return {s: anchor + 400 * math.log10(v) for s, v in strength.items()}


def win_rates(comparisons: Sequence[Comparison]) -> dict[str, float]:
    """Share of comparisons each system won (ties count half)."""
    won: dict[str, float] = defaultdict(float)
    played: dict[str, int] = defaultdict(int)
    for c in comparisons:
        won[c.first] += c.outcome
        won[c.second] += 1 - c.outcome
        played[c.first] += 1
        played[c.second] += 1
    return {s: won[s] / played[s] for s in played}


def bootstrap_win_rate_ci(
    groups: Sequence[Sequence[Comparison]], samples: int = 2000, level: float = 0.95, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Percentile confidence interval for each system's win rate.

    `groups` holds the comparisons for each task. Tasks are resampled whole,
    because several raters judging the same task are not independent.
    """
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        resampled = [c for _ in groups for c in rng.choice(groups)]
        for system, rate in win_rates(resampled).items():
            draws[system].append(rate)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for system, values in draws.items():
        values.sort()
        out[system] = (values[int(lo_q * (len(values) - 1))], values[int(hi_q * (len(values) - 1))])
    return out
