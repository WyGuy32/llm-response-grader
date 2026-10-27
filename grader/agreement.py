"""Inter-rater agreement statistics, implemented from their definitions.

* Cohen's kappa: two raters, categorical labels.
* Weighted kappa: two raters, ordinal labels; near misses count partly.
* Fleiss' kappa: any fixed number of raters per item, categorical labels.
* Krippendorff's alpha: any number of raters, missing ratings allowed, with
  nominal, ordinal-as-interval, or interval distance. It is the most general of
  the four and the headline number in the report.

Each statistic is 1 for perfect agreement, 0 for chance-level agreement, and
negative when raters disagree more than chance would predict.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from typing import Hashable, Sequence


def percent_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    po = percent_agreement(a, b)
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / n**2
    if pe == 1:
        return 1.0  # both raters used one identical label throughout
    return (po - pe) / (1 - pe)


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int], power: int = 2) -> float:
    """Weighted kappa; power=2 is the quadratic weighting used for ordinal scales."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1 / n
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    span = (k - 1) ** power or 1
    weight = lambda i, j: abs(i - j) ** power / span  # noqa: E731  (disagreement weight)
    obs = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    exp = sum(weight(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    if exp == 0:
        return 1.0
    return 1 - obs / exp


def fleiss_kappa(counts: Sequence[Sequence[int]]) -> float:
    """`counts[i][j]` = number of raters who put item i in category j.

    Every item must have the same number of raters.
    """
    if not counts:
        raise ValueError("no items")
    n = sum(counts[0])
    if n < 2 or any(sum(row) != n for row in counts):
        raise ValueError("every item needs the same number (>= 2) of raters")
    items = len(counts)
    p_j = [sum(row[j] for row in counts) / (items * n) for j in range(len(counts[0]))]
    p_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts]
    p_bar = sum(p_i) / items
    pe = sum(p * p for p in p_j)
    if pe == 1:
        return 1.0
    return (p_bar - pe) / (1 - pe)


def krippendorff_alpha(units: Sequence[Sequence[float | Hashable | None]], level: str = "interval") -> float:
    """Krippendorff's alpha.

    `units` is one list per rated item, holding each rater's value (None when
    that rater skipped the item). `level` is "nominal" or "interval".
    Items with fewer than two values carry no information and are ignored.
    """
    if level == "nominal":
        delta = lambda x, y: 0.0 if x == y else 1.0  # noqa: E731
    elif level == "interval":
        delta = lambda x, y: (x - y) ** 2  # noqa: E731
    else:
        raise ValueError("level must be 'nominal' or 'interval'")

    pairable = [[v for v in unit if v is not None] for unit in units]
    pairable = [u for u in pairable if len(u) >= 2]
    n = sum(len(u) for u in pairable)
    if n < 2:
        raise ValueError("need at least one item rated twice")

    d_observed = sum(sum(delta(x, y) for x, y in permutations(u, 2)) / (len(u) - 1) for u in pairable) / n

    values = Counter(v for u in pairable for v in u)
    d_expected = sum(nc * nk * delta(c, k) for c, nc in values.items() for k, nk in values.items() if c != k)
    d_expected /= n * (n - 1)
    if d_expected == 0:
        return 1.0
    return 1 - d_observed / d_expected
