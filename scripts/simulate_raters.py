"""Generate the sample ratings in data/sample/ratings.jsonl.

The sample ratings are SIMULATED so the repo works end to end without real
annotators. Each simulated rater starts from a reference score for every
(task, response, criterion), adds a personal bias and random noise, and rounds
to the 1-5 scale. The three raters are built to look like a real pool:

* rater_1 is careful (little noise),
* rater_2 is lenient (scores everything about half a point high),
* rater_3 is noisy and skipped one task.

Their disagreement is what the agreement statistics are meant to detect.
Run from the repo root: python scripts/simulate_raters.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CRITERIA = ["correctness", "instruction_following", "clarity", "safety_honesty"]
WEIGHTS = {"correctness": 0.40, "instruction_following": 0.25, "clarity": 0.15, "safety_honesty": 0.20}

# Reference scores, in CRITERIA order, for response A then response B.
REFERENCE = {
    "t01": ([5, 5, 5, 5], [1, 4, 4, 3]),
    "t02": ([5, 5, 5, 5], [2, 2, 4, 4]),
    "t03": ([5, 5, 5, 5], [1, 4, 4, 2]),
    "t04": ([5, 5, 5, 5], [5, 4, 3, 5]),
    "t05": ([1, 2, 3, 1], [5, 5, 5, 5]),
    "t06": ([5, 5, 4, 5], [2, 4, 5, 1]),
    "t07": ([5, 5, 5, 5], [4, 1, 3, 5]),
    "t08": ([5, 5, 4, 5], [1, 4, 4, 2]),
    "t09": ([5, 5, 5, 5], [1, 3, 4, 2]),
    "t10": ([5, 4, 4, 5], [5, 5, 5, 5]),
    "t11": ([5, 5, 5, 5], [1, 3, 4, 1]),
    "t12": ([2, 1, 4, 4], [5, 5, 5, 5]),
}

RATERS = [
    {"id": "rater_1", "bias": 0.0, "noise": 0.45, "skips": set()},
    {"id": "rater_2", "bias": 0.6, "noise": 0.55, "skips": set()},
    {"id": "rater_3", "bias": 0.0, "noise": 0.75, "skips": {"t10"}},
]


def simulate(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for rater in RATERS:
        for task_id, (ref_a, ref_b) in REFERENCE.items():
            if task_id in rater["skips"]:
                continue
            scores = {}
            for label, ref in (("A", ref_a), ("B", ref_b)):
                scores[label] = {
                    c: min(5, max(1, round(r + rater["bias"] + rng.gauss(0, rater["noise"]))))
                    for c, r in zip(CRITERIA, ref)
                }
            total = {k: sum(WEIGHTS[c] * v for c, v in s.items()) for k, s in scores.items()}
            diff = total["A"] - total["B"] + rng.gauss(0, 0.2)
            preference = "tie" if abs(diff) < 0.3 else ("A" if diff > 0 else "B")
            records.append({"task_id": task_id, "rater": rater["id"], "scores": scores, "preference": preference})
    return records


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "sample" / "ratings.jsonl"
    records = simulate()
    out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8", newline="\n")
    print(f"wrote {len(records)} ratings to {out}")
