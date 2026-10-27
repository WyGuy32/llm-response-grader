"""Loading and validating tasks, rubrics, human ratings and judge output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LABELS = ("A", "B")
PREFERENCES = ("A", "B", "tie")


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    weight: float
    description: str
    anchors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Rubric:
    name: str
    scale: tuple[int, int]
    criteria: list[Criterion]

    def weighted(self, scores: dict[str, float]) -> float:
        """Weighted average on the rubric's own scale."""
        total_weight = sum(c.weight for c in self.criteria)
        return sum(c.weight * scores[c.id] for c in self.criteria) / total_weight


@dataclass(frozen=True)
class Response:
    system: str
    text: str


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    responses: tuple[Response, Response]  # shown to raters as A, B

    def system(self, label: str) -> str:
        return self.responses[LABELS.index(label)].system


@dataclass(frozen=True)
class Rating:
    task_id: str
    rater: str
    scores: dict[str, dict[str, int]]  # label -> criterion -> score
    preference: str  # "A", "B" or "tie"


@dataclass(frozen=True)
class Dataset:
    rubric: Rubric
    tasks: list[Task]
    ratings: list[Rating]
    judgments: list[dict]

    @property
    def task_by_id(self) -> dict[str, Task]:
        return {t.id: t for t in self.tasks}

    @property
    def raters(self) -> list[str]:
        return sorted({r.rater for r in self.ratings})


class DataError(ValueError):
    pass


def load_rubric(path: Path) -> Rubric:
    raw = json.loads(path.read_text(encoding="utf-8"))
    criteria = [Criterion(**c) for c in raw["criteria"]]
    if not criteria:
        raise DataError("rubric has no criteria")
    if len({c.id for c in criteria}) != len(criteria):
        raise DataError("rubric criterion ids must be unique")
    return Rubric(name=raw["name"], scale=tuple(raw["scale"]), criteria=criteria)


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        if len(raw["responses"]) != 2:
            raise DataError(f"task {raw['id']}: exactly two responses are required for pairwise rating")
        tasks.append(
            Task(
                id=raw["id"],
                category=raw.get("category", ""),
                prompt=raw["prompt"],
                responses=tuple(Response(**r) for r in raw["responses"]),
            )
        )
    return tasks


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataError(f"{path.name} line {n}: {e}") from e
    return rows


def validate_rating(rating: Rating, rubric: Rubric, task_ids: set[str]) -> None:
    lo, hi = rubric.scale
    where = f"rating by {rating.rater} on {rating.task_id}"
    if rating.task_id not in task_ids:
        raise DataError(f"{where}: unknown task")
    if rating.preference not in PREFERENCES:
        raise DataError(f"{where}: preference must be one of {PREFERENCES}")
    for label in LABELS:
        for c in rubric.criteria:
            score = rating.scores.get(label, {}).get(c.id)
            if not isinstance(score, int) or not lo <= score <= hi:
                raise DataError(f"{where}: {label}.{c.id} must be an integer from {lo} to {hi}")


def load_dataset(directory: Path) -> Dataset:
    """Read rubric.json, tasks.json, ratings.jsonl and (optionally) judgments.jsonl."""
    rubric = load_rubric(directory / "rubric.json")
    tasks = load_tasks(directory / "tasks.json")
    task_ids = {t.id for t in tasks}
    ratings = [Rating(**row) for row in _read_jsonl(directory / "ratings.jsonl")]
    seen = set()
    for r in ratings:
        validate_rating(r, rubric, task_ids)
        if (r.task_id, r.rater) in seen:
            raise DataError(f"duplicate rating by {r.rater} on {r.task_id}")
        seen.add((r.task_id, r.rater))
    return Dataset(rubric, tasks, ratings, _read_jsonl(directory / "judgments.jsonl"))
