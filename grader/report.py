"""Build the evaluation report (Markdown) from a dataset."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from itertools import combinations

from .agreement import cohen_kappa, fleiss_kappa, krippendorff_alpha, percent_agreement, weighted_kappa
from .data import LABELS, PREFERENCES, Dataset
from .ranking import Comparison, bootstrap_win_rate_ci, bradley_terry, elo_scale, win_rates

OUTCOME = {"A": 1.0, "tie": 0.5, "B": 0.0}


def md_table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def reliability(alpha: float) -> str:
    """Krippendorff's own guidance for reading alpha."""
    if alpha >= 0.8:
        return "reliable"
    if alpha >= 0.667:
        return "tentative"
    return "unreliable"


def majority(prefs: list[str]) -> str:
    """Most common preference; a split vote with no single winner counts as a tie."""
    counts = Counter(prefs).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "tie"
    return counts[0][0]


def human_comparisons(ds: Dataset) -> list[list[Comparison]]:
    by_task = defaultdict(list)
    tasks = ds.task_by_id
    for r in ds.ratings:
        t = tasks[r.task_id]
        by_task[r.task_id].append(Comparison(t.system("A"), t.system("B"), OUTCOME[r.preference]))
    return list(by_task.values())


def score_units(ds: Dataset, criterion: str | None = None) -> list[list[float | None]]:
    """One unit per (task, response[, criterion]) with each rater's score or None."""
    raters = ds.raters
    lookup = {(r.task_id, r.rater): r for r in ds.ratings}
    criteria = [criterion] if criterion else [c.id for c in ds.rubric.criteria]
    units = []
    for t in ds.tasks:
        for label in LABELS:
            for c in criteria:
                units.append([lookup[t.id, rt].scores[label][c] if (t.id, rt) in lookup else None for rt in raters])
    return units


def build_report(ds: Dataset) -> str:
    rubric, raters, tasks = ds.rubric, ds.raters, ds.task_by_id
    lookup = {(r.task_id, r.rater): r for r in ds.ratings}
    out = [
        "# Evaluation report",
        "",
        f"**Rubric:** {rubric.name} · **Tasks:** {len(ds.tasks)} · **Raters:** {len(raters)} · "
        f"**Ratings:** {len(ds.ratings)}",
        "",
    ]

    # ---- leaderboard ----
    groups = human_comparisons(ds)
    flat = [c for g in groups for c in g]
    strength = bradley_terry(flat)
    elo = elo_scale(strength)
    rates = win_rates(flat)
    ci = bootstrap_win_rate_ci(groups)
    system_scores = defaultdict(list)
    for r in ds.ratings:
        for label in LABELS:
            system_scores[tasks[r.task_id].system(label)].append(rubric.weighted(r.scores[label]))
    games = Counter(s for c in flat for s in (c.first, c.second))
    order = sorted(strength, key=strength.get, reverse=True)
    out += [
        "## Leaderboard",
        "",
        "Human preferences fitted with a Bradley-Terry model (shown on an Elo scale), plus the mean "
        "weighted rubric score. Win-rate intervals come from resampling whole tasks.",
        "",
        md_table(
            ["System", "Elo", "Win rate", "95% CI", "Mean rubric score", "Judgments"],
            [
                [
                    f"**{s}**",
                    f"{elo[s]:.0f}",
                    f"{rates[s]:.0%}",
                    f"{ci[s][0]:.0%} to {ci[s][1]:.0%}",
                    f"{statistics.mean(system_scores[s]):.2f} / {rubric.scale[1]}",
                    str(games[s]),
                ]
                for s in order
            ],
        ),
        "",
    ]
    if len(order) >= 2:
        best, second = order[0], order[1]
        p = strength[best] / (strength[best] + strength[second])
        out += [f"The model gives **{best}** a {p:.0%} chance of being preferred over **{second}** on a new task.", ""]

    # ---- agreement ----
    alpha_rows = []
    for c in rubric.criteria:
        a = krippendorff_alpha(score_units(ds, c.id), "interval")
        alpha_rows.append([c.name, f"{a:.2f}", reliability(a)])
    overall = krippendorff_alpha(score_units(ds), "interval")
    alpha_rows.append(["**All criteria**", f"**{overall:.2f}**", f"**{reliability(overall)}**"])
    pref_units = [[lookup[t.id, rt].preference if (t.id, rt) in lookup else None for rt in raters] for t in ds.tasks]
    pref_alpha = krippendorff_alpha(pref_units, "nominal")
    alpha_rows.append(["Overall preference (A / B / tie)", f"{pref_alpha:.2f}", reliability(pref_alpha)])

    out += [
        "## Inter-rater agreement",
        "",
        "Krippendorff's alpha handles any number of raters and missing ratings. Read it as "
        "≥ 0.80 reliable, 0.667 to 0.80 tentative, below 0.667 unreliable.",
        "",
        md_table(["Measure", "Alpha", "Reading"], alpha_rows, "lrl"),
        "",
    ]

    complete = [t for t in ds.tasks if all((t.id, rt) in lookup for rt in raters)]
    if len(raters) >= 2 and complete:
        counts = [[sum(lookup[t.id, rt].preference == p for rt in raters) for p in PREFERENCES] for t in complete]
        out += [
            f"Fleiss' kappa on preferences, over the {len(complete)} tasks every rater completed: "
            f"**{fleiss_kappa(counts):.2f}**.",
            "",
        ]

    pair_rows = []
    lo, hi = rubric.scale
    for r1, r2 in combinations(raters, 2):
        shared = [t.id for t in ds.tasks if (t.id, r1) in lookup and (t.id, r2) in lookup]
        if not shared:
            continue
        p1 = [lookup[t, r1].preference for t in shared]
        p2 = [lookup[t, r2].preference for t in shared]
        s1 = [lookup[t, r1].scores[lb][c.id] for t in shared for lb in LABELS for c in rubric.criteria]
        s2 = [lookup[t, r2].scores[lb][c.id] for t in shared for lb in LABELS for c in rubric.criteria]
        pair_rows.append(
            [
                f"{r1} · {r2}",
                str(len(shared)),
                f"{percent_agreement(p1, p2):.0%}",
                f"{cohen_kappa(p1, p2):.2f}",
                f"{percent_agreement(s1, s2):.0%}",
                f"{weighted_kappa(s1, s2, list(range(lo, hi + 1))):.2f}",
            ]
        )
    if pair_rows:
        out += [
            "### Rater pairs",
            "",
            md_table(
                ["Pair", "Shared tasks", "Pref. agree", "Cohen κ", "Exact score agree", "Quadratic κ"],
                pair_rows,
            ),
            "",
        ]

    # ---- calibration ----
    cal_rows = []
    for rt in raters:
        mine = [r for r in ds.ratings if r.rater == rt]
        mean = statistics.mean(s for r in mine for lb in LABELS for s in r.scores[lb].values())
        # Bias is measured against the other raters on the same tasks, so it isn't diluted by the
        # rater's own scores or skewed by which tasks they happened to rate.
        mine_ids = {r.task_id for r in mine}
        others_mean = statistics.mean(
            s for r in ds.ratings if r.rater != rt and r.task_id in mine_ids for lb in LABELS for s in r.scores[lb].values()
        )
        # Agreement with the other raters' majority, so a rater isn't graded against themself.
        agree = []
        for r in mine:
            others = [lookup[r.task_id, o].preference for o in raters if o != rt and (r.task_id, o) in lookup]
            if others:
                agree.append(r.preference == majority(others))
        bias = mean - others_mean
        flag = "lenient" if bias > 0.3 else "harsh" if bias < -0.3 else ""
        cal_rows.append(
            [rt, str(len(mine)), f"{mean:.2f}", f"{bias:+.2f}", f"{statistics.mean(agree):.0%}" if agree else "n/a", flag]
        )
    out += [
        "## Rater calibration",
        "",
        "A rater whose average sits well above or below the others on the same tasks is miscalibrated, even when "
        "their rankings are sensible. Consensus agreement compares each rater with the *other* raters' majority.",
        "",
        md_table(["Rater", "Tasks", "Mean score", "vs. others", "Agrees with consensus", "Flag"], cal_rows, "lrrrrl"),
        "",
    ]

    # ---- adjudication queue ----
    queue = []
    for t in ds.tasks:
        rs = [lookup[t.id, rt] for rt in raters if (t.id, rt) in lookup]
        if len(rs) < 2:
            continue
        prefs = [r.preference for r in rs]
        spread = statistics.mean(
            statistics.pstdev([r.scores[lb][c.id] for r in rs]) for lb in LABELS for c in rubric.criteria
        )
        split = len(set(prefs)) > 1
        if split or spread >= 0.6:
            queue.append((split, spread, t, prefs))
    queue.sort(key=lambda q: (not q[0], -q[1]))
    if queue:
        out += [
            "## Needs adjudication",
            "",
            "Tasks where raters split on the preferred response or their scores spread widely. A lead reviewer "
            "should settle these before the data is used for training.",
            "",
            md_table(
                ["Task", "Category", "Preferences", "Score spread (σ)"],
                [[t.id, t.category, " / ".join(prefs), f"{spread:.2f}"] for _, spread, t, prefs in queue],
                "llll",
            ),
            "",
        ]

    # ---- judge ----
    pairwise = {j["task_id"]: j for j in ds.judgments if j.get("kind") == "pairwise"}
    if pairwise:
        human = {t.id: majority([lookup[t.id, rt].preference for rt in raters if (t.id, rt) in lookup]) for t in ds.tasks}
        shared = [tid for tid in pairwise if tid in human]
        jv = [pairwise[t]["verdict"] for t in shared]
        hv = [human[t] for t in shared]
        consistent = sum(pairwise[t]["position_consistent"] for t in shared)
        out += [
            "## LLM judge vs. humans",
            "",
            md_table(
                ["Measure", "Value"],
                [
                    ["Tasks judged", str(len(shared))],
                    ["Agrees with human majority", f"{percent_agreement(jv, hv):.0%}"],
                    ["Cohen κ vs. human majority", f"{cohen_kappa(jv, hv):.2f}"],
                    ["Same verdict when answer order is swapped", f"{consistent / len(shared):.0%}"],
                ],
            ),
            "",
        ]
        rubric_j = [j for j in ds.judgments if j.get("kind") == "rubric"]
        if rubric_j:
            errors = defaultdict(list)
            for j in rubric_j:
                rs = [lookup[j["task_id"], rt] for rt in raters if (j["task_id"], rt) in lookup]
                for c in rubric.criteria:
                    human_mean = statistics.mean(r.scores[j["label"]][c.id] for r in rs)
                    errors[c.name].append(j["scores"][c.id] - human_mean)
            out += [
                md_table(
                    ["Criterion", "Judge − human mean", "Mean absolute gap"],
                    [[k, f"{statistics.mean(v):+.2f}", f"{statistics.mean(abs(x) for x in v):.2f}"] for k, v in errors.items()],
                ),
                "",
            ]
        disagreements = [t for t in shared if pairwise[t]["verdict"] != human[t]]
        if disagreements:
            out += ["Judge and humans disagree on: " + ", ".join(disagreements) + ".", ""]
    else:
        out += [
            "## LLM judge vs. humans",
            "",
            "No judge results yet. Run `python -m grader judge <dataset>` to have Claude grade the same tasks; "
            "this section will then compare its verdicts with the human raters.",
            "",
        ]

    return "\n".join(out).rstrip() + "\n"
