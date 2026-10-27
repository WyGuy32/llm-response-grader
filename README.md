# LLM Response Grader

[![tests](https://github.com/WyGuy32/llm-response-grader/actions/workflows/tests.yml/badge.svg)](https://github.com/WyGuy32/llm-response-grader/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![core dependencies](https://img.shields.io/badge/core%20dependencies-none-brightgreen)

Tools for the human side of AI evaluation: rating model responses against a rubric, checking whether the raters actually agree, ranking models from head-to-head preferences, and testing whether an LLM judge can stand in for people.

**[Try the rating tool →](https://wyguy32.github.io/llm-response-grader/)** Grade 12 pairs of AI answers yourself, then see how your judgments compare with other raters.

## Why

Training and evaluating language models depends on human judgments: which answer is better, is this one correct, is that one safe. Those judgments are only useful if they are consistent. If two annotators can't agree on a label, a model trained on it learns noise. This project covers the full workflow:

1. **Rate.** Annotators score each response on weighted criteria (correctness, instruction following, clarity, safety and honesty) and pick the better one. The answers appear in random order with model names hidden.
2. **Audit the raters.** Krippendorff's alpha, Fleiss' kappa, and Cohen's and quadratic-weighted kappa show which criteria are reliable. Calibration flags lenient or harsh raters. An adjudication queue lists the tasks where raters disagree.
3. **Rank the models.** Preferences are fitted with a Bradley-Terry model (the method behind chatbot leaderboards), with bootstrapped confidence intervals.
4. **Test an LLM judge.** Claude grades the same tasks. Each comparison runs twice with the answers swapped, so the report can say how often the judge's verdict changed with the order and how closely it agrees with people.

## Quick start

```bash
git clone https://github.com/WyGuy32/llm-response-grader.git
cd llm-response-grader
python -m grader report data/sample          # no install needed for this
```

The output is a Markdown report. Here is part of the [full sample report](data/sample/REPORT.md):

> | Measure | Alpha | Reading |
> | --- | ---: | --- |
> | Correctness | 0.92 | reliable |
> | Instruction following | 0.79 | tentative |
> | Clarity | 0.47 | unreliable |
> | Safety and honesty | 0.83 | reliable |
> | **All criteria** | **0.83** | **reliable** |
> | Overall preference (A / B / tie) | 0.87 | reliable |
>
> | Rater | Tasks | Mean score | vs. others | Agrees with consensus | Flag |
> | --- | ---: | ---: | ---: | ---: | --- |
> | rater_1 | 12 | 3.90 | -0.15 | 92% |  |
> | rater_2 | 12 | 4.25 | +0.39 | 92% | lenient |
> | rater_3 | 11 | 3.82 | -0.18 | 91% |  |

These numbers lead to concrete actions. Raters agree on what's *correct* but not on what's *clear*, so the clarity guidelines need sharper anchors before that label is used. `rater_2` picks the same winners as everyone else but scores about 0.4 points high, so their absolute scores need recalibrating even though their preferences are fine.

### Run the LLM judge

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # or: ant auth login
python -m grader judge data/sample  # writes data/sample/judgments.jsonl
python -m grader report data/sample # now includes "LLM judge vs. humans"
```

The judge uses Claude (`claude-opus-5-5`) with structured outputs, so every verdict is schema-valid JSON, and server-side refusal fallbacks are turned on. The report then adds how often the judge agrees with the human majority, its Cohen's kappa against that majority, how often its verdict survives swapping the answer order, and how far its rubric scores sit from the human mean on each criterion.

## Sample data

[`data/sample/`](data/sample) holds 12 tasks written to test the failure modes annotators are trained to catch:

| Task | What separates the two answers |
| --- | --- |
| Pens at 3 for $2 | Rounding error gives a confident wrong total |
| Palindrome function | One ignores the "ignore case and punctuation" requirement |
| Why seasons happen | Common misconception (distance from the Sun) |
| APR vs. APY | Both correct; one is padded. Raters split on this, as they should |
| Bleach + ammonia exposure | One minimizes a real emergency |
| One-sentence summary | One invents a statistic that isn't in the source |
| Two-sentence email | One ignores the length limit |
| Syllogism | Invalid inference stated with confidence |
| Cap rate | Wrong formula (subtracts mortgage payments) |
| Top-3 SQL | Both work; they handle ties differently |
| Fictional court case | One admits it can't find the case; the other invents a ruling |
| Three primes, no other text | Extra text, and 57 isn't prime |

**The sample ratings are simulated**, so the repo runs without real annotators. [`scripts/simulate_raters.py`](scripts/simulate_raters.py) builds three raters from reference scores with set noise and bias (one careful, one lenient, one noisy who skipped a task). The report is then checked on whether it catches the problems that were planted. To use real data, replace `ratings.jsonl` with your annotators' output, or rate in the browser and export.

## Data format

```
rubric.json      criteria with weights, descriptions and score anchors
tasks.json       [{id, category, prompt, responses: [{system, text}, {system, text}]}]
ratings.jsonl    {"task_id", "rater", "scores": {"A": {criterion: 1-5}, "B": {...}}, "preference": "A"|"B"|"tie"}
judgments.jsonl  written by `grader judge`
```

Ratings are validated on load: scores must be on the rubric's scale, task ids must exist, and each rater may rate a task only once.

## Statistics, and how they're tested

Everything is implemented from the definitions in plain Python, and the tests check the results against published worked examples:

| Method | Used for | Test |
| --- | --- | --- |
| Krippendorff's alpha (nominal, interval) | Headline reliability; handles missing ratings | Krippendorff's 4-coder example: **0.743** nominal, **0.849** interval |
| Fleiss' kappa | Preference agreement across all raters | Fleiss (1971) example: **0.210** |
| Cohen's kappa | Rater pairs | Textbook 2×2 table: **0.40** |
| Quadratic weighted kappa | Ordinal scores; near misses earn partial credit | Equals Cohen's kappa with two categories |
| Bradley-Terry (MM algorithm) | Ranking from pairwise preferences | Recovers exact 3:1 odds; ranks transitively through a shared opponent |
| Task-level bootstrap | Win-rate confidence intervals | Resamples whole tasks, because ratings of one task aren't independent |
| Swapped-order judging | LLM-judge position bias | Fake client that always picks the first answer is flagged |

```bash
python -m unittest discover -s tests -t .
```

The judge tests use a fake client, so CI needs no API key.

## Layout

```
grader/
  data.py        loading and validation
  agreement.py   kappa and alpha statistics
  ranking.py     Bradley-Terry, win rates, bootstrap
  judge.py       LLM-as-judge with Claude, run in both answer orders
  report.py      Markdown report
  __main__.py    command line: report | judge | export-web
docs/            browser rating tool (GitHub Pages)
data/sample/     rubric, tasks, simulated ratings, generated report
scripts/         rater simulation
tests/
```

## License

MIT
