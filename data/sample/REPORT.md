# Evaluation report

**Rubric:** General assistant quality v1 · **Tasks:** 12 · **Raters:** 3 · **Ratings:** 35

## Leaderboard

Human preferences fitted with a Bradley-Terry model (shown on an Elo scale), plus the mean weighted rubric score. Win-rate intervals come from resampling whole tasks.

| System | Elo | Win rate | 95% CI | Mean rubric score | Judgments |
| --- | ---: | ---: | ---: | ---: | ---: |
| **model-a** | 1327 | 98% | 93% to 100% | 4.84 / 5 | 24 |
| **model-b** | 883 | 33% | 6% to 66% | 3.71 / 5 | 23 |
| **model-c** | 790 | 17% | 0% to 44% | 3.07 / 5 | 23 |

The model gives **model-a** a 93% chance of being preferred over **model-b** on a new task.

## Inter-rater agreement

Krippendorff's alpha handles any number of raters and missing ratings. Read it as ≥ 0.80 reliable, 0.667 to 0.80 tentative, below 0.667 unreliable.

| Measure | Alpha | Reading |
| --- | ---: | --- |
| Correctness | 0.92 | reliable |
| Instruction following | 0.79 | tentative |
| Clarity | 0.47 | unreliable |
| Safety and honesty | 0.83 | reliable |
| **All criteria** | **0.83** | **reliable** |
| Overall preference (A / B / tie) | 0.87 | reliable |

Fleiss' kappa on preferences, over the 11 tasks every rater completed: **0.82**.

### Rater pairs

| Pair | Shared tasks | Pref. agree | Cohen κ | Exact score agree | Quadratic κ |
| --- | ---: | ---: | ---: | ---: | ---: |
| rater_1 · rater_2 | 12 | 92% | 0.82 | 64% | 0.87 |
| rater_1 · rater_3 | 11 | 100% | 1.00 | 53% | 0.84 |
| rater_2 · rater_3 | 11 | 91% | 0.76 | 53% | 0.78 |

## Rater calibration

A rater whose average sits well above or below the others on the same tasks is miscalibrated, even when their rankings are sensible. Consensus agreement compares each rater with the *other* raters' majority.

| Rater | Tasks | Mean score | vs. others | Agrees with consensus | Flag |
| --- | ---: | ---: | ---: | ---: | --- |
| rater_1 | 12 | 3.90 | -0.15 | 92% |  |
| rater_2 | 12 | 4.25 | +0.39 | 92% | lenient |
| rater_3 | 11 | 3.82 | -0.18 | 91% |  |

## Needs adjudication

Tasks where raters split on the preferred response or their scores spread widely. A lead reviewer should settle these before the data is used for training.

| Task | Category | Preferences | Score spread (σ) |
| --- | --- | --- | --- |
| t04 | finance | A / tie / A | 0.35 |

## LLM judge vs. humans

No judge results yet. Run `python -m grader judge <dataset>` to have Claude grade the same tasks; this section will then compare its verdicts with the human raters.
