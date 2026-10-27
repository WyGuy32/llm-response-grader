"""LLM-as-judge with Claude, with a check for position bias.

LLM judges tend to favor whichever answer they read first. To catch that,
every pairwise comparison runs twice with the answers swapped. A verdict
only counts when both orders agree. When they don't, the result is recorded
as a tie and flagged `position_consistent: false`, so the report can say how
often the judge's choice depended on the order.

Requires the `anthropic` package and a credential (ANTHROPIC_API_KEY or an
`ant auth login` profile). Everything else in this repo runs without it.
"""

from __future__ import annotations

import json
from typing import Any

from .data import Rubric, Task

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are an expert evaluator of AI assistant responses, working the way a careful \
human annotator would. Judge substance over style: a confident, polished answer that is wrong \
is worse than a plain answer that is right. Fabricated facts or citations, and advice that could \
cause harm, are serious failures. Length is not quality; padding counts against clarity. Do not \
let the order in which responses appear influence you."""


class JudgeError(RuntimeError):
    pass


def _rubric_text(rubric: Rubric) -> str:
    lo, hi = rubric.scale
    lines = [f"Score each criterion from {lo} to {hi}."]
    for c in rubric.criteria:
        anchors = "; ".join(f"{k} = {v}" for k, v in sorted(c.anchors.items()))
        lines.append(f"- {c.id} ({c.name}): {c.description}" + (f" Anchors: {anchors}." if anchors else ""))
    return "\n".join(lines)


def _call(client: Any, prompt: str, schema: dict) -> dict:
    """One structured-output request. Returns the parsed JSON object."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        # If a safety classifier declines, re-run on Anthropic's recommended fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise JudgeError("the judge model declined this item")
    if response.stop_reason == "max_tokens":
        raise JudgeError("the judge's answer was cut off (max_tokens)")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise JudgeError("no text in judge response")
    return json.loads(text)


def score_response(client: Any, task: Task, label: str, rubric: Rubric) -> dict:
    """Rubric scores for one response, judged on its own."""
    response = task.responses["AB".index(label)]
    schema = {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            **{c.id: {"type": "integer"} for c in rubric.criteria},
        },
        "required": ["rationale", *[c.id for c in rubric.criteria]],
        "additionalProperties": False,
    }
    prompt = (
        f"{_rubric_text(rubric)}\n\n<user_prompt>\n{task.prompt}\n</user_prompt>\n\n"
        f"<response>\n{response.text}\n</response>\n\n"
        "Give a brief rationale, then the score for each criterion."
    )
    raw = _call(client, prompt, schema)
    lo, hi = rubric.scale
    scores = {c.id: min(hi, max(lo, int(raw[c.id]))) for c in rubric.criteria}
    return {"task_id": task.id, "kind": "rubric", "label": label, "scores": scores, "rationale": raw["rationale"]}


def _compare_once(client: Any, task: Task, first: str, second: str, rubric: Rubric) -> str:
    """Ask which of two responses is better. Returns "first", "second" or "tie"."""
    schema = {
        "type": "object",
        "properties": {"rationale": {"type": "string"}, "verdict": {"type": "string", "enum": ["first", "second", "tie"]}},
        "required": ["rationale", "verdict"],
        "additionalProperties": False,
    }
    prompt = (
        f"Compare two responses to the same prompt using this rubric.\n{_rubric_text(rubric)}\n\n"
        f"<user_prompt>\n{task.prompt}\n</user_prompt>\n\n"
        f"<first_response>\n{first}\n</first_response>\n\n<second_response>\n{second}\n</second_response>\n\n"
        "Which response is better overall? Answer \"tie\" only if they are genuinely equivalent."
    )
    return _call(client, prompt, schema)["verdict"]


def compare_pair(client: Any, task: Task, rubric: Rubric) -> dict:
    """Pairwise verdict for a task, run in both orders to detect position bias."""
    a, b = (r.text for r in task.responses)
    forward = _compare_once(client, task, a, b, rubric)  # A shown first
    backward = _compare_once(client, task, b, a, rubric)  # B shown first

    as_label = {"first": "A", "second": "B", "tie": "tie"}
    swapped = {"first": "B", "second": "A", "tie": "tie"}
    v1, v2 = as_label[forward], swapped[backward]
    consistent = v1 == v2
    return {
        "task_id": task.id,
        "kind": "pairwise",
        "verdict": v1 if consistent else "tie",
        "position_consistent": consistent,
        "raw": [forward, backward],
    }


def judge_dataset(client: Any, tasks: list[Task], rubric: Rubric, rubric_scores: bool = True, log=print) -> list[dict]:
    """Run the judge over every task. Items the judge can't handle are logged and skipped."""
    results = []
    for task in tasks:
        try:
            results.append(compare_pair(client, task, rubric))
            if rubric_scores:
                results.extend(score_response(client, task, label, rubric) for label in "AB")
            log(f"judged {task.id}")
        except (JudgeError, json.JSONDecodeError, KeyError) as e:
            log(f"skipped {task.id}: {e}")
    return results


def make_client():
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit("The judge needs the Anthropic SDK: pip install anthropic") from e
    return anthropic.Anthropic()
