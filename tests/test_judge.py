"""The judge is tested against a fake client, so no API key or network is needed."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from grader.data import load_dataset
from grader.judge import MODEL, JudgeError, compare_pair, judge_dataset, score_response

SAMPLE = load_dataset(Path(__file__).resolve().parent.parent / "data" / "sample")
TASK = SAMPLE.tasks[0]


class FakeClient:
    """Mimics client.beta.messages.create; `answer(prompt)` returns the JSON object to send back."""

    def __init__(self, answer, stop_reason="end_turn"):
        self.answer = answer
        self.stop_reason = stop_reason
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        text = json.dumps(self.answer(prompt))
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)])


def first_text(prompt: str) -> str:
    return prompt.split("<first_response>\n", 1)[1].split("\n</first_response>", 1)[0]


class PairwiseTests(unittest.TestCase):
    def test_consistent_judge(self):
        good = TASK.responses[0].text
        client = FakeClient(lambda p: {"rationale": "", "verdict": "first" if first_text(p) == good else "second"})
        result = compare_pair(client, TASK, SAMPLE.rubric)
        self.assertEqual(result["verdict"], "A")
        self.assertTrue(result["position_consistent"])
        self.assertEqual(len(client.calls), 2)

    def test_position_biased_judge_is_caught(self):
        # Always picks whatever it reads first: the swap exposes it.
        client = FakeClient(lambda p: {"rationale": "", "verdict": "first"})
        result = compare_pair(client, TASK, SAMPLE.rubric)
        self.assertEqual(result["verdict"], "tie")
        self.assertFalse(result["position_consistent"])
        self.assertEqual(result["raw"], ["first", "first"])

    def test_both_orders_are_actually_shown(self):
        client = FakeClient(lambda p: {"rationale": "", "verdict": "tie"})
        compare_pair(client, TASK, SAMPLE.rubric)
        firsts = [first_text(c["messages"][0]["content"]) for c in client.calls]
        self.assertEqual(firsts, [TASK.responses[0].text, TASK.responses[1].text])


class RequestShapeTests(unittest.TestCase):
    def test_request_uses_structured_output_and_fallbacks(self):
        client = FakeClient(lambda p: {"rationale": "", "verdict": "tie"})
        compare_pair(client, TASK, SAMPLE.rubric)
        call = client.calls[0]
        self.assertEqual(call["model"], MODEL)
        self.assertEqual(call["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(call["fallbacks"], "default")
        self.assertNotIn("temperature", call)


class RubricScoreTests(unittest.TestCase):
    def test_scores_are_clamped_to_scale(self):
        ids = [c.id for c in SAMPLE.rubric.criteria]
        client = FakeClient(lambda p: {"rationale": "ok", **dict(zip(ids, [9, 0, 3, 5]))})
        r = score_response(client, TASK, "B", SAMPLE.rubric)
        self.assertEqual(list(r["scores"].values()), [5, 1, 3, 5])
        self.assertIn(TASK.responses[1].text, client.calls[0]["messages"][0]["content"])

    def test_refusal_raises(self):
        client = FakeClient(lambda p: {}, stop_reason="refusal")
        with self.assertRaises(JudgeError):
            score_response(client, TASK, "A", SAMPLE.rubric)

    def test_dataset_run_skips_failures_and_continues(self):
        client = FakeClient(lambda p: {}, stop_reason="refusal")
        log = []
        self.assertEqual(judge_dataset(client, SAMPLE.tasks[:2], SAMPLE.rubric, log=log.append), [])
        self.assertEqual(len(log), 2)


if __name__ == "__main__":
    unittest.main()
