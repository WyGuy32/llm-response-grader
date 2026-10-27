import json
import shutil
import tempfile
import unittest
from pathlib import Path

from grader.__main__ import export_web
from grader.data import DataError, load_dataset
from grader.report import build_report, majority

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample"


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for name in ("rubric.json", "tasks.json", "ratings.jsonl"):
            shutil.copy(SAMPLE_DIR / name, self.tmp / name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _append_rating(self, record: dict) -> None:
        with open(self.tmp / "ratings.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def test_sample_loads(self):
        ds = load_dataset(SAMPLE_DIR)
        self.assertEqual(len(ds.tasks), 12)
        self.assertEqual(ds.raters, ["rater_1", "rater_2", "rater_3"])

    def test_out_of_range_score_rejected(self):
        bad = json.loads((SAMPLE_DIR / "ratings.jsonl").read_text().splitlines()[0])
        bad["rater"] = "new_rater"
        bad["scores"]["A"]["clarity"] = 7
        self._append_rating(bad)
        with self.assertRaisesRegex(DataError, "clarity"):
            load_dataset(self.tmp)

    def test_duplicate_rating_rejected(self):
        self._append_rating(json.loads((SAMPLE_DIR / "ratings.jsonl").read_text().splitlines()[0]))
        with self.assertRaisesRegex(DataError, "duplicate"):
            load_dataset(self.tmp)

    def test_unknown_task_rejected(self):
        bad = json.loads((SAMPLE_DIR / "ratings.jsonl").read_text().splitlines()[0])
        bad["task_id"] = "nope"
        self._append_rating(bad)
        with self.assertRaisesRegex(DataError, "unknown task"):
            load_dataset(self.tmp)


class ReportTests(unittest.TestCase):
    def test_report_sections(self):
        text = build_report(load_dataset(SAMPLE_DIR))
        for heading in ("## Leaderboard", "## Inter-rater agreement", "## Rater calibration", "## LLM judge vs. humans"):
            self.assertIn(heading, text)

    def test_simulated_lenient_rater_is_flagged(self):
        text = build_report(load_dataset(SAMPLE_DIR))
        row = next(line for line in text.splitlines() if line.startswith("| rater_2 | "))
        self.assertIn("lenient", row)

    def test_judge_section_uses_judgments(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for name in ("rubric.json", "tasks.json", "ratings.jsonl"):
                shutil.copy(SAMPLE_DIR / name, tmp / name)
            judgments = [
                {"task_id": "t01", "kind": "pairwise", "verdict": "A", "position_consistent": True},
                {"task_id": "t02", "kind": "pairwise", "verdict": "tie", "position_consistent": False},
            ]
            (tmp / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments))
            text = build_report(load_dataset(tmp))
            self.assertIn("Same verdict when answer order is swapped | 50%", text)
            self.assertIn("Judge and humans disagree on: t02", text)
        finally:
            shutil.rmtree(tmp)

    def test_majority(self):
        self.assertEqual(majority(["A", "A", "B"]), "A")
        self.assertEqual(majority(["A", "B"]), "tie")


class WebBundleTests(unittest.TestCase):
    def test_docs_bundle_is_in_sync_with_sample_data(self):
        committed = (ROOT / "docs" / "sample.json").read_text(encoding="utf-8")
        self.assertEqual(committed, export_web(SAMPLE_DIR), "run: python -m grader export-web data/sample")


if __name__ == "__main__":
    unittest.main()
