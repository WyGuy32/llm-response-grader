"""Command line.

    python -m grader report data/sample              print the evaluation report
    python -m grader report data/sample -o REPORT.md write it to a file
    python -m grader judge data/sample               grade the tasks with Claude (needs an API key)
    python -m grader export-web data/sample          refresh docs/sample.json for the web rater
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import DataError, load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grader", description="Rate, compare and audit LLM responses.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="build the agreement and leaderboard report")
    p_report.add_argument("dataset", type=Path)
    p_report.add_argument("-o", "--output", type=Path)

    p_judge = sub.add_parser("judge", help="grade every task with Claude as an LLM judge")
    p_judge.add_argument("dataset", type=Path)
    p_judge.add_argument("--pairwise-only", action="store_true", help="skip per-response rubric scores")
    p_judge.add_argument("--limit", type=int, help="only judge the first N tasks")

    p_web = sub.add_parser("export-web", help="bundle a dataset for the browser rating tool")
    p_web.add_argument("dataset", type=Path)
    p_web.add_argument("-o", "--output", type=Path, default=Path("docs/sample.json"))

    args = parser.parse_args(argv)
    try:
        ds = load_dataset(args.dataset)
    except (DataError, FileNotFoundError, KeyError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.command == "report":
        from .report import build_report

        text = build_report(ds)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            sys.stdout.reconfigure(encoding="utf-8")
            print(text)

    elif args.command == "judge":
        from .judge import judge_dataset, make_client

        tasks = ds.tasks[: args.limit] if args.limit else ds.tasks
        results = judge_dataset(make_client(), tasks, ds.rubric, rubric_scores=not args.pairwise_only)
        out = args.dataset / "judgments.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
        print(f"wrote {len(results)} judgments to {out}")

    elif args.command == "export-web":
        args.output.write_text(export_web(args.dataset), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


def export_web(dataset: Path) -> str:
    """The web rater's data file: rubric, tasks and the sample ratings, as one JSON document."""
    bundle = {
        "rubric": json.loads((dataset / "rubric.json").read_text(encoding="utf-8")),
        "tasks": json.loads((dataset / "tasks.json").read_text(encoding="utf-8")),
        "ratings": [
            json.loads(line)
            for line in (dataset / "ratings.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
    }
    return json.dumps(bundle, ensure_ascii=False, indent=1) + "\n"


if __name__ == "__main__":
    sys.exit(main())
