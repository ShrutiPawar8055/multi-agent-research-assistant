"""Command line: `researchcrew ask "question"`."""

from __future__ import annotations

import argparse
import json
import sys

from .bootstrap import build_crew
from .models import Result


def render(result: Result) -> str:
    lines = [
        f"Route: {result.route}" + (f" (used {result.effective_route})" if result.route != result.effective_route else ""),
        f"Sub-questions: {len(result.subquestions)}",
        "",
        result.answer,
        "",
        f"Verdict: {result.verdict} after {result.revisions} revision(s)",
    ]
    lines += [f"  ! {issue}" for issue in result.issues]
    lines += [f"  note: {note}" for note in result.notes]
    if result.evidence:
        lines += ["", "Sources:"]
        lines += [f"  [{e.id}] {e.source}" for e in result.evidence]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="researchcrew", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--docs", help="folder of .md/.txt/.pdf files (default: $CREW_DOCS_DIR or sample_docs)")
    ask.add_argument("--llm", choices=["auto", "gemini", "none"], default="auto")
    ask.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)

    try:
        crew = build_crew(docs=args.docs, llm_mode=args.llm)
        result = crew.ask(args.question)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2) if args.json else render(result))
    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
