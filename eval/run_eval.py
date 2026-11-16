"""Offline evaluation of the pipeline against eval/dataset.jsonl.

Metrics (all deterministic, no API key needed):
  routing accuracy    planner route == expected route
  source hit rate     the expected document was retrieved (corpus questions)
  keyword recall      expected key terms appear in the answer (corpus questions)
  grounded rate       critic verdict PASS (corpus questions)

Usage:  python eval/run_eval.py [--llm none|gemini] [--write eval/results.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from researchcrew.bootstrap import build_crew

ROOT = Path(__file__).resolve().parent


def run(llm_mode: str) -> tuple[list[dict], dict]:
    crew = build_crew(docs=ROOT.parent / "sample_docs", llm_mode=llm_mode)
    rows = []
    for line in (ROOT / "dataset.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        result = crew.ask(case["question"])
        corpus_case = case["expected_source"] is not None
        answer = result.answer.lower()
        rows.append(
            {
                "question": case["question"],
                "route_ok": result.route == case["expected_route"],
                "corpus_case": corpus_case,
                "source_hit": corpus_case and case["expected_source"] in {e.source for e in result.evidence},
                "keyword_recall": (
                    sum(k in answer for k in case["must_mention"]) / len(case["must_mention"])
                    if case["must_mention"] else None
                ),
                "grounded": result.verdict == "PASS",
            }
        )
    corpus = [r for r in rows if r["corpus_case"]]
    recalls = [r["keyword_recall"] for r in corpus if r["keyword_recall"] is not None]
    summary = {
        "questions": len(rows),
        "routing_accuracy": sum(r["route_ok"] for r in rows) / len(rows),
        "source_hit_rate": sum(r["source_hit"] for r in corpus) / len(corpus),
        "keyword_recall": sum(recalls) / len(recalls),
        "grounded_rate": sum(r["grounded"] for r in corpus) / len(corpus),
    }
    return rows, summary


def to_markdown(rows: list[dict], s: dict, llm_mode: str) -> str:
    out = [
        f"# Evaluation results (llm={llm_mode})", "",
        "| Metric | Score |", "|---|---|",
        f"| routing accuracy (all {s['questions']}) | {s['routing_accuracy']:.0%} |",
        f"| source hit rate | {s['source_hit_rate']:.0%} |",
        f"| keyword recall | {s['keyword_recall']:.0%} |",
        f"| grounded rate | {s['grounded_rate']:.0%} |", "",
        "| Route ok | Source hit | Keyword recall | Grounded | Question |", "|---|---|---|---|---|",
    ]
    for r in rows:
        kr = "-" if r["keyword_recall"] is None else f"{r['keyword_recall']:.0%}"
        hit = "-" if not r["corpus_case"] else ("yes" if r["source_hit"] else "NO")
        out.append(f"| {'yes' if r['route_ok'] else 'NO'} | {hit} | {kr} | {'yes' if r['grounded'] else 'NO'} | {r['question']} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["none", "gemini", "auto"], default="none")
    parser.add_argument("--write", help="write a markdown report to this path")
    args = parser.parse_args()
    rows, summary = run(args.llm)
    report = to_markdown(rows, summary, args.llm)
    print(report)
    if args.write:
        Path(args.write).write_text(report)
