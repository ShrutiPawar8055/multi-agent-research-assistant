from pathlib import Path

import pytest

from researchcrew import KnowledgeBase, ResearchCrew, Settings
from researchcrew.cli import main
from researchcrew.models import NO_EVIDENCE, Evidence

from helpers import FakeWeb, RoleLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.from_directory(SAMPLE)


def test_offline_end_to_end_is_grounded_and_cited(kb):
    result = ResearchCrew(kb).ask("What is retrieval-augmented generation and why does it reduce hallucination?")
    assert result.verdict == "PASS" and result.revisions == 0
    assert "[" in result.answer and result.evidence
    assert "retrieval_augmented_generation.md" in {e.source for e in result.evidence}
    assert [t["step"] for t in result.trace][:2] == ["plan", "research"]


def test_unanswerable_question_abstains_instead_of_guessing(kb):
    result = ResearchCrew(kb).ask("How tall is the Eiffel tower in metres?")
    assert result.answer == NO_EVIDENCE and result.verdict == "PASS" and result.evidence == []


def test_web_route_without_web_tool_falls_back_with_a_note(kb):
    result = ResearchCrew(kb).ask("What are the latest developments in retrieval systems this year?")
    assert result.route == "WEB" and result.effective_route == "CORPUS"
    assert any("no web search" in n for n in result.notes)


def test_web_evidence_is_used_and_numbered_after_dedup(kb):
    web = FakeWeb([Evidence("web", "http://news", "Retrieval systems now commonly combine keyword and vector search.")] * 2)
    result = ResearchCrew(kb, web=web).ask("What are the latest retrieval systems news?")
    web_items = [e for e in result.evidence if e.origin == "web"]
    assert len(web_items) == 1
    assert [e.id for e in result.evidence] == list(range(1, len(result.evidence) + 1))


def test_revision_loop_recovers_from_a_bad_first_draft(kb):
    llm = RoleLLM(
        planner=['{"route": "CORPUS", "subquestions": ["What is retrieval-augmented generation?"]}'],
        writer=[
            "Retrieval-augmented generation is a technique that many people find very useful.",  # uncited
            "RAG gives a language model access to external documents at answer time [1].",
        ],
        critic=["PASS"],
    )
    result = ResearchCrew(kb, llm=llm).ask("What is RAG?")
    assert result.verdict == "PASS" and result.revisions == 1
    assert any(c[0] == "writer" and "Uncited claim" in c[1] for c in llm.calls)


def test_loop_is_bounded_and_reports_failure(kb):
    llm = RoleLLM(
        planner=['{"route": "CORPUS", "subquestions": ["What is retrieval-augmented generation?"]}'],
        writer=["This sentence has no citation but is long enough to count."],
    )
    result = ResearchCrew(kb, llm=llm, settings=Settings(max_revisions=2)).ask("What is RAG?")
    assert result.verdict == "FAIL" and result.revisions == 2 and result.issues
    assert sum(1 for c in llm.calls if c[0] == "writer") == 3


def test_blank_question_rejected(kb):
    with pytest.raises(ValueError):
        ResearchCrew(kb).ask("   ")


def test_cli_prints_answer_and_exit_code(capsys):
    code = main(["ask", "What is BM25 and how does it rank documents?", "--docs", str(SAMPLE), "--llm", "none"])
    out = capsys.readouterr().out
    assert code == 0 and "Verdict: PASS" in out and "Sources:" in out


def test_cli_reports_bad_docs_folder(capsys):
    assert main(["ask", "q?", "--docs", "/no/such/dir", "--llm", "none"]) == 2
    assert "not found" in capsys.readouterr().err
