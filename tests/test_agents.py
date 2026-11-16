from researchcrew.agents import Critic, Planner, Researcher, Writer, grounding_issues, heuristic_plan
from researchcrew.models import NO_EVIDENCE, Evidence
from researchcrew.retrieval import KnowledgeBase, chunk_text
from researchcrew.tools import parse_tavily

from helpers import FakeWeb, RoleLLM

EV = [
    Evidence("corpus", "rag.md", "RAG places retrieved passages in the prompt so answers can be grounded.", id=1),
    Evidence("corpus", "bm25.md", "BM25 scores passages using rare query terms and document length.", id=2),
]


# --- planner -----------------------------------------------------------------
def test_heuristic_routes_fresh_questions_to_web():
    assert heuristic_plan("What are the latest model releases in 2026?").route == "WEB"
    assert heuristic_plan("What is BM25?").route == "CORPUS"


def test_heuristic_splits_compound_questions_and_caps_them():
    plan = heuristic_plan("What is RAG and how does BM25 rank passages? Why use chunk overlap?", max_sub=2)
    assert len(plan.subquestions) == 2
    assert plan.subquestions[0].startswith("What is RAG")


def test_planner_uses_valid_llm_json():
    llm = RoleLLM(planner=['Sure! {"route": "both", "subquestions": ["a one two", "b one two"]}'])
    plan = Planner(llm).plan("anything")
    assert (plan.route, plan.subquestions) == ("BOTH", ["a one two", "b one two"])


def test_planner_falls_back_on_garbage_or_errors():
    assert Planner(RoleLLM(planner=["not json"])).plan("What is BM25?").route == "CORPUS"
    assert Planner(RoleLLM(planner=['{"route": "MOON", "subquestions": ["x"]}'])).plan("What is BM25?").route == "CORPUS"

    class Broken:
        def complete(self, system, prompt):
            raise RuntimeError("down")

    assert Planner(Broken()).plan("latest news today").route == "WEB"


# --- researcher --------------------------------------------------------------
def test_researcher_combines_corpus_and_web_and_tolerates_failures():
    kb = KnowledgeBase(chunk_text("bm25 ranks passages by rare terms", "k.md"))
    web = FakeWeb([Evidence("web", "http://x", "fresh news")])
    out = Researcher(kb, web).research(["bm25 ranking?", "news?"], "BOTH")
    assert {e.origin for e in out.evidence} == {"corpus", "web"}
    assert len(web.queries) == 2

    failing = Researcher(kb, FakeWeb(fail=True)).research(["bm25 ranking?"], "BOTH")
    assert [e.origin for e in failing.evidence] == ["corpus"]
    assert "web search failed" in failing.errors[0]


def test_researcher_skips_web_when_unconfigured():
    kb = KnowledgeBase(chunk_text("bm25 ranks passages", "k.md"))
    assert Researcher(kb, None).research(["bm25?"], "WEB").evidence == []


def test_parse_tavily_skips_empty_results():
    payload = {"results": [{"url": "http://a", "content": " text ", "score": 0.5}, {"url": "http://b", "content": ""}]}
    found = parse_tavily(payload)
    assert [(e.source, e.text, e.origin) for e in found] == [("http://a", "text", "web")]


# --- writer ------------------------------------------------------------------
def test_extractive_writer_cites_and_abstains():
    answer = Writer().write("How does BM25 score passages?", EV)
    assert answer.endswith("document length [2].")  # quotes the best-matching sentence, cited
    assert Writer().write("How does BM25 score passages?", EV[:1]).endswith("[1].")
    assert Writer().write("anything", []) == NO_EVIDENCE
    assert Writer().write("zebra migration patterns", EV) == NO_EVIDENCE


def test_writer_passes_feedback_to_llm():
    llm = RoleLLM(writer=["Draft [1]."])
    Writer(llm).write("q?", EV, feedback=["Uncited claim"])
    assert "Uncited claim" in llm.calls[0][1]


# --- critic ------------------------------------------------------------------
def test_critic_accepts_grounded_answer_and_abstention():
    good = "RAG places retrieved passages in the prompt so answers can be grounded [1]."
    assert grounding_issues(good, EV) == []
    assert grounding_issues(NO_EVIDENCE, EV) == []


def test_critic_flags_uncited_missing_and_unsupported_claims():
    uncited = "RAG places retrieved passages in the prompt so answers are grounded."
    assert "Uncited" in grounding_issues(uncited, EV)[0]
    assert "missing evidence" in grounding_issues("Some long claim about retrieval systems today [9].", EV)[0]
    unsupported = "Volcanic eruptions rearrange tectonic plates beneath oceans dramatically [1]."
    assert "Not supported" in grounding_issues(unsupported, EV)[0]


def test_llm_critic_can_only_tighten_the_verdict():
    good = "RAG places retrieved passages in the prompt so answers can be grounded [1]."
    assert Critic(RoleLLM(critic=["PASS"])).review("q", good, EV).passed
    review = Critic(RoleLLM(critic=["FAIL: misses nuance"])).review("q", good, EV)
    assert not review.passed and review.issues == ["misses nuance"]
    # deterministic failure is never overridden by an optimistic LLM
    assert not Critic(RoleLLM(critic=["PASS"])).review("q", "A long uncited claim about many things here.", EV).passed
