from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from researchcrew import KnowledgeBase, ResearchCrew  # noqa: E402
from researchcrew.api import create_app  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "sample_docs"


def client() -> TestClient:
    return TestClient(create_app(ResearchCrew(KnowledgeBase.from_directory(SAMPLE))))


def test_health_and_ask():
    c = client()
    assert c.get("/health").json() == {"status": "ok"}
    body = c.post("/ask", json={"question": "What is BM25?"}).json()
    assert body["verdict"] == "PASS" and body["evidence"]


def test_blank_and_missing_questions_are_rejected():
    c = client()
    assert c.post("/ask", json={"question": "   "}).status_code == 422
    assert c.post("/ask", json={}).status_code == 422
