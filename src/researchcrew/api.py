"""HTTP API (needs the `api` extra). Run: uvicorn researchcrew.api:app --port 8080"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .bootstrap import build_crew
from .pipeline import ResearchCrew


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def create_app(crew: ResearchCrew | None = None) -> FastAPI:
    app = FastAPI(title="ResearchCrew", version="0.1.0")
    state: dict[str, ResearchCrew] = {}
    if crew is not None:
        state["crew"] = crew

    def get_crew() -> ResearchCrew:
        if "crew" not in state:  # built lazily so importing the module needs no config
            state["crew"] = build_crew()
        return state["crew"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be blank")
        return get_crew().ask(question).to_dict()

    return app


app = create_app()
