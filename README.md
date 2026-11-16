# ResearchCrew

A multi-agent research assistant. It plans a question, gathers evidence from a local
document collection and (optionally) the live web, writes a cited answer, and has an
independent critic verify that every claim is grounded before the answer is returned.

The core pipeline is **standard-library only** and runs fully offline. Gemini (LLM) and
Tavily (web search) are optional plug-ins.

```
question
   │
   ▼
Planner ──► route (CORPUS | WEB | BOTH) + 1-3 sub-questions
   │
   ▼
Researcher ──► one lookup per sub-question and source, run in parallel
   │            BM25 over local docs · Tavily web search
   ▼
number + de-duplicate evidence  [1] [2] [3] …
   │
   ▼
Writer ──► answer with [n] citations ◄──┐
   │                                     │ feedback (max 2 revisions)
   ▼                                     │
Critic ──► PASS / FAIL ──────────────────┘
   │
   ▼
Result: answer, verdict, sources, notes, per-step trace
```

## Design choices

- **Deterministic guardrails + optional LLM judgment.** The critic always checks that
  citations exist, that no claim is uncited, and that each cited sentence overlaps with the
  evidence it cites. An LLM review can only make the verdict stricter, never override a failure.
- **Abstain rather than guess.** No usable evidence → "Insufficient evidence…" (counts as grounded).
- **Every agent degrades gracefully.** With no LLM, the planner uses heuristics and the writer
  quotes the best-matching sentences. A bad LLM reply or a failing search never crashes a request.
- **Bounded loop.** Writer ⇄ critic runs at most `max_revisions` times (default 2); a still-failing
  answer is returned with `verdict: FAIL` and the reasons.
- **Pure-Python BM25** instead of an embedding index: no vector database to deploy, and exact
  terms are matched reliably. Swapping in embeddings means replacing `retrieval/bm25.py`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

researchcrew ask "What is retrieval-augmented generation and why does it reduce hallucination?"
researchcrew ask "How does BM25 rank documents?" --json
pytest
python eval/run_eval.py
```

Your own documents: `researchcrew ask "…" --docs path/to/folder` (`.md`, `.txt`; `.pdf` with `pip install -e ".[pdf]"`).

### With Gemini and web search

```bash
pip install -e ".[gemini,api]"
cp .env.example .env     # set GOOGLE_API_KEY and TAVILY_API_KEY, then export them
researchcrew ask "What are the latest developments in AI agent frameworks?"
uvicorn researchcrew.api:app --port 8080       # POST /ask {"question": "..."}
```

Without `TAVILY_API_KEY`, a question the planner sends to the web falls back to the local
documents and says so in `notes`.

## Layout

```
src/researchcrew/
  agents/        planner.py  researcher.py  writer.py  critic.py
  retrieval/     chunking.py  bm25.py  store.py
  tools/web.py   Tavily search behind a small interface
  pipeline.py    ResearchCrew: orchestration, revision loop, trace
  llm.py         LLM protocol + Gemini client
  cli.py  api.py  bootstrap.py  config.py  models.py  text.py
sample_docs/     five short original notes used by the demo, tests and eval
tests/           pytest suite (LLM and web are faked)
eval/            dataset.jsonl, run_eval.py, results.md
```

## Evaluation

`eval/results.md` is produced by `python eval/run_eval.py` on the 14-question set in
`eval/dataset.jsonl` (offline mode). It is a small, self-written set over the bundled sample
docs, so treat it as a regression check, not a benchmark. Offline answers quote source
sentences, which is why keyword recall is below 100%.

## Status and limits

- Tested: retrieval, planner, researcher, writer, critic, pipeline, CLI, evaluation (offline).
- Not exercised in development (no network or keys available): the Gemini client, the live
  Tavily call, and the FastAPI test (skipped unless `fastapi` and `httpx` are installed).
  Run `pytest` after installing the extras to cover the API.
- The lexical support check cannot detect a claim that is fluent, uses the right words, and is
  still wrong; that is what the optional LLM critic is for.
- In-process only: no auth, rate limiting or persistent sessions.

## License

MIT — see [LICENSE](LICENSE).
