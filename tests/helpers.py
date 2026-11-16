"""Test doubles."""

from __future__ import annotations

from researchcrew.models import Evidence


class RoleLLM:
    """Replies per agent role (agents put `ROLE:<name>` on the first line of their system prompt)."""

    def __init__(self, **replies: list[str]) -> None:
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        role = system.split("\n", 1)[0].removeprefix("ROLE:").strip()
        self.calls.append((role, prompt))
        queue = self.replies.get(role, [])
        if not queue:
            raise AssertionError(f"no scripted reply left for role {role!r}")
        return queue.pop(0) if len(queue) > 1 else queue[0]


class FakeWeb:
    def __init__(self, results: list[Evidence] | None = None, fail: bool = False) -> None:
        self.results, self.fail, self.queries = results or [], fail, []

    def search(self, query: str, max_results: int = 3) -> list[Evidence]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("boom")
        return list(self.results)
