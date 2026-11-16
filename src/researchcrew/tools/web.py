"""Live web search behind a tiny interface. Tavily is the bundled provider."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol

from ..models import Evidence


class WebSearch(Protocol):
    def search(self, query: str, max_results: int = 3) -> list[Evidence]: ...


def parse_tavily(payload: dict[str, Any]) -> list[Evidence]:
    """Turn a Tavily /search response into Evidence (skips empty results)."""
    found: list[Evidence] = []
    for item in payload.get("results", []):
        text = (item.get("content") or "").strip()
        if text:
            found.append(
                Evidence(
                    origin="web",
                    source=item.get("url") or item.get("title") or "web",
                    text=text,
                    score=float(item.get("score") or 0.0),
                )
            )
    return found


class TavilySearch:
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    def search(self, query: str, max_results: int = 3) -> list[Evidence]:  # pragma: no cover - network
        body = json.dumps({"query": query, "max_results": max_results}).encode()
        request = urllib.request.Request(
            self.URL,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return parse_tavily(json.load(response))
