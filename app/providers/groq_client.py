"""Thin wrapper around the Groq chat completions API."""

from __future__ import annotations

import os

from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq:
    # Lazy singleton: importing this module (and therefore the router,
    # and therefore main.py) shouldn't require GROQ_API_KEY to be set --
    # a given request might never actually reach Groq, e.g. it's served
    # from cache or routed to Ollama.
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it before making a request that routes to Groq."
            )
        _client = Groq(api_key=api_key)
    return _client


def complete(messages: list[dict], model: str, temperature: float) -> str:
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
