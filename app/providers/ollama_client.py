"""Thin wrapper around a local Ollama server's chat API.

Not wired into the router yet (router.OLLAMA_MODELS is empty until Ollama
is actually set up locally) -- written now so adding support later is just
populating that set, no code changes needed here.
"""

from __future__ import annotations

import os

import ollama

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def complete(messages: list[dict], model: str, temperature: float) -> str:
    client = ollama.Client(host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]
