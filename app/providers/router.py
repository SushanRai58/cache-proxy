"""Routes a completion request to the right provider based on model name."""

from __future__ import annotations

from app.providers import groq_client, ollama_client

# Locally-pulled Ollama model names. Anything not in this set falls through
# to Groq. main.py and both provider clients treat the two identically,
# since they share one interface: complete(messages, model, temperature) -> str.
OLLAMA_MODELS: set[str] = {"llama3.2:1b"}


def complete(messages: list[dict], model: str, temperature: float) -> str:
    if model in OLLAMA_MODELS:
        return ollama_client.complete(messages, model, temperature)
    return groq_client.complete(messages, model, temperature)
