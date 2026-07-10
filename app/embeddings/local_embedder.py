"""Local text embedding and similarity utilities backed by sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """Wraps a local sentence-transformers model for embedding and comparing text."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a normalized vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes cosine similarity equivalent to a plain
        # dot product downstream, and matches the convention Redis's COSINE
        # distance metric expects.
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # Written out explicitly (rather than calling util.cos_sim) so the math
        # driving cache-hit decisions is visible, not hidden behind a library call.
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
