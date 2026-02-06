"""Lazy-loaded embedding function for ChromaDB integration."""

from __future__ import annotations

from typing import Any

import chromadb.api.types as chroma_types


class LazyEmbeddingFunction(chroma_types.EmbeddingFunction[list[str]]):
    """Embedding function that lazy-loads SentenceTransformer on first use.

    Implements ChromaDB's EmbeddingFunction protocol for seamless integration.
    The model is loaded only when embeddings are first requested, keeping
    server startup fast.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding function.

        Args:
            model_name: Name of the SentenceTransformer model to use.
        """
        self._model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Lazy-load the SentenceTransformer model on first use.

        Returns:
            Loaded SentenceTransformer model instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            input: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        model = self._load_model()
        embeddings = model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding dimension (384 for all-MiniLM-L6-v2).

        Returns:
            Dimensionality of the embedding vectors.
        """
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
