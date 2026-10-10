"""Sentence Transformer embedding generation."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingEncoder(Protocol):
    """Interface supported by real and test embedding encoders."""

    @property
    def model_name(self) -> str:
        """Return the embedding model name."""

    @property
    def dimension(self) -> int:
        """Return the generated vector dimension."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text into normalized float32 vectors."""


class SentenceTransformerEncoder:
    """Generate normalized vectors using Sentence Transformers."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._load_model()

        if self._dimension is None:
            raise RuntimeError("Embedding dimension could not be determined")

        return self._dimension

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            device=self.device,
        )

        dimension = self._model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "The embedding model did not provide a vector dimension"
            )

        self._dimension = int(dimension)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode non-empty text into normalized float32 vectors."""

        if not texts:
            dimension = self.dimension

            return np.empty(
                shape=(0, dimension),
                dtype=np.float32,
            )

        if any(not text.strip() for text in texts):
            raise ValueError("Embedding text cannot be empty")

        self._load_model()

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise RuntimeError(
                "Embedding model returned an invalid vector shape"
            )

        if vectors.shape[0] != len(texts):
            raise RuntimeError(
                "Embedding count does not match the input text count"
            )

        self._dimension = int(vectors.shape[1])

        return vectors
