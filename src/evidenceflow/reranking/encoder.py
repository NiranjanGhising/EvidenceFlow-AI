"""Pretrained cross-encoder passage scoring."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PassageReranker(Protocol):
    """Interface implemented by passage reranking models."""

    @property
    def model_name(self) -> str:
        """Return the reranking model name."""

    def score(
        self,
        query: str,
        passages: list[str],
    ) -> np.ndarray:
        """Score passages against one query."""


class CrossEncoderReranker:
    """Score query-passage pairs using a pretrained cross-encoder."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        self._model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model_name(self) -> str:
        """Return the model name."""

        return self._model_name

    def _load_model(self) -> None:
        """Load the cross-encoder only when first required."""

        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            model_name_or_path=self._model_name,
            device=self.device,
        )

    def score(
        self,
        query: str,
        passages: list[str],
    ) -> np.ndarray:
        """Return one float score for every passage."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Reranking query cannot be empty")

        if not passages:
            return np.empty(
                shape=(0,),
                dtype=np.float32,
            )

        if any(not passage.strip() for passage in passages):
            raise ValueError(
                "Reranking passages cannot be empty"
            )

        self._load_model()

        pairs = [
            (normalized_query, passage)
            for passage in passages
        ]

        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scores_array = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)

        if scores_array.size != len(passages):
            raise RuntimeError(
                "Reranker score count does not match passage count"
            )

        return scores_array