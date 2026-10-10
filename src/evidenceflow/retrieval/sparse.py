"""BM25 sparse retrieval over persisted chunks."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from evidenceflow.domain.models import Chunk
from evidenceflow.retrieval.models import SparseSearchResult
from evidenceflow.retrieval.tokenizer import tokenize_for_search
from evidenceflow.storage.chunks import ChunkRepository


class BM25Retriever:
    """Build and search an in-memory BM25 index from stored chunks."""

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        heading_weight: int = 2,
    ) -> None:
        if heading_weight < 0:
            raise ValueError("heading_weight cannot be negative")

        self.chunk_repository = chunk_repository
        self.heading_weight = heading_weight

    def _chunk_tokens(self, chunk: Chunk) -> list[str]:
        """Tokenize chunk text with optional heading weighting."""

        text_tokens = tokenize_for_search(chunk.text)

        heading_tokens = tokenize_for_search(
            " ".join(chunk.heading_path)
        )

        return (
            heading_tokens * self.heading_weight
            + text_tokens
        )

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SparseSearchResult]:
        """Search all stored chunks using BM25."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query cannot be empty")

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        query_tokens = tokenize_for_search(normalized_query)

        if not query_tokens:
            return []

        chunks = self.chunk_repository.list_all()

        if not chunks:
            return []

        tokenized_corpus = [
            self._chunk_tokens(chunk)
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: (
                -float(item[1]),
                item[0].chunk_id,
            ),
        )

        query_token_set = set(query_tokens)
        results: list[SparseSearchResult] = []

        for chunk, raw_score in ranked:
            chunk_token_set = set(
                self._chunk_tokens(chunk)
            )

            has_lexical_match = bool(
                query_token_set.intersection(
                    chunk_token_set
                )
            )

            if not has_lexical_match:
                continue

            score = float(raw_score)

            results.append(
                SparseSearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_name=chunk.source_name,
                    text=chunk.text,
                    score=score,
                    rank=len(results) + 1,
                    chunk_order=chunk.chunk_order,
                    heading_path=chunk.heading_path,
                )
            )

            if len(results) >= limit:
                break

        return results