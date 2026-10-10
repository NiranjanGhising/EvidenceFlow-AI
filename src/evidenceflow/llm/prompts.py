"""Prompt construction for evidence-grounded LLM generation."""

from __future__ import annotations

from evidenceflow.llm.models import LLMGenerationRequest

REFUSAL_TEXT = (
    "I do not have enough evidence in the indexed documents "
    "to answer this question."
)


def build_grounded_prompt(
    request: LLMGenerationRequest,
) -> str:
    """Build a strict grounded-answer prompt."""

    question = request.question.strip()
    evidence_context = request.evidence_context.strip()

    if not question:
        raise ValueError("question cannot be empty")

    if not evidence_context:
        raise ValueError("evidence_context cannot be empty")

    if not request.allowed_citation_ids:
        raise ValueError(
            "allowed_citation_ids cannot be empty"
        )

    citation_list = "\n".join(
        f"- [{citation_id}]"
        for citation_id in request.allowed_citation_ids
    )

    return f"""
You are the grounded answer generator for EvidenceFlow AI.

Answer the user's question using only the evidence provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts, sources, or citation IDs.
3. Every factual statement must be supported by the evidence.
4. Add a citation immediately after each supported statement.
5. Use only citation IDs from the allowed citation list.
6. Preserve citation IDs exactly, including square brackets.
7. Do not mention retrieval scores or internal system behavior.
8. If the evidence is insufficient, reply exactly with:
{REFUSAL_TEXT}
9. Keep the answer concise, clear, and natural.
10. Do not add a separate references section.

Allowed citations:
{citation_list}

Evidence:
{evidence_context}

Question:
{question}

Grounded answer:
""".strip()


def build_evidence_context(
    evidence_items: list[
        tuple[str, str, str, tuple[str, ...]]
    ],
) -> str:
    """Format reranked evidence for grounded generation."""

    if not evidence_items:
        raise ValueError("evidence_items cannot be empty")

    formatted_items: list[str] = []

    for (
        chunk_id,
        source_name,
        text,
        heading_path,
    ) in evidence_items:
        normalized_chunk_id = chunk_id.strip()
        normalized_source_name = source_name.strip()
        normalized_text = text.strip()

        if not normalized_chunk_id:
            raise ValueError(
                "Evidence chunk ID cannot be empty"
            )

        if not normalized_source_name:
            raise ValueError(
                "Evidence source name cannot be empty"
            )

        if not normalized_text:
            raise ValueError(
                "Evidence text cannot be empty"
            )

        heading = (
            " > ".join(heading_path)
            if heading_path
            else "No heading"
        )

        formatted_items.append(
            "\n".join(
                [
                    f"Citation: [{normalized_chunk_id}]",
                    f"Source: {normalized_source_name}",
                    f"Heading: {heading}",
                    "Content:",
                    normalized_text,
                ]
            )
        )

    return "\n\n---\n\n".join(formatted_items)