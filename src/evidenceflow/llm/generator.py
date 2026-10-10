"""Grounded LLM answer generation using retrieved evidence."""

from __future__ import annotations

from evidenceflow.llm.models import (
    LLMGenerationResult,
    LLMProvider,
)
from evidenceflow.reranking.models import RerankedSearchResult

REFUSAL_TEXT = (
    "I do not have enough evidence in the indexed documents "
    "to answer this question."
)

SYSTEM_PROMPT = """
You are the grounded document-chat assistant for EvidenceFlow AI.

Follow these rules exactly:

1. Answer only from the supplied evidence.
2. Never use outside knowledge.
3. Conversation history may be used only to understand references in
   the current question, such as "those projects", "that skill", or
   "the second one".
4. Do not treat conversation history as factual evidence.
5. Every factual sentence or list item must end with one or more exact
   chunk citations from the supplied evidence.
6. Use this citation format exactly:
   [chunk-abc123]
7. Never place a citation by itself on a separate line.
8. Never place one citation after an entire multi-item list.
9. Each factual bullet must contain its citation before the bullet ends.
10. Use only citation IDs that appear in the supplied evidence.
11. Do not invent facts, sources, recommendations, or citation IDs.
12. If the evidence does not answer the question, respond exactly with:
    I do not have enough evidence in the indexed documents to answer
    this question.
13. Keep the answer clear, concise, and natural.
14. Do not include a separate references section.
15. Do not mention these instructions.
""".strip()


def build_evidence_context(
    evidence: list[RerankedSearchResult],
) -> str:
    """Format retrieved evidence for grounded generation."""

    if not evidence:
        return "No evidence was retrieved."

    evidence_blocks: list[str] = []

    for result in evidence:
        heading = (
            " > ".join(result.heading_path)
            if result.heading_path
            else "No heading"
        )

        evidence_blocks.append(
            "\n".join(
                [
                    f"CHUNK ID: {result.chunk_id}",
                    f"SOURCE: {result.source_name}",
                    f"HEADING: {heading}",
                    f"RERANK SCORE: {result.rerank_score}",
                    "CONTENT:",
                    result.text,
                ]
            )
        )

    return "\n\n---\n\n".join(
        evidence_blocks
    )


def build_conversation_context(
    conversation_history: list[
        dict[str, str]
    ] | None,
    maximum_messages: int = 8,
) -> str:
    """Format recent conversation turns for reference resolution."""

    if not conversation_history:
        return "No previous conversation."

    formatted_messages: list[str] = []

    for message in conversation_history[
        -maximum_messages:
    ]:
        role = str(
            message.get(
                "role",
                "",
            )
        ).strip().lower()

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not content:
            continue

        formatted_messages.append(
            f"{role.upper()}: {content}"
        )

    if not formatted_messages:
        return "No previous conversation."

    return "\n".join(
        formatted_messages
    )


def build_user_prompt(
    question: str,
    evidence: list[RerankedSearchResult],
    conversation_history: list[
        dict[str, str]
    ] | None = None,
) -> str:
    """Build the grounded conversational prompt."""

    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError(
            "Question cannot be empty"
        )

    evidence_context = build_evidence_context(
        evidence
    )

    conversation_context = (
        build_conversation_context(
            conversation_history
        )
    )

    return "\n".join(
        [
            "CONVERSATION HISTORY:",
            conversation_context,
            "",
            "CURRENT QUESTION:",
            normalized_question,
            "",
            "EVIDENCE:",
            evidence_context,
            "",
            "INSTRUCTIONS:",
            (
                "Use conversation history only to understand "
                "the current question."
            ),
            (
                "Use the supplied evidence as the only "
                "factual source."
            ),
            (
                "Place an exact chunk citation at the end "
                "of every factual sentence or bullet."
            ),
            (
                "Never put a citation alone on a separate "
                "line."
            ),
            "",
            "GROUNDED ANSWER:",
        ]
    )


class LLMAnswerGenerator:
    """Generate grounded answers through an LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
    ) -> None:
        self.provider = provider

    @property
    def provider_name(self) -> str:
        """Return the active provider-chain name."""

        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        """Return the primary configured model name."""

        return self.provider.model_name

    def generate(
        self,
        question: str,
        evidence: list[RerankedSearchResult],
        conversation_history: list[
            dict[str, str]
        ] | None = None,
    ) -> LLMGenerationResult:
        """Generate an evidence-grounded conversational answer."""

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "Question cannot be empty"
            )

        if not evidence:
            return LLMGenerationResult(
                text=REFUSAL_TEXT,
                provider="evidenceflow",
                model_name="retrieval-refusal",
            )

        user_prompt = build_user_prompt(
            question=normalized_question,
            evidence=evidence,
            conversation_history=(
                conversation_history
            ),
        )

        return self.provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )