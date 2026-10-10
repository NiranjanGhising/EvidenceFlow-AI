"""Streamlit user interface for EvidenceFlow AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from evidenceflow.ui.api_client import (
    EvidenceFlowAPIClient,
    EvidenceFlowAPIError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENT_DIRECTORY = PROJECT_ROOT / "data" / "documents"

DOCUMENT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def initialize_session_state() -> None:
    """Initialize Streamlit session values."""

    default_values: dict[str, Any] = {
        "processed_document": None,
        "answer_result": None,
        "search_result": None,
        "chat_messages": [],
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_api_client() -> EvidenceFlowAPIClient:
    """Create the EvidenceFlow API client."""

    base_url = os.getenv(
        "EVIDENCEFLOW_API_URL",
        "http://127.0.0.1:8000",
    )

    return EvidenceFlowAPIClient(
        base_url=base_url,
        timeout_seconds=180.0,
    )


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Save an uploaded document to the documents directory."""

    filename = Path(uploaded_file.name).name

    if not filename:
        raise ValueError(
            "The uploaded file does not have a valid filename."
        )

    destination = DOCUMENT_DIRECTORY / filename

    destination.write_bytes(
        uploaded_file.getvalue()
    )

    return destination.resolve()


def render_sidebar(
    client: EvidenceFlowAPIClient,
) -> tuple[int, int, int, int, float | None]:
    """Render API status and pipeline settings."""

    with st.sidebar:
        st.header("EvidenceFlow AI")

        st.caption(
            "Document intelligence with hybrid retrieval, "
            "reranking, and validated citations."
        )

        st.divider()
        st.subheader("API status")

        try:
            health = client.health()

            st.success(
                f"{health.get('service', 'EvidenceFlow AI')} is online"
            )

            st.caption(
                f"Version: {health.get('version', 'unknown')}"
            )

            st.caption(
                f"Environment: "
                f"{health.get('environment', 'unknown')}"
            )

        except EvidenceFlowAPIError as exc:
            st.error(str(exc))

        st.divider()
        st.subheader("Pipeline settings")

        chunk_size = st.number_input(
            "Chunk size",
            min_value=100,
            max_value=4000,
            value=800,
            step=100,
        )

        overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=1000,
            value=120,
            step=20,
        )

        candidate_limit = st.number_input(
            "Retrieval candidates",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
        )

        result_limit = st.number_input(
            "Final result limit",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )

        minimum_rerank_score = None

        with st.expander(
            "Advanced settings",
            expanded=False,
        ):
            use_rerank_threshold = st.checkbox(
                "Enable manual rerank threshold",
                value=False,
                help=(
                    "Leave disabled to use retrieval signals "
                    "and LLM evidence refusal automatically."
                ),
            )

            if use_rerank_threshold:
                minimum_rerank_score = st.number_input(
                    "Minimum rerank score",
                    value=0.0,
                    step=0.5,
                    format="%.2f",
                    help=(
                        "Advanced override for experimentation "
                        "and evaluation."
                    ),
                )

    return (
        int(chunk_size),
        int(overlap),
        int(candidate_limit),
        int(result_limit),
                (
            float(minimum_rerank_score)
            if minimum_rerank_score is not None
            else None
        ),
    )


def render_document_processing(
    client: EvidenceFlowAPIClient,
    chunk_size: int,
    overlap: int,
) -> None:
    """Render document upload and processing."""

    st.subheader("Document processing")

    st.write(
        "Upload a PDF, Markdown, or text document. "
        "EvidenceFlow will ingest, chunk, and index it."
    )

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "md", "txt"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info(
            "Upload a supported document to begin processing."
        )
        return

    file_size_kb = len(uploaded_file.getvalue()) / 1024

    file_column, size_column = st.columns(2)

    file_column.metric(
        "File",
        uploaded_file.name,
    )

    size_column.metric(
        "Size",
        f"{file_size_kb:.1f} KB",
    )

    if overlap >= chunk_size:
        st.error(
            "Chunk overlap must be smaller than chunk size."
        )
        return

    if st.button(
        "Process document",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.status(
                "Processing document...",
                expanded=True,
            ) as processing_status:
                st.write("Saving uploaded file")

                document_path = save_uploaded_file(
                    uploaded_file
                )

                st.write("Ingesting document")

                ingestion = client.ingest_document(
                    str(document_path)
                )

                document_id = str(
                    ingestion["document_id"]
                )

                st.write("Creating chunks")

                chunking = client.chunk_document(
                    document_id=document_id,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                st.write("Generating embeddings")

                indexing = client.index_document(
                    document_id=document_id
                )

                processing_status.update(
                    label="Document processing complete",
                    state="complete",
                    expanded=False,
                )

            st.session_state.processed_document = {
                "path": str(document_path),
                "ingestion": ingestion,
                "chunking": chunking,
                "indexing": indexing,
            }

            st.success(
                f"Processed {ingestion['source_name']} successfully."
            )

        except (
            EvidenceFlowAPIError,
            OSError,
            ValueError,
            KeyError,
        ) as exc:
            st.error(str(exc))

    processed_document = (
        st.session_state.processed_document
    )

    if processed_document is None:
        return

    ingestion = processed_document["ingestion"]
    chunking = processed_document["chunking"]
    indexing = processed_document["indexing"]

    st.divider()
    st.markdown("#### Processing summary")

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Outcome",
        str(ingestion["outcome"]).title(),
    )

    metric_columns[1].metric(
        "Chunks",
        int(chunking["chunk_count"]),
    )

    metric_columns[2].metric(
        "Vectors",
        int(indexing["vector_count"]),
    )

    metric_columns[3].metric(
        "Dimensions",
        int(indexing["dimension"]),
    )

    with st.expander(
        "Document and indexing details"
    ):
        st.json(processed_document)


def render_citations(
    citations: list[dict[str, Any]],
) -> None:
    """Render structured citations."""

    if not citations:
        st.info("No citations were returned.")
        return

    for citation in citations:
        heading_path = citation.get(
            "heading_path",
            [],
        )

        heading_text = (
            " > ".join(heading_path)
            if heading_path
            else "No heading"
        )

        with st.expander(
            f"{citation['source_name']} · {heading_text}"
        ):
            st.write(
                f"Citation ID: `{citation['citation_id']}`"
            )

            st.write(
                f"Document ID: `{citation['document_id']}`"
            )

            st.write(
                f"Chunk ID: `{citation['chunk_id']}`"
            )


def render_evidence(
    evidence_items: list[dict[str, Any]],
) -> None:
    """Render evidence and score diagnostics."""

    if not evidence_items:
        st.info("No supporting evidence was selected.")
        return

    for evidence in evidence_items:
        title = (
            f"Rank {evidence['rerank_rank']} · "
            f"{evidence['source_name']}"
        )

        with st.expander(title):
            columns = st.columns(3)

            columns[0].metric(
                "Rerank score",
                f"{evidence['rerank_score']:.4f}",
            )

            columns[1].metric(
                "Fusion rank",
                evidence["fusion_rank"],
            )

            columns[2].metric(
                "Chunk order",
                evidence.get(
                    "chunk_order",
                    "N/A",
                ),
            )

            st.markdown("**Evidence text**")
            st.write(evidence["text"])

            st.caption(
                f"Chunk ID: `{evidence['chunk_id']}`"
            )


def render_answer_panel(
    client: EvidenceFlowAPIClient,
    candidate_limit: int,
    result_limit: int,
    minimum_rerank_score: float | None,
) -> None:
    """Render the document-scoped RAG chatbot."""

    st.subheader("Chat with your document")

    processed_document = (
        st.session_state.processed_document
    )

    if processed_document is None:
        st.info(
            "Process a document first, then return here "
            "to start a grounded conversation."
        )
        return

    ingestion = processed_document.get(
        "ingestion",
        {},
    )

    document_id = str(
        ingestion.get(
            "document_id",
            "",
        )
    ).strip()

    source_name = str(
        ingestion.get(
            "source_name",
            "Active document",
        )
    )

    if not document_id:
        st.error(
            "The active document does not have a valid "
            "document ID. Process the document again."
        )
        return

    if result_limit > candidate_limit:
        st.error(
            "Final result limit cannot exceed "
            "the candidate limit."
        )
        return

    document_column, clear_column = st.columns(
        [4, 1]
    )

    with document_column:
        st.success(
            f"Active document: {source_name}"
        )

        st.caption(
            f"Questions are restricted to document "
            f"`{document_id}`."
        )

    with clear_column:
        if st.button(
            "Clear chat",
            use_container_width=True,
            key="clear_document_chat",
        ):
            st.session_state.chat_messages = []
            st.session_state.answer_result = None
            st.rerun()

    chat_messages = (
        st.session_state.chat_messages
    )

    if not chat_messages:
        with st.chat_message("assistant"):
            st.write(
                "Your document is ready. Ask a question "
                "about its content."
            )

            st.caption(
                "EvidenceFlow will answer only from the "
                "active document and include citations."
            )

    for message_index, message in enumerate(
        chat_messages
    ):
        role = str(
            message.get(
                "role",
                "assistant",
            )
        )

        content = str(
            message.get(
                "content",
                "",
            )
        )

        with st.chat_message(role):
            st.write(content)

            if role != "assistant":
                continue

            result = message.get("result")

            if not isinstance(result, dict):
                continue

            if result.get("supported"):
                st.caption(
                    "Supported by the active document"
                )
            else:
                refusal_reason = result.get(
                    "refusal_reason"
                )

                st.caption(
                    "EvidenceFlow could not find enough "
                    "supporting evidence in the active document."
                )

                if refusal_reason:
                    st.caption(
                        f"Reason: `{refusal_reason}`"
                    )

            citations = result.get(
                "citations",
                [],
            )

            if citations:
                with st.expander(
                    "Sources",
                    expanded=False,
                ):
                    render_citations(
                        citations
                    )

            evidence_items = result.get(
                "evidence",
                [],
            )

            if evidence_items:
                with st.expander(
                    "Retrieved evidence",
                    expanded=False,
                ):
                    render_evidence(
                        evidence_items
                    )

            with st.expander(
                "Technical details",
                expanded=False,
            ):
                citation_validation = result.get(
                    "citation_validation",
                    {},
                )

                metric_columns = st.columns(4)

                metric_columns[0].metric(
                    "Supported",
                    (
                        "Yes"
                        if result.get("supported")
                        else "No"
                    ),
                )

                metric_columns[1].metric(
                    "Citations valid",
                    (
                        "Yes"
                        if citation_validation.get(
                            "valid",
                            False,
                        )
                        else "No"
                    ),
                )

                metric_columns[2].metric(
                    "Coverage",
                    (
                        f"{citation_validation.get('coverage', 0):.0%}"
                    ),
                )

                total_latency = (
                    result.get(
                        "latency_ms",
                        {},
                    ).get(
                        "total",
                        0.0,
                    )
                )

                metric_columns[3].metric(
                    "Latency",
                    f"{total_latency:.0f} ms",
                )

                st.json(
                    {
                        "document_id": document_id,
                        "retrieval_mode": result.get(
                            "retrieval_mode"
                        ),
                        "reranker_model": result.get(
                            "reranker_model"
                        ),
                        "generation_provider": result.get(
                            "generation_provider"
                        ),
                        "generation_model": result.get(
                            "generation_model"
                        ),
                        "used_reranker_fallback": (
                            result.get(
                                "used_reranker_fallback"
                            )
                        ),
                        "used_llm_fallback": result.get(
                            "used_llm_fallback"
                        ),
                        "citation_validation": (
                            citation_validation
                        ),
                        "latency_ms": result.get(
                            "latency_ms",
                            {},
                        ),
                        "warnings": result.get(
                            "warnings",
                            [],
                        ),
                        "message_index": message_index,
                    }
                )

    question = st.chat_input(
        "Ask a question about the active document"
    )

    if question is None:
        return

    normalized_question = question.strip()

    if not normalized_question:
        return

    conversation_history = [
        {
            "role": str(
                message.get(
                    "role",
                    "user",
                )
            ),
            "content": str(
                message.get(
                    "content",
                    "",
                )
            ),
        }
        for message in chat_messages
        if message.get("content")
    ]

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": normalized_question,
        }
    )

    try:
        with st.chat_message("assistant"):
            with st.spinner(
                "Retrieving evidence and generating "
                "a grounded answer..."
            ):
                result = client.answer_question(
                    question=normalized_question,
                    candidates=candidate_limit,
                    limit=result_limit,
                    minimum_rerank_score=(
                        minimum_rerank_score
                    ),
                    document_id=document_id,
                    conversation_history=(
                        conversation_history
                    ),
                )

        answer = str(
            result.get(
                "answer",
                (
                    "I could not generate a grounded "
                    "answer."
                ),
            )
        )

        st.session_state.answer_result = result

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "result": result,
            }
        )

    except EvidenceFlowAPIError as exc:
        error_message = str(exc)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": (
                    "I could not complete the request. "
                    f"{error_message}"
                ),
            }
        )

    st.rerun()


def render_search_panel(
    client: EvidenceFlowAPIClient,
    candidate_limit: int,
    result_limit: int,
) -> None:
    """Render hybrid search diagnostics."""

    st.subheader("Hybrid retrieval explorer")

    query = st.text_input(
        "Search query",
        placeholder=(
            "Example: FastAPI request validation"
        ),
    )

    if result_limit > candidate_limit:
        st.error(
            "Final result limit cannot exceed "
            "the candidate limit."
        )
        return

    if st.button(
        "Run hybrid search",
        use_container_width=True,
    ):
        if not query.strip():
            st.warning("Enter a search query first.")
        else:
            try:
                with st.spinner(
                    "Running hybrid retrieval..."
                ):
                    result = client.hybrid_search(
                        query=query,
                        limit=result_limit,
                        candidates=candidate_limit,
                    )

                st.session_state.search_result = result

            except EvidenceFlowAPIError as exc:
                st.error(str(exc))

    result = st.session_state.search_result

    if result is None:
        return

    st.divider()

    summary_columns = st.columns(3)

    summary_columns[0].metric(
        "Results",
        result["result_count"],
    )

    summary_columns[1].metric(
        "Mode",
        result["retrieval_mode"],
    )

    summary_columns[2].metric(
        "Total latency",
        f"{result['latency_ms']['total']:.0f} ms",
    )

    for search_result in result["results"]:
        title = (
            f"Rank {search_result['rank']} · "
            f"{search_result['source_name']}"
        )

        with st.expander(title):
            score_columns = st.columns(3)

            fusion_score = search_result.get(
                "fusion_score"
            )

            sparse_score = search_result.get(
                "sparse_score"
            )

            dense_score = search_result.get(
                "dense_score"
            )

            score_columns[0].metric(
                "Fusion score",
                (
                    f"{fusion_score:.6f}"
                    if fusion_score is not None
                    else "N/A"
                ),
            )

            score_columns[1].metric(
                "BM25 score",
                (
                    f"{sparse_score:.6f}"
                    if sparse_score is not None
                    else "N/A"
                ),
            )

            score_columns[2].metric(
                "Dense score",
                (
                    f"{dense_score:.6f}"
                    if dense_score is not None
                    else "N/A"
                ),
            )

            st.write(search_result["text"])

            st.caption(
                f"Chunk ID: "
                f"`{search_result['chunk_id']}`"
            )


def main() -> None:
    """Render the EvidenceFlow Streamlit application."""

    st.set_page_config(
        page_title="EvidenceFlow AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    client = get_api_client()

    (
        chunk_size,
        overlap,
        candidate_limit,
        result_limit,
        minimum_rerank_score,
    ) = render_sidebar(client)

    st.title("EvidenceFlow AI")

    st.caption(
        "Evidence-grounded document intelligence with "
        "hybrid retrieval, reranking, and citations."
    )

    processing_tab, answer_tab, search_tab = st.tabs(
        [
            "Process documents",
            "Ask questions",
            "Explore retrieval",
        ]
    )

    with processing_tab:
        render_document_processing(
            client=client,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    with answer_tab:
        render_answer_panel(
            client=client,
            candidate_limit=candidate_limit,
            result_limit=result_limit,
            minimum_rerank_score=minimum_rerank_score,
        )

    with search_tab:
        render_search_panel(
            client=client,
            candidate_limit=candidate_limit,
            result_limit=result_limit,
        )


if __name__ == "__main__":
    main()