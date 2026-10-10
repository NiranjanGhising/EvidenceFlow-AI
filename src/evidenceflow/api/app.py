"""FastAPI application factory for EvidenceFlow AI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from evidenceflow import __version__
from evidenceflow.api.dependencies import (
    get_application_services,
    get_settings,
)
from evidenceflow.api.routes.pipeline import router as pipeline_router
from evidenceflow.api.schemas import HealthResponse
from evidenceflow.core.logging import configure_logging


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """Initialize shared services during application startup."""

    del application

    settings = get_settings()
    configure_logging(settings.log_level)

    get_application_services()

    yield


def create_app() -> FastAPI:
    """Create and configure the EvidenceFlow FastAPI application."""

    settings = get_settings()

    application = FastAPI(
        title="EvidenceFlow AI",
        description=(
            "Evidence-grounded document ingestion, retrieval, "
            "reranking, answering, and evaluation API"
        ),
        version=__version__,
        lifespan=lifespan,
    )

    application.include_router(pipeline_router)

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
    )
    def health_check() -> HealthResponse:
        """Return service health and version information."""

        return HealthResponse(
            status="ok",
            service="EvidenceFlow AI",
            version=__version__,
            environment=settings.environment,
        )

    return application


app = create_app()