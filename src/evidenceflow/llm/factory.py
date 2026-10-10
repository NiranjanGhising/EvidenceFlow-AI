"""Create the configured Groq and Gemini provider chain."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from evidenceflow.llm.fallback import FallbackLLMProvider
from evidenceflow.llm.generator import LLMAnswerGenerator
from evidenceflow.llm.providers import (
    GeminiProvider,
    GroqProvider,
)


def get_required_environment_variable(
    variable_name: str,
) -> str:
    """Return a required environment variable."""

    value = os.getenv(
        variable_name,
        "",
    ).strip()

    if not value:
        raise ValueError(
            f"Required environment variable "
            f"'{variable_name}' is not configured"
        )

    return value


def build_llm_answer_generator() -> LLMAnswerGenerator:
    """Build Groq primary with optional Gemini fallback."""

    load_dotenv()
    groq_api_key = get_required_environment_variable(
        "GROQ_API_KEY"
    )

    groq_model = os.getenv(
        "EVIDENCEFLOW_GROQ_MODEL",
        "llama-3.3-70b-versatile",
    ).strip()

    primary_provider = GroqProvider(
        api_key=groq_api_key,
        model_name=groq_model,
        temperature=0.0,
        max_tokens=1024,
    )

    gemini_api_key = os.getenv(
        "GEMINI_API_KEY",
        "",
    ).strip()

    fallback_provider = None

    if gemini_api_key:
        gemini_model = os.getenv(
            "EVIDENCEFLOW_GEMINI_MODEL",
            "gemini-2.5-flash",
        ).strip()

        fallback_provider = GeminiProvider(
            api_key=gemini_api_key,
            model_name=gemini_model,
            temperature=0.0,
            max_output_tokens=1024,
        )

    provider_chain = FallbackLLMProvider(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )

    return LLMAnswerGenerator(
        provider=provider_chain
    )