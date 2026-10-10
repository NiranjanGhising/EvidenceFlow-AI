"""Primary LLM provider with automatic fallback support."""

from __future__ import annotations

import logging

from evidenceflow.llm.models import (
    LLMGenerationResult,
    LLMProvider,
)

logger = logging.getLogger(__name__)


class FallbackLLMProvider:
    """Use a primary LLM and switch to a fallback when required."""

    def __init__(
        self,
        primary_provider: LLMProvider,
        fallback_provider: LLMProvider | None = None,
    ) -> None:
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider

    @property
    def provider_name(self) -> str:
        """Return the configured provider-chain name."""

        if self.fallback_provider is None:
            return self.primary_provider.provider_name

        return (
            f"{self.primary_provider.provider_name}"
            f"_with_{self.fallback_provider.provider_name}_fallback"
        )

    @property
    def model_name(self) -> str:
        """Return the primary provider model name."""

        return self.primary_provider.model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMGenerationResult:
        """Generate with the primary provider or use the fallback."""

        if not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty"
            )

        if not user_prompt.strip():
            raise ValueError(
                "user_prompt cannot be empty"
            )

        try:
            return self.primary_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        except Exception as primary_error:
            if self.fallback_provider is None:
                raise

            logger.warning(
                "primary_llm_failed_using_fallback",
                extra={
                    "primary_provider": (
                        self.primary_provider.provider_name
                    ),
                    "primary_model": (
                        self.primary_provider.model_name
                    ),
                    "fallback_provider": (
                        self.fallback_provider.provider_name
                    ),
                    "fallback_model": (
                        self.fallback_provider.model_name
                    ),
                    "primary_error_type": (
                        type(primary_error).__name__
                    ),
                    "primary_error_message": str(
                        primary_error
                    ),
                },
            )

            try:
                fallback_result = (
                    self.fallback_provider.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                )

            except Exception as fallback_error:
                raise RuntimeError(
                    "Both the primary and fallback LLM "
                    "providers failed. "
                    f"Primary error: "
                    f"{type(primary_error).__name__}: "
                    f"{primary_error}. "
                    f"Fallback error: "
                    f"{type(fallback_error).__name__}: "
                    f"{fallback_error}."
                ) from fallback_error

            return LLMGenerationResult(
                text=fallback_result.text,
                provider=fallback_result.provider,
                model_name=fallback_result.model_name,
                used_fallback=True,
                fallback_reason=(
                    f"{type(primary_error).__name__}: "
                    f"{primary_error}"
                ),
            )