"""Groq and Gemini LLM provider implementations."""

from __future__ import annotations

from evidenceflow.llm.models import (
    LLMGenerationResult,
)


class GroqProvider:
    """Generate grounded answers using Groq."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key cannot be empty")

        if not model_name.strip():
            raise ValueError(
                "Groq model name cannot be empty"
            )

        if temperature < 0:
            raise ValueError(
                "temperature cannot be negative"
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero"
            )

        self.api_key = api_key
        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""

        return "groq"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""

        return self._model_name

    def _get_client(self):
        """Create the Groq client when first required."""

        if self._client is None:
            from groq import Groq

            self._client = Groq(
                api_key=self.api_key
            )

        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMGenerationResult:
        """Generate text using Groq chat completions."""

        if not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty"
            )

        if not user_prompt.strip():
            raise ValueError(
                "user_prompt cannot be empty"
            )

        client = self._get_client()

        completion = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not completion.choices:
            raise RuntimeError(
                "Groq returned no completion choices"
            )

        content = completion.choices[0].message.content

        if content is None or not content.strip():
            raise RuntimeError(
                "Groq returned an empty response"
            )

        return LLMGenerationResult(
            text=content.strip(),
            provider=self.provider_name,
            model_name=self.model_name,
        )


class GeminiProvider:
    """Generate grounded answers using Google Gemini."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> None:
        if not api_key.strip():
            raise ValueError(
                "Gemini API key cannot be empty"
            )

        if not model_name.strip():
            raise ValueError(
                "Gemini model name cannot be empty"
            )

        if temperature < 0:
            raise ValueError(
                "temperature cannot be negative"
            )

        if max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens must be greater than zero"
            )

        self.api_key = api_key
        self._model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""

        return "gemini"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""

        return self._model_name

    def _get_client(self):
        """Create the Gemini client when first required."""

        if self._client is None:
            from google import genai

            self._client = genai.Client(
                api_key=self.api_key
            )

        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMGenerationResult:
        """Generate text using Gemini."""

        if not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty"
            )

        if not user_prompt.strip():
            raise ValueError(
                "user_prompt cannot be empty"
            )

        from google.genai import types

        client = self._get_client()

        response = client.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        response_text = response.text

        if response_text is None or not response_text.strip():
            raise RuntimeError(
                "Gemini returned an empty response"
            )

        return LLMGenerationResult(
            text=response_text.strip(),
            provider=self.provider_name,
            model_name=self.model_name,
        )