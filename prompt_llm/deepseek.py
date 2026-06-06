#!/usr/bin/env python3
"""
DeepSeek Module
Provides functions to interact with DeepSeek models via the OpenAI-compatible API.
"""

from openai import OpenAI
from typing import Optional, Dict, Any, List
import os


# Models that route through DeepSeek's reasoning ("thinking") path. They
# reject custom temperatures and instead accept reasoning_effort plus an
# extra_body thinking flag. Match by prefix so future variants are picked
# up automatically (e.g. deepseek-v4-pro-mini).
_REASONING_MODEL_PREFIXES = (
    "deepseek-r",
    "deepseek-reasoner",
    "deepseek-v4-pro",
)


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


class DeepSeekClient:
    """Wrapper class for DeepSeek API interactions (OpenAI-compatible)."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY environment variable.
            base_url: Optional base URL for API. Defaults to https://api.deepseek.com.
        """
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )
        url = base_url or self.DEFAULT_BASE_URL

        # Mirror gpt.py's per-request HTTP timeout and retry budget so a
        # single hung reasoning request can't stall the whole eval run.
        timeout_s = float(os.environ.get("DEEPSEEK_HTTP_TIMEOUT", "600"))
        max_retries = int(os.environ.get("DEEPSEEK_MAX_RETRIES", "2"))
        self.client = OpenAI(
            api_key=key,
            base_url=url,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    def generate_response(
        self,
        prompt: str,
        model: str = "deepseek-v4-pro",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from DeepSeek using chat completion.

        Args:
            prompt: The user prompt/question to send to the model.
            model: Model name (default: "deepseek-v4-pro").
            temperature: Sampling temperature. Pass None to omit (required for reasoning models).
            max_tokens: Maximum tokens in response. None for model default.
            system_message: Optional system message to set context/behavior.
            **kwargs: Additional parameters forwarded to the API.

        Returns:
            str: The generated response text.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        api_params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            api_params["temperature"] = temperature
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        api_params.update(kwargs)

        response = self.client.chat.completions.create(**api_params)
        return response.choices[0].message.content


def generate_response(
    prompt: str,
    model: str = "deepseek-v4-pro",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """Simple function to generate a response from a DeepSeek model."""
    client = DeepSeekClient(api_key=api_key)
    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_message=system_message,
        **kwargs,
    )


def generate_code(
    prompt: str,
    model: str = "deepseek-v4-pro",
    language: Optional[str] = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate code based on prompt. Reasoning models drop the temperature
    knob and accept reasoning_effort + extra_body.thinking instead; non-
    reasoning models (e.g. deepseek-chat) keep the standard temperature path.
    """
    system_msg = "You are an expert programmer. Generate clean, efficient, well-commented code."
    if language:
        system_msg += f" Always use {language}."

    client = DeepSeekClient(api_key=api_key)

    extra: Dict[str, Any] = {}
    if _is_reasoning_model(model):
        # Reasoning models reject temperature != 1.0. Forward an effort
        # hint (default "high") and turn on the thinking body so the
        # model actually uses the reasoning path. Override the default
        # via DEEPSEEK_REASONING_EFFORT.
        effort = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high")
        extra["reasoning_effort"] = effort
        extra["extra_body"] = {"thinking": {"type": "enabled"}}
        temp_to_use: Optional[float] = None
    else:
        temp_to_use = temperature

    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=temp_to_use,
        system_message=system_msg,
        **extra,
    )


def main():
    """Quick smoke test for the DeepSeek module."""
    print("Example 1: Simple response generation")
    print("-" * 60)
    print(generate_response("Hello"))
    print()

    print("Example 2: Code generation")
    print("-" * 60)
    print(generate_code(
        "Write a simple CUDA kernel for element-wise array addition",
        language="CUDA",
    ))


if __name__ == "__main__":
    main()
