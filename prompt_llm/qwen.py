#!/usr/bin/env python3
"""
Qwen Module
Provides functions to interact with Qwen models via an OpenAI-compatible API.
"""

from openai import OpenAI
from typing import Optional, Dict, Any
import os


class QwenClient:
    """Wrapper class for Qwen API interactions."""

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Qwen client.

        Args:
            api_key: Qwen API key. If None, uses QWEN_API_KEY environment variable.
            base_url: Optional OpenAI-compatible API base URL. Defaults to DashScope.
        """
        key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set QWEN_API_KEY or DASHSCOPE_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        timeout_s = float(os.environ.get("QWEN_HTTP_TIMEOUT", "600"))
        max_retries = int(os.environ.get("QWEN_MAX_RETRIES", "2"))
        url = base_url or os.environ.get("QWEN_BASE_URL") or self.DEFAULT_BASE_URL
        self.client = OpenAI(api_key=key, base_url=url, timeout=timeout_s, max_retries=max_retries)

    def generate_response(
        self,
        prompt: str,
        model: str = "qwen2.5-7B",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a response from a Qwen model using chat completion."""
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

        # Disable thinking mode for qwen3 models (thinking generates very long
        # internal reasoning tokens that cause connection timeouts on large prompts).
        model_lower = model.lower()
        if any(model_lower.startswith(p) for p in ("qwen3", "qwen3.")):
            extra_body = kwargs.pop("extra_body", {})
            extra_body.setdefault("enable_thinking", False)
            kwargs["extra_body"] = extra_body

        api_params.update(kwargs)
        response = self.client.chat.completions.create(**api_params)
        return response.choices[0].message.content


def generate_response(
    prompt: str,
    model: str = "qwen2.5-7B",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    """Simple function to generate a response from a Qwen model."""
    client = QwenClient(api_key=api_key)
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
    model: str = "qwen2.5-7B",
    language: Optional[str] = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
) -> str:
    """Generate code based on prompt."""
    system_msg = "You are an expert programmer. Generate clean, efficient, well-commented code."
    if language:
        system_msg += f" Always use {language}."

    client = QwenClient(api_key=api_key)
    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_message=system_msg,
    )


def main():
    """Quick smoke test for the Qwen module."""
    print(generate_response("Hello"))


if __name__ == "__main__":
    main()
