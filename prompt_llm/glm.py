#!/usr/bin/env python3
"""
GLM Module
Provides functions to interact with GLM models via an OpenAI-compatible API.
"""

from openai import OpenAI
from typing import Optional, Dict, Any
import os


class GLMClient:
    """Wrapper class for GLM API interactions."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize GLM client.

        Args:
            api_key: GLM API key. If None, uses GLM_API_KEY environment variable.
            base_url: Optional OpenAI-compatible API base URL. Defaults to OpenRouter.
        """
        key = api_key or os.environ.get("GLM_API_KEY")
        if not key:
            raise ValueError("No API key provided. Set GLM_API_KEY environment variable or pass api_key parameter.")

        timeout_s = float(os.environ.get("GLM_HTTP_TIMEOUT", "600"))
        max_retries = int(os.environ.get("GLM_MAX_RETRIES", "2"))
        url = base_url or os.environ.get("GLM_BASE_URL") or self.DEFAULT_BASE_URL
        self.client = OpenAI(api_key=key, base_url=url, timeout=timeout_s, max_retries=max_retries)

    def generate_response(
        self,
        prompt: str,
        model: str = "z-ai/glm-4.5-air:free",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a response from a GLM model using chat completion."""
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
    model: str = "z-ai/glm-4.5-air:free",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    """Simple function to generate a response from a GLM model."""
    client = GLMClient(api_key=api_key)
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
    model: str = "z-ai/glm-4.5-air:free",
    language: Optional[str] = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
) -> str:
    """Generate code based on prompt."""
    system_msg = "You are an expert programmer. Generate clean, efficient, well-commented code."
    if language:
        system_msg += f" Always use {language}."

    client = GLMClient(api_key=api_key)
    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_message=system_msg,
    )


def main():
    """Quick smoke test for the GLM module."""
    print(generate_response("Hello"))


if __name__ == "__main__":
    main()
