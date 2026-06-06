#!/usr/bin/env python3
"""
Moonshot Module
Provides functions to interact with Kimi models via the Moonshot OpenAI-compatible API.
"""

from openai import OpenAI
from typing import Optional, Dict, Any
import os


class MoonshotClient:
    """Wrapper class for Moonshot API interactions."""

    DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set MOONSHOT_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        timeout_s = float(os.environ.get("MOONSHOT_HTTP_TIMEOUT", "600"))
        max_retries = int(os.environ.get("MOONSHOT_MAX_RETRIES", "2"))
        url = base_url or os.environ.get("MOONSHOT_BASE_URL") or self.DEFAULT_BASE_URL
        self.client = OpenAI(api_key=key, base_url=url, timeout=timeout_s, max_retries=max_retries)

    def generate_response(
        self,
        prompt: str,
        model: str = "kimi-k2.6",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        api_params: Dict[str, Any] = {"model": model, "messages": messages}
        # kimi-k2.x models only accept temperature=1
        if model.lower().startswith("kimi-k2"):
            temperature = 1
        if temperature is not None:
            api_params["temperature"] = temperature
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        api_params.update(kwargs)

        response = self.client.chat.completions.create(**api_params)
        return response.choices[0].message.content


def generate_response(
    prompt: str,
    model: str = "kimi-k2.6",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    client = MoonshotClient(api_key=api_key)
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
    model: str = "kimi-k2.6",
    language: Optional[str] = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
) -> str:
    system_msg = "You are an expert programmer. Generate clean, efficient, well-commented code."
    if language:
        system_msg += f" Always use {language}."

    client = MoonshotClient(api_key=api_key)
    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_message=system_msg,
    )


if __name__ == "__main__":
    print(generate_response("Hello"))
