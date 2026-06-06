#!/usr/bin/env python3
"""
xAI Grok Module
Provides functions to interact with xAI's Grok models via the OpenAI-compatible API.
"""

from openai import OpenAI
from typing import Optional, Dict, Any, List
import os


class GrokClient:
    """Wrapper class for xAI Grok API interactions."""

    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Grok client.

        Args:
            api_key: xAI API key. If None, uses XAI_API_KEY or GROK_API_KEY environment variable.
            base_url: Optional base URL for API. Defaults to https://api.x.ai/v1.
        """
        key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set XAI_API_KEY or GROK_API_KEY environment variable, "
                "or pass api_key parameter."
            )
        url = base_url or self.DEFAULT_BASE_URL
        self.client = OpenAI(api_key=key, base_url=url)

    def generate_response(
        self,
        prompt: str,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from Grok model using chat completion.

        Args:
            prompt: The user prompt/question to send to the model.
            model: Model name (default: "grok-3").
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response. None for model default.
            system_message: Optional system message to set context/behavior.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            str: The generated response text.
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        api_params.update(kwargs)

        response = self.client.chat.completions.create(**api_params)

        return response.choices[0].message.content

    def generate_response_with_context(
        self,
        prompt: str,
        context_messages: List[Dict[str, str]],
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response with conversation context.

        Args:
            prompt: The user prompt/question.
            context_messages: List of previous messages in format:
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional parameters.

        Returns:
            str: The generated response text.
        """
        messages = context_messages + [{"role": "user", "content": prompt}]

        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        api_params.update(kwargs)

        response = self.client.chat.completions.create(**api_params)

        return response.choices[0].message.content

    def generate_response_full(
        self,
        prompt: str,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response and return full response object with metadata.

        Args:
            prompt: The user prompt/question.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            system_message: Optional system message.
            **kwargs: Additional parameters.

        Returns:
            Dict containing:
                - content: The response text
                - model: Model used
                - usage: Token usage statistics
                - finish_reason: Why the generation stopped
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        api_params.update(kwargs)

        response = self.client.chat.completions.create(**api_params)

        result = {
            "content": response.choices[0].message.content,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
        }

        if hasattr(response, 'usage') and response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return result


# Convenience functions for simple use cases

def generate_response(
    prompt: str,
    model: str = "grok-3",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Simple function to generate a response from Grok model.

    Args:
        prompt: The user prompt/question.
        model: Model name (default: "grok-3").
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens in response.
        system_message: Optional system message.
        api_key: Optional API key. Uses environment variable if not provided.
        **kwargs: Additional parameters.

    Returns:
        str: The generated response text.
    """
    client = GrokClient(api_key=api_key)
    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_message=system_message,
        **kwargs
    )


def generate_code(
    prompt: str,
    model: str = "grok-3",
    language: Optional[str] = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate code based on prompt. Uses lower temperature for more deterministic output.

    Args:
        prompt: Description of the code to generate.
        model: Model name.
        language: Programming language (e.g., "Python", "C++", "CUDA").
        temperature: Sampling temperature (default 0.3 for code generation).
        api_key: Optional API key.

    Returns:
        str: Generated code.
    """
    system_msg = "You are an expert programmer. Generate clean, efficient, well-commented code."
    if language:
        system_msg += f" Always use {language}."

    client = GrokClient(api_key=api_key)
    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_message=system_msg,
    )


def ask_question(
    question: str,
    context: Optional[str] = None,
    model: str = "grok-3",
    api_key: Optional[str] = None,
) -> str:
    """
    Ask a question with optional context.

    Args:
        question: The question to ask.
        context: Optional context information.
        model: Model name.
        api_key: Optional API key.

    Returns:
        str: The answer.
    """
    if context:
        full_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        full_prompt = question

    client = GrokClient(api_key=api_key)
    return client.generate_response(prompt=full_prompt, model=model)


def main():
    """Example usage of the Grok module."""
    print("Example 1: Simple response generation")
    print("-" * 60)
    response = generate_response("Explain all-reduce in multi-GPU systems in one paragraph.")
    print(response)
    print()

    print("Example 2: Code generation")
    print("-" * 60)
    code = generate_code(
        "Write a simple CUDA kernel for element-wise array addition",
        language="CUDA",
    )
    print(code)
    print()

    print("Example 3: Using GrokClient class with full response")
    print("-" * 60)
    client = GrokClient()
    result = client.generate_response_full(
        prompt="What is GPU memory coalescing?",
        system_message="You are a GPU programming expert. Be concise.",
        max_tokens=150,
    )
    print(f"Response: {result['content']}")
    if 'usage' in result:
        print(f"Tokens used: {result['usage']['total_tokens']}")
    print()


if __name__ == "__main__":
    main()
