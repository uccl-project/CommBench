#!/usr/bin/env python3
"""
OpenAI GPT Module
Provides functions to interact with OpenAI's GPT models.
"""

from openai import OpenAI
from typing import Optional, Dict, Any, List
import os


class GPTClient:
    """Wrapper class for OpenAI GPT API interactions."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize GPT client.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
            base_url: Optional base URL for API. Useful for custom endpoints.
        """
        # Per-request HTTP timeout and retry budget. Without this the SDK's
        # default is no timeout, which can make a single request hang for
        # tens of minutes on reasoning models. Override via env vars
        # OPENAI_HTTP_TIMEOUT / OPENAI_MAX_RETRIES.
        timeout_s = float(os.environ.get("OPENAI_HTTP_TIMEOUT", "600"))
        max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
        client_kwargs: Dict[str, Any] = {
            "timeout": timeout_s,
            "max_retries": max_retries,
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def generate_response(
        self,
        prompt: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from GPT model using chat completion.

        Args:
            prompt: The user prompt/question to send to the model.
            model: Model name (default: "gpt-4o").
            temperature: Sampling temperature (0-2). Higher values make output more random.
            max_tokens: Maximum tokens in response. None for model default.
            system_message: Optional system message to set context/behavior.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            str: The generated response text.

        Example:
            >>> client = GPTClient()
            >>> response = client.generate_response("Explain GPU memory hierarchy")
            >>> print(response)
        """
        messages = []

        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        # Prepare API parameters
        api_params = {
            "model": model,
            "messages": messages,
        }
        # Temperature is only set when explicitly provided. Reasoning models
        # (gpt-5*, o1, o3, ...) reject any temperature != 1.0, so callers
        # pass temperature=None to skip it.
        if temperature is not None:
            api_params["temperature"] = temperature

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        # Add any additional parameters
        api_params.update(kwargs)

        # Make API call
        response = self.client.chat.completions.create(**api_params)

        return response.choices[0].message.content

    def generate_response_with_context(
        self,
        prompt: str,
        context_messages: List[Dict[str, str]],
        model: str = "gpt-4o",
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

        Example:
            >>> client = GPTClient()
            >>> context = [
            ...     {"role": "user", "content": "What is CUDA?"},
            ...     {"role": "assistant", "content": "CUDA is NVIDIA's parallel computing platform..."}
            ... ]
            >>> response = client.generate_response_with_context(
            ...     "How does it compare to HIP?",
            ...     context
            ... )
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
        model: str = "gpt-4o",
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

        Example:
            >>> client = GPTClient()
            >>> result = client.generate_response_full("Explain GPU kernels")
            >>> print(result['content'])
            >>> print(f"Tokens used: {result['usage']['total_tokens']}")
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

        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "finish_reason": response.choices[0].finish_reason,
        }


# Convenience functions for simple use cases

def generate_response(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Simple function to generate a response from GPT model.

    Args:
        prompt: The user prompt/question.
        model: Model name (default: "gpt-4o").
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens in response.
        system_message: Optional system message.
        api_key: Optional API key. Uses environment variable if not provided.
        **kwargs: Additional parameters.

    Returns:
        str: The generated response text.

    Example:
        >>> from gpt import generate_response
        >>> response = generate_response("Explain GPU P2P communication")
        >>> print(response)
    """
    client = GPTClient(api_key=api_key)
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
    model: str = "gpt-4o",
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

    Example:
        >>> from gpt import generate_code
        >>> code = generate_code(
        ...     "Write a CUDA kernel for vector addition",
        ...     language="CUDA"
        ... )
        >>> print(code)
    """
    system_msg = "You are an expert programmer. Generate clean, efficient, well-commented code."
    if language:
        system_msg += f" Always use {language}."

    client = GPTClient(api_key=api_key)

    # Reasoning models (gpt-5*, o1, o3, ...) don't support arbitrary
    # temperature, but they do accept an optional reasoning_effort hint
    # that controls how much chain-of-thought to spend. Detect those and:
    #   - drop the temperature parameter,
    #   - forward reasoning_effort if OPENAI_REASONING_EFFORT is set.
    model_lower = model.lower()
    is_reasoning = (
        model_lower.startswith("gpt-5")
        or model_lower.startswith("o1")
        or model_lower.startswith("o3")
        or model_lower.startswith("o4")
    )

    extra: Dict[str, Any] = {}
    effort = os.environ.get("OPENAI_REASONING_EFFORT")
    if is_reasoning and effort:
        extra["reasoning_effort"] = effort

    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=None if is_reasoning else temperature,
        system_message=system_msg,
        **extra,
    )


def ask_question(
    question: str,
    context: Optional[str] = None,
    model: str = "gpt-4o",
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

    Example:
        >>> from gpt import ask_question
        >>> answer = ask_question(
        ...     "How does this work?",
        ...     context="CUDA streams allow concurrent kernel execution..."
        ... )
        >>> print(answer)
    """
    if context:
        full_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        full_prompt = question

    client = GPTClient(api_key=api_key)
    return client.generate_response(prompt=full_prompt, model=model)


def main():
    """Example usage of the GPT module."""
    # Example 1: Simple response generation
    print("Example 1: Simple response generation")
    print("-" * 60)
    response = generate_response("Explain rANS in one paragraph for a GPU engineer.")
    print(response)
    print()

    # Example 2: Code generation
    print("Example 2: Code generation")
    print("-" * 60)
    code = generate_code(
        "Write a simple CUDA kernel for element-wise array addition",
        language="CUDA",
    )
    print(code)
    print()

    # Example 3: Using GPTClient class
    print("Example 3: Using GPTClient class with full response")
    print("-" * 60)
    client = GPTClient()
    result = client.generate_response_full(
        prompt="What is GPU memory coalescing?",
        system_message="You are a GPU programming expert. Be concise.",
        max_tokens=150,
    )
    print(f"Response: {result['content']}")
    print(f"Tokens used: {result['usage']['total_tokens']}")
    print()

    # Example 4: Conversation with context
    print("Example 4: Conversation with context")
    print("-" * 60)
    context = [
        {"role": "user", "content": "What is CUDA?"},
        {
            "role": "assistant",
            "content": "CUDA is NVIDIA's parallel computing platform and programming model.",
        },
    ]
    follow_up = client.generate_response_with_context(
        "How does it differ from OpenCL?", context
    )
    print(follow_up)


if __name__ == "__main__":
    main()
