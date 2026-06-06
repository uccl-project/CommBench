#!/usr/bin/env python3
"""
Google Gemini Module
Provides functions to interact with Google's Gemini models.
"""

from google import genai
from google.genai import types
from typing import Optional, Dict, Any, List
import os


class GeminiClient:
    """Wrapper class for Google Gemini API interactions."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY environment variable.
        """
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        self.client = genai.Client(api_key=key)

    def generate_response(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from Gemini model.

        Args:
            prompt: The user prompt/question to send to the model.
            model: Model name (default: "gemini-2.0-flash").
            temperature: Sampling temperature (0-2). Higher values make output more random.
            max_tokens: Maximum tokens in response. None for model default.
            system_message: Optional system message to set context/behavior.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            str: The generated response text.

        Example:
            >>> client = GeminiClient()
            >>> response = client.generate_response("Explain GPU memory hierarchy")
            >>> print(response)
        """
        # Configure generation settings
        config_kwargs = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_message:
            config_kwargs["system_instruction"] = system_message

        config = types.GenerateContentConfig(**config_kwargs)

        # Generate response
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        return response.text

    def generate_response_with_context(
        self,
        prompt: str,
        context_messages: List[Dict[str, str]],
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response with conversation context.

        Args:
            prompt: The user prompt/question.
            context_messages: List of previous messages in format:
                [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional parameters.

        Returns:
            str: The generated response text.
        """
        config_kwargs = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        config = types.GenerateContentConfig(**config_kwargs)

        # Convert context to Gemini format
        contents = []
        for msg in context_messages:
            role = msg["role"]
            # Gemini uses "model" instead of "assistant"
            if role == "assistant":
                role = "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(msg["content"])]
            ))

        # Add the current prompt
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(prompt)]
        ))

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        return response.text

    def generate_response_full(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash",
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
                - usage: Token usage statistics (if available)
                - finish_reason: Why the generation stopped
        """
        config_kwargs = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_message:
            config_kwargs["system_instruction"] = system_message

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        result = {
            "content": response.text,
            "model": model,
            "finish_reason": response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN",
        }

        # Add usage info if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            result["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return result


# Convenience functions for simple use cases

def generate_response(
    prompt: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Simple function to generate a response from Gemini model.

    Args:
        prompt: The user prompt/question.
        model: Model name (default: "gemini-2.0-flash").
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens in response.
        system_message: Optional system message.
        api_key: Optional API key. Uses environment variable if not provided.
        **kwargs: Additional parameters.

    Returns:
        str: The generated response text.

    Example:
        >>> from gemini import generate_response
        >>> response = generate_response("Explain GPU P2P communication")
        >>> print(response)
    """
    client = GeminiClient(api_key=api_key)
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
    model: str = "gemini-2.0-flash",
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
        >>> from gemini import generate_code
        >>> code = generate_code(
        ...     "Write a CUDA kernel for vector addition",
        ...     language="CUDA"
        ... )
        >>> print(code)
    """
    system_msg = "You are an expert programmer. Generate clean, efficient, well-commented code."
    if language:
        system_msg += f" Always use {language}."

    client = GeminiClient(api_key=api_key)
    return client.generate_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_message=system_msg,
    )


def ask_question(
    question: str,
    context: Optional[str] = None,
    model: str = "gemini-2.0-flash",
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
        >>> from gemini import ask_question
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

    client = GeminiClient(api_key=api_key)
    return client.generate_response(prompt=full_prompt, model=model)


def main():
    """Example usage of the Gemini module."""
    # Example 1: Simple response generation
    print("Example 1: Simple response generation")
    print("-" * 60)
    response = generate_response("Explain all-reduce in multi-GPU systems in one paragraph.")
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

    # Example 3: Using GeminiClient class
    print("Example 3: Using GeminiClient class with full response")
    print("-" * 60)
    client = GeminiClient()
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
