#!/usr/bin/env python3
"""
LLM Factory Module
Provides a unified interface to interact with different LLM providers (OpenAI, Google Gemini, etc.)
"""

from typing import Optional, Callable
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    GROK = "grok"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GLM = "glm"
    QWEN = "qwen"
    MOONSHOT = "moonshot"


# Model name to provider mapping
MODEL_PROVIDER_MAP = {
    # OpenAI models
    "gpt-4o": LLMProvider.OPENAI,
    "gpt-4o-mini": LLMProvider.OPENAI,
    "gpt-4-turbo": LLMProvider.OPENAI,
    "gpt-4": LLMProvider.OPENAI,
    "gpt-3.5-turbo": LLMProvider.OPENAI,
    "o1": LLMProvider.OPENAI,
    "o1-mini": LLMProvider.OPENAI,
    "o1-preview": LLMProvider.OPENAI,
    "o3-mini": LLMProvider.OPENAI,
    # Future OpenAI models (pattern matching)
    "gpt-5": LLMProvider.OPENAI,
    "gpt-5.2": LLMProvider.OPENAI,

    # Google Gemini models
    "gemini-1.5-pro": LLMProvider.GEMINI,
    "gemini-1.5-flash": LLMProvider.GEMINI,
    "gemini-1.0-pro": LLMProvider.GEMINI,
    "gemini-pro": LLMProvider.GEMINI,
    "gemini-2.0-flash": LLMProvider.GEMINI,
    "gemini-2.0-flash-exp": LLMProvider.GEMINI,
    # Future Gemini models (pattern matching)
    "gemini-pro-3": LLMProvider.GEMINI,
    "gemini-3": LLMProvider.GEMINI,
    "gemini-3-pro-preview": LLMProvider.GEMINI,

    # xAI Grok models
    "grok-3": LLMProvider.GROK,
    "grok-3-fast": LLMProvider.GROK,
    "grok-3-mini": LLMProvider.GROK,
    "grok-3-mini-fast": LLMProvider.GROK,
    "grok-2": LLMProvider.GROK,
    "grok-2-mini": LLMProvider.GROK,

    # Anthropic Claude models
    "claude-opus-4-20250514": LLMProvider.ANTHROPIC,
    "claude-sonnet-4-20250514": LLMProvider.ANTHROPIC,
    "claude-3-7-sonnet-20250219": LLMProvider.ANTHROPIC,
    "claude-3-5-sonnet-20241022": LLMProvider.ANTHROPIC,
    "claude-3-5-haiku-20241022": LLMProvider.ANTHROPIC,
    "claude-3-opus-20240229": LLMProvider.ANTHROPIC,
    "claude-3-haiku-20240307": LLMProvider.ANTHROPIC,

    # DeepSeek models
    "deepseek-chat": LLMProvider.DEEPSEEK,
    "deepseek-reasoner": LLMProvider.DEEPSEEK,
    "deepseek-v4-pro": LLMProvider.DEEPSEEK,

    # GLM models
    "z-ai/glm-4.5-air:free": LLMProvider.GLM,
    "glm-4.5": LLMProvider.GLM,
    "glm-4.5-air": LLMProvider.GLM,

    # Qwen models
    "qwen2.5-7B": LLMProvider.QWEN,
    "qwen3.6-27B": LLMProvider.QWEN,
    "qwen3-32b": LLMProvider.QWEN,
    "qwen-plus": LLMProvider.QWEN,
    "qwen-max": LLMProvider.QWEN,

    # Moonshot / Kimi models
    "kimi-k2.6": LLMProvider.MOONSHOT,
    "kimi-k2.5": LLMProvider.MOONSHOT,
    "kimi-k2-0711-preview": LLMProvider.MOONSHOT,
}


def get_provider_for_model(model: str) -> LLMProvider:
    """
    Determine the LLM provider based on model name.

    Args:
        model: The model name (e.g., "gpt-4o", "gemini-1.5-pro")

    Returns:
        LLMProvider enum value

    Raises:
        ValueError: If the model is not recognized
    """
    # Direct mapping first
    if model in MODEL_PROVIDER_MAP:
        return MODEL_PROVIDER_MAP[model]

    # Pattern-based matching for flexibility
    model_lower = model.lower()

    if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
        return LLMProvider.OPENAI

    if model_lower.startswith("gemini"):
        return LLMProvider.GEMINI

    if model_lower.startswith("grok"):
        return LLMProvider.GROK

    if model_lower.startswith("claude"):
        return LLMProvider.ANTHROPIC

    if model_lower.startswith("deepseek"):
        return LLMProvider.DEEPSEEK

    if model_lower.startswith("glm") or model_lower.startswith("z-ai/glm"):
        return LLMProvider.GLM

    if model_lower.startswith("qwen"):
        return LLMProvider.QWEN

    if model_lower.startswith("kimi") or model_lower.startswith("moonshot"):
        return LLMProvider.MOONSHOT

    raise ValueError(
        f"Unknown model: {model}. "
        f"Supported prefixes: 'gpt', 'o1', 'o3' (OpenAI), 'gemini' (Google), "
        f"'grok' (xAI), 'claude' (Anthropic), 'deepseek' (DeepSeek), "
        f"'glm'/'z-ai/glm' (GLM), 'qwen' (Qwen). "
        f"Or specify --api explicitly."
    )


def get_generate_code_function(provider: LLMProvider) -> Callable:
    """
    Get the generate_code function for the specified provider.

    Args:
        provider: LLMProvider enum value

    Returns:
        The generate_code function from the appropriate module
    """
    if provider == LLMProvider.OPENAI:
        from prompt_llm.gpt import generate_code
        return generate_code
    elif provider == LLMProvider.GEMINI:
        from prompt_llm.gemini import generate_code
        return generate_code
    elif provider == LLMProvider.GROK:
        from prompt_llm.grok import generate_code
        return generate_code
    elif provider == LLMProvider.ANTHROPIC:
        from prompt_llm.anthropic_llm import generate_code
        return generate_code
    elif provider == LLMProvider.DEEPSEEK:
        from prompt_llm.deepseek import generate_code
        return generate_code
    elif provider == LLMProvider.GLM:
        from prompt_llm.glm import generate_code
        return generate_code
    elif provider == LLMProvider.QWEN:
        from prompt_llm.qwen import generate_code
        return generate_code
    elif provider == LLMProvider.MOONSHOT:
        from prompt_llm.moonshot import generate_code
        return generate_code
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def generate_code(
    prompt: str,
    model: str = "gpt-4o",
    language: Optional[str] = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Unified interface to generate code using any supported LLM provider.

    Automatically detects the provider based on the model name,
    or uses the explicitly specified provider.

    Args:
        prompt: Description of the code to generate.
        model: Model name (e.g., "gpt-4o", "gemini-1.5-pro").
        language: Programming language (e.g., "Python", "C++", "CUDA").
        temperature: Sampling temperature (default 0.3 for code generation).
        api_key: Optional API key.
        provider: Optional explicit provider ("openai" or "gemini").
                 If not specified, auto-detected from model name.

    Returns:
        str: Generated code.

    Example:
        >>> from llm_factory import generate_code
        >>> # Auto-detect provider from model name
        >>> code = generate_code("Write a CUDA kernel", model="gpt-4o")
        >>> code = generate_code("Write a CUDA kernel", model="gemini-1.5-pro")
        >>> # Explicit provider
        >>> code = generate_code("Write a CUDA kernel", model="my-custom-model", provider="openai")
    """
    # Determine provider
    if provider:
        llm_provider = LLMProvider(provider.lower())
    else:
        llm_provider = get_provider_for_model(model)

    # Get the appropriate generate_code function
    generate_fn = get_generate_code_function(llm_provider)

    # Call the provider-specific function
    return generate_fn(
        prompt=prompt,
        model=model,
        language=language,
        temperature=temperature,
        api_key=api_key,
    )


def list_supported_models() -> dict:
    """
    List all supported models grouped by provider.

    Returns:
        Dict mapping provider names to lists of model names.
    """
    result = {
        "openai": [],
        "gemini": [],
        "grok": [],
        "anthropic": [],
        "deepseek": [],
        "glm": [],
        "qwen": [],
    }

    for model, provider in MODEL_PROVIDER_MAP.items():
        result[provider.value].append(model)

    return result


def main():
    """Example usage of the LLM factory."""
    print("Supported Models:")
    print("-" * 60)

    models = list_supported_models()
    for provider, model_list in models.items():
        print(f"\n{provider.upper()}:")
        for model in model_list:
            print(f"  - {model}")

    print("\n" + "-" * 60)
    print("\nProvider detection examples:")
    test_models = [
        "gpt-4o", "gemini-1.5-pro", "gpt-5.2", "gemini-pro-3",
        "grok-3", "claude-opus-4-20250514", "z-ai/glm-4.5-air:free",
        "qwen3.6-27B",
    ]
    for model in test_models:
        try:
            provider = get_provider_for_model(model)
            print(f"  {model} -> {provider.value}")
        except ValueError as e:
            print(f"  {model} -> ERROR: {e}")


if __name__ == "__main__":
    main()
