import os


def provider_from_name(name):
    provider_name = (name or "openai").strip().lower()
    if provider_name == "openai":
        from skeleton_generation.providers.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if provider_name == "gemini":
        from skeleton_generation.providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    if provider_name == "deepseek":
        from skeleton_generation.providers.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider()
    raise ValueError(f"Unsupported provider: {provider_name}")


def get_caption_provider():
    provider_name = os.getenv("CAPTION_PROVIDER", "openai")
    return provider_from_name(provider_name)


def get_image_provider():
    provider_name = os.getenv("IMAGE_PROVIDER", "openai")
    return provider_from_name(provider_name)


def provider_status():
    configured_caption = os.getenv("CAPTION_PROVIDER", "openai")
    configured_image = os.getenv("IMAGE_PROVIDER", "openai")
    status = {
        "caption_provider": configured_caption,
        "image_provider": configured_image,
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "deepseek_key_set": bool(os.getenv("DEEPSEEK_API_KEY")),
    }
    return status
