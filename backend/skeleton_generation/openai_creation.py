from dotenv import load_dotenv
import os
import requests
from io import BytesIO
from PIL import Image

from skeleton_generation.providers.router import (
    get_caption_provider,
    get_image_provider,
    provider_from_name,
    provider_status,
)

load_dotenv()

def generate_image(prompt):
    provider = get_image_provider()
    return provider.generate_image_url(prompt)

def save_generation(prompt, save_path):
    try:
        image_url = generate_image(prompt)
    except NotImplementedError:
        # Keep prompt-to-image endpoint functional by falling back to OpenAI.
        image_url = provider_from_name("openai").generate_image_url(prompt)

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image.save(save_path)

def describe_image(image_path):
    preferred = get_caption_provider()
    fallback_name = os.getenv("CAPTION_FALLBACK_PROVIDER", "openai")

    try:
        return preferred.describe_image(image_path)
    except Exception as exc:
        try:
            return provider_from_name(fallback_name).describe_image(image_path)
        except Exception:
            status = provider_status()
            return (
                f"Caption generation failed. Preferred provider error: {exc}. "
                f"Provider status: {status}"
            )

def get_provider_status():
    return provider_status()
