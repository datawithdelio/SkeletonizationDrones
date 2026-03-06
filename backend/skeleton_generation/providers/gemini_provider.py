import base64
import os

import requests


DEFAULT_VISION_PROMPT = (
    "Analyze this image conservatively and avoid overconfident claims. "
    "Return a concise response with these sections: "
    "1) Object guess (drone/bird/aircraft/unknown), "
    "2) Confidence (low/medium/high), "
    "3) Motion estimate (direction/angle only if visible; otherwise 'not inferable'), "
    "4) Evidence from visible features, "
    "5) Uncertainty notes. "
    "If evidence is weak, explicitly say so."
)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key=None, vision_model=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.vision_model = vision_model or os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash")

    def generate_image_url(self, prompt):
        raise NotImplementedError("Gemini image generation is not wired in this project.")

    def describe_image(self, image_path, prompt=DEFAULT_VISION_PROMPT, max_tokens=200):
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.vision_model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2,
            },
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return "No response returned by Gemini."
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        out = "\n".join([t for t in text_parts if t]).strip()
        return out or "No textual description returned by Gemini."
