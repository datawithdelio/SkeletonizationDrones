import base64
import os

from openai import OpenAI


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


class DeepSeekProvider:
    name = "deepseek"

    def __init__(self, api_key=None, vision_model=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.vision_model = vision_model or os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-chat")

    def generate_image_url(self, prompt):
        raise NotImplementedError("DeepSeek image generation is not wired in this project.")

    def describe_image(self, image_path, prompt=DEFAULT_VISION_PROMPT, max_tokens=200):
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
