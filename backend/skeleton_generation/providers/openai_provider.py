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


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key=None, image_model=None, vision_model=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.image_model = image_model or os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
        self.vision_model = vision_model or os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

    def generate_image_url(self, prompt):
        response = self.client.images.generate(
            model=self.image_model,
            prompt=(
                f"{prompt}, clearly visible and centered in the frame, "
                "with high contrast against the background, well-lit, and "
                "no occlusions or background clutter. Ensure the entire subject is within view and that subjects don't overlap with each other."
            ),
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url

    def describe_image(self, image_path, prompt=DEFAULT_VISION_PROMPT, max_tokens=200):
        with open(image_path, "rb") as img_file:
            base64_img = base64.b64encode(img_file.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
