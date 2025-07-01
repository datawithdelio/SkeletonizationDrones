from openai import OpenAI 
from dotenv import load_dotenv
import os
import requests
from io import BytesIO
from PIL import Image
import base64

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=API_KEY)

def generate_image(prompt):
    response = client.images.generate(
        model='dall-e-3',
        prompt = (
            f"{prompt}, clearly visible and centered in the frame, "
            "with high contrast against the background, well-lit, and "
            "no occlusions or background clutter. Ensure the entire subject is within view and that subjects don't overlap with each other."
        ),
        size='1024x1024',
        quality='standard',
        n=1
    )
    image_url = response.data[0].url
    return image_url

def save_generation(prompt, save_path):
    image_url = generate_image(prompt)
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image.save(save_path)

def describe_image(image_path):
    with open(image_path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. Try to identify the type of drone in the image "
                            "(guess if unsure), and explain what the drone is doing or what is happening in the scene."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    }
                ]
            }
        ],
        max_tokens=200
    )

    return response.choices[0].message.content

