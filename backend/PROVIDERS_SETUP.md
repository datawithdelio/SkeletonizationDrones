# LLM Provider Setup

This backend supports multiple providers for image captioning:
- `openai`
- `gemini`
- `deepseek`

## Environment Variables

Add these to `backend/.env`:

```env
# Provider selection
CAPTION_PROVIDER=openai
CAPTION_FALLBACK_PROVIDER=openai
IMAGE_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=...
OPENAI_VISION_MODEL=gpt-4o
OPENAI_IMAGE_MODEL=dall-e-3

# Gemini
GEMINI_API_KEY=...
GEMINI_VISION_MODEL=gemini-1.5-flash

# DeepSeek
DEEPSEEK_API_KEY=...
DEEPSEEK_VISION_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## Runtime Behavior

- `/api/upload`: uses `CAPTION_PROVIDER` to describe the generated skeleton image.
- `/api/openai/upload`: prompt-to-image path uses `IMAGE_PROVIDER`; if unsupported, falls back to OpenAI image generation.
- `/api/providers/status`: returns configured provider names and whether provider keys are set.

## Quick Check

```bash
curl http://127.0.0.1:5001/api/providers/status
```
