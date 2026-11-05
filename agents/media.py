"""
agents/media.py - voice and photo helpers for the customer portal.

transcribe_audio(): turns a recorded voice message into text with OpenAI speech-to-text.
                    Works in many languages (for example Tamil), and the customer can
                    check and edit the text before sending it.
check_photo():      looks at a photo the customer attached (for example a cracked screen)
                    and records, in one sentence, what it shows and whether it supports the
                    complaint. A photo that does NOT match the complaint is escalated.
"""
import base64
import json

from config import MODEL_NAME, TRANSCRIBE_MODEL, get_api_key

PHOTO_PROMPT = """You check photos that customers attach to support complaints for an online store.
1. Describe in one short sentence what the photo shows: the product and any visible damage.
2. Decide whether the photo supports the complaint:
   "yes" if it clearly shows the problem described,
   "no" if it shows something unrelated or no problem at all,
   "unclear" if you cannot tell.
Only describe what is visible. Do not guess.
Answer as JSON: {"description": "...", "supports": "yes" | "no" | "unclear"}"""


def _client():
    from openai import OpenAI
    return OpenAI(api_key=get_api_key())


def transcribe_audio(audio_bytes: bytes, filename: str = "voice.wav") -> str:
    """Speech to text. Tries the configured model, then whisper-1 as a fallback."""
    client, last_error = _client(), None
    for model in dict.fromkeys([TRANSCRIBE_MODEL, "whisper-1"]):
        try:
            result = client.audio.transcriptions.create(model=model, file=(filename, audio_bytes))
            return (result.text or "").strip()
        except Exception as exc:  # try the next model
            last_error = exc
    raise RuntimeError(f"Could not transcribe the voice message: {last_error}")


def check_photo(image_bytes: bytes, mime_type: str, complaint: str) -> dict:
    """Describe the photo and say whether it supports the complaint (yes / no / unclear)."""
    encoded = base64.b64encode(image_bytes).decode()
    try:
        response = _client().chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PHOTO_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Customer complaint: {complaint}"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": "low"}},
                ]},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        supports = str(data.get("supports", "unclear")).strip().lower()
        return {
            "description": str(data.get("description", "")).strip()[:300],
            "supports": supports if supports in ("yes", "no", "unclear") else "unclear",
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
    except Exception as exc:  # the ticket still goes ahead without the check
        return {"description": f"The photo could not be checked ({type(exc).__name__}).",
                "supports": "unclear", "prompt_tokens": 0, "completion_tokens": 0}
