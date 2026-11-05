"""
config.py - one place for settings shared by the whole project.

- Loads secrets from the .env file (never hard-code keys in code).
- Defines the important folders (knowledge base, logs).
- Builds the OpenAI model client that every agent uses.
"""
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Folder that contains this file = the project root
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# AutoGen + the newest Pydantic print a harmless warning for structured output.
# It does not affect results, so we hide it to keep the output readable.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

KB_DIR = PROJECT_ROOT / "knowledge_base"
LOG_DIR = PROJECT_ROOT / "logs"

# Model can be changed in .env with OPENAI_MODEL=... (default is cheap and fast)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Speech-to-text model for voice messages (falls back to whisper-1)
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

# Prices used for the cost estimate (US dollars per 1 million tokens).
# Defaults are OpenAI's list prices for gpt-4o-mini; change them in .env if you
# use another model or the prices change.
PRICE_INPUT_PER_M = float(os.getenv("PRICE_INPUT_PER_M", "0.15"))
PRICE_OUTPUT_PER_M = float(os.getenv("PRICE_OUTPUT_PER_M", "0.60"))
USD_TO_INR = float(os.getenv("USD_TO_INR", "88"))


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated cost of the language-model calls for one ticket."""
    return (prompt_tokens * PRICE_INPUT_PER_M + completion_tokens * PRICE_OUTPUT_PER_M) / 1_000_000


def get_api_key() -> str:
    """Read the API key from .env and fail with a clear message if it is missing."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Open the .env file and add a line: "
            "OPENAI_API_KEY=sk-..."
        )
    return key


def get_model_client(temperature: float = 0.2) -> OpenAIChatCompletionClient:
    """
    Create the LLM client used by all agents.
    Low temperature = more consistent, factual answers (good for support).
    """
    api_key = get_api_key()
    try:
        return OpenAIChatCompletionClient(
            model=MODEL_NAME, api_key=api_key, temperature=temperature
        )
    except ValueError:
        # AutoGen does not know very new model names yet.
        # In that case we describe the model's abilities ourselves.
        return OpenAIChatCompletionClient(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=temperature,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": "unknown",
            },
        )
