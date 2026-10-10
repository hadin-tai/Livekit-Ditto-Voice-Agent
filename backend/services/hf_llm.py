import os
import logging
from huggingface_hub import InferenceClient
from app.config import settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("hf_llm_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Initialize HuggingFace client
HF_TOKEN = os.getenv("HF_TOKEN")
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = settings.HF_LLM_MODEL
# MODEL_NAME = "deepseek-ai/DeepSeek-R1"
logger.info("Using HuggingFace LLM")
logger.info(f"Model: {MODEL_NAME}")

if not HF_TOKEN:
    logger.error("HF_TOKEN environment variable not found")
    raise ValueError("HF_TOKEN is required")

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


def generate_response(messages, temperature=0.7, max_tokens=1024):
    logger.info(f"Messages: {messages}")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        logger.info(f"Response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        logger.exception("HF LLM Error")
        return "I'm sorry, I couldn't generate a response."
