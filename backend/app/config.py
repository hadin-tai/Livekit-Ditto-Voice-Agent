from pydantic_settings import BaseSettings
import logging
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()



logger = logging.getLogger("config")

class Settings(BaseSettings):
    LIVEKIT_URL: str
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    GOOGLE_API_KEY: str
    DEEPGRAM_API_KEY: str
    CARTESIA_API_KEY: str
    SPEECHMATICS_API_KEY: str
    ANAM_API_KEY: str
    ANAM_AVATAR_ID: str
    PINECONE_API_KEY: Optional[str] = None
    ASSISTANT_NAME: str = "voice-agent-rag"
    RAG_API_URL: str = "http://localhost:8000/api/rag/search"
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    HF_LLM_MODEL: str = os.getenv("HF_LLM_MODEL")
    CARTESIA_VOICE_ID: str = os.getenv("CARTESIA_VOICE_ID")
    DITTO_AVATAR_IDENTITY: str = "ditto-avatar-worker"
    DITTO_DEBUG_DISPLAY: bool = False
    DITTO_ENABLED: bool = True

    class Config:
        env_file = ".env"


logger.info("Loading environment variables from .env file")
try:
    settings = Settings()
    logger.info("Environment variables loaded successfully", extra={
        "livekit_url_set": bool(settings.LIVEKIT_URL),
        "livekit_api_key_set": bool(settings.LIVEKIT_API_KEY),
        "livekit_api_secret_set": bool(settings.LIVEKIT_API_SECRET),
        "google_api_key_set": bool(settings.GOOGLE_API_KEY),
        "deepgram_api_key_set": bool(settings.DEEPGRAM_API_KEY),
        "cartesia_api_key_set": bool(settings.CARTESIA_API_KEY),
        "speechmatics_api_key_set": bool(settings.SPEECHMATICS_API_KEY),
        "pinecone_api_key_set": bool(settings.PINECONE_API_KEY),
        "rag_api_url_set": bool(settings.RAG_API_URL)
    })
except Exception as e:
    logger.exception("Failed to load environment variables")
    raise
