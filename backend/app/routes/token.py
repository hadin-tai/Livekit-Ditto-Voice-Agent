from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from livekit import api
from app.config import settings
import logging
import time
import json

router = APIRouter()
logger = logging.getLogger("token-router")


class TokenRequest(BaseModel):
    identity: str
    room_name: str
    metadata: dict | None = None


class TokenResponse(BaseModel):
    token: str
    url: str


@router.post("/token", response_model=TokenResponse)
async def generate_token(request: TokenRequest):
    start_time = time.perf_counter()
    logger.info("=============== TOKEN GENERATION REQUEST ===============")
    logger.info("Token generation request received", extra={
        "identity": request.identity,
        "room_name": request.room_name,
        "metadata": request.metadata
    })
    try:
        logger.info(f"[TOKEN] Building token for room_name: '{request.room_name}'")
        token_builder = (
            api.AccessToken(
                api_key=settings.LIVEKIT_API_KEY,
                api_secret=settings.LIVEKIT_API_SECRET,
            )
            .with_identity(request.identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=request.room_name,
                )
            )
        )
        print("Room name (in token builder):", request.room_name)
        logger.info(f"[TOKEN] Token builder grants: room = '{request.room_name}'")
        if request.metadata:
            token_builder = token_builder.with_metadata(json.dumps(request.metadata))

        jwt = token_builder.to_jwt()

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        logger.info("[TOKEN GENERATED]", extra={
            "identity": request.identity,
            "room_name": request.room_name,
            "latency_ms": round(latency_ms, 2),
            "livekit_url": settings.LIVEKIT_URL
        })

        return TokenResponse(
            token=jwt,
            url=settings.LIVEKIT_URL,
        )

    except Exception as e:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        logger.exception("Failed to generate token", extra={
            "identity": request.identity,
            "room_name": request.room_name,
            "latency_ms": round(latency_ms, 2),
            "error": str(e)
        })
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate token: {str(e)}",
        )