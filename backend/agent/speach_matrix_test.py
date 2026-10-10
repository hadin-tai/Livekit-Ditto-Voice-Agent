import asyncio
import logging

from speaker_utils import load_known_speakers
# from app.config import settings
from livekit.plugins import speechmatics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speaker_test")


async def main():
    known_speakers = load_known_speakers()

    stt = speechmatics.STT(
        turn_detection_mode=speechmatics.TurnDetectionMode.SMART_TURN,
        enable_diarization=True,
        speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
        speaker_passive_format="<PASSIVE><{speaker_id}>{text}</{speaker_id}></PASSIVE>",
        known_speakers=known_speakers,
        # api_key=settings.SPEECHMATICS_API_KEY,
        api_key="jr57zQRlU7kFPnwz2tpnugrgbjKfFh6e",
    )

    logger.info(f"STT object type: {type(stt)}")

    logger.info(f"Has get_speaker_ids: {hasattr(stt, 'get_speaker_ids')}")

    if hasattr(stt, "get_speaker_ids"):
        logger.info("Method exists.")
        logger.info(f"Method: {stt.get_speaker_ids}")
    else:
        logger.error("This version of Speechmatics STT does NOT have get_speaker_ids().")


if __name__ == "__main__":
    asyncio.run(main())