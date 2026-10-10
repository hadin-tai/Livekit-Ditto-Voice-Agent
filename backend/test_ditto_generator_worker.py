import os
import sys
import time
import asyncio
import logging
from pathlib import Path
import numpy as np
import librosa

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_ditto_worker")

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "avatar" / "ditto"))

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd
from avatar.ditto.integration.ditto_video_generator import DittoVideoGenerator

AUDIO_PATH = BACKEND_DIR / "avatar" / "ditto" / "example" / "audio.wav"


async def test_video_generator():
    logger.info("============================================================")
    logger.info("TEST: DITTO VIDEO GENERATOR PRODUCER/CONSUMER LIFECYCLE")
    logger.info("============================================================")

    logger.info(f"Loading audio from: {AUDIO_PATH}")
    audio_full, sr = librosa.load(str(AUDIO_PATH), sr=16000)
    logger.info(f"Loaded audio: {len(audio_full)} samples ({len(audio_full)/sr:.2f}s) at {sr}Hz")

    generator = DittoVideoGenerator(debug_display=False)
    await generator.start()
    logger.info("Generator started.")

    # Task to consume from generator.__aiter__()
    audio_frames_received = []
    video_frames_received = []
    segment_ends_received = []
    consumer_done = asyncio.Event()

    async def consumer():
        try:
            async for item in generator:
                if isinstance(item, rtc.AudioFrame):
                    audio_frames_received.append(item)
                elif isinstance(item, rtc.VideoFrame):
                    video_frames_received.append(item)
                    if len(video_frames_received) % 10 == 1 or len(video_frames_received) <= 5:
                        logger.info(
                            f"[TEST CONSUMER] Received VideoFrame #{len(video_frames_received)}: "
                            f"{item.width}x{item.height}, type={item.type}"
                        )
                elif isinstance(item, AudioSegmentEnd):
                    segment_ends_received.append(item)
                    logger.info(f"[TEST CONSUMER] Received AudioSegmentEnd #{len(segment_ends_received)}")
                    consumer_done.set()
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consumer())

    # TURN 1: Push 1.5s of speech (24000 samples) in variable chunk sizes (640, 1280, 3200)
    test_audio_segment = audio_full[:24000]
    chunk_sizes = [640, 1280, 3200, 1600, 3200, 3200, 3200, 3200, 3200, 1160]
    offset = 0
    t0 = time.time()

    logger.info("--- PUSHING AUDIO FRAMES (TURN 1) ---")
    for chunk_size in chunk_sizes:
        if offset >= len(test_audio_segment):
            break
        chunk = test_audio_segment[offset : offset + chunk_size]
        offset += len(chunk)

        # Convert float32 to int16 bytes to simulate real LiveKit AudioFrame
        int16_bytes = (chunk * 32767.0).astype(np.int16).tobytes()
        audio_frame = rtc.AudioFrame(
            data=int16_bytes,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(chunk),
        )

        await generator.push_audio(audio_frame)
        await asyncio.sleep(0.02)  # small delay simulating streaming

    logger.info(f"All audio frames pushed. Samples pushed: {offset}. Sending AudioSegmentEnd...")
    await generator.push_audio(AudioSegmentEnd())

    # Wait for consumer to receive AudioSegmentEnd
    try:
        await asyncio.wait_for(consumer_done.wait(), timeout=40.0)
    except asyncio.TimeoutError:
        logger.error("Timed out waiting for AudioSegmentEnd in consumer!")

    logger.info(f"Turn 1 completed in {time.time() - t0:.2f}s")
    logger.info(f"Audio frames received: {len(audio_frames_received)}")
    logger.info(f"Video frames received: {len(video_frames_received)}")
    logger.info(f"Segment ends received: {len(segment_ends_received)}")

    assert len(audio_frames_received) > 0, "No audio frames received!"
    assert len(video_frames_received) > 0, "No video frames received!"
    assert len(segment_ends_received) == 1, "Expected 1 AudioSegmentEnd!"

    first_video = video_frames_received[0]
    assert first_video.width == 540 and first_video.height == 360, f"Unexpected dimensions: {first_video.width}x{first_video.height}"
    assert first_video.type == rtc.VideoBufferType.RGB24, f"Unexpected buffer type: {first_video.type}"

    # TEST: clear_buffer() during speech (Turn 2 interruption test)
    logger.info("--- TESTING INTERRUPTION / CLEAR_BUFFER ---")
    consumer_done.clear()
    for chunk_size in [3200, 3200]:
        chunk = audio_full[24000 : 24000 + chunk_size]
        int16_bytes = (chunk * 32767.0).astype(np.int16).tobytes()
        audio_frame = rtc.AudioFrame(
            data=int16_bytes,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(chunk),
        )
        await generator.push_audio(audio_frame)

    generator.clear_buffer()
    logger.info("clear_buffer() executed successfully.")

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    await generator.aclose()
    logger.info("============================================================")
    logger.info("DITTO VIDEO GENERATOR ACCEPTANCE TEST PASSED!")
    logger.info("============================================================")


if __name__ == "__main__":
    asyncio.run(test_video_generator())
