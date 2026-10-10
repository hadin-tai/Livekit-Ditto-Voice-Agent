import os
import sys
import time
import asyncio
import logging
from pathlib import Path
import numpy as np
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_live_avatar")

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "avatar" / "ditto"))

from livekit import rtc, api
from livekit.agents.voice.avatar import DataStreamAudioOutput
from app.config import settings
from avatar.ditto.ditto_avatar_worker import DittoAvatarWorker

AUDIO_PATH = BACKEND_DIR / "avatar" / "ditto" / "example" / "audio.wav"


async def run_live_test():
    test_room_name = f"test-avatar-room-{int(time.time())}"
    agent_identity = "mock-voice-agent"
    client_identity = "test-client-listener"

    logger.info("============================================================")
    logger.info(f"STARTING LIVE ROOM AVATAR WORKER TEST | Room: {test_room_name}")
    logger.info("============================================================")

    # 1. Connect Mock Agent Room
    agent_token = (
        api.AccessToken(
            api_key=settings.LIVEKIT_API_KEY,
            api_secret=settings.LIVEKIT_API_SECRET,
        )
        .with_kind("agent")
        .with_identity(agent_identity)
        .with_grants(api.VideoGrants(room_join=True, room=test_room_name))
        .to_jwt()
    )

    agent_room = rtc.Room()
    logger.info(f"Agent connecting to room '{test_room_name}'...")
    await agent_room.connect(settings.LIVEKIT_URL, agent_token)
    logger.info(f"Agent connected as '{agent_identity}'.")

    # 2. Connect Client Listener Room (to subscribe to avatar tracks)
    client_token = (
        api.AccessToken(
            api_key=settings.LIVEKIT_API_KEY,
            api_secret=settings.LIVEKIT_API_SECRET,
        )
        .with_identity(client_identity)
        .with_grants(api.VideoGrants(room_join=True, room=test_room_name))
        .to_jwt()
    )

    client_room = rtc.Room()
    tracks_subscribed = []
    video_frames_rendered = []

    @client_room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        logger.info(
            f"[CLIENT LISTENER] Track subscribed: kind={track.kind}, name={publication.name}, participant={participant.identity}"
        )
        tracks_subscribed.append((track.kind, publication.name, participant.identity))
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            video_stream = rtc.VideoStream(track)
            async def read_video():
                async for event in video_stream:
                    video_frames_rendered.append(event.frame)
                    if len(video_frames_rendered) % 10 == 1 or len(video_frames_rendered) <= 3:
                        logger.info(
                            f"[CLIENT LISTENER] Received Live Video Frame #{len(video_frames_rendered)}: "
                            f"{event.frame.width}x{event.frame.height}"
                        )
            asyncio.create_task(read_video())

    logger.info("Client listener connecting...")
    await client_room.connect(settings.LIVEKIT_URL, client_token)
    logger.info(f"Client listener connected as '{client_identity}'.")

    # 3. Start DittoAvatarWorker
    logger.info("--- STARTING DITTO AVATAR WORKER ---")
    worker = DittoAvatarWorker(debug_display=False)
    await worker.start(room_name=test_room_name, runtime_agent_identity=agent_identity)
    logger.info("DittoAvatarWorker started and connected.")

    # 4. Configure DataStreamAudioOutput on Agent Room
    audio_output = DataStreamAudioOutput(
        room=agent_room,
        destination_identity=settings.DITTO_AVATAR_IDENTITY,
        sample_rate=16000,
        wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        wait_playback_start=True,
    )

    # 5. Push Audio via DataStreamAudioOutput
    logger.info("Loading test audio...")
    audio_full, sr = librosa.load(str(AUDIO_PATH), sr=16000)
    test_audio = audio_full[:24000]  # 1.5 seconds

    logger.info("--- STREAMING AUDIO VIA DATASTREAM TO DITTO WORKER ---")
    chunk_size = 1600
    for i in range(0, len(test_audio), chunk_size):
        chunk = test_audio[i : i + chunk_size]
        int16_bytes = (chunk * 32767.0).astype(np.int16).tobytes()
        frame = rtc.AudioFrame(
            data=int16_bytes,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(chunk),
        )
        await audio_output.capture_frame(frame)
        await asyncio.sleep(0.05)

    logger.info("Flushing audio_output (marking segment end)...")
    audio_output.flush()

    # Wait for frames to be rendered and published
    logger.info("Waiting for video frames to arrive at client listener...")
    t_wait = time.time()
    while len(video_frames_rendered) < 30 and (time.time() - t_wait) < 30.0:
        await asyncio.sleep(0.5)

    logger.info(f"Test completed. Subscribed tracks: {tracks_subscribed}")
    logger.info(f"Video frames received by client: {len(video_frames_rendered)}")

    # Assertions
    assert any(p == settings.DITTO_AVATAR_IDENTITY for _, _, p in tracks_subscribed), "No track from ditto-avatar-worker!"
    assert len(video_frames_rendered) > 0, "No video frames received by client listener!"

    logger.info("Cleaning up test rooms and worker...")
    await worker.aclose()
    await client_room.disconnect()
    await agent_room.disconnect()

    logger.info("============================================================")
    logger.info("LIVE ROOM AVATAR WORKER END-TO-END TEST PASSED!")
    logger.info("============================================================")


if __name__ == "__main__":
    asyncio.run(run_live_test())
