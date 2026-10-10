import os
import sys
import time
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional

from livekit import rtc, api
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
)
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

BACKEND_DIR = Path(__file__).parent.parent.parent
DITTO_DIR = Path(__file__).parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(DITTO_DIR) not in sys.path:
    sys.path.insert(0, str(DITTO_DIR))

from app.config import settings
from avatar.ditto.integration.ditto_video_generator import DittoVideoGenerator

logger = logging.getLogger("ditto_avatar_worker")
logger.setLevel(logging.INFO)


class DittoAvatarWorker:
    """
    Dedicated LiveKit Custom Avatar Worker for Ditto TalkingHead.
    Connects to the LiveKit room with the 'ditto-avatar-worker' identity,
    publishes synchronized audio & video tracks on behalf of the voice agent,
    and synchronizes speech with Ditto inference.
    """

    def __init__(
        self,
        *,
        identity: Optional[str] = None,
        debug_display: Optional[bool] = None,
    ):
        self.identity = identity or settings.DITTO_AVATAR_IDENTITY
        self.debug_display = (
            debug_display if debug_display is not None else settings.DITTO_DEBUG_DISPLAY
        )
        self.room: Optional[rtc.Room] = None
        self.audio_receiver: Optional[DataStreamAudioReceiver] = None
        self.video_generator: Optional[DittoVideoGenerator] = None
        self.avatar_runner: Optional[AvatarRunner] = None
        self._stopped = False
        self._main_task: Optional[asyncio.Task] = None

    def generate_token(self, room_name: str, runtime_agent_identity: str) -> str:
        """Generate LiveKit worker token with dynamic publish_on_behalf attribute."""
        logger.info(
            f"Generating token | Room: '{room_name}' | Worker Identity: '{self.identity}' | Publish-on-behalf: '{runtime_agent_identity}'"
        )
        token_builder = (
            api.AccessToken(
                api_key=settings.LIVEKIT_API_KEY,
                api_secret=settings.LIVEKIT_API_SECRET,
            )
            .with_kind("agent")
            .with_identity(self.identity)
            .with_name("Ditto Avatar")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                )
            )
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: runtime_agent_identity})
        )
        return token_builder.to_jwt()

    async def start(
        self,
        room_name: str,
        runtime_agent_identity: str,
        livekit_url: Optional[str] = None,
    ):
        """Connect worker to LiveKit room and start AvatarRunner pipeline."""
        url = livekit_url or settings.LIVEKIT_URL
        token = self.generate_token(room_name, runtime_agent_identity)

        logger.info("================ DITTO WORKER STARTING ================")
        logger.info(f"Connecting to room: '{room_name}' at {url}")
        logger.info(f"Worker identity: '{self.identity}'")
        logger.info(f"Publish-on-behalf: '{runtime_agent_identity}'")

        self.room = rtc.Room()

        @self.room.on("connected")
        def on_connected():
            logger.info(f"Successfully connected to room '{room_name}' as '{self.identity}'")

        @self.room.on("disconnected")
        def on_disconnected(reason):
            logger.info(f"Room disconnected (reason: {reason})")

        @self.room.on("local_track_published")
        def on_local_track_published(publication, track):
            logger.info(
                f"Published local track: kind={track.kind}, sid={publication.sid}, name={publication.name}"
            )

        # Connect to room
        await self.room.connect(url, token)
        logger.info("Room connection established.")

        # Initialize DataStreamAudioReceiver
        logger.info(f"Creating DataStreamAudioReceiver for sender '{runtime_agent_identity}'...")
        self.audio_receiver = DataStreamAudioReceiver(
            room=self.room,
            sender_identity=runtime_agent_identity,
            frame_size_ms=20,
        )

        # Initialize DittoVideoGenerator
        logger.info("Creating DittoVideoGenerator...")
        self.video_generator = DittoVideoGenerator(
            debug_display=self.debug_display,
        )
        await self.video_generator.start()

        # Configure AvatarOptions
        avatar_options = AvatarOptions(
            video_width=self.video_generator.video_width,
            video_height=self.video_generator.video_height,
            video_fps=25.0,
            audio_sample_rate=16000,
            audio_channels=1,
        )
        logger.info(
            f"AvatarOptions configured: {avatar_options.video_width}x{avatar_options.video_height} @ {avatar_options.video_fps}fps, Audio: {avatar_options.audio_sample_rate}Hz mono"
        )

        # Initialize AvatarRunner
        logger.info("Creating AvatarRunner...")
        self.avatar_runner = AvatarRunner(
            room=self.room,
            audio_recv=self.audio_receiver,
            video_gen=self.video_generator,
            options=avatar_options,
            _queue_size_ms=100,
            _lazy_publish=False,  # publish tracks upfront
        )

        # Start AvatarRunner
        logger.info("Starting AvatarRunner...")
        await self.avatar_runner.start()
        logger.info("AvatarRunner started successfully. Audio and Video tracks active.")
        logger.info("================ DITTO WORKER READY ================")

    async def aclose(self):
        """Clean shutdown of AvatarRunner, DittoVideoGenerator, and Room connection."""
        if self._stopped:
            return
        self._stopped = True
        logger.info("================ DITTO WORKER STOPPING ================")

        if self.avatar_runner is not None:
            try:
                logger.info("Closing AvatarRunner...")
                await self.avatar_runner.aclose()
                logger.info("AvatarRunner closed.")
            except Exception as e:
                logger.warning(f"Error closing AvatarRunner: {e}")

        if self.video_generator is not None:
            try:
                logger.info("Closing DittoVideoGenerator...")
                await self.video_generator.aclose()
                logger.info("DittoVideoGenerator closed.")
            except Exception as e:
                logger.warning(f"Error closing DittoVideoGenerator: {e}")

        if self.room is not None and self.room.isconnected():
            try:
                logger.info("Disconnecting from LiveKit room...")
                await self.room.disconnect()
                logger.info("Room disconnected.")
            except Exception as e:
                logger.warning(f"Error disconnecting room: {e}")

        logger.info("================ DITTO WORKER STOPPED ================")


async def main():
    parser = argparse.ArgumentParser(description="Ditto LiveKit Custom Avatar Worker")
    parser.add_argument("--room", type=str, required=True, help="LiveKit room name to join")
    parser.add_argument("--agent-id", type=str, required=True, help="Voice agent participant identity")
    parser.add_argument("--debug-display", action="store_true", help="Enable OpenCV debug window")
    args = parser.parse_args()

    worker = DittoAvatarWorker(debug_display=args.debug_display)
    try:
        await worker.start(room_name=args.room, runtime_agent_identity=args.agent_id)
        # Keep worker alive until interrupted
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested")
    finally:
        await worker.aclose()


if __name__ == "__main__":
    asyncio.run(main())
