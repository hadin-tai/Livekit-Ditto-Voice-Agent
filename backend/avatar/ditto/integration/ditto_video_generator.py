import os
import sys
import time
import math
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional, AsyncIterator
import numpy as np

from livekit import rtc
from livekit.agents.voice.avatar import VideoGenerator, AudioSegmentEnd

# Setup logger
logger = logging.getLogger("ditto_video_generator")
logger.setLevel(logging.INFO)

BACKEND_DIR = Path(__file__).parent.parent.parent.parent
DITTO_DIR = Path(__file__).parent.parent
if str(DITTO_DIR) not in sys.path:
    sys.path.insert(0, str(DITTO_DIR))

from avatar.ditto.integration.ditto_adapter import (
    DittoEngine,
    EngineDisplayWriter,
    CHUNKSIZE,
    SPLIT_LEN,
    ADVANCE,
    INITIAL_SILENCE_SAMPLES,
    DITTO_CHECKPOINTS,
    DITTO_CFG_PKL,
    DITTO_SOURCE_IMAGE,
    OPENCV_WINDOW_NAME,
)
import cv2


class _SegmentEndMarker:
    """Internal marker placed in the inference queue to signal end of an audio segment."""
    def __init__(self, segment_id: str = ""):
        self.segment_id = segment_id


class LiveKitFrameWriter:
    """
    Custom frame writer attached to Ditto StreamSDK.
    Receives generated RGB frames from Ditto's putback worker and pushes them
    as rtc.VideoFrame objects to the generator's output queue.
    """
    def __init__(
        self,
        output_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        debug_display: bool = False,
        video_width: int = 540,
        video_height: int = 360,
    ):
        self.output_queue = output_queue
        self.loop = loop
        self.debug_display = debug_display
        self.video_width = video_width
        self.video_height = video_height
        self.frame_count = 0
        self._window_created = False
        self.closed = False

    def __call__(self, img: np.ndarray, fmt: str = "rgb"):
        if self.closed:
            return

        self.frame_count += 1
        h, w = img.shape[:2]

        # Optional debug OpenCV display on the same thread
        if self.debug_display:
            try:
                if not self._window_created:
                    cv2.namedWindow(OPENCV_WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(OPENCV_WINDOW_NAME, w, h)
                    self._window_created = True

                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if fmt == "rgb" else img
                cv2.imshow(OPENCV_WINDOW_NAME, bgr_img)
                cv2.waitKey(1)
            except Exception as cv_e:
                logger.warning(f"[DITTO VIDEO] OpenCV debug display error: {cv_e}")

        try:
            # Ensure uint8 contiguous RGB buffer
            if not img.flags["C_CONTIGUOUS"]:
                img = np.ascontiguousarray(img)
            
            raw_bytes = img.tobytes()

            video_frame = rtc.VideoFrame(
                width=w,
                height=h,
                type=rtc.VideoBufferType.RGB24,
                data=raw_bytes,
            )

            # Thread-safely push into asyncio output queue
            self.loop.call_soon_threadsafe(self.output_queue.put_nowait, video_frame)

            if self.frame_count % 25 == 1 or self.frame_count <= 5:
                logger.info(
                    f"[DITTO VIDEO] Generated frame #{self.frame_count} | {w}x{h} RGB24 -> LiveKit VideoFrame"
                )
        except Exception as e:
            logger.exception(f"[DITTO VIDEO ERROR] Failed converting/queueing video frame: {e}")

    def close(self):
        self.closed = True
        if self.debug_display and self._window_created:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


class DittoVideoGenerator(VideoGenerator):
    """
    LiveKit-native VideoGenerator wrapping Ditto StreamSDK.
    Implements a robust Producer/Consumer architecture:
    - push_audio() converts & buffers AudioFrame objects, forwards original AudioFrame to output queue
    - Background inference worker feeds accumulated chunks to StreamSDK.run_chunk()
    - LiveKitFrameWriter collects generated frames and pushes rtc.VideoFrame to output queue
    - __aiter__() streams rtc.AudioFrame | rtc.VideoFrame | AudioSegmentEnd to AvatarRunner
    """

    def __init__(
        self,
        *,
        source_image: Optional[Path] = None,
        cfg_pkl: Optional[Path] = None,
        checkpoints_dir: Optional[Path] = None,
        debug_display: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._loop = loop or asyncio.get_event_loop()
        self.debug_display = debug_display
        self.source_image = source_image or DITTO_SOURCE_IMAGE
        self.cfg_pkl = cfg_pkl or DITTO_CFG_PKL
        self.checkpoints_dir = checkpoints_dir or DITTO_CHECKPOINTS

        # Determine dimensions from source image
        self.video_width = 540
        self.video_height = 360
        if self.source_image.exists():
            try:
                sample_img = cv2.imread(str(self.source_image))
                if sample_img is not None:
                    self.video_height, self.video_width = sample_img.shape[:2]
            except Exception:
                pass

        logger.info(
            f"[DITTO GENERATOR] Initializing DittoVideoGenerator | Dimensions: {self.video_width}x{self.video_height} | Debug display: {debug_display}"
        )

        # Output queue for AvatarRunner
        self._output_queue: asyncio.Queue[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()

        # Custom Frame Writer
        self._frame_writer = LiveKitFrameWriter(
            output_queue=self._output_queue,
            loop=self._loop,
            debug_display=self.debug_display,
            video_width=self.video_width,
            video_height=self.video_height,
        )

        # Initialize underlying DittoEngine
        self._engine = DittoEngine(
            checkpoints_dir=self.checkpoints_dir,
            cfg_pkl=self.cfg_pkl,
            source_image=self.source_image,
        )

        # Audio buffering state
        self._audio_buffer: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._chunk_position: int = 0
        self._initial_silence_prepended: bool = False
        self._buffer_lock = threading.Lock()
        self._segment_counter: int = 0
        self._current_segment_id: str = "seg_init"

        # Inference worker queue & thread
        self._infer_queue: asyncio.Queue[np.ndarray | _SegmentEndMarker] = asyncio.Queue()
        self._infer_task: Optional[asyncio.Task] = None
        self._closed: bool = False
        self._started: bool = False

    async def start(self):
        """Initialize Ditto engine in a background thread to prevent blocking event loop."""
        if self._started:
            return
        
        logger.info("[DITTO GENERATOR] Starting DittoEngine & StreamSDK...")
        await asyncio.to_thread(self._init_engine_sync)
        
        # Start background inference processing task
        self._infer_task = asyncio.create_task(self._inference_worker())
        self._started = True
        logger.info("[DITTO GENERATOR] DittoVideoGenerator successfully started and ready.")

    def _init_engine_sync(self):
        """Synchronous engine initialization running in worker thread."""
        self._engine.start()
        # Override sdk.writer with our LiveKitFrameWriter
        if self._engine.sdk is not None:
            self._engine.sdk.writer = self._frame_writer
            logger.info("[DITTO GENERATOR] StreamSDK writer attached to LiveKitFrameWriter.")

    def _ensure_initial_silence(self):
        if self._initial_silence_prepended:
            return
        silence = np.zeros((INITIAL_SILENCE_SAMPLES,), dtype=np.float32)
        self._audio_buffer = np.concatenate([silence, self._audio_buffer], 0)
        self._chunk_position = 0
        self._initial_silence_prepended = True
        logger.info(
            f"[DITTO AUDIO] Prepended initial silence: {INITIAL_SILENCE_SAMPLES} samples (120ms)"
        )

    def _convert_audio_frame_to_numpy(self, frame: rtc.AudioFrame) -> Optional[np.ndarray]:
        """Convert incoming LiveKit AudioFrame to 16kHz mono float32 NumPy array."""
        try:
            raw_data = frame.data
            if isinstance(raw_data, memoryview):
                raw_bytes = raw_data.tobytes()
            elif isinstance(raw_data, (bytes, bytearray)):
                raw_bytes = bytes(raw_data)
            elif isinstance(raw_data, np.ndarray):
                if raw_data.dtype == np.int16:
                    raw_bytes = raw_data.tobytes()
                elif raw_data.dtype == np.float32:
                    return raw_data.flatten()
                else:
                    raw_bytes = raw_data.astype(np.int16).tobytes()
            else:
                raw_bytes = bytes(raw_data)

            int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
            if frame.num_channels > 1:
                int16_arr = int16_arr.reshape(-1, frame.num_channels)[:, 0]

            float32_arr = int16_arr.astype(np.float32) / 32768.0
            return float32_arr
        except Exception as e:
            logger.exception(f"[DITTO AUDIO ERROR] Failed converting AudioFrame: {e}")
            return None

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """
        Push incoming AudioFrame or AudioSegmentEnd into the generator.
        - AudioFrames are immediately queued to output for AV sync AND fed to audio buffer.
        - AudioSegmentEnd flushes buffer and signals completion.
        """
        if self._closed:
            return

        if not self._started:
            await self.start()

        if isinstance(frame, rtc.AudioFrame):
            # 1. Forward original AudioFrame directly to output queue for synchronized playback
            await self._output_queue.put(frame)

            # 2. Convert to float32 NumPy array
            audio_np = self._convert_audio_frame_to_numpy(frame)
            if audio_np is None or len(audio_np) == 0:
                return

            with self._buffer_lock:
                self._audio_buffer = np.concatenate([self._audio_buffer, audio_np], 0)
                self._ensure_initial_silence()

                # Extract available full processing chunks
                while True:
                    available = len(self._audio_buffer) - self._chunk_position
                    if available >= SPLIT_LEN:
                        chunk_start = self._chunk_position
                        chunk_end = chunk_start + SPLIT_LEN
                        chunk = self._audio_buffer[chunk_start:chunk_end]
                        self._chunk_position += ADVANCE
                        # Enqueue chunk for background inference
                        self._infer_queue.put_nowait(chunk)
                    else:
                        break

        elif isinstance(frame, AudioSegmentEnd):
            logger.info("[DITTO AUDIO] Received AudioSegmentEnd from LiveKit receiver.")
            self._segment_counter += 1
            seg_id = f"seg_{self._segment_counter}"
            self._current_segment_id = seg_id

            with self._buffer_lock:
                available = len(self._audio_buffer) - self._chunk_position
                if available > 0:
                    chunk_start = self._chunk_position
                    chunk = self._audio_buffer[chunk_start:]
                    pad_len = SPLIT_LEN - len(chunk)
                    if pad_len > 0:
                        chunk = np.pad(chunk, (0, pad_len), mode="constant")
                    self._chunk_position = len(self._audio_buffer)
                    self._infer_queue.put_nowait(chunk)

            # Enqueue segment end marker
            self._infer_queue.put_nowait(_SegmentEndMarker(segment_id=seg_id))

    async def _inference_worker(self):
        """Background task running Ditto inference without blocking asyncio loop."""
        logger.info("[DITTO GENERATOR] Inference worker loop started.")
        while not self._closed:
            try:
                item = await self._infer_queue.get()
                if item is None:
                    break

                if isinstance(item, np.ndarray):
                    # Run chunk inference in thread
                    await asyncio.to_thread(self._run_chunk_sync, item)
                elif isinstance(item, _SegmentEndMarker):
                    # Flush segment in thread
                    await asyncio.to_thread(self._flush_segment_sync, item.segment_id)
                    # Once all frames are flushed, notify AvatarRunner
                    await self._output_queue.put(AudioSegmentEnd())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"[DITTO GENERATOR ERROR] Error in inference worker: {e}")

    def _run_chunk_sync(self, chunk: np.ndarray):
        if self._engine.sdk is None or self._closed:
            return
        try:
            self._engine.sdk.run_chunk(chunk, CHUNKSIZE)
        except Exception as e:
            logger.exception(f"[DITTO INFERENCE ERROR] SDK.run_chunk failed: {e}")

    def _flush_segment_sync(self, segment_id: str):
        if self._engine.sdk is None or self._closed:
            return
        try:
            logger.info(f"[DITTO GENERATOR] Flushing segment '{segment_id}' through StreamSDK...")
            self._engine.sdk.flush_segment(segment_id=segment_id, timeout=30.0)
            logger.info(f"[DITTO GENERATOR] Segment '{segment_id}' successfully flushed.")

            # Reset per-segment audio buffer
            with self._buffer_lock:
                self._audio_buffer = np.zeros((0,), dtype=np.float32)
                self._chunk_position = 0
                self._initial_silence_prepended = False
        except Exception as e:
            logger.exception(f"[DITTO GENERATOR ERROR] SDK.flush_segment failed: {e}")

    def clear_buffer(self) -> None:
        """
        Clear audio and video buffers upon interruption.
        Discards pending inference chunks and queued output frames so new speech
        can start immediately without lag.
        """
        logger.info("[DITTO GENERATOR] clear_buffer() called - clearing pending audio/video queues")
        with self._buffer_lock:
            self._audio_buffer = np.zeros((0,), dtype=np.float32)
            self._chunk_position = 0
            self._initial_silence_prepended = False

        # Drain inference queue
        while not self._infer_queue.empty():
            try:
                self._infer_queue.get_nowait()
            except Exception:
                break

        # Drain output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except Exception:
                break

        logger.info("[DITTO GENERATOR] clear_buffer() completed cleanly.")

    async def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """Stream video frames, audio frames, and segment end markers to AvatarRunner."""
        while not self._closed:
            try:
                item = await self._output_queue.get()
                if item is None:
                    break
                yield item
            except asyncio.CancelledError:
                break

    async def aclose(self):
        """Clean teardown of Ditto generator and engine."""
        self._closed = True
        self._frame_writer.close()

        if self._infer_task is not None:
            self._infer_task.cancel()
            try:
                await self._infer_task
            except asyncio.CancelledError:
                pass

        await asyncio.to_thread(self._engine.stop)
        logger.info("[DITTO GENERATOR] DittoVideoGenerator closed cleanly.")
