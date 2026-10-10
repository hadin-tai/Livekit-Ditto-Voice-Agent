import os
import sys
import time
import math
import traceback
import threading
import queue
import logging
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2

FRAME_WIDTH = 1100
FRAME_HEIGHT = 600

BACKEND_DIR = Path(__file__).parent.parent.parent.parent
DITTO_DIR = Path(__file__).parent.parent
LOGS_DIR = BACKEND_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "ditto_integration.log"

ditto_logger = logging.getLogger("ditto_integration")
ditto_logger.setLevel(logging.DEBUG)
ditto_logger.propagate = False

if not ditto_logger.handlers:
    file_handler = logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(formatter)
    ditto_logger.addHandler(file_handler)

DITTO_CHECKPOINTS = DITTO_DIR / "checkpoints" / "ditto_pytorch"
DITTO_CFG_PKL = DITTO_DIR / "checkpoints" / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"
DITTO_SOURCE_IMAGE = DITTO_DIR / "example" / "image.png"
DUMMY_OUTPUT_PATH = "dummy_output.mp4"

CHUNKSIZE = (3, 5, 2)
SPLIT_LEN = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80
ADVANCE = CHUNKSIZE[1] * 640
INITIAL_SILENCE_SAMPLES = CHUNKSIZE[0] * 640

INITIAL_ND_FRAMES = 50000
WATCHDOG_INTERVAL_SEC = 5.0
OPENCV_WINDOW_NAME = "Ditto Avatar Live"

DITTO_DEBUG_DIR = LOGS_DIR / "ditto_debug"
DITTO_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_CHUNK_NPY = DITTO_DEBUG_DIR / "ref_chunk_00_silence_plus_audio.npy"


class EngineState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    PROCESSING = "PROCESSING"
    IDLE = "IDLE"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class SynthesizedAudioInfo:
    frame_data: Any
    sample_rate: int
    num_channels: int
    samples_per_channel: int
    duration: float
    request_id: str
    segment_id: str
    is_final: bool


@dataclass
class EngineStats:
    total_audio_samples_received: int = 0
    total_chunks_processed: int = 0
    total_frames_generated: int = 0
    total_segments_processed: int = 0
    engine_start_time: Optional[float] = None
    last_audio_time: Optional[float] = None
    last_chunk_time: Optional[float] = None
    last_frame_time: Optional[float] = None
    current_segment_id: Optional[str] = None
    current_request_id: Optional[str] = None
    current_operation: str = "idle"
    audio_buffer_size: int = 0
    chunk_position: int = 0


class EngineDisplayWriter:
    def __init__(self, video_path, fps=25, **kwargs):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_print_time = time.time()
        self.fps = 0.0
        self.frame_info_printed = False
        self._engine_ref: Optional["DittoEngine"] = None
        self._window_created = False
        self._window_owner_tid: Optional[int] = None
        cur_tid = threading.current_thread().ident
        cur_tname = threading.current_thread().name
        self.video_path = video_path
        self.fps_configured = fps
        ditto_logger.info(
            "[EngineDisplayWriter __INIT__] Constructor called | "
            "obj_id=%s cls=%s@%s | "
            "video_path=%s | fps=%d | TID=%s thread=%s | kwargs=%s",
            id(self),
            type(self).__name__,
            type(self).__module__,
            video_path,
            fps,
            cur_tid,
            cur_tname,
            kwargs,
        )

    def set_engine_ref(self, engine: "DittoEngine"):
        self._engine_ref = engine
        ditto_logger.info(
            "[EngineDisplayWriter set_engine_ref] obj_id=%s | engine_id=%s",
            id(self),
            id(engine) if engine is not None else "None",
        )

    def __call__(self, img, fmt="rgb"):
        """
        Receives a frame, converts it to BGR (for OpenCV), and displays it.
        Matches test_ditto_live.py DisplayWriter.__call__() behavior:
        - Window is created on THIS thread (writer_worker) on the first frame
        - cv2.imshow() and cv2.waitKey() always run on the same thread that owns the window
        """
        t0 = time.time()
        cur_tid = threading.current_thread().ident
        cur_tname = threading.current_thread().name
        try:
            self.frame_count += 1
            if self._engine_ref is not None:
                self._engine_ref._stats.total_frames_generated = self.frame_count
                self._engine_ref._stats.last_frame_time = t0

            ditto_logger.info(
                "[EngineDisplayWriter CALL] Frame #%d START | TID=%s thread=%s | writer_id=%s cls=%s@%s | "
                "fmt=%s | img_shape=%s | img_dtype=%s | window_owner_TID=%s",
                self.frame_count,
                cur_tid,
                cur_tname,
                id(self),
                type(self).__name__,
                type(self).__module__,
                fmt,
                getattr(img, "shape", "?"),
                getattr(img, "dtype", "?"),
                self._window_owner_tid,
            )

            # --- Create window on FIRST frame (on the writer_worker thread) ---
            # This matches test_ditto_live.py where DisplayWriter.__call__()
            # implicitly creates the window via the first cv2.imshow() call.
            # All subsequent cv2.imshow()/cv2.waitKey() calls happen on this
            # same thread, avoiding the cross-thread HighGUI issue on Windows.
            if not self._window_created:
                ditto_logger.info(
                    "[EngineDisplayWriter] WINDOW CREATED ON FIRST FRAME | "
                    "Creating cv2.namedWindow('%s') on TID=%s thread=%s "
                    "(this is the writer_worker thread - matches test_ditto_live.py)",
                    OPENCV_WINDOW_NAME,
                    cur_tid,
                    cur_tname,
                )
                try:
                    cv2.namedWindow(OPENCV_WINDOW_NAME, cv2.WINDOW_NORMAL)
                    self._window_created = True
                    self._window_owner_tid = cur_tid
                    ditto_logger.info(
                        "[EngineDisplayWriter] cv2.namedWindow('%s') SUCCESS | "
                        "window_owner_TID=%s thread=%s",
                        OPENCV_WINDOW_NAME,
                        self._window_owner_tid,
                        cur_tname,
                    )
                except Exception as win_e:
                    ditto_logger.warning(
                        "[EngineDisplayWriter] cv2.namedWindow raised (continuing with implicit create): %s",
                        win_e,
                    )
                    # Even if namedWindow fails, imshow will create it implicitly
                    self._window_created = True
                    self._window_owner_tid = cur_tid

            # Warn if cv2.imshow is called from a different thread than window owner
            if self._window_owner_tid is not None and cur_tid != self._window_owner_tid:
                ditto_logger.warning(
                    "[EngineDisplayWriter CROSS-THREAD WARNING] cv2.imshow() called from TID=%s "
                    "but window was created on TID=%s — this causes blank/frozen window on Windows!",
                    cur_tid,
                    self._window_owner_tid,
                )

            if img is not None:
                ditto_logger.info(
                    "[EngineDisplayWriter CALL] Frame #%d stats | min=%s max=%s mean=%.6f std=%.6f",
                    self.frame_count,
                    float(np.min(img)) if hasattr(img, "shape") else "N/A",
                    float(np.max(img)) if hasattr(img, "shape") else "N/A",
                    float(np.mean(img)) if hasattr(img, "shape") else "N/A",
                    float(np.std(img)) if hasattr(img, "shape") else "N/A",
                )

            # Convert RGB to BGR for OpenCV (identical to test_ditto_live.py)
            if fmt == "rgb":
                frame_bgr = img[..., ::-1]
            else:
                frame_bgr = img

            frame_bgr = cv2.resize(
                frame_bgr,
                (FRAME_WIDTH, FRAME_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            
            # Create window
            if not self._window_created:
                cv2.namedWindow(OPENCV_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    OPENCV_WINDOW_NAME,
                    FRAME_WIDTH,
                    FRAME_HEIGHT,
                )
                self._window_created = True
                self._window_owner_tid = cur_tid

            cv2.imshow(OPENCV_WINDOW_NAME, frame_bgr)
            cv2.waitKey(1)


            # Print frame information once (matches test_ditto_live.py)
            if not self.frame_info_printed:
                ditto_logger.info(
                    "[EngineDisplayWriter] FIRST GENERATED FRAME | "
                    "shape=%s | dtype=%s | fmt_src=%s | frame_idx=%d | TID=%s | "
                    "window_owner_TID=%s (same_thread=%s)",
                    frame_bgr.shape,
                    frame_bgr.dtype,
                    fmt,
                    self.frame_count,
                    cur_tid,
                    self._window_owner_tid,
                    cur_tid == self._window_owner_tid,
                )
                self.frame_info_printed = True

            # Display the frame (identical to test_ditto_live.py)
            ditto_logger.info(
                "[EngineDisplayWriter cv2.imshow ENTER] window=%s | frame shape=%s dtype=%s | TID=%s thread=%s",
                OPENCV_WINDOW_NAME,
                frame_bgr.shape,
                frame_bgr.dtype,
                cur_tid,
                cur_tname,
            )
            t_imshow0 = time.time()
            cv2.imshow(OPENCV_WINDOW_NAME, frame_bgr)
            t_imshow1 = time.time()
            ditto_logger.info(
                "[EngineDisplayWriter cv2.imshow EXIT] window=%s | elapsed=%.2f ms | TID=%s",
                OPENCV_WINDOW_NAME,
                (t_imshow1 - t_imshow0) * 1000.0,
                cur_tid,
            )

            # Calculate and print FPS every second (identical to test_ditto_live.py)
            current_time = time.time()
            if current_time - self.last_fps_print_time >= 1.0:
                elapsed = current_time - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0.0
                ditto_logger.info(
                    "[EngineDisplayWriter FPS] | FPS=%.1f | Total Frames=%d | Elapsed=%.2fs | TID=%s",
                    self.fps,
                    self.frame_count,
                    elapsed,
                    cur_tid,
                )
                self.last_fps_print_time = current_time

            # Check for exit key (identical to test_ditto_live.py)
            t_wait0 = time.time()
            key = cv2.waitKey(1) & 0xFF
            t_wait1 = time.time()
            if key == ord("q"):
                ditto_logger.info("[EngineDisplayWriter] Quit key (q) pressed")
            ditto_logger.debug(
                "[EngineDisplayWriter cv2.waitKey] retval=%d | took %.2f ms | TID=%s",
                key,
                (t_wait1 - t_wait0) * 1000.0,
                cur_tid,
            )

            render_latency_ms = (time.time() - t0) * 1000.0
            ditto_logger.info(
                "[EngineDisplayWriter CALL] Frame #%d END | total_render_latency=%.2f ms | TID=%s",
                self.frame_count,
                render_latency_ms,
                cur_tid,
            )

        except Exception as e:
            ditto_logger.exception(
                "[EngineDisplayWriter ERROR] Frame #%d render failed TID=%s: %s\n%s",
                self.frame_count,
                cur_tid,
                e,
                traceback.format_exc(),
            )

    def close(self):
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0.0
        ditto_logger.info(
            "[EngineDisplayWriter CLOSE] START | Total Time=%.2fs | Total Frames=%d | Avg FPS=%.1f",
            total_time,
            self.frame_count,
            avg_fps,
        )
        try:
            cv2.destroyAllWindows()
            ditto_logger.info("[EngineDisplayWriter CLOSE] OpenCV windows destroyed successfully")
        except Exception as e:
            ditto_logger.exception(
                "[EngineDisplayWriter CLOSE ERROR] destroyAllWindows failed: %s",
                e,
            )
        ditto_logger.info("[EngineDisplayWriter CLOSE] END")


class DittoEngine:
    def __init__(
        self,
        checkpoints_dir: Optional[Path] = None,
        cfg_pkl: Optional[Path] = None,
        source_image: Optional[Path] = None,
        initial_nd_frames: int = INITIAL_ND_FRAMES,
    ):
        ditto_logger.info("=" * 80)
        ditto_logger.info("[DittoEngine __init__] START")
        self.checkpoints_dir = checkpoints_dir or DITTO_CHECKPOINTS
        self.cfg_pkl = cfg_pkl or DITTO_CFG_PKL
        self.source_image = source_image or DITTO_SOURCE_IMAGE
        self.initial_nd_frames = initial_nd_frames

        self._state: EngineState = EngineState.STARTING
        self._state_lock = threading.Lock()

        self.sdk: Optional[Any] = None
        self._spo_module = None
        self._StreamSDK_cls = None
        self._display_writer: Optional[EngineDisplayWriter] = None

        self._audio_buffer: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._chunk_position: int = 0
        self._initial_silence_prepended: bool = False

        self._stats: EngineStats = EngineStats()

        self._stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._closed: bool = False

        ditto_logger.info(
            "[DittoEngine __init__] Config | checkpoints_dir=%s | cfg_pkl=%s | source_image=%s",
            str(self.checkpoints_dir),
            str(self.cfg_pkl),
            str(self.source_image),
        )
        ditto_logger.info(
            "[DittoEngine __init__] Chunk config | chunksize=%s | split_len=%d | advance=%d | initial_silence=%d",
            CHUNKSIZE,
            SPLIT_LEN,
            ADVANCE,
            INITIAL_SILENCE_SAMPLES,
        )
        ditto_logger.info(
            "[DittoEngine __init__] initial_nd_frames=%d",
            self.initial_nd_frames,
        )
        ditto_logger.info("[DittoEngine __init__] END")
        ditto_logger.info("=" * 80)

    def _set_state(self, new_state: EngineState, operation: Optional[str] = None):
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            if operation is not None:
                self._stats.current_operation = operation
        ditto_logger.info(
            "[DittoEngine STATE] %s -> %s | operation=%s",
            old_state.value,
            new_state.value,
            operation or self._stats.current_operation,
        )

    @property
    def state(self) -> EngineState:
        with self._state_lock:
            return self._state

    def _verify_files(self):
        ditto_logger.info("[DittoEngine _verify_files] START")
        t0 = time.time()

        if not self.cfg_pkl.exists():
            raise FileNotFoundError(f"Ditto config not found: {self.cfg_pkl}")
        ditto_logger.info("[DittoEngine _verify_files] cfg_pkl OK: %s", str(self.cfg_pkl))

        if not self.source_image.exists():
            raise FileNotFoundError(f"Ditto source image not found: {self.source_image}")
        ditto_logger.info("[DittoEngine _verify_files] source_image OK: %s", str(self.source_image))

        if not self.checkpoints_dir.exists():
            raise FileNotFoundError(f"Ditto checkpoints dir not found: {self.checkpoints_dir}")
        ditto_logger.info("[DittoEngine _verify_files] checkpoints_dir OK: %s", str(self.checkpoints_dir))

        ckpt_files = list(self.checkpoints_dir.glob("**/*.pth")) + list(self.checkpoints_dir.glob("**/*.pt"))
        ditto_logger.info(
            "[DittoEngine _verify_files] Found %d checkpoint files in %s",
            len(ckpt_files),
            str(self.checkpoints_dir),
        )

        elapsed = (time.time() - t0) * 1000.0
        ditto_logger.info(
            "[DittoEngine _verify_files] END | elapsed=%.2f ms",
            elapsed,
        )

    def _apply_monkey_patch(self):
        cur_tid = threading.current_thread().ident
        cur_tname = threading.current_thread().name
        ditto_logger.info("[DittoEngine _apply_monkey_patch] START | TID=%s thread=%s", cur_tid, cur_tname)
        t0 = time.time()

        if str(DITTO_DIR) not in sys.path:
            sys.path.insert(0, str(DITTO_DIR))
            ditto_logger.info("[DittoEngine _apply_monkey_patch] Added DITTO_DIR to sys.path[0]: %s", str(DITTO_DIR))

        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] [STEP 1/3 matching test_ditto_live.py] "
            ">>> import stream_pipeline_online (TID=%s)",
            cur_tid,
        )
        import stream_pipeline_online as _spo
        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] stream_pipeline_online imported | module_id=%s | file=%s",
            id(_spo),
            getattr(_spo, "__file__", "?"),
        )

        orig_vw_cls = getattr(_spo, "VideoWriterByImageIO", None)
        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] BEFORE patch - VideoWriterByImageIO current value BEFORE patch: name=%s module=%s id=%s",
            orig_vw_cls.__name__ if orig_vw_cls is not None else "None",
            orig_vw_cls.__module__ if orig_vw_cls is not None else "N/A",
            id(orig_vw_cls) if orig_vw_cls is not None else "N/A",
        )

        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] [STEP 2/3 matching test_ditto_live.py] "
            ">>> stream_pipeline_online.VideoWriterByImageIO = EngineDisplayWriter (TID=%s)",
            cur_tid,
        )
        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] EngineDisplayWriter identity: name=%s module=%s id=%s",
            EngineDisplayWriter.__name__,
            EngineDisplayWriter.__module__,
            id(EngineDisplayWriter),
        )
        _spo.VideoWriterByImageIO = EngineDisplayWriter
        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] AFTER patch - stream_pipeline_online.VideoWriterByImageIO is: name=%s module=%s id=%s (expected=%s match=%s",
            _spo.VideoWriterByImageIO.__name__,
            _spo.VideoWriterByImageIO.__module__,
            id(_spo.VideoWriterByImageIO),
            id(EngineDisplayWriter),
            _spo.VideoWriterByImageIO is EngineDisplayWriter,
        )

        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] [STEP 3/3 matching test_ditto_live.py] "
            ">>> from stream_pipeline_online import StreamSDK (TID=%s)",
            cur_tid,
        )
        from stream_pipeline_online import StreamSDK
        self._spo_module = _spo
        self._StreamSDK_cls = StreamSDK
        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] StreamSDK imported: name=%s module=%s id=%s | uses patched module=%s",
            StreamSDK.__name__,
            StreamSDK.__module__,
            id(StreamSDK),
            StreamSDK.__module__ == _spo.__name__,
        )
        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] FINAL VERIFY: patched module VideoWriterByImageIO is EngineDisplayWriter? -> %s (id=%s vs id=%s)",
            _spo.VideoWriterByImageIO is EngineDisplayWriter,
            id(_spo.VideoWriterByImageIO),
            id(EngineDisplayWriter),
        )
        if "core.atomic_components.writer" in sys.modules:
            core_writer_module = sys.modules["core.atomic_components.writer"]
            core_vw = getattr(core_writer_module, "VideoWriterByImageIO", None)
            ditto_logger.info(
                "[DittoEngine _apply_monkey_patch] core.atomic_components.writer VideoWriterByImageIO: id=%s name=%s (same_as_EngineDisplayWriter?=%s)",
                id(core_vw) if core_vw is not None else "N/A",
                core_vw.__name__ if core_vw is not None else "N/A",
                core_vw is EngineDisplayWriter,
            )

        elapsed = (time.time() - t0) * 1000.0
        ditto_logger.info(
            "[DittoEngine _apply_monkey_patch] END | elapsed=%.2f ms | TID=%s",
            elapsed,
            cur_tid,
        )

    def _create_sdk_instance(self):
        cur_tid = threading.current_thread().ident
        cur_tname = threading.current_thread().name
        ditto_logger.info(
            "[DittoEngine _create_sdk_instance] START | TID=%s thread=%s",
            cur_tid,
            cur_tname,
        )
        t0 = time.time()

        ditto_logger.info(
            "[DittoEngine _create_sdk_instance] Creating StreamSDK(cfg_pkl=%s, data_root=%s) | TID=%s",
            str(self.cfg_pkl),
            str(self.checkpoints_dir),
            cur_tid,
        )
        self.sdk = self._StreamSDK_cls(
            str(self.cfg_pkl),
            str(self.checkpoints_dir),
        )
        ditto_logger.info(
            "[DittoEngine _create_sdk_instance] StreamSDK instance created | sdk_id=%s | TID=%s",
            id(self.sdk),
            cur_tid,
        )

        elapsed = (time.time() - t0) * 1000.0
        ditto_logger.info(
            "[DittoEngine _create_sdk_instance] END | StreamSDK created | elapsed=%.2f ms | TID=%s",
            elapsed,
            cur_tid,
        )

    def _diagnose_pipeline_state(self, tag: str = "diagnose"):
        if self.sdk is None:
            ditto_logger.warning(
                "[DittoEngine PIPELINE %s] SDK is None - cannot diagnose",
                tag,
            )
            return

        stop_set = getattr(self.sdk, "stop_event", None)
        stop_set = stop_set.is_set() if stop_set is not None else "N/A"
        worker_exc = getattr(self.sdk, "worker_exception", None)
        thread_list = getattr(self.sdk, "thread_list", [])
        threads_alive = sum(1 for t in thread_list if t.is_alive())
        threads_total = len(thread_list)

        queue_names = [
            "audio2motion_queue",
            "motion_stitch_queue",
            "warp_f3d_queue",
            "decode_f3d_queue",
            "putback_queue",
            "writer_queue",
        ]
        q_sizes = {}
        for qn in queue_names:
            q = getattr(self.sdk, qn, None)
            q_sizes[qn] = q.qsize() if q is not None else "N/A"

        writer_obj = getattr(self.sdk, "writer", None)
        writer_cls_name = type(writer_obj).__name__ if writer_obj is not None else "None"
        writer_module = type(writer_obj).__module__ if writer_obj is not None else "N/A"

        ditto_logger.info(
            "[DittoEngine PIPELINE %s] stop_event=%s | worker_exception=%s | "
            "threads_alive=%d/%d | writer_cls=%s@%s | "
            "q_sizes: audio2motion=%s | motion_stitch=%s | warp_f3d=%s | decode_f3d=%s | putback=%s | writer=%s",
            tag,
            stop_set,
            type(worker_exc).__name__ if worker_exc is not None else "None",
            threads_alive,
            threads_total,
            writer_cls_name,
            writer_module,
            q_sizes["audio2motion_queue"],
            q_sizes["motion_stitch_queue"],
            q_sizes["warp_f3d_queue"],
            q_sizes["decode_f3d_queue"],
            q_sizes["putback_queue"],
            q_sizes["writer_queue"],
        )
        if worker_exc is not None:
            ditto_logger.error(
                "[DittoEngine PIPELINE %s] worker_exception traceback:\n%s",
                tag,
                "".join(
                    traceback.format_exception(
                        type(worker_exc),
                        worker_exc,
                        getattr(worker_exc, "__traceback__", None),
                    )
                ),
            )
        for i, t in enumerate(thread_list):
            ditto_logger.debug(
                "[DittoEngine PIPELINE %s] thread[%d] name=%s ident=%s alive=%s",
                tag,
                i,
                getattr(t, "name", "?"),
                getattr(t, "ident", "?"),
                t.is_alive() if hasattr(t, "is_alive") else "?",
            )

    def _raise_if_worker_failed(self, context: str = "run_chunk"):
        worker_exc = getattr(self.sdk, "worker_exception", None)
        if worker_exc is not None:
            ditto_logger.exception(
                "[DittoEngine PIPELINE CRASH %s] Detected worker_exception: %s\n%s",
                context,
                worker_exc,
                "".join(
                    traceback.format_exception(
                        type(worker_exc),
                        worker_exc,
                        getattr(worker_exc, "__traceback__", None),
                    )
                ),
            )
            raise RuntimeError(
                f"Ditto SDK pipeline worker crashed in [{context}]: {worker_exc!r}"
            ) from worker_exc
        stop_set = getattr(self.sdk, "stop_event", None)
        stop_set_flag = stop_set.is_set() if stop_set is not None else False
        if stop_set_flag:
            ditto_logger.error(
                "[DittoEngine PIPELINE STOPPED %s] SDK.stop_event is already set but worker_exception is None - pipeline halted unexpectedly",
                context,
            )
            raise RuntimeError(
                f"Ditto SDK pipeline stop_event set unexpectedly in [{context}]"
            )

    def _call_sdk_setup(self):
        cur_tid = threading.current_thread().ident
        cur_tname = threading.current_thread().name
        ditto_logger.info(
            "[DittoEngine _call_sdk_setup] START | TID=%s thread=%s | OUTPUT_PATH='%s' (exact same string as test_ditto_live.py)",
            cur_tid,
            cur_tname,
            DUMMY_OUTPUT_PATH,
        )
        t0 = time.time()

        ditto_logger.info(
            "[DittoEngine _call_sdk_setup] >>> Calling SDK.setup(source_path=%s, output_path='%s', online_mode=True, emo=4, N_d=%d, overlap_v2=70, sampling_timesteps=25) | TID=%s",
            str(self.source_image),
            DUMMY_OUTPUT_PATH,
            self.initial_nd_frames,
            cur_tid,
        )
        self.sdk.setup(
            str(self.source_image),
            DUMMY_OUTPUT_PATH,
            online_mode=True,
            emo=4,
            N_d=self.initial_nd_frames,
            overlap_v2=70,
            sampling_timesteps=25,
        )
        t_setup = time.time()
        ditto_logger.info(
            "[DittoEngine _call_sdk_setup] SDK.setup() returned after %.2f ms - checking worker_exception/stop_event immediately (TID=%s)",
            (t_setup - t0) * 1000.0,
            cur_tid,
        )
        self._raise_if_worker_failed(context="sdk_setup_post")

        if hasattr(self.sdk, "tmp_output_path"):
            ditto_logger.info(
                "[DittoEngine _call_sdk_setup] sdk.tmp_output_path=%s | sdk.output_path=%s",
                getattr(self.sdk, "tmp_output_path", "N/A"),
                getattr(self.sdk, "output_path", "N/A"),
            )

        ditto_logger.info(
            "[DittoEngine _call_sdk_setup] WRITER VALIDATION - inspecting sdk.writer class AFTER setup()"
        )
        raw_writer = getattr(self.sdk, "writer", None)
        ditto_logger.info(
            "[DittoEngine _call_sdk_setup] raw writer from sdk.writer: type=%s module=%s id=%s",
            type(raw_writer).__name__ if raw_writer is not None else "None",
            type(raw_writer).__module__ if raw_writer is not None else "N/A",
            id(raw_writer) if raw_writer is not None else "N/A",
        )
        ditto_logger.info(
            "[DittoEngine _call_sdk_setup] EngineDisplayWriter class identity: name=%s module=%s id=%s",
            EngineDisplayWriter.__name__,
            EngineDisplayWriter.__module__,
            id(EngineDisplayWriter),
        )
        is_engine_writer = isinstance(raw_writer, EngineDisplayWriter)
        ditto_logger.info(
            "[DittoEngine _call_sdk_setup] isinstance(raw_writer, EngineDisplayWriter) = %s",
            is_engine_writer,
        )

        if not is_engine_writer:
            ditto_logger.warning(
                "[DittoEngine _call_sdk_setup] WRITER MISMATCH - monkey-patch did not resolve to EngineDisplayWriter. "
                "EXPLICITLY replacing sdk.writer with a fresh EngineDisplayWriter instance."
            )
            replacement_writer = EngineDisplayWriter(DUMMY_OUTPUT_PATH, fps=25)
            replacement_writer.set_engine_ref(self)
            self.sdk.writer = replacement_writer
            self._display_writer = replacement_writer
            ditto_logger.info(
                "[DittoEngine _call_sdk_setup] WRITER REPLACED OK: sdk.writer now = %s id=%s",
                type(self.sdk.writer).__name__,
                id(self.sdk.writer),
            )
        else:
            ditto_logger.info(
                "[DittoEngine _call_sdk_setup] WRITER OK - monkey-patch correctly produced EngineDisplayWriter"
            )
            self._display_writer = raw_writer
            if hasattr(self._display_writer, "set_engine_ref"):
                self._display_writer.set_engine_ref(self)
                ditto_logger.info(
                    "[DittoEngine _call_sdk_setup] Engine reference wired to existing display writer"
                )

        self._diagnose_pipeline_state(tag="AFTER_SDK_SETUP")
        self._raise_if_worker_failed(context="after_sdk_setup_diagnose")

        elapsed = (time.time() - t0) * 1000.0
        ditto_logger.info(
            "[DittoEngine _call_sdk_setup] END | SDK.setup() completed + writer verified + workers alive | elapsed=%.2f ms | TID=%s",
            elapsed,
            cur_tid,
        )

    def _call_setup_nd(self, n_d_frames: int, context: str = "initial"):
        cur_tid = threading.current_thread().ident
        cur_tname = threading.current_thread().name
        ditto_logger.info(
            "[DittoEngine _call_setup_nd] START | context=%s | N_d=%d | TID=%s thread=%s",
            context,
            n_d_frames,
            cur_tid,
            cur_tname,
        )
        t0 = time.time()

        self.sdk.setup_Nd(
            N_d=n_d_frames,
            fade_in=-1,
            fade_out=-1,
            ctrl_info={},
        )
        actual_sdk_Nd = getattr(self.sdk, "N_d", None)
        ditto_logger.info(
            "[DittoEngine _call_setup_nd] SDK.N_d after setup_Nd() call: %s",
            actual_sdk_Nd,
        )

        elapsed = (time.time() - t0) * 1000.0
        ditto_logger.info(
            "[DittoEngine _call_setup_nd] END | context=%s | N_d=%d (sdk.N_d=%s) | elapsed=%.2f ms | TID=%s",
            context,
            n_d_frames,
            actual_sdk_Nd,
            elapsed,
            cur_tid,
        )

    def _initialize_opencv_window(self):
        """Verify source image exists but do NOT create the OpenCV window here.

        The window must be created on the writer_worker thread to avoid
        cross-thread HighGUI issues on Windows (window message queue belongs
        to the thread that created it). Window creation is deferred to
        EngineDisplayWriter.__call__() on the first frame — matching exactly
        how test_ditto_live.py works.
        """
        cur_tid = threading.current_thread().ident
        cur_tname = threading.current_thread().name
        ditto_logger.info(
            "[DittoEngine _initialize_opencv_window] START | TID=%s thread=%s",
            cur_tid,
            cur_tname,
        )
        t0 = time.time()

        # --- DO NOT call cv2.namedWindow / cv2.imshow / cv2.waitKey here ---
        # Reason: This method runs on the job_thread_runner (TID=%s), but
        # EngineDisplayWriter.__call__() runs on the writer_worker thread.
        # On Windows, OpenCV HighGUI binds the window message queue to the
        # creating thread. Cross-thread imshow causes blank/frozen window
        # and waitKey returns 255 instead of -1.
        ditto_logger.info(
            "[DittoEngine _initialize_opencv_window] SKIPPING premature cv2.namedWindow/imshow | "
            "Reason: Window must be created on writer_worker thread to avoid "
            "cross-thread OpenCV issue | Window will be created by "
            "EngineDisplayWriter.__call__() on first frame | "
            "init_thread_TID=%s (%s) - this thread must NOT own the OpenCV window",
            cur_tid,
            cur_tname,
        )

        # Only verify the source image exists (no GUI calls)
        ditto_logger.info(
            "[DittoEngine _initialize_opencv_window] Verifying source image exists: %s",
            str(self.source_image),
        )
        idle_frame_bgr = cv2.imread(str(self.source_image), cv2.IMREAD_COLOR)
        if idle_frame_bgr is None:
            ditto_logger.error(
                "[DittoEngine _initialize_opencv_window] FAILED to load source image: %s",
                str(self.source_image),
            )
        else:
            ditto_logger.info(
                "[DittoEngine _initialize_opencv_window] Source image verified OK | shape=%s | dtype=%s",
                idle_frame_bgr.shape,
                idle_frame_bgr.dtype,
            )

        elapsed = (time.time() - t0) * 1000.0
        ditto_logger.info(
            "[DittoEngine _initialize_opencv_window] END | Window creation deferred to writer_worker thread | elapsed=%.2f ms | TID=%s",
            elapsed,
            cur_tid,
        )

    def _start_watchdog(self):
        ditto_logger.info("[DittoEngine _start_watchdog] START")
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="DittoEngineWatchdog",
            daemon=True,
        )
        self._watchdog_thread.start()
        ditto_logger.info(
            "[DittoEngine _start_watchdog] Watchdog thread started (TID=%d, interval=%.1fs)",
            self._watchdog_thread.ident,
            WATCHDOG_INTERVAL_SEC,
        )

    def _watchdog_loop(self):
        ditto_logger.info(
            "[DittoEngine WATCHDOG] Loop START (TID=%d)",
            threading.current_thread().ident,
        )
        log_count = 0
        prev_frames_gen = 0
        prev_chunks_proc = 0
        prev_samples_recv = 0
        try:
            while not self._stop_event.is_set():
                log_count += 1
                with self._state_lock:
                    state_val = self._state.value
                with self._buffer_lock:
                    buf_size = len(self._audio_buffer)
                    chunk_pos = self._chunk_position

                elapsed_s = (
                    (time.time() - self._stats.engine_start_time)
                    if self._stats.engine_start_time is not None
                    else 0.0
                )

                stop_set = "N/A"
                worker_exc = "None"
                threads_alive = "N/A"
                threads_total = "N/A"
                q_sizes = {}
                queue_names = [
                    "audio2motion_queue",
                    "motion_stitch_queue",
                    "warp_f3d_queue",
                    "decode_f3d_queue",
                    "putback_queue",
                    "writer_queue",
                ]
                writer_cls = "N/A"
                if self.sdk is not None:
                    stop_evt = getattr(self.sdk, "stop_event", None)
                    stop_set = stop_evt.is_set() if stop_evt is not None else "N/A"
                    wexc = getattr(self.sdk, "worker_exception", None)
                    worker_exc = type(wexc).__name__ if wexc is not None else "None"
                    if wexc is not None and log_count % 2 == 1:
                        ditto_logger.error(
                            "[DittoEngine WATCHDOG #%d] worker_exception FULL:\n%s",
                            log_count,
                            "".join(
                                traceback.format_exception(
                                    type(wexc),
                                    wexc,
                                    getattr(wexc, "__traceback__", None),
                                )
                            ),
                        )
                    tlist = getattr(self.sdk, "thread_list", [])
                    threads_total = len(tlist)
                    threads_alive = sum(1 for t in tlist if t.is_alive())
                    for qn in queue_names:
                        q = getattr(self.sdk, qn, None)
                        q_sizes[qn] = q.qsize() if q is not None else "N/A"
                    wobj = getattr(self.sdk, "writer", None)
                    writer_cls = type(wobj).__name__ if wobj is not None else "None"

                frames_gen = self._stats.total_frames_generated
                chunks_proc = self._stats.total_chunks_processed
                samples_recv = self._stats.total_audio_samples_received
                delta_frames = frames_gen - prev_frames_gen
                delta_chunks = chunks_proc - prev_chunks_proc
                delta_samples = samples_recv - prev_samples_recv
                prev_frames_gen = frames_gen
                prev_chunks_proc = chunks_proc
                prev_samples_recv = samples_recv

                ditto_logger.info(
                    "[DittoEngine WATCHDOG #%d] | alive=%s | state=%s | "
                    "stop_event=%s | worker_exception=%s | threads=%s/%s | writer_cls=%s | "
                    "segment_id=%s | request_id=%s | operation=%s | "
                    "audio_buf=%d | chunk_pos=%d | "
                    "samples_recv=%d (+%d) | chunks_proc=%d (+%d) | frames_gen=%d (+%d) | segments_proc=%d | "
                    "q_audio2motion=%s q_motion_stitch=%s q_warp_f3d=%s q_decode_f3d=%s q_putback=%s q_writer=%s | "
                    "elapsed=%.1fs",
                    log_count,
                    not self._stop_event.is_set() and not self._closed,
                    state_val,
                    stop_set,
                    worker_exc,
                    threads_alive,
                    threads_total,
                    writer_cls,
                    self._stats.current_segment_id or "None",
                    self._stats.current_request_id or "None",
                    self._stats.current_operation,
                    buf_size,
                    chunk_pos,
                    samples_recv,
                    delta_samples,
                    chunks_proc,
                    delta_chunks,
                    frames_gen,
                    delta_frames,
                    self._stats.total_segments_processed,
                    q_sizes.get("audio2motion_queue", "N/A"),
                    q_sizes.get("motion_stitch_queue", "N/A"),
                    q_sizes.get("warp_f3d_queue", "N/A"),
                    q_sizes.get("decode_f3d_queue", "N/A"),
                    q_sizes.get("putback_queue", "N/A"),
                    q_sizes.get("writer_queue", "N/A"),
                    elapsed_s,
                )
                self._stop_event.wait(WATCHDOG_INTERVAL_SEC)
        except Exception as e:
            ditto_logger.exception(
                "[DittoEngine WATCHDOG ERROR] Watchdog crashed: %s\n%s",
                e,
                traceback.format_exc(),
            )
        ditto_logger.info(
            "[DittoEngine WATCHDOG] Loop END | total_watchdog_logs=%d",
            log_count,
        )

    def start(self):
        ditto_logger.info("[DittoEngine start] ================ START CALLED ================")
        t_total0 = time.time()
        self._stats.engine_start_time = t_total0
        self._set_state(EngineState.STARTING, operation="engine_initializing")

        try:
            self._set_state(EngineState.STARTING, operation="verifying_files")
            self._verify_files()

            self._set_state(EngineState.STARTING, operation="applying_monkey_patch")
            self._apply_monkey_patch()

            self._set_state(EngineState.STARTING, operation="creating_stream_sdk")
            self._create_sdk_instance()

            self._set_state(EngineState.STARTING, operation="calling_sdk_setup")
            self._call_sdk_setup()

            self._set_state(EngineState.STARTING, operation="calling_setup_nd_initial")
            self._call_setup_nd(self.initial_nd_frames, context="initial_engine_start")

            self._set_state(EngineState.STARTING, operation="warming_up_sdk")
            ditto_logger.info("[DittoEngine start] >>> Triggering StreamSDK.warmup() (one-time pre-warm)...")
            t_warm0 = time.time()
            self.sdk.warmup(timeout=15.0)
            t_warm1 = time.time()
            ditto_logger.info(
                "[DittoEngine start] StreamSDK.warmup() completed in %.2f ms | d0 established | Ditto READY",
                (t_warm1 - t_warm0) * 1000.0,
            )

            self._set_state(EngineState.STARTING, operation="initializing_opencv")
            self._initialize_opencv_window()

            self._set_state(EngineState.STARTING, operation="starting_watchdog")
            self._start_watchdog()

            total_elapsed = (time.time() - t_total0) * 1000.0
            ditto_logger.info(
                "[DittoEngine start] ================ INITIALIZATION COMPLETE ================"
            )
            ditto_logger.info(
                "[DittoEngine start] Total initialization elapsed=%.2f ms",
                total_elapsed,
            )
            self._set_state(EngineState.READY, operation="waiting_for_audio")
            ditto_logger.info(
                "[DittoEngine start] ================ READY / WAITING FOR AUDIO ================"
            )
            return True

        except Exception as e:
            total_elapsed = (time.time() - t_total0) * 1000.0
            ditto_logger.exception(
                "[DittoEngine start FATAL] Initialization FAILED after %.2f ms: %s\n%s",
                total_elapsed,
                e,
                traceback.format_exc(),
            )
            self._set_state(EngineState.ERROR, operation=f"init_failed:{type(e).__name__}")
            raise

    def _ensure_initial_silence(self):
        if self._initial_silence_prepended:
            return
        cur_tid = threading.current_thread().ident
        ditto_logger.info(
            "[DittoEngine _ensure_initial_silence] [MATCHING test_ditto_live.py line 140] Prepending initial silence: chunksize[0]*640 = %d*640 = %d samples (%.2f ms) | TID=%s",
            CHUNKSIZE[0],
            INITIAL_SILENCE_SAMPLES,
            INITIAL_SILENCE_SAMPLES / 16000 * 1000.0,
            cur_tid,
        )
        silence = np.zeros((INITIAL_SILENCE_SAMPLES,), dtype=np.float32)
        self._audio_buffer = np.concatenate([silence, self._audio_buffer], 0)
        self._chunk_position = 0
        self._initial_silence_prepended = True
        prefix_stats = self._audio_buffer[:INITIAL_SILENCE_SAMPLES]
        zero_count = int(np.sum(prefix_stats == 0.0))
        ditto_logger.info(
            "[DittoEngine _ensure_initial_silence] VERIFY PREFIX | "
            "buffer[0:%d] zero_count=%d/%d expected=%d (100%% match=%s) | "
            "prefix_min=%.6f prefix_max=%.6f prefix_mean=%.6f | "
            "Buffer total after prepend: %d samples | TID=%s",
            INITIAL_SILENCE_SAMPLES,
            zero_count,
            INITIAL_SILENCE_SAMPLES,
            INITIAL_SILENCE_SAMPLES,
            zero_count == INITIAL_SILENCE_SAMPLES,
            float(np.min(prefix_stats)),
            float(np.max(prefix_stats)),
            float(np.mean(prefix_stats)),
            len(self._audio_buffer),
            cur_tid,
        )

    def _update_nd_if_needed(self, additional_samples: int):
        total_samples_including_silence = (
            self._stats.total_audio_samples_received + INITIAL_SILENCE_SAMPLES
        )
        new_frames_estimate_test_formula = math.ceil(total_samples_including_silence / 16000 * 25)
        new_frames_estimate = new_frames_estimate_test_formula + 100
        current_nd = getattr(self.sdk, "N_d", self.initial_nd_frames)
        if new_frames_estimate_test_formula > current_nd:
            ditto_logger.info(
                "[DittoEngine _update_nd_if_needed] [MATCHING test_ditto_live.py line 111 formula: ceil(audio_len / 16000 * 25)] "
                "N_d: current=%d -> proposed_test_formula=%d (ceil(total_samples_with_silence=%d / 16000 * 25)) "
                "allocating=%d (with 100 frame safety margin) | samples_recv=%d",
                current_nd,
                new_frames_estimate_test_formula,
                total_samples_including_silence,
                new_frames_estimate,
                self._stats.total_audio_samples_received,
            )
            try:
                self._call_setup_nd(new_frames_estimate, context="streaming_adjustment_test_formula")
            except Exception as e:
                ditto_logger.exception(
                    "[DittoEngine _update_nd_if_needed WARN] setup_Nd update failed (continuing): %s",
                    e,
                )

    def submit_audio(
        self,
        audio_np: np.ndarray,
        segment_id: str,
        request_id: str,
        is_final: bool = False,
    ):
        t0 = time.time()
        ditto_logger.info(
            "[DittoEngine submit_audio] START | segment_id=%s | request_id=%s | is_final=%s | "
            "samples_in=%d | dtype=%s | shape=%s",
            segment_id,
            request_id,
            is_final,
            len(audio_np),
            audio_np.dtype,
            audio_np.shape,
        )

        if self._closed:
            ditto_logger.warning(
                "[DittoEngine submit_audio] Engine already closed - DROPPING audio | "
                "segment_id=%s | samples=%d",
                segment_id,
                len(audio_np),
            )
            return

        if self.sdk is None:
            ditto_logger.error(
                "[DittoEngine submit_audio] SDK is None (not started?) - DROPPING audio | segment_id=%s",
                segment_id,
            )
            return

        self._stats.current_segment_id = segment_id
        self._stats.current_request_id = request_id
        self._stats.last_audio_time = t0

        self._set_state(EngineState.PROCESSING, operation="appending_audio_to_buffer")

        if len(audio_np) > 0:
            with self._buffer_lock:
                old_len = len(self._audio_buffer)
                self._audio_buffer = np.concatenate([self._audio_buffer, audio_np], 0)
                self._stats.total_audio_samples_received += len(audio_np)
                new_len = len(self._audio_buffer)
                self._stats.audio_buffer_size = new_len

            ditto_logger.info(
                "[DittoEngine submit_audio] Buffer updated | before=%d | after=%d | delta=%+d | "
                "total_samples_received=%d",
                old_len,
                new_len,
                new_len - old_len,
                self._stats.total_audio_samples_received,
            )

            self._ensure_initial_silence()
            self._update_nd_if_needed(len(audio_np))

        # Process complete audio chunks; if is_final=True, force flush remaining audio samples
        self._process_available_chunks(force_flush=is_final)

        if is_final:
            ditto_logger.info(
                "[DittoEngine submit_audio] is_final=True received for segment_id=%s. Triggering SDK.flush_segment()...",
                segment_id,
            )
            t_flush0 = time.time()
            flush_ok = self.sdk.flush_segment(segment_id=segment_id, timeout=30.0)
            t_flush1 = time.time()
            ditto_logger.info(
                "[DittoEngine submit_audio] SDK.flush_segment() finished (success=%s) in %.2f ms for segment_id=%s",
                flush_ok,
                (t_flush1 - t_flush0) * 1000.0,
                segment_id,
            )

            # Reset per-segment audio buffer now that all frames from this segment are flushed
            with self._buffer_lock:
                self._audio_buffer = np.zeros((0,), dtype=np.float32)
                self._chunk_position = 0
                self._initial_silence_prepended = False
                self._stats.audio_buffer_size = 0
                self._stats.chunk_position = 0

            self._stats.total_segments_processed += 1
            self._set_state(EngineState.IDLE, operation=f"segment_complete:{segment_id[:8]}")
            ditto_logger.info(
                "[DittoEngine submit_audio] SEGMENT FLUSH COMPLETE | segment_id=%s | total_segments=%d | Ditto READY for next segment",
                segment_id,
                self._stats.total_segments_processed,
            )
        else:
            if self.state == EngineState.PROCESSING:
                self._set_state(EngineState.IDLE, operation="waiting_for_more_audio")

        total_elapsed = (time.time() - t0) * 1000.0
        ditto_logger.info(
            "[DittoEngine submit_audio] END | segment_id=%s | is_final=%s | "
            "total_elapsed=%.2f ms | buffer_size=%d | chunk_position=%d",
            segment_id,
            is_final,
            total_elapsed,
            len(self._audio_buffer),
            self._chunk_position,
        )

    def _process_available_chunks(self, force_flush: bool = False):
        t_start = time.time()
        ditto_logger.info(
            "[DittoEngine _process_available_chunks] START | force_flush=%s | "
            "buffer_size=%d | chunk_position=%d | split_len=%d | advance=%d",
            force_flush,
            len(self._audio_buffer),
            self._chunk_position,
            SPLIT_LEN,
            ADVANCE,
        )

        self._set_state(EngineState.PROCESSING, operation="processing_ditto_chunks")
        chunks_processed_this_call = 0

        while True:
            available = len(self._audio_buffer) - self._chunk_position
            ditto_logger.debug(
                "[DittoEngine _process_available_chunks] Loop | chunk_position=%d | available=%d | "
                "need=%d (split_len) | force_flush=%s",
                self._chunk_position,
                available,
                SPLIT_LEN,
                force_flush,
            )

            if available >= SPLIT_LEN:
                chunk_start = self._chunk_position
                chunk_end = chunk_start + SPLIT_LEN
                audio_chunk = self._audio_buffer[chunk_start:chunk_end]
                self._run_single_chunk(
                    audio_chunk,
                    chunk_idx=self._stats.total_chunks_processed + 1,
                    call_idx=chunks_processed_this_call,
                    is_final_chunk=False,
                )
                self._chunk_position += ADVANCE
                self._stats.chunk_position = self._chunk_position
                chunks_processed_this_call += 1
            elif force_flush and available > 0:
                chunk_start = self._chunk_position
                audio_chunk = self._audio_buffer[chunk_start:]
                pad_len = SPLIT_LEN - len(audio_chunk)
                ditto_logger.info(
                    "[DittoEngine _process_available_chunks FLUSH] Final chunk flush | "
                    "samples=%d | pad_len=%d | total_padded=%d",
                    len(audio_chunk),
                    pad_len,
                    SPLIT_LEN,
                )
                audio_chunk = np.pad(audio_chunk, (0, pad_len), mode="constant")
                self._run_single_chunk(
                    audio_chunk,
                    chunk_idx=self._stats.total_chunks_processed + 1,
                    call_idx=chunks_processed_this_call,
                    is_final_chunk=True,
                )
                self._chunk_position = len(self._audio_buffer)
                self._stats.chunk_position = self._chunk_position
                chunks_processed_this_call += 1
                break
            else:
                break

        total_latency = (time.time() - t_start) * 1000.0
        self._stats.last_chunk_time = time.time()

        if chunks_processed_this_call > 0:
            ditto_logger.info(
                "[DittoEngine _process_available_chunks] BATCH COMPLETE | chunks=%d | "
                "total_chunks_lifetime=%d | batch_latency=%.2f ms | avg_per_chunk=%.2f ms",
                chunks_processed_this_call,
                self._stats.total_chunks_processed,
                total_latency,
                total_latency / chunks_processed_this_call if chunks_processed_this_call > 0 else 0.0,
            )
        else:
            ditto_logger.info(
                "[DittoEngine _process_available_chunks] NO CHUNKS READY | available=%d | need=%d | force_flush=%s",
                len(self._audio_buffer) - self._chunk_position,
                SPLIT_LEN,
                force_flush,
            )

        ditto_logger.info(
            "[DittoEngine _process_available_chunks] END | total_elapsed=%.2f ms",
            total_latency,
        )

    def _run_single_chunk(
        self,
        audio_chunk: np.ndarray,
        chunk_idx: int,
        call_idx: int,
        is_final_chunk: bool = False,
    ):
        t0 = time.time()
        chunk_samples = len(audio_chunk)
        cur_tid = threading.current_thread().ident
        cur_tname = threading.current_thread().name

        split_len_expected = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80
        advance_expected = CHUNKSIZE[1] * 640

        ditto_logger.info(
            "[DittoEngine RUN_CHUNK #%d START] | (batch_call_idx=%d) | "
            "samples=%d expected_split_len=%d match=%s | chunksize=%s | "
            "advance_this_window=%d expected_advance=%d match=%s | "
            "is_final_chunk=%s | dtype=%s | shape=%s | TID=%s thread=%s",
            chunk_idx,
            call_idx,
            chunk_samples,
            split_len_expected,
            chunk_samples == split_len_expected,
            CHUNKSIZE,
            ADVANCE,
            advance_expected,
            ADVANCE == advance_expected,
            is_final_chunk,
            audio_chunk.dtype,
            audio_chunk.shape,
            cur_tid,
            cur_tname,
        )
        cmin = float(np.min(audio_chunk))
        cmax = float(np.max(audio_chunk))
        cmean = float(np.mean(audio_chunk))
        cstd = float(np.std(audio_chunk))
        ditto_logger.info(
            "[DittoEngine RUN_CHUNK #%d] Audio stats | min=%.6f | max=%.6f | mean=%.6f | std=%.6f | "
            "TID=%s | sdk_id=%s writer_id=%s writer_cls=%s | "
            "SDK.N_d=%s",
            chunk_idx,
            cmin,
            cmax,
            cmean,
            cstd,
            cur_tid,
            id(self.sdk) if self.sdk is not None else "N/A",
            id(getattr(self.sdk, "writer", None)) if self.sdk is not None else "N/A",
            type(getattr(self.sdk, "writer", None)).__name__ if self.sdk is not None and getattr(self.sdk, "writer", None) is not None else "None",
            getattr(self.sdk, "N_d", None),
        )

        if chunk_idx == 1:
            silence_prefix = audio_chunk[:INITIAL_SILENCE_SAMPLES]
            prefix_zeros = int(np.sum(silence_prefix == 0.0))
            ditto_logger.info(
                "[DittoEngine RUN_CHUNK #%d] FIRST CHUNK SILENCE VALIDATION [MATCH test_ditto_live.py line 140] | "
                "prefix[:%d] zero_count=%d/%d expected=%d match=%s | "
                "prefix_min=%.6f prefix_max=%.6f prefix_mean=%.6f | "
                "Chunk absolute window start index (incl silence)=%d | TID=%s",
                chunk_idx,
                INITIAL_SILENCE_SAMPLES,
                prefix_zeros,
                INITIAL_SILENCE_SAMPLES,
                INITIAL_SILENCE_SAMPLES,
                prefix_zeros == INITIAL_SILENCE_SAMPLES,
                float(np.min(silence_prefix)),
                float(np.max(silence_prefix)),
                float(np.mean(silence_prefix)),
                self._chunk_position,
                cur_tid,
            )
            npy_path = str(REFERENCE_CHUNK_NPY)
            try:
                np.save(npy_path, audio_chunk.astype(np.float32))
                ditto_logger.info(
                    "[DittoEngine RUN_CHUNK #%d] FIRST CHUNK SAVED TO NPY FOR A/B: %s | "
                    "shape=%s dtype=%s bytes=%d | min=%.6f max=%.6f mean=%.6f std=%.6f",
                    chunk_idx,
                    npy_path,
                    audio_chunk.shape,
                    audio_chunk.dtype,
                    audio_chunk.astype(np.float32).nbytes,
                    cmin,
                    cmax,
                    cmean,
                    cstd,
                )
                meta_path = npy_path + ".txt"
                with open(meta_path, "w", encoding="utf-8") as meta_fp:
                    meta_fp.write(f"DITTO FIRST CHUNK REFERENCE - for A/B test_ditto_live.py comparison\n")
                    meta_fp.write(f"chunk_idx=1\n")
                    meta_fp.write(f"saved_at_epoch={time.time():.4f}\n")
                    meta_fp.write(f"chunk_position_absolute_in_buffer={self._chunk_position}\n")
                    meta_fp.write(f"INITIAL_SILENCE_SAMPLES={INITIAL_SILENCE_SAMPLES}\n")
                    meta_fp.write(f"CHUNKSIZE={CHUNKSIZE}\n")
                    meta_fp.write(f"SPLIT_LEN={SPLIT_LEN}\n")
                    meta_fp.write(f"ADVANCE={ADVANCE}\n")
                    meta_fp.write(f"chunk.shape={audio_chunk.shape}\n")
                    meta_fp.write(f"chunk.dtype={audio_chunk.dtype}\n")
                    meta_fp.write(f"chunk_samples={chunk_samples}\n")
                    meta_fp.write(f"chunk_min={cmin:.10f}\n")
                    meta_fp.write(f"chunk_max={cmax:.10f}\n")
                    meta_fp.write(f"chunk_mean={cmean:.10f}\n")
                    meta_fp.write(f"chunk_std={cstd:.10f}\n")
                    meta_fp.write(f"chunk_first_1920_prefix_zeros={prefix_zeros}/{INITIAL_SILENCE_SAMPLES}\n")
                    meta_fp.write(f"TID_run_chunk={cur_tid} ({cur_tname})\n")
                    meta_fp.write(f"SDK.N_d_at_run={getattr(self.sdk, 'N_d', None)}\n")
                    meta_fp.write(f"writer_id={id(getattr(self.sdk, 'writer', None)) if self.sdk is not None else 'N/A'}\n")
                    meta_fp.write(f"writer_cls={type(getattr(self.sdk, 'writer', None)).__name__ if self.sdk is not None and getattr(self.sdk, 'writer', None) is not None else 'None'}\n")
                ditto_logger.info(
                    "[DittoEngine RUN_CHUNK #%d] Metadata file written: %s",
                    chunk_idx,
                    meta_path,
                )
            except Exception as save_exc:
                ditto_logger.exception(
                    "[DittoEngine RUN_CHUNK #%d WARN] Failed saving reference chunk .npy: %s",
                    chunk_idx,
                    save_exc,
                )

        try:
            self._set_state(EngineState.PROCESSING, operation=f"sdk_run_chunk_{chunk_idx}")
            self._raise_if_worker_failed(context=f"pre_run_chunk_{chunk_idx}")

            t_run0 = time.time()
            ditto_logger.info(
                "[DittoEngine RUN_CHUNK #%d] >>> SDK.run_chunk(audio_chunk, chunksize=%s) | "
                "writer_id=%s cls=%s | TID=%s",
                chunk_idx,
                CHUNKSIZE,
                id(getattr(self.sdk, "writer", None)) if self.sdk is not None else "N/A",
                type(getattr(self.sdk, "writer", None)).__name__ if self.sdk is not None and getattr(self.sdk, "writer", None) is not None else "None",
                cur_tid,
            )
            self.sdk.run_chunk(audio_chunk, CHUNKSIZE)
            t_run1 = time.time()
            run_latency_ms = (t_run1 - t_run0) * 1000.0

            ditto_logger.info(
                "[DittoEngine RUN_CHUNK #%d] SDK.run_chunk() returned in %.2f ms. "
                "Checking worker_exception + diagnosing pipeline progress | TID=%s",
                chunk_idx,
                run_latency_ms,
                cur_tid,
            )
            self._raise_if_worker_failed(context=f"post_run_chunk_{chunk_idx}")
            self._diagnose_pipeline_state(tag=f"AFTER_RUN_CHUNK_#{chunk_idx}")

            frames_before = self._stats.total_frames_generated
            post_check_time = time.time()
            sleep_s = 0.020
            time.sleep(sleep_s)
            frames_after = self._stats.total_frames_generated
            self._diagnose_pipeline_state(tag=f"AFTER_RUN_CHUNK_#{chunk_idx}_+{int(sleep_s*1000)}ms")
            writer_q_after = getattr(self.sdk, "writer_queue", None)
            writer_q_size = writer_q_after.qsize() if writer_q_after is not None else "N/A"
            ditto_logger.info(
                "[DittoEngine RUN_CHUNK #%d] Frame-arrival quick check | "
                "sleep=%.0fms | frames_before=%d | frames_after=%d | delta=%+d | "
                "writer_queue.qsize()=%s | check_latency=%.1f ms | TID=%s",
                chunk_idx,
                sleep_s * 1000.0,
                frames_before,
                frames_after,
                frames_after - frames_before,
                writer_q_size,
                (time.time() - post_check_time) * 1000.0,
                cur_tid,
            )

            self._stats.total_chunks_processed += 1
            total_elapsed_ms = (time.time() - t0) * 1000.0

            ditto_logger.info(
                "[DittoEngine RUN_CHUNK #%d END] | run_chunk_latency=%.2f ms | "
                "total_chunk_processing=%.2f ms | total_chunks=%d | frames_gen_so_far=%d | TID=%s",
                chunk_idx,
                run_latency_ms,
                total_elapsed_ms,
                self._stats.total_chunks_processed,
                self._stats.total_frames_generated,
                cur_tid,
            )
        except Exception as e:
            ditto_logger.exception(
                "[DittoEngine RUN_CHUNK #%d ERROR TID=%s] Failed: %s\n%s",
                chunk_idx,
                cur_tid,
                e,
                traceback.format_exc(),
            )
            raise

    def stop(self):
        ditto_logger.info(
            "[DittoEngine stop] ================ STOP CALLED ================"
        )
        t0 = time.time()
        if self._closed:
            ditto_logger.warning("[DittoEngine stop] Already closed - ignoring duplicate call")
            return

        self._set_state(EngineState.STOPPING, operation="stopping_engine")
        self._stop_event.set()

        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            ditto_logger.info(
                "[DittoEngine stop] Joining watchdog thread (TID=%d, timeout=10s)...",
                self._watchdog_thread.ident,
            )
            self._watchdog_thread.join(timeout=10.0)
            if self._watchdog_thread.is_alive():
                ditto_logger.warning(
                    "[DittoEngine stop] Watchdog thread did not terminate within 10s - abandoning"
                )
            else:
                ditto_logger.info("[DittoEngine stop] Watchdog thread joined successfully")

        if self.sdk is not None:
            ditto_logger.info("[DittoEngine stop] Calling SDK.close()...")
            try:
                t_sdk0 = time.time()
                self.sdk.close()
                t_sdk1 = time.time()
                ditto_logger.info(
                    "[DittoEngine stop] SDK.close() completed in %.2f ms",
                    (t_sdk1 - t_sdk0) * 1000.0,
                )
            except Exception as e:
                ditto_logger.exception(
                    "[DittoEngine stop WARN] SDK.close() raised (continuing cleanup): %s",
                    e,
                )

        ditto_logger.info("[DittoEngine stop] Shutting down OpenCV windows...")
        try:
            cv2.destroyAllWindows()
            ditto_logger.info("[DittoEngine stop] OpenCV windows destroyed successfully")
        except Exception as e:
            ditto_logger.exception(
                "[DittoEngine stop WARN] OpenCV destroyAllWindows raised: %s",
                e,
            )

        total_elapsed = (time.time() - t0) * 1000.0
        total_uptime = (
            (time.time() - self._stats.engine_start_time)
            if self._stats.engine_start_time is not None
            else 0.0
        )
        self._closed = True
        self._set_state(EngineState.STOPPED, operation="shutdown_complete")

        ditto_logger.info(
            "[DittoEngine stop] FINAL STATS | "
            "samples_recv=%d | chunks_proc=%d | frames_gen=%d | segments_proc=%d | "
            "stop_elapsed=%.2f ms | total_uptime=%.2fs",
            self._stats.total_audio_samples_received,
            self._stats.total_chunks_processed,
            self._stats.total_frames_generated,
            self._stats.total_segments_processed,
            total_elapsed,
            total_uptime,
        )
        ditto_logger.info(
            "[DittoEngine stop] ================ STOP COMPLETE ================"
        )


class DittoAdapter:
    def __init__(
        self,
        checkpoints_dir: Optional[Path] = None,
        cfg_pkl: Optional[Path] = None,
        source_image: Optional[Path] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled

        self._engine: Optional[DittoEngine] = (
            DittoEngine(
                checkpoints_dir=checkpoints_dir,
                cfg_pkl=cfg_pkl,
                source_image=source_image,
            )
            if enabled
            else None
        )

        self._queue: "queue.Queue[Optional[SynthesizedAudioInfo]]" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stopped = False
        self._stop_lock = threading.Lock()
        self._start_time = None
        self._initialized = False
        self._init_lock = threading.Lock()

        ditto_logger.info("=" * 80)
        ditto_logger.info("[DittoAdapter] Initializing DittoAdapter")
        ditto_logger.info("[DittoAdapter] enabled=%s", self.enabled)
        if self._engine is not None:
            ditto_logger.info(
                "[DittoAdapter] DittoEngine instance created (start() will initialize SDK + OpenCV)"
            )
        ditto_logger.info("=" * 80)

    def start(self):
        ditto_logger.info("[DittoAdapter START] ================ START CALLED ================")
        caller = inspect.stack()[1]
        caller_info = f"{caller.filename}:{caller.lineno} in {caller.function}"
        ditto_logger.info("[DittoAdapter START] called by %s", caller_info)

        if not self.enabled:
            ditto_logger.warning("[DittoAdapter START] Disabled - not starting worker/engine")
            return

        if self._worker_thread is not None and self._worker_thread.is_alive():
            ditto_logger.warning(
                "[DittoAdapter START] Worker already running (TID=%s) - skipping start",
                self._worker_thread.ident,
            )
            return

        with self._init_lock:
            if not self._initialized and self._engine is not None:
                ditto_logger.info(
                    "[DittoAdapter START] Initializing DittoEngine (StreamSDK + OpenCV + Watchdog)..."
                )
                t_init0 = time.time()
                try:
                    self._engine.start()
                    t_init1 = time.time()
                    self._initialized = True
                    ditto_logger.info(
                        "[DittoAdapter START] DittoEngine initialized in %.2f ms",
                        (t_init1 - t_init0) * 1000.0,
                    )
                except Exception as e:
                    ditto_logger.exception(
                        "[DittoAdapter START FATAL] DittoEngine.start() failed: %s\n%s",
                        e,
                        traceback.format_exc(),
                    )
                    raise

        self._start_time = time.time()
        self._stop_event.clear()
        with self._stop_lock:
            self._stopped = False

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="DittoAdapterWorker",
            daemon=True,
        )
        self._worker_thread.start()

        ditto_logger.info(
            "[DittoAdapter START] SUCCESS | Worker thread started (TID=%d) | called by %s",
            self._worker_thread.ident,
            caller_info,
        )
        ditto_logger.info("[DittoAdapter START] ================ START COMPLETE ================")

    def stop(self, reason: str = "unspecified"):
        caller = inspect.stack()[1]
        caller_info = f"{caller.filename}:{caller.lineno} in {caller.function}"
        tb_str = "".join(traceback.format_stack()[:-1])

        with self._stop_lock:
            if self._stopped:
                ditto_logger.warning(
                    "[DittoAdapter STOP] IGNORED duplicate stop() call | reason=%s | caller=%s | already stopped",
                    reason,
                    caller_info,
                )
                return
            self._stopped = True

        uptime = (time.time() - self._start_time) if self._start_time else 0.0
        queue_size = self._queue.qsize()

        ditto_logger.warning(
            "[DittoAdapter STOP] ===== STOP EXECUTING ===== | reason=%s | caller=%s | uptime=%.3fs | queue_size_at_stop=%d",
            reason,
            caller_info,
            uptime,
            queue_size,
        )
        ditto_logger.warning(
            "[DittoAdapter STOP] Full call stack at stop() invocation:\n%s",
            tb_str,
        )

        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
            ditto_logger.info(
                "[DittoAdapter STOP] Stop sentinel enqueued (wakes worker if idle)"
            )
        except queue.Full:
            ditto_logger.warning(
                "[DittoAdapter STOP] Queue full, could not enqueue stop sentinel (relying on _stop_event)"
            )

        if self._worker_thread is not None:
            ditto_logger.info(
                "[DittoAdapter STOP] Joining worker thread (TID=%s, timeout=30s)...",
                self._worker_thread.ident,
            )
            t_join0 = time.time()
            self._worker_thread.join(timeout=30.0)
            join_duration = time.time() - t_join0
            if self._worker_thread.is_alive():
                ditto_logger.warning(
                    "[DittoAdapter STOP] Worker thread did not terminate within 30s (join took %.2fs) - abandoning",
                    join_duration,
                )
            else:
                ditto_logger.info(
                    "[DittoAdapter STOP] Worker thread joined successfully in %.2fs",
                    join_duration,
                )

        if self._engine is not None:
            ditto_logger.info("[DittoAdapter STOP] Stopping DittoEngine (SDK.close + OpenCV)...")
            try:
                self._engine.stop()
            except Exception as e:
                ditto_logger.exception(
                    "[DittoAdapter STOP ERROR] DittoEngine.stop() raised: %s",
                    e,
                )

        total_uptime = (time.time() - self._start_time) if self._start_time else 0.0
        ditto_logger.warning(
            "[DittoAdapter STOP] ===== STOP COMPLETE ===== | reason=%s | caller=%s | total_uptime=%.3fs",
            reason,
            caller_info,
            total_uptime,
        )

    def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """
        Waits until all queued audio chunks and segment flushes have been completely
        processed, rendered, and displayed by the OpenCV writer.
        Returns True if idle, False if timed out.
        """
        t0 = time.time()
        while not self._stop_event.is_set():
            if self._queue.empty() and self._engine is not None and self._engine.state == EngineState.IDLE:
                return True
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.02)
        return True


    def submit_synthesized_audio(
        self,
        frame_data: Any,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
        duration: float,
        request_id: str,
        segment_id: str,
        is_final: bool,
    ) -> bool:
        if not self.enabled:
            return False

        info = SynthesizedAudioInfo(
            frame_data=frame_data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
            duration=duration,
            request_id=str(request_id),
            segment_id=str(segment_id),
            is_final=bool(is_final),
        )

        ditto_logger.info(
            "[DittoAdapter QUEUE] Enqueuing SynthesizedAudio | segment_id=%s | request_id=%s | is_final=%s | "
            "sr=%d | ch=%d | samples_per_ch=%d | duration=%.3fs | data_type=%s",
            info.segment_id,
            info.request_id,
            info.is_final,
            info.sample_rate,
            info.num_channels,
            info.samples_per_channel,
            info.duration,
            type(info.frame_data).__name__,
        )

        queue_size_before = self._queue.qsize()

        if self._stop_event.is_set():
            ditto_logger.warning(
                "[DittoAdapter QUEUE ENQUEUE] STOP already requested - DROPPING event | "
                "segment_id=%s | request_id=%s | queue_size_before=%d",
                info.segment_id,
                info.request_id,
                queue_size_before,
            )
            return False

        try:
            self._queue.put_nowait(info)
            queue_size_after = self._queue.qsize()
            ditto_logger.info(
                "[DittoAdapter QUEUE ENQUEUE] SUCCESS | segment_id=%s | request_id=%s | is_final=%s | "
                "queue_size_before=%d | queue_size_after=%d | delta=%+d",
                info.segment_id,
                info.request_id,
                info.is_final,
                queue_size_before,
                queue_size_after,
                queue_size_after - queue_size_before,
            )
            return True
        except queue.Full:
            ditto_logger.error(
                "[DittoAdapter QUEUE ENQUEUE ERROR] Queue FULL (size=%d) - DROPPING event | "
                "segment_id=%s | request_id=%s",
                queue_size_before,
                info.segment_id,
                info.request_id,
            )
            return False
        except Exception as e:
            ditto_logger.exception(
                "[DittoAdapter QUEUE ENQUEUE FATAL] %s | segment_id=%s | request_id=%s\n%s",
                e,
                info.segment_id,
                info.request_id,
                traceback.format_exc(),
            )
            return False

    def _worker_loop(self):
        ditto_logger.info(
            "[DittoAdapter WORKER START] Worker loop starting (TID=%s, name=%s)",
            threading.current_thread().ident,
            threading.current_thread().name,
        )
        processed_count = 0
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                queue_size_after_dequeue = self._queue.qsize()

                if item is None:
                    ditto_logger.info(
                        "[DittoAdapter WORKER DEQUEUE] Received stop sentinel | queue_size_after=%d | processed_so_far=%d",
                        queue_size_after_dequeue,
                        processed_count,
                    )
                    break

                processed_count += 1
                ditto_logger.info(
                    "[DittoAdapter WORKER DEQUEUE] SUCCESS | item #%d | segment_id=%s | request_id=%s | is_final=%s | "
                    "queue_size_after_dequeue=%d",
                    processed_count,
                    item.segment_id,
                    item.request_id,
                    item.is_final,
                    queue_size_after_dequeue,
                )

                t_proc_start = time.time()
                ditto_logger.info(
                    "[DittoAdapter WORKER PROCESS START] item #%d | segment_id=%s | request_id=%s",
                    processed_count,
                    item.segment_id,
                    item.request_id,
                )

                try:
                    self._handle_audio_info(item)
                except Exception as e:
                    ditto_logger.exception(
                        "[DittoAdapter WORKER PROCESS ERROR] item #%d (segment_id=%s) failed:\n%s",
                        processed_count,
                        item.segment_id,
                        traceback.format_exc(),
                    )
                finally:
                    proc_duration_ms = (time.time() - t_proc_start) * 1000.0
                    ditto_logger.info(
                        "[DittoAdapter WORKER PROCESS END] item #%d | segment_id=%s | duration=%.2f ms | total_processed=%d",
                        processed_count,
                        item.segment_id,
                        proc_duration_ms,
                        processed_count,
                    )

            ditto_logger.info(
                "[DittoAdapter WORKER END] Loop exiting normally | total_items_processed=%d | stop_event_set=%s",
                processed_count,
                self._stop_event.is_set(),
            )
        except Exception as e:
            ditto_logger.exception(
                "[DittoAdapter WORKER FATAL] Worker thread crashed (processed=%d):\n%s",
                processed_count,
                traceback.format_exc(),
            )

    def _handle_audio_info(self, info: SynthesizedAudioInfo):
        t0 = time.time()
        ditto_logger.info(
            "[DittoAdapter HANDLE START] Processing event | segment_id=%s | request_id=%s | is_final=%s",
            info.segment_id,
            info.request_id,
            info.is_final,
        )

        if self._engine is None:
            ditto_logger.error(
                "[DittoAdapter HANDLE] DittoEngine is None - skipping event | segment_id=%s",
                info.segment_id,
            )
            return

        audio_np = self._convert_frame_to_numpy(info)
        if audio_np is None:
            ditto_logger.error(
                "[DittoAdapter HANDLE] Audio conversion FAILED, skipping event | segment_id=%s",
                info.segment_id,
            )
            return

        self._engine.submit_audio(
            audio_np=audio_np,
            segment_id=info.segment_id,
            request_id=info.request_id,
            is_final=info.is_final,
        )

        ditto_logger.info(
            "[DittoAdapter HANDLE END] Event processed in %.2f ms | segment_id=%s | is_final=%s",
            (time.time() - t0) * 1000.0,
            info.segment_id,
            info.is_final,
        )

    def _convert_frame_to_numpy(self, info: SynthesizedAudioInfo) -> Optional[np.ndarray]:
        t0 = time.time()
        ditto_logger.info(
            "[DittoAdapter CONVERT START] Frame data analysis | type=%s | sr=%d | ch=%d | samples_per_ch=%d | duration=%.3fs",
            type(info.frame_data).__name__,
            info.sample_rate,
            info.num_channels,
            info.samples_per_channel,
            info.duration,
        )

        try:
            data = info.frame_data
            if isinstance(data, memoryview):
                ditto_logger.debug(
                    "[DittoAdapter CONVERT] data is memoryview -> bytes -> numpy int16"
                )
                raw_bytes = bytes(data)
                int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
            elif isinstance(data, (bytes, bytearray)):
                ditto_logger.debug("[DittoAdapter CONVERT] data is bytes/bytearray -> numpy int16")
                int16_arr = np.frombuffer(data, dtype=np.int16)
            elif isinstance(data, np.ndarray):
                ditto_logger.debug(
                    "[DittoAdapter CONVERT] data is numpy ndarray | dtype=%s | shape=%s",
                    data.dtype,
                    data.shape,
                )
                if data.dtype == np.int16:
                    int16_arr = data
                elif data.dtype in (np.float32, np.float64):
                    float_arr = data.astype(np.float32)
                    if float_arr.ndim > 1:
                        float_arr = (
                            float_arr[:, 0]
                            if info.num_channels >= 2
                            else float_arr.flatten()
                        )
                    float_arr = float_arr.astype(np.float32)
                    ditto_logger.info(
                        "[DittoAdapter CONVERT END] Direct float path | dtype=float32 | shape=%s | samples=%d | latency=%.2f ms",
                        float_arr.shape,
                        len(float_arr),
                        (time.time() - t0) * 1000.0,
                    )
                    return float_arr
                else:
                    ditto_logger.error(
                        "[DittoAdapter CONVERT] Unsupported numpy dtype: %s", data.dtype
                    )
                    return None
            else:
                ditto_logger.error(
                    "[DittoAdapter CONVERT] Unknown frame.data type: %s",
                    type(data).__name__,
                )
                return None

            ditto_logger.debug(
                "[DittoAdapter CONVERT] int16 array shape=%s | len=%d",
                int16_arr.shape,
                len(int16_arr),
            )

            expected_samples = info.samples_per_channel * info.num_channels
            if len(int16_arr) != expected_samples:
                ditto_logger.warning(
                    "[DittoAdapter CONVERT] Sample count mismatch: expected=%d (samples_per_ch=%d * ch=%d) | actual=%d | using actual",
                    expected_samples,
                    info.samples_per_channel,
                    info.num_channels,
                    len(int16_arr),
                )

            if info.num_channels > 1:
                ditto_logger.debug(
                    "[DittoAdapter CONVERT] Multi-channel (%d) - downmixing to mono by taking channel 0",
                    info.num_channels,
                )
                int16_arr = int16_arr.reshape(-1, info.num_channels)[:, 0]

            float32_arr = int16_arr.astype(np.float32) / 32768.0

            if info.sample_rate != 16000:
                ditto_logger.warning(
                    "[DittoAdapter CONVERT] Sample rate is %d Hz, expected 16000 Hz. No resampling implemented - passing through as-is.",
                    info.sample_rate,
                )

            ditto_logger.info(
                "[DittoAdapter CONVERT END] SUCCESS | int16->float32 | input_dtype=int16 | output_dtype=%s | "
                "output_shape=%s | output_samples=%d | min=%.6f | max=%.6f | latency=%.2f ms",
                float32_arr.dtype,
                float32_arr.shape,
                len(float32_arr),
                float(np.min(float32_arr)) if len(float32_arr) > 0 else 0.0,
                float(np.max(float32_arr)) if len(float32_arr) > 0 else 0.0,
                (time.time() - t0) * 1000.0,
            )

            return float32_arr.astype(np.float32)

        except Exception as e:
            ditto_logger.exception(
                "[DittoAdapter CONVERT FATAL ERROR] %s\n%s",
                e,
                traceback.format_exc(),
            )
            return None


class DittoSynthesizedAudioConsumer:
    def __init__(self, adapter: DittoAdapter):
        self.adapter = adapter
        ditto_logger.info(
            "[DittoSynthesizedAudioConsumer] Created with adapter enabled=%s",
            adapter.enabled,
        )

    def on_synthesized_audio(self, ev) -> None:
        try:
            frame = ev.frame
            ditto_logger.info(
                "[DittoSynthesizedAudioConsumer] SynthesizedAudio event received | "
                "segment_id=%s | request_id=%s | is_final=%s | sr=%d | ch=%d | samples_per_ch=%d | duration=%.3fs | data_type=%s",
                getattr(ev, "segment_id", "unknown"),
                getattr(ev, "request_id", "unknown"),
                getattr(ev, "is_final", False),
                getattr(frame, "sample_rate", "?"),
                getattr(frame, "num_channels", "?"),
                getattr(frame, "samples_per_channel", "?"),
                getattr(frame, "duration", "?"),
                type(getattr(frame, "data", None)).__name__,
            )
            self.adapter.submit_synthesized_audio(
                frame_data=frame.data,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=frame.samples_per_channel,
                duration=frame.duration,
                request_id=getattr(ev, "request_id", "unknown"),
                segment_id=getattr(ev, "segment_id", "unknown"),
                is_final=getattr(ev, "is_final", False),
            )
        except Exception as e:
            ditto_logger.exception(
                "[DittoSynthesizedAudioConsumer ERROR] on_synthesized_audio callback failed: %s\n%s",
                e,
                traceback.format_exc(),
            )
