import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
import librosa

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_lifecycle")

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "avatar" / "ditto"))

from avatar.ditto.integration.ditto_adapter import DittoAdapter, DittoSynthesizedAudioConsumer, SynthesizedAudioInfo

AUDIO_PATH = BACKEND_DIR / "avatar" / "ditto" / "example" / "audio.wav"

def run_tests():
    logger.info("============================================================")
    logger.info("STARTING DITTO LIFECYCLE ACCEPTANCE TESTS (4+ CONSECUTIVE TURNS)")
    logger.info("============================================================")

    logger.info("Loading test audio from: %s", AUDIO_PATH)
    audio_full, sr = librosa.load(str(AUDIO_PATH), sr=16000)
    logger.info("Loaded full audio: %d samples (%.2fs) at %d Hz", len(audio_full), len(audio_full)/sr, sr)

    metrics_table = []

    # 1. Initialize and Start DittoAdapter
    logger.info("\n--- STEP 1: INITIALIZE & WARM-UP DITTO ---")
    t0 = time.time()
    adapter = DittoAdapter(enabled=True)
    adapter.start()
    t_warmed = time.time() - t0
    logger.info("Ditto initialized and warmed up in %.2fs. Engine state: %s", t_warmed, adapter._engine.state.value)

    assert adapter._engine.state.value in ("READY", "IDLE"), f"Expected READY/IDLE, got {adapter._engine.state}"
    initial_frames = adapter._engine._stats.total_frames_generated
    logger.info("Initial generated frames (warmup only): %d (expected 0 visible frames)", initial_frames)
    assert initial_frames == 0, f"Expected 0 frames emitted during warmup, got {initial_frames}"

    def push_sentence_with_metrics(sentence_audio: np.ndarray, segment_id: str, test_name: str):
        logger.info("\n--- PUSHING %s: segment_id=%s (samples=%d, duration=%.2fs) ---", test_name, segment_id, len(sentence_audio), len(sentence_audio)/16000)
        chunk_len = 1600 # 100ms per synthesized audio packet
        num_chunks = int(np.ceil(len(sentence_audio) / chunk_len))
        
        t_start = time.time()
        first_frame_latency = None
        frames_before = adapter._engine._stats.total_frames_generated
        
        for i in range(num_chunks):
            start = i * chunk_len
            end = min(start + chunk_len, len(sentence_audio))
            chunk = sentence_audio[start:end]
            is_final = (i == num_chunks - 1)
            
            adapter.submit_synthesized_audio(
                frame_data=chunk,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(chunk),
                duration=len(chunk) / 16000.0,
                request_id=f"req_{segment_id}",
                segment_id=segment_id,
                is_final=is_final,
            )

            if first_frame_latency is None:
                cur_frames = adapter._engine._stats.total_frames_generated
                if cur_frames > frames_before:
                    first_frame_latency = (time.time() - t_start) * 1000.0

            time.sleep(0.01)

        # Wait for first frame if not yet generated during chunk push
        t_wait_start = time.time()
        while first_frame_latency is None and (time.time() - t_wait_start) < 20.0:
            cur_frames = adapter._engine._stats.total_frames_generated
            if cur_frames > frames_before:
                first_frame_latency = (time.time() - t_start) * 1000.0
                break
            time.sleep(0.05)

        idle_ok = adapter.wait_for_idle(timeout=60.0)
        assert idle_ok, f"FAILED: Timed out waiting for {test_name} ({segment_id}) to finish rendering!"
        
        frames_after = adapter._engine._stats.total_frames_generated
        segment_frames = frames_after - frames_before
        total_time = (time.time() - t_start) * 1000.0
        
        if first_frame_latency is None:
            first_frame_latency = total_time

        logger.info(
            "%s RESULT: frames=%d | first_frame_latency=%.2f ms | total_elapsed=%.2f ms",
            test_name, segment_frames, first_frame_latency, total_time
        )
        assert segment_frames > 0, f"{test_name} FAILED: Produced 0 frames!"

        metrics_table.append({
            "Turn": test_name,
            "Segment ID": segment_id,
            "Duration (s)": f"{len(sentence_audio)/16000:.2f}s",
            "Generated Frames": segment_frames,
            "First Frame Latency": f"{first_frame_latency:.1f} ms",
            "Status": "PASS"
        })
        return segment_frames

    # TEST 1: First sentence (approx 1.5s)
    push_sentence_with_metrics(audio_full[:24000], "segment_1_first_sentence", "Turn 1 (First sentence)")

    # TEST 2: Second sentence (approx 1.5s)
    push_sentence_with_metrics(audio_full[24000:48000], "segment_2_second_sentence", "Turn 2 (Second sentence)")

    # TEST 3: Short sentence (approx 1.0s)
    push_sentence_with_metrics(audio_full[48000:64000], "segment_3_short_hello", "Turn 3 (Short sentence)")

    # TEST 4: Fourth sentence (approx 1.5s)
    push_sentence_with_metrics(audio_full[64000:88000], "segment_4_consecutive", "Turn 4 (Fourth sentence)")

    # TEST 5: Clean Shutdown
    logger.info("\n============================================================")
    logger.info("CLEAN SHUTDOWN (NO DELAYED POPUP)")
    logger.info("============================================================")
    frames_before_stop = adapter._engine._stats.total_frames_generated
    adapter.stop(reason="acceptance test completed")
    frames_after_stop = adapter._engine._stats.total_frames_generated
    shutdown_delta = frames_after_stop - frames_before_stop
    logger.info("Frames generated during shutdown: %d", shutdown_delta)
    assert shutdown_delta <= 5, f"Unexpected burst of {shutdown_delta} frames during shutdown!"

    logger.info("\n============================================================")
    logger.info("MULTI-TURN ACCEPTANCE TEST SUMMARY")
    logger.info("============================================================")
    logger.info(f"{'Turn':<25} | {'Duration':<10} | {'Frames':<8} | {'First Frame Latency':<20} | {'Status':<6}")
    logger.info("-" * 80)
    for row in metrics_table:
        logger.info(f"{row['Turn']:<25} | {row['Duration (s)']:<10} | {row['Generated Frames']:<8} | {row['First Frame Latency']:<20} | {row['Status']:<6}")
    logger.info("============================================================")
    logger.info("ALL 4+ CONSECUTIVE TURNS PASSED SUCCESSFULLY!")
    logger.info("============================================================")

if __name__ == "__main__":
    run_tests()
