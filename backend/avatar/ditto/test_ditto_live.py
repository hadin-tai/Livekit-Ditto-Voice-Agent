import os
import sys
import math
import time
import cv2
import numpy as np
import librosa

# Add the ditto directory to the path for imports
# TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# AVATAR_DIR = os.path.dirname(TEST_DIR)
# DITTO_DIR = os.path.join(AVATAR_DIR, "ditto")
# sys.path.insert(0, DITTO_DIR)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DITTO_DIR = TEST_DIR
sys.path.insert(0, DITTO_DIR)

# Monkey patch VideoWriterByImageIO with our custom DisplayWriter before importing StreamSDK
class DisplayWriter:
    """
    Custom writer that displays frames with OpenCV instead of writing to an MP4 file.
    Reuses the same interface as VideoWriterByImageIO.
    """
    def __init__(self, video_path, fps=25, **kwargs):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_print_time = time.time()
        self.fps = 0.0
        self.frame_info_printed = False

    def __call__(self, img, fmt="rgb"):
        """
        Receives a frame, converts it to BGR (for OpenCV), and displays it.
        """
        self.frame_count += 1
        
        # Convert RGB to BGR for OpenCV
        if fmt == "rgb":
            frame_bgr = img[..., ::-1]
        else:
            frame_bgr = img
        
        # Print frame information once
        if not self.frame_info_printed:
            print(f"Frame Shape: {frame_bgr.shape}")
            print(f"Frame Dtype: {frame_bgr.dtype}")
            self.frame_info_printed = True
        
        # Display the frame
        cv2.imshow("Ditto Avatar Live", frame_bgr)
        
        # Calculate and print FPS every second
        current_time = time.time()
        if current_time - self.last_fps_print_time >= 1.0:
            elapsed = current_time - self.start_time
            self.fps = self.frame_count / elapsed
            print(f"FPS: {self.fps:.1f}")
            self.last_fps_print_time = current_time
        
        # Check for exit key (q) - we need a small wait time for OpenCV window to update
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    def close(self):
        """
        Cleanup: close all OpenCV windows and print total execution time.
        """
        total_time = time.time() - self.start_time
        print(f"\nTotal Execution Time: {total_time:.2f} seconds")
        print(f"Total Frames: {self.frame_count}")
        print(f"Average FPS: {self.frame_count / total_time:.1f}")
        cv2.destroyAllWindows()

# Perform monkey patching
import stream_pipeline_online
stream_pipeline_online.VideoWriterByImageIO = DisplayWriter
from stream_pipeline_online import StreamSDK


def main():
    try:
        # Configuration (absolute paths for reliability)
        DATA_ROOT = os.path.join(
            DITTO_DIR,
            "checkpoints",
            "ditto_pytorch"
        )

        CFG_PKL = os.path.join(
            DITTO_DIR,
            "checkpoints",
            "ditto_cfg",
            "v0.4_hubert_cfg_pytorch.pkl"
        )

        SOURCE_PATH = os.path.join(DITTO_DIR, "example", "image.png")
        AUDIO_PATH = os.path.join(DITTO_DIR, "example", "audio.wav")

        # -----------------------------
        # Step 1: Initialize StreamSDK
        # -----------------------------
        print("Initializing SDK...")
        SDK = StreamSDK(CFG_PKL, DATA_ROOT)

        # -----------------------------
        # Step 2: Load and prepare audio
        # -----------------------------
        print("Loading Audio...")
        audio, sr = librosa.core.load(AUDIO_PATH, sr=16000)
        num_f = math.ceil(len(audio) / 16000 * 25)  # Calculate number of frames at 25 FPS

        # -----------------------------
        # Step 3: Setup SDK
        # -----------------------------
        print("Setting up Avatar...")
        SDK.setup(
            SOURCE_PATH,
            OUTPUT_PATH,
            online_mode=True,  # Enable online mode
            emo=4  # Default emotion
        )

        # -----------------------------
        # Step 4: Setup N_d
        # -----------------------------
        SDK.setup_Nd(
            N_d=num_f,
            fade_in=-1,
            fade_out=-1,
            ctrl_info={}
        )

        # -----------------------------
        # Step 5: Process audio chunks
        # -----------------------------
        print("Processing Chunks...")
        chunksize = (3, 5, 2)
        # Prepend silence as done in the repository
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480

        # Process each chunk
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            # Pad final chunk if needed
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)

        # -----------------------------
        # Step 6: Cleanup
        # -----------------------------
        print("Cleaning up...")
        SDK.close()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, cleaning up...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure OpenCV windows are closed
        cv2.destroyAllWindows()


if __name__ == "__main__":
    OUTPUT_PATH = "dummy_output.mp4"
    main()
