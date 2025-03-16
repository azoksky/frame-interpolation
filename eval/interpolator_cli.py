"""
This script performs recursive frame interpolation on all directories matching the given pattern.
It is designed to work on both single‐GPU and multi‐GPU systems with a unified graphical progress bar.
It no longer creates videos automatically – only the interpolated frames are saved (and zipped).

Usage (from command line):
    python3 -m eval.interpolator_cli \
      --pattern /kaggle/working/frames \
      --model_path /kaggle/frame-interpolation/pretrained_models/film_net/Style/saved_model \
      --times_to_interpolate 3
"""

import os
import sys
import math
import shutil
import threading
import functools
from typing import List, Iterable, Generator, Sequence
import tensorflow as tf
import numpy as np
from absl import app, flags, logging
from tqdm import tqdm
import natsort
import zipfile

# Set TensorFlow log level (prints TF messages to stderr)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define command-line flags.
_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='Glob pattern to match directories containing input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='Path to the TF2 saved model to use for interpolation.',
    required=True)
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=3,
    help='Number of times to run recursive midpoint interpolation. '
         'For each adjacent pair, (2^(times_to_interpolate)-1) new frames are generated.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=1,
    help='Alignment value to pad input dimensions if needed.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='Number of patches along the image height.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='Number of patches along the image width.')

# -----------------------------------------------------------------------------
# Interpolator wrapper class
# -----------------------------------------------------------------------------
class Interpolator:
    """
    Loads the saved model and wraps its call so that it accepts a dictionary of inputs.
    This avoids the error “Could not find matching concrete function …” when passing positional args.
    """
    def __init__(self, model_path: str, align: int, blocks: List[int]):
        self.model = tf.saved_model.load(model_path)
        self.align = align
        self.blocks = blocks

    def __call__(self, frame1: np.ndarray, frame2: np.ndarray, t: np.ndarray) -> np.ndarray:
        # The saved model expects a dict input with keys "x0", "x1", and "time".
        # Here, frame1 and frame2 are expected to have shape (1, H, W, 3) and t shape (1, 1).
        result = self.model({"x0": frame1, "x1": frame2, "time": t}, training=False)
        # Return as a numpy array.
        return result.numpy()

# -----------------------------------------------------------------------------
# Image I/O functions
# -----------------------------------------------------------------------------
def read_image(filename: str) -> np.ndarray:
    """Reads an image from file and returns a float32 RGB array with values in [0,1]."""
    image_data = tf.io.read_file(filename)
    image = tf.io.decode_image(image_data, channels=3)
    image = tf.cast(image, tf.float32).numpy()
    return image / 255.0

def write_image(filename: str, image: np.ndarray) -> None:
    """Writes an RGB image (values in [0,1]) to file as PNG (or JPEG if extension is .jpg)."""
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        encoded = tf.io.encode_jpeg(image_uint8)
    else:
        encoded = tf.io.encode_png(image_uint8)
    tf.io.write_file(filename, encoded)

# -----------------------------------------------------------------------------
# Recursive interpolation functions with a global progress bar
# -----------------------------------------------------------------------------
def _recursive_generator(frame1: np.ndarray,
                         frame2: np.ndarray,
                         num_recursions: int,
                         interpolator: Interpolator,
                         global_bar: tqdm,
                         gpu_info: str) -> Generator[np.ndarray, None, None]:
    """
    Recursively computes the mid-frame between frame1 and frame2.
    Updates the shared global progress bar for each interpolation call.
    """
    if num_recursions == 0:
        yield frame1
    else:
        # Prepare a time input of 0.5 (midpoint) with shape (1,1)
        t = np.full((1, 1), 0.5, dtype=np.float32)
        mid_frame = interpolator(frame1[np.newaxis, ...], frame2[np.newaxis, ...], t)[0]
        global_bar.update(1)
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1, interpolator, global_bar, gpu_info)
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1, interpolator, global_bar, gpu_info)

def interpolate_recursively_from_files(frames: List[str],
                                       times_to_interpolate: int,
                                       interpolator: Interpolator,
                                       global_bar: tqdm,
                                       gpu_info: str) -> Iterable[np.ndarray]:
    """
    Iterates over adjacent pairs of image file paths, interpolating between them recursively.
    Yields all interpolated frames (including original input frames).
    """
    n = len(frames)
    logging.info("Starting recursive interpolation on %s for %d adjacent pairs.", gpu_info, n - 1)
    for i in range(1, n):
        yield from _recursive_generator(
            read_image(frames[i - 1]),
            read_image(frames[i]),
            times_to_interpolate,
            interpolator,
            global_bar,
            gpu_info
        )
    # Yield the very last frame
    yield read_image(frames[-1])

# -----------------------------------------------------------------------------
# Functions to save output frames and create a ZIP archive.
# -----------------------------------------------------------------------------
def _output_frames(frames: List[np.ndarray], output_dir: str) -> None:
    """Writes all frames as PNG files into the specified directory."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for idx, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"frame_{idx:03d}.png")
        write_image(filename, frame)
    logging.info("Output frames saved in %s.", output_dir)

# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def _process_directory(directory: str) -> None:
    """
    For a given directory, gathers input frames, performs interpolation (on one or two GPUs),
    writes the output frames to a subfolder, and creates a ZIP archive of the results.
    The splitting for multi-GPU is done so that (if n is even) both GPUs process an equal number
    of frames (with an overlapping region for continuity), and if n is odd, the split is nearly equal.
    """
    # Gather input frames (supporting png, jpg, jpeg)
    exts = ['png', 'jpg', 'jpeg']
    frames = []
    for ext in exts:
        frames.extend(natsort.natsorted(tf.io.gfile.glob(os.path.join(directory, f"*.{ext}"))))
    if len(frames) < 2:
        logging.error("Not enough frames in %s to interpolate.", directory)
        return

    logging.info("Found %d input frames in %s.", len(frames), directory)
    times = _TIMES_TO_INTERPOLATE.value

    # Calculate total expected interpolation calls (for a single GPU run):
    total_expected = (len(frames) - 1) * (2 ** times - 1)
    # Create a shared global progress bar.
    global_bar = tqdm(total=total_expected, desc="Total Progress", position=0)

    # Determine available GPUs.
    physical_gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_gpus)
    logging.info("Number of GPUs available: %d", num_gpus)

    if num_gpus > 1:
        # Multi-GPU mode: split the input frames into two segments with overlap.
        n = len(frames)
        # For equal load with overlap, use:
        #   Segment1: frames[0:mid+1]
        #   Segment2: frames[mid-1:n]
        mid = n // 2 if n % 2 == 0 else (n // 2 + 1)
        segment1 = frames[:mid + 1]
        segment2 = frames[mid - 1:]
        logging.info("Splitting frames into two segments: Segment1: %d frames, Segment2: %d frames", len(segment1), len(segment2))

        results = [None, None]
        threads = []

        def worker(segment: List[str], gpu_index: int, result_list: list):
            try:
                with tf.device(f'/GPU:{gpu_index}'):
                    interp = Interpolator(_MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
                    res = list(interpolate_recursively_from_files(segment, times, interp, global_bar, gpu_info=f"/GPU:{gpu_index}"))
                result_list[gpu_index] = res
            except Exception as e:
                logging.error("Error in GPU %d worker: %s", gpu_index, e)
                result_list[gpu_index] = None

        for gpu_idx, segment in enumerate([segment1, segment2]):
            t = threading.Thread(target=worker, args=(segment, gpu_idx, results))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        global_bar.close()

        if results[0] is None or results[1] is None:
            logging.error("One of the GPU workers failed.")
            return

        # Merge results by taking full result from GPU0 and appending GPU1's result excluding its first frame.
        combined_frames = results[0] + results[1][1:]
    else:
        # Single GPU (or CPU) scenario.
        with tf.device('/GPU:0' if num_gpus == 1 else '/CPU:0'):
            interp = Interpolator(_MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
            combined_frames = list(interpolate_recursively_from_files(frames, times, interp, global_bar,
                                                                       gpu_info="/GPU:0" if num_gpus == 1 else "CPU"))
        global_bar.close()

    # Save output frames to a subdirectory.
    output_dir = os.path.join(directory, "interpolated")
    _output_frames(combined_frames, output_dir)

    # Create a ZIP archive of the output directory.
    zip_path = os.path.join(directory, "interpolated.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(base_name=os.path.join(directory, "interpolated"), format='zip', root_dir=output_dir)
    logging.info("Interpolated frames have been zipped at %s.", zip_path)

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main(argv: Sequence[str]) -> None:
    del argv  # Unused.
    directories = tf.io.gfile.glob(_PATTERN.value)
    if not directories:
        logging.error("No directories found matching pattern %s", _PATTERN.value)
        return
    for directory in directories:
        logging.info("Processing directory: %s", directory)
        _process_directory(directory)

if __name__ == '__main__':
    app.run(main)
