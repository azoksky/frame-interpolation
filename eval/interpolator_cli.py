"""
eval/interpolator_cli.py

A combined solution that performs recursive frame interpolation using a
saved TensorFlow model. Supports both single‐ and multi‐GPU scenarios with
graphical (tqdm) progress bars. Writes the interpolated frames to an output
directory and creates a ZIP archive for convenient download.

Usage:
    python3 -m eval.interpolator_cli \
      --pattern="/path/to/input_frames_dir/*" \
      --model_path="/path/to/saved_model" \
      --times_to_interpolate=5 \
      --align=1 \
      --block_height=1 \
      --block_width=1
"""

import os
import sys
import shutil
import functools
import threading
import logging
from typing import List, Iterable, Generator, Optional, Sequence

import numpy as np
import tensorflow as tf
from absl import app, flags, logging as absl_logging
import natsort
from tqdm.auto import tqdm

# Set TensorFlow log level and warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

# Command-line flags.
pattern_flag = flags.DEFINE_string(
    "pattern", None,
    "The pattern to determine the directories with the input frames.",
    required=True)
model_path_flag = flags.DEFINE_string(
    "model_path", None,
    "The path of the TF2 saved model to use.",
    required=True)
times_to_interpolate_flag = flags.DEFINE_integer(
    "times_to_interpolate", 5,
    "The number of times to run recursive midpoint interpolation.")
align_flag = flags.DEFINE_integer(
    "align", 1,
    "Alignment value (pad input size to be divisible by this value).")
block_height_flag = flags.DEFINE_integer(
    "block_height", 1,
    "Number of patches along height.")
block_width_flag = flags.DEFINE_integer(
    "block_width", 1,
    "Number of patches along width.")

# Setup logger.
logger = logging.getLogger("frame_interpolation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)


# ============================================================================
# Utility functions and classes.
# ============================================================================

def read_image(filename: str) -> np.ndarray:
    """Reads an image file into an [0,1] float32 numpy array."""
    image_data = tf.io.read_file(filename)
    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F


def write_image(filename: str, image: np.ndarray) -> None:
    """Writes a float32 [0,1] image array to a file (PNG/JPG)."""
    image_in_uint8_range = np.clip(image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
    image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.jpg':
        encoded = tf.io.encode_jpeg(image_in_uint8)
    else:
        encoded = tf.io.encode_png(image_in_uint8)
    tf.io.write_file(filename, encoded)


def _recursive_generator(
    frame1: np.ndarray,
    frame2: np.ndarray,
    num_recursions: int,
    interpolator,
    total: int,
    progress: dict,
    tqdm_bar: Optional[tqdm],
    gpu_info: str
) -> Generator[np.ndarray, None, None]:
    """
    Recursively interpolates between two frames.
    Updates the provided tqdm progress bar.
    """
    if num_recursions == 0:
        yield frame1
    else:
        # Prepare time input for midpoint interpolation.
        t_val = np.full((1,), 0.5, dtype=np.float32)
        mid_frame = interpolator(
            frame1[np.newaxis, ...],
            frame2[np.newaxis, ...],
            t_val
        )[0]
        progress["count"] += 1
        if tqdm_bar is not None:
            tqdm_bar.update(1)
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                        interpolator, total, progress, tqdm_bar, gpu_info)
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                        interpolator, total, progress, tqdm_bar, gpu_info)


def interpolate_recursively_from_files(
    frames: List[str],
    times_to_interpolate: int,
    interpolator,
    gpu_info: Optional[str] = None
) -> Iterable[np.ndarray]:
    """
    Loads frames from file paths and performs recursive midpoint interpolation.
    Displays a tqdm progress bar with GPU information.
    """
    if gpu_info is None:
        gpu_info = tf.test.gpu_device_name() or "CPU"
    n = len(frames)
    total = (n - 1) * (2**times_to_interpolate - 1)
    tqdm_bar = tqdm(total=total, ncols=100, desc=f"GPU {gpu_info}", colour='green')
    progress = {"count": 0}
    for i in range(1, n):
        yield from _recursive_generator(
            read_image(frames[i - 1]),
            read_image(frames[i]),
            times_to_interpolate,
            interpolator,
            total,
            progress,
            tqdm_bar,
            gpu_info
        )
    yield read_image(frames[-1])
    tqdm_bar.close()


def _output_frames(frames: List[np.ndarray], frames_dir: str):
    """Writes a list of frames as PNG files into a directory."""
    if tf.io.gfile.isdir(frames_dir):
        old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
        if old_frames:
            logger.info("Removing existing frames from %s", frames_dir)
            for f in old_frames:
                tf.io.gfile.remove(f)
    else:
        tf.io.gfile.makedirs(frames_dir)
    for idx, frame in enumerate(frames):
        write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
    logger.info("Output frames saved in %s", frames_dir)


class Interpolator:
    """
    A simple wrapper for the saved model. Adjust the __call__ method
    if your model requires different inputs.
    """
    def __init__(self, model_path: str, align: int, block_size: List[int]):
        self.model = tf.saved_model.load(model_path)
        self.align = align
        self.block_size = block_size

    def __call__(self, frame1, frame2, t):
        # Call your saved model. Adjust as necessary.
        return self.model(frame1, frame2, t)


# ============================================================================
# Main processing.
# ============================================================================

def _process_directory(directory: str):
    """
    Processes a directory of input frames:
      1. Reads all frames (supported extensions).
      2. If more than one GPU is available, splits the frames into two segments.
         GPU0 processes the first segment, GPU1 processes the second (overlapping one frame).
      3. Performs recursive interpolation on each segment with graphical logging.
      4. Combines the outputs (dropping the duplicate boundary frame) and writes the frames.
      5. Creates a ZIP archive of the output.
    """
    # Gather input frame file paths.
    extensions = ['png', 'jpg', 'jpeg']
    frames_list = []
    for ext in extensions:
        frames_list.extend(tf.io.gfile.glob(os.path.join(directory, f"*.{ext}")))
    frames_list = natsort.natsorted(frames_list)
    if len(frames_list) < 2:
        logger.error("Not enough frames in %s to interpolate.", directory)
        return

    logger.info("Generating in-between frames for %s.", directory)
    times = times_to_interpolate_flag.value

    physical_gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_gpus)
    if num_gpus > 1:
        num_frames = len(frames_list)
        # For even split: GPU0 gets frames[0..mid-1] extra one if odd.
        mid = (num_frames + 1) // 2
        segment0 = frames_list[:mid]
        segment1 = frames_list[mid-1:]  # Overlap last frame of seg0.
        logger.info("Total %d frames split as follows: Segment 0: %d frames, Segment 1: %d frames.",
                    num_frames, len(segment0), len(segment1))
        results = [None, None]
        threads = []

        def worker(segment: List[str], gpu_index: int):
            logger.info("GPU %d processing frames %s to %s", gpu_index,
                        os.path.basename(segment[0]),
                        os.path.basename(segment[-1]))
            with tf.device(f'/GPU:{gpu_index}'):
                interp = Interpolator(
                    model_path_flag.value,
                    align_flag.value,
                    [block_height_flag.value, block_width_flag.value])
                res = list(interpolate_recursively_from_files(segment, times, interp, gpu_info=f"/GPU:{gpu_index}"))
            results[gpu_index] = res
            logger.info("GPU %d finished processing.", gpu_index)

        for i, seg in enumerate([segment0, segment1]):
            t = threading.Thread(target=worker, args=(seg, i))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        combined_frames = results[0] + results[1][1:]
    else:
        device = '/GPU:0' if num_gpus >= 1 else '/CPU:0'
        with tf.device(device):
            interp = Interpolator(
                model_path_flag.value,
                align_flag.value,
                [block_height_flag.value, block_width_flag.value])
            combined_frames = list(interpolate_recursively_from_files(frames_list, times, interp))
    output_dir = os.path.join(directory, "interpolated")
    _output_frames(combined_frames, output_dir)
    zip_path = os.path.join(directory, "interpolated.zip")
    if tf.io.gfile.exists(zip_path):
        tf.io.gfile.remove(zip_path)
    shutil.make_archive(os.path.join(directory, "interpolated"), 'zip', root_dir=output_dir)
    logger.info("Interpolated frames have been zipped at %s", zip_path)


def main(argv: Sequence[str]) -> None:
    del argv  # Unused.
    directories = tf.io.gfile.glob(pattern_flag.value)
    if not directories:
        logger.error("No directories found matching pattern %s", pattern_flag.value)
        return
    for directory in directories:
        _process_directory(directory)


if __name__ == '__main__':
    app.run(main)
