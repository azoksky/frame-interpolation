#!/usr/bin/env python3
"""
This script performs recursive frame interpolation on directories of input frames.
It supports both single-GPU and multi-GPU setups with a unified progress bar.
No video is generated automatically; only the interpolated frames are saved (and then zipped).

Usage:
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

# Set TensorFlow log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Command-line flags.
_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='Glob pattern for directories containing input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='Path to the TF2 saved model to use for interpolation.',
    required=True)
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=3,
    help='Number of recursive midpoint interpolations to perform.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=1,
    help='Alignment for input dimensions.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='Number of patches along image height.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='Number of patches along image width.')

# -----------------------------------------------------------------------------
# Interpolator class wrapper.
# -----------------------------------------------------------------------------
class Interpolator:
    def __init__(self, model_path: str, align: int, blocks: List[int]):
        self.model = tf.saved_model.load(model_path)
        self.align = align
        self.blocks = blocks

    def __call__(self, frame1: np.ndarray, frame2: np.ndarray, t: np.ndarray) -> np.ndarray:
        # Ensure inputs have the correct shape:
        # If an image is 3D (H,W,3), add a batch dimension.
        if frame1.ndim == 3:
            frame1 = frame1[np.newaxis, ...]
        if frame2.ndim == 3:
            frame2 = frame2[np.newaxis, ...]
        inputs = {"x0": frame1, "x1": frame2, "time": t}
        result = self.model(inputs, training=False)
        # The model may return a dict or list; take the first value if so.
        if isinstance(result, dict):
            result = list(result.values())[0]
        elif isinstance(result, list):
            result = result[0]
        # If the result is a tensor, convert to numpy.
        if hasattr(result, "numpy"):
            return result.numpy()
        return result

# -----------------------------------------------------------------------------
# Image I/O functions.
# -----------------------------------------------------------------------------
def read_image(filename: str) -> np.ndarray:
    """
    Reads an image file and returns a float32 RGB image with values in [0,1].
    Uses tf.image.decode_png or decode_jpeg to ensure 3 channels.
    """
    image_data = tf.io.read_file(filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.png':
        image = tf.image.decode_png(image_data, channels=3)
    else:
        image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32).numpy()
    return image / 255.0

def write_image(filename: str, image: np.ndarray) -> None:
    """Writes a float32 RGB image (values in [0,1]) to file as PNG (or JPEG)."""
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        encoded = tf.io.encode_jpeg(image_uint8)
    else:
        encoded = tf.io.encode_png(image_uint8)
    tf.io.write_file(filename, encoded)

# -----------------------------------------------------------------------------
# Recursive interpolation functions.
# -----------------------------------------------------------------------------
def _recursive_generator(frame1: np.ndarray,
                         frame2: np.ndarray,
                         num_recursions: int,
                         interpolator: Interpolator,
                         global_bar: tqdm,
                         gpu_info: str) -> Generator[np.ndarray, None, None]:
    """
    Recursively computes the mid-frame between frame1 and frame2.
    Each interpolation call updates the shared progress bar.
    """
    if num_recursions == 0:
        yield frame1
    else:
        t = np.full((1, 1), 0.5, dtype=np.float32)
        # Call the model. (The __call__ method adds batch dimensions if needed.)
        mid_frame = interpolator(frame1, frame2, t)[0]  # remove batch dimension
        global_bar.update(1)
        yield from _recursive_generator(frame1, mid_frame, num_recursions - 1, interpolator, global_bar, gpu_info)
        yield from _recursive_generator(mid_frame, frame2, num_recursions - 1, interpolator, global_bar, gpu_info)

def interpolate_recursively_from_files(frames: List[str],
                                       times_to_interpolate: int,
                                       interpolator: Interpolator,
                                       global_bar: tqdm,
                                       gpu_info: str) -> Iterable[np.ndarray]:
    """
    Iterates over adjacent pairs of image file paths and yields all interpolated frames.
    """
    logging.info("Starting interpolation on %s for %d adjacent pairs.", gpu_info, len(frames)-1)
    for i in range(1, len(frames)):
        yield from _recursive_generator(
            read_image(frames[i-1]),
            read_image(frames[i]),
            times_to_interpolate,
            interpolator,
            global_bar,
            gpu_info
        )
    yield read_image(frames[-1])

# -----------------------------------------------------------------------------
# Function to output frames and create a ZIP archive.
# -----------------------------------------------------------------------------
def _output_frames(frames: List[np.ndarray], output_dir: str) -> None:
    """Writes all frames as PNG files into the specified directory."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for idx, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"frame_{idx:03d}.png")
        write_image(filename, frame)
    logging.info("Saved output frames to %s", output_dir)

# -----------------------------------------------------------------------------
# Main processing function.
# -----------------------------------------------------------------------------
def _process_directory(directory: str) -> None:
    """
    Processes one directory of input frames:
      - Reads all image files (png/jpg/jpeg)
      - If multiple GPUs are available, splits frames evenly (with one overlapping frame)
      - Runs recursive interpolation (using a shared progress bar)
      - Saves interpolated frames to a subdirectory and creates a ZIP archive.
    """
    exts = ['png', 'jpg', 'jpeg']
    frames = []
    for ext in exts:
        frames.extend(natsort.natsorted(tf.io.gfile.glob(os.path.join(directory, f"*.{ext}"))))
    if len(frames) < 2:
        logging.error("Not enough frames in %s to interpolate.", directory)
        return
    logging.info("Found %d input frames in %s", len(frames), directory)
    times = _TIMES_TO_INTERPOLATE.value
    total_expected = (len(frames)-1) * (2**times - 1)
    global_bar = tqdm(total=total_expected, desc="Total Progress", position=0)

    physical_gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_gpus)
    logging.info("Number of GPUs available: %d", num_gpus)

    if num_gpus > 1:
        # Split frames into two segments (with one frame overlap)
        n = len(frames)
        mid = n//2 if n % 2 == 0 else (n//2 + 1)
        segment1 = frames[:mid+1]
        segment2 = frames[mid-1:]
        logging.info("Segment1: %d frames, Segment2: %d frames", len(segment1), len(segment2))
        results = [None, None]
        threads = []
        def worker(segment: List[str], gpu_index: int, res_list: list):
            try:
                with tf.device(f'/GPU:{gpu_index}'):
                    interp = Interpolator(_MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
                    res = list(interpolate_recursively_from_files(segment, times, interp, global_bar,
                                                                   gpu_info=f"/GPU:{gpu_index}"))
                res_list[gpu_index] = res
            except Exception as e:
                logging.error("Error in GPU %d worker: %s", gpu_index, e)
                res_list[gpu_index] = None
        for idx, segment in enumerate([segment1, segment2]):
            t_thread = threading.Thread(target=worker, args=(segment, idx, results))
            t_thread.start()
            threads.append(t_thread)
        for t in threads:
            t.join()
        global_bar.close()
        if results[0] is None or results[1] is None:
            logging.error("One of the GPU workers failed.")
            return
        # Merge results; skip the first frame of the second segment to avoid duplication.
        combined_frames = results[0] + results[1][1:]
    else:
        with tf.device('/GPU:0' if num_gpus==1 else '/CPU:0'):
            interp = Interpolator(_MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
            combined_frames = list(interpolate_recursively_from_files(frames, times, interp, global_bar,
                                                                       gpu_info="/GPU:0" if num_gpus==1 else "CPU"))
        global_bar.close()

    output_dir = os.path.join(directory, "interpolated")
    _output_frames(combined_frames, output_dir)

    # Create ZIP archive.
    zip_path = os.path.join(directory, "interpolated.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(base_name=os.path.join(directory, "interpolated"), format='zip', root_dir=output_dir)
    logging.info("Zipped interpolated frames at %s", zip_path)

# -----------------------------------------------------------------------------
# Main entry point.
# -----------------------------------------------------------------------------
def main(argv: Sequence[str]) -> None:
    del argv
    directories = tf.io.gfile.glob(_PATTERN.value)
    if not directories:
        logging.error("No directories found matching pattern %s", _PATTERN.value)
        return
    for directory in directories:
        logging.info("Processing directory: %s", directory)
        _process_directory(directory)

if __name__ == '__main__':
    app.run(main)
