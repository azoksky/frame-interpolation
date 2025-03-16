import functools
import math
import os
import shutil
from typing import List, Sequence
import threading

from . import interpolator as interpolator_lib
from . import util
from absl import app, flags, logging
import natsort
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# Fallback for logging_redirect_tqdm in case it's not available.
try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def logging_redirect_tqdm():
        yield

# Set TensorFlow log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Command-line flags.
_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='Glob pattern for directories containing input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.',
    required=True)
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=5,
    help='Number of recursive midpoint interpolations to perform.')
# Unused FPS flag (video generation removed)
_FPS = flags.DEFINE_integer(
    name='fps',
    default=30,
    help='(Unused) Frames per second for video.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=1,
    help='Pad input size so it is divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='Number of patches along height (1 means no tiling).')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='Number of patches along width (1 means no tiling).')

# Allowed file extensions.
_INPUT_EXT = ['png', 'jpg', 'jpeg']

def _output_frames(frames: List[np.ndarray], frames_dir: str):
    """Saves a list of frames as PNG files in the given directory."""
    if tf.io.gfile.isdir(frames_dir):
        old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
        if old_frames:
            logging.info('Removing existing frames from %s.', frames_dir)
            for old_frame in old_frames:
                tf.io.gfile.remove(old_frame)
    else:
        tf.io.gfile.makedirs(frames_dir)
    # Use tqdm to show saving progress.
    for idx, frame in tqdm(enumerate(frames), total=len(frames),
                           ncols=100, desc="Saving frames", unit="frame"):
        util.write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
    logging.info('Output frames saved in %s.', frames_dir)

def _process_directory(directory: str):
    """Process one directory of frames: perform interpolation and save results."""
    # Gather input frame file paths (with allowed extensions).
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
        for ext in _INPUT_EXT
    ]
    input_frames = functools.reduce(lambda x, y: x + y, input_frames_list, [])
    if len(input_frames) < 2:
        logging.error('Not enough frames in %s to interpolate.', directory)
        return

    logging.info('Generating in-between frames for %s.', directory)
    num_input_frames = len(input_frames)
    times = _TIMES_TO_INTERPOLATE.value

    # Determine available GPUs.
    physical_gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_gpus)

    if num_gpus >= 2:
        # Equal split: for even frames, split equally; for odd, assign the extra frame to GPU0.
        split_index = math.ceil(num_input_frames / 2)
        segment1 = input_frames[:split_index]
        segment2 = input_frames[split_index:]
        logging.info("Total %d frames split as follows: Segment 1: %d frames, Segment 2: %d frames.",
                     num_input_frames, len(segment1), len(segment2))
        threads = []
        results = [None, None]

        def interpolate_segment(seg_frames: List[str], gpu_index: int):
            device_str = f'/GPU:{gpu_index}'
            first_frame = os.path.basename(seg_frames[0])
            last_frame = os.path.basename(seg_frames[-1])
            logging.info("GPU %d processing frames %s through %s", gpu_index, first_frame, last_frame)
            with tf.device(device_str):
                interpolator = interpolator_lib.Interpolator(
                    _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
                num_pairs = len(seg_frames) - 1
                total_steps = num_pairs * (2 ** times - 1)
                progress = {"count": 0}
                seg_result = []
                with tqdm(total=total_steps, desc=f"GPU {gpu_index} Interpolation", ncols=100) as pbar:
                    for i in range(1, len(seg_frames)):
                        pair_frames = list(util._recursive_generator(
                            util.read_image(seg_frames[i - 1]),
                            util.read_image(seg_frames[i]),
                            times, interpolator, total_steps, progress, device_str,
                            pbar=pbar))
                        # Append all but the last frame.
                        seg_result.extend(pair_frames[:-1])
                    seg_result.append(util.read_image(seg_frames[-1]))
            results[gpu_index] = seg_result
            logging.info("GPU %d finished processing frames %s through %s", gpu_index, first_frame, last_frame)

        # Start threads for each segment.
        for gpu_idx, seg in enumerate([segment1, segment2]):
            t = threading.Thread(target=interpolate_segment, args=(seg, gpu_idx))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # Process the boundary pair (last frame of segment1 and first frame of segment2).
        logging.info("Processing boundary interpolation between frame %s and %s",
                     os.path.basename(segment1[-1]), os.path.basename(segment2[0]))
        with tf.device('/GPU:0'):
            interpolator = interpolator_lib.Interpolator(
                _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
            total_steps = 2 ** times - 1
            progress = {"count": 0}
            with tqdm(total=total_steps, desc="Boundary Interpolation", ncols=100) as pbar:
                boundary_frames = list(util._recursive_generator(
                    util.read_image(segment1[-1]),
                    util.read_image(segment2[0]),
                    times, interpolator, total_steps, progress, '/GPU:0',
                    pbar=pbar))
        # Merge segments by removing the duplicate boundary frame from GPU0's result and inserting the boundary interpolation.
        combined_frames = results[0][:-1] + boundary_frames + results[1]

    else:
        # Single GPU or CPU: process sequentially.
        logging.info("Using single GPU/CPU for processing.")
        with tf.device('/GPU:0' if num_gpus == 1 else '/CPU:0'):
            interpolator = interpolator_lib.Interpolator(
                _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
            num_pairs = len(input_frames) - 1
            total_steps = num_pairs * (2 ** times - 1)
            progress = {"count": 0}
            pbar = tqdm(total=total_steps, desc="Interpolation progress", ncols=100)
            combined_frames = list(util._recursive_generator_chain(
                input_frames, times, interpolator, total_steps, progress, pbar))
            pbar.close()

    # Save output frames.
    output_dir = os.path.join(directory, "interpolated_frames")
    _output_frames(combined_frames, output_dir)
    # Zip the interpolated frames for download.
    zip_path = os.path.join(directory, "interpolated_frames.zip")
    if tf.io.gfile.exists(zip_path):
        tf.io.gfile.remove(zip_path)
    shutil.make_archive(os.path.join(directory, "interpolated_frames"), 'zip', root_dir=output_dir)
    logging.info('Interpolated frames have been zipped at %s.', zip_path)

def main(argv: Sequence[str]) -> None:
    del argv
    with logging_redirect_tqdm():
        directories = tf.io.gfile.glob(_PATTERN.value)
        if not directories:
            logging.error('No directories found matching pattern %s', _PATTERN.value)
            return
        for directory in directories:
            _process_directory(directory)

if __name__ == '__main__':
    app.run(main)
