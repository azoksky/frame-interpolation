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
    """Process one directory of frames: perform interpolation in parallel across GPUs."""
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
    times = _TIMES_TO_INTERPOLATE.value
    total_frames = len(input_frames)
    total_pairs = total_frames - 1  # each adjacent pair is independent
    steps_per_pair = 2 ** times - 1  # each pair requires these many interpolation steps

    # Prepare a container to store the result for each pair.
    pair_results = [None] * total_pairs

    # Create a lock for thread-safe tqdm updates (tqdm is thread-safe when using its built-in lock).
    tqdm_lock = threading.Lock()

    def process_pair(pair_index: int, gpu_id: int):
        """Process a single pair (frame_i, frame_i+1) on the given GPU."""
        with tf.device(f'/GPU:{gpu_id}'):
            interpolator = interpolator_lib.Interpolator(
                _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
            progress = {"count": 0}
            # Create a dedicated progress bar for this pair.
            # (Alternatively, you could use one global progress bar per GPU.)
            with tqdm(total=steps_per_pair,
                      desc=f"GPU {gpu_id} Pair {pair_index+1}/{total_pairs}",
                      ncols=100, position=gpu_id, leave=False) as pbar:
                # Use the shared lock for thread-safe writing.
                with tqdm_lock:
                    result = list(util._recursive_generator(
                        util.read_image(input_frames[pair_index]),
                        util.read_image(input_frames[pair_index+1]),
                        times,
                        interpolator,
                        steps_per_pair,
                        progress,
                        f'/GPU:{gpu_id}',
                        pbar=pbar))
            pair_results[pair_index] = result

    # Worker function for each GPU thread: process all pairs assigned to that GPU.
    def gpu_worker(gpu_id: int):
        for i in range(total_pairs):
            if i % 2 == gpu_id:  # assign pair by round-robin (ensuring near-equal load)
                process_pair(i, gpu_id)

    # Launch two threads, one per GPU.
    threads = []
    for gpu_id in [0, 1]:
        t = threading.Thread(target=gpu_worker, args=(gpu_id,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # Merge the pair results in order.
    # For the first pair, include all frames; for subsequent pairs, drop the first frame to avoid duplicates.
    combined_frames = []
    for idx, pair_frames in enumerate(pair_results):
        if idx == 0:
            combined_frames.extend(pair_frames)
        else:
            combined_frames.extend(pair_frames[1:])
    
    # Save the final sequence.
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
