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

# Fallback for logging_redirect_tqdm.
try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def logging_redirect_tqdm():
        yield

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
    for idx, frame in tqdm(enumerate(frames), total=len(frames),
                           ncols=100, desc="Saving frames", unit="frame"):
        util.write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
    logging.info('Output frames saved in %s.', frames_dir)

def _process_directory(directory: str):
    """Distributes adjacent-pair interpolation across GPUs using one shared progress bar.
    
    The input frames are split into two segments (non-overlapping) for the GPUs.
    After interpolation, the segments are merged (dropping the duplicate boundary frame).
    """
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
        for ext in _INPUT_EXT
    ]
    input_frames = functools.reduce(lambda x, y: x + y, input_frames_list, [])
    n = len(input_frames)
    if n < 2:
        logging.error('Not enough frames in %s to interpolate.', directory)
        return

    logging.info('Generating in-between frames for %s.', directory)
    times = _TIMES_TO_INTERPOLATE.value
    total_pairs = n - 1
    global_total_steps = total_pairs * (2 ** times - 1)

    # Split adjacent pairs nearly equally:
    # GPU0 processes pairs corresponding to frames[0...m] and GPU1 processes frames[m...n-1].
    m = total_pairs // 2
    logging.info("Total input frames: %d (i.e., %d pairs). GPU0 gets pairs 0..%d, GPU1 gets pairs %d..%d.",
                 n, total_pairs, m, m, total_pairs - 1)

    results = [None, None]

    # Create one shared progress bar with blue description.
    shared_pbar = tqdm(total=global_total_steps,
                       desc="\033[94mOverall Interpolation\033[0m", ncols=100)

    def process_segment(frames_subset: List[str], gpu_id: int, segment_name: str, seg_total_steps: int):
        device_str = f'/GPU:{gpu_id}'
        first_frame = os.path.basename(frames_subset[0])
        last_frame = os.path.basename(frames_subset[-1])
        logging.info("GPU %d processing %s: frames %s through %s", gpu_id, segment_name, first_frame, last_frame)
        with tf.device(device_str):
            interpolator = interpolator_lib.Interpolator(
                _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
            progress = {"count": 0}
            seg_result = list(util._recursive_generator_chain(frames_subset, times, interpolator, seg_total_steps, progress, shared_pbar))
        results[gpu_id] = seg_result
        logging.info("GPU %d finished processing %s.", gpu_id, segment_name)

    threads = []

    # GPU0: process frames[0...m] (m pairs).
    gpu0_frames = input_frames[:m+1]
    seg0_steps = m * (2 ** times - 1)
    t0 = threading.Thread(target=process_segment, args=(gpu0_frames, 0, "Segment GPU0", seg0_steps))
    t0.start()
    threads.append(t0)

    # GPU1: process frames[m...n-1] ((total_pairs - m) pairs).
    gpu1_frames = input_frames[m:]
    seg1_steps = (total_pairs - m) * (2 ** times - 1)
    t1 = threading.Thread(target=process_segment, args=(gpu1_frames, 1, "Segment GPU1", seg1_steps))
    t1.start()
    threads.append(t1)

    for t in threads:
        t.join()

    shared_pbar.close()

    # Merge the two segments; drop the duplicate boundary frame.
    combined_frames = results[0] + results[1][1:]
    expected_total_frames = (n - 1) * (2 ** times - 1) + 1
    logging.info("Total output frames: %d (expected: %d)", len(combined_frames), expected_total_frames)

    _output_frames(combined_frames, os.path.join(directory, "interpolated_frames"))
    zip_path = os.path.join(directory, "interpolated_frames.zip")
    if tf.io.gfile.exists(zip_path):
        tf.io.gfile.remove(zip_path)
    shutil.make_archive(os.path.join(directory, "interpolated_frames"), 'zip', root_dir=os.path.join(directory, "interpolated_frames"))
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
