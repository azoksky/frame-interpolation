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
    for idx, frame in tqdm(enumerate(frames), total=len(frames),
                           ncols=100, desc="Saving frames", unit="frame"):
        util.write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
    logging.info('Output frames saved in %s.', frames_dir)

def _process_directory(directory: str):
    """Process one directory by distributing adjacent pair interpolation across GPUs.
    
    The total number of output frames is (n-1)*(2^T-1) + 1, where T is the
    number of interpolation recursions and n is the number of input frames.
    
    We split the n-1 adjacent pairs into two contiguous blocks:
      - GPU0 processes pairs 0..m-1 (input frames 0..m)
      - GPU1 processes pairs m..(n-2) (input frames m..n-1)
    and then merge the results, dropping the duplicate boundary frame.
    """
    # Gather input frame file paths.
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
    total_pairs = n - 1  # each adjacent pair is a work unit

    # Split adjacent pairs nearly equally.
    m = total_pairs // 2  # GPU0 processes pairs 0..m-1; GPU1 processes pairs m..total_pairs-1
    # Note: GPU0’s segment will use frames[0...m] and GPU1’s segment will use frames[m...n-1].
    logging.info("Total input frames: %d (i.e., %d pairs). GPU0 gets pairs 0..%d, GPU1 gets pairs %d..%d.",
                 n, total_pairs, m, m, total_pairs - 1)

    results = [None, None]  # to store each GPU's interpolated segment

    def process_segment(frames_subset: List[str], gpu_id: int, segment_name: str):
        device_str = f'/GPU:{gpu_id}'
        first_frame = os.path.basename(frames_subset[0])
        last_frame = os.path.basename(frames_subset[-1])
        logging.info("GPU %d processing %s: frames %s through %s", gpu_id, segment_name, first_frame, last_frame)
        with tf.device(device_str):
            interpolator = interpolator_lib.Interpolator(
                _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
            # Use the existing helper to process adjacent pairs in the given subset.
            # This function processes all pairs and returns a continuous output sequence.
            seg_result = list(util.interpolate_recursively_from_files(frames_subset, times, interpolator))
        results[gpu_id] = seg_result
        logging.info("GPU %d finished processing %s.", gpu_id, segment_name)

    threads = []

    # GPU0: process frames[0...m] (i.e. pairs 0 to m-1)
    gpu0_frames = input_frames[:m+1]
    t0 = threading.Thread(target=process_segment, args=(gpu0_frames, 0, "Segment GPU0"))
    t0.start()
    threads.append(t0)

    # GPU1: process frames[m...n-1] (i.e. pairs m to n-2)
    gpu1_frames = input_frames[m:]
    t1 = threading.Thread(target=process_segment, args=(gpu1_frames, 1, "Segment GPU1"))
    t1.start()
    threads.append(t1)

    for t in threads:
        t.join()

    # Merge the results:
    # GPU0’s output ends with the frame at index m, which is the same as GPU1’s first frame.
    # So we drop the first frame of GPU1’s output.
    combined_frames = results[0] + results[1][1:]
    logging.info("Total output frames: %d (expected: %d)", len(combined_frames),
                 (n - 1) * (2 ** times - 1) + 1)

    # Save and zip output frames.
    output_dir = os.path.join(directory, "interpolated_frames")
    _output_frames(combined_frames, output_dir)
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
