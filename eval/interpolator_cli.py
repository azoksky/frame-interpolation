import functools
import os
import shutil
from typing import List, Sequence
import threading

from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
from absl import logging
import natsort
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# Control TensorFlow C++ logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define flags.
_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='The glob pattern to find directories with input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=5,
    help='The number of times to run recursive midpoint interpolation. '
         'For each pair, the number of output frames is (2^times_to_interpolate - 1) + 1.')
_FPS = flags.DEFINE_integer(
    name='fps',
    default=30,
    help='(Unused now; video generation is removed).')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=1,
    help='If >1, pad the input so that its dimensions are evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='Number of patches along the image height.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='Number of patches along the image width.')
_INPUT_EXT = flags.DEFINE_list(
    name='input_ext',
    default=['png', 'jpg', 'jpeg'],
    help='List of allowed input image file extensions.')

def _process_directory(directory: str):
    """For a given directory of images, run frame interpolation and zip the result."""
    # Gather input frame file paths (all allowed extensions) in sorted order.
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
        for ext in _INPUT_EXT.value
    ]
    input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
    if len(input_frames) < 2:
        logging.error('Not enough frames in %s to interpolate.', directory)
        return

    logging.info('Generating in-between frames for directory: %s', directory)
    times = _TIMES_TO_INTERPOLATE.value
    N = len(input_frames)
    logging.info('Total input frames: %d', N)

    physical_gpus = tf.config.list_physical_devices('GPU')
    if len(physical_gpus) < 2:
        logging.info('Less than 2 GPUs found; using single GPU processing.')
        interp_instance = interpolator_lib.Interpolator(
            _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
        combined_frames = list(util.interpolate_recursively_from_files(
            input_frames, times, interp_instance))
    else:
        # Split the list into two non-overlapping segments.
        # For even number, both segments have equal length.
        # For odd, first segment gets one extra frame.
        split_index = (N + 1) // 2
        segment0 = input_frames[0:split_index]
        segment1 = input_frames[split_index:]
        logging.info('Segment split: segment0 has %d frames; segment1 has %d frames.',
                     len(segment0), len(segment1))
        results = [None, None]
        threads = []

        def process_segment(segment_frames, gpu_index, result_index):
            gpu_info = f'/GPU:{gpu_index}'
            with tf.device(gpu_info):
                interp_inst = interpolator_lib.Interpolator(
                    _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
                if len(segment_frames) < 2:
                    # If only one frame, just load it.
                    res = [util.read_image(segment_frames[0])]
                else:
                    res = list(util.interpolate_recursively_from_files(
                        segment_frames, times, interp_inst, gpu_info=gpu_info))
                results[result_index] = res

        t0 = threading.Thread(target=process_segment, args=(segment0, 0, 0))
        t0.start()
        threads.append(t0)
        if len(segment1) >= 1:
            t1 = threading.Thread(target=process_segment, args=(segment1, 1, 1))
            t1.start()
            threads.append(t1)
        else:
            results[1] = []
        for t in threads:
            t.join()

        # Compute boundary interpolation for the pair (last frame of segment0, first frame of segment1).
        if len(segment1) >= 1:
            boundary_pair = [input_frames[split_index - 1], input_frames[split_index]]
            with tf.device('/GPU:0'):
                interp_boundary = interpolator_lib.Interpolator(
                    _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
                boundary_frames = list(util.interpolate_recursively_from_files(
                    boundary_pair, times, interp_boundary, gpu_info='/GPU:0'))
        else:
            boundary_frames = []

        # Merge the results:
        # results[0] already ends with input_frames[split_index-1] and results[1] begins with input_frames[split_index].
        # We compute the boundary interpolation and remove its first and last frames (which duplicate the boundary).
        if len(segment1) >= 1 and boundary_frames:
            boundary_mid = boundary_frames[1:-1]
            seg1_tail = results[1][1:]  # Drop the first frame from GPU1's segment.
            combined_frames = results[0] + boundary_mid + seg1_tail
        else:
            combined_frames = results[0]

    # Write the combined frames to a folder.
    output_dir = os.path.join(directory, "interpolated_frames")
    util._output_frames(combined_frames, output_dir)
    # Zip the output frames.
    zip_path = os.path.join(directory, "interpolated_frames.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(os.path.join(directory, "interpolated_frames"), 'zip', root_dir=output_dir)
    logging.info('Interpolated frames have been zipped at %s.', zip_path)

def main(argv: Sequence[str]) -> None:
    del argv
    directories = tf.io.gfile.glob(_PATTERN.value)
    if not directories:
        logging.error('No directories found matching pattern %s', _PATTERN.value)
        return
    for directory in directories:
        _process_directory(directory)

if __name__ == '__main__':
    app.run(main)
