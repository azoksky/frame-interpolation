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

# Flags (video-related flags have been removed).
_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='Glob pattern to find directories containing input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=5,
    help='The number of times to do recursive midpoint interpolation.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=1,
    help='Pad the input so its dimensions are divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='Number of patches along height.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='Number of patches along width.')

def _process_directory(directory: str):
    """Processes one directory of frames: interpolates and saves the result."""
    # Gather input frame file paths (allowed extensions: png, jpg, jpeg) in sorted order.
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
        for ext in ['png', 'jpg', 'jpeg']
    ]
    # Flatten the list of lists.
    input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
    if len(input_frames) < 2:
        logging.error('Not enough frames in %s to interpolate.', directory)
        return

    logging.info('Generating in-between frames for directory: %s', directory)
    N = len(input_frames)
    logging.info('Total input frames: %d', N)

    physical_gpus = tf.config.list_physical_devices('GPU')
    if len(physical_gpus) < 2:
        logging.info('Less than 2 GPUs found; using single GPU processing.')
        device = physical_gpus[0].name if physical_gpus else '/CPU:0'
        with tf.device(device):
            interp_instance = interpolator_lib.Interpolator(
                _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
            combined_frames = list(util.interpolate_recursively_from_files(
                input_frames, _TIMES_TO_INTERPOLATE.value, interp_instance))
    else:
        # There are N-1 frame pairs.
        num_pairs = N - 1
        tasks_gpu0 = num_pairs // 2 + (num_pairs % 2)  # GPU0 gets extra pair if odd.
        # GPU0 processes frames[0:tasks_gpu0+1]; GPU1 processes frames[tasks_gpu0:].
        segment0 = input_frames[0 : tasks_gpu0 + 1]
        segment1 = input_frames[tasks_gpu0 :]
        logging.info('Segment split: GPU0 will process %d frames; GPU1 will process %d frames.',
                     len(segment0), len(segment1))
        results = [None, None]
        threads = []

        def process_segment(segment_frames: List[str], gpu_index: int, result_index: int):
            gpu_info = f'/GPU:{gpu_index}'
            with tf.device(gpu_info):
                interp_inst = interpolator_lib.Interpolator(
                    _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
                total_steps = (len(segment_frames) - 1) * (2**_TIMES_TO_INTERPOLATE.value - 1)
                pbar = tqdm(total=total_steps, desc=f"GPU {gpu_info}", ncols=100, colour='green')
                seg_result = []
                for frame in util.interpolate_recursively_from_files(segment_frames, _TIMES_TO_INTERPOLATE.value, interp_inst, gpu_info=gpu_info):
                    seg_result.append(frame)
                    pbar.update(1)
                pbar.close()
                results[result_index] = seg_result
                logging.info("GPU %d finished processing segment; generated %d frames.", gpu_index, len(seg_result))

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
        # Merge results: drop the first frame from GPU1's result to avoid duplication.
        if results[1]:
            combined_frames = results[0] + results[1][1:]
        else:
            combined_frames = results[0]

    # Save all frames into an output directory.
    output_dir = os.path.join(directory, "interpolated_frames")
    util.output_frames(combined_frames, output_dir)
    # Zip the output frames.
    zip_path = os.path.join(directory, "interpolated_frames.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(os.path.join(directory, "interpolated_frames"), 'zip', root_dir=output_dir)
    logging.info('Interpolated frames have been zipped at %s.', zip_path)

def main(argv: Sequence[str]) -> None:
    del argv  # Unused.
    directories = tf.io.gfile.glob(_PATTERN.value)
    if not directories:
        logging.error('No directories found matching pattern %s', _PATTERN.value)
        return
    for directory in directories:
        _process_directory(directory)

if __name__ == '__main__':
    app.run(main)
