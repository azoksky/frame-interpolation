import functools
import os
import shutil
from typing import List, Sequence
import threading  # Added for multi-GPU threading

from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
from absl import logging
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# Controls TF_CPP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='The pattern to determine the directories with the input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=5,
    help='The number of times to run recursive midpoint interpolation. '
         'The number of output frames will be 2^times_to_interpolate+1.')
_FPS = flags.DEFINE_integer(
    name='fps',
    default=30,
    help='Frames per second to play interpolated videos in slow motion.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=1,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='An int >= 1, number of patches along height, '
         'patch_height = height//block_height, should be evenly divisible.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='An int >= 1, number of patches along width, '
         'patch_width = width//block_width, should be evenly divisible.')
_OUTPUT_VIDEO = flags.DEFINE_boolean(
    name='output_video',
    default=False,
    help='If true, creates a video of the frames in the interpolated_frames/ subdirectory')

# List of allowable image file extensions
_INPUT_EXT = ['png', 'jpg', 'jpeg']

def _output_frames(frames: List[np.ndarray], frames_dir: str):
    """Writes a list of frames to PNG files in a directory."""
    if tf.io.gfile.isdir(frames_dir):
        old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
        if old_frames:
            logging.info('Removing existing frames from %s.', frames_dir)
            for old_frame in old_frames:
                tf.io.gfile.remove(old_frame)
    else:
        tf.io.gfile.makedirs(frames_dir)
    # Save each frame as PNG with an index-based filename.
    for idx, frame in tqdm(enumerate(frames), total=len(frames), ncols=100, colour='green'):
        util.write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
    logging.info('Output frames saved in %s.', frames_dir)

def _process_directory(directory: str):
    """Process a single directory of frames: interpolate and save results."""
    # Gather input frame file paths with supported extensions, in sorted order.
    input_frames_list = [
        natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
        for ext in _INPUT_EXT
    ]
    # Flatten the list of lists into one list of file paths
    input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
    if len(input_frames) < 2:
        logging.error('Not enough frames in %s to interpolate.', directory)
        return

    logging.info('Generating in-between frames for %s.', directory)
    num_input_frames = len(input_frames)
    times = _TIMES_TO_INTERPOLATE.value

    # Determine the number of available GPUs
    physical_gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_gpus)
    # If more than one GPU, split workload between them
    if num_gpus > 1:
        # Calculate number of interpolation tasks (adjacent frame pairs) 
        num_pairs = num_input_frames - 1
        # Distribute pairs as evenly as possible among GPUs. If an odd number of tasks, GPU 0 gets one extra.
        base_tasks = num_pairs // num_gpus
        remainder = num_pairs % num_gpus
        tasks_per_gpu = [base_tasks + (1 if i < remainder else 0) for i in range(num_gpus)]

        # Prepare threads for each GPU
        threads = []
        results = [None] * num_gpus

        # Function to interpolate a segment of frames on a specific GPU
        def interpolate_segment(seg_frames: List[str], gpu_index: int, start_idx: int, end_idx: int):
            # Log which frames this GPU will process (using file names for clarity)
            first_frame = os.path.basename(seg_frames[0])
            last_frame = os.path.basename(seg_frames[-1])
            logging.info("GPU %d processing frames %s through %s", gpu_index, first_frame, last_frame)
            with tf.device(f'/GPU:{gpu_index}'):
                # Create a new interpolator for this GPU (load model on that device)
                interpolator = interpolator_lib.Interpolator(
                    _MODEL_PATH.value, _ALIGN.value,
                    [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
                # Perform recursive interpolation on this segment
                seg_result = list(util.interpolate_recursively_from_files(seg_frames, times, interpolator))
            results[gpu_index] = seg_result
            logging.info("GPU %d finished processing frames %s through %s", gpu_index, first_frame, last_frame)

        # Determine frame index ranges for each GPU segment (with overlapping boundary frames)
        start_index = 0
        for gpu_idx, task_count in enumerate(tasks_per_gpu):
            if task_count == 0:
                continue  # Skip if no tasks assigned (in case of more GPUs than tasks)
            end_index = start_index + task_count  # end_index is index of last frame in this segment
            segment_frame_paths = input_frames[start_index : end_index + 1]  # include end_index frame
            # Launch a thread for this GPU segment
            t = threading.Thread(target=interpolate_segment,
                                  args=(segment_frame_paths, gpu_idx, start_index, end_index))
            t.start()
            threads.append(t)
            # Overlap: next segment starts at the last frame of this segment (to avoid missing the boundary frame)
            start_index = end_index

        # Wait for all GPU threads to finish
        for t in threads:
            t.join()

        # Combine results from all GPUs, dropping duplicate overlapping frames at segment boundaries
        # Start with GPU0's full output frames
        combined_frames = results[0]
        for gpu_idx in range(1, num_gpus):
            if results[gpu_idx] is None:
                continue
            # Each subsequent segment starts with a frame that was the last frame of the previous segment.
            # Skip the first frame of this segment's result to avoid duplication.
            segment_frames = results[gpu_idx]
            combined_frames.extend(segment_frames[1:])
    else:
        # Single GPU or CPU: process all frames sequentially on available device
        interpolator = interpolator_lib.Interpolator(
            _MODEL_PATH.value, _ALIGN.value,
            [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
        combined_frames = list(util.interpolate_recursively_from_files(input_frames, times, interpolator))

    # Save all interpolated (and original) frames into the output directory
    _output_frames(combined_frames, f'{directory}/interpolated_frames')
    # **No automatic video generation**: skip creating interpolated.mp4
    # Instead, optionally we could create a video manually if needed, but we disable this per requirements.
    # if _OUTPUT_VIDEO.value:
    #     media.write_video(f'{directory}/interpolated.mp4', combined_frames, fps=_FPS.value)
    #     logging.info('Output video saved at %s/interpolated.mp4.', directory)

    # **Zip the interpolated frames** for convenient download or storage
    zip_path = f'{directory}/interpolated_frames.zip'
    if tf.io.gfile.exists(zip_path):
        tf.io.gfile.remove(zip_path)
    # Create a ZIP archive of the entire interpolated_frames directory
    shutil.make_archive(f'{directory}/interpolated_frames', 'zip', root_dir=f'{directory}/interpolated_frames')
    logging.info('Interpolated frames have been zipped at %s.', zip_path)

def main(argv: Sequence[str]) -> None:
    """Main function to run frame interpolation on all directories matching the pattern."""
    del argv  # Unused
    directories = tf.io.gfile.glob(_PATTERN.value)
    if not directories:
        logging.error('No directories found matching pattern %s', _PATTERN.value)
        return
    for directory in directories:
        _process_directory(directory)

if __name__ == '__main__':
    app.run(main)
