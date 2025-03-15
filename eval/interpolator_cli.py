import functools
import os
from typing import List, Sequence

# Import internal modules
from . import interpolator as interpolator_lib
from . import util

from absl import app
from absl import flags
from absl import logging
# Removed: import apache_beam as beam (no longer used for multi-GPU inference)
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# Control TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define command-line flags
_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    required=True,
    help='The glob pattern for directories with input frames.'
)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='Path of the TF2 SavedModel to use for frame interpolation.'
)
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=5,
    help=(
        'Number of recursive midpoint interpolations. '
        'Output frames = 2^times_to_interpolate + 1 (for two input frames).'
    )
)
_FPS = flags.DEFINE_integer(
    name='fps',
    default=30,
    help='Frame rate for the output video (if --output_video is True).'
)
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=64,
    help='If >1, pad input dimensions to be divisible by this value during inference.'
)
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='Number of patches along the image height (for high-resolution splitting).'
)
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='Number of patches along the image width (for high-resolution splitting).'
)
_OUTPUT_VIDEO = flags.DEFINE_boolean(
    name='output_video',
    default=False,
    help='If true, save an MP4 video of the interpolated frames in each directory.'
)

# Acceptable input image extensions
_INPUT_EXT = ['png', 'jpg', 'jpeg']

def _output_frames(frames: List[np.ndarray], frames_dir: str):
    """Writes a list of RGB frames (numpy arrays) as PNG files to a directory."""
    if tf.io.gfile.isdir(frames_dir):
        # Remove any existing output frames in the directory
        old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
        if old_frames:
            logging.info('Removing existing frames from %s.', frames_dir)
            for old_frame in old_frames:
                tf.io.gfile.remove(old_frame)
    else:
        tf.io.gfile.makedirs(frames_dir)
    # Save each frame with a sequential filename
    for idx, frame in tqdm(enumerate(frames), total=len(frames), ncols=100, colour='green'):
        util.write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
    logging.info('Output frames saved in %s.', frames_dir)

def _run_inference():
    """Runs frame interpolation on all directories matched by the pattern, using multi-GPU if available."""
    # Find all directories matching the pattern
    directories = tf.io.gfile.glob(_PATTERN.value)
    if not directories:
        logging.error('No input directories found with pattern: %s', _PATTERN.value)
        return

    # Determine if multiple GPUs are available
    physical_gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_gpus)
    if num_gpus > 1:
        # Use MirroredStrategy for multi-GPU parallelism
        strategy = tf.distribute.MirroredStrategy()
        logging.info('Found %d GPUs. Using MirroredStrategy for parallel inference.', num_gpus)
    else:
        # Use default strategy (single GPU or CPU)
        strategy = tf.distribute.get_strategy()  # This will be a SingleDevice or default strategy
        if num_gpus == 1:
            logging.info('Found 1 GPU. Running on single GPU without MirroredStrategy.')
        else:
            logging.info('No GPU found. Running on CPU.')

    # Create the interpolator model within the strategy scope (model gets replicated to each GPU)
    with strategy.scope():
        interpolator = interpolator_lib.Interpolator(
            _MODEL_PATH.value, _ALIGN.value, [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value]
        )
    logging.info('Loaded interpolation model from %s', _MODEL_PATH.value if _MODEL_PATH.value else "TFHub default")

    # If video output is requested, configure ffmpeg for mediapy (do this once)
    if _OUTPUT_VIDEO.value:
        ffmpeg_path = util.get_ffmpeg_path()
        media.set_ffmpeg(ffmpeg_path)

    def process_directory(dir_path: str):
        """Process a single directory: read frames, interpolate, and save outputs."""
        # Gather and sort input frame file paths (for all supported extensions)
        input_frames_lists = [
            natsort.natsorted(tf.io.gfile.glob(f'{dir_path}/*.{ext}')) 
            for ext in _INPUT_EXT
        ]
        # Combine all found files into a single list
        input_frames = functools.reduce(lambda x, y: x + y, input_frames_lists)
        if not input_frames:
            logging.warning('No images found in %s, skipping.', dir_path)
            return
        logging.info('Generating in-between frames for %s.', dir_path)
        # Perform recursive interpolation to get intermediate frames
        frames = list(util.interpolate_recursively_from_files(
            input_frames, _TIMES_TO_INTERPOLATE.value, interpolator
        ))
        # Save interpolated frames to disk
        _output_frames(frames, f'{dir_path}/interpolated_frames')
        # Optionally, save a video of the interpolated sequence
        if _OUTPUT_VIDEO.value:
            output_video_path = f'{dir_path}/interpolated.mp4'
            media.write_video(output_video_path, frames, fps=_FPS.value)
            logging.info('Output video saved at %s.', output_video_path)

    if num_gpus > 1:
        # Multi-GPU execution: distribute directories among GPUs in batches
        batch_size = strategy.num_replicas_in_sync  # this should equal num_gpus
        for i in range(0, len(directories), batch_size):
            batch_dirs = directories[i : i + batch_size]
            # Create distributed inputs for this batch: one directory path per replica
            dist_dirs = strategy.experimental_distribute_values_from_function(
                lambda ctx: batch_dirs[ctx.replica_id_in_sync_group] 
                            if ctx.replica_id_in_sync_group < len(batch_dirs) else None
            )
            # Run processing in parallel on each GPU
            strategy.run(lambda dir_path: process_directory(dir_path) 
                         if dir_path is not None else None, args=(dist_dirs,))
    else:
        # Single-device execution: loop through directories sequentially
        for directory in directories:
            process_directory(directory)

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    _run_inference()

if __name__ == '__main__':
    app.run(main)
