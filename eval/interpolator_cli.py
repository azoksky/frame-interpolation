import os
import math
import logging
import numpy as np
import mediapy as media
import tensorflow as tf
from absl import app, flags
from tqdm.auto import tqdm
from eval import interpolator as interpolator_lib, util

# Define command-line flags for inputs (pattern, model path, etc.)
_PATTERN = flags.DEFINE_string(
    'pattern', None,
    'Glob pattern for directories with input frames (each directory is an independent task).',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    'model_path', None,
    'Path to the TF2 saved model to use for interpolation.', required=True)
_TIMES = flags.DEFINE_integer(
    'times_to_interpolate', 1,
    'Number of recursive midpoint interpolations (output frames = 2^times + 1).')
_FPS = flags.DEFINE_integer(
    'fps', 30, 'Frames per second for the output video if generated.')
_ALIGN = flags.DEFINE_integer(
    'align', 64, 'If >1, pad input size so it is divisible by this value (model requirement).')
_BLOCK_H = flags.DEFINE_integer(
    'block_height', 1, 'Number of patches along height for high-res images (1 = no tiling).')
_BLOCK_W = flags.DEFINE_integer(
    'block_width', 1, 'Number of patches along width for high-res images (1 = no tiling).')
_OUTPUT_VIDEO = flags.DEFINE_boolean(
    'output_video', False, 'If true, also save an MP4 video of the interpolated frames.')

# Utility to write frames to an output directory
def _output_frames(frames: list[np.ndarray], out_dir: str):
    if tf.io.gfile.isdir(out_dir):
        # Remove old frames if any
        for old_frame in tf.io.gfile.glob(f'{out_dir}/frame_*.png'):
            tf.io.gfile.remove(old_frame)
    else:
        tf.io.gfile.makedirs(out_dir)
    # Save each frame as PNG
    for idx, frame in tqdm(enumerate(frames), total=len(frames), unit="frame"):
        util.write_image(f'{out_dir}/frame_{idx:03d}.png', frame)
    logging.info('Output frames saved in %s.', out_dir)

def main(argv):
    del argv  # Unused
    # Prepare list of directories to process
    directories = tf.io.gfile.glob(_PATTERN.value)
    if not directories:
        raise ValueError(f"No directories found for pattern {_PATTERN.value}")

    # Enable memory growth for GPUs to allow multiple GPUs usage without conflicts
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    num_gpus = len(gpus)
    logging.info("Found %d GPUs for inference.", num_gpus)

    # Load the interpolation model under distribution scope
    strategy = tf.distribute.MirroredStrategy() if num_gpus > 1 else None
    if strategy:
        logging.info("Using MirroredStrategy with %d replicas.", strategy.num_replicas_in_sync)
        with strategy.scope():
            # Initialize the interpolator (this loads the model, e.g., from SavedModel)
            interp = interpolator_lib.Interpolator(_MODEL_PATH.value, _ALIGN.value,
                                                   [_BLOCK_H.value, _BLOCK_W.value])
    else:
        # Single GPU/CPU: no strategy, just load normally
        interp = interpolator_lib.Interpolator(_MODEL_PATH.value, _ALIGN.value,
                                               [_BLOCK_H.value, _BLOCK_W.value])

    # Define a python function to process one directory (one task)
    def process_directory(dir_path: str):
        if not dir_path:
            # Empty path used for padding – do nothing
            return 0
        logging.info("Processing directory: %s", dir_path)
        # Gather input frame file paths (supports png/jpg/jpeg)
        input_frames_lists = [
            tf.io.gfile.glob(f'{dir_path}/*.{ext}') for ext in ['png', 'jpg', 'jpeg']
        ]
        # Flatten and sort file list
        input_frames = util.sort_files(input_frames_lists) if hasattr(util, "sort_files") else \
                       sum(input_frames_lists, [])
        input_frames = sorted(input_frames)  # ensure sorted order by name
        if len(input_frames) < 2:
            raise ValueError(f"Directory {dir_path} does not contain at least two frames.")
        # Perform recursive interpolation on this sequence of frames
        frames = list(util.interpolate_recursively_from_files(
            input_frames, _TIMES.value, interp))
        # Save interpolated frames to disk
        out_dir = os.path.join(dir_path, "interpolated_frames")
        _output_frames(frames, out_dir)
        # Optionally, save video
        if _OUTPUT_VIDEO.value:
            video_path = os.path.join(dir_path, "interpolated.mp4")
            media.write_video(video_path, frames, fps=_FPS.value)
            logging.info("Output video saved at %s.", video_path)
        return 0  # return dummy result

    if strategy:
        # Create a tf.function to run distributed inference
        @tf.function
        def distributed_infer(per_replica_dirs):
            # Each replica runs the processing function on its assigned directory
            return strategy.run(lambda dir_path: tf.numpy_function(
                    func=lambda p: process_directory(p.decode('utf-8')),
                    inp=[dir_path], Tout=tf.int64),
                args=(per_replica_dirs,))

        # Process directories in batches of `num_gpus` for parallel execution
        total_dirs = len(directories)
        results = []
        for i in range(0, total_dirs, strategy.num_replicas_in_sync):
            batch_dirs = directories[i:i+strategy.num_replicas_in_sync]
            # Pad batch to have exactly num_replicas elements (if needed)
            if len(batch_dirs) < strategy.num_replicas_in_sync:
                batch_dirs += [""] * (strategy.num_replicas_in_sync - len(batch_dirs))
            # Create a per-replica value for the batch
            batch_dirs_tensor = tf.convert_to_tensor(batch_dirs, dtype=tf.string)
            per_replica_dirs = strategy.experimental_distribute_values_from_function(
                lambda ctx: batch_dirs_tensor[ctx.replica_id_in_sync_group])
            # Run distributed inference on this batch
            distributed_infer(per_replica_dirs)
            results.extend(batch_dirs[:len(batch_dirs)])  # record processed directories
        logging.info("Processed %d directories with multi-GPU MirroredStrategy.", len(results))
    else:
        # Single GPU/CPU: loop through directories sequentially
        for d in directories:
            process_directory(d)
            logging.info("Processed directory: %s", d)

if __name__ == '__main__':
    app.run(main)
