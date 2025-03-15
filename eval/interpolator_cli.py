import os
import math
import glob
import logging
import tensorflow as tf
import mediapy as media
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from absl import app, flags, logging as absl_logging
from eval import interpolator as interpolator_lib, util

# Configure logging to print to stdout.
absl_logging.set_verbosity(absl_logging.INFO)

# Define command-line flags.
FLAGS = flags.FLAGS
flags.DEFINE_string('input_pattern', None, 'Glob pattern for input image files (e.g., "/kaggle/working/Downloads/*.png").')
flags.DEFINE_string('model_path', None, 'Path to the saved model for interpolation.')
flags.DEFINE_integer('times_to_interpolate', 1, 'Number of recursive interpolations (output frames = 2^times + 1).')
flags.DEFINE_integer('fps', 30, 'Frames per second for the output video.')
flags.DEFINE_string('output_path', 'interpolated.mp4', 'Output video file path.')
flags.DEFINE_integer('align', 64, 'Alignment value to pad image dimensions.')
flags.DEFINE_integer('block_height', 1, 'Number of patches along image height (1 means no tiling).')
flags.DEFINE_integer('block_width', 1, 'Number of patches along image width (1 means no tiling).')

def process_frames(frame_paths, times_to_interpolate, model_path, output_fps=30,
                   align=64, block_height=1, block_width=1) -> str:
    """
    Interpolate between frames using manual GPU assignment.
    Splits the input frames into segments, processes each segment on a separate GPU,
    then merges the results into a single video.
    """
    if len(frame_paths) < 2:
        raise ValueError("Need at least two frames to interpolate.")
    frame_paths = sorted(frame_paths)
    
    # Enable GPU memory growth.
    physical_gpus = tf.config.list_physical_devices('GPU')
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    num_gpus = len(physical_gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU available.")
    
    total_pairs = len(frame_paths) - 1
    if num_gpus == 1 or total_pairs == 1:
        absl_logging.info("Using single-GPU mode")
        interpolator = interpolator_lib.Interpolator(model_path, align, [block_height, block_width])
        frames = list(util.interpolate_recursively_from_files(frame_paths, times_to_interpolate, interpolator))
        media.write_video(FLAGS.output_path, frames, fps=output_fps)
        return FLAGS.output_path

    num_gpus_to_use = min(num_gpus, total_pairs)
    pairs_per_gpu = math.ceil(total_pairs / num_gpus_to_use)
    
    segments = []
    for i in range(num_gpus_to_use):
        start = i * pairs_per_gpu
        end = min(total_pairs, (i + 1) * pairs_per_gpu)
        seg = frame_paths[start:end + 1]  # include one extra frame for overlap
        segments.append(seg)
    
    # Enforce overlap between segments.
    for j in range(1, len(segments)):
        segments[j][0] = segments[j-1][-1]
    
    absl_logging.info("Splitting %d frame pairs across %d GPUs.", total_pairs, num_gpus_to_use)
    absl_logging.info("Segment sizes: %s", [len(seg) for seg in segments])
    
    # Load one interpolator per GPU.
    interpolators = []
    for gpu_index in range(num_gpus_to_use):
        with tf.device(f'/GPU:{gpu_index}'):
            interpolators.append(interpolator_lib.Interpolator(model_path, align, [block_height, block_width]))
    
    # Worker function for each GPU segment with enhanced logging.
    def process_segment(idx: int) -> list:
        try:
            absl_logging.info("GPU %d: Starting segment %d with %d frames.", idx, idx, len(segments[idx]))
            with tf.device(f'/GPU:{idx}'):
                interp = interpolators[idx]
                seg_frames = segments[idx]
                result = list(util.interpolate_recursively_from_files(seg_frames, times_to_interpolate, interp))
            absl_logging.info("GPU %d: Finished segment %d. Generated %d frames.", idx, idx, len(result))
            return result
        except Exception as e:
            absl_logging.error("GPU %d: Exception in segment %d: %s", idx, idx, str(e))
            raise

    # Execute segment processing in parallel.
    results = [None] * len(segments)
    with ThreadPoolExecutor(max_workers=len(segments)) as executor:
        future_to_idx = {executor.submit(process_segment, idx): idx for idx in range(len(segments))}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                absl_logging.error("Segment %d generated an exception: %s", idx, exc)
                raise

    # Merge segments, removing duplicate overlap frames.
    final_frames = results[0]
    for seg in results[1:]:
        final_frames.extend(seg[1:])  # drop the first frame (overlap)
    
    media.write_video(FLAGS.output_path, final_frames, fps=output_fps)
    absl_logging.info("Video saved at: %s", FLAGS.output_path)
    return FLAGS.output_path

def main(argv):
    del argv  # Unused.
    # Use the input pattern flag to get frame file paths.
    frame_files = sorted(glob.glob(FLAGS.input_pattern))
    if not frame_files:
        raise ValueError('No frame files found for pattern: ' + FLAGS.input_pattern)
    absl_logging.info("Found %d frame files.", len(frame_files))
    
    video_path = process_frames(frame_files,
                                times_to_interpolate=FLAGS.times_to_interpolate,
                                model_path=FLAGS.model_path,
                                output_fps=FLAGS.fps,
                                align=FLAGS.align,
                                block_height=FLAGS.block_height,
                                block_width=FLAGS.block_width)
    print("Video saved at:", video_path)

if __name__ == '__main__':
    app.run(main)
